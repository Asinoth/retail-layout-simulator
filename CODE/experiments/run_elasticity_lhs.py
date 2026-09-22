"""Elasticity-uncertainty robustness sweep (paper audit item #7).

The GA composite layout score is ELASTICITY-INDEPENDENT; the cited
elasticity coefficients only map score -> (conversion, impulse, basket)
lifts -> projected revenue. The headline "GA beats baseline" claim is
therefore robust iff the PROJECTED LIFT stays positive across the
literature-implied coefficient bands -- which we can evaluate in closed
form without re-running the GA per draw.

Design:
  * N scenarios x S seeds: run the (moderate-budget) GA once per case and
    build, on the same scenario and the same feasible set, the comparators
    the paper's headline comparisons use -- the popularity_rank baseline,
    the analytical reference solution, and the two equal-budget
    metaheuristics (random search and simulated annealing).
  * Score every layout once (elasticity-free composite + breakdown).
  * Latin Hypercube (scipy.stats.qmc) over the three total elasticities,
    each within its cited band [BASE, BASE + GAIN_MAX]:
        conversion  (Hui 2009),
        impulse     (Hui, Inman 2013),
        basket size (conservative band; flagged in retail_literature).
  * For each draw, closed-form expected horizon revenue for every layout
    (same structure as the audited _ga_fitness: relative abandonment,
    net base revenue, multiplicative basket, additive impulse delta),
    -> paired lift distribution across the band, per comparison.

Outputs (experiments/results/elasticity_lhs_<ts>/):
    results.csv   one row per (scenario, seed, draw)
    summary.json  fraction of positive-lift draws and lift percentiles for
                  the popularity baseline, and the same per comparison
                  under 'comparisons'
    lhs_hist.png  lift histogram across all draws
    sidecar.json  git SHA, seeds, elasticity snapshot

Smoke:   python -m experiments.run_elasticity_lhs --n-scenarios 3 --n-seeds 1 --n-draws 64
Paper:   python -m experiments.run_elasticity_lhs --n-scenarios 12 --n-seeds 3 --n-draws 256 \
             --n-gens 15 --pop-size 24 --mc-iters 500
         (the GA budget flags are required: the argparse defaults are a
         smaller budget than the published run; each scenario-seed now also
         runs the two equal-budget searches, so it costs about three GA runs)
         ``--workers N`` runs scenarios in N processes; the CSV is unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from scipy.stats import qmc

from synthetic_shops import generate_synthetic_shop
from baselines import popularity_rank, assert_no_strict_overlap
from oracle import solve_oracle
from experiments._common import (build_headless_shop, base_params_for,
                                 run_ga_headless, feasible_layout,
                                 chromosome_to_layout, layout_to_chromosome,
                                 write_sidecar, make_run_dir)
from experiments.metaheuristics import random_search, simulated_annealing
from retail_literature import (
    ELASTICITY_CONV_BASE, ELASTICITY_CONV_GAIN_MAX,
    ELASTICITY_IMP_BASE, ELASTICITY_IMP_GAIN_MAX,
    ELASTICITY_BSK_BASE, ELASTICITY_BSK_GAIN_MAX,
    ABANDON_FLOW_COEF, ABANDON_SECTION_COEF,
    ABANDON_BOTTLENECK_COEF, ABANDON_FLOOR_FRAC,
    DEFAULT_OP_HOURS_PER_DAY, DEFAULT_WEEKEND_MULTIPLIER,
)

BANDS = {
    'conv': (ELASTICITY_CONV_BASE, ELASTICITY_CONV_BASE + ELASTICITY_CONV_GAIN_MAX),
    'imp':  (ELASTICITY_IMP_BASE,  ELASTICITY_IMP_BASE + ELASTICITY_IMP_GAIN_MAX),
    'bsk':  (ELASTICITY_BSK_BASE,  ELASTICITY_BSK_BASE + ELASTICITY_BSK_GAIN_MAX),
}

# The comparators the GA is differenced against, in report order. The first
# is the popularity-rank baseline this sweep has always used; the other
# three are the comparisons the paper's headline table reports, so the
# elasticity bands are swept over all of them rather than one.
COMPARATORS = ('popularity_rank', 'oracle', 'random_search',
               'simulated_annealing')

# Length of the projected horizon, and its customer-hours counted the way
# mc_engine counts them: it takes day d of the horizon as day d % 7 of the
# week, so a 30-day horizon holds 8 weekend days, not 30 * 2/7 of them.
# Using the average-week share instead would scale every projected sum here
# about 0.7% above the MC objective this closed form mirrors.
MC_DAYS = 30
HORIZON_CUST_HOURS = DEFAULT_OP_HOURS_PER_DAY * sum(
    DEFAULT_WEEKEND_MULTIPLIER if d % 7 in (5, 6) else 1.0
    for d in range(MC_DAYS))

# results.csv layout. ``rev30_baseline``, ``lift30`` and ``lift_pct`` keep
# their names for the popularity baseline and the GA-minus-popularity
# difference the lift figure is drawn from; the other comparators follow.
COLUMNS = (['scenario', 'seed', 'draw', 'conv_e', 'imp_e', 'bsk_e',
            'rev30_ga', 'rev30_baseline']
           + [f'rev30_{m}' for m in COMPARATORS[1:]]
           + ['lift30', 'lift_pct']
           + [f'lift30_{m}' for m in COMPARATORS[1:]])


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-scenarios', type=int, default=12)
    ap.add_argument('--n-seeds', type=int, default=3)
    ap.add_argument('--n-draws', type=int, default=256)
    ap.add_argument('--n-items', type=int, default=10)
    ap.add_argument('--n-gens', type=int, default=12)
    ap.add_argument('--pop-size', type=int, default=20)
    ap.add_argument('--mc-iters', type=int, default=300)
    ap.add_argument('--mc-days', type=int, default=14)
    ap.add_argument('--workers', type=int, default=1,
                    help='Processes running scenarios in parallel')
    ap.add_argument('--out-root', type=str,
                    default=os.path.join(_HERE, 'results'))
    args = ap.parse_args()
    if args.workers < 1:
        ap.error('--workers must be >= 1')
    return args


def projected_horizon_revenue(score, bkd, bp, conv_e, imp_e, bsk_e):
    """Closed-form expected revenue over the projection horizon under drawn
    elasticities.

    Mirrors the audited fitness structure: conversion lift on the
    composite score with RELATIVE abandonment, multiplicative basket
    lift on NET base revenue, additive impulse delta."""
    conv_adj = bp['conversion_rate'] * (1.0 + score * conv_e)
    conv_adj = min(max(conv_adj, 0.01), 0.99)

    ab = bp.get('abandonment_rate', 0.0)
    abandon_adj = ab * max(ABANDON_FLOOR_FRAC,
                           1.0 - bkd.get('flow', 0) * ABANDON_FLOW_COEF
                           - bkd.get('section_compliance', 0) * ABANDON_SECTION_COEF
                           + bkd.get('bottleneck_penalty', 0) * ABANDON_BOTTLENECK_COEF)
    abandon_adj = min(max(abandon_adj, 0.0), 0.5)
    conv_final = conv_adj * (1.0 - abandon_adj) / max(1.0 - min(ab, 0.95), 1e-6)
    conv_final = min(max(conv_final, 0.01), 0.99)

    imp_adj = bp['impulse_rate'] * (1.0 + bkd.get('impulse', 0) * imp_e)
    imp_adj = min(max(imp_adj, 0.0), 0.99)

    bsk_mult = 1.0 + score * bsk_e
    horizon_cust = bp['customers_per_hour'] * HORIZON_CUST_HOURS
    return horizon_cust * conv_final * (
        bp['rev_per_converting_customer'] * bsk_mult
        + imp_adj * bp['avg_impulse_value'])


def lhs_scenario(sc, args, draws, t0):
    """Result rows of one scenario: every (seed, draw) pair, in order."""
    rows = []
    scenario = generate_synthetic_shop(
        name=f'lhs_{sc:03d}', seed=10_000 + sc, n_items=args.n_items)
    # The analytical reference depends on the scenario alone (its solver
    # starts are seeded from it), so it is solved once for every seed.
    oracle_layout = solve_oracle(scenario, n_restarts=24).layout
    for seed in range(args.n_seeds):
        shop = build_headless_shop(scenario)
        bp = base_params_for(scenario)
        item_names = [it.name for it in scenario.items]
        init_layout = {n: tuple(shop.floors[1]['items'][n]['position'])
                       for n in item_names}

        ga_out = run_ga_headless(
            shop, item_names, bp,
            n_gens=args.n_gens, pop_size=args.pop_size,
            mc_iters=args.mc_iters, mc_days=args.mc_days,
            rng_seed=seed, init_layout=init_layout)

        # Every comparator is scored on the GA's feasible set, like every
        # GA candidate, so a lift is not a by-product of the repair itself.
        # The two searches get the GA's own evaluation budget and seed
        # block, as in the paired-MC comparison they come from.
        budget = args.pop_size * args.n_gens
        layouts = {
            'GA': chromosome_to_layout(ga_out['best_chrom'], item_names),
            'popularity_rank': popularity_rank(scenario, seed=seed),
            'oracle': oracle_layout,
            'random_search': random_search(
                shop, scenario, item_names, bp, seed=seed, budget=budget,
                block=args.pop_size, mc_iters=args.mc_iters,
                mc_days=args.mc_days),
            'simulated_annealing': simulated_annealing(
                shop, scenario, item_names, bp, seed=seed, budget=budget,
                block=args.pop_size, mc_iters=args.mc_iters,
                mc_days=args.mc_days),
        }
        scores = {}
        for name, lay in layouts.items():
            lay = feasible_layout(shop, item_names, lay)
            if name == 'oracle':
                # Held to zero overlap as in the paired-MC comparison: the
                # layout score penalizes any positive overlap, so solver
                # contact residue would sink the reference on a technicality.
                assert_no_strict_overlap(scenario, lay)
            chrom = layout_to_chromosome(lay, item_names)
            scores[name] = shop._ga_compute_layout_score(chrom, item_names, bp)

        for di, (ce, ie, be) in enumerate(draws):
            rev = {name: projected_horizon_revenue(s, b, bp, ce, ie, be)
                   for name, (s, b) in scores.items()}
            r_ga = rev['GA']
            r_bl = rev['popularity_rank']
            pct = (r_ga / max(r_bl, 1e-9) - 1.0) * 100.0
            rows.append((sc, seed, di, ce, ie, be, r_ga,
                         *(rev[m] for m in COMPARATORS),
                         r_ga - r_bl, pct,
                         *(r_ga - rev[m] for m in COMPARATORS[1:])))
    print(f'[lhs] scenario {sc + 1}/{args.n_scenarios} done '
          f'({time.time() - t0:.1f}s)', flush=True)
    return rows


def _map_scenarios(fn, n_scenarios, workers, *fn_args):
    """``[fn(s, *fn_args) for s in range(n_scenarios)]``, in scenario order.

    With ``workers`` > 1 the scenarios run in separate processes. They are
    independent (each builds its own shop and reseeds every RNG it draws
    from), and results are put back in scenario order, so the output is
    identical to the serial run."""
    if workers <= 1:
        return [fn(s, *fn_args) for s in range(n_scenarios)]
    done = {}
    pool = ProcessPoolExecutor(max_workers=workers)
    try:
        futures = {pool.submit(fn, s, *fn_args): s for s in range(n_scenarios)}
        for fut in as_completed(futures):
            done[futures[fut]] = fut.result()
    except BaseException:
        # Leaving the pool's context manager would wait for every queued
        # scenario first, so a scenario that fails minutes into a run of
        # hours would only report at the end. Drop what has not started and
        # let the error out now.
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown(wait=True)
    return [done[s] for s in sorted(done)]


def main():
    args = parse_args()
    t0 = time.time()
    out_dir = make_run_dir(args.out_root, 'elasticity_lhs')

    sampler = qmc.LatinHypercube(d=3, seed=123)
    unit = sampler.random(args.n_draws)
    lo = np.array([BANDS['conv'][0], BANDS['imp'][0], BANDS['bsk'][0]])
    hi = np.array([BANDS['conv'][1], BANDS['imp'][1], BANDS['bsk'][1]])
    draws = qmc.scale(unit, lo, hi)

    rows = [row
            for sc_rows in _map_scenarios(lhs_scenario, args.n_scenarios,
                                          args.workers, args, draws, t0)
            for row in sc_rows]

    def _lift_stats(values):
        v = np.asarray(values, dtype=float)
        return {'frac_positive': float((v > 0).mean()),
                'lift_median': float(np.median(v)),
                'lift_p5': float(np.percentile(v, 5)),
                'lift_p95': float(np.percentile(v, 95)),
                'n_draws_total': int(v.size)}

    def _column(name):
        i = COLUMNS.index(name)
        return [row[i] for row in rows]

    # One entry per headline comparison, so the bands are swept over the
    # differences the paper reports and not only over GA-minus-popularity.
    comparisons = {
        m: _lift_stats(_column('lift30' if m == COMPARATORS[0]
                               else f'lift30_{m}'))
        for m in COMPARATORS
    }
    lifts = np.array(_column('lift30'), dtype=float)
    summary = {
        'n_draws_total': int(lifts.size),
        # Resolution of the LHS design in the 3-D elasticity box. The
        # total above multiplies this by scenarios x seeds, which is
        # replication, not additional coverage of the coefficient space
        # -- keep them distinct so the paper can quote the right one
        # (audit R67).
        'n_lhs_points': int(args.n_draws),
        'n_scenarios': int(args.n_scenarios),
        'n_seeds': int(args.n_seeds),
        # Top-level figures stay the GA-minus-popularity comparison this
        # sweep has always reported; the rest are under 'comparisons'.
        'frac_positive': float((lifts > 0).mean()),
        'lift_median': float(np.median(lifts)),
        'lift_p5': float(np.percentile(lifts, 5)),
        'lift_p95': float(np.percentile(lifts, 95)),
        'comparisons': comparisons,
        'horizon_days': MC_DAYS,
        'horizon_customer_hours': float(HORIZON_CUST_HOURS),
        'bands': {k: list(v) for k, v in BANDS.items()},
        'config': vars(args),
    }

    with open(os.path.join(out_dir, 'results.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        w.writerows(rows)
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.hist(lifts, bins=50, color='#4E79A7', edgecolor='white')
        ax.axvline(0, color='crimson', lw=1.2)
        ax.set_xlabel('30-day projected lift, GA − popularity baseline ($)')
        ax.set_ylabel('LHS draws')
        ax.set_title(f'Elasticity-band robustness: {summary["frac_positive"]*100:.1f}% '
                     f'of draws positive (n={lifts.size})')
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'lhs_hist.png'), dpi=130)
    except Exception as e:
        print('[lhs] plot skipped:', e)

    write_sidecar(out_dir, {'kind': 'elasticity_lhs', 'summary': summary,
                            'workers': args.workers})
    for m, st in comparisons.items():
        print(f'[lhs] GA - {m:20s} positive in {st["frac_positive"]*100:5.1f}% '
              f'of draws  median ${st["lift_median"]:,.0f}  '
              f'p5..p95 = [${st["lift_p5"]:,.0f}, ${st["lift_p95"]:,.0f}]')
    print('Artifacts in:', out_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())
