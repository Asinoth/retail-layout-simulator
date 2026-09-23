"""Elasticity- and weight-uncertainty robustness sweeps.

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
    metaheuristics (random search and simulated annealing). Both searches
    start from the as-built layout the GA's population is seeded from, so
    the three searches leave from the same point.
  * Score every layout once (elasticity-free composite + breakdown).
  * Latin Hypercube (scipy.stats.qmc) over the three total elasticities,
    each within its cited band [BASE, BASE + GAIN_MAX]:
        conversion  (Hui 2009),
        impulse     (Hui, Inman 2013),
        basket size (conservative band; flagged in retail_literature).
  * For each draw, closed-form expected horizon revenue for every layout
    (same structure as _ga_fitness: relative abandonment, net base
    revenue, multiplicative basket, additive impulse delta),
    -> paired lift distribution across the band, per comparison.

Weight sweep -- a second design over the same scored layouts:
  The composite's seven weights are fixed in every experiment: the five
  spatial criteria (traffic, cross-merchandising, impulse, flow, revenue
  placement) share 0.80 equally, section compliance holds 0.16 and
  accessibility 0.04 (retail_literature). The equal split of the five
  rests on Dawes (1979), an argument for improper linear models, not a
  measurement. This design tests it without any extra GA or search run:
  * W spatial weight vectors drawn uniformly on the simplex scaled to the
    spatial total, Dirichlet(1,1,1,1,1) x 0.80, from a dedicated generator
    (``--weight-seed``). Section compliance, accessibility and both
    penalties keep their values.
  * Each layout's composite is recomposed from the per-criterion
    breakdown ``_ga_compute_layout_score`` already returned. Before any
    draw is evaluated, recomposing at the default weights must reproduce
    the returned score to 1e-9, so the sweep measures the weights and not
    the recomposer.
  * The recomposed composite goes through the same closed-form horizon
    revenue, at the elasticity band midpoints. Only the composite moves:
    the impulse elasticity still reads the impulse sub-score, and
    abandonment the flow, section and bottleneck sub-scores.
  What it does and does not test: every layout was SEARCHED under the
  default weights. The sweep asks whether the ranking of the layouts in
  hand survives when they are SCORED under other weightings of the
  criteria. It does not ask whether the GA would still win if it searched
  under other weights; that needs one search per weight vector, which is
  a different experiment. A (scenario, seed) pair counts as a sign flip
  when its lift takes both signs over the default weights and the W
  draws. W points only sample the simplex, so the count is a lower bound
  on the pairs whose lift changes sign somewhere in it.

Outputs (experiments/results/elasticity_lhs_<ts>/):
    results.csv       one row per (scenario, seed, draw)
    weight_sweep.csv  one row per (scenario, seed, weight draw,
                      comparator): the five spatial weights, the
                      GA-minus-comparator lift, and the pair's lift at the
                      default weights
    summary.json      fraction of positive-lift draws and lift percentiles
                      for the popularity baseline, and the same per
                      comparison under 'comparisons'; the weight design and
                      its per-comparison results under 'weight_sweep'
    lhs_hist.png      lift histogram across all draws
    sidecar.json      git SHA, seeds, elasticity snapshot

Smoke:   python -m experiments.run_elasticity_lhs --n-scenarios 3 --n-seeds 1 --n-draws 64
Paper:   python -m experiments.run_elasticity_lhs --n-scenarios 12 --n-seeds 3 --n-draws 256 \
             --n-gens 15 --pop-size 24 --mc-iters 500
         (the GA budget flags are required: the argparse defaults are a
         smaller budget than the published run; each scenario-seed now also
         runs the two equal-budget searches, so it costs about three GA runs)
         ``--workers N`` runs scenarios in N processes; the CSVs are unchanged.
         ``--n-weight-draws`` (default 256) and ``--weight-seed`` set the
         weight design; it re-scores the layouts above and adds no search.
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
                                 write_sidecar, make_run_dir, portable_paths)
from experiments.metaheuristics import random_search, simulated_annealing
from retail_literature import (
    ELASTICITY_CONV_BASE, ELASTICITY_CONV_GAIN_MAX,
    ELASTICITY_IMP_BASE, ELASTICITY_IMP_GAIN_MAX,
    ELASTICITY_BSK_BASE, ELASTICITY_BSK_GAIN_MAX,
    ABANDON_FLOW_COEF, ABANDON_SECTION_COEF,
    ABANDON_BOTTLENECK_COEF, ABANDON_FLOOR_FRAC,
    DEFAULT_OP_HOURS_PER_DAY, DEFAULT_WEEKEND_MULTIPLIER,
    GA_W_TRAFFIC, GA_W_CROSS_MERCH, GA_W_IMPULSE, GA_W_FLOW,
    GA_W_REVENUE_PLACEMENT, GA_W_SECTION_COMPLIANCE, GA_W_ACCESSIBILITY,
    GA_PEN_OVERLAP, GA_PEN_BOTTLENECK,
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

# --- weight sweep ---------------------------------------------------------
# The five spatial criteria, in the order their weights are drawn and
# written, with the weights the composite uses today. They share
# SPATIAL_TOTAL (0.80); section compliance, accessibility and the two
# penalties are held at their retail_literature values throughout.
SPATIAL_CRITERIA = ('traffic', 'cross_merch', 'impulse', 'flow',
                    'revenue_placement')
DEFAULT_SPATIAL_WEIGHTS = (GA_W_TRAFFIC, GA_W_CROSS_MERCH, GA_W_IMPULSE,
                           GA_W_FLOW, GA_W_REVENUE_PLACEMENT)
SPATIAL_TOTAL = float(sum(DEFAULT_SPATIAL_WEIGHTS))

# How closely the recomposed default-weight composite must match the score
# _ga_compute_layout_score returned before the sweep may run.
RECOMPOSE_TOL = 1e-9

# The weight sweep holds the elasticities at their band midpoints, the
# values the GA's own fitness (viz_ga_run._ga_fitness) searches under, so
# only the weights vary.
ELASTICITY_MIDPOINTS = {k: 0.5 * (lo + hi) for k, (lo, hi) in BANDS.items()}

WEIGHT_SEED = 1979

# weight_sweep.csv layout. ``lift30_default`` repeats the pair's lift at the
# default weights on every row, so the sign-flip count in summary.json can
# be recomputed from this file alone.
WEIGHT_COLUMNS = (['scenario', 'seed', 'draw', 'comparator']
                  + [f'w_{c}' for c in SPATIAL_CRITERIA]
                  + ['lift30', 'lift30_default'])


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
    ap.add_argument('--n-weight-draws', type=int, default=256,
                    help='Spatial weight vectors drawn for the weight sweep')
    ap.add_argument('--weight-seed', type=int, default=WEIGHT_SEED,
                    help='Seed of the weight-sweep generator')
    ap.add_argument('--out-root', type=str,
                    default=os.path.join(_HERE, 'results'))
    args = ap.parse_args()
    if args.workers < 1:
        ap.error('--workers must be >= 1')
    if args.n_weight_draws < 1:
        ap.error('--n-weight-draws must be >= 1')
    return args


def projected_horizon_revenue(score, bkd, bp, conv_e, imp_e, bsk_e):
    """Closed-form expected revenue over the projection horizon under drawn
    elasticities.

    Mirrors the _ga_fitness structure: conversion lift on the
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


def recompose_score(bkd, spatial_weights):
    """Composite layout score of a breakdown under other spatial weights.

    ``bkd`` is the per-criterion breakdown ``_ga_compute_layout_score``
    returns and ``spatial_weights`` the five weights in SPATIAL_CRITERIA
    order. Section compliance, accessibility and both penalties keep their
    retail_literature weights. The terms are summed in the order
    ``_ga_compute_layout_score`` sums them, so at the default weights the
    result is that function's own score, not an approximation of it."""
    w = spatial_weights
    return float(bkd['traffic']            * w[0]
                 + bkd['cross_merch']        * w[1]
                 + bkd['impulse']            * w[2]
                 + bkd['flow']               * w[3]
                 + bkd['revenue_placement']  * w[4]
                 + bkd['section_compliance'] * GA_W_SECTION_COMPLIANCE
                 + bkd['accessibility']      * GA_W_ACCESSIBILITY
                 - bkd['overlap_penalty']    * GA_PEN_OVERLAP
                 - bkd['bottleneck_penalty'] * GA_PEN_BOTTLENECK)


def check_recomposition(score, bkd, tol=RECOMPOSE_TOL):
    """Raise AssertionError unless recomposing ``bkd`` at the default
    weights gives back ``score``.

    A recomposer that drifted from the composite (a criterion added,
    renamed or re-weighted in viz_ga) would otherwise turn the weight sweep
    into a measurement of the drift. Raised explicitly so ``python -O``
    cannot strip the check."""
    got = recompose_score(bkd, DEFAULT_SPATIAL_WEIGHTS)
    if not abs(got - float(score)) <= tol:
        raise AssertionError(
            f'recomposed composite {got!r} differs from the layout score '
            f'{float(score)!r} by more than {tol:g}; the weight sweep would '
            f'measure the recomposer rather than the weights')


def draw_spatial_weights(n_draws, seed):
    """``n_draws`` x 5 spatial weight vectors, uniform on the simplex scaled
    to SPATIAL_TOTAL: Dirichlet(1,1,1,1,1) x 0.80, from a generator of its
    own so the draws depend on ``seed`` alone."""
    rng = np.random.default_rng(seed)
    return rng.dirichlet(np.ones(len(SPATIAL_CRITERIA)),
                         size=n_draws) * SPATIAL_TOTAL


def weight_sweep_lifts(scores, weight_draws, bp):
    """GA-minus-comparator horizon lifts under each spatial weight vector.

    ``scores`` maps layout name -> ``(score, breakdown)`` as
    ``_ga_compute_layout_score`` returned them, 'GA' and every entry of
    COMPARATORS among them. Every layout's recomposition is checked before
    any draw is evaluated. Revenue is ``projected_horizon_revenue`` at the
    elasticity midpoints with the recomposed composite and the breakdown
    unchanged, so impulse and abandonment read their own sub-scores exactly
    as in the elasticity design.

    Returns ``{comparator: (lift at the default weights, array of lifts,
    one per row of weight_draws)}``."""
    for s, b in scores.values():
        check_recomposition(s, b)
    ce = ELASTICITY_MIDPOINTS['conv']
    ie = ELASTICITY_MIDPOINTS['imp']
    be = ELASTICITY_MIDPOINTS['bsk']

    def _rev(b, w):
        return projected_horizon_revenue(recompose_score(b, w), b, bp,
                                         ce, ie, be)

    at_default = {name: _rev(b, DEFAULT_SPATIAL_WEIGHTS)
                  for name, (_, b) in scores.items()}
    swept = {name: np.array([_rev(b, w) for w in weight_draws], dtype=float)
             for name, (_, b) in scores.items()}
    return {m: (float(at_default['GA'] - at_default[m]),
                swept['GA'] - swept[m])
            for m in COMPARATORS}


def weight_sweep_summary(weight_rows):
    """Per-comparator results of the weight sweep from weight_sweep.csv
    rows (WEIGHT_COLUMNS order).

    The lift percentiles pool every (scenario, seed, draw). A (scenario,
    seed) pair counts in ``n_pairs_sign_flip`` when its lift is strictly
    positive somewhere and strictly negative somewhere over the default
    weights and the drawn ones, i.e. the ranking of that pair's two
    layouts depends on how the criteria are weighted."""
    col = {c: i for i, c in enumerate(WEIGHT_COLUMNS)}
    out = {}
    for m in COMPARATORS:
        mine = [r for r in weight_rows if r[col['comparator']] == m]
        lifts = np.array([r[col['lift30']] for r in mine], dtype=float)
        by_pair = {}
        for r in mine:
            key = (r[col['scenario']], r[col['seed']])
            by_pair.setdefault(key, [r[col['lift30_default']]]).append(
                r[col['lift30']])
        out[m] = {
            'frac_positive': float((lifts > 0).mean()),
            'median': float(np.median(lifts)),
            'p5': float(np.percentile(lifts, 5)),
            'p95': float(np.percentile(lifts, 95)),
            'n_pairs_sign_flip': int(sum(1 for v in by_pair.values()
                                         if max(v) > 0 and min(v) < 0)),
            'n_pairs_positive_at_default': int(sum(
                1 for v in by_pair.values() if v[0] > 0)),
            'n_pairs': len(by_pair),
            'n_values': int(lifts.size),
        }
    return out


def lhs_scenario(sc, args, draws, weight_draws, t0):
    """Result rows of one scenario, in order: the elasticity rows (every
    (seed, draw) pair) and the weight-sweep rows (every (seed, weight draw,
    comparator))."""
    rows = []
    weight_rows = []
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
        # block, as in the paired-MC comparison they come from, and start
        # from the as-built snapshot the GA is seeded from: random search
        # spends its first evaluation on it and SA anneals away from it, so
        # no search is handed a better starting point than the others.
        budget = args.pop_size * args.n_gens
        layouts = {
            'GA': chromosome_to_layout(ga_out['best_chrom'], item_names),
            'popularity_rank': popularity_rank(scenario, seed=seed),
            'oracle': oracle_layout,
            'random_search': random_search(
                shop, scenario, item_names, bp, seed=seed, budget=budget,
                block=args.pop_size, mc_iters=args.mc_iters,
                mc_days=args.mc_days, init_layout=init_layout),
            'simulated_annealing': simulated_annealing(
                shop, scenario, item_names, bp, seed=seed, budget=budget,
                block=args.pop_size, mc_iters=args.mc_iters,
                mc_days=args.mc_days, init_layout=init_layout,
                start='asbuilt'),
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

        # Weight sweep: the same scored layouts, re-weighted. No layout is
        # searched or scored again; weight_sweep_lifts checks the
        # recomposition against these scores before it evaluates a draw.
        sweep = weight_sweep_lifts(scores, weight_draws, bp)
        for wi, w in enumerate(weight_draws):
            for m in COMPARATORS:
                at_default, lifts = sweep[m]
                weight_rows.append((sc, seed, wi, m, *(float(x) for x in w),
                                    float(lifts[wi]), at_default))
    print(f'[lhs] scenario {sc + 1}/{args.n_scenarios} done '
          f'({time.time() - t0:.1f}s)', flush=True)
    return rows, weight_rows


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
    # Drawn once, here, and shared by every scenario and seed, so each
    # (scenario, seed) pair is re-weighted by the same vectors and the
    # draws do not depend on --workers.
    weight_draws = draw_spatial_weights(args.n_weight_draws, args.weight_seed)

    per_scenario = _map_scenarios(lhs_scenario, args.n_scenarios,
                                  args.workers, args, draws, weight_draws, t0)
    rows = [row for sc_rows, _ in per_scenario for row in sc_rows]
    weight_rows = [row for _, sc_rows in per_scenario for row in sc_rows]

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
        # -- keep them distinct so the paper can quote the right one.
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
        # The second design: the same layouts under other weightings of the
        # five spatial criteria. The layouts were searched under the
        # default weights, so this tests whether their ranking survives
        # re-weighting when scored, not whether a search under other
        # weights would rank the methods the same way.
        'weight_sweep': {
            'design': {
                'n_weight_draws': int(args.n_weight_draws),
                'seed': int(args.weight_seed),
                'distribution': ('Dirichlet(1,1,1,1,1) x spatial_total over '
                                 'the spatial criteria'),
                'spatial_criteria': list(SPATIAL_CRITERIA),
                'spatial_total': SPATIAL_TOTAL,
                'default_spatial_weights': dict(zip(SPATIAL_CRITERIA,
                                                    DEFAULT_SPATIAL_WEIGHTS)),
                'fixed': {
                    'section_compliance': GA_W_SECTION_COMPLIANCE,
                    'accessibility': GA_W_ACCESSIBILITY,
                    'overlap_penalty': GA_PEN_OVERLAP,
                    'bottleneck_penalty': GA_PEN_BOTTLENECK,
                    'elasticities': dict(ELASTICITY_MIDPOINTS),
                    'layouts': ('searched under the default weights and '
                                're-scored from their breakdowns; no '
                                'search is re-run'),
                },
                'recompose_tolerance': RECOMPOSE_TOL,
                'sign_flip': ('lift strictly positive and strictly negative '
                              'somewhere over the default weights and the '
                              'drawn ones'),
            },
            'n_pairs': int(args.n_scenarios * args.n_seeds),
            'comparisons': weight_sweep_summary(weight_rows),
        },
    }

    with open(os.path.join(out_dir, 'results.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        w.writerows(rows)
    with open(os.path.join(out_dir, 'weight_sweep.csv'), 'w',
              newline='') as f:
        w = csv.writer(f)
        w.writerow(WEIGHT_COLUMNS)
        w.writerows(weight_rows)
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(portable_paths(summary), f, indent=2)

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
    for m, st in summary['weight_sweep']['comparisons'].items():
        print(f'[weights] GA - {m:20s} positive in '
              f'{st["frac_positive"]*100:5.1f}% of draws  '
              f'median ${st["median"]:,.0f}  '
              f'p5..p95 = [${st["p5"]:,.0f}, ${st["p95"]:,.0f}]  '
              f'sign flips in {st["n_pairs_sign_flip"]}/{st["n_pairs"]} pairs')
    print('Artifacts in:', out_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())
