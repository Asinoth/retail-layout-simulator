"""Elasticity-uncertainty robustness sweep (paper audit item #7).

The GA composite layout score is ELASTICITY-INDEPENDENT; the cited
elasticity coefficients only map score -> (conversion, impulse, basket)
lifts -> projected revenue. The headline "GA beats baseline" claim is
therefore robust iff the PROJECTED LIFT stays positive across the
literature-implied coefficient bands -- which we can evaluate in closed
form without re-running the GA per draw.

Design:
  * N scenarios x S seeds: run the (moderate-budget) GA once per case and
    take the popularity_rank baseline for the same scenario.
  * Score both layouts once (elasticity-free composite + breakdown).
  * Latin Hypercube (scipy.stats.qmc) over the three total elasticities,
    each within its cited band [BASE, BASE + GAIN_MAX]:
        conversion  (Hui 2009),
        impulse     (Hui, Inman 2013),
        basket size (conservative band; flagged in retail_literature).
  * For each draw, closed-form expected daily revenue for both layouts
    (same structure as the audited _ga_fitness: relative abandonment,
    net base revenue, multiplicative basket, additive impulse delta),
    -> paired lift distribution across the band.

Outputs (experiments/results/elasticity_lhs_<ts>/):
    results.csv   one row per (scenario, seed, draw)
    summary.json  fraction of positive-lift draws, lift percentiles
    lhs_hist.png  lift histogram across all draws
    sidecar.json  git SHA, seeds, elasticity snapshot

Smoke:   python -m experiments.run_elasticity_lhs --n-scenarios 3 --n-seeds 1 --n-draws 64
Paper:   python -m experiments.run_elasticity_lhs --n-scenarios 12 --n-seeds 3 --n-draws 256
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from scipy.stats import qmc

from synthetic_shops import generate_synthetic_shop
from baselines import popularity_rank
from experiments._common import (build_headless_shop, base_params_for,
                                 run_ga_headless, write_sidecar, make_run_dir)
from retail_literature import (
    ELASTICITY_CONV_BASE, ELASTICITY_CONV_GAIN_MAX,
    ELASTICITY_IMP_BASE, ELASTICITY_IMP_GAIN_MAX,
    ELASTICITY_BSK_BASE, ELASTICITY_BSK_GAIN_MAX,
    ABANDON_FLOW_COEF, ABANDON_SECTION_COEF,
    ABANDON_BOTTLENECK_COEF, ABANDON_FLOOR_FRAC,
)

BANDS = {
    'conv': (ELASTICITY_CONV_BASE, ELASTICITY_CONV_BASE + ELASTICITY_CONV_GAIN_MAX),
    'imp':  (ELASTICITY_IMP_BASE,  ELASTICITY_IMP_BASE + ELASTICITY_IMP_GAIN_MAX),
    'bsk':  (ELASTICITY_BSK_BASE,  ELASTICITY_BSK_BASE + ELASTICITY_BSK_GAIN_MAX),
}


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
    ap.add_argument('--out-root', type=str,
                    default=os.path.join(_HERE, 'results'))
    return ap.parse_args()


def projected_daily_revenue(score, bkd, bp, conv_e, imp_e, bsk_e):
    """Closed-form expected daily revenue under drawn elasticities.

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
    op_eff_hours = 10.0 * ((5 + 2 * 1.4) / 7.0)
    daily_cust = bp['customers_per_hour'] * op_eff_hours
    return daily_cust * conv_final * (
        bp['rev_per_converting_customer'] * bsk_mult
        + imp_adj * bp['avg_impulse_value'])


def main():
    args = parse_args()
    t0 = time.time()
    out_dir = make_run_dir(args.out_root, 'elasticity_lhs')

    sampler = qmc.LatinHypercube(d=3, seed=123)
    unit = sampler.random(args.n_draws)
    lo = np.array([BANDS['conv'][0], BANDS['imp'][0], BANDS['bsk'][0]])
    hi = np.array([BANDS['conv'][1], BANDS['imp'][1], BANDS['bsk'][1]])
    draws = qmc.scale(unit, lo, hi)

    rows = []
    all_lifts = []
    for sc in range(args.n_scenarios):
        scenario = generate_synthetic_shop(
            name=f'lhs_{sc:03d}', seed=10_000 + sc, n_items=args.n_items)
        for seed in range(args.n_seeds):
            shop = build_headless_shop(scenario)
            bp = base_params_for(scenario)
            item_names = [it.name for it in scenario.items]

            ga_out = run_ga_headless(
                shop, item_names, bp,
                n_gens=args.n_gens, pop_size=args.pop_size,
                mc_iters=args.mc_iters, mc_days=args.mc_days,
                rng_seed=seed)
            ga_chrom = ga_out['best_chrom']
            base_layout = popularity_rank(scenario, seed=seed)
            base_chrom = np.array([base_layout[it.name] for it in scenario.items],
                                  dtype=np.float64)

            s_ga, b_ga = shop._ga_compute_layout_score(ga_chrom, item_names, bp)
            s_bl, b_bl = shop._ga_compute_layout_score(base_chrom, item_names, bp)

            for di, (ce, ie, be) in enumerate(draws):
                r_ga = projected_daily_revenue(s_ga, b_ga, bp, ce, ie, be)
                r_bl = projected_daily_revenue(s_bl, b_bl, bp, ce, ie, be)
                lift = (r_ga - r_bl) * 30.0
                pct = (r_ga / max(r_bl, 1e-9) - 1.0) * 100.0
                rows.append((sc, seed, di, ce, ie, be, r_ga * 30, r_bl * 30,
                             lift, pct))
                all_lifts.append(lift)
        print(f'[lhs] scenario {sc + 1}/{args.n_scenarios} done '
              f'({time.time() - t0:.1f}s)', flush=True)

    lifts = np.array(all_lifts)
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
        'frac_positive': float((lifts > 0).mean()),
        'lift_median': float(np.median(lifts)),
        'lift_p5': float(np.percentile(lifts, 5)),
        'lift_p95': float(np.percentile(lifts, 95)),
        'bands': {k: list(v) for k, v in BANDS.items()},
        'config': vars(args),
    }

    with open(os.path.join(out_dir, 'results.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['scenario', 'seed', 'draw', 'conv_e', 'imp_e', 'bsk_e',
                    'rev30_ga', 'rev30_baseline', 'lift30', 'lift_pct'])
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

    write_sidecar(out_dir, {'kind': 'elasticity_lhs', 'summary': summary})
    print(f'[lhs] frac positive = {summary["frac_positive"]*100:.1f}%  '
          f'median lift = ${summary["lift_median"]:,.0f}  '
          f'p5..p95 = [${summary["lift_p5"]:,.0f}, ${summary["lift_p95"]:,.0f}]')
    print('Artifacts in:', out_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())
