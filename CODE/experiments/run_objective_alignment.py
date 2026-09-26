"""Objective-alignment diagnostic: the GA's objective against the analytical one.

Figure A reports two things about the synthetic scenarios: a positive but
partial Spearman correlation between the GA's fitness and the analytical
revenue model, and a cross-objective reversal -- each objective prefers
its own optimizer's layout. Any two objectives with rank correlation below
one would show the reversal, so the question worth asking is where the
disagreement comes from. One candidate is structural: the analytical model
rewards wall proximity (``synthetic_shops.analytical_revenue``, its
perimeter term, Larson 2005), but the synthetic scenarios' traffic prior,
which the GA's traffic and revenue-placement criteria read, has no wall
term; only the calibrated store's prior does (retail_literature
HEAT_PRIOR_*).

This runner measures, on the synthetic scenarios, with closed forms only:

  (i)  ``as_now``        the synthetic prior as every experiment uses it,
                         5/(1 + d_entrance) + 3/(1 + d_checkout);
  (ii) ``wall_aligned``  the same prior plus the calibrated prior's wall
                         term, HEAT_PRIOR_CAL_WALL / (1 + max(d_wall, 0.1)),
                         so both objectives reward wall proximity;

and, in each variant, per scenario:

  * the Spearman rank correlation between the GA's fitness -- its exact
    mean, ``experiments.closed_form.expected_revenue``, anchored at the
    repaired as-built layout -- and the analytical revenue, over the same
    held-out sample of repaired random layouts Figure A draws (seed
    20_000 + scenario);
  * the cross-objective reversal: the GA run on its exact fitness (the
    GA's own operators, ``run_ga_headless`` with ``_ga_fitness`` replaced by
    ``expected_revenue``) and the analytical reference solution
    (``oracle.solve_oracle``), both mapped through ``feasible_layout``, each
    scored on the other's objective. ``gap_fitness_pct`` is how much the
    GA's objective prefers the GA's layout, ``gap_analytical_pct`` how
    much the analytical objective prefers the reference's; both positive is
    a reversal.

The analytical reference does not depend on the prior, so it is solved once
per scenario. No Monte Carlo runs anywhere: the fitness is its exact mean.

Outputs (experiments/results/objective_alignment_<ts>/):
    results.csv    one row per (scenario, variant)
    summary.json   per variant: the correlation (Fisher-z mean, median,
                   range), the reversal count and the median gaps; and the
                   paired change from (i) to (ii)
    sidecar.json   git SHA, seeds, elasticity snapshot, base parameters

Smoke:   python -m experiments.run_objective_alignment --n-scenarios 2 \
             --n-samples 20 --n-gens 3 --pop-size 8 --oracle-restarts 4
Full:    python -m experiments.run_objective_alignment
         (30 scenarios, 100 layouts each, the Figure A GA budget)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List

import numpy as np
from scipy.stats import spearmanr

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from synthetic_shops import generate_synthetic_shop, analytical_revenue
from baselines import _repair_within_section
from oracle import solve_oracle
from experiments._common import (ORACLE_RESTARTS, oracle_record,
                                 oracle_summary,
                                 anchor_base_params, base_params_for,
                                 base_params_record, build_headless_shop,
                                 checked_layout,
                                 chromosome_to_layout, feasible_layout,
                                 layout_to_chromosome, make_run_dir,
                                 paint_synthetic_heatmap, portable_paths,
                                 provenance_snapshot, run_ga_headless,
                                 write_sidecar)
from experiments.closed_form import expected_revenue
from experiments.run_synthetic_gt import random_within_section_bounds
from retail_literature import HEAT_PRIOR_CAL_WALL

#: The two traffic priors compared, in report order, with the wall
#: coefficient each paints (None: the --wall-coef argument).
VARIANTS = (('as_now', 0.0), ('wall_aligned', None))

COLUMNS = ['scenario', 'variant', 'wall_coef', 'spearman_rho', 'spearman_p',
           'n_layouts', 'fitness_ga', 'fitness_oracle', 'analytical_ga',
           'analytical_oracle', 'gap_fitness_pct', 'gap_analytical_pct',
           'reversal']


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--n-scenarios', type=int, default=30)
    p.add_argument('--n-items', type=int, default=10)
    p.add_argument('--n-samples', type=int, default=100,
                   help='Held-out repaired random layouts per scenario')
    p.add_argument('--horizon-days', type=int, default=30)
    p.add_argument('--n-gens', type=int, default=25)
    p.add_argument('--pop-size', type=int, default=30)
    p.add_argument('--ga-seed', type=int, default=0)
    p.add_argument('--oracle-restarts', '--n-restarts',
                   dest='oracle_restarts', type=int, default=ORACLE_RESTARTS,
                   help='L-BFGS-B restarts of the analytical reference '
                        '(oracle.solve_oracle); Figures A and B\'s default')
    p.add_argument('--wall-coef', type=float, default=HEAT_PRIOR_CAL_WALL,
                   help='Wall coefficient of the aligned prior (default: '
                        'the calibrated store prior\'s)')
    p.add_argument('--workers', type=int, default=1,
                   help='Processes running scenarios in parallel')
    p.add_argument('--out-root', type=str,
                   default=os.path.join(_HERE, 'results'))
    args = p.parse_args()
    if args.workers < 1:
        p.error('--workers must be >= 1')
    if args.n_samples < 3:
        p.error('--n-samples must be >= 3')
    if args.oracle_restarts < 1:
        p.error('--oracle-restarts must be >= 1')
    return args


def _closed_form_ga(shop, item_names, bp, args, init_layout):
    """The GA's own operators on its exact fitness: ``run_ga_headless`` with
    the instance's ``_ga_fitness`` replaced by ``expected_revenue``. The
    Monte Carlo seeds the GA sets are then irrelevant, so selection sees the
    objective itself."""
    def fitness(chrom, names, base_params, mc_days, mc_iters):
        return expected_revenue(base_params, mc_days, shop=shop,
                                chromosome=chrom, item_names=names)
    shop._ga_fitness = fitness
    try:
        out = run_ga_headless(shop, item_names, bp, pop_size=args.pop_size,
                              n_gens=args.n_gens, mc_iters=1,
                              mc_days=args.horizon_days,
                              rng_seed=args.ga_seed, init_layout=init_layout)
    finally:
        del shop._ga_fitness
    return chromosome_to_layout(out['best_chrom'], item_names)


def one_scenario(sc: int, args) -> Dict:
    synth = generate_synthetic_shop(name=f'align_{sc:03d}', seed=10_000 + sc,
                                    n_items=args.n_items, width=12.0,
                                    height=10.0)
    item_names = [it.name for it in synth.items]
    oracle_result = solve_oracle(synth, n_restarts=args.oracle_restarts)
    oracle_layout = oracle_result.layout
    rows, base_params = [], {}
    for variant, coef in VARIANTS:
        wall = args.wall_coef if coef is None else coef
        shop = build_headless_shop(synth)
        if wall:
            paint_synthetic_heatmap(shop, synth, wall_coef=wall)
        init_layout = {n: tuple(shop.floors[1]['items'][n]['position'])
                       for n in item_names}
        bp = anchor_base_params(shop, item_names, base_params_for(synth),
                                init_layout)
        base_params[variant] = base_params_record(bp)

        def fitness(lay):
            return expected_revenue(bp, args.horizon_days, shop=shop,
                                    chromosome=layout_to_chromosome(
                                        lay, item_names),
                                    item_names=item_names)

        # Figure A's held-out sample, drawn from the same seed in both
        # variants so the two correlations are paired layout for layout.
        rng = np.random.default_rng(20_000 + sc)
        sample = [feasible_layout(shop, item_names,
                                  _repair_within_section(
                                      synth,
                                      random_within_section_bounds(synth, rng)))
                  for _ in range(args.n_samples)]
        f_vals = np.array([fitness(lay) for lay in sample])
        a_vals = np.array([analytical_revenue(synth, lay) for lay in sample])
        rho, p_val = spearmanr(f_vals, a_vals)

        ga = feasible_layout(shop, item_names,
                             _closed_form_ga(shop, item_names, bp, args,
                                             init_layout))
        ref = feasible_layout(shop, item_names, oracle_layout)
        for label, lay in (('GA', ga), ('reference', ref)):
            checked_layout(shop, item_names, lay,
                           f"align scenario {sc} {variant} {label}")
        f_ga, f_ref = fitness(ga), fitness(ref)
        a_ga, a_ref = analytical_revenue(synth, ga), analytical_revenue(synth, ref)
        gap_f = (f_ga - f_ref) / max(abs(f_ref), 1e-9) * 100.0
        gap_a = (a_ref - a_ga) / max(abs(a_ga), 1e-9) * 100.0
        rows.append({'scenario': sc, 'variant': variant, 'wall_coef': wall,
                     'spearman_rho': float(rho), 'spearman_p': float(p_val),
                     'n_layouts': len(sample),
                     'fitness_ga': f_ga, 'fitness_oracle': f_ref,
                     'analytical_ga': a_ga, 'analytical_oracle': a_ref,
                     'gap_fitness_pct': gap_f, 'gap_analytical_pct': gap_a,
                     'reversal': bool(gap_f > 0 and gap_a > 0)})
        print(f'[align {sc:>2d} {variant:12s}] rho={rho:+.3f}  '
              f'fitness prefers GA by {gap_f:+.3f}%  analytical prefers '
              f'reference by {gap_a:+.3f}%', flush=True)
    return {'rows': rows, 'base_params': base_params,
            'oracle': oracle_record(oracle_result)}


def _map_scenarios(fn, n_scenarios, workers, *fn_args) -> list:
    """``[fn(s, *fn_args) for s in range(n_scenarios)]``, in scenario order;
    identical output for any worker count (each scenario seeds every
    generator it draws from)."""
    if workers <= 1:
        return [fn(s, *fn_args) for s in range(n_scenarios)]
    done = {}
    pool = ProcessPoolExecutor(max_workers=workers)
    try:
        futures = {pool.submit(fn, s, *fn_args): s for s in range(n_scenarios)}
        for fut in as_completed(futures):
            done[futures[fut]] = fut.result()
    except BaseException:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown(wait=True)
    return [done[s] for s in sorted(done)]


def _fisher_mean(rhos) -> float:
    r = np.clip(np.asarray(rhos, dtype=float), -0.999999, 0.999999)
    return float(np.tanh(np.arctanh(r).mean()))


def variant_summary(rows: List[Dict]) -> Dict:
    rho = np.array([r['spearman_rho'] for r in rows])
    gf = np.array([r['gap_fitness_pct'] for r in rows])
    ga = np.array([r['gap_analytical_pct'] for r in rows])
    return {
        'n_scenarios': len(rows),
        'wall_coef': rows[0]['wall_coef'] if rows else None,
        'spearman_fisher_mean': _fisher_mean(rho),
        'spearman_median': float(np.median(rho)),
        'spearman_min': float(rho.min()),
        'spearman_max': float(rho.max()),
        'n_reversal': int(sum(r['reversal'] for r in rows)),
        'n_fitness_prefers_ga': int((gf > 0).sum()),
        'n_analytical_prefers_reference': int((ga > 0).sum()),
        'gap_fitness_pct_median': float(np.median(gf)),
        'gap_analytical_pct_median': float(np.median(ga)),
    }


def main() -> int:
    args = parse_args()
    prov = provenance_snapshot()
    out_dir = make_run_dir(args.out_root, 'objective_alignment')
    print(f'output dir: {out_dir}', flush=True)
    t0 = time.perf_counter()
    results = _map_scenarios(one_scenario, args.n_scenarios, args.workers,
                             args)
    wall = time.perf_counter() - t0
    rows = [r for res in results for r in res['rows']]

    with open(os.path.join(out_dir, 'results.csv'), 'w', newline='',
              encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for r in rows:
            w.writerow([r[c] for c in COLUMNS])

    per_variant = {v: variant_summary([r for r in rows if r['variant'] == v])
                   for v, _ in VARIANTS}
    by_key = {(r['scenario'], r['variant']): r for r in rows}
    d_rho = np.array([by_key[(sc, 'wall_aligned')]['spearman_rho']
                      - by_key[(sc, 'as_now')]['spearman_rho']
                      for sc in range(args.n_scenarios)])
    summary = {
        'experiment': 'objective_alignment',
        'variants': per_variant,
        'change_as_now_to_wall_aligned': {
            'spearman_diff_mean': float(d_rho.mean()),
            'spearman_diff_median': float(np.median(d_rho)),
            'n_scenarios_rho_up': int((d_rho > 0).sum()),
            'n_reversal_as_now': per_variant['as_now']['n_reversal'],
            'n_reversal_wall_aligned': per_variant['wall_aligned']['n_reversal'],
        },
        'design': {
            'scenarios': 'generate_synthetic_shop(seed=10_000 + scenario), '
                         'as in Figure A',
            'sample': 'Figure A\'s held-out repaired random layouts '
                      '(seed 20_000 + scenario), the same in both variants',
            'fitness': 'experiments.closed_form.expected_revenue (exact mean '
                       'of the GA fitness), anchored at the repaired '
                       'as-built layout',
            'ga': 'run_ga_headless on the exact fitness, '
                  f'pop {args.pop_size} x {args.n_gens} gens, seed '
                  f'{args.ga_seed}',
            'reference': f'oracle.solve_oracle(n_restarts='
                         f'{args.oracle_restarts})',
            'reversal': 'the fitness prefers the GA layout and the '
                        'analytical model prefers the reference',
        },
        'oracle': oracle_summary([res['oracle'] for res in results],
                                 args.oracle_restarts),
        'config': vars(args),
    }
    write_sidecar(out_dir, {'experiment': 'objective_alignment',
                            'args': vars(args), 'wall_seconds': wall,
                            'workers': args.workers,
                            'base_params': {str(sc): res['base_params']
                                            for sc, res in enumerate(results)},
                            'summary': summary}, provenance=prov)
    # Written last: its presence marks a finished run.
    with open(os.path.join(out_dir, 'summary.json'), 'w',
              encoding='utf-8') as f:
        json.dump(portable_paths(summary), f, indent=2)
    for v, st in per_variant.items():
        print(f'[align] {v:12s} rho Fisher mean {st["spearman_fisher_mean"]:+.3f} '
              f'(median {st["spearman_median"]:+.3f}, '
              f'{st["spearman_min"]:+.3f}..{st["spearman_max"]:+.3f}); '
              f'reversal in {st["n_reversal"]}/{st["n_scenarios"]}')
    print(f'Artifacts in: {out_dir}  (wall {wall:.0f}s)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
