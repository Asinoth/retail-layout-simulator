"""Figure A: GA recovers known optima above baselines (synthetic GT).

For each of N synthetic shops, we:
  1. Solve the closed-form analytical optimum via ``oracle.solve_oracle``
     (the "true" optimum under literature-cited elasticities).
  2. Run the simulator-backed GA from K random seeds.
  3. Evaluate each GA result's *true* revenue with the same analytical
     model (NOT the GA's own objective -- regret measured on the objective
     the GA searched would be circular).
  4. Compute regret = (R_oracle - R_GA) / R_oracle.

The analytical reference is the best of ``--oracle-restarts`` (96) local
searches; each scenario's best value after 1, 2, ... restarts is recorded
(``oracle.best_curve``) together with every failed restart, so a reader can
see whether the reference had stopped improving before the regret was
taken against it.

Outputs (in ``experiments/results/synthetic_gt_<timestamp>/``):
  * results.csv        -- per-(scenario, seed) row with regret + metadata,
                          the GA layout's MC fitness (``ga_mc_fit``, the
                          selection's own estimate) and its closed-form
                          expected revenue (``ga_cf_R``)
  * regret_boxplot.png -- one box per scenario
  * spearman.csv       -- per-scenario Spearman rank correlation between
                          GA-fitness ordering and oracle-true-revenue
                          ordering on a held-out random layout sample, under
                          the Monte Carlo fitness and under its closed form
  * layouts.json       -- per scenario the reference layout (unrepaired and
                          repaired) and every seed's GA layout and trace
  * sidecar.json       -- seed, git SHA, elasticity snapshot, wall time,
                          and the summary
  * summary.json       -- aggregates, the reference's restart curves, the
                          scenarios whose reference the repair moved and
                          the closed-form check; written last

Smoke run (verify wiring):
    python -m experiments.run_synthetic_gt --n-scenarios 3 --n-seeds 2
                                            --mc-iters 200 --n-gens 8

Paper-grade run:
    python -m experiments.run_synthetic_gt --n-scenarios 30 --n-seeds 10
                                            --mc-iters 2000 --n-gens 25

``--workers N`` runs scenarios in N processes; the numeric columns are
unchanged (only wall_seconds differs between runs).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import spearmanr

# Make sibling project modules importable when run as ``python -m ...``.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from synthetic_shops import (
    generate_synthetic_shop,
    analytical_revenue,
    grid_layout_within_sections,
)
from oracle import solve_oracle
from experiments._common import (
    ORACLE_RESTARTS,
    build_headless_shop,
    base_params_for,
    anchor_base_params,
    base_params_record,
    checked_layout,
    closed_form_revenue,
    layout_json,
    combine_repair_stats,
    repair_stats,
    run_ga_headless,
    chromosome_to_layout,
    feasible_layout,
    make_run_dir,
    write_json,
    write_sidecar,
)

#: Restart counts at which the reference's best value is summarized (the
#: full curve is saved too); 24 is the count the paper-grade runs used
#: before the default became 96.
RESTART_CHECKPOINTS = (6, 12, 24, 48, 96)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--n-scenarios', type=int, default=30,
                   help="Number of synthetic shops")
    p.add_argument('--n-seeds', type=int, default=10,
                   help="GA seeds per scenario")
    p.add_argument('--n-items', type=int, default=10)
    p.add_argument('--mc-iters', type=int, default=2000)
    p.add_argument('--mc-days', type=int, default=30)
    p.add_argument('--n-gens', type=int, default=25)
    p.add_argument('--pop-size', type=int, default=30)
    p.add_argument('--n-spearman-samples', type=int, default=100,
                   help="Held-out random layouts for Spearman correlation")
    p.add_argument('--oracle-restarts', type=int, default=ORACLE_RESTARTS,
                   help="L-BFGS-B restarts of the analytical reference")
    p.add_argument('--workers', type=int, default=1,
                   help="Processes running scenarios in parallel")
    p.add_argument('--out-root', type=str,
                   default=os.path.join(_HERE, 'results'))
    args = p.parse_args()
    if args.workers < 1:
        p.error("--workers must be >= 1")
    if args.oracle_restarts < 1:
        p.error("--oracle-restarts must be >= 1")
    return args


def random_within_section_bounds(shop_synth, rng: np.random.Generator
                                  ) -> Dict[str, Tuple[float, float]]:
    """Sample a random valid layout (positions inside each item's
    section bounds, no overlap repair). Used to build the held-out
    sample for the Spearman test."""
    layout = {}
    for it in shop_synth.items:
        sec = shop_synth.sections.get(it.category)
        if sec is None:
            layout[it.name] = (rng.uniform(0.1, shop_synth.width - it.size[0] - 0.1),
                                rng.uniform(0.1, shop_synth.height - it.size[1] - 0.1))
            continue
        sx, sy, sw, sh = sec
        iw, ih = it.size
        pad = 0.05
        lo_x, hi_x = sx + pad, max(sx + pad, sx + sw - iw - pad)
        lo_y, hi_y = sy + pad, max(sy + pad, sy + sh - ih - pad)
        layout[it.name] = (float(rng.uniform(lo_x, hi_x)),
                           float(rng.uniform(lo_y, hi_y)))
    return layout


def run_one_scenario(scenario_idx: int,
                     seeds: List[int],
                     args: argparse.Namespace) -> Dict:
    """Returns a dict with per-seed regrets + Spearman rho."""
    shop_synth = generate_synthetic_shop(
        name=f"synth_{scenario_idx:03d}",
        seed=10_000 + scenario_idx,    # deterministic per-scenario
        n_items=args.n_items,
        width=12.0, height=10.0,
    )
    print(f"[scenario {scenario_idx:>3d}] "
          f"shop dims {shop_synth.width:.1f}x{shop_synth.height:.1f}, "
          f"{len(shop_synth.items)} items, "
          f"{len(shop_synth.sections)} sections",
          flush=True)

    # 1) Oracle (analytical optimum)
    t0 = time.perf_counter()
    oracle_result = solve_oracle(shop_synth, n_restarts=args.oracle_restarts)
    oracle_R = oracle_result.true_revenue
    grid_R = analytical_revenue(shop_synth, grid_layout_within_sections(shop_synth))
    print(f"  oracle R = {oracle_R:.2f}  "
          f"(grid R = {grid_R:.2f}, lift = {(oracle_R-grid_R)/grid_R*100:+.2f}%, "
          f"oracle in {time.perf_counter()-t0:.2f}s)",
          flush=True)

    # 2) Build the headless shop once per scenario. Every GA run starts
    #    from the as-built layout, not from positions left by earlier calls.
    shop = build_headless_shop(shop_synth)
    item_names = [it.name for it in shop_synth.items]
    init_layout = {n: tuple(shop.floors[1]['items'][n]['position'])
                   for n in item_names}
    # The elasticities act on score differences from the repaired as-built
    # layout, which therefore reproduces the calibrated inputs.
    base_params = anchor_base_params(shop, item_names,
                                     base_params_for(shop_synth), init_layout)

    # Regret stays measured against the unrepaired analytical optimum. Its
    # image under the GA's repair chain is recorded alongside, since that
    # is the best reference layout the GA's feasible set actually contains.
    oracle_feasible_layout = feasible_layout(shop, item_names,
                                             oracle_result.layout)
    oracle_R_feasible = analytical_revenue(shop_synth, oracle_feasible_layout)
    oracle_invariants = checked_layout(shop, item_names, oracle_feasible_layout,
                                       f"scenario {scenario_idx} reference")
    print(f"  oracle R on the GA feasible set = {oracle_R_feasible:.2f}",
          flush=True)

    # 3) Run GA across seeds
    seed_records = []
    ga_layouts = {}
    for seed in seeds:
        t1 = time.perf_counter()
        ga_out = run_ga_headless(
            shop, item_names, base_params,
            pop_size=args.pop_size,
            n_gens=args.n_gens,
            mut_rate=0.18,
            elite_frac=0.20,
            mc_iters=args.mc_iters,
            mc_days=args.mc_days,
            rng_seed=seed,
            init_layout=init_layout,
        )
        ga_layout = chromosome_to_layout(ga_out['best_chrom'], item_names)
        inv = checked_layout(shop, item_names, ga_layout,
                             f"scenario {scenario_idx} seed {seed} GA")
        ga_true_R = analytical_revenue(shop_synth, ga_layout)
        # The exact mean of the GA's own objective at its layout, beside the
        # selection's Monte Carlo estimate of it.
        ga_cf_R = closed_form_revenue(shop, item_names, ga_layout,
                                      base_params, args.mc_days)
        regret = (oracle_R - ga_true_R) / max(oracle_R, 1e-6)
        wall = time.perf_counter() - t1
        ga_layouts[str(seed)] = {'layout': layout_json(ga_layout),
                                 'invariants': inv,
                                 'trace': ga_out['trace']}
        print(f"  seed {seed:>3d}: GA true R = {ga_true_R:.2f}  "
              f"regret = {regret*100:+.3f}%  ({wall:.1f}s)",
              flush=True)
        seed_records.append({
            'scenario': scenario_idx,
            'seed': seed,
            'oracle_R': oracle_R,
            'oracle_R_feasible': oracle_R_feasible,
            'grid_R': grid_R,
            'ga_true_R': ga_true_R,
            'ga_mc_fit': ga_out['best_fit'],
            'ga_cf_R': ga_cf_R,
            'regret': regret,
            'wall_seconds': wall,
        })

    # 4) Spearman rank correlation on overlap-repaired random layouts.
    #    Layouts must be valid (no within-section overlap) so the GA
    #    fitness's huge overlap-penalty doesn't dominate the comparison.
    #    Real GA outputs avoid overlap; the Spearman sample should too.
    #    Each sample is mapped onto the GA's feasible set, and both the
    #    MC fitness and the analytical revenue are taken on that layout.
    from baselines import _repair_within_section
    from experiments._common import paired_mc_revenue
    rng = np.random.default_rng(20_000 + scenario_idx)
    sample_layouts = [
        feasible_layout(
            shop, item_names,
            _repair_within_section(shop_synth,
                                   random_within_section_bounds(shop_synth, rng)))
        for _ in range(args.n_spearman_samples)
    ]
    ga_fits = np.array([
        paired_mc_revenue(shop, item_names, lay, base_params,
                          seed=99, mc_iters=max(200, args.mc_iters // 4),
                          mc_days=args.mc_days)
        for lay in sample_layouts
    ])
    true_revs = np.array([analytical_revenue(shop_synth, lay)
                          for lay in sample_layouts])
    rho, p_val = spearmanr(ga_fits, true_revs)
    # The same correlation with the GA's objective at its exact mean: what
    # is left once the Monte Carlo noise in the fitness is taken out.
    cf_fits = np.array([closed_form_revenue(shop, item_names, lay,
                                            base_params, args.mc_days)
                        for lay in sample_layouts])
    rho_cf, p_cf = spearmanr(cf_fits, true_revs)
    print(f"  Spearman (GA-fit, true-R) over {len(sample_layouts)} "
          f"valid layouts: rho = {rho:.3f}, p = {p_val:.3g} "
          f"(closed form: rho = {rho_cf:.3f})",
          flush=True)

    return {
        'scenario': scenario_idx,
        'oracle_R': oracle_R,
        'oracle_R_feasible': oracle_R_feasible,
        'grid_R': grid_R,
        'spearman_rho': float(rho),
        'spearman_p': float(p_val),
        'spearman_rho_cf': float(rho_cf),
        'spearman_p_cf': float(p_cf),
        'seed_records': seed_records,
        'base_params': base_params_record(base_params),
        'oracle': {
            'n_restarts': oracle_result.n_restarts,
            'n_failed_restarts': oracle_result.n_failed,
            'failures': [list(f) for f in oracle_result.failures],
            'converged': bool(oracle_result.converged),
            'note': oracle_result.note,
            'best_curve': oracle_result.best_curve,
            'restart_values': oracle_result.restart_values,
            'R_unrepaired': float(oracle_R),
            'R_feasible': float(oracle_R_feasible),
            'moved_by_repair': bool(oracle_R_feasible < oracle_R),
        },
        'layouts': {
            'reference_unrepaired': layout_json(oracle_result.layout),
            'reference_feasible': layout_json(oracle_feasible_layout),
            'reference_invariants': oracle_invariants,
            'ga': ga_layouts,
        },
        'repair_stats': repair_stats(shop),
    }


def _map_scenarios(fn, n_scenarios: int, workers: int, *fn_args) -> list:
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


def write_results_csv(out_dir: str, scenario_results: List[Dict]) -> str:
    path = os.path.join(out_dir, 'results.csv')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['scenario', 'seed', 'oracle_R', 'oracle_R_feasible',
                    'grid_R', 'ga_true_R', 'ga_mc_fit', 'ga_cf_R',
                    'regret_pct', 'wall_seconds'])
        for sc in scenario_results:
            for rec in sc['seed_records']:
                w.writerow([rec['scenario'], rec['seed'],
                            f"{rec['oracle_R']:.4f}",
                            f"{rec['oracle_R_feasible']:.4f}",
                            f"{rec['grid_R']:.4f}",
                            f"{rec['ga_true_R']:.4f}",
                            f"{rec['ga_mc_fit']:.4f}",
                            f"{rec['ga_cf_R']:.4f}",
                            f"{rec['regret']*100:.6f}",
                            f"{rec['wall_seconds']:.3f}"])
    return path


def write_spearman_csv(out_dir: str, scenario_results: List[Dict]) -> str:
    path = os.path.join(out_dir, 'spearman.csv')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['scenario', 'spearman_rho', 'spearman_p',
                    'spearman_rho_cf', 'spearman_p_cf'])
        for sc in scenario_results:
            w.writerow([sc['scenario'],
                        f"{sc['spearman_rho']:.6f}",
                        f"{sc['spearman_p']:.6g}",
                        f"{sc['spearman_rho_cf']:.6f}",
                        f"{sc['spearman_p_cf']:.6g}"])
    return path


def make_boxplot(out_dir: str, scenario_results: List[Dict]) -> str:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    data = []
    labels = []
    for sc in scenario_results:
        regrets_pct = [rec['regret'] * 100 for rec in sc['seed_records']]
        data.append(regrets_pct)
        labels.append(f"S{sc['scenario']}")

    fig, ax = plt.subplots(figsize=(max(8, len(data) * 0.35), 4.5))
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('#4ECDC4')
        patch.set_alpha(0.7)
    ax.axhline(0, color='black', linewidth=0.6)
    ax.set_xlabel('Synthetic scenario')
    ax.set_ylabel('GA regret vs. oracle  (%)')
    ax.set_title('Figure A. GA recovers known optima '
                 f'(N = {len(data)} scenarios x {len(data[0])} seeds)')
    plt.setp(ax.get_xticklabels(), rotation=60, fontsize=7)
    fig.tight_layout()
    path = os.path.join(out_dir, 'regret_boxplot.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _fisher_mean(rho: np.ndarray) -> float:
    """Fisher-z mean of correlations (clipped away from +/-1)."""
    z = np.arctanh(np.clip(np.asarray(rho, dtype=float), -0.999999, 0.999999))
    return float(np.tanh(z.mean()))


def summarize(scenario_results: List[Dict], args: argparse.Namespace,
              regrets: np.ndarray, rhos: np.ndarray,
              rhos_cf: np.ndarray) -> Dict:
    """The run's summary: regret and Spearman aggregates, the analytical
    reference's convergence (its best value against the number of restarts,
    per scenario, and how much the restarts past each checkpoint added), the
    scenarios whose reference the shared repair moved, and the closed-form
    check.

    Figure A's regret is taken on the analytical objective, which involves
    no Monte Carlo at all, so no regret moves under the closed form. What
    does involve the MC fitness is the Spearman agreement: it is recomputed
    with the fitness at its exact mean, and the conclusion (a positive
    Fisher-z mean correlation) is checked under both."""
    curves = {}
    gains = {c: [] for c in RESTART_CHECKPOINTS}
    for sc in scenario_results:
        cur = sc['oracle']['best_curve']
        final = next((v for v in reversed(cur) if v is not None), None)
        at = {}
        for c in RESTART_CHECKPOINTS:
            if c <= len(cur) and cur[c - 1] is not None and final:
                at[str(c)] = cur[c - 1]
                gains[c].append((final - cur[c - 1]) / abs(final) * 100.0)
        curves[str(sc['scenario'])] = {'best_curve': cur, 'at': at,
                                       'final': final}
    moved = [sc['scenario'] for sc in scenario_results
             if sc['oracle']['moved_by_repair']]
    ga_mc = np.array([r['ga_mc_fit'] for sc in scenario_results
                      for r in sc['seed_records']])
    ga_cf = np.array([r['ga_cf_R'] for sc in scenario_results
                      for r in sc['seed_records']])
    rel = (ga_mc - ga_cf) / np.maximum(np.abs(ga_cf), 1e-9) * 100.0
    fm, fcf = _fisher_mean(rhos), _fisher_mean(rhos_cf)
    repair_totals = combine_repair_stats(sc['repair_stats']
                                         for sc in scenario_results)
    return {
        'n_scenarios': args.n_scenarios,
        'n_seeds_per_scenario': args.n_seeds,
        'regret_pct': {
            'mean': float(regrets.mean() * 100),
            'median': float(np.median(regrets) * 100),
            'p5': float(np.percentile(regrets, 5) * 100),
            'p95': float(np.percentile(regrets, 95) * 100),
            'max': float(regrets.max() * 100)},
        'spearman': {'fisher_mean': fm, 'median': float(np.median(rhos)),
                     'min': float(rhos.min()), 'max': float(rhos.max()),
                     'n_positive': int((rhos > 0).sum())},
        'oracle': {
            'n_restarts': args.oracle_restarts,
            'n_failed_restarts': int(sum(sc['oracle']['n_failed_restarts']
                                         for sc in scenario_results)),
            'failures': {str(sc['scenario']): sc['oracle']['failures']
                         for sc in scenario_results
                         if sc['oracle']['failures']},
            'n_not_converged': int(sum(not sc['oracle']['converged']
                                       for sc in scenario_results)),
            'restart_checkpoints': list(RESTART_CHECKPOINTS),
            # Per checkpoint c: how much the restarts after the c-th raised
            # the best value, as % of the final best (median and max over
            # scenarios), and in how many scenarios they raised it at all.
            'gain_after_checkpoint_pct': {
                str(c): ({'median': float(np.median(g)),
                          'max': float(np.max(g)),
                          'n_improved': int(sum(x > 1e-9 for x in g)),
                          'n_scenarios': len(g)} if g else None)
                for c, g in gains.items()},
            'curves': curves,
        },
        # Figure A's 'moved runs', as scenarios: the reference is one layout
        # per scenario, so counting runs counts each scenario once per seed.
        'reference_moved_by_repair': {
            'n_scenarios': len(moved), 'of_scenarios': len(scenario_results),
            'scenarios': moved,
            'n_runs': len(moved) * args.n_seeds,
            'definition': 'analytical revenue of the reference after the '
                          'shared repair below its unrepaired value'},
        'closed_form': {
            'spearman_fisher_mean_cf': fcf,
            'spearman_median_cf': float(np.median(rhos_cf)),
            'spearman_min_cf': float(rhos_cf.min()),
            'spearman_max_cf': float(rhos_cf.max()),
            'spearman_n_positive_cf': int((rhos_cf > 0).sum()),
            # The GA's reported fitness is the maximum of its final pool's
            # multi-seed means, so it sits above the exact mean of the layout
            # it picked; this is by how much.
            'ga_mc_fit_minus_cf_pct': {'mean': float(rel.mean()),
                                       'median': float(np.median(rel)),
                                       'min': float(rel.min()),
                                       'max': float(rel.max())},
            'regret_uses_mc': False,
            'conclusions_unchanged': {
                'regret': True,
                'spearman_sign': bool(np.sign(fm) == np.sign(fcf)),
                'all_unchanged': bool(np.sign(fm) == np.sign(fcf))},
        },
        'repair_stats': repair_totals,
    }


def main() -> int:
    args = parse_args()
    out_dir = make_run_dir(args.out_root, 'synthetic_gt')
    print(f"output dir: {out_dir}", flush=True)

    seeds = list(range(args.n_seeds))
    wall_t0 = time.perf_counter()

    scenario_results: List[Dict] = _map_scenarios(
        run_one_scenario, args.n_scenarios, args.workers, seeds, args)

    wall = time.perf_counter() - wall_t0
    print(f"\nTotal wall: {wall:.1f}s", flush=True)

    csv_path = write_results_csv(out_dir, scenario_results)
    spr_path = write_spearman_csv(out_dir, scenario_results)
    png_path = make_boxplot(out_dir, scenario_results)

    # Aggregate stats for the sidecar
    all_regrets = np.array([rec['regret']
                            for sc in scenario_results
                            for rec in sc['seed_records']])
    rhos = np.array([sc['spearman_rho'] for sc in scenario_results])
    rhos_cf = np.array([sc['spearman_rho_cf'] for sc in scenario_results])
    summary = summarize(scenario_results, args, all_regrets, rhos, rhos_cf)

    layouts_path = write_json(out_dir, 'layouts.json', {
        str(sc['scenario']): sc['layouts'] for sc in scenario_results})

    write_sidecar(out_dir, {
        'experiment': 'synthetic_gt',
        'args': vars(args),
        'wall_seconds': wall,
        'csv_path': os.path.relpath(csv_path, out_dir),
        'spearman_csv_path': os.path.relpath(spr_path, out_dir),
        'figure_path': os.path.relpath(png_path, out_dir),
        'layouts_path': os.path.relpath(layouts_path, out_dir),
        'aggregate_regret_pct': {
            'mean':   float(all_regrets.mean() * 100),
            'median': float(np.median(all_regrets) * 100),
            'p5':     float(np.percentile(all_regrets, 5) * 100),
            'p95':    float(np.percentile(all_regrets, 95) * 100),
        },
        'aggregate_spearman': {
            'mean':   float(rhos.mean()),
            'median': float(np.median(rhos)),
            'min':    float(rhos.min()),
            'max':    float(rhos.max()),
        },
        'n_scenarios': args.n_scenarios,
        'n_seeds_per_scenario': args.n_seeds,
        'workers': args.workers,
        # The inputs every layout of each scenario was scored with, anchor
        # included (lists as length, moments and a hash).
        'base_params': {str(sc['scenario']): sc['base_params']
                        for sc in scenario_results},
        'summary': summary,
    })
    # Written last: its presence marks a finished run.
    write_json(out_dir, 'summary.json', summary)

    print(f"\nRegret summary (%):  mean={all_regrets.mean()*100:.3f}  "
          f"median={np.median(all_regrets)*100:.3f}  "
          f"p5={np.percentile(all_regrets, 5)*100:.3f}  "
          f"p95={np.percentile(all_regrets, 95)*100:.3f}")
    print(f"Spearman rho:        mean={rhos.mean():.3f}  "
          f"median={np.median(rhos):.3f}  "
          f"min={rhos.min():.3f}  max={rhos.max():.3f}")
    print(f"\nArtifacts in: {out_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
