"""Figure A: GA recovers known optima above baselines (synthetic GT).

For each of N synthetic shops, we:
  1. Solve the closed-form analytical optimum via ``oracle.solve_oracle``
     (the "true" optimum under literature-cited elasticities).
  2. Run the simulator-backed GA from K random seeds.
  3. Evaluate each GA result's *true* revenue with the same analytical
     model (NOT the simulator's MC engine -- that would be circular).
  4. Compute regret = (R_oracle - R_GA) / R_oracle.

Outputs (in ``experiments/results/synthetic_gt_<timestamp>/``):
  * results.csv        -- per-(scenario, seed) row with regret + metadata
  * regret_boxplot.png -- one box per scenario
  * spearman.csv       -- per-scenario Spearman rank correlation between
                          GA-fitness ordering and oracle-true-revenue
                          ordering on a held-out random layout sample
  * sidecar.json       -- seed, git SHA, elasticity snapshot, wall time

Smoke run (verify wiring):
    python -m experiments.run_synthetic_gt --n-scenarios 3 --n-seeds 2
                                            --mc-iters 200 --n-gens 8

Paper-grade run:
    python -m experiments.run_synthetic_gt --n-scenarios 30 --n-seeds 10
                                            --mc-iters 2000 --n-gens 25
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
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
    build_headless_shop,
    base_params_for,
    run_ga_headless,
    chromosome_to_layout,
    make_run_dir,
    write_sidecar,
)


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
    p.add_argument('--out-root', type=str,
                   default=os.path.join(_HERE, 'results'))
    return p.parse_args()


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
    oracle_result = solve_oracle(shop_synth, n_restarts=24)
    oracle_R = oracle_result.true_revenue
    grid_R = analytical_revenue(shop_synth, grid_layout_within_sections(shop_synth))
    print(f"  oracle R = {oracle_R:.2f}  "
          f"(grid R = {grid_R:.2f}, lift = {(oracle_R-grid_R)/grid_R*100:+.2f}%, "
          f"oracle in {time.perf_counter()-t0:.2f}s)",
          flush=True)

    # 2) Build the headless shop once per scenario; the GA mutates the
    #    item positions but we always overwrite before evaluating.
    shop = build_headless_shop(shop_synth)
    item_names = [it.name for it in shop_synth.items]
    base_params = base_params_for(shop_synth)

    # 3) Run GA across seeds
    seed_records = []
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
        )
        ga_layout = chromosome_to_layout(ga_out['best_chrom'], item_names)
        ga_true_R = analytical_revenue(shop_synth, ga_layout)
        regret = (oracle_R - ga_true_R) / max(oracle_R, 1e-6)
        wall = time.perf_counter() - t1
        print(f"  seed {seed:>3d}: GA true R = {ga_true_R:.2f}  "
              f"regret = {regret*100:+.3f}%  ({wall:.1f}s)",
              flush=True)
        seed_records.append({
            'scenario': scenario_idx,
            'seed': seed,
            'oracle_R': oracle_R,
            'grid_R': grid_R,
            'ga_true_R': ga_true_R,
            'ga_mc_fit': ga_out['best_fit'],
            'regret': regret,
            'wall_seconds': wall,
        })

    # 4) Spearman rank correlation on overlap-repaired random layouts.
    #    Layouts must be valid (no within-section overlap) so the GA
    #    fitness's huge overlap-penalty doesn't dominate the comparison.
    #    Real GA outputs avoid overlap; the Spearman sample should too.
    from baselines import _repair_within_section
    from experiments._common import paired_mc_revenue
    rng = np.random.default_rng(20_000 + scenario_idx)
    sample_layouts = [
        _repair_within_section(shop_synth,
                               random_within_section_bounds(shop_synth, rng))
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
    print(f"  Spearman (GA-fit, true-R) over {len(sample_layouts)} "
          f"valid layouts: rho = {rho:.3f}, p = {p_val:.3g}",
          flush=True)

    return {
        'scenario': scenario_idx,
        'oracle_R': oracle_R,
        'grid_R': grid_R,
        'spearman_rho': float(rho),
        'spearman_p': float(p_val),
        'seed_records': seed_records,
    }


def write_results_csv(out_dir: str, scenario_results: List[Dict]) -> str:
    path = os.path.join(out_dir, 'results.csv')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['scenario', 'seed', 'oracle_R', 'grid_R',
                    'ga_true_R', 'ga_mc_fit', 'regret_pct', 'wall_seconds'])
        for sc in scenario_results:
            for rec in sc['seed_records']:
                w.writerow([rec['scenario'], rec['seed'],
                            f"{rec['oracle_R']:.4f}",
                            f"{rec['grid_R']:.4f}",
                            f"{rec['ga_true_R']:.4f}",
                            f"{rec['ga_mc_fit']:.4f}",
                            f"{rec['regret']*100:.6f}",
                            f"{rec['wall_seconds']:.3f}"])
    return path


def write_spearman_csv(out_dir: str, scenario_results: List[Dict]) -> str:
    path = os.path.join(out_dir, 'spearman.csv')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['scenario', 'spearman_rho', 'spearman_p'])
        for sc in scenario_results:
            w.writerow([sc['scenario'],
                        f"{sc['spearman_rho']:.6f}",
                        f"{sc['spearman_p']:.6g}"])
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


def main() -> int:
    args = parse_args()
    out_dir = make_run_dir(args.out_root, 'synthetic_gt')
    print(f"output dir: {out_dir}", flush=True)

    seeds = list(range(args.n_seeds))
    wall_t0 = time.perf_counter()

    scenario_results: List[Dict] = []
    for s in range(args.n_scenarios):
        sc = run_one_scenario(s, seeds, args)
        scenario_results.append(sc)

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

    write_sidecar(out_dir, {
        'experiment': 'synthetic_gt',
        'args': vars(args),
        'wall_seconds': wall,
        'csv_path': os.path.relpath(csv_path, out_dir),
        'spearman_csv_path': os.path.relpath(spr_path, out_dir),
        'figure_path': os.path.relpath(png_path, out_dir),
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
    })

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
