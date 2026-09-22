"""GA hyperparameter sensitivity + population-diversity trace (audit R4.5).

Reviewer 4 noted that the GA's operators are asserted, never justified,
and that the "plateau by generation 10" convergence is equally
consistent with genuine convergence or with premature diversity
collapse. Two artifacts answer this:

  1. A sensitivity sweep over mutation rate x population size. Each run's
     returned layout is re-scored on a block of held-out MC seeds, shared
     by every setting of a scenario, and we report the relative spread of
     those scores across the grid: if the GA's result barely moves as these
     knobs vary, the reported settings are not cherry-picked and the
     conclusion is robust to them.

  2. A population-diversity trace (mean per-coordinate spread, normalized
     by the shop diagonal, recorded per generation by run_ga_headless).
     A nonzero plateau -- rather than a collapse to ~0 -- shows the
     search retains exploratory spread, so the fitness plateau reflects
     convergence, not diversity loss.

Writes ../figs/ga_diversity.{pdf,png} and ../figs/ga_sensitivity.json (both
under ``--figs-dir``), a run record with the design and the checkout it ran
from under ``--out-root``, and prints macro-ready summary numbers.

    python -m experiments.run_ga_sensitivity --n-scenarios 3 --n-seeds 2 \
        --mc-iters 800
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

import figstyle  # noqa: E402  (shared style, audit R9)
figstyle.apply()

from synthetic_shops import generate_synthetic_shop        # noqa: E402
from experiments._common import (build_headless_shop,       # noqa: E402
                                 base_params_for, run_ga_headless,
                                 layout_to_chromosome, make_run_dir,
                                 package_versions, provenance_snapshot,
                                 write_sidecar)

MUT_RATES = [0.10, 0.18, 0.30]
POP_SIZES = [20, 30, 40]
DEFAULT = (0.18, 30)

# Base of the held-out seed block the settings are re-scored on. Every GA
# run of scenario ``sc``, seed ``sd`` searches and selects under
# ``(1000*sc + sd) * 1000 + [0, n_gens + 6)``; ``parse_args`` keeps that
# range below this base so no setting is judged on a seed it optimized
# against.
HELDOUT_SEED_BASE = 100_000_000


def _figs_dir():
    root = os.path.dirname(os.path.dirname(_HERE))
    d = os.path.join(root, 'figs')
    os.makedirs(d, exist_ok=True)
    return d


def _paired_best_gain(shop, names, bp, ref_chrom, history_best, rng_seed,
                      mc_days, mc_iters):
    """Per-generation population-best fitness minus the fitness of a fixed
    reference layout scored under the same MC seed.

    run_ga_headless scores generation ``gen`` under seed
    ``rng_seed * 1000 + gen``, so raw ``history_best`` values carry a
    seed-level shift shared by every layout, and a running max over them
    ratchets up on that noise. Differencing against the reference under
    the identical seed cancels the shared shift."""
    gains = np.empty(len(history_best), dtype=np.float64)
    for gen, best in enumerate(history_best):
        np.random.seed(rng_seed * 1000 + gen)
        ref = shop._ga_fitness(ref_chrom, names, bp, mc_days, mc_iters)
        gains[gen] = best - ref
    return gains


def _heldout_fitness(shop, names, bp, chrom, sc, args):
    """Mean fitness of one chromosome over the scenario's held-out seeds.

    ``best_fit`` is the largest of the final population's multi-seed means,
    taken over the very seeds that picked it, so it carries the upward bias
    of a maximum of noisy estimates -- and that bias grows with the
    population size, one of the axes this sweep varies. Re-scoring the
    returned layout on seeds no run searched or selected under, the same
    block for every setting of a scenario so the comparison stays paired,
    measures the layouts rather than the selection."""
    vals = []
    for k in range(args.n_heldout_seeds):
        np.random.seed(HELDOUT_SEED_BASE + 1000 * sc + k)
        vals.append(shop._ga_fitness(chrom, names, bp,
                                     args.mc_days, args.mc_iters))
    return float(np.mean(vals))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--n-scenarios', type=int, default=3)
    p.add_argument('--n-seeds', type=int, default=2)
    p.add_argument('--n-items', type=int, default=10)
    p.add_argument('--n-gens', type=int, default=25)
    p.add_argument('--mc-iters', type=int, default=800)
    p.add_argument('--mc-days', type=int, default=30)
    p.add_argument('--n-heldout-seeds', type=int, default=10,
                   help="Seeds each setting's result is re-scored on, "
                        "outside every seed the GA searched under")
    p.add_argument('--figs-dir', type=str, default=None,
                   help="Directory for ga_diversity.{pdf,png} and "
                        "ga_sensitivity.json (default: repo figs/)")
    p.add_argument('--out-root', type=str,
                   default=os.path.join(_HERE, 'results'),
                   help="Parent of the run directory holding the sidecar")
    args = p.parse_args()
    if args.n_heldout_seeds < 1:
        p.error("--n-heldout-seeds must be >= 1")
    top_search_seed = ((1000 * (args.n_scenarios - 1) + args.n_seeds - 1)
                       * 1000 + args.n_gens + 6)
    if (top_search_seed >= HELDOUT_SEED_BASE
            or args.n_heldout_seeds > 1000):
        p.error("the sweep's search seeds must stay below the held-out "
                f"block at {HELDOUT_SEED_BASE} and --n-heldout-seeds must "
                "be <= 1000, so the re-scoring seeds are disjoint from "
                "every seed a run optimized against")
    return args


def main():
    args = parse_args()
    figs = args.figs_dir or _figs_dir()
    os.makedirs(figs, exist_ok=True)
    out_dir = make_run_dir(args.out_root, 'ga_sensitivity')
    prov = provenance_snapshot()

    # scenario/seed -> per-setting held-out fitness; plus diversity traces
    # for the default setting.
    per_scen_spread: List[float] = []
    per_scen_default_gap: List[float] = []
    default_div_traces: List[np.ndarray] = []
    default_best_traces: List[np.ndarray] = []

    t0 = time.perf_counter()
    for sc in range(args.n_scenarios):
        shop_synth = generate_synthetic_shop(
            name=f"sens_{sc:03d}", seed=30_000 + sc,
            n_items=args.n_items, width=12.0, height=10.0)
        shop = build_headless_shop(shop_synth)
        names = [it.name for it in shop_synth.items]
        bp = base_params_for(shop_synth)
        # Every run in the sweep starts from the as-built layout. Its
        # chromosome is the GA's generation-0 layout, used as the paired
        # reference for the convergence trace.
        init_layout = {n: tuple(shop.floors[1]['items'][n]['position'])
                       for n in names}
        init_chrom = layout_to_chromosome(init_layout, names)

        setting_fit: Dict[tuple, List[float]] = {}
        for mr in MUT_RATES:
            for ps in POP_SIZES:
                fits = []
                for sd in range(args.n_seeds):
                    out = run_ga_headless(
                        shop, names, bp, pop_size=ps, n_gens=args.n_gens,
                        mut_rate=mr, elite_frac=0.20,
                        mc_iters=args.mc_iters, mc_days=args.mc_days,
                        rng_seed=1000 * sc + sd, init_layout=init_layout)
                    fits.append(_heldout_fitness(shop, names, bp,
                                                 out['best_chrom'], sc, args))
                    if (mr, ps) == DEFAULT:
                        default_div_traces.append(
                            np.asarray(out['history_diversity'], float))
                        default_best_traces.append(_paired_best_gain(
                            shop, names, bp, init_chrom,
                            out['history_best'], 1000 * sc + sd,
                            args.mc_days, args.mc_iters))
                setting_fit[(mr, ps)] = fits
        means = {k: float(np.mean(v)) for k, v in setting_fit.items()}
        vals = np.array(list(means.values()))
        spread = (vals.max() - vals.min()) / max(abs(vals.mean()), 1e-9) * 100
        best = vals.max()
        default_gap = (best - means[DEFAULT]) / max(abs(best), 1e-9) * 100
        per_scen_spread.append(spread)
        per_scen_default_gap.append(default_gap)
        print(f"[sens {sc}] held-out-fitness spread across "
              f"{len(means)} settings = {spread:.2f}%  "
              f"(default is {default_gap:.2f}% below best setting)",
              flush=True)

    # -- Diversity + convergence figure -------------------------------
    G = min(len(t) for t in default_div_traces)
    div = np.stack([t[:G] for t in default_div_traces], 0)
    bst = np.stack([t[:G] for t in default_best_traces], 0)
    gens = np.arange(1, G + 1)
    div_m = div.mean(0)
    bst_m = bst.mean(0)
    bst_norm = (bst_m - bst_m.min()) / max(bst_m.max() - bst_m.min(), 1e-9)

    fig, ax1 = plt.subplots(figsize=(7.2, 4.2))
    ax1.plot(gens, bst_norm, color='#4E79A7', lw=2,
             label='best fitness gain over initial layout (normalized)')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('best gain over initial layout (normalized)',
                   color='#4E79A7')
    ax1.tick_params(axis='y', labelcolor='#4E79A7')
    ax2 = ax1.twinx()
    ax2.plot(gens, div_m * 100, color='#E15759', lw=2, ls='--',
             label='population diversity')
    ax2.set_ylabel('population diversity (% of shop diagonal)',
                   color='#E15759')
    ax2.tick_params(axis='y', labelcolor='#E15759')
    ax2.set_ylim(0, max(div_m.max() * 100 * 1.25, 1e-3))
    ax1.set_title('GA convergence vs population diversity '
                  f'(default {DEFAULT[0]}, pop {DEFAULT[1]}; '
                  f'{len(default_div_traces)} runs)')
    fig.tight_layout()
    figstyle.save(fig, 'ga_diversity', out_dir=figs)
    plt.close(fig)

    summary = {
        'hyper_spread_median_pct': float(np.median(per_scen_spread)),
        'hyper_spread_max_pct': float(np.max(per_scen_spread)),
        'default_gap_median_pct': float(np.median(per_scen_default_gap)),
        'diversity_start': float(div_m[0]),
        'diversity_end': float(div_m[-1]),
        'diversity_retained_pct': float(div_m[-1] / max(div_m[0], 1e-9) * 100),
        'n_settings': len(MUT_RATES) * len(POP_SIZES),
        'mut_rates': MUT_RATES, 'pop_sizes': POP_SIZES,
        # Both statistics above are computed from the held-out re-scores,
        # not from the fitness each run selected on.
        'fitness_statistic': 'heldout_mean',
        'heldout_seeds': int(args.n_heldout_seeds),
        'heldout_seed_rule': f'{HELDOUT_SEED_BASE} + 1000*scenario + k',
        # The design behind these numbers, so a reader (and the macro
        # builder) can tell a paper-grade sweep from a quick check.
        'config': vars(args),
        'provenance': {**prov, 'python': sys.version.split()[0],
                       'packages': package_versions()},
    }
    with open(os.path.join(figs, 'ga_sensitivity.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    write_sidecar(out_dir, {'experiment': 'ga_sensitivity',
                            'args': vars(args),
                            'figs_dir': figs,
                            'wall_seconds': time.perf_counter() - t0,
                            'summary': summary},
                  provenance=prov)
    print(f"\nHyperparameter spread (median across scenarios): "
          f"{summary['hyper_spread_median_pct']:.2f}%")
    print(f"Default setting gap below best (median): "
          f"{summary['default_gap_median_pct']:.2f}%")
    print(f"Diversity: start {summary['diversity_start']*100:.2f}% -> "
          f"end {summary['diversity_end']*100:.2f}% of diagonal "
          f"({summary['diversity_retained_pct']:.0f}% retained)")
    print(f"wall {time.perf_counter() - t0:.0f}s; wrote "
          f"{os.path.join(figs, 'ga_diversity.png')} + "
          f"{os.path.join(figs, 'ga_sensitivity.json')}; "
          f"run record in {out_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
