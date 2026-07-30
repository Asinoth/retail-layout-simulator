"""GA hyperparameter sensitivity + population-diversity trace (audit R4.5).

Reviewer 4 noted that the GA's operators are asserted, never justified,
and that the "plateau by generation 10" convergence is equally
consistent with genuine convergence or with premature diversity
collapse. Two artifacts answer this:

  1. A sensitivity sweep over mutation rate x population size. We report
     the relative spread of final fitness across the grid: if the GA's
     result barely moves as these knobs vary, the reported settings are
     not cherry-picked and the conclusion is robust to them.

  2. A population-diversity trace (mean per-coordinate spread, normalized
     by the shop diagonal, recorded per generation by run_ga_headless).
     A nonzero plateau -- rather than a collapse to ~0 -- shows the
     search retains exploratory spread, so the fitness plateau reflects
     convergence, not diversity loss.

Writes ../figs/ga_diversity.png and prints macro-ready summary numbers.

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
                                 base_params_for, run_ga_headless)

MUT_RATES = [0.10, 0.18, 0.30]
POP_SIZES = [20, 30, 40]
DEFAULT = (0.18, 30)


def _figs_dir():
    root = os.path.dirname(os.path.dirname(_HERE))
    d = os.path.join(root, 'figs')
    os.makedirs(d, exist_ok=True)
    return d


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--n-scenarios', type=int, default=3)
    p.add_argument('--n-seeds', type=int, default=2)
    p.add_argument('--n-items', type=int, default=10)
    p.add_argument('--n-gens', type=int, default=25)
    p.add_argument('--mc-iters', type=int, default=800)
    p.add_argument('--mc-days', type=int, default=30)
    return p.parse_args()


def main():
    args = parse_args()
    figs = _figs_dir()

    # scenario/seed -> per-setting final fitness; plus diversity traces for
    # the default setting.
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

        setting_fit: Dict[tuple, List[float]] = {}
        for mr in MUT_RATES:
            for ps in POP_SIZES:
                fits = []
                for sd in range(args.n_seeds):
                    out = run_ga_headless(
                        shop, names, bp, pop_size=ps, n_gens=args.n_gens,
                        mut_rate=mr, elite_frac=0.20,
                        mc_iters=args.mc_iters, mc_days=args.mc_days,
                        rng_seed=1000 * sc + sd)
                    fits.append(out['best_fit'])
                    if (mr, ps) == DEFAULT:
                        default_div_traces.append(
                            np.asarray(out['history_diversity'], float))
                        bh = np.maximum.accumulate(out['history_best'])
                        default_best_traces.append(bh)
                setting_fit[(mr, ps)] = fits
        means = {k: float(np.mean(v)) for k, v in setting_fit.items()}
        vals = np.array(list(means.values()))
        spread = (vals.max() - vals.min()) / max(abs(vals.mean()), 1e-9) * 100
        best = vals.max()
        default_gap = (best - means[DEFAULT]) / max(abs(best), 1e-9) * 100
        per_scen_spread.append(spread)
        per_scen_default_gap.append(default_gap)
        print(f"[sens {sc}] final-fitness spread across "
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
             label='best-so-far fitness (normalized)')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('best-so-far fitness (normalized)', color='#4E79A7')
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
    figstyle.save(fig, 'ga_diversity')
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
    }
    with open(os.path.join(figs, 'ga_sensitivity.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nHyperparameter spread (median across scenarios): "
          f"{summary['hyper_spread_median_pct']:.2f}%")
    print(f"Default setting gap below best (median): "
          f"{summary['default_gap_median_pct']:.2f}%")
    print(f"Diversity: start {summary['diversity_start']*100:.2f}% -> "
          f"end {summary['diversity_end']*100:.2f}% of diagonal "
          f"({summary['diversity_retained_pct']:.0f}% retained)")
    print(f"wall {time.perf_counter() - t0:.0f}s; wrote figs/ga_diversity.png"
          f" + figs/ga_sensitivity.json")
    return 0


if __name__ == '__main__':
    sys.exit(main())
