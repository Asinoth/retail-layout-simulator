"""Regenerate the headless paper figures into ``../figs/``.

Produces, from a single fixed synthetic scenario (seed 10007) so the
figures are reproducible run-to-run:

  * ga_convergence.png   -- best-so-far and population-mean fitness over
                            generations, averaged across 5 GA seeds with a
                            seed-range band (paired-seed MC fitness).
  * score_components.png -- composite-score decomposition, popularity
                            baseline vs GA-optimized layout, per criterion.

The GUI screenshots and the emergent traffic heat map require a live Tk
display and live simulation run; they are produced by the companion
``CODE/make_gui_figures.py`` (run on a machine with a desktop).

Usage:
    python -m experiments.make_paper_figures
    python -m experiments.make_paper_figures --n-seeds 5 --mc-iters 2000
    python -m experiments.make_paper_figures --figs-dir /tmp/figs

``realized_scores.json`` carries the design it was produced at, so the
macro generator can tell a paper-grade run from a quick check.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import figstyle  # noqa: E402  (shared color-blind-safe style)
figstyle.apply()

from synthetic_shops import generate_synthetic_shop        # noqa: E402
from baselines import popularity_rank                       # noqa: E402
from dataset_paths import repo_relative                     # noqa: E402
from experiments._common import (build_headless_shop,       # noqa: E402
                                 base_params_for, package_versions,
                                 provenance_snapshot, run_ga_headless)

# The fixed scenario the figures and realized scores are drawn from. The
# macro generator reads these to tell a run of this scenario from one of
# another, whose numbers the paper does not quote.
SCENARIO_SEED = 10007
N_ITEMS = 10
MC_DAYS = 30

# Criteria in display order, with human labels.
SCORE_KEYS = ['traffic', 'cross_merch', 'impulse', 'flow',
              'revenue_placement',
              'section_compliance', 'accessibility']
SCORE_LABELS = ['traffic', 'cross-\nmerch', 'impulse', 'flow',
                'revenue\nplacement', 'section\ncompliance',
                'access-\nibility']


def _figs_dir():
    root = os.path.dirname(_HERE) if os.path.basename(_HERE) != 'experiments' \
        else os.path.dirname(os.path.dirname(_HERE))
    d = os.path.join(root, 'figs')
    os.makedirs(d, exist_ok=True)
    return d


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario-seed', type=int, default=SCENARIO_SEED)
    ap.add_argument('--n-items', type=int, default=N_ITEMS)
    ap.add_argument('--n-seeds', type=int, default=5)
    ap.add_argument('--pop-size', type=int, default=30)
    ap.add_argument('--n-gens', type=int, default=25)
    ap.add_argument('--mc-iters', type=int, default=2000)
    ap.add_argument('--mc-days', type=int, default=MC_DAYS)
    ap.add_argument('--figs-dir', type=str, default=None,
                    help="Where to write the figures and "
                         "realized_scores.json (default: repo figs/)")
    return ap.parse_args()


def main():
    args = parse_args()
    figs = args.figs_dir or _figs_dir()
    os.makedirs(figs, exist_ok=True)
    # The checkout as the run starts: the GA runs below take a while at
    # the paper design, and the tree may change before the scores are
    # written.
    prov = provenance_snapshot()

    scenario = generate_synthetic_shop(name='paper_fig',
                                       seed=args.scenario_seed,
                                       n_items=args.n_items)
    shop = build_headless_shop(scenario)
    bp = base_params_for(scenario)
    names = [it.name for it in scenario.items]

    best_curves, avg_curves, best_out = [], [], None
    for seed in range(args.n_seeds):
        out = run_ga_headless(shop, names, bp, pop_size=args.pop_size,
                              n_gens=args.n_gens, mc_iters=args.mc_iters,
                              mc_days=args.mc_days, rng_seed=seed)
        # Best-so-far (cumulative max) so the convergence curve is monotone
        # despite paired-MC noise in per-generation best.
        best_curves.append(np.maximum.accumulate(out['history_best']))
        avg_curves.append(out['history_avg'])
        if best_out is None or out['best_fit'] > best_out['best_fit']:
            best_out = out
        print(f'  GA seed {seed} done', flush=True)

    # -- Figure: convergence ------------------------------------------
    B = np.array(best_curves)
    A = np.array(avg_curves)
    gens = np.arange(1, B.shape[1] + 1)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(gens, B.mean(0), color='#4E79A7', lw=2,
            label=f'Best-so-far fitness (mean of {args.n_seeds} seeds)')
    ax.fill_between(gens, B.min(0), B.max(0), color='#4E79A7', alpha=0.18,
                    label='Best-so-far (seed range)')
    ax.plot(gens, A.mean(0), color='#F28E2B', lw=1.6, ls='--',
            label=f'Population mean (mean of {args.n_seeds} seeds)')
    ax.set_xlabel('Generation')
    ax.set_ylabel(f'Fitness ({args.mc_days}-day MC mean revenue, $)')
    ax.set_title(f'GA convergence (paired-seed MC fitness; pop '
                 f'{args.pop_size}, {args.mc_iters} MC iters)')
    ax.legend(frameon=False, fontsize=9, loc='lower right')
    ax.grid(alpha=0.25)
    fig.tight_layout()
    figstyle.save(fig, 'ga_convergence', out_dir=figs)
    plt.close(fig)

    # -- Figure: score-component decomposition ------------------------
    base_layout = popularity_rank(scenario, seed=0)
    base_chrom = np.array([base_layout[it.name] for it in scenario.items])
    s_bl, bkd_bl = shop._ga_compute_layout_score(base_chrom, names, bp)
    s_ga, bkd_ga = shop._ga_compute_layout_score(best_out['best_chrom'], names, bp)
    x = np.arange(len(SCORE_KEYS))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    ax.bar(x - w / 2, [bkd_bl.get(k, 0) for k in SCORE_KEYS], w,
           color='#BAB0AC', label=f'Popularity baseline (composite {s_bl:.3f})')
    ax.bar(x + w / 2, [bkd_ga.get(k, 0) for k in SCORE_KEYS], w,
           color='#4E79A7', label=f'GA-optimized (composite {s_ga:.3f})')
    ax.set_xticks(x)
    ax.set_xticklabels(SCORE_LABELS, fontsize=8)
    ax.set_ylabel('Criterion score [0,1]')
    ax.set_title('Composite-score decomposition: baseline vs GA-optimized layout')
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    figstyle.save(fig, 'score_components', out_dir=figs)
    plt.close(fig)

    # -- Realized elasticities at realized scores ---------------------
    # Report the effect actually APPLIED at the realized composite scores,
    # not the score=1 band endpoint (which no layout reaches).
    from retail_literature import (
        ELASTICITY_CONV_BASE as ECB, ELASTICITY_CONV_GAIN_MAX as ECG,
        ELASTICITY_IMP_BASE as EIB, ELASTICITY_IMP_GAIN_MAX as EIG,
        ELASTICITY_BSK_BASE as EBB, ELASTICITY_BSK_GAIN_MAX as EBG)
    ce, ie, be = ECB + 0.5 * ECG, EIB + 0.5 * EIG, EBB + 0.5 * EBG
    imp_bl = bkd_bl.get('impulse', s_bl)
    imp_ga = bkd_ga.get('impulse', s_ga)
    realized = {
        'score_baseline': round(float(s_bl), 4),
        'score_ga': round(float(s_ga), 4),
        'conv_lift_pct': round(((1 + s_ga * ce) / (1 + s_bl * ce) - 1) * 100, 1),
        'impulse_lift_pct': round(((1 + imp_ga * ie) / (1 + imp_bl * ie) - 1) * 100, 1),
        'basket_lift_pct': round(((1 + s_ga * be) / (1 + s_bl * be) - 1) * 100, 1),
        'elasticity_midpoints': {'conv': ce, 'impulse': ie, 'basket': be},
        # The design behind these numbers, so a reader (and the macro
        # builder) can tell a paper-grade run from a quick check. The
        # figures directory is the one actually written to, named
        # relative to the repository when it lies inside it, so the
        # shipped file does not carry this checkout's absolute path.
        'config': {**vars(args), 'figs_dir': repo_relative(figs)},
        'provenance': {**prov,
                       'python': sys.version.split()[0],
                       'packages': package_versions()},
    }
    with open(os.path.join(figs, 'realized_scores.json'), 'w') as f:
        json.dump(realized, f, indent=2)

    # -- Figure: Monte Carlo convergence trace ------------------------
    # Running mean of the per-iteration revenue totals with a +/-1.96 SE
    # band, showing the estimate has stabilized well before the iteration
    # count the experiments use.
    from sim_calibration import mc_engine
    from retail_literature import (DEFAULT_OP_HOURS_PER_DAY,
                                   DEFAULT_WEEKEND_MULTIPLIER)
    n_big = 20000
    res = mc_engine(
        cph=bp['customers_per_hour'], conv=bp['conversion_rate'],
        rev_mean=bp['rev_per_converting_customer'], rev_std=bp['rev_std'],
        imp_rate=bp['impulse_rate'], imp_val=bp['avg_impulse_value'],
        avg_bsk=bp['avg_basket_size'], std_bsk=bp['std_basket_size'],
        observed_baskets=bp['basket_sizes_observed'],
        n_days=args.mc_days, n_iter=n_big,
        op_hours=DEFAULT_OP_HOURS_PER_DAY,
        wknd_mult=DEFAULT_WEEKEND_MULTIPLIER,
        monthly_growth=0.0, rng=np.random.default_rng(0))
    totals = np.asarray(res['totals'], dtype=float)
    m = np.arange(1, totals.size + 1)
    run_mean = np.cumsum(totals) / m
    # Sample variance (ddof=1) about the running mean at m. Summing squared
    # deviations from each iteration's own earlier running mean instead
    # biases it low; the clamp absorbs round-off at small m.
    run_var = np.maximum(np.cumsum(totals ** 2) - m * run_mean ** 2, 0.0) \
        / np.maximum(m - 1, 1)
    se = np.sqrt(run_var / m)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    lo_m, hi_m = 200, n_big
    ax.plot(m[lo_m:], run_mean[lo_m:], color='#4E79A7', lw=1.6,
            label='running mean estimate')
    ax.fill_between(m[lo_m:], (run_mean - 1.96 * se)[lo_m:],
                    (run_mean + 1.96 * se)[lo_m:], color='#4E79A7',
                    alpha=0.2, label=r'$\pm 1.96\,\mathrm{SE}$')
    n_used = min(args.mc_iters, n_big)
    ax.axvline(n_used, color='#E15759', ls='--', lw=1.2,
               label=f'iterations used ({n_used})')
    ax.set_xscale('log')
    ax.set_xlabel('Monte Carlo iterations (log scale)')
    ax.set_ylabel(f'estimated {args.mc_days}-day revenue ($)')
    ax.set_title('Monte Carlo estimator convergence ($O(1/\\sqrt{M})$)')
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, which='both')
    fig.tight_layout()
    figstyle.save(fig, 'mc_convergence', out_dir=figs)
    plt.close(fig)
    # Relative SE at the iteration count the experiments run at, which is
    # what the paper quotes -- read at that index rather than at a fixed
    # 2000 so the two cannot drift apart.
    rse_used = float(se[n_used - 1] / run_mean[n_used - 1] * 100)
    realized['mc_rse_pct'] = round(rse_used, 3)
    realized['mc_rse_iters'] = int(n_used)
    with open(os.path.join(figs, 'realized_scores.json'), 'w') as f:
        json.dump(realized, f, indent=2)

    print(f'Wrote ga_convergence.png, score_components.png, '
          f'mc_convergence.png, realized_scores.json to {figs}')
    print(f'  realized: {realized}')
    print(f'  MC relative SE at {n_used} iters: {rse_used:.3f}%')
    return 0


if __name__ == '__main__':
    sys.exit(main())
