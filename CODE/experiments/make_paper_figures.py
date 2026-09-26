"""Regenerate the headless paper figures into ``../figs/``.

Produces, from a single fixed synthetic scenario (seed 10007) so the
figures are reproducible run-to-run:

  * ga_convergence   (Fig. 9)  -- best-so-far and population-mean fitness
                                  over generations, averaged across 5 GA
                                  seeds with a seed-range band (paired-seed
                                  MC fitness).
  * score_components (Fig. 10) -- composite-score decomposition,
                                  popularity baseline vs GA-optimized
                                  layout, per criterion.
  * mc_convergence   (Fig. 8)  -- running mean of the MC estimator with its
                                  +/-1.96 SE band.

Each is drawn at the width it prints at (``figstyle.PRINT_FRAC``), with no
text below ``figstyle.MIN_FONT_PT`` and series told apart by line style,
marker or hatching as well as hue (review R56); ``figstyle.save`` refuses
a figure that is not.

The run writes what the figures plot to ``paper_figures_data.json`` beside
them, so ``--replot`` redraws all three from that file without re-running
the GA. ``realized_scores.json`` carries the realized lifts and the MC
relative SE the macros quote, and the design it was produced at, so the
macro generator can tell a paper-grade run from a quick check.

The GUI screenshots and the emergent traffic heat map require a live Tk
display and live simulation run; they are produced by the companion
``CODE/make_gui_figures.py`` (run on a machine with a desktop).

Usage:
    python -m experiments.make_paper_figures
    python -m experiments.make_paper_figures --n-seeds 5 --mc-iters 2000
    python -m experiments.make_paper_figures --figs-dir /tmp/figs
    python -m experiments.make_paper_figures --replot
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

import figstyle  # noqa: E402  (shared color-blind-safe style)

from synthetic_shops import generate_synthetic_shop        # noqa: E402
from baselines import popularity_rank                       # noqa: E402
from dataset_paths import repo_relative                     # noqa: E402
from experiments._common import (build_headless_shop,       # noqa: E402
                                 base_params_for, anchor_base_params,
                                 base_params_record, checked_layout,
                                 chromosome_to_layout, elasticity_snapshot,
                                 hardware_record, package_versions,
                                 provenance_snapshot, run_ga_headless,
                                 with_git_state_check)

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
SCORE_LABELS = ['traffic', 'cross-merchandising', 'impulse', 'flow',
                'revenue placement', 'section compliance', 'accessibility']

#: File the plotted data is written to, beside the figures.
DATA_FILE = 'paper_figures_data.json'
#: Iteration counts the MC trace keeps (log-spaced), enough for a smooth
#: curve on a log axis without storing 20,000 points.
N_TRACE_POINTS = 400

_THOUSANDS = FuncFormatter(lambda v, _: f"{v:,.0f}")


def _figs_dir():
    root = os.path.dirname(_HERE) if os.path.basename(_HERE) != 'experiments' \
        else os.path.dirname(os.path.dirname(_HERE))
    d = os.path.join(root, 'figs')
    os.makedirs(d, exist_ok=True)
    return d


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario-seed', type=int, default=SCENARIO_SEED)
    ap.add_argument('--n-items', type=int, default=N_ITEMS)
    ap.add_argument('--n-seeds', type=int, default=5)
    ap.add_argument('--pop-size', type=int, default=30)
    ap.add_argument('--n-gens', type=int, default=25)
    ap.add_argument('--mc-iters', type=int, default=2000)
    ap.add_argument('--mc-days', type=int, default=MC_DAYS)
    ap.add_argument('--figs-dir', type=str, default=None,
                    help="Where to write the figures, "
                         f"{DATA_FILE} and realized_scores.json "
                         "(default: repo figs/)")
    ap.add_argument('--replot', action='store_true',
                    help=f"Redraw the figures from {DATA_FILE} in the "
                         "figures directory, without re-running anything")
    return ap.parse_args(argv)


# -- plotting (from the saved data only) ---------------------------------

def plot_ga_convergence(data, figs):
    g = data['ga_convergence']
    B = np.asarray(g['best_so_far'], dtype=float)
    A = np.asarray(g['pop_mean'], dtype=float)
    gens = np.arange(1, B.shape[1] + 1)
    best, mean = figstyle.SERIES[0], figstyle.SERIES[1]
    every = max(1, B.shape[1] // 8)
    fig, ax = figstyle.print_figure('ga_convergence', height_in=2.6)
    ax.fill_between(gens, B.min(0), B.max(0), color=best['color'],
                    alpha=0.18, linewidth=0, label='best-so-far, seed range')
    ax.plot(gens, B.mean(0), color=best['color'], ls=best['linestyle'],
            marker=best['marker'], markevery=every,
            label=f"best-so-far, mean of {g['n_seeds']} seeds")
    ax.plot(gens, A.mean(0), color=mean['color'], ls=mean['linestyle'],
            marker=mean['marker'], markevery=every,
            label=f"population mean, mean of {g['n_seeds']} seeds")
    ax.set_xlabel('Generation')
    ax.set_ylabel(f"Fitness: {g['mc_days']}-day MC\nmean revenue ($)")
    ax.yaxis.set_major_formatter(_THOUSANDS)
    ax.set_title(f"GA convergence (population {g['pop_size']}, "
                 f"{g['mc_iters']:,} MC iterations)")
    ax.legend(loc='lower right')
    figstyle.save(fig, 'ga_convergence', out_dir=figs)
    plt.close(fig)


def plot_score_components(data, figs):
    """Horizontal grouped bars, one row per criterion, so each criterion's
    name fits on one line at print size."""
    s = data['score_components']
    y = np.arange(len(SCORE_KEYS))[::-1]
    h = 0.38
    fig, ax = figstyle.print_figure('score_components', height_in=3.0)
    ax.barh(y + h / 2, [s['baseline'][k] for k in SCORE_KEYS], h,
            color=figstyle.BASELINE, hatch=figstyle.HATCHES[0],
            edgecolor='white', linewidth=0.4,
            label=f"popularity baseline (composite "
                  f"{s['composite_baseline']:.3f})")
    ax.barh(y - h / 2, [s['ga'][k] for k in SCORE_KEYS], h,
            color=figstyle.GA, hatch=figstyle.HATCHES[1],
            edgecolor=figstyle.BLACK, linewidth=0.4,
            label=f"GA-optimized (composite {s['composite_ga']:.3f})")
    ax.set_yticks(y)
    ax.set_yticklabels(SCORE_LABELS)
    ax.set_xlabel('Criterion score [0, 1]')
    ax.set_xlim(0, 1.0)
    ax.set_title('Composite-score decomposition by criterion')
    ax.legend(loc='upper center', bbox_to_anchor=(0.4, -0.18), ncol=1)
    ax.grid(axis='y', visible=False)
    figstyle.save(fig, 'score_components', out_dir=figs)
    plt.close(fig)


def plot_mc_convergence(data, figs):
    c = data['mc_convergence']
    m = np.asarray(c['m'], dtype=float)
    mean = np.asarray(c['run_mean'], dtype=float)
    se = np.asarray(c['se'], dtype=float)
    line, used = figstyle.SERIES[0], figstyle.SERIES[3]
    fig, ax = figstyle.print_figure('mc_convergence', height_in=2.5)
    ax.fill_between(m, mean - 1.96 * se, mean + 1.96 * se,
                    color=line['color'], alpha=0.2, linewidth=0,
                    label='±1.96 SE')
    ax.plot(m, mean, color=line['color'], ls=line['linestyle'],
            label='running mean estimate')
    ax.axvline(c['n_used'], color=used['color'], ls='--', lw=1.0,
               label=f"iterations used ({c['n_used']:,})")
    ax.set_xscale('log')
    # Plain tick labels: a 10^k label sets its exponent in type smaller
    # than the 7-pt floor.
    ax.xaxis.set_major_formatter(_THOUSANDS)
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ''))
    ax.yaxis.set_major_formatter(_THOUSANDS)
    ax.set_xlabel('Monte Carlo iterations M (log scale)')
    ax.set_ylabel(f"estimated {c['mc_days']}-day\nrevenue ($)")
    ax.set_title('Monte Carlo estimator convergence, SE ∝ 1/√M')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.25, which='both')
    figstyle.save(fig, 'mc_convergence', out_dir=figs)
    plt.close(fig)


def plot_all(data, figs):
    figstyle.apply_print()
    plot_ga_convergence(data, figs)
    plot_score_components(data, figs)
    plot_mc_convergence(data, figs)


# -- the run ---------------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)
    figs = args.figs_dir or _figs_dir()
    os.makedirs(figs, exist_ok=True)
    if args.replot:
        with open(os.path.join(figs, DATA_FILE)) as f:
            plot_all(json.load(f), figs)
        print(f'Redrew ga_convergence, score_components, mc_convergence '
              f'from {os.path.join(figs, DATA_FILE)}')
        return 0

    # The checkout as the run starts: the GA runs below take a while at
    # the paper design, and the tree may change before the scores are
    # written.
    prov = provenance_snapshot()
    wall_t0 = time.perf_counter()

    scenario = generate_synthetic_shop(name='paper_fig',
                                       seed=args.scenario_seed,
                                       n_items=args.n_items)
    shop = build_headless_shop(scenario)
    names = [it.name for it in scenario.items]
    init_layout = {n: tuple(shop.floors[1]['items'][n]['position'])
                   for n in names}
    # The elasticities act on score differences from the repaired as-built
    # layout, which therefore reproduces the calibrated inputs.
    bp = anchor_base_params(shop, names, base_params_for(scenario),
                            init_layout)

    best_curves, avg_curves, best_out = [], [], None
    for seed in range(args.n_seeds):
        out = run_ga_headless(shop, names, bp, pop_size=args.pop_size,
                              n_gens=args.n_gens, mc_iters=args.mc_iters,
                              mc_days=args.mc_days, rng_seed=seed,
                              init_layout=init_layout)
        # Best-so-far (cumulative max) so the convergence curve is monotone
        # despite paired-MC noise in per-generation best.
        best_curves.append(np.maximum.accumulate(out['history_best']))
        avg_curves.append(out['history_avg'])
        if best_out is None or out['best_fit'] > best_out['best_fit']:
            best_out = out
        # Every layout a figure draws from keeps the floor-plan invariants.
        checked_layout(shop, names,
                       chromosome_to_layout(out['best_chrom'], names),
                       f'paper figures GA seed {seed}')
        print(f'  GA seed {seed} done', flush=True)

    # -- score-component decomposition --------------------------------
    base_layout = popularity_rank(scenario, seed=0)
    base_chrom = np.array([base_layout[it.name] for it in scenario.items])
    s_bl, bkd_bl = shop._ga_compute_layout_score(base_chrom, names, bp)
    s_ga, bkd_ga = shop._ga_compute_layout_score(best_out['best_chrom'], names, bp)

    # -- Realized elasticities at realized scores ---------------------
    # Report the effect actually APPLIED at the realized composite scores,
    # not the score=1 band endpoint (which no layout reaches). The drivers
    # come from the objective's own transform (layout_objective), anchored
    # at the repaired as-built layout like the GA's fitness, so these are
    # the ratios the fitness applies between the two layouts.
    from layout_objective import ELASTICITY_MIDPOINTS, layout_drivers
    d_bl = layout_drivers(s_bl, bkd_bl, bp)
    d_ga = layout_drivers(s_ga, bkd_ga, bp)

    def _pct(key):
        return round((d_ga[key] / d_bl[key] - 1) * 100, 1)

    realized = {
        'score_baseline': round(float(s_bl), 4),
        'score_ga': round(float(s_ga), 4),
        'score_anchor': round(float(bp['score_anchor']['score']), 4),
        'conv_lift_pct': _pct('conv_lifted'),
        'impulse_lift_pct': _pct('imp_rate'),
        'basket_lift_pct': _pct('avg_bsk'),
        'elasticity_midpoints': {'conv': ELASTICITY_MIDPOINTS['conv'],
                                 'impulse': ELASTICITY_MIDPOINTS['imp'],
                                 'basket': ELASTICITY_MIDPOINTS['bsk']},
        'base_params': base_params_record(bp),
        # The objective the scores were made under -- weights, elasticity
        # bands, every objective constant, the Monte Carlo spend law -- as
        # the runners' sidecars stamp it, so the macro generator can refuse
        # a record made under an older objective.
        'elasticities': elasticity_snapshot(),
        # The design behind these numbers, so a reader (and the macro
        # builder) can tell a paper-grade run from a quick check. The
        # figures directory is the one actually written to, named
        # relative to the repository when it lies inside it, so the
        # shipped file does not carry this checkout's absolute path.
        'config': {**{k: v for k, v in vars(args).items() if k != 'replot'},
                   'figs_dir': repo_relative(figs)},
        'provenance': {**prov,
                       'python': sys.version.split()[0],
                       'packages': package_versions()},
    }

    # -- Monte Carlo convergence trace ---------------------------------
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
    n_used = min(args.mc_iters, n_big)
    # Relative SE at the iteration count the experiments run at, which is
    # what the paper quotes -- read at that index rather than at a fixed
    # 2000 so the two cannot drift apart.
    rse_used = float(se[n_used - 1] / run_mean[n_used - 1] * 100)
    realized['mc_rse_pct'] = round(rse_used, 3)
    realized['mc_rse_iters'] = int(n_used)
    realized['mc_trace_iters'] = int(n_big)
    # As a runner's sidecar records them (write_sidecar): whether the
    # checkout moved while the run was in flight -- the macro generator
    # quotes these numbers only from a run whose code did not -- and what
    # the run cost, on what machine (review R53).
    realized['provenance'] = with_git_state_check(realized['provenance'])
    realized['hardware'] = hardware_record()
    realized['wall_seconds'] = time.perf_counter() - wall_t0
    with open(os.path.join(figs, 'realized_scores.json'), 'w') as f:
        json.dump(realized, f, indent=2)

    # The trace from iteration 200 on, at log-spaced counts plus the one
    # the experiments use.
    lo_m = 200
    idx = np.unique(np.concatenate([
        np.geomspace(lo_m, n_big, N_TRACE_POINTS).astype(int), [n_used]]))
    data = {
        'config': realized['config'],
        'provenance': realized['provenance'],
        'ga_convergence': {
            'best_so_far': np.asarray(best_curves, float).tolist(),
            'pop_mean': np.asarray(avg_curves, float).tolist(),
            'n_seeds': args.n_seeds, 'pop_size': args.pop_size,
            'mc_iters': args.mc_iters, 'mc_days': args.mc_days,
        },
        'score_components': {
            'baseline': {k: float(bkd_bl.get(k, 0.0)) for k in SCORE_KEYS},
            'ga': {k: float(bkd_ga.get(k, 0.0)) for k in SCORE_KEYS},
            'composite_baseline': float(s_bl),
            'composite_ga': float(s_ga),
        },
        'mc_convergence': {
            'm': idx.tolist(),
            'run_mean': run_mean[idx - 1].tolist(),
            'se': se[idx - 1].tolist(),
            'n_used': int(n_used), 'n_big': n_big, 'mc_days': args.mc_days,
        },
    }
    with open(os.path.join(figs, DATA_FILE), 'w') as f:
        json.dump(data, f, indent=1)
    plot_all(data, figs)

    print(f'Wrote ga_convergence, score_components, mc_convergence, '
          f'{DATA_FILE}, realized_scores.json to {figs}')
    print(f'  realized: {realized}')
    print(f'  MC relative SE at {n_used} iters: {rse_used:.3f}%')
    return 0


if __name__ == '__main__':
    sys.exit(main())
