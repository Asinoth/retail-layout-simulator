"""Regenerate the experiment plot figures as vector PDF (audit R9, R56).

Re-plots the four experiment figures from their saved ``results.csv``
artifacts with the shared color-blind-safe style, so no paper-grade
experiment needs to be re-run. Each is drawn at the width it prints at
(``figstyle.PRINT_FRAC``) with no text below ``figstyle.MIN_FONT_PT``, and
``figstyle.save`` refuses one that is not. Writes both ``.pdf`` (for the
manuscript) and ``.png`` into ``../figs/``:

    regret_boxplot  (Fig. 6)    methods_bar  (Fig. 7)
    lhs_hist        (Fig. 12)   figure_c     (Fig. 13)

figure_c shows the paired differences of the UCI worked example -- the GA
minus the as-built layout (the lift) and, when the run scored them, minus
equal-budget random search and simulated annealing -- each with its
interval. It no longer histograms the per-replicate revenue levels: each
replicate is a Monte Carlo mean over many simulated months, so those
histograms showed Monte Carlo error on a 'projected revenue' axis, and two
of them 'not overlapping' said only that the error is smaller than the
lift, which the paired interval states directly.

    python -m experiments.restyle_figures
    python -m experiments.restyle_figures --figs-dir /tmp/figs
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import figstyle  # noqa: E402
# Pick artifacts with the same validators the macro generator uses, so a
# smoke run or an in-progress directory cannot redraw a paper figure while
# the macros still quote the paper-grade run.
from make_results_macros import (  # noqa: E402
    latest, cluster_boot_ci, boot_ci, figc_comparator_stats,
    FIGC_COMPARATOR_COLUMNS, _figa_big_enough, _figb_big_enough,
    _lhs_big_enough, _figc_big_enough)
from matplotlib.ticker import FuncFormatter, MaxNLocator  # noqa: E402

_THOUSANDS = FuncFormatter(lambda v, _: f"{v:,.0f}")


def _rows(d):
    with open(os.path.join(d, "results.csv"), newline="") as f:
        return list(csv.DictReader(f))


def regret_boxplot(figs=None):
    d = latest("synthetic_gt_", _figa_big_enough)
    if not d:
        return False
    rows = _rows(d)
    by = defaultdict(list)
    for r in rows:
        by[r["scenario"]].append(float(r["regret_pct"]))
    scen = sorted(by, key=lambda s: np.median(by[s]))
    data = [by[s] for s in scen]
    allreg = np.concatenate(data)
    fig, ax = figstyle.print_figure("regret_boxplot", height_in=2.5)
    bp = ax.boxplot(data, patch_artist=True, widths=0.6,
                    medianprops=dict(color=figstyle.BLACK, lw=1.0),
                    whiskerprops=dict(lw=0.7), capprops=dict(lw=0.7),
                    flierprops=dict(marker="o", ms=2.0, alpha=0.6,
                                    markerfacecolor=figstyle.GREY,
                                    markeredgecolor="none"))
    for patch in bp["boxes"]:
        patch.set(facecolor=figstyle.GA, alpha=0.55, edgecolor=figstyle.BLUE,
                  linewidth=0.7)
    ax.axhline(np.median(allreg), color=figstyle.VERMILLION, ls="--", lw=1.0,
               label=f"overall median {np.median(allreg):.2f}%")
    ax.set_xticks([])
    ax.set_xlabel(f"{len(scen)} synthetic scenarios, sorted by median regret "
                  f"({len(data[0])} GA seeds each)")
    ax.set_ylabel("GA regret vs. reference (%)")
    ax.legend(loc="upper left")
    figstyle.save(fig, "regret_boxplot", out_dir=figs)
    plt.close(fig)
    return True


def methods_bar(figs=None):
    """Paired GA-minus-method differences with the table's intervals.

    Plotting the per-method revenue LEVELS put most of the plotted
    variation into differences between scenarios, which the paired design
    removes: the same shop is laid out by every method under the same MC
    seeds. The intervals are therefore the ones the results table
    reports -- a scenario-level cluster bootstrap, Bonferroni-corrected
    over the comparator family -- so figure and table say the same thing.
    The analytical reference is hatched as well as coloured, so it stands
    apart in grey-scale print too.
    """
    d = latest("baseline_comparison_", _figb_big_enough)
    if not d:
        return False
    rows = _rows(d)
    order = ["random_valid", "perimeter_only", "popularity_rank",
             "greedy_swap", "random_search", "simulated_annealing", "oracle"]
    labels = {"random_valid": "random", "perimeter_only": "perimeter",
              "popularity_rank": "popularity", "greedy_swap": "greedy",
              "random_search": "random\nsearch",
              "simulated_annealing": "simulated\nannealing",
              "oracle": "analytical\nreference"}
    by = defaultdict(dict)          # (scenario, seed) -> {method: revenue}
    for r in rows:
        by[(r["scenario"], r["seed"])][r["method"]] = float(r["mc_revenue"])
    methods = [m for m in order
               if any("GA" in v and m in v for v in by.values())]
    if not methods:
        return False
    alpha_bonf = 0.05 / len(methods)
    means, los, his = [], [], []
    for m in methods:
        clusters = defaultdict(list)
        for (scen, _seed), v in by.items():
            if "GA" in v and m in v:
                clusters[scen].append(v["GA"] - v[m])
        mean, lo, hi = cluster_boot_ci(list(clusters.values()),
                                       alpha=alpha_bonf)
        means.append(mean); los.append(mean - lo); his.append(hi - mean)
    x = np.arange(len(methods))
    fig, ax = figstyle.print_figure("methods_bar", height_in=2.6)
    for i, m in enumerate(methods):
        ref = m == "oracle"
        ax.bar(x[i], means[i], yerr=[[los[i]], [his[i]]],
               color=figstyle.REFERENCE if ref else figstyle.BASELINE,
               hatch="///" if ref else "", alpha=0.9, capsize=3,
               edgecolor=figstyle.BLACK if ref else "white", linewidth=0.5,
               error_kw=dict(elinewidth=0.8, capthick=0.8))
    ax.axhline(0, color=figstyle.BLACK, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[m] for m in methods])
    ax.yaxis.set_major_formatter(_THOUSANDS)
    ax.set_ylabel("GA minus method, paired\nMC revenue ($)")
    ax.set_title(f"Paired differences, "
                 f"{100 * (1 - alpha_bonf):.1f}% cluster-bootstrap intervals")
    ax.grid(axis="x", visible=False)
    figstyle.save(fig, "methods_bar", out_dir=figs)
    plt.close(fig)
    return True


def lhs_hist(figs=None):
    d = latest("elasticity_lhs_", _lhs_big_enough)
    if not d:
        return False
    rows = _rows(d)
    lifts = np.array([float(r["lift30"]) for r in rows])
    fig, ax = figstyle.print_figure("lhs_hist", height_in=2.4)
    ax.hist(lifts, bins=40, color=figstyle.GA, alpha=0.8, edgecolor="white",
            linewidth=0.3)
    ax.axvline(0, color=figstyle.BLACK, lw=0.8)
    ax.axvline(np.median(lifts), color=figstyle.VERMILLION, ls="--", lw=1.1,
               label=f"median ${np.median(lifts):,.0f}")
    frac_pos = (lifts > 0).mean() * 100
    ax.xaxis.set_major_formatter(_THOUSANDS)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_xlabel("projected 30-day lift, GA minus popularity ($)")
    ax.set_ylabel("Latin-hypercube draws")
    ax.set_title(f"{frac_pos:.0f}% of {lifts.size:,} draws positive")
    ax.legend()
    figstyle.save(fig, "lhs_hist", out_dir=figs)
    plt.close(fig)
    return True


def figure_c(figs=None):
    """Paired differences of the UCI worked example, one row each: the GA
    minus the as-built layout (the lift), and minus the equal-budget random
    search and simulated annealing when the run scored them.

    Every replicate's difference is a faint point and the mean carries its
    95% bootstrap interval -- the lift's from the same bootstrap as its
    macro, each search's as the runner reports it -- so figure and text
    cannot disagree. The replicates are Monte Carlo re-evaluations of one
    layout per method, so the intervals are evaluation noise, not
    search-to-search variation. Rows differ by marker shape as well as
    colour, and each is labelled on the axis."""
    d = latest("real_data_uci_", _figc_big_enough)
    if not d:
        return False
    rows = _rows(d)
    base = np.array([float(r["baseline_revenue"]) for r in rows])
    diffs = np.array([float(r["diff"]) for r in rows])
    lo, hi = boot_ci(diffs)
    entries = [("as-built\n(the lift)", diffs, diffs.mean(), lo, hi,
                diffs.mean() / max(base.mean(), 1e-9) * 100,
                figstyle.SERIES[0])]
    # The equal-budget searches scored under the same replicates. A run
    # from before they were added has no such columns and shows the lift
    # alone.
    comps = (figc_comparator_stats(d, rows)
             if all(c in rows[0] for c in FIGC_COMPARATOR_COLUMNS) else {})
    for i, (meth, col, label) in enumerate(
            (("random_search", "diff_rs", "random\nsearch"),
             ("simulated_annealing", "diff_sa", "simulated\nannealing")),
            start=1):
        if meth in comps:
            mean, c_lo, c_hi, pct = comps[meth]
            entries.append((label, np.array([float(r[col]) for r in rows]),
                            mean, c_lo, c_hi, pct, figstyle.SERIES[i]))
    fig, ax = figstyle.print_figure("figure_c",
                                    height_in=0.55 + 0.5 * len(entries))
    rng = np.random.default_rng(0)
    ys = np.arange(len(entries))[::-1]
    for y, (_, vals, mean, c_lo, c_hi, pct, style) in zip(ys, entries):
        ax.scatter(vals, y + rng.uniform(-0.16, 0.16, vals.size), s=6,
                   color=figstyle.GREY, alpha=0.5, edgecolor="none",
                   zorder=2)
        ax.errorbar(mean, y, xerr=[[mean - c_lo], [c_hi - mean]],
                    fmt=style["marker"], ms=5, color=style["color"],
                    ecolor=style["color"], elinewidth=1.4, capsize=3,
                    zorder=3)
        # Past the rightmost replicate as well as the interval, so the
        # label never sits on a point.
        ax.annotate(f"{pct:+.2f}%", xy=(max(c_hi, vals.max()), y),
                    xytext=(5, 0), textcoords="offset points", va="center")
    ax.axvline(0, color=figstyle.BLACK, lw=0.8)
    ax.set_yticks(ys)
    ax.set_yticklabels([e[0] for e in entries])
    ax.set_ylim(-0.6, len(entries) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.12)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.xaxis.set_major_formatter(_THOUSANDS)
    ax.set_xlabel("GA minus layout, projected 30-day revenue (GBP)")
    ax.set_title(f"UCI store, paired over {len(rows)} held-out MC "
                 f"re-evaluations (95% intervals)")
    figstyle.save(fig, "figure_c", out_dir=figs)
    plt.close(fig)
    return True


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--figs-dir", type=str, default=None,
                    help="Where to write the figures (default: repo figs/)")
    args = ap.parse_args(argv)
    figstyle.apply_print()
    figs = args.figs_dir
    for fn in (regret_boxplot, methods_bar, lhs_hist, figure_c):
        # A figure whose run fails the design check is left as it is, so
        # say so rather than report it as redrawn.
        if fn(figs):
            print(f"  regenerated {fn.__name__}", flush=True)
        else:
            print(f"  skipped {fn.__name__}: no run passes the design "
                  "check", flush=True)
    print(f"wrote vector PDFs to {figs or figstyle.figs_dir()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
