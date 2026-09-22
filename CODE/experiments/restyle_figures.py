"""Regenerate the experiment plot figures as vector PDF (audit R9).

Re-plots the four experiment figures from their saved ``results.csv``
artifacts with the shared color-blind-safe style, so no paper-grade
experiment needs to be re-run. Writes both ``.pdf`` (for the manuscript)
and ``.png`` into ``../figs/``:

    regret_boxplot, methods_bar, lhs_hist, figure_c

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
    latest, cluster_boot_ci, _figa_big_enough, _figb_big_enough,
    _lhs_big_enough, _figc_big_enough)


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
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    bp = ax.boxplot(data, patch_artist=True, widths=0.6,
                    medianprops=dict(color=figstyle.BLACK, lw=1.2),
                    flierprops=dict(marker="o", ms=2.5, alpha=0.5,
                                    markerfacecolor=figstyle.GREY,
                                    markeredgecolor="none"))
    for patch in bp["boxes"]:
        patch.set(facecolor=figstyle.GA, alpha=0.55, edgecolor=figstyle.BLUE)
    ax.axhline(np.median(allreg), color=figstyle.VERMILLION, ls="--", lw=1.2,
               label=f"overall median {np.median(allreg):.2f}%")
    ax.set_xticks([])
    ax.set_xlabel(f"{len(scen)} synthetic scenarios "
                  f"(sorted by median regret)")
    ax.set_ylabel("GA regret vs. reference (%)")
    ax.set_title("Optimum recovery: GA regret per scenario")
    ax.legend(loc="upper left")
    fig.tight_layout()
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
    """
    d = latest("baseline_comparison_", _figb_big_enough)
    if not d:
        return False
    rows = _rows(d)
    order = ["random_valid", "perimeter_only", "popularity_rank",
             "greedy_swap", "random_search", "simulated_annealing", "oracle"]
    labels = {"random_valid": "random", "perimeter_only": "perimeter",
              "popularity_rank": "popularity", "greedy_swap": "greedy",
              "random_search": "rand. search", "simulated_annealing": "sim. anneal.",
              "oracle": "analytic\\nreference"}
    by = defaultdict(dict)          # (scenario, seed) -> {method: revenue}
    for r in rows:
        by[(r["scenario"], r["seed"])][r["method"]] = float(r["mc_revenue"])
    methods = [m for m in order
               if any("GA" in v and m in v for v in by.values())]
    if not methods:
        return False
    alpha_bonf = 0.05 / len(methods)
    means, los, his, colors = [], [], [], []
    for m in methods:
        clusters = defaultdict(list)
        for (scen, _seed), v in by.items():
            if "GA" in v and m in v:
                clusters[scen].append(v["GA"] - v[m])
        mean, lo, hi = cluster_boot_ci(list(clusters.values()),
                                       alpha=alpha_bonf)
        means.append(mean); los.append(mean - lo); his.append(hi - mean)
        colors.append(figstyle.REFERENCE if m == "oracle"
                      else figstyle.BASELINE)
    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    ax.bar(x, means, yerr=[los, his], color=colors, alpha=0.9, capsize=4,
           edgecolor="white", linewidth=0.6)
    ax.axhline(0, color=figstyle.BLACK, lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[m].replace("\\n", "\n") for m in methods],
                       fontsize=9)
    ax.set_ylabel("GA $-$ method, paired MC revenue (\\$)")
    ax.set_title(f"Layout method comparison: paired differences "
                 f"({100 * (1 - alpha_bonf):.1f}% cluster-bootstrap CIs)")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    figstyle.save(fig, "methods_bar", out_dir=figs)
    plt.close(fig)
    return True


def lhs_hist(figs=None):
    d = latest("elasticity_lhs_", _lhs_big_enough)
    if not d:
        return False
    rows = _rows(d)
    lifts = np.array([float(r["lift30"]) for r in rows])
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.hist(lifts, bins=40, color=figstyle.GA, alpha=0.8, edgecolor="white",
            linewidth=0.3)
    ax.axvline(0, color=figstyle.BLACK, lw=1.0)
    ax.axvline(np.median(lifts), color=figstyle.VERMILLION, ls="--", lw=1.3,
               label=f"median \\${np.median(lifts):,.0f}")
    frac_pos = (lifts > 0).mean() * 100
    ax.set_xlabel("projected 30-day lift, GA $-$ popularity (\\$)")
    ax.set_ylabel("Latin-hypercube draws")
    ax.set_title(f"Elasticity-band robustness "
                 f"({frac_pos:.0f}% of draws positive)")
    ax.legend()
    fig.tight_layout()
    figstyle.save(fig, "lhs_hist", out_dir=figs)
    plt.close(fig)
    return True


def figure_c(figs=None):
    d = latest("real_data_uci_", _figc_big_enough)
    if not d:
        return False
    rows = _rows(d)
    base = np.array([float(r["baseline_revenue"]) for r in rows]) / 1e6
    opt = np.array([float(r["optimized_revenue"]) for r in rows]) / 1e6
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    bins = np.linspace(min(base.min(), opt.min()), max(base.max(), opt.max()), 18)
    ax.hist(base, bins=bins, color=figstyle.BASELINE, alpha=0.75,
            label=f"naive baseline (mean {base.mean():.2f}M)", edgecolor="white")
    ax.hist(opt, bins=bins, color=figstyle.GA, alpha=0.75,
            label=f"GA-optimized (mean {opt.mean():.2f}M)", edgecolor="white")
    lift_pct = (opt.mean() - base.mean()) / base.mean() * 100
    ax.set_xlabel("projected 30-day revenue (GBP, millions)")
    ax.set_ylabel("paired-MC replicates")
    ax.set_title(f"UCI worked example: model-conditional lift "
                 f"{lift_pct:+.2f}%")
    ax.legend(loc="upper center")
    fig.tight_layout()
    figstyle.save(fig, "figure_c", out_dir=figs)
    plt.close(fig)
    return True


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--figs-dir", type=str, default=None,
                    help="Where to write the figures (default: repo figs/)")
    args = ap.parse_args(argv)
    figstyle.apply()
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
