"""Regenerate the experiment plot figures as vector PDF (audit R9).

Re-plots the four experiment figures from their saved ``results.csv``
artifacts with the shared color-blind-safe style, so no paper-grade
experiment needs to be re-run. Writes both ``.pdf`` (for the manuscript)
and ``.png`` into ``../figs/``:

    regret_boxplot, methods_bar, lhs_hist, figure_c

    python -m experiments.restyle_figures
"""

from __future__ import annotations

import csv
import glob
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

_RES = os.path.join(_HERE, "results")


def _latest(prefix):
    ds = sorted(glob.glob(os.path.join(_RES, prefix + "*")))
    return ds[-1] if ds else None


def _rows(d):
    with open(os.path.join(d, "results.csv"), newline="") as f:
        return list(csv.DictReader(f))


def _boot_ci(x, n=2000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    m = np.array([rng.choice(x, x.size, replace=True).mean() for _ in range(n)])
    return float(m.mean()), float(np.percentile(m, 100 * alpha / 2)), \
        float(np.percentile(m, 100 * (1 - alpha / 2)))


def regret_boxplot():
    d = _latest("synthetic_gt_")
    if not d:
        return
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
    figstyle.save(fig, "regret_boxplot")
    plt.close(fig)


def methods_bar():
    d = _latest("baseline_comparison_")
    if not d:
        return
    rows = _rows(d)
    order = ["random_valid", "perimeter_only", "popularity_rank",
             "greedy_swap", "random_search", "simulated_annealing",
             "oracle", "GA"]
    labels = {"random_valid": "random", "perimeter_only": "perimeter",
              "popularity_rank": "popularity", "greedy_swap": "greedy",
              "random_search": "rand. search", "simulated_annealing": "sim. anneal.",
              "oracle": "analytic\\nreference", "GA": "GA"}
    vals = defaultdict(list)
    for r in rows:
        vals[r["method"]].append(float(r["mc_revenue"]))
    methods = [m for m in order if m in vals]
    means, los, his, colors = [], [], [], []
    for m in methods:
        mean, lo, hi = _boot_ci(vals[m])
        means.append(mean); los.append(mean - lo); his.append(hi - mean)
        colors.append(figstyle.GA if m == "GA"
                      else figstyle.REFERENCE if m == "oracle"
                      else figstyle.BASELINE)
    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    ax.bar(x, means, yerr=[los, his], color=colors, alpha=0.9, capsize=4,
           edgecolor="white", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[m].replace("\\n", "\n") for m in methods],
                       fontsize=9)
    ax.set_ylabel("mean MC revenue (\\$)")
    ax.set_title("Layout method comparison (paired-seed MC)")
    lo_y = min(means) * 0.985
    ax.set_ylim(lo_y, max(h + m for h, m in zip(his, means)) * 1.005)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    figstyle.save(fig, "methods_bar")
    plt.close(fig)


def lhs_hist():
    d = _latest("elasticity_lhs_")
    if not d:
        return
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
    figstyle.save(fig, "lhs_hist")
    plt.close(fig)


def figure_c():
    d = _latest("real_data_uci_")
    if not d:
        return
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
    figstyle.save(fig, "figure_c")
    plt.close(fig)


def main():
    figstyle.apply()
    for fn in (regret_boxplot, methods_bar, lhs_hist, figure_c):
        fn()
        print(f"  regenerated {fn.__name__}", flush=True)
    print("wrote vector PDFs to ../figs/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
