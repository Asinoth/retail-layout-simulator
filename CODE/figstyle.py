"""Shared figure style for the TOMACS paper figures (audit R9).

One place for a color-blind-safe palette (Okabe & Ito 2008), consistent
typography, and vector output, so every plot reads as one family and
prints cleanly. Raster screenshots and the heat map keep PNG; every
line/bar plot goes to vector PDF.

Usage:
    import figstyle
    figstyle.apply()
    fig, ax = plt.subplots(...)
    ...
    figstyle.save(fig, "ga_convergence")   # writes ../figs/ga_convergence.{pdf,png}
"""

from __future__ import annotations

import os

import matplotlib

# Okabe-Ito color-blind-safe qualitative palette.
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILLION = "#D55E00"
PURPLE = "#CC79A7"
SKY = "#56B4E9"
YELLOW = "#F0E442"
GREY = "#999999"
BLACK = "#000000"

# Ordered cycle for multi-series plots (hue-distinct for all common CVD types).
CYCLE = [BLUE, ORANGE, GREEN, VERMILLION, PURPLE, SKY, "#666666", YELLOW]

# Semantic roles used across the paper's figures.
GA = BLUE
REFERENCE = ORANGE
BASELINE = GREY
ACCENT = VERMILLION
BAND = "#B3CDE3"        # light blue fill for CI/SE bands


def apply():
    """Set global rcParams for consistent, print-quality figures."""
    matplotlib.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,          # embed TrueType (editable, no type-3)
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.prop_cycle": matplotlib.cycler(color=CYCLE),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#DDDDDD",
        "grid.linewidth": 0.7,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 1.8,
    })


def figs_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here) if os.path.basename(here).upper() == "CODE" else here
    d = os.path.join(root, "figs")
    os.makedirs(d, exist_ok=True)
    return d


def save(fig, stem, also_png=True):
    """Save ``fig`` as vector PDF (and optionally PNG) into ../figs/."""
    d = figs_dir()
    fig.savefig(os.path.join(d, stem + ".pdf"))
    if also_png:
        fig.savefig(os.path.join(d, stem + ".png"), dpi=200)
    return os.path.join(d, stem + ".pdf")
