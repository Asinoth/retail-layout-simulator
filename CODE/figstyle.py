"""Shared figure style for the TOMACS paper figures (audit R9, review R56).

One place for a color-blind-safe palette (Okabe & Ito 2008), consistent
typography, and vector output, so every plot reads as one family and
prints cleanly. Raster screenshots and the heat map keep PNG; every
line/bar plot goes to vector PDF.

Print width. A figure drawn at 7-8 inches and scaled into the 5.5-inch
text block of the ACM small format prints its 10-pt labels at 6 pt or less
(review R56 measured 3.5-6.4 pt). The paper figures are therefore drawn AT
the width they print at: ``print_figure(stem)`` makes a figure exactly
``PRINT_FRAC[stem] * TEXTWIDTH_IN`` wide, ``apply_print()`` sets the type
sizes for that scale, and ``save(..., stem)`` refuses a figure whose width
differs from its print width or whose smallest text is below
``MIN_FONT_PT``. The manuscript must include each figure at
``width=<PRINT_FRAC[stem]>\\textwidth`` so it prints at scale 1.

Series are told apart by line style and marker (``SERIES``) or hatching
(``HATCHES``) as well as by hue, so a grey-scale print or a reader with a
colour-vision deficiency loses nothing.

Usage:
    import figstyle
    figstyle.apply_print()
    fig, ax = figstyle.print_figure("ga_convergence", height_in=2.6)
    ...
    figstyle.save(fig, "ga_convergence")   # writes ../figs/ga_convergence.{pdf,png}

``apply()`` keeps the earlier screen-size style as the base the print
style starts from; the queue and heat-map figures check their print size
themselves (``ACMSMALL_TEXTWIDTH_IN``, ``MIN_PRINT_FONT_PT``).
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

#: Line series: colour, line style and marker together, so no two series
#: differ by hue alone.
SERIES = [
    dict(color=BLUE, linestyle='-', marker='o'),
    dict(color=ORANGE, linestyle='--', marker='s'),
    dict(color=GREEN, linestyle='-.', marker='^'),
    dict(color=VERMILLION, linestyle=':', marker='D'),
    dict(color=PURPLE, linestyle=(0, (5, 1, 1, 1)), marker='v'),
]

#: Bar and patch fills, in the same order as ``SERIES``.
HATCHES = ['', '///', '...', 'xxx', '\\\\\\']

#: Text block width of the ACM small format (acmart.cls, acmsmall): a
#: 6.75 in page less two 46 pt margins.
TEXTWIDTH_IN = 6.75 - 2 * 46 / 72.27

#: Smallest type the printed figures may carry.
MIN_FONT_PT = 7.0

#: The share of the text width each paper figure prints at: the value its
#: \includegraphics must use. Figure numbers are those of the manuscript
#: built from commit fe545bd.
PRINT_FRAC = {
    'layout_previews': 1.0,             # Fig. 4
    'dataset_layout_previews': 1.0,     # Fig. 5
    'regret_boxplot': 0.9,              # Fig. 6
    'methods_bar': 0.9,                 # Fig. 7
    'mc_convergence': 0.82,             # Fig. 8
    'ga_convergence': 0.78,             # Fig. 9
    'score_components': 0.85,           # Fig. 10
    'ga_diversity': 0.8,                # Fig. 11
    'lhs_hist': 0.72,                   # Fig. 12
    'figure_c': 0.95,                   # Fig. 13
    'figc_layouts': 1.0,                # Figure C's layouts (review R06)
}

#: The same two constants under the names the live figures
#: (``experiments/make_heatmap_figure.py``, ``measure_queueing.py``) read;
#: one definition, so the two print checks cannot disagree.
ACMSMALL_TEXTWIDTH_IN = TEXTWIDTH_IN
MIN_PRINT_FONT_PT = MIN_FONT_PT


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


def apply_print():
    """The paper style at print scale: type between 7.5 and 9 pt, thin
    lines, and a saved figure exactly as large as it was drawn (no tight
    bounding box, which would change the width the fonts were set for;
    constrained layout keeps the labels inside it instead)."""
    apply()
    matplotlib.rcParams.update({
        "savefig.bbox": "standard",
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.03,
        "figure.constrained_layout.w_pad": 0.03,
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.linewidth": 0.6,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "hatch.linewidth": 0.5,
        "legend.handlelength": 2.2,
    })


def print_width(stem):
    """Printed width of figure ``stem`` in inches."""
    return PRINT_FRAC[stem] * TEXTWIDTH_IN


def print_figure(stem, height_in, nrows=1, ncols=1, **kwargs):
    """``plt.subplots`` at the width figure ``stem`` prints at."""
    import matplotlib.pyplot as plt
    return plt.subplots(nrows, ncols,
                        figsize=(print_width(stem), height_in), **kwargs)


def _visible_texts(fig):
    from matplotlib.text import Text
    fig.canvas.draw()
    return [t for t in fig.findobj(Text)
            if t.get_visible() and t.get_text().strip()]


def min_font_pt(fig):
    """Size of the smallest visible, non-empty text in ``fig``, in points
    at the size the figure is drawn (tick labels included)."""
    sizes = [t.get_fontsize() for t in _visible_texts(fig)]
    return min(sizes) if sizes else float('inf')


def overflow_in(fig, tol_in=0.01):
    """How far, in inches, what the figure draws extends past its edge
    (0 when everything fits). A saved figure of fixed size cuts that part
    off.

    Two measures, the larger counts: the tight bounding box of the drawn
    artists, which covers the tick labels actually drawn (not those past
    the axis limits, which exist but are never drawn) but ignores how wide
    an axis label is; and the extent of every other text (axis labels,
    titles, legends, annotations) taken directly."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    w, h = fig.get_size_inches()
    boxes = [fig.get_tightbbox(renderer)]
    tick_labels = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                tick_labels.update((id(tick.label1), id(tick.label2)))
    to_inches = fig.dpi_scale_trans.inverted()
    for t in _visible_texts(fig):
        if id(t) not in tick_labels:
            boxes.append(t.get_window_extent(renderer).transformed(to_inches))
    over = max(max(-b.x0, -b.y0, b.x1 - w, b.y1 - h) for b in boxes)
    return over if over > tol_in else 0.0


def check_print(fig, stem):
    """Raise unless ``fig`` is exactly as wide as ``stem`` prints, all of
    its text is at least ``MIN_FONT_PT`` there, and none of it runs past
    the figure's edge. Returns the smallest font."""
    width = fig.get_size_inches()[0]
    target = print_width(stem)
    if abs(width - target) > 1e-6:
        raise ValueError(f"{stem}: drawn {width:.3f} in wide, prints at "
                         f"{target:.3f} in")
    smallest = min_font_pt(fig)
    if smallest < MIN_FONT_PT - 1e-9:
        raise ValueError(f"{stem}: smallest text {smallest:.1f} pt, below "
                         f"{MIN_FONT_PT} pt at print size")
    over = overflow_in(fig)
    if over:
        raise ValueError(f"{stem}: drawing runs {over:.2f} in past the "
                         f"figure edge")
    return smallest


def figs_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here) if os.path.basename(here).upper() == "CODE" else here
    d = os.path.join(root, "figs")
    os.makedirs(d, exist_ok=True)
    return d


def pdf_size_in(path):
    """(width, height) in inches of the first page of the PDF at ``path``,
    from its MediaBox (PDF points, 1/72 in). ``savefig.bbox = 'tight'``
    trims or grows a figure past its ``figsize``, so this is the size the
    saved file actually has -- the size it prints at when included at its
    natural size."""
    import re
    with open(path, 'rb') as f:
        data = f.read()
    m = re.search(rb'/MediaBox\s*\[\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+'
                  rb'([-\d.]+)\s*\]', data)
    if m is None:
        return None
    x0, y0, x1, y1 = (float(v) for v in m.groups())
    return (x1 - x0) / 72.0, (y1 - y0) / 72.0


def print_record(pdf_path, min_font_pt, include_width_in=None):
    """How a saved figure prints: its size, the smallest font as drawn, and
    the smallest font it prints at when included ``include_width_in``
    inches wide (default: its natural size, which keeps every font as
    drawn)."""
    size = pdf_size_in(pdf_path)
    if size is None:
        return {'saved_width_in': None, 'saved_height_in': None,
                'min_font_pt': min_font_pt}
    w, h = size
    target = include_width_in if include_width_in else w
    return {'saved_width_in': round(w, 3), 'saved_height_in': round(h, 3),
            'min_font_pt': min_font_pt,
            'include_width_in': round(target, 3),
            'min_font_pt_printed': round(min_font_pt * target / w, 2)}


def save(fig, stem, also_png=True, out_dir=None, png_dpi=200):
    """Save ``fig`` as vector PDF (and optionally PNG) into ``out_dir``,
    by default ../figs/. A figure listed in ``PRINT_FRAC`` is checked
    against its print width and the minimum type size first."""
    if stem in PRINT_FRAC:
        check_print(fig, stem)
    d = out_dir or figs_dir()
    os.makedirs(d, exist_ok=True)
    fig.savefig(os.path.join(d, stem + ".pdf"))
    if also_png:
        fig.savefig(os.path.join(d, stem + ".png"), dpi=png_dpi)
    return os.path.join(d, stem + ".pdf")
