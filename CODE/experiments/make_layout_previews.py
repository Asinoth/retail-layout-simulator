"""Regenerate the floor-plan preview figures into ``../figs/``.

  * layout_previews.png          -- the three archetypes at two floor
                                    sizes, from the architecture engine.
  * dataset_layout_previews.png  -- the same engine driven by a dataset
                                    calibration (UCI Online Retail II,
                                    Omnichannel).

Both figures are drawn from the engine at fixed seeds, so what the paper
shows is what the shipped code produces, and every panel title reports
the counts the engine actually placed rather than a remembered number.

The dataset panels need the UCI workbook and the Omnichannel bundle, found
by ``dataset_paths`` (under ``DATASETS/``, or where ``UCI_RETAIL_XLSX`` /
``OMNICHANNEL_DIR`` point). They are skipped, with a message, when those
are absent, so this still runs on a checkout without the data.

    python -m experiments.make_layout_previews
    python -m experiments.make_layout_previews --figs-dir /tmp/figs
"""

from __future__ import annotations

import argparse
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.patches import Polygon, Rectangle    # noqa: E402

import figstyle                                      # noqa: E402

import dataset_adapters as DA                        # noqa: E402
import dataset_calibration as DC                     # noqa: E402
import dataset_paths                                 # noqa: E402
from dataset_layout import build_layout_from_calibration  # noqa: E402
from shop_architecture import (generate_architecture,     # noqa: E402
                               scale_catalog)
from experiments._common import HeadlessShop         # noqa: E402

# One shop type per archetype, at the small and large floor sizes the
# size-scaling heuristics cover.
ARCHETYPE_SHOPS = ('Grocery Store', 'Electronics Store', 'Clothing Store')
FLOOR_SIZES = ((18.0, 13.0), (32.0, 22.0))
# Fixed per-panel seeds: the Generate path is random by design, so the
# figure pins a draw instead of showing whichever one came out last.
PREVIEW_SEED = 4207

# Rows of the most recent UCI sheet the preview calibrates on -- the same
# sheet and sample the live diagnostics use. The floor plan depends only on
# how many categories there are and how many products each one carries,
# and this sample gives the same counts, and so the same plan, as the
# whole sheet or as both sheets together (the Dataset button's default),
# at a fraction of the read time. Which products sit on the fixtures does
# differ, but the preview does not label them. Pass 0 to use all of it.
UCI_SAMPLE_ROWS = 60000

# Plan colors, chosen for what they denote rather than from the plot
# palette: these are floor plans, not data series.
FIXTURE = '#6B7160'        # shelving / display fixtures
BOUNDARY = '#1A1A1A'       # perimeter walls
CHECKOUT = '#F2A93B'       # checkout lanes
RESTROOM = '#8CC3E8'       # restroom
DOOR = '#D62246'           # entrance marker


def _figs_dir():
    root = os.path.dirname(os.path.dirname(_HERE))
    d = os.path.join(root, 'figs')
    os.makedirs(d, exist_ok=True)
    return d


def _draw_plan(ax, walls, items, door_position, width, height, title):
    """Draw one floor plan: department zones, fixtures, checkout, WC."""
    ax.set_xlim(-0.8, width + 0.8)
    ax.set_ylim(-0.8, height + 0.8)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=8)

    # Department zones first, as tinted backgrounds.
    for nm, w in walls.items():
        if not nm.startswith('Section_'):
            continue
        (x, y), (ww, hh) = w['position'], w['size']
        ax.add_patch(Rectangle((x, y), ww, hh,
                               facecolor=w.get('color', figstyle.GREY),
                               alpha=0.14, edgecolor=w.get('color',
                                                           figstyle.GREY),
                               lw=0.4, ls=':', zorder=1))

    for nm, w in walls.items():
        (x, y), (ww, hh) = w['position'], w['size']
        if nm.startswith('Section_'):
            continue
        if nm.startswith('Checkout'):
            ax.add_patch(Rectangle((x, y), ww, hh, facecolor=CHECKOUT,
                                   edgecolor=BOUNDARY, lw=0.5, zorder=4))
        elif nm == 'WC':
            ax.add_patch(Rectangle((x, y), ww, hh, facecolor=RESTROOM,
                                   edgecolor=BOUNDARY, lw=0.5, zorder=4))
            ax.text(x + ww / 2, y + hh / 2, 'WC', ha='center', va='center',
                    fontsize=6, zorder=5)
        else:
            ax.add_patch(Rectangle((x, y), ww, hh, facecolor=BOUNDARY,
                                   edgecolor='none', zorder=3))

    for it in items.values():
        (x, y), (ww, hh) = it['position'], it['size']
        ax.add_patch(Rectangle((x, y), ww, hh, facecolor=FIXTURE,
                               edgecolor='white', lw=0.35, zorder=2))

    if door_position:
        dx, dy = door_position
        s = max(width, height) * 0.02
        ax.add_patch(Polygon([(dx - s, dy - 1.6 * s), (dx + s, dy - 1.6 * s),
                              (dx, dy)], closed=True, facecolor=DOOR,
                             edgecolor='none', zorder=5))


def archetype_previews(figs):
    """The three archetypes at two floor sizes, one panel each."""
    fig, axes = plt.subplots(len(FLOOR_SIZES), len(ARCHETYPE_SHOPS),
                             figsize=(16.0, 9.0))
    for r, (W, H) in enumerate(FLOOR_SIZES):
        for c, shop_type in enumerate(ARCHETYPE_SHOPS):
            sections = scale_catalog(shop_type, W, H)
            rng = random.Random(PREVIEW_SEED + 100 * r + c)
            plan = generate_architecture(W, H, shop_type, sections, rng=rng)
            lanes = sum(1 for nm in plan['walls']
                        if nm.startswith('Checkout'))
            title = (f"{shop_type} [{plan['archetype']}] {W:.0f}x{H:.0f}m - "
                     f"{len(sections)} sections, {len(plan['items'])} items, "
                     f"{lanes} lanes")
            _draw_plan(axes[r][c], plan['walls'], plan['items'],
                       plan['door_position'], W, H, title)
    fig.suptitle('Size-scaled generated layouts - small (top) vs large '
                 '(bottom); checkout banks in gold', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95], h_pad=2.5)
    # The paper includes these previews as raster panels; the vector copy
    # of a dense floor plan buys nothing and is several MB.
    out = os.path.join(figs, 'layout_previews.png')
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def _uci_params(sample_rows):
    try:
        path = dataset_paths.uci_workbook()
    except FileNotFoundError:
        return None
    sheet = DA.list_excel_sheets(path)[-1][0]
    df, _ = DA.read_excel_sheets(path, [sheet])
    if sample_rows and len(df) > sample_rows:
        df = df.sample(n=sample_rows, random_state=0).reset_index(drop=True)
    normalized, report = DA.OnlineRetailIIAdapter().adapt(df)
    return DC.calibrate_transactional(
        normalized, assumed_conversion_rate=0.30,
        currency=report.extra.get('currency', 'GBP'))


def _omnichannel_params():
    try:
        path = dataset_paths.omnichannel_dir()
    except FileNotFoundError:
        return None
    families, arrivals, _ = DA.load_omnichannel_bundle(path)
    return DC.calibrate_omnichannel(families, arrivals)


def dataset_previews(figs, sample_rows=UCI_SAMPLE_ROWS):
    """Dataset-calibrated stores, or None when the data is not present."""
    panels = []
    for label, params in (('UCI Online Retail II', _uci_params(sample_rows)),
                          ('Omnichannel Retail', _omnichannel_params())):
        if params is None:
            print(f'  [previews] {label} data not found (DATASETS/ or its '
                  'environment override) - panel skipped', flush=True)
            continue
        shop = HeadlessShop(width=20.0, height=15.0)
        stats = build_layout_from_calibration(shop, params)
        panels.append((label, shop, stats))
    if not panels:
        return None

    fig, axes = plt.subplots(1, len(panels), figsize=(9.5 * len(panels), 7.0),
                             squeeze=False)
    for ax, (label, shop, stats) in zip(axes[0], panels):
        f1 = shop.floors[1]
        lanes = sum(1 for nm in f1['walls'] if nm.startswith('Checkout'))
        title = (f"{label} - derived {shop.width:.0f}x{shop.height:.0f}m, "
                 f"{stats['sections']} departments, {stats['zones']} zones, "
                 f"{stats['items']} items, {lanes} lanes")
        _draw_plan(ax, f1['walls'], f1['items'], f1['door_position'],
                   shop.width, shop.height, title)
    fig.suptitle('Imported-dataset layouts on the architecture engine '
                 '(dimensions derived from assortment size)', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(figs, 'dataset_layout_previews.png')
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--figs-dir', type=str, default=None,
                    help="Where to write the previews (default: repo figs/)")
    ap.add_argument('--sample-rows', type=int, default=UCI_SAMPLE_ROWS,
                    help="Rows sampled from the UCI sheet for the dataset "
                         "panel (0 = the whole sheet)")
    ap.add_argument('--skip-datasets', action='store_true',
                    help="Only redraw the archetype previews")
    args = ap.parse_args(argv)

    figstyle.apply()
    figs = args.figs_dir or _figs_dir()
    os.makedirs(figs, exist_ok=True)

    written = [archetype_previews(figs)]
    if not args.skip_datasets:
        out = dataset_previews(figs, args.sample_rows)
        if out:
            written.append(out)
    for p in written:
        print(f'wrote {p}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
