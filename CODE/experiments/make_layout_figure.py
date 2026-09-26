"""Figure C's store before and after the GA (review R06), display-free.

Draws the calibrated UCI store of the Figure C run twice, side by side at
print width: the repaired as-built layout the searches start from and the
GA's optimized layout, both read from the committed run's ``layouts.json``.
Department zones (each item's GA zone) are tinted, fixtures are drawn at
their recorded positions, so the aisles are the untinted gaps between them;
checkout lanes, the restroom and the entrance are marked, with a 10 m scale
bar.

What the GA changes is which PRODUCT stands in which slot, so every fixture
is shaded by its product's popularity rank -- the number of invoices that
contain it, the calibration's ``popular_items`` the GA's flow criterion
reads -- in classes: the flow criterion's own products (the
``FLOW_TOUR_ITEMS`` most bought), the rest of the top quarter, the second
quarter and the bottom half. The same shades reappear permuted within the
fixture runs in the optimized panel. The flow criterion scores the walk
entrance -> those products in rank order -> checkout; that tour is drawn in
both panels with its stops numbered and its length in each panel's title.
The tour is computed here from its definition in ``viz_ga`` and checked
against the flow score the GA itself gives each drawn layout, so the lines
are the criterion, not a sketch of it.

Fixtures that moved at least ``MOVE_HIGHLIGHT_M`` (the minimum aisle,
1.6 m) are outlined in both panels. Nearly every fixture moves a little
(the GA perturbs every coordinate), so a smaller threshold would outline
the whole store. A move that long is not a move across an aisle: on this
store most of them exchange slots within one run of shelving. The record
therefore counts, among them, the slot exchanges -- a fixture whose centre
lands within ``SLOT_TOL_M`` of another fixture's as-built centre -- and the
moves along the fixture's own run -- displaced across its long axis by
less than ``ALONG_RUN_TOL_M``.

The store is rebuilt exactly as Figure C built it -- the run's recorded
store options through ``run_real_data_example.build_store`` -- and checked
against the run: the same fixtures, and a repaired as-built layout equal to
the recorded one. Both drawn layouts are then checked against the floor-plan
engine's invariants (``_common.checked_layout``: inside their zones, no
overlap, the as-built store's aisles kept at ``MIN_AISLE`` between zones it
keeps apart, every fixture's access point reachable from the door) and the
record -- the smallest kept-apart clearance and every violation count --
goes into ``figs/figc_layouts.json`` beside the counts the caption quotes,
the tour lengths, the run it was drawn from and the checkout it was drawn
with.

    python -m experiments.make_layout_figure
    python -m experiments.make_layout_figure --figc-dir experiments/results/real_data_uci_20260926-111819

Reads the UCI workbook once (both sheets, Figure C's default); no search and
no simulation.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Polygon, Rectangle  # noqa: E402

import figstyle  # noqa: E402
from dataset_paths import uci_workbook  # noqa: E402
from experiments._common import (checked_layout, layout_score,  # noqa: E402
                                 portable_paths, provenance_snapshot,
                                 with_git_state_check)
from experiments._feasibility import MIN_AISLE  # noqa: E402
from experiments.run_real_data_example import (build_store,  # noqa: E402
                                               calibrate_from_file)
from sim_calibration import _cval  # noqa: E402

STEM = 'figc_layouts'

#: A fixture that moved at least this far is outlined: the minimum aisle.
#: On this store such a move is mostly an exchange of slots within one run
#: of shelving, not a move across an aisle (see SLOT_TOL_M, ALONG_RUN_TOL_M).
MOVE_HIGHLIGHT_M = float(MIN_AISLE)
#: A fixture that moved less than this counts as not moved at all.
MOVE_TOL_M = 0.01
#: A moved fixture whose centre lands within this distance of another
#: fixture's as-built centre has taken that fixture's slot: under a fifth
#: of the minimum aisle and under half the smallest fixture side (0.7 m).
SLOT_TOL_M = 0.3
#: A moved fixture displaced across its long axis by less than this stayed
#: in its own run of shelving: half the depth of a gondola (1.0 m).
ALONG_RUN_TOL_M = 0.5

#: How many of the most-bought products the GA's flow criterion routes
#: through (``viz_ga._ga_compute_layout_score``, flow efficiency: the
#: entrance, the five most popular products in rank order, the checkout).
#: The drawn tour is checked against the GA's own flow score, so a change
#: there cannot leave this figure drawing a different tour.
FLOW_TOUR_ITEMS = 5

#: The scale bar's length, metres.
SCALE_BAR_M = 10.0

#: The Figure C runs this reads by default (the newest one).
FIGC_PREFIX = 'real_data_uci_'

# Plan colours, as in make_layout_previews: what they denote, not a series.
BOUNDARY = '#1A1A1A'
CHECKOUT = '#F2A93B'
RESTROOM = '#8CC3E8'
DOOR = '#D62246'
CHECKOUT_HATCH = '/////'
RESTROOM_HATCH = '....'
ZONE = figstyle.BLUE
FIXTURE_EDGE = '#FFFFFF'
#: Popularity classes, most bought first: the flow tour's products, then
#: three greys from dark to light.
RANK_COLOURS = (figstyle.VERMILLION, '#3D3D3D', '#8C8C8C', '#CDCDCD')
#: Fixtures without a popularity rank (no calibration).
UNRANKED = '#8C8C8C'
MOVED_EDGE = figstyle.BLUE
TOUR = '#000000'


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


def default_figc_dir(results: str) -> Optional[str]:
    """The newest Figure C run under ``results`` that saved its layouts."""
    for d in sorted(glob.glob(os.path.join(results, FIGC_PREFIX + '*')),
                    reverse=True):
        if (os.path.isfile(os.path.join(d, 'layouts.json'))
                and os.path.isfile(os.path.join(d, 'sidecar.json'))):
            return d
    return None


def rebuild_store(figc_dir: str, retail_path: Optional[str]):
    """Figure C's store from the run's own recorded store options, checked
    against the run: the same fixtures and the same repaired as-built
    layout. Returns ``(store, layouts, side)``."""
    side = json.load(open(os.path.join(figc_dir, 'sidecar.json')))
    layouts = json.load(open(os.path.join(figc_dir, 'layouts.json')))
    a = side['args']
    args = argparse.Namespace(
        retail_path=uci_workbook(retail_path), sheets=a['sheets'],
        max_items_per_category=int(a['max_items_per_category']),
        assumed_conversion=float(a['assumed_conversion']),
        exclude_anonymous=bool(a.get('exclude_anonymous', False)))
    params, _report, _reader = calibrate_from_file(args)
    store = build_store(params, args.max_items_per_category, verbose=False)
    if set(store.item_names) != set(layouts['baseline']):
        raise RuntimeError("the rebuilt store does not place the run's "
                           "fixtures")
    worst = max(max(abs(store.baseline_layout[n][0] - xy[0]),
                    abs(store.baseline_layout[n][1] - xy[1]))
                for n, xy in layouts['baseline'].items())
    if worst > 1e-9:
        raise RuntimeError(f"the rebuilt as-built layout differs from the "
                           f"run's by up to {worst:.3g} m")
    return store, layouts, side


def _centre(xy: Sequence[float], size: Sequence[float]) -> np.ndarray:
    return np.array([float(xy[0]) + float(size[0]) / 2,
                     float(xy[1]) + float(size[1]) / 2])


def moves(baseline: Mapping[str, Sequence[float]],
          optimized: Mapping[str, Sequence[float]]) -> Dict[str, float]:
    """How far each fixture moved between the two layouts, in metres."""
    return {n: float(np.hypot(optimized[n][0] - baseline[n][0],
                              optimized[n][1] - baseline[n][1]))
            for n in baseline}


def move_kinds(baseline: Mapping[str, Sequence[float]],
               optimized: Mapping[str, Sequence[float]],
               sizes: Mapping[str, Sequence[float]],
               threshold: float = MOVE_HIGHLIGHT_M
               ) -> Dict[str, Dict[str, bool]]:
    """What each move of at least ``threshold`` is: a slot exchange (its
    centre lands within ``SLOT_TOL_M`` of another fixture's as-built
    centre) and/or a move along its own run (displaced across its long
    axis -- the longer side of ``sizes[n]`` -- by less than
    ``ALONG_RUN_TOL_M``). Keyed by fixture, moved fixtures only."""
    names = list(baseline)
    before = {n: _centre(baseline[n], sizes[n]) for n in names}
    out = {}
    for n, dist in moves(baseline, optimized).items():
        if dist < threshold:
            continue
        after = _centre(optimized[n], sizes[n])
        nearest = min((float(np.hypot(*(after - before[m])))
                       for m in names if m != n), default=math.inf)
        w, h = float(sizes[n][0]), float(sizes[n][1])
        shift = after - before[n]
        across = abs(shift[0]) if h >= w else abs(shift[1])
        out[n] = {'slot_exchange': bool(nearest <= SLOT_TOL_M),
                  'along_run': bool(across < ALONG_RUN_TOL_M)}
    return out


def move_stats(baseline: Mapping[str, Sequence[float]],
               optimized: Mapping[str, Sequence[float]],
               sizes: Mapping[str, Sequence[float]]) -> Dict[str, Any]:
    """The moves the caption quotes: how many fixtures moved at all, how
    many by at least the minimum aisle, and of those how many exchanged
    slots and how many stayed in their own run; the distances' median,
    90th percentile and largest."""
    dist = moves(baseline, optimized)
    d = np.array(list(dist.values()), dtype=float)
    kinds = move_kinds(baseline, optimized, sizes)
    return {'n_fixtures': int(d.size),
            'n_moved': int((d >= MOVE_TOL_M).sum()),
            'n_moved_at_least_aisle': int((d >= MOVE_HIGHLIGHT_M).sum()),
            'n_slot_exchange': int(sum(k['slot_exchange']
                                       for k in kinds.values())),
            'n_along_run': int(sum(k['along_run'] for k in kinds.values())),
            'n_slot_exchange_along_run': int(sum(
                k['slot_exchange'] and k['along_run']
                for k in kinds.values())),
            'move_tol_m': MOVE_TOL_M,
            'highlight_m': MOVE_HIGHLIGHT_M,
            'slot_tol_m': SLOT_TOL_M,
            'along_run_tol_m': ALONG_RUN_TOL_M,
            'median_m': float(np.median(d)),
            'p90_m': float(np.percentile(d, 90)),
            'max_m': float(d.max())}


def popularity_order(popularity: Mapping[str, float]) -> List[str]:
    """Products most bought first, ties in the calibration's own order --
    the sort the GA's flow criterion makes."""
    return [n for n, _ in sorted(popularity.items(), key=lambda kv: kv[1],
                                 reverse=True)]


def rank_classes(n_items: int, top: int = FLOW_TOUR_ITEMS) -> List[int]:
    """Upper rank of each shading class: the flow tour's products, the rest
    of the top quarter, the second quarter, the bottom half."""
    edges = [top, max(top, n_items // 4), max(top, n_items // 2), n_items]
    out = []
    for e in edges:
        if not out or e > out[-1]:
            out.append(e)
    return out


def item_ranks(popularity: Mapping[str, float],
               names: Sequence[str]) -> Dict[str, int]:
    """Each placed fixture's popularity rank (1 = most bought) among the
    placed fixtures."""
    placed = set(names)
    order = [n for n in popularity_order(popularity) if n in placed]
    return {n: i + 1 for i, n in enumerate(order)}


def flow_tour(shop, layout: Mapping[str, Sequence[float]],
              names: Sequence[str]) -> Dict[str, Any]:
    """The walk the GA's flow criterion scores on ``layout``: the entrance,
    the ``FLOW_TOUR_ITEMS`` most-bought products in rank order, the
    checkout, straight lines between them (one floor), exactly as
    ``viz_ga._ga_compute_layout_score`` builds it. Returns the points, the
    products and the length, and the flow score that length implies,
    ``1 - length / (diagonal x points)`` floored at 0."""
    A = shop.customer_simulation.analytics
    items = shop.floors[1]['items']
    placed = set(names)
    # A calibration name resolves to the fixture itself or to the first
    # fixture stocking it, as viz_ga's key_for_name does.
    by_source = {}
    for n in names:
        by_source.setdefault(items[n].get('source_name', n), n)
    keys = []
    for name in popularity_order(_cval(A, 'popular_items', {}) or {}
                                 )[:FLOW_TOUR_ITEMS]:
        k = name if name in placed else by_source.get(name)
        if k is not None:
            keys.append(k)
    entrance, _ = shop._ga_get_entrance_center()
    checkout, _ = shop._ga_find_special_center('Checkout')
    if entrance is None or checkout is None:
        return {'items': keys, 'points': [], 'length_m': None,
                'flow_score': 0.0}
    pts = ([tuple(map(float, entrance))]
           + [tuple(_centre(layout[k], items[k]['size'])) for k in keys]
           + [tuple(map(float, checkout))])
    length = float(sum(math.dist(pts[i], pts[i + 1])
                       for i in range(len(pts) - 1)))
    diag = math.hypot(float(shop.width), float(shop.height))
    return {'items': keys, 'points': pts, 'length_m': length,
            'flow_score': max(0.0, 1.0 - length / max(diag * len(pts),
                                                      1e-6))}


def checked_tour(shop, layout: Mapping[str, Sequence[float]],
                 names: Sequence[str], tol: float = 1e-9) -> Dict[str, Any]:
    """``flow_tour`` of ``layout``, refused unless its flow score is the one
    the GA's own score gives the layout (to ``tol``)."""
    tour = flow_tour(shop, layout, names)
    _score, breakdown = layout_score(shop, list(names), dict(layout))
    ga_flow = float(breakdown['flow'])
    if abs(tour['flow_score'] - ga_flow) > tol:
        raise RuntimeError(f"the drawn flow tour scores "
                           f"{tour['flow_score']:.12f}, the GA's flow "
                           f"criterion {ga_flow:.12f}")
    tour['ga_flow_score'] = ga_flow
    return tour


def _rank_colour(rank: Optional[int], edges: Sequence[int]) -> str:
    if rank is None:
        return UNRANKED
    for colour, edge in zip(RANK_COLOURS, edges):
        if rank <= edge:
            return colour
    return RANK_COLOURS[-1]


def _scale_bar(ax, x0: float, y0: float, length: float = SCALE_BAR_M):
    ax.plot([x0, x0 + length], [y0, y0], color=BOUNDARY, lw=1.0,
            solid_capstyle='butt', zorder=6)
    for x in (x0, x0 + length):
        ax.plot([x, x], [y0 - 0.35, y0 + 0.35], color=BOUNDARY, lw=0.6,
                zorder=6)
    ax.text(x0 + length / 2, y0 + 0.5, f'{length:g} m', ha='center',
            va='bottom', fontsize=7.5, zorder=6)


def _draw(ax, shop, layout, title, highlight=(), ranks=None, edges=(),
          tour=None, scale_bar=False):
    f1 = shop.floors[1]
    w, h = float(shop.width), float(shop.height)
    ax.set_xlim(-0.8, w + 0.8)
    ax.set_ylim(-0.8, h + 0.8)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=8)
    for nm, wall in f1['walls'].items():
        (x, y), (ww, hh) = wall['position'], wall['size']
        if nm.startswith('Section_'):
            ax.add_patch(Rectangle((x, y), ww, hh, facecolor=ZONE,
                                   alpha=0.10, edgecolor=ZONE, lw=0.4,
                                   ls=':', zorder=1))
        elif nm.startswith('Checkout'):
            ax.add_patch(Rectangle((x, y), ww, hh, facecolor=CHECKOUT,
                                   hatch=CHECKOUT_HATCH, edgecolor=BOUNDARY,
                                   lw=0.4, zorder=4))
        elif nm == 'WC':
            ax.add_patch(Rectangle((x, y), ww, hh, facecolor=RESTROOM,
                                   hatch=RESTROOM_HATCH, edgecolor=BOUNDARY,
                                   lw=0.4, zorder=4))
        else:
            ax.add_patch(Rectangle((x, y), ww, hh, facecolor=BOUNDARY,
                                   edgecolor='none', zorder=3))
    hl = set(highlight)
    ranks = ranks or {}
    for n, data in f1['items'].items():
        (x, y), (ww, hh) = layout[n], data['size']
        ax.add_patch(Rectangle((x, y), ww, hh,
                               facecolor=_rank_colour(ranks.get(n), edges),
                               edgecolor=FIXTURE_EDGE, lw=0.25, zorder=2))
        if n in hl:
            ax.add_patch(Rectangle((x, y), ww, hh, facecolor='none',
                                   edgecolor=MOVED_EDGE, lw=0.8, zorder=5))
    if tour and tour.get('points'):
        xs, ys = zip(*tour['points'])
        ax.plot(xs, ys, color=TOUR, lw=0.7, ls=(0, (3, 1.5)), zorder=7)
        halo = [pe.withStroke(linewidth=1.8, foreground='white')]
        for i, (px, py) in enumerate(tour['points'][1:-1], start=1):
            ax.text(px, py, str(i), ha='center', va='center',
                    fontsize=7.5, fontweight='bold', color=TOUR,
                    path_effects=halo, zorder=8)
    door = f1.get('door_position') or getattr(shop, 'door_position', None)
    if door:
        dx, dy = door
        s = max(w, h) * 0.03
        ax.add_patch(Polygon([(dx - s, dy - 1.6 * s), (dx + s, dy - 1.6 * s),
                              (dx, dy)], closed=True, facecolor=DOOR,
                             edgecolor='none', zorder=6))
    if scale_bar:
        _scale_bar(ax, 1.0, 0.9)


def _rank_label(lo: int, hi: int, first: bool) -> str:
    span = f'{lo}–{hi}' if hi > lo else f'{lo}'
    return f'most bought {span}' if first else f'rank {span}'


def draw(store, layouts, dist, out_dir=None, ranks=None, tours=None):
    """The two-panel figure, saved through ``figstyle.save`` (print width
    and minimum type checked). ``ranks`` maps a fixture to its product's
    popularity rank (shading), ``tours`` each layout key to its
    ``flow_tour``. Returns ``(pdf path, smallest font)``."""
    figstyle.apply_print()
    shop = store.shop
    n_items = len(shop.floors[1]['items'])
    edges = rank_classes(n_items) if ranks else []
    tours = tours or {}
    height = figstyle.print_width(STEM) / 2 * shop.height / shop.width + 0.8
    fig, axes = figstyle.print_figure(STEM, height_in=height, nrows=1,
                                      ncols=2)
    hl = [n for n, d in dist.items() if d >= MOVE_HIGHLIGHT_M]

    def title(label, key):
        t = tours.get(key) or {}
        if t.get('length_m') is None:
            return label
        return f"{label}: tour {t['length_m']:.0f} m"

    _draw(axes[0], shop, layouts['baseline'],
          title('As built (repaired)', 'baseline'), highlight=hl,
          ranks=ranks, edges=edges, tour=tours.get('baseline'),
          scale_bar=True)
    _draw(axes[1], shop, layouts['optimized'],
          title('GA-optimized', 'optimized'), highlight=hl, ranks=ranks,
          edges=edges, tour=tours.get('optimized'))
    handles = []
    lo = 1
    for i, (colour, hi) in enumerate(zip(RANK_COLOURS, edges)):
        handles.append(Patch(facecolor=colour, edgecolor='white',
                             label=_rank_label(lo, hi, i == 0)))
        lo = hi + 1
    if not edges:
        handles.append(Patch(facecolor=UNRANKED, edgecolor='white',
                             label='fixture'))
    handles += [
        Patch(facecolor='none', edgecolor=MOVED_EDGE, lw=0.8,
              label=f'moved ≥ {MOVE_HIGHLIGHT_M:g} m'),
    ]
    if any((t or {}).get('points') for t in tours.values()):
        handles.append(Line2D([], [], color=TOUR, lw=0.7, ls=(0, (3, 1.5)),
                              label='flow tour'))
    handles += [
        Patch(facecolor=ZONE, alpha=0.2, edgecolor=ZONE, ls=':',
              label='zone'),
        Patch(facecolor=CHECKOUT, hatch=CHECKOUT_HATCH, edgecolor=BOUNDARY,
              lw=0.4, label='checkout'),
        Patch(facecolor=RESTROOM, hatch=RESTROOM_HATCH, edgecolor=BOUNDARY,
              lw=0.4, label='restroom'),
        Line2D([], [], marker='^', ls='none', color=DOOR, markersize=5,
               label='entrance'),
    ]
    fig.legend(handles=handles, loc='outside lower center',
               ncol=math.ceil(len(handles) / 2), handlelength=1.4,
               columnspacing=0.9)
    smallest = figstyle.check_print(fig, STEM)
    path = figstyle.save(fig, STEM, out_dir=out_dir, png_dpi=300)
    plt.close(fig)
    return path, smallest


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--figc-dir', default=None,
                   help='Figure C run to draw (default: the newest '
                        'real_data_uci_* with saved layouts)')
    p.add_argument('--results', default=os.path.join(_HERE, 'results'))
    p.add_argument('--retail-path', default=None,
                   help='the UCI workbook (default: dataset_paths)')
    p.add_argument('--figs-dir', default=None,
                   help='where to write (default: the repository\'s figs/)')
    args = p.parse_args(argv)
    if args.figc_dir is None:
        args.figc_dir = default_figc_dir(args.results)
        if args.figc_dir is None:
            p.error(f'no Figure C run with saved layouts under '
                    f'{args.results}')
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    prov = provenance_snapshot()
    t0 = time.perf_counter()
    store, layouts, side = rebuild_store(args.figc_dir, args.retail_path)
    names = list(store.item_names)
    inv = {k: checked_layout(store.shop, names, layouts[k], f'figure {k} ')
           for k in ('baseline', 'optimized')}
    items = store.shop.floors[1]['items']
    sizes = {n: items[n]['size'] for n in names}
    dist = moves(layouts['baseline'], layouts['optimized'])
    A = store.shop.customer_simulation.analytics
    popularity = dict(_cval(A, 'popular_items', {}) or {})
    ranks = item_ranks(popularity, names)
    if set(ranks) != set(names):
        raise RuntimeError("the calibration does not rank every placed "
                           "fixture's product")
    tours = {k: checked_tour(store.shop, layouts[k], names)
             for k in ('baseline', 'optimized')}
    out_dir = args.figs_dir or figstyle.figs_dir()
    pdf, smallest = draw(store, layouts, dist, out_dir=out_dir, ranks=ranks,
                         tours=tours)
    edges = rank_classes(len(names))
    record = {
        'figure': STEM,
        'source': {'figc_dir': os.path.abspath(args.figc_dir),
                   'layouts_sha256': _sha256(os.path.join(args.figc_dir,
                                                          'layouts.json')),
                   'summary_sha256': _sha256(os.path.join(args.figc_dir,
                                                          'summary.json')),
                   'figc_git_sha': side.get('git_sha'),
                   'search_seed': int(side['args']['ga_seed'])},
        'store': {'width_m': float(store.shop.width),
                  'height_m': float(store.shop.height),
                  'n_items': len(names),
                  'n_zones': store.n_sections},
        'moves': move_stats(layouts['baseline'], layouts['optimized'],
                            sizes),
        'shading': {'by': 'popularity rank: invoices containing the '
                          'product (calibration popular_items)',
                    'class_upper_ranks': edges,
                    'most_bought': popularity_order(
                        {n: popularity[n] for n in names}
                    )[:FLOW_TOUR_ITEMS]},
        'flow_tour': {'n_items': FLOW_TOUR_ITEMS,
                      'items': tours['optimized']['items'],
                      'length_m': {k: t['length_m']
                                   for k, t in tours.items()},
                      'flow_score': {k: t['ga_flow_score']
                                     for k, t in tours.items()},
                      'matches_ga_flow_score': True},
        'scale_bar_m': SCALE_BAR_M,
        'invariants': inv,
        'invariants_hold': all(
            all(v == 0 for k, v in rec.items() if k.startswith('n_'))
            for rec in inv.values()),
        'min_clearance_m': min(float(rec['min_clearance_kept_apart_m'])
                               for rec in inv.values()),
        'required_clearance_m': float(MIN_AISLE),
        'print': figstyle.print_record(pdf, smallest,
                                       figstyle.print_width(STEM)),
        'print_frac': figstyle.PRINT_FRAC[STEM],
        'wall_seconds': time.perf_counter() - t0,
        'provenance': with_git_state_check(prov),
    }
    path = os.path.join(out_dir, STEM + '.json')
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(portable_paths(record), f, indent=1)
    m, t = record['moves'], record['flow_tour']['length_m']
    print(f"[layout figure] {pdf}: {m['n_moved']}/{m['n_fixtures']} "
          f"fixtures moved, {m['n_moved_at_least_aisle']} by >= "
          f"{MOVE_HIGHLIGHT_M:g} m ({m['n_slot_exchange']} slot exchanges, "
          f"{m['n_along_run']} along their own run); flow tour "
          f"{t['baseline']:.1f} m -> {t['optimized']:.1f} m; invariants "
          f"hold: {record['invariants_hold']} (min clearance "
          f"{record['min_clearance_m']:.2f} m); smallest text "
          f"{smallest:.1f} pt")
    return 0


if __name__ == '__main__':
    sys.exit(main())
