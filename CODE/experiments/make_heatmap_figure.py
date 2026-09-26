"""The emergent traffic heat map, from a seeded headless run (review R52).

The paper's heat map used to be a screenshot of an unseeded GUI session on
an earlier store (``make_gui_figures.py``), so nobody could regenerate the
figure the paper shows, and it predated the stocked-invoice list law. This
script draws it display-free from ONE seeded fixed-step replication of the
store the live diagnostics measure (``experiments._live_store``, the UCI
workbook's current period, naive layout) under their nominal protocol:
arrivals at ``NOMINAL_SPAWN`` per second with at most ``NOMINAL_CAP`` in the
store, the first ``WARMUP_S`` simulated seconds deleted, and the map
accumulated over the next ``COLLECT_S`` -- the increment of the
simulation's own heat map (one sample per agent per simulated second, per
5 cm cell) over the window, exactly what ``run_abm_diagnostics`` measures
its perimeter ratio on. The seed is disjoint from every live runner's.

For display only, the map is summed into 25 cm cells (DISPLAY_BIN_M, the
pathfinding grid's step; at print size a 5 cm cell is a speck), and the
colour scale is clipped at the 99th percentile of the occupied cells so a
few queue cells do not flatten the rest (the colour bar says so).
Fixtures, walls, checkout lanes and the door are outlined. The figure is
drawn at the width it is printed at -- INCLUDE_FRACTION of the ACM
small-format text width, the manuscript's include -- with no text below
7 pt, so it must be included at its natural size (``figure.include_hint``
in the summary); scaling it down when the paper includes it would shrink the
text below 7 pt again (review R56).

Writes ``emergent_heatmap.{pdf,png}`` to ``--figs-dir`` (the repo figs/ by
default) and, in a run directory under ``--out-root``, the figure again, a
sidecar and ``summary.json``: the protocol, the seed, the store, and this
run's perimeter ratio with its geometric null, so the figure's caption can
quote numbers from the same run.

    python -m experiments.make_heatmap_figure
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import figstyle  # noqa: E402
figstyle.apply()

from experiments import _live_store as LS  # noqa: E402
from experiments._common import (make_run_dir, provenance_snapshot,  # noqa: E402
                                 write_sidecar)
from experiments.run_abm_diagnostics import (  # noqa: E402
    COLLECT_S, NOMINAL_CAP, NOMINAL_SPAWN, WARMUP_S, _floor1_walkable,
    perimeter_ratio, perimeter_ratio_geometric_null)

# Seed of the one replication drawn; disjoint from every live runner's
# (ABM 1000.., structural 2000.., queueing 3000.., GoF 4000..) and from the
# protocol study's.
FIGURE_SEED = 5000
# Display: the share of occupied cells the colour scale covers before it
# clips, and the figure's size and smallest text.
CLIP_PERCENTILE = 99.0
DISPLAY_BIN_M = 0.25          # display cell: the pathfinding grid's step
# Printed at 0.62 of the text width, the share of the page the paper
# gives the heat map; drawn at that size so no text is scaled.
INCLUDE_FRACTION = 0.62
PRINT_WIDTH_IN = round(INCLUDE_FRACTION * figstyle.ACMSMALL_TEXTWIDTH_IN, 2)
MIN_FONT_PT = figstyle.MIN_PRINT_FONT_PT
_RC = {'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8,
       'xtick.labelsize': MIN_FONT_PT, 'ytick.labelsize': MIN_FONT_PT,
       'legend.fontsize': MIN_FONT_PT}


def window_heat(params, spawn, cap, warmup, seconds, dt, seed):
    """One seeded fixed-step run of the live store; returns the heat-map
    increment over the collection window, the walkable cells, the shop and
    the run's arrival load."""
    shop = LS.build_live_store(params)
    sim = shop.customer_simulation
    sim.max_customers = cap
    sim.spawn_rate = spawn
    boundary = {}

    def _mark(s):
        if boundary:
            return
        boundary['heat'] = np.array(s.heat_raw, dtype=float)
        boundary['arrivals'] = (int(s.analytics.get('total_customers', 0)),
                                int(s.analytics.get('balked_arrivals', 0)))

    sim.run_headless(warmup + seconds, dt=dt, seed=seed, callback=_mark,
                     callback_every_s=warmup)
    errs = dict(getattr(sim, 'suppressed_errors', {}))
    if errs:
        raise RuntimeError(
            f"live loop suppressed exceptions during the run: {errs}")
    heat = np.array(sim.heat_raw, dtype=float) - boundary['heat']
    admitted = int(sim.analytics.get('total_customers', 0))
    balked = int(sim.analytics.get('balked_arrivals', 0))
    load = {'arrivals': (admitted + balked) - sum(boundary['arrivals']),
            'balked': balked - boundary['arrivals'][1]}
    return heat, _floor1_walkable(sim, heat.shape), shop, load


def _binned(heat, res, bin_m=DISPLAY_BIN_M):
    """The heat map summed into square display cells of ``bin_m`` metres
    (whole numbers of the simulator's cells): at print size a 5 cm cell is
    a speck, and a walked aisle would show as a dotted line."""
    k = max(1, int(round(bin_m * res)))
    nx, ny = heat.shape
    px, py = (-nx) % k, (-ny) % k
    padded = np.pad(heat, ((0, px), (0, py)))
    return padded.reshape(padded.shape[0] // k, k,
                          padded.shape[1] // k, k).sum(axis=(1, 3)), k / res


def draw(heat, shop, out_dir, stem='emergent_heatmap', res=20):
    """The figure: the window's heat map over the floor plan."""
    W, H = shop.width, shop.height
    shown, cell_m = _binned(heat, res)
    occupied = shown[shown > 0]
    vmax = float(np.percentile(occupied, CLIP_PERCENTILE)) if occupied.size \
        else 1.0
    floor = shop.floors[1]
    with matplotlib.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(PRINT_WIDTH_IN,
                                        PRINT_WIDTH_IN * H / W * 0.92))
        im = ax.imshow(shown.T, origin='lower', cmap='inferno',
                       extent=[0, shown.shape[0] * cell_m, 0,
                               shown.shape[1] * cell_m],
                       aspect='equal', vmin=0.0, vmax=vmax,
                       interpolation='nearest')
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.grid(False)
        for name, d in (floor.get('items') or {}).items():
            x, y = d['position']
            w, h = d['size']
            ax.add_patch(plt.Rectangle((x, y), w, h, fill=False,
                                       edgecolor='white', lw=0.4, alpha=0.8))
        for name, d in (floor.get('walls') or {}).items():
            if name.startswith('Section_'):
                continue
            x, y = d['position']
            w, h = d['size']
            lane = name.startswith('Checkout') or name == 'WC'
            ax.add_patch(plt.Rectangle(
                (x, y), w, h, fill=False, lw=0.8 if lane else 0.5,
                edgecolor=figstyle.SKY if lane else 'white',
                linestyle='--' if lane else '-'))
        door = floor.get('door_position') or getattr(shop, 'door_position',
                                                      None)
        if door is not None:
            ax.plot([door[0]], [door[1]], marker='^', ms=5, color=figstyle.SKY,
                    mec='black', mew=0.4, ls='none', label='door')
        cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cb.set_label(f'agent-seconds per {100 * cell_m:.0f} cm cell\n'
                     f'(clipped at the {CLIP_PERCENTILE:g}th percentile)')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        fig.tight_layout()
        path = figstyle.save(fig, stem, out_dir=out_dir)
        plt.close(fig)
    return path, vmax


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=FIGURE_SEED)
    p.add_argument('--spawn', type=float, default=NOMINAL_SPAWN)
    p.add_argument('--cap', type=int, default=NOMINAL_CAP)
    p.add_argument('--warmup', type=float, default=WARMUP_S)
    p.add_argument('--seconds', type=float, default=COLLECT_S)
    p.add_argument('--dt', type=float, default=0.04)
    p.add_argument('--retail-path', type=str, default=None)
    p.add_argument('--figs-dir', type=str, default=None,
                   help="Where emergent_heatmap.{pdf,png} go (default: "
                        "repo figs/)")
    p.add_argument('--out-root', type=str,
                   default=os.path.join(_HERE, 'results'))
    args = p.parse_args(argv)
    if args.warmup <= 0 or args.seconds <= 0 or args.dt <= 0:
        p.error("--warmup, --seconds and --dt must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    figs = args.figs_dir or figstyle.figs_dir()
    os.makedirs(figs, exist_ok=True)
    out_dir = make_run_dir(args.out_root, 'heatmap_figure')
    prov = provenance_snapshot()
    params = LS.calibrate_live_store(args.retail_path, 'current')
    data = LS.period_record(args.retail_path, 'current')
    t0 = time.perf_counter()
    heat, walkable, shop, load = window_heat(
        params, args.spawn, args.cap, args.warmup, args.seconds, args.dt,
        args.seed)
    wall = time.perf_counter() - t0
    path, vmax = draw(heat, shop, figs,
                      res=shop.customer_simulation.heat_map_resolution)
    for ext in ('.pdf', '.png'):
        shutil.copy2(os.path.splitext(path)[0] + ext, out_dir)
    pr = perimeter_ratio(heat, shop.width, shop.height)
    null = perimeter_ratio_geometric_null(walkable, shop.width, shop.height,
                                          pr['band_m'])
    printed = figstyle.print_record(path, MIN_FONT_PT)
    summary = {
        'figure': {'stem': 'emergent_heatmap', 'width_in': PRINT_WIDTH_IN,
                   'min_font_pt': MIN_FONT_PT, 'cmap': 'inferno',
                   'clip_percentile': CLIP_PERCENTILE, 'vmax': vmax,
                   'display_cell_m': DISPLAY_BIN_M,
                   'include_fraction_of_textwidth': INCLUDE_FRACTION,
                   'textwidth_in': round(figstyle.ACMSMALL_TEXTWIDTH_IN, 3),
                   # Include the vector file at its natural size: that
                   # prints every font as drawn.
                   'include_hint': '\\includegraphics{figs/'
                                   'emergent_heatmap.pdf}',
                   **printed},
        'protocol': {'mode': 'fixed_step', 'dt': args.dt, 'seed': args.seed,
                     'spawn': args.spawn, 'cap': args.cap,
                     'warmup_s': args.warmup, 'collect_s': args.seconds,
                     'reps': 1, 'heat': 'increment over the window, '
                                        'one sample per agent per second'},
        'store': {'width_m': shop.width, 'height_m': shop.height,
                  'n_items': len(shop.floors[1].get('items') or {})},
        'load': load,
        'agent_seconds': float(heat.sum()),
        'perimeter_ratio': pr,
        'perimeter_interior_ratio_geometric_null': (
            None if null is None else round(float(null), 3)),
        'data': data,
    }
    write_sidecar(out_dir, {'experiment': 'heatmap_figure',
                            'args': vars(args), 'wall_seconds': wall,
                            'figs_dir': figs}, provenance=prov)
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {path} (and .png); perimeter ratio "
          f"{pr['perimeter_interior_ratio']} (geometric null "
          f"{summary['perimeter_interior_ratio_geometric_null']}); "
          f"artifacts in {out_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
