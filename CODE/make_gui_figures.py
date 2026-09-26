"""Regenerate the GUI screenshots.

Requires a live desktop (Tk display): it opens the real application,
calibrates a UCI Online Retail II shop, captures the layout canvas, runs
the live simulation briefly and captures it with agents overlaid at their
real positions (colored by behavioral state).

The emergent traffic heat map is no longer drawn here: a few seconds of an
unseeded GUI session cannot be regenerated. It comes from a seeded
headless run of the live store at the live diagnostics' protocol,
``python -m experiments.make_heatmap_figure`` (review R52).

Screenshots are taken from the application's OWN matplotlib canvas
(``fig.savefig``), never a desktop screen-grab, so no other window can
appear in the capture.

Usage (from CODE/, on a machine with a display):
    python make_gui_figures.py

The workbook is found by ``dataset_paths`` (``DATASETS/``, or the path in
``UCI_RETAIL_XLSX``).
"""

from __future__ import annotations

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE) if os.path.basename(HERE).upper() == 'CODE' else HERE
FIGS = os.path.join(ROOT, 'figs')
os.makedirs(FIGS, exist_ok=True)
sys.path.insert(0, HERE)

import matplotlib
matplotlib.use('TkAgg')

from visualizer import ShopVisualizer                       # noqa: E402
import dataset_adapters as DA                               # noqa: E402
import dataset_calibration as DC                            # noqa: E402
import dataset_paths                                        # noqa: E402
from dataset_layout import build_layout_from_calibration    # noqa: E402


def _save_canvas(root, v, path):
    for _ in range(6):
        root.update_idletasks(); root.update(); time.sleep(0.04)
    v.fig.savefig(path, dpi=130, facecolor=v.fig.get_facecolor())
    print('saved', path)


def main():
    v = ShopVisualizer((32.0, 22.0))
    root = v.tk_root
    try:
        root.geometry('1500x950+30+30')
        root.update()

        uci = dataset_paths.uci_workbook()
        sheets = DA.list_excel_sheets(uci)
        df, _ = DA.read_excel_sheets(uci, [sheets[-1][0]])
        df = df.sample(n=60000, random_state=0).reset_index(drop=True)
        norm, _ = DA.OnlineRetailIIAdapter().adapt(df)
        params = DC.calibrate_transactional(norm, currency='GBP')
        # small per-category cap keeps item labels legible in the figure
        build_layout_from_calibration(v, params, max_items_per_category=4)
        params.seed_into(v.customer_simulation, shop=v)
        v.current_tab = 'Layout'
        v.redraw()
        _save_canvas(root, v, os.path.join(FIGS, 'gui_layout.png'))

        sim = v.customer_simulation
        sim.spawn_rate = 1.5
        sim.start_simulation()
        t0 = time.time()
        while time.time() - t0 < 16.0:
            root.update(); time.sleep(0.03)
        print('live customers:', len(sim.customers))
        state_col = {'shopping': '#7CE577', 'checking_out': '#FFD54A',
                     'exiting': '#FF8A65'}
        for c in list(sim.customers):
            v.ax.plot(c.position[0], c.position[1], 'o', ms=7,
                      color=state_col.get(c.state, '#FFFFFF'),
                      mec='black', mew=0.8, zorder=60)
        v.fig.canvas.draw()
        _save_canvas(root, v, os.path.join(FIGS, 'gui_simulation.png'))

        sim.hard_stop()
    finally:
        try:
            v.customer_simulation.hard_stop()
        except Exception:
            pass
        try:
            root.destroy()
        except Exception:
            pass
    return 0


if __name__ == '__main__':
    sys.exit(main())
