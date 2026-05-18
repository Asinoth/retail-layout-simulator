"""Generate a shop floor plan from a CalibratedParams object.

The output is wired *directly* into the multi-floor visualizer state
(``self.floors[fid]['items'|'walls'|'prices']``). It explicitly:

  - clears EVERY floor (avoids the historical "floor 2 stale data" bug
    where ``self.items.clear()`` only touched the current floor via the
    property proxy),
  - groups items by inferred category and builds a ``Section_<category>``
    wall per category,
  - places items in a row-major grid inside their section,
  - adds boundary walls + entrance + checkout on floor 1,
  - resizes the simulation's heat-map buffers to match the new shop dims.

It does NOT touch ``customer_simulation.analytics`` — that's the
calibration step's responsibility (``CalibratedParams.seed_into``).
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

from dataset_calibration import CalibratedParams


# ─── Tunables (kept as module constants so a TOMACS reviewer can see what
# default geometry was used without spelunking the source.)

ITEM_SIZE   = (1.2, 0.8)      # (width_m, depth_m) of every product fixture
ITEM_GAP    = (0.6, 0.6)      # (x_gap, y_gap) between items in a section
SECTION_PAD = 0.6             # interior padding inside each section
AISLE_WIDTH = 1.5             # space between sections
ENTRANCE_WIDTH = 2.0
CHECKOUT_WIDTH = 3.0
CHECKOUT_DEPTH = 1.5
FRONT_BUFFER = 4.0            # depth of clear area between entrance and sections
BOUNDARY_WALL_THICKNESS = 0.2


# ─── Helpers ──────────────────────────────────────────────────────────────

def _items_per_category(params: CalibratedParams,
                        max_items_per_category: int) -> Dict[str, List[str]]:
    """Bucket product_ids by inferred category, keep the top-N by visit
    count per category so we don't try to lay out 4,000 products in 25m²."""
    by_cat: Dict[str, List[Tuple[str, int]]] = {}
    for pid, cat in params.item_categories.items():
        visits = params.item_visit_counts.get(pid, 0)
        by_cat.setdefault(cat, []).append((pid, visits))
    out: Dict[str, List[str]] = {}
    for cat, lst in by_cat.items():
        lst.sort(key=lambda t: t[1], reverse=True)
        out[cat] = [pid for pid, _ in lst[:max_items_per_category]]
    return out


def _grid_for(n: int) -> Tuple[int, int]:
    """Return (rows, cols) for laying out n items roughly square."""
    if n <= 0:
        return 0, 0
    cols = max(1, int(round(math.sqrt(n))))
    rows = int(math.ceil(n / cols))
    return rows, cols


def _section_size_for(n_items: int) -> Tuple[float, float]:
    rows, cols = _grid_for(n_items)
    w = cols * ITEM_SIZE[0] + (cols - 1) * ITEM_GAP[0] + 2 * SECTION_PAD
    h = rows * ITEM_SIZE[1] + (rows - 1) * ITEM_GAP[1] + 2 * SECTION_PAD
    return max(w, 2.0), max(h, 2.0)


# ─── Main entry point ─────────────────────────────────────────────────────

def build_layout_from_calibration(shop,
                                  params: CalibratedParams,
                                  max_items_per_category: int = 12,
                                  ) -> Dict[str, int]:
    """Materialize ``params`` as a single-floor shop on ``shop`` (the
    ShopVisualizer). Returns a small ``stats`` dict for the UI:
    {'sections': N, 'items': M, 'width_m': W, 'height_m': H}.

    Multi-floor note: this REPLACES the contents of every floor. Multi-floor
    layouts are inherently a design decision — once the user has a working
    single-floor calibrated layout, they can add floors via the existing
    `+ Floor` button and the connector tooling.
    """
    cats = _items_per_category(params, max_items_per_category)
    if not cats:
        raise ValueError("build_layout_from_calibration: no categories found")

    # 1) Decide section sizes
    sections: List[Tuple[str, List[str], float, float]] = []  # (cat, pids, w, h)
    for cat, pids in sorted(cats.items(), key=lambda kv: -len(kv[1])):
        w, h = _section_size_for(len(pids))
        sections.append((cat, pids, w, h))

    # 2) Pack sections row-by-row. Pick a target shop width from the
    #    cumulative widths so we have ~3 sections per row on average.
    n = len(sections)
    target_cols = max(1, int(round(math.sqrt(n * 1.4))))
    rows_layout: List[List[Tuple[str, List[str], float, float]]] = []
    row: List[Tuple[str, List[str], float, float]] = []
    for sec in sections:
        row.append(sec)
        if len(row) >= target_cols:
            rows_layout.append(row)
            row = []
    if row:
        rows_layout.append(row)

    # Per-row width & height
    row_widths = [
        sum(s[2] for s in r) + (len(r) - 1) * AISLE_WIDTH for r in rows_layout
    ]
    row_heights = [max(s[3] for s in r) for r in rows_layout]

    # 3) Compute final shop dimensions
    content_w = max(row_widths) if row_widths else 8.0
    content_h = (sum(row_heights)
                 + (len(rows_layout) - 1) * AISLE_WIDTH
                 + FRONT_BUFFER)
    shop_w = max(20.0, content_w + 4.0)
    shop_h = max(15.0, content_h + 2.0)

    # 4) Apply to shop
    shop.width = shop_w
    shop.height = shop_h

    # CRITICAL: clear EVERY floor, not just the current one. The
    # ``self.items / .walls / .prices`` proxies only touch
    # ``floors[current_floor]`` — the bug that motivated this rewrite.
    for fid in list(shop.floors.keys()):
        shop.floors[fid]['items'] = {}
        shop.floors[fid]['walls'] = {}
        shop.floors[fid]['prices'] = {}
        shop.floors[fid]['door_position'] = None
        shop.floors[fid]['door_side'] = None

    # Drop any extra floors so the loaded dataset starts on a clean slate.
    extra_floors = [fid for fid in shop.floors if fid != 1]
    for fid in extra_floors:
        del shop.floors[fid]
    shop.connectors = {}
    shop.num_floors = 1
    shop.current_floor = 1

    f1 = shop.floors[1]

    # Boundary walls
    t = BOUNDARY_WALL_THICKNESS
    f1['walls']['Wall_Bot']  = {'position': [0.0, 0.0],        'size': [shop_w, t]}
    f1['walls']['Wall_Top']  = {'position': [0.0, shop_h - t], 'size': [shop_w, t]}
    f1['walls']['Wall_Left'] = {'position': [0.0, 0.0],        'size': [t, shop_h]}
    f1['walls']['Wall_Right']= {'position': [shop_w - t, 0.0], 'size': [t, shop_h]}

    # Entrance + checkout on front wall (y = 0). Entrance on the right,
    # checkout to the left of it.
    entrance_x = shop_w - ENTRANCE_WIDTH - 0.5
    entrance_y = 0.0
    door_position = (entrance_x + ENTRANCE_WIDTH / 2, entrance_y + 0.2)
    door_side = 'bottom'
    f1['door_position'] = door_position
    f1['door_side'] = door_side
    shop.door_position = door_position
    shop.door_side = door_side

    f1['walls']['Entrance'] = {
        'position': [entrance_x, entrance_y],
        'size':     [ENTRANCE_WIDTH, 0.4],
        'category': 'Entrance',
    }
    f1['walls']['Checkout'] = {
        'position': [entrance_x - CHECKOUT_WIDTH - 1.0, 0.4],
        'size':     [CHECKOUT_WIDTH, CHECKOUT_DEPTH],
        'category': 'Checkout',
    }

    # 5) Place section walls + items, row by row from front to back so the
    #    customer sim's "deeper = harder to reach" heuristic still works.
    n_sections_placed = 0
    n_items_placed = 0
    y_cursor = FRONT_BUFFER
    for r_idx, r in enumerate(rows_layout):
        row_h = row_heights[r_idx]
        # Center the row horizontally
        row_w = (sum(s[2] for s in r)
                 + (len(r) - 1) * AISLE_WIDTH)
        x_cursor = (shop_w - row_w) / 2.0
        for cat, pids, sec_w, sec_h in r:
            f1['walls'][f'Section_{cat}'] = {
                'position': [x_cursor, y_cursor],
                'size':     [sec_w, sec_h],
                'category': cat,
            }
            n_sections_placed += 1

            # Place items in a grid inside the section.
            rows_i, cols_i = _grid_for(len(pids))
            for k, pid in enumerate(pids):
                rr = k // cols_i
                cc = k % cols_i
                ix = x_cursor + SECTION_PAD + cc * (ITEM_SIZE[0] + ITEM_GAP[0])
                iy = y_cursor + SECTION_PAD + rr * (ITEM_SIZE[1] + ITEM_GAP[1])
                display = params.item_names.get(pid) or pid
                # Items can collide on display name (multiple SKUs called
                # "MUG"). Make the catalog key unique while preserving the
                # display name on the patch.
                key = display.strip()[:32] or pid
                if key in f1['items']:
                    key = f"{key} ({pid})"
                f1['items'][key] = {
                    'position': [float(ix), float(iy)],
                    'size':     list(ITEM_SIZE),
                    'category': cat,
                    'source_name': key,
                    'product_id': pid,
                }
                f1['prices'][key] = float(params.item_prices.get(pid, 0.0) or 0.0)
                n_items_placed += 1

            x_cursor += sec_w + AISLE_WIDTH
        y_cursor += row_h + AISLE_WIDTH

    # 6) Resize heat-map buffers to match new shop dims
    sim = getattr(shop, 'customer_simulation', None)
    if sim is not None:
        sim.door_position = door_position
        sim.door_side = door_side
        sim.geometry_dirty = True
        try:
            res = sim.heat_map_resolution
            wc = int(shop_w * res) + 1
            hc = int(shop_h * res) + 1
            sim.heat_map_data = np.zeros((wc, hc))
            sim.heat_raw = np.zeros((wc, hc), dtype=np.float32)
            # Floor-1 heatmap too
            if hasattr(sim, '_floor_heat_raw'):
                sim._floor_heat_raw = {1: sim.heat_raw}
        except Exception:
            pass
        try:
            sim.invalidate_zones_cache()
        except Exception:
            pass

    return {
        'sections': n_sections_placed,
        'items': n_items_placed,
        'width_m': shop_w,
        'height_m': shop_h,
    }
