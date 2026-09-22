"""Generate a realistic shop floor plan from a CalibratedParams object.

The geometry comes from ``shop_architecture`` (grid archetype): perimeter
wall fixtures, parallel gondola aisles, a front power aisle, a multi-lane
checkout bank by the entrance, and a WC -- the same realistic architecture
as the Generate button -- while the ITEMS remain the dataset's calibrated
products (``product_id``, price, and category are preserved on every
fixture, so analytics / GA / validation see the real assortment).

Two things are deliberately data-driven and two deliberately not:

  * DATA-DRIVEN -- which sections and items exist (from calibration), and
    the SHOP DIMENSIONS: ``dimensions_for_assortment`` sizes the floor
    from how many categories/products the dataset carries, so a dataset
    with a rich assortment yields a proportionally larger retail space.
  * NOT data-driven -- the geometry itself (aisle widths, fixture depths,
    archetype) and, under ``naive=True``, the ORDER of placement.

``naive=True`` (GUI default): sections alphabetical, items sorted by
``product_id`` -- a traffic/value-neutral starting layout the Optimize
pipeline can measurably improve. ``naive=False``: sections by revenue
share, items by visit count (legacy popularity baseline). The old gold
impulse-shelf relocation is gone -- the architecture engine owns
placement now, and impulse positioning is the GA's job to discover.

The output is wired directly into the multi-floor visualizer state and
explicitly clears EVERY floor first (the historical "floor 2 stale
data" bug).
"""

from __future__ import annotations

import random as _random_mod
import warnings
from typing import Dict, List, Tuple

import numpy as np

from dataset_calibration import CalibratedParams
from shop_architecture import (generate_architecture,
                               dimensions_for_assortment)

# Seeded RNG for the (vary=False) engine call -- layouts are reproducible
# run-to-run for the same calibration.
COSMETIC_SEED = 12345

# When the engine cannot place every product on the heuristic floor size,
# the floor grows by GROW_FACTOR per side (aspect ratio kept), at most
# GROW_STEPS times.
GROW_FACTOR = 1.1
GROW_STEPS = 8


# --- Helpers --------------------------------------------------------------

def _items_per_category(params: CalibratedParams,
                        max_items_per_category: int) -> Dict[str, List[str]]:
    """Bucket product_ids by inferred category, keep the top-N by visit
    count per category so we don't try to lay out 4,000 products."""
    by_cat: Dict[str, List[Tuple[str, int]]] = {}
    for pid, cat in params.item_categories.items():
        visits = params.item_visit_counts.get(pid, 0)
        by_cat.setdefault(cat, []).append((pid, visits))
    out: Dict[str, List[str]] = {}
    for cat, lst in by_cat.items():
        lst.sort(key=lambda t: t[1], reverse=True)
        out[cat] = [pid for pid, _ in lst[:max_items_per_category]]
    return out


# --- Main entry point -----------------------------------------------------

def build_layout_from_calibration(shop,
                                  params: CalibratedParams,
                                  max_items_per_category: int = 12,
                                  naive: bool = True,
                                  ) -> dict:
    """Materialize ``params`` as a single-floor realistic shop on ``shop``
    (the ShopVisualizer or a HeadlessShop). Returns a ``stats`` dict:
    {'sections': departments placed, 'zones': Section_ zone walls,
     'items': M, 'dropped': [item names not placed], 'width_m': W,
     'height_m': H}. A department split over two gondolas counts once in
    'sections' and twice in 'zones'.

    Multi-floor note: this REPLACES the contents of every floor.
    """
    cats = _items_per_category(params, max_items_per_category)
    if not cats:
        raise ValueError("build_layout_from_calibration: no categories found")

    # Placement order: the naive baseline is deliberately neutral w.r.t.
    # traffic/value; the non-naive variant is the popularity-ordered
    # legacy baseline.
    if naive:
        ordered = [(c, sorted(cats[c])) for c in sorted(cats.keys())]
    else:
        ordered = sorted(cats.items(),
                         key=lambda kv: -params.category_revenue_share.get(kv[0], 0.0))
        ordered = [(c, list(pids)) for c, pids in ordered]

    # Unique display keys per product (multiple SKUs can share a display
    # name like "MUG"); the engine uses these as item-dict keys.
    key_to_pid: Dict[str, str] = {}
    sections: List[Tuple[str, List[str]]] = []
    for cat, pids in ordered:
        names: List[str] = []
        for pid in pids:
            display = (params.item_names.get(pid) or str(pid)).strip()[:32] \
                or str(pid)
            key = display
            if key in key_to_pid:
                key = f"{display} ({pid})"
            n = 2
            while key in key_to_pid:
                key = f"{display} ({pid}-{n})"
                n += 1
            key_to_pid[key] = pid
            names.append(key)
        sections.append((cat, names))

    # SHOP DIMENSIONS follow from the assortment: more categories/products
    # in the dataset -> a proportionally larger floor. The area heuristic
    # only estimates fixture capacity, so when the engine cannot place
    # every product the floor grows and the plan is rebuilt.
    n_sections = len(sections)
    n_items_max = max(len(nm) for _, nm in sections)
    shop_w, shop_h = dimensions_for_assortment(n_sections, n_items_max)

    # Realistic architecture (grid archetype for a generic retail dataset);
    # vary=False preserves our section order exactly, so the naive
    # alphabetical baseline survives into the wall insertion order.
    def _build(w, h):
        return generate_architecture(w, h, '__dataset_grid__', sections,
                                     rng=_random_mod.Random(COSMETIC_SEED),
                                     is_main_floor=True, vary=False)

    arch = _build(shop_w, shop_h)
    for _ in range(GROW_STEPS):
        if not arch['dropped']:
            break
        shop_w = round(shop_w * GROW_FACTOR, 1)
        shop_h = round(shop_h * GROW_FACTOR, 1)
        arch = _build(shop_w, shop_h)

    shop.width = shop_w
    shop.height = shop_h

    # CRITICAL: clear EVERY floor, not just the current one (the
    # ``self.items / .walls / .prices`` proxies only touch floor 1).
    for fid in list(shop.floors.keys()):
        shop.floors[fid]['items'] = {}
        shop.floors[fid]['walls'] = {}
        shop.floors[fid]['prices'] = {}
        shop.floors[fid]['door_position'] = None
        shop.floors[fid]['door_side'] = None
    extra_floors = [fid for fid in shop.floors if fid != 1]
    for fid in extra_floors:
        del shop.floors[fid]
    shop.connectors = {}
    shop.num_floors = 1
    shop.current_floor = 1

    f1 = shop.floors[1]

    for nm, wd in arch['walls'].items():
        f1['walls'][nm] = wd

    n_items_placed = 0
    for nm, it in arch['items'].items():
        pid = key_to_pid.get(nm)
        f1['items'][nm] = {
            'position': [float(it['position'][0]), float(it['position'][1])],
            'size':     [float(it['size'][0]), float(it['size'][1])],
            'category': it['category'],
            # The Section_ wall this fixture sits in: a department split
            # over two gondolas has one zone per gondola.
            'zone': it.get('zone'),
            'source_name': nm,
            'product_id': pid,
        }
        f1['prices'][nm] = float(params.item_prices.get(pid, 0.0) or 0.0)
        n_items_placed += 1

    # If the floor still cannot hold the assortment after growing, the
    # engine trims the tail of shelving runs. Say so instead of silently
    # placing fewer products than were requested.
    dropped = [k for k in key_to_pid if k not in arch['items']]
    if dropped:
        warnings.warn(
            f"build_layout_from_calibration: {len(dropped)} of "
            f"{len(key_to_pid)} products trimmed by fixture capacity on the "
            f"{shop_w:.0f}x{shop_h:.0f} m floor: "
            f"{', '.join(dropped[:10])}{' ...' if len(dropped) > 10 else ''}",
            stacklevel=2)

    door_position = arch['door_position']
    door_side = arch['door_side']
    f1['door_position'] = door_position
    f1['door_side'] = door_side
    shop.door_position = door_position
    shop.door_side = door_side

    # Departments, not zone walls: a split department has one Section_
    # zone per gondola.
    n_departments = len({it['category'] for it in f1['items'].values()})
    n_zones = sum(1 for nm in f1['walls'] if nm.startswith('Section_'))

    # Resize heat-map buffers to match the new shop dims + sync the sim.
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
            if hasattr(sim, '_floor_heat_raw'):
                sim._floor_heat_raw = {1: sim.heat_raw}
        except Exception:
            pass
        try:
            sim.invalidate_zones_cache()
        except Exception:
            pass

    return {
        'sections': n_departments,
        'zones': n_zones,
        'items': n_items_placed,
        'dropped': dropped,
        'width_m': shop_w,
        'height_m': shop_h,
    }
