"""Parameterized synthetic shops for the TOMACS methodology validation.

Each ``SyntheticShop`` carries:
  * geometry (width, height, entrance, checkout, sections)
  * a small set of items (typically 8-12) with category, size, base
    revenue, and per-item type tag (regular/impulse)
  * a cross-merchandising affinity matrix
  * the *literature-cited elasticity parameters* used by the oracle's
    closed-form revenue model

These are not a replacement for the dataset-loaded shop that
dataset_layout.build_layout_from_calibration produces. They're small,
controlled instances for one job -- the synthetic-ground-truth pipeline:
generate -> solve analytically (oracle) -> run the GA on the simulator
fitness -> compare GA vs oracle.

oracle.py works only off these dataclasses and analytical_revenue below.
The simulator fitness in viz_ga._ga_compute_layout_score is structurally
independent (agent paths, heatmap, ...), so any agreement between the two
is real evidence that the geometric scoring captures the same placement
principles as the closed-form model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# --- Data classes ---------------------------------------------------------

@dataclass
class SyntheticItem:
    """A single product fixture in a synthetic shop."""
    name: str
    category: str
    size: Tuple[float, float]          # (width_m, depth_m)
    base_revenue: float                # GBP / per-converting-customer
    is_impulse: bool = False
    # Per-item demand sensitivities used by ``analytical_revenue``.
    # Defaults come from retail_literature.py midpoints; the generator
    # perturbs them within the cited bands.
    detour_elasticity: float = 0.40    # Hui 2009 midpoint
    impulse_elasticity: float = 0.50   # Hui Inman 2013 midpoint
    perimeter_elasticity: float = 0.20 # Larson 2005 lower bound


@dataclass
class SyntheticShop:
    """A fully-specified synthetic shop with a known analytical revenue
    model and a fixed entrance/checkout. Ready to (a) hand to the oracle
    for closed-form optimization and (b) materialize into a headless
    simulator instance via ``experiments._common.build_headless_shop``."""
    name: str                            # human label, e.g. "synth_shop_3"
    width: float
    height: float
    items: List[SyntheticItem]
    entrance: Tuple[float, float]        # (x, y) -- center of door
    checkout: Tuple[float, float]        # (x, y) -- center of checkout area
    sections: Dict[str, Tuple[float, float, float, float]]  # cat -> (x, y, w, h)
    affinity: Dict[Tuple[str, str], float] = field(default_factory=dict)
    # Per-shop scaling factors -- not random; deterministic from ``seed``.
    daily_customers: float = 200.0
    rng_seed: int = 0

    def item_by_name(self, name: str) -> SyntheticItem:
        for it in self.items:
            if it.name == name:
                return it
        raise KeyError(name)

    def section_bounds(self, category: str) -> Optional[Tuple[float, float, float, float]]:
        return self.sections.get(category)

    def diag(self) -> float:
        return math.hypot(self.width, self.height)


# --- Analytical revenue model (oracle's objective) ------------------------

def analytical_revenue(shop: SyntheticShop,
                       layout: Dict[str, Tuple[float, float]]) -> float:
    """Closed-form per-item revenue under literature-cited elasticities.

    For each item i at position (x_i, y_i):

        r_i = base_rev_i * daily_customers * (1 + L_traffic + L_impulse
                                                 + L_perimeter + L_cross)

    where:
        L_traffic   = -detour_elasticity * d(i, entrance) / diag
                      (Hui 2009: revenue decreases with detour from entrance)
        L_impulse   = impulse_elasticity * (1 - d(i, checkout) / diag)
                      if is_impulse else 0
                      (Hui Inman 2013: impulse SKUs lift sharply near checkout)
        L_perimeter = perimeter_elasticity * (1 - d(i, nearest_wall) / wall_diag)
                      (Larson 2005: perimeter racetrack gets more traffic)
        L_cross     = sum_{j != i} affinity[(i,j)] * (1 - d(i,j) / diag)
                      (Hui 2009: items with high co-purchase affinity benefit
                       from spatial proximity)

    Plus a HARD physical penalty for any within-section overlap (items
    cannot occupy the same physical space; this is a constraint, not a
    learned preference). Per overlapping pair we subtract twice the
    smaller item's base revenue * daily_customers -- large enough to
    dominate any positional gain from stacking.

    Total: R(layout) = sum_i r_i - overlap_penalty

    This formula is *structurally independent* of the simulator's GA
    scoring function on what makes a layout *good* (heatmap, agent
    paths, queue) -- their elasticities come from the same literature
    (Hui/Larson) but the functional forms differ. They agree only on
    what makes a layout *physically valid* (no overlapping items), which
    is a constraint both must respect. This is what keeps synthetic-GT
    non-circular: structural agreement on validity, divergence on value.
    """
    diag = shop.diag()
    rev_total = 0.0
    by_name = {it.name: it for it in shop.items}
    layout_pos = {name: layout[name] for name in by_name}

    for it in shop.items:
        x, y = layout_pos[it.name]
        w, h = it.size
        cx, cy = x + w / 2.0, y + h / 2.0

        # Traffic / detour penalty
        d_ent = math.hypot(cx - shop.entrance[0], cy - shop.entrance[1])
        L_traffic = -it.detour_elasticity * (d_ent / max(diag, 1e-6))

        # Impulse adjacency
        L_impulse = 0.0
        if it.is_impulse:
            d_chk = math.hypot(cx - shop.checkout[0], cy - shop.checkout[1])
            L_impulse = it.impulse_elasticity * (1.0 - d_chk / max(diag, 1e-6))

        # Perimeter proximity (closer to any wall is better)
        d_wall = min(cx, cy, shop.width - cx, shop.height - cy)
        L_perimeter = it.perimeter_elasticity * (
            1.0 - 2.0 * d_wall / max(min(shop.width, shop.height), 1e-6)
        )
        L_perimeter = max(0.0, L_perimeter)   # only reward, never punish

        # Cross-merchandising
        L_cross = 0.0
        for (a, b), aff in shop.affinity.items():
            if a == it.name:
                jx, jy = layout_pos[b]
                jw, jh = by_name[b].size
                jcx, jcy = jx + jw / 2.0, jy + jh / 2.0
                d_ij = math.hypot(cx - jcx, cy - jcy)
                L_cross += aff * (1.0 - d_ij / max(diag, 1e-6))
            elif b == it.name:
                jx, jy = layout_pos[a]
                jw, jh = by_name[a].size
                jcx, jcy = jx + jw / 2.0, jy + jh / 2.0
                d_ij = math.hypot(cx - jcx, cy - jcy)
                L_cross += aff * (1.0 - d_ij / max(diag, 1e-6))

        r_i = it.base_revenue * shop.daily_customers * (
            1.0 + L_traffic + L_impulse + L_perimeter + L_cross
        )
        rev_total += max(r_i, 0.0)   # revenue is non-negative

    # Physical overlap penalty (within-section only). Penalty is
    # *smooth* in overlap area so a gradient-based optimizer
    # (L-BFGS-B) can navigate out of overlapping configurations
    # cleanly. Scale: 2 * smaller_base * daily_customers per unit of
    # smaller-item area overlapped -- guaranteed to dominate any
    # positional gain from stacking.
    by_cat: Dict[str, List[SyntheticItem]] = {}
    for it in shop.items:
        by_cat.setdefault(it.category, []).append(it)
    overlap_pen = 0.0
    for cat, items in by_cat.items():
        for i in range(len(items)):
            it_i = items[i]
            x_i, y_i = layout[it_i.name]
            w_i, h_i = it_i.size
            for j in range(i + 1, len(items)):
                it_j = items[j]
                x_j, y_j = layout[it_j.name]
                w_j, h_j = it_j.size
                ox = max(0.0, min(x_i + w_i, x_j + w_j) - max(x_i, x_j))
                oy = max(0.0, min(y_i + h_i, y_j + h_j) - max(y_i, y_j))
                if ox > 0 and oy > 0:
                    overlap_area = ox * oy
                    smaller_area = min(w_i * h_i, w_j * h_j)
                    smaller_base = min(it_i.base_revenue, it_j.base_revenue)
                    overlap_pen += (
                        2.0 * smaller_base * shop.daily_customers
                        * overlap_area / max(smaller_area, 1e-6)
                    )

    return rev_total - overlap_pen


# --- Generator ------------------------------------------------------------

# Per-Larson/Sorensen/Hui defaults; categories deliberately small so we
# can vary the shop's item mix without exploding the oracle's search
# dimensionality.
_DEFAULT_CATEGORIES = [
    ("Beverages",   False, (1.2, 0.8), (8.0, 25.0)),
    ("Snacks",      False, (1.2, 0.8), (3.0, 12.0)),
    ("Produce",     False, (1.5, 1.0), (2.0, 8.0)),
    ("Electronics", False, (1.4, 0.9), (40.0, 200.0)),
    ("Stationery",  False, (1.0, 0.7), (2.0, 15.0)),
    ("Impulse",     True,  (0.6, 0.4), (1.0, 6.0)),
]


def _section_min_size_for(items_in_cat: List[SyntheticItem]
                          ) -> Tuple[float, float]:
    """Smallest section that fits these items in a 2-column grid with
    0.2m padding on each side and at least 0.2m gap between items."""
    if not items_in_cat:
        return (2.0, 2.0)
    max_w = max(it.size[0] for it in items_in_cat)
    max_h = max(it.size[1] for it in items_in_cat)
    n = len(items_in_cat)
    cols = max(1, min(2, n))
    rows = int(math.ceil(n / cols))
    gap = 0.25
    pad = 0.20
    w = 2 * pad + cols * max_w + (cols - 1) * gap
    h = 2 * pad + rows * max_h + (rows - 1) * gap
    return (w, h)


def _build_sections(width: float, height: float,
                    items_by_cat: Dict[str, List["SyntheticItem"]],
                    rng: np.random.Generator,
                    checkout: Tuple[float, float] = (0.0, 0.7),
                    front_buffer: float = 4.0,
                    aisle: float = 1.5,
                    pad: float = 0.6) -> Dict[str, Tuple[float, float, float, float]]:
    """Lay out one section per category in a row-major grid leaving a
    front-of-shop buffer for entrance/checkout. Each section is sized
    to fit its items in a 2-col grid with proper padding.

    Design rule: if "Impulse" is among the categories, place its section
    in the front row adjacent to checkout (Larson 2005, Hui Inman 2013).
    """
    cats = list(items_by_cat.keys())
    if "Impulse" in cats:
        cats.remove("Impulse")
        cats.insert(0, "Impulse")

    # Compute per-section minimum sizes
    min_sizes = {c: _section_min_size_for(items_by_cat[c]) for c in cats}
    max_sw = max(sz[0] for sz in min_sizes.values())
    max_sh = max(sz[1] for sz in min_sizes.values())

    # Layout grid: pick cols so the total horizontal footprint fits.
    n = len(cats)
    cols = max(1, int(round(math.sqrt(n * 1.4))))
    while cols > 1 and (cols * max_sw + (cols - 1) * aisle + 2 * pad) > width:
        cols -= 1
    rows = int(math.ceil(n / cols))

    # Use uniform per-section size (= max needed) so the grid layout is
    # clean; sections that don't need the full size simply have extra
    # interior space (which the GA can use to move items around).
    sec_w = max_sw
    sec_h = max_sh

    # Auto-expand shop dimensions if the grid doesn't fit at the
    # caller-supplied (width, height).
    needed_w = pad + cols * sec_w + (cols - 1) * aisle + pad
    needed_h = front_buffer + rows * sec_h + (rows - 1) * aisle + pad
    if needed_w > width or needed_h > height:
        # Caller-visible side effect: the SyntheticShop's width/height
        # are updated by generate_synthetic_shop() after this returns.
        pass   # we return the section dict; generator handles dims.

    sections: Dict[str, Tuple[float, float, float, float]] = {}
    for i, cat in enumerate(cats):
        r = i // cols
        c = i % cols
        x = pad + c * (sec_w + aisle)
        y = front_buffer + r * (sec_h + aisle)
        sections[cat] = (x, y, sec_w, sec_h)
    sections["__layout_dims__"] = (needed_w, needed_h, 0, 0)  # sentinel
    return sections


def generate_synthetic_shop(
    name: str,
    seed: int,
    n_items: int = 10,
    width: float = 12.0,
    height: float = 10.0,
    affinity_density: float = 0.15,
    daily_customers: float = 200.0,
) -> SyntheticShop:
    """Generate one parameterized synthetic shop.

    ``n_items`` keeps the oracle's continuous-optimization dimensionality
    at 2*n_items (default 20), which is well within scipy.optimize's
    reliable range when warm-started from a section-grid initialization.

    The per-item elasticities are drawn from cited literature bands
    (see ``retail_literature.py`` ELASTICITY_* constants). A
    deterministic seed produces a reproducible scenario.
    """
    rng = np.random.default_rng(seed)

    # Round-robin across categories so even small shops contain at least
    # one item per category (including Impulse -- otherwise the
    # impulse-elasticity branch of analytical_revenue never fires).
    items: List[SyntheticItem] = []
    cat_idx = 0
    next_in_cat = {cat: 0 for cat, *_ in _DEFAULT_CATEGORIES}
    while len(items) < n_items:
        cat, is_impulse_cat, size, (rev_lo, rev_hi) = (
            _DEFAULT_CATEGORIES[cat_idx % len(_DEFAULT_CATEGORIES)]
        )
        base_rev = float(rng.uniform(rev_lo, rev_hi))
        det = float(rng.uniform(0.30, 0.50))    # Hui 2009 midpoint +/-25%
        imp = float(rng.uniform(0.40, 0.70))    # Hui Inman 2013 band
        per = float(rng.uniform(0.15, 0.30))    # Larson 2005 perimeter
        suffix = next_in_cat[cat]
        next_in_cat[cat] += 1
        items.append(SyntheticItem(
            name=f"{cat[:3].upper()}_{suffix:02d}",
            category=cat,
            size=size,
            base_revenue=base_rev,
            is_impulse=is_impulse_cat,
            detour_elasticity=det,
            impulse_elasticity=imp,
            perimeter_elasticity=per,
        ))
        cat_idx += 1

    # Section walls per category, sized to fit their items.
    items_by_cat: Dict[str, List[SyntheticItem]] = {}
    for it in items:
        items_by_cat.setdefault(it.category, []).append(it)
    sections = _build_sections(width, height, items_by_cat, rng)
    # Extract layout sentinel + auto-expand shop dims if needed.
    needed_w, needed_h, _, _ = sections.pop("__layout_dims__")
    width = max(width, needed_w)
    height = max(height, needed_h)

    # Geometry: entrance bottom-right, checkout adjacent left
    entrance = (width - 1.0, 0.2)
    checkout = (width - 3.5, 0.7)

    # Cross-merchandising affinity: sparse random graph
    affinity: Dict[Tuple[str, str], float] = {}
    names = [it.name for it in items]
    n_pairs = int(affinity_density * len(names) * (len(names) - 1) / 2)
    for _ in range(n_pairs):
        a, b = rng.choice(names, size=2, replace=False)
        if a == b:
            continue
        key = (min(a, b), max(a, b))
        if key in affinity:
            continue
        affinity[key] = float(rng.uniform(0.10, 0.30))   # modest affinity

    return SyntheticShop(
        name=name,
        width=width,
        height=height,
        items=items,
        entrance=entrance,
        checkout=checkout,
        sections=sections,
        affinity=affinity,
        daily_customers=daily_customers,
        rng_seed=seed,
    )


def grid_layout_within_sections(shop: SyntheticShop) -> Dict[str, Tuple[float, float]]:
    """Deterministic 'naive' layout: place each item in a grid inside
    its category section. Used as the GA's initial chromosome and as
    the oracle's warm-start."""
    by_cat: Dict[str, List[SyntheticItem]] = {}
    for it in shop.items:
        by_cat.setdefault(it.category, []).append(it)
    layout: Dict[str, Tuple[float, float]] = {}
    for cat, cat_items in by_cat.items():
        bounds = shop.sections.get(cat)
        if bounds is None:
            continue
        sx, sy, sw, sh = bounds
        n = len(cat_items)
        cols = max(1, int(math.ceil(math.sqrt(n))))
        rows = int(math.ceil(n / cols))
        pad = 0.2
        cell_w = max((sw - 2 * pad) / cols, 0.1)
        cell_h = max((sh - 2 * pad) / rows, 0.1)
        for k, it in enumerate(cat_items):
            r = k // cols
            c = k % cols
            iw, ih = it.size
            x = sx + pad + c * cell_w + (cell_w - iw) / 2.0
            y = sy + pad + r * cell_h + (cell_h - ih) / 2.0
            layout[it.name] = (max(x, sx + 0.05), max(y, sy + 0.05))
    return layout
