"""Realistic shop-architecture generator.

Produces floor plans that follow the three canonical retail layout
archetypes named in the literature the paper cites (Sorensen 2009;
Larson 2005):

  * ``grid``      -- supermarket-style parallel gondola aisles with
                     perimeter wall fixtures, a front power aisle, a
                     checkout bank by the entrance, and the classic
                     "staple category on the back wall" placement.
                     Used for Grocery Store / Pharmacy / Hardware Store.
  * ``racetrack`` -- department-store loop: perimeter departments along
                     the walls, a central island block, and a walkable
                     loop corridor between them.
                     Used for Electronics Store / Bookstore.
  * ``freeform``  -- boutique-style: perimeter wall racks plus
                     free-standing display tables on a relaxed,
                     jittered grid. Used for Clothing Store / Cafe.

The module is deliberately Tk-free and side-effect-free so the smoke
test can validate every invariant headlessly (no overlaps, aisle
clearances, door-to-item reachability). ``viz_generate`` calls
``generate_architecture`` and copies the result into the visualizer's
floor dicts.

Output contract (matches the visualizer's existing dict shapes):

    {
      'walls': {name: {'position': (x, y), 'size': (w, h), ...}},
      'items': {name: {'position': (x, y), 'size': (w, h),
                       'category': section_name}},
      'door_position': (x, y) | None,
      'door_side': 'bottom' | None,
    }

Section zones are emitted as ``Section_<name>`` walls (non-blocking by
project convention) with a color from a fixed color-blind-safe palette.
"""

from __future__ import annotations

import math
import random as _random_mod
from typing import Dict, List, Optional, Sequence, Tuple

# --- Tunables (metres) ----------------------------------------------------

T = 0.2                 # boundary wall thickness
DOOR_LEN = 2.2          # entrance opening width
FIX_D = 0.7             # perimeter wall-fixture depth
GONDOLA_D = 1.0         # double-sided gondola depth
MIN_AISLE = 1.6         # minimum clear aisle between fixtures
SEG_GAP = 0.15          # gap between item segments on one run
MIN_SEG = 0.7           # minimum segment length worth placing
MAX_SEG = 3.2           # cap a single shelving segment (visual realism)
WC_W, WC_H = 2.0, 1.5   # restroom footprint
LANE_W, LANE_D = 0.9, 1.9    # one checkout lane (counter) footprint
LANE_GAP = 1.1               # customer pass-gap between checkout lanes

SECTION_PALETTE = (
    '#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F', '#EDC948',
    '#B07AA1', '#FF9DA7', '#9C755F', '#BAB0AC',
)

ARCHETYPE_BY_SHOP = {
    'Grocery Store':     'grid',
    'Pharmacy':          'grid',
    'Hardware Store':    'grid',
    'Electronics Store': 'racetrack',
    'Bookstore':         'racetrack',
    'Clothing Store':    'freeform',
    'Cafe':              'freeform',
}


# --- Small geometry helpers -----------------------------------------------

def _rects_overlap(a, b, eps=1e-9):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (ax + aw <= bx + eps or ax >= bx + bw - eps
                or ay + ah <= by + eps or ay >= by + bh - eps)


def _trim_interval(lo: float, hi: float,
                   keepouts: Sequence[Tuple[float, float]]) -> Tuple[float, float]:
    """Largest sub-interval of [lo, hi] that avoids every 1-D keepout."""
    pieces = [(lo, hi)]
    for klo, khi in keepouts:
        nxt = []
        for plo, phi in pieces:
            if khi <= plo or klo >= phi:
                nxt.append((plo, phi))
                continue
            if klo > plo:
                nxt.append((plo, klo))
            if khi < phi:
                nxt.append((khi, phi))
        pieces = nxt
    if not pieces:
        return lo, lo
    return max(pieces, key=lambda p: p[1] - p[0])


def _segments_along(names: Sequence[str], lo: float, hi: float,
                    fixed: float, depth: float, horizontal: bool,
                    category: str) -> Dict[str, dict]:
    """Place item segments along the run [lo, hi].

    ``fixed`` is the run's constant coordinate (y for horizontal runs,
    x for vertical). Capacity-aware: if the run cannot hold every name at
    MIN_SEG it places as many as fit (the caller's catalog is scaled to
    the shop, so trimming only occurs at extreme aspect ratios).
    Segments are capped at MAX_SEG and spread evenly so a short list on
    a long run still reads as continuous shelving, not one giant slab."""
    span = hi - lo
    if not names or span < MIN_SEG:
        return {}
    cap = int((span + SEG_GAP) // (MIN_SEG + SEG_GAP))
    use = list(names)[:cap]
    n = len(use)
    if n == 0:
        return {}
    seg = (span - (n - 1) * SEG_GAP) / n
    gap = SEG_GAP
    start = lo
    if seg > MAX_SEG:
        seg = MAX_SEG
        if n > 1:
            gap = (span - n * seg) / (n - 1)
        else:
            start = lo + (span - seg) / 2.0
    out = {}
    for i, nm in enumerate(use):
        a = start + i * (seg + gap)
        if horizontal:
            out[nm] = {'position': (a, fixed), 'size': (seg, depth),
                       'category': category}
        else:
            out[nm] = {'position': (fixed, a), 'size': (depth, seg),
                       'category': category}
    return out


def _zone(name: str, x, y, w, h, color) -> Tuple[str, dict]:
    return (f'Section_{name}', {
        'position': (x, y), 'size': (w, h),
        'color': color, 'label_loc': 'top',
    })


# --- Shared frame: boundary, entrance, checkout, WC -----------------------

def _frame(W, H, rng, is_main, wc_side='left', front_jitter=0.0):
    """Boundary walls (+entrance gap), checkout bank, WC.

    Returns (walls, door_position, door_side, front_h, keepouts) where
    ``keepouts`` are rects that fixture runs must avoid. ``wc_side``
    puts the restroom in the back-left or back-right corner."""
    walls: Dict[str, dict] = {}
    keepouts: List[Tuple[float, float, float, float]] = []

    walls['Wall_Left'] = {'position': (0, 0), 'size': (T, H)}
    walls['Wall_Right'] = {'position': (W - T, 0), 'size': (T, H)}
    walls['Wall_Top'] = {'position': (0, H - T), 'size': (W, T)}

    if not is_main:
        walls['Wall_Bot'] = {'position': (0, 0), 'size': (W, T)}
        return walls, None, None, max(1.2, H * 0.06), keepouts

    # Entrance on the FRONT (bottom) wall, offset right of centre -- the
    # standard configuration in the layouts the cited literature studies.
    gap_lo = W * 0.55
    gap_hi = W - T - DOOR_LEN - 0.6
    gap_x = rng.uniform(gap_lo, max(gap_lo, gap_hi))
    walls['Wall_Bot1'] = {'position': (0, 0), 'size': (gap_x, T)}
    walls['Wall_Bot2'] = {'position': (gap_x + DOOR_LEN, 0),
                          'size': (W - gap_x - DOOR_LEN, T)}
    door_position = (gap_x + DOOR_LEN / 2, 0)
    door_side = 'bottom'
    # Keep the doorway approach clear.
    keepouts.append((gap_x - 0.6, 0, DOOR_LEN + 1.2, 3.0))

    # Front power aisle depth (scaled to the shop, clamped to sane range).
    front_h = max(2.4, min(3.4, H * 0.18 + front_jitter))

    # Checkout bank left of the entrance, front-facing lanes.
    n_lanes = 3 if W >= 20 else 2
    bank_right = gap_x - 1.0
    x = bank_right - LANE_W
    placed = 0
    for k in range(n_lanes):
        if x < T + 0.8:
            break
        nm = 'Checkout' if placed == 0 else f'Checkout_Lane{placed + 1}'
        walls[nm] = {'position': (x, T + 0.25), 'size': (LANE_W, LANE_D),
                     'category': 'Checkout'}
        keepouts.append((x - 0.4, T, LANE_W + 0.8, LANE_D + 1.2))
        placed += 1
        x -= (LANE_W + LANE_GAP)

    # Restroom in a back corner (left by default; right as a variation).
    wc_x = T if wc_side == 'left' else W - T - WC_W
    walls['WC'] = {'position': (wc_x, H - T - WC_H), 'size': (WC_W, WC_H),
                   'category': 'WC'}
    keepouts.append((wc_x - 0.8, H - T - WC_H - 0.8,
                     WC_W + 1.6, WC_H + 0.8))

    return walls, door_position, door_side, front_h, keepouts


def _keepouts_1d(keepouts, horizontal, band_lo, band_hi):
    """Project the keepout rects that intersect a fixture band onto the
    run axis, as 1-D intervals."""
    out = []
    for kx, ky, kw, kh in keepouts:
        if horizontal:
            if ky < band_hi and ky + kh > band_lo:
                out.append((kx, kx + kw))
        else:
            if kx < band_hi and kx + kw > band_lo:
                out.append((ky, ky + kh))
    return out


def _wall_interval(side, W, H, front_h, keepouts):
    """(lo, hi, fixed, horizontal) of the usable fixture band on a wall."""
    if side == 'top':
        band_lo = H - T - FIX_D
        lo, hi = _trim_interval(T + 0.3, W - T - 0.3,
                                _keepouts_1d(keepouts, True, band_lo, H - T))
        return lo, hi, band_lo, True
    if side == 'left':
        lo, hi = _trim_interval(front_h, H - T - FIX_D - 0.8,
                                _keepouts_1d(keepouts, False, T, T + FIX_D))
        return lo, hi, T, False
    band_lo = W - T - FIX_D
    lo, hi = _trim_interval(front_h, H - T - FIX_D - 0.8,
                            _keepouts_1d(keepouts, False, band_lo, W - T))
    return lo, hi, band_lo, False


def _zone_for_run(side, lo, hi, fixed):
    if side == 'top':
        return (lo, fixed - MIN_AISLE - 0.2, hi - lo, FIX_D + MIN_AISLE + 0.2)
    if side == 'left':
        return (T, lo, FIX_D + MIN_AISLE + 0.2, hi - lo)
    return (fixed - MIN_AISLE - 0.2, lo, FIX_D + MIN_AISLE + 0.2, hi - lo)


def _multi_wall_run(depts, side, W, H, front_h, keepouts, items, zones):
    """One or more departments sharing a perimeter fixture wall; the
    usable band is subdivided among them with a 0.8 m break."""
    if not depts:
        return
    lo, hi, fixed, horizontal = _wall_interval(side, W, H, front_h, keepouts)
    k = len(depts)
    share = (hi - lo - (k - 1) * 0.8) / k if k else 0.0
    if share < MIN_SEG:
        depts = depts[:1]
        k, share = 1, hi - lo
    cur = lo
    for cat, cat_items in depts:
        items.update(_segments_along(cat_items, cur, cur + share,
                                     fixed, FIX_D, horizontal, cat))
        zones.append((cat, _zone_for_run(side, cur, cur + share, fixed)))
        cur += share + 0.8


# --- Archetypes -----------------------------------------------------------

def _grid(W, H, sections, rng, walls, items, zones, front_h, keepouts,
          vary=False):
    """Supermarket grid: perimeter departments (back wall always -- the
    classic staple-at-the-back -- plus left, and right on larger
    assortments), the rest as parallel vertical gondola runs."""
    cats = list(sections)
    use_right = len(cats) >= 7
    perim_sides = ['top', 'left'] + (['right'] if use_right else [])
    for side, dept in zip(perim_sides, cats):
        _multi_wall_run([dept], side, W, H, front_h, keepouts, items, zones)

    interior = cats[len(perim_sides):]
    if not interior:
        return

    # Interior gondola zone: right of the left fixtures, under the back
    # fixtures, above the front power aisle.
    x0 = T + FIX_D + MIN_AISLE
    x1 = W - T - ((FIX_D + MIN_AISLE) if use_right else MIN_AISLE)
    y0 = front_h
    y1 = H - T - FIX_D - MIN_AISLE
    zone_w = x1 - x0
    n = len(interior)

    # Fit as many gondolas as aisle width allows. Overflow departments
    # share a gondola (front half / back half); conversely, when there is
    # width to spare a department SPLITS across two gondolas (its items
    # divided between them) so big shops read as dense supermarkets
    # instead of a few runs in an empty hall.
    g_fit = max(1, int((zone_w + MIN_AISLE) // (GONDOLA_D + MIN_AISLE)))
    if g_fit >= 2 * n:
        assign = []
        for cat, its in interior:
            half = (len(its) + 1) // 2
            a, b = its[:half], its[half:]
            assign.append([(cat, a)])
            if b:
                assign.append([(cat, b)])
        g = len(assign)
    else:
        g = max(1, min(n, g_fit))
        assign = [[] for _ in range(g)]
        for i, dept in enumerate(interior):
            assign[i % g].append(dept)

    pitch = zone_w / g
    for gi in range(g):
        gx = x0 + gi * pitch + (pitch - GONDOLA_D) / 2
        depts = assign[gi]
        if not depts:
            continue
        run_lo, run_hi = y0, y1
        share = (run_hi - run_lo - (len(depts) - 1) * (MIN_AISLE * 0.6)) / len(depts)
        cur = run_lo
        for cat, cat_items in depts:
            segs = _segments_along(cat_items, cur, cur + share,
                                   gx, GONDOLA_D, False, cat)
            items.update(segs)
            zones.append((cat, (gx - (pitch - GONDOLA_D) / 2, cur,
                                pitch, share)))
            cur += share + MIN_AISLE * 0.6


def _racetrack(W, H, sections, rng, walls, items, zones, front_h, keepouts,
               vary=False):
    """Department-store loop: departments distributed round-robin over
    five slots -- the three perimeter walls (top / left / right) and the
    two faces of a central island block -- with a walkable loop corridor
    all the way around. Slots holding several departments subdivide
    their run, so the archetype scales from 5 to a dozen departments."""
    cats = list(sections)

    # Island sized so the loop corridor stays >= MIN_AISLE + 0.1 wide;
    # under ``vary`` its proportions jitter a little between runs.
    corr = MIN_AISLE + 0.1
    jw = rng.uniform(0.92, 1.08) if vary else 1.0
    jh = rng.uniform(0.92, 1.08) if vary else 1.0
    iw_max = W - 2 * (T + FIX_D + corr)
    ih_max = (H - T - FIX_D - corr) - (front_h + corr)
    iw = max(1.8, min(W * 0.36 * jw, iw_max))
    ih = max(1.5, min(H * 0.34 * jh, ih_max))
    ix = (W - iw) / 2
    iy = front_h + corr + max(0.0, (ih_max - ih) / 2)

    slot_order = ['top', 'left', 'right', 'islandL', 'islandR']
    assign = {s: [] for s in slot_order}
    for i, dept in enumerate(cats):
        assign[slot_order[i % len(slot_order)]].append(dept)

    for side in ('top', 'left', 'right'):
        _multi_wall_run(assign[side], side, W, H, front_h, keepouts,
                        items, zones)

    for side, sx, zx in (('islandL', ix, ix - 0.6),
                         ('islandR', ix + iw - 0.9, ix + iw / 2)):
        depts = assign[side]
        if not depts:
            continue
        lo, hi = iy + 0.2, iy + ih - 0.2
        k = len(depts)
        share = (hi - lo - (k - 1) * 0.6) / k
        if share < MIN_SEG:
            depts, k, share = depts[:1], 1, hi - lo
        cur = lo
        for cat, cat_items in depts:
            items.update(_segments_along(cat_items, cur, cur + share,
                                         sx, 0.9, False, cat))
            zones.append((cat, (zx, cur, iw / 2 + 0.6, share)))
            cur += share + 0.6


def _freeform(W, H, sections, rng, walls, items, zones, front_h, keepouts,
              vary=False):
    """Boutique free-flow: wall racks on the back and left walls plus
    free-standing display tables on a relaxed, jittered grid. Scales to
    many departments by tiling the floor in up to two zone rows and by
    packing tables in multiple columns inside a zone."""
    cats = list(sections)
    _multi_wall_run([cats[0]], 'top', W, H, front_h, keepouts, items, zones)
    _multi_wall_run([cats[1]], 'left', W, H, front_h, keepouts, items, zones)

    floor = cats[2:]
    if not floor:
        return
    x0 = T + FIX_D + MIN_AISLE * 0.9
    x1 = W - T - 0.9
    y0 = front_h
    y1 = H - T - FIX_D - MIN_AISLE * 0.9

    tab_w, tab_h = 1.5, 0.9
    nf = len(floor)
    n_rows = 2 if (nf > 4 and (y1 - y0) >= 2 * (tab_h + 1.6) + 0.8) else 1
    n_cols = int(math.ceil(nf / n_rows))
    col_w = (x1 - x0) / n_cols
    row_h = (y1 - y0) / n_rows

    for ci, (cat, cat_items) in enumerate(floor):
        r_i, c_i = divmod(ci, n_cols)
        cx0 = x0 + c_i * col_w
        cy0 = y0 + r_i * row_h
        zones.append((cat, (cx0, cy0, col_w, row_h)))
        k = len(cat_items)

        # Tables in up to two columns; column split only when the spacing
        # keeps >= 1.3 m clear between neighbouring tables after jitter.
        # Under ``vary`` a feasible split is taken or skipped at random.
        can_split = (k > 3 and col_w >= 2 * (tab_w + 1.3))
        t_cols = 2 if (can_split and (not vary or rng.random() < 0.7)) else 1
        t_rows = int(math.ceil(k / t_cols))
        slot_w = col_w / t_cols
        slot_h = row_h / t_rows
        if slot_h < tab_h + 1.2 or slot_w < tab_w + 1.0:
            # Not enough clearance for free tables -- fall back to a
            # vertical shelving run (still free-standing).
            gx = cx0 + (col_w - 0.9) / 2
            items.update(_segments_along(cat_items, cy0 + 0.3,
                                         cy0 + row_h - 0.3, gx, 0.9,
                                         False, cat))
            continue
        for ki, nm in enumerate(cat_items):
            rr, cc = divmod(ki, t_cols)
            jx = rng.uniform(-0.25, 0.25) + (0.3 if (rr % 2 and t_cols == 1) else 0.0)
            jy = rng.uniform(-0.2, 0.2)
            px = cx0 + cc * slot_w + (slot_w - tab_w) / 2 + jx
            py = cy0 + rr * slot_h + (slot_h - tab_h) / 2 + jy
            px = max(cx0 + cc * slot_w + 0.5,
                     min(px, cx0 + (cc + 1) * slot_w - tab_w - 0.5))
            py = max(cy0 + rr * slot_h + 0.3,
                     min(py, cy0 + (rr + 1) * slot_h - tab_h - 0.3))
            items[nm] = {'position': (px, py), 'size': (tab_w, tab_h),
                         'category': cat}


# --- Entry point ----------------------------------------------------------

def generate_architecture(width: float,
                          height: float,
                          shop_type: str,
                          sections: Sequence[Tuple[str, Sequence[str]]],
                          rng=None,
                          is_main_floor: bool = True,
                          vary: bool = True) -> dict:
    """Generate a realistic floor plan for ``shop_type``.

    ``sections`` is the [(section_name, [item names]), ...] list from the
    Generate dialog (or ``scale_catalog``). Deterministic given the same
    ``rng`` state.

    ``vary=True`` (the Generate-button default) adds structural variety
    between runs of the same shop type: the section-to-slot assignment is
    shuffled, the restroom corner and front-aisle depth vary, and the
    racetrack island proportions jitter. ``vary=False`` (used by the
    dataset-import path) preserves the caller's section order exactly, so
    e.g. a naive-baseline alphabetical ordering survives into the wall
    insertion order."""
    rng = rng or _random_mod
    W, H = float(width), float(height)
    archetype = ARCHETYPE_BY_SHOP.get(shop_type, 'grid')

    secs = list(sections)
    wc_side = 'left'
    front_jitter = 0.0
    if vary:
        rng.shuffle(secs)
        wc_side = 'left' if rng.random() < 0.5 else 'right'
        front_jitter = rng.uniform(-0.35, 0.35)

    walls, door_position, door_side, front_h, keepouts = _frame(
        W, H, rng, is_main_floor, wc_side=wc_side, front_jitter=front_jitter)

    items: Dict[str, dict] = {}
    zones: List[Tuple[str, Tuple[float, float, float, float]]] = []

    build = {'grid': _grid, 'racetrack': _racetrack,
             'freeform': _freeform}[archetype]
    build(W, H, secs, rng, walls, items, zones, front_h, keepouts,
          vary=vary)

    # Emit section zones (clamped inside the boundary), colored from the
    # fixed palette so departments are visually distinct + reproducible.
    for i, (cat, (zx, zy, zw, zh)) in enumerate(zones):
        zx2, zy2 = max(T, zx), max(T, zy)
        zw2 = min(W - T, zx + zw) - zx2
        zh2 = min(H - T, zy + zh) - zy2
        if zw2 <= 0.2 or zh2 <= 0.2:
            continue
        name, wall = _zone(cat, zx2, zy2, zw2, zh2,
                           SECTION_PALETTE[i % len(SECTION_PALETTE)])
        # Two zones for the same category (split gondola) get suffixes.
        if name in walls:
            name = f'{name}_{i}'
        walls[name] = wall

    return {
        'walls': walls,
        'items': items,
        'door_position': door_position,
        'door_side': door_side,
        'archetype': archetype,
    }


# --- Section / item catalog + size-based scaling --------------------------
# Up to 10 sections x 8 items per shop type. ``scale_catalog`` picks how
# many of each the shop can carry from the floor area the user chose, so
# a bigger retail space automatically gets a richer assortment. Item
# names are unique within a shop type and deliberately do not collide
# with the impulse-SKU lists in viz_generate.

FULL_CATALOG: Dict[str, List[Tuple[str, List[str]]]] = {
    'Grocery Store': [
        ('Fresh Produce', ["Apples", "Bananas", "Carrots", "Tomatoes",
                           "Lettuce", "Potatoes", "Onions", "Grapes"]),
        ('Dairy & Refrigerated', ["Milk", "Cheese", "Yogurt", "Butter",
                                  "Eggs", "Cream", "Deli Meats", "Fresh Juice"]),
        ('Pantry & Dry Goods', ["Rice", "Pasta", "Canned Beans", "Flour",
                                "Sugar", "Cereal", "Canned Soup", "Olive Oil"]),
        ('Beverages', ["Water", "Soda", "Juice", "Iced Tea",
                       "Sports Drinks", "Sparkling Water", "Beer", "Wine"]),
        ('Snacks', ["Potato Chips", "Chocolate Boxes", "Nuts", "Crackers",
                    "Popcorn", "Granola Bars", "Pretzels", "Dried Fruit"]),
        ('Frozen Foods', ["Frozen Pizza", "Ice Cream", "Frozen Vegetables",
                          "Frozen Meals", "Frozen Fish", "Frozen Fries",
                          "Frozen Berries", "Frozen Desserts"]),
        ('Bakery', ["Fresh Bread", "Baguettes", "Croissants", "Muffins",
                    "Bagels", "Cake Slices", "Donuts", "Pita"]),
        ('Meat & Seafood', ["Chicken", "Ground Beef", "Pork Chops", "Salmon",
                            "Shrimp", "Sausages", "Turkey", "Steaks"]),
        ('Household', ["Detergent", "Paper Towels", "Dish Soap", "Sponges",
                       "Trash Bags", "Cleaners", "Foil & Wrap", "Air Freshener"]),
        ('Health & Beauty', ["Shampoo", "Toothpaste", "Soap Bars", "Lotion",
                             "Razors", "Deodorant", "Cotton Pads", "Sunscreen"]),
    ],
    'Bookstore': [
        ('Fiction', ["Novels", "Short Stories", "Poetry", "Mystery",
                     "Sci-Fi", "Fantasy", "Romance", "Thrillers"]),
        ('Non-Fiction', ["Biographies", "History", "Science", "Philosophy",
                         "Politics", "True Crime", "Economics", "Nature"]),
        ('Children', ["Picture Books", "Young Adult", "Coloring Books",
                      "Early Readers", "Middle Grade", "Pop-Up Books",
                      "Activity Books", "Board Books"]),
        ('Magazines & Press', ["Fashion Mags", "Tech Mags", "Health Mags",
                               "News Weeklies", "Comics", "Art Journals",
                               "Travel Mags", "Puzzle Books"]),
        ('Stationery', ["Notebooks", "Fountain Pens", "Calendars", "Planners",
                        "Sketchpads", "Envelopes", "Washi Tape", "Folders"]),
        ('Art & Design', ["Art Books", "Photography", "Architecture",
                          "Graphic Novels", "Design Annuals", "Craft Guides",
                          "Fashion Books", "Museum Editions"]),
        ('Academic', ["Textbooks", "Reference", "Dictionaries", "Test Prep",
                      "Language Courses", "Atlases", "Encyclopedias", "Law"]),
        ('Cooking & Home', ["Cookbooks", "Baking Books", "Wine Guides",
                            "Gardening", "Interior Design", "DIY Manuals",
                            "Diet Books", "Coffee-Table Books"]),
        ('Audio & Media', ["Audiobooks", "Vinyl Records", "Board Games",
                           "Jigsaw Puzzles", "E-Readers", "Music CDs",
                           "Film Books", "Calendars Deluxe"]),
        ('Local & Travel', ["City Guides", "Maps", "Local Authors",
                            "Souvenirs Books", "Hiking Guides", "Phrasebooks",
                            "Travel Essays", "Postcard Books"]),
    ],
    'Electronics Store': [
        ('Mobile Phones', ["Smartphone A", "Feature Phone", "Smartphone B",
                           "Flagship Phone", "Budget Phone", "Refurb Phones",
                           "Kids Phone", "Rugged Phone"]),
        ('Computers', ["Laptop X", "Tablet", "Desktop Y", "Gaming Laptop",
                       "Ultrabook", "Mini PC", "Workstation", "Chromebook"]),
        ('Audio', ["Headphones", "Speakers", "Microphone", "Soundbars Mini",
                   "Turntables", "Studio Monitors", "Wireless Buds Pro",
                   "Home Audio Amp"]),
        ('TV & Theatre', ["LED TVs", "OLED TVs", "Projectors", "TV Stands",
                          "Streaming Boxes", "AV Receivers", "Antennas",
                          "Home Cinema Kits"]),
        ('Accessories', ["Charging Docks", "Laptop Sleeves", "Camera Bags",
                         "Keyboards", "Mice", "Monitor Arms", "Hubs",
                         "Surge Protectors"]),
        ('Gaming', ["Consoles", "Controllers", "Gaming Headsets", "VR Kits",
                    "Racing Wheels", "Game Titles", "Gaming Chairs",
                    "Capture Cards"]),
        ('Cameras', ["DSLR Bodies", "Mirrorless", "Lenses", "Tripods",
                     "Action Cams", "Drones", "Instant Cameras", "Lighting Kits"]),
        ('Smart Home', ["Smart Bulbs", "Smart Plugs", "Thermostats",
                        "Security Cams", "Video Doorbells", "Smart Locks",
                        "Hubs & Bridges", "Robot Vacuums"]),
        ('Appliances', ["Microwaves", "Coffee Makers", "Blenders", "Kettles",
                        "Air Fryers", "Toasters", "Vacuums", "Fans"]),
        ('Networking', ["Routers", "Mesh Systems", "Switches Net", "NAS Drives",
                        "Range Extenders", "Modems", "Ethernet Kits",
                        "Powerline Kits"]),
    ],
    'Clothing Store': [
        ('Men', ["Shirts", "Trousers", "Jackets", "Suits",
                 "Polo Shirts", "Sweaters", "Shorts Men", "Coats Men"]),
        ('Women', ["Dresses", "Blouses", "Skirts", "Cardigans",
                   "Leggings", "Blazers", "Jumpsuits", "Coats Women"]),
        ('Kids', ["T-Shirts", "Shorts", "Jeans Kids", "Hoodies Kids",
                  "School Wear", "Baby Sets", "Pajamas Kids", "Rainwear"]),
        ('Shoes', ["Sneakers", "Boots", "Loafers", "Heels",
                   "Sandals", "Running Shoes", "Dress Shoes", "Slippers"]),
        ('Sale', ["Clearance1", "Clearance2", "Clearance3", "Clearance4",
                  "Last Sizes", "Season End", "Outlet Rack", "Final Reductions"]),
        ('Sportswear', ["Track Pants", "Sports Tops", "Gym Shorts",
                        "Training Jackets", "Yoga Wear", "Swim Wear",
                        "Compression Wear", "Team Jerseys"]),
        ('Denim', ["Slim Jeans", "Regular Jeans", "Denim Jackets",
                   "Denim Skirts", "Bootcut Jeans", "Denim Shirts",
                   "Raw Denim", "Stretch Jeans"]),
        ('Underwear & Sleep', ["Boxers", "Briefs", "Bras", "Sleep Sets",
                               "Thermal Wear", "Robes", "Sock Packs",
                               "Tights"]),
        ('Bags & Luggage', ["Handbags", "Backpacks", "Totes", "Wallets",
                            "Suitcases", "Duffels", "Crossbody Bags",
                            "Laptop Bags"]),
        ('Formal & Occasion', ["Evening Gowns", "Tuxedos", "Ties & Bowties",
                               "Waistcoats", "Formal Shirts", "Clutches",
                               "Dress Pants", "Occasion Dresses"]),
    ],
    'Pharmacy': [
        ('Prescription', ["Drug A", "Drug B", "Insulin", "Antibiotics",
                          "Heart Medication", "Asthma Inhalers",
                          "Blood Pressure Meds", "Thyroid Meds"]),
        ('OTC', ["Pain Relievers", "Cold/Flu", "Allergy Meds", "Antacids",
                 "Cough Syrup", "Sleep Aids", "Motion Sickness", "Fever Reducers"]),
        ('Wellness', ["Multivitamins", "Supplements", "Protein Bars",
                      "Omega-3", "Probiotics", "Electrolytes",
                      "Herbal Remedies", "Energy Gels"]),
        ('Personal Care', ["Shampoo Rx", "Toothpaste Rx", "Deodorant Rx",
                           "Mouthwash", "Dental Floss", "Body Wash",
                           "Hair Brushes", "Nail Care"]),
        ('Medical Equipment', ["Bandages Roll", "Thermometers", "Wheelchair",
                               "Blood Pressure Monitors", "Crutches",
                               "Glucose Meters", "Nebulizers", "Pulse Oximeters"]),
        ('Baby & Child', ["Diapers", "Baby Formula", "Baby Wipes",
                          "Teething Gel", "Kids Vitamins", "Baby Lotion",
                          "Pacifiers", "Baby Monitors"]),
        ('Skin Care', ["Moisturizers", "Acne Care", "Eczema Cream",
                       "SPF Cream", "Anti-Aging", "Cleansers",
                       "Face Masks", "Serums"]),
        ('Optical & Hearing', ["Reading Glasses Rack", "Contact Solution",
                               "Eye Wash", "Lens Cases", "Hearing Aid Batteries",
                               "Ear Drops", "Eye Masks", "Blue-Light Glasses"]),
        ('First Aid', ["Gauze", "Antiseptic", "Burn Gel", "Cold Packs",
                       "Medical Tape", "Splints", "Tweezers Kits",
                       "First Aid Kits"]),
        ('Home Health', ["Compression Socks", "Heating Pads", "Massagers",
                         "Pill Organizers", "Bath Safety", "Orthopedic Pillows",
                         "TENS Units", "Scales"]),
    ],
    'Hardware Store': [
        ('Hand Tools', ["Hammers", "Screwdrivers", "Wrenches", "Pliers",
                        "Hand Saws", "Chisels", "Clamps", "Levels"]),
        ('Paint', ["Interior Paint", "Exterior Paint", "Brushes", "Rollers",
                   "Primer", "Stains", "Spray Paint", "Drop Cloths"]),
        ('Electrical', ["Wiring", "Switches", "Outlets", "Breakers",
                        "Conduit", "Light Fixtures", "Extension Cords",
                        "Smart Switches"]),
        ('Plumbing', ["Pipes", "Fittings", "Valves", "Faucets",
                      "Drain Cleaner", "Seals & Washers", "Water Heaters",
                      "Hose Kits"]),
        ('Garden', ["Seeds", "Soil", "Plants", "Fertilizer",
                    "Garden Hoses", "Pruners", "Planters", "Mulch"]),
        ('Power Tools', ["Drills", "Circular Saws", "Sanders", "Grinders",
                         "Jigsaws", "Impact Drivers", "Rotary Tools",
                         "Tool Combos"]),
        ('Fasteners', ["Wood Screws", "Bolts", "Nails", "Anchors",
                       "Washers", "Hinges", "Brackets", "Hooks"]),
        ('Building Materials', ["Lumber", "Drywall", "Plywood", "Insulation",
                                "Cement Bags", "Tiles", "Molding", "Rebar"]),
        ('Safety & Workwear', ["Hard Hats", "Ear Protection", "Respirators",
                               "Knee Pads", "Safety Vests", "Steel-Toe Boots",
                               "Goggles", "First Aid Wall Kits"]),
        ('Storage & Organization', ["Tool Boxes", "Shelving Units", "Bins",
                                    "Pegboards", "Cabinets", "Ladders",
                                    "Workbenches", "Sawhorses"]),
    ],
    'Cafe': [
        ('Seating', ["Tables", "Chairs", "Benches", "Window Bar",
                     "Lounge Sofas", "Outdoor Tables", "High Stools",
                     "Corner Booths"]),
        ('Counter', ["Cash Register", "Display", "POS System",
                     "Order Pickup", "Menu Boards", "Barista Station",
                     "Tip Jar Stand", "Loyalty Kiosk"]),
        ('Kitchen', ["Coffee Machine", "Oven", "Refrigerator", "Grinder",
                     "Dishwasher", "Prep Station", "Freezer", "Ice Machine"]),
        ('Pastry Case', ["Cakes", "Tarts", "Bread Loaves", "Brownies",
                         "Macarons", "Scones", "Cheesecakes", "Eclairs"]),
        ('Retail Shelf', ["Whole Beans", "Ground Coffee", "Teas", "Mugs Retail",
                          "Tumblers", "Brew Kits", "Syrups", "Chocolate Gifts"]),
        ('Deli & Sandwiches', ["Paninis", "Wraps", "Salads", "Soups",
                               "Baguette Subs", "Quiches", "Bagel Melts",
                               "Breakfast Boxes"]),
        ('Beverage Station', ["Juice Bar", "Smoothie Station", "Iced Coffee",
                              "Kombucha Tap", "Milk Alternatives", "Water Station",
                              "Hot Chocolate", "Seasonal Drinks"]),
        ('Kids & Family', ["Kids Corner", "High Chairs", "Board Game Shelf",
                           "Kids Menu Stand", "Crayon Station", "Family Booth",
                           "Story Books", "Baby Change Station"]),
        ('Workspace', ["Laptop Bar", "Power Outlets Desk", "Meeting Nook",
                       "Phone Booths", "Print Station", "Reading Rack",
                       "Standing Desks", "Quiet Zone"]),
        ('Events & Local', ["Local Art Wall", "Event Stage", "Community Board",
                            "Vinyl Corner", "Book Swap", "Merch Stand",
                            "Plant Shelf", "Workshop Table"]),
    ],
}


def scale_catalog(shop_type: str, width: float, height: float,
                  rng=None) -> List[Tuple[str, List[str]]]:
    """Pick how many sections and items-per-section a shop of the given
    dimensions carries. Bigger floor area -> richer assortment, so a
    large retail space generates like a real one instead of a 5-shelf
    kiosk. At the smallest sizes this reduces to the classic 5x3.

    With an ``rng``, the assortment itself varies between runs: the
    section/item counts jitter by one, and beyond the first two "core"
    entries the sections and items are SAMPLED from the full catalog
    (order-preserving) -- so two generated groceries differ in which
    departments and products they carry, not just where the door is."""
    cat = FULL_CATALOG.get(shop_type)
    if not cat:
        cat = next(iter(FULL_CATALOG.values()))
    area = float(width) * float(height)
    n_sections = int(round(3 + area / 60.0))
    n_items = int(round(2 + area / 110.0))
    if rng is not None:
        n_sections += rng.choice((-1, 0, 0, 1))
        n_items += rng.choice((-1, 0, 0, 1))
    n_sections = max(5, min(len(cat), n_sections))
    n_items = max(3, min(8, n_items))

    def _sample_prefix(seq, n, n_core):
        """First ``n_core`` entries always in; the rest sampled from the
        remainder, original order preserved."""
        if rng is None or n >= len(seq):
            return list(seq[:n])
        core = list(seq[:n_core])
        rest = list(seq[n_core:])
        picked_idx = sorted(rng.sample(range(len(rest)),
                                       max(0, n - len(core))))
        return core + [rest[i] for i in picked_idx]

    picked_sections = _sample_prefix(cat, n_sections, n_core=2)
    return [(sec, [str(x) for x in _sample_prefix(itms, n_items, n_core=2)])
            for sec, itms in picked_sections]


def dimensions_for_assortment(n_sections: int,
                              max_items_per_section: int,
                              aspect: float = 1.4) -> Tuple[float, float]:
    """Shop (width, height) sized to carry the given assortment.

    Inverse of the ``scale_catalog`` area heuristics, plus a 15% safety
    margin so the archetype engines never have to trim item segments.
    Used by the dataset-import path: the floor dimensions follow from
    how many sections/items the DATASET has, not from a fixed default."""
    n_sections = max(1, int(n_sections))
    n_items = max(1, int(max_items_per_section))
    area = max(60.0 * (n_sections - 3),
               110.0 * (n_items - 2),
               220.0) * 1.15
    w = math.sqrt(area * aspect)
    h = area / w
    w = max(14.0, min(44.0, w))
    h = max(10.0, min(32.0, h))
    return round(w, 1), round(h, 1)
