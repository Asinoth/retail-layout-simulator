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
                       'category': section_name,
                       'zone': 'Section_...' wall containing the centre}},
      'door_position': (x, y) | None,
      'door_side': 'bottom' | None,
      'archetype': 'grid' | 'racetrack' | 'freeform',
      'dropped': [requested item names that did not fit],
      'keepouts': [(x, y, w, h), ...],
      'door_keepout': (x, y, w, h) | None,
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

# Smallest floor the archetypes can furnish. Below this the mandatory
# frame (entrance approach, checkout bank, restroom, back-wall fixtures
# and their aisles) leaves no room for an interior, and the archetypes
# start folding fixtures onto one another instead of failing.
MIN_FLOOR_W, MIN_FLOOR_H = 12.0, 9.0

# Build passes allowed while an assortment is trimmed to what the floor
# holds (the first build plus up to three trimmed rebuilds).
MAX_FIT_PASSES = 4

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


def _run_capacity(span: float) -> int:
    """How many MIN_SEG segments (SEG_GAP apart) a run of ``span`` holds."""
    if span < MIN_SEG:
        return 0
    return int((span + SEG_GAP) // (MIN_SEG + SEG_GAP))


def _segments_along(names: Sequence[str], lo: float, hi: float,
                    fixed: float, depth: float, horizontal: bool,
                    category: str) -> Dict[str, dict]:
    """Place item segments along the run [lo, hi].

    ``fixed`` is the run's constant coordinate (y for horizontal runs,
    x for vertical). If the run cannot hold every name at MIN_SEG it
    places as many as fit; ``generate_architecture`` reports the rest in
    ``dropped``.
    Segments are capped at MAX_SEG and spread evenly so a short list on
    a long run still reads as continuous shelving, not one giant slab."""
    span = hi - lo
    if not names or span < MIN_SEG:
        return {}
    cap = _run_capacity(span)
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

    Returns (walls, door_position, door_side, front_h, keepouts,
    door_keepout) where ``keepouts`` are rects that fixture runs must
    avoid and ``door_keepout`` is the doorway-approach one of them, named
    separately because callers that place their own fixtures (the
    impulse racks) must keep the entrance clear without also avoiding the
    gaps between checkout lanes. ``wc_side`` puts the restroom in the
    back-left or back-right corner."""
    walls: Dict[str, dict] = {}
    keepouts: List[Tuple[float, float, float, float]] = []

    walls['Wall_Left'] = {'position': (0, 0), 'size': (T, H)}
    walls['Wall_Right'] = {'position': (W - T, 0), 'size': (T, H)}
    walls['Wall_Top'] = {'position': (0, H - T), 'size': (W, T)}

    if not is_main:
        walls['Wall_Bot'] = {'position': (0, 0), 'size': (W, T)}
        return walls, None, None, max(1.2, H * 0.06), keepouts, None

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
    door_keepout = (gap_x - 0.6, 0, DOOR_LEN + 1.2, 3.0)
    keepouts.append(door_keepout)

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

    return walls, door_position, door_side, front_h, keepouts, door_keepout


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
    # Side runs stop where the back cross-aisle begins (the WC corner's
    # keepout already ends them there), so the side and back perimeter
    # zones meet at the corner instead of overlapping.
    if side == 'left':
        lo, hi = _trim_interval(front_h, H - T - FIX_D - MIN_AISLE,
                                _keepouts_1d(keepouts, False, T, T + FIX_D))
        return lo, hi, T, False
    band_lo = W - T - FIX_D
    lo, hi = _trim_interval(front_h, H - T - FIX_D - MIN_AISLE,
                            _keepouts_1d(keepouts, False, band_lo, W - T))
    return lo, hi, band_lo, False


def _zone_for_run(side, lo, hi, fixed, aisle=MIN_AISLE):
    """Zone of a perimeter run: the fixture band plus the aisle in front
    of it, ending exactly where the archetype's interior zones begin."""
    if side == 'top':
        return (lo, fixed - aisle, hi - lo, FIX_D + aisle)
    if side == 'left':
        return (T, lo, FIX_D + aisle, hi - lo)
    return (fixed - aisle, lo, FIX_D + aisle, hi - lo)


def _slot_share(length, k, brk):
    """Equal share of a run of ``length`` split among ``k`` departments
    with ``brk`` breaks between them."""
    return (length - (k - 1) * brk) / k


def _slot_fits(depts, length, brk):
    """True when every department's items fit its equal share of the run."""
    if not depts:
        return True
    cap = _run_capacity(_slot_share(length, len(depts), brk))
    return all(len(its) <= cap for _, its in depts)


def _multi_wall_run(depts, side, W, H, front_h, keepouts, items, zones,
                    aisle=MIN_AISLE):
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
        zones.append((cat, _zone_for_run(side, cur, cur + share, fixed,
                                         aisle)))
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
    # instead of a few runs in an empty hall. Each gondola is centred in
    # a pitch cell of zone_w / g, so the aisle between neighbours is
    # pitch - GONDOLA_D; capping g at zone_w // (GONDOLA_D + MIN_AISLE)
    # keeps that aisle >= MIN_AISLE.
    g_fit = max(1, int(zone_w // (GONDOLA_D + MIN_AISLE)))
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
        # A gondola in front of the checkout bank or the doorway starts
        # behind that approach, the same clearance the perimeter runs keep.
        run_lo = max([y0] + [ky + kh for kx, ky, kw, kh in keepouts
                             if kx < gx + GONDOLA_D and kx + kw > gx
                             and ky < y0 + MIN_AISLE])
        run_hi = y1
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

    # Round-robin can give a short island face more items than its run
    # holds. Departments on such a slot move to the slot that keeps the
    # largest share after taking them; slots that already fit are left
    # alone, so shops with room keep the round-robin distribution.
    run_len = {'islandL': ih - 0.4, 'islandR': ih - 0.4}
    for s in ('top', 'left', 'right'):
        lo, hi, _, _ = _wall_interval(s, W, H, front_h, keepouts)
        run_len[s] = hi - lo
    brk = {'top': 0.8, 'left': 0.8, 'right': 0.8,
           'islandL': 0.6, 'islandR': 0.6}
    moved = True
    while moved:
        moved = False
        for s in slot_order:
            if _slot_fits(assign[s], run_len[s], brk[s]):
                continue
            for di in range(len(assign[s]) - 1, -1, -1):
                dept = assign[s][di]
                targets = [t for t in slot_order if t != s and _slot_fits(
                    assign[t] + [dept], run_len[t], brk[t])]
                if not targets:
                    continue
                best = max(targets, key=lambda t: _slot_share(
                    run_len[t], len(assign[t]) + 1, brk[t]))
                assign[best].append(assign[s].pop(di))
                moved = True
                break

    # A face whose departments still do not fit gets a taller island, up
    # to the height the loop corridor allows.
    need = 0.0
    for s in ('islandL', 'islandR'):
        if assign[s]:
            k = len(assign[s])
            m = max(len(its) for _, its in assign[s])
            need = max(need, k * max(0.0, m * MIN_SEG + (m - 1) * SEG_GAP)
                       + (k - 1) * 0.6 + 0.4 + 1e-6)
    if need > ih:
        ih = max(ih, min(ih_max, need))
        iy = front_h + corr + max(0.0, (ih_max - ih) / 2)

    for side in ('top', 'left', 'right'):
        _multi_wall_run(assign[side], side, W, H, front_h, keepouts,
                        items, zones)

    # The island zones reach into the loop corridor by up to 0.6 m, but
    # never past where the side perimeter zones end.
    reach = min(0.6, max(0.0, ix - (T + FIX_D + MIN_AISLE)))
    for side, sx, zx in (('islandL', ix, ix - reach),
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
            zones.append((cat, (zx, cur, iw / 2 + reach, share)))
            cur += share + 0.6


def _freeform(W, H, sections, rng, walls, items, zones, front_h, keepouts,
              vary=False):
    """Boutique free-flow: wall racks on the back and left walls plus
    free-standing display tables on a relaxed, jittered grid. Scales to
    many departments by tiling the floor in up to two zone rows and by
    packing tables in multiple columns inside a zone."""
    cats = list(sections)
    aisle = MIN_AISLE * 0.9
    # Slicing rather than indexing: a boutique with one or two departments
    # simply leaves the second wall (and the floor) empty.
    _multi_wall_run(cats[:1], 'top', W, H, front_h, keepouts, items, zones,
                    aisle=aisle)
    _multi_wall_run(cats[1:2], 'left', W, H, front_h, keepouts, items, zones,
                    aisle=aisle)

    floor = cats[2:]
    if not floor:
        return
    x0 = T + FIX_D + aisle
    x1 = W - T - 0.9
    y0 = front_h
    y1 = H - T - FIX_D - aisle

    tab_w, tab_h = 1.5, 0.9
    clear = 1.3             # clear gap kept between neighbouring tables
    nf = len(floor)
    n_rows = 2 if (nf > 4 and (y1 - y0) >= 2 * (tab_h + 1.6) + 0.8) else 1
    n_cols = int(math.ceil(nf / n_rows))
    # A zone column narrower than a shelving run plus its clearance puts
    # the neighbouring runs closer together than a customer can pass, so
    # the floor is split by width first and the overflow goes into extra
    # zone rows (a narrow, deep unit becomes a long file of zones).
    max_cols = max(1, int((x1 - x0) // (0.9 + clear)))
    if n_cols > max_cols:
        n_cols = max_cols
        n_rows = int(math.ceil(nf / n_cols))
    col_w = (x1 - x0) / n_cols
    row_h = (y1 - y0) / n_rows

    for ci, (cat, cat_items) in enumerate(floor):
        r_i, c_i = divmod(ci, n_cols)
        cx0 = x0 + c_i * col_w
        cy0 = y0 + r_i * row_h
        zone_h = row_h
        if r_i == 0:
            # Front-row zones start behind the checkout and doorway
            # approaches, the same clearance the perimeter runs keep.
            cy1 = cy0 + row_h
            cy0 = max([cy0] + [ky + kh for kx, ky, kw, kh in keepouts
                               if kx < cx0 + col_w and kx + kw > cx0
                               and ky < y0 + MIN_AISLE])
            zone_h = cy1 - cy0
        zones.append((cat, (cx0, cy0, col_w, zone_h)))
        k = len(cat_items)

        # Tables in up to two columns. Every table stays clear/2 inside its
        # slot, so neighbours in this zone or the next are >= ``clear``
        # apart whatever the jitter. Under ``vary`` a feasible split is
        # taken or skipped at random.
        can_split = (k > 3 and col_w >= 2 * (tab_w + clear))
        t_cols = 2 if (can_split and (not vary or rng.random() < 0.7)) else 1
        t_rows = int(math.ceil(k / t_cols))
        slot_w = col_w / t_cols
        slot_h = zone_h / t_rows
        if slot_h < tab_h + clear or slot_w < tab_w + clear:
            # Not enough clearance for free tables -- fall back to vertical
            # shelving runs (still free-standing): one run centred in the
            # zone, or as many parallel runs as the items need while
            # neighbouring runs keep ``clear`` between them.
            per_run = max(1, _run_capacity(zone_h - 0.6))
            n_runs = max(1, min(int(math.ceil(k / per_run)),
                                int(col_w // (0.9 + clear))))
            pitch = col_w / n_runs
            per = int(math.ceil(k / n_runs))
            for j in range(n_runs):
                gx = cx0 + (j + 0.5) * pitch - 0.45
                items.update(_segments_along(cat_items[j * per:(j + 1) * per],
                                             cy0 + 0.3, cy0 + zone_h - 0.3,
                                             gx, 0.9, False, cat))
            continue
        for ki, nm in enumerate(cat_items):
            rr, cc = divmod(ki, t_cols)
            jx = rng.uniform(-0.25, 0.25) + (0.3 if (rr % 2 and t_cols == 1) else 0.0)
            jy = rng.uniform(-0.2, 0.2)
            px = cx0 + cc * slot_w + (slot_w - tab_w) / 2 + jx
            py = cy0 + rr * slot_h + (slot_h - tab_h) / 2 + jy
            px = max(cx0 + cc * slot_w + clear / 2,
                     min(px, cx0 + (cc + 1) * slot_w - tab_w - clear / 2))
            py = max(cy0 + rr * slot_h + clear / 2,
                     min(py, cy0 + (rr + 1) * slot_h - tab_h - clear / 2))
            items[nm] = {'position': (px, py), 'size': (tab_w, tab_h),
                         'category': cat}


# --- Entry point ----------------------------------------------------------

def _build_plan(W, H, shop_type, sections, rng, is_main_floor, vary) -> dict:
    """One generation pass: frame, archetype, zones. See
    ``generate_architecture`` for the output contract."""
    archetype = ARCHETYPE_BY_SHOP.get(shop_type, 'grid')

    secs = list(sections)
    requested = [nm for _, names in secs for nm in names]
    wc_side = 'left'
    front_jitter = 0.0
    if vary:
        rng.shuffle(secs)
        wc_side = 'left' if rng.random() < 0.5 else 'right'
        front_jitter = rng.uniform(-0.35, 0.35)

    walls, door_position, door_side, front_h, keepouts, door_keepout = _frame(
        W, H, rng, is_main_floor, wc_side=wc_side, front_jitter=front_jitter)

    items: Dict[str, dict] = {}
    zones: List[Tuple[str, Tuple[float, float, float, float]]] = []

    build = {'grid': _grid, 'racetrack': _racetrack,
             'freeform': _freeform}[archetype]
    build(W, H, secs, rng, walls, items, zones, front_h, keepouts,
          vary=vary)

    # Emit section zones (clamped inside the boundary), colored from the
    # fixed palette so departments are visually distinct + reproducible.
    zone_walls: Dict[str, List[Tuple[str, Tuple[float, float, float, float]]]] = {}
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
        zone_walls.setdefault(cat, []).append((name, (zx2, zy2, zw2, zh2)))

    # A department split over several fixtures has one zone per fixture,
    # so the category alone does not say which zone an item belongs to.
    # Stamp the zone containing the item's centre (the nearest zone of its
    # category if clamping left the centre outside every one).
    for it in items.values():
        cx = it['position'][0] + it['size'][0] / 2.0
        cy = it['position'][1] + it['size'][1] / 2.0
        best, best_d = None, math.inf
        for name, (zx, zy, zw, zh) in zone_walls.get(it['category'], []):
            d = math.hypot(max(zx - cx, 0.0, cx - (zx + zw)),
                           max(zy - cy, 0.0, cy - (zy + zh)))
            if d < best_d:
                best, best_d = name, d
        it['zone'] = best

    return {
        'walls': walls,
        'items': items,
        'door_position': door_position,
        'door_side': door_side,
        'archetype': archetype,
        'dropped': [nm for nm in requested if nm not in items],
        'keepouts': list(keepouts),
        'door_keepout': door_keepout,
    }


def generate_architecture(width: float,
                          height: float,
                          shop_type: str,
                          sections: Sequence[Tuple[str, Sequence[str]]],
                          rng=None,
                          is_main_floor: bool = True,
                          vary: bool = True,
                          fit_assortment: bool = False) -> dict:
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
    insertion order.

    ``fit_assortment=True`` makes the plan carry what it says it carries:
    the item-count heuristics only estimate fixture capacity, so when a
    pass leaves items over they are removed from the assortment and the
    floor is rebuilt from the same random state. The surviving items then
    spread over the whole fixture length instead of the runs being cut
    short, and the caller still learns the full shortfall from
    ``dropped``.

    Every item carries ``zone``: the name of the ``Section_`` wall that
    contains its centre. ``dropped`` lists requested item names the floor
    could not hold (empty whenever the whole assortment fits),
    ``keepouts`` the door/checkout/WC approach rects fixtures avoid, and
    ``door_keepout`` the doorway-approach rect on a main floor.

    Raises ``ValueError`` for floors below ``MIN_FLOOR_W`` x
    ``MIN_FLOOR_H``."""
    rng = rng or _random_mod
    W, H = float(width), float(height)
    if W < MIN_FLOOR_W - 1e-9 or H < MIN_FLOOR_H - 1e-9:
        raise ValueError(
            f'{W:.6g}x{H:.6g} m is too small for a realistic layout; '
            f'the archetypes need at least '
            f'{MIN_FLOOR_W:.6g}x{MIN_FLOOR_H:.6g} m')

    requested = [nm for _, names in sections for nm in names]
    secs = [(cat, list(names)) for cat, names in sections]
    # Rewinding the generator starts every retry from the first attempt's
    # random state, so the retry keeps its slot shuffle, restroom corner
    # and front depth unless a whole department drops out, and trimming
    # changes the assortment rather than the shop. Generators without
    # getstate/setstate (e.g. numpy's) take a single pass.
    state = (rng.getstate() if fit_assortment and hasattr(rng, 'getstate')
             else None)
    best = None
    for _ in range(MAX_FIT_PASSES):
        if state is not None:
            rng.setstate(state)
        plan = _build_plan(W, H, shop_type, secs, rng, is_main_floor, vary)
        # A smaller assortment can be spread differently over the slots, so
        # on a floor that is short of fixture length either way a retry can
        # come out one item worse; keep the fullest plan seen. A retry that
        # places as many items wins the tie: it was laid out for exactly
        # the items it holds, where the earlier pass cut its runs short.
        if best is None or len(plan['items']) >= len(best['items']):
            best = plan
        if state is None or not plan['dropped']:
            break
        left_out = set(plan['dropped'])
        secs = [(cat, [nm for nm in names if nm not in left_out])
                for cat, names in secs]
        secs = [(cat, names) for cat, names in secs if names]
        if not secs:
            break
    best['dropped'] = [nm for nm in requested if nm not in best['items']]
    return best


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
    margin for fixture capacity. Used by the dataset-import path: the
    floor dimensions follow from how many sections/items the DATASET has,
    not from a fixed default. Only a lower bound is applied: capping the
    size would silently shrink the floor below what the assortment needs."""
    n_sections = max(1, int(n_sections))
    n_items = max(1, int(max_items_per_section))
    area = max(60.0 * (n_sections - 3),
               110.0 * (n_items - 2),
               220.0) * 1.15
    w = math.sqrt(area * aspect)
    h = area / w
    w = max(14.0, w)
    h = max(10.0, h)
    return round(w, 1), round(h, 1)
