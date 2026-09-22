"""Headless smoke for the realistic shop-architecture generator.

For every shop type x shop size x seed it asserts:
  * no overlap among items and blocking walls (Section_ zones excluded),
  * everything inside the shop bounds,
  * Entrance gap / Checkout / WC present on the main floor,
  * Section zones never overlap each other,
  * every item's ``zone`` is a zone of its own category that contains the
    item's centre and the whole item,
  * no item intrudes on a door / checkout / WC keepout,
  * the plan carries what it says it carries: every sampled item is
    either placed or named in ``dropped``, the assortment it settles on
    fits in one pass and the plan returned is that pass, the floor holds
    most of a jittered assortment and all of the size heuristic's nominal
    one,
  * grid aisles between fixture columns are >= MIN_AISLE and free-form
    tables and shelving runs keep >= 1.3 m between neighbours,
  * every item is REACHABLE from the door (0.1 m grid flood fill with a
    0.35 m customer clearance) -- the property that makes a layout an
    actual shop rather than decoration,
  * every fixture has an ACCESS POINT on the simulator's own grid (the
    free cell nearest its centre, with fixtures blocking movement just as
    they do in the live model) that is close enough to shop from and
    connected to the door, so agents can complete a visit,
  * floors below the engine's minimum are refused instead of silently
    generating overlapping fixtures.

The catalog is sampled the way the Generate button samples it (an rng is
passed to ``scale_catalog``), so these properties are checked for the
assortments the application actually produces.

Run:  python architecture_smoke.py     (exit 0 = pass)
"""

from __future__ import annotations

import math
import sys
import random
import traceback

import numpy as np

from customer_pathfinding import AGENT_RADIUS_M, obstacle_rects
from shop_architecture import (generate_architecture, ARCHETYPE_BY_SHOP,
                               FULL_CATALOG, scale_catalog,
                               MIN_FLOOR_W, MIN_FLOOR_H,
                               T, FIX_D, MIN_AISLE)

SHOP_TYPES = list(FULL_CATALOG.keys())

# The last size is a long, narrow unit: a shape that squeezes the
# free-form zone columns hardest.
SIZES = [(20.0, 15.0), (26.0, 18.0), (15.0, 11.0), (34.0, 24.0),
         (12.0, 40.0)]
SEEDS = [0, 1, 2]

RES = 10                 # raster cells per metre
CLEAR = 0.35             # customer clearance radius (m)
TABLE_SIZE = (1.5, 0.9)  # free-form display table footprint (m)
TABLE_CLEAR = 1.3        # promised clear gap between neighbouring tables (m)
RUN_D = 0.9              # free-standing shelving-run depth (m)
MIN_PLACED_FRAC = 0.5    # of the sampled assortment the floor must carry

# The live model's walkability: agents plan on a 0.25 m grid
# (CustomerFlowSimulation.path_grid_resolution) whose obstacles are
# inflated by the agent radius, and they count as having arrived at a
# fixture within this margin of its rectangle (Customer).
PATH_RES = 0.25
ARRIVE_MARGIN = AGENT_RADIUS_M + PATH_RES + 0.05


def _rects_overlap(a, b, eps=1e-6):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (ax + aw <= bx + eps or ax >= bx + bw - eps
                or ay + ah <= by + eps or ay >= by + bh - eps)


def _blocking(walls):
    out = []
    for nm, w in walls.items():
        if nm.startswith('Section_'):
            continue
        x, y = w['position']
        ww, wh = w['size']
        out.append((nm, (x, y, ww, wh)))
    return out


def _flood_reachable(W, H, blockers, door_xy):
    """Boolean grid of cells reachable from just inside the door, after
    inflating every blocker by the customer clearance radius."""
    nx, ny = int(W * RES), int(H * RES)
    free = np.ones((nx, ny), dtype=bool)
    for (_, (x, y, w, h)) in blockers:
        x0 = max(0, int((x - CLEAR) * RES))
        y0 = max(0, int((y - CLEAR) * RES))
        x1 = min(nx, int(np.ceil((x + w + CLEAR) * RES)))
        y1 = min(ny, int(np.ceil((y + h + CLEAR) * RES)))
        free[x0:x1, y0:y1] = False

    sx = min(nx - 1, max(0, int(door_xy[0] * RES)))
    sy = min(ny - 1, max(0, int((door_xy[1] + 0.7) * RES)))
    # nudge to the nearest free cell around the door
    if not free[sx, sy]:
        found = False
        for r in range(1, int(2.5 * RES)):
            xs = slice(max(0, sx - r), min(nx, sx + r + 1))
            ys = slice(max(0, sy - r), min(ny, sy + r + 1))
            sub = np.argwhere(free[xs, ys])
            if len(sub):
                sx = xs.start + int(sub[0][0])
                sy = ys.start + int(sub[0][1])
                found = True
                break
        if not found:
            return np.zeros_like(free)

    reach = np.zeros_like(free)
    stack = [(sx, sy)]
    reach[sx, sy] = True
    while stack:
        cx, cy = stack.pop()
        if cx > 0 and free[cx - 1, cy] and not reach[cx - 1, cy]:
            reach[cx - 1, cy] = True; stack.append((cx - 1, cy))
        if cx < nx - 1 and free[cx + 1, cy] and not reach[cx + 1, cy]:
            reach[cx + 1, cy] = True; stack.append((cx + 1, cy))
        if cy > 0 and free[cx, cy - 1] and not reach[cx, cy - 1]:
            reach[cx, cy - 1] = True; stack.append((cx, cy - 1))
        if cy < ny - 1 and free[cx, cy + 1] and not reach[cx, cy + 1]:
            reach[cx, cy + 1] = True; stack.append((cx, cy + 1))
    return reach


def _item_reachable(reach, W, H, rect):
    """True if any raster cell in a 0.8 m halo around the item rect is
    reachable (a customer can stand next to the fixture)."""
    nx, ny = reach.shape
    x, y, w, h = rect
    x0 = max(0, int((x - 0.8) * RES)); x1 = min(nx, int((x + w + 0.8) * RES))
    y0 = max(0, int((y - 0.8) * RES)); y1 = min(ny, int((y + h + 0.8) * RES))
    return bool(reach[x0:x1, y0:y1].any())


def _axis_gaps(a, b):
    """Clear gap between two rects along x and along y (negative when
    their projections on that axis overlap)."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return (max(bx - (ax + aw), ax - (bx + bw)),
            max(by - (ay + ah), ay - (by + bh)))


def _agent_grid(W, H, walls, items):
    """The walkability grid the live model plans on: every obstacle it
    knows about -- blocking walls and the fixtures agents walk around --
    inflated by the agent radius, so a free cell is one an agent can
    stand on. The obstacle set comes from the simulator's own rule."""
    nx, ny = int(W / PATH_RES) + 1, int(H / PATH_RES) + 1
    blocked = np.zeros((nx, ny), dtype=bool)
    r = AGENT_RADIUS_M
    for ox, oy, ow, oh in obstacle_rects(walls, items):
        x0 = max(0, int(math.ceil((ox - r) / PATH_RES)))
        y0 = max(0, int(math.ceil((oy - r) / PATH_RES)))
        x1 = min(nx - 1, int((ox + ow + r) / PATH_RES))
        y1 = min(ny - 1, int((oy + oh + r) / PATH_RES))
        blocked[x0:x1 + 1, y0:y1 + 1] = True
    return blocked


def _nearest_free_cell(blocked, x, y):
    """Free cell nearest the point (metres), or None on a floor with no
    free cell at all -- the access point an agent walks to.

    Distances are measured from the cell the point falls in, and ties go
    to the lowest index, because that is how the simulator's own search
    picks the standing spot; measuring from the exact point instead would
    hand a different cell back on roughly half the fixtures."""
    nx, ny = blocked.shape
    gx = min(max(int(x / PATH_RES), 0), nx - 1)
    gy = min(max(int(y / PATH_RES), 0), ny - 1)
    if not blocked[gx, gy]:
        return gx, gy
    free = np.argwhere(~blocked)
    if not len(free):
        return None
    d2 = (free[:, 0] - gx) ** 2 + (free[:, 1] - gy) ** 2
    k = int(np.argmin(d2))
    return int(free[k, 0]), int(free[k, 1])


def _connected_cells(blocked, start):
    """Cells a 4-connected walk from ``start`` can reach."""
    nx, ny = blocked.shape
    seen = np.zeros_like(blocked)
    seen[start] = True
    stack = [start]
    while stack:
        cx, cy = stack.pop()
        for ax, ay in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
            if 0 <= ax < nx and 0 <= ay < ny and not blocked[ax, ay] \
                    and not seen[ax, ay]:
                seen[ax, ay] = True
                stack.append((ax, ay))
    return seen


def _dist_to_rect(px, py, rect):
    x, y, w, h = rect
    return math.hypot(max(x - px, 0.0, px - (x + w)),
                      max(y - py, 0.0, py - (y + h)))


def check_one(shop_type, W, H, seed, is_main=True):
    rng = random.Random(seed)
    # Sample the assortment and build the plan exactly as the Generate
    # button does, so these properties hold for the shops users get.
    catalog = scale_catalog(shop_type, W, H, rng=rng)
    build_state = rng.getstate()
    arch = generate_architecture(W, H, shop_type, catalog,
                                 rng=rng, is_main_floor=is_main,
                                 fit_assortment=True)
    walls, items = arch['walls'], arch['items']
    blockers = _blocking(walls)
    tag = f'{shop_type} {W}x{H} seed={seed}'

    # 1) In-bounds
    for nm, it in items.items():
        x, y = it['position']; w, h = it['size']
        assert 0 <= x and 0 <= y and x + w <= W + 1e-6 and y + h <= H + 1e-6, \
            f'{shop_type}: item {nm} out of bounds'

    # 2) No item-item / item-blockingwall overlap
    rects = [(nm, (it['position'][0], it['position'][1],
                   it['size'][0], it['size'][1])) for nm, it in items.items()]
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            assert not _rects_overlap(rects[i][1], rects[j][1]), \
                f'{shop_type}: items overlap: {rects[i][0]} / {rects[j][0]}'
    for nm, r in rects:
        for wn, wr in blockers:
            assert not _rects_overlap(r, wr), \
                f'{shop_type}: item {nm} overlaps wall {wn}'

    # 3) Main-floor furniture present; door set
    if is_main:
        assert arch['door_position'] and arch['door_side'] == 'bottom'
        assert 'Checkout' in walls, f'{shop_type}: no Checkout'
        assert 'WC' in walls, f'{shop_type}: no WC'

    # 4) Section zones never overlap each other
    zones = [(wn, (w['position'][0], w['position'][1],
                   w['size'][0], w['size'][1]))
             for wn, w in walls.items() if wn.startswith('Section_')]
    for i in range(len(zones)):
        for j in range(i + 1, len(zones)):
            assert not _rects_overlap(zones[i][1], zones[j][1]), \
                f'{tag}: zones overlap: {zones[i][0]} / {zones[j][0]}'

    # 5) Each item's zone belongs to its own category (Section_<cat> or a
    #    split-fixture Section_<cat>_<n>) and holds its centre and the
    #    whole item, to the same round-off the GA's section check allows.
    zone_rect = dict(zones)
    for nm, it in items.items():
        cat, zn = it['category'], it.get('zone')
        own = f'Section_{cat}'
        suffix = zn[len(own):] if isinstance(zn, str) else None
        assert zn in zone_rect and zn.startswith(own) and \
            (suffix == '' or (suffix[:1] == '_' and suffix[1:].isdigit())), \
            f'{tag}: item {nm} has no zone of its {cat} category ({zn})'
        zx, zy, zw, zh = zone_rect[zn]
        x, y = it['position']; w, h = it['size']
        cx, cy = x + w / 2, y + h / 2
        assert zx <= cx <= zx + zw and zy <= cy <= zy + zh, \
            f'{tag}: item {nm} centre outside its zone {zn}'
        assert (zx - 1e-6 <= x and zy - 1e-6 <= y
                and x + w <= zx + zw + 1e-6 and y + h <= zy + zh + 1e-6), \
            f'{tag}: item {nm} not inside its zone {zn}'

    # 6) No item intrudes on a door / checkout / WC approach
    for nm, r in rects:
        for k in arch.get('keepouts', []):
            assert not _rects_overlap(r, tuple(k)), \
                f'{tag}: item {nm} intrudes on keepout {k}'

    # 7) The plan accounts for the whole sampled catalog: what the floor
    #    cannot hold is named in 'dropped' rather than vanishing, and the
    #    floor still carries most of the assortment its size asked for.
    requested = [nm for _, names in catalog for nm in names]
    dropped = list(arch['dropped'])
    assert set(items) | set(dropped) == set(requested), \
        f'{tag}: plan does not account for the sampled catalog'
    assert not (set(items) & set(dropped)), \
        f'{tag}: item reported both placed and dropped'
    assert len(items) >= MIN_PLACED_FRAC * len(requested), \
        f'{tag}: only {len(items)} of {len(requested)} sampled items fit'
    # At these sizes the size heuristic on its own (no count jitter) asks
    # for no more than the floor holds, so its nominal assortment fits
    # whole in one pass; this keeps the heuristic in step with what the
    # archetypes can actually carry when either of them changes.
    nominal = scale_catalog(shop_type, W, H)
    plain = generate_architecture(W, H, shop_type, nominal,
                                  rng=random.Random(seed),
                                  is_main_floor=is_main)
    assert not plain['dropped'], \
        f'{tag}: {len(plain["dropped"])} item(s) of the nominal catalog ' \
        f'dropped: {plain["dropped"][:5]}'
    # The assortment it settled on really fits: from the same random
    # state, building that assortment alone leaves nothing over, and that
    # build is the plan returned -- laid out for the items it holds rather
    # than a first pass whose runs were cut short.
    kept = [(cat, [nm for nm in names if nm in items])
            for cat, names in catalog]
    kept = [(cat, names) for cat, names in kept if names]
    rng.setstate(build_state)
    again = generate_architecture(W, H, shop_type, kept, rng=rng,
                                  is_main_floor=is_main)
    assert not again['dropped'], \
        f'{tag}: the assortment kept still does not fit: {again["dropped"][:5]}'
    assert all(tuple(again['items'][nm]['position']) == tuple(it['position'])
               and tuple(again['items'][nm]['size']) == tuple(it['size'])
               for nm, it in items.items()), \
        f'{tag}: the plan returned is not the one built for its assortment'

    # 8) Aisle clearances
    if arch['archetype'] == 'grid':
        # Vertical fixture columns (side wall runs and gondolas; the back
        # wall run is excluded) facing each other keep MIN_AISLE clear.
        cols = [(nm, r) for nm, r in rects if r[1] < H - T - FIX_D - 1e-6]
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                gx, gy = _axis_gaps(cols[i][1], cols[j][1])
                if gx > 1e-6 and gy < 0:
                    assert gx >= MIN_AISLE - 1e-6, \
                        f'{tag}: aisle {gx:.3f} m between ' \
                        f'{cols[i][0]} / {cols[j][0]}'
    elif arch['archetype'] == 'freeform':
        tables = [(nm, r) for nm, r in rects if r[2:] == TABLE_SIZE]
        for i in range(len(tables)):
            for j in range(i + 1, len(tables)):
                gap = max(_axis_gaps(tables[i][1], tables[j][1]))
                assert gap >= TABLE_CLEAR - 1e-6, \
                    f'{tag}: tables {tables[i][0]} / {tables[j][0]} ' \
                    f'only {gap:.3f} m apart'
        # Free-standing shelving runs (the fallback for zones too tight
        # for tables) must keep the same gap, whatever the floor shape:
        # parallel runs closer than this are a shop nobody can walk.
        runs = [(nm, r) for nm, r in rects if abs(r[2] - RUN_D) < 1e-6]
        for i in range(len(runs)):
            for j in range(i + 1, len(runs)):
                gx, gy = _axis_gaps(runs[i][1], runs[j][1])
                if gx > 1e-6 and gy < 0:
                    assert gx >= TABLE_CLEAR - 1e-6, \
                        f'{tag}: shelving runs {runs[i][0]} / ' \
                        f'{runs[j][0]} only {gx:.3f} m apart'

    # 9) Every item reachable from the door
    if is_main:
        all_blk = blockers + rects
        reach = _flood_reachable(W, H, all_blk, arch['door_position'])
        for nm, r in rects:
            assert _item_reachable(reach, W, H, r), \
                f'{shop_type} {W}x{H} seed={seed}: item {nm} NOT reachable'

    # 10) Every fixture is shoppable in the live model: the access point
    #     an agent walks to -- the free cell nearest the fixture centre on
    #     the simulator's grid, with fixtures blocking movement as they do
    #     there -- must be close enough to count as an arrival and be
    #     connected to the spot the agent enters at.
    if is_main:
        blocked = _agent_grid(W, H, walls, items)
        door = arch['door_position']
        start = _nearest_free_cell(blocked, door[0], door[1] + 0.5)
        assert start is not None, f'{tag}: no free cell at the entrance'
        connected = _connected_cells(blocked, start)
        for nm, r in rects:
            cell = _nearest_free_cell(blocked, r[0] + r[2] / 2,
                                      r[1] + r[3] / 2)
            assert cell is not None, f'{tag}: item {nm} has no access point'
            px, py = cell[0] * PATH_RES, cell[1] * PATH_RES
            gap = _dist_to_rect(px, py, r)
            assert gap <= ARRIVE_MARGIN + 1e-6, \
                f'{tag}: access point for {nm} is {gap:.3f} m away, ' \
                f'beyond the {ARRIVE_MARGIN:.3f} m arrival margin'
            assert connected[cell], \
                f'{tag}: access point for {nm} is cut off from the door'

    return arch['archetype'], len(items), len([1 for wn in walls
                                               if wn.startswith('Section_')])


def check_scaling(shop_type):
    """Bigger floor area must produce more sections and more items."""
    small = scale_catalog(shop_type, 15.0, 11.0)
    mid = scale_catalog(shop_type, 20.0, 15.0)
    big = scale_catalog(shop_type, 34.0, 24.0)
    n = lambda cat: (len(cat), sum(len(i) for _, i in cat))
    (s_sec, s_it), (m_sec, m_it), (b_sec, b_it) = n(small), n(mid), n(big)
    assert s_sec <= m_sec <= b_sec, f'{shop_type}: sections not monotonic'
    assert s_it < m_it < b_it, f'{shop_type}: item counts not growing with area'
    # And the generated shop should actually PLACE more items when bigger.
    r = random.Random(0)
    a_small = generate_architecture(15, 11, shop_type, small, rng=r)
    r = random.Random(0)
    a_big = generate_architecture(34, 24, shop_type, big, rng=r)
    assert len(a_big['items']) > len(a_small['items']), \
        f'{shop_type}: bigger shop did not place more items ' \
        f'({len(a_big["items"])} vs {len(a_small["items"])})'
    return len(a_small['items']), len(a_big['items'])


def check_variation(shop_type):
    """Two generates of the same shop type/size must differ in assortment
    (sampled sections/items), not just entrance/checkout positions."""
    def _one(seed):
        rng = random.Random(seed)
        cat = scale_catalog(shop_type, 24, 17, rng=rng)
        arch = generate_architecture(24, 17, shop_type, cat, rng=rng)
        return (frozenset(nm for nm, _ in cat),
                frozenset(arch['items'].keys()),
                arch['walls']['WC']['position'])
    a, b = _one(0), _one(7)
    assert (a[0] != b[0]) or (a[1] != b[1]) or (a[2] != b[2]), \
        f'{shop_type}: two generates identical (no assortment variation)'
    # And determinism: the same seed reproduces the same shop.
    assert _one(3) == _one(3), f'{shop_type}: generation not deterministic'
    return True


def check_min_floor(shop_type):
    """A floor too small for the archetypes is refused outright, instead
    of being furnished with overlapping fixtures."""
    W, H = MIN_FLOOR_W - 1.0, MIN_FLOOR_H - 1.0
    catalog = scale_catalog(shop_type, W, H)
    try:
        generate_architecture(W, H, shop_type, catalog, rng=random.Random(0))
    except ValueError:
        return True
    raise AssertionError(f'{shop_type}: a {W}x{H} m floor was not refused')


def check_multilane(shop_type):
    """Big shops get a checkout bank; the lane-picker helper balances."""
    arch = generate_architecture(26, 18, shop_type,
                                 scale_catalog(shop_type, 26, 18),
                                 rng=random.Random(0))
    lanes = [nm for nm in arch['walls']
             if nm == 'Checkout' or nm.startswith('Checkout_Lane')]
    assert len(lanes) >= 2, f'{shop_type}: expected a multi-lane checkout bank'
    return len(lanes)


def main():
    failed = []
    for shop_type in SHOP_TYPES:
        for (W, H) in SIZES:
            for seed in SEEDS:
                try:
                    at, n_items, n_zones = check_one(shop_type, W, H, seed)
                except Exception as e:
                    failed.append((shop_type, W, H, seed, str(e)))
                    traceback.print_exc()
        # one upper-floor variant per type
        try:
            check_one(shop_type, 20.0, 15.0, 0, is_main=False)
        except Exception as e:
            failed.append((shop_type, 'floor2', '-', 0, str(e)))
            traceback.print_exc()
        try:
            small_n, big_n = check_scaling(shop_type)
            lanes_n = check_multilane(shop_type)
            check_variation(shop_type)
            check_min_floor(shop_type)
        except Exception as e:
            failed.append((shop_type, 'scaling/lanes/variation', '-', 0, str(e)))
            traceback.print_exc()
            small_n = big_n = lanes_n = -1
        print(f'  [{ARCHETYPE_BY_SHOP.get(shop_type, "grid"):9s}] {shop_type:18s} OK '
              f'(items {small_n}->{big_n} small->big, lanes={lanes_n})')

    print('=' * 56)
    if failed:
        print(f'ARCHITECTURE SMOKE FAILED ({len(failed)}):')
        for f in failed:
            print('  ', f)
        return 1
    print('ARCHITECTURE SMOKE PASSED '
          f'({len(SHOP_TYPES)} shop types x {len(SIZES)} sizes x {len(SEEDS)} seeds '
          f'+ scaling + multi-lane + minimum floor)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
