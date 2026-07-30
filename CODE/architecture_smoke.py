"""Headless smoke for the realistic shop-architecture generator.

For every shop type x shop size x seed it asserts:
  * no overlap among items and blocking walls (Section_ zones excluded),
  * everything inside the shop bounds,
  * Entrance gap / Checkout / WC present on the main floor,
  * every item sits inside a Section zone of its own category,
  * every item is REACHABLE from the door (0.1 m grid flood fill with a
    0.35 m customer clearance) -- the property that makes a layout an
    actual shop rather than decoration.

Run:  python architecture_smoke.py     (exit 0 = pass)
"""

from __future__ import annotations

import sys
import random
import traceback

import numpy as np

from shop_architecture import (generate_architecture, ARCHETYPE_BY_SHOP,
                               FULL_CATALOG, scale_catalog)

SHOP_TYPES = list(FULL_CATALOG.keys())

SIZES = [(20.0, 15.0), (26.0, 18.0), (15.0, 11.0), (34.0, 24.0)]
SEEDS = [0, 1, 2]

RES = 10                 # raster cells per metre
CLEAR = 0.35             # customer clearance radius (m)


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


def check_one(shop_type, W, H, seed, is_main=True):
    rng = random.Random(seed)
    arch = generate_architecture(W, H, shop_type, scale_catalog(shop_type, W, H),
                                 rng=rng, is_main_floor=is_main)
    walls, items = arch['walls'], arch['items']
    blockers = _blocking(walls)

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

    # 4) Items inside a Section zone of their category
    zones = {}
    for wn, w in walls.items():
        if wn.startswith('Section_'):
            base = wn[len('Section_'):].rsplit('_', 1)[0] \
                if wn[len('Section_'):].rsplit('_', 1)[-1].isdigit() \
                else wn[len('Section_'):]
            zones.setdefault(base, []).append(
                (w['position'][0], w['position'][1], w['size'][0], w['size'][1]))
    for nm, it in items.items():
        cat = it['category']
        x, y = it['position']; w, h = it['size']
        inside = any(zx - 0.05 <= x and zy - 0.05 <= y
                     and x + w <= zx + zw + 0.05 and y + h <= zy + zh + 0.05
                     for zx, zy, zw, zh in zones.get(cat, []))
        assert inside, f'{shop_type}: item {nm} outside its {cat} zone'

    # 5) Every item reachable from the door
    if is_main:
        all_blk = blockers + rects
        reach = _flood_reachable(W, H, all_blk, arch['door_position'])
        for nm, r in rects:
            assert _item_reachable(reach, W, H, r), \
                f'{shop_type} {W}x{H} seed={seed}: item {nm} NOT reachable'

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
          f'+ scaling + multi-lane)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
