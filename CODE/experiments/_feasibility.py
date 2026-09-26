"""The floor-plan invariants of the shared feasibility repair.

The floor-plan engine (``shop_architecture``) promises two things about
every store it builds: fixtures that face each other across an aisle keep
at least ``MIN_AISLE`` (1.6 m) clear between them, and every fixture can be
shopped -- the free grid cell an agent walks to beside it (its access
point, ``sim_geometry``) lies within arrival distance of it and is
connected to the door. The zones the GA confines items to include the
aisles, though: a perimeter zone is the wall fixture band plus the aisle in
front of it, and a gondola zone is the gondola plus half an aisle on each
side. A repair that only clips items into their zone and removes overlaps
inside a zone can therefore push two zones' fixtures to 0.1 m apart and
close the aisle between them, or wall a fixture in, and an optimum found on
that set is not a layout the engine would accept. This module adds the
engine's invariants to the repair every method shares
(``experiments._common.feasible_layout`` and the GA's own chain).

The aisle plan (``aisle_plan``) is read once per store from its as-built
layout (``shop.as_built_layout``, set by the headless builders; the layout
on the floor otherwise):

  * a zone's BAND is the bounding box of its movement-blocking fixtures as
    built. Impulse displays, which agents reach across, block nothing and
    are left out, as ``customer_pathfinding.item_blocks_movement`` leaves
    them out of every obstacle set;
  * two zones are KEPT APART when their bands, as built, are at least
    ``MIN_AISLE`` apart along x or along y -- the aisles of the engine's
    plan. Zones closer than that (two departments on one wall with the
    engine's 0.8 m break between them, or on one gondola with its 0.96 m
    cross-aisle) only have to not overlap, which disjoint zones guarantee;
  * each zone gets a REGION: its band (after the repair's zone clip) grown
    toward each side by as much of the zone as the aisles allow. Between
    two zones kept apart along one axis, the slack beyond ``MIN_AISLE`` is
    shared in proportion to the room each zone has on that side -- all of
    it when both rooms fit inside the slack -- so any two positions in the
    two regions stay ``MIN_AISLE`` apart. A pair kept apart along both axes
    is constrained only where the regions from the single-axis pairs would
    come closer than that on both, and then along the axis that is closer
    to holding.

The region always contains the as-built band, so the as-built layout stays
feasible. On a store whose zones exclude their aisles -- the synthetic
scenarios, whose sections stand 1.5 m apart -- the rooms always fit inside
the slack, every region is the zone less the repair's pad, and the repair
returns exactly what it returned before this module existed.

The repair (``repair_positions``, called by ``_repair_chrom_overlaps``):
clip each item into its region, remove overlaps inside each zone with the
rank-preserving grid snap run inside the region, then check every
fixture's access point on the agent grid. A zone holding a fixture that
cannot be shopped is re-packed single-file along its run -- its fixtures
back on their as-built line, in the order the layout gave them, spread over
the region -- and, if that still fails, returned to its as-built positions;
if the store still fails, the whole layout is. Each step is deterministic
and leaves a feasible layout where it is, so the repair stays idempotent.
``repair_stats`` counts how often the region clip and each fallback bound.

Every search's operators work inside the regions, not the zones: random
search's uniform draws and the annealer's moves (``position_bounds``, via
``_common.zone_sampler`` / ``zone_neighbor``) and the GA's initial noise and
mutation (``move_box``, via ``HeadlessShop._ga_move_boxes``) are scaled to
and clipped into them, so no search spends proposals the repair would
clip onto an aisle edge, and all three are held to the same space. The rule
therefore binds by narrowing that space -- ``aisle_rule_summary`` measures
by how much -- and the region clip in the repair is a safety net a search
leaves at or near zero. On the synthetic template the regions are the zones
less the pad and every operator is the one it always was.

The invariant check (``layout_violations`` / ``check_layout_invariants``)
tests a layout against all of it -- containment in the zone, no overlap,
the aisles between zones kept apart, and every access point shoppable --
and every runner applies it to every layout it reports.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage

from customer_pathfinding import (AGENT_RADIUS_M, item_blocks_movement,
                                  wall_blocks_movement)
from shop_architecture import MIN_AISLE
from sim_geometry import _nearest_free_cell

#: Inset of every position from its zone edge: the pad ``_ga_repair`` clips
#: items to and the grid snap leaves at a zone's edge (the value the repair
#: has always used).
ZONE_PAD = 0.05

#: Gap the grid snap leaves between neighbouring fixtures in a zone (the
#: value the repair has always used).
SNAP_GAP = 0.15

#: Round-off allowed on the clearance, containment and arrival checks, in
#: metres: float error at shop scale, far below any physical tolerance.
GEOM_TOL = 1e-6

#: Where an agent enters: this far inside the door, as the floor-plan
#: engine's own reachability check (``architecture_smoke``) enters.
DOOR_STEP_IN_M = 0.5

#: Slack below which a proportional share is not worth the arithmetic: two
#: rooms that fit inside the slack to within this are granted whole, so the
#: repair on a store whose zones exclude their aisles is bit-identical to the
#: zone clip.
_ROOM_FIT_TOL = 1e-9

# Sides of a rectangle, in the order extras are stored.
_LEFT, _RIGHT, _BOTTOM, _TOP = 0, 1, 2, 3


class LayoutInvariantError(AssertionError):
    """A reported layout breaks the floor-plan engine's invariants."""


# --- The plan --------------------------------------------------------------

@dataclass
class AislePlan:
    """What the repair and the check read about one store.

    Per item (in ``names`` order): its zone key and zone rectangle, its
    size, whether it blocks movement, and its as-built position after the
    repair's zone and region clips. Per zone: the extra inset beyond
    ``ZONE_PAD`` on each side (``extras``: left, right, bottom, top) that
    turns the zone into its region, and the axis its fixtures run along.
    ``kept_apart`` lists the zone pairs whose fixtures keep an aisle, with
    the clearance required of them. ``stats`` counts what the repair did."""
    key: Any
    names: Tuple[str, ...]
    zone: Tuple[Optional[str], ...]
    zone_rect: Dict[str, Tuple[float, float, float, float]]
    sizes: np.ndarray
    blocking: np.ndarray
    extras: Dict[str, Tuple[float, float, float, float]]
    run_axis: Dict[str, int]
    kept_apart: Tuple[Tuple[str, str, float], ...]
    as_built: np.ndarray
    width: float
    height: float
    res: float
    wall_blocked: np.ndarray
    door: Tuple[float, float]
    members: Dict[str, List[int]] = field(default_factory=dict)
    stats: Dict[str, float] = field(default_factory=dict)

    def region(self, zone: str) -> Tuple[float, float, float, float]:
        """``(x0, y0, x1, y1)``: where the zone's fixture rectangles may lie."""
        sx, sy, sw, sh = self.zone_rect[zone]
        el, er, eb, et = self.extras[zone]
        return (sx + ZONE_PAD + el, sy + ZONE_PAD + eb,
                sx + sw - ZONE_PAD - er, sy + sh - ZONE_PAD - et)

    def move_box(self, zone: str) -> Tuple[float, float, float, float]:
        """``(sx, sy, sw, sh)``: the zone less the aisle rule's extras -- the
        box every search's operators are scaled to and clip into, each
        adding the repair's ``ZONE_PAD`` inside it as it always has, which
        makes it the region exactly. With zero extras it is the zone."""
        sx, sy, sw, sh = self.zone_rect[zone]
        el, er, eb, et = self.extras[zone]
        return (sx + el, sy + eb, sw - el - er, sh - eb - et)

    def position_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """``(lo, hi)`` of every item's position (lower-left corner) that
        keeps its rectangle where the repair keeps it: inside the region on
        every side the aisle rule narrows (extras above zero), so a position
        drawn in these bounds is never moved by the region clip; on a side
        the rule leaves alone, at the zone edge, where ``_ga_repair``'s zone
        clip adds its 0.05 m pad afterwards, as it always has. Items without
        a zone get the floor less 0.1 m, the box ``_ga_repair`` clips them
        to. With zero extras (a store whose zones exclude their aisles)
        these are the zone's own bounds, operation for operation."""
        n = len(self.names)
        lo = np.empty((n, 2)); hi = np.empty((n, 2))
        for i in range(n):
            w, h = self.sizes[i]
            z = self.zone[i]
            if z is None:
                sx, sy, sw, sh = 0.1, 0.1, self.width - 0.2, self.height - 0.2
                el = er = eb = et = 0.0
            else:
                sx, sy, sw, sh = self.zone_rect[z]
                el, er, eb, et = self.extras[z]
            # The pad the region clip adds on a narrowed side.
            pl, pr, pb, pt = (ZONE_PAD if e > 0.0 else 0.0
                              for e in (el, er, eb, et))
            lo_x, lo_y = sx + el + pl, sy + eb + pb
            lo[i] = (lo_x, lo_y)
            hi[i] = (max(lo_x, sx + sw - w - er - pr),
                     max(lo_y, sy + sh - h - et - pt))
        return lo, hi


def _zone_key(data: Mapping[str, Any]) -> str:
    """The zone the repair groups an item by: its stamped zone, else
    ``Section_<category>`` -- the rule ``_repair_chrom_overlaps`` has
    always grouped by."""
    return data.get('zone') or f"Section_{data.get('category', '')}"


def _as_built_positions(shop, item_names: Sequence[str]) -> np.ndarray:
    """The store as built: ``shop.as_built_layout`` when the builder
    recorded it, else the positions on the floor."""
    ab = getattr(shop, 'as_built_layout', None)
    items = shop.floors[1]['items']
    out = np.empty((len(item_names), 2), dtype=np.float64)
    for i, n in enumerate(item_names):
        if ab is not None and n in ab:
            out[i] = ab[n]
        else:
            out[i] = items[n]['position']
    return out


def _fingerprint(shop, item_names: Sequence[str],
                 as_built: np.ndarray) -> Any:
    """What the plan depends on, as a hashable key: the items (as-built
    position, size, zone, category), the walls, the floor and the door."""
    items = shop.floors[1]['items']
    walls = shop.floors[1].get('walls', {})
    it = tuple((n, float(as_built[i, 0]), float(as_built[i, 1]),
                tuple(map(float, items[n].get('size', (1.0, 1.0)))),
                items[n].get('zone'), items[n].get('category'))
               for i, n in enumerate(item_names))
    wl = tuple((wn, tuple(map(float, w['position'])),
                tuple(map(float, w['size'])), w.get('category'))
               for wn, w in walls.items())
    door = shop.door_position or shop.floors[1].get('door_position')
    return (it, wl, float(shop.width), float(shop.height),
            None if door is None else (float(door[0]), float(door[1])))


def _band(rects: np.ndarray) -> Tuple[float, float, float, float]:
    """Bounding box ``(x0, y0, x1, y1)`` of rows ``(x, y, w, h)``."""
    return (float(rects[:, 0].min()), float(rects[:, 1].min()),
            float((rects[:, 0] + rects[:, 2]).max()),
            float((rects[:, 1] + rects[:, 3]).max()))


def _axis_gap(a, b, axis: int) -> float:
    """Clear gap between boxes ``(x0, y0, x1, y1)`` along ``axis``
    (negative when their projections on it overlap)."""
    return max(b[axis] - a[axis + 2], a[axis] - b[axis + 2])


def _grid_shape(width: float, height: float, res: float) -> Tuple[int, int]:
    # The simulator's grid (``_rebuild_geometry_caches``).
    return int(width / res) + 1, int(height / res) + 1


def _rasterize(blocked: np.ndarray, rects, res: float) -> None:
    """Mark the cells an agent standing on would collide with each
    ``(x, y, w, h)``: the rule ``sim_geometry`` builds its grid by."""
    nx, ny = blocked.shape
    r = AGENT_RADIUS_M
    for ox, oy, ow, oh in rects:
        x0 = max(0, int(math.ceil((ox - r) / res)))
        y0 = max(0, int(math.ceil((oy - r) / res)))
        x1 = min(nx - 1, int((ox + ow + r) / res))
        y1 = min(ny - 1, int((oy + oh + r) / res))
        if x1 >= x0 and y1 >= y0:
            blocked[x0:x1 + 1, y0:y1 + 1] = True


def aisle_plan(shop, item_names: Sequence[str]) -> AislePlan:
    """The store's aisle plan (see the module docstring), cached on the
    shop per item list and as-built layout."""
    names = tuple(item_names)
    as_built_raw = _as_built_positions(shop, names)
    key = (names, _fingerprint(shop, names, as_built_raw))
    cache = shop.__dict__.setdefault('_aisle_plan_cache', {})
    plan = cache.get(key)
    if plan is None:
        plan = _build_plan(shop, names, as_built_raw, key)
        cache.clear()           # one store layout at a time
        cache[key] = plan
    return plan


def _build_plan(shop, names: Tuple[str, ...], as_built_raw: np.ndarray,
                key: Any) -> AislePlan:
    walls = shop.floors[1].get('walls', {})
    data = [shop._ga_get_item_data(n) for n in names]
    sizes = np.array([tuple(d.get('size', (1.0, 1.0))) for d in data],
                     dtype=np.float64)
    blocking = np.array([item_blocks_movement(d) for d in data], dtype=bool)
    zone: List[Optional[str]] = []
    zone_rect: Dict[str, Tuple[float, float, float, float]] = {}
    for d in data:
        z = _zone_key(d)
        w = walls.get(z)
        if w is None:
            zone.append(None)
            continue
        zone.append(z)
        zone_rect[z] = (float(w['position'][0]), float(w['position'][1]),
                        float(w['size'][0]), float(w['size'][1]))
    members: Dict[str, List[int]] = {}
    for i, z in enumerate(zone):
        if z is not None:
            members.setdefault(z, []).append(i)

    # The as-built layout through the repair's zone clip: the positions the
    # regions must contain.
    clipped = shop._ga_repair(as_built_raw.copy(), list(names))

    raw_band: Dict[str, Tuple[float, float, float, float]] = {}
    cut_band: Dict[str, Tuple[float, float, float, float]] = {}
    for z, idx in members.items():
        blk = [i for i in idx if blocking[i]]
        if not blk:
            continue
        raw_band[z] = _band(np.column_stack([as_built_raw[blk], sizes[blk]]))
        cut_band[z] = _band(np.column_stack([clipped[blk], sizes[blk]]))

    # Room on each side: from the clipped band to the zone less its pad.
    room: Dict[str, List[float]] = {}
    for z, b in cut_band.items():
        sx, sy, sw, sh = zone_rect[z]
        room[z] = [max(0.0, b[0] - (sx + ZONE_PAD)),
                   max(0.0, (sx + sw - ZONE_PAD) - b[2]),
                   max(0.0, b[1] - (sy + ZONE_PAD)),
                   max(0.0, (sy + sh - ZONE_PAD) - b[3])]
    grow = {z: list(r) for z, r in room.items()}

    zones = sorted(cut_band)
    single, double, kept = [], [], []
    for a_i, a in enumerate(zones):
        for b in zones[a_i + 1:]:
            axes = [ax for ax in (0, 1)
                    if _axis_gap(raw_band[a], raw_band[b], ax)
                    >= MIN_AISLE - GEOM_TOL]
            if not axes:
                continue
            # Never demand more than the repaired as-built layout keeps.
            need = min(MIN_AISLE,
                       max(_axis_gap(cut_band[a], cut_band[b], ax)
                           for ax in axes))
            kept.append((a, b, float(need)))
            (single if len(axes) == 1 else double).append((a, b, axes, need))

    def _sides(a, b, ax):
        """(side of a facing b, side of b facing a) along ``ax``."""
        if cut_band[a][ax + 2] <= cut_band[b][ax]:
            return (_RIGHT, _LEFT) if ax == 0 else (_TOP, _BOTTOM)
        return (_LEFT, _RIGHT) if ax == 0 else (_BOTTOM, _TOP)

    # Pairs kept apart along one axis: share the slack beyond the aisle.
    for a, b, axes, need in single:
        ax = axes[0]
        sa, sb = _sides(a, b, ax)
        slack = max(0.0, _axis_gap(cut_band[a], cut_band[b], ax) - need)
        ra, rb = room[a][sa], room[b][sb]
        if ra + rb <= slack + _ROOM_FIT_TOL:
            ga, gb = ra, rb
        else:
            ga, gb = slack * ra / (ra + rb), slack * rb / (ra + rb)
        grow[a][sa] = min(grow[a][sa], ga)
        grow[b][sb] = min(grow[b][sb], gb)

    def _grown(z):
        b, g = cut_band[z], grow[z]
        return (b[0] - g[_LEFT], b[1] - g[_BOTTOM],
                b[2] + g[_RIGHT], b[3] + g[_TOP])

    # Pairs kept apart along both axes: constrain only if the regions now
    # come closer than the aisle on both, along the axis closer to holding.
    for a, b, axes, need in double:
        ra, rb = _grown(a), _grown(b)
        gaps = {ax: _axis_gap(ra, rb, ax) for ax in axes}
        if max(gaps.values()) >= need - _ROOM_FIT_TOL:
            continue
        ax = max(axes, key=lambda k: (gaps[k], -k))
        sa, sb = _sides(a, b, ax)
        short = need - gaps[ax]
        ea, eb = grow[a][sa], grow[b][sb]
        if ea + eb <= 0.0:
            continue
        grow[a][sa] = max(0.0, ea - short * ea / (ea + eb))
        grow[b][sb] = max(0.0, eb - short * eb / (ea + eb))

    extras: Dict[str, Tuple[float, float, float, float]] = {}
    for z in zone_rect:
        if z in room:
            extras[z] = tuple(max(0.0, room[z][s] - grow[z][s])
                              for s in range(4))
        else:
            extras[z] = (0.0, 0.0, 0.0, 0.0)

    # The axis each zone's fixtures run along: the one their band is long
    # on, relative to one fixture (a zone with one fixture: its long side).
    run_axis: Dict[str, int] = {}
    for z, idx in members.items():
        blk = [i for i in idx if blocking[i]] or idx
        sx, sy, sw, sh = zone_rect[z]
        if len(blk) > 1:
            b = cut_band.get(z) or _band(np.column_stack(
                [clipped[blk], sizes[blk]]))
            spread_x = (b[2] - b[0]) - sizes[blk, 0].max()
            spread_y = (b[3] - b[1]) - sizes[blk, 1].max()
            run_axis[z] = 1 if spread_y >= spread_x else 0
        else:
            run_axis[z] = 1 if sh >= sw else 0

    res = float(getattr(getattr(shop, 'customer_simulation', None),
                        'path_grid_resolution', 0.25) or 0.25)
    nx, ny = _grid_shape(shop.width, shop.height, res)
    wall_blocked = np.zeros((nx, ny), dtype=bool)
    _rasterize(wall_blocked,
               [(float(w['position'][0]), float(w['position'][1]),
                 float(w['size'][0]), float(w['size'][1]))
                for wn, w in walls.items() if wall_blocks_movement(wn, w)],
               res)
    door = shop.door_position or shop.floors[1].get('door_position') \
        or (shop.width / 2.0, 0.0)

    plan = AislePlan(key=key, names=names, zone=tuple(zone),
                     zone_rect=zone_rect, sizes=sizes, blocking=blocking,
                     extras=extras, run_axis=run_axis,
                     kept_apart=tuple(kept), as_built=clipped.copy(),
                     width=float(shop.width), height=float(shop.height),
                     res=res, wall_blocked=wall_blocked,
                     door=(float(door[0]), float(door[1])),
                     members=members,
                     stats=_fresh_stats())
    # The as-built layout, clipped into its regions, is the fallback of last
    # resort; clip it once here so it is a fixed point of the region clip.
    _clip_to_regions(plan, plan.as_built)
    return plan


# --- The repair ------------------------------------------------------------

def _fresh_stats() -> Dict[str, float]:
    """The repair's counters (``repair_stats``), all zero."""
    return {'calls': 0, 'aisle_clip_calls': 0, 'aisle_clip_items': 0,
            'aisle_clip_max_m': 0.0, 'snap_zones': 0,
            'reach_single_file': 0, 'reach_as_built_zone': 0,
            'reach_as_built_store': 0}


def _clip_to_regions(plan: AislePlan, out: np.ndarray) -> Tuple[int, float]:
    """Clip every item of a zone with a region tighter than the zone into
    it, in place; returns the number of items moved by more than
    ``GEOM_TOL`` and the largest distance one moved (metres, per axis). A
    zone whose extras are all zero is skipped: ``_ga_repair`` has already
    clipped its items to exactly that box, so skipping keeps the arithmetic
    identical."""
    moved = 0
    worst = 0.0
    for z, idx in plan.members.items():
        el, er, eb, et = plan.extras[z]
        if el == 0.0 and er == 0.0 and eb == 0.0 and et == 0.0:
            continue
        sx, sy, sw, sh = plan.zone_rect[z]
        for i in idx:
            w, h = plan.sizes[i]
            lo_x, hi_x = sx + ZONE_PAD + el, sx + sw - w - ZONE_PAD - er
            lo_y, hi_y = sy + ZONE_PAD + eb, sy + sh - h - ZONE_PAD - et
            if hi_x < lo_x:
                hi_x = lo_x
            if hi_y < lo_y:
                hi_y = lo_y
            x = min(max(out[i, 0], lo_x), hi_x)
            y = min(max(out[i, 1], lo_y), hi_y)
            if x != out[i, 0] or y != out[i, 1]:
                # A move below GEOM_TOL is float round-off between two
                # spellings of the same edge (an operator's box plus its
                # pad), not the rule binding; it is applied, not counted.
                d = max(abs(x - out[i, 0]), abs(y - out[i, 1]))
                if d > GEOM_TOL:
                    moved += 1
                    worst = max(worst, d)
                out[i, 0], out[i, 1] = x, y
    return moved, worst


def _zone_overlaps(out: np.ndarray, sizes: np.ndarray,
                   idx: Sequence[int]) -> bool:
    for k in range(len(idx)):
        i = idx[k]
        xi, yi = out[i, 0], out[i, 1]
        wi, hi = sizes[i]
        for k2 in range(k + 1, len(idx)):
            j = idx[k2]
            xj, yj = out[j, 0], out[j, 1]
            wj, hj = sizes[j]
            if not (xi + wi <= xj or xi >= xj + wj
                    or yi + hi <= yj or yi >= yj + hj):
                return True
    return False


def _snap_zone(out: np.ndarray, sizes: np.ndarray, idx: Sequence[int],
               rect: Tuple[float, float, float, float],
               extras: Tuple[float, float, float, float]) -> None:
    """The rank-preserving 2D grid snap of ``_repair_chrom_overlaps``, run
    inside the zone's region: the zone ``rect`` less ``ZONE_PAD`` and the
    region's ``extras``. With zero extras it is the snap the repair has
    always run, operation for operation.

    Items within a zone are sorted by current position into a row-major
    grid (y picks the row, x the column within it) and snapped to evenly
    spaced non-overlapping slots that preserve that order, so the layout's
    positional information survives as rank.

    A run the engine laid end to end with ``SEG_GAP`` between segments is
    0.1 m longer than the region (the zone less its pad at both ends), so
    it no longer fits at the full snap gap. Where the items still fit end
    to end, the snap keeps them in one line and spreads them evenly over
    the room there is -- a gap narrower than ``SNAP_GAP`` but no overlap --
    instead of stacking the run into a second column or letting the last
    slot overflow onto its neighbour. Where the full gap fits, as on every
    zone of the synthetic scenarios, both rules give the same slots."""
    sx, sy, sw, sh = rect
    el, er, eb, et = extras
    sx, sw = sx + el, sw - el - er
    sy, sh = sy + eb, sh - eb - et
    n = len(idx)
    max_w = max(sizes[i][0] for i in idx)
    max_h = max(sizes[i][1] for i in idx)
    gap, pad = SNAP_GAP, ZONE_PAD
    avail_w = sw - 2 * pad
    avail_h = sh - 2 * pad
    cols = max(1, int(math.floor((avail_w + gap) / (max_w + gap))))
    rows_needed = int(math.ceil(n / cols))
    if rows_needed * max_h > avail_h + GEOM_TOL:
        # Doesn't fit in this many cols even end to end; try the other
        # orientation
        rows_needed = max(1, int(math.floor((avail_h + gap) / (max_h + gap))))
        cols = int(math.ceil(n / rows_needed))
    # Preserve the rank the layout chose on both axes: y decides the row,
    # x decides the column within that row. A plain (y, x) sort would
    # order columns by y as well, since real-valued y almost never ties.
    by_y = sorted(idx, key=lambda i: out[i, 1])
    rows = [sorted(by_y[r * cols:(r + 1) * cols], key=lambda i: out[i, 0])
            for r in range(rows_needed)]
    step_x = (avail_w - max_w) / max(cols - 1, 1) if cols > 1 else 0.0
    step_y = (avail_h - max_h) / max(rows_needed - 1, 1) if rows_needed > 1 else 0.0
    # Spread evenly when the items fit end to end; below that, a step of
    # item dim + gap, clipped at the far edge (else overlap).
    step_x = step_x if step_x >= max_w else max_w + gap
    step_y = step_y if step_y >= max_h else max_h + gap
    for r, row in enumerate(rows):
        for c, i in enumerate(row):
            wi, hi = sizes[i]
            x = sx + pad + c * step_x
            y = sy + pad + r * step_y
            # Clip to section (in case step pushed past)
            x = max(sx + pad, min(x, sx + sw - wi - pad))
            y = max(sy + pad, min(y, sy + sh - hi - pad))
            out[i, 0] = x
            out[i, 1] = y


def _single_file(plan: AislePlan, out: np.ndarray, z: str) -> bool:
    """Re-pack zone ``z`` single-file along its run: each fixture back on
    its as-built line across the run, the fixtures in the order the layout
    gave them along it, spread evenly over the region. False, with ``out``
    untouched, when they do not fit end to end."""
    idx = plan.members[z]
    if len(idx) < 2:
        return False
    a = plan.run_axis[z]
    lat = 1 - a
    reg = plan.region(z)
    lo, hi = reg[a], reg[a + 2]
    lengths = [float(plan.sizes[i][a]) for i in idx]
    total = sum(lengths)
    if total > (hi - lo) + GEOM_TOL:
        return False
    order = sorted(idx, key=lambda i: (out[i, a], i))
    step_gap = ((hi - lo) - total) / (len(idx) - 1)
    pos = lo
    for i in order:
        out[i, a] = min(pos, hi - float(plan.sizes[i][a]))
        out[i, lat] = plan.as_built[i, lat]
        pos += float(plan.sizes[i][a]) + step_gap
    return not _zone_overlaps(out, plan.sizes, idx)


def repair_positions(shop, chrom: np.ndarray,
                     item_names: Sequence[str]) -> np.ndarray:
    """``chrom`` (N, 2) mapped onto the store's feasible set, after the
    zone clip (``_ga_repair``): region clip, overlap snap inside the region,
    then the reachability fallbacks. Deterministic, no RNG draws, and a
    feasible layout is a fixed point."""
    plan = aisle_plan(shop, item_names)
    out = np.array(chrom, dtype=np.float64, copy=True)
    plan.stats['calls'] += 1
    moved, worst = _clip_to_regions(plan, out)
    if moved:
        plan.stats['aisle_clip_calls'] += 1
        plan.stats['aisle_clip_items'] += moved
        plan.stats['aisle_clip_max_m'] = max(plan.stats['aisle_clip_max_m'],
                                             worst)
    for z, idx in plan.members.items():
        if len(idx) < 2:
            continue
        if not _zone_overlaps(out, plan.sizes, idx):
            continue
        plan.stats['snap_zones'] += 1
        _snap_zone(out, plan.sizes, idx, plan.zone_rect[z], plan.extras[z])

    bad = _unshoppable(plan, out)
    if not bad:
        return out
    # Fallback 1: the zones of the fixtures that cannot be shopped go
    # single-file along their run.
    for z in sorted({plan.zone[i] for i in bad if plan.zone[i] is not None}):
        trial = out.copy()
        if _single_file(plan, trial, z):
            out = trial
            plan.stats['reach_single_file'] += 1
    bad = _unshoppable(plan, out)
    if not bad:
        return out
    # Fallback 2: those zones back to their as-built positions.
    for z in sorted({plan.zone[i] for i in bad if plan.zone[i] is not None}):
        idx = plan.members[z]
        out[idx] = plan.as_built[idx]
        plan.stats['reach_as_built_zone'] += 1
    if not _unshoppable(plan, out):
        return out
    # Fallback 3: the store as built.
    plan.stats['reach_as_built_store'] += 1
    return plan.as_built.copy()


def repair_stats(shop) -> Dict[str, float]:
    """Counters of the repair on this store since its plan was built (or
    ``reset_repair_stats``): calls; calls in which the region clip moved an
    item (``aisle_clip_calls``), the items it moved and the farthest it moved
    one (``aisle_clip_max_m``); zones snapped; each reachability fallback.

    Every search's operators draw inside the regions (``move_box``,
    ``position_bounds``), so the region clip is a safety net that a search
    run leaves at or near zero: the aisle rule binds by bounding where the
    operators may put a fixture, which ``aisle_rule_summary`` measures, not
    by clipping what they propose."""
    cache = shop.__dict__.get('_aisle_plan_cache') or {}
    return combine_repair_stats(plan.stats for plan in cache.values())


def combine_repair_stats(records) -> Dict[str, float]:
    """Several ``repair_stats`` records as one: counts summed, the
    distances (keys ending ``_m``) their maximum."""
    out: Dict[str, float] = {}
    for rec in records:
        for k, v in rec.items():
            if k.endswith('_m'):
                out[k] = max(out.get(k, 0.0), float(v))
            else:
                out[k] = out.get(k, 0) + int(v)
    return out


def reset_repair_stats(shop) -> None:
    for plan in (shop.__dict__.get('_aisle_plan_cache') or {}).values():
        plan.stats.update(_fresh_stats())


def aisle_rule_summary(shop, item_names: Sequence[str]) -> Dict[str, Any]:
    """How much the aisle rule takes from the search space of this store.

    Per item and axis, the ROOM is the length of the range of positions its
    rectangle can take: in its zone less the repair's pad (what the repair
    allowed before the rule), and in its region (what it allows now). The
    rule binds where the region's room is shorter. Reported: the zones and
    the pairs kept apart; the items the rule narrows on some axis; per axis,
    the zone and region room (median over narrowed items) and the share of
    its zone room each narrowed item keeps (median and minimum); the items
    the rule PINS on an axis (no room left there, as on a run whose aisle
    slack is all taken); and the share of the zone's position area (the
    product of the per-axis shares, an axis without room in the zone
    counting as kept) the regions keep -- median over narrowed items, and
    the geometric mean over the items not pinned, the per-item factor by
    which the joint search space shrinks. On the synthetic template every
    share is 1 and nothing is pinned."""
    plan = aisle_plan(shop, item_names)
    zone_room, region_room = [], []
    for i, z in enumerate(plan.zone):
        if z is None:
            continue
        w, h = plan.sizes[i]
        sx, sy, sw, sh = plan.zone_rect[z]
        x0, y0, x1, y1 = plan.region(z)
        zone_room.append((max(0.0, sw - 2 * ZONE_PAD - w),
                          max(0.0, sh - 2 * ZONE_PAD - h)))
        region_room.append((max(0.0, x1 - x0 - w), max(0.0, y1 - y0 - h)))
    zr = np.asarray(zone_room, dtype=np.float64).reshape(-1, 2)
    rr = np.asarray(region_room, dtype=np.float64).reshape(-1, 2)
    narrowed_axis = rr < zr - GEOM_TOL
    narrowed = narrowed_axis.any(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        keep = np.where(zr > GEOM_TOL, rr / np.where(zr > GEOM_TOL, zr, 1.0),
                        1.0)
    area = keep.prod(axis=1)
    pinned = area <= GEOM_TOL
    per_axis = {}
    for ax, name in ((0, 'x'), (1, 'y')):
        sel = narrowed_axis[:, ax]
        per_axis[name] = {
            'n_items_narrowed': int(sel.sum()),
            'zone_room_median_m': (float(np.median(zr[sel, ax]))
                                   if sel.any() else None),
            'region_room_median_m': (float(np.median(rr[sel, ax]))
                                     if sel.any() else None),
            'share_kept_median': (float(np.median(keep[sel, ax]))
                                  if sel.any() else 1.0),
            'share_kept_min': (float(keep[sel, ax].min())
                               if sel.any() else 1.0)}
    unpinned = area[~pinned]
    return {
        'n_items': int(len(zr)),
        'n_zones': int(len(plan.members)),
        'n_zones_narrowed': int(sum(1 for ex in plan.extras.values()
                                    if any(e > 0 for e in ex))),
        'n_pairs_kept_apart': int(len(plan.kept_apart)),
        'min_aisle_m': float(MIN_AISLE),
        'n_items_narrowed': int(narrowed.sum()),
        'per_axis': per_axis,
        'area_share_kept_median_narrowed': (float(np.median(area[narrowed]))
                                            if narrowed.any() else 1.0),
        'area_share_kept_geomean_unpinned': (
            float(np.exp(np.mean(np.log(unpinned)))) if unpinned.size
            else None),
        'n_items_pinned': int(pinned.sum()),
    }


# --- Reachability ------------------------------------------------------------

def _dist_to_rect(px: float, py: float, x: float, y: float,
                  w: float, h: float) -> float:
    return math.hypot(max(x - px, 0.0, px - (x + w)),
                      max(y - py, 0.0, py - (y + h)))


# Cell offsets within ``_OFFSET_RADIUS`` cells, nearest first; equal
# distances in (dx, dy) order, which is the order of the cells' own indices
# for a fixed centre -- the tie-break ``sim_geometry._nearest_free_cell``
# applies (the lowest index among the nearest free cells).
_OFFSET_RADIUS = 8
_OFF = np.array([(dx, dy)
                 for dx in range(-_OFFSET_RADIUS, _OFFSET_RADIUS + 1)
                 for dy in range(-_OFFSET_RADIUS, _OFFSET_RADIUS + 1)
                 if dx * dx + dy * dy <= _OFFSET_RADIUS ** 2])
_OFF = _OFF[np.lexsort((_OFF[:, 1], _OFF[:, 0],
                        _OFF[:, 0] ** 2 + _OFF[:, 1] ** 2))]


def _access_cells(blocked: np.ndarray, cx: np.ndarray,
                  cy: np.ndarray) -> List[Optional[Tuple[int, int]]]:
    """``sim_geometry._nearest_free_cell`` for many points at once (cell
    units): the free cell nearest the cell each point falls in, lowest
    index on ties, or None on a floor with no free cell. The offsets within
    ``_OFFSET_RADIUS`` cells are tried nearest first; a point with no free
    cell that close falls back to the simulator's own search. Same answer,
    cell for cell."""
    nx, ny = blocked.shape
    gx = np.clip(cx.astype(np.int64), 0, nx - 1)
    gy = np.clip(cy.astype(np.int64), 0, ny - 1)
    px = gx[:, None] + _OFF[None, :, 0]
    py = gy[:, None] + _OFF[None, :, 1]
    inside = (px >= 0) & (px < nx) & (py >= 0) & (py < ny)
    free = np.zeros(px.shape, dtype=bool)
    free[inside] = ~blocked[px[inside], py[inside]]
    first = free.argmax(axis=1)
    out: List[Optional[Tuple[int, int]]] = []
    for k in range(len(gx)):
        if free[k, first[k]]:
            out.append((int(px[k, first[k]]), int(py[k, first[k]])))
        else:
            out.append(_nearest_free_cell(blocked, float(gx[k]),
                                          float(gy[k])))
    return out


def _shoppable(plan: AislePlan, pos: np.ndarray) -> np.ndarray:
    """Per item: True when its access point -- the free cell of the agent
    grid nearest its centre, the spot ``sim_geometry`` sends an agent to --
    lies within the arrival margin of the fixture and is connected to the
    cell an agent enters at."""
    res = plan.res
    blocked = plan.wall_blocked.copy()
    rects = [(pos[i, 0], pos[i, 1], plan.sizes[i, 0], plan.sizes[i, 1])
             for i in range(len(plan.names)) if plan.blocking[i]]
    _rasterize(blocked, rects, res)
    labels, _ = ndimage.label(~blocked)
    start = _nearest_free_cell(blocked, plan.door[0] / res,
                               (plan.door[1] + DOOR_STEP_IN_M) / res)
    ok = np.zeros(len(plan.names), dtype=bool)
    if start is None:
        return ok
    home = labels[start]
    # An agent that has arrived stands within its radius plus one grid cell
    # (plus the step it stops on) of the fixture: Customer._item_arrival_margin.
    margin = AGENT_RADIUS_M + res + 0.05
    cells = _access_cells(blocked, (pos[:, 0] + plan.sizes[:, 0] / 2.0) / res,
                          (pos[:, 1] + plan.sizes[:, 1] / 2.0) / res)
    for i, cell in enumerate(cells):
        if cell is None or labels[cell] != home:
            continue
        x, y = pos[i]
        w, h = plan.sizes[i]
        ok[i] = (_dist_to_rect(cell[0] * res, cell[1] * res, x, y, w, h)
                 <= margin + GEOM_TOL)
    return ok


def _unshoppable(plan: AislePlan, pos: np.ndarray) -> List[int]:
    return [int(i) for i in np.flatnonzero(~_shoppable(plan, pos))]


def as_built_unshoppable(plan: AislePlan) -> Tuple[str, ...]:
    """The fixtures that cannot be shopped in the store as built (the
    plan's as-built layout, after the repair's clips), cached on the plan.

    Empty on every store the floor-plan engine builds, which guarantees
    door-to-fixture reachability. Not empty on a GUI floor edited into an
    unshoppable state (a fixture walled into a pocket, fixtures packed
    shut): there no candidate can pass the repair's reachability check
    while that fixture stays where it is, so ``repair_positions`` would
    hand back the whole as-built store for every candidate. The GUI reads
    this to fall back to the zone-only repair instead
    (``GARunMixin._ga_shared_plan``)."""
    cached = plan.__dict__.get('_as_built_unshoppable')
    if cached is None:
        cached = tuple(plan.names[i] for i in _unshoppable(plan,
                                                           plan.as_built))
        plan.__dict__['_as_built_unshoppable'] = cached
    return cached


# --- The invariant check ---------------------------------------------------

def layout_violations(shop, item_names: Sequence[str],
                      layout: Mapping[str, Tuple[float, float]]
                      ) -> Dict[str, list]:
    """Every way ``layout`` breaks the floor-plan engine's invariants on
    this store (empty lists when it breaks none):

      * ``outside_zone`` -- items not inside their zone;
      * ``overlap`` -- pairs of items overlapping with positive area (the
        fitness charges a full penalty for any);
      * ``clearance`` -- ``(a, b, gap, required)`` for fixtures of two zones
        the as-built store keeps an aisle apart that stand closer than it;
      * ``unshoppable`` -- fixtures whose access point is cut off from the
        door or out of arrival distance."""
    plan = aisle_plan(shop, item_names)
    names = plan.names
    pos = np.array([layout[n] for n in names], dtype=np.float64)
    sizes = plan.sizes
    out: Dict[str, list] = {'outside_zone': [], 'overlap': [],
                            'clearance': [], 'unshoppable': []}
    for i, n in enumerate(names):
        z = plan.zone[i]
        if z is None:
            continue
        sx, sy, sw, sh = plan.zone_rect[z]
        x, y = pos[i]
        w, h = sizes[i]
        if not (x >= sx - GEOM_TOL and y >= sy - GEOM_TOL
                and x + w <= sx + sw + GEOM_TOL
                and y + h <= sy + sh + GEOM_TOL):
            out['outside_zone'].append(n)
    x0, y0 = pos[:, 0], pos[:, 1]
    x1, y1 = x0 + sizes[:, 0], y0 + sizes[:, 1]
    sep = ((x1[:, None] <= x0[None, :]) | (x0[:, None] >= x1[None, :])
           | (y1[:, None] <= y0[None, :]) | (y0[:, None] >= y1[None, :]))
    for i, j in zip(*np.nonzero(np.triu(~sep, k=1))):
        out['overlap'].append((names[i], names[j]))
    gx = np.maximum(x0[None, :] - x1[:, None], x0[:, None] - x1[None, :])
    gy = np.maximum(y0[None, :] - y1[:, None], y0[:, None] - y1[None, :])
    cheb = np.maximum(gx, gy)
    for za, zb, need in plan.kept_apart:
        ia = [i for i in plan.members[za] if plan.blocking[i]]
        ib = [i for i in plan.members[zb] if plan.blocking[i]]
        if not ia or not ib:
            continue
        sub = cheb[np.ix_(ia, ib)]
        for a, b in zip(*np.nonzero(sub < need - GEOM_TOL)):
            out['clearance'].append((names[ia[a]], names[ib[b]],
                                     float(sub[a, b]), float(need)))
    ok = _shoppable(plan, pos)
    out['unshoppable'] = [names[i] for i in np.flatnonzero(~ok)]
    return out


def check_layout_invariants(shop, item_names: Sequence[str],
                            layout: Mapping[str, Tuple[float, float]],
                            label: str = 'layout') -> None:
    """Raise ``LayoutInvariantError`` naming the first few violations when
    ``layout`` breaks any invariant of ``layout_violations``."""
    v = layout_violations(shop, item_names, layout)
    bad = {k: x for k, x in v.items() if x}
    if bad:
        parts = [f"{k}: {len(x)} ({x[:3]}{' ...' if len(x) > 3 else ''})"
                 for k, x in bad.items()]
        raise LayoutInvariantError(
            f"{label} breaks the floor-plan invariants -- " + '; '.join(parts))


def invariant_summary(shop, item_names: Sequence[str],
                      layout: Mapping[str, Tuple[float, float]]
                      ) -> Dict[str, Any]:
    """What a run records about one reported layout: the counts of each
    violation (all zero, or the run would have stopped), and the smallest
    clearance between fixtures of zones kept apart."""
    plan = aisle_plan(shop, item_names)
    v = layout_violations(shop, item_names, layout)
    pos = np.array([layout[n] for n in plan.names], dtype=np.float64)
    sizes = plan.sizes
    min_clear = None
    for za, zb, _need in plan.kept_apart:
        for i in plan.members[za]:
            if not plan.blocking[i]:
                continue
            for j in plan.members[zb]:
                if not plan.blocking[j]:
                    continue
                g = max(max(pos[j, 0] - pos[i, 0] - sizes[i, 0],
                            pos[i, 0] - pos[j, 0] - sizes[j, 0]),
                        max(pos[j, 1] - pos[i, 1] - sizes[i, 1],
                            pos[i, 1] - pos[j, 1] - sizes[j, 1]))
                min_clear = g if min_clear is None else min(min_clear, g)
    return {**{f'n_{k}': len(x) for k, x in v.items()},
            'min_clearance_kept_apart_m': (None if min_clear is None
                                           else float(min_clear)),
            'required_clearance_m': float(MIN_AISLE)}
