import time
import math
from threading import Thread
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from customer import Customer
from customer_pathfinding import AGENT_RADIUS_M, obstacle_rects
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import random


def _nearest_free_cell(blocked, gx, gy):
    """Free cell nearest the cell that holds the point (gx, gy), in cell
    units, or None when the whole floor is blocked.

    The point is mapped to its cell the way the A* planner maps a goal,
    and distances are counted from that cell. The window grows until the
    winner is closer than its own radius, so a cell just outside a small
    window cannot be nearer than the one returned. Ties go to the lowest
    index, which keeps the answer the same however the search is sized.
    """
    nx, ny = blocked.shape
    gx = min(max(int(gx), 0), nx - 1)
    gy = min(max(int(gy), 0), ny - 1)
    if not blocked[gx, gy]:
        return gx, gy

    limit = max(nx, ny)
    rad = 2
    while True:
        x0, x1 = max(0, gx - rad), min(nx, gx + rad + 1)
        y0, y1 = max(0, gy - rad), min(ny, gy + rad + 1)
        free = np.argwhere(~blocked[x0:x1, y0:y1])
        if len(free):
            dx = free[:, 0] + x0 - gx
            dy = free[:, 1] + y0 - gy
            d2 = dx * dx + dy * dy
            k = int(np.argmin(d2))
            if d2[k] <= rad * rad or rad >= limit:
                return int(free[k, 0]) + x0, int(free[k, 1]) + y0
        if rad >= limit:
            return None
        rad = min(rad * 2, limit)


def _item_access_points(items, blocked, res):
    """Where an agent stands to shop each fixture on a floor: the free grid
    cell nearest the fixture's centre.

    Cached alongside the grid it is read from, because the standing spot
    moves whenever the layout does. A contiguous run of shelving resolves
    to the aisle beside it. On a floor with no free cell at all a fixture
    keeps its centre, so the agent still has somewhere to walk.
    """
    out = {}
    for name, data in items.items():
        pos, size = data['position'], data['size']
        cx = pos[0] + size[0] / 2.0
        cy = pos[1] + size[1] / 2.0
        cell = _nearest_free_cell(blocked, cx / res, cy / res)
        out[name] = ((cx, cy) if cell is None
                     else (cell[0] * res, cell[1] * res))
    return out


class GeometryMixin:
    """Geometry caches, wall bins, and heatmap helpers for
    CustomerFlowSimulation -- split out of simulation.py to keep that file
    manageable."""

    def _rebuild_geometry_caches(self):
        """Build the pathfinding grid, obstacle bins and fixture access
        points per floor.

        Fixtures block movement exactly as walls do, so both go into the
        same caches, each inflated by the agent radius: a grid cell is free
        only when an agent standing on it clears the obstacle.
        """
        res = self.path_grid_resolution
        W, H = self.shop.width, self.shop.height
        nx = int(W / res) + 1
        ny = int(H / res) + 1
        bin_size = self.wall_bin_size
        r = AGENT_RADIUS_M

        floors = getattr(self.shop, 'floors', None)
        if not floors:
            floors = {1: {'walls': self.shop.walls,
                          'items': getattr(self.shop, 'items', {})}}

        # Fill local dicts and publish them together at the end: a rebuild
        # that raises partway then leaves the previous, complete set in
        # place instead of a half-filled one that agents on the missing
        # floors would plan against. Each floor's dicts are snapshotted
        # because a Tk-thread edit can resize them mid-iteration.
        grids, bins_by_floor, access_by_floor = {}, {}, {}

        for fid, fdata in list(floors.items()):
            walls = dict(fdata.get('walls', {}) or {})
            items = dict(fdata.get('items', {}) or {})
            blocked = np.zeros((nx, ny), dtype=np.bool_)
            bins = {}
            for ox, oy, ow, oh in obstacle_rects(walls, items):
                # A cell is blocked exactly when an agent standing on it
                # would collide, so the free cell next to a fixture is a
                # spot the agent can both reach and shop from. Rounding
                # the low edge outward instead would push it a whole cell
                # further away on that side only.
                x0 = max(0, int(math.ceil((ox - r) / res)))
                y0 = max(0, int(math.ceil((oy - r) / res)))
                x1 = min(nx - 1, int((ox + ow + r) / res))
                y1 = min(ny - 1, int((oy + oh + r) / res))
                blocked[x0:x1+1, y0:y1+1] = True

                i0 = int(ox // bin_size); j0 = int(oy // bin_size)
                i1 = int((ox + ow) // bin_size); j1 = int((oy + oh) // bin_size)
                for i in range(i0, i1 + 1):
                    for j in range(j0, j1 + 1):
                        bins.setdefault((i, j), []).append((ox, oy, ow, oh))

            grids[fid] = blocked
            bins_by_floor[fid] = bins
            access_by_floor[fid] = _item_access_points(items, blocked, res)

        self.path_blocked_grid_by_floor = grids
        self.wall_bins_by_floor = bins_by_floor
        self.item_access_points_by_floor = access_by_floor

        # Backwards-compat: keep "current floor" pointers populated.
        cur = getattr(self.shop, 'current_floor', 1)
        self.path_blocked_grid = grids.get(cur)
        self.path_grid_shape = (nx, ny)
        self.wall_bins = bins_by_floor.get(cur, {})

    def item_access_point(self, name, data, floor=None):
        """Where an agent stands to shop `name`, so it stops in the aisle
        instead of inside the shelving.

        An impulse display is not an obstacle, so the spot beside it is
        normally the cell it stands on itself. Falls back to the fixture's
        centre for a floor whose caches have not been built yet.
        """
        fid = floor if floor is not None else data.get('floor', 1)
        by_floor = getattr(self, 'item_access_points_by_floor', None) or {}
        points = by_floor.get(fid) or {}
        point = points.get(name)
        if point is None:
            # The flattened catalogue renames items duplicated across
            # floors; the cache is keyed by the name on the floor itself.
            point = points.get(data.get('source_name'))
        if point is not None:
            return [point[0], point[1]]
        pos, size = data['position'], data['size']
        return [pos[0] + size[0] / 2.0, pos[1] + size[1] / 2.0]

    def nearest_free_position(self, x, y, floor=1):
        """Closest spot on this floor an agent can stand on, or None when
        the floor has no grid yet."""
        grids = getattr(self, 'path_blocked_grid_by_floor', None) or {}
        blocked = grids.get(floor)
        if blocked is None:
            return None
        res = self.path_grid_resolution
        cell = _nearest_free_cell(blocked, x / res, y / res)
        if cell is None:
            return None
        return [cell[0] * res, cell[1] * res]

    def _free_trapped_customers(self):
        """Lift any agent that a layout change left inside a fixture onto
        the nearest free spot.

        An agent inside an obstacle cannot step anywhere, because every
        direction it tries is still inside it, so it would stand there
        until its stall window ran out. Its current waypoints are kept: if
        they are no longer walkable the usual stall recovery re-targets it.
        """
        floors = getattr(self.shop, 'floors', None) or {}
        for cust in list(self.customers):
            walls = (floors.get(cust.floor) or {}).get('walls', self.shop.walls)
            if not cust._collides_with_interior(cust.position[0],
                                                cust.position[1], walls):
                continue
            spot = self.nearest_free_position(cust.position[0],
                                              cust.position[1], cust.floor)
            if spot is not None:
                cust.position[0], cust.position[1] = spot
                cust._reset_progress()

    def _candidate_walls(self, x, y, radius, floor=None):
        """Obstacles near a point on the given floor: blocking walls and the
        fixtures agents walk around."""
        bins_by_floor = getattr(self, 'wall_bins_by_floor', None)
        if floor is not None:
            # Only this floor's bins; another floor's would report its
            # obstacles at our coordinates.
            bins = bins_by_floor.get(floor) if bins_by_floor else None
        else:
            bins = self.wall_bins or {}

        if not bins:
            # fallback: scan that floor's obstacles directly
            floors = getattr(self.shop, 'floors', None)
            fdata = (floors.get(floor) if floors and floor is not None
                     else None)
            if fdata is None:
                walls = self.shop.walls
                items = getattr(self.shop, 'items', {})
            else:
                walls = fdata.get('walls', {})
                items = fdata.get('items', {})
            return list(obstacle_rects(walls, items))

        bin_size = self.wall_bin_size
        i = int(x // bin_size); j = int(y // bin_size)
        r_bins = max(1, int(math.ceil(radius / bin_size)))
        seen, out = set(), []
        for di in range(-r_bins, r_bins + 1):
            for dj in range(-r_bins, r_bins + 1):
                for rect in bins.get((i + di, j + dj), ()):
                    if rect not in seen:
                        seen.add(rect); out.append(rect)
        return out

    def _refresh_smoothed_heatmap(self):
        """Blur raw heatmap"""
        if self.optimize_smoothing == False:
            K = self._heat_kernel_1d
            A = self.heat_raw
            tmp = np.apply_along_axis(lambda m: np.convolve(m, K, mode='same'), 0, A)
            smoothed = np.apply_along_axis(lambda m: np.convolve(m, K, mode='same'), 1, tmp)
            self.heat_map_data = smoothed
            return

        # optimized blur
        A = self.heat_raw

        A0 = np.pad(A, ((1, 1), (0, 0)), mode='edge')
        tmp = (A0[:-2, :] + 2.0 * A0[1:-1, :] + A0[2:, :]) * 0.25

        tmp0 = np.pad(tmp, ((0, 0), (1, 1)), mode='edge')
        smoothed = (tmp0[:, :-2] + 2.0 * tmp0[:, 1:-1] + tmp0[:, 2:]) * 0.25

        self.heat_map_data = smoothed


    def reset_heatmap(self):
        """Clear heatmap data"""
        try:
            if hasattr(self, 'heat_raw') and self.heat_raw is not None:
                self.heat_raw.fill(0)
            else:
                res = getattr(self, 'heat_map_resolution', 20)
                wcells = int(self.shop.width  * res) + 1
                hcells = int(self.shop.height * res) + 1
                self.heat_raw = np.zeros((wcells, hcells), dtype=np.float32)

            if hasattr(self, 'heat_map_data') and self.heat_map_data is not None:
                self.heat_map_data.fill(0)
            else:
                self.heat_map_data = np.zeros_like(self.heat_raw, dtype=np.float32)
        except Exception:
            res = getattr(self, 'heat_map_resolution', 20)
            wcells = int(self.shop.width  * res) + 1
            hcells = int(self.shop.height * res) + 1
            self.heat_raw      = np.zeros((wcells, hcells), dtype=np.float32)
            self.heat_map_data = np.zeros((wcells, hcells), dtype=np.float32)

        self.last_heat_update = 0.0

        # Clear per-floor heatmap storage
        if hasattr(self, '_floor_heat_raw'):
            self._floor_heat_raw.clear()

        if 'foot_traffic_density' in self.analytics:
            self.analytics['foot_traffic_density'].clear()
        if 'bottlenecks' in self.analytics:
            self.analytics['bottlenecks'].clear()
