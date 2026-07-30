import time
import math
from threading import Thread
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from customer import Customer
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import random


class GeometryMixin:
    """Geometry caches, wall bins, and heatmap helpers for
    CustomerFlowSimulation -- split out of simulation.py to keep that file
    manageable."""

    def _rebuild_geometry_caches(self):
        """Build pathfinding grid + wall bins per-floor."""
        res = self.path_grid_resolution
        W, H = self.shop.width, self.shop.height
        nx = int(W / res) + 1
        ny = int(H / res) + 1
        bin_size = self.wall_bin_size
        r = 0.15

        self.path_blocked_grid_by_floor = {}
        self.wall_bins_by_floor = {}

        floors = getattr(self.shop, 'floors', None)
        if not floors:
            floors = {1: {'walls': self.shop.walls}}

        for fid, fdata in floors.items():
            walls = fdata.get('walls', {})
            blocked = np.zeros((nx, ny), dtype=np.bool_)
            bins = {}
            for nm, data in walls.items():
                # All checkout lanes are walk-up counters (walkable), like WC.
                if nm.startswith('Section_') or nm.startswith('Checkout') or nm == 'WC':
                    continue
                if data.get('category') == 'Connector':
                    continue
                wx, wy = data['position']
                ww, wh = data['size']
                x0 = max(0, int((wx - r) / res))
                y0 = max(0, int((wy - r) / res))
                x1 = min(nx - 1, int((wx + ww + r) / res))
                y1 = min(ny - 1, int((wy + wh + r) / res))
                blocked[x0:x1+1, y0:y1+1] = True

                i0 = int(wx // bin_size); j0 = int(wy // bin_size)
                i1 = int((wx + ww) // bin_size); j1 = int((wy + wh) // bin_size)
                for i in range(i0, i1 + 1):
                    for j in range(j0, j1 + 1):
                        bins.setdefault((i, j), []).append((wx, wy, ww, wh))

            self.path_blocked_grid_by_floor[fid] = blocked
            self.wall_bins_by_floor[fid] = bins

        # Backwards-compat: keep "current floor" pointers populated.
        cur = getattr(self.shop, 'current_floor', 1)
        self.path_blocked_grid = self.path_blocked_grid_by_floor.get(cur)
        self.path_grid_shape = (nx, ny)
        self.wall_bins = self.wall_bins_by_floor.get(cur, {})


    def _candidate_walls(self, x, y, radius, floor=None):
        """Get walls near point on the given floor."""
        bins_by_floor = getattr(self, 'wall_bins_by_floor', None)
        if floor is not None and bins_by_floor and floor in bins_by_floor:
            bins = bins_by_floor[floor]
        else:
            bins = self.wall_bins or {}

        if not bins:
            # fallback: scan that floor's walls directly
            floors = getattr(self.shop, 'floors', None)
            walls = (floors[floor]['walls']
                    if floors and floor in floors
                    else self.shop.walls)
            out = []
            for nm, data in walls.items():
                if nm.startswith('Section_') or nm.startswith('Checkout') or nm == 'WC':
                    continue
                if data.get('category') == 'Connector':
                    continue
                wx, wy = data['position']; ww, wh = data['size']
                out.append((wx, wy, ww, wh))
            return out

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
