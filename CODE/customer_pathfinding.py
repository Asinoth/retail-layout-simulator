import heapq


class PathfindingMixin:
   """A* pathfinding for the Customer class -- split out of customer.py to
   keep that file manageable."""

   def _compute_path(self, start, goal, walls):
        """A* pathfinding. returns list of waypoints"""
        import heapq
        sim = getattr(self, "_simulation_ref", None)
        blocked = None
        if sim is not None:
            grids = getattr(sim, 'path_blocked_grid_by_floor', None)
            if grids and self.floor in grids:
                blocked = grids[self.floor]
            elif sim.path_blocked_grid is not None:
                blocked = sim.path_blocked_grid
            if blocked is not None:
                res = sim.path_grid_resolution
                nx, ny = blocked.shape
        else:
            res = 0.25
            w_shop, h_shop = self.shop_dimensions
            nx = int(w_shop / res) + 1
            ny = int(h_shop / res) + 1
            blocked = [[False] * ny for _ in range(nx)]
            r = self.size
            for nm, data in walls.items():
                if nm.startswith('Section_') or nm.startswith('Checkout') or nm == 'WC':
                    continue
                # Connectors are walkable: skip them when building the
                # blocked grid (mirrors sim_geometry._rebuild_geometry_caches).
                if data.get('category') == 'Connector':
                    continue
                wx, wy = data['position']
                ww, wh = data['size']
                x0 = max(0, int((wx - r) / res))
                y0 = max(0, int((wy - r) / res))
                x1 = min(nx - 1, int((wx + ww + r) / res))
                y1 = min(ny - 1, int((wy + wh + r) / res))
                for ix in range(x0, x1 + 1):
                    col = blocked[ix]
                    for iy in range(y0, y1 + 1):
                        col[iy] = True

        def to_grid(pt):
            gx = min(max(int(pt[0] / res), 0), nx - 1)
            gy = min(max(int(pt[1] / res), 0), ny - 1)
            return (gx, gy)

        sx, sy = to_grid(start)
        gx, gy = to_grid(goal)

        if (sx, sy) == (gx, gy):
            return []

        if isinstance(blocked, list):
            def is_blocked(x, y):
                if (x, y) == (sx, sy) or (x,y) == (gx, gy):
                    return False
                return blocked[x][y]
        else:
            def is_blocked(x,y):
                if (x, y) == (sx,sy) or (x, y) == (gx, gy):
                    return False
                return bool(blocked[x, y])

        # A* 4-neighbor
        open_set = [(0, sx, sy)]
        g_score = {(sx, sy): 0}
        came_from = {}
        
        def hcost(x, y):
            return abs(x - gx) + abs(y - gy)

        while len(open_set) > 0:
            _, x, y = heapq.heappop(open_set)
            if (x, y) == (gx, gy):
                break
            for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                nx0, ny0 = x + dx, y + dy
                if 0 <= nx0 < nx and 0 <= ny0 < ny and not is_blocked(nx0, ny0):
                    tentative = g_score[(x, y)] + 1
                    if tentative < g_score.get((nx0, ny0), float('inf')):
                        g_score[(nx0, ny0)] = tentative
                        f = tentative + hcost(nx0, ny0)
                        heapq.heappush(open_set, (f, nx0, ny0))
                        came_from[(nx0, ny0)] = (x, y)
        else:
            return []

        # reconstruct path
        path = []
        node = (gx, gy)
        while node != (sx, sy):
            path.append((node[0] * res, node[1] * res))
            node = came_from.get(node)
            if node is None:
                return []
        return list(reversed(path))

