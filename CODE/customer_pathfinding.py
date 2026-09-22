import heapq
import math


# Radius an obstacle is inflated by so a grid cell counts as free only when
# an agent standing on it clears the obstacle. Matches ``Customer.size``.
AGENT_RADIUS_M = 0.15


def wall_blocks_movement(name, data):
    """True when a wall stops an agent.

    Section rectangles are bookkeeping zones, every checkout lane and the
    WC are walk-up counters agents step up to, and connectors are the
    stair tiles they stand on to change floor.
    """
    if name.startswith('Section_') or name.startswith('Checkout') or name == 'WC':
        return False
    if data.get('category') == 'Connector':
        return False
    return True


def item_blocks_movement(data):
    """True when a fixture stops an agent.

    Shelving, gondolas and display tables are furniture: agents walk
    around them and shop from the aisle beside them. Impulse displays are
    the low racks at the lane, small enough to reach across, and the
    checkout catchment measures distance to them rather than a walk.
    """
    return 'impulse' not in str(data.get('category') or '').lower()


def obstacle_rects(walls, items=None):
    """(x, y, w, h) for everything on one floor an agent must walk around.

    The one place the obstacle set is defined; the collision test, the
    wall bins, the A* grid and the fallback grid all read it, so they
    cannot drift apart.
    """
    for name, data in (walls or {}).items():
        if wall_blocks_movement(name, data):
            wx, wy = data['position']
            ww, wh = data['size']
            yield (wx, wy, ww, wh)
    for name, data in (items or {}).items():
        if item_blocks_movement(data):
            ix, iy = data['position']
            iw, ih = data['size']
            yield (ix, iy, iw, ih)


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
            if grids is not None:
                # Per-floor grids exist, so use this floor's or none at
                # all: another floor's grid describes another floor's
                # obstacles, and planning on it walks agents into walls.
                blocked = grids.get(self.floor)
            elif sim.path_blocked_grid is not None:
                blocked = sim.path_blocked_grid
            if blocked is not None:
                res = sim.path_grid_resolution
                nx, ny = blocked.shape
        # No simulation, or one without a grid for this floor (its geometry
        # rebuild has not run or failed): plan on a grid built here.
        if blocked is None:
            res = 0.25
            w_shop, h_shop = self.shop_dimensions
            nx = int(w_shop / res) + 1
            ny = int(h_shop / res) + 1
            blocked = [[False] * ny for _ in range(nx)]
            r = self.size
            # Fixtures block movement too. An agent driven without a
            # simulation has no catalogue to read them from, and plans
            # around the walls alone.
            items = None
            if sim is not None:
                floors = getattr(getattr(sim, 'shop', None), 'floors', None) or {}
                items = (floors.get(self.floor) or {}).get('items')
            for wx, wy, ww, wh in obstacle_rects(walls, items):
                # Same rule as the simulation's grid: a cell is blocked
                # exactly when an agent standing on it would collide.
                x0 = max(0, int(math.ceil((wx - r) / res)))
                y0 = max(0, int(math.ceil((wy - r) / res)))
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

        # A* 4-neighbor. Equal-f ties go to the larger g, so the search
        # presses toward the goal across the wide equal-f plateaus of an
        # open shop floor instead of flooding them in coordinate order.
        open_set = [(0, 0, sx, sy)]
        g_score = {(sx, sy): 0}
        came_from = {}
        
        def hcost(x, y):
            return abs(x - gx) + abs(y - gy)

        while len(open_set) > 0:
            _, neg_g, x, y = heapq.heappop(open_set)
            if (x, y) == (gx, gy):
                break
            if -neg_g > g_score[(x, y)]:
                continue   # superseded by a cheaper entry for this cell
            for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                nx0, ny0 = x + dx, y + dy
                if 0 <= nx0 < nx and 0 <= ny0 < ny and not is_blocked(nx0, ny0):
                    tentative = g_score[(x, y)] + 1
                    if tentative < g_score.get((nx0, ny0), float('inf')):
                        g_score[(nx0, ny0)] = tentative
                        f = tentative + hcost(nx0, ny0)
                        heapq.heappush(open_set, (f, -tentative, nx0, ny0))
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
        path = list(reversed(path))
        # Waypoints sit on grid-cell corners, up to res*sqrt(2) from the goal.
        # On a top or right door that corner can fall outside the door's
        # despawn radius, so exit paths end on the door itself.
        if getattr(self, 'current_target_type', None) == 'exit':
            path[-1] = (float(goal[0]), float(goal[1]))
        return path

