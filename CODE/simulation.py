import time
import math
from threading import Thread
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import queue as _queue
from customer import Customer
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import random

from sim_analytics import AnalyticsMixin
from sim_geometry import GeometryMixin


class CustomerFlowSimulation(AnalyticsMixin, GeometryMixin):
    """Manages customer flow sim for shop."""

    def __init__(self, shop_visualizer):
        self.shop = shop_visualizer
        self.customers = []
        self.customer_counter = 0
        self.running = False
        self.paused = False
        self.simulation_speed = 1.0
        self.spawn_rate = 0.2
        # Maximum number of concurrently simulated customers. Replaces the
        # former hard-coded cap of 40. The Simulation tab exposes a Spinbox
        # bound to this attribute (1-100, default 20). Keep it as a plain
        # int -- the spawn check reads it on every tick so changes take
        # effect live without restarting the loop.
        self.max_customers = 20
        self.last_spawn_time = 0
        self.last_update_time = time.perf_counter()
        self.target_fps = 20.0
        self.last_draw_time = time.perf_counter()
        self.draw_interval = 1.0 / self.target_fps
        self.last_heat_update = time.perf_counter()
        self.heat_update_interval = 1.0

        self.heat_map_resolution = 20
        wcells = int(self.shop.width  * self.heat_map_resolution) + 1
        hcells = int(self.shop.height * self.heat_map_resolution) + 1
        self.heat_map_data = np.zeros((wcells, hcells))

        self.door_position = None
        self.door_side = None
        self.seed = None

        self.path_grid_resolution = 0.25
        self.path_blocked_grid = None
        self.path_grid_shape = None
        self.geometry_dirty = True

        self.wall_bin_size = 1.0
        self.wall_bins = None

        self.heat_raw = np.zeros_like(self.heat_map_data, dtype=np.float32)
        self._heat_kernel_1d = np.array([1.0, 2.0, 1.0], dtype=np.float32)
        self._heat_kernel_1d = self._heat_kernel_1d / self._heat_kernel_1d.sum()

        self.reset_heatmap_on_start = False

        self.analytics = {
            'total_customers': 0,
            'average_time_in_shop': 0,
            'popular_items': defaultdict(int),
            'area_visits': defaultdict(int),
            'exit_traffic_by_minute': defaultdict(int),
            'total_revenue': 0.0,
            'completed_purchases': 0,
            'abandoned_carts': 0,
            'basket_sizes': [],
            'impulse_purchases': 0,
            'impulse_item_sales': defaultdict(int),
            'revenue_by_area': defaultdict(float),
            'dwell_times_by_zone': defaultdict(list),
            'customer_paths': [],
            'return_customers': 0,
            'customer_lifetime_values': defaultdict(float),
            'cross_merchandising': defaultdict(int),
            'item_conversion_rates': defaultdict(lambda: {'visits': 0, 'purchases': 0}),
            'foot_traffic_density': defaultdict(float),
            'bottlenecks': defaultdict(int),
            'queue_wait_times': [],
            'processing_times': [],
            'data_points_collected': 0,
            'accuracy_metrics': [],
            'optimization_history': [],
            'pre_optimization_revenue': 0.0,
            'post_optimization_revenue': 0.0,
            'optimization_impact': 0.0,
        }

        self.analytics['measurement_duration'] = 90
        self.analytics['pre_window_revenue']   = 0.0
        self.analytics['post_window_revenue']  = 0.0
        self.analytics['pre_window_start']     = None
        self.analytics['post_window_start']    = None
        self.analytics['phase']                = None
        self.analytics['customers_over_time']  = defaultdict(int)
        self.analytics['floor_visits']         = defaultdict(int)

        self.bottleneck_threshold = 50
        self.run_time = 0.0
        self.prev_state_by_cust = {}

        # --- Non-homogeneous Poisson arrival profile ----------------
        # 24-vector of hour-of-day arrival-rate multipliers (mean 1.0
        # over the operating window). Defaults to uniform; gets replaced
        # by the calibration step's empirical hour-of-day distribution
        # when a transactional dataset is loaded.
        #
        # ``sim_clock_start_hour`` controls what time of the simulated
        # day the run starts at. Default 09:00 matches typical UK retail
        # (and UCI Online Retail II open hours).
        self.hourly_profile = np.ones(24, dtype=np.float64)
        self.sim_clock_start_hour = 9.0

        # Gap to the next arrival, drawn from Exp(effective_rate). Set to
        # None so the first tick draws one. Arrivals used to fire on a
        # fixed 1/rate cadence, which reproduced the mean rate but had
        # zero inter-arrival variance -- and understated queueing, since
        # regular arrivals queue far less than Poisson ones at the same
        # utilization (audit R22).
        self._next_spawn_gap = None
        self.arrival_rng = np.random.default_rng()

        # The live loop deliberately swallows per-agent exceptions so one
        # bad agent can't kill a GUI session. That is fine interactively
        # and dangerous for measurement: a silently skipped update biases
        # any statistic collected from the run. Count them so a headless
        # diagnostic can assert the run was clean (audit R38).
        self.suppressed_errors = defaultdict(int)

        # Queue used by the simulation worker thread to hand callables
        # off to the Tk main thread (which drains via after-tick in the
        # visualizer). Never call into Tk from the worker thread directly.
        self._gui_queue = _queue.Queue(maxsize=256)

        self.simulation_start_time = time.time()
        self.last_analytics_update = time.time()

        self.simulation_thread = None
        self.last_update_time = time.time()
        self._executor = None
        self._running = False
        self.executor_type = 'thread'

        self.use_executor = False
        self.optimize_smoothing = True
        self._last_auto_tune = time.perf_counter()
        self._min_fps = 12.0
        self._max_fps = 25.0
        self._min_heat_ivl = 0.8
        self._max_heat_ivl = 2.0

        # Per-floor heatmap storage
        self._floor_heat_raw = {}

    # ------------------------------------------------------------------
    def _auto_tune_performance(self):
        now = time.perf_counter()
        if now - self._last_auto_tune < 2.0:
            return
        self._last_auto_tune = now
        times = self.analytics.get('processing_times', [])
        if len(times) < 50:
            return
        avg_proc = sum(times[-50:]) / 50.0
        frame_budget = 1.0 / max(self.target_fps, 1e-6)
        changed = False
        if avg_proc > 0.75 * frame_budget:
            new_fps = max(self._min_fps, self.target_fps - 2.0)
            if new_fps != self.target_fps:
                self.target_fps = new_fps
                self.draw_interval = 1.0 / self.target_fps
                changed = True
            self.heat_update_interval = min(self._max_heat_ivl, self.heat_update_interval + 0.2)
        elif avg_proc < 0.40 * frame_budget:
            new_fps = min(self._max_fps, self.target_fps + 1.0)
            if new_fps != self.target_fps:
                self.target_fps = new_fps
                self.draw_interval = 1.0 / self.target_fps
                changed = True
            self.heat_update_interval = max(self._min_heat_ivl, self.heat_update_interval - 0.1)
        if changed:
            del times[:-100]

    # ------------------------------------------------------------------
    # Start / Pause / Stop
    # ------------------------------------------------------------------
    def start_simulation(self, executor_type=None):
        if executor_type in ('thread', 'process'):
            self.executor_type = executor_type

        # Resume from pause without resetting anything
        if self.paused and self._running:
            self.paused = False
            self.geometry_dirty = True   # <-- critical: layout may have changed
            self.last_update_time = time.perf_counter()
            self.last_spawn_time  = self.last_update_time
            self._next_spawn_gap  = None   # redraw; the old gap predates the pause
            # Refresh reachable floors for everyone, since user may have added one.
            try:
                reach = self._reachable_floors(start_floor=1)
                for c in self.customers:
                    c._reachable_floors = set(reach)
            except Exception:
                pass
            return

        if getattr(self, 'simulation_thread', None) is not None:
            self.geometry_dirty = True
            if self.simulation_thread.is_alive():
                self.running = False
                self._running = False
                try:
                    self.simulation_thread.join(timeout=1.0)
                except Exception:
                    pass
            self.simulation_thread = None

        if self._running:
            return

        if getattr(self, 'reset_heatmap_on_start', False):
            try:
                self.reset_heatmap()
            except Exception:
                pass
            self.reset_heatmap_on_start = False

        try:
            self.shop._assign_default_prices()
        except Exception:
            pass

        if self.seed is not None:
            try:
                np.random.seed(self.seed)
            except Exception:
                pass

        self.geometry_dirty = True

        try:
            self._sync_entrance_from_shop()
        except Exception:
            if hasattr(self.shop, 'door_position'):
                self.door_position = self.shop.door_position
                self.door_side     = self.shop.door_side

        if self._executor:
            try:
                self._executor.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                self._executor.shutdown(wait=False)
            self._executor = None

        if self.use_executor:
            try:
                self._executor = ThreadPoolExecutor()
                self.executor_type = 'thread'
            except Exception:
                self._executor = None
        else:
            self._executor = None

        self.paused = False
        self.running = True
        self._running = True
        self.last_update_time = time.perf_counter()
        self.last_spawn_time  = self.last_update_time
        self._next_spawn_gap  = None   # fresh run, fresh inter-arrival draw
        self.simulation_thread = Thread(target=self._simulation_loop, daemon=True)
        self.simulation_thread.start()

    def _gui_post(self, fn):
        """Submit a 0-arg callable to be executed on the Tk main thread.
        Worker threads must not touch Tk widgets directly."""
        try:
            self._gui_queue.put_nowait(fn)
        except _queue.Full:
            # Queue is saturated (GUI is behind); drop the request rather
            # than blocking the simulation tick.
            pass

    def pause_simulation(self):
        """Freeze in place -- customers stay visible; resume via start_simulation()."""
        if not (self._running or self.running):
            return
        self.paused = True

    def stop_simulation(self):
        """Backwards-compatible alias: pauses (does NOT clear customers)."""
        self.pause_simulation()

    def hard_stop(self):
        """Fully terminate, clear all customers and patches."""
        if not (self._running or self.running):
            # Still wipe any leftover patches in case sim never started
            if hasattr(self, 'customer_patches_by_id'):
                for pid, patch in list(self.customer_patches_by_id.items()):
                    try: patch.remove()
                    except Exception: pass
                self.customer_patches_by_id.clear()
            self.customers = []
            return

        self.running = False
        self._running = False
        self.paused = False

        if self.simulation_thread is not None:
            try:
                # Join WITHOUT a timeout: the worker loop checks self.running
                # at the top of every iteration and exits within one frame,
                # so this never blocks for more than ~1/target_fps seconds.
                self.simulation_thread.join()
            except Exception:
                pass
            self.simulation_thread = None

        if self._executor:
            try: self._executor.shutdown(wait=False, cancel_futures=True)
            except TypeError: self._executor.shutdown(wait=False)
            self._executor = None

        if hasattr(self, 'customer_patches_by_id'):
            for pid, patch in list(self.customer_patches_by_id.items()):
                try: patch.remove()
                except Exception: pass
            self.customer_patches_by_id.clear()

        self.customers = []
        self.prev_state_by_cust = {}
        self.geometry_dirty = True

        try:
            if self.shop and getattr(self.shop, 'canvas', None) is not None:
                self._gui_post(self.shop.canvas.draw_idle)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Floor reachability via connectors
    # ------------------------------------------------------------------
    def _reachable_floors(self, start_floor=1):
        reachable = {start_floor}
        try:
            connectors = getattr(self.shop, 'connectors', {}) or {}
        except Exception:
            return reachable
        adj = defaultdict(set)
        for cid, mapping in connectors.items():
            floors = list(mapping.keys())
            for a in floors:
                for b in floors:
                    if a != b:
                        adj[a].add(b)
        frontier = [start_floor]
        while frontier:
            f = frontier.pop()
            for nb in adj.get(f, ()):
                if nb not in reachable:
                    reachable.add(nb)
                    frontier.append(nb)
        return reachable

    def _connector_path(self, start_floor=1, dest_floor=1):
        """Return a shortest floor path using connector graph, or [] if none."""
        if start_floor == dest_floor:
            return [start_floor]
        connectors = getattr(self.shop, 'connectors', {}) or {}
        adj = defaultdict(set)
        for mapping in connectors.values():
            floors = list(mapping.keys())
            for a in floors:
                for b in floors:
                    if a != b:
                        adj[a].add(b)
        queue = [(start_floor, [start_floor])]
        seen = {start_floor}
        for floor, path in queue:
            for nb in sorted(adj.get(floor, ())):
                if nb in seen:
                    continue
                if nb == dest_floor:
                    return path + [nb]
                seen.add(nb)
                queue.append((nb, path + [nb]))
        return []

    def _infer_entrance_from_floor1_walls(self):
        """Infer a generated entrance by finding a gap in floor-1 boundary walls."""
        floor1 = getattr(self.shop, 'floors', {}).get(1, {})
        walls = floor1.get('walls', {})
        if not walls:
            return None, None
        W, H = self.shop.width, self.shop.height
        tol = 0.35

        def merge_intervals(intervals):
            intervals = sorted((max(0.0, a), max(0.0, b)) for a, b in intervals if b > a)
            merged = []
            for a, b in intervals:
                if not merged or a > merged[-1][1] + 1e-6:
                    merged.append([a, b])
                else:
                    merged[-1][1] = max(merged[-1][1], b)
            return merged

        sides = {'bottom': [], 'top': [], 'left': [], 'right': []}
        for nm, wall in walls.items():
            if nm.startswith('Section_') or wall.get('category') == 'Connector':
                continue
            x, y = wall.get('position', (0, 0))
            w, h = wall.get('size', (0, 0))
            if h <= tol and y <= tol:
                sides['bottom'].append((x, x + w))
            if h <= tol and y + h >= H - tol:
                sides['top'].append((x, x + w))
            if w <= tol and x <= tol:
                sides['left'].append((y, y + h))
            if w <= tol and x + w >= W - tol:
                sides['right'].append((y, y + h))

        for side, intervals in sides.items():
            limit = W if side in ('bottom', 'top') else H
            cur = 0.0
            for a, b in merge_intervals(intervals):
                if a - cur >= 0.5:
                    mid = (cur + a) / 2
                    if side == 'bottom':
                        return (mid, 0), side
                    if side == 'top':
                        return (mid, H), side
                    if side == 'left':
                        return (0, mid), side
                    return (W, mid), side
                cur = max(cur, b)
            if limit - cur >= 0.5:
                mid = (cur + limit) / 2
                if side == 'bottom':
                    return (mid, 0), side
                if side == 'top':
                    return (mid, H), side
                if side == 'left':
                    return (0, mid), side
                return (W, mid), side
        return None, None

    def _sync_entrance_from_shop(self):
        """Keep simulation door state synchronized with floor-1 metadata."""
        floor1 = getattr(self.shop, 'floors', {}).get(1, {})
        door = floor1.get('door_position') or getattr(self.shop, 'door_position', None)
        side = floor1.get('door_side') or getattr(self.shop, 'door_side', None)
        if door is None:
            door, side = self._infer_entrance_from_floor1_walls()
        if door is not None:
            door = tuple(door)
            self.door_position = door
            self.door_side = side
            self.shop.door_position = door
            self.shop.door_side = side
            try:
                floor1['is_main_floor'] = True
                floor1['door_position'] = door
                floor1['door_side'] = side
            except Exception:
                pass
        return self.door_position, self.door_side

    # ------------------------------------------------------------------
    # Spawning
    # ------------------------------------------------------------------
    def _spawn_customer(self):
        # Read the cap live each call so the Simulation-tab Spinbox can
        # raise/lower it without restarting the loop. Defensive fallback
        # keeps legacy callers working if the attribute was deleted.
        try:
            cap = int(getattr(self, 'max_customers', 20))
        except Exception:
            cap = 20
        if cap < 1:
            cap = 1
        if len(self.customers) >= cap:
            return
        entrance_pos = self._find_entrance_position()
        reachable = self._reachable_floors(start_floor=1)
        try:
            all_items = self.shop.all_items_across_floors()
        except Exception:
            all_items = dict(self.shop.items)
        filtered_items = {nm: data for nm, data in all_items.items()
                          if data.get('floor', 1) in reachable}

        cust = Customer(
            self.customer_counter, entrance_pos, filtered_items,
            (self.shop.width, self.shop.height),
            door_position=self.door_position, door_side=self.door_side,
            floor=1,
        )
        cust._simulation_ref = self
        cust._reachable_floors = set(reachable)

        # Trajectory-calibrated walking speed: when a spatial dataset
        # (OpenTraj / ATC) has been loaded, agents draw their speed from
        # the EMPIRICAL distribution (normal, clipped to the observed
        # p5-p95 band) instead of the uniform(0.8, 1.5) default -- so the
        # spatial calibration genuinely shapes agent behavior, not just
        # the Validation-tab overlays.
        try:
            cal = self.analytics.get('calibration', {})
            mu = cal.get('empirical_speed_mean')
            if mu:
                sd = cal.get('empirical_speed_std') or 0.2
                lo = cal.get('empirical_speed_p5', mu - 2 * sd)
                hi = cal.get('empirical_speed_p95', mu + 2 * sd)
                s = float(np.random.normal(mu, sd))
                cust.speed = float(min(max(s, max(0.3, lo)), min(2.5, hi)))
        except Exception:
            pass

        self.customers.append(cust)
        self.customer_counter += 1
        self.analytics['total_customers'] += 1
        self.analytics.setdefault('floor_visits', defaultdict(int))[1] += 1

    def _find_entrance_position(self):
        if self.door_position is None:
            try:
                self._sync_entrance_from_shop()
            except Exception:
                pass
        if self.door_position is None:
            return [self.shop.width / 2, 0.2]
        door_x, door_y = self.door_position
        if self.door_side == 'bottom':
            return [door_x, door_y + 0.3]
        elif self.door_side == 'top':
            return [door_x, door_y - 0.3]
        elif self.door_side == 'left':
            return [door_x + 0.3, door_y]
        else:
            return [door_x - 0.3, door_y]

    # ------------------------------------------------------------------
    # Anti-stacking separation force
    # ------------------------------------------------------------------
    def _apply_separation(self, customers):
        # Overridable for structural-sensitivity sweeps (audit R6.5); the
        # defaults reproduce the shipped behavior exactly.
        min_dist = getattr(self, 'separation_min_dist', 0.35)
        strength = getattr(self, 'separation_strength', 0.08)
        n = len(customers)
        if n < 2:
            return
        for i in range(n):
            ci = customers[i]
            if ci.state == 'checking_out':
                continue
            for j in range(i + 1, n):
                cj = customers[j]
                if ci.floor != cj.floor or cj.state == 'checking_out':
                    continue
                dx = ci.position[0] - cj.position[0]
                dy = ci.position[1] - cj.position[1]
                dist = math.sqrt(dx * dx + dy * dy)
                if 1e-6 < dist < min_dist:
                    overlap = (min_dist - dist) * strength
                    nx = dx / dist; ny = dy / dist
                    w, h = ci.shop_dimensions

                    new_ix = max(0.1, min(w - 0.1, ci.position[0] + nx * overlap))
                    new_iy = max(0.1, min(h - 0.1, ci.position[1] + ny * overlap))
                    new_jx = max(0.1, min(w - 0.1, cj.position[0] - nx * overlap))
                    new_jy = max(0.1, min(h - 0.1, cj.position[1] - ny * overlap))

                    # Reject any push that would shove a customer through a wall.
                    fwalls_i = self.shop.floors.get(ci.floor, {}).get('walls', self.shop.walls)
                    fwalls_j = self.shop.floors.get(cj.floor, {}).get('walls', self.shop.walls)
                    if not ci._collides_with_interior(new_ix, new_iy, fwalls_i):
                        ci.position[0], ci.position[1] = new_ix, new_iy
                    if not cj._collides_with_interior(new_jx, new_jy, fwalls_j):
                        cj.position[0], cj.position[1] = new_jx, new_jy

    # ------------------------------------------------------------------
    def _simulation_loop(self):
        try:
            from concurrent.futures import ThreadPoolExecutor as _TPE
        except Exception:
            _TPE = None

        while self.running:
            loop_start = time.perf_counter()

            # ---- PAUSE: keep thread alive, redraw existing customers ----
            if self.paused:
                if (getattr(self.shop, 'current_tab', None) == 'Layout'
                        and self.shop.canvas):
                    self._gui_post(self._update_visualization)
                time.sleep(1.0 / self.target_fps)
                continue

            if self.geometry_dirty:
                try: self._rebuild_geometry_caches()
                except Exception: pass
                self.geometry_dirty = False

            now = loop_start
            real_dt = now - self.last_update_time
            dt = real_dt * self.simulation_speed
            self.last_update_time = now
            self.run_time += real_dt

            # Non-homogeneous Poisson spawn: scale spawn_rate by the
            # hour-of-day multiplier so peak hours produce more arrivals
            # and off-hours produce fewer. ``hourly_profile`` defaults to
            # uniform-1.0 so existing layouts behave unchanged until a
            # transactional dataset's empirical hour-of-day shape is
            # seeded into it.
            #
            # Inter-arrival gaps are drawn from Exp(effective_rate) rather
            # than fired on a fixed cadence. Because the hour-of-day
            # profile is piecewise constant, exponential gaps redrawn at
            # each arrival are an exact NHPP realization within the hour;
            # the only approximation is a gap that straddles an hour
            # boundary, which keeps the rate it was drawn under.
            sim_clock_h = (self.sim_clock_start_hour
                           + self.run_time / 3600.0 * self.simulation_speed)
            hod_mult = float(self.hourly_profile[int(sim_clock_h) % 24])
            effective_rate = self.spawn_rate * hod_mult
            if effective_rate > 0:
                if self._next_spawn_gap is None:
                    self._next_spawn_gap = float(
                        self.arrival_rng.exponential(1.0 / effective_rate))
                if (now - self.last_spawn_time) >= self._next_spawn_gap:
                    self._spawn_customer()
                    self.last_spawn_time = now
                    self._next_spawn_gap = float(
                        self.arrival_rng.exponential(1.0 / effective_rate))

            customers = list(self.customers)
            try:
                all_items_for_update = self.shop.all_items_across_floors()
            except Exception:
                all_items_for_update = dict(self.shop.items)
            used_executor = False

            if self.use_executor and getattr(self, '_executor', None) is None and _TPE is not None:
                try:
                    self._executor = _TPE()
                    self.executor_type = 'thread'
                except Exception:
                    self._executor = None

            if self.use_executor and self._executor is not None:
                futures = []
                try:
                    for cust in customers:
                        fwalls = self.shop.floors.get(cust.floor, {}).get('walls', self.shop.walls)
                        futures.append(self._executor.submit(
                            cust.update, dt, all_items_for_update,
                            fwalls, customers))
                    used_executor = True
                except Exception:
                    used_executor = False
                    self._executor = None
                if used_executor:
                    for f in futures:
                        try: f.result()
                        except Exception: pass

            if not used_executor:
                for cust in customers:
                    try:
                        fwalls = self.shop.floors.get(cust.floor, {}).get('walls', self.shop.walls)
                        cust.update(dt, all_items_for_update,
                                    fwalls, customers)
                    except Exception:
                        self.suppressed_errors['agent_update'] += 1

            # Anti-stack separation
            try: self._apply_separation(customers)
            except Exception: self.suppressed_errors['separation'] += 1

            for cust in customers:
                try: self._update_analytics_realtime(cust, dt)
                except Exception: self.suppressed_errors['analytics'] += 1

            if now - self.last_heat_update >= self.heat_update_interval:
                try: self._update_floor_heatmaps()
                except Exception: self.suppressed_errors['heatmap'] += 1
                self.last_heat_update = now

            to_remove = []
            for cust in self.customers:
                if cust.state == 'exiting' and self._customer_reached_door(cust):
                    try: self._process_customer_exit(cust)
                    except Exception: self.suppressed_errors['customer_exit'] += 1
                    to_remove.append(cust)
            for cust in to_remove:
                try: self.customers.remove(cust)
                except ValueError: pass

            minute = int(self.run_time // 60)
            prev_peak = self.analytics['customers_over_time'][minute]
            self.analytics['customers_over_time'][minute] = max(prev_peak, len(self.customers))

            if (getattr(self.shop, 'current_tab', None) == 'Layout'
                    and self.shop.canvas
                    and (now - self.last_draw_time) >= self.draw_interval):
                self._gui_post(self._update_visualization)
                self.last_draw_time = now

            elapsed = time.perf_counter() - loop_start
            sleep_for = max(0.0, 1.0 / self.target_fps - elapsed)
            time.sleep(sleep_for)

            total_loop = time.perf_counter() - loop_start
            self.analytics['processing_times'].append(elapsed)
            ideal_ivl = 1.0 / self.target_fps
            err_pct = abs(total_loop - ideal_ivl) / ideal_ivl * 100.0
            self.analytics['accuracy_metrics'].append(err_pct)
            self.analytics['data_points_collected'] += 1

            self._auto_tune_performance()

    # ------------------------------------------------------------------
    # Per-floor heatmap accumulation
    # ------------------------------------------------------------------
    def _get_floor_heat_raw(self, floor_id):
        if floor_id not in self._floor_heat_raw:
            res = self.heat_map_resolution
            wcells = int(self.shop.width  * res) + 1
            hcells = int(self.shop.height * res) + 1
            self._floor_heat_raw[floor_id] = np.zeros((wcells, hcells), dtype=np.float32)
        return self._floor_heat_raw[floor_id]

    def _update_floor_heatmaps(self):
        res = self.heat_map_resolution
        wcells = int(self.shop.width  * res) + 1
        hcells = int(self.shop.height * res) + 1

        for cust in self.customers:
            raw = self._get_floor_heat_raw(cust.floor)
            if raw.shape[0] < wcells or raw.shape[1] < hcells:
                new_raw = np.zeros((wcells, hcells), dtype=np.float32)
                ow, oh = raw.shape
                new_raw[:ow, :oh] = raw
                raw = new_raw
                self._floor_heat_raw[cust.floor] = raw
            xi = int(cust.position[0] * res)
            yi = int(cust.position[1] * res)
            if 0 <= xi < raw.shape[0] and 0 <= yi < raw.shape[1]:
                raw[xi, yi] += 1.0

        cur_floor = getattr(self.shop, 'current_floor', 1)
        cur_raw = self._get_floor_heat_raw(cur_floor)
        if cur_raw.shape[0] < wcells or cur_raw.shape[1] < hcells:
            new_raw = np.zeros((wcells, hcells), dtype=np.float32)
            ow, oh = cur_raw.shape
            new_raw[:ow, :oh] = cur_raw
            cur_raw = new_raw
            self._floor_heat_raw[cur_floor] = cur_raw

        if self.heat_raw.shape[0] < wcells or self.heat_raw.shape[1] < hcells:
            self.heat_raw = np.zeros((wcells, hcells), dtype=np.float32)

        self.heat_raw[:cur_raw.shape[0], :cur_raw.shape[1]] = cur_raw

        try: self._refresh_smoothed_heatmap()
        except Exception: pass

    def switch_floor_heatmap(self, new_floor):
        """Sync heat_raw to the requested floor's accumulated data."""
        cur_raw = self._get_floor_heat_raw(new_floor)
        res = self.heat_map_resolution
        wcells = int(self.shop.width  * res) + 1
        hcells = int(self.shop.height * res) + 1
        if self.heat_raw.shape[0] < wcells or self.heat_raw.shape[1] < hcells:
            self.heat_raw = np.zeros((wcells, hcells), dtype=np.float32)
        self.heat_raw[:] = 0.0
        w = min(cur_raw.shape[0], self.heat_raw.shape[0])
        h = min(cur_raw.shape[1], self.heat_raw.shape[1])
        self.heat_raw[:w, :h] = cur_raw[:w, :h]
        try: self._refresh_smoothed_heatmap()
        except Exception: pass

    # ------------------------------------------------------------------
    def _customer_reached_door(self, cust):
        if getattr(cust, 'current_target_type', None) != 'exit':
            return False
        if self.door_position is None:
            return False
        # Door is always on floor 1 -- customer must be on floor 1 to exit
        if getattr(cust, 'floor', 1) != 1:
            return False
        dx = cust.position[0] - self.door_position[0]
        dy = cust.position[1] - self.door_position[1]
        return (dx * dx + dy * dy) < 0.09

    def set_seed(self, seed):
        self.seed = int(seed)

    # ------------------------------------------------------------------
    def _update_visualization(self):
        if not hasattr(self, 'customer_patches_by_id'):
            self.customer_patches_by_id = {}

        ax = getattr(self.shop, 'ax', None)
        canvas = getattr(self.shop, 'canvas', None)
        if ax is None or canvas is None:
            return

        cur_floor = getattr(self.shop, 'current_floor', 1)

        existing_ids = set(self.customer_patches_by_id.keys())
        current_ids  = {c.id for c in self.customers}
        for dead_id in (existing_ids - current_ids):
            patch = self.customer_patches_by_id.pop(dead_id, None)
            if patch is not None:
                try: patch.remove()
                except Exception: pass

        for c in self.customers:
            if c.floor != cur_floor:
                p = self.customer_patches_by_id.get(c.id)
                if p is not None:
                    try: p.set_visible(False)
                    except Exception: pass
                continue

            p = self.customer_patches_by_id.get(c.id)
            if p is None or p.axes is None:
                try:
                    p = plt.Circle(c.position, c.size, color=c.get_color(),
                                   alpha=0.8, zorder=10)
                    ax.add_patch(p)
                    self.customer_patches_by_id[c.id] = p
                except Exception:
                    continue
            else:
                try:
                    p.set_visible(True)
                    p.center = tuple(c.position)
                    new_color = c.get_color()
                    if p.get_facecolor() != plt.matplotlib.colors.to_rgba(
                            new_color, alpha=p.get_alpha()):
                        p.set_color(new_color)
                except Exception:
                    pass

        try: canvas.draw_idle()
        except Exception: pass