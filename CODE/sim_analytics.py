import math
from collections import defaultdict
import numpy as np

from sim_calibration import new_markov_transition_counts, record_transition


# Movement paths are only ever exported, so sample them coarsely and keep a
# bounded history: per-tick samples for every exited customer grow without
# limit over a long GUI session.
PATH_SAMPLE_INTERVAL_S = 0.5
MAX_STORED_PATHS = 500

# A cell is congested while at least this many agents stand in it at the
# same time. Congestion is a property of how many agents share the cell,
# not of how much traffic it has seen since the run started.
BOTTLENECK_MIN_AGENTS = 3


class AnalyticsMixin:
    """Analytics / reporting methods for CustomerFlowSimulation."""

    def _update_heat_map(self, cust):
        res = self.heat_map_resolution
        xi = int(cust.position[0] * res)
        yi = int(cust.position[1] * res)
        if 0 <= xi < self.heat_raw.shape[0] and 0 <= yi < self.heat_raw.shape[1]:
            self.heat_raw[xi, yi] += 1.0

    # ------------------------------------------------------------------
    # Multi-floor lookups
    # ------------------------------------------------------------------
    def _split_floor_item_key(self, item_name):
        text = str(item_name)
        if text.startswith('F') and ':' in text:
            prefix, source = text.split(':', 1)
            if prefix[1:].isdigit():
                return int(prefix[1:]), source
        return None, item_name

    def _resolve_item_floor(self, item_name):
        """Return (floor id, per-floor name) for a flattened item key.

        Mirrors all_items_across_floors(): a plain name belongs to the
        lowest floor holding it and later duplicates are 'F<fid>:name'.
        shop.items / shop.prices follow the floor on screen, so they cannot
        tell which floor a purchased duplicate came from.
        """
        floors = getattr(self.shop, 'floors', None) or {}
        for fid in sorted(floors):
            if item_name in floors[fid].get('items', {}):
                return fid, item_name
        key_floor, source_name = self._split_floor_item_key(item_name)
        if key_floor in floors and source_name in floors[key_floor].get('items', {}):
            return key_floor, source_name
        if key_floor is not None:
            # Collision-suffixed keys ('F2:Milk_2') only round-trip through
            # the flattening itself.
            d = self.shop.all_items_across_floors().get(item_name)
            if d is not None and d.get('floor') in floors:
                return d['floor'], d.get('source_name', source_name)
        return None, source_name

    def _get_item_price(self, item_name):
        try:
            floors = getattr(self.shop, 'floors', None) or {}
            key_floor, source_name = self._resolve_item_floor(item_name)
            if key_floor is not None:
                p = floors[key_floor].get('prices', {}).get(source_name)
                if p is not None:
                    return p
            for fid in sorted(floors):
                p = floors[fid].get('prices', {}).get(source_name)
                if p is not None:
                    return p
        except Exception as _e:
            print(f'[sim_analytics] _get_item_price({item_name!r}) failed: {_e}')
        price = self.shop.prices.get(item_name)
        if price is not None:
            return price
        return 0.0

    def _get_item_category(self, item_name):
        try:
            floors = getattr(self.shop, 'floors', None) or {}
            key_floor, source_name = self._resolve_item_floor(item_name)
            if key_floor is not None:
                c2 = floors[key_floor].get('items', {}).get(source_name, {}).get('category')
                if c2 is not None:
                    return c2
            for fid in sorted(floors):
                c2 = floors[fid].get('items', {}).get(source_name, {}).get('category')
                if c2 is not None:
                    return c2
        except Exception as _e:
            print(f'[sim_analytics] _get_item_category({item_name!r}) failed: {_e}')
        cat = self.shop.items.get(item_name, {}).get('category')
        if cat is not None:
            return cat
        return 'Unknown'

    # ------------------------------------------------------------------
    def _process_customer_exit(self, cust):
        # Simulated seconds: the clock agents stamp their spawn and state
        # times with, and the one the optimize windows are opened on.
        now = float(getattr(self, 'sim_time', 0.0))
        A = self.analytics

        minute = int(now // 60)
        A['exit_traffic_by_minute'][minute] += 1

        if getattr(cust, 'loyalty_level', 'new') != 'new':
            A['return_customers'] += 1

        # queue_wait_times is filled when service starts
        # (Customer._start_service), where the wait is known.

        # Close the agent's state sequence with its absorbing outcome.
        outcome = 'purchased' if cust.has_checked_out else 'abandoned'
        states = list(getattr(cust, 'state_history', None) or [cust.state])
        counts = self._log_state_transitions(cust)
        record_transition(counts, states[-1], outcome)
        if getattr(self, 'record_state_histories', False):
            if getattr(self, 'completed_state_histories', None) is None:
                self.completed_state_histories = []
            self.completed_state_histories.append({
                'spawn_t': float(getattr(cust, 'spawn_sim_time', 0.0)),
                'exit_t': float(getattr(self, 'sim_time', 0.0)),
                'states': states + [outcome],
            })

        # Running mean over exits only: total_customers also counts agents
        # still in the store, which would dilute every early exit.
        duration = now - cust.start_shop_time
        n_exited = A.get('n_exited', 0) + 1
        A['n_exited'] = n_exited
        prev_avg = A.get('average_time_in_shop', 0.0)
        A['average_time_in_shop'] = prev_avg + (duration - prev_avg) / n_exited

        # How each exit came about (a tally, read by no decision): unpaid
        # exits by the route that sent them out -- the spawn-time
        # abandonment draw, no route left to walk, or anything else -- and
        # whether the agent had drawn abandonment at all, plus what the
        # stall detector did to the exited agents. abandoned_carts below
        # counts every unpaid exit alike; this is what splits it.
        ex = A.setdefault('exit_routes', defaultdict(int))
        if cust.has_checked_out:
            ex['paid'] += 1
        else:
            ex['unpaid_' + (getattr(cust, 'exit_route', None) or 'other')] += 1
            if getattr(cust, 'abandon_cart', False):
                ex['unpaid_abandon_drawn'] += 1
        if getattr(cust, 'abandon_cart', False):
            ex['abandon_drawn'] += 1
        stalls = int(getattr(cust, 'moving_stalls', 0) or 0)
        ex['moving_stalls'] += stalls
        ex['exits_after_moving_stall'] += int(stalls > 0)
        ex['exit_stall_teleports'] += int(
            getattr(cust, 'exit_stall_teleports', 0) or 0)

        if cust.has_checked_out:
            A['completed_purchases'] += 1
            # visited_items is a set of names, and string-set iteration order
            # changes with each process's hash seed. A fixed order keeps the
            # revenue sums and counter key order bit-identical when a seeded
            # run is repeated in another process.
            visited = sorted(cust.visited_items, key=str)
            items_bought = visited + cust.impulse_items
            rev = 0.0
            for it in items_bought:
                rev += self._get_item_price(it)
            A['total_revenue'] += rev

            # Measurement windows are timed on the simulated clock at both
            # ends: the exit is credited from the same clock the optimize
            # pipeline opens and closes the window on, so the credited span
            # is the window itself at any simulation speed. The phase is
            # also required, so exits after the pipeline finished cannot
            # keep growing a window's revenue.
            phase = A.get('phase')
            dur   = A.get('measurement_duration', 0)
            if phase in ('pre', 'post'):
                start = A.get(phase + '_window_start', 0) or 0
                if now - start <= dur:
                    A[phase + '_window_revenue'] = A.get(
                        phase + '_window_revenue', 0.0) + rev

            cnt = len(visited) + len(cust.impulse_items)
            A['basket_sizes'].append(cnt)
            # Per-visit spend, the companion sample of basket_sizes: the
            # validation revenue test and the projections' revenue spread
            # both need the per-customer values, not just the running total.
            A.setdefault('customer_revenues', []).append(rev)
            # The items themselves, one list per visit in the same order as
            # basket_sizes and customer_revenues (so a window is cut from all
            # three by one index). The category goodness-of-fit test needs a
            # visit's purchases kept together: items bought on one trip are
            # not independent draws, and the per-item counters below cannot
            # be split back into the visits they came from. Keys are stored
            # as strings so the record stays JSON-serialisable.
            A.setdefault('visit_purchases', []).append(
                [str(it) for it in items_bought])

            imp_cnt = len(cust.impulse_items)
            A['impulse_purchases'] += imp_cnt
            for imp in cust.impulse_items:
                A['impulse_item_sales'][imp] += 1

            if cnt > 0:
                # Credit each category what its own items cost; an even split
                # would let cheap basket-mates claim an expensive item's revenue.
                for item_name in items_bought:
                    cat = self._get_item_category(item_name)
                    A['revenue_by_area'][cat] += self._get_item_price(item_name)

            for itm in visited:
                A['item_conversion_rates'][itm]['purchases'] += 1
                A['popular_items'][itm] += 1

            iv = visited
            for i in range(len(iv)):
                for j in range(i + 1, len(iv)):
                    key = "|".join(sorted((iv[i], iv[j])))
                    A['cross_merchandising'][key] += 1

            clv_mult = {'new': 1.0, 'regular': 2.5, 'vip': 4.0}.get(
                cust.loyalty_level, 1.0)
            A['customer_lifetime_values'][cust.loyalty_level] += rev * clv_mult
            A.setdefault('clv_counts', defaultdict(int))[cust.loyalty_level] += 1
        else:
            A['abandoned_carts'] += 1

        for zone, dwell_t in cust.zone_dwell_times.items():
            A['dwell_times_by_zone'][zone].append(dwell_t)

        if len(cust.movement_path) > 5:
            A['customer_paths'].append(cust.movement_path.copy())
            del A['customer_paths'][:-MAX_STORED_PATHS]

        cust.visited_zones.clear()
        cust.zone_dwell_times.clear()
        cust.movement_path.clear()
        self.prev_state_by_cust.pop(cust.id, None)

    # ------------------------------------------------------------------
    def _update_analytics_realtime(self, cust, dt):
        # dt and the path-sample stamps are simulated seconds.
        now = float(getattr(self, 'sim_time', 0.0))
        path = cust.movement_path
        if not path or now - path[-1][2] >= PATH_SAMPLE_INTERVAL_S:
            path.append((cust.position[0], cust.position[1], now))

        zone = self._get_customer_zone(cust)
        if zone:
            if zone not in cust.visited_zones:
                cust.visited_zones.add(zone)
                self.analytics['area_visits'][zone] += 1
            if cust.state == 'shopping':
                cust.zone_dwell_times[zone] += dt

        prev_state = self.prev_state_by_cust.get(cust.id)
        if cust.state == 'shopping' and prev_state != 'shopping':
            itm   = getattr(cust, 'current_target_item', None)
            ttype = getattr(cust, 'current_target_type', None)
            if itm and ttype in ('item', 'impulse'):
                self.analytics['item_conversion_rates'][itm]['visits'] += 1

        # Traffic and congestion accumulate in agent-seconds. A tick is a
        # different amount of simulated time at every speed setting, so
        # per-tick counters would make the same walk register different
        # traffic, and the GA's bottleneck penalty would follow the speed
        # slider rather than the layout.
        cell = self._traffic_cell(cust.position)
        self.analytics['foot_traffic_density'][cell] += dt
        occupancy = self._cell_occupancy().get(
            (getattr(cust, 'floor', 1), cell), 0)
        if occupancy >= BOTTLENECK_MIN_AGENTS:
            self.analytics['bottlenecks'][cell] += dt

        # dt is simulated seconds, so occupancy is in sim time.
        self._log_state_transitions(cust)
        occ = self.analytics.get('markov_state_occupancy')
        if occ is None:
            occ = self.analytics['markov_state_occupancy'] = defaultdict(float)
        occ[cust.state] += dt

        self.prev_state_by_cust[cust.id] = cust.state

    @staticmethod
    def _traffic_cell(position):
        return "{},{}".format(int(position[0]), int(position[1]))

    def _cell_occupancy(self):
        """Agents per (floor, 1 m traffic cell) for the tick being processed.

        Every agent has already moved by the time the analytics pass runs,
        so the counts are built once per tick and shared by the agents of
        that tick. Agents at the same spot on different floors do not crowd
        each other, so the floor is part of the key.
        """
        now = float(getattr(self, 'sim_time', 0.0))
        counts = getattr(self, '_cell_occupancy_cache', None)
        if counts is None or getattr(self, '_cell_occupancy_time', None) != now:
            counts = defaultdict(int)
            for c in list(self.customers):
                counts[(getattr(c, 'floor', 1),
                        self._traffic_cell(c.position))] += 1
            self._cell_occupancy_cache = counts
            self._cell_occupancy_time = now
        return counts

    def _log_state_transitions(self, cust):
        """Feed the agent's not-yet-counted state jumps to the empirical
        Markov estimator and return the transition-count table.

        Reads state_history rather than comparing cust.state between
        ticks, so several jumps within one tick are all counted.
        """
        A = self.analytics
        counts = A.get('markov_transition_counts')
        if counts is None:
            counts = A['markov_transition_counts'] = new_markov_transition_counts()
        hist = getattr(cust, 'state_history', None) or ()
        start = max(getattr(cust, '_markov_logged', 1), 1)
        for i in range(start, len(hist)):
            record_transition(counts, hist[i - 1], hist[i])
        cust._markov_logged = max(len(hist), 1)
        return counts

    # ------------------------------------------------------------------
    # Floor-aware zone detection
    # ------------------------------------------------------------------
    def _get_customer_zone(self, cust):
        x, y = cust.position
        floor = getattr(cust, 'floor', 1)
        cached_floor = getattr(self, '_zones_cache_floor', None)
        # _rebuild_geometry_caches assigns a fresh wall_bins_by_floor dict, so
        # a different object means the walls changed (Generate, import, ...)
        # and the section rectangles must be re-read. Holding the reference
        # keeps the identity check reliable.
        geom = getattr(self, 'wall_bins_by_floor', None)
        if (getattr(self, '_zones_cache', None) is None or cached_floor != floor
                or getattr(self, '_zones_cache_geom', None) is not geom):
            self._zones_cache = self._build_zones_cache(floor)
            self._zones_cache_floor = floor
            self._zones_cache_geom = geom
        for name, wx, wy, ww, wh in self._zones_cache:
            if wx <= x <= wx + ww and wy <= y <= wy + wh:
                return name
        return 'General Area'

    def _build_zones_cache(self, floor_id=1):
        cache = []
        try:
            floor_walls = (self.shop.floors.get(floor_id, {}).get('walls')
                           or self.shop.walls)
            for wall_name, wall_data in floor_walls.items():
                wx, wy = wall_data['position']
                ww, wh = wall_data['size']
                if wall_name.startswith('Section_'):
                    cache.append((wall_name[8:], wx, wy, ww, wh))
            for special in ('Checkout', 'WC'):
                if special in floor_walls:
                    wx, wy = floor_walls[special]['position']
                    ww, wh = floor_walls[special]['size']
                    cache.append((special, wx, wy, ww, wh))
        except Exception as _e:
            print(f'[sim_analytics] zone cache build failed: {_e}')
        return cache

    def invalidate_zones_cache(self):
        self._zones_cache = None
        self._zones_cache_floor = None

    # ------------------------------------------------------------------
    def get_analytics_summary(self):
        # Simulated minutes: a headless run advances sim_time but no wall
        # clock, and at any other speed wall time misstates the run length.
        sim_minutes = self.sim_time / 60.0
        # The worker thread inserts keys into these dicts while this runs on
        # the Tk thread, so the Python-level loops below iterate list()
        # snapshots; a live view can raise mid-iteration.
        A = self.analytics
        total_cust = max(1, A.get('total_customers', 0))

        lines = []
        lines.append("🏪 COMPREHENSIVE SHOP ANALYTICS")
        lines.append("=" * 50)
        lines.append("")

        lines.append("💰 SALES PERFORMANCE")
        lines.append("-" * 25)
        # Rates are per customer who has LEFT the shop. Agents still inside
        # have not decided yet, and counting them as non-buyers reads as a
        # far lower conversion than the projections show, which take the
        # same exit-based rate.
        baskets       = A.get('basket_sizes', [])
        avg_basket    = sum(baskets) / len(baskets) if baskets else 0.0
        shop_area     = self.shop.width * self.shop.height
        sales_per_sqm = A['total_revenue'] / shop_area if shop_area else 0.0
        completed     = A.get('completed_purchases', 0)
        abandon       = A.get('abandoned_carts', 0)
        exited        = completed + abandon

        lines.append("Total Revenue:         ${:.2f}".format(A['total_revenue']))
        if exited:
            lines.append("Conversion Rate:       {:.1f}% of {} exited".format(
                completed / exited * 100, exited))
        else:
            lines.append("Conversion Rate:       n/a (no customer has left yet)")
        lines.append("Avg Basket Size:       {:.1f} items".format(avg_basket))
        if exited:
            lines.append("Revenue/Exited Cust.:  ${:.2f}".format(
                A['total_revenue'] / exited))
        else:
            lines.append("Revenue/Exited Cust.:  n/a")
        lines.append("Sales per m²:          ${:.2f}".format(sales_per_sqm))
        if exited:
            lines.append("Cart Abandonment:      {} ({:.1f}% of exited)".format(
                abandon, abandon / exited * 100))
        else:
            lines.append("Cart Abandonment:      {} (n/a)".format(abandon))
        # The Optimize pipeline's live windows, shown as observed. No lift is
        # taken between them: each is one short run from an empty store;
        # the projected lift is in the optimization report.
        lines.append("PRE-window revenue (live):  ${:.2f}".format(A.get('pre_optimization_revenue', 0.0)))
        lines.append("POST-window revenue (live): ${:.2f}".format(A.get('post_optimization_revenue', 0.0)))
        lines.append("")

        floor_visits = A.get('floor_visits', {})
        if floor_visits:
            lines.append("🏢 FLOOR TRAFFIC")
            lines.append("-" * 25)
            for fid, cnt in sorted(floor_visits.items()):
                lines.append("  Floor {}: {} customer visits".format(fid, cnt))
            lines.append("")

        lines.append("🛒 IMPULSE PURCHASE PERFORMANCE")
        lines.append("-" * 35)
        total_impulse = A.get('impulse_purchases', 0)
        completed     = A.get('completed_purchases', 0)
        imp_conv      = total_impulse / completed * 100 if completed else 0.0
        avg_imp_per   = total_impulse / completed if completed else 0.0
        lines.append("Total Impulse Items Sold:    {}".format(total_impulse))
        lines.append("Impulse Conversion Rate:     {:.1f}%".format(imp_conv))
        lines.append("Avg Impulse Items/Customer:  {:.1f}".format(avg_imp_per))
        impulse_sales = A.get('impulse_item_sales', {})
        if impulse_sales:
            top_impulse = sorted(impulse_sales.items(), key=lambda x: x[1], reverse=True)[:3]
            lines.append("Top Impulse Items:")
            for item, cnt in top_impulse:
                lines.append("  {}: {} sold".format(item, cnt))
        lines.append("")

        lines.append("👥 CUSTOMER BEHAVIOR")
        lines.append("-" * 25)
        lines.append("Total Customers:       {} arrived".format(A['total_customers']))
        lines.append("Active Customers:      {}".format(len(self.customers)))
        lines.append("Avg Time in Shop:      {:.1f}s".format(A.get('average_time_in_shop', 0.0)))
        # Return customers are tallied as they leave, so they are a share of
        # the exited customers, like the rates above. WC use is tallied when
        # an agent walks in, which can happen before it leaves, so that one
        # stays per arrival.
        returning = A.get('return_customers', 0)
        if exited:
            lines.append("Return Customers:      {} ({:.1f}% of exited)".format(
                returning, returning / exited * 100))
        else:
            lines.append("Return Customers:      {} (n/a)".format(returning))

        wc_visits = A['area_visits'].get('WC', 0)
        lines.append("WC Usage Rate:         {:.1f}% of arrivals ({} visits)".format(
            wc_visits / total_cust * 100, wc_visits))

        # CLV only accrues at checkout, so average over each segment's own
        # purchasers rather than an assumed equal share of all arrivals.
        clv_counts = A.get('clv_counts', {})
        clv_list = []
        for segment, total_clv in list(A.get('customer_lifetime_values', {}).items()):
            seg_cnt = max(1, clv_counts.get(segment, 0))
            avg_clv = total_clv / seg_cnt
            clv_list.append((segment.capitalize(), avg_clv))
        clv_list.sort(key=lambda x: x[1], reverse=True)
        for seg_name, avg_clv in clv_list:
            lines.append("  CLV/Purchaser ({}): ${:.2f}".format(seg_name, avg_clv))
        lines.append("Exit Traffic by Minute:")
        for mn, cnt in sorted(A.get('exit_traffic_by_minute', {}).items()):
            lines.append("  {:>2} min → {} exits".format(mn, cnt))
        lines.append("")

        lines.append("🎯 PRODUCT PERFORMANCE")
        lines.append("-" * 25)
        item_convs = {}
        for itm, data in list(A.get('item_conversion_rates', {}).items()):
            visits = data.get('visits', 0)
            buys   = data.get('purchases', 0)
            if visits > 0:
                item_convs[itm] = buys / visits * 100
        top_items = sorted(item_convs.items(), key=lambda x: x[1], reverse=True)[:3]
        lines.append("Top Item Conversions:")
        for itm, rate in top_items:
            lines.append("  {}: {:.1f}%".format(itm, rate))

        raw_cm = A.get('cross_merchandising', {})
        if raw_cm:
            parsed = []
            for key, cnt in list(raw_cm.items()):
                parts = (str(key).split('|') if not isinstance(key, (list, tuple))
                         else [str(x) for x in key])
                pair_str = ("{} + {}".format(parts[0], parts[1])
                            if len(parts) >= 2 else str(key))
                parsed.append((pair_str, cnt))
            parsed.sort(key=lambda x: x[1], reverse=True)
            if parsed:
                lines.append("Top Product Combos:")
                for pair_str, cnt in parsed[:2]:
                    lines.append("  {}: {}x".format(pair_str, cnt))
        lines.append("")

        lines.append("🚶 OPERATIONAL METRICS")
        lines.append("-" * 25)
        custs_over_time = A.get('customers_over_time', {})
        if custs_over_time:
            peak_minute, peak_count = max(list(custs_over_time.items()), key=lambda kv: kv[1])
            lines.append("Peak Time: {} min ({} customers)".format(peak_minute, peak_count))
        if A['dwell_times_by_zone']:
            avg_dwell = {z: sum(ts) / len(ts)
                         for z, ts in list(A['dwell_times_by_zone'].items()) if ts}
            lines.append("Avg Section Dwell (s):")
            for zone, val in sorted(avg_dwell.items(), key=lambda kv: kv[1], reverse=True):
                lines.append("  {}: {:.1f}s".format(zone, val))

        checkout_times = A.get('queue_wait_times', [])
        if checkout_times:
            avg_chk = sum(checkout_times) / len(checkout_times)
            lines.append("Avg Queue Wait:        {:.1f}s over {} customers".format(
                avg_chk, len(checkout_times)))

        bn = A.get('bottlenecks', {})
        if bn:
            lines.append("Top Bottleneck Cells (>= {} agents at once):".format(
                BOTTLENECK_MIN_AGENTS))
            for cell, secs in sorted(bn.items(), key=lambda x: x[1], reverse=True)[:3]:
                lines.append("  {}: {:.1f} agent-seconds".format(cell, secs))
        lines.append("")

        lines.append("💻 SYSTEM PERFORMANCE")
        lines.append("-" * 25)
        pt = A.get('processing_times', [])
        if pt:
            avg_proc = sum(pt) / len(pt)
            lines.append("Avg Processing Lat.:   {:.2f} ms".format(avg_proc * 1000))
            target_ivl = 1.0 / self.target_fps
            acc = min(target_ivl / avg_proc, 1.0) * 100
            lines.append("Tracking Accuracy:     {:.1f}%".format(acc))
        lines.append("Data Points Collected: {}".format(A.get('data_points_collected', 0)))
        lines.append("")

        lines.append("⚙️ SIMULATION STATUS")
        lines.append("-" * 25)
        lines.append("Runtime:               {:.2f} minutes".format(sim_minutes))
        lines.append("Speed:                 {:.1f}×".format(self.simulation_speed))
        lines.append("Spawn Rate:            {:.2f}/sec".format(self.spawn_rate))
        paused = getattr(self, 'paused', False)
        if self.running and not paused:
            state = "🟢 Running"
        elif self.running and paused:
            state = "⏸ Paused"
        else:
            state = "🔴 Stopped"
        lines.append("Status:                {}".format(state))

        return "\n".join(lines)