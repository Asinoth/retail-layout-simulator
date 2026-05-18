import time
import math
from collections import defaultdict
import numpy as np


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

    def _get_item_price(self, item_name):
        price = self.shop.prices.get(item_name)
        if price is not None:
            return price
        key_floor, source_name = self._split_floor_item_key(item_name)
        try:
            if key_floor in self.shop.floors:
                p = self.shop.floors[key_floor].get('prices', {}).get(source_name)
                if p is not None:
                    return p
            for fid, fdata in self.shop.floors.items():
                p = fdata.get('prices', {}).get(source_name)
                if p is not None:
                    return p
        except Exception as _e:
            print(f'[sim_analytics] _get_item_price({item_name!r}) failed: {_e}')
        return 0.0

    def _get_item_category(self, item_name):
        cat = self.shop.items.get(item_name, {}).get('category')
        if cat is not None:
            return cat
        key_floor, source_name = self._split_floor_item_key(item_name)
        try:
            if key_floor in self.shop.floors:
                c2 = self.shop.floors[key_floor].get('items', {}).get(source_name, {}).get('category')
                if c2 is not None:
                    return c2
            for fid, fdata in self.shop.floors.items():
                c2 = fdata.get('items', {}).get(source_name, {}).get('category')
                if c2 is not None:
                    return c2
        except Exception as _e:
            print(f'[sim_analytics] _get_item_category({item_name!r}) failed: {_e}')
        return 'Unknown'

    # ------------------------------------------------------------------
    def _process_customer_exit(self, cust):
        now = time.time()
        A = self.analytics

        minute = int(self.run_time // 60)
        A['exit_traffic_by_minute'][minute] += 1

        if getattr(cust, 'loyalty_level', 'new') != 'new':
            A['return_customers'] += 1

        if hasattr(cust, 'checkout_start_time') and hasattr(cust, 'checkout_end_time'):
            wait = max(0.0, cust.checkout_end_time - cust.checkout_start_time)
            A['queue_wait_times'].append(wait)

        duration = now - cust.start_shop_time
        total_cust = max(1, A['total_customers'])
        prev_avg = A.get('average_time_in_shop', 0.0)
        A['average_time_in_shop'] = (prev_avg * (total_cust - 1) + duration) / total_cust

        if cust.has_checked_out:
            A['completed_purchases'] += 1
            items_bought = list(cust.visited_items) + cust.impulse_items
            rev = 0.0
            for it in items_bought:
                rev += self._get_item_price(it)
            A['total_revenue'] += rev

            phase = A.get('phase')
            dur   = A.get('measurement_duration', 0)
            if phase == 'pre':
                start = A.get('pre_window_start', 0) or 0
                if now - start <= dur:
                    A['pre_window_revenue'] += rev
            elif phase == 'post':
                start = A.get('post_window_start', 0) or 0
                if now - start <= dur:
                    A['post_window_revenue'] += rev

            cnt = len(cust.visited_items) + len(cust.impulse_items)
            A['basket_sizes'].append(cnt)

            imp_cnt = len(cust.impulse_items)
            A['impulse_purchases'] += imp_cnt
            for imp in cust.impulse_items:
                A['impulse_item_sales'][imp] += 1

            if cnt > 0:
                rev_each = rev / cnt
                combined = list(cust.visited_items) + cust.impulse_items
                for item_name in combined:
                    cat = self._get_item_category(item_name)
                    A['revenue_by_area'][cat] += rev_each

            for itm in cust.visited_items:
                A['item_conversion_rates'][itm]['purchases'] += 1
                A['popular_items'][itm] += 1

            iv = list(cust.visited_items)
            for i in range(len(iv)):
                for j in range(i + 1, len(iv)):
                    key = "|".join(sorted((iv[i], iv[j])))
                    A['cross_merchandising'][key] += 1

            clv_mult = {'new': 1.0, 'regular': 2.5, 'vip': 4.0}.get(
                cust.loyalty_level, 1.0)
            A['customer_lifetime_values'][cust.loyalty_level] += rev * clv_mult
        else:
            A['abandoned_carts'] += 1

        for zone, dwell_t in cust.zone_dwell_times.items():
            A['dwell_times_by_zone'][zone].append(dwell_t)

        if len(cust.movement_path) > 5:
            A['customer_paths'].append(cust.movement_path.copy())

        cust.visited_zones.clear()
        cust.zone_dwell_times.clear()
        cust.movement_path.clear()
        self.prev_state_by_cust.pop(cust.id, None)

    # ------------------------------------------------------------------
    def _update_analytics_realtime(self, cust, real_dt):
        cust.movement_path.append((cust.position[0], cust.position[1], time.time()))

        zone = self._get_customer_zone(cust)
        if zone:
            if zone not in cust.visited_zones:
                cust.visited_zones.add(zone)
                self.analytics['area_visits'][zone] += 1
            if cust.state == 'shopping':
                cust.zone_dwell_times[zone] += real_dt

        prev_state = self.prev_state_by_cust.get(cust.id)
        if cust.state == 'shopping' and prev_state != 'shopping':
            itm   = getattr(cust, 'current_target_item', None)
            ttype = getattr(cust, 'current_target_type', None)
            if itm and ttype in ('item', 'impulse'):
                self.analytics['item_conversion_rates'][itm]['visits'] += 1

        cell = "{},{}".format(int(cust.position[0]), int(cust.position[1]))
        self.analytics['foot_traffic_density'][cell] += 1
        if self.analytics['foot_traffic_density'][cell] >= self.bottleneck_threshold:
            self.analytics['bottlenecks'][cell] += 1

        self.prev_state_by_cust[cust.id] = cust.state

    # ------------------------------------------------------------------
    # Floor-aware zone detection
    # ------------------------------------------------------------------
    def _get_customer_zone(self, cust):
        x, y = cust.position
        floor = getattr(cust, 'floor', 1)
        cached_floor = getattr(self, '_zones_cache_floor', None)
        if getattr(self, '_zones_cache', None) is None or cached_floor != floor:
            self._zones_cache = self._build_zones_cache(floor)
            self._zones_cache_floor = floor
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
        sim_minutes = self.run_time / 60.0
        A = self.analytics
        total_cust = max(1, A.get('total_customers', 0))

        lines = []
        lines.append("🏪 COMPREHENSIVE SHOP ANALYTICS")
        lines.append("=" * 50)
        lines.append("")

        lines.append("💰 SALES PERFORMANCE")
        lines.append("-" * 25)
        conv_rate     = A['completed_purchases'] / total_cust * 100
        baskets       = A.get('basket_sizes', [])
        avg_basket    = sum(baskets) / len(baskets) if baskets else 0.0
        rev_per_cust  = A['total_revenue'] / total_cust
        shop_area     = self.shop.width * self.shop.height
        sales_per_sqm = A['total_revenue'] / shop_area if shop_area else 0.0
        abandon       = A.get('abandoned_carts', 0)
        abandon_rate  = abandon / total_cust * 100

        lines.append("Total Revenue:         ${:.2f}".format(A['total_revenue']))
        lines.append("Conversion Rate:       {:.1f}%".format(conv_rate))
        lines.append("Avg Basket Size:       {:.1f} items".format(avg_basket))
        lines.append("Revenue/Customer:      ${:.2f}".format(rev_per_cust))
        lines.append("Sales per m²:          ${:.2f}".format(sales_per_sqm))
        lines.append("Cart Abandonment:      {} ({:.1f}%)".format(abandon, abandon_rate))
        lines.append("Pre-Optimization Revenue:   ${:.2f}".format(A.get('pre_optimization_revenue', 0.0)))
        lines.append("Post-Optimization Revenue:  ${:.2f}".format(A.get('post_optimization_revenue', 0.0)))
        lines.append("Optimization Impact:        {:+.1f}%".format(A.get('optimization_impact', 0.0)))
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
        lines.append("Total Customers:       {}".format(A['total_customers']))
        lines.append("Active Customers:      {}".format(len(self.customers)))
        lines.append("Avg Time in Shop:      {:.1f}s".format(A.get('average_time_in_shop', 0.0)))
        lines.append("Return Customers:      {} ({:.1f}%)".format(
            A.get('return_customers', 0),
            A.get('return_customers', 0) / total_cust * 100))

        wc_visits = A['area_visits'].get('WC', 0)
        lines.append("WC Usage Rate:         {:.1f}% ({} visits)".format(
            wc_visits / total_cust * 100, wc_visits))

        clv_list = []
        for segment, total_clv in A.get('customer_lifetime_values', {}).items():
            seg_cnt = max(1, total_cust * 0.33)
            avg_clv = total_clv / seg_cnt
            clv_list.append((segment.capitalize(), avg_clv))
        clv_list.sort(key=lambda x: x[1], reverse=True)
        for seg_name, avg_clv in clv_list:
            lines.append("  CLV ({}): ${:.2f}".format(seg_name, avg_clv))
        lines.append("Exit Traffic by Minute:")
        for mn, cnt in sorted(A.get('exit_traffic_by_minute', {}).items()):
            lines.append("  {:>2} min → {} exits".format(mn, cnt))
        lines.append("")

        lines.append("🎯 PRODUCT PERFORMANCE")
        lines.append("-" * 25)
        item_convs = {}
        for itm, data in A.get('item_conversion_rates', {}).items():
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
            for key, cnt in raw_cm.items():
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
            peak_minute, peak_count = max(custs_over_time.items(), key=lambda kv: kv[1])
            lines.append("Peak Time: {} min ({} customers)".format(peak_minute, peak_count))
        if A['dwell_times_by_zone']:
            avg_dwell = {z: sum(ts) / len(ts)
                         for z, ts in A['dwell_times_by_zone'].items() if ts}
            lines.append("Avg Section Dwell (s):")
            for zone, val in sorted(avg_dwell.items(), key=lambda kv: kv[1], reverse=True):
                lines.append("  {}: {:.1f}s".format(zone, val))

        checkout_times = A.get('queue_wait_times', [])
        if checkout_times:
            avg_chk = sum(checkout_times) / len(checkout_times)
            lines.append("Avg Checkout Time:     {:.1f}s over {} customers".format(
                avg_chk, len(checkout_times)))

        bn = A.get('bottlenecks', {})
        if bn:
            lines.append("Top Bottleneck Cells:")
            for cell, hits in sorted(bn.items(), key=lambda x: x[1], reverse=True)[:3]:
                lines.append("  {}: {} hits".format(cell, hits))
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