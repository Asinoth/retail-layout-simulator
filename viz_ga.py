import time
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import json
import os

import tkinter as tk
from tkinter import simpledialog, messagebox, colorchooser, ttk, font as tkfont, filedialog
from collections import defaultdict
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap
from drag import DraggableRectangle
from simulation import CustomerFlowSimulation

import copy
from scipy import stats as sp_stats


class GAMixin:
    """Genetic-algorithm layout optimization tab plus encoders/decoders/scoring."""

    def _create_ga_tab(self, notebook):
        tab = tk.Frame(notebook, bg='#002747')
        notebook.add(tab, text="GA Optimizer")

        ctrl = tk.Frame(tab, bg='#002747')
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=10, pady=8)

        lbl_kw = dict(fg='white', bg='#002747', font=('Arial', 10))
        ent_kw = dict(width=7)

        tk.Label(ctrl, text="Pop size:", **lbl_kw).grid(row=0, column=0, padx=3)
        self._ga_pop = tk.IntVar(value=40)
        tk.Entry(ctrl, textvariable=self._ga_pop, **ent_kw).grid(row=0, column=1, padx=3)

        tk.Label(ctrl, text="Generations:", **lbl_kw).grid(row=0, column=2, padx=3)
        self._ga_gens = tk.IntVar(value=50)
        tk.Entry(ctrl, textvariable=self._ga_gens, **ent_kw).grid(row=0, column=3, padx=3)

        tk.Label(ctrl, text="Mutation %:", **lbl_kw).grid(row=0, column=4, padx=3)
        self._ga_mut = tk.DoubleVar(value=15.0)
        tk.Entry(ctrl, textvariable=self._ga_mut, **ent_kw).grid(row=0, column=5, padx=3)

        tk.Label(ctrl, text="Elite %:", **lbl_kw).grid(row=0, column=6, padx=3)
        self._ga_elite = tk.DoubleVar(value=10.0)
        tk.Entry(ctrl, textvariable=self._ga_elite, **ent_kw).grid(row=0, column=7, padx=3)

        tk.Label(ctrl, text="MC iters/eval:", **lbl_kw).grid(row=0, column=8, padx=3)
        self._ga_mc_iters = tk.IntVar(value=500)
        tk.Entry(ctrl, textvariable=self._ga_mc_iters, **ent_kw).grid(row=0, column=9, padx=3)

        tk.Label(ctrl, text="MC days:", **lbl_kw).grid(row=0, column=10, padx=3)
        self._ga_mc_days = tk.IntVar(value=30)
        tk.Entry(ctrl, textvariable=self._ga_mc_days, **ent_kw).grid(row=0, column=11, padx=3)

        self._ga_progress = tk.StringVar(value="")
        tk.Label(ctrl, textvariable=self._ga_progress, fg='#4ECDC4',
                 bg='#002747', font=('Arial', 10, 'bold')).grid(row=0, column=14, padx=8)

        tk.Button(
            ctrl, text="Run GA", command=self._run_ga_optimization,
            bg='#e67e22', fg='white', font=('Arial', 11, 'bold'),
            relief=tk.RAISED, bd=3
        ).grid(row=0, column=12, padx=8)

        tk.Button(
            ctrl, text="Apply Best Layout", command=self._apply_ga_best,
            bg='#27ae60', fg='white', font=('Arial', 11, 'bold'),
            relief=tk.RAISED, bd=3
        ).grid(row=0, column=13, padx=8)

        body = tk.Frame(tab, bg='#002747')
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=3)
        body.rowconfigure(0, weight=1)

        self._ga_text = tk.Text(
            body, bg='#001a33', fg='white', font=('Courier', 10),
            wrap=tk.WORD, state=tk.DISABLED, width=48
        )
        self._ga_text.grid(row=0, column=0, sticky='nsew', padx=(0, 5))

        chart_frame = tk.Frame(body, bg='#002747')
        chart_frame.grid(row=0, column=1, sticky='nsew')

        self._ga_fig = Figure(figsize=(12, 8), dpi=100)
        self._ga_fig.patch.set_facecolor('#002747')
        self._ga_canvas_fig = FigureCanvasTkAgg(self._ga_fig, master=chart_frame)
        self._ga_canvas_fig.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._ga_best_chromosome = None
        self._ga_best_layout = None

    def _ga_accessible_floors(self):
        """Return ALL floors defined in the visualizer for optimization.

        The connector-reachable subset is still computed for reference and
        stored in ``self._ga_reachable_floors``, but optimization considers
        every floor in ``self.floors`` regardless of connectivity so that
        unreachable upper floors are not silently excluded from the GA.
        """
        # Keep reachable-floors info available for diagnostics / scoring.
        try:
            self._ga_reachable_floors = set(
                self.customer_simulation._reachable_floors(start_floor=1)
            )
        except Exception:
            self._ga_reachable_floors = {getattr(self, 'current_floor', 1)}

        try:
            all_floors = set(getattr(self, 'floors', {}).keys())
        except Exception:
            all_floors = set()
        if all_floors:
            return all_floors
        return {getattr(self, 'current_floor', 1)}

    def _ga_build_item_catalog(self, accessible_floors=None):
        """Build duplicate-safe optimization keys for items across floors."""
        if accessible_floors is not None:
            accessible_floors = set(accessible_floors)
        catalog = {}
        source_map = {}
        floor_map = {}
        for fid, fdata in sorted(getattr(self, 'floors', {}).items()):
            if accessible_floors is not None and fid not in accessible_floors:
                continue
            for source_name, data in fdata.get('items', {}).items():
                key = source_name
                if key in catalog:
                    key = f"F{fid}:{source_name}"
                    suffix = 2
                    while key in catalog:
                        key = f"F{fid}:{source_name}_{suffix}"
                        suffix += 1
                rec = dict(data)
                rec['floor'] = fid
                rec['source_name'] = source_name
                catalog[key] = rec
                source_map[key] = (fid, source_name)
                floor_map[key] = fid
        self._ga_item_catalog = catalog
        self._ga_item_source_map = source_map
        self._ga_item_floor_map = floor_map
        return catalog

    def _ga_resolve_item_key(self, key):
        source_map = getattr(self, '_ga_item_source_map', {}) or {}
        if key in source_map:
            return source_map[key]
        text = str(key)
        if text.startswith('F') and ':' in text:
            prefix, source = text.split(':', 1)
            if prefix[1:].isdigit():
                fid = int(prefix[1:])
                if fid in self.floors and source in self.floors[fid].get('items', {}):
                    return fid, source
        for fid, fdata in getattr(self, 'floors', {}).items():
            if key in fdata.get('items', {}):
                return fid, key
        return getattr(self, 'current_floor', 1), key

    def _ga_get_item_data(self, key):
        catalog = getattr(self, '_ga_item_catalog', {}) or {}
        if key in catalog:
            return catalog[key]
        fid, source = self._ga_resolve_item_key(key)
        data = self.floors.get(fid, {}).get('items', {}).get(source)
        if data is None:
            data = self.items.get(source, self.items.get(key, {}))
        rec = dict(data)
        rec.setdefault('floor', fid)
        rec.setdefault('source_name', source)
        return rec

    def _ga_get_item_floor(self, key):
        return self._ga_get_item_data(key).get('floor', self._ga_resolve_item_key(key)[0])

    def _ga_find_special_center(self, special_name):
        """Find Checkout/WC/Entrance center across floors; walls take precedence."""
        for fid, fdata in sorted(getattr(self, 'floors', {}).items()):
            walls = fdata.get('walls', {})
            if special_name in walls:
                pos = walls[special_name]['position']; size = walls[special_name]['size']
                return (pos[0] + size[0] / 2, pos[1] + size[1] / 2), fid
        for fid, fdata in sorted(getattr(self, 'floors', {}).items()):
            items = fdata.get('items', {})
            if special_name in items:
                pos = items[special_name]['position']; size = items[special_name]['size']
                return (pos[0] + size[0] / 2, pos[1] + size[1] / 2), fid
        return None, None

    def _ga_get_entrance_center(self):
        floor1 = getattr(self, 'floors', {}).get(1, {})
        door = floor1.get('door_position') or getattr(self, 'door_position', None)
        if door:
            return (float(door[0]), float(door[1])), 1
        return self._ga_find_special_center('Entrance')

    def _ga_connector_center(self, fid, connector_id):
        mapping = getattr(self, 'connectors', {}).get(connector_id, {})
        wname = mapping.get(fid)
        if wname is None:
            return None
        wall = self.floors.get(fid, {}).get('walls', {}).get(wname)
        if not wall:
            return None
        pos = wall['position']; size = wall['size']
        return (pos[0] + size[0] / 2, pos[1] + size[1] / 2)

    def _ga_floor_distance(self, start_pos, start_floor, dest_pos, dest_floor):
        """Approximate customer walking distance, using connectors between floors."""
        if start_floor == dest_floor:
            dx = dest_pos[0] - start_pos[0]
            dy = dest_pos[1] - start_pos[1]
            return (dx * dx + dy * dy) ** 0.5
        connectors = getattr(self, 'connectors', {}) or {}
        if not connectors:
            return math.sqrt(self.width ** 2 + self.height ** 2) * 10.0
        import heapq
        diag = math.sqrt(self.width ** 2 + self.height ** 2)
        heap = [(0.0, start_floor, tuple(start_pos))]
        best = {}
        while heap:
            cost, floor, pos = heapq.heappop(heap)
            state = (floor, round(pos[0], 2), round(pos[1], 2))
            if cost >= best.get(state, float('inf')):
                continue
            best[state] = cost
            if floor == dest_floor:
                return cost + math.hypot(dest_pos[0] - pos[0], dest_pos[1] - pos[1])
            for cid, mapping in connectors.items():
                if floor not in mapping:
                    continue
                cur_c = self._ga_connector_center(floor, cid)
                if cur_c is None:
                    continue
                to_connector = math.hypot(cur_c[0] - pos[0], cur_c[1] - pos[1])
                for nb in mapping:
                    if nb == floor:
                        continue
                    nb_c = self._ga_connector_center(nb, cid)
                    if nb_c is None:
                        continue
                    heapq.heappush(heap, (cost + to_connector + 1.0, nb, tuple(nb_c)))
        return diag * 10.0


    def _ga_get_movable_items(self):
        accessible = self._ga_accessible_floors()
        catalog = self._ga_build_item_catalog(accessible)
        fixed_names = {'Entrance', 'Checkout', 'WC'}
        out = []
        for key, data in catalog.items():
            source = data.get('source_name', key)
            cat = str(data.get('category', '')).lower()
            if source in fixed_names or key in fixed_names:
                continue
            if cat in ('entrance', 'checkout', 'wc', 'connector'):
                continue
            out.append(key)
        return out

    def _ga_encode(self, item_names):
        return np.array([
            list(self._ga_get_item_data(n).get('position', (0.0, 0.0)))
            for n in item_names
        ], dtype=np.float64)

    def _ga_decode(self, chromosome, item_names):
        layout = {}
        for i, n in enumerate(item_names):
            layout[n] = (float(chromosome[i, 0]), float(chromosome[i, 1]))
        return layout

    def _ga_compute_layout_score(self, positions, item_names, base_params):
        A = self.customer_simulation.analytics
        if len(item_names) == 0:
            return 0.0, {'composite': 0.0}

        positions = np.asarray(positions, dtype=np.float64)
        item_data = {n: self._ga_get_item_data(n) for n in item_names}
        item_floor = {n: int(item_data[n].get('floor', self._ga_get_item_floor(n))) for n in item_names}
        source_to_keys = defaultdict(list)
        for n in item_names:
            source_to_keys[item_data[n].get('source_name', n)].append(n)
        index_by_key = {n: i for i, n in enumerate(item_names)}

        def item_center(i, n):
            x, y = positions[i]
            w, h = item_data[n].get('size', (0.0, 0.0))
            return (x + w / 2, y + h / 2)

        def key_for_name(name):
            if name in index_by_key:
                return name
            keys = source_to_keys.get(name)
            return keys[0] if keys else None

        # ── 1. TRAFFIC SCORE (floor-aware heat-map alignment) ─────────
        traffic_score = 0.0
        traffic_count = 0
        sim = self.customer_simulation
        floor_heat = getattr(sim, '_floor_heat_raw', {}) or {}
        res = getattr(sim, 'heat_map_resolution', 20)
        for i, n in enumerate(item_names):
            fid = item_floor[n]
            heat = floor_heat.get(fid)
            if heat is None and fid == getattr(self, 'current_floor', 1):
                heat = getattr(sim, 'heat_map_data', None)
            if heat is None or not hasattr(heat, 'size') or heat.size == 0 or float(np.max(heat)) <= 0:
                continue
            cx, cy = item_center(i, n)
            xi = max(0, min(int(cx * res), heat.shape[0] - 1))
            yi = max(0, min(int(cy * res), heat.shape[1] - 1))
            traffic_score += float(heat[xi, yi]) / max(float(np.max(heat)), 1e-6)
            traffic_count += 1
        traffic_score = traffic_score / max(traffic_count, 1)

        # ── 2. CROSS-MERCHANDISING SCORE ───────────────────────────
        cross_score = 0.0
        cross_weight = 0.0
        for pair, count in dict(A.get('cross_merchandising', {})).items():
            if isinstance(pair, str) and '|' in pair:
                a, b = pair.split('|', 1)
            elif isinstance(pair, tuple) and len(pair) == 2:
                a, b = pair
            else:
                continue
            ka, kb = key_for_name(a), key_for_name(b)
            if ka is None or kb is None:
                continue
            ia, ib = index_by_key[ka], index_by_key[kb]
            ca, cb = item_center(ia, ka), item_center(ib, kb)
            dist = self._ga_floor_distance(ca, item_floor[ka], cb, item_floor[kb])
            max_dist = math.sqrt(self.width ** 2 + self.height ** 2) * (1 + abs(item_floor[ka] - item_floor[kb]))
            cross_score += max(0.0, 1.0 - dist / max(max_dist, 1e-6)) * max(count, 1)
            cross_weight += max(count, 1)
        if cross_weight > 0:
            cross_score /= cross_weight

        # ── 3. IMPULSE PLACEMENT SCORE ─────────────────────────────
        impulse_score = 0.0
        checkout_pos, checkout_floor = self._ga_find_special_center('Checkout')
        if checkout_pos:
            n_impulse = 0
            for i, n in enumerate(item_names):
                cat = str(item_data[n].get('category', '')).lower()
                if 'impulse' in cat:
                    c = item_center(i, n)
                    dist = self._ga_floor_distance(c, item_floor[n], checkout_pos, checkout_floor)
                    max_d = math.sqrt(self.width ** 2 + self.height ** 2) * (1 + abs(item_floor[n] - checkout_floor))
                    impulse_score += max(0.0, 1.0 - dist / max(max_d, 1e-6))
                    n_impulse += 1
            if n_impulse > 0:
                impulse_score /= n_impulse

        # ── 4. FLOW EFFICIENCY SCORE ──────────────────────────────
        flow_score = 0.0
        entrance_pos, entrance_floor = self._ga_get_entrance_center()
        if entrance_pos and checkout_pos:
            top_items = sorted(A.get('popular_items', {}).items(), key=lambda kv: kv[1], reverse=True)[:5]
            waypoint_keys = [key_for_name(name) for name, _ in top_items]
            waypoint_keys = [k for k in waypoint_keys if k is not None]
            waypoints = [(entrance_pos, entrance_floor)]
            for key in waypoint_keys:
                waypoints.append((item_center(index_by_key[key], key), item_floor[key]))
            waypoints.append((checkout_pos, checkout_floor))
            total_dist = 0.0
            for k in range(len(waypoints) - 1):
                total_dist += self._ga_floor_distance(waypoints[k][0], waypoints[k][1],
                                                       waypoints[k + 1][0], waypoints[k + 1][1])
            max_path = math.sqrt(self.width ** 2 + self.height ** 2) * max(len(waypoints), 1)
            flow_score = max(0.0, 1.0 - total_dist / max(max_path, 1e-6))

        # ── 5. OVERLAP PENALTY (same-floor only; walls floor-aware) ──
        overlap_penalty = 0.0
        for i, ni in enumerate(item_names):
            x1, y1 = positions[i]
            w1, h1 = item_data[ni].get('size', (0.0, 0.0))
            fid = item_floor[ni]
            if x1 < 0 or y1 < 0 or x1 + w1 > self.width or y1 + h1 > self.height:
                overlap_penalty += 1.0
            for j in range(i + 1, len(item_names)):
                nj = item_names[j]
                if item_floor[nj] != fid:
                    continue
                x2, y2 = positions[j]
                w2, h2 = item_data[nj].get('size', (0.0, 0.0))
                if not (x1 + w1 <= x2 or x1 >= x2 + w2 or y1 + h1 <= y2 or y1 >= y2 + h2):
                    overlap_penalty += 1.0
            for wname, wd in self.floors.get(fid, {}).get('walls', {}).items():
                if wname.startswith('Section_'):
                    continue
                wx, wy = wd['position']; ww, wh = wd['size']
                if not (x1 + w1 <= wx or x1 >= wx + ww or y1 + h1 <= wy or y1 >= wy + wh):
                    overlap_penalty += 0.5

        # ── 6. REVENUE-WEIGHTED PLACEMENT ─────────────────────────
        revenue_placement_score = 0.0
        conv_rates = dict(A.get('item_conversion_rates', {}))
        rev_by_area = dict(A.get('revenue_by_area', {}))
        item_revenues = []
        for n in item_names:
            cat = item_data[n].get('category', '')
            source = item_data[n].get('source_name', n)
            area_rev = rev_by_area.get(cat, 0.0)
            cr = conv_rates.get(n, conv_rates.get(source, {'visits': 0, 'purchases': 0}))
            conv_r = cr.get('purchases', 0) / max(cr.get('visits', 0), 1)
            item_revenues.append(area_rev * (1.0 + conv_r))
        max_rev = max(item_revenues) if item_revenues and max(item_revenues) > 0 else 1.0
        rev_count = 0
        for i, n in enumerate(item_names):
            fid = item_floor[n]
            heat = floor_heat.get(fid)
            if heat is None and fid == getattr(self, 'current_floor', 1):
                heat = getattr(sim, 'heat_map_data', None)
            if heat is None or not hasattr(heat, 'size') or heat.size == 0 or float(np.max(heat)) <= 0:
                continue
            cx, cy = item_center(i, n)
            xi = max(0, min(int(cx * res), heat.shape[0] - 1))
            yi = max(0, min(int(cy * res), heat.shape[1] - 1))
            revenue_placement_score += (item_revenues[i] / max_rev) * (float(heat[xi, yi]) / max(float(np.max(heat)), 1e-6))
            rev_count += 1
        revenue_placement_score = revenue_placement_score / max(rev_count, 1)

        # ── 7. DWELL-TIME ALIGNMENT ───────────────────────────────
        dwell_score = 0.0
        dwell_data = dict(A.get('dwell_times_by_zone', {}))
        if dwell_data:
            all_dwells = [float(np.mean(times)) for times in dwell_data.values() if times]
            max_dwell = max(all_dwells) if all_dwells else 1.0
            n_dwell = 0
            for n in item_names:
                cat = item_data[n].get('category', '')
                zone_times = dwell_data.get(cat, [])
                if zone_times:
                    dwell_score += float(np.mean(zone_times)) / max(max_dwell, 1e-6)
                    n_dwell += 1
            if n_dwell > 0:
                dwell_score /= n_dwell

        # ── 8. BOTTLENECK AVOIDANCE ───────────────────────────────
        bottleneck_penalty = 0.0
        bn_data = dict(A.get('bottlenecks', {}))
        if bn_data:
            max_bn = max(bn_data.values()) if bn_data.values() else 1
            for i, n in enumerate(item_names):
                x, y = positions[i]
                w, h = item_data[n].get('size', (0.0, 0.0))
                cx, cy = x + w / 2, y + h / 2
                cell_key = f"{int(cx)},{int(cy)}"
                cat = item_data[n].get('category', '')
                bn_val = max(bn_data.get(cell_key, 0), bn_data.get(cat, 0))
                bottleneck_penalty += bn_val / max(max_bn, 1)
            bottleneck_penalty /= max(len(item_names), 1)

        # ── 9. SECTION COMPLIANCE ─────────────────────────────────
        section_score = 0.0
        n_sec = 0
        for i, n in enumerate(item_names):
            cat = item_data[n].get('category', '')
            sec_wall = self.floors.get(item_floor[n], {}).get('walls', {}).get(f"Section_{cat}")
            if not sec_wall:
                continue
            n_sec += 1
            sx, sy = sec_wall['position']; sw, sh = sec_wall['size']
            ix, iy = positions[i]; iw, ih = item_data[n].get('size', (0.0, 0.0))
            if ix >= sx and iy >= sy and ix + iw <= sx + sw and iy + ih <= sy + sh:
                section_score += 1.0
            else:
                dx = max(sx - ix, 0, ix + iw - (sx + sw))
                dy = max(sy - iy, 0, iy + ih - (sy + sh))
                section_score -= math.sqrt(dx ** 2 + dy ** 2) / max(math.sqrt(self.width ** 2 + self.height ** 2), 1e-6)
        if n_sec > 0:
            section_score /= n_sec

        # ── 10. CONVERSION-WEIGHTED ACCESSIBILITY ─────────────────
        accessibility_score = 0.0
        if entrance_pos:
            weighted_sum = 0.0
            weight_total = 0.0
            for i, n in enumerate(item_names):
                source = item_data[n].get('source_name', n)
                cr = conv_rates.get(n, conv_rates.get(source, {'visits': 0, 'purchases': 0}))
                conv_r = cr.get('purchases', 0) / max(cr.get('visits', 0), 1)
                dist = self._ga_floor_distance(entrance_pos, entrance_floor, item_center(i, n), item_floor[n])
                max_d = math.sqrt(self.width ** 2 + self.height ** 2) * (1 + abs(item_floor[n] - entrance_floor))
                weighted_sum += conv_r * max(0.0, 1.0 - dist / max(max_d, 1e-6))
                weight_total += conv_r
            if weight_total > 0:
                accessibility_score = weighted_sum / weight_total

        # ── COMPOSITE ─────────────────────────────────────────────
        # Weights and citations live in retail_literature.py — see that
        # module for the methodology footnote a TOMACS reviewer will want
        # to read alongside the paper's Methods section.
        from retail_literature import (
            GA_W_TRAFFIC, GA_W_CROSS_MERCH, GA_W_IMPULSE, GA_W_FLOW,
            GA_W_REVENUE_PLACEMENT, GA_W_DWELL, GA_W_SECTION_COMPLIANCE,
            GA_W_ACCESSIBILITY, GA_PEN_OVERLAP, GA_PEN_BOTTLENECK,
        )
        score = (traffic_score           * GA_W_TRAFFIC
               + cross_score             * GA_W_CROSS_MERCH
               + impulse_score           * GA_W_IMPULSE
               + flow_score              * GA_W_FLOW
               + revenue_placement_score * GA_W_REVENUE_PLACEMENT
               + dwell_score             * GA_W_DWELL
               + section_score           * GA_W_SECTION_COMPLIANCE
               + accessibility_score     * GA_W_ACCESSIBILITY
               - overlap_penalty         * GA_PEN_OVERLAP
               - bottleneck_penalty      * GA_PEN_BOTTLENECK)

        return score, {
            'traffic':              traffic_score,
            'cross_merch':          cross_score,
            'impulse':              impulse_score,
            'flow':                 flow_score,
            'revenue_placement':    revenue_placement_score,
            'dwell_alignment':      dwell_score,
            'bottleneck_penalty':   bottleneck_penalty,
            'section_compliance':   section_score,
            'accessibility':        accessibility_score,
            'overlap_penalty':      overlap_penalty,
            'composite':            score,
        }
