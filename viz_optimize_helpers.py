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


class OptimizeHelpersMixin:
    """Optimization helpers: progress dialog, traffic zones, placement, repair, etc."""

    def _create_optimization_progress_dialog(self):
        """Create a progress dialog for the optimization pipeline.

        The dialog hosts a ``ttk.Progressbar`` in *determinate* mode plus
        labels for the current phase and a numeric percentage. All UI calls
        are wrapped in try/except so the pipeline survives the user closing
        the dialog mid-run.

        Callers update progress via ``update_status(text, pct=None)`` —
        passing ``pct`` (0-100) drives the progress bar; otherwise only
        the phase label changes. ``pump_events()`` must be called inside
        long synchronous loops so the dialog repaints and Windows does not
        mark the window 'not responding'.
        """
        class OptimizationProgress:
            def __init__(self, parent):
                self._alive = True
                self.dialog = tk.Toplevel(parent)
                self.dialog.title("Optimizing Layout...")
                self.dialog.geometry("460x200")
                self.dialog.transient(parent)

                # Center the dialog
                try:
                    self.dialog.update_idletasks()
                except tk.TclError:
                    pass
                x = (self.dialog.winfo_screenwidth() // 2) - 230
                y = (self.dialog.winfo_screenheight() // 2) - 100
                self.dialog.geometry(f"460x200+{x}+{y}")

                # Header
                tk.Label(self.dialog, text="Optimizing Shop Layout",
                         font=('Arial', 14, 'bold')).pack(pady=(12, 4))

                # Phase / status line
                self.status_label = tk.Label(
                    self.dialog, text="Initializing...",
                    font=('Arial', 10), wraplength=420, justify='center')
                self.status_label.pack(pady=2)

                # Numeric percentage (updated alongside the bar)
                self.pct_label = tk.Label(
                    self.dialog, text="0%",
                    font=('Arial', 11, 'bold'))
                self.pct_label.pack(pady=2)

                # Determinate progress bar driven by update_status(..., pct)
                self.progress_bar = ttk.Progressbar(
                    self.dialog, mode='determinate', maximum=100, value=0)
                self.progress_bar.pack(pady=(4, 14), padx=20, fill=tk.X)

                # Hint that the work runs on a worker thread; the dialog
                # stays responsive but cannot be cancelled.
                tk.Label(
                    self.dialog,
                    text="(Window may be moved while optimization runs.)",
                    font=('Arial', 8), fg='#555555').pack(pady=(0, 8))

                self._current_pct = 0.0
                self.dialog.protocol("WM_DELETE_WINDOW", self._on_user_close)

            def _on_user_close(self):
                self._alive = False
                try:
                    self.dialog.destroy()
                except tk.TclError:
                    pass

            def update_status(self, status, pct=None):
                """Update phase label and (optionally) the percentage bar."""
                if not self._alive:
                    return
                try:
                    self.status_label.config(text=status)
                    if pct is not None:
                        # Monotonic clamp: never let the bar go backwards.
                        p = max(self._current_pct, min(100.0, float(pct)))
                        self._current_pct = p
                        self.progress_bar.config(value=p)
                        self.pct_label.config(text=f"{p:0.0f}%")
                    # Full update (not just update_idletasks) so the new label
                    # and bar value are flushed immediately — matches the old
                    # working dialog's behavior.
                    self.dialog.update()
                except tk.TclError:
                    self._alive = False

            def update_progress(self, pct):
                """Advance the percentage without changing the phase text."""
                if not self._alive:
                    return
                try:
                    p = max(self._current_pct, min(100.0, float(pct)))
                    self._current_pct = p
                    self.progress_bar.config(value=p)
                    self.pct_label.config(text=f"{p:0.0f}%")
                    self.dialog.update()
                except tk.TclError:
                    self._alive = False

            def pump_events(self):
                """Pump the Tk event loop so the dialog stays responsive
                during long synchronous compute (e.g. inside the GA
                fitness loop). Without this Windows marks the window
                'not responding' after ~5 seconds of blocked main thread.
                """
                if not self._alive:
                    return
                try:
                    self.dialog.update()
                except tk.TclError:
                    self._alive = False

            def close(self):
                if not self._alive:
                    return
                # Drive the bar to 100 % so the final repaint shows completion.
                try:
                    self.progress_bar.config(value=100)
                    self.pct_label.config(text="100%")
                    self.dialog.update_idletasks()
                except tk.TclError:
                    pass
                self._alive = False
                try:
                    self.dialog.destroy()
                except tk.TclError:
                    pass

        return OptimizationProgress(self.tk_root)



    def _analyze_current_performance(self):
        """Analyze current shop performance metrics."""
        analytics = self.customer_simulation.analytics
        
        # Calculate item performance scores across ALL floors so that items
        # on connector-unreachable floors are still represented in the report
        # (matches the GA's all-floors optimization scope).
        item_scores = {}
        try:
            items_catalog = {k: dict(v) for k, v in self.all_items_across_floors().items()}
        except Exception:
            items_catalog = {k: dict(v) for k, v in self.items.items()}
        for item_name, item_data in items_catalog.items():
            source_name = item_data.get('source_name', item_name)
            score = 0
            
            # Popularity score (visits)
            popularity = max(analytics['popular_items'].get(item_name, 0),
                             analytics['popular_items'].get(source_name, 0))
            score += popularity * 2
            
            # Conversion rate score
            conversion_data = analytics['item_conversion_rates'].get(
                item_name,
                analytics['item_conversion_rates'].get(source_name, {'visits': 0, 'purchases': 0})
            )
            if conversion_data['visits'] > 0:
                conversion_rate = conversion_data['purchases'] / conversion_data['visits']
                score += conversion_rate * 100
            
            # Revenue contribution score
            revenue_contribution = 0
            item_category = item_data.get('category', 'Unknown')
            if item_category in analytics['revenue_by_area']:
                revenue_contribution = analytics['revenue_by_area'][item_category]
            score += revenue_contribution
            
            item_scores[item_name] = score
        
        # Identify top performers
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        
        performance_data = {
            'item_scores': item_scores,
            'top_performers': [item for item, score in sorted_items[:5]],
            'low_performers': [item for item, score in sorted_items[-5:]],
            'cross_merchandising': dict(analytics['cross_merchandising']),
            'total_revenue': analytics['total_revenue'],
            'conversion_rate': analytics['completed_purchases'] / max(1, analytics['total_customers'])
        }
        
        return performance_data

    def _identify_traffic_zones(self):
        """Identify high, medium, and low traffic zones from heat map data."""
        if not hasattr(self.customer_simulation, 'heat_map_data'):
            return {'high': [], 'medium': [], 'low': []}
        
        heat_data = self.customer_simulation.heat_map_data
        # Use the same resolution the simulation uses to map cells → meters
        res = getattr(self.customer_simulation, 'heat_map_resolution', 20)
        
        # Calculate percentiles for zone classification
        non_zero_values = heat_data[heat_data > 0]
        if len(non_zero_values) == 0:
            return {'high': [], 'medium': [], 'low': []}
        
        high_threshold = np.percentile(non_zero_values, 80)
        medium_threshold = np.percentile(non_zero_values, 50)
        
        zones = {'high': [], 'medium': [], 'low': []}
        
        # Create zones based on heat map data
        for x_idx in range(heat_data.shape[0]):
            for y_idx in range(heat_data.shape[1]):
                heat_value = heat_data[x_idx, y_idx]
                if heat_value > 0:
                    # Convert grid coordinates back to shop coordinates (meters)
                    x = x_idx / res
                    y = y_idx / res
                    
                    if heat_value >= high_threshold:
                        zones['high'].append((x, y, heat_value))
                    elif heat_value >= medium_threshold:
                        zones['medium'].append((x, y, heat_value))
                    else:
                        zones['low'].append((x, y, heat_value))
        
        return zones



    def _clamp_position_to_section(self, item_name, x, y):
        """
        Ensure that the top-left (x,y) of item_name stays within its Section_<category> wall.
        If the item has no matching section or already fits, returns (x,y) unchanged.
        Ensures that during Optimize Layout items dont move outside their section
        """
        data = self.items[item_name]
        sec      = data['category']
        sec_wall = self.walls.get(f"Section_{sec}")
        if not sec_wall:
            return x, y
        sx, sy = sec_wall['position']
        sw, sh = sec_wall['size']
        iw, ih = data['size']
        # clamp so item fully fits inside that rect
        cx = max(sx, min(x, sx+sw-iw))
        cy = max(sy, min(y, sy+sh-ih))
        return cx, cy


    def _optimize_product_placement(self, performance_data, traffic_zones):
        """
        Optimize product placement based on performance and traffic data.
        Now respects section boundaries: no item leaves its original section.
        Ensures final positions are collision-free (no item overlap).
        """
        changes = []

        # Strategy 1: Move top performers to high-traffic zones (skip impulse items)
        top_performers = performance_data['top_performers']
        high_positions = [(x, y) for x, y, _ in traffic_zones['high'][:5]]

        for i, item_name in enumerate(top_performers[:len(high_positions)]):
            item = self.items.get(item_name)
            if not item or item.get('category') == 'Impulse':
                continue

            old_x, old_y = item['position']
            iw, ih = item['size']
            tx, ty = high_positions[i]

            # find a safe spot near the high-traffic cell, then clamp to section
            cand_x, cand_y = self._find_safe_position(tx, ty, item['size'], exclude_item=item_name)
            new_x, new_y = self._clamp_position_to_section(item_name, cand_x, cand_y)

            # If clamp caused collision, search again inside the section
            if not self._is_position_safe(new_x, new_y, iw, ih, exclude_item=item_name):
                new_x, new_y = self._find_safe_position_in_section(item_name, tx, ty)

            if (new_x, new_y) != (old_x, old_y) and self._is_position_safe(new_x, new_y, iw, ih, exclude_item=item_name):
                self.items[item_name]['position'] = (new_x, new_y)
                changes.append(f"Moved high-performer '{item_name}' within section '{item['category']}'")

        # Strategy 2: Move low performers away from prime locations but keep them in their own section
        low_performers = performance_data['low_performers']
        for item_name in low_performers:
            item = self.items.get(item_name)
            if not item or item.get('category') == 'Impulse':
                continue

            if self._is_in_high_traffic_zone(item['position'], traffic_zones):
                iw, ih = item['size']
                old_pos = item['position']
                medium = traffic_zones['medium']
                if not medium:
                    continue
                tx, ty, _ = medium[0]

                cand_x, cand_y = self._find_safe_position(tx, ty, item['size'], exclude_item=item_name)
                new_x, new_y = self._clamp_position_to_section(item_name, cand_x, cand_y)
                if not self._is_position_safe(new_x, new_y, iw, ih, exclude_item=item_name):
                    new_x, new_y = self._find_safe_position_in_section(item_name, tx, ty)

                if (new_x, new_y) != old_pos and self._is_position_safe(new_x, new_y, iw, ih, exclude_item=item_name):
                    self.items[item_name]['position'] = (new_x, new_y)
                    changes.append(f"Relocated low-performer '{item_name}' inside section '{item['category']}'")

        return changes



    def _implement_cross_merchandising(self):
        """
        Implement cross-merchandising by grouping complementary products.
        Never move an item outside its own Section_<category>, and 
        always use safe positions to prevent overlap.
        """
        changes = []
        analytics = self.customer_simulation.analytics

        # Top 3 combos by frequency (keys may be tuples or 'A|B' strings)
        top_combos = sorted(
            analytics['cross_merchandising'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]

        for key, frequency in top_combos:
            if frequency < 2:
                continue

            # Parse key into two item names
            if isinstance(key, (list, tuple)):
                if len(key) != 2:
                    continue
                item1, item2 = key
            else:
                parts = str(key).split('|')
                if len(parts) != 2:
                    continue
                item1, item2 = parts

            if item1 not in self.items or item2 not in self.items:
                continue

            sec1 = self.items[item1]['category']
            sec2 = self.items[item2]['category']
            if sec1 != sec2:
                continue

            section_center = self._get_section_center(sec1)
            if not section_center:
                continue

            size2 = self.items[item2]['size']
            old_x, old_y = self.items[item2]['position']
            iw, ih = size2

            # first candidate near center (global safe), then enforce section bounds
            cand_x, cand_y = self._find_safe_position(section_center[0], section_center[1], size2, exclude_item=item2)
            new_x, new_y = self._clamp_position_to_section(item2, cand_x, cand_y)

            # ensure final is collision-free; if not, search inside the section around center
            if not self._is_position_safe(new_x, new_y, iw, ih, exclude_item=item2):
                new_x, new_y = self._find_safe_position_in_section(item2, section_center[0], section_center[1])

            if (new_x, new_y) != (old_x, old_y) and self._is_position_safe(new_x, new_y, iw, ih, exclude_item=item2):
                self.items[item2]['position'] = (new_x, new_y)
                changes.append(
                    f"Cross-merch: Moved '{item2}' closer to '{item1}' within section '{sec1}'"
                )

        return changes



    def _optimize_customer_journey(self, traffic_zones):
        """Optimize customer journey by strategic product placement."""
        changes = []
        
        # Strategy: Place popular essential items deeper in store
        # This increases exposure to other products
        analytics = self.customer_simulation.analytics
        popular_items = sorted(analytics['popular_items'].items(), 
                            key=lambda x: x[1], reverse=True)[:3]
        
        # Find positions away from entrance
        entrance_pos = self.customer_simulation.door_position or (self.width/2, 0)
        
        for item_name, visits in popular_items:
            if item_name in self.items and visits > 5:  # Only for truly popular items
                current_pos = self.items[item_name]['position']
                
                # Calculate distance from entrance
                distance_from_entrance = ((current_pos[0] - entrance_pos[0])**2 + 
                                        (current_pos[1] - entrance_pos[1])**2)**0.5
                
                # If item is too close to entrance, move it deeper
                if distance_from_entrance < self.height * 0.4:  # Less than 40% of shop depth
                    # Find position deeper in store
                    target_y = self.height * 0.7  # 70% depth
                    target_x = current_pos[0]  # Keep same x position
                    
                    new_pos = self._find_safe_position(target_x, target_y,
                                                    self.items[item_name]['size'],
                                                    exclude_item=item_name)
                    
                    if new_pos != current_pos:
                        self.items[item_name]['position'] = new_pos
                        changes.append(f"Moved popular item '{item_name}' deeper to increase product exposure")
    
        return changes

    
    
    def _optimize_checkout_area(self, performance_data):
        """
        Reposition only existing impulse items into the 3m high-traffic zone
        around checkout. Do not introduce new impulse items or change categories.
        """
        changes = []
        if 'Checkout' not in self.walls:
            return changes

        # Checkout centre + size
        cx_co, cy_co = self.walls['Checkout']['position']
        cw, ch      = self.walls['Checkout']['size']
        # Define a 3m radius impulse zone around the geometric centre
        center_x = cx_co + cw/2
        center_y = cy_co + ch/2
        zone_radius = 3.0

        # Collect conversion rates for existing impulse items
        impulse_stats = []
        rates = self.customer_simulation.analytics['item_conversion_rates']
        for name, data in self.items.items():
            if data.get('category') != 'Impulse':
                continue
            conv = rates.get(name, {'visits': 0, 'purchases': 0})
            if conv['visits'] > 0:
                rate = conv['purchases'] / conv['visits']
                impulse_stats.append((name, rate))

        # Sort descending and take top-2
        impulse_stats.sort(key=lambda x: x[1], reverse=True)
        for name, _ in impulse_stats[:2]:
            x0, y0 = self.items[name]['position']
            w, h   = self.items[name]['size']
            cx_it, cy_it = x0 + w/2, y0 + h/2
            dist = math.hypot(center_x - cx_it, center_y - cy_it)
            # if outside the 3m circle, slide it onto the circle boundary
            if dist > zone_radius:
                # direction vector from item to checkout centre
                dx, dy = center_x - cx_it, center_y - cy_it
                mag = math.hypot(dx, dy) or 1.0
                # target centre on the circle
                tx = center_x - (dx/mag) * zone_radius
                ty = center_y - (dy/mag) * zone_radius
                # adjust for rectangle origin
                new_x = tx - w/2
                new_y = ty - h/2
                new_pos = self._find_safe_position(new_x, new_y, (w, h), exclude_item=name)
                if new_pos != (x0, y0):
                    self.items[name]['position'] = new_pos
                    changes.append(f"Repositioned impulse item '{name}' near checkout")

        return changes
    

    def _find_safe_position(self, target_x, target_y, item_size, exclude_item=None):
        """Find a safe position near target coordinates that doesn't cause overlaps."""
        item_w, item_h = item_size
        
        # Try the exact target position first
        if self._is_position_safe(target_x, target_y, item_w, item_h, exclude_item):
            return (target_x, target_y)
        
        # Try positions in expanding circles around target
        for radius in [0.5, 1.0, 1.5, 2.0]:
            for angle in range(0, 360, 30):  # Try every 30 degrees
                x = target_x + radius * np.cos(np.radians(angle))
                y = target_y + radius * np.sin(np.radians(angle))
                
                # Ensure within shop bounds
                x = max(0.1, min(x, self.width - item_w - 0.1))
                y = max(0.1, min(y, self.height - item_h - 0.1))
                
                if self._is_position_safe(x, y, item_w, item_h, exclude_item):
                    return (x, y)
        
        # Fallback: return original position if no safe position found
        if exclude_item and exclude_item in self.items:
            return self.items[exclude_item]['position']
        
        return (target_x, target_y)

    def _is_position_safe(self, x, y, w, h, exclude_item=None):
        """Check if a position is safe (no overlaps with walls or other items)."""
        # Check bounds
        if x < 0 or y < 0 or x + w > self.width or y + h > self.height:
            return False
        
        # Check overlap with walls
        for wall_name, wall_data in self.walls.items():
            if wall_name.startswith('Section_'):
                continue  # Allow overlap with sections
            
            wx, wy = wall_data['position']
            ww, wh = wall_data['size']
            
            if not (x + w <= wx or x >= wx + ww or y + h <= wy or y >= wy + wh):
                return False
        
        # Check overlap with other items
        for item_name, item_data in self.items.items():
            if item_name == exclude_item:
                continue
            
            ix, iy = item_data['position']
            iw, ih = item_data['size']
            
            if not (x + w <= ix or x >= ix + iw or y + h <= iy or y >= iy + ih):
                return False
        
        return True

    def _find_safe_position_in_section(self, item_name, target_x, target_y):
        """
        Find a collision-free position for item_name near (target_x, target_y),
        constrained to its own Section_<category> rectangle. Falls back to the
        global search if the item has no section.

        Returns (x, y) which keeps the full item rect inside the section and
        does not overlap any walls (except sections) or other items.
        """
        if item_name not in self.items:
            return (target_x, target_y)

        data = self.items[item_name]
        iw, ih = data['size']
        sec = data.get('category')
        sec_wall = self.walls.get(f"Section_{sec}") if sec else None

        # No section → use global search
        if not sec_wall:
            return self._find_safe_position(target_x, target_y, (iw, ih), exclude_item=item_name)

        sx, sy = sec_wall['position']
        sw, sh = sec_wall['size']

        # Clamp a point inside the section bounds with a small margin
        pad = 0.05
        def clamp_to_section(x, y):
            x = max(sx + pad, min(x, sx + sw - iw - pad))
            y = max(sy + pad, min(y, sy + sh - ih - pad))
            return x, y

        # 1) Try the clamped target point first
        cx, cy = clamp_to_section(target_x, target_y)
        if self._is_position_safe(cx, cy, iw, ih, exclude_item=item_name):
            return (cx, cy)

        # 2) Try positions in expanding rings around the target
        for radius in [0.3, 0.6, 1.0, 1.5, 2.0]:
            for angle in range(0, 360, 30):  # every 30°
                nx = target_x + radius * np.cos(np.radians(angle))
                ny = target_y + radius * np.sin(np.radians(angle))
                nx, ny = clamp_to_section(nx, ny)
                if self._is_position_safe(nx, ny, iw, ih, exclude_item=item_name):
                    return (nx, ny)

        # 3) Grid scan inside the section (coarse → fine)
        for grid_size in (6, 8):
            for i in range(grid_size):
                for j in range(grid_size):
                    nx = sx + pad + (sw - iw - 2*pad) * (i / (grid_size - 1))
                    ny = sy + pad + (sh - ih - 2*pad) * (j / (grid_size - 1))
                    if self._is_position_safe(nx, ny, iw, ih, exclude_item=item_name):
                        return (nx, ny)

        # 4) Give up and keep current position as last resort
        return tuple(data['position'])
    
    def _repair_item_overlaps(self, max_passes=3):
        """
        Post-optimization repair pass: detect any overlapping items and resolve them
        by moving the smaller-area item to the nearest safe position (within its
        own section if available). Runs up to max_passes passes.
        """
        def overlaps(a_pos, a_size, b_pos, b_size):
            ax, ay = a_pos; aw, ah = a_size
            bx, by = b_pos; bw, bh = b_size
            return not (ax + aw <= bx or ax >= bx + bw or ay + ah <= by or ay >= by + bh)

        items_list = list(self.items.keys())
        if not items_list:
            return

        for _ in range(max_passes):
            any_moved = False

            # Collect all overlapping pairs
            pairs = []
            for i in range(len(items_list)):
                ni = items_list[i]
                pi = self.items[ni]['position']
                si = self.items[ni]['size']
                for j in range(i + 1, len(items_list)):
                    nj = items_list[j]
                    pj = self.items[nj]['position']
                    sj = self.items[nj]['size']
                    if overlaps(pi, si, pj, sj):
                        pairs.append((ni, nj))

            if not pairs:
                break

            # Resolve each pair by moving the smaller-area item
            for a, b in pairs:
                aiw, aih = self.items[a]['size']; a_area = aiw * aih
                biw, bih = self.items[b]['size']; b_area = biw * bih
                move_name = a if a_area <= b_area else b
                cur_x, cur_y = self.items[move_name]['position']
                iw, ih = self.items[move_name]['size']

                # Prefer keeping within section
                nx, ny = self._find_safe_position_in_section(move_name, cur_x, cur_y)
                if (nx, ny) != (cur_x, cur_y) and self._is_position_safe(nx, ny, iw, ih, exclude_item=move_name):
                    self.items[move_name]['position'] = (nx, ny)
                    any_moved = True
                else:
                    # Fallback: global safe search near current
                    gx, gy = self._find_safe_position(cur_x, cur_y, (iw, ih), exclude_item=move_name)
                    if (gx, gy) != (cur_x, cur_y) and self._is_position_safe(gx, gy, iw, ih, exclude_item=move_name):
                        self.items[move_name]['position'] = (gx, gy)
                        any_moved = True

            if not any_moved:
                break

        # Mark metrics dirty after repairs
        self._metrics_dirty = True

    def _is_in_high_traffic_zone(self, position, traffic_zones):
        """Check if a position is in a high-traffic zone."""
        x, y = position
        
        for tx, ty, _ in traffic_zones['high']:
            # Check if position is within 1m of a high-traffic point
            if ((x - tx)**2 + (y - ty)**2)**0.5 < 1.0:
                return True
        
        return False

    def _get_section_center(self, section_name):
        sec_wall = self.walls.get(f"Section_{section_name}")
        if not sec_wall: return None
        x, y = sec_wall['position']
        w, h = sec_wall['size']
        return (x + w/2, y + h/2)

