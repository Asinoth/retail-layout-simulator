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


class GARunMixin:
    """Genetic-algorithm fitness, mutation, repair, run loop, results, apply."""

    def _ga_fitness(self, chromosome, item_names, base_params, mc_days, mc_iters):
        # Elasticities cited from retail_literature.py (Hui 2009 /
        # Hui Inman 2013); midpoint of the literature band when no SA
        # weight is available in this context. Matches the formulation
        # used in viz_optimize.py and viz_whatif.py.
        from retail_literature import (
            ELASTICITY_CONV_BASE as _ECB,
            ELASTICITY_CONV_GAIN_MAX as _ECG,
            ELASTICITY_IMP_BASE  as _EIB,
            ELASTICITY_IMP_GAIN_MAX  as _EIG,
            ELASTICITY_BSK_BASE  as _EBB,
            ELASTICITY_BSK_GAIN_MAX  as _EBG,
            ABANDON_FLOW_COEF, ABANDON_SECTION_COEF,
            ABANDON_BOTTLENECK_COEF, ABANDON_FLOOR_FRAC,
        )
        _conv_e = _ECB + 0.5 * _ECG    # ~0.35 at midpoint
        _imp_e  = _EIB + 0.5 * _EIG    # ~0.50 at midpoint
        _bsk_e  = _EBB + 0.5 * _EBG    # ~0.20 at midpoint

        score, breakdown = self._ga_compute_layout_score(chromosome, item_names, base_params)

        conv_adj = base_params['conversion_rate'] * (1.0 + score * _conv_e)
        conv_adj = min(max(conv_adj, 0.01), 0.99)

        imp_adj = base_params['impulse_rate'] * (1.0 + breakdown['impulse'] * _imp_e)
        imp_adj = min(max(imp_adj, 0.0), 0.99)

        bsk_adj = base_params['avg_basket_size'] * (1.0 + score * _bsk_e)

        abandon_base = base_params.get('abandonment_rate', 0.0)
        abandon_adj = abandon_base * max(ABANDON_FLOOR_FRAC,
            1.0 - breakdown.get('flow', 0) * ABANDON_FLOW_COEF
                - breakdown.get('section_compliance', 0) * ABANDON_SECTION_COEF
                + breakdown.get('bottleneck_penalty', 0) * ABANDON_BOTTLENECK_COEF)
        abandon_adj = min(max(abandon_adj, 0.0), 0.5)

        queue_base = base_params.get('avg_queue_time', 5.0)
        queue_factor = 1.0 + breakdown.get('bottleneck_penalty', 0) * 0.3
        queue_adj = queue_base * queue_factor

        conv_after_abandon = conv_adj * (1.0 - abandon_adj)
        conv_after_abandon = min(max(conv_after_abandon, 0.01), 0.99)

        queue_penalty = 1.0
        if queue_adj > 0:
            queue_penalty = max(0.85, 1.0 - (queue_adj - queue_base) / max(queue_base, 1e-6) * 0.15)

        res = self._mc_engine(
            cph=base_params['customers_per_hour'],
            conv=conv_after_abandon,
            rev_mean=base_params['rev_per_converting_customer'] * queue_penalty,
            rev_std=base_params['rev_std'],
            imp_rate=imp_adj,
            imp_val=base_params['avg_impulse_value'],
            avg_bsk=bsk_adj,
            std_bsk=base_params['std_basket_size'],
            observed_baskets=base_params['basket_sizes_observed'],
            n_days=mc_days, n_iter=mc_iters,
            op_hours=10.0, wknd_mult=1.4, monthly_growth=0.0,
        )
        return res['mean']
    

    def _ga_crossover(self, p1, p2):
        mask = np.random.random(p1.shape[0]) < 0.5
        c1 = np.where(mask[:, None], p1, p2)
        c2 = np.where(mask[:, None], p2, p1)
        alpha = np.random.uniform(0.1, 0.5)
        blend = alpha * p1 + (1.0 - alpha) * p2
        return c1, c2, blend

    def _ga_get_section_bounds(self, item_name):
        """Return (sx, sy, sw, sh) for the item's section on its own floor."""
        data = self._ga_get_item_data(item_name) if hasattr(self, '_ga_get_item_data') else self.items[item_name]
        fid = data.get('floor', getattr(self, 'current_floor', 1))
        cat = data.get('category', '')
        sec_wall = self.floors.get(fid, {}).get('walls', {}).get(f"Section_{cat}")
        if sec_wall:
            return (*sec_wall['position'], *sec_wall['size'])
        return None

    def _ga_get_impulse_max_dist(self):
        """Max allowed distance from checkout for impulse items (25% of store diagonal)."""
        return math.sqrt(self.width ** 2 + self.height ** 2) * 0.25

    def _ga_mutate(self, chrom, item_names, mut_rate):
        mutated = chrom.copy()
        checkout_pos, checkout_floor = self._ga_find_special_center('Checkout') if hasattr(self, '_ga_find_special_center') else (None, None)
        impulse_max = self._ga_get_impulse_max_dist()

        for i, n in enumerate(item_names):
            if np.random.random() >= mut_rate:
                continue
            data = self._ga_get_item_data(n) if hasattr(self, '_ga_get_item_data') else self.items[n]
            w, h = data.get('size', (1.0, 1.0))
            fid = data.get('floor', getattr(self, 'current_floor', 1))
            bounds = self._ga_get_section_bounds(n)

            if bounds:
                sx, sy, sw, sh = bounds
                pad = 0.05
                lo_x, hi_x = sx + pad, sx + sw - w - pad
                lo_y, hi_y = sy + pad, sy + sh - h - pad
                sigma_x = max(sw * 0.12, 0.2)
                sigma_y = max(sh * 0.12, 0.2)
            else:
                lo_x, lo_y = 0.1, 0.1
                hi_x, hi_y = self.width - w - 0.1, self.height - h - 0.1
                sigma_x = max(self.width, self.height) * 0.08
                sigma_y = sigma_x

            if hi_x < lo_x:
                hi_x = lo_x
            if hi_y < lo_y:
                hi_y = lo_y

            mutated[i, 0] = np.clip(mutated[i, 0] + np.random.normal(0, sigma_x), lo_x, hi_x)
            mutated[i, 1] = np.clip(mutated[i, 1] + np.random.normal(0, sigma_y), lo_y, hi_y)

            cat = str(data.get('category', '')).lower()
            if 'impulse' in cat and checkout_pos and fid == checkout_floor:
                cx = mutated[i, 0] + w / 2
                cy = mutated[i, 1] + h / 2
                dist = math.hypot(cx - checkout_pos[0], cy - checkout_pos[1])
                if dist > impulse_max:
                    scale = impulse_max / max(dist, 1e-6) * 0.95
                    mutated[i, 0] = checkout_pos[0] + (cx - checkout_pos[0]) * scale - w / 2
                    mutated[i, 1] = checkout_pos[1] + (cy - checkout_pos[1]) * scale - h / 2
        return self._ga_repair(mutated, item_names)

    def _ga_repair(self, chrom, item_names):
        checkout_pos, checkout_floor = self._ga_find_special_center('Checkout') if hasattr(self, '_ga_find_special_center') else (None, None)
        impulse_max = self._ga_get_impulse_max_dist()

        for i, n in enumerate(item_names):
            data = self._ga_get_item_data(n) if hasattr(self, '_ga_get_item_data') else self.items[n]
            w, h = data.get('size', (1.0, 1.0))
            fid = data.get('floor', getattr(self, 'current_floor', 1))
            bounds = self._ga_get_section_bounds(n)

            if bounds:
                sx, sy, sw, sh = bounds
                pad = 0.05
                lo_x, hi_x = sx + pad, sx + sw - w - pad
                lo_y, hi_y = sy + pad, sy + sh - h - pad
            else:
                lo_x, hi_x = 0.1, self.width - w - 0.1
                lo_y, hi_y = 0.1, self.height - h - 0.1
            if hi_x < lo_x:
                hi_x = lo_x
            if hi_y < lo_y:
                hi_y = lo_y
            chrom[i, 0] = np.clip(chrom[i, 0], lo_x, hi_x)
            chrom[i, 1] = np.clip(chrom[i, 1], lo_y, hi_y)

            cat = str(data.get('category', '')).lower()
            if 'impulse' in cat and checkout_pos and fid == checkout_floor:
                cx = chrom[i, 0] + w / 2
                cy = chrom[i, 1] + h / 2
                dist = math.hypot(cx - checkout_pos[0], cy - checkout_pos[1])
                if dist > impulse_max:
                    scale = impulse_max / max(dist, 1e-6) * 0.95
                    chrom[i, 0] = checkout_pos[0] + (cx - checkout_pos[0]) * scale - w / 2
                    chrom[i, 1] = checkout_pos[1] + (cy - checkout_pos[1]) * scale - h / 2
                    chrom[i, 0] = np.clip(chrom[i, 0], lo_x, hi_x)
                    chrom[i, 1] = np.clip(chrom[i, 1], lo_y, hi_y)
        return chrom

    def _run_ga_optimization(self):
        A = self.customer_simulation.analytics
        if A.get('total_customers', 0) < 5:
            messagebox.showwarning("Insufficient Data",
                                   "Run the simulation first (>= 5 customers).",
                                   parent=self.tk_root)
            return

        item_names = self._ga_get_movable_items()
        if len(item_names) < 2:
            messagebox.showwarning("Too Few Items",
                                   "Need at least 2 movable items for GA.",
                                   parent=self.tk_root)
            return

        base_params = self._extract_simulation_parameters()
        pop_size = max(10, self._ga_pop.get())
        n_gens = max(5, self._ga_gens.get())
        mut_rate = max(0.01, self._ga_mut.get() / 100.0)
        elite_frac = max(0.05, self._ga_elite.get() / 100.0)
        mc_iters = max(100, self._ga_mc_iters.get())
        mc_days = max(1, self._ga_mc_days.get())


        n_elite = max(1, int(pop_size * elite_frac))

        self._ga_progress.set("Initializing population...")
        self.tk_root.update_idletasks()

        current = self._ga_encode(item_names)
        population = [current.copy()]
        for _ in range(pop_size - 1):
            noisy = current.copy()
            for i, n in enumerate(item_names):
                data = self._ga_get_item_data(n) if hasattr(self, '_ga_get_item_data') else self.items[n]
                w, h = data.get('size', (1.0, 1.0))
                bounds = self._ga_get_section_bounds(n)
                if bounds:
                    sx, sy, sw, sh = bounds
                    pad = 0.05
                    lo_x, hi_x = sx + pad, sx + sw - w - pad
                    lo_y, hi_y = sy + pad, sy + sh - h - pad
                    sigma_x = max(sw * 0.15, 0.3)
                    sigma_y = max(sh * 0.15, 0.3)
                else:
                    lo_x, lo_y = 0.1, 0.1
                    hi_x = self.width - w - 0.1
                    hi_y = self.height - h - 0.1
                    sigma_x = self.width * 0.15
                    sigma_y = self.height * 0.15
                if hi_x < lo_x:
                    hi_x = lo_x
                if hi_y < lo_y:
                    hi_y = lo_y
                noisy[i, 0] = np.clip(
                    current[i, 0] + np.random.normal(0, sigma_x), lo_x, hi_x)
                noisy[i, 1] = np.clip(
                    current[i, 1] + np.random.normal(0, sigma_y), lo_y, hi_y)
            noisy = self._ga_repair(noisy, item_names)
            population.append(noisy)

        history_best = []
        history_avg = []
        history_worst = []
        gen_scores_all = []

        for gen in range(n_gens):
            self._ga_progress.set(f"Gen {gen + 1}/{n_gens} — evaluating...")
            self.tk_root.update_idletasks()

            fitness = np.array([
                self._ga_fitness(p, item_names, base_params, mc_days, mc_iters)
                for p in population
            ])

            rank = np.argsort(fitness)[::-1]
            population = [population[r] for r in rank]
            fitness = fitness[rank]

            history_best.append(fitness[0])
            history_avg.append(fitness.mean())
            history_worst.append(fitness[-1])
            gen_scores_all.append(fitness.copy())

            self._ga_progress.set(
                f"Gen {gen + 1}/{n_gens} — best=${fitness[0]:,.0f}  avg=${fitness.mean():,.0f}"
            )
            self.tk_root.update_idletasks()

            new_pop = [p.copy() for p in population[:n_elite]]

            while len(new_pop) < pop_size:
                t_size = min(5, pop_size)
                t_idx = np.random.choice(pop_size, t_size, replace=False)
                t_fit = fitness[t_idx]
                p1 = population[t_idx[np.argmax(t_fit)]]

                t_idx2 = np.random.choice(pop_size, t_size, replace=False)
                t_fit2 = fitness[t_idx2]
                p2 = population[t_idx2[np.argmax(t_fit2)]]

                c1, c2, blend = self._ga_crossover(p1, p2)
                for child in [c1, c2, blend]:
                    child = self._ga_mutate(child, item_names, mut_rate)
                    child = self._ga_repair(child, item_names)
                    new_pop.append(child)
                    if len(new_pop) >= pop_size:
                        break

            population = new_pop[:pop_size]

        self._ga_progress.set("Final evaluation...")
        self.tk_root.update_idletasks()
        final_fitness = np.array([
            self._ga_fitness(p, item_names, base_params, mc_days, mc_iters)
            for p in population
        ])
        best_idx = np.argmax(final_fitness)
        best_chrom = population[best_idx]

        self._ga_best_chromosome = best_chrom
        self._ga_best_layout = self._ga_decode(best_chrom, item_names)
        self._ga_item_names = item_names

        best_score, best_breakdown = self._ga_compute_layout_score(
            best_chrom, item_names, base_params
        )
        current_chrom = self._ga_encode(item_names)
        current_fit = self._ga_fitness(current_chrom, item_names, base_params, mc_days, mc_iters)

        self._display_ga_results(
            item_names, base_params, best_chrom, best_breakdown,
            final_fitness[best_idx], current_fit,
            history_best, history_avg, history_worst,
            gen_scores_all, n_gens, pop_size, mc_days, mc_iters
        )
        self._ga_progress.set("Done. Click 'Apply Best Layout' to use it.")

    def _display_ga_results(self, item_names, base_params, best_chrom, breakdown,
                            best_rev, current_rev, h_best, h_avg, h_worst,
                            gen_scores, n_gens, pop_size, mc_days, mc_iters):
        improvement = (best_rev - current_rev) / max(current_rev, 1e-6) * 100

        lines = [
            "=" * 50,
            "   GENETIC ALGORITHM OPTIMIZATION RESULTS",
            "=" * 50,
            "",
            f"  Population:        {pop_size}",
            f"  Generations:       {n_gens}",
            f"  MC iters/eval:     {mc_iters}",
            f"  MC projection:     {mc_days} days",
            f"  Movable items:     {len(item_names)}",
            "",
            "REVENUE COMPARISON",
            "-" * 50,
            f"  Current layout:    ${current_rev:>12,.2f}",
            f"  GA-optimized:      ${best_rev:>12,.2f}",
            f"  Improvement:       {improvement:>+11.1f}%",
            f"  Daily delta:       ${(best_rev - current_rev) / mc_days:>+12,.2f}",
            f"  Annual projection: ${(best_rev - current_rev) / mc_days * 365:>+12,.0f}",
            "",
            "LAYOUT QUALITY BREAKDOWN",
            "-" * 50,
            f"  Traffic alignment:   {breakdown['traffic']:.4f}",
            f"  Cross-merch:         {breakdown['cross_merch']:.4f}",
            f"  Impulse placement:   {breakdown['impulse']:.4f}",
            f"  Flow efficiency:     {breakdown['flow']:.4f}",
            f"  Revenue placement:   {breakdown['revenue_placement']:.4f}",
            f"  Dwell alignment:     {breakdown['dwell_alignment']:.4f}",
            f"  Accessibility:       {breakdown['accessibility']:.4f}",
            f"  Section compliance:  {breakdown['section_compliance']:.4f}",
            f"  Bottleneck penalty:  {breakdown['bottleneck_penalty']:.4f}",
            f"  Overlap penalty:     {breakdown['overlap_penalty']:.4f}",
            f"  Composite score:     {breakdown['composite']:.4f}",
            "",
            "ITEM MOVEMENTS (current -> optimized)",
            "-" * 50,
        ]

        moved = 0
        item_meta = {name: (self._ga_get_item_data(name) if hasattr(self, '_ga_get_item_data') else self.items[name])
                     for name in item_names}
        for i, name in enumerate(item_names):
            old = item_meta[name].get('position', (0.0, 0.0))
            new = (best_chrom[i, 0], best_chrom[i, 1])
            dist = np.sqrt((new[0] - old[0]) ** 2 + (new[1] - old[1]) ** 2)
            if dist > 0.3:
                moved += 1
                fid = item_meta[name].get('floor', self.current_floor)
                label = f"F{fid}:{item_meta[name].get('source_name', name)}"
                lines.append(
                    f"  {label[:24]:<24s}  ({old[0]:.1f},{old[1]:.1f})"
                    f" -> ({new[0]:.1f},{new[1]:.1f})  d={dist:.1f}m"
                )

        lines.append(f"\n  Items moved: {moved}/{len(item_names)}")

        self._ga_text.config(state=tk.NORMAL)
        self._ga_text.delete('1.0', tk.END)
        self._ga_text.insert(tk.END, "\n".join(lines))
        self._ga_text.config(state=tk.DISABLED)

        self._ga_fig.clear()
        wc = 'white'

        ax1 = self._ga_fig.add_subplot(2, 2, 1)
        gens = range(1, len(h_best) + 1)
        ax1.plot(gens, h_best, color='#2ecc71', linewidth=2, label='Best')
        ax1.plot(gens, h_avg, color='#3498db', linewidth=1.5, label='Average')
        ax1.plot(gens, h_worst, color='#e74c3c', linewidth=1, alpha=0.6, label='Worst')
        ax1.fill_between(gens, h_worst, h_best, alpha=0.1, color='#3498db')
        ax1.set_xlabel("Generation", color=wc, fontsize=9)
        ax1.set_ylabel("MC Revenue ($)", color=wc, fontsize=9)
        ax1.set_title("Fitness Convergence", color=wc, fontsize=10)
        ax1.legend(fontsize=7)
        ax1.set_facecolor('#001a33')
        ax1.tick_params(colors=wc, labelsize=7)
        ax1.grid(True, alpha=0.15, color='white')

        ax2 = self._ga_fig.add_subplot(2, 2, 2)
        categories = [k for k in breakdown.keys() if k != 'composite']
        vals = [breakdown[c] for c in categories]
        n_bars = len(categories)
        colors_bar = ['#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#e74c3c',
                       '#1abc9c', '#e67e22', '#2980b9', '#c0392b', '#8e44ad'][:n_bars]
        short_cats = [c.replace('_', '\n') for c in categories]
        bars = ax2.bar(short_cats, vals, color=colors_bar[:len(categories)])
        ax2.axhline(0, color='white', linewidth=0.5)
        ax2.set_title("Layout Quality Breakdown", color=wc, fontsize=10)
        ax2.set_facecolor('#001a33')
        ax2.tick_params(colors=wc, labelsize=7)
        for bar, val in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f'{val:.3f}', ha='center', va='bottom', fontsize=7, color=wc)

        ax3 = self._ga_fig.add_subplot(2, 2, 3)
        display_floor = getattr(self, 'current_floor', 1)
        for i, name in enumerate(item_names):
            meta = item_meta.get(name) or (self._ga_get_item_data(name) if hasattr(self, '_ga_get_item_data') else self.items[name])
            if meta.get('floor', display_floor) != display_floor:
                continue
            old = meta.get('position', (0.0, 0.0))
            s = meta.get('size', (1.0, 1.0))
            new = (best_chrom[i, 0], best_chrom[i, 1])
            dist = np.sqrt((new[0] - old[0]) ** 2 + (new[1] - old[1]) ** 2)
            ax3.add_patch(plt.Rectangle(old, s[0], s[1],
                                         fill=False, edgecolor='#e74c3c',
                                         linewidth=0.8, linestyle='--', alpha=0.5))
            ax3.add_patch(plt.Rectangle(new, s[0], s[1],
                                         fill=False, edgecolor='#2ecc71',
                                         linewidth=1.2))
            if dist > 0.3:
                ax3.annotate('', xy=(new[0] + s[0] / 2, new[1] + s[1] / 2),
                             xytext=(old[0] + s[0] / 2, old[1] + s[1] / 2),
                             arrowprops=dict(arrowstyle='->', color='#f39c12',
                                             lw=0.8, alpha=0.7))
        for wname, wd in self.floors.get(display_floor, {}).get('walls', self.walls).items():
            if not wname.startswith('Section_'):
                wp = wd['position']
                ws = wd['size']
                ax3.add_patch(plt.Rectangle(wp, ws[0], ws[1],
                                             fill=True, facecolor='#555555',
                                             edgecolor='white', linewidth=0.5, alpha=0.6))
        ax3.set_xlim(-0.5, self.width + 0.5)
        ax3.set_ylim(-0.5, self.height + 0.5)
        ax3.set_aspect('equal')
        ax3.set_title("Movement Map (red=old, green=new)", color=wc, fontsize=10)
        ax3.set_facecolor('#001a33')
        ax3.tick_params(colors=wc, labelsize=7)

        ax4 = self._ga_fig.add_subplot(2, 2, 4)
        if gen_scores:
            last_gen = gen_scores[-1]
            ax4.hist(last_gen, bins=min(20, pop_size // 2), color='#3498db',
                     alpha=0.7, edgecolor='white', linewidth=0.5)
            ax4.axvline(best_rev, color='#2ecc71', linewidth=2,
                        linestyle='--', label=f'Best ${best_rev:,.0f}')
            ax4.axvline(current_rev, color='#e74c3c', linewidth=2,
                        linestyle='--', label=f'Current ${current_rev:,.0f}')
        ax4.set_xlabel("MC Revenue ($)", color=wc, fontsize=9)
        ax4.set_ylabel("Count", color=wc, fontsize=9)
        ax4.set_title("Final Generation Distribution", color=wc, fontsize=10)
        ax4.legend(fontsize=7)
        ax4.set_facecolor('#001a33')
        ax4.tick_params(colors=wc, labelsize=7)
        ax4.grid(True, alpha=0.15, color='white')

        self._ga_fig.tight_layout(pad=2.0)
        self._ga_canvas_fig.draw_idle()

    def _apply_ga_best(self):
        if self._ga_best_layout is None:
            messagebox.showinfo("No Result", "Run the GA optimizer first.",
                                parent=self.tk_root)
            return

        old_positions = {}
        touched_floors = set()
        for key, (x, y) in self._ga_best_layout.items():
            fid, source = self._ga_resolve_item_key(key) if hasattr(self, '_ga_resolve_item_key') else (self.current_floor, key)
            floor_items = self.floors.get(fid, {}).get('items', {})
            if source not in floor_items:
                continue
            old_positions[(fid, source)] = tuple(floor_items[source]['position'])
            w, h = floor_items[source]['size']
            safe_x = max(0.1, min(x, self.width - w - 0.1))
            safe_y = max(0.1, min(y, self.height - h - 0.1))
            floor_items[source]['position'] = (safe_x, safe_y)
            touched_floors.add(fid)

        prev_floor = self.current_floor
        try:
            for fid in touched_floors or {self.current_floor}:
                self.current_floor = fid
                self._repair_item_overlaps()
        finally:
            self.current_floor = prev_floor
        self.redraw()

        moved = 0
        for (fid, source), old in old_positions.items():
            new_pos = self.floors.get(fid, {}).get('items', {}).get(source, {}).get('position', old)
            if np.sqrt((new_pos[0] - old[0]) ** 2 + (new_pos[1] - old[1]) ** 2) > 0.3:
                moved += 1
        messagebox.showinfo(
            "Layout Applied",
            f"GA-optimized layout applied.\n{moved} items repositioned.",
            parent=self.tk_root
        )

    # ═════════════════════════════════════════════════════════════
    #  WHAT-IF SCENARIO COMPARISON (A/B Testing) TAB
    # ═════════════════════════════════════════════════════════════

