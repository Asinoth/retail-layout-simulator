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

from sim_calibration import _calib


class WhatIfMixin:
    """What-if (A/B) layout comparison tab."""

    def _create_whatif_tab(self, notebook):
        tab = tk.Frame(notebook, bg='#002747')
        notebook.add(tab, text="What-If A/B Test")

        ctrl = tk.Frame(tab, bg='#002747')
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=10, pady=8)

        lbl_kw = dict(fg='white', bg='#002747', font=('Arial', 10))
        ent_kw = dict(width=7)

        tk.Button(
            ctrl, text="Save Snapshot A", command=lambda: self._save_snapshot('A'),
            bg='#e74c3c', fg='white', font=('Arial', 10, 'bold'),
            relief=tk.RAISED, bd=2
        ).grid(row=0, column=0, padx=6)

        tk.Button(
            ctrl, text="Save Snapshot B", command=lambda: self._save_snapshot('B'),
            bg='#3498db', fg='white', font=('Arial', 10, 'bold'),
            relief=tk.RAISED, bd=2
        ).grid(row=0, column=1, padx=6)

        tk.Button(
            ctrl, text="Save GA Result as B",
            command=self._save_ga_as_b,
            bg='#8e44ad', fg='white', font=('Arial', 10, 'bold'),
            relief=tk.RAISED, bd=2
        ).grid(row=0, column=2, padx=6)

        self._ab_snap_a_lbl = tk.StringVar(value="A: (empty)")
        self._ab_snap_b_lbl = tk.StringVar(value="B: (empty)")
        tk.Label(ctrl, textvariable=self._ab_snap_a_lbl, fg='#e74c3c',
                 bg='#002747', font=('Arial', 9)).grid(row=0, column=3, padx=6)
        tk.Label(ctrl, textvariable=self._ab_snap_b_lbl, fg='#3498db',
                 bg='#002747', font=('Arial', 9)).grid(row=0, column=4, padx=6)

        tk.Label(ctrl, text="MC iters:", **lbl_kw).grid(row=0, column=5, padx=3)
        self._ab_iters = tk.IntVar(value=3000)
        tk.Entry(ctrl, textvariable=self._ab_iters, **ent_kw).grid(row=0, column=6, padx=3)

        tk.Label(ctrl, text="MC days:", **lbl_kw).grid(row=0, column=7, padx=3)
        self._ab_days = tk.IntVar(value=30)
        tk.Entry(ctrl, textvariable=self._ab_days, **ent_kw).grid(row=0, column=8, padx=3)

        # No significance level: the comparison reports the paired lift with
        # its Monte Carlo interval, and no test decides anything
        # (layout_comparison).
        self._ab_progress = tk.StringVar(value="")
        tk.Label(ctrl, textvariable=self._ab_progress, fg='#4ECDC4',
                 bg='#002747', font=('Arial', 10, 'bold')).grid(row=0, column=10, padx=8)

        tk.Button(
            ctrl, text="Run A/B Comparison", command=self._run_ab_comparison,
            bg='#27ae60', fg='white', font=('Arial', 11, 'bold'),
            relief=tk.RAISED, bd=3
        ).grid(row=0, column=9, padx=10)

        body = tk.Frame(tab, bg='#002747')
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=3)
        body.rowconfigure(0, weight=1)

        self._ab_text = tk.Text(
            body, bg='#001a33', fg='white', font=('Courier', 10),
            wrap=tk.WORD, state=tk.DISABLED, width=52
        )
        self._ab_text.grid(row=0, column=0, sticky='nsew', padx=(0, 5))

        chart_frame = tk.Frame(body, bg='#002747')
        chart_frame.grid(row=0, column=1, sticky='nsew')

        self._ab_fig = Figure(figsize=(12, 8), dpi=100)
        self._ab_fig.patch.set_facecolor('#002747')
        self._ab_canvas_fig = FigureCanvasTkAgg(self._ab_fig, master=chart_frame)
        self._ab_canvas_fig.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._snapshots = {}

    def _snapshot_layout(self):
        A = self.customer_simulation.analytics
        # A trajectory-only dataset fills the calibration block with spatial
        # keys but no arrival or conversion rate, so it cannot stand in for
        # observed customers.
        cal = _calib(A)
        has_tx = bool(cal.get('arrivals_per_hour')) and 'conversion_rate' in cal
        # The same test as the Optimize entry gate. total_customers counts
        # arrivals and is not reduced when the store is cleared, so once the
        # pipeline has passed that gate every snapshot it takes carries
        # parameters.
        observed = int(A.get('total_customers', 0) or 0)
        return {
            'items': copy.deepcopy(self.items),
            'walls': copy.deepcopy(self.walls),
            'prices': copy.deepcopy(self.prices),
            'floors': copy.deepcopy(getattr(self, 'floors', {})),
            'connectors': copy.deepcopy(getattr(self, 'connectors', {})),
            'current_floor': getattr(self, 'current_floor', 1),
            'door_position': copy.deepcopy(getattr(self, 'door_position', None)),
            'door_side': getattr(self, 'door_side', None),
            'params': self._extract_simulation_parameters()
                      if (has_tx or observed >= 5)
                      else None,
            'timestamp': time.time(),
        }

    def _save_snapshot(self, label):
        snap = self._snapshot_layout()
        if snap['params'] is None:
            messagebox.showwarning("Insufficient Data",
                                   "Run the simulation first (>= 5 customers).",
                                   parent=self.tk_root)
            return
        self._snapshots[label] = snap
        n = len(snap['items'])
        getattr(self, f'_ab_snap_{label.lower()}_lbl').set(
            f"{label}: {n} items @ {time.strftime('%H:%M:%S')}"
        )

    def _save_ga_as_b(self):
        if not hasattr(self, '_ga_best_layout') or self._ga_best_layout is None:
            messagebox.showinfo("No GA Result",
                                "Run the GA optimizer first.",
                                parent=self.tk_root)
            return
        snap = self._snapshot_layout()
        if snap['params'] is None:
            messagebox.showwarning("Insufficient Data",
                                   "Run the simulation first.",
                                   parent=self.tk_root)
            return
        ga_floors = copy.deepcopy(getattr(self, 'floors', {}))
        for key, (x, y) in self._ga_best_layout.items():
            fid, source = self._ga_resolve_item_key(key) if hasattr(self, '_ga_resolve_item_key') else (self.current_floor, key)
            items = ga_floors.get(fid, {}).get('items', {})
            if source in items:
                w, h = items[source]['size']
                items[source]['position'] = (
                    max(0.1, min(x, self.width - w - 0.1)),
                    max(0.1, min(y, self.height - h - 0.1))
                )
        snap['floors'] = ga_floors
        snap['items'] = copy.deepcopy(ga_floors.get(self.current_floor, {}).get('items', {}))
        self._snapshots['B'] = snap
        n = sum(len(f.get('items', {})) for f in ga_floors.values())
        self._ab_snap_b_lbl.set(f"B: GA layout ({n} items) @ {time.strftime('%H:%M:%S')}")

    def _ab_layout_score(self, snap, with_breakdown=False):
        floors_backup = copy.deepcopy(getattr(self, 'floors', {}))
        connectors_backup = copy.deepcopy(getattr(self, 'connectors', {}))
        current_floor_backup = getattr(self, 'current_floor', 1)
        try:
            if snap.get('floors'):
                self.floors = copy.deepcopy(snap['floors'])
                self.connectors = copy.deepcopy(snap.get('connectors', self.connectors))
                self.current_floor = int(snap.get('current_floor', current_floor_backup))
            else:
                self.items = copy.deepcopy(snap['items'])
                self.walls = copy.deepcopy(snap.get('walls', self.walls))

            item_names = self._ga_get_movable_items() if hasattr(self, '_ga_get_movable_items') else [n for n in self.items if n not in {'Entrance', 'Checkout', 'WC'}]
            if not item_names:
                return (0.0, {}) if with_breakdown else 0.0
            positions = self._ga_encode(item_names) if hasattr(self, '_ga_encode') else np.array([list(self.items[n]['position']) for n in item_names])
            score, bkd = self._ga_compute_layout_score(positions, item_names, snap['params'])
            return (score, bkd) if with_breakdown else score
        finally:
            self.floors = floors_backup
            self.connectors = connectors_backup
            self.current_floor = current_floor_backup
            # A running simulation may have rebuilt its obstacle grid or
            # zone table from the snapshot while it was swapped in; items
            # are obstacles, so have both rebuilt from the restored layout.
            try:
                self.customer_simulation.geometry_dirty = True
                self.customer_simulation.invalidate_zones_cache()
            except Exception:
                pass
            if hasattr(self, '_ga_item_catalog'):
                try:
                    self._ga_build_item_catalog(self._ga_accessible_floors())
                except Exception:
                    pass

    def _whatif_comparison(self, n_iter, n_days, seed):
        """The What-If comparison of snapshots A and B, without any Tk.

        Both layouts are differenced against the same anchor: the layout on
        the floor as the comparison starts, which reproduces the calibrated
        conversion, basket and spend, as it does in the Optimize pipeline
        and the GA tab. The anchor and both layouts are scored here, before
        any Monte Carlo run, so a simulation still adding to the heat map
        cannot score them under different analytics. A floor with no
        movable items scores zero, so its anchor is zero -- stated
        explicitly, since the objective refuses parameters without one.

        Both layouts are projected under snapshot A's calibrated inputs, so
        they differ by the layout alone even if the simulation ran between
        the two snapshots, through the same score -> driver transform as
        the Optimize pipeline and the headless experiments (the band
        midpoints; ``_opt_layout_drivers``), and compared under common
        random numbers (``layout_comparison.paired_comparison``). Both
        arms run ``mc_engine`` itself, so a day's spend is the engine's
        law (retail_literature MC_SPEND_LAW, the moment-matched lognormal
        of ``sim_calibration.lognormal_spend_total``), not a zero-floored
        normal.

        Returns the comparison with each arm's score and drivers added."""
        from layout_objective import make_anchor
        from layout_comparison import paired_comparison

        params = self._snapshots['A']['params']
        names_now = self._ga_get_movable_items()
        anchor = (self._ga_score_anchor(names_now, params,
                                        source='floor_at_whatif_start')
                  if names_now
                  else make_anchor(0.0, {}, source='empty_floor_at_whatif_start'))
        ab_scores = {label: self._ab_layout_score(self._snapshots[label],
                                                  with_breakdown=True)
                     for label in ('A', 'B')}
        drivers = {label: self._opt_layout_drivers(*ab_scores[label], params,
                                                   anchor=anchor)
                   for label in ('A', 'B')}
        kws = {label: self._opt_layout_mc_kwargs(*ab_scores[label], params,
                                                 n_days=n_days, n_iter=n_iter,
                                                 anchor=anchor)
               for label in ('A', 'B')}
        result = paired_comparison(kws['A'], kws['B'], seed)
        result['scores'] = {label: float(ab_scores[label][0])
                            for label in ('A', 'B')}
        result['drivers'] = drivers
        result['anchor'] = anchor
        return result

    def _run_ab_comparison(self):
        if 'A' not in self._snapshots or 'B' not in self._snapshots:
            messagebox.showwarning("Missing Snapshots",
                                   "Save both Snapshot A and Snapshot B first.",
                                   parent=self.tk_root)
            return

        n_iter = max(500, self._ab_iters.get())
        n_days = max(1, self._ab_days.get())

        self._ab_progress.set("Running paired Monte Carlo...")
        self.tk_root.update_idletasks()
        # The seed is printed with the result, so a comparison can be
        # reproduced exactly.
        seed = int(np.random.randint(0, 2**31 - 1))
        result = self._whatif_comparison(n_iter, n_days, seed)
        self._display_ab_results(result)
        self._ab_progress.set("Done.")

    def _display_ab_results(self, result):
        from layout_comparison import comparison_lines
        ra = result['A']
        rb = result['B']
        p = result['paired']
        n_days = result['n_days']
        da, db = result['drivers']['A'], result['drivers']['B']
        winner = result['direction']
        lvl = int(round(p['level'] * 100))

        lines = [
            "=" * 52,
            "   WHAT-IF A/B COMPARISON REPORT",
            "=" * 52,
            "",
            f"  MC iterations:     {result['n_iter']:,} paired draws",
            f"  Projection:        {n_days} days",
            "  Elasticities:      midpoints of the cited bands",
            "                     (as in the Optimize pipeline and",
            "                     the headless experiments)",
            "  Anchor:            the layout on the floor at the start;",
            "                     it reproduces the calibrated inputs",
            "  Inputs:            snapshot A's calibration, for both",
            f"  Random numbers:    common to A and B (seed {result['seed']})",
            "",
            "REVENUE SUMMARY ({}-day total)".format(n_days),
            "-" * 52,
            f"  {'Metric':<22s} {'Layout A':>12s} {'Layout B':>12s}",
            f"  {'Mean':.<22s} ${ra['mean']:>11,.2f} ${rb['mean']:>11,.2f}",
            f"  {'Exact mean (model)':.<22s} ${ra['exact_mean']:>11,.2f} "
            f"${rb['exact_mean']:>11,.2f}",
            f"  {'Median':.<22s} ${ra['median']:>11,.2f} ${rb['median']:>11,.2f}",
            f"  {'Std Dev':.<22s} ${ra['std']:>11,.2f} ${rb['std']:>11,.2f}",
            f"  {'5th pct':.<22s} ${ra['p5']:>11,.2f} ${rb['p5']:>11,.2f}",
            f"  {'95th pct':.<22s} ${ra['p95']:>11,.2f} ${rb['p95']:>11,.2f}",
            "  (the spread of simulated outcomes, not the",
            "   uncertainty of the lift)",
            "",
            "PAIRED LIFT",
            "-" * 52,
        ]
        lines += comparison_lines(result, 'A', 'B')
        lines += [
            f"  Daily delta:       ${p['lift_mean'] / n_days:>+12,.2f}",
            f"  Annual projection: ${p['lift_mean'] / n_days * 365:>+12,.0f}",
            "",
            "LAYOUT QUALITY SCORES",
            "-" * 52,
            f"  Layout A score:    {result['scores']['A']:.4f}",
            f"  Layout B score:    {result['scores']['B']:.4f}",
            f"  Anchor score:      {result['anchor']['score']:.4f}",
            f"  Conv rate A:       {da['conv']:.4f}",
            f"  Conv rate B:       {db['conv']:.4f}",
            f"  Impulse rate A:    {da['imp_rate']:.4f}",
            f"  Impulse rate B:    {db['imp_rate']:.4f}",
            f"  Spend mult. A:     {da['rev_mult']:.4f}",
            f"  Spend mult. B:     {db['rev_mult']:.4f}",
        ]

        self._ab_text.config(state=tk.NORMAL)
        self._ab_text.delete('1.0', tk.END)
        self._ab_text.insert(tk.END, "\n".join(lines))
        self._ab_text.config(state=tk.DISABLED)

        self._ab_fig.clear()
        wc = 'white'

        # Spread of each layout's simulated totals. These overlap heavily
        # whatever the lift, because a month's revenue varies far more than
        # the layouts differ; the lift is read off the paired panel below.
        ax1 = self._ab_fig.add_subplot(2, 2, 1)
        bins = np.linspace(
            min(ra['totals'].min(), rb['totals'].min()),
            max(ra['totals'].max(), rb['totals'].max()),
            50)
        ax1.hist(ra['totals'], bins=bins, alpha=0.6, color='#e74c3c',
                 label=f"A (mean=${ra['mean']:,.0f})", edgecolor='white', linewidth=0.3)
        ax1.hist(rb['totals'], bins=bins, alpha=0.6, color='#3498db',
                 label=f"B (mean=${rb['mean']:,.0f})", edgecolor='white',
                 linewidth=0.3, hatch='//')
        ax1.axvline(ra['mean'], color='#e74c3c', linewidth=2, linestyle='--')
        ax1.axvline(rb['mean'], color='#3498db', linewidth=2, linestyle=':')
        ax1.set_xlabel(f"Simulated {n_days}-day total ($)", color=wc, fontsize=9)
        ax1.set_ylabel("Draws", color=wc, fontsize=9)
        ax1.set_title("Spread of simulated outcomes", color=wc, fontsize=10)
        ax1.legend(fontsize=7)
        ax1.set_facecolor('#001a33')
        ax1.tick_params(colors=wc, labelsize=7)
        ax1.grid(True, alpha=0.15, color='white')

        ax2 = self._ab_fig.add_subplot(2, 2, 2)
        days = np.arange(1, n_days + 1)
        ax2.plot(days, ra['daily_means'], color='#e74c3c', linewidth=1.5,
                 marker='o', markersize=2.5, label='A daily')
        ax2.plot(days, rb['daily_means'], color='#3498db', linewidth=1.5,
                 linestyle='--', marker='s', markersize=2.5, label='B daily')
        ax2.set_xlabel("Day", color=wc, fontsize=9)
        ax2.set_ylabel("Mean daily revenue ($)", color=wc, fontsize=9)
        ax2.set_title("Mean daily revenue", color=wc, fontsize=10)
        ax2.legend(fontsize=7, loc='upper left')
        ax2.set_facecolor('#001a33')
        ax2.tick_params(colors=wc, labelsize=7)
        ax2.grid(True, alpha=0.15, color='white')

        # The paired differences: what the lift and its interval come from.
        ax3 = self._ab_fig.add_subplot(2, 2, 3)
        diff = p['diffs']
        lift_color = {'B': '#2ecc71', 'A': '#e74c3c'}.get(winner, '#95a5a6')
        ax3.hist(diff, bins=50, color=lift_color,
                 alpha=0.7, edgecolor='white', linewidth=0.3)
        ax3.axvline(0, color='white', linewidth=1.5, linestyle='--')
        ax3.axvline(p['lift_mean'], color='#f39c12', linewidth=2,
                    label=f"Mean: ${p['lift_mean']:+,.0f}")
        ax3.axvspan(*p['lift_ci'], color='#f39c12', alpha=0.25,
                    label=f"{lvl}% interval of the mean")
        ax3.axvline(result['exact']['lift'], color='white', linewidth=1.2,
                    linestyle=':', label=f"Exact: ${result['exact']['lift']:+,.0f}")
        ax3.set_xlabel("Paired difference B - A ($)", color=wc, fontsize=9)
        ax3.set_ylabel("Paired draws", color=wc, fontsize=9)
        ax3.set_title("Paired differences (common random numbers)",
                      color=wc, fontsize=10)
        ax3.legend(fontsize=7)
        ax3.set_facecolor('#001a33')
        ax3.tick_params(colors=wc, labelsize=7)
        ax3.grid(True, alpha=0.15, color='white')

        ax4 = self._ab_fig.add_subplot(2, 2, 4)
        metrics = ['Mean\nRevenue', 'Median\nRevenue', 'P5', 'P95',
                   'Std Dev']
        a_vals = [ra['mean'], ra['median'], ra['p5'], ra['p95'], ra['std']]
        b_vals = [rb['mean'], rb['median'], rb['p5'], rb['p95'], rb['std']]
        x = np.arange(len(metrics))
        w = 0.35
        ax4.bar(x - w / 2, a_vals, w, color='#e74c3c', alpha=0.8, label='Layout A')
        ax4.bar(x + w / 2, b_vals, w, color='#3498db', alpha=0.8,
                label='Layout B', hatch='//')
        ax4.set_xticks(x)
        ax4.set_xticklabels(metrics, fontsize=7, color=wc)
        ax4.set_ylabel("$ Revenue", color=wc, fontsize=9)
        ax4.set_title("Side-by-Side Metrics", color=wc, fontsize=10)
        ax4.legend(fontsize=7)
        ax4.set_facecolor('#001a33')
        ax4.tick_params(colors=wc, labelsize=7)
        ax4.grid(True, alpha=0.15, color='white', axis='y')

        self._ab_fig.tight_layout(pad=2.0)
        self._ab_canvas_fig.draw_idle()
