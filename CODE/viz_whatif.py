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

        tk.Label(ctrl, text="Significance:", **lbl_kw).grid(row=0, column=9, padx=3)
        self._ab_alpha = tk.DoubleVar(value=0.05)
        tk.Entry(ctrl, textvariable=self._ab_alpha, **ent_kw).grid(row=0, column=10, padx=3)

        self._ab_progress = tk.StringVar(value="")
        tk.Label(ctrl, textvariable=self._ab_progress, fg='#4ECDC4',
                 bg='#002747', font=('Arial', 10, 'bold')).grid(row=0, column=12, padx=8)

        tk.Button(
            ctrl, text="Run A/B Comparison", command=self._run_ab_comparison,
            bg='#27ae60', fg='white', font=('Arial', 11, 'bold'),
            relief=tk.RAISED, bd=3
        ).grid(row=0, column=11, padx=10)

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

    def _run_ab_comparison(self):
        if 'A' not in self._snapshots or 'B' not in self._snapshots:
            messagebox.showwarning("Missing Snapshots",
                                   "Save both Snapshot A and Snapshot B first.",
                                   parent=self.tk_root)
            return

        n_iter = max(500, self._ab_iters.get())
        n_days = max(1, self._ab_days.get())
        alpha = max(0.001, min(self._ab_alpha.get(), 0.5))

        from retail_literature import DEFAULT_WEEKEND_MULTIPLIER as _WKND

        # Both layouts are drawn from the same random stream, restarted per
        # label, so two identical layouts compare exactly equal instead of
        # differing by Monte Carlo noise. The binomial draws consume a
        # parameter-dependent number of variates, so once the layouts differ
        # the two streams drift apart and the draws are close to
        # independent; the interval below is valid either way.
        ab_seed = int(np.random.randint(0, 2**31 - 1))

        results = {}
        for label in ['A', 'B']:
            self._ab_progress.set(f"Running MC for layout {label}...")
            self.tk_root.update_idletasks()

            snap = self._snapshots[label]
            p = snap['params']
            score, score_bkd = self._ab_layout_score(snap, with_breakdown=True)

            # The same score -> parameter transform as the optimize pipeline
            # and the GA fitness: conversion, the impulse component of the
            # score, layout-induced abandonment, and basket size as a
            # revenue-per-converter multiplier. Evaluated at the midpoint of
            # each cited elasticity band (the pipeline uses its
            # sensitivity-derived weights instead), so a layout pair gets
            # comparable lifts here and in the Optimize report.
            mid_weights = {'Conversion rate': 0.5, 'Impulse rate': 0.5,
                           'Basket size': 0.5}
            conv_adj, imp_adj, _bsk_adj, rev_mult = self._opt_layout_revenue_params(
                score, score_bkd, p, mid_weights, p['conversion_rate'])
            rev_mean = p['rev_per_converting_customer'] * rev_mult
            rev_sd   = max(p['rev_std'] * rev_mult, 0.01)

            day_mult = np.ones(7)
            day_mult[5] = _WKND
            day_mult[6] = _WKND

            rs = np.random.RandomState(ab_seed)
            daily_rev = np.zeros((n_iter, n_days))
            for d in range(n_days):
                dow = d % 7
                lam = max(p['customers_per_hour'] * 10.0 * day_mult[dow], 0.1)
                n_cust = rs.poisson(lam, n_iter)
                n_conv = rs.binomial(n_cust, conv_adj)
                # A day's spend is the sum of n independent per-customer
                # draws, N(n*mu, sqrt(n)*sigma), as in mc_engine; scaling a
                # single draw by n would inflate the spread by sqrt(n).
                nc = n_conv.astype(np.float64)
                base = np.where(
                    n_conv > 0,
                    rs.normal(nc * rev_mean, np.sqrt(nc) * rev_sd, n_iter),
                    0.0
                )
                base = np.maximum(base, 0.0)
                n_imp = rs.binomial(np.maximum(n_conv, 0), imp_adj)
                ni = n_imp.astype(np.float64)
                imp_std = p.get('impulse_value_std', p['avg_impulse_value'] * 0.3)
                imp = np.where(
                    n_imp > 0,
                    rs.normal(ni * max(p['avg_impulse_value'], 0.01),
                              np.sqrt(ni) * max(imp_std, 0.01), n_iter),
                    0.0
                )
                imp = np.maximum(imp, 0.0)
                daily_rev[:, d] = base + imp

            totals = daily_rev.sum(axis=1)
            results[label] = {
                'totals': totals,
                'daily_rev': daily_rev,
                'mean': totals.mean(),
                'std': totals.std(),
                'median': np.median(totals),
                'p5': np.percentile(totals, 5),
                'p95': np.percentile(totals, 95),
                'daily_means': daily_rev.mean(axis=0),
                'score': score,
                'conv_adj': conv_adj,
                'imp_adj': imp_adj,
            }

        self._ab_progress.set("Running statistical tests...")
        self.tk_root.update_idletasks()

        t_stat, p_value = sp_stats.ttest_ind(
            results['A']['totals'], results['B']['totals'],
            equal_var=False
        )

        ks_stat, ks_p = sp_stats.ks_2samp(
            results['A']['totals'], results['B']['totals']
        )

        pooled_std = np.sqrt(
            (results['A']['std'] ** 2 + results['B']['std'] ** 2) / 2
        )
        cohens_d = (results['B']['mean'] - results['A']['mean']) / max(pooled_std, 1e-6)

        significant = p_value < alpha

        n_b_wins = np.sum(results['B']['totals'] > results['A']['totals'])
        p_b_better = n_b_wins / n_iter * 100

        # Iteration i of A and of B come from the same stream position, and
        # the pairs are independent across iterations, so the bootstrap of
        # the paired differences is an interval for the mean lift under the
        # model. The verdict below only names a direction when that interval
        # stays on one side of zero.
        diffs = results['B']['totals'] - results['A']['totals']
        rng_bs = np.random.default_rng(ab_seed)
        boots = np.array([
            rng_bs.choice(diffs, size=diffs.size, replace=True).mean()
            for _ in range(2000)])
        lift_mean = float(diffs.mean())
        lift_ci95 = (float(np.percentile(boots, 2.5)),
                     float(np.percentile(boots, 97.5)))

        test_results = {
            't_stat': t_stat, 'p_value': p_value,
            'ks_stat': ks_stat, 'ks_p': ks_p,
            'cohens_d': cohens_d, 'significant': significant,
            'alpha': alpha, 'p_b_better': p_b_better,
            'lift_mean': lift_mean, 'lift_ci95': lift_ci95,
        }

        self._display_ab_results(results, test_results, n_days, n_iter)
        self._ab_progress.set("Done.")

    def _display_ab_results(self, results, tests, n_days, n_iter):
        ra = results['A']
        rb = results['B']
        delta = rb['mean'] - ra['mean']
        pct = delta / max(ra['mean'], 1e-6) * 100

        sig_str = "YES" if tests['significant'] else "NO"
        effect = ("Large" if abs(tests['cohens_d']) >= 0.8 else
                  "Medium" if abs(tests['cohens_d']) >= 0.5 else
                  "Small" if abs(tests['cohens_d']) >= 0.2 else "Negligible")

        lines= [
            "=" * 52,
            "   WHAT-IF A/B COMPARISON REPORT",
            "=" * 52,
            "",
            f"  MC iterations:     {n_iter:,}",
            f"  Projection:        {n_days} days",
            f"  Significance:      alpha = {tests['alpha']}",
            f"  Elasticities:      band midpoints (the Optimize report",
            f"                     uses its sensitivity-derived weights)",
            f"  Random numbers:    one stream shared by A and B",
            "",
            "REVENUE SUMMARY ({}-day total)".format(n_days),
            "-" * 52,
            f"  {'Metric':<22s} {'Layout A':>12s} {'Layout B':>12s}",
            f"  {'Mean':.<22s} ${ra['mean']:>11,.2f} ${rb['mean']:>11,.2f}",
            f"  {'Median':.<22s} ${ra['median']:>11,.2f} ${rb['median']:>11,.2f}",
            f"  {'Std Dev':.<22s} ${ra['std']:>11,.2f} ${rb['std']:>11,.2f}",
            f"  {'5th pct':.<22s} ${ra['p5']:>11,.2f} ${rb['p5']:>11,.2f}",
            f"  {'95th pct':.<22s} ${ra['p95']:>11,.2f} ${rb['p95']:>11,.2f}",
            "",
            f"  Delta (B - A):     ${delta:>+12,.2f}  ({pct:+.1f}%)",
            f"  Daily delta:       ${delta / n_days:>+12,.2f}",
            f"  Annual projection: ${delta / n_days * 365:>+12,.0f}",
            "",
            "STATISTICAL TESTS",
            "-" * 52,
            f"  Welch's t-test:",
            f"    t-statistic:     {tests['t_stat']:.4f}",
            f"    p-value:         {tests['p_value']:.6f}",
            f"    p < alpha:       {sig_str} (alpha={tests['alpha']})",
            "",
            f"  KS test:",
            f"    KS statistic:    {tests['ks_stat']:.4f}",
            f"    p-value:         {tests['ks_p']:.6f}",
            "",
            f"  Effect Size:",
            f"    Cohen's d:       {tests['cohens_d']:.4f}  ({effect})",
            f"    P(B > A):        {tests['p_b_better']:.1f}%",
            "",
            "LAYOUT QUALITY SCORES",
            "-" * 52,
            f"  Layout A score:    {ra['score']:.4f}",
            f"  Layout B score:    {rb['score']:.4f}",
            f"  Conv rate A:       {ra['conv_adj']:.4f}",
            f"  Conv rate B:       {rb['conv_adj']:.4f}",
            f"  Impulse rate A:    {ra['imp_adj']:.4f}",
            f"  Impulse rate B:    {rb['imp_adj']:.4f}",
            "",
            "VERDICT",
            "-" * 52,]

        # Both layouts are simulated under parameters that differ by
        # construction, so p shrinks with the MC iteration count and says
        # nothing about a real-world effect; the verdict reports magnitude
        # and only names a direction the paired-lift interval supports.
        # With thousands of iterations that interval excludes zero even for
        # changes too small to matter, so a winner is also
        # required to move the mean by at least a small effect (|d| >= 0.2)
        # relative to the spread of simulated outcomes.
        lift_mean = tests.get('lift_mean', delta)
        lift_ci = tests.get('lift_ci95')
        ahead = None
        if lift_ci and lift_ci[0] > 0:
            ahead = 'B'
        elif lift_ci and lift_ci[1] < 0:
            ahead = 'A'
        winner = ahead if (ahead and abs(tests['cohens_d']) >= 0.2) else None
        if winner:
            lines.append(f"  Layout {winner} projects higher mean revenue")
        elif ahead:
            lines.append(f"  No practically relevant difference")
            lines.append(f"  (Layout {ahead} is ahead beyond MC noise, but by")
            lines.append(f"  less than 0.2 SD of the simulated outcomes)")
        else:
            lines.append(f"  No difference distinguishable from MC noise")
        lines.append(f"  ({pct:+.1f}% B vs A, d={tests['cohens_d']:.3f}, {effect})")
        lines.append(f"  Paired lift (B - A):  ${lift_mean:+,.2f}")
        if lift_ci:
            lines.append(f"  95% CI (MC precision under the model):")
            lines.append(f"    ${lift_ci[0]:+,.2f}  to  ${lift_ci[1]:+,.2f}")
        lines.append(f"  The interval covers MC precision under the")
        lines.append(f"  model only, not real-world uncertainty")
        lines.append(f"  about the change.")
        lines.append(f"  p-values compare two simulated distributions")
        lines.append(f"  and shrink as MC iterations grow; read the")
        lines.append(f"  effect size, not p.")

        self._ab_text.config(state=tk.NORMAL)
        self._ab_text.delete('1.0', tk.END)
        self._ab_text.insert(tk.END, "\n".join(lines))
        self._ab_text.config(state=tk.DISABLED)

        self._ab_fig.clear()
        wc = 'white'

        ax1 = self._ab_fig.add_subplot(2, 2, 1)
        bins= np.linspace(
            min(ra['totals'].min(), rb['totals'].min()),
            max(ra['totals'].max(), rb['totals'].max()),
            50)
        ax1.hist(ra['totals'], bins=bins, alpha=0.6, color='#e74c3c',
                 label=f"A (mean=${ra['mean']:,.0f})", edgecolor='white', linewidth=0.3)
        ax1.hist(rb['totals'], bins=bins, alpha=0.6, color='#3498db',
                 label=f"B (mean=${rb['mean']:,.0f})", edgecolor='white', linewidth=0.3)
        ax1.axvline(ra['mean'], color='#e74c3c', linewidth=2, linestyle='--')
        ax1.axvline(rb['mean'], color='#3498db', linewidth=2, linestyle='--')
        ax1.set_xlabel(f"{n_days}-Day Total Revenue ($)", color=wc, fontsize=9)
        ax1.set_ylabel("Frequency", color=wc, fontsize=9)
        # No significance marker: p here only tracks the MC iteration count.
        title1 = f"Revenue Distributions"
        ax1.set_title(title1, color=wc, fontsize=10)
        ax1.legend(fontsize=7)
        ax1.set_facecolor('#001a33')
        ax1.tick_params(colors=wc, labelsize=7)
        ax1.grid(True, alpha=0.15, color='white')

        ax2 = self._ab_fig.add_subplot(2, 2, 2)
        days = np.arange(1, n_days + 1)
        ax2.plot(days, ra['daily_means'], color='#e74c3c', linewidth=1.5, label='A daily')
        ax2.plot(days, rb['daily_means'], color='#3498db', linewidth=1.5, label='B daily')
        cum_a = np.cumsum(ra['daily_means'])
        cum_b = np.cumsum(rb['daily_means'])
        ax2t = ax2.twinx()
        ax2t.plot(days, cum_a, color='#e74c3c', linewidth=1, linestyle=':', alpha=0.6,
                  label='A cumul.')
        ax2t.plot(days, cum_b, color='#3498db', linewidth=1, linestyle=':', alpha=0.6,
                  label='B cumul.')
        ax2t.set_ylabel("Cumulative ($)", color=wc, fontsize=8)
        ax2t.tick_params(colors=wc, labelsize=7)
        ax2.set_xlabel("Day", color=wc, fontsize=9)
        ax2.set_ylabel("Daily Revenue ($)", color=wc, fontsize=9)
        ax2.set_title("Daily & Cumulative Revenue", color=wc, fontsize=10)
        ax2.legend(fontsize=6, loc='upper left')
        ax2t.legend(fontsize=6, loc='lower right')
        ax2.set_facecolor('#001a33')
        ax2.tick_params(colors=wc, labelsize=7)
        ax2.grid(True, alpha=0.15, color='white')

        ax3 = self._ab_fig.add_subplot(2, 2, 3)
        diff = rb['totals'] - ra['totals']
        # Coloured by the verdict, not the sign of the mean difference.
        lift_color = {'B': '#2ecc71', 'A': '#e74c3c'}.get(winner, '#95a5a6')
        ax3.hist(diff, bins=50, color=lift_color,
                 alpha=0.7, edgecolor='white', linewidth=0.3)
        ax3.axvline(0, color='white', linewidth=1.5, linestyle='--')
        ax3.axvline(diff.mean(), color='#f39c12', linewidth=2,
                    label=f'Mean diff: ${diff.mean():+,.0f}')
        pct_positive = (diff > 0).sum() / len(diff) * 100
        ax3.set_xlabel("Revenue Difference B - A ($)", color=wc, fontsize=9)
        ax3.set_ylabel("Frequency", color=wc, fontsize=9)
        ax3.set_title(f"Lift Distribution (B>A in {pct_positive:.1f}% of sims)",
                      color=wc,fontsize=10)
        ax3.legend(fontsize=7)
        ax3.set_facecolor('#001a33')
        ax3.tick_params(colors=wc, labelsize=7)
        ax3.grid(True, alpha=0.15, color='white')

        ax4 = self._ab_fig.add_subplot(2, 2, 4)
        metrics = ['Mean\nRevenue', 'Median\nRevenue', 'P5\n(Worst)', 'P95\n(Best)',
                   'Std Dev']
        a_vals = [ra['mean'], ra['median'], ra['p5'], ra['p95'], ra['std']]
        b_vals = [rb['mean'], rb['median'], rb['p5'], rb['p95'], rb['std']]
        x = np.arange(len(metrics))
        w = 0.35
        ax4.bar(x - w / 2, a_vals, w, color='#e74c3c', alpha=0.8, label='Layout A')
        ax4.bar(x + w / 2, b_vals, w, color='#3498db', alpha=0.8, label='Layout B')
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
        
        

