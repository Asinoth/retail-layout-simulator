"""Validation tab: empirical-vs-simulated goodness-of-fit dashboard.

Shows the loaded dataset's provenance + three side-by-side distribution
plots (basket size, per-visit revenue, category visit shares) with KS /
chi-square test results. The "Run validation now" button takes a snapshot
of the live simulation analytics and re-tests against the last calibrated
distribution.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np

import tkinter as tk
from tkinter import ttk, messagebox

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from dataset_validation import validate_against_simulation


class ValidationMixin:
    """Adds the Validation tab + refresh helpers to ShopVisualizer."""

    def _create_validation_tab(self, notebook):
        tab = tk.Frame(notebook, bg='#001a33')
        notebook.add(tab, text='Validation')

        # -- Header / provenance --------------------------------
        hdr = tk.Frame(tab, bg='#001a33')
        hdr.pack(fill=tk.X, padx=10, pady=8)
        tk.Label(hdr, text="Dataset calibration & goodness-of-fit",
                 font=('Arial', 14, 'bold'),
                 bg='#001a33', fg='white').pack(anchor='w')

        self._val_prov_text = tk.Text(
            tab, height=10, wrap=tk.NONE, bg='#002747', fg='white',
            font=('Courier New', 9))
        self._val_prov_text.pack(fill=tk.X, padx=10, pady=(0, 8))
        self._set_val_prov_text(
            "No dataset loaded.\n\n"
            "Open the Layout tab → Dataset button to load a transactional "
            "dataset (UCI Online Retail II recommended). After calibration "
            "this tab shows the empirical distributions of the dataset and "
            "the goodness-of-fit between them and the simulator's output."
        )

        # -- Action row -----------------------------------------
        act = tk.Frame(tab, bg='#001a33')
        act.pack(fill=tk.X, padx=10, pady=4)
        tk.Button(act, text="Run validation now",
                  command=self._run_validation_now,
                  bg='#2d7a2d', fg='white',
                  font=('Arial', 10, 'bold')).pack(side=tk.LEFT)
        tk.Label(act,
                 text="Compares the currently calibrated dataset against "
                      "the live simulation analytics.",
                 fg='#9fb6c9', bg='#001a33',
                 font=('Arial', 9, 'italic')).pack(side=tk.LEFT, padx=10)

        # -- Goodness-of-fit table ------------------------------
        tbl_frame = tk.Frame(tab, bg='#001a33')
        tbl_frame.pack(fill=tk.X, padx=10, pady=(8, 4))

        cols = ('test', 'statistic', 'p_value', 'n_obs', 'n_sim', 'verdict')
        self._val_table = ttk.Treeview(tbl_frame, columns=cols,
                                       show='headings', height=5)
        for c, w in zip(cols, (260, 90, 90, 70, 70, 80)):
            self._val_table.heading(c, text=c.replace('_', ' ').title())
            self._val_table.column(c, width=w, anchor='w')
        self._val_table.pack(fill=tk.X)

        # -- Distribution plots ---------------------------------
        self._val_fig = Figure(figsize=(12, 4), dpi=100)
        self._val_fig.patch.set_facecolor('#001a33')
        self._val_canvas = FigureCanvasTkAgg(self._val_fig, master=tab)
        self._val_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True,
                                              padx=10, pady=8)

    # -- Helpers -------------------------------------------------------
    def _set_val_prov_text(self, msg: str) -> None:
        self._val_prov_text.config(state=tk.NORMAL)
        self._val_prov_text.delete('1.0', tk.END)
        self._val_prov_text.insert(tk.END, msg)
        self._val_prov_text.config(state=tk.DISABLED)

    def _refresh_validation_tab_metadata(self) -> None:
        """Called by viz_dataset after a successful load (transactional or
        trajectory). Shows whichever provenance records are present and
        redraws the relevant plot panes."""
        A = self.customer_simulation.analytics
        tx_prov = A.get('provenance', {}) or {}
        sp_prov = A.get('spatial_provenance', {}) or {}
        params = getattr(self, '_last_calibration_params', None)
        sparams = getattr(self, '_last_spatial_params', None)
        if not tx_prov and not sp_prov:
            return

        lines = []
        if tx_prov and params is not None:
            lines += [
                "TRANSACTIONAL DATASET",
                "-" * 60,
                f"Source:           {tx_prov.get('source_path', '?')}",
                f"  bytes:          {tx_prov.get('source_bytes', 0):,}",
                f"  sha256:         {tx_prov.get('source_sha256', '?')}",
                f"Adapter:          {tx_prov.get('adapter_name','?')} "
                f"v{tx_prov.get('adapter_version','?')}",
                f"Rows in/kept:     {tx_prov.get('rows_in',0):,} / "
                f"{tx_prov.get('rows_kept',0):,}",
                f"Invoices:         {params.n_invoices:,}",
                f"Unique products:  {params.n_unique_products:,}",
                f"Categories:       {params.n_unique_categories}",
                f"Arrival rate:     {params.arrivals_per_hour:.1f} invoices/hr",
                f"Assumed conv.:    {params.assumed_conversion_rate*100:.0f} %  "
                f"(asserted; not measured from a buy/no-buy split)",
                "",
            ]
        if sp_prov and sparams is not None:
            lines += [
                "TRAJECTORY (SPATIAL) DATASET",
                "-" * 60,
                f"Source:           {sp_prov.get('source_path', '?')}",
                f"  sha256:         {sp_prov.get('source_sha256', '?')}",
                f"Adapter:          {sp_prov.get('adapter_name','?')} "
                f"v{sp_prov.get('adapter_version','?')}",
                f"Rows in/kept:     {sp_prov.get('rows_in',0):,} / "
                f"{sp_prov.get('rows_kept',0):,}",
                f"Tracks:           {sparams.n_tracks:,}",
                f"Span:             {sparams.span_seconds/3600.0:.1f} hours",
                f"Speed (m/s):      mean={sparams.speeds_m_s.mean():.2f}, "
                f"std={sparams.speeds_m_s.std():.2f}"
                if sparams.speeds_m_s.size else "Speed (m/s):      -",
                "",
            ]
        if not lines:
            lines = ["No dataset loaded."]
        self._set_val_prov_text("\n".join(lines))
        # Empirical-only plots -- overlays come on Run validation now.
        self._draw_validation_plots(simulated_baskets=None,
                                    simulated_revs=None,
                                    simulated_categories=None)

    def _run_validation_now(self) -> None:
        params = getattr(self, '_last_calibration_params', None)
        if params is None:
            messagebox.showinfo(
                "No dataset",
                "Load a dataset first (Layout tab → Dataset button).",
                parent=self.tk_root)
            return

        res = validate_against_simulation(params, self.customer_simulation)

        # Populate the table
        for row in self._val_table.get_children():
            self._val_table.delete(row)
        for t in res.tests:
            verdict = "PASS" if t.pass_at(res.alpha) else "FAIL"
            self._val_table.insert(
                '', tk.END,
                values=(f"{t.name}  ({t.test})",
                        f"{t.statistic:.4f}",
                        f"{t.p_value:.4g}",
                        t.n_observed, t.n_simulated, verdict),
            )

        # Redraw plots with simulated overlay
        A = self.customer_simulation.analytics
        self._draw_validation_plots(
            simulated_baskets=np.asarray(A.get('basket_sizes', []),
                                         dtype=np.float64),
            simulated_revs=np.asarray(A.get('customer_revenues', []),
                                       dtype=np.float64),
            simulated_categories=dict(A.get('area_visits', {})),
        )

        if res.summary_lines:
            messagebox.showinfo("Validation notes",
                                "\n".join(res.summary_lines),
                                parent=self.tk_root)

    # -- Plotting ------------------------------------------------------
    def _draw_validation_plots(self,
                               simulated_baskets: Optional[np.ndarray],
                               simulated_revs: Optional[np.ndarray],
                               simulated_categories: Optional[Dict[str, int]],
                               ) -> None:
        params = getattr(self, '_last_calibration_params', None)
        sparams = getattr(self, '_last_spatial_params', None)
        fig = self._val_fig
        fig.clear()

        # 1x3 if only transactional, 1x4 if spatial too.
        n_panels = 3 + (1 if sparams is not None else 0)
        ax_b = fig.add_subplot(1, n_panels, 1)
        ax_r = fig.add_subplot(1, n_panels, 2)
        ax_c = fig.add_subplot(1, n_panels, 3)
        ax_s = fig.add_subplot(1, n_panels, 4) if n_panels == 4 else None

        # Styling
        axes = [ax_b, ax_r, ax_c] + ([ax_s] if ax_s is not None else [])
        for ax in axes:
            ax.set_facecolor('#002747')
            ax.tick_params(colors='white', labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor('#557')
            ax.title.set_color('white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')

        if params is None and sparams is None:
            for ax in axes:
                ax.text(0.5, 0.5, 'No dataset loaded',
                        ha='center', va='center', color='#9fb6c9',
                        transform=ax.transAxes)
            self._val_canvas.draw_idle()
            return
        if params is None:
            # Trajectory-only: blank the transactional panels.
            for ax in (ax_b, ax_r, ax_c):
                ax.text(0.5, 0.5, 'No transactional\ndata',
                        ha='center', va='center', color='#9fb6c9',
                        transform=ax.transAxes)
            self._draw_speed_panel(ax_s, sparams)
            fig.tight_layout()
            self._val_canvas.draw_idle()
            return

        # 1) Basket size histogram
        obs = params.basket_sizes
        bmax = int(np.percentile(obs, 99)) if obs.size else 10
        bins = np.arange(0, max(bmax + 2, 3))
        ax_b.hist(obs, bins=bins, density=True, alpha=0.6,
                  color='#4ECDC4', label='Observed')
        if simulated_baskets is not None and simulated_baskets.size > 0:
            ax_b.hist(simulated_baskets, bins=bins, density=True, alpha=0.4,
                      color='#FF6B6B', label='Simulated')
        ax_b.set_title('Basket size  (items / visit)')
        ax_b.set_xlabel('items')
        ax_b.set_ylabel('density')
        ax_b.legend(facecolor='#001a33', edgecolor='#557', labelcolor='white',
                    fontsize=8)

        # 2) Per-visit revenue
        obs_r = params.invoice_revenues
        if obs_r.size:
            top = float(np.percentile(obs_r, 99))
            r_bins = np.linspace(0, max(top, 1.0), 40)
            ax_r.hist(obs_r, bins=r_bins, density=True, alpha=0.6,
                      color='#4ECDC4', label='Observed')
            if simulated_revs is not None and simulated_revs.size > 0:
                ax_r.hist(np.clip(simulated_revs, 0, top), bins=r_bins,
                          density=True, alpha=0.4,
                          color='#FF6B6B', label='Simulated')
            ax_r.set_title(f'Per-visit revenue  ({params.currency})')
            ax_r.set_xlabel(params.currency)
            ax_r.set_ylabel('density')
            ax_r.legend(facecolor='#001a33', edgecolor='#557',
                        labelcolor='white', fontsize=8)

        # 3) Category visit shares
        cats = sorted(params.category_revenue_share.keys())
        obs_share = np.array([params.category_revenue_share.get(c, 0)
                              for c in cats])
        x = np.arange(len(cats))
        width = 0.4 if simulated_categories else 0.7
        ax_c.bar(x - width/2 if simulated_categories else x,
                 obs_share, width=width,
                 color='#4ECDC4', label='Observed')
        if simulated_categories:
            tot = max(sum(simulated_categories.values()), 1)
            sim_share = np.array(
                [simulated_categories.get(c, 0) / tot for c in cats])
            ax_c.bar(x + width/2, sim_share, width=width,
                     color='#FF6B6B', label='Simulated')
            ax_c.legend(facecolor='#001a33', edgecolor='#557',
                        labelcolor='white', fontsize=8)
        ax_c.set_xticks(x)
        ax_c.set_xticklabels(cats, rotation=30, ha='right', fontsize=7)
        ax_c.set_title('Category share')

        # 4) Walking speed (only when trajectory data has been loaded)
        if ax_s is not None and sparams is not None:
            self._draw_speed_panel(ax_s, sparams)

        fig.tight_layout()
        self._val_canvas.draw_idle()

    def _draw_speed_panel(self, ax, sparams):
        """Histogram of empirical walking speeds from the loaded trajectory
        dataset, with the simulator's nominal speed band (0.8-1.5 m/s,
        per Customer.__init__) overlaid as a shaded reference."""
        speeds = sparams.speeds_m_s
        if speeds is None or speeds.size == 0:
            ax.text(0.5, 0.5, 'No velocity samples',
                    ha='center', va='center', color='#9fb6c9',
                    transform=ax.transAxes)
            return
        bins = np.linspace(0, max(float(np.percentile(speeds, 99)), 2.0), 40)
        ax.hist(speeds, bins=bins, density=True, alpha=0.7,
                color='#4ECDC4', label='Empirical (dataset)')
        # Simulator's hard-coded speed band (Customer.__init__).
        ax.axvspan(0.8, 1.5, color='#FF6B6B', alpha=0.2,
                   label='Simulator speed band')
        ax.set_title('Walking speed')
        ax.set_xlabel('m / s')
        ax.set_ylabel('density')
        ax.legend(facecolor='#001a33', edgecolor='#557',
                  labelcolor='white', fontsize=8)
