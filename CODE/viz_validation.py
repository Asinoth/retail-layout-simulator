"""Validation tab: empirical-vs-simulated goodness-of-fit dashboard.

Shows the loaded dataset's provenance + three side-by-side distribution
plots (basket size, per-visit revenue, category purchase shares) with KS /
chi-square test results. The "Run validation now" button takes a snapshot
of the live simulation analytics and re-tests against the last calibrated
distribution. Each panel draws the same two samples its test compares, and
leaves the simulated overlay out when the test cannot run.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np

import tkinter as tk
from tkinter import ttk, messagebox

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from dataset_validation import (validate_against_simulation,
                                observed_category_counts,
                                simulated_category_purchases,
                                placed_product_ids,
                                observed_distinct_baskets,
                                observed_distinct_revenues)


def _is_aggregate_source(params) -> bool:
    """True for aggregate sources (Omnichannel), whose basket and revenue
    arrays are a parametric realization of measured purchase probabilities
    rather than observed visits -- the same rule the goodness-of-fit tests
    use to leave those two tests out."""
    extra = getattr(params, 'calibration_extra', None) or {}
    return (str(extra.get('basket_size_source', '')).startswith('parametric')
            or extra.get('source_kind') == 'aggregate_retail_omnichannel')


def _conversion_wording(params) -> str:
    """How the calibrated conversion rate was obtained, as the dataset
    load popup states it."""
    if (getattr(params, 'conversion_rate_source', 'assumption')
            == 'estimated_from_purchase_probabilities'):
        return "estimated from purchase probabilities"
    return "asserted; not measured from a buy/no-buy split"


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
        if tx_prov and params is not None and _is_aggregate_source(params):
            # An aggregate source has no invoices: n_invoices holds the
            # arrivals table's weekly in-store visitor total, and the rate
            # the simulator runs on is visitors, not buyers.
            lines += [
                "AGGREGATE (BEHAVIORAL) DATASET",
                "-" * 60,
                f"Source:           {tx_prov.get('source_path', '?')}",
                f"  bytes:          {tx_prov.get('source_bytes', 0):,}",
                f"  sha256:         {tx_prov.get('source_sha256', '?')}",
                f"Adapter:          {tx_prov.get('adapter_name','?')} "
                f"v{tx_prov.get('adapter_version','?')}",
                f"Rows in/kept:     {tx_prov.get('rows_in',0):,} / "
                f"{tx_prov.get('rows_kept',0):,}",
                f"In-store visitors / week: {params.n_invoices:,}",
                f"Product families: {params.n_unique_products:,}",
                f"In-store zones:   {params.n_unique_categories}",
                f"Arrival rate:     {params.visitors_per_hour:.1f} visitors/hr",
                f"Conversion:       {params.assumed_conversion_rate*100:.0f} %  "
                f"({_conversion_wording(params)})",
                "",
            ]
        elif tx_prov and params is not None:
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
                f"({_conversion_wording(params)})",
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
            verdict = t.verdict(res.alpha)
            self._val_table.insert(
                '', tk.END,
                values=(f"{t.name}  ({t.test})",
                        f"{t.statistic:.4f}",
                        f"{t.p_value:.4g}",
                        t.n_observed, t.n_simulated, verdict),
            )

        # Redraw plots with simulated overlay. The category panel shows item
        # purchases per category over the products placed in the shop, the
        # sample the chi-square test uses; zone entries also count agents
        # walking through a department and are not what is tested.
        sim = self.customer_simulation
        A = sim.analytics
        pids = placed_product_ids(getattr(sim, 'shop', None))
        sim_cat_purchases, _ = simulated_category_purchases(
            sim, product_ids=pids or None)
        self._draw_validation_plots(
            simulated_baskets=np.asarray(A.get('basket_sizes', []),
                                         dtype=np.float64),
            simulated_revs=np.asarray(A.get('customer_revenues', []),
                                       dtype=np.float64),
            simulated_categories=sim_cat_purchases,
        )

        # Tests that could not run show N/A; their note says why.
        notes = list(res.summary_lines)
        notes += [f"{t.name}: {t.note.strip()}" for t in res.tests
                  if not t.applicable() and t.note.strip()]
        if notes:
            messagebox.showinfo("Validation notes",
                                "\n".join(notes),
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

        # Aggregate sources carry a parametric realization of measured
        # purchase probabilities, not observed visits; their KS tests are
        # left out as circular, so the panels draw no simulated overlay.
        aggregate = _is_aggregate_source(params)

        # 1) Basket size histogram
        # The simulator counts distinct items per visit; params.basket_sizes
        # is units per invoice, so the simulated overlay is only drawn
        # against a distinct-items-per-invoice sample.
        obs = None if aggregate else observed_distinct_baskets(params)
        comparable = obs is not None
        if not comparable:
            obs = np.asarray(params.basket_sizes, dtype=np.float64)
        bmax = int(np.percentile(obs, 99)) if obs.size else 10
        bins = np.arange(0, max(bmax + 2, 3))
        ax_b.hist(obs, bins=bins, density=True, alpha=0.6,
                  color='#4ECDC4', label='Observed')
        if (comparable and simulated_baskets is not None
                and simulated_baskets.size > 0):
            ax_b.hist(simulated_baskets, bins=bins, density=True, alpha=0.4,
                      color='#FF6B6B', label='Simulated')
        if comparable:
            ax_b.set_title('Basket size  (items / visit)')
            ax_b.set_xlabel('items')
        elif aggregate:
            # Distinct families bought per synthetic visitor.
            ax_b.set_title('Basket size  (families / visit, parametric)')
            ax_b.set_xlabel('families')
        else:
            ax_b.set_title('Basket size  (units / invoice)')
            ax_b.set_xlabel('units')
        ax_b.set_ylabel('density')
        ax_b.legend(facecolor='#001a33', edgecolor='#557', labelcolor='white',
                    fontsize=8)

        # 2) Per-visit revenue
        # A simulated visit buys one unit of each item it picks up, so the
        # overlay is only drawn against the dataset's one-unit-per-product
        # invoice sums; params.invoice_revenues is sum(quantity x price),
        # which a wholesale line inflates.
        obs_r = None if aggregate else observed_distinct_revenues(params)[0]
        comparable_r = obs_r is not None
        if not comparable_r:
            obs_r = np.asarray(params.invoice_revenues, dtype=np.float64)
        if obs_r.size:
            top = float(np.percentile(obs_r, 99))
            r_bins = np.linspace(0, max(top, 1.0), 40)
            ax_r.hist(obs_r, bins=r_bins, density=True, alpha=0.6,
                      color='#4ECDC4', label='Observed')
            if (comparable_r and simulated_revs is not None
                    and simulated_revs.size > 0):
                ax_r.hist(np.clip(simulated_revs, 0, top), bins=r_bins,
                          density=True, alpha=0.4,
                          color='#FF6B6B', label='Simulated')
            if comparable_r:
                ax_r.set_title(f'Per-visit revenue  ({params.currency}, '
                               f'one unit / product)')
            elif aggregate:
                ax_r.set_title(f'Per-visit revenue  ({params.currency}, '
                               f'parametric)')
            else:
                ax_r.set_title(f'Invoice revenue  ({params.currency}, '
                               f'quantity x price)')
            ax_r.set_xlabel(params.currency)
            ax_r.set_ylabel('density')
            ax_r.legend(facecolor='#001a33', edgecolor='#557',
                        labelcolor='white', fontsize=8)

        # 3) Category purchase shares
        # Same quantities the chi-square test compares: product-invoice
        # touches per category over the products placed in the shop, against
        # the live item purchases of those products per category, each
        # normalized over its own total.
        pids = placed_product_ids(getattr(self.customer_simulation, 'shop',
                                          None))
        obs_counts = observed_category_counts(params,
                                              product_ids=pids or None)
        sim_counts = dict(simulated_categories or {})
        cats = sorted(set(obs_counts) | set(sim_counts))
        obs_tot = sum(obs_counts.values()) or 1.0
        obs_share = np.array([obs_counts.get(c, 0.0) / obs_tot for c in cats])
        x = np.arange(len(cats))
        width = 0.4 if sim_counts else 0.7
        ax_c.bar(x - width/2 if sim_counts else x,
                 obs_share, width=width,
                 color='#4ECDC4', label='Observed')
        if sim_counts:
            tot = max(sum(sim_counts.values()), 1)
            sim_share = np.array([sim_counts.get(c, 0) / tot for c in cats])
            ax_c.bar(x + width/2, sim_share, width=width,
                     color='#FF6B6B', label='Simulated')
            ax_c.legend(facecolor='#001a33', edgecolor='#557',
                        labelcolor='white', fontsize=8)
        ax_c.set_xticks(x)
        ax_c.set_xticklabels(cats, rotation=30, ha='right', fontsize=7)
        ax_c.set_title('Category purchase share')

        # 4) Walking speed (only when trajectory data has been loaded)
        if ax_s is not None and sparams is not None:
            self._draw_speed_panel(ax_s, sparams)

        fig.tight_layout()
        self._val_canvas.draw_idle()

    def _draw_speed_panel(self, ax, sparams):
        """Histogram of empirical walking speeds from the loaded trajectory
        dataset, with the range the simulator draws agent speeds from
        overlaid as a shaded reference."""
        speeds = sparams.speeds_m_s
        if speeds is None or speeds.size == 0:
            ax.text(0.5, 0.5, 'No velocity samples',
                    ha='center', va='center', color='#9fb6c9',
                    transform=ax.transAxes)
            return
        bins = np.linspace(0, max(float(np.percentile(speeds, 99)), 2.0), 40)
        ax.hist(speeds, bins=bins, density=True, alpha=0.7,
                color='#4ECDC4', label='Empirical (dataset)')
        # Once trajectory calibration is seeded, the spawn loop draws each
        # agent's speed from N(mean, std) clipped to [max(0.3, p5),
        # min(2.5, p95)]; without it Customer.__init__ uses uniform(0.8, 1.5).
        cal = self.customer_simulation.analytics.get('calibration', {}) or {}
        mu = cal.get('empirical_speed_mean')
        if mu:
            sd = cal.get('empirical_speed_std') or 0.2
            lo = max(0.3, cal.get('empirical_speed_p5', mu - 2 * sd))
            hi = min(2.5, cal.get('empirical_speed_p95', mu + 2 * sd))
            band_label = 'Simulator speed range (calibrated)'
        else:
            lo, hi = 0.8, 1.5
            band_label = 'Simulator speed band (default)'
        ax.axvspan(lo, hi, color='#FF6B6B', alpha=0.2,
                   label=band_label)
        ax.set_title('Walking speed')
        ax.set_xlabel('m / s')
        ax.set_ylabel('density')
        ax.legend(facecolor='#001a33', edgecolor='#557',
                  labelcolor='white', fontsize=8)
