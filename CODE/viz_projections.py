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
from sim_calibration import extract_simulation_parameters, mc_engine, _calibrated

import copy
from scipy import stats as sp_stats


class ProjectionsMixin:
    """Projections tab + Monte Carlo simulation engine."""

    def _create_projections_tab(self, notebook):
        tab = tk.Frame(notebook, bg='#002747')
        notebook.add(tab, text="Projections")

        # -- Controls bar -----------------------------------------
        ctrl = tk.Frame(tab, bg='#002747')
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=10, pady=8)

        lbl_kw = dict(fg='white', bg='#002747', font=('Arial', 10))
        ent_kw = dict(width=8)

        tk.Label(ctrl, text="Operating hrs/day:", **lbl_kw).grid(row=0, column=0, padx=4)
        self._proj_hours = tk.DoubleVar(value=10.0)
        tk.Entry(ctrl, textvariable=self._proj_hours, **ent_kw).grid(row=0, column=1, padx=4)

        tk.Label(ctrl, text="Projection days:", **lbl_kw).grid(row=0, column=2, padx=4)
        self._proj_days = tk.IntVar(value=30)
        tk.Entry(ctrl, textvariable=self._proj_days, **ent_kw).grid(row=0, column=3, padx=4)

        tk.Label(ctrl, text="MC iterations:", **lbl_kw).grid(row=0, column=4, padx=4)
        self._proj_iters = tk.IntVar(value=5000)
        tk.Entry(ctrl, textvariable=self._proj_iters, **ent_kw).grid(row=0, column=5, padx=4)

        tk.Label(ctrl, text="Weekend multiplier:", **lbl_kw).grid(row=0, column=6, padx=4)
        self._proj_wknd = tk.DoubleVar(value=1.4)
        tk.Entry(ctrl, textvariable=self._proj_wknd, **ent_kw).grid(row=0, column=7, padx=4)

        tk.Label(ctrl, text="Monthly growth %:", **lbl_kw).grid(row=0, column=8, padx=4)
        self._proj_growth = tk.DoubleVar(value=0.0)
        tk.Entry(ctrl, textvariable=self._proj_growth, **ent_kw).grid(row=0, column=9, padx=4)

        tk.Button(
            ctrl, text="Run Projection", command=self._run_monte_carlo,
            bg='#FF6B35', fg='white', font=('Arial', 11, 'bold'),
            relief=tk.RAISED, bd=3
        ).grid(row=0, column=10, padx=12)

        # -- Results area: left = text, right = charts ------------
        body = tk.Frame(tab, bg='#002747')
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)

        self._proj_text = tk.Text(
            body, bg='#001a33', fg='white', font=('Courier', 10),
            wrap=tk.WORD, state=tk.DISABLED, width=52
        )
        self._proj_text.grid(row=0, column=0, sticky='nsew', padx=(0, 5))

        chart_frame = tk.Frame(body, bg='#002747')
        chart_frame.grid(row=0, column=1, sticky='nsew')

        self._proj_fig = Figure(figsize=(9, 7), dpi=100)
        self._proj_fig.patch.set_facecolor('#002747')
        self._proj_canvas = FigureCanvasTkAgg(self._proj_fig, master=chart_frame)
        self._proj_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


    def _extract_simulation_parameters(self):
        return extract_simulation_parameters(self.customer_simulation)


    def _run_monte_carlo(self):
        A = self.customer_simulation.analytics
        if not _calibrated(A) and A.get('total_customers', 0) < 5:
            messagebox.showwarning(
                "Insufficient Data",
                "Run the real-time simulation for a while first to gather\n"
                "enough data for meaningful projections (at least 5 customers).",
                parent=self.tk_root
            )
            return

        params = self._extract_simulation_parameters()
        n_days = max(1, self._proj_days.get())
        n_iter = max(100, self._proj_iters.get())
        op_hours = max(1.0, self._proj_hours.get())
        wknd_mult = max(1.0, self._proj_wknd.get())
        monthly_growth = self._proj_growth.get() / 100.0

        cph = params['customers_per_hour']
        conv = params['conversion_rate']
        rev_mean = params['rev_per_converting_customer']
        rev_std = params['rev_std']
        imp_rate = params['impulse_rate']
        imp_val = params['avg_impulse_value']
        observed_baskets = params['basket_sizes_observed']
        avg_bsk = params['avg_basket_size']
        std_bsk = params['std_basket_size']

        imp_std = params.get('impulse_value_std', max(imp_val * 0.3, 0.01))
        engine = self._mc_engine(
            cph, conv, rev_mean, rev_std, imp_rate, imp_val,
            avg_bsk, std_bsk, observed_baskets,
            n_days, n_iter, op_hours, wknd_mult, monthly_growth,
            impulse_value_std=imp_std,
        )['raw']
        daily_rev = engine['daily_revenue']
        daily_cust = engine['daily_customers']
        daily_converting = engine['daily_converting']
        daily_impulse_rev = engine['daily_impulse_rev']
        daily_baskets = engine['daily_baskets']

        weekly_rev = np.array([
            daily_rev[:, i:i+7].sum(axis=1) for i in range(0, n_days, 7)
        ]).T
        monthly_rev = np.array([
            daily_rev[:, i:i+30].sum(axis=1) for i in range(0, n_days, 30)
        ]).T

        cumulative_rev = np.cumsum(daily_rev, axis=1)

        results = {
            'params': params,
            'n_days': n_days,
            'n_iter': n_iter,
            'op_hours': op_hours,
            'daily_revenue': daily_rev,
            'daily_customers': daily_cust,
            'daily_converting': daily_converting,
            'daily_impulse_rev': daily_impulse_rev,
            'daily_baskets': daily_baskets,
            'weekly_revenue': weekly_rev,
            'monthly_revenue': monthly_rev,
            'cumulative_revenue': cumulative_rev,
        }

        self._display_projection_results(results)


    def _display_projection_results(self, R):
        p = R['params']
        dr = R['daily_revenue']
        dc = R['daily_customers']
        dcv = R['daily_converting']
        di = R['daily_impulse_rev']
        cr = R['cumulative_revenue']
        nd = R['n_days']
        ni = R['n_iter']

        mean_daily_rev = dr.mean(axis=0)
        lo_daily = np.percentile(dr, 5, axis=0)
        hi_daily = np.percentile(dr, 95, axis=0)

        overall_daily_mean = dr.mean()
        overall_daily_med = np.median(dr)
        overall_daily_std = dr.std()
        total_mean = cr[:, -1].mean()
        total_lo = np.percentile(cr[:, -1], 5)
        total_hi = np.percentile(cr[:, -1], 95)

        mean_daily_cust = dc.mean()
        mean_conv = dcv.mean()
        mean_impulse_daily = di.mean()
        impulse_share = mean_impulse_daily / max(overall_daily_mean, 1e-6) * 100

        if R['monthly_revenue'].shape[1] > 0:
            m1 = R['monthly_revenue'][:, 0]
            m1_mean, m1_lo, m1_hi = m1.mean(), np.percentile(m1, 5), np.percentile(m1, 95)
        else:
            m1_mean = m1_lo = m1_hi = 0.0

        wr = R['weekly_revenue']
        if wr.shape[1] > 0:
            w1 = wr[:, 0]
            w1_mean, w1_lo, w1_hi = w1.mean(), np.percentile(w1, 5), np.percentile(w1, 95)
        else:
            w1_mean = w1_lo = w1_hi = 0.0

        best_day_idx = int(np.argmax(mean_daily_rev))
        worst_day_idx = int(np.argmin(mean_daily_rev))
        annual_proj = overall_daily_mean * 365
        annual_lo = np.percentile(dr.mean(axis=0), 5) * 365
        annual_hi = np.percentile(dr.mean(axis=0), 95) * 365

        area_shares = p.get('area_revenue_shares', {})

        txt_lines = [
            "=" * 50,
            "   MONTE CARLO PROJECTION REPORT",
            "=" * 50,
            "",
            "INPUT PARAMETERS (from live simulation)",
            "-" * 42,
            f"  Observed customers/hr:    {p['customers_per_hour']:.2f}",
            f"  Conversion rate:          {p['conversion_rate']*100:.1f}%",
            f"  Rev/converting customer:  ${p['rev_per_converting_customer']:.2f}",
            f"  Avg basket size:          {p['avg_basket_size']:.1f} items",
            f"  Impulse purchase rate:    {p['impulse_rate']*100:.1f}%",
            f"  Avg impulse value:        ${p['avg_impulse_value']:.2f}",
            f"  Abandonment rate:         {p['abandonment_rate']*100:.1f}%",
            f"  Based on {p['total_observed_customers']} customers,",
            f"    ${p['total_observed_revenue']:.2f} revenue",
            f"    over {p['run_hours']*60:.1f} min sim clock",
            f"    ({p.get('run_wall_hours', p['run_hours'])*60:.1f} min wall)",
            "",
            f"MC: {ni:,} iterations x {nd} days",
            f"    {R['op_hours']:.0f} operating hrs/day",
            "",
            "DAILY REVENUE",
            "-" * 42,
            f"  Mean:     ${overall_daily_mean:>10,.2f}",
            f"  Median:   ${overall_daily_med:>10,.2f}",
            f"  Std Dev:  ${overall_daily_std:>10,.2f}",
            f"  90% CI:   ${lo_daily.mean():>10,.2f}  –"
            f"  ${hi_daily.mean():>10,.2f}",
            f"  Best day  (day {best_day_idx+1}):"
            f" ${mean_daily_rev[best_day_idx]:,.2f}",
            f"  Worst day (day {worst_day_idx+1}):"
            f" ${mean_daily_rev[worst_day_idx]:,.2f}",
            "",
            "WEEKLY REVENUE (week 1)",
            "-" * 42,
            f"  Mean:     ${w1_mean:>10,.2f}",
            f"  90% CI:   ${w1_lo:>10,.2f}  –  ${w1_hi:>10,.2f}",
            "",
            "MONTHLY REVENUE (month 1)",
            "-" * 42,
            f"  Mean:     ${m1_mean:>10,.2f}",
            f"  90% CI:   ${m1_lo:>10,.2f}  –  ${m1_hi:>10,.2f}",
            "",
            f"TOTAL OVER {nd} DAYS",
            "-" * 42,
            f"  Mean:     ${total_mean:>10,.2f}",
            f"  90% CI:   ${total_lo:>10,.2f}  –  ${total_hi:>10,.2f}",
            "",
            "ANNUALISED ESTIMATE",
            "-" * 42,
            f"  Mean:     ${annual_proj:>10,.2f}",
            f"  Range:    ${annual_lo:>10,.2f}  –  ${annual_hi:>10,.2f}",
            "",
            "CUSTOMER METRICS (daily avg)",
            "-" * 42,
            f"  Customers/day:       {mean_daily_cust:>8.1f}",
            f"  Converting/day:      {mean_conv:>8.1f}",
            f"  Impulse rev share:   {impulse_share:>7.1f}%",
            "",
        ]

        if area_shares:
            txt_lines.append("PROJECTED REVENUE BY CATEGORY")
            txt_lines.append("-" * 42)
            for area, share in sorted(area_shares.items(),
                                    key=lambda x: x[1], reverse=True):
                proj = share * overall_daily_mean
                txt_lines.append(f"  {area:<22s} ${proj:>8,.2f}/day"
                                f"  ({share*100:.1f}%)")
            txt_lines.append("")

        self._proj_text.config(state=tk.NORMAL)
        self._proj_text.delete('1.0', tk.END)
        self._proj_text.insert(tk.END, "\n".join(txt_lines))
        self._proj_text.config(state=tk.DISABLED)

        # -- Charts -----------------------------------------------
        self._proj_fig.clear()
        axes_color = 'white'

        # 1) Daily revenue with 90% CI band
        ax1 = self._proj_fig.add_subplot(2, 2, 1)
        days_x = np.arange(1, nd + 1)
        ax1.fill_between(days_x, lo_daily, hi_daily,
                        alpha=0.3, color='#FF6B35', label='90% CI')
        ax1.plot(days_x, mean_daily_rev, color='#FF6B35',
                linewidth=1.5, label='Mean')
        ax1.set_title("Daily Revenue Projection", color=axes_color, fontsize=10)
        ax1.set_xlabel("Day", color=axes_color, fontsize=8)
        ax1.set_ylabel("Revenue ($)", color=axes_color, fontsize=8)
        ax1.legend(fontsize=7)
        ax1.set_facecolor('#001a33')
        ax1.tick_params(colors=axes_color, labelsize=7)

        # 2) Cumulative revenue with CI
        ax2 = self._proj_fig.add_subplot(2, 2, 2)
        cum_mean = cr.mean(axis=0)
        cum_lo = np.percentile(cr, 5, axis=0)
        cum_hi = np.percentile(cr, 95, axis=0)
        ax2.fill_between(days_x, cum_lo, cum_hi,
                        alpha=0.3, color='#4ECDC4')
        ax2.plot(days_x, cum_mean, color='#4ECDC4', linewidth=1.5)
        ax2.set_title("Cumulative Revenue", color=axes_color, fontsize=10)
        ax2.set_xlabel("Day", color=axes_color, fontsize=8)
        ax2.set_ylabel("Cumulative ($)", color=axes_color, fontsize=8)
        ax2.set_facecolor('#001a33')
        ax2.tick_params(colors=axes_color, labelsize=7)

        # 3) Distribution of total revenue (histogram)
        ax3 = self._proj_fig.add_subplot(2, 2, 3)
        totals = cr[:, -1]
        ax3.hist(totals, bins=60, color='#45B7D1', alpha=0.8,
                edgecolor='#002747')
        ax3.axvline(total_mean, color='#FF6B35', linestyle='--',
                    linewidth=1.5, label=f'Mean ${total_mean:,.0f}')
        ax3.axvline(total_lo, color='white', linestyle=':',
                    linewidth=1, label=f'5th %ile ${total_lo:,.0f}')
        ax3.axvline(total_hi, color='white', linestyle=':',
                    linewidth=1, label=f'95th %ile ${total_hi:,.0f}')
        ax3.set_title(f"Total Revenue Distribution ({nd}d)",
                    color=axes_color, fontsize=10)
        ax3.set_xlabel("Revenue ($)", color=axes_color, fontsize=8)
        ax3.set_ylabel("Frequency", color=axes_color, fontsize=8)
        ax3.legend(fontsize=6)
        ax3.set_facecolor('#001a33')
        ax3.tick_params(colors=axes_color, labelsize=7)

        # 4) Revenue by category (pie or bar)
        ax4 = self._proj_fig.add_subplot(2, 2, 4)
        if area_shares:
            sorted_areas = sorted(area_shares.items(),
                                key=lambda x: x[1], reverse=True)
            labels = [a for a, _ in sorted_areas]
            sizes = [s for _, s in sorted_areas]
            colors_pie = plt.cm.Set3(np.linspace(0, 1, len(labels)))
            wedges, texts, autotexts = ax4.pie(
                sizes, labels=labels, autopct='%1.1f%%',
                colors=colors_pie, textprops={'fontsize': 7}
            )
            for t in texts:
                t.set_color(axes_color)
            for t in autotexts:
                t.set_color('#002747')
                t.set_fontweight('bold')
            ax4.set_title("Revenue by Category", color=axes_color,
                        fontsize=10)
        else:
            ax4.text(0.5, 0.5, "No category data\navailable yet",
                    ha='center', va='center', color='white',
                    fontsize=11, transform=ax4.transAxes)
            ax4.set_facecolor('#001a33')

        self._proj_fig.tight_layout(pad=2.0)
        self._proj_canvas.draw_idle()





    # -------------------------------------------------------------
    # REUSABLE MC ENGINE
    # -------------------------------------------------------------

    def _mc_engine(self, cph, conv, rev_mean, rev_std, imp_rate, imp_val,
               avg_bsk, std_bsk, observed_baskets,
               n_days, n_iter, op_hours, wknd_mult, monthly_growth,
               impulse_value_std=None):
        if impulse_value_std is None:
            impulse_value_std = max(imp_val * 0.3, 0.01)
        raw = mc_engine(
            cph, conv, rev_mean, rev_std, imp_rate, imp_val,
            avg_bsk, std_bsk, observed_baskets,
            n_days, n_iter, op_hours, wknd_mult, monthly_growth,
            impulse_value_std=impulse_value_std,
        )
        daily_rev = raw['daily_revenue']            # shape: (n_iter, n_days)
        totals = daily_rev.sum(axis=1)              # per-iteration totals
        return {
            'mean':        float(totals.mean()),
            'std':         float(totals.std()),
            'median':      float(np.median(totals)),
            'p5':          float(np.percentile(totals, 5)),
            'p95':         float(np.percentile(totals, 95)),
            'totals':      totals,
            'daily_means': daily_rev.mean(axis=0),  # shape: (n_days,)
            'raw':         raw,                     # keep originals for projections tab
        }

    # -------------------------------------------------------------
    # SENSITIVITY ANALYSIS TAB (Tornado Chartsigma)
    # -------------------------------------------------------------

