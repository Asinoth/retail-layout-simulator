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

from sim_calibration import _calib, mc_engine
from retail_literature import DEFAULT_WEEKEND_MULTIPLIER


def interpretation_lines(pct):
    """The INTERPRETATION block of the Sensitivity report.

    It names no "largest" or "top" driver. Revenue is visitors (customers
    per hour x operating hours) x conversion x spend per converter, plus a
    small impulse term, so a +/-pct change in any of those four moves it
    by about +/-pct on every store: which of them comes first is decided by
    the impulse share and by clamps, not by the store, and the Optimize
    report's tornado says the same (review R42). The tab is for exploring
    the engine's local response; it feeds nothing else."""
    p = f"{pct * 100:.0f}"
    return [
        "INTERPRETATION",
        "-" * 44,
        "  Revenue is the product of visitors",
        "  (customers/hr x op. hours), conversion",
        "  and spend per converter, plus a small",
        "  impulse term. A +/-" + p + "% change in any of",
        "  those four moves it by about +/-" + p + "% on",
        "  any store, so their order above is set by",
        "  the impulse share and the clamps and is",
        "  not a finding about this store. The",
        "  impulse bars are small because impulse",
        "  spend is a small share of revenue.",
        "",
        # With a transactional calibration the visitor rate is derived as
        # buyers / assumed conversion, so moving the conversion alone is a
        # change in how many visitors buy, not in the assumed rate.
        "  Conversion is varied at a fixed visitor",
        "  rate. On a transactional calibration the",
        "  visitor rate is buyers / assumed conversion,",
        "  so this bar varies only one of the rate's",
        "  two uses, not the assumption itself.",
        "",
        "  For exploration only: the optimizer's",
        "  fitness does not read these swings.",
    ]


class SensitivityMixin:
    """Sensitivity analysis tab."""

    def _create_sensitivity_tab(self, notebook):
        tab = tk.Frame(notebook, bg='#002747')
        notebook.add(tab, text="Sensitivity Analysis")

        ctrl = tk.Frame(tab, bg='#002747')
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=10, pady=8)

        lbl_kw = dict(fg='white', bg='#002747', font=('Arial', 10))
        ent_kw = dict(width=8)

        tk.Label(ctrl, text="Perturbation %:", **lbl_kw).grid(row=0, column=0, padx=4)
        self._sa_pct = tk.DoubleVar(value=20.0)
        tk.Entry(ctrl, textvariable=self._sa_pct, **ent_kw).grid(row=0, column=1, padx=4)

        tk.Label(ctrl, text="Spider steps:", **lbl_kw).grid(row=0, column=2, padx=4)
        self._sa_steps = tk.IntVar(value=5)
        tk.Entry(ctrl, textvariable=self._sa_steps, **ent_kw).grid(row=0, column=3, padx=4)

        tk.Label(ctrl, text="MC iters/point:", **lbl_kw).grid(row=0, column=4, padx=4)
        self._sa_iters = tk.IntVar(value=500)
        tk.Entry(ctrl, textvariable=self._sa_iters, **ent_kw).grid(row=0, column=5, padx=4)

        tk.Label(ctrl, text="Projection days:", **lbl_kw).grid(row=0, column=6, padx=4)
        self._sa_days = tk.IntVar(value=30)
        tk.Entry(ctrl, textvariable=self._sa_days, **ent_kw).grid(row=0, column=7, padx=4)

        tk.Label(ctrl, text="Op. hours:", **lbl_kw).grid(row=0, column=8, padx=4)
        self._sa_hours = tk.DoubleVar(value=10.0)
        tk.Entry(ctrl, textvariable=self._sa_hours, **ent_kw).grid(row=0, column=9, padx=4)

        self._sa_progress = tk.StringVar(value="")
        tk.Label(ctrl, textvariable=self._sa_progress, fg='#4ECDC4',
                 bg='#002747', font=('Arial', 10, 'bold')).grid(row=0, column=11, padx=8)

        tk.Button(
            ctrl, text="Run Analysis", command=self._run_sensitivity_analysis,
            bg='#FF6B35', fg='white', font=('Arial', 11, 'bold'),
            relief=tk.RAISED, bd=3
        ).grid(row=0, column=10, padx=12)

        body = tk.Frame(tab, bg='#002747')
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=3)
        body.rowconfigure(0, weight=1)

        self._sa_text = tk.Text(
            body, bg='#001a33', fg='white', font=('Courier', 10),
            wrap=tk.WORD, state=tk.DISABLED, width=48
        )
        self._sa_text.grid(row=0, column=0, sticky='nsew', padx=(0, 5))

        chart_frame = tk.Frame(body, bg='#002747')
        chart_frame.grid(row=0, column=1, sticky='nsew')

        self._sa_fig = Figure(figsize=(11, 7), dpi=100)
        self._sa_fig.patch.set_facecolor('#002747')
        self._sa_canvas = FigureCanvasTkAgg(self._sa_fig, master=chart_frame)
        self._sa_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _run_sensitivity_analysis(self):
        A = self.customer_simulation.analytics
        # A trajectory-only dataset fills the calibration block with spatial
        # keys but no arrival or conversion rate, so it cannot stand in for
        # observed customers.
        cal = _calib(A)
        has_tx = bool(cal.get('arrivals_per_hour')) and 'conversion_rate' in cal
        if not has_tx and A.get('total_customers', 0) < 5:
            messagebox.showwarning(
                "Insufficient Data",
                "Run the real-time simulation first (at least 5 customers).",
                parent=self.tk_root
            )
            return

        params = self._extract_simulation_parameters()
        pct = max(5.0, self._sa_pct.get()) / 100.0
        n_steps = max(3, self._sa_steps.get())
        n_iter = max(100, self._sa_iters.get())
        n_days = max(1, self._sa_days.get())
        op_hours = max(1.0, self._sa_hours.get())
        wknd_mult = DEFAULT_WEEKEND_MULTIPLIER
        growth = 0.0

        param_defs = {
            'Customers/hr':    ('customers_per_hour',          None,       None),
            'Conversion rate': ('conversion_rate',             0.001,      0.999),
            'Revenue/customer':('rev_per_converting_customer', 0.01,       None),
            'Impulse rate':    ('impulse_rate',                0.0,        0.999),
            'Impulse value':   ('avg_impulse_value',           0.01,       None),
            # mc_engine draws basket sizes but never prices them, so this
            # parameter has no path to revenue.
            'Basket size (not monetized)': ('avg_basket_size',   0.5,        None),
            'Op. hours':       (None,                          1.0,        24.0),
            'Weekend mult.':   (None,                          1.0,        3.0),
        }

        def get_baseline(name):
            if name == 'Op. hours':
                return op_hours
            if name == 'Weekend mult.':
                return wknd_mult
            return params[param_defs[name][0]]

        def build_mc_kwargs(overrides):
            return dict(
                cph=overrides.get('customers_per_hour', params['customers_per_hour']),
                conv=overrides.get('conversion_rate', params['conversion_rate']),
                rev_mean=overrides.get('rev_per_converting_customer',
                                       params['rev_per_converting_customer']),
                # Scale observed rev_std proportionally when the mean is
                # overridden so we don't accidentally lose the calibrated
                # variance ratio. When mean is not perturbed, keep params['rev_std'].
                rev_std=(
                    params['rev_std'] * (
                        overrides['rev_per_converting_customer']
                        / max(params['rev_per_converting_customer'], 1e-6)
                    )
                    if 'rev_per_converting_customer' in overrides
                    else params['rev_std']
                ),
                imp_rate=overrides.get('impulse_rate', params['impulse_rate']),
                imp_val=overrides.get('avg_impulse_value', params['avg_impulse_value']),
                avg_bsk=overrides.get('avg_basket_size', params['avg_basket_size']),
                std_bsk=overrides.get('std_basket_size', params['std_basket_size']),
                observed_baskets=params['basket_sizes_observed'],
                n_days=n_days, n_iter=n_iter,
                op_hours=overrides.get('_op_hours', op_hours),
                wknd_mult=overrides.get('_wknd_mult', wknd_mult),
                monthly_growth=growth,
            )

        # Every evaluation replays the same random stream (common random
        # numbers), so a swing measures the parameter change rather than
        # independent Monte Carlo noise, and a parameter with no path to
        # revenue shows no swing. A private RandomState also keeps the live
        # simulation thread's draws out of the stream.
        sa_seed = 12345

        def run_mc(overrides):
            return mc_engine(**build_mc_kwargs(overrides),
                             rng=np.random.RandomState(sa_seed))

        self._sa_progress.set("Running baseline...")
        self.tk_root.update_idletasks()

        baseline = run_mc({})
        base_rev = baseline['mean']

        tornado = {}
        spider = {}
        # A parameter whose baseline is 0 (impulse rate, for instance, in a
        # live session before any impulse purchase) cannot be moved by a
        # multiplicative perturbation, and an additive one would add revenue
        # the base revenue per customer already carries. Report those
        # separately instead of ranking them at a zero swing, which reads as
        # "this channel does not matter".
        not_perturbable = [label for label in param_defs
                           if abs(get_baseline(label)) <= 0.0]
        active = {label: defs for label, defs in param_defs.items()
                  if label not in not_perturbable}
        total_runs = len(active) * (2 + n_steps)
        run_count = 0

        for label, (pkey, lo_clamp, hi_clamp) in active.items():
            bval = get_baseline(label)
            lo_val = bval * (1.0 - pct)
            hi_val = bval * (1.0 + pct)
            if lo_clamp is not None:
                lo_val = max(lo_clamp, lo_val)
            if hi_clamp is not None:
                hi_val = min(hi_clamp, hi_val)

            def make_override(val):
                ov = {}
                if label == 'Op. hours':
                    ov['_op_hours'] = val
                elif label == 'Weekend mult.':
                    ov['_wknd_mult'] = val
                else:
                    ov[pkey] = val
                return ov

            run_count += 1
            self._sa_progress.set(f"Tornado {label} low... ({run_count}/{total_runs})")
            self.tk_root.update_idletasks()
            lo_res = run_mc(make_override(lo_val))

            run_count += 1
            self._sa_progress.set(f"Tornado {label} high... ({run_count}/{total_runs})")
            self.tk_root.update_idletasks()
            hi_res = run_mc(make_override(hi_val))

            tornado[label] = {
                'baseline': bval,
                'lo_val': lo_val, 'hi_val': hi_val,
                'lo_rev': lo_res['mean'], 'hi_rev': hi_res['mean'],
                'swing': abs(hi_res['mean'] - lo_res['mean']),
            }

            fracs = np.linspace(-pct, pct, n_steps)
            spider_x = []
            spider_y = []
            for frac in fracs:
                run_count += 1
                if abs(frac) < 1e-12:
                    # The unperturbed point is the baseline run itself.
                    spider_x.append(0.0)
                    spider_y.append(base_rev)
                    continue
                val = bval * (1.0 + frac)
                if lo_clamp is not None:
                    val = max(lo_clamp, val)
                if hi_clamp is not None:
                    val = min(hi_clamp, val)

                self._sa_progress.set(f"Spider {label} {frac:+.0%}... ({run_count}/{total_runs})")
                self.tk_root.update_idletasks()

                res = run_mc(make_override(val))
                # Plot the change actually applied, which is smaller than the
                # requested fraction whenever a clamp binds.
                spider_x.append((val / bval - 1.0) * 100)
                spider_y.append(res['mean'])
            spider[label] = (spider_x, spider_y)

        self._sa_progress.set("Rendering...")
        self.tk_root.update_idletasks()

        self._display_sensitivity_results(tornado, spider, base_rev, pct, n_days,
                                          n_iter, not_perturbable)
        self._sa_progress.set("Done.")

    def _display_sensitivity_results(self, tornado, spider, base_rev, pct, n_days,
                                     n_iter, not_perturbable=()):
        sorted_t = sorted(tornado.items(), key=lambda kv: kv[1]['swing'], reverse=True)
        total_swing = sum(v['swing'] for v in tornado.values()) or 1.0

        lines = [
            "=" * 48,
            "   SENSITIVITY ANALYSIS REPORT",
            "=" * 48,
            "",
            f"Baseline {n_days}-day revenue:  ${base_rev:,.2f}",
            f"Perturbation range:          +/-{pct*100:.0f}%",
            f"MC iterations per point:     {n_iter:,}",
            "",
            "PARAMETER RANKING (by revenue swing)",
            "-" * 44,
        ]

        for rank, (label, d) in enumerate(sorted_t, 1):
            pct_impact = d['swing'] / max(base_rev, 1e-6) * 100
            contribution = d['swing'] / total_swing * 100
            lines.append(f"  #{rank}  {label}")
            lines.append(f"       Baseline:    {d['baseline']:.4g}")
            lines.append(f"       Low  ({d['lo_val']:.4g}):  ${d['lo_rev']:>10,.2f}")
            lines.append(f"       High ({d['hi_val']:.4g}):  ${d['hi_rev']:>10,.2f}")
            lines.append(f"       Swing:       ${d['swing']:>10,.2f}"
                         f"  ({pct_impact:.1f}% of baseline)")
            lines.append(f"       Contribution: {contribution:.1f}%")
            lines.append("")

        if not_perturbable:
            lines.append("NOT PERTURBABLE (baseline 0)")
            lines.append("-" * 44)
            for label in not_perturbable:
                lines.append(f"  {label}")
            lines.append("  A +/- % change of zero is still zero, so these")
            lines.append("  parameters are left out of the ranking rather")
            lines.append("  than shown with no influence on revenue.")
            lines.append("")

        lines.extend(interpretation_lines(pct))

        self._sa_text.config(state=tk.NORMAL)
        self._sa_text.delete('1.0', tk.END)
        self._sa_text.insert(tk.END, "\n".join(lines))
        self._sa_text.config(state=tk.DISABLED)

        self._sa_fig.clear()
        ax_c = 'white'

        # -- TORNADO CHART (top) ------------------------------
        ax1 = self._sa_fig.add_subplot(2, 1, 1)
        labels_t = [s[0] for s in sorted_t]
        y_pos = np.arange(len(labels_t))

        lo_deltas = [tornado[l]['lo_rev'] - base_rev for l in labels_t]
        hi_deltas = [tornado[l]['hi_rev'] - base_rev for l in labels_t]

        bars_lo = ax1.barh(y_pos, lo_deltas, height=0.5, color='#FF6B35',
                           alpha=0.85, label=f'-{pct*100:.0f}%')
        bars_hi = ax1.barh(y_pos, hi_deltas, height=0.5, color='#4ECDC4',
                           alpha=0.85, label=f'+{pct*100:.0f}%')

        ax1.axvline(0, color='white', linewidth=0.8, linestyle='--')
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(labels_t, fontsize=8)
        ax1.set_xlabel("Revenue change from baseline ($)", color=ax_c, fontsize=9)
        ax1.set_title("Tornado Chart — Revenue Sensitivity", color=ax_c, fontsize=11)
        ax1.legend(fontsize=7, loc='lower right')
        ax1.set_facecolor('#001a33')
        ax1.tick_params(colors=ax_c, labelsize=7)
        ax1.invert_yaxis()

        for bar, val in zip(bars_lo, lo_deltas):
            if abs(val) > 0:
                ax1.text(val - abs(val)*0.05, bar.get_y() + bar.get_height()/2,
                         f'${val:+,.0f}', va='center', ha='right',
                         fontsize=6, color='white')
        for bar, val in zip(bars_hi, hi_deltas):
            if abs(val) > 0:
                ax1.text(val + abs(val)*0.05, bar.get_y() + bar.get_height()/2,
                         f'${val:+,.0f}', va='center', ha='left',
                         fontsize=6, color='white')

        # -- SPIDER / SENSITIVITY CURVES (bottom) -------------
        ax2 = self._sa_fig.add_subplot(2, 1, 2)
        cmap = plt.cm.tab10
        for idx, (label, (sx, sy)) in enumerate(spider.items()):
            pct_change_rev = [(v - base_rev) / max(base_rev, 1e-6) * 100 for v in sy]
            ax2.plot(sx, pct_change_rev, marker='o', markersize=3,
                     linewidth=1.5, color=cmap(idx / max(len(spider)-1, 1)),
                     label=label)

        ax2.axhline(0, color='white', linewidth=0.6, linestyle='--')
        ax2.axvline(0, color='white', linewidth=0.6, linestyle='--')
        ax2.set_xlabel("Parameter change (%)", color=ax_c, fontsize=9)
        ax2.set_ylabel("Revenue change (%)", color=ax_c, fontsize=9)
        ax2.set_title("Sensitivity Curves — All Parameters", color=ax_c, fontsize=11)
        ax2.legend(fontsize=6, loc='best', ncol=2)
        ax2.set_facecolor('#001a33')
        ax2.tick_params(colors=ax_c, labelsize=7)
        ax2.grid(True, alpha=0.15, color='white')

        self._sa_fig.tight_layout(pad=2.5)
        self._sa_canvas.draw_idle()












        # -------------------------------------------------------------
    # MARKOV CHAIN REVENUE MODEL TAB
    # -------------------------------------------------------------

