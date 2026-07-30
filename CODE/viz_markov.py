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
from sim_calibration import (
    build_empirical_transition_matrix,
    compute_absorbing_analysis,
    extract_simulation_parameters,
    transient_occupancy_distribution,
    _calibrated,
)

import copy
from scipy import stats as sp_stats


class MarkovMixin:
    """Markov-chain customer-flow modeling tab."""

    def _create_markov_tab(self, notebook):
        tab = tk.Frame(notebook, bg='#002747')
        notebook.add(tab, text="Markov Chain Model")

        ctrl = tk.Frame(tab, bg='#002747')
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=10, pady=8)

        lbl_kw = dict(fg='white', bg='#002747', font=('Arial', 10))
        ent_kw = dict(width=8)

        tk.Label(ctrl, text="Customers/batch:", **lbl_kw).grid(row=0, column=0, padx=4)
        self._mk_batch = tk.IntVar(value=1000)
        tk.Entry(ctrl, textvariable=self._mk_batch, **ent_kw).grid(row=0, column=1, padx=4)

        tk.Label(ctrl, text="Max steps:", **lbl_kw).grid(row=0, column=2, padx=4)
        self._mk_steps = tk.IntVar(value=200)
        tk.Entry(ctrl, textvariable=self._mk_steps, **ent_kw).grid(row=0, column=3, padx=4)

        tk.Label(ctrl, text="Projection days:", **lbl_kw).grid(row=0, column=4, padx=4)
        self._mk_days = tk.IntVar(value=30)
        tk.Entry(ctrl, textvariable=self._mk_days, **ent_kw).grid(row=0, column=5, padx=4)

        tk.Label(ctrl, text="Op. hours:", **lbl_kw).grid(row=0, column=6, padx=4)
        self._mk_hours = tk.DoubleVar(value=10.0)
        tk.Entry(ctrl, textvariable=self._mk_hours, **ent_kw).grid(row=0, column=7, padx=4)

        self._mk_progress = tk.StringVar(value="")
        tk.Label(ctrl, textvariable=self._mk_progress, fg='#4ECDC4',
                 bg='#002747', font=('Arial', 10, 'bold')).grid(row=0, column=9, padx=8)

        tk.Button(
            ctrl, text="Run Markov Model", command=self._run_markov_model,
            bg='#9B59B6', fg='white', font=('Arial', 11, 'bold'),
            relief=tk.RAISED, bd=3
        ).grid(row=0, column=8, padx=12)

        body = tk.Frame(tab, bg='#002747')
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=3)
        body.rowconfigure(0, weight=1)

        self._mk_text = tk.Text(
            body, bg='#001a33', fg='white', font=('Courier', 10),
            wrap=tk.WORD, state=tk.DISABLED, width=52
        )
        self._mk_text.grid(row=0, column=0, sticky='nsew', padx=(0, 5))

        chart_frame = tk.Frame(body, bg='#002747')
        chart_frame.grid(row=0, column=1, sticky='nsew')

        self._mk_fig = Figure(figsize=(12, 8), dpi=100)
        self._mk_fig.patch.set_facecolor('#002747')
        self._mk_canvas = FigureCanvasTkAgg(self._mk_fig, master=chart_frame)
        self._mk_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _build_transition_matrix(self):
        states, T, meta = build_empirical_transition_matrix(
            self.customer_simulation.analytics
        )
        self.customer_simulation.analytics['markov_matrix_meta'] = meta
        return states, T

    def _compute_absorbing_analysis(self, states, T):
        return compute_absorbing_analysis(states, T)

    def _simulate_markov_batch(self, states, T, n_customers, max_steps):
        si = {s: i for i, s in enumerate(states)}
        n = len(states)

        current = np.zeros(n_customers, dtype=int)

        cumT = np.cumsum(T, axis=1)

        state_counts = np.zeros((max_steps, n))
        state_counts[0, si['entering']] = n_customers

        absorbing_mask = np.array([
            states[i] in ('purchased', 'abandoned') for i in range(n)
        ])

        for step in range(1, max_steps):
            rng = np.random.random(n_customers)
            for i in range(n):
                mask = current == i
                if not mask.any():
                    continue
                r = rng[mask]
                next_states = np.searchsorted(cumT[i], r)
                next_states = np.clip(next_states, 0, n - 1)
                current[mask] = next_states

            for i in range(n):
                state_counts[step, i] = np.sum(current == i)

            active = ~absorbing_mask[current]
            if not active.any():
                state_counts[step:] = state_counts[step]
                break

        return state_counts

    def _run_markov_model(self):
        A = self.customer_simulation.analytics
        if not _calibrated(A) and A.get('total_customers', 0) < 5:
            messagebox.showwarning(
                "Insufficient Data",
                "Run the real-time simulation first (at least 5 customers).",
                parent=self.tk_root
            )
            return

        self._mk_progress.set("Building transition matrix...")
        self.tk_root.update_idletasks()

        states, T = self._build_transition_matrix()
        si = {s: i for i, s in enumerate(states)}

        self._mk_progress.set("Absorbing chain analysis...")
        self.tk_root.update_idletasks()
        absorb = self._compute_absorbing_analysis(states, T)

        n_batch = max(100, self._mk_batch.get())
        max_steps = max(20, self._mk_steps.get())
        n_days = max(1, self._mk_days.get())
        op_hours = max(1.0, self._mk_hours.get())

        self._mk_progress.set(f"Simulating {n_batch:,} customers...")
        self.tk_root.update_idletasks()
        state_counts = self._simulate_markov_batch(states, T, n_batch, max_steps)

        params = extract_simulation_parameters(self.customer_simulation)
        cph = params['customers_per_hour']
        rev_per = params['rev_per_converting_customer']
        imp_rate = params['impulse_rate']
        imp_val = params['avg_impulse_value']

        p_purchase = absorb['p_purchase_from_entering']

        daily_customers = cph * op_hours
        daily_purchasers = daily_customers * p_purchase
        daily_base_rev = daily_purchasers * rev_per
        daily_impulse_rev = daily_purchasers * imp_rate * imp_val
        daily_total_rev = daily_base_rev + daily_impulse_rev

        steady_dict = transient_occupancy_distribution(
            self.customer_simulation.analytics
        )
        steady = np.zeros(len(states))
        si = {s: i for i, s in enumerate(states)}
        for s, p in steady_dict.items():
            if s in si:
                steady[si[s]] = p
        for s in ('purchased', 'abandoned'):
            if s in si:
                steady[si[s]] = 0.0

        projection_days = np.arange(1, n_days + 1)
        cumulative_rev = np.cumsum(np.full(n_days, daily_total_rev))

        growth_scenarios = {}
        for label, growth_rate in [("Pessimistic (-5%/mo)", -0.05),
                                    ("Baseline (0%)", 0.0),
                                    ("Optimistic (+5%/mo)", 0.05),
                                    ("Aggressive (+10%/mo)", 0.10)]:
            daily = np.array([
                daily_total_rev * (1.0 + growth_rate * (d / 30.0))
                for d in range(n_days)
            ])
            growth_scenarios[label] = np.cumsum(daily)

        results = {
            'states': states,
            'T': T,
            'absorb': absorb,
            'state_counts': state_counts,
            'steady': steady,
            'daily_customers': daily_customers,
            'daily_purchasers': daily_purchasers,
            'daily_base_rev': daily_base_rev,
            'daily_impulse_rev': daily_impulse_rev,
            'daily_total_rev': daily_total_rev,
            'cumulative_rev': cumulative_rev,
            'projection_days': projection_days,
            'growth_scenarios': growth_scenarios,
            'n_days': n_days,
            'rev_per': rev_per,
            'imp_rate': imp_rate,
            'imp_val': imp_val,
            'p_purchase': p_purchase,
            'markov_meta': self.customer_simulation.analytics.get(
                'markov_matrix_meta', {}
            ),
        }

        self._display_markov_results(results)
        self._mk_progress.set("Done.")

    def _display_markov_results(self, R):
        states = R['states']
        T = R['T']
        ab = R['absorb']
        sc = R['state_counts']
        steady = R['steady']
        si = {s: i for i, s in enumerate(states)}
        nd = R['n_days']

        lines = [
            "=" * 50,
            "   MARKOV CHAIN REVENUE MODEL",
            "=" * 50,
            "",
            "TRANSITION MATRIX",
            "-" * 50,
        ]

        hdr = "          " + " ".join(f"{s[:6]:>7s}" for s in states)
        lines.append(hdr)
        for i, s in enumerate(states):
            row_str = " ".join(f"{T[i,j]:7.3f}" for j in range(len(states)))
            lines.append(f"  {s[:8]:<8s}  {row_str}")

        lines += [
            "",
            "ABSORBING CHAIN ANALYSIS",
            "-" * 50,
            f"  P(Purchase | Enter):    {ab['p_purchase_from_entering']:.4f}"
            f"  ({ab['p_purchase_from_entering']*100:.1f}%)",
            f"  P(Abandon  | Enter):    {ab['p_abandon_from_entering']:.4f}"
            f"  ({ab['p_abandon_from_entering']*100:.1f}%)",
            f"  Expected steps to absorb: {ab['expected_steps_from_entering']:.1f}",
            "",
            "  Expected time in each state (from entering):",
        ]
        for s, t in ab['time_in_each_state'].items():
            bar = "#" * min(int(t * 3), 30)
            lines.append(f"    {s:<14s} {t:6.2f} steps  {bar}")

        meta = R.get('markov_meta', {})
        lines += [
            "",
            "TRANSITION MATRIX SOURCE",
            "-" * 50,
            f"  Empirical (logged ABM):  {'yes' if meta.get('empirical') else 'no (prior)'}",
            f"  Logged transitions:      {meta.get('total_transitions', 0)}",
            "",
            "TRANSIENT STATE OCCUPANCY (empirical, pre-absorption)",
            "-" * 50,
        ]
        for i, s in enumerate(states):
            if s in ('purchased', 'abandoned'):
                continue
            pct = steady[i] * 100
            bar = "#" * min(int(pct * 2), 40)
            lines.append(f"  {s:<14s} {pct:6.2f}%  {bar}")

        lines += [
            "",
            "REVENUE PROJECTIONS (Analytical)",
            "-" * 50,
            f"  Daily customers:       {R['daily_customers']:>8.1f}",
            f"  Daily purchasers:      {R['daily_purchasers']:>8.1f}",
            f"  Daily base revenue:    ${R['daily_base_rev']:>10,.2f}",
            f"  Daily impulse revenue: ${R['daily_impulse_rev']:>10,.2f}",
            f"  Daily TOTAL revenue:   ${R['daily_total_rev']:>10,.2f}",
            "",
            f"  Weekly projection:     ${R['daily_total_rev']*7:>10,.2f}",
            f"  Monthly projection:    ${R['daily_total_rev']*30:>10,.2f}",
            f"  Annual projection:     ${R['daily_total_rev']*365:>10,.2f}",
            "",
            "GROWTH SCENARIOS (cumulative at day {})".format(nd),
            "-" * 50,
        ]
        for label, cum in R['growth_scenarios'].items():
            lines.append(f"  {label:<26s} ${cum[-1]:>12,.2f}")

        lines += [
            "",
            "KEY INSIGHT",
            "-" * 50,
            f"  Each 1% increase in conversion rate adds",
            f"  ~${R['daily_customers'] * 0.01 * R['rev_per']:,.2f}/day"
            f"  (${R['daily_customers'] * 0.01 * R['rev_per'] * 365:,.0f}/year)",
            f"  to base revenue.",
        ]

        self._mk_text.config(state=tk.NORMAL)
        self._mk_text.delete('1.0', tk.END)
        self._mk_text.insert(tk.END, "\n".join(lines))
        self._mk_text.config(state=tk.DISABLED)

        self._mk_fig.clear()
        wc = 'white'

        # -- 1) Transition matrix heatmap (top-left) ----------
        ax1 = self._mk_fig.add_subplot(2, 2, 1)
        short = [s[:5] for s in states]
        im = ax1.imshow(T, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
        ax1.set_xticks(range(len(states)))
        ax1.set_xticklabels(short, fontsize=7, rotation=45, color=wc)
        ax1.set_yticks(range(len(states)))
        ax1.set_yticklabels(short, fontsize=7, color=wc)
        for i in range(len(states)):
            for j in range(len(states)):
                v = T[i, j]
                if v > 0.005:
                    ax1.text(j, i, f"{v:.2f}", ha='center', va='center',
                             fontsize=6, color='black' if v > 0.5 else 'white')
        ax1.set_title("Transition Matrix", color=wc, fontsize=10)
        ax1.set_facecolor('#001a33')
        self._mk_fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

        # -- 2) State evolution over steps (top-right) ---------
        ax2 = self._mk_fig.add_subplot(2, 2, 2)
        colors_map = {
            'entering': '#00ff00', 'moving': '#3498db', 'shopping': '#f39c12',
            'checking_out': '#e74c3c', 'exiting': '#9b59b6',
            'purchased': '#2ecc71', 'abandoned': '#e74c3c'
        }
        max_step = min(sc.shape[0], int(np.where(
            np.abs(np.diff(sc[:, si['purchased']])) > 0)[0][-1] + 20)
            if np.any(np.diff(sc[:, si['purchased']]) > 0) else sc.shape[0])

        for idx, s in enumerate(states):
            if s in ('purchased', 'abandoned'):
                continue
            ax2.plot(range(max_step), sc[:max_step, idx] / sc[0].sum() * 100,
                     label=s, color=colors_map.get(s, '#ffffff'), linewidth=1.5)
        ax2.set_xlabel("Markov Steps", color=wc, fontsize=9)
        ax2.set_ylabel("% of Customers", color=wc, fontsize=9)
        ax2.set_title("State Evolution (Transient)", color=wc, fontsize=10)
        ax2.legend(fontsize=6, loc='upper right')
        ax2.set_facecolor('#001a33')
        ax2.tick_params(colors=wc, labelsize=7)
        ax2.grid(True, alpha=0.15, color='white')

        # -- 3) Absorption curves (bottom-left) ---------------
        ax3 = self._mk_fig.add_subplot(2, 2, 3)
        purchased_pct = sc[:max_step, si['purchased']] / sc[0].sum() * 100
        abandoned_pct = sc[:max_step, si['abandoned']] / sc[0].sum() * 100
        ax3.fill_between(range(max_step), purchased_pct,
                         color='#2ecc71', alpha=0.6, label='Purchased')
        ax3.fill_between(range(max_step), purchased_pct,
                         purchased_pct + abandoned_pct,
                         color='#e74c3c', alpha=0.6, label='Abandoned')
        ax3.set_xlabel("Markov Steps", color=wc, fontsize=9)
        ax3.set_ylabel("Cumulative %", color=wc, fontsize=9)
        ax3.set_title("Absorption: Purchase vs Abandon", color=wc, fontsize=10)
        ax3.legend(fontsize=7)
        ax3.set_facecolor('#001a33')
        ax3.tick_params(colors=wc, labelsize=7)
        ax3.set_ylim(0, 105)
        ax3.grid(True, alpha=0.15, color='white')

        # -- 4) Growth scenario projections (bottom-right) ----
        ax4 = self._mk_fig.add_subplot(2, 2, 4)
        scenario_colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
        for idx, (label, cum) in enumerate(R['growth_scenarios'].items()):
            ax4.plot(R['projection_days'], cum, label=label,
                     color=scenario_colors[idx % len(scenario_colors)],
                     linewidth=1.8)
        ax4.set_xlabel("Day", color=wc, fontsize=9)
        ax4.set_ylabel("Cumulative Revenue ($)", color=wc, fontsize=9)
        ax4.set_title("Revenue Scenarios (Markov-based)", color=wc, fontsize=10)
        ax4.legend(fontsize=6, loc='upper left')
        ax4.set_facecolor('#001a33')
        ax4.tick_params(colors=wc, labelsize=7)
        ax4.grid(True, alpha=0.15, color='white')

        self._mk_fig.tight_layout(pad=2.0)
        self._mk_canvas.draw_idle()



        # -------------------------------------------------------------
    #  GENETIC ALGORITHM LAYOUT OPTIMIZATION TAB
    # -----------------------------------------

