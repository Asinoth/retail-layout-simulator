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


class OptimizeResultsMixin:
    """Detailed optimization-results report dialog (single very large method)."""

    def _show_optimization_results(self, changes, performance_data):
        """
        Comprehensive sectioned optimization report dialog,
        structured like an academic paper with numbered sections and bibliography.
        """
        sim = self.customer_simulation
        A   = sim.analytics
        pipe = A.get('_opt_pipeline_data', {})

        results_dialog = tk.Toplevel(self.tk_root)
        results_dialog.title("Layout Optimization — Full Analysis Report")
        results_dialog.geometry("950x800")
        results_dialog.transient(self.tk_root)

        results_dialog.update_idletasks()
        x = (results_dialog.winfo_screenwidth() // 2) - 475
        y = (results_dialog.winfo_screenheight() // 2) - 400


        results_dialog.geometry(f"950x800+{x}+{y}")

        main_frame = tk.Frame(results_dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        scrollbar = tk.Scrollbar(main_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        report_text = tk.Text(
            main_frame, yscrollcommand=scrollbar.set,
            font=('Courier', 10), wrap=tk.WORD, bg='#1a1a2e', fg='#e0e0e0',
            insertbackground='white', padx=15, pady=10
        )
        report_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=report_text.yview)

        report_text.tag_configure('title', font=('Arial', 16, 'bold'), foreground='#FF6B35',
                                   justify='center')
        report_text.tag_configure('section', font=('Arial', 12, 'bold'), foreground='#4ECDC4')
        report_text.tag_configure('subsection', font=('Arial', 10, 'bold'), foreground='#f39c12')
        report_text.tag_configure('good', foreground='#2ecc71')
        report_text.tag_configure('bad', foreground='#e74c3c')
        report_text.tag_configure('neutral', foreground='#bdc3c7')
        report_text.tag_configure('highlight', foreground='#f1c40f', font=('Courier', 10, 'bold'))
        report_text.tag_configure('divider', foreground='#4ECDC4')
        report_text.tag_configure('cite', foreground='#a29bfe', font=('Courier', 9, 'italic'))
        report_text.tag_configure('bibref', foreground='#74b9ff', font=('Courier', 9))

        def divider():
            report_text.insert(tk.END, "\n" + "=" * 80 + "\n\n", 'divider')

        def section(num, title):
            report_text.insert(tk.END, f"\n{num}. {title}\n", 'section')
            report_text.insert(tk.END, "-" * 70 + "\n", 'divider')

        def subsection(num, title):
            report_text.insert(tk.END, f"\n  {num} {title}\n", 'subsection')

        def line(text, tag='neutral'):
            report_text.insert(tk.END, f"    {text}\n", tag)

        def cite(text):


            report_text.insert(tk.END, f"    {text}\n", 'cite')

        def bibline(text):
            report_text.insert(tk.END, f"    {text}\n", 'bibref')

        real_pre  = pipe.get('real_pre_rev', 0)
        real_post = pipe.get('real_post_rev', 0)
        # The live lift is undefined when the PRE window collected no
        # revenue; the pipeline records the reason in the analytics.
        real_lift = (real_post - real_pre) / real_pre * 100 if real_pre > 0 else None
        real_lift_note = A.get('optimization_impact_note', '')

        # Two different baselines: mc_baseline is the CURRENT layout scored
        # through the same score -> parameter transform as the optimized
        # projection (so the lift isolates the layout change), while
        # mc_calibrated_baseline is the run on the raw calibrated
        # parameters, which is what the tornado swings were measured around.
        mc_base = pipe.get('mc_baseline', {})
        mc_cal  = pipe.get('mc_calibrated_baseline', mc_base)
        mc_opt  = pipe.get('mc_optimized', {})
        mc_lift = (mc_opt.get('mean', 0) - mc_base.get('mean', 0)) / max(mc_base.get('mean', 1), 1e-6) * 100

        ga_best_fit = pipe.get('ga_best_fit', 0)
        ga_curr_fit = pipe.get('ga_current_fit', 0)
        ga_lift = (ga_best_fit - ga_curr_fit) / max(ga_curr_fit, 1e-6) * 100

        n_changes = len(changes) if changes else 0
        base_params = pipe.get('pre_snapshot', {}).get('params', {})
        tornado = pipe.get('tornado', {})
        sorted_t = sorted(tornado.items(), key=lambda kv: kv[1]['swing'], reverse=True)
        total_swing = sum(v['swing'] for v in tornado.values()) or 1.0
        markov_absorb = pipe.get('markov_absorb', {})
        markov_steady = pipe.get('markov_steady', {})
        curr_bk = pipe.get('ga_current_breakdown', {})
        best_bk = pipe.get('ga_best_breakdown', {})

        # --------------------------------------------------------
        # TITLE
        # --------------------------------------------------------
        report_text.insert(tk.END, "\n\n", 'neutral')
        report_text.insert(tk.END, "SHOP LAYOUT OPTIMIZATION\nFULL ANALYSIS REPORT\n", 'title')
        report_text.insert(tk.END, f"\n    Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n", 'neutral')
        divider()

        # --------------------------------------------------------
        # 1. EXECUTIVE SUMMARY
        # --------------------------------------------------------
        section("1", "EXECUTIVE SUMMARY")

        line(f"Items repositioned:          {n_changes}")
        if real_lift is not None:
            line(f"Real-time revenue lift:      {real_lift:+.1f}%  (${real_pre:.2f} -> ${real_post:.2f})",
                 'good' if real_lift > 0 else 'bad')
        else:
            line(f"Real-time revenue lift:      n/a  (${real_pre:.2f} -> ${real_post:.2f})")
            if real_lift_note:
                line(f"  {real_lift_note}")
        line(f"MC projected 30-day lift:    {mc_lift:+.1f}%  (${mc_base.get('mean',0):,.0f} -> ${mc_opt.get('mean',0):,.0f})",
             'good' if mc_lift > 0 else 'bad')
        line("  (current vs optimized layout, both under the layout model)")
        line(f"GA fitness improvement:      {ga_lift:+.1f}%", 'good' if ga_lift > 0 else 'bad')
        line(f"Conversion rate:             {performance_data.get('conversion_rate',0)*100:.1f}%")
        line(f"Current total revenue:       ${performance_data.get('total_revenue',0):,.2f}")
        line("")
        line("Methodology: PRE/POST measurement windows with GA-optimized")
        line("layout placement, validated by Monte Carlo simulation,")
        line("Markov chain customer flow analysis, sensitivity analysis,")
        line("and A/B statistical testing (Welch's t, KS, Cohen's d).")

        # --------------------------------------------------------
        # 2. PRE-OPTIMIZATION BASELINE
        # --------------------------------------------------------
        section("2","PRE-OPTIMIZATION BASELINE")

        subsection("2.1", "Observed Simulation Metrics")
        line(f"Customers/hour:              {base_params.get('customers_per_hour',0):.1f}")
        line(f"Conversion rate:             {base_params.get('conversion_rate',0)*100:.1f}%")
        line(f"Abandonment rate:            {base_params.get('abandonment_rate',0)*100:.1f}%")
        line(f"Avg basket size:             {base_params.get('avg_basket_size',0):.1f} items")
        line(f"Revenue/converting customer: ${base_params.get('rev_per_converting_customer',0):.2f}")
        line(f"Impulse rate:                {base_params.get('impulse_rate',0)*100:.1f}%")
        line(f"Avg impulse value:           ${base_params.get('avg_impulse_value',0):.2f}")
        line(f"Avg queue wait time:         {base_params.get('avg_queue_time',0):.1f}s")
        line(f"PRE-window revenue:          ${real_pre:.2f}")

        subsection("2.2", "Monte Carlo Baseline (raw calibrated parameters, "
                          "30-day projection)")
        line(f"Mean revenue:                ${mc_cal.get('mean',0):>12,.2f}")
        line(f"Std deviation:               ${mc_cal.get('std',0):>12,.2f}")
        line(f"5th percentile (worst):      ${mc_cal.get('p5',0):>12,.2f}")
        line(f"95th percentile (best):      ${mc_cal.get('p95',0):>12,.2f}")
        line(f"Daily average:               ${mc_cal.get('mean',0)/30:>12,.2f}")
        line(f"Annual projection:           ${mc_cal.get('mean',0)/30*365:>12,.0f}")
        line("")
        line("Run on the parameters in 2.1 as measured, without the layout")
        line("score -> conversion / impulse / basket transform that Secs. 7")
        line("and 8 apply to both compared layouts.")

        subsection("2.3", "Area Revenue Distribution")
        area_shares = base_params.get('area_revenue_shares', {})
        if area_shares:
            sorted_areas = sorted(area_shares.items(), key=lambda kv: kv[1], reverse=True)
            for area, share in sorted_areas:
                bar = "#" * min(int(share * 50), 40)
                line(f"  {area:<20s} {share*100:5.1f}%  {bar}")

        subsection("2.4", "Dwell Time by Zone")
        dwell_data = base_params.get('avg_dwell_by_zone', {})
        if dwell_data:
            for zone, avg_t in sorted(dwell_data.items(), key=lambda kv: kv[1], reverse=True):
                bar = "#" * min(int(avg_t / 2), 30)
                line(f"  {zone:<20s} {avg_t:5.1f}s  {bar}")
        else:
            line("  No dwell time data collected yet.")

        # --------------------------------------------------------
        # 3. SENSITIVITY ANALYSIS
        # --------------------------------------------------------
        section("3", "SENSITIVITY ANALYSIS")

        line("Parameter ranking by revenue impact (+/-20% perturbation):", 'highlight')
        cite("[Uses tornado-chart methodology per Saltelli et al., 2008]")
        line("Percentages are shares of the Sec. 2.2 raw calibrated baseline.")
        line("")
        # The swings were measured by perturbing the raw calibrated
        # parameters, so they are a share of the baseline those runs used
        # (Sec. 2.2), not of the layout-model projection in Secs. 7 and 8.
        for rank, (lbl, d) in enumerate(sorted_t, 1):
            pct_impact = d['swing'] / max(mc_cal.get('mean', 1), 1e-6) * 100
            contrib = d['swing'] / total_swing * 100
            line(f"  #{rank}  {lbl:<24s}  swing=${d['swing']:>10,.0f}  ({pct_impact:.1f}%)  contrib={contrib:.0f}%")

        if sorted_t:
            top = sorted_t[0][0]
            line("")


            line(f"Key insight: '{top}' is the single largest revenue driver.", 'highlight')
            if len(sorted_t) >= 3:
                top3 = [s[0] for s in sorted_t[:3]]
                line(f"Top 3 drivers: {', '.join(top3)}")
                top3_pct = sum(tornado[k]['swing'] for k in top3) / total_swing * 100
                line(f"Together they account for {top3_pct:.0f}% of total sensitivity.")

        line("")
        cite("These sensitivity weights dynamically adjust GA fitness scoring,")
        cite("so the optimizer focuses on the parameters with most leverage.")

        # --------------------------------------------------------
        # 4. MARKOV CHAIN ANALYSIS
        # --------------------------------------------------------
        section("4", "MARKOV CHAIN CUSTOMER FLOW ANALYSIS")
        cite("[Absorbing Markov chain model per Kemeny & Snell, 1976]")

        subsection("4.1", "Absorbing Chain Results")
        line(f"P(Purchase | Enter):         {markov_absorb.get('p_purchase_from_entering',0)*100:.1f}%",
             'good' if markov_absorb.get('p_purchase_from_entering',0) > 0.5 else 'neutral')
        line(f"P(Abandon  | Enter):         {markov_absorb.get('p_abandon_from_entering',0)*100:.1f}%",
             'bad' if markov_absorb.get('p_abandon_from_entering',0) > 0.3 else 'neutral')
        line(f"Expected steps to absorb:    {markov_absorb.get('expected_steps_from_entering',0):.1f}")

        subsection("4.2", "Time in Each State (from entering)")
        for state, t_val in markov_absorb.get('time_in_each_state', {}).items():
            bar = "#" * min(int(t_val * 3), 30)
            line(f"  {state:<16s} {t_val:6.2f} steps  {bar}")

        subsection("4.3", "Steady-State Distribution")
        for state, prob in markov_steady.items():
            bar = "#" * min(int(prob * 100 * 2), 40)
            line(f"  {state:<16s} {prob*100:6.2f}%  {bar}")

        subsection("4.4", "Markov Revenue Projection")
        from retail_literature import DEFAULT_WEEKEND_MULTIPLIER as _WKND
        cph = base_params.get('customers_per_hour', 10)
        # Gross spend per purchaser (base plus expected impulse) and the MC
        # engine's weekday/weekend traffic mix, so this projection is
        # comparable with the MC baseline in Secs. 2.2 and 7.
        rev_per = base_params.get(
            'rev_per_customer_gross',
            base_params.get('rev_per_converting_customer', 20)
            + base_params.get('impulse_rate', 0) * base_params.get('avg_impulse_value', 0))
        mk_chain = markov_absorb.get('p_purchase_from_entering', 0.5)
        # With a transactional dataset the visitor rate is the observed buyer
        # rate divided by the assumed conversion, so purchasers must use that
        # same conversion; the chain's absorption probability (live agents or
        # a prior) would count each observed buyer mk_chain / conversion times.
        cal = _calib(A)
        has_tx = bool(cal.get('arrivals_per_hour')) and 'conversion_rate' in cal
        if has_tx:
            mk_p = base_params.get('conversion_rate', mk_chain)
            mk_src = "calibrated conversion"
        else:
            mk_p = mk_chain
            mk_src = "Markov chain, Sec. 4.1"
        daily_cust = cph * 10 * (5 + 2 * _WKND) / 7
        daily_rev_mk = daily_cust * mk_p * rev_per
        line(f"Purchase probability used:   {mk_p*100:.1f}%  ({mk_src})")
        line(f"Daily customers:             {daily_cust:.0f}")
        line(f"Daily purchasers:            {daily_cust * mk_p:.0f}")
        line(f"Markov daily revenue:        ${daily_rev_mk:,.2f}")
        line(f"Markov monthly projection:   ${daily_rev_mk * 30:,.0f}")
        line(f"Markov annual projection:    ${daily_rev_mk * 365:,.0f}")
        line("")
        if has_tx:
            cite("The visitor rate was derived from the dataset's buyers with the")
            cite("calibrated conversion, so these purchasers use that rate, not")
            cite("the chain's P(Purchase | Enter) from 4.1.")
        else:
            cite("With no transactional calibration the chain's P(Purchase | Enter)")
            cite("is the observed flow's own purchase estimate. It is reported here")
            cite("and does not rescale the GA's conversion rate.")

        # --------------------------------------------------------
        # 5. GA OPTIMIZATION RESULTS
        # --------------------------------------------------------
        section("5", "GENETIC ALGORITHM OPTIMIZATION")
        from retail_literature import GA_W_SECTION_COMPLIANCE, GA_W_ACCESSIBILITY
        cite("[Composite criteria per Larson 2005 / Hui 2009 / Hui-Inman 2013 /")
        cite(" Sorensen 2009; five spatial criteria equal-weighted per Dawes 1979,")
        cite(f" section compliance {GA_W_SECTION_COMPLIANCE:.2f}, entrance proximity "
             f"{GA_W_ACCESSIBILITY:.2f} — retail_literature.py]")

        subsection("5.1", "GA Configuration")
        line(f"Population size:             {pipe.get('ga_pop_size', 0)}")
        line(f"Generations:                 {pipe.get('ga_n_gens', 0)}")
        fit_kind = pipe.get('ga_fitness_kind', 'analytic_surrogate')
        if fit_kind == 'analytic_surrogate':
            line("Fitness evaluation:          analytic surrogate "
                 "(layout score -> cited elasticities -> expected revenue)")
        else:
            line(f"MC iterations/evaluation:    {pipe.get('ga_mc_iters', 0)}")
        line(f"Projection horizon:          {pipe.get('ga_mc_days', 30)} days")
        line(f"Movable items:               {len(pipe.get('ga_item_names', []))}")
        line(f"Selection:                   Tournament (k=5)")
        line(f"Crossover:                   uniform per-gene mask + arithmetic blend child "
             f"(alpha~U(0.1,0.5))")
        line(f"Mutation:                    Gaussian, sigma = 12% of section extent "
             f"(min 0.2 m; unsectioned items 8% of max(W,H))")
        line(f"Fitness:                     layout-adjusted conversion x "
             f"SA-weighted cited elasticities")

        subsection("5.2", "Revenue Comparison (GA)")
        line(f"Current layout proj revenue: ${ga_curr_fit:>12,.2f}")
        line(f"Optimized layout projection: ${ga_best_fit:>12,.2f}")
        line(f"Improvement:                 {ga_lift:+.1f}%", 'good' if ga_lift> 0 else 'bad')
        ga_mc_d = max(pipe.get('ga_mc_days', 30), 1)
        line(f"Daily delta:                 ${(ga_best_fit - ga_curr_fit) / ga_mc_d:+,.2f}")
        line(f"Annual projection delta:     ${(ga_best_fit - ga_curr_fit) / ga_mc_d * 365:+,.0f}",
             'good' if ga_lift > 0 else 'bad')

        subsection("5.3", "Layout Quality Breakdown (Before -> After)")
        for key in best_bk:
            if key == 'composite':
                continue
            cv = curr_bk.get(key, 0)
            bv = best_bk.get(key, 0)
            delta_v = bv - cv
            tag = 'good' if delta_v > 0.001 else ('bad' if delta_v < -0.001 else 'neutral')
            if 'penalty' in key:
                tag = 'good' if delta_v < -0.001 else ('bad' if delta_v > 0.001 else 'neutral')
            line(f"  {key:<24s}  {cv:.4f} -> {bv:.4f}  ({delta_v:+.4f})", tag)
        line(f"  {'COMPOSITE':<24s}  {curr_bk.get('composite',0):.4f} -> {best_bk.get('composite',0):.4f}  "
             f"({best_bk.get('composite',0) - curr_bk.get('composite',0):+.4f})", 'highlight')

        subsection("5.4", "GA Convergence")
        h_best = pipe.get('ga_history_best', [])
        h_avg  = pipe.get('ga_history_avg', [])
        h_worst = pipe.get('ga_history_worst', [])
        if h_best:
            line(f"Gen 1 best:                  ${h_best[0]:>12,.2f}")
            line(f"Final best:                  ${h_best[-1]:>12,.2f}")
            line(f"Convergence gain:            ${h_best[-1] - h_best[0]:+12,.2f}")
            if len(h_best) >= 2:
                early_gain = h_best[min(4, len(h_best)-1)] - h_best[0]
                late_gain  = h_best[-1] - h_best[max(len(h_best)-5, 0)]
                line(f"Early-stage gain (gen 1-5):  ${early_gain:+12,.2f}")
                line(f"Late-stage gain (last 5):    ${late_gain:+12,.2f}")
                # Check the flat run first: with zero gains both ratio tests
                # would otherwise fail and report a search that never moved
                # as still improving.
                if abs(h_best[-1] - h_best[0]) <= 1e-9 * max(abs(h_best[0]), 1.0):
                    line("Convergence: NO IMPROVEMENT AFTER GENERATION 1 (flat search)", 'neutral')
                elif abs(late_gain) <= abs(early_gain) * 0.1:
                    line("Convergence: FULLY CONVERGED (flat tail)", 'good')
                elif abs(late_gain) <= abs(early_gain) * 0.3:
                    line("Convergence: NEARLY CONVERGED", 'good')
                else:
                    line("Convergence: STILL IMPROVING (more generations may help)", 'highlight')

        # --------------------------------------------------------
        # 6. ITEM MOVEMENTS
        # --------------------------------------------------------
        section("6", "ITEM REPOSITIONING LOG")

        if changes:
            for i, change in enumerate(changes, 1):
                line(f"{i:>3d}. {change}")
            line("")
            line(f"Total items moved: {n_changes}", 'highlight')
        else:
            line("No items were moved — layout already optimal.", 'good')

        # --------------------------------------------------------
        # 7. POST-OPTIMIZATION MC PROJECTION
        # --------------------------------------------------------
        section("7", "POST-OPTIMIZATION MONTE CARLO PROJECTION")
        cite("[Monte Carlo revenue simulation per Metropolis & Ulam, 1949; Rubinstein & Kroese, 2016]")

        line(f"Mean 30-day revenue:         ${mc_opt.get('mean',0):>12,.2f}")
        line(f"Std deviation:               ${mc_opt.get('std',0):>12,.2f}")
        line(f"5th percentile:              ${mc_opt.get('p5',0):>12,.2f}")
        line(f"95th percentile:             ${mc_opt.get('p95',0):>12,.2f}")
        line(f"Daily average:               ${mc_opt.get('mean',0)/30:>12,.2f}")
        line(f"Annual projection:           ${mc_opt.get('mean',0)/30*365:>12,.0f}")
        line("")
        mc_delta = mc_opt.get('mean', 0) - mc_base.get('mean', 0)
        line(f"vs current layout, 30-day:   ${mc_delta:+12,.2f}  ({mc_lift:+.1f}%)",
             'good' if mc_delta > 0 else 'bad')
        line(f"vs current layout, annual:   ${mc_delta/30*365:+12,.0f}",
             'good' if mc_delta > 0 else 'bad')
        line("  (both layouts under the layout model; the raw calibrated")
        line("   baseline is Sec. 2.2)")

        line("")
        # 5th-95th percentiles of the simulated 30-day totals: the spread of
        # outcomes the model produces, which is far wider than the Monte
        # Carlo uncertainty about the projected mean.
        line("90% range of simulated outcomes (5th-95th pct):", 'highlight')
        line(f"  Current:   ${mc_base.get('p5',0):>10,.0f}  to  ${mc_base.get('p95',0):>10,.0f}")
        line(f"  Optimized: ${mc_opt.get('p5',0):>10,.0f}  to  ${mc_opt.get('p95',0):>10,.0f}")

        # --------------------------------------------------------
        # 8. A/B TEST (PRE vs POST)
        # --------------------------------------------------------
        section("8", "A/B STATISTICAL COMPARISON (PRE vs POST LAYOUT)")
        cite("[Welch's t-test: Welch, 1947; KS test: Massey, 1951; Cohen's d: Cohen, 1988]")

        ab_r = pipe.get('ab_results', {})
        ab_t = pipe.get('ab_tests', {})

        if 'A' in ab_r and 'B' in ab_r:
            ra = ab_r['A']
            rb = ab_r['B']
            ab_delta = rb['mean'] - ra['mean']
            ab_pct = ab_delta / max(ra['mean'], 1e-6) * 100
            # Direction (and colour) comes from the paired-lift interval, not
            # the sign of the mean, so Monte Carlo noise alone never shows up
            # as a better or worse layout.
            lift_mean = ab_t.get('lift_mean', ab_delta)
            lift_ci = ab_t.get('lift_ci95')
            if lift_ci and lift_ci[0] > 0:
                ab_dir_tag = 'good'
            elif lift_ci and lift_ci[1] < 0:
                ab_dir_tag = 'bad'
            else:
                ab_dir_tag = 'neutral'

            subsection("8.1", "Revenue Summary (30-day MC)")
            line(f"  {'Metric':<22s} {'PRE (A)':>12s} {'POST (B)':>12s}")
            line(f"  {'Mean':.<22s} ${ra['mean']:>11,.2f} ${rb['mean']:>11,.2f}")
            line(f"  {'Median':.<22s} ${ra['median']:>11,.2f} ${rb['median']:>11,.2f}")
            line(f"  {'Std Dev':.<22s} ${ra['std']:>11,.2f} ${rb['std']:>11,.2f}")
            line(f"  {'5th pct':.<22s} ${ra['p5']:>11,.2f} ${rb['p5']:>11,.2f}")
            line(f"  {'95th pct':.<22s} ${ra['p95']:>11,.2f} ${rb['p95']:>11,.2f}")
            line("")
            line(f"  Delta (POST - PRE):    ${ab_delta:+12,.2f}  ({ab_pct:+.1f}%)",
                 ab_dir_tag)
            line(f"  Daily delta:           ${ab_delta/30:+12,.2f}")
            line(f"  Annual projection:     ${ab_delta/30*365:+12,.0f}",
                 ab_dir_tag)

            subsection("8.2", "Statistical Tests")
            sig_str = "YES" if ab_t.get('significant') else "NO"
            effect = ("Large" if abs(ab_t.get('cohens_d', 0)) >= 0.8 else
                      "Medium" if abs(ab_t.get('cohens_d', 0)) >= 0.5 else
                      "Small" if abs(ab_t.get('cohens_d', 0)) >= 0.2 else "Negligible")
            line(f"  Welch's t-test:")
            line(f"    t-statistic:         {ab_t.get('t_stat',0):.4f}")
            line(f"    p-value:             {ab_t.get('p_value',1):.6f}")
            line(f"    p < alpha:           {sig_str} (alpha={ab_t.get('alpha',0.05)})")
            line("")
            line(f"  Kolmogorov-Smirnov test:")
            line(f"    KS statistic:        {ab_t.get('ks_stat',0):.4f}")
            line(f"    p-value:             {ab_t.get('ks_p',1):.6f}")
            if ab_t.get('p_value_caveat'):
                line(f"    Note: {ab_t['p_value_caveat']}")
            line("")
            line(f"  Effect Size:")
            line(f"    Cohen's d:           {ab_t.get('cohens_d',0):.4f}  ({effect})")
            line(f"    P(POST > PRE):       {ab_t.get('p_b_better',50):.1f}%",
                 ab_dir_tag)

            subsection("8.3", "Layout Quality Scores")
            line(f"  PRE layout score:      {ra.get('score',0):.4f}")
            line(f"  POST layout score:     {rb.get('score',0):.4f}")
            line(f"  Score improvement:     {rb.get('score',0) - ra.get('score',0):+.4f}",
                 'good' if rb.get('score',0) > ra.get('score',0) else 'bad')
            line(f"  Conv rate PRE:         {ra.get('conv_adj',0)*100:.2f}%")
            line(f"  Conv rate POST:        {rb.get('conv_adj',0)*100:.2f}%")
            line(f"  Impulse rate PRE:      {ra.get('imp_adj',0)*100:.2f}%")
            line(f"  Impulse rate POST:     {rb.get('imp_adj',0)*100:.2f}%")

            subsection("8.4", "Verdict")
            # PRE and POST are simulated under parameters that differ by
            # construction, so both the p-values and the lift CI narrow as
            # MC iterations grow; the verdict reports magnitude, not
            # significance. The direction is the one set from the lift
            # interval above.
            if ab_dir_tag == 'good':
                line(f"  POST (B) projects higher 30-day revenue than PRE (A)", 'good')
            elif ab_dir_tag == 'bad':
                line(f"  POST (B) projects lower 30-day revenue than PRE (A)", 'bad')
            else:
                line(f"  No difference distinguishable from MC noise", 'neutral')
            line(f"  Mean lift (POST - PRE): ${lift_mean:+12,.2f}")
            if lift_ci:
                line(f"  95% CI (MC precision under the model): "
                     f"${lift_ci[0]:+,.2f} to ${lift_ci[1]:+,.2f}")
            line(f"  Effect magnitude:       Cohen's d = {ab_t.get('cohens_d', 0):.3f} ({effect})")
            line(f"  The CI reflects simulation noise only, not real-world")
            line(f"  uncertainty about the layout change.")
        else:
            line("A/B comparison could not be run (insufficient snapshot data).", 'bad')

        # --------------------------------------------------------
        # 9. REAL-TIME VALIDATION
        # --------------------------------------------------------
        section("9", "REAL-TIME SIMULATION VALIDATION")

        dur = A.get('measurement_duration', 120)
        line(f"Measurement window:          {dur}s ({dur//60} min each)")
        line(f"PRE-window revenue:          ${real_pre:.2f}")
        line(f"POST-window revenue:         ${real_post:.2f}")
        if real_lift is not None:
            line(f"Real-time lift:              {real_lift:+.1f}%",
                 'good' if real_lift > 0 else 'bad')
        else:
            line(f"Real-time lift:              n/a")
            if real_lift_note:
                line(f"  {real_lift_note}")
        line("")
        line("Note: Short measurement windows have high stochastic variance.")
        line("The MC/GA projections (Sections 5, 7, 8) provide statistically")
        line("robust estimates. Real-time results serve as a directional sanity check.")
        line("")
        cite("In practice, A/B tests in retail require 2-4 weeks of data to")
        cite("reach statistical power >0.80 (Kohavi et al., 2020).")

        # --------------------------------------------------------
        # 10. STRATEGIC RECOMMENDATIONS
        # --------------------------------------------------------
        section("10", "STRATEGIC RECOMMENDATIONS")

        recs = []
        if sorted_t:
            top_driver = sorted_t[0][0]
            recs.append(f"Focus on improving '{top_driver}' — it has the largest revenue impact "
                        f"(Sec. 3).")
        if markov_absorb.get('p_abandon_from_entering', 0) > 0.25:
            recs.append(f"Reduce abandonment rate (currently "
                        f"{markov_absorb['p_abandon_from_entering']*100:.0f}%) through better "
                        f"flow and checkout optimization (Sec. 4).")
        if best_bk.get('bottleneck_penalty', 0) > 0.1:
            recs.append("Address congestion hotspots — the bottleneck penalty is still "
                        "significant (Sec. 5.3).")
        if best_bk.get('section_compliance', 0) < 0.7:
            recs.append("Improve section compliance — some items placed outside their "
                        "category zones reduce wayfinding efficiency [10][11].")
        if best_bk.get('cross_merch', 0) < 0.3:
            recs.append("Strengthen cross-merchandising — frequently co-purchased items "
                        "are still far apart. Adjacent placement lifts basket size 10-30% [5][6].")
        if best_bk.get('impulse', 0) < 0.4:
            recs.append("Reposition impulse items closer to checkout — checkout-adjacent "
                        "placement increases impulse purchases by 25-45% [7][8].")
        if ga_lift > 0:
            recs.append(f"The GA found a {ga_lift:.1f}% revenue improvement — "
                        f"apply and monitor for 2+ weeks [17].")
        if not recs:
            recs.append("Layout appears well-optimized. Monitor metrics monthly for drift.")

        for i, rec in enumerate(recs, 1):
            line(f"{i}. {rec}", 'highlight')

        # --------------------------------------------------------
        # 11. THEORETICAL FRAMEWORK & EXPECTED IMPROVEMENTS
        # --------------------------------------------------------
        section("11", "THEORETICAL FRAMEWORK — RETAIL SCIENCE BENCHMARKS")

        cite("The optimization strategies applied in this report are grounded in")
        cite("established retail science and consumer behavior research. Each")
        cite("layout decision maps to empirically validated principles:")
        line("")

        subsection("11.1", "Eye-Level Placement & Vertical Merchandising")
        line("Products placed at eye level (120-160cm) receive 35% more")
        line("visual attention and 8-15% higher purchase rates than items")
        line("on lower or upper shelves.")
        cite("[1] Chandon et al. (2009), Journal of Marketing, 73(6), 1-17.")
        cite("    'Does In-Store Marketing Work? Effects of the Number and")
        cite("    Position of Shelf Facings on Brand Attention and Evaluation'")
        cite("[2] Dreze, Hoch & Purk (1994), Journal of Retailing, 70(4), 301-326.")
        cite("    'Shelf Management and Space Elasticity'")
        line("")
        line("APPLIED: Revenue-weighted placement score ensures high-value")
        line("items are positioned in high-traffic zones (hot cells on heat map).")
        conv_improvement = best_bk.get('revenue_placement', 0) - curr_bk.get('revenue_placement', 0)
        if abs(conv_improvement) > 0.001:
            line(f"YOUR RESULT: Revenue placement score {conv_improvement:+.4f}",
                 'good' if conv_improvement > 0 else 'bad')

        subsection("11.2", "Traffic Flow & Store Layout Optimization")
        line("Strategic product placement along natural traffic flow increases")
        line("exposure time by 20-40%, directlycorrelating with 7-19% higher")
        line("conversion rates.")
        cite("[3] Sorensen (2009), 'Inside the Mind of the Shopper', Pearson.")
        cite("    Found that shoppers traverse only 33% of store area on average;")
        cite("    optimized flow increases this to 45-60%.")
        cite("[4] Underhill (2009), 'Why We Buy: The Science of Shopping', Simon")
        cite("    & Schuster. Documents the 'transition zone' and 'butt-brush")
        cite("    factor' as key flow design principles.")
        line("")
        line("APPLIED: Flow efficiency score measures entrance -> top items ->")
        line("checkout path length.")
        flow_improvement = best_bk.get('flow', 0) - curr_bk.get('flow', 0)
        if abs(flow_improvement) > 0.001:
            line(f"YOUR RESULT: Flow efficiency score {flow_improvement:+.4f}",
                 'good' if flow_improvement > 0 else 'bad')

        subsection("11.3", "Cross-Merchandising & Complementary Placement")
        line("Placing frequently co-purchased items adjacent to each other")
        line("increases basket size by 10-30% and cross-category revenue by")
        line("up to 18%.")
        cite("[5] Russell & Petersen (2000), Journal of Retailing,")
        cite("    76(3), 367-392. 'Analysis of Cross Category Dependence in")
        cite("    Market Basket Selection'")
        cite("[6] Bezawada et al. (2009), Journal of Marketing, 73(3), 99-117.")
        cite("    'Cross-Category Effects of Aisle and Display Placements'")
        line("")
        line("APPLIED: Cross-merchandising score rewards proximity-weighted")
        line("placement of observed co-purchase pairs from analytics data.")
        cross_improvement = best_bk.get('cross_merch', 0) - curr_bk.get('cross_merch', 0)
        if abs(cross_improvement) > 0.001:
            line(f"YOUR RESULT: Cross-merch score {cross_improvement:+.4f}",
                 'good' if cross_improvement > 0 else 'bad')

        subsection("11.4", "Impulse Purchase Zone Optimization")
        line("Checkout-adjacent placement of impulse items (confections,")
        line("magazines, small accessories) increases unplanned purchases")
        line("by 25-45%. The 'waiting time' at checkout creates a captive")
        line("browsing window of 60-180 seconds.")
        cite("[7] Inman, Winer & Ferraro (2009), Journal of Marketing, 73(5),")
        cite("    19-29. 'The Interplay Among Category Characteristics,")
        cite("    Customer Characteristics, and Customer Activities on")
        cite("    In-Store Decision Making'")
        cite("[8] Hui, Bradlow & Fader (2009), Journal of Consumer Research,")
        cite("    36(3), 478-493. 'Testing Behavioral Hypotheses Using an")
        cite("    Integrated Model of Grocery Store Shopping Path and")
        cite("    Purchase Behavior'")
        line("")
        line("APPLIED: Impulse placement score penalises impulse items far")
        line("from checkout; GA evolves their positions toward the register.")
        impulse_improvement = best_bk.get('impulse', 0) - curr_bk.get('impulse', 0)
        if abs(impulse_improvement) > 0.001:
            line(f"YOUR RESULT: Impulse placement score {impulse_improvement:+.4f}",
                 'good' if impulse_improvement > 0 else 'bad')

        subsection("11.5", "Congestion & Bottleneck Avoidance")
        line("Crowding and narrow-aisle congestion decrease dwell time by")
        line("15-25% and increase abandonment. 'Butt-brush effect' research")
        line("shows customers leave aisles when jostled, reducing purchases.")
        cite("[9] Harrell, Hutt & Anderson (1980), Journal of Marketing Research,")
        cite("    17(1), 45-51. 'Path Analysis of Buyer Behavior Under")
        cite("    Conditions of Crowding'")
        cite("[4] Underhill (2009) — see above.")
        line("")
        line("APPLIED: Bottleneck penalty in layout scoring; items avoid")
        line("known congestion hotspots identified by heat map analysis.")
        bn_improvement = curr_bk.get('bottleneck_penalty', 0) - best_bk.get('bottleneck_penalty', 0)
        if abs(bn_improvement) > 0.001:


            line(f"YOUR RESULT: Bottleneck penalty reduced by {bn_improvement:+.4f}",
                 'good' if bn_improvement > 0 else 'bad')
                 

        subsection("11.6", "Section Compliance & Wayfinding")
        line("Consistent category grouping reduces search time by 20-35%")
        line("and increases customer satisfaction. Items placed outside their")
        line("logical section create confusion and reduce conversion.")
        cite("[10] Titus & Everett (1995), Journal of the Academy of Marketing")
        cite("     Science, 23(2), 106-119. 'The Consumer Retail Search Process'")
        cite("[11] Chebat, Gelinas-Chebat & Therrien (2005), Journal of Business")
        cite("     Research, 58(11), 1590-1598. 'Lost in a Mall'")
        line("")
        line("APPLIED: Section compliance score rewards items staying in their")
        line("Section_<category> wall and penalises boundary violations.")
        sec_improvement = best_bk.get('section_compliance', 0) - curr_bk.get('section_compliance', 0)


        if abs(sec_improvement) > 0.001:
            line(f"YOUR RESULT: Section compliance score {sec_improvement:+.4f}",
                 'good' if sec_improvement > 0 else 'bad')

        subsection("11.7", "Dwell Time & Conversion Correlation")
        line("Longer dwell times in product zones correlate positively with")
        line("purchase probability (r = 0.42-0.68 depending on category).")
        line("Each additional 30s of dwell time increases conversion by 2-5%.")
        cite("[12] Hui, Fader & Bradlow (2009), Marketing Science,")
        cite("     28(3), 566-572. 'The Traveling Salesman Goes Shopping'")
        cite("[13] Larson, Bradlow & Fader (2005), International Journal of")
        cite("     Research in Marketing, 22(4), 395-414. 'An Exploratory Look at")
        cite("     Supermarket Shopping Paths'")
        line("")
        line("NOT SCORED: dwell times are averaged per category, so they do not")
        line("change when items move; the layout score has no dwell term.")

        # --------------------------------------------------------
        # 12. EXPECTED IMPROVEMENTS SUMMARY (with citations)
        # --------------------------------------------------------
        section("12", "EXPECTED IMPROVEMENTS SUMMARY")

        line("Based on the retail science literature and the optimization", 'highlight')
        line("scores achieved, the following improvements are expected:", 'highlight')
        line("")
        line(f"  {'Strategy':<35s} {'Expected Range':>18s}  {'Source':>8s}")
        line(f"  {'-'*35:<35s} {'-'*18:>18s}  {'-'*8:>8s}")
        line(f"  {'Eye-level / traffic placement':<35s} {'7-19% conv. lift':>18s}  {'[1][2]':>8s}")
        line(f"  {'Cross-merchandising adjacency':<35s} {'10-30% basket lift':>18s}  {'[5][6]':>8s}")
        line(f"  {'Impulse zone at checkout':<35s} {'25-45% impulse lift':>18s}  {'[7][8]':>8s}")
        line(f"  {'Optimized customer flow':<35s} {'20-40% more exposure':>18s}  {'[3][4]':>8s}")
        line(f"  {'Bottleneck elimination':<35s} {'15-25% less abandon':>18s}  {'[9][4]':>8s}")
        line(f"  {'Section compliance':<35s} {'20-35% less search':>18s}  {'[10][11]':>8s}")
        line("")

        composite_lift = best_bk.get('composite', 0) - curr_bk.get('composite', 0)
        line(f"Composite layout quality improvement: {composite_lift:+.4f}", 'highlight')
        line(f"MC projected revenue lift (30-day):   {mc_lift:+.1f}%", 'highlight')
        line(f"GA projected revenue lift:            {ga_lift:+.1f}%", 'highlight')
        line("")
        cite("Note: Expected ranges are literature benchmarks. Actual results")
        cite("depend on shop size, product mix, customerdemographics, and")
        cite("baseline layout quality. The MC simulation projects site-specific impact.")

        # --------------------------------------------------------
        # 13. REFERENCES
        # --------------------------------------------------------
        section("13", "REFERENCES")

        bibline("[1]  Chandon, P., Hutchinson, J.W., Bradlow, E.T. & Young, S.H.")
        bibline("     (2009). 'Does In-Store Marketing Work? Effects of the Number")
        bibline("     and Position of Shelf Facings on Brand Attention and")
        bibline("     Evaluation at the Point of Purchase'. Journal of Marketing,")
        bibline("     73(6), pp.1-17.")
        bibline("")
        bibline("[2]  Dreze, X., Hoch, S.J. & Purk, M.E. (1994). 'Shelf")
        bibline("     Management and Space Elasticity'. Journal of Retailing,")
        bibline("     70(4), pp.301-326.")
        bibline("")
        bibline("[3]  Sorensen, H. (2009). Inside the Mind of the Shopper: The")
        bibline("     Science of Retailing. Upper Saddle River, NJ: Pearson/FT")
        bibline("     Press.")
        bibline("")
        bibline("[4]  Underhill, P. (2009). Why We Buy: The Science of Shopping")
        bibline("     (Updated and Revised). New York: Simon & Schuster.")
        bibline("")
        bibline("[5]  Russell, G.J. & Petersen, A. (2000). 'Analysis of Cross")
        bibline("     Category Dependence in Market Basket Selection'. Journal of")
        bibline("     Retailing, 76(3), pp.367-392.")
        bibline("")
        bibline("[6]  Bezawada, R., Balachander, S., Kannan, P.K. & Shankar, V.")
        bibline("     (2009). 'Cross-Category Effects of Aisle and Display")
        bibline("     Placements'. Journal of Marketing, 73(3), pp.99-117.")
        bibline("")
        bibline("[7]  Inman, J.J., Winer, R.S. & Ferraro, R. (2009). 'The")
        bibline("     Interplay Among Category Characteristics, Customer")
        bibline("     Characteristics, and Customer Activities on In-Store")
        bibline("     Decision Making'. Journal of Marketing, 73(5), pp.19-29.")
        bibline("")
        bibline("[8]  Hui, S.K., Bradlow, E.T. & Fader, P.S. (2009). 'Testing")
        bibline("     Behavioral Hypotheses Using an Integrated Model of Grocery")
        bibline("     Store Shopping Path and Purchase Behavior'. Journal of")
        bibline("     Consumer Research, 36(3), pp.478-493.")
        bibline("")
        bibline("[9]  Harrell, G.D., Hutt, M.D. & Anderson, J.C. (1980). 'Path")
        bibline("     Analysis of Buyer Behavior Under Conditions of Crowding'.")
        bibline("     Journal of Marketing Research, 17(1), pp.45-51.")
        bibline("")
        bibline("[10] Titus, P.A. & Everett, P.B. (1995). 'The Consumer Retail")
        bibline("     Search Process: A Conceptual Model and Research Agenda'.")
        bibline("     Journal of the Academy of Marketing Science, 23(2),")
        bibline("     pp.106-119.")
        bibline("")
        bibline("[11] Chebat, J.C., Gelinas-Chebat, C. & Therrien, K. (2005).")
        bibline("     'Lost in a Mall, the Effects of Gender, Familiarity with")
        bibline("     the Shopping Mall and the Shopping Values on Shoppers'")
        bibline("     Wayfinding Processes'. Journal of Business Research, 58(11),")
        bibline("     pp.1590-1598.")
        bibline("")
        bibline("[12] Hui, S.K., Fader, P.S. & Bradlow, E.T. (2009). 'The")
        bibline("     Traveling Salesman Goes Shopping: The Systematic Deviations")
        bibline("     of Grocery Paths from TSP Optimality'. Marketing Science,")
        bibline("     28(3), pp.566-572.")
        bibline("")
        bibline("[13] Larson, J.S., Bradlow, E.T. & Fader, P.S. (2005). 'An")
        bibline("     Exploratory Look at Supermarket Shopping Paths'. International")
        bibline("     Journal of Research in Marketing, 22(4), pp.395-414.")
        bibline("")
        bibline("[14] Saltelli, A., Ratto, M., Andres, T. et al. (2008). Global")
        bibline("     Sensitivity Analysis:The Primer. Chichester: John Wiley &")
        bibline("     Sons.")
        bibline("")
        bibline("[15] Kemeny, J.G. & Snell, J.L. (1976). Finite Markov Chains.")
        bibline("     New York: Springer-Verlag.")
        bibline("")
        bibline("[16] Rubinstein, R.Y. & Kroese, D.P. (2016). Simulation and the")
        bibline("     Monte Carlo Method (3rd ed.). Hoboken, NJ: John Wiley & Sons.")
        bibline("")
        bibline("[17] Kohavi, R., Tang, D. & Xu, Y. (2020). Trustworthy Online")
        bibline("     Controlled Experiments: A Practical Guide to A/B Testing.")
        bibline("     Cambridge: Cambridge University Press.")
        bibline("")
        bibline("[18] Gonzalez-Cruz, M.C. & Fernandez-Llatas, C. (2011). 'Genetic")
        bibline("     Algorithm-Based Optimization of Retail Store Layout'.")
        bibline("     Proceedings of the European Modeling and Simulation Symposium.")
        bibline("")
        bibline("[19] Cohen, J. (1988). Statistical Power Analysis for the")


        bibline("     Behavioral Sciences (2nd ed.). Hillsdale, NJ: Lawrence")
        bibline("     Erlbaum Associates.")
        bibline("")
        bibline("[20] Metropolis, N. & Ulam, S. (1949). 'The Monte Carlo Method'.")
        bibline("     Journal of the American Statistical Association, 44(247),")
        bibline("     pp.335-341.")

        divider()
        report_text.insert(tk.END, "\n", 'neutral')
        report_text.insert(tk.END, "    END OF REPORT\n\n", 'section')

        report_text.config(state=tk.DISABLED)

        btn_frame = tk.Frame(results_dialog, bg='#1a1a2e')
        btn_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Button(btn_frame, text="Close",
                  command=results_dialog.destroy,
                  bg='#4CAF50', fg='white', font=('Arial', 12, 'bold'),
                  pady=5).pack(side=tk.RIGHT, padx=5)

        results_dialog.grab_set()

        # The saved copy is a convenience: an unwritable working directory
        # must not abort the dialog before its Close button and grab exist.
        report_content = report_text.get("1.0", tk.END)
        report_path = f"optimization_report_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report_content)
            print(f"Report saved to {report_path}")
        except OSError as e:
            print(f"Could not save report to {report_path}: {e}")

