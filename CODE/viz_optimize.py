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
from sim_calibration import transient_occupancy_distribution, _calib, mc_engine
from retail_literature import DEFAULT_WEEKEND_MULTIPLIER

import copy
from scipy import stats as sp_stats


class OptimizeMixin:
    """Optimization workflow start, post-window measurement, finalization metrics."""

    def _optimize_layout(self):
        """
        Master optimization pipeline:
          Phase 1 -- PRE-window: collect baseline simulation data
          Phase 2 -- Run all analytical engines (MC, Sensitivity, Markov, GA)
          Phase 3 -- Apply GA-optimized layout
          Phase 4 -- POST-window: collect post-optimization data
          Phase 5 -- A/B test PRE vs POST, build sectioned report
        """
        sim       = self.customer_simulation
        analytics = sim.analytics
        dur       = analytics.get('measurement_duration', 120)

        # One pipeline at a time. The PRE/POST windows leave the GUI
        # interactive, and a second run would reset the window counters
        # and orphan the first run's pending after-callback.
        if getattr(self, '_opt_running', False):
            return

        # A trajectory-only dataset fills the calibration block with spatial
        # keys but no arrival or conversion rate, so it cannot stand in for
        # observed customers -- the same predicate the other tabs gate on,
        # and the one the PRE snapshot's parameter extraction applies.
        cal = _calib(analytics)
        has_tx = bool(cal.get('arrivals_per_hour')) and 'conversion_rate' in cal
        if not has_tx and analytics.get('total_customers', 0) < 5:
            messagebox.showwarning(
                "Insufficient Data",
                "Run the simulation first (at least 5 customers).",
                parent=self.tk_root
            )
            return

        self._opt_running = True

        try:
            messagebox.showinfo(
                "Optimization",
                f"Phase 1/5: Collecting PRE-optimization baseline for {dur // 60} min...",
                parent=self.tk_root
            )

            # Empty the store right before the window opens, after the dialog:
            # a running simulation keeps spawning while it is on screen.
            self._clear_simulation()
            self._start_simulation()
        except Exception:
            # Without this the guard would stay set and Optimize would do
            # nothing for the rest of the session.
            self._opt_running = False
            raise
        if not getattr(sim, 'running', False) or getattr(sim, 'paused', False):
            # _start_simulation refused (no items / entrance / checkout) and
            # has already told the user why; there is nothing to measure.
            self._opt_running = False
            return

        # Open the PRE window only once the simulation is running, so its
        # clock excludes the time the dialog above was on screen (the POST
        # window is stamped the same way). Setting the phase before the
        # dialog would also credit exits from a still-running simulation.
        # Window starts are simulated seconds, the clock exits are timed on.
        analytics.update({
            'phase':               'pre',
            'pre_window_start':    sim.sim_time,
            'pre_window_revenue':  0.0,
            'post_window_revenue': 0.0,
            'optimization_history': [],
            '_opt_pipeline_data':  {},
        })

        prev_after_id = getattr(self, '_opt_after_id', None)
        if prev_after_id is not None:
            try:
                self.tk_root.after_cancel(prev_after_id)
            except Exception:
                pass
        self._schedule_window_end('pre_window_start', self._on_pre_window_end)

    #: How often the measurement window checks the simulated clock, in
    #: wall-clock milliseconds.
    _OPT_WINDOW_POLL_MS = 250

    def _schedule_window_end(self, start_key, callback):
        """Call ``callback`` once the simulated clock has advanced one
        measurement duration past ``analytics[start_key]``.

        Exits are credited to the window on the simulated clock, so ending it
        after a fixed number of WALL seconds would measure a different span at
        any speed other than 1x: above 1x most of the window's revenue falls
        outside the credited span, below 1x the window closes before the
        agents that spawned in it have finished shopping. Polling the
        simulated clock also lets a paused simulation extend the window rather
        than silently truncate it."""
        sim       = self.customer_simulation
        analytics = sim.analytics
        dur       = analytics.get('measurement_duration', 120)
        start     = analytics.get(start_key) or 0.0
        worker_gone = not (getattr(sim, 'running', False)
                           or getattr(sim, '_running', False))
        # A hard-stopped worker never advances the clock again, so waiting for
        # the rest of the window would hang the pipeline; measure what the
        # window did collect instead.
        if sim.sim_time - start >= dur or worker_gone:
            self._opt_after_id = None
            callback()
            return
        self._opt_after_id = self.tk_root.after(
            self._OPT_WINDOW_POLL_MS,
            lambda: self._schedule_window_end(start_key, callback))

    def _on_pre_window_end(self):
        """Public entry point for the PRE-window callback.

        Wraps the real implementation in try/except so any failure in
        the GA/MC/Markov/Sensitivity pipeline (including the user
        manually closing the progress dialog) is reported via a
        messagebox instead of taking down the GUI."""
        self._opt_progress = None
        try:
            self._on_pre_window_end_inner()
        except Exception as e:
            # The run never reached the POST window, so nothing else will
            # release the one-pipeline-at-a-time guard.
            self._opt_running = False
            import traceback as _tb
            tb_text = _tb.format_exc()
            try:
                if self._opt_progress is not None:
                    self._opt_progress.close()
            except Exception:
                pass
            pipe = self.customer_simulation.analytics.get('_opt_pipeline_data') or {}
            layout_note = (
                "\n\nThe optimized layout had already been written to the "
                "store and was left in place; it was not reverted to the "
                "PRE layout."
                if pipe.get('layout_applied') else "")
            try:
                messagebox.showerror(
                    "Optimization failed",
                    f"Optimization aborted due to:\n\n{e}{layout_note}\n\n"
                    f"{tb_text[-800:]}",
                    parent=self.tk_root,
                )
            except Exception:
                pass

    def _on_pre_window_end_inner(self):
        """
        Phase 2-3: PRE window finished.
        Iteration counts scale with the Hi-Fi toggle from the toolbar.
        """
        sim       = self.customer_simulation
        analytics = sim.analytics
        dur       = analytics.get('measurement_duration', 120)
        pipe      = analytics.get('_opt_pipeline_data', {})

        print('[opt] pre-window ended; stopping simulation…', flush=True)
        self._stop_simulation()
        # Settle the agents still in the store through _clear_simulation
        # BEFORE the worker drops them: it records them as cleared and retires
        # their state history, whereas hard_stop empties the store with no
        # bookkeeping, and afterwards there is nothing left to record. They
        # stay in the arrival count -- they did arrive inside the window --
        # and never enter the exit-based rates.
        self._clear_simulation()
        # Fully terminate the worker thread before the optimization pipeline
        # starts. ``_stop_simulation`` only pauses it, which leaves the
        # thread alive and posting canvas-redraw callables into
        # ``_gui_queue``. Those callables get processed inside every
        # ``pump_events()`` call below and can starve the main loop on
        # complex layouts. ``hard_stop`` joins the thread within one frame.
        try:
            self.customer_simulation.hard_stop()
        except Exception as _e:
            print(f'[opt] hard_stop failed (continuing): {_e}', flush=True)
        # Drain any callables the worker thread queued before exiting so
        # the first pump_events doesn't have to chew through a backlog of
        # matplotlib redraws.
        try:
            q = getattr(self.customer_simulation, '_gui_queue', None)
            if q is not None:
                drained = 0
                while True:
                    try:
                        q.get_nowait()
                        drained += 1
                    except Exception:
                        break
                if drained:
                    print(f'[opt] drained {drained} pending GUI callbacks',
                          flush=True)
        except Exception:
            pass
        print('[opt] simulation stopped; creating progress dialog…', flush=True)
        progress = self._create_optimization_progress_dialog()
        self._opt_progress = progress
        print('[opt] progress dialog created.', flush=True)

        # -- SNAPSHOT PRE-LAYOUT (A) -----------------------------
        progress.update_status("Saving PRE-layout snapshot (A)...", pct=2)
        progress.pump_events()
        print('[opt] taking PRE-layout snapshot…', flush=True)
        pre_snapshot = self._snapshot_layout()
        print('[opt] PRE-layout snapshot done.', flush=True)
        pipe['pre_snapshot'] = pre_snapshot
        base_params = pre_snapshot['params']
        if base_params is None:
            # Without a transactional calibration the parameters come from the
            # window itself, and a window that saw almost no customers cannot
            # supply an arrival or conversion rate for the engines below.
            progress.close()
            self._opt_progress = None
            self._opt_running = False
            messagebox.showwarning(
                "Optimization stopped",
                "Too few customers were observed during the PRE window to "
                "estimate the arrival and conversion rates the optimization "
                "needs.\n\nRun a longer window, raise the spawn rate, or load "
                "a transactional dataset first.",
                parent=self.tk_root)
            return

         # -- 1. MONTE CARLO BASELINE (chunked so the bar visibly moves) -----
        mc_days     = 30
        mc_iters    = 400
        mc_chunks   = 8
        chunk_iters = max(1, mc_iters // mc_chunks)

        # Iteration counts for the analytical engines further down. Defined
        # here (rather than near each phase) so closures like _make_kw can
        # see them. Values mirror the proven old-codebase defaults.
        sa_days      = mc_days
        sa_iters     = 300
        pop_size     = 40
        n_gens       = 20
        ga_mc_iters  = 300
        ga_mc_days   = 14
        import time as _time
        print('[opt]    cph=%.2f, conv=%.3f, baskets_n=%d'
              % (base_params['customers_per_hour'],
                 base_params['conversion_rate'],
                 len(base_params.get('basket_sizes_observed') or [])),
              flush=True)
        mc_means, mc_p5s, mc_p95s, mc_stds = [], [], [], []
        all_totals = []
        all_daily  = []
        for ci in range(mc_chunks):
            pct_now = 5 + 6.0 * ci / mc_chunks   # 5 .. 11
            progress.update_status(
                f"Phase 2a: Monte Carlo baseline {ci+1}/{mc_chunks}…",
                pct=pct_now)
            progress.pump_events()
            _t0 = _time.perf_counter()
            print('[opt]    MC chunk %d/%d START' % (ci + 1, mc_chunks), flush=True)
            r = self._mc_engine(
                cph=base_params['customers_per_hour'],
                conv=base_params['conversion_rate'],
                rev_mean=base_params['rev_per_converting_customer'],
                rev_std=base_params['rev_std'],
                imp_rate=base_params['impulse_rate'],
                imp_val=base_params['avg_impulse_value'],
                avg_bsk=base_params['avg_basket_size'],
                std_bsk=base_params['std_basket_size'],
                observed_baskets=base_params['basket_sizes_observed'],
                n_days=mc_days, n_iter=chunk_iters,
                op_hours=10.0, wknd_mult=DEFAULT_WEEKEND_MULTIPLIER,
                monthly_growth=0.0,
            )
            print('[opt]    MC chunk %d/%d END in %.3fs'
                  % (ci + 1, mc_chunks, _time.perf_counter() - _t0),
                  flush=True)
            mc_means.append(r['mean']); mc_stds.append(r['std'])
            mc_p5s.append(r.get('p5', r['mean']))
            mc_p95s.append(r.get('p95', r['mean']))
            if 'totals' in r:
                all_totals.append(r['totals'])
            if 'daily_means' in r:
                all_daily.append(r['daily_means'])
            progress.pump_events()

        # Aggregate chunks into a single baseline dict matching _mc_engine's shape
        import numpy as _np
        totals_concat = _np.concatenate(all_totals) if all_totals else _np.array(mc_means)
        mc_baseline = {
            'mean':   float(_np.mean(totals_concat)) if totals_concat.size else float(_np.mean(mc_means)),
            'std':    float(_np.std(totals_concat))  if totals_concat.size else float(_np.mean(mc_stds)),
            'median': float(_np.median(totals_concat)) if totals_concat.size else float(_np.mean(mc_means)),
            'p5':     float(_np.percentile(totals_concat, 5))  if totals_concat.size else float(_np.mean(mc_p5s)),
            'p95':    float(_np.percentile(totals_concat, 95)) if totals_concat.size else float(_np.mean(mc_p95s)),
            'totals': totals_concat,
            'daily_means': _np.mean(all_daily, axis=0) if all_daily else _np.zeros(mc_days),
        }
        pipe['mc_baseline'] = mc_baseline
        progress.update_progress(11)
        progress.pump_events()

        # -- 2. SENSITIVITY ANALYSIS -----------------------------
        progress.update_status("Phase 2b: Sensitivity analysis...", pct=12)
        progress.pump_events()
        sa_pct = 0.20
        sa_param_defs = {
            'Customers/hr':    ('customers_per_hour',          None,   None),
            'Conversion rate': ('conversion_rate',             0.001,  0.999),
            'Revenue/customer':('rev_per_converting_customer', 0.01,   None),
            'Impulse rate':    ('impulse_rate',                0.0,    0.999),
            'Impulse value':   ('avg_impulse_value',           0.01,   None),
            'Basket size':     ('avg_basket_size',             0.5,    None),
        }
        tornado = {}
        sa_labels = list(sa_param_defs.items())
        total_sa = len(sa_labels) * 2
        sa_done = 0
        # Common random numbers: each low/high pair replays one stream, so a
        # swing reflects the parameter change rather than MC noise.
        sa_seed = int(np.random.randint(0, 2**31 - 1))
        for label, (pkey, lo_clamp, hi_clamp) in sa_labels:
            bval = base_params[pkey]
            lo_val = bval * (1.0 - sa_pct)
            hi_val = bval * (1.0 + sa_pct)
            if lo_clamp is not None:
                lo_val = max(lo_clamp, lo_val)
            if hi_clamp is not None:
                hi_val = min(hi_clamp, hi_val)

            def _make_kw(override_key, override_val):
                kw = dict(
                    cph=base_params['customers_per_hour'],
                    conv=base_params['conversion_rate'],
                    rev_mean=base_params['rev_per_converting_customer'],
                    rev_std=base_params['rev_std'],
                    imp_rate=base_params['impulse_rate'],
                    imp_val=base_params['avg_impulse_value'],
                    avg_bsk=base_params['avg_basket_size'],
                    std_bsk=base_params['std_basket_size'],
                    observed_baskets=base_params['basket_sizes_observed'],
                    n_days=sa_days, n_iter=sa_iters,
                    op_hours=10.0, wknd_mult=DEFAULT_WEEKEND_MULTIPLIER,
                    monthly_growth=0.0,
                )
                key_map = {
                    'customers_per_hour': 'cph',
                    'conversion_rate': 'conv',
                    'rev_per_converting_customer': 'rev_mean',
                    'impulse_rate': 'imp_rate',
                    'avg_impulse_value': 'imp_val',
                    'avg_basket_size': 'avg_bsk',
                }
                kw[key_map[override_key]] = override_val
                if override_key == 'rev_per_converting_customer':
                    base_rm = base_params['rev_per_converting_customer']
                    kw['rev_std'] = base_params['rev_std'] * (
                        override_val / max(base_rm, 1e-6)
                    )
                if override_key == 'avg_basket_size':
                    # mc_engine never turns basket size into revenue, so
                    # perturb it through the channel the GA fitness uses:
                    # revenue per converter scales with basket size at a
                    # fixed average item price.
                    base_bsk = base_params['avg_basket_size']
                    ratio = override_val / base_bsk if base_bsk > 0 else 1.0
                    kw['rev_mean'] = base_params['rev_per_converting_customer'] * ratio
                    kw['rev_std'] = base_params['rev_std'] * ratio
                return kw

            sa_done += 1
            progress.update_status(
                f"Phase 2b: Sensitivity {sa_done}/{total_sa} — {label} (low)…",
                pct=12 + 18.0 * sa_done / max(total_sa, 1))
            progress.pump_events()
            lo_res = mc_engine(**_make_kw(pkey, lo_val),
                               rng=np.random.RandomState(sa_seed))
            progress.pump_events()

            sa_done += 1
            progress.update_status(
                f"Phase 2b: Sensitivity {sa_done}/{total_sa} — {label} (high)…",
                pct=12 + 18.0 * sa_done / max(total_sa, 1))
            progress.pump_events()
            hi_res = mc_engine(**_make_kw(pkey, hi_val),
                               rng=np.random.RandomState(sa_seed))
            progress.pump_events()

            tornado[label] = {
                'baseline': bval,
                'lo_val': lo_val, 'hi_val': hi_val,
                'lo_rev': lo_res['mean'], 'hi_rev': hi_res['mean'],
                'swing': abs(hi_res['mean'] - lo_res['mean']),
            }
        pipe['tornado'] = tornado

        total_swing = sum(v['swing'] for v in tornado.values()) or 1.0
        sa_weights = {lbl: d['swing'] / total_swing for lbl, d in tornado.items()}
        pipe['sa_weights'] = sa_weights
        pipe['top_drivers'] = sorted(sa_weights.items(),
                                     key=lambda kv: kv[1], reverse=True)

        # -- 3. MARKOV CHAIN ANALYSIS ----------------------------
        progress.update_status("Phase 2c: Markov chain analysis...", pct=32)
        progress.pump_events()
        states, T = self._build_transition_matrix()
        absorb = self._compute_absorbing_analysis(states, T)
        pipe['markov_states'] = states
        pipe['markov_T'] = T
        pipe['markov_absorb'] = absorb
        pipe['markov_steady'] = transient_occupancy_distribution(analytics)

        markov_p_purchase = absorb['p_purchase_from_entering']
        markov_p_abandon  = absorb['p_abandon_from_entering']
        pipe['markov_p_purchase'] = markov_p_purchase
        pipe['markov_p_abandon']  = markov_p_abandon

        # -- 4. GA OPTIMIZATION ----------------------------------
        progress.update_status(
            f"Phase 2d: Genetic Algorithm ({pop_size} pop × {n_gens} gens)…",
            pct=38)
        progress.pump_events()
        item_names = self._ga_get_movable_items()
        if len(item_names) < 2:
            progress.close()
            messagebox.showinfo("Optimization",
                                "Not enough movable items to optimize.",
                                parent=self.tk_root)
            self._start_simulation()
            analytics['post_window_start'] = sim.sim_time
            analytics['phase'] = 'post'
            self._schedule_window_end('post_window_start',
                                      self._finalize_optimization_metrics)
            return

        mut_rate    = 0.18
        elite_frac  = 0.20
        n_elite     = max(1, int(pop_size * elite_frac))
        # (rest of the GA loop unchanged -- uses pop_size / n_gens / ga_mc_iters / ga_mc_days)

        accessible_floors = (
            self._ga_accessible_floors()
            if hasattr(self, '_ga_accessible_floors')
            else {self.current_floor}
        )
        item_floor_map  = dict(getattr(self, '_ga_item_floor_map',  {}) or {})
        item_source_map = dict(getattr(self, '_ga_item_source_map', {}) or {})
        pipe['ga_accessible_floors'] = sorted(accessible_floors)
        pipe['ga_item_floor_map']   = item_floor_map
        pipe['ga_item_source_map']  = item_source_map

        current_chrom = self._ga_encode(item_names)

        # Pre-fetch per-item size to avoid dict lookups in seeding loop
        item_sizes = []
        for n in item_names:
            src = (self._ga_get_item_data(n)
                   if hasattr(self, '_ga_get_item_data')
                   else self.items.get(n, {'size': (1.0, 1.0)}))
            item_sizes.append(src.get('size', (1.0, 1.0)))

        # The as-built layout enters through the same repair chain as every
        # other candidate, so the whole population is scored on one feasible
        # set; ``current_chrom`` itself stays unrepaired because the report
        # compares against the store as the user built it.
        population = [self._ga_resolve_overlaps(
            self._ga_repair(current_chrom.copy(), item_names), item_names)]
        for _ in range(pop_size - 1):
            noisy = current_chrom.copy()
            for i, (w, h) in enumerate(item_sizes):
                # Perturb inside the item's own section and repair, as the GA
                # tab does. Shop-scale noise clipped only to the shop bounds
                # pushes nearly every item out of its section, so the seeds
                # would be infeasible and carried forward as elites.
                bounds = self._ga_get_section_bounds(item_names[i])
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
                    current_chrom[i, 0] + np.random.normal(0, sigma_x), lo_x, hi_x)
                noisy[i, 1] = np.clip(
                    current_chrom[i, 1] + np.random.normal(0, sigma_y), lo_y, hi_y)
            noisy = self._ga_repair(noisy, item_names)
            noisy = self._ga_resolve_overlaps(noisy, item_names)
            population.append(noisy)

        # Pre-compute a base expected daily revenue once (no per-chrom MC).
        cph_base   = base_params['customers_per_hour']
        rev_base   = base_params['rev_per_converting_customer']
        impv_base  = base_params['avg_impulse_value']
        # 30 days @ 10 op hours, weekend mult from retail_literature
        # -> effective 10*(5 + 2*wknd)/7 ~ 11.14 at the cited 1.4
        op_eff_hours = 10.0 * ((5 + 2 * DEFAULT_WEEKEND_MULTIPLIER) / 7.0)
        base_daily_cust = cph_base * op_eff_hours

        def _fast_fitness(chromosome):
            """Analytic surrogate of MC mean revenue -- O(score) only.

            The score -> conversion / impulse / basket transform (cited per
            retail_literature.py) is _opt_layout_revenue_params, shared with
            the MC projection and the A/B comparison.
            """
            score, bkd = self._ga_compute_layout_score(
                chromosome, item_names, base_params)
            final_conv, imp_adj, _bsk_adj, rev_mult = \
                self._opt_layout_revenue_params(
                    score, bkd, base_params, sa_weights, markov_p_purchase)

            # Analytic expected 30-day revenue (matches _mc_engine mean
            # structure).
            exp_converters = base_daily_cust * final_conv
            rev_adj   = rev_base * rev_mult
            base_rev  = exp_converters * rev_adj
            imp_rev   = exp_converters * imp_adj * impv_base
            return (base_rev + imp_rev) * 30.0

        # Elite-fitness cache keyed by chromosome bytes
        fit_cache = {}
        def _eval(chrom):
            key = chrom.tobytes()
            v = fit_cache.get(key)
            if v is None:
                v = _fast_fitness(chrom)
                fit_cache[key] = v
            return v

        history_best, history_avg, history_worst = [], [], []

        ga_pct_start = 40.0
        ga_pct_end   = 86.0
        for gen in range(n_gens):
            gen_pct = ga_pct_start + (ga_pct_end - ga_pct_start) * (gen / max(n_gens, 1))
            progress.update_status(
                f"Phase 2d: GA generation {gen + 1}/{n_gens}…",
                pct=gen_pct)
            progress.pump_events()

            fitness = np.empty(len(population), dtype=np.float64)
            next_gen_pct = ga_pct_start + (ga_pct_end - ga_pct_start) * ((gen + 1) / max(n_gens, 1))
            for _ci, _chrom in enumerate(population):
                fitness[_ci] = _eval(_chrom)
                # Advance the bar smoothly across the population so it never
                # appears stuck while a generation evaluates.
                if (_ci & 3) == 0:
                    progress.update_progress(
                        gen_pct + (next_gen_pct - gen_pct) * (_ci + 1) / max(len(population), 1)
                    )
            progress.pump_events()

            rank = np.argsort(fitness)[::-1]
            population = [population[r] for r in rank]
            fitness = fitness[rank]

            history_best.append(fitness[0])
            history_avg.append(fitness.mean())
            history_worst.append(fitness[-1])

            new_pop = [p.copy() for p in population[:n_elite]]
            while len(new_pop) < pop_size:
                t_size = min(5, pop_size)
                t_idx  = np.random.choice(pop_size, t_size, replace=False)
                p1 = population[t_idx[np.argmax(fitness[t_idx])]]
                t_idx2 = np.random.choice(pop_size, t_size, replace=False)
                p2 = population[t_idx2[np.argmax(fitness[t_idx2])]]

                c1, c2, blend = self._ga_crossover(p1, p2)
                for child in [c1, c2, blend]:
                    child = self._ga_mutate(child, item_names, mut_rate)
                    child = self._ga_repair(child, item_names)
                    # Separate items that landed on top of each other, so the
                    # child is judged on its placement instead of collapsing
                    # to the overlap-penalty floor and leaving the starting
                    # layout as the only viable candidate.
                    child = self._ga_resolve_overlaps(child, item_names)
                    new_pop.append(child)
                    if len(new_pop) >= pop_size:
                        break
            population = new_pop[:pop_size]

            # Keep cache from growing without bound
            if len(fit_cache) > 4 * pop_size:
                fit_cache.clear()

        progress.update_status("Phase 2d: Final GA evaluation…", pct=ga_pct_end)
        progress.pump_events()
        final_fitness = np.array([_eval(c) for c in population], dtype=np.float64)
        best_idx   = int(np.argmax(final_fitness))
        best_chrom = population[best_idx]
        best_fit   = float(final_fitness[best_idx])

        current_fit = _fast_fitness(current_chrom)
        best_layout = self._ga_decode(best_chrom, item_names)

        best_score, best_breakdown = self._ga_compute_layout_score(
            best_chrom, item_names, base_params)
        current_score, current_breakdown = self._ga_compute_layout_score(
            current_chrom, item_names, base_params)

        pipe['ga_item_names']        = item_names
        pipe['ga_best_chrom']        = best_chrom
        pipe['ga_best_layout']       = best_layout
        # The chromosome's own values; ga_best_fit / _score / _breakdown are
        # set after the apply step from the layout actually placed.
        pipe['ga_chrom_fit']         = best_fit
        pipe['ga_current_fit']       = current_fit
        pipe['ga_chrom_score']       = best_score
        pipe['ga_chrom_breakdown']   = best_breakdown
        pipe['ga_current_score']     = current_score
        pipe['ga_current_breakdown'] = current_breakdown
        pipe['ga_history_best']      = history_best
        pipe['ga_history_avg']       = history_avg
        pipe['ga_history_worst']     = history_worst
        pipe['ga_pop_size']  = pop_size
        pipe['ga_n_gens']    = n_gens
        # The pipeline GA uses the ANALYTIC surrogate fitness (score ->
        # cited elasticities -> expected revenue), not per-chromosome MC --
        # record that honestly so the report doesn't claim MC evaluations.
        pipe['ga_fitness_kind'] = 'analytic_surrogate'
        pipe['ga_mc_iters']  = 0
        pipe['ga_mc_days']   = 30

        # -- APPLY BEST LAYOUT -----------------------------------
        # From here on the store is being modified. Closing the progress
        # window must no longer abort the run: the layout would stay moved
        # while the POST window, the A/B test and the report never happen,
        # and the user would be told the run was cancelled.
        progress.disable_cancellation()
        progress.update_status("Phase 3: Applying optimized layout…", pct=90)
        progress.pump_events()
        changes = []
        # Lets a failure from here on tell the user the store was changed.
        pipe['layout_applied'] = True

        # Impulse items are placed exactly where the GA put them: the
        # chromosome already passed the impulse-near-checkout projection in
        # _ga_repair, which keeps the pull only when the item's own zone can
        # reach the radius. Pulling again here would override the GA's choice
        # for an item whose zone lies outside that radius and pin every such
        # item to the zone corner nearest the checkout.
        for opt_key, (nx, ny) in best_layout.items():
            fid, source_name = item_source_map.get(
                opt_key,
                self._ga_resolve_item_key(opt_key)
                if hasattr(self, '_ga_resolve_item_key')
                else (item_floor_map.get(opt_key, self.current_floor), opt_key),
            )
            floor_items = self.floors.get(fid, {}).get('items', {})
            if source_name not in floor_items:
                continue
            old = floor_items[source_name]['position']
            dist = math.hypot(nx - old[0], ny - old[1])
            if dist > 0.3:
                prev_floor = self.current_floor
                try:
                    self.current_floor = fid
                    safe = self._find_safe_position_in_section(source_name, nx, ny)
                finally:
                    self.current_floor = prev_floor

                # Stored positions may be lists while the search returns
                # tuples, and (x, y) != [x, y] is always True; compare the
                # coordinates so an unchanged fallback is not logged as a move.
                moved = math.hypot(safe[0] - old[0], safe[1] - old[1])
                if moved > 1e-9:
                    floor_items[source_name]['position'] = safe
                    display = opt_key if opt_key == source_name else f"{source_name} ({opt_key})"
                    changes.append(
                        f"[F{fid}] Moved '{display}' ({old[0]:.1f},{old[1]:.1f}) -> "
                        f"({safe[0]:.1f},{safe[1]:.1f})  d={moved:.1f}m"
                    )

        prev_floor = self.current_floor
        try:
            for fid in self.floors:
                self.current_floor = fid
                try:
                    self._repair_item_overlaps()
                except Exception:
                    import traceback as _tb
                    print(f'[optimize] _repair_item_overlaps failed on floor {fid}:')
                    _tb.print_exc()
        finally:
            self.current_floor = prev_floor

        # Items are obstacles for the live agents and carry the zone
        # attribution, so the POST window needs both the path grid and the
        # zone table rebuilt from the new positions.
        try:
            self.customer_simulation.geometry_dirty = True
            self.customer_simulation.invalidate_zones_cache()
        except Exception:
            pass

        self.redraw()
        pipe['changes'] = changes
        analytics['optimization_history'] = changes.copy()

        self._ga_best_chromosome = best_chrom
        self._ga_best_layout     = best_layout
        self._ga_item_names      = item_names

        post_snapshot = self._snapshot_layout()
        pipe['post_snapshot'] = post_snapshot

        # Score the layout that was actually placed rather than best_chrom:
        # the apply step keeps sub-0.3 m moves in place, falls back to
        # collision-free spots and repairs overlaps, so the store measured in
        # the POST window can differ from the chromosome. Rebuilding the
        # catalog refreshes the positions _ga_encode reads.
        self._ga_get_movable_items()
        applied_chrom = self._ga_encode(item_names)
        applied_fit   = _fast_fitness(applied_chrom)
        applied_score, applied_breakdown = self._ga_compute_layout_score(
            applied_chrom, item_names, base_params)
        pipe['ga_best_fit']       = applied_fit
        pipe['ga_best_score']     = applied_score
        pipe['ga_best_breakdown'] = applied_breakdown

        # -- MC PROJECTION: CURRENT vs APPLIED LAYOUT (chunked so the bar moves) --
        # Both layouts go through the GA fitness's score -> parameter
        # transform (conversion, layout-induced abandonment, impulse, and
        # basket size as a revenue-per-converter multiplier, since
        # mc_engine does not turn basket size into revenue). Each chunk
        # pair replays one random stream, so the lift reflects the layout
        # change, not how each side was built or MC noise.
        def _layout_mc_kwargs(score, bkd):
            conv, imp, bsk, rev_mult = self._opt_layout_revenue_params(
                score, bkd, base_params, sa_weights, markov_p_purchase)
            return dict(
                cph=base_params['customers_per_hour'],
                conv=conv,
                rev_mean=base_params['rev_per_converting_customer'] * rev_mult,
                rev_std=base_params['rev_std'] * rev_mult,
                imp_rate=imp,
                imp_val=base_params['avg_impulse_value'],
                avg_bsk=bsk,
                std_bsk=base_params['std_basket_size'],
                observed_baskets=base_params['basket_sizes_observed'],
                n_days=mc_days, n_iter=opt_chunk_iters,
                op_hours=10.0, wknd_mult=DEFAULT_WEEKEND_MULTIPLIER,
                monthly_growth=0.0,
            )

        def _summarize(totals_list, daily_list):
            t = np.concatenate(totals_list)
            return {
                'mean':   float(np.mean(t)),
                'std':    float(np.std(t)),
                'median': float(np.median(t)),
                'p5':     float(np.percentile(t, 5)),
                'p95':    float(np.percentile(t, 95)),
                'totals': t,
                'daily_means': np.mean(daily_list, axis=0),
            }

        opt_chunks  = 4
        opt_chunk_iters = max(1, mc_iters // opt_chunks)
        cur_kw = _layout_mc_kwargs(current_score, current_breakdown)
        opt_kw = _layout_mc_kwargs(applied_score, applied_breakdown)
        proj_seed = int(np.random.randint(0, 2**31 - 1))
        cur_totals, cur_daily, opt_totals, opt_daily = [], [], [], []
        for ci in range(opt_chunks):
            pct_now = 96 + 3.0 * ci / opt_chunks   # 96 .. 99
            progress.update_status(
                f"Phase 2e: MC projection, current vs optimized layout {ci+1}/{opt_chunks}…",
                pct=pct_now)
            progress.pump_events()
            r_cur = mc_engine(**cur_kw, rng=np.random.RandomState(proj_seed + ci))
            r_opt = mc_engine(**opt_kw, rng=np.random.RandomState(proj_seed + ci))
            cur_totals.append(r_cur['totals']); cur_daily.append(r_cur['daily_means'])
            opt_totals.append(r_opt['totals']); opt_daily.append(r_opt['daily_means'])
            progress.pump_events()

        # The Phase 2a run on the raw calibrated parameters is kept for
        # reference; the baseline the report compares against must be the
        # current layout under the same model as the optimized projection.
        pipe['mc_calibrated_baseline'] = mc_baseline
        pipe['mc_baseline']  = _summarize(cur_totals, cur_daily)
        pipe['mc_optimized'] = _summarize(opt_totals, opt_daily)

        progress.update_status("Phase 2 complete.", pct=100)
        progress.pump_events()
        progress.close()
        analytics['_opt_pipeline_data'] = pipe

        improvement = (applied_fit - current_fit) / max(current_fit, 1e-6) * 100
        messagebox.showinfo(
            "Optimization",
            f"Phase 2 complete. {len(changes)} items moved.\n"
            f"GA projected lift: {improvement:+.1f}%\n\n"
            f"Phase 4/5: Collecting POST-optimization data for {dur // 60} min...",
            parent=self.tk_root
        )

        self._start_simulation()
        analytics['post_window_start'] = sim.sim_time
        analytics['phase'] = 'post'
        self._schedule_window_end('post_window_start',
                                  self._finalize_optimization_metrics)

    def _opt_layout_revenue_params(self, score, bkd, p, sa_weights,
                                   markov_p_purchase):
        """Layout score -> (conversion, impulse rate, basket size, revenue
        multiplier) for the calibrated parameters ``p``.

        The single score-to-parameter transform behind the pipeline GA
        fitness, the MC projection and the A/B comparison, so the lifts the
        report shows side by side all come from the same model.

        score-to-parameter elasticities cited per retail_literature.py:
          conversion lift ~ Hui (2009) detour-distance elasticity
          impulse lift    ~ Hui, Inman (2013) in-zone gap
          basket lift     ~ Hui, Inman (2013) per-meter unplanned

        ``markov_p_purchase`` is the chain's P(Purchase | Enter), which the
        report shows as a flow diagnostic; it does not rescale the
        conversion. ``p['conversion_rate']`` already is the purchase
        probability per visitor, while the chain's absorption share comes
        from the live agents' list-driven behaviour (or a prior), so a
        P / conversion factor would move the conversion level -- by up to
        about 3x in a calibrated session -- for reasons unrelated to the
        layout.
        """
        from retail_literature import (
            ELASTICITY_CONV_BASE, ELASTICITY_CONV_GAIN_MAX,
            ELASTICITY_IMP_BASE,  ELASTICITY_IMP_GAIN_MAX,
            ELASTICITY_BSK_BASE,  ELASTICITY_BSK_GAIN_MAX,
            ABANDON_FLOW_COEF, ABANDON_SECTION_COEF,
            ABANDON_BOTTLENECK_COEF, ABANDON_FLOOR_FRAC,
        )
        conv_weight    = sa_weights.get('Conversion rate', 0.15)
        impulse_weight = sa_weights.get('Impulse rate',    0.10)
        basket_weight  = sa_weights.get('Basket size',     0.10)
        conv_base  = p['conversion_rate']
        bsk_base   = p['avg_basket_size']
        imp_base   = p['impulse_rate']
        aband_base = p.get('abandonment_rate', 0.0)

        # Conversion: midpoint of 50-75% lift band, modulated by the
        # SA-derived weight on conversion-rate sensitivity.
        conv_adj = conv_base * (1.0 + score *
            (ELASTICITY_CONV_BASE + conv_weight * ELASTICITY_CONV_GAIN_MAX))
        conv_adj = min(max(conv_adj, 0.01), 0.99)

        # Impulse: midpoint of Hui Inman 2013 50-70% in-zone gap.
        imp_adj = imp_base * (1.0 + bkd.get('impulse', 0) *
            (ELASTICITY_IMP_BASE + impulse_weight * ELASTICITY_IMP_GAIN_MAX))
        imp_adj = min(max(imp_adj, 0.0), 0.99)

        # Basket: conservative 10-30% lift band (largest standing
        # uncertainty; flagged in retail_literature.py).
        bsk_adj = bsk_base * (1.0 + score *
            (ELASTICITY_BSK_BASE + basket_weight * ELASTICITY_BSK_GAIN_MAX))

        # Abandonment recovery: see ABANDON_* in retail_literature.py.
        # RELATIVE to the baseline abandonment rate: the calibrated
        # conversion is already net of baseline abandonment, so only the
        # layout-induced change may move it.
        abandon_adj = aband_base * max(ABANDON_FLOOR_FRAC,
            1.0 - bkd.get('flow', 0) * ABANDON_FLOW_COEF
                - bkd.get('section_compliance', 0) * ABANDON_SECTION_COEF
                + bkd.get('bottleneck_penalty', 0) * ABANDON_BOTTLENECK_COEF)
        abandon_adj = min(max(abandon_adj, 0.0), 0.5)

        final_conv = conv_adj * (1.0 - abandon_adj) \
            / max(1.0 - min(aband_base, 0.95), 1e-6)
        final_conv = min(max(final_conv, 0.01), 0.99)

        # Basket-size lift enters MULTIPLICATIVELY on revenue-per-customer
        # (more items at the same average item price => proportionally
        # higher spend); adding the basket-size COUNT as dollars would be a
        # unit error that inflates the projected lift.
        rev_mult = bsk_adj / max(bsk_base, 1e-9)
        return final_conv, imp_adj, bsk_adj, rev_mult

    def _finalize_optimization_metrics(self):
        """Public entry point for the POST-window callback.

        Releases the one-pipeline-at-a-time guard however finalization
        ends: early return, report shown, or an exception."""
        try:
            self._finalize_optimization_metrics_inner()
        finally:
            self._opt_running = False

    def _finalize_optimization_metrics_inner(self):
        """
        Phase 5: POST window done.
          - Collect post revenue
          - Run A/B test (PRE vs POST) with statistical tests
          - Build comprehensive sectioned report
        """
        # Bail silently if the user closed the window before the
        # measurement-duration after-callback fired.
        try:
            if not self.tk_root.winfo_exists():
                return
        except Exception:
            return

        sim = self.customer_simulation
        A   = sim.analytics
        pipe = A.get('_opt_pipeline_data', {})

        self._stop_simulation()

        pre  = A.get('pre_window_revenue', 0.0)
        post = A.get('post_window_revenue', 0.0)
        A['pre_optimization_revenue']  = pre
        A['post_optimization_revenue'] = post
        if pre > 0:
            A['optimization_impact'] = (post - pre) / pre * 100.0
            A.pop('optimization_impact_note', None)
        else:
            # A silent 0% would have masked a real failure: PRE collected no
            # revenue (no customers spawned in the measurement window).
            # Surface the cause explicitly so reports + the user know.
            A['optimization_impact'] = 0.0
            A['optimization_impact_note'] = (
                "PRE-window revenue was $0 — no customers spawned during "
                "the measurement window. The live A/B lift is not "
                "computable; see the MC-projected lift instead.")

        performance_data = self._analyze_current_performance()

        # -- A/B STATISTICAL COMPARISON (MC-based) ---------------
        ab_results = {}
        mc_ab_iters = 1000
        mc_ab_days  = 30
        for label, snap_key in [('A', 'pre_snapshot'), ('B', 'post_snapshot')]:
            snap = pipe.get(snap_key)
            if not snap or not snap.get('params'):
                continue
            p = snap['params']
            score, score_bkd = self._ab_layout_score(snap, with_breakdown=True)

            # Same score -> parameter transform and SA weights as the GA
            # fitness and the MC projection, so PRE and POST are evaluated
            # on the same footing as the other lifts in the report:
            # conversion, layout-induced abandonment, the IMPULSE
            # component driving the impulse lift, and basket size as a
            # revenue-per-converter multiplier.
            conv_adj, imp_adj, _bsk_adj, rev_mult = self._opt_layout_revenue_params(
                score, score_bkd, p, pipe.get('sa_weights', {}),
                pipe.get('markov_p_purchase', p['conversion_rate']))
            rev_mean = p['rev_per_converting_customer'] * rev_mult
            rev_sd   = max(p['rev_std'] * rev_mult, 0.01)
            imp_val  = max(p['avg_impulse_value'], 0.01)
            imp_sd   = max(p['avg_impulse_value'] * 0.3, 0.01)

            day_mult = np.ones(7)
            day_mult[5] = DEFAULT_WEEKEND_MULTIPLIER
            day_mult[6] = DEFAULT_WEEKEND_MULTIPLIER
            daily_rev = np.zeros((mc_ab_iters, mc_ab_days))
            for d in range(mc_ab_days):
                dow = d % 7
                lam = max(p['customers_per_hour'] * 10.0 * day_mult[dow], 0.1)
                n_cust = np.random.poisson(lam, mc_ab_iters)
                n_conv = np.random.binomial(n_cust, conv_adj)
                # A day's revenue is a sum of n i.i.d. customer spends,
                # N(n*mu, sqrt(n)*sigma) as in mc_engine. Scaling a single
                # per-customer draw by n would make the spread grow like n.
                nc_f = n_conv.astype(np.float64)
                base = np.where(
                    n_conv > 0,
                    np.random.normal(nc_f * rev_mean, np.sqrt(nc_f) * rev_sd,
                                     mc_ab_iters),
                    0.0)
                base = np.maximum(base, 0.0)
                n_imp = np.random.binomial(np.maximum(n_conv, 0), imp_adj)
                ni_f = n_imp.astype(np.float64)
                imp = np.where(
                    n_imp > 0,
                    np.random.normal(ni_f * imp_val, np.sqrt(ni_f) * imp_sd,
                                     mc_ab_iters),
                    0.0)
                imp = np.maximum(imp, 0.0)
                daily_rev[:, d] = base + imp
            totals = daily_rev.sum(axis=1)
            ab_results[label] = {
                'totals': totals, 'daily_rev': daily_rev,
                'mean': totals.mean(), 'std': totals.std(),
                'median': np.median(totals),
                'p5': np.percentile(totals, 5),
                'p95': np.percentile(totals, 95),
                'daily_means': daily_rev.mean(axis=0),
                'score': score, 'conv_adj': conv_adj, 'imp_adj': imp_adj,
            }

        ab_tests = {}
        if 'A' in ab_results and 'B' in ab_results:
            t_stat, p_value = sp_stats.ttest_ind(
                ab_results['A']['totals'], ab_results['B']['totals'], equal_var=False)
            ks_stat, ks_p = sp_stats.ks_2samp(
                ab_results['A']['totals'], ab_results['B']['totals'])
            pooled_std = np.sqrt(
                (ab_results['A']['std'] ** 2 + ab_results['B']['std'] ** 2) / 2)
            cohens_d = (ab_results['B']['mean'] - ab_results['A']['mean']) / max(pooled_std, 1e-6)
            n_b_wins = np.sum(ab_results['B']['totals'] > ab_results['A']['totals'])

            # PRIMARY metric: effect size + bootstrap CI on the mean lift
            # (audit #1). The Welch p-value on two simulated distributions
            # shrinks toward 0 as MC iterations grow BY CONSTRUCTION (the
            # parameter shift is deterministic), so it must not be read as
            # real-world statistical significance; it is retained only as
            # a distribution-separation diagnostic.
            diffs = ab_results['B']['totals'] - ab_results['A']['totals']
            rng_bs = np.random.default_rng(0)
            boots = np.array([
                rng_bs.choice(diffs, size=diffs.size, replace=True).mean()
                for _ in range(2000)])
            lift_mean = float(diffs.mean())
            lift_lo, lift_hi = (float(np.percentile(boots, 2.5)),
                                float(np.percentile(boots, 97.5)))
            ab_tests = {
                't_stat': t_stat, 'p_value': p_value,
                'ks_stat': ks_stat, 'ks_p': ks_p,
                'cohens_d': cohens_d,
                'significant': p_value < 0.05,
                'alpha': 0.05,
                'p_b_better': n_b_wins / mc_ab_iters * 100,
                'lift_mean': lift_mean,
                'lift_ci95': (lift_lo, lift_hi),
                'primary_metric': 'cohens_d + lift_ci95',
                'p_value_caveat': (
                    'p-value compares two SIMULATED distributions whose '
                    'parameters differ by construction; report effect size '
                    'and the lift CI, not p, as the substantive result.'),
            }

        pipe['ab_results'] = ab_results
        pipe['ab_tests']   = ab_tests
        pipe['performance_data'] = performance_data
        pipe['real_pre_rev']  = pre
        pipe['real_post_rev'] = post

        self._show_optimization_results(
            A.get('optimization_history', []),
            performance_data
        )
        

