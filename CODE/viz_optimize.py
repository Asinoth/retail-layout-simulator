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
from retail_literature import (DEFAULT_OP_HOURS_PER_DAY,
                               DEFAULT_WEEKEND_MULTIPLIER)

import copy


class OptimizeMixin:
    """Optimization workflow start, post-window measurement, finalization metrics."""

    def _optimize_layout(self):
        """
        Master optimization pipeline:
          Phase 1 -- PRE-window: collect baseline simulation data
          Phase 2 -- Run all analytical engines (MC, Sensitivity, Markov, GA)
          Phase 3 -- Apply GA-optimized layout
          Phase 4 -- POST-window: collect post-optimization data
          Phase 5 -- Re-score the as-built and applied layouts under the
                     analytics the POST window added to, compare them
                     under common random numbers, build the report
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

        # Iteration counts for the analytical engines further down, defined
        # in one place. Values mirror the proven old-codebase defaults.
        sa_days      = mc_days
        sa_iters     = 300
        pop_size     = 40
        n_gens       = 20
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
                op_hours=DEFAULT_OP_HOURS_PER_DAY, wknd_mult=DEFAULT_WEEKEND_MULTIPLIER,
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
        # A diagnostic for the report's Sec. 3; the optimizer's fitness does
        # not read it (_opt_sensitivity_tornado).
        progress.update_status("Phase 2b: Sensitivity analysis...", pct=12)
        progress.pump_events()
        n_sa_runs = 2 * len(self._OPT_TORNADO_PARAMS)

        def _tornado_step(done, label, side):
            progress.update_status(
                f"Phase 2b: Sensitivity {done}/{n_sa_runs} — {label} ({side})…",
                pct=12 + 18.0 * done / max(n_sa_runs, 1))
            progress.pump_events()

        pipe['tornado'] = self._opt_sensitivity_tornado(
            base_params, n_days=sa_days, n_iter=sa_iters,
            seed=int(np.random.randint(0, 2**31 - 1)), on_step=_tornado_step)

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
            # No GA, no projection and no re-scored comparison: the report
            # says why and prints no lift (``pipe['mc_baseline']`` is still
            # the Phase-2a run on the raw calibrated parameters, which no
            # layout was compared against).
            pipe['not_optimized_reason'] = self._OPT_TOO_FEW_ITEMS
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
        # The elasticities act on the score difference from the layout on
        # the floor as the optimization starts (the PRE layout), so that
        # layout reproduces the calibrated conversion, basket and spend in
        # the fitness and the MC projection, which run while the
        # simulation is stopped. The re-scored comparison of Phase 5
        # re-anchors at the same PRE layout, re-scored under the analytics
        # the live POST window has since added to (``_ab_arm_scores``).
        from layout_objective import SCORE_ANCHOR_KEY
        score_anchor = self._ga_score_anchor(
            item_names, base_params, source='floor_at_optimize_start')
        base_params[SCORE_ANCHOR_KEY] = score_anchor
        pipe['score_anchor'] = score_anchor

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
        # Whether the candidates are held to the shared repair's aisle and
        # reach rules on this floor (``_ga_repair_note``); the report says so
        # when they are not.
        pipe['ga_repair_note'] = self._ga_repair_note(item_names)
        population = [self._ga_resolve_overlaps(
            self._ga_repair(current_chrom.copy(), item_names), item_names)]
        # Each item's move box: its section less the aisles the shared
        # repair keeps (``_ga_move_boxes``), as in the GA tab and the
        # headless GA, so the seeds start inside the feasible set's space.
        boxes = self._ga_move_boxes(item_names)
        for _ in range(pop_size - 1):
            noisy = current_chrom.copy()
            for i, (w, h) in enumerate(item_sizes):
                # Perturb inside the item's own section and repair, as the GA
                # tab does. Shop-scale noise clipped only to the shop bounds
                # pushes nearly every item out of its section, so the seeds
                # would be infeasible and carried forward as elites.
                bounds = boxes[i]
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

        # The fitness is the exact expected revenue of the Monte Carlo
        # projection over the same horizon (experiments.closed_form): the
        # layout reaches the engine only through deterministic drivers, so
        # that mean has a closed form, and it is the quantity the headless
        # experiments score layouts with. The projection below therefore
        # converges to exactly the lift the GA reports.
        from experiments.closed_form import expected_revenue
        ga_horizon_days = mc_days

        def _fast_fitness(chromosome):
            """Exact expected ``ga_horizon_days`` revenue of one layout.

            The score -> driver transform is ``layout_objective.
            layout_drivers`` at the band-midpoint elasticities, anchored at
            the PRE layout -- the transform the MC projection and the
            re-scored comparison use (``_opt_layout_drivers``)."""
            score, bkd = self._ga_compute_layout_score(
                chromosome, item_names, base_params)
            return expected_revenue(base_params, ga_horizon_days,
                                    score=score, breakdown=bkd)

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
        # The pipeline GA scores each chromosome by the closed-form mean of
        # the MC projection, not by per-chromosome MC -- recorded so the
        # report does not claim MC evaluations.
        pipe['ga_fitness_kind'] = 'closed_form'
        pipe['ga_horizon_days'] = ga_horizon_days

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
        # Both layouts go through the fitness's score -> driver transform
        # (conversion, layout-induced abandonment, impulse, and basket size
        # as a spend-per-converter multiplier, since mc_engine does not
        # turn basket size into revenue) and are compared under common
        # random numbers (layout_comparison), so the paired lift reflects
        # the layout change and its interval is Monte Carlo precision.
        # Both arms run mc_engine itself, so a day's spend is the engine's
        # law (MC_SPEND_LAW: sim_calibration.lognormal_spend_total, the
        # moment-matched lognormal), not a zero-floored normal.
        from layout_comparison import paired_comparison
        opt_chunks  = 4
        opt_chunk_iters = max(1, mc_iters // opt_chunks)
        cur_kw = self._opt_layout_mc_kwargs(
            current_score, current_breakdown, base_params,
            n_days=mc_days, n_iter=opt_chunk_iters)
        opt_kw = self._opt_layout_mc_kwargs(
            applied_score, applied_breakdown, base_params,
            n_days=mc_days, n_iter=opt_chunk_iters)
        proj_seed = int(np.random.randint(0, 2**31 - 1))

        def _projection_chunk_done(ci):
            progress.update_status(
                f"Phase 2e: MC projection, current vs optimized layout "
                f"{ci + 1}/{opt_chunks}…",
                pct=96 + 3.0 * (ci + 1) / opt_chunks)     # 96 .. 99
            progress.pump_events()

        progress.update_status(
            "Phase 2e: MC projection, current vs optimized layout…", pct=96)
        progress.pump_events()
        projection = paired_comparison(cur_kw, opt_kw, proj_seed,
                                       n_chunks=opt_chunks,
                                       between_chunks=_projection_chunk_done)

        # The Phase 2a run on the raw calibrated parameters is kept for
        # reference; the baseline the report compares against must be the
        # current layout under the same model as the optimized projection.
        pipe['mc_calibrated_baseline'] = mc_baseline
        pipe['projection']   = projection
        pipe['mc_baseline']  = projection['A']
        pipe['mc_optimized'] = projection['B']

        progress.update_status("Phase 2 complete.", pct=100)
        progress.pump_events()
        progress.close()
        analytics['_opt_pipeline_data'] = pipe

        improvement = (applied_fit - current_fit) / max(current_fit, 1e-6) * 100
        messagebox.showinfo(
            "Optimization",
            f"Phase 2 complete. {len(changes)} items moved.\n"
            f"Projected {ga_horizon_days}-day lift under the layout model: "
            f"{improvement:+.2f}%\n\n"
            f"Phase 4/5: Collecting POST-optimization data for {dur // 60} min...",
            parent=self.tk_root
        )

        self._start_simulation()
        analytics['post_window_start'] = sim.sim_time
        analytics['phase'] = 'post'
        self._schedule_window_end('post_window_start',
                                  self._finalize_optimization_metrics)

    #: Why the report prints no lift when the GA was skipped (read by
    #: ``_show_optimization_results`` from ``pipe['not_optimized_reason']``).
    _OPT_TOO_FEW_ITEMS = ("fewer than two movable items, no layout was "
                          "optimized")

    #: The tornado's bars: label -> (base_params key, lower clamp, upper
    #: clamp). Basket size has no bar: mc_engine never prices it, and
    #: perturbing it through spend per converter only repeated the
    #: Revenue/customer bar.
    _OPT_TORNADO_PARAMS = {
        'Customers/hr':    ('customers_per_hour',          None,   None),
        'Conversion rate': ('conversion_rate',             0.001,  0.999),
        'Revenue/customer':('rev_per_converting_customer', 0.01,   None),
        'Impulse rate':    ('impulse_rate',                0.0,    0.999),
        'Impulse value':   ('avg_impulse_value',           0.01,   None),
    }
    _OPT_TORNADO_ENGINE_KEYS = {
        'customers_per_hour': 'cph',
        'conversion_rate': 'conv',
        'rev_per_converting_customer': 'rev_mean',
        'impulse_rate': 'imp_rate',
        'avg_impulse_value': 'imp_val',
    }

    def _opt_sensitivity_tornado(self, base_params, n_days, n_iter, seed,
                                 pct=0.20, on_step=None):
        """One-at-a-time +/-``pct`` tornado over the engine's revenue
        inputs, at the calibrated values ``base_params``.

        A diagnostic the report shows in its Sec. 3, and nothing more: the
        optimizer's fitness runs at the midpoints of the cited elasticity
        bands, as every headless experiment does. (These swings once set
        the elasticities inside their bands, but revenue is a product of
        the drivers, so each swing is about the same share of revenue on
        any store and the weights encoded that identity rather than
        anything about the store.) Every low/high run replays one stream
        (common random numbers), so a swing reflects the parameter change
        rather than Monte Carlo noise.

        'Conversion rate' moves the conversion at a fixed visitor rate. On
        a transactional calibration the visitor rate is itself derived as
        buyers / assumed conversion, so this bar varies only one of the
        rate's two uses; it is the sensitivity to a change in how many
        visitors buy, not to the assumed rate.

        ``on_step(done, label, side)`` is called before each run (the GUI
        updates its progress bar there). Returns label -> ``{'baseline',
        'lo_val', 'hi_val', 'lo_rev', 'hi_rev', 'swing'}``."""
        def _kw(override_key, override_val):
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
                n_days=n_days, n_iter=n_iter,
                op_hours=DEFAULT_OP_HOURS_PER_DAY,
                wknd_mult=DEFAULT_WEEKEND_MULTIPLIER,
                monthly_growth=0.0,
            )
            kw[self._OPT_TORNADO_ENGINE_KEYS[override_key]] = override_val
            if override_key == 'rev_per_converting_customer':
                # Keep the calibrated coefficient of variation of spend.
                base_rm = base_params['rev_per_converting_customer']
                kw['rev_std'] = base_params['rev_std'] * (
                    override_val / max(base_rm, 1e-6))
            return kw

        tornado = {}
        done = 0
        for label, (pkey, lo_clamp, hi_clamp) in self._OPT_TORNADO_PARAMS.items():
            bval = base_params[pkey]
            lo_val = bval * (1.0 - pct)
            hi_val = bval * (1.0 + pct)
            if lo_clamp is not None:
                lo_val = max(lo_clamp, lo_val)
            if hi_clamp is not None:
                hi_val = min(hi_clamp, hi_val)
            res = {}
            for side, val in (('low', lo_val), ('high', hi_val)):
                done += 1
                if on_step is not None:
                    on_step(done, label, side)
                res[side] = mc_engine(**_kw(pkey, val),
                                      rng=np.random.RandomState(seed))
            tornado[label] = {
                'baseline': bval,
                'lo_val': lo_val, 'hi_val': hi_val,
                'lo_rev': res['low']['mean'], 'hi_rev': res['high']['mean'],
                'swing': abs(res['high']['mean'] - res['low']['mean']),
            }
        return tornado

    def _opt_layout_drivers(self, score, bkd, p, anchor=None):
        """Revenue drivers of a layout scored ``(score, bkd)`` under the
        calibrated parameters ``p``: ``layout_objective.layout_drivers`` at
        the midpoints of the cited elasticity bands, queue channel
        included.

        The one transform behind the pipeline GA fitness, its MC
        projection, the re-scored comparison after the POST window and the
        What-If tab, and the one the headless experiments and the closed
        form use, so a layout pair gets the same projected lift in every
        one of them. (The pipeline once set the elasticities inside their
        bands from the tornado's swing weights; those weights encode the
        product form of revenue, not the store, and made the same layout
        pair project a different lift here than in What-If.)

        ``anchor`` is the layout the elasticities are differenced against
        (``layout_objective.make_anchor``); by default the one ``p``
        carries. The pipeline and the What-If tab pass the layout that was
        on the floor when they started, which therefore reproduces the
        calibrated conversion, basket and spend. ``rev_mult`` scales both
        the mean and the SD of spend per converter.

        The Markov chain's P(Purchase | Enter) is a flow diagnostic in the
        report and does not enter: ``p['conversion_rate']`` already is the
        purchase probability per visitor, while the chain's absorption
        share comes from the live agents' list-driven behaviour (or a
        prior), so a P / conversion factor would move the conversion level
        -- by up to about 3x in a calibrated session -- for reasons
        unrelated to the layout."""
        from layout_objective import layout_drivers
        return layout_drivers(score, bkd, p, anchor=anchor)

    def _opt_layout_mc_kwargs(self, score, bkd, p, n_days, n_iter,
                              anchor=None):
        """``mc_engine`` keyword arguments for a layout scored
        ``(score, bkd)``: ``_opt_layout_drivers`` through
        ``layout_objective.layout_mc_kwargs``, as ``_ga_fitness`` builds
        them, so ``experiments.closed_form.mc_expected_total`` of the
        result is the layout's exact expected revenue."""
        from layout_objective import layout_mc_kwargs
        return layout_mc_kwargs(
            self._opt_layout_drivers(score, bkd, p, anchor=anchor), p,
            n_days=n_days, n_iter=n_iter)

    def _ab_arm_scores(self, pipe):
        """Scores of the Phase-5 comparison's arms -- A the as-built (PRE)
        layout, B the applied one -- and the anchor they share.

        The traffic, revenue-placement and bottleneck criteria read the heat
        map and the bottleneck counts, and the live POST window has added to
        both since ``pipe['score_anchor']`` was scored at optimize start. So
        the PRE snapshot no longer scores that anchor, and differencing
        against it would move the PRE arm off the calibrated conversion,
        basket and spend and give both arms the same analytics-drift
        offset. Both arms are therefore scored now, under one state of the
        analytics, and anchored at the PRE layout scored under that same
        state: arm A reproduces the calibrated inputs exactly and arm B
        differs from it by the layout alone. ``pipe['score_anchor']`` is
        left as it was for the GA and the projection.

        Returns ``(scores, anchor)``: ``scores`` maps 'A' / 'B' to
        ``(score, breakdown)`` for each snapshot that carries parameters.
        Without a PRE snapshot (the A/B then does not run) the anchor falls
        back to ``pipe['score_anchor']``."""
        from layout_objective import make_anchor
        scores = {}
        for label, snap_key in (('A', 'pre_snapshot'), ('B', 'post_snapshot')):
            snap = pipe.get(snap_key)
            if snap and snap.get('params'):
                scores[label] = self._ab_layout_score(snap, with_breakdown=True)
        if 'A' in scores:
            anchor = make_anchor(*scores['A'], source='pre_snapshot_at_ab')
        else:
            anchor = pipe.get('score_anchor')
        return scores, anchor

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
          - Record the live PRE / POST window revenues (descriptive only)
          - Re-score the as-built and applied layouts under the analytics
            the POST window added to, and compare them under common
            random numbers (paired lift with its interval)
          - Build comprehensive sectioned report

        The comparison is a model projection of two layouts, not a
        measurement of the two windows: each window is one short live run
        that starts from an empty store, and no lift is computed from them.
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

        # The live windows' revenues are kept as a description of what the
        # two short runs saw. No lift is taken between them: each window
        # is one run of measurement_duration simulated seconds that starts
        # from an empty store, so its revenue is dominated by which few
        # customers happened to finish inside it, and the comparison below
        # is what estimates the layout effect.
        pre  = A.get('pre_window_revenue', 0.0)
        post = A.get('post_window_revenue', 0.0)
        A['pre_optimization_revenue']  = pre
        A['post_optimization_revenue'] = post

        performance_data = self._analyze_current_performance()

        # -- RE-SCORED COMPARISON: AS-BUILT vs APPLIED LAYOUT ------
        # Both arms scored under one state of the analytics and anchored at
        # the as-built (PRE) layout scored under it (see _ab_arm_scores),
        # and projected under the PRE snapshot's calibrated inputs, so the
        # arms differ by the layout alone; common random numbers pair the
        # draws (layout_comparison). Both arms run mc_engine itself, so a
        # day's spend is the engine's law (MC_SPEND_LAW:
        # sim_calibration.lognormal_spend_total), not a zero-floored normal.
        from layout_comparison import paired_comparison
        mc_ab_iters = 1000
        mc_ab_days  = 30
        ab_scores, ab_anchor = self._ab_arm_scores(pipe)
        pipe['ab_score_anchor'] = ab_anchor
        comparison = None
        if 'A' in ab_scores and 'B' in ab_scores:
            p = pipe['pre_snapshot']['params']
            drivers = {label: self._opt_layout_drivers(*ab_scores[label], p,
                                                       anchor=ab_anchor)
                       for label in ('A', 'B')}
            kws = {label: self._opt_layout_mc_kwargs(
                       *ab_scores[label], p, n_days=mc_ab_days,
                       n_iter=mc_ab_iters, anchor=ab_anchor)
                   for label in ('A', 'B')}
            comparison = paired_comparison(
                kws['A'], kws['B'], int(np.random.randint(0, 2**31 - 1)))
            comparison['scores'] = {label: float(ab_scores[label][0])
                                    for label in ('A', 'B')}
            comparison['drivers'] = drivers

        pipe['ab_comparison'] = comparison
        pipe['performance_data'] = performance_data
        pipe['real_pre_rev']  = pre
        pipe['real_post_rev'] = post

        self._show_optimization_results(
            A.get('optimization_history', []),
            performance_data
        )
        

