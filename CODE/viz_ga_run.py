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


def item_zone_name(data, walls):
    """Name of the zone wall an item is constrained to, or None.

    Items carry a stamped ``zone`` because a department split over two
    gondolas has one ``Section_<cat>_<i>`` wall per gondola, and looking up
    ``Section_<category>`` alone would pull the second gondola's items into
    the first. The stamp is honoured only while it still belongs to the
    item's category and the wall exists on the item's own floor: changing an
    item's category (marking it as impulse, for example) leaves the old
    department's stamp behind, and clipping the item back into that
    department would undo the move the user just made. ``walls`` is the wall
    dict of the floor the item sits on."""
    cat = data.get('category', '')
    base = f"Section_{cat}"
    zone = data.get('zone')
    if zone and zone in walls:
        suffix = zone[len(base):] if zone.startswith(base) else None
        if suffix == '' or (suffix and suffix.startswith('_') and suffix[1:].isdigit()):
            return zone
    return base if base in walls else None


class GARunMixin:
    """Genetic-algorithm fitness, mutation, repair, run loop, results, apply."""

    def _ga_fitness(self, chromosome, item_names, base_params, mc_days, mc_iters):
        """Monte Carlo mean revenue of one layout over ``mc_days``.

        The layout score goes through ``layout_objective.layout_drivers``
        -- the one score-to-driver transform, at the midpoint of each cited
        elasticity band (retail_literature) and anchored at
        ``base_params['score_anchor']`` -- and the drivers through
        ``mc_engine``. The Optimize pipeline and the What-If tab use the
        same transform at the same elasticities
        (``viz_optimize._opt_layout_drivers``), and
        ``experiments.closed_form`` gives this function's exact
        expectation."""
        from layout_objective import layout_drivers, layout_mc_kwargs

        score, breakdown = self._ga_compute_layout_score(chromosome, item_names, base_params)
        drivers = layout_drivers(score, breakdown, base_params)
        res = self._mc_engine(**layout_mc_kwargs(drivers, base_params,
                                                 n_days=mc_days,
                                                 n_iter=mc_iters))
        return res['mean']

    def _ga_score_anchor(self, item_names, base_params, source='floor'):
        """Anchor record of the layout on the floor now: the store the
        optimization or projection starts from, scored as it stands. Its
        score is what the elasticities are differenced against, so the
        layout the user built reproduces the calibrated inputs exactly."""
        from layout_objective import make_anchor
        chrom = self._ga_encode(item_names)
        score, breakdown = self._ga_compute_layout_score(chrom, item_names,
                                                         base_params)
        return make_anchor(score, breakdown, source)


    def _ga_crossover(self, p1, p2):
        mask = np.random.random(p1.shape[0]) < 0.5
        c1 = np.where(mask[:, None], p1, p2)
        c2 = np.where(mask[:, None], p2, p1)
        alpha = np.random.uniform(0.1, 0.5)
        blend = alpha * p1 + (1.0 - alpha) * p2
        return c1, c2, blend

    def _ga_get_section_bounds(self, item_name):
        """Return (sx, sy, sw, sh) for the item's zone on its own floor.

        The zone is resolved by ``item_zone_name``: the stamped zone while it
        still matches the item's category, else Section_<category>, else no
        bounds at all."""
        data = self._ga_get_item_data(item_name) if hasattr(self, '_ga_get_item_data') else self.items[item_name]
        fid = data.get('floor', getattr(self, 'current_floor', 1))
        walls = self.floors.get(fid, {}).get('walls', {})
        zone = item_zone_name(data, walls)
        sec_wall = walls.get(zone) if zone else None
        if sec_wall:
            return (*sec_wall['position'], *sec_wall['size'])
        return None

    def _ga_get_impulse_max_dist(self):
        """Radius around the checkout that impulse items are pulled into (25% of store diagonal)."""
        return math.sqrt(self.width ** 2 + self.height ** 2) * 0.25

    def _ga_project_impulse(self, x, y, w, h, lo_x, hi_x, lo_y, hi_y,
                            checkout_pos, impulse_max):
        """Pull an impulse item toward the checkout when its bounds allow it.

        The centre is moved to 0.95 of the radius along the line to the
        checkout and clipped back into [lo, hi]. The move is kept only if the
        clipped position really lies within the radius. When the section
        cannot reach the disc, the clip would send every input to the same
        corner and the item would stop being searched, so the incoming
        position is returned unchanged instead."""
        cx, cy = x + w / 2, y + h / 2
        dist = math.hypot(cx - checkout_pos[0], cy - checkout_pos[1])
        if dist <= impulse_max:
            return x, y
        scale = impulse_max / max(dist, 1e-6) * 0.95
        px = np.clip(checkout_pos[0] + (cx - checkout_pos[0]) * scale - w / 2, lo_x, hi_x)
        py = np.clip(checkout_pos[1] + (cy - checkout_pos[1]) * scale - h / 2, lo_y, hi_y)
        if math.hypot(px + w / 2 - checkout_pos[0], py + h / 2 - checkout_pos[1]) <= impulse_max:
            return px, py
        return x, y

    def _ga_aisle_plan(self, item_names):
        """The shared repair's aisle plan for these items
        (``experiments._feasibility.aisle_plan``: each zone's region, the
        zone pairs that keep an aisle, the door and the agent grid), or None
        where it does not apply.

        The plan is read from the ground floor, as every headless store is
        built, with the layout on the floor as the store as built (a
        headless shop records its builder's layout in ``as_built_layout``
        instead). It applies when every item is a ground-floor item under its
        own name; a multi-floor GUI store, whose upper-floor items carry
        ``F<n>:`` keys and zones on their own floor, keeps the zone-only
        repair (``_ga_zone_snap``)."""
        f1 = self.floors.get(1, {}).get('items', {})
        for n in item_names:
            if n not in f1 or self._ga_get_item_data(n).get('floor', 1) != 1:
                return None
        from experiments._feasibility import aisle_plan
        return aisle_plan(self, item_names)

    def _ga_plan_matches_zones(self, plan, item_names):
        """True when the plan puts every item in the zone ``_ga_repair``
        clips it to. They differ only on a GUI-edited store whose zone stamp
        no longer belongs to the item's category (``item_zone_name``
        ignores such a stamp, the plan reads it), and there the shared repair
        would drag the item back into its old department."""
        for i, n in enumerate(item_names):
            z = plan.zone[i]
            rect = plan.zone_rect[z] if z is not None else None
            b = self._ga_get_section_bounds(n)
            if (tuple(float(v) for v in b) if b else None) != rect:
                return False
        return True

    def _ga_shared_plan(self, item_names):
        """The aisle plan when the shared repair's rules can hold on this
        store, else None.

        They cannot hold when the store as built already breaks them: a
        fixture that cannot be shopped from the door as built (walled into
        a pocket, packed shut by its neighbours; ``_feasibility.
        as_built_unshoppable``) stays unshoppable in every candidate that
        leaves it where it is, so the shared repair's reachability
        fallbacks would end, for every candidate, at the whole as-built
        store -- the GA would score only its starting layout and report no
        lift. The floor-plan engine never builds such a store, so every
        headless run passes this check; a GUI floor edited into that state
        is repaired zone by zone instead (``_ga_zone_snap``), as it was
        before the shared repair, and the GA tab and the Optimize report
        say so (``_ga_repair_note``). Cached with the plan."""
        plan = self._ga_aisle_plan(item_names)
        if plan is None:
            return None
        from experiments._feasibility import as_built_unshoppable
        if as_built_unshoppable(plan):
            return None
        return plan

    def _ga_repair_note(self, item_names):
        """Why the GA's candidates were not held to the shared repair's
        aisle and reach rules on this store (a sentence for the GA tab and
        the Optimize report), or None when they were."""
        plan = self._ga_aisle_plan(item_names)
        if plan is None:
            return ("aisle and reach rules not applied: they need every "
                    "movable item on the ground floor; candidates were "
                    "repaired zone by zone (overlaps only)")
        from experiments._feasibility import as_built_unshoppable
        bad = as_built_unshoppable(plan)
        if bad:
            shown = ', '.join(bad[:3]) + (' ...' if len(bad) > 3 else '')
            return (f"aisle and reach rules not applied: {len(bad)} "
                    f"fixture(s) cannot be shopped from the door as the "
                    f"store stands ({shown}); candidates were repaired zone "
                    f"by zone (overlaps only)")
        if not self._ga_plan_matches_zones(plan, item_names):
            return ("aisle and reach rules not applied: an item's zone "
                    "stamp no longer matches its category; candidates were "
                    "repaired zone by zone (overlaps only)")
        return None

    def _ga_move_boxes(self, item_names):
        """``[(sx, sy, sw, sh) or None]`` per item: the box the GA's initial
        noise and mutation scale their steps to and clip into (inset by
        0.05 m): each item's zone (``_ga_get_section_bounds``) less the
        aisle rule of the shared repair (``experiments._feasibility.
        AislePlan.move_box``), so the GA's moves stay inside the space the
        repair allows -- the space random search draws from and the
        annealer moves in. Without it the GA would spend its moves in the
        aisles, only to have the repair clip them onto an aisle edge. The
        GUI and the headless experiments (``experiments._common.
        HeadlessShop``) run this same method. An item whose zone the plan
        does not resolve to the same rectangle keeps its zone, as does
        every item where the plan does not apply (``_ga_aisle_plan``) or
        cannot hold (``_ga_shared_plan``: a store with a fixture that
        cannot be shopped as built, which the repair then treats zone by
        zone). On a store whose zones exclude their aisles (the synthetic
        scenarios) every box is the zone itself."""
        boxes = [self._ga_get_section_bounds(n) for n in item_names]
        plan = self._ga_shared_plan(item_names)
        if plan is None:
            return boxes
        out = []
        for i, b in enumerate(boxes):
            z = plan.zone[i]
            if (b is None or z is None
                    or tuple(float(v) for v in b) != plan.zone_rect[z]):
                out.append(b)
            else:
                out.append(plan.move_box(z))
        return out

    def _ga_mutate(self, chrom, item_names, mut_rate):
        mutated = chrom.copy()
        checkout_pos, checkout_floor = self._ga_find_special_center('Checkout') if hasattr(self, '_ga_find_special_center') else (None, None)
        impulse_max = self._ga_get_impulse_max_dist()
        boxes = self._ga_move_boxes(item_names)

        for i, n in enumerate(item_names):
            if np.random.random() >= mut_rate:
                continue
            data = self._ga_get_item_data(n) if hasattr(self, '_ga_get_item_data') else self.items[n]
            w, h = data.get('size', (1.0, 1.0))
            fid = data.get('floor', getattr(self, 'current_floor', 1))
            bounds = boxes[i]

            if bounds:
                sx, sy, sw, sh = bounds
                pad = 0.05
                lo_x, hi_x = sx + pad, sx + sw - w - pad
                lo_y, hi_y = sy + pad, sy + sh - h - pad
                sigma_x = max(sw * 0.12, 0.2)
                sigma_y = max(sh * 0.12, 0.2)
            else:
                lo_x, lo_y = 0.1, 0.1
                hi_x, hi_y = self.width - w - 0.1, self.height - h - 0.1
                sigma_x = max(self.width, self.height) * 0.08
                sigma_y = sigma_x

            if hi_x < lo_x:
                hi_x = lo_x
            if hi_y < lo_y:
                hi_y = lo_y

            mutated[i, 0] = np.clip(mutated[i, 0] + np.random.normal(0, sigma_x), lo_x, hi_x)
            mutated[i, 1] = np.clip(mutated[i, 1] + np.random.normal(0, sigma_y), lo_y, hi_y)

            cat = str(data.get('category', '')).lower()
            if 'impulse' in cat and checkout_pos and fid == checkout_floor:
                mutated[i, 0], mutated[i, 1] = self._ga_project_impulse(
                    mutated[i, 0], mutated[i, 1], w, h,
                    lo_x, hi_x, lo_y, hi_y, checkout_pos, impulse_max)
        return self._ga_repair(mutated, item_names)

    def _ga_repair(self, chrom, item_names):
        checkout_pos, checkout_floor = self._ga_find_special_center('Checkout') if hasattr(self, '_ga_find_special_center') else (None, None)
        impulse_max = self._ga_get_impulse_max_dist()

        for i, n in enumerate(item_names):
            data = self._ga_get_item_data(n) if hasattr(self, '_ga_get_item_data') else self.items[n]
            w, h = data.get('size', (1.0, 1.0))
            fid = data.get('floor', getattr(self, 'current_floor', 1))
            bounds = self._ga_get_section_bounds(n)

            if bounds:
                sx, sy, sw, sh = bounds
                pad = 0.05
                lo_x, hi_x = sx + pad, sx + sw - w - pad
                lo_y, hi_y = sy + pad, sy + sh - h - pad
            else:
                lo_x, hi_x = 0.1, self.width - w - 0.1
                lo_y, hi_y = 0.1, self.height - h - 0.1
            if hi_x < lo_x:
                hi_x = lo_x
            if hi_y < lo_y:
                hi_y = lo_y
            chrom[i, 0] = np.clip(chrom[i, 0], lo_x, hi_x)
            chrom[i, 1] = np.clip(chrom[i, 1], lo_y, hi_y)

            cat = str(data.get('category', '')).lower()
            if 'impulse' in cat and checkout_pos and fid == checkout_floor:
                chrom[i, 0], chrom[i, 1] = self._ga_project_impulse(
                    chrom[i, 0], chrom[i, 1], w, h,
                    lo_x, hi_x, lo_y, hi_y, checkout_pos, impulse_max)
        return chrom

    def _ga_resolve_overlaps(self, chrom, item_names):
        """The second half of the GA's repair, after ``_ga_repair``'s zone
        clip: the repair every headless method shares
        (``experiments._feasibility.repair_positions``, which the headless
        GA reaches through ``experiments._common._repair_chrom_overlaps``).
        It keeps the floor-plan engine's invariants -- fixtures of two zones
        the store keeps an aisle apart stay ``MIN_AISLE`` apart, overlaps
        inside a zone are snapped apart inside the zone less its aisles, and
        every fixture stays shoppable from the door, with the single-file
        and as-built fallbacks -- so the GUI's GA tab and Optimize pipeline
        search the same feasible set as the headless experiments (review
        R06). Deterministic, no RNG draws; a feasible layout is a fixed
        point.

        Where the shared plan does not apply (a multi-floor store, or a zone
        stamp the GUI no longer honours; ``_ga_aisle_plan`` /
        ``_ga_plan_matches_zones``) or cannot hold (a fixture that cannot be
        shopped as the store stands, ``_ga_shared_plan``: every candidate
        would fall back to the as-built store) the zone-only snap
        (``_ga_zone_snap``) is used, as before; ``_ga_repair_note`` says
        which."""
        plan = self._ga_shared_plan(item_names)
        if plan is not None and self._ga_plan_matches_zones(plan, item_names):
            from experiments._feasibility import repair_positions
            return repair_positions(self, chrom, item_names)
        return self._ga_zone_snap(chrom, item_names)

    def _ga_zone_snap(self, chrom, item_names):
        """Resolve item-vs-item overlaps inside each zone of a chromosome.

        ``_ga_repair`` only clamps each item to its zone; it never separates
        two items that landed on top of each other. Fixtures are laid out
        0.15 m apart, so almost every perturbed or crossed-over child
        overlaps a neighbour, and one overlapping pair costs more score than
        the whole positive composite -- without this step the unperturbed
        starting layout is the only viable candidate and the GA cannot move
        anything.

        The repair is 2D and rank-preserving: the items of a zone are read
        row-major from their current positions (y picks the row, x the column
        within it) and snapped onto evenly spaced, non-overlapping slots, so
        the chromosome keeps its positional information as rank. Items are
        grouped by (floor, zone) -- an upper floor generated for the same
        shop type reuses section names, and its items must not be snapped
        into the ground floor's boxes. Deterministic: no RNG draws."""
        out = chrom.copy()

        by_zone = {}
        sizes = []
        for i, n in enumerate(item_names):
            d = (self._ga_get_item_data(n)
                 if hasattr(self, '_ga_get_item_data') else self.items[n])
            fid = d.get('floor', getattr(self, 'current_floor', 1))
            walls = self.floors.get(fid, {}).get('walls', {})
            sizes.append(tuple(d.get('size', (1.0, 1.0))))
            zone = item_zone_name(d, walls)
            if zone is None:
                continue
            by_zone.setdefault((fid, zone), []).append(i)

        for (fid, zone), idx_list in by_zone.items():
            if len(idx_list) < 2:
                continue
            sec_wall = self.floors.get(fid, {}).get('walls', {}).get(zone)
            if sec_wall is None:
                continue
            sx, sy = sec_wall['position']
            sw, sh = sec_wall['size']

            any_overlap = False
            for k in range(len(idx_list)):
                i = idx_list[k]
                xi, yi = out[i, 0], out[i, 1]
                wi, hi = sizes[i]
                for k2 in range(k + 1, len(idx_list)):
                    j = idx_list[k2]
                    xj, yj = out[j, 0], out[j, 1]
                    wj, hj = sizes[j]
                    if not (xi + wi <= xj or xi >= xj + wj
                            or yi + hi <= yj or yi >= yj + hj):
                        any_overlap = True
                        break
                if any_overlap:
                    break
            if not any_overlap:
                continue

            n_z = len(idx_list)
            max_w = max(sizes[i][0] for i in idx_list)
            max_h = max(sizes[i][1] for i in idx_list)
            gap = 0.15
            pad = 0.05
            avail_w = sw - 2 * pad
            avail_h = sh - 2 * pad
            cols = max(1, int(math.floor((avail_w + gap) / (max_w + gap))))
            rows_needed = int(math.ceil(n_z / cols))
            if rows_needed * (max_h + gap) - gap > avail_h:
                # Doesn't fit in this many columns; try the other orientation
                rows_needed = max(1, int(math.floor((avail_h + gap) / (max_h + gap))))
                cols = int(math.ceil(n_z / rows_needed))
            # Preserve the rank the GA chose on both axes: y decides the row,
            # x decides the column within that row. A plain (y, x) sort would
            # order columns by y as well, since real-valued y almost never ties.
            by_y = sorted(idx_list, key=lambda i: out[i, 1])
            rows = [sorted(by_y[r * cols:(r + 1) * cols], key=lambda i: out[i, 0])
                    for r in range(rows_needed)]
            step_x = (avail_w - max_w) / max(cols - 1, 1) if cols > 1 else 0.0
            step_y = (avail_h - max_h) / max(rows_needed - 1, 1) if rows_needed > 1 else 0.0
            # Steps below item dimension + gap would re-create the overlap.
            step_x = max(step_x, max_w + gap)
            step_y = max(step_y, max_h + gap)
            for r, row in enumerate(rows):
                for c, i in enumerate(row):
                    wi, hi = sizes[i]
                    x = sx + pad + c * step_x
                    y = sy + pad + r * step_y
                    # Clip back into the zone in case a step pushed past it
                    x = max(sx + pad, min(x, sx + sw - wi - pad))
                    y = max(sy + pad, min(y, sy + sh - hi - pad))
                    out[i, 0] = x
                    out[i, 1] = y
        return out

    def _run_ga_optimization(self):
        # The Optimize pipeline leaves the GUI interactive while it measures
        # its PRE/POST windows, so this button can be clicked mid-window.
        # Stopping the worker below would freeze that measurement halfway and
        # the pipeline would go on to compare a partial window with a full one.
        if getattr(self, '_opt_running', False):
            messagebox.showinfo("Optimization running",
                                "Wait for the Optimize measurement to finish.",
                                parent=self.tk_root)
            return

        A = self.customer_simulation.analytics
        # A trajectory-only dataset fills the calibration block with spatial
        # keys but no arrival or conversion rate, so it cannot stand in for
        # observed customers.
        cal = _calib(A)
        has_tx = bool(cal.get('arrivals_per_hour')) and 'conversion_rate' in cal
        if not has_tx and A.get('total_customers', 0) < 5:
            messagebox.showwarning("Insufficient Data",
                                   "Run the simulation first (>= 5 customers).",
                                   parent=self.tk_root)
            return

        item_names = self._ga_get_movable_items()
        if len(item_names) < 2:
            messagebox.showwarning("Too Few Items",
                                   "Need at least 2 movable items for GA.",
                                   parent=self.tk_root)
            return

        # The live simulation worker draws from the same global numpy RNG as
        # the Monte Carlo fitness, and inserts keys into the analytics dicts
        # the parameter extraction reads; stop it before either. Both gates
        # above have passed at this point, so a refused click never wipes a
        # running simulation.
        sim = self.customer_simulation
        if sim.running or sim._running:
            self._stop_simulation()
            # Settle the agents still in the store BEFORE hard_stop drops
            # them: _clear_simulation records them as cleared and retires
            # their state history, hard_stop empties the store with no
            # bookkeeping at all.
            self._clear_simulation()
            sim.hard_stop()

        base_params = self._extract_simulation_parameters()
        # Anchor the elasticities at the layout on the floor now, so the
        # store as built reproduces the calibrated inputs and the GA's
        # fitness moves them only by a candidate's score difference from it.
        from layout_objective import SCORE_ANCHOR_KEY
        base_params[SCORE_ANCHOR_KEY] = self._ga_score_anchor(
            item_names, base_params, source='floor_at_ga_start')
        pop_size = max(10, self._ga_pop.get())
        n_gens = max(5, self._ga_gens.get())
        mut_rate = max(0.01, self._ga_mut.get() / 100.0)
        elite_frac = max(0.05, self._ga_elite.get() / 100.0)
        mc_iters = max(100, self._ga_mc_iters.get())
        mc_days = max(1, self._ga_mc_days.get())

        n_elite = max(1, int(pop_size * elite_frac))

        self._ga_progress.set("Initializing population...")
        self.tk_root.update_idletasks()

        current = self._ga_encode(item_names)
        # The starting layout enters through the same repair chain as every
        # other candidate, so the population is scored on one feasible set:
        # an as-built fixture can sit flush on its zone edge, inside the
        # clearance the repair enforces.
        population = [self._ga_resolve_overlaps(
            self._ga_repair(current.copy(), item_names), item_names)]
        # The initial noise is scaled to, and clipped into, the box the
        # mutation uses: the zone less the aisle rule (``_ga_move_boxes``),
        # as in the headless GA.
        boxes = self._ga_move_boxes(item_names)
        for _ in range(pop_size - 1):
            noisy = current.copy()
            for i, n in enumerate(item_names):
                data = self._ga_get_item_data(n) if hasattr(self, '_ga_get_item_data') else self.items[n]
                w, h = data.get('size', (1.0, 1.0))
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
                    current[i, 0] + np.random.normal(0, sigma_x), lo_x, hi_x)
                noisy[i, 1] = np.clip(
                    current[i, 1] + np.random.normal(0, sigma_y), lo_y, hi_y)
            noisy = self._ga_repair(noisy, item_names)
            noisy = self._ga_resolve_overlaps(noisy, item_names)
            population.append(noisy)

        history_best = []
        history_avg = []
        history_worst = []
        gen_scores_all = []

        # Paired Monte Carlo: every candidate in a round is scored on the
        # same seed, so the ranking reflects layout differences rather than
        # independent MC noise. The global RNG state is restored after each
        # round so selection and mutation keep drawing from the unseeded
        # stream.
        mc_base_seed = int(np.random.randint(0, 2 ** 20))

        def _paired_fitness(pop, mc_seed):
            saved_state = np.random.get_state()
            out = np.empty(len(pop), dtype=np.float64)
            for k, p in enumerate(pop):
                np.random.seed(mc_seed)
                out[k] = self._ga_fitness(p, item_names, base_params, mc_days, mc_iters)
            np.random.set_state(saved_state)
            return out

        for gen in range(n_gens):
            self._ga_progress.set(f"Gen {gen + 1}/{n_gens} — evaluating...")
            self.tk_root.update_idletasks()

            fitness = _paired_fitness(population, mc_base_seed * 1000 + gen)

            rank = np.argsort(fitness)[::-1]
            population = [population[r] for r in rank]
            fitness = fitness[rank]

            history_best.append(fitness[0])
            history_avg.append(fitness.mean())
            history_worst.append(fitness[-1])
            gen_scores_all.append(fitness.copy())

            self._ga_progress.set(
                f"Gen {gen + 1}/{n_gens} — best=${fitness[0]:,.0f}  avg=${fitness.mean():,.0f}"
            )
            self.tk_root.update_idletasks()

            new_pop = [p.copy() for p in population[:n_elite]]

            while len(new_pop) < pop_size:
                t_size = min(5, pop_size)
                t_idx = np.random.choice(pop_size, t_size, replace=False)
                t_fit = fitness[t_idx]
                p1 = population[t_idx[np.argmax(t_fit)]]

                t_idx2 = np.random.choice(pop_size, t_size, replace=False)
                t_fit2 = fitness[t_idx2]
                p2 = population[t_idx2[np.argmax(t_fit2)]]

                c1, c2, blend = self._ga_crossover(p1, p2)
                for child in [c1, c2, blend]:
                    child = self._ga_mutate(child, item_names, mut_rate)
                    child = self._ga_repair(child, item_names)
                    # Separate items that landed on top of each other, so the
                    # child can be judged on its placement instead of
                    # collapsing to the overlap-penalty floor.
                    child = self._ga_resolve_overlaps(child, item_names)
                    new_pop.append(child)
                    if len(new_pop) >= pop_size:
                        break

            population = new_pop[:pop_size]

        self._ga_progress.set("Final evaluation...")
        self.tk_root.update_idletasks()
        # Average the survivors over several shared seeds so a single
        # lucky draw cannot decide the winner.
        n_final_seeds = 5
        final_fitness = np.mean([
            _paired_fitness(population, mc_base_seed * 1000 + n_gens + 1 + s)
            for s in range(n_final_seeds)
        ], axis=0)
        best_idx = int(np.argmax(final_fitness))
        best_chrom = population[best_idx]

        self._ga_best_chromosome = best_chrom
        self._ga_best_layout = self._ga_decode(best_chrom, item_names)
        self._ga_item_names = item_names

        best_score, best_breakdown = self._ga_compute_layout_score(
            best_chrom, item_names, base_params
        )
        # The argmax over noisy means is biased upward, so the reported
        # comparison re-scores the chosen layout and the current one
        # together on fresh shared seeds.
        current_chrom = self._ga_encode(item_names)
        best_fit, current_fit = np.mean([
            _paired_fitness([best_chrom, current_chrom],
                            mc_base_seed * 1000 + n_gens + 1 + n_final_seeds + s)
            for s in range(n_final_seeds)
        ], axis=0)

        # Whether the candidates were held to the shared repair's aisle and
        # reach rules; the results say so when they were not.
        self._ga_repair_notice = self._ga_repair_note(item_names)
        self._display_ga_results(
            item_names, base_params, best_chrom, best_breakdown,
            best_fit, current_fit,
            history_best, history_avg, history_worst,
            gen_scores_all, n_gens, pop_size, mc_days, mc_iters
        )
        self._ga_progress.set(
            "Done. Click 'Apply Best Layout' to use it."
            + (" (Aisle and reach rules not applied: see the results.)"
               if self._ga_repair_notice else ""))

    def _display_ga_results(self, item_names, base_params, best_chrom, breakdown,
                            best_rev, current_rev, h_best, h_avg, h_worst,
                            gen_scores, n_gens, pop_size, mc_days, mc_iters):
        improvement = (best_rev - current_rev) / max(current_rev, 1e-6) * 100

        lines = [
            "=" * 50,
            "   GENETIC ALGORITHM OPTIMIZATION RESULTS",
            "=" * 50,
            "",
            f"  Population:        {pop_size}",
            f"  Generations:       {n_gens}",
            f"  MC iters/eval:     {mc_iters}",
            f"  MC projection:     {mc_days} days",
            f"  Movable items:     {len(item_names)}",
        ]
        note = getattr(self, '_ga_repair_notice', None)
        if note:
            lines += ["", "  NOTE: " + note]
        lines += [
            "",
            "REVENUE COMPARISON",
            "-" * 50,
            f"  Current layout:    ${current_rev:>12,.2f}",
            f"  GA-optimized:      ${best_rev:>12,.2f}",
            f"  Improvement:       {improvement:>+11.1f}%",
            f"  Daily delta:       ${(best_rev - current_rev) / mc_days:>+12,.2f}",
            f"  Annual projection: ${(best_rev - current_rev) / mc_days * 365:>+12,.0f}",
            "",
            "LAYOUT QUALITY BREAKDOWN",
            "-" * 50,
            f"  Traffic alignment:   {breakdown['traffic']:.4f}",
            f"  Cross-merch:         {breakdown['cross_merch']:.4f}",
            f"  Impulse placement:   {breakdown['impulse']:.4f}",
            f"  Flow efficiency:     {breakdown['flow']:.4f}",
            f"  Revenue placement:   {breakdown['revenue_placement']:.4f}",
            f"  Accessibility:       {breakdown['accessibility']:.4f}",
            f"  Section compliance:  {breakdown['section_compliance']:.4f}",
            f"  Bottleneck penalty:  {breakdown['bottleneck_penalty']:.4f}",
            f"  Overlap penalty:     {breakdown['overlap_penalty']:.4f}",
            f"  Composite score:     {breakdown['composite']:.4f}",
            "",
            "ITEM MOVEMENTS (current -> optimized)",
            "-" * 50,
        ]

        moved = 0
        item_meta = {name: (self._ga_get_item_data(name) if hasattr(self, '_ga_get_item_data') else self.items[name])
                     for name in item_names}
        for i, name in enumerate(item_names):
            old = item_meta[name].get('position', (0.0, 0.0))
            new = (best_chrom[i, 0], best_chrom[i, 1])
            dist = np.sqrt((new[0] - old[0]) ** 2 + (new[1] - old[1]) ** 2)
            if dist > 0.3:
                moved += 1
                fid = item_meta[name].get('floor', self.current_floor)
                label = f"F{fid}:{item_meta[name].get('source_name', name)}"
                lines.append(
                    f"  {label[:24]:<24s}  ({old[0]:.1f},{old[1]:.1f})"
                    f" -> ({new[0]:.1f},{new[1]:.1f})  d={dist:.1f}m"
                )

        lines.append(f"\n  Items moved: {moved}/{len(item_names)}")

        self._ga_text.config(state=tk.NORMAL)
        self._ga_text.delete('1.0', tk.END)
        self._ga_text.insert(tk.END, "\n".join(lines))
        self._ga_text.config(state=tk.DISABLED)

        self._ga_fig.clear()
        wc = 'white'

        ax1 = self._ga_fig.add_subplot(2, 2, 1)
        gens = range(1, len(h_best) + 1)
        ax1.plot(gens, h_best, color='#2ecc71', linewidth=2, label='Best')
        ax1.plot(gens, h_avg, color='#3498db', linewidth=1.5, label='Average')
        ax1.plot(gens, h_worst, color='#e74c3c', linewidth=1, alpha=0.6, label='Worst')
        ax1.fill_between(gens, h_worst, h_best, alpha=0.1, color='#3498db')
        ax1.set_xlabel("Generation", color=wc, fontsize=9)
        ax1.set_ylabel("MC Revenue ($)", color=wc, fontsize=9)
        ax1.set_title("Fitness Convergence", color=wc, fontsize=10)
        ax1.legend(fontsize=7)
        ax1.set_facecolor('#001a33')
        ax1.tick_params(colors=wc, labelsize=7)
        ax1.grid(True, alpha=0.15, color='white')

        ax2 = self._ga_fig.add_subplot(2, 2, 2)
        categories = [k for k in breakdown.keys() if k != 'composite']
        vals = [breakdown[c] for c in categories]
        n_bars = len(categories)
        colors_bar = ['#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#e74c3c',
                       '#1abc9c', '#e67e22', '#2980b9', '#c0392b', '#8e44ad'][:n_bars]
        short_cats = [c.replace('_', '\n') for c in categories]
        bars = ax2.bar(short_cats, vals, color=colors_bar[:len(categories)])
        ax2.axhline(0, color='white', linewidth=0.5)
        ax2.set_title("Layout Quality Breakdown", color=wc, fontsize=10)
        ax2.set_facecolor('#001a33')
        ax2.tick_params(colors=wc, labelsize=7)
        for bar, val in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f'{val:.3f}', ha='center', va='bottom', fontsize=7, color=wc)

        ax3 = self._ga_fig.add_subplot(2, 2, 3)
        display_floor = getattr(self, 'current_floor', 1)
        for i, name in enumerate(item_names):
            meta = item_meta.get(name) or (self._ga_get_item_data(name) if hasattr(self, '_ga_get_item_data') else self.items[name])
            if meta.get('floor', display_floor) != display_floor:
                continue
            old = meta.get('position', (0.0, 0.0))
            s = meta.get('size', (1.0, 1.0))
            new = (best_chrom[i, 0], best_chrom[i, 1])
            dist = np.sqrt((new[0] - old[0]) ** 2 + (new[1] - old[1]) ** 2)
            ax3.add_patch(plt.Rectangle(old, s[0], s[1],
                                         fill=False, edgecolor='#e74c3c',
                                         linewidth=0.8, linestyle='--', alpha=0.5))
            ax3.add_patch(plt.Rectangle(new, s[0], s[1],
                                         fill=False, edgecolor='#2ecc71',
                                         linewidth=1.2))
            if dist > 0.3:
                ax3.annotate('', xy=(new[0] + s[0] / 2, new[1] + s[1] / 2),
                             xytext=(old[0] + s[0] / 2, old[1] + s[1] / 2),
                             arrowprops=dict(arrowstyle='->', color='#f39c12',
                                             lw=0.8, alpha=0.7))
        for wname, wd in self.floors.get(display_floor, {}).get('walls', self.walls).items():
            if not wname.startswith('Section_'):
                wp = wd['position']
                ws = wd['size']
                ax3.add_patch(plt.Rectangle(wp, ws[0], ws[1],
                                             fill=True, facecolor='#555555',
                                             edgecolor='white', linewidth=0.5, alpha=0.6))
        ax3.set_xlim(-0.5, self.width + 0.5)
        ax3.set_ylim(-0.5, self.height + 0.5)
        ax3.set_aspect('equal')
        ax3.set_title("Movement Map (red=old, green=new)", color=wc, fontsize=10)
        ax3.set_facecolor('#001a33')
        ax3.tick_params(colors=wc, labelsize=7)

        ax4 = self._ga_fig.add_subplot(2, 2, 4)
        if gen_scores:
            last_gen = gen_scores[-1]
            ax4.hist(last_gen, bins=min(20, pop_size // 2), color='#3498db',
                     alpha=0.7, edgecolor='white', linewidth=0.5)
            ax4.axvline(best_rev, color='#2ecc71', linewidth=2,
                        linestyle='--', label=f'Best ${best_rev:,.0f}')
            ax4.axvline(current_rev, color='#e74c3c', linewidth=2,
                        linestyle='--', label=f'Current ${current_rev:,.0f}')
        ax4.set_xlabel("MC Revenue ($)", color=wc, fontsize=9)
        ax4.set_ylabel("Count", color=wc, fontsize=9)
        ax4.set_title("Final Generation Distribution", color=wc, fontsize=10)
        ax4.legend(fontsize=7)
        ax4.set_facecolor('#001a33')
        ax4.tick_params(colors=wc, labelsize=7)
        ax4.grid(True, alpha=0.15, color='white')

        self._ga_fig.tight_layout(pad=2.0)
        self._ga_canvas_fig.draw_idle()

    def _apply_ga_best(self):
        # Moving fixtures mid-window would change the store the Optimize
        # pipeline is measuring, so its PRE baseline and POST result would no
        # longer describe the same two layouts.
        if getattr(self, '_opt_running', False):
            messagebox.showinfo("Optimization running",
                                "Wait for the Optimize measurement to finish.",
                                parent=self.tk_root)
            return

        if self._ga_best_layout is None:
            messagebox.showinfo("No Result", "Run the GA optimizer first.",
                                parent=self.tk_root)
            return

        old_positions = {}
        touched_floors = set()
        for key, (x, y) in self._ga_best_layout.items():
            fid, source = self._ga_resolve_item_key(key) if hasattr(self, '_ga_resolve_item_key') else (self.current_floor, key)
            floor_items = self.floors.get(fid, {}).get('items', {})
            if source not in floor_items:
                continue
            old_positions[(fid, source)] = tuple(floor_items[source]['position'])
            w, h = floor_items[source]['size']
            safe_x = max(0.1, min(x, self.width - w - 0.1))
            safe_y = max(0.1, min(y, self.height - h - 0.1))
            floor_items[source]['position'] = (safe_x, safe_y)
            touched_floors.add(fid)

        prev_floor = self.current_floor
        try:
            for fid in touched_floors or {self.current_floor}:
                self.current_floor = fid
                self._repair_item_overlaps()
        finally:
            self.current_floor = prev_floor

        # Items are obstacles for the live agents and carry the zone
        # attribution, so both the path grid and the zone table have to be
        # rebuilt from the new positions.
        sim = getattr(self, 'customer_simulation', None)
        if sim is not None:
            sim.geometry_dirty = True
            sim.invalidate_zones_cache()

        self.redraw()

        moved = 0
        for (fid, source), old in old_positions.items():
            new_pos = self.floors.get(fid, {}).get('items', {}).get(source, {}).get('position', old)
            if np.sqrt((new_pos[0] - old[0]) ** 2 + (new_pos[1] - old[1]) ** 2) > 0.3:
                moved += 1
        messagebox.showinfo(
            "Layout Applied",
            f"GA-optimized layout applied.\n{moved} items repositioned.",
            parent=self.tk_root
        )

    # -------------------------------------------------------------
    #  WHAT-IF SCENARIO COMPARISON (A/B Testing) TAB
    # -------------------------------------------------------------

