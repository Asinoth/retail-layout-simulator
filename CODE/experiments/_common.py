"""Shared infrastructure for the TOMACS Tier-1 experiments.

Provides:
  * ``build_headless_shop(synthetic_shop)`` -- materialize a SyntheticShop
    into a Tk-free shop instance that the existing GA / fitness / MC
    code can run against without ever instantiating a Tk root.
  * ``base_params_for(synthetic_shop)`` -- construct the ``base_params``
    dict that ``_ga_fitness`` and ``_mc_engine`` consume, derived from
    the synthetic shop's parameters.
  * ``anchor_base_params(shop, item_names, base_params, as_built)`` --
    anchor those parameters at the store's repaired as-built layout, so
    the elasticities act on score differences from it
    (``layout_objective``); every runner does this before scoring.
    ``base_params_record`` is the compact form the sidecars record.
  * ``criterion_spread(shop, item_names)`` -- each score criterion's SD
    over repaired random layouts and its effective weight (weight x SD).
  * ``apply_layout_to_shop(shop, layout, item_names)`` -- write a layout
    dict back into the headless shop's floor-1 items.
  * ``paired_mc_revenue(shop, item_names, layout, base_params, seed,
    mc_iters, mc_days)`` -- evaluate one layout's revenue under the
    simulator's fitness function with a FIXED RNG seed; paired across
    layouts to remove MC noise from method comparisons. Leaves the
    shop's item positions as it found them.
  * ``feasible_layout(shop, item_names, layout)`` -- the GA's repair
    chain as a layout -> layout map; every method's layouts go through
    it before MC scoring. It keeps the floor-plan engine's invariants --
    the aisles between zones the as-built store keeps apart, and every
    fixture shoppable (``experiments._feasibility``) -- and
    ``checked_layout`` checks them on every layout a runner reports.
  * ``zone_sampler(shop, item_names)`` / ``zone_neighbor(shop,
    item_names, n_move=1)`` -- random layouts and moves of one or
    ``n_move`` items inside the zones the GA constrains each item to (less
    the aisles), for the ``sampler`` / ``neighbor`` hooks of
    ``experiments.metaheuristics`` on a shop that has no SyntheticShop
    behind it (the calibrated store); ``k_move_items`` scales the move
    with the store.
  * ``closed_form_revenue`` -- a layout's exact expected revenue
    (``experiments.closed_form``), recorded beside its Monte Carlo value.
  * ``bootstrap_ci(samples, alpha, n_boot)`` -- percentile bootstrap CI.
  * ``write_sidecar(out_dir, payload)`` -- JSON sidecar with seed +
    git SHA + elasticity snapshot + the resolved package closure and the
    lock file's hash (``environment_lock``) + the machine, for
    reproducibility. The git state comes from a ``provenance_snapshot()``
    taken when the run started, not from the tree as it stands hours later
    when the file is written. Paths in the payload are written relative to
    the repository root (``portable_paths``).

Anti-Tk guarantee: ``build_headless_shop`` constructs an instance of
``HeadlessShop`` (defined here) which inherits the GA / projection /
heatmap mixins but skips every Tk-touching one. An assertion at the
end of the constructor verifies that ``getattr(shop, 'tk_root', None)
is None``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Project imports (mixins). The experiments harness inherits only the
# pure-Python mixins; the Tk-bound ones are excluded so no root is
# ever constructed.
from viz_ga import GAMixin
from viz_ga_run import GARunMixin
from viz_projections import ProjectionsMixin
from simulation import CustomerFlowSimulation

from synthetic_shops import SyntheticShop, SyntheticItem
from dataset_calibration import CalibratedParams
from dataset_paths import repo_relative
from dataset_layout import build_layout_from_calibration
import retail_literature as RL
from retail_literature import (
    GaWeights,
    ELASTICITY_CONV_BASE, ELASTICITY_CONV_GAIN_MAX,
    ELASTICITY_IMP_BASE,  ELASTICITY_IMP_GAIN_MAX,
    ELASTICITY_BSK_BASE,  ELASTICITY_BSK_GAIN_MAX,
    ABANDON_FRAC_OF_NONCONVERTERS, ABANDON_RATE_SYNTHETIC,
    ASSUMED_CONVERSION_RATE, DEFAULT_OP_HOURS_PER_DAY, DEFAULT_QUEUE_TIME_S,
    HEAT_PRIOR_CAL_CHECKOUT, HEAT_PRIOR_CAL_ENTRANCE, HEAT_PRIOR_CAL_WALL,
    HEAT_PRIOR_SYNTH_CHECKOUT, HEAT_PRIOR_SYNTH_ENTRANCE,
    HEAT_PRIOR_WALL_MIN_D, IMPULSE_RATE_STANDIN, IMPULSE_VALUE_FRAC_OF_GROSS,
    REV_STD_FRAC_OF_MEAN, STANDIN_AVG_BASKET, STANDIN_STD_BASKET,
)
from layout_objective import CRITERION_WEIGHTS, require_anchor, with_anchor
from experiments._feasibility import (                       # noqa: F401
    LayoutInvariantError, aisle_plan, aisle_rule_summary,
    check_layout_invariants, combine_repair_stats, invariant_summary,
    layout_violations, repair_positions, repair_stats, reset_repair_stats,
)


#: Tags mixed into the GA's array seeds. An array seed goes through
#: MT19937's init_by_array, a different initialisation from the plain
#: integer seeding used for the Monte Carlo evaluation seeds
#: (``rng_seed*1000 + gen``), so the population and operator streams of one
#: replication cannot silently repeat another replication's evaluation
#: stream.
_INIT_POP_STREAM_TAG = 0x1A17
_OPERATOR_STREAM_TAG = 0x06A0

#: Seeds the GA's surviving population is re-evaluated under before the
#: winner is picked (``run_ga_headless``). The equal-budget comparators are
#: given the same number so their final selection matches the GA's.
GA_N_FINAL_SEEDS = 5

#: L-BFGS-B restarts of the analytical reference (``oracle.solve_oracle``) in
#: every runner that uses it -- Figures A and B, the elasticity LHS and the
#: objective alignment -- so no comparison is made against a weaker
#: reference than another. A protocol setting, not a model coefficient: at
#: 24 restarts the best value still rose with more in 14 of 15 scenarios
#: (review finding R36); ``run_synthetic_gt`` records the best-value curve
#: that shows what the restarts past each checkpoint add.
ORACLE_RESTARTS = 96


def oracle_record(result: Any) -> Dict[str, Any]:
    """What a runner records about one analytical-reference solve: its
    restarts, the ones that failed (with their reasons), whether it
    converged and its note."""
    return {'n_restarts': int(result.n_restarts),
            'n_failed_restarts': int(result.n_failed),
            'failures': [list(f) for f in result.failures],
            'converged': bool(result.converged),
            'note': result.note}


def oracle_summary(records: List[Dict[str, Any]],
                   n_restarts: int) -> Dict[str, Any]:
    """The analytical references of a run, summed over its scenarios: the
    restarts each was given, how many failed in all, and how many
    references did not converge. A validator checks ``n_restarts``."""
    return {'n_restarts': int(n_restarts),
            'n_failed_restarts': int(sum(r['n_failed_restarts']
                                         for r in records)),
            'n_not_converged': int(sum(not r['converged'] for r in records)),
            'n_scenarios': len(records)}


# --- Headless shop class -------------------------------------------------

class HeadlessShop(GAMixin, GARunMixin, ProjectionsMixin):
    """Tk-free shop with just the attributes the GA, fitness, and MC
    engine actually read. The Tk-bound mixins (SimTabMixin, LayoutMixin,
    etc.) are deliberately NOT inherited, so no ``tk.Tk()`` ever runs.

    Required attributes the inherited methods access:
      width, height, floors, current_floor, num_floors, connectors,
      door_position, door_side, customer_simulation, items (proxy),
      walls (proxy), prices (proxy).
    """

    def __init__(self, width: float, height: float):
        self.width = float(width)
        self.height = float(height)
        self.current_floor = 1
        self.num_floors = 1
        self.connectors: Dict[int, Dict[int, str]] = {}
        self.floors: Dict[int, Dict[str, Any]] = {
            1: {
                'items':  {},
                'walls':  {},
                'prices': {},
                'is_main_floor': True,
                'door_position': None,
                'door_side': None,
            }
        }
        self.door_position: Optional[Tuple[float, float]] = None
        self.door_side: Optional[str] = None
        self.canvas = None          # any code that checks for canvas finds None
        self.tk_root = None         # explicit: no Tk anywhere
        self.current_tab = None
        # The store as its builder laid it out, {name: (x, y)}: what the
        # repair's aisle plan is read from (``_feasibility.aisle_plan``),
        # whatever layout a later call leaves on the floor.
        self.as_built_layout: Optional[Dict[str, Tuple[float, float]]] = None
        # Now construct the simulation; it stores self as ``self.shop``.
        self.customer_simulation = CustomerFlowSimulation(self)
        # The sim creates a _gui_queue and heat buffers but never starts
        # the worker thread (start_simulation is not called).

        # Anti-Tk guarantee
        assert self.tk_root is None, "HeadlessShop must not own a Tk root"

    # GAMixin's ``items``/``walls``/``prices`` access pattern relies on
    # the ShopVisualizer property proxies; replicate them here.
    @property
    def items(self):
        return self.floors[self.current_floor]['items']

    @items.setter
    def items(self, value):
        self.floors[self.current_floor]['items'] = value

    @property
    def walls(self):
        return self.floors[self.current_floor]['walls']

    @walls.setter
    def walls(self, value):
        self.floors[self.current_floor]['walls'] = value

    @property
    def prices(self):
        return self.floors[self.current_floor]['prices']

    @prices.setter
    def prices(self, value):
        self.floors[self.current_floor]['prices'] = value

    def all_items_across_floors(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for fid, fdata in sorted(self.floors.items()):
            for name, data in fdata.get('items', {}).items():
                d = dict(data)
                d['floor'] = fid
                d.setdefault('source_name', name)
                out[name] = d
        return out

    def redraw(self): pass   # no-op; no canvas

    # ``_ga_move_boxes`` -- the zone less the aisle rule of the shared
    # repair, the box the GA's initial noise and mutation are scaled to
    # and clipped into -- is ``GARunMixin._ga_move_boxes``, the method the
    # GUI runs too, so the two cannot drift apart.


# --- Build a headless shop from a SyntheticShop --------------------------

def build_headless_shop(synthetic: SyntheticShop) -> HeadlessShop:
    """Materialize a SyntheticShop's items, sections, entrance, and
    checkout into a HeadlessShop. Populates the simulator's analytics
    with synthetic counts so the GA's scoring function has data to read."""
    shop = HeadlessShop(synthetic.width, synthetic.height)
    f1 = shop.floors[1]

    # Boundary walls
    t = 0.2
    f1['walls']['Wall_Bot']   = {'position': [0.0, 0.0],                    'size': [synthetic.width, t]}
    f1['walls']['Wall_Top']   = {'position': [0.0, synthetic.height - t],   'size': [synthetic.width, t]}
    f1['walls']['Wall_Left']  = {'position': [0.0, 0.0],                    'size': [t, synthetic.height]}
    f1['walls']['Wall_Right'] = {'position': [synthetic.width - t, 0.0],    'size': [t, synthetic.height]}

    # Sections
    for cat, (sx, sy, sw, sh) in synthetic.sections.items():
        f1['walls'][f'Section_{cat}'] = {
            'position': [sx, sy],
            'size':     [sw, sh],
            'category': cat,
        }

    # Entrance + checkout (as wall fixtures, matching the existing layout)
    ent_x = synthetic.entrance[0] - 1.0
    f1['walls']['Entrance'] = {
        'position': [ent_x, 0.0],
        'size':     [2.0, 0.4],
        'category': 'Entrance',
    }
    chk_x = synthetic.checkout[0] - 1.5
    f1['walls']['Checkout'] = {
        'position': [chk_x, 0.4],
        'size':     [3.0, 1.5],
        'category': 'Checkout',
    }
    shop.door_position = synthetic.entrance
    shop.door_side = 'bottom'
    f1['door_position'] = shop.door_position
    f1['door_side'] = 'bottom'

    # Items at their initial (grid) positions; the experiment runner
    # will overwrite positions before each fitness evaluation.
    from synthetic_shops import grid_layout_within_sections
    init_layout = grid_layout_within_sections(synthetic)
    for it in synthetic.items:
        pos = init_layout.get(it.name, (0.5, 0.5))
        f1['items'][it.name] = {
            'position': [float(pos[0]), float(pos[1])],
            'size':     list(it.size),
            'category': it.category,
            'source_name': it.name,
        }
        f1['prices'][it.name] = float(it.base_revenue)

    # Populate analytics with synthetic counts so _ga_compute_layout_score
    # has data to score against (otherwise traffic_score, cross_score,
    # impulse_score etc. all collapse to zero).
    A = shop.customer_simulation.analytics
    n_visitors = max(50, int(synthetic.daily_customers))
    A['total_customers'] = n_visitors
    A['completed_purchases'] = int(n_visitors * 0.30)
    A['popular_items'] = {
        it.name: max(1, int(it.base_revenue / 5)) for it in synthetic.items
    }
    # Per-item conversion rates vary -- impulse items see many visits
    # but lower per-visit conversion; high-base-revenue items see fewer
    # visits but higher conversion. This gives the GA scoring a
    # non-uniform signal across items (otherwise revenue_placement
    # collapses to a constant).
    rng_anal = np.random.default_rng(synthetic.rng_seed + 99)
    A['item_conversion_rates'] = {}
    for it in synthetic.items:
        if it.is_impulse:
            visits = max(2, int(it.base_revenue / 1.5))
            conv = float(rng_anal.uniform(0.10, 0.20))
        else:
            visits = max(2, int(it.base_revenue / 5))
            conv = float(rng_anal.uniform(0.20, 0.50))
        A['item_conversion_rates'][it.name] = {
            'visits':    visits,
            'purchases': max(1, int(visits * conv)),
        }
    A['revenue_by_area'] = {}
    for it in synthetic.items:
        A['revenue_by_area'][it.category] = (
            A['revenue_by_area'].get(it.category, 0.0) + it.base_revenue
        )
    A['cross_merchandising'] = {
        f"{a}|{b}": max(1, int(round(aff * n_visitors)))
        for (a, b), aff in synthetic.affinity.items()
    }
    # Per-category dwell time varies. Use literature-grounded ranges:
    # food categories ~15-25s, electronics ~60-120s (browse-heavy),
    # impulse ~5-10s. The GA composite does not score dwell; these
    # samples only keep the analytics block complete for parameter
    # extraction, which reads dwell_times_by_zone.
    dwell_ranges = {
        'Beverages':   (15.0, 25.0),
        'Snacks':      (15.0, 25.0),
        'Produce':     (20.0, 35.0),
        'Electronics': (60.0, 120.0),
        'Stationery':  (25.0, 50.0),
        'Impulse':     (5.0, 10.0),
    }
    A['dwell_times_by_zone'] = {}
    for cat in synthetic.sections.keys():
        lo, hi = dwell_ranges.get(cat, (20.0, 40.0))
        A['dwell_times_by_zone'][cat] = [
            float(rng_anal.uniform(lo, hi)) for _ in range(20)
        ]
    A['bottlenecks'] = {}
    A['basket_sizes'] = [3] * n_visitors

    # Heat map: bias toward entrance + checkout to give traffic_score
    # a non-trivial gradient.
    paint_synthetic_heatmap(shop, synthetic)
    shop.customer_simulation.geometry_dirty = True
    shop.as_built_layout = {n: (float(d['position'][0]), float(d['position'][1]))
                            for n, d in f1['items'].items()}
    return shop


def paint_synthetic_heatmap(shop: HeadlessShop, synthetic: SyntheticShop,
                            wall_coef: float = 0.0) -> None:
    """Paint the synthetic scenarios' cold-start traffic prior:

        heat = HEAT_PRIOR_SYNTH_ENTRANCE / (1 + d_entrance)
             + HEAT_PRIOR_SYNTH_CHECKOUT / (1 + d_checkout)
             [+ wall_coef / (1 + max(d_wall, HEAT_PRIOR_WALL_MIN_D))]

    (retail_literature, OPERATIONAL ASSUMPTION), which the traffic and
    revenue-placement criteria read. The wall term is off by default: the
    prior every synthetic experiment uses has none, unlike the calibrated
    store's (``_paint_synthetic_heatmap``), so on the synthetic scenarios
    the GA's objective does not reward wall proximity while the analytical
    one does. ``run_objective_alignment`` passes the calibrated prior's
    wall coefficient to measure what aligning the two changes. With
    ``wall_coef`` 0 the map is the one this code has always painted, value
    for value."""
    res = shop.customer_simulation.heat_map_resolution
    wc = int(synthetic.width * res) + 1
    hc = int(synthetic.height * res) + 1
    shop.customer_simulation.heat_map_data = np.zeros((wc, hc), dtype=np.float64)
    shop.customer_simulation.heat_raw = np.zeros((wc, hc), dtype=np.float32)
    # Higher density near (entrance, checkout) decaying with distance.
    for ix in range(wc):
        for iy in range(hc):
            x = ix / res; y = iy / res
            d_ent = math.hypot(x - synthetic.entrance[0], y - synthetic.entrance[1])
            d_chk = math.hypot(x - synthetic.checkout[0], y - synthetic.checkout[1])
            heat = (HEAT_PRIOR_SYNTH_ENTRANCE / (1.0 + d_ent)
                    + HEAT_PRIOR_SYNTH_CHECKOUT / (1.0 + d_chk))
            if wall_coef:
                d_wall = min(x, y, synthetic.width - x, synthetic.height - y)
                heat += wall_coef / (1.0 + max(d_wall, HEAT_PRIOR_WALL_MIN_D))
            shop.customer_simulation.heat_raw[ix, iy] = heat
    shop.customer_simulation.heat_map_data = (
        shop.customer_simulation.heat_raw.astype(np.float64)
    )
    shop.customer_simulation._floor_heat_raw = {
        1: shop.customer_simulation.heat_raw
    }


# --- base_params for the simulator's MC engine ---------------------------

def base_params_for(synthetic: SyntheticShop) -> Dict[str, Any]:
    """Construct the dict ``_ga_fitness`` / ``_mc_engine`` consume from
    the synthetic shop's parameters.

    Only the arrival rate and the spend come from the scenario; the rest
    are retail_literature stand-ins (conversion, impulse rate, spend SD,
    basket, abandonment, queue time). The result carries no anchor: a
    runner anchors it at the scenario's as-built layout with
    ``anchor_base_params`` before any layout is scored."""
    from sim_calibration import net_base_revenue
    mean_rev = float(np.mean([it.base_revenue for it in synthetic.items]))
    imp_val = max(
        np.mean([it.base_revenue for it in synthetic.items if it.is_impulse])
        if any(it.is_impulse for it in synthetic.items)
        else mean_rev * IMPULSE_VALUE_FRAC_OF_GROSS,
        0.5
    )
    return {
        'customers_per_hour': synthetic.daily_customers / DEFAULT_OP_HOURS_PER_DAY,
        'conversion_rate': ASSUMED_CONVERSION_RATE,
        # NET of the expected impulse spend so mc_engine's
        # additive impulse term doesn't double-count the baseline.
        'rev_per_converting_customer': net_base_revenue(
            mean_rev, IMPULSE_RATE_STANDIN, imp_val),
        'rev_per_customer_gross': mean_rev,
        'rev_std': max(mean_rev * REV_STD_FRAC_OF_MEAN, 0.5),
        'impulse_rate': IMPULSE_RATE_STANDIN,
        'avg_impulse_value': imp_val,
        'avg_basket_size': STANDIN_AVG_BASKET,
        'std_basket_size': STANDIN_STD_BASKET,
        'basket_sizes_observed': [int(STANDIN_AVG_BASKET)] * 30,
        'abandonment_rate': ABANDON_RATE_SYNTHETIC,
        'avg_queue_time': DEFAULT_QUEUE_TIME_S,
    }


# --- Build a headless shop from a calibrated real dataset ---------------

def build_headless_shop_from_calibration(
    params: CalibratedParams,
    max_items_per_category: int = 12,
    naive: bool = False,
) -> "HeadlessShop":
    """Materialize a real-dataset-calibrated shop on a Tk-free
    ``HeadlessShop`` instance.

    Reuses the existing dataset pipeline:
      * ``dataset_layout.build_layout_from_calibration`` lays out the
        sections + items based on the calibrated category mix.
      * ``CalibratedParams.seed_into(sim, shop=shop)`` populates
        analytics with the empirical basket / co-purchase /
        hourly-profile / dwell data, re-keys the per-item dicts from
        product id to the shop's item keys (which is how the GA score
        looks items up), and writes the shopping-list length sample
        -- the stocked part of each invoice -- that agents draw from,
        sets ``sim.run_time`` and ``sim.sim_time`` so
        ``extract_simulation_parameters`` reports the right
        ``customers_per_hour``, and stamps the calibration provenance
        block.

    The returned shop is immediately usable by ``run_ga_headless`` /
    ``paired_mc_revenue`` -- the GA's ``_ga_compute_layout_score`` reads
    from ``shop.customer_simulation.analytics`` which has been
    populated by ``seed_into``, and ``_ga_fitness`` calls the simulator's
    MC engine with no Tk dependency.
    """
    # HeadlessShop's constructor needs initial dimensions; the layout
    # builder will overwrite ``self.width`` / ``self.height`` to fit the
    # calibrated section grid. Start with a placeholder; the builder
    # sizes it correctly.
    shop = HeadlessShop(width=20.0, height=15.0)

    # Geometry: sections + items + entrance + checkout + heat-map buffer.
    # ``naive`` default False preserves the documented Tier-1 paper-grade
    # results (popularity-rank baseline). Figure C / the real-data worked
    # example pass ``naive=True`` to mirror the GUI default and show the
    # positive lift over an un-optimized starting layout.
    layout_stats = build_layout_from_calibration(
        shop, params, max_items_per_category=max_items_per_category,
        naive=naive,
    )

    # Analytics: empirical distributions seeded into sim.analytics so
    # the GA / MC pipelines see real values from t=0.
    params.seed_into(shop.customer_simulation, shop=shop)

    # Heat map: paint a literature-inspired bias (perimeter + entrance
    # + checkout proximity) so the traffic / revenue_placement
    # components of the GA score have non-trivial spatial gradient.
    # This mirrors what the live simulator's heat map would build up
    # from real customer trajectories, but lets the GA score
    # immediately rather than after a long sim warm-up.
    _paint_synthetic_heatmap(shop)

    # ``geometry_dirty`` was already set by build_layout_from_calibration.
    shop.as_built_layout = {n: (float(d['position'][0]), float(d['position'][1]))
                            for n, d in shop.floors[1]['items'].items()}
    return shop


def _paint_synthetic_heatmap(shop: "HeadlessShop") -> None:
    """Initial heat-map prior for a freshly built shop. Bias toward
    (entrance, checkout) and the perimeter racetrack. Mirrors what
    Larson 2005 reports as the dominant traffic pattern in real shops,
    so the GA's traffic_score / revenue_placement_score read a
    meaningful spatial gradient instead of an all-zero array.

    The live simulator's heat-map (driven by actual customer paths)
    overwrites this once a simulation runs; this is only the
    cold-start prior for headless GA optimization. Its coefficients are
    retail_literature's HEAT_PRIOR_CAL_* (OPERATIONAL ASSUMPTION): the
    shape is motivated by the cited traffic patterns, the sizes are ours."""
    import math
    sim = shop.customer_simulation
    res = sim.heat_map_resolution
    wc = sim.heat_raw.shape[0]
    hc = sim.heat_raw.shape[1]
    door = shop.door_position or (shop.width / 2, 0.2)
    # Find checkout center
    chk = None
    walls = shop.floors[1].get('walls', {})
    if 'Checkout' in walls:
        cp = walls['Checkout']['position']; cs = walls['Checkout']['size']
        chk = (cp[0] + cs[0] / 2, cp[1] + cs[1] / 2)
    if chk is None:
        chk = (door[0] - 3.0, door[1] + 0.5)

    heat = np.zeros((wc, hc), dtype=np.float32)
    for ix in range(wc):
        for iy in range(hc):
            x = ix / res; y = iy / res
            d_door = math.hypot(x - door[0], y - door[1])
            d_chk  = math.hypot(x - chk[0], y - chk[1])
            d_wall = min(x, y, shop.width - x, shop.height - y)
            heat[ix, iy] = (
                HEAT_PRIOR_CAL_ENTRANCE / (1.0 + d_door)      # entrance proximity
                + HEAT_PRIOR_CAL_CHECKOUT / (1.0 + d_chk)     # checkout proximity
                + HEAT_PRIOR_CAL_WALL
                / (1.0 + max(d_wall, HEAT_PRIOR_WALL_MIN_D))  # perimeter racetrack
            )
    sim.heat_raw = heat
    sim.heat_map_data = heat.astype(np.float64)
    if hasattr(sim, '_floor_heat_raw'):
        sim._floor_heat_raw = {1: heat}


def base_params_for_calibration(params: CalibratedParams) -> Dict[str, Any]:
    """``base_params`` dict for the simulator's MC engine, derived from a
    real-dataset calibration. Same shape as ``base_params_for`` (the
    synthetic version) so ``run_ga_headless`` / ``paired_mc_revenue``
    work without modification.

    Notes on the per-field derivation:
      * ``customers_per_hour`` is the calibrated VISITOR rate. The
        transactional data only records buyers (invoices per open hour),
        so visitors = buyers / assumed conversion; the MC engine applies
        the conversion rate again, which makes simulated purchases
        reproduce the observed invoice volume.
      * ``conversion_rate`` is the asserted assumption (not measured);
        same caveat as ``seed_into`` records under
        ``analytics['calibration']['conversion_rate_source']``.
      * basket-size mean/std come from the empirical distribution, and
        the full list goes into ``basket_sizes_observed``. None of the
        three reaches revenue: ``mc_engine`` reads only the list's mean
        and SD, for its tracked item-count series. Basket size affects
        revenue only as the ratio b(s)/b0 that the layout transform
        applies to spend per converter (``layout_objective``), so its
        level cancels.
      * ``rev_per_converting_customer`` and ``rev_std`` come from the
        empirical per-invoice revenue distribution (the mean net of the
        expected impulse spend).
      * ``impulse_rate`` / ``avg_impulse_value`` are stand-ins (real
        transactional data doesn't distinguish planned vs. impulse):
        retail_literature's IMPULSE_RATE_STANDIN and
        IMPULSE_VALUE_FRAC_OF_GROSS, so the MC projection's impulse term
        does not collapse to zero. A store with no impulse fixtures (the
        UCI store) never moves this term.
      * ``abandonment_rate`` is ABANDON_FRAC_OF_NONCONVERTERS of the
        non-converting visitors, an assumption.
    The result carries no anchor: ``anchor_base_params`` adds the store's
    as-built one.
    """
    from sim_calibration import net_base_revenue
    baskets = np.asarray(params.basket_sizes, dtype=np.float64)
    revs = np.asarray(params.invoice_revenues, dtype=np.float64)
    gross = float(revs.mean()) if revs.size else 1.0
    imp_val = max(gross * IMPULSE_VALUE_FRAC_OF_GROSS, 0.5)
    return {
        'customers_per_hour': float(params.visitors_per_hour),
        'conversion_rate': float(params.assumed_conversion_rate),
        # NET of the expected impulse spend, as above.
        'rev_per_converting_customer': net_base_revenue(
            gross, IMPULSE_RATE_STANDIN, imp_val),
        'rev_per_customer_gross': gross,
        'rev_std': float(revs.std(ddof=1)) if revs.size > 1 else 1.0,
        'impulse_rate': IMPULSE_RATE_STANDIN,   # UCI doesn't label impulse
        'avg_impulse_value': imp_val,
        'avg_basket_size': (float(baskets.mean()) if baskets.size
                            else STANDIN_AVG_BASKET),
        'std_basket_size': (float(baskets.std(ddof=1)) if baskets.size > 1
                            else STANDIN_STD_BASKET),
        'basket_sizes_observed': (list(map(int, baskets.astype(int)))
                                  if baskets.size
                                  else [int(STANDIN_AVG_BASKET)] * 30),
        'abandonment_rate': (max(0.0, 1.0 - float(params.assumed_conversion_rate))
                             * ABANDON_FRAC_OF_NONCONVERTERS),
        'avg_queue_time': DEFAULT_QUEUE_TIME_S,
    }


# --- Apply a layout to the headless shop ---------------------------------

def apply_layout(shop: HeadlessShop,
                 layout: Dict[str, Tuple[float, float]]) -> None:
    """Write the (x, y) for each item back into the shop's floor-1 items
    dict. Used between fitness evaluations."""
    for name, (x, y) in layout.items():
        if name in shop.floors[1]['items']:
            shop.floors[1]['items'][name]['position'] = [float(x), float(y)]


def layout_to_chromosome(layout: Dict[str, Tuple[float, float]],
                         item_names: List[str]) -> np.ndarray:
    """Pack a layout dict into the (N, 2) chromosome the GA uses."""
    return np.array(
        [[layout[n][0], layout[n][1]] for n in item_names],
        dtype=np.float64,
    )


def chromosome_to_layout(chrom: np.ndarray,
                         item_names: List[str]) -> Dict[str, Tuple[float, float]]:
    return {n: (float(chrom[i, 0]), float(chrom[i, 1]))
            for i, n in enumerate(item_names)}


# --- Paired-MC revenue evaluator -----------------------------------------

def paired_mc_revenue(shop: HeadlessShop,
                      item_names: List[str],
                      layout: Dict[str, Tuple[float, float]],
                      base_params: Dict[str, Any],
                      seed: int,
                      mc_iters: int = 500,
                      mc_days: int = 30) -> float:
    """Apply ``layout`` to ``shop``, then evaluate the simulator's
    fitness function with ``np.random`` seeded to ``seed``. Returning
    a single revenue number (MC mean).

    Paired-MC: callers MUST use the same ``seed`` across all candidates
    on the same scenario; this removes MC noise from the comparison
    so the per-method differences reflect real fitness gaps, not RNG
    variance. ``base_params`` must carry the store's anchor
    (``anchor_base_params``; ``layout_objective.MissingAnchorError``
    otherwise, before the shop is touched); the value is an unbiased
    estimate of ``experiments.closed_form.expected_revenue`` for the same
    inputs.

    The shop's previous item positions are restored before returning.
    ``run_ga_headless`` seeds its population from the shop's positions
    when no ``init_layout`` is given, so leaving the evaluated layout in
    place would let one evaluation warm-start a later, supposedly
    independent GA run."""
    require_anchor(base_params, 'paired_mc_revenue')
    f1_items = shop.floors[1]['items']
    saved = {n: f1_items[n]['position'] for n in layout if n in f1_items}
    try:
        apply_layout(shop, layout)
        chrom = layout_to_chromosome(layout, item_names)
        np.random.seed(seed)   # _mc_engine via _ga_fitness uses np.random
        return float(shop._ga_fitness(chrom, item_names, base_params,
                                      mc_days=mc_days, mc_iters=mc_iters))
    finally:
        # apply_layout rebinds 'position' to a new list, so the saved
        # objects are the untouched originals.
        for n, pos in saved.items():
            f1_items[n]['position'] = pos


# --- Headless GA loop ----------------------------------------------------

def _repair_chrom_overlaps(shop: "HeadlessShop",
                           chrom: np.ndarray,
                           item_names: List[str]) -> np.ndarray:
    """The second half of the shared repair, after ``_ga_repair``'s zone
    clip: keep the store's aisles, resolve overlaps, keep every fixture
    shoppable (``experiments._feasibility.repair_positions``).

    The GA's own ``_ga_repair`` only clamps each item to its zone; it does
    not check pair-wise overlap, and the zones include the aisles. So:

      * each item is clipped into its zone's REGION -- the zone less what
        the aisles between zones the as-built store keeps apart need
        (``_feasibility.aisle_plan``), so fixtures of two such zones stay
        ``shop_architecture.MIN_AISLE`` apart whatever the layout;
      * within-zone overlaps are resolved by the 2D-aware grid snap, run
        inside the region: items within a zone are sorted by current
        position into a row-major grid, then snapped to evenly spaced
        non-overlapping slots that preserve relative order. The
        chromosome's positional information is therefore retained as
        *rank* (rows by y, then left-to-right by x within each row);
        without this every random perturbation puts items of one zone on
        top of each other, the overlap penalty crushes the score and no
        search can leave the initial layout;
      * a zone holding a fixture whose access point is cut off from the
        door is re-packed single-file along its run, or returned to its
        as-built positions.

    Items are grouped by the section wall they belong to (``zone``, or
    ``Section_<category>`` when the item carries no zone). A department
    split over several zones is repaired zone by zone, so its items are
    not pulled into one of them. On a store whose zones exclude their
    aisles (the synthetic scenarios) every region is the zone less the
    repair's pad, and the result is the one the snap alone always gave."""
    return repair_positions(shop, chrom, item_names)


def feasible_layout(shop: "HeadlessShop",
                    item_names: List[str],
                    layout: Dict[str, Tuple[float, float]]
                    ) -> Dict[str, Tuple[float, float]]:
    """Map a layout onto the GA's feasible set.

    Runs exactly the repair chain every GA candidate goes through
    (``_ga_repair``: zone clipping and the impulse projection toward the
    checkout; then ``_repair_chrom_overlaps``: the aisle regions, the
    overlap snap and the reachability fallbacks). Every method's layouts
    pass through this before MC scoring so all methods are compared on,
    and search over, the same feasible set, and every layout in it keeps
    the floor-plan engine's invariants (``check_layout_invariants``).
    Deterministic: no RNG draws; idempotent."""
    chrom = layout_to_chromosome(layout, item_names)
    chrom = shop._ga_repair(chrom, item_names)
    chrom = _repair_chrom_overlaps(shop, chrom, item_names)
    return chromosome_to_layout(chrom, item_names)


# --- Search moves inside each item's zone --------------------------------

def _zone_position_bounds(shop: "HeadlessShop",
                          item_names: List[str]
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """``(lo, hi)``, each of shape (N, 2): the range of each item's (x, y)
    position that keeps the whole fixture inside the part of its zone the
    repair lets it occupy -- its REGION, the zone less what the aisles to
    the zones the as-built store keeps apart need, pad included
    (``_feasibility.AislePlan.position_bounds``), on every side the aisle
    rule narrows. On a side it leaves alone the range runs to the zone edge
    and ``_ga_repair`` adds its 0.05 m pad afterwards, as it always did.

    Drawing and moving inside the region, rather than the whole zone, keeps
    a uniform draw uniform over the positions the repair keeps instead of
    piling it on the edges of the aisle rule: the region clip never moves a
    position drawn here (``tests/test_feasibility_parity``). An item that
    resolves to no zone gets the box ``_ga_repair`` clips such an item to,
    the floor less 0.1 m on each side, so the search space stays the GA's on
    every item. A position is the fixture's lower-left corner, so this range
    is the zone (less the aisle rule) inset by the item's size. An item larger
    than its zone on an axis has a single position there, the zone's lower
    edge; ``feasible_layout`` then pins it where ``_ga_repair`` pins such an
    item, 0.05 m inside. On a store whose zones exclude their aisles (the
    synthetic scenarios) the aisle rule takes nothing, and these are the
    zone's own bounds."""
    plan = aisle_plan(shop, item_names)
    return plan.position_bounds()


def zone_sampler(shop: "HeadlessShop", item_names: List[str]):
    """Random-layout sampler for ``random_search``'s ``sampler`` hook on a
    shop whose items carry zones -- the calibrated store, which has no
    SyntheticShop for ``baselines.random_valid`` to draw from.

    ``sample(rng)`` draws every item's position uniformly over the range
    that keeps it inside its zone less the aisle rule (its region,
    ``_zone_position_bounds``), independently per item and axis, and returns
    ``{name: (x, y)}`` for ``item_names``.
    Items may overlap; callers map the layout through ``feasible_layout``,
    as ``random_search`` does with every draw, so the search runs over the
    GA's feasible set. Only ``rng`` is drawn from, never the global numpy
    RNG, so a draw cannot disturb the seeded Monte Carlo evaluation stream
    and the same generator state gives the same layout. The zones are read
    once, when the sampler is made."""
    names = list(item_names)
    if not names:
        raise ValueError("zone_sampler needs at least one item")
    lo, hi = _zone_position_bounds(shop, names)

    def sample(rng: np.random.Generator) -> Dict[str, Tuple[float, float]]:
        pos = rng.uniform(lo, hi)
        return {n: (float(pos[i, 0]), float(pos[i, 1]))
                for i, n in enumerate(names)}

    return sample


def zone_neighbor(shop: "HeadlessShop", item_names: List[str],
                  step_frac: float = 0.25, n_move: int = 1):
    """Move for ``simulated_annealing``'s ``neighbor`` hook on a shop whose
    items carry zones, the zone-based counterpart of
    ``metaheuristics._neighbor``.

    ``neighbor(layout, rng)`` picks ``n_move`` distinct items uniformly
    (one by default), moves each by a step drawn uniformly from
    +/- ``step_frac`` of the room it has on each axis (the part of its zone
    the aisle rule leaves it, less the item's size, as ``_neighbor`` scales
    its step by the section's inner bounds; ``_zone_position_bounds``), and
    clips it back inside that room. Every other item keeps its position.
    With ``n_move`` = 1 the draws are the one-item move's, call for call.
    No overlap repair happens here: callers map the result through
    ``feasible_layout``, as ``simulated_annealing`` does with every
    proposal. Only ``rng`` is drawn from, never the global numpy RNG.

    A move of several items is the annealer's answer to a store with many
    of them: on the 108-fixture store one-item moves give it about seven
    proposals per item at the paper's budget, and ``k_move_items`` scales
    the move with the item count."""
    if not step_frac > 0.0:
        raise ValueError("step_frac must be positive")
    names = list(item_names)
    if not names:
        raise ValueError("zone_neighbor needs at least one item")
    if not 1 <= int(n_move) <= len(names):
        raise ValueError("n_move must be between 1 and the number of items")
    n_move = int(n_move)
    lo, hi = _zone_position_bounds(shop, names)
    reach = step_frac * np.maximum(hi - lo, 1e-3)

    def neighbor(layout: Dict[str, Tuple[float, float]],
                 rng: np.random.Generator) -> Dict[str, Tuple[float, float]]:
        lay = dict(layout)
        if n_move == 1:
            picks = [int(rng.integers(0, len(names)))]
        else:
            picks = [int(i) for i in rng.choice(len(names), size=n_move,
                                                replace=False)]
        for i in picks:
            step = rng.uniform(-reach[i], reach[i])
            x, y = lay[names[i]]
            lay[names[i]] = (float(np.clip(x + step[0], lo[i, 0], hi[i, 0])),
                             float(np.clip(y + step[1], lo[i, 1], hi[i, 1])))
        return lay

    return neighbor


#: Items per annealing move on a large store: one item per this many
#: (``k_move_items``). With ten, the 108-fixture store moves ten items at a
#: time, which gives the annealer about as many proposals per unit of item
#: movement as one-item moves give it on the 8-10-item synthetic scenarios.
ITEMS_PER_MOVED_ITEM = 10


def k_move_items(n_items: int) -> int:
    """Items one annealing move displaces on a store of ``n_items``:
    ``max(1, n_items // ITEMS_PER_MOVED_ITEM)`` -- one on the synthetic
    scenarios, ten on the 108-fixture store."""
    return max(1, int(n_items) // ITEMS_PER_MOVED_ITEM)


# --- The anchor layout and the criteria's spread -------------------------

def layout_score(shop: "HeadlessShop", item_names: List[str],
                 layout: Dict[str, Tuple[float, float]]
                 ) -> Tuple[float, Dict[str, float]]:
    """``(score, breakdown)`` of ``layout`` as the GA scores it."""
    return shop._ga_compute_layout_score(
        layout_to_chromosome(layout, item_names), item_names, None)


def anchor_base_params(shop: "HeadlessShop", item_names: List[str],
                       base_params: Dict[str, Any],
                       as_built: Dict[str, Tuple[float, float]]
                       ) -> Dict[str, Any]:
    """A copy of ``base_params`` anchored at the store's as-built layout.

    The anchor is the score and breakdown of ``feasible_layout(as_built)``:
    the as-built layout after the shared repair, which is the layout every
    search starts from and the baseline the comparisons are made against.
    The elasticities then act on each layout's score DIFFERENCE from it
    (``layout_objective``), so the as-built store reproduces the calibrated
    conversion, basket, spend and daily volume exactly. Every runner calls
    this once per store, before anything is scored."""
    score, bkd = layout_score(shop, item_names,
                              feasible_layout(shop, item_names, as_built))
    return with_anchor(base_params, score, bkd, source='as_built_repaired')


def base_params_record(base_params: Dict[str, Any]) -> Dict[str, Any]:
    """``base_params`` as a runner records it in its sidecar: every scalar
    and the anchor as they are, and each list (the observed basket sizes,
    tens of thousands of entries on the calibrated store) as its length,
    mean, SD and a SHA-256 of its values, so the record stays small and
    still pins the input."""
    out: Dict[str, Any] = {}
    for k, v in base_params.items():
        if isinstance(v, (list, tuple, np.ndarray)):
            arr = np.asarray(v, dtype=np.float64)
            out[k] = {'n': int(arr.size),
                      'mean': float(arr.mean()) if arr.size else None,
                      'std': float(arr.std(ddof=1)) if arr.size > 1 else None,
                      'sha256': hashlib.sha256(arr.tobytes()).hexdigest()}
        elif isinstance(v, np.generic):
            out[k] = v.item()
        else:
            out[k] = v
    return out


def closed_form_revenue(shop: "HeadlessShop", item_names: List[str],
                        layout: Dict[str, Tuple[float, float]],
                        base_params: Dict[str, Any], horizon_days: int,
                        **kw: Any) -> float:
    """The exact mean of ``paired_mc_revenue`` for ``layout`` over
    ``horizon_days`` (``experiments.closed_form.expected_revenue``): what
    the Monte Carlo value converges to as its iterations grow. ``kw`` goes
    to ``expected_revenue`` (elasticity overrides, basket exclusions)."""
    from experiments.closed_form import expected_revenue
    return float(expected_revenue(
        base_params, int(horizon_days), shop=shop,
        chromosome=layout_to_chromosome(layout, item_names),
        item_names=list(item_names), **kw))


def layout_json(layout: Dict[str, Tuple[float, float]]) -> Dict[str, list]:
    """A layout as a run saves it: ``{name: [x, y]}`` (lower-left corners,
    metres), floats written by JSON in their shortest exact form."""
    return {n: [float(x), float(y)] for n, (x, y) in layout.items()}


def write_json(out_dir: str, name: str, obj: Any) -> str:
    """Write ``obj`` as ``out_dir/name`` (LF line ends, UTF-8); returns the
    path."""
    path = os.path.join(out_dir, name)
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(obj, f, indent=1, default=str)
    return path


def checked_layout(shop: "HeadlessShop", item_names: List[str],
                   layout: Dict[str, Tuple[float, float]],
                   label: str) -> Dict[str, Any]:
    """Check a layout a run is about to report against the floor-plan
    engine's invariants (``check_layout_invariants``: inside its zones, no
    overlap, the as-built aisles kept, every fixture shoppable), raising
    ``LayoutInvariantError`` on any violation, and return the record the
    run keeps (``invariant_summary``)."""
    check_layout_invariants(shop, item_names, layout, label=label)
    return invariant_summary(shop, item_names, layout)


#: The criteria of the composite, in the order ``_ga_compute_layout_score``
#: sums them.
SCORE_CRITERIA = ('traffic', 'cross_merch', 'impulse', 'flow',
                  'revenue_placement', 'section_compliance', 'accessibility',
                  'overlap_penalty', 'bottleneck_penalty')

#: Reference sample of ``criterion_spread``: its size (the number the
#: review's spread estimates used) and its seed.
SPREAD_N_LAYOUTS = 150
SPREAD_SEED = 7_000


def criterion_spread(shop: "HeadlessShop", item_names: List[str],
                     n_layouts: int = SPREAD_N_LAYOUTS,
                     seed: int = SPREAD_SEED) -> Dict[str, Any]:
    """Spread of each score criterion over repaired random layouts, and the
    effective weight it carries.

    Draws ``n_layouts`` layouts uniformly inside each item's zone
    (``zone_sampler``, drawing from ``np.random.default_rng(seed)`` only, so
    the global stream is untouched), maps each through ``feasible_layout``
    -- the set every method searches -- and scores it. A criterion's
    EFFECTIVE weight is its nominal weight times its SD over that sample:
    how far the composite actually moves with it between feasible layouts.
    Equal nominal weights (Dawes 1979) are equal effective weights only for
    standardized criteria. Here a criterion that barely varies (traffic,
    revenue placement) or not at all (section compliance, which the repair
    holds at 1 for every layout) carries almost none, whatever its nominal
    weight.

    Returns ``{'n_layouts', 'seed', 'sampler', 'mean', 'sd', 'weight',
    'effective_weight', 'effective_share'}``, each per-criterion mapping
    keyed by ``SCORE_CRITERIA``. ``weight`` is signed as the criterion
    enters the score; ``effective_weight`` is |weight| x SD; the share is
    each positive criterion's effective weight over their sum (the
    penalties are reported, not shared)."""
    if n_layouts < 2:
        raise ValueError("criterion_spread needs at least two layouts")
    sample = zone_sampler(shop, item_names)
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_layouts):
        lay = feasible_layout(shop, item_names, sample(rng))
        rows.append(layout_score(shop, item_names, lay)[1])
    mean = {k: float(np.mean([r[k] for r in rows])) for k in SCORE_CRITERIA}
    sd = {k: float(np.std([r[k] for r in rows], ddof=1))
          for k in SCORE_CRITERIA}
    weight = {k: float(CRITERION_WEIGHTS[k]) for k in SCORE_CRITERIA}
    eff = {k: abs(weight[k]) * sd[k] for k in SCORE_CRITERIA}
    positive = [k for k in SCORE_CRITERIA if weight[k] > 0]
    total = sum(eff[k] for k in positive)
    share = {k: (eff[k] / total if total > 0 else float('nan'))
             for k in positive}
    return {'n_layouts': int(n_layouts), 'seed': int(seed),
            'sampler': 'zone_sampler + feasible_layout',
            'mean': mean, 'sd': sd, 'weight': weight,
            'effective_weight': eff, 'effective_share': share}


def run_ga_headless(shop: HeadlessShop,
                    item_names: List[str],
                    base_params: Dict[str, Any],
                    pop_size: int = 30,
                    n_gens: int = 25,
                    mut_rate: float = 0.18,
                    elite_frac: float = 0.20,
                    mc_iters: int = 500,
                    mc_days: int = 30,
                    rng_seed: int = 0,
                    verbose: bool = False,
                    init_layout: Optional[Dict[str, Tuple[float, float]]] = None,
                    ) -> Dict[str, Any]:
    """Run the same GA the GUI does, headlessly, with a fixed seed.

    Mirrors ``GARunMixin._run_ga_optimization`` minus the Tk progress
    chatter, with two differences on a store whose zones include aisles:
    its overlap repair is the shared one (``_repair_chrom_overlaps``: the
    GUI's rank-preserving snap plus the aisle rule and the reachability
    fallbacks), and its initial noise and mutation are scaled to the zone
    less the aisles that repair keeps (``HeadlessShop._ga_move_boxes``) --
    the space random search and the annealer draw and move in. On a store
    whose zones exclude their aisles (the synthetic scenarios) both are the
    GUI's own. Returns:
        {
          'best_chrom':   (N, 2) array,
          'best_fit':     float,
          'history_best': [float] of length n_gens,
          'history_avg':  [float],
          'history_diversity': [float],
          'n_search_evals': n_gens * pop_size,
          'n_final_evals':  n_final_seeds * pop_size,
          'n_evals':        total fitness evaluations,
        }

    ``init_layout`` fixes the starting point: the initial population is
    seeded from it instead of the shop's current item positions, so a
    run does not depend on whatever layout earlier code left on the shop.
    Search evaluations use MC seeds ``rng_seed*1000 + gen`` for gen in
    [0, n_gens); final selection uses ``rng_seed*1000 + n_gens + 1 + s``
    for s in [0, n_final_seeds). ``experiments.metaheuristics`` uses the
    same namespaces and budget.

    The pool the final selection re-scores is the population the last
    generation BRED, not the one it scored: the ``n_elite`` best layouts
    of the last generation, ranked under that generation's one seed, plus
    ``pop_size - n_elite`` children crossed and mutated from it that no
    search evaluation has scored. All ``pop_size`` of them are re-scored
    under the ``n_final_seeds`` selection seeds and the best mean wins; the
    comparators re-score the same number of candidates under the same
    seeds (``experiments.metaheuristics``).

    Determinism / threading contract: reproducibility from
    ``rng_seed`` relies on the GLOBAL numpy RNG (``np.random.seed`` inside
    ``_paired_fitness`` and the GA operators). It is therefore guaranteed
    only in a SINGLE-THREADED process with no other concurrent consumer of
    the global RNG -- which is exactly the headless experiment path used
    for every number in the paper. The live GUI runs a simulation worker
    thread, so bit-identical reproducibility is NOT claimed for interactive
    runs; the paper's figures come from this headless path. See
    ``tests/test_ga_determinism.py``.

    The population and operator streams are array-seeded so they stay
    disjoint from the integer MC evaluation seeds above (see
    ``_INIT_POP_STREAM_TAG``).

    ``base_params`` must carry the store's anchor (``anchor_base_params``);
    without one the run stops here with
    ``layout_objective.MissingAnchorError`` instead of searching the
    unanchored objective.
    """
    require_anchor(base_params, 'run_ga_headless')
    np.random.seed([rng_seed, _INIT_POP_STREAM_TAG])

    n_elite = max(1, int(pop_size * elite_frac))
    if init_layout is not None:
        current = layout_to_chromosome(init_layout, item_names)
    else:
        current = shop._ga_encode(item_names)
    # The starting layout enters through the same repair as every other
    # candidate: an as-built fixture can sit flush on its zone edge, inside
    # the clearance the repair enforces, and the GA must only ever score
    # and return layouts from the feasible set its comparators are mapped to.
    start = shop._ga_repair(current.copy(), item_names)
    population: List[np.ndarray] = [
        _repair_chrom_overlaps(shop, start, item_names)]
    # The initial noise is scaled to, and clipped into, the box the GA's
    # mutation uses: the zone less the aisle rule (``_ga_move_boxes``).
    boxes = shop._ga_move_boxes(item_names)
    for _ in range(pop_size - 1):
        noisy = current.copy()
        for i, n in enumerate(item_names):
            data = shop._ga_get_item_data(n)
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
                hi_x = shop.width - w - 0.1
                hi_y = shop.height - h - 0.1
                sigma_x = shop.width * 0.15
                sigma_y = shop.height * 0.15
            if hi_x < lo_x: hi_x = lo_x
            if hi_y < lo_y: hi_y = lo_y
            noisy[i, 0] = float(np.clip(
                current[i, 0] + np.random.normal(0, sigma_x), lo_x, hi_x))
            noisy[i, 1] = float(np.clip(
                current[i, 1] + np.random.normal(0, sigma_y), lo_y, hi_y))
        noisy = shop._ga_repair(noisy, item_names)
        noisy = _repair_chrom_overlaps(shop, noisy, item_names)
        population.append(noisy)

    history_best: List[float] = []
    history_avg:  List[float] = []
    history_diversity: List[float] = []
    _ga_diag = float(np.hypot(shop.width, shop.height))
    n_evals = [0]

    def _paired_fitness(pop: List[np.ndarray], mc_seed: int) -> np.ndarray:
        """Evaluate every chromosome in ``pop`` under the SAME RNG seed
        in ``mc_engine``. Without this, MC noise (~2-3% SE at mc_iters=500)
        dominates real fitness differences, and the GA's selection
        becomes random. Paired-MC across the population is the
        within-GA analogue of the paired-MC we already use across
        methods in ``paired_mc_revenue``.
        """
        out = np.empty(len(pop), dtype=np.float64)
        for k, p in enumerate(pop):
            np.random.seed(mc_seed)
            out[k] = shop._ga_fitness(p, item_names, base_params,
                                      mc_days, mc_iters)
        n_evals[0] += len(pop)
        return out

    # GA-loop RNG (used for crossover/mutation/tournament) is separated
    # from the MC-evaluation RNG so the latter can be paired across
    # candidates without disturbing the former. We use Python's random
    # for GA-loop choices via numpy, and reseed np.random freshly inside
    # ``_paired_fitness`` for each MC eval.
    ga_rng = np.random.RandomState([rng_seed, _OPERATOR_STREAM_TAG])

    for gen in range(n_gens):
        # Every candidate in this generation shares one MC seed -> the
        # fitness differences reflect real layout quality, not noise.
        mc_seed_gen = rng_seed * 1000 + gen
        fitness = _paired_fitness(population, mc_seed_gen)
        rank = np.argsort(fitness)[::-1]
        population = [population[r] for r in rank]
        fitness = fitness[rank]

        history_best.append(float(fitness[0]))
        history_avg.append(float(fitness.mean()))
        # Population diversity: mean per-coordinate spread of
        # the population, normalized by the shop diagonal. A collapse to ~0
        # would signal premature convergence; a nonzero plateau shows the
        # search retains exploratory spread through the run.
        pop_arr = np.stack(population, axis=0)          # (P, N, 2)
        history_diversity.append(
            float(pop_arr.std(axis=0).mean() / max(_ga_diag, 1e-9)))
        if verbose:
            print(f"  gen {gen+1:>3d}/{n_gens}: "
                  f"best={fitness[0]:>10.2f}  avg={fitness.mean():>10.2f}",
                  flush=True)

        new_pop: List[np.ndarray] = [p.copy() for p in population[:n_elite]]
        while len(new_pop) < pop_size:
            t_size = min(5, pop_size)
            t_idx  = ga_rng.choice(pop_size, t_size, replace=False)
            p1 = population[t_idx[np.argmax(fitness[t_idx])]]
            t_idx2 = ga_rng.choice(pop_size, t_size, replace=False)
            p2 = population[t_idx2[np.argmax(fitness[t_idx2])]]
            # Crossover / mutate use module np.random; isolate them by
            # temporarily setting the global state from ga_rng.
            saved_state = np.random.get_state()
            np.random.set_state(ga_rng.get_state())
            c1, c2, blend = shop._ga_crossover(p1, p2)
            children = []
            for child in (c1, c2, blend):
                child = shop._ga_mutate(child, item_names, mut_rate)
                child = shop._ga_repair(child, item_names)
                # Resolve within-section overlaps so the GA can actually
                # explore the section bound -- without this every mutated
                # candidate's fitness collapses to the overlap-penalty
                # floor and the GA can't escape the initial layout.
                child = _repair_chrom_overlaps(shop, child, item_names)
                children.append(child)
            ga_rng.set_state(np.random.get_state())
            np.random.set_state(saved_state)
            for child in children:
                new_pop.append(child)
                if len(new_pop) >= pop_size:
                    break
        population = new_pop[:pop_size]

    # Final evaluation on the population the last generation bred (its
    # elites plus children never scored): average over multiple paired-MC
    # seeds so a single unlucky seed doesn't decide the winner. With one
    # seed, the chromosome that happens to score best under THAT specific
    # RNG state wins -- and that's often the initial (grid) chromosome
    # since it's always feasible. Averaging across ``n_final_seeds`` seeds
    # reduces SE by sqrt(n) and lets the GA's genuine improvements show
    # through.
    n_final_seeds = GA_N_FINAL_SEEDS
    n_search_evals = n_evals[0]
    final_fits_stack = np.zeros((n_final_seeds, len(population)), dtype=np.float64)
    for s_idx in range(n_final_seeds):
        seed_s = rng_seed * 1000 + n_gens + 1 + s_idx
        final_fits_stack[s_idx] = _paired_fitness(population, seed_s)
    final_fit = final_fits_stack.mean(axis=0)
    best_idx = int(np.argmax(final_fit))
    return {
        'best_chrom':   population[best_idx],
        'best_fit':     float(final_fit[best_idx]),
        'history_best': history_best,
        'history_avg':  history_avg,
        'history_diversity': history_diversity,
        # Per generation: its MC seed and the population's best and mean
        # fitness under it, and its diversity -- the convergence trace.
        'trace': [{'gen': g, 'seed': rng_seed * 1000 + g,
                   'best': history_best[g], 'mean': history_avg[g],
                   'diversity': history_diversity[g]}
                  for g in range(len(history_best))],
        'n_elite':      n_elite,
        'final_pool_means': [float(v) for v in final_fit],
        'n_search_evals': n_search_evals,
        'n_final_evals':  n_evals[0] - n_search_evals,
        'n_evals':        n_evals[0],
    }


# --- Bootstrap CI --------------------------------------------------------

def bootstrap_ci(samples: np.ndarray,
                 alpha: float = 0.05,
                 n_boot: int = 2000,
                 seed: int = 0) -> Tuple[float, float, float]:
    """Percentile bootstrap CI for the mean. Returns (mean, lo, hi)."""
    samples = np.asarray(samples, dtype=np.float64)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return (float('nan'), float('nan'), float('nan'))
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    n = samples.size
    for k in range(n_boot):
        idx = rng.integers(0, n, n)
        means[k] = samples[idx].mean()
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return (float(samples.mean()), lo, hi)


# --- Provenance / sidecar ------------------------------------------------

def git_sha() -> str:
    try:
        out = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        return out.decode().strip()
    except Exception:
        return "no-git"


def git_worktree_state() -> Dict[str, Any]:
    """Uncommitted state of the checkout the run was made from.

    ``git_sha`` names HEAD only; a run made from a working tree with edits
    on top of it would otherwise point at a commit that does not contain
    the code that produced its numbers. Tracked edits are captured by
    hashing ``git diff HEAD`` over the executable files only (``*.py`` plus
    the build entry points), so that two runs of the same code hash the
    same even when the manuscript or the notes have moved in between;
    the recorded list of modified paths is scoped the same way.
    Untracked ``.py`` files are hashed as well, since a new module can
    change results without appearing in that diff. Other untracked paths
    (result directories, including this run's own) are ignored so that a
    clean checkout does not read as dirty.
    """
    cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _git(*args: str) -> bytes:
        return subprocess.check_output(['git', *args],
                                       stderr=subprocess.DEVNULL, cwd=cwd)

    try:
        top = _git('rev-parse', '--show-toplevel').decode().strip()
        modified = _git('status', '--porcelain', '--untracked-files=no',
                        '--', ':(top)*.py', ':(top)Makefile',
                        ':(top)reproduce.ps1',
                        ':(top)requirements.txt',
                        ':(top)requirements-lock.txt').decode().splitlines()
        untracked = [p for p in _git('ls-files', '--others',
                                     '--exclude-standard', '--full-name',
                                     '-z', '--', ':(top)*.py')
                     .decode().split('\0') if p]
        h = hashlib.sha256(_git('diff', 'HEAD', '--no-ext-diff', '--',
                                ':(top)*.py', ':(top)Makefile',
                                ':(top)reproduce.ps1',
                                ':(top)requirements.txt',
                                ':(top)requirements-lock.txt'))
        for rel in sorted(untracked):
            h.update(rel.encode())
            with open(os.path.join(top, rel), 'rb') as f:
                h.update(f.read())
        return {
            'git_dirty':        bool(modified or untracked),
            'git_diff_sha256':  h.hexdigest(),
            'git_modified':     modified,
            'git_untracked_py': sorted(untracked),
        }
    except Exception:
        return {'git_dirty': None, 'git_diff_sha256': None}


#: retail_literature constants the objective reads besides the weights and
#: elasticities -- the Monte Carlo engine's guards included -- and the
#: synthetic scenarios' analytical ground truth, stamped into every sidecar
#: next to them.
OBJECTIVE_CONSTANTS = (
    'ABANDON_FLOW_COEF', 'ABANDON_SECTION_COEF', 'ABANDON_BOTTLENECK_COEF',
    'ABANDON_FLOOR_FRAC', 'CONV_CLAMP_LO', 'CONV_CLAMP_HI',
    'IMPULSE_RATE_CLAMP_HI', 'ABANDON_RATE_CLAMP_HI', 'ABANDON_BASE_CAP',
    'NET_BASE_REVENUE_MIN_FRAC', 'QUEUE_BOTTLENECK_FACTOR',
    'QUEUE_PENALTY_SLOPE', 'QUEUE_PENALTY_FLOOR', 'DEFAULT_QUEUE_TIME_S',
    'ASSUMED_CONVERSION_RATE', 'IMPULSE_RATE_STANDIN',
    'IMPULSE_VALUE_FRAC_OF_GROSS', 'IMPULSE_VALUE_CV', 'REV_STD_FRAC_OF_MEAN',
    'ABANDON_RATE_SYNTHETIC', 'ABANDON_FRAC_OF_NONCONVERTERS',
    'STANDIN_AVG_BASKET', 'STANDIN_STD_BASKET',
    'HEAT_PRIOR_SYNTH_ENTRANCE', 'HEAT_PRIOR_SYNTH_CHECKOUT',
    'HEAT_PRIOR_CAL_ENTRANCE', 'HEAT_PRIOR_CAL_CHECKOUT',
    'HEAT_PRIOR_CAL_WALL', 'HEAT_PRIOR_WALL_MIN_D',
    'DEFAULT_OP_HOURS_PER_DAY', 'DEFAULT_WEEKEND_MULTIPLIER',
    'DEFAULT_DAY_NOISE_STD',
    # mc_engine's numerical guards, which the closed form applies too
    'MC_LAMBDA_FLOOR', 'MC_LAMBDA_CAP', 'MC_CONV_CLAMP_LO',
    'MC_CONV_CLAMP_HI', 'MC_SD_FLOOR',
    # mc_engine's day-spend law: artifacts made under the former floored
    # normal lack it, which is how a validator tells them apart
    'MC_SPEND_LAW',
    # the synthetic scenarios' analytical ground truth (synthetic_shops)
    'SYNTH_DETOUR_ELASTICITY_RANGE', 'SYNTH_DETOUR_ELASTICITY_DEFAULT',
    'SYNTH_IMPULSE_ELASTICITY_RANGE', 'SYNTH_IMPULSE_ELASTICITY_DEFAULT',
    'SYNTH_PERIMETER_ELASTICITY_RANGE', 'SYNTH_PERIMETER_ELASTICITY_DEFAULT',
    'SYNTH_AFFINITY_RANGE', 'SYNTH_AFFINITY_DENSITY', 'SYNTH_DAILY_CUSTOMERS',
)


def elasticity_snapshot() -> Dict[str, Any]:
    return {
        'GA_weights': asdict(GaWeights()),
        'ELASTICITY_CONV_BASE':     ELASTICITY_CONV_BASE,
        'ELASTICITY_CONV_GAIN_MAX': ELASTICITY_CONV_GAIN_MAX,
        'ELASTICITY_IMP_BASE':      ELASTICITY_IMP_BASE,
        'ELASTICITY_IMP_GAIN_MAX':  ELASTICITY_IMP_GAIN_MAX,
        'ELASTICITY_BSK_BASE':      ELASTICITY_BSK_BASE,
        'ELASTICITY_BSK_GAIN_MAX':  ELASTICITY_BSK_GAIN_MAX,
        # How the elasticities act: on the score difference from the anchor
        # layout each run records in its base_params (layout_objective).
        'anchoring': 'difference from base_params["score_anchor"]',
        'objective_constants': {k: getattr(RL, k)
                                for k in OBJECTIVE_CONSTANTS},
    }


def package_versions() -> Dict[str, str]:
    """Resolved version of every package the experiments can load: the
    direct dependencies of ``requirements.txt`` and everything they
    require, transitively, as installed (``environment_lock``). The
    paper's numbers depend on these, so they are stamped alongside the
    seed and git SHA; ``lock_record`` says whether they match the shipped
    ``requirements-lock.txt``. Uses importlib.metadata so no heavy import
    is forced."""
    from environment_lock import dependency_closure
    return {name: (rec['version'] if rec['version'] is not None
                   else 'not-installed')
            for name, rec in dependency_closure().items()}


def lock_record() -> Dict[str, Any]:
    """The lock file as a sidecar records it: path (relative to the
    repository), SHA-256, and whether the environment matched it."""
    from environment_lock import lock_record as _lock_record
    return _lock_record()


def hardware_record() -> Dict[str, Any]:
    """The machine a run was made on, so a family's wall time can be read
    against it: logical CPUs, the processor string, the architecture and
    the physical memory in GiB (None where the platform does not say)."""
    ram = None
    try:
        if sys.platform == 'win32':
            import ctypes

            class _MemStatus(ctypes.Structure):
                _fields_ = [('dwLength', ctypes.c_ulong),
                            ('dwMemoryLoad', ctypes.c_ulong),
                            ('ullTotalPhys', ctypes.c_ulonglong),
                            ('ullAvailPhys', ctypes.c_ulonglong),
                            ('ullTotalPageFile', ctypes.c_ulonglong),
                            ('ullAvailPageFile', ctypes.c_ulonglong),
                            ('ullTotalVirtual', ctypes.c_ulonglong),
                            ('ullAvailVirtual', ctypes.c_ulonglong),
                            ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]
            st = _MemStatus()
            st.dwLength = ctypes.sizeof(_MemStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                ram = st.ullTotalPhys / 2 ** 30
        elif hasattr(os, 'sysconf'):
            ram = (os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
                   / 2 ** 30)
    except Exception:
        ram = None
    return {'cpu_count': os.cpu_count(), 'processor': platform.processor(),
            'machine': platform.machine(),
            'ram_gib': None if ram is None else round(float(ram), 1)}


def provenance_snapshot() -> Dict[str, Any]:
    """HEAD plus the uncommitted state of the checkout, as of right now.

    Runners take this at start-up and hand it to ``write_sidecar``: a
    paper-grade run can finish hours after its code was imported, and the
    working tree may have moved on in between, so reading the tree when the
    sidecar is written would describe code the run never executed."""
    return {'git_sha': git_sha(), **git_worktree_state()}


# Fallback for callers that do not take their own snapshot: the state as
# the experiment code was imported, which is at least no later than the
# first line of the run.
_PROVENANCE_AT_IMPORT: Dict[str, Any] = provenance_snapshot()


#: Keys whose string values are file-system paths (``out_root``,
#: ``retail_path``, ``figs_dir``, ``source_path``, ...).
_PATH_KEY_RE = re.compile(r'(?:^|_)(?:path|paths|dir|dirs|root|file|files)$')

#: Path keys that name a folder (``out_root``, ``figs_dir``, ...). No runner
#: records a folder relative to its run directory -- only its own output
#: files, under ``*_path`` keys -- so a relative folder was given relative to
#: the working directory, wherever in the payload it sits.
_DIR_KEY_RE = re.compile(r'(?:^|_)(?:dir|dirs|root)$')

#: Mappings that hold a run's argparse namespace. A relative path in them
#: was typed relative to the working directory the run started in.
_ARGS_KEYS = frozenset({'args', 'config'})


def _portable_str(s: str, path_key: bool, cwd_relative: bool) -> str:
    if not s or '\n' in s:
        return s
    if os.path.isabs(s):
        rel = repo_relative(s)
        # Outside the repository: kept exactly as recorded.
        return s if os.path.isabs(rel) else rel
    if path_key and cwd_relative:
        return repo_relative(s)
    return s


def _portable(obj: Any, key: Any, cwd_relative: bool) -> Any:
    if isinstance(obj, dict):
        return {k: (v if isinstance(k, str) and k.startswith('git_')
                    else _portable(v, k, cwd_relative or k in _ARGS_KEYS))
                for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_portable(v, key, cwd_relative) for v in obj]
    if isinstance(obj, os.PathLike):
        obj, path_key = os.fspath(obj), True
    else:
        path_key = isinstance(key, str) and bool(_PATH_KEY_RE.search(key))
    if isinstance(obj, str):
        dir_key = isinstance(key, str) and bool(_DIR_KEY_RE.search(key))
        return _portable_str(obj, path_key,
                             cwd_relative or dir_key or key == 'source_path')
    return obj


def portable_paths(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Copy of ``payload`` with the paths it records made portable.

    Every absolute path inside the repository, at any depth, becomes
    relative to the repository root with forward slashes
    (``dataset_paths.repo_relative``); an absolute path outside it is left
    as it is. A relative path under a path-valued key
    (``*_path``, ``*_dir``, ``*_root``, ...) is resolved against the working
    directory first when it is known to be relative to it -- inside a run's
    ``args`` / ``config`` namespace, under a folder key (``*_dir``,
    ``*_root``) anywhere, and a dataset ``source_path``, which
    ``dataset_provenance.stamp`` records as it was handed. Any other
    relative path is left alone: the runners record their own outputs
    (``csv_path``, ``figure_path``) relative to the run directory.
    ``provenance_snapshot``'s ``git_*`` fields are copied unchanged. The
    input is not modified."""
    return _portable(payload, None, False)


def with_git_state_check(provenance: Dict[str, Any]) -> Dict[str, Any]:
    """A copy of a start-of-run ``provenance_snapshot()`` with
    ``git_state_changed_during_run``: whether HEAD or the uncommitted diff
    differs now, as the run's results are written. A record without it
    cannot show its commit is the code behind its numbers, and the macro
    generator refuses it."""
    prov = dict(provenance)
    at_write = provenance_snapshot()
    prov['git_state_changed_during_run'] = bool(
        at_write.get('git_sha') != prov.get('git_sha')
        or at_write.get('git_diff_sha256') != prov.get('git_diff_sha256'))
    return prov


def write_sidecar(out_dir: str, payload: Dict[str, Any],
                  provenance: Optional[Dict[str, Any]] = None) -> str:
    """Write a JSON sidecar with the merged payload + provenance fields.
    Returns the absolute path.

    ``provenance`` is a ``provenance_snapshot()`` taken when the run
    started; without it the import-time snapshot is used. The checkout is
    read again here, and ``git_state_changed_during_run`` records whether
    it moved while the run was in flight.

    Paths in ``payload`` (``out_root``, ``retail_path``, ``figs_dir``, a
    dataset's ``source_path``, ...) are written by ``portable_paths``:
    relative to the repository root when they lie inside it, so an artifact
    shipped with the code does not carry the folder it was produced in."""
    os.makedirs(out_dir, exist_ok=True)
    prov = with_git_state_check(
        provenance if provenance is not None else _PROVENANCE_AT_IMPORT)
    full = {
        **prov,
        'python':            sys.version.split()[0],
        'platform':          platform.platform(),
        'packages':          package_versions(),
        'requirements_lock': lock_record(),
        'hardware':          hardware_record(),
        'iso_time':          time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime()),
        'elasticities':      elasticity_snapshot(),
        **portable_paths(payload),
    }
    path = os.path.join(out_dir, 'sidecar.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(full, f, indent=2, default=str)
    return path


def make_run_dir(parent: str, prefix: str) -> str:
    """Create a timestamped subdirectory under ``parent`` for one run.
    Returns the absolute path.

    The directory is created exclusively: two runs of the same experiment
    started within the same second would otherwise share a directory and
    overwrite each other's results, with nothing to show for it. The
    disambiguating suffix goes AFTER the timestamp, zero-padded so that it
    stays in numeric order past the ninth, and the newest-by-name ordering
    the macro tooling relies on still holds."""
    stamp = time.strftime('%Y%m%d-%H%M%S')
    base = os.path.join(parent, f'{prefix}_{stamp}')
    for attempt in range(1, 100):
        out = base if attempt == 1 else f'{base}-{attempt:02d}'
        try:
            os.makedirs(out)
            return out
        except FileExistsError:
            continue
    raise RuntimeError(f"could not create a fresh run directory for {base}")
