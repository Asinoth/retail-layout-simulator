"""Shared infrastructure for the TOMACS Tier-1 experiments.

Provides:
  * ``build_headless_shop(synthetic_shop)`` -- materialize a SyntheticShop
    into a Tk-free shop instance that the existing GA / fitness / MC
    code can run against without ever instantiating a Tk root.
  * ``base_params_for(synthetic_shop)`` -- construct the ``base_params``
    dict that ``_ga_fitness`` and ``_mc_engine`` consume, derived from
    the synthetic shop's parameters.
  * ``apply_layout_to_shop(shop, layout, item_names)`` -- write a layout
    dict back into the headless shop's floor-1 items.
  * ``paired_mc_revenue(shop, item_names, layout, base_params, seed,
    mc_iters, mc_days)`` -- evaluate one layout's revenue under the
    simulator's fitness function with a FIXED RNG seed; paired across
    layouts to remove MC noise from method comparisons.
  * ``bootstrap_ci(samples, alpha, n_boot)`` -- percentile bootstrap CI.
  * ``write_sidecar(out_dir, payload)`` -- JSON sidecar with seed +
    git SHA + elasticity snapshot for reproducibility.

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
from retail_literature import (
    GaWeights,
    ELASTICITY_CONV_BASE, ELASTICITY_CONV_GAIN_MAX,
    ELASTICITY_IMP_BASE,  ELASTICITY_IMP_GAIN_MAX,
    ELASTICITY_BSK_BASE,  ELASTICITY_BSK_GAIN_MAX,
)


# ─── Headless shop class ─────────────────────────────────────────────────

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


# ─── Build a headless shop from a SyntheticShop ──────────────────────────

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
    # impulse ~5-10s. Random within these bands so the dwell_alignment
    # term in the GA composite isn't a constant.
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
    res = shop.customer_simulation.heat_map_resolution
    wc = int(synthetic.width * res) + 1
    hc = int(synthetic.height * res) + 1
    shop.customer_simulation.heat_map_data = np.zeros((wc, hc), dtype=np.float64)
    shop.customer_simulation.heat_raw = np.zeros((wc, hc), dtype=np.float32)
    # Higher density near (entrance, checkout) decaying with distance.
    ent_i = (int(synthetic.entrance[0] * res), int(synthetic.entrance[1] * res))
    chk_i = (int(synthetic.checkout[0] * res), int(synthetic.checkout[1] * res))
    for ix in range(wc):
        for iy in range(hc):
            x = ix / res; y = iy / res
            d_ent = math.hypot(x - synthetic.entrance[0], y - synthetic.entrance[1])
            d_chk = math.hypot(x - synthetic.checkout[0], y - synthetic.checkout[1])
            heat = 5.0 / (1.0 + d_ent) + 3.0 / (1.0 + d_chk)
            shop.customer_simulation.heat_raw[ix, iy] = heat
    shop.customer_simulation.heat_map_data = (
        shop.customer_simulation.heat_raw.astype(np.float64)
    )
    shop.customer_simulation._floor_heat_raw = {
        1: shop.customer_simulation.heat_raw
    }
    shop.customer_simulation.geometry_dirty = True
    return shop


# ─── base_params for the simulator's MC engine ───────────────────────────

def base_params_for(synthetic: SyntheticShop) -> Dict[str, Any]:
    """Construct the dict ``_ga_fitness`` / ``_mc_engine`` consume from
    the synthetic shop's parameters."""
    mean_rev = float(np.mean([it.base_revenue for it in synthetic.items]))
    return {
        'customers_per_hour': synthetic.daily_customers / 10.0,  # 10h day
        'conversion_rate': 0.30,
        'rev_per_converting_customer': mean_rev,
        'rev_std': max(mean_rev * 0.35, 0.5),
        'impulse_rate': 0.20,
        'avg_impulse_value': max(
            np.mean([it.base_revenue for it in synthetic.items if it.is_impulse])
            if any(it.is_impulse for it in synthetic.items) else mean_rev * 0.15,
            0.5
        ),
        'avg_basket_size': 3.0,
        'std_basket_size': 1.0,
        'basket_sizes_observed': [3] * 30,
        'abandonment_rate': 0.05,
        'avg_queue_time': 5.0,
    }


# ─── Apply a layout to the headless shop ─────────────────────────────────

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


# ─── Paired-MC revenue evaluator ─────────────────────────────────────────

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
    variance."""
    apply_layout(shop, layout)
    chrom = layout_to_chromosome(layout, item_names)
    np.random.seed(seed)   # _mc_engine via _ga_fitness uses np.random
    return float(shop._ga_fitness(chrom, item_names, base_params,
                                  mc_days=mc_days, mc_iters=mc_iters))


# ─── Headless GA loop ────────────────────────────────────────────────────

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
                    ) -> Dict[str, Any]:
    """Run the same GA the GUI does, headlessly, with a fixed seed.

    Mirrors ``GARunMixin._run_ga_optimization`` minus the Tk progress
    chatter. Returns:
        {
          'best_chrom':   (N, 2) array,
          'best_fit':     float,
          'history_best': [float] of length n_gens,
          'history_avg':  [float],
        }
    """
    np.random.seed(rng_seed)

    n_elite = max(1, int(pop_size * elite_frac))
    current = shop._ga_encode(item_names)
    population: List[np.ndarray] = [current.copy()]
    for _ in range(pop_size - 1):
        noisy = current.copy()
        for i, n in enumerate(item_names):
            data = shop._ga_get_item_data(n)
            w, h = data.get('size', (1.0, 1.0))
            bounds = shop._ga_get_section_bounds(n)
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
        population.append(noisy)

    history_best: List[float] = []
    history_avg:  List[float] = []

    for gen in range(n_gens):
        fitness = np.array([
            shop._ga_fitness(p, item_names, base_params, mc_days, mc_iters)
            for p in population
        ])
        rank = np.argsort(fitness)[::-1]
        population = [population[r] for r in rank]
        fitness = fitness[rank]

        history_best.append(float(fitness[0]))
        history_avg.append(float(fitness.mean()))
        if verbose:
            print(f"  gen {gen+1:>3d}/{n_gens}: "
                  f"best={fitness[0]:>10.2f}  avg={fitness.mean():>10.2f}",
                  flush=True)

        new_pop: List[np.ndarray] = [p.copy() for p in population[:n_elite]]
        while len(new_pop) < pop_size:
            t_size = min(5, pop_size)
            t_idx  = np.random.choice(pop_size, t_size, replace=False)
            p1 = population[t_idx[np.argmax(fitness[t_idx])]]
            t_idx2 = np.random.choice(pop_size, t_size, replace=False)
            p2 = population[t_idx2[np.argmax(fitness[t_idx2])]]
            c1, c2, blend = shop._ga_crossover(p1, p2)
            for child in (c1, c2, blend):
                child = shop._ga_mutate(child, item_names, mut_rate)
                child = shop._ga_repair(child, item_names)
                new_pop.append(child)
                if len(new_pop) >= pop_size:
                    break
        population = new_pop[:pop_size]

    # Final evaluation on the surviving population
    final_fit = np.array([
        shop._ga_fitness(p, item_names, base_params, mc_days, mc_iters)
        for p in population
    ])
    best_idx = int(np.argmax(final_fit))
    return {
        'best_chrom':   population[best_idx],
        'best_fit':     float(final_fit[best_idx]),
        'history_best': history_best,
        'history_avg':  history_avg,
    }


# ─── Bootstrap CI ────────────────────────────────────────────────────────

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


# ─── Provenance / sidecar ────────────────────────────────────────────────

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


def elasticity_snapshot() -> Dict[str, Any]:
    return {
        'GA_weights': asdict(GaWeights()),
        'ELASTICITY_CONV_BASE':     ELASTICITY_CONV_BASE,
        'ELASTICITY_CONV_GAIN_MAX': ELASTICITY_CONV_GAIN_MAX,
        'ELASTICITY_IMP_BASE':      ELASTICITY_IMP_BASE,
        'ELASTICITY_IMP_GAIN_MAX':  ELASTICITY_IMP_GAIN_MAX,
        'ELASTICITY_BSK_BASE':      ELASTICITY_BSK_BASE,
        'ELASTICITY_BSK_GAIN_MAX':  ELASTICITY_BSK_GAIN_MAX,
    }


def write_sidecar(out_dir: str, payload: Dict[str, Any]) -> str:
    """Write a JSON sidecar with the merged payload + provenance fields.
    Returns the absolute path."""
    os.makedirs(out_dir, exist_ok=True)
    full = {
        'git_sha':           git_sha(),
        'python':            sys.version.split()[0],
        'platform':          platform.platform(),
        'iso_time':          time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime()),
        'elasticities':      elasticity_snapshot(),
        **payload,
    }
    path = os.path.join(out_dir, 'sidecar.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(full, f, indent=2, default=str)
    return path


def make_run_dir(parent: str, prefix: str) -> str:
    """Create a timestamped subdirectory under ``parent`` for one run.
    Returns the absolute path."""
    stamp = time.strftime('%Y%m%d-%H%M%S')
    out = os.path.join(parent, f'{prefix}_{stamp}')
    os.makedirs(out, exist_ok=True)
    return out
