"""Equal-budget metaheuristic comparators for Figure B (audit R4.1, R4.2).

Reviewer 4 (optimization/OR) correctly objected that the informed
baselines do not control for *search budget*: ``random_valid`` etc. are
single-shot constructions, so "GA beats them" only shows that 750
evaluations of the true objective beat one draw. These two comparators
close that gap by searching the SAME Monte Carlo objective the GA
optimizes, under the SAME evaluation budget (``budget`` = GA
``pop_size * n_gens``):

  * ``random_search``       -- best of ``budget`` random valid layouts
                               (R4.1: the honest equal-budget baseline).
  * ``simulated_annealing`` -- Metropolis SA over within-section moves
                               (R4.2: a peer metaheuristic; if the GA
                               cannot beat plain SA at equal budget, the
                               "genetic algorithm" framing is not earned).

Fairness details, mirroring ``run_ga_headless`` exactly:
  * a MOVING common-random-number seed during search (``seed*1000 + i``),
    so no single noise realization is exploited;
  * final selection by re-evaluating the top candidates under
    ``n_final_seeds`` fresh shared seeds and taking the best mean --- this
    removes the max-order-statistic ("winner's curse") bias that would
    otherwise let a layout with one lucky low-noise evaluation win and
    then regress on the held-out paired seed.

Both return a plain ``{name: (x, y)}`` layout, scored downstream by the
harness under the shared paired seed just like the GA's output.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from baselines import (random_valid, popularity_rank,
                       _repair_within_section, _section_inner_bounds)
from experiments._common import paired_mc_revenue

Layout = Dict[str, Tuple[float, float]]


def _fit(shop, item_names, layout, base_params, seed, mc_iters, mc_days):
    return paired_mc_revenue(shop, item_names, layout, base_params,
                             seed=seed, mc_iters=mc_iters, mc_days=mc_days)


def _final_select(shop, item_names, candidates: List[Layout], base_params,
                  base_seed, mc_iters, mc_days, n_final_seeds=5) -> Layout:
    """Re-evaluate candidate layouts under ``n_final_seeds`` fresh shared
    seeds; return the layout with the best mean. Mirrors the GA's final
    multi-seed selection and removes single-seed winner's-curse bias."""
    best_layout, best_mean = candidates[0], -np.inf
    for lay in candidates:
        vals = [_fit(shop, item_names, lay, base_params,
                     base_seed + 100 + k, mc_iters, mc_days)
                for k in range(n_final_seeds)]
        m = float(np.mean(vals))
        if m > best_mean:
            best_mean, best_layout = m, lay
    return best_layout


def random_search(shop, shop_synth, item_names, base_params, *, seed,
                  budget, mc_iters, mc_days, keep_top=None,
                  n_final_seeds=5) -> Layout:
    """Best of ``budget`` random valid layouts under the MC objective,
    with a moving CRN seed per draw and a multi-seed final selection over
    the top ``keep_top`` candidates (default ~ a GA population)."""
    rng = np.random.default_rng(seed * 7919 + 13)
    keep_top = keep_top or max(5, budget // 25)
    scored: List[Tuple[float, Layout]] = []
    for i in range(budget):
        lay = random_valid(shop_synth, seed=int(rng.integers(0, 2 ** 31 - 1)))
        f = _fit(shop, item_names, lay, base_params,
                 seed * 1000 + i, mc_iters, mc_days)
        scored.append((f, lay))
    scored.sort(key=lambda t: t[0], reverse=True)
    top = [lay for _, lay in scored[:keep_top]]
    return _final_select(shop, item_names, top, base_params,
                         seed * 1000 + budget, mc_iters, mc_days,
                         n_final_seeds)


def _neighbor(shop_synth, layout: Layout, rng, step_frac=0.25) -> Layout:
    """Perturb one random item's position within its section bounds, then
    repair overlaps (same feasibility rules the GA repair enforces)."""
    lay = dict(layout)
    it = shop_synth.items[int(rng.integers(0, len(shop_synth.items)))]
    lo_x, lo_y, hi_x, hi_y = _section_inner_bounds(shop_synth, it)
    sx, sy = max(hi_x - lo_x, 1e-3), max(hi_y - lo_y, 1e-3)
    x, y = lay[it.name]
    nx = float(np.clip(x + rng.normal(0, step_frac * sx), lo_x, hi_x))
    ny = float(np.clip(y + rng.normal(0, step_frac * sy), lo_y, hi_y))
    lay[it.name] = (nx, ny)
    np.random.seed(int(rng.integers(0, 2 ** 31 - 1)))
    return _repair_within_section(shop_synth, lay)


def simulated_annealing(shop, shop_synth, item_names, base_params, *, seed,
                        budget, mc_iters, mc_days, n_final_seeds=5) -> Layout:
    """Metropolis SA on the MC objective, ``budget`` proposal evaluations,
    geometric cooling from ``T0`` (5% of the warm-start revenue) to 1% of
    ``T0``. Warm-started from ``popularity_rank`` (same start greedy_swap
    uses) so SA and greedy_swap probe the same basin from equal footing."""
    rng = np.random.default_rng(seed * 104729 + 7)
    cur = popularity_rank(shop_synth, seed=seed)
    cur_f = _fit(shop, item_names, cur, base_params,
                 seed * 1000 + 0, mc_iters, mc_days)
    best, best_f = cur, cur_f
    T0 = max(abs(cur_f) * 0.05, 1.0)
    cool = (0.01) ** (1.0 / max(budget - 1, 1))   # T0 -> 0.01*T0 over budget
    T = T0
    for i in range(1, budget):
        cand = _neighbor(shop_synth, cur, rng)
        cand_f = _fit(shop, item_names, cand, base_params,
                      seed * 1000 + i, mc_iters, mc_days)
        dE = cand_f - cur_f
        if dE >= 0 or rng.random() < np.exp(dE / max(T, 1e-9)):
            cur, cur_f = cand, cand_f
            if cur_f > best_f:
                best, best_f = cur, cur_f
        T *= cool
    return _final_select(shop, item_names, [best, cur], base_params,
                         seed * 1000 + budget, mc_iters, mc_days,
                         n_final_seeds)
