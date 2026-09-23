"""Equal-budget metaheuristic comparators for Figure B.

The informed baselines do not control for *search budget*:
``random_valid`` etc. are single-shot constructions, so "GA beats them"
only shows that 750 evaluations of the true objective beat one draw.
These two comparators close that gap by searching the SAME Monte Carlo
objective the GA optimizes, under the SAME evaluation budget
(``budget`` = GA ``pop_size * n_gens``):

  * ``random_search``       -- best of ``budget`` random valid layouts
                               (the equal-budget undirected baseline).
  * ``simulated_annealing`` -- Metropolis SA over within-section moves
                               (a peer metaheuristic; if the GA cannot
                               beat plain SA at equal budget, the
                               "genetic algorithm" framing is not earned).

Fairness details, mirroring ``run_ga_headless``:
  * the same starting information: the GA seeds its population from the
    as-built layout, random search scores that layout as its first
    candidate (``init_layout``) and SA starts from it
    (``start='asbuilt'``). SA's former warm start from ``popularity_rank``
    is kept as a named sensitivity variant, ``start='popularity'``;
  * the same feasible set: every candidate (each random draw, SA's start
    and each neighbour) goes through ``feasible_layout``, the GA's own
    repair chain, before it is evaluated;
  * common random numbers in blocks: search evaluation ``k`` uses MC seed
    ``seed*1000 + k // block``, with ``block`` the GA population size, the
    analogue of a GA generation sharing one seed. SA re-scores its
    incumbent whenever the block seed changes, so every Metropolis
    comparison is made under one seed;
  * the same budget: ``budget`` search evaluations (SA's start evaluation
    and incumbent re-scores included), then final selection over ``block``
    candidates x ``n_final_seeds`` seeds, which is the GA's re-evaluation
    of its final population. ``stats`` receives the counts;
  * the same seed namespaces: search seeds ``seed*1000 + [0, n_blocks)``,
    final-selection seeds ``seed*1000 + n_blocks + 1 + s`` for s in
    [0, n_final_seeds), where ``n_blocks`` = GA ``n_gens`` when ``budget``
    is ``n_gens * block``;
  * final selection by re-evaluating the candidates under the shared
    final seeds and taking the best mean --- this removes the
    max-order-statistic ("winner's curse") bias that would otherwise let
    a layout with one lucky low-noise evaluation win and then regress on
    the held-out paired seed.

Both return a plain ``{name: (x, y)}`` layout, scored downstream by the
harness under the shared paired seed just like the GA's output.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from baselines import (random_valid, popularity_rank,
                       _repair_within_section, _section_inner_bounds)
from experiments._common import paired_mc_revenue, feasible_layout

Layout = Dict[str, Tuple[float, float]]

#: Starting points ``simulated_annealing`` accepts. ``'asbuilt'`` is the
#: start every search in the equal-budget comparison shares;
#: ``'popularity'`` is the sensitivity variant.
SA_STARTS = ('asbuilt', 'popularity')

#: Probability of accepting a median worsening move at the starting
#: temperature. 0.8 is the usual warm start for a Metropolis schedule that
#: still has room to quench; ``simulated_annealing`` exposes it so the
#: setting can be swept.
DEFAULT_INITIAL_ACCEPT = 0.8

#: Tag mixed into the neighbour-repair seed. Array seeds go through
#: MT19937's init_by_array, a different initialisation from the plain
#: integer seeding used for the Monte Carlo evaluation seeds, so this
#: stream cannot silently coincide with an evaluation stream.
_NEIGHBOR_REPAIR_TAG = 0x5A0E


def _fit(shop, item_names, layout, base_params, seed, mc_iters, mc_days):
    return paired_mc_revenue(shop, item_names, layout, base_params,
                             seed=seed, mc_iters=mc_iters, mc_days=mc_days)


def _search_seed(seed: int, k: int, block: int) -> int:
    """MC seed of search evaluation ``k``: one shared seed per block."""
    return seed * 1000 + k // block


def _n_blocks(budget: int, block: int) -> int:
    return -(-budget // block)


def _record_counts(stats: Optional[dict], n_search: int, n_final: int) -> None:
    if stats is not None:
        stats.update(n_search_evals=n_search, n_final_evals=n_final,
                     n_evals=n_search + n_final)


def _final_select(shop, item_names, candidates: List[Layout], base_params,
                  base_seed, mc_iters, mc_days, n_final_seeds=5,
                  stats: Optional[dict] = None) -> Layout:
    """Re-evaluate candidate layouts under the shared seeds
    ``base_seed + k`` for k in [0, n_final_seeds); return the layout with
    the best mean. Mirrors the GA's final multi-seed selection and removes
    single-seed winner's-curse bias. Adds the evaluations spent to
    ``stats['n_final_evals']`` when ``stats`` is given."""
    best_layout, best_mean = candidates[0], -np.inf
    for lay in candidates:
        vals = [_fit(shop, item_names, lay, base_params,
                     base_seed + k, mc_iters, mc_days)
                for k in range(n_final_seeds)]
        m = float(np.mean(vals))
        if m > best_mean:
            best_mean, best_layout = m, lay
    if stats is not None:
        stats['n_final_evals'] = (stats.get('n_final_evals', 0)
                                  + len(candidates) * n_final_seeds)
    return best_layout


def random_search(shop, shop_synth, item_names, base_params, *, seed,
                  budget, mc_iters, mc_days, block=30,
                  n_final_seeds=5, stats: Optional[dict] = None,
                  init_layout: Optional[Layout] = None,
                  sampler: Optional[Callable[[np.random.Generator], Layout]]
                  = None) -> Layout:
    """Best of ``budget`` random valid layouts under the MC objective.

    With ``init_layout`` the first candidate is
    ``feasible_layout(init_layout)`` -- the as-built layout the GA seeds its
    population from -- and ``budget - 1`` random draws follow. The search
    then starts from the same information as the GA and SA, and the layout
    it was handed is charged to the budget like any draw rather than
    scored for free. Without ``init_layout`` the search starts cold.

    ``sampler(rng)`` replaces the default draw, ``random_valid`` on
    ``shop_synth`` with its seed taken from ``rng``, the search's own
    ``np.random.Generator``. ``shop_synth`` is read only by the default
    draw, so it may be ``None`` when a sampler is given.

    Each candidate is mapped through ``feasible_layout`` and scored under
    its block's shared seed; the top ``block`` candidates go to the
    multi-seed final selection. ``stats`` receives the evaluation counts
    and the start used (``rs_start``)."""
    if budget < 1:
        raise ValueError("budget must be >= 1 search evaluation")
    if sampler is None:
        if shop_synth is None:
            raise ValueError("random_search needs shop_synth to draw "
                             "random_valid layouts when no sampler is given")

        def sampler(rng: np.random.Generator) -> Layout:
            return random_valid(shop_synth,
                                seed=int(rng.integers(0, 2 ** 31 - 1)))
    rng = np.random.default_rng(seed * 7919 + 13)
    scored: List[Tuple[float, Layout]] = []
    for k in range(budget):
        if k == 0 and init_layout is not None:
            lay = init_layout
        else:
            lay = sampler(rng)
        lay = feasible_layout(shop, item_names, lay)
        f = _fit(shop, item_names, lay, base_params,
                 _search_seed(seed, k, block), mc_iters, mc_days)
        scored.append((f, lay))
    scored.sort(key=lambda t: t[0], reverse=True)
    top = [lay for _, lay in scored[:block]]
    final_counts: dict = {}
    best = _final_select(shop, item_names, top, base_params,
                         seed * 1000 + _n_blocks(budget, block) + 1,
                         mc_iters, mc_days, n_final_seeds,
                         stats=final_counts)
    _record_counts(stats, budget, final_counts.get('n_final_evals', 0))
    if stats is not None:
        stats['rs_start'] = 'asbuilt' if init_layout is not None else 'cold'
    return best


def _neighbor(shop_synth, layout: Layout, rng, step_frac=0.25) -> Layout:
    """Perturb one random item's position within its section bounds, then
    spread any within-section overlap. Callers map the result through
    ``feasible_layout`` before evaluating it."""
    lay = dict(layout)
    it = shop_synth.items[int(rng.integers(0, len(shop_synth.items)))]
    lo_x, lo_y, hi_x, hi_y = _section_inner_bounds(shop_synth, it)
    sx, sy = max(hi_x - lo_x, 1e-3), max(hi_y - lo_y, 1e-3)
    x, y = lay[it.name]
    nx = float(np.clip(x + rng.normal(0, step_frac * sx), lo_x, hi_x))
    ny = float(np.clip(y + rng.normal(0, step_frac * sy), lo_y, hi_y))
    lay[it.name] = (nx, ny)
    np.random.seed([int(rng.integers(0, 2 ** 31 - 1)), _NEIGHBOR_REPAIR_TAG])
    return _repair_within_section(shop_synth, lay)


def simulated_annealing(shop, shop_synth, item_names, base_params, *, seed,
                        budget, mc_iters, mc_days, block=30,
                        n_final_seeds=5,
                        initial_accept: float = DEFAULT_INITIAL_ACCEPT,
                        stats: Optional[dict] = None,
                        init_layout: Optional[Layout] = None,
                        start: str = 'asbuilt',
                        neighbor: Optional[Callable[
                            [Layout, np.random.Generator], Layout]] = None
                        ) -> Layout:
    """Metropolis SA on the MC objective with ``budget`` search evaluations.

    By default (``start='asbuilt'``) the annealer starts from
    ``feasible_layout(init_layout)``: the as-built layout, the one the GA
    seeds its population from and random search scores first. In a
    budget-parity comparison the starting point is information a method
    receives before it spends any budget. A warm start from a constructive
    heuristic hands the annealer, for free, what that heuristic already
    encodes -- for ``popularity_rank``, the revenue ranking of the
    assortment -- so a gap between two searches would mix what each did
    with its budget with how good a seed it was given. With one shared
    start and one budget, what each search does with that budget is the
    only thing that differs between them.

    ``start='popularity'`` keeps the former warm start,
    ``feasible_layout(popularity_rank(shop_synth, seed=seed))``, as a named
    sensitivity variant: set beside the as-built run it shows how much of
    the annealer's result the warm start was carrying. ``init_layout`` is
    not read in that case.

    ``neighbor(layout, rng)`` replaces the default move (``_neighbor``: one
    item perturbed within its section of ``shop_synth``); ``rng`` is the
    annealer's own ``np.random.Generator``. ``shop_synth`` is read only by
    the default move and the popularity start, so it may be ``None`` when
    a neighbour is given and the start is as-built.

    The temperature is calibrated to the MOVES, not to the objective's
    level: the Metropolis test compares T against ``dE``, the revenue
    difference of a one-item move, which is orders of magnitude smaller
    than the revenue itself, so a T taken as a fraction of the starting
    revenue accepts nearly every worsening move for the whole budget and
    the search degenerates into a random walk. Instead the first block
    accepts every proposal and records what the worsening ones cost, then
    ``T0 = median(|dE|) / ln(1 / initial_accept)`` puts the acceptance
    probability of a median worsening move at ``initial_accept``.
    Geometric cooling takes T to 1% of ``T0`` over the evaluations left
    after that block, so the run ends in a quench phase. The calibration
    evaluations are charged to ``budget`` like any other, so the total
    stays equal to the GA's.

    At the first evaluation of each block the incumbent is re-scored under
    the new seed, so the candidate and the incumbent in every Metropolis
    test share a seed. Every evaluated state enters an archive scored by
    the mean of its evaluations; the best ``block`` distinct states go to
    the multi-seed final selection. ``stats`` receives the evaluation
    counts, the schedule (``sa_T0``, ``sa_initial_accept``) and the start
    used (``sa_start``)."""
    if budget < 1:
        raise ValueError("budget must be >= 1 search evaluation")
    if block < 2:
        raise ValueError("block must be >= 2: the incumbent is re-scored at "
                         "the start of every block before any proposal")
    if not 0.0 < initial_accept < 1.0:
        raise ValueError("initial_accept must lie strictly between 0 and 1: "
                         "it is the probability of accepting a median "
                         "worsening move at T0")
    if start not in SA_STARTS:
        raise ValueError(f"start must be one of {SA_STARTS}, got {start!r}")
    if start == 'asbuilt' and init_layout is None:
        raise ValueError("start='asbuilt' needs init_layout, the as-built "
                         "layout every search in the comparison starts from")
    if start == 'popularity' and shop_synth is None:
        raise ValueError("start='popularity' needs shop_synth to build the "
                         "popularity_rank layout")
    if neighbor is None:
        if shop_synth is None:
            raise ValueError("simulated_annealing needs shop_synth for its "
                             "default within-section move when no neighbor "
                             "is given")

        def neighbor(layout: Layout, rng: np.random.Generator) -> Layout:
            return _neighbor(shop_synth, layout, rng)
    rng = np.random.default_rng(seed * 104729 + 7)
    archive: Dict[tuple, list] = {}   # state -> [sum of fits, count, layout]

    def _evaluate(lay: Layout, k: int) -> float:
        f = _fit(shop, item_names, lay, base_params,
                 _search_seed(seed, k, block), mc_iters, mc_days)
        entry = archive.setdefault(tuple(lay[n] for n in item_names),
                                   [0.0, 0, lay])
        entry[0] += f
        entry[1] += 1
        return f

    if start == 'asbuilt':
        cur = feasible_layout(shop, item_names, init_layout)
    else:
        cur = feasible_layout(shop, item_names,
                              popularity_rank(shop_synth, seed=seed))
    cur_f = _evaluate(cur, 0)
    k, cur_block = 1, 0
    # Proposals up to and including ``k_cal`` -- the first block -- are the
    # calibration phase: accepted unconditionally, and the worsening ones
    # measured to set T0.
    k_cal = min(block - 1, budget - 1)
    worsening: List[float] = []
    T0: Optional[float] = None
    cool = 0.01 ** (1.0 / max(budget - 1 - k_cal, 1))  # T0 -> 0.01*T0
    while k < budget:
        if k // block != cur_block:
            cur_block = k // block
            cur_f = _evaluate(cur, k)
            k += 1
            continue
        cand = feasible_layout(shop, item_names, neighbor(cur, rng))
        cand_f = _evaluate(cand, k)
        dE = cand_f - cur_f
        if k <= k_cal:
            if dE < 0.0:
                worsening.append(-dE)
            accept = True
        else:
            if T0 is None:
                if worsening:
                    T0 = max(float(np.median(worsening))
                             / float(np.log(1.0 / initial_accept)), 1e-9)
                else:
                    # Nothing worsened during calibration; a unit
                    # temperature keeps the schedule well-defined.
                    T0 = 1.0
            # Temperature follows the evaluations spent, so the incumbent
            # re-scores do not stretch the schedule past the budget.
            T = T0 * cool ** (k - 1 - k_cal)
            accept = dE >= 0.0 or rng.random() < np.exp(dE / max(T, 1e-9))
        k += 1
        if accept:
            cur, cur_f = cand, cand_f

    ranked = sorted(archive.values(), key=lambda e: e[0] / e[1], reverse=True)
    pool = [e[2] for e in ranked[:block]]
    final_counts: dict = {}
    best = _final_select(shop, item_names, pool, base_params,
                         seed * 1000 + _n_blocks(budget, block) + 1,
                         mc_iters, mc_days, n_final_seeds,
                         stats=final_counts)
    if stats is not None:
        # The schedule the run actually used, so a reader can check the
        # comparator was annealing rather than drifting.
        stats.update(sa_T0=(float(T0) if T0 is not None else None),
                     sa_initial_accept=float(initial_accept),
                     sa_start=start)
    _record_counts(stats, k, final_counts.get('n_final_evals', 0))
    return best
