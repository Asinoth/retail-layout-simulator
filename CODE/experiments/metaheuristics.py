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
    and incumbent re-scores included), then final selection over exactly
    ``block`` candidates x ``n_final_seeds`` seeds -- the number the GA
    spends re-evaluating its final population of ``block`` layouts under
    the same selection seeds. ``stats`` receives the counts;
  * the same seed namespaces: search seeds ``seed*1000 + [0, n_blocks)``,
    final-selection seeds ``seed*1000 + n_blocks + 1 + s`` for s in
    [0, n_final_seeds), where ``n_blocks`` = GA ``n_gens`` when ``budget``
    is ``n_gens * block``;
  * candidates for the final selection are ranked only WITHIN the block
    they were scored in. A block's seed shifts every layout scored under it
    by more than the layouts differ from each other, so ranking raw scores
    across blocks prefers whatever drew a kind seed. Random search ranks
    each candidate by its standing in its own block (its score over the
    block's median score, both under the block's one seed; the blocks are
    exchangeable draws, so the median carries the seed and nothing else);
    the annealer's blocks are not exchangeable -- later ones sit in the
    cooled, better part of the walk -- so it fills its pool rank by rank,
    every block's best state first, the latest blocks first within a rank
    (``_pool_by_block_rank``);
  * final selection by re-evaluating the candidates under the shared
    final seeds and taking the best mean --- this removes the
    max-order-statistic ("winner's curse") bias that would otherwise let
    a layout with one lucky low-noise evaluation win and then regress on
    the held-out paired seed.

The GA's pool, for comparison (``run_ga_headless``): its final population,
which is the ``n_elite`` best layouts of the last generation (ranked under
that generation's one seed) plus ``block - n_elite`` children bred from it
that no search evaluation has scored.

Both return a plain ``{name: (x, y)}`` layout, scored downstream by the
harness under the shared paired seed just like the GA's output, and put a
per-block convergence trace in ``stats['trace']``.
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

#: Fraction of T0 the annealer's geometric schedule reaches at its last
#: evaluation, so the run ends in a quench phase. ``simulated_annealing``
#: exposes it so the setting can be swept.
DEFAULT_FINAL_TEMP_FRAC = 0.01

#: Step of the default within-section move, as a fraction of the room the
#: item's section gives it on each axis (``_neighbor``).
DEFAULT_STEP_FRAC = 0.25


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


def _key(layout: Layout, item_names) -> tuple:
    return tuple(layout[n] for n in item_names)


def _pad_pool(pool: List[Layout], size: int) -> List[Layout]:
    """``pool`` cycled from its head up to ``size`` candidates.

    The final selection re-scores exactly as many candidates as the GA
    re-scores layouts in its final population, whatever the search found:
    a search that visited fewer distinct layouts than that (only possible
    with a tiny budget or a move that rarely moves) repeats its best ones,
    as the GA's population repeats its elites. A repeat scores what its
    original scored, so it cannot change the choice."""
    if not pool:
        raise ValueError("no candidate layouts for the final selection")
    out = list(pool)
    k = 0
    while len(out) < size:
        out.append(pool[k % len(pool)])
        k += 1
    return out[:size]


def _final_select(shop, item_names, candidates: List[Layout], base_params,
                  base_seed, mc_iters, mc_days, n_final_seeds=5,
                  stats: Optional[dict] = None,
                  means_out: Optional[List[float]] = None) -> Layout:
    """Re-evaluate candidate layouts under the shared seeds
    ``base_seed + k`` for k in [0, n_final_seeds); return the layout with
    the best mean (the first of equal means). Mirrors the GA's final
    multi-seed selection and removes single-seed winner's-curse bias. Adds
    the evaluations spent to ``stats['n_final_evals']`` when ``stats`` is
    given; ``means_out`` receives each candidate's mean, in order."""
    best_layout, best_mean = candidates[0], -np.inf
    for lay in candidates:
        vals = [_fit(shop, item_names, lay, base_params,
                     base_seed + k, mc_iters, mc_days)
                for k in range(n_final_seeds)]
        m = float(np.mean(vals))
        if means_out is not None:
            means_out.append(m)
        if m > best_mean:
            best_mean, best_layout = m, lay
    if stats is not None:
        stats['n_final_evals'] = (stats.get('n_final_evals', 0)
                                  + len(candidates) * n_final_seeds)
    return best_layout


def _standing(scores: List[float], blocks: List[int]) -> List[float]:
    """Each score over the median score of its block (the difference when
    that median is not positive): its standing among the candidates scored
    under the same seed, comparable across blocks whose candidates are
    exchangeable draws."""
    groups: Dict[int, List[float]] = {}
    for s, b in zip(scores, blocks):
        groups.setdefault(b, []).append(s)
    med = {b: float(np.median(v)) for b, v in groups.items()}
    return [s / med[b] if med[b] > 0 else s - med[b]
            for s, b in zip(scores, blocks)]


def _pool_by_block_rank(entries, size: int) -> Tuple[List[Layout], List[int]]:
    """Pool of up to ``size`` distinct layouts, compared only within the
    block they were scored in.

    ``entries`` holds ``(block, key, score, layout, k)`` per evaluation. In
    each block the distinct states are ranked by their score under that
    block's seed (the mean of their evaluations in the block, the earliest
    first on ties). The pool takes every block's rank-1 state, the latest
    block first, then every block's rank-2 state, and so on: no state is
    preferred for the seed its block drew, and within a rank the latest
    (coolest) part of an annealing walk comes first. Returns the pool and
    the block each member came from."""
    by_block: Dict[int, Dict[tuple, list]] = {}
    for b, key, f, lay, k in entries:
        e = by_block.setdefault(b, {}).setdefault(key, [0.0, 0, lay, k, key])
        e[0] += f
        e[1] += 1
    ranked = {b: sorted(d.values(), key=lambda e: (-(e[0] / e[1]), e[3]))
              for b, d in by_block.items()}
    order = sorted(ranked, reverse=True)
    pool: List[Layout] = []
    src: List[int] = []
    seen = set()
    depth = max((len(v) for v in ranked.values()), default=0)
    for r in range(depth):
        for b in order:
            if len(pool) >= size:
                return pool, src
            if r < len(ranked[b]):
                e = ranked[b][r]
                if e[4] not in seen:
                    seen.add(e[4])
                    pool.append(e[2])
                    src.append(b)
    return pool, src


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
    its block's shared seed. The ``block`` distinct candidates with the
    highest standing in their own block (score over the block's median,
    ``_standing``) go to the multi-seed final selection -- exactly
    ``block`` of them, the number the GA re-scores. ``stats`` receives the
    evaluation counts, the start used (``rs_start``), the pool rule and the
    per-block trace (``trace``: each block's seed, size, best and median
    score and best standing)."""
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
    scores: List[float] = []
    blocks: List[int] = []
    layouts: List[Layout] = []
    for k in range(budget):
        if k == 0 and init_layout is not None:
            lay = init_layout
        else:
            lay = sampler(rng)
        lay = feasible_layout(shop, item_names, lay)
        f = _fit(shop, item_names, lay, base_params,
                 _search_seed(seed, k, block), mc_iters, mc_days)
        scores.append(f)
        blocks.append(k // block)
        layouts.append(lay)
    stand = _standing(scores, blocks)
    order = sorted(range(budget), key=lambda k: (-stand[k], k))
    top: List[Layout] = []
    seen = set()
    for k in order:
        key = _key(layouts[k], item_names)
        if key in seen:
            continue
        seen.add(key)
        top.append(layouts[k])
        if len(top) == block:
            break
    final_counts: dict = {}
    best = _final_select(shop, item_names, _pad_pool(top, block), base_params,
                         seed * 1000 + _n_blocks(budget, block) + 1,
                         mc_iters, mc_days, n_final_seeds,
                         stats=final_counts)
    _record_counts(stats, budget, final_counts.get('n_final_evals', 0))
    if stats is not None:
        stats['rs_start'] = 'asbuilt' if init_layout is not None else 'cold'
        stats['pool_rule'] = 'within-block standing (score / block median)'
        stats['pool_distinct'] = len(top)
        trace = []
        by_block: Dict[int, List[int]] = {}
        for k, b in enumerate(blocks):
            by_block.setdefault(b, []).append(k)
        for b in sorted(by_block):
            ks = by_block[b]
            sb = [scores[k] for k in ks]
            trace.append({'block': b,
                          'seed': _search_seed(seed, ks[0], block),
                          'n': len(ks), 'best': float(max(sb)),
                          'median': float(np.median(sb)),
                          'best_standing': float(max(stand[k] for k in ks))})
        stats['trace'] = trace
    return best


def _neighbor(shop_synth, layout: Layout, rng,
              step_frac: float = DEFAULT_STEP_FRAC) -> Layout:
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
                        final_temp_frac: float = DEFAULT_FINAL_TEMP_FRAC,
                        step_frac: float = DEFAULT_STEP_FRAC,
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
    item perturbed within its section of ``shop_synth`` by a normal step of
    SD ``step_frac`` of the section's room); ``rng`` is the annealer's own
    ``np.random.Generator``. ``shop_synth`` is read only by the default
    move and the popularity start, so it may be ``None`` when a neighbour
    is given and the start is as-built. ``step_frac`` is read only by the
    default move: a caller's ``neighbor`` carries its own step.

    The temperature is calibrated to the MOVES, not to the objective's
    level: the Metropolis test compares T against ``dE``, the revenue
    difference of a one-item move, which is orders of magnitude smaller
    than the revenue itself, so a T taken as a fraction of the starting
    revenue accepts nearly every worsening move for the whole budget and
    the search degenerates into a random walk. Instead the first block
    accepts every proposal and records what the worsening ones cost, then
    ``T0 = median(|dE|) / ln(1 / initial_accept)`` puts the acceptance
    probability of a median worsening move at ``initial_accept``.
    Geometric cooling takes T to ``final_temp_frac`` of ``T0`` over the
    evaluations left after that block, so the run ends in a quench phase.
    The calibration evaluations are charged to ``budget`` like any other,
    so the total stays equal to the GA's.

    At the first evaluation of each block the incumbent is re-scored under
    the new seed, so the candidate and the incumbent in every Metropolis
    test share a seed. The final selection re-scores exactly ``block``
    distinct evaluated states, compared only within the block they were
    scored in (``_pool_by_block_rank``). ``stats`` receives the evaluation
    counts, the schedule (``sa_T0``, ``sa_initial_accept``,
    ``sa_final_temp_frac``), the start used (``sa_start``), the pool rule
    and the per-block trace (``trace``: each block's seed, the incumbent's
    score as the block opened, the best and median score in it, the
    proposals made and accepted, and the temperature it ended at)."""
    if budget < 1:
        raise ValueError("budget must be >= 1 search evaluation")
    if block < 2:
        raise ValueError("block must be >= 2: the incumbent is re-scored at "
                         "the start of every block before any proposal")
    if not 0.0 < initial_accept < 1.0:
        raise ValueError("initial_accept must lie strictly between 0 and 1: "
                         "it is the probability of accepting a median "
                         "worsening move at T0")
    if not 0.0 < final_temp_frac <= 1.0:
        raise ValueError("final_temp_frac must lie in (0, 1]: it is the "
                         "fraction of T0 the schedule ends at")
    if not step_frac > 0.0:
        raise ValueError("step_frac must be positive")
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
            return _neighbor(shop_synth, layout, rng, step_frac=step_frac)
    rng = np.random.default_rng(seed * 104729 + 7)
    entries: List[tuple] = []      # (block, key, score, layout, k)
    trace: Dict[int, dict] = {}

    def _evaluate(lay: Layout, k: int) -> float:
        f = _fit(shop, item_names, lay, base_params,
                 _search_seed(seed, k, block), mc_iters, mc_days)
        entries.append((k // block, _key(lay, item_names), f, lay, k))
        return f

    if start == 'asbuilt':
        cur = feasible_layout(shop, item_names, init_layout)
    else:
        cur = feasible_layout(shop, item_names,
                              popularity_rank(shop_synth, seed=seed))
    cur_f = _evaluate(cur, 0)
    trace[0] = {'block': 0, 'seed': _search_seed(seed, 0, block),
                'incumbent_start': float(cur_f), 'proposals': 0,
                'accepted': 0, 'T_end': None}
    k, cur_block = 1, 0
    # Proposals up to and including ``k_cal`` -- the first block -- are the
    # calibration phase: accepted unconditionally, and the worsening ones
    # measured to set T0.
    k_cal = min(block - 1, budget - 1)
    worsening: List[float] = []
    T0: Optional[float] = None
    T: Optional[float] = None
    cool = final_temp_frac ** (1.0 / max(budget - 1 - k_cal, 1))
    while k < budget:
        if k // block != cur_block:
            cur_block = k // block
            cur_f = _evaluate(cur, k)
            trace[cur_block] = {'block': cur_block,
                                'seed': _search_seed(seed, k, block),
                                'incumbent_start': float(cur_f),
                                'proposals': 0, 'accepted': 0, 'T_end': T}
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
        tr = trace[cur_block]
        tr['proposals'] += 1
        tr['accepted'] += int(bool(accept))
        tr['T_end'] = T
        k += 1
        if accept:
            cur, cur_f = cand, cand_f

    pool, pool_blocks = _pool_by_block_rank(entries, block)
    final_counts: dict = {}
    best = _final_select(shop, item_names, _pad_pool(pool, block), base_params,
                         seed * 1000 + _n_blocks(budget, block) + 1,
                         mc_iters, mc_days, n_final_seeds,
                         stats=final_counts)
    if stats is not None:
        # The schedule the run actually used, so a reader can check the
        # comparator was annealing rather than drifting.
        stats.update(sa_T0=(float(T0) if T0 is not None else None),
                     sa_initial_accept=float(initial_accept),
                     sa_final_temp_frac=float(final_temp_frac),
                     sa_start=start,
                     pool_rule='within-block rank, latest block first',
                     pool_distinct=len(pool),
                     pool_blocks=pool_blocks)
        out_trace = []
        scores_by_block: Dict[int, List[float]] = {}
        for e in entries:
            scores_by_block.setdefault(e[0], []).append(e[2])
        for b in sorted(trace):
            sb = scores_by_block[b]
            out_trace.append({**trace[b], 'n': len(sb),
                              'best': float(max(sb)),
                              'median': float(np.median(sb))})
        stats['trace'] = out_trace
    _record_counts(stats, k, final_counts.get('n_final_evals', 0))
    return best

