"""Layout feasibility regression tests.

Guards the constraint machinery every optimizer relies on: baselines and
the equal-budget metaheuristics must return layouts that place every item
inside its section with no within-section overlaps.
"""

import numpy as np
import pytest

import oracle
from synthetic_shops import (generate_synthetic_shop, analytical_revenue,
                             grid_layout_within_sections)
from baselines import (random_valid, perimeter_only, popularity_rank,
                       greedy_swap, assert_layout_valid,
                       assert_no_strict_overlap)
from oracle import (solve_oracle, _separate_contacts, _bounds_for,
                    _layout_to_vec, _vec_to_layout)
from experiments._common import (build_headless_shop, base_params_for,
                                 anchor_base_params,
                                 feasible_layout, run_ga_headless,
                                 paired_mc_revenue)
from experiments.metaheuristics import random_search, simulated_annealing


def _shop(seed=1, n_items=8):
    return generate_synthetic_shop(name=f'feas_{seed}', seed=40_000 + seed,
                                   n_items=n_items, width=12.0, height=10.0)


def _as_built(shop, names):
    """The as-built layout every search in the comparison starts from."""
    return {n: tuple(shop.floors[1]['items'][n]['position']) for n in names}


def test_constructive_baselines_feasible():
    ss = _shop()
    names = {it.name for it in ss.items}
    for fn in (random_valid, perimeter_only, popularity_rank, greedy_swap):
        lay = fn(ss, seed=0)
        assert set(lay.keys()) == names, fn.__name__
        assert_layout_valid(ss, lay)          # raises on infeasibility


def test_metaheuristics_feasible_and_complete():
    ss = _shop(seed=2)
    shop = build_headless_shop(ss)
    names = [it.name for it in ss.items]
    init = _as_built(shop, names)
    bp = anchor_base_params(shop, names, base_params_for(ss), init)
    for fn in (random_search, simulated_annealing):
        lay = fn(shop, ss, names, bp, seed=0, budget=12, block=4,
                 mc_iters=60, mc_days=10, init_layout=init)
        assert set(lay.keys()) == set(names), fn.__name__
        assert_layout_valid(ss, lay)
        assert_no_strict_overlap(ss, lay)
        # Candidates are searched inside the GA's feasible set.
        assert feasible_layout(shop, names, lay) == lay, fn.__name__


def test_equal_evaluation_budgets():
    """GA, random search and SA spend the same number of fitness
    evaluations: n_gens*pop search + n_final_seeds*pop final selection.
    The reported counts are checked against the calls actually made."""
    ss = _shop(seed=3)
    shop = build_headless_shop(ss)
    names = [it.name for it in ss.items]
    init = _as_built(shop, names)
    bp = anchor_base_params(shop, names, base_params_for(ss), init)
    pop, gens = 4, 3

    calls = []
    real_fitness = shop._ga_fitness

    def _counting_fitness(*args, **kwargs):
        calls.append(1)
        return real_fitness(*args, **kwargs)

    shop._ga_fitness = _counting_fitness

    ga = run_ga_headless(shop, names, bp, pop_size=pop, n_gens=gens,
                         mc_iters=40, mc_days=5, rng_seed=0)
    counts = {'GA': ga['n_evals']}
    assert ga['n_evals'] == len(calls)
    assert ga['n_search_evals'] == pop * gens
    for fn in (random_search, simulated_annealing):
        del calls[:]
        stats = {}
        fn(shop, ss, names, bp, seed=0, budget=pop * gens, block=pop,
           mc_iters=40, mc_days=5, stats=stats, init_layout=init)
        assert stats['n_evals'] == len(calls), fn.__name__
        assert stats['n_search_evals'] == pop * gens, fn.__name__
        counts[fn.__name__] = stats['n_evals']
    assert len(set(counts.values())) == 1, counts


def test_sa_temperature_is_scaled_to_move_differences():
    """SA's Metropolis temperature must sit on the scale of the one-move
    revenue differences it is compared against, not on the scale of the
    revenue itself -- at the revenue scale every worsening move is accepted
    for the whole budget and the comparator stops annealing."""
    ss = _shop(seed=4)
    shop = build_headless_shop(ss)
    names = [it.name for it in ss.items]
    init = _as_built(shop, names)
    bp = anchor_base_params(shop, names, base_params_for(ss), init)
    mc = dict(mc_iters=60, mc_days=10, init_layout=init)

    stats = {}
    simulated_annealing(shop, ss, names, bp, seed=0, budget=24, block=6,
                        n_final_seeds=2, stats=stats, **mc)
    # The revenue level at the start the annealer ran from.
    level = paired_mc_revenue(shop, names, feasible_layout(shop, names, init),
                              bp, seed=0, mc_iters=mc['mc_iters'],
                              mc_days=mc['mc_days'])
    assert stats['sa_initial_accept'] == 0.8
    assert stats['sa_T0'] is not None
    assert 0.0 < stats['sa_T0'] < 0.01 * abs(level)

    # A higher acceptance target must give a hotter start.
    hot = {}
    simulated_annealing(shop, ss, names, bp, seed=0, budget=24, block=6,
                        n_final_seeds=2, initial_accept=0.95, stats=hot, **mc)
    assert hot['sa_T0'] > stats['sa_T0']

    with pytest.raises(ValueError):
        simulated_annealing(shop, ss, names, bp, seed=0, budget=24, block=6,
                            initial_accept=1.0, **mc)


def test_oracle_reports_unresolvable_overlap():
    """When separation cannot clear a section, the reference solution is
    reported as not converged instead of being handed on as clean: the
    harness repair would otherwise grid-snap it and score a layout the
    solver never proposed."""
    ss = generate_synthetic_shop(name='synth_000', seed=10_000, n_items=10,
                                 width=12.0, height=10.0)
    saved = oracle._separate_contacts
    # The raw solver output of this scenario leaves a flush contact, so
    # skipping separation exercises the postcondition.
    oracle._separate_contacts = (
        lambda shop, names, vec, bounds, max_passes=20:
        np.array(vec, dtype=np.float64, copy=True))
    try:
        res = oracle.solve_oracle(ss, n_restarts=24)
    finally:
        oracle._separate_contacts = saved
    assert not res.converged
    assert res.note.startswith('residual overlap after separation')


def test_oracle_layout_has_no_strict_overlap():
    """The simulator fitness penalizes any positive overlap, so the
    analytical reference must leave no contact residue. With the current
    section geometry the raw solver output of scenarios 0, 23 and 27
    overlaps (1e-9 m contacts and a 2 mm overlap), so those exercise the
    separation; 12 and 25 are the paper's reference cases."""
    for idx in (0, 12, 23, 25, 27):
        ss = generate_synthetic_shop(name=f'synth_{idx:03d}',
                                     seed=10_000 + idx, n_items=10,
                                     width=12.0, height=10.0)
        res = solve_oracle(ss, n_restarts=24)
        assert_no_strict_overlap(ss, res.layout)
        assert_layout_valid(ss, res.layout)
        assert res.true_revenue == analytical_revenue(ss, res.layout)
        assert res.converged and res.note == ''


def test_separate_contacts_clears_residue():
    """Flush contact, a pair blocked by its bound, and a pair that can only
    separate on the other axis all end with zero strict overlap."""
    ss = generate_synthetic_shop(name='contact', seed=10_012, n_items=10,
                                 width=12.0, height=10.0)
    names = [it.name for it in ss.items]
    bounds = _bounds_for(ss, names)
    a, b = 'BEV_00', 'BEV_01'
    sx, sy, _, _ = ss.sections['Beverages']
    w = ss.item_by_name(a).size[0]
    hi_x = bounds[2 * names.index(b)][1]
    cases = [
        ((sx + 0.05, sy + 0.3), (sx + 0.05 + w - 2e-8, sy + 0.3)),
        ((hi_x - 0.9, sy + 0.3), (hi_x, sy + 0.3)),
        ((sx + 0.05, sy + 0.3), (sx + 0.05 + 1e-8, sy + 0.3)),
    ]
    for pa, pb in cases:
        lay = dict(grid_layout_within_sections(ss))
        lay[a], lay[b] = pa, pb
        vec = _separate_contacts(ss, names, _layout_to_vec(lay, names),
                                 np.asarray(bounds))
        out = _vec_to_layout(vec, names)
        assert_no_strict_overlap(ss, out)
        assert_layout_valid(ss, out, tol=0.0)
