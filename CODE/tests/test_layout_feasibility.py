"""Layout feasibility regression tests (audit R5.3).

Guards the constraint machinery every optimizer relies on: baselines and
the equal-budget metaheuristics must return layouts that place every item
inside its section with no within-section overlaps.
"""

from synthetic_shops import generate_synthetic_shop
from baselines import (random_valid, perimeter_only, popularity_rank,
                       greedy_swap, assert_layout_valid)
from experiments._common import build_headless_shop, base_params_for
from experiments.metaheuristics import random_search, simulated_annealing


def _shop(seed=1, n_items=8):
    return generate_synthetic_shop(name=f'feas_{seed}', seed=40_000 + seed,
                                   n_items=n_items, width=12.0, height=10.0)


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
    bp = base_params_for(ss)
    for fn in (random_search, simulated_annealing):
        lay = fn(shop, ss, names, bp, seed=0, budget=12,
                 mc_iters=60, mc_days=10)
        assert set(lay.keys()) == set(names), fn.__name__
        assert_layout_valid(ss, lay)
