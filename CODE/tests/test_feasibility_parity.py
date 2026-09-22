"""Feasibility parity between the GA and every other layout method.

Every method's layouts are mapped through ``feasible_layout`` (the GA's
own repair chain) before MC scoring. For that to put all methods on the
same feasible set, the map must be idempotent (a repaired layout is a
fixed point) and its output must carry no overlap under the simulator
fitness's own predicate.
"""

import numpy as np

from synthetic_shops import generate_synthetic_shop
from baselines import (random_valid, perimeter_only, popularity_rank,
                       greedy_swap, assert_layout_valid,
                       assert_no_strict_overlap, _section_inner_bounds)
from experiments._common import (build_headless_shop, base_params_for,
                                 feasible_layout, layout_to_chromosome)


def _raw_random_layout(ss, rng):
    """Uniform positions inside section bounds with no overlap repair, so
    the repair chain has overlaps to resolve."""
    out = {}
    for it in ss.items:
        lo_x, lo_y, hi_x, hi_y = _section_inner_bounds(ss, it)
        out[it.name] = (float(rng.uniform(lo_x, hi_x)),
                        float(rng.uniform(lo_y, hi_y)))
    return out


def test_feasible_layout_idempotent_and_overlap_free():
    rng = np.random.default_rng(0)
    for idx in (0, 3, 12, 25):
        for n_items in (10, 12):
            ss = generate_synthetic_shop(name=f'parity_{idx}_{n_items}',
                                         seed=10_000 + idx, n_items=n_items,
                                         width=12.0, height=10.0)
            shop = build_headless_shop(ss)
            names = [it.name for it in ss.items]
            bp = base_params_for(ss)
            layouts = [fn(ss, seed=idx) for fn in (random_valid, perimeter_only,
                                                  popularity_rank, greedy_swap)]
            layouts += [_raw_random_layout(ss, rng) for _ in range(8)]
            for lay in layouts:
                once = feasible_layout(shop, names, lay)
                assert feasible_layout(shop, names, once) == once
                assert_layout_valid(ss, once)
                assert_no_strict_overlap(ss, once)
                _, breakdown = shop._ga_compute_layout_score(
                    layout_to_chromosome(once, names), names, bp)
                assert breakdown['overlap_penalty'] == 0
