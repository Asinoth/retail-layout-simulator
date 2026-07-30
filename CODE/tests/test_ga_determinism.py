"""GA determinism regression test (audit R5.3 / R5.4).

The paper claims every reported number is reproducible from a recorded
seed. This encodes that contract for the optimizer: two headless GA runs
with the same seed, in the same (single-threaded) process, produce a
bit-identical result. It also documents the boundary of the guarantee ---
the determinism relies on the global numpy RNG not being touched
concurrently, which the single-threaded experiment path ensures and the
live multi-threaded GUI path does not (see run_ga_headless docstring).
"""

import numpy as np

from synthetic_shops import generate_synthetic_shop
from experiments._common import (build_headless_shop, base_params_for,
                                 run_ga_headless)


def _setup():
    ss = generate_synthetic_shop(name='det', seed=40_007, n_items=8,
                                 width=12.0, height=10.0)
    shop = build_headless_shop(ss)
    names = [it.name for it in ss.items]
    bp = base_params_for(ss)
    return shop, names, bp


def test_ga_bit_identical_under_fixed_seed():
    shop, names, bp = _setup()
    kw = dict(pop_size=10, n_gens=4, mut_rate=0.18, elite_frac=0.20,
              mc_iters=60, mc_days=10, rng_seed=7)
    a = run_ga_headless(shop, names, bp, **kw)
    b = run_ga_headless(shop, names, bp, **kw)
    assert a['best_fit'] == b['best_fit']
    assert np.array_equal(a['best_chrom'], b['best_chrom'])
    assert a['history_best'] == b['history_best']


def test_ga_different_seeds_differ():
    shop, names, bp = _setup()
    kw = dict(pop_size=10, n_gens=4, mut_rate=0.18, elite_frac=0.20,
              mc_iters=60, mc_days=10)
    a = run_ga_headless(shop, names, bp, rng_seed=1, **kw)
    b = run_ga_headless(shop, names, bp, rng_seed=2, **kw)
    # Different seeds should explore differently (not a hard guarantee of
    # different optima, but the search trajectories must not be identical).
    assert a['history_best'] != b['history_best']
