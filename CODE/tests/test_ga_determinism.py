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
from baselines import random_valid
from experiments._common import (build_headless_shop, base_params_for,
                                 anchor_base_params,
                                 run_ga_headless, paired_mc_revenue,
                                 apply_layout, _INIT_POP_STREAM_TAG,
                                 _OPERATOR_STREAM_TAG)


def _synth():
    return generate_synthetic_shop(name='det', seed=40_007, n_items=8,
                                   width=12.0, height=10.0)


def _setup():
    ss = _synth()
    shop = build_headless_shop(ss)
    names = [it.name for it in ss.items]
    as_built = {n: tuple(shop.floors[1]['items'][n]['position'])
                for n in names}
    bp = anchor_base_params(shop, names, base_params_for(ss), as_built)
    return shop, names, bp


_KW = dict(pop_size=10, n_gens=4, mut_rate=0.18, elite_frac=0.20,
           mc_iters=60, mc_days=10, rng_seed=7)


def _positions(shop, names):
    return {n: list(shop.floors[1]['items'][n]['position']) for n in names}


def test_ga_bit_identical_under_fixed_seed():
    shop, names, bp = _setup()
    a = run_ga_headless(shop, names, bp, **_KW)
    b = run_ga_headless(shop, names, bp, **_KW)
    assert a['best_fit'] == b['best_fit']
    assert np.array_equal(a['best_chrom'], b['best_chrom'])
    assert a['history_best'] == b['history_best']


def test_ga_different_seeds_differ():
    shop, names, bp = _setup()
    kw = dict(_KW)
    del kw['rng_seed']
    a = run_ga_headless(shop, names, bp, rng_seed=1, **kw)
    b = run_ga_headless(shop, names, bp, rng_seed=2, **kw)
    # Different seeds should explore differently (not a hard guarantee of
    # different optima, but the search trajectories must not be identical).
    assert a['history_best'] != b['history_best']


def test_ga_independent_of_prior_paired_evaluations():
    """Scoring other layouts on a shop must not warm-start a later GA run
    on that shop: paired_mc_revenue leaves the item positions untouched."""
    fresh, names, bp = _setup()
    ref = run_ga_headless(fresh, names, bp, **_KW)

    shop, _, _ = _setup()
    before = _positions(shop, names)
    other = random_valid(_synth(), seed=5)
    assert any(tuple(before[n]) != other[n] for n in names)
    paired_mc_revenue(shop, names, other, bp, seed=11,
                      mc_iters=60, mc_days=10)
    assert _positions(shop, names) == before

    out = run_ga_headless(shop, names, bp, **_KW)
    assert out['best_fit'] == ref['best_fit']
    assert np.array_equal(out['best_chrom'], ref['best_chrom'])


def test_ga_streams_disjoint_from_mc_evaluation_seeds():
    """The population and operator streams are array-seeded, so neither can
    turn out to be the same Mersenne-Twister sequence as some other
    replication's integer-seeded evaluation stream."""
    def _draws(seed):
        return tuple(np.random.RandomState(seed).normal(0, 1, 8).round(12))

    n_seeds, n_gens = 10, 30
    mc = {_draws(s * 1000 + g)
          for s in range(n_seeds) for g in range(n_gens + 1 + 5)}
    for s in range(n_seeds):
        for tag in (_INIT_POP_STREAM_TAG, _OPERATOR_STREAM_TAG):
            assert _draws([s, tag]) not in mc


def test_ga_init_layout_overrides_shop_positions():
    """With ``init_layout`` the run starts from that layout whatever
    positions the shop currently holds."""
    fresh, names, bp = _setup()
    init = {n: tuple(fresh.floors[1]['items'][n]['position']) for n in names}
    ref = run_ga_headless(fresh, names, bp, **_KW)

    shop, _, _ = _setup()
    apply_layout(shop, random_valid(_synth(), seed=5))
    out = run_ga_headless(shop, names, bp, init_layout=init, **_KW)
    assert out['best_fit'] == ref['best_fit']
    assert np.array_equal(out['best_chrom'], ref['best_chrom'])
