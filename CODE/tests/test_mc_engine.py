"""Monte Carlo engine regression tests (audit R5.3).

Encodes the paper's Verification-section claims for the MC engine as
runnable assertions: (1) with day noise disabled the empirical mean
matches the closed-form expectation, and (2) basket size is *tracked*,
not monetized, so it does not move revenue.
"""

import numpy as np

from sim_calibration import mc_engine

_PARAMS = dict(
    cph=20.0, op_hours=10.0, conv=0.30,
    rev_mean=25.0, rev_std=2.0, imp_rate=0.40, imp_val=5.0,
    avg_bsk=4.0, std_bsk=1.0, n_days=30,
)


def _run(n_iter, avg_bsk=None, seed=0):
    p = dict(_PARAMS)
    if avg_bsk is not None:
        p['avg_bsk'] = avg_bsk
    return mc_engine(
        cph=p['cph'], conv=p['conv'], rev_mean=p['rev_mean'],
        rev_std=p['rev_std'], imp_rate=p['imp_rate'], imp_val=p['imp_val'],
        avg_bsk=p['avg_bsk'], std_bsk=p['std_bsk'], observed_baskets=None,
        n_days=p['n_days'], n_iter=n_iter, op_hours=p['op_hours'],
        wknd_mult=1.0, monthly_growth=0.0, day_noise_std=0.0,
        rng=np.random.default_rng(seed))


def test_mc_mean_matches_closed_form():
    """E[total] = n_days * (cph*op_hours) * conv * (rev_mean + imp_rate*imp_val)
    with weekend multiplier 1 and day noise off."""
    p = _PARAMS
    expected = (p['n_days'] * p['cph'] * p['op_hours'] * p['conv']
                * (p['rev_mean'] + p['imp_rate'] * p['imp_val']))
    res = _run(n_iter=40000, seed=0)
    rel_err = abs(res['mean'] - expected) / expected
    assert rel_err < 0.01, f"MC mean {res['mean']:.1f} vs closed-form {expected:.1f} (rel err {rel_err:.4f})"


def test_basket_size_does_not_move_revenue():
    """Basket statistics are tracked but not double-monetized: changing
    avg_bsk must leave the revenue totals identical (the basket draw
    consumes the same number of RNG values regardless of its mean, so the
    revenue stream is bit-identical)."""
    small = _run(n_iter=5000, avg_bsk=4.0, seed=1)
    large = _run(n_iter=5000, avg_bsk=40.0, seed=1)
    assert small['mean'] == large['mean']
    assert np.array_equal(small['totals'], large['totals'])


def test_mc_seed_reproducible():
    """Same seed -> identical totals (determinism of the engine itself)."""
    a = _run(n_iter=3000, seed=5)
    b = _run(n_iter=3000, seed=5)
    assert np.array_equal(a['totals'], b['totals'])
