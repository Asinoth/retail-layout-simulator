"""The goodness-of-fit runner's size-matched replica yardstick.

A held-out statistic is only meaningful beside what a model that carried
its calibration period over perfectly would score at the SAME sample size:
a chi-square grows with the sample, so the two periods' full-size
statistic is no yardstick for a simulator that produced a few thousand
visits. ``_replicas`` draws as many store-period invoices as the pooled
test had simulated visits and scores them with the row's own test. These
tests pin that it is the null when the store period is the reference, has
power against a shifted period, reports where the simulator falls among
the replicas, and is reproducible.
"""
import numpy as np

from experiments import run_validation_gof as G


def _pooled(basket=None, revenue=None, category=None, n=400):
    rows = []
    for name, stat in (('Basket size (items/visit)', basket),
                       ('Per-visit revenue', revenue),
                       ('Category purchase shares', category)):
        if stat is not None:
            rows.append({'test': name, 'statistic': stat, 'n_simulated': n})
    return rows


def _clusters(rng, n, probs):
    """Baskets of 1-6 purchases over len(probs) categories, clustered: each
    basket leans towards one category."""
    k = len(probs)
    out = np.zeros((n, k))
    for i in range(n):
        lean = rng.choice(k, p=probs)
        for _ in range(rng.integers(1, 7)):
            c = lean if rng.random() < 0.5 else rng.choice(k, p=probs)
            out[i, c] += 1
    return out


def _samples(rng, n, mean_size):
    sizes = rng.poisson(mean_size, n) + 1
    return {'sizes': sizes.astype(float),
            'revenues': sizes * rng.gamma(4.0, 2.0, n)}


def test_replicas_of_the_reference_are_the_null():
    rng = np.random.default_rng(1)
    ref = _samples(rng, 5000, 3.0)
    probs = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
    ref_c = _clusters(rng, 5000, probs)
    out = G._replicas(ref, ref, ref_c, ref_c,
                      _pooled(0.02, 0.02, 5.0), 0.05, n_rep=60, seed=7)
    for name in ('basket', 'revenue', 'category'):
        r = out[name]
        assert r['n_replicas'] == 60 and r['n_per_replica'] == 400
        # A size-level test on a model that replays the reference; KS on
        # tied sizes is conservative, so only an upper bound is pinned.
        assert r['reject_rate'] <= 0.15
        assert r['lo'] <= r['median'] <= r['hi']
        assert 0.0 <= r['simulator_percentile'] <= 1.0


def test_replicas_of_a_shifted_period_carry_the_shift():
    rng = np.random.default_rng(2)
    ref = _samples(rng, 5000, 3.0)
    store = _samples(rng, 5000, 4.0)
    ref_c = _clusters(rng, 5000, np.array([0.3, 0.25, 0.2, 0.15, 0.1]))
    store_c = _clusters(rng, 5000, np.array([0.1, 0.15, 0.2, 0.25, 0.3]))
    out = G._replicas(store, ref, store_c, ref_c,
                      _pooled(0.0, 0.0, 0.0), 0.05, n_rep=40, seed=7)
    for name in ('basket', 'revenue', 'category'):
        r = out[name]
        assert r['reject_rate'] > 0.9
        # A simulator scoring 0 sits below every replica of the shift.
        assert r['simulator_percentile'] == 0.0
        assert r['median'] > 0.0


def test_replicas_are_reproducible_and_skip_rows_the_design_did_not_run():
    rng = np.random.default_rng(3)
    ref = _samples(rng, 2000, 3.0)
    ref_c = _clusters(rng, 2000, np.array([0.5, 0.3, 0.2]))
    pooled = _pooled(basket=0.05, revenue=None, category=3.0, n=300)
    a = G._replicas(ref, ref, ref_c, ref_c, pooled, 0.05, n_rep=20, seed=11)
    b = G._replicas(ref, ref, ref_c, ref_c, pooled, 0.05, n_rep=20, seed=11)
    assert a == b
    assert a['revenue'] is None
    assert a['basket']['n_per_replica'] == 300
    assert a['basket']['simulator_statistic'] == 0.05
