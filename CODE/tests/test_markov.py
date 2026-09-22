"""Absorbing Markov-chain regression tests (audit R5.3).

Encodes the paper's Verification claims: estimated transition matrices
are row-stochastic, absorbing rows are identity, absorption
probabilities sum to one, and expected steps to absorption are positive.
Logged transitions are turned into Laplace-smoothed row frequencies, and
a singular I - Q falls back to the pseudo-inverse.
"""

import numpy as np

from sim_calibration import (build_empirical_transition_matrix,
                             compute_absorbing_analysis,
                             MARKOV_TRANSIENT, MARKOV_ABSORBING)


def _analytics():
    # Enough transitions to exercise the prior path deterministically.
    return {'total_customers': 1000, 'completed_purchases': 300,
            'abandoned_carts': 150, 'total_items_browsed': 2500}


def test_transition_matrix_row_stochastic():
    states, T, meta = build_empirical_transition_matrix(_analytics())
    assert np.allclose(T.sum(axis=1), 1.0, atol=1e-9)
    assert (T >= -1e-12).all()


def test_absorbing_rows_are_identity():
    states, T, meta = build_empirical_transition_matrix(_analytics())
    si = {s: i for i, s in enumerate(states)}
    for s in MARKOV_ABSORBING:
        i = si[s]
        assert T[i, i] == 1.0
        others = np.delete(T[i], i)
        assert np.allclose(others, 0.0)


def test_absorption_analysis_valid():
    states, T, meta = build_empirical_transition_matrix(_analytics())
    res = compute_absorbing_analysis(states, T)
    B = res['absorption_B']
    # Every transient state is absorbed with total probability one.
    assert np.allclose(B.sum(axis=1), 1.0, atol=1e-6)
    assert (B >= -1e-9).all()
    # Expected steps from entry are finite and positive.
    assert res['expected_steps_from_entering'] > 0
    assert np.isfinite(res['expected_steps_from_entering'])
    # Only two absorbing states, so the two entry probabilities sum to one.
    assert abs(res['p_purchase_from_entering']
               + res['p_abandon_from_entering'] - 1.0) < 1e-6


_COUNTS = {
    ('entering', 'moving'): 40,
    ('moving', 'shopping'): 30, ('moving', 'checking_out'): 12,
    ('moving', 'exiting'): 5,
    ('shopping', 'moving'): 18, ('shopping', 'checking_out'): 10,
    ('shopping', 'abandoned'): 3,
    ('checking_out', 'purchased'): 20, ('checking_out', 'abandoned'): 2,
    ('exiting', 'abandoned'): 4,
}


def test_empirical_matrix_matches_laplace_frequencies():
    # Compare exact cell values, not just row sums: the builder
    # re-normalizes transient rows afterwards, which would hide a
    # mis-normalized or unsmoothed estimate from a row-sum check.
    analytics = dict(_analytics(), markov_transition_counts=dict(_COUNTS))
    states, T, meta = build_empirical_transition_matrix(analytics)
    assert meta['empirical']
    n = len(states)
    for i, a in enumerate(states):
        if a not in MARKOV_TRANSIENT:
            continue
        row_total = sum(c for (src, _), c in _COUNTS.items() if src == a)
        for j, b in enumerate(states):
            expected = (_COUNTS.get((a, b), 0) + 1.0) / (row_total + n)
            assert abs(T[i, j] - expected) < 1e-12, (a, b)


def test_singular_chain_uses_pseudo_inverse(monkeypatch):
    # A transient state that returns to itself with probability one is
    # never absorbed, so I - Q is singular. The estimator cannot produce
    # this (smoothing keeps every row off the identity), so the matrix is
    # edited by hand.
    states, T, _ = build_empirical_transition_matrix(_analytics())
    k = states.index('shopping')
    T = T.copy()
    T[k, :] = 0.0
    T[k, k] = 1.0

    calls = []
    real_pinv = np.linalg.pinv

    def _spy(a, *args, **kwargs):
        calls.append(a.shape)
        return real_pinv(a, *args, **kwargs)

    monkeypatch.setattr(np.linalg, 'pinv', _spy)
    res = compute_absorbing_analysis(states, T)
    assert calls
    assert np.isfinite(res['absorption_B']).all()
    assert np.isfinite(res['expected_steps_from_entering'])
