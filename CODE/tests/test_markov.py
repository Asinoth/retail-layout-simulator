"""Absorbing Markov-chain regression tests (audit R5.3).

Encodes the paper's Verification claims: estimated transition matrices
are row-stochastic, absorbing rows are identity, absorption
probabilities sum to one, and expected steps to absorption are positive.
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
