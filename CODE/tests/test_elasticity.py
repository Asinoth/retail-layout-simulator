"""Elasticity-mapping and cited-coefficient regression tests (audit R5.3).

Guards the load-bearing coefficient module: the positive GA weights sum
to one (asserted at import) and the cited elasticity bands are
non-negative. The score-to-parameter mapping is checked on the production
``_ga_fitness`` itself: conversion and basket rise with the composite
score, impulse rises with the impulse sub-score, the clamps hold, and a
layout scoring what the anchor scores leaves the base parameters unchanged
(these parameters opt into the zero anchor explicitly, since the objective
refuses parameters without one; the as-built anchor itself is pinned in
tests/test_objective_anchor.py).
"""

import importlib

import pytest

import retail_literature as RL
from layout_objective import SCORE_ANCHOR_KEY, zero_anchor

_ELAS = ('ELASTICITY_CONV_BASE', 'ELASTICITY_CONV_GAIN_MAX',
         'ELASTICITY_IMP_BASE', 'ELASTICITY_IMP_GAIN_MAX',
         'ELASTICITY_BSK_BASE', 'ELASTICITY_BSK_GAIN_MAX')


def test_module_import_weight_assertion():
    # Re-executes the module-level assertion that the positive GA weights
    # sum to 1.0; a broken weight set raises AssertionError here.
    importlib.reload(RL)


def test_elasticity_constants_nonnegative():
    for name in _ELAS:
        assert getattr(RL, name) >= 0.0, name


_BP = {
    'conversion_rate': 0.30, 'impulse_rate': 0.20,
    'avg_basket_size': 4.0, 'std_basket_size': 1.5,
    'basket_sizes_observed': [], 'abandonment_rate': 0.10,
    'avg_queue_time': 5.0, 'rev_per_converting_customer': 40.0,
    'rev_std': 10.0, 'customers_per_hour': 50.0, 'avg_impulse_value': 3.0,
    SCORE_ANCHOR_KEY: zero_anchor(),
}


def _mapped(score, impulse, **overrides):
    """Run the production ``_ga_fitness`` with a fixed layout score and
    return the parameters it hands to the MC engine. Penalty and
    abandonment sub-scores are zero so only the elasticity terms act."""
    from experiments._common import HeadlessShop
    shop = HeadlessShop(12.0, 10.0)
    bkd = {'impulse': impulse, 'flow': 0.0, 'section_compliance': 0.0,
           'bottleneck_penalty': 0.0}
    shop._ga_compute_layout_score = lambda chrom, names, bp: (score, bkd)
    seen = {}

    def _engine(**kw):
        seen.update(kw)
        return {'mean': 0.0}

    shop._mc_engine = _engine
    shop._ga_fitness(None, [], dict(_BP, **overrides), mc_days=1, mc_iters=1)
    return seen


def _strictly_increasing(vals):
    return all(a < b for a, b in zip(vals, vals[1:]))


def test_score_mapping_increases_with_composite_score():
    runs = [_mapped(s, impulse=0.5) for s in (0.0, 0.25, 0.5, 0.75, 1.0)]
    for key in ('conv', 'avg_bsk', 'rev_mean'):
        assert _strictly_increasing([r[key] for r in runs]), key


def test_impulse_mapping_increases_with_impulse_subscore():
    runs = [_mapped(0.5, impulse=u) for u in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert _strictly_increasing([r['imp_rate'] for r in runs])


def test_zero_score_leaves_base_parameters():
    r = _mapped(0.0, impulse=0.0)
    assert r['conv'] == pytest.approx(_BP['conversion_rate'])
    assert r['imp_rate'] == pytest.approx(_BP['impulse_rate'])
    assert r['avg_bsk'] == pytest.approx(_BP['avg_basket_size'])
    assert r['rev_mean'] == pytest.approx(_BP['rev_per_converting_customer'])


def test_score_mapping_respects_clamps():
    # Base rates near one push the lifted values past the upper clamps.
    hi = _mapped(1.0, impulse=1.0, conversion_rate=0.98, impulse_rate=0.95)
    assert 0.01 <= hi['conv'] <= 0.99
    assert 0.0 <= hi['imp_rate'] <= 0.99
    lo = _mapped(0.0, impulse=0.0, conversion_rate=0.0, impulse_rate=0.0)
    assert lo['conv'] >= 0.01
    assert lo['imp_rate'] >= 0.0


def test_ga_positive_weights_sum_to_one():
    from experiments._common import elasticity_snapshot
    w = elasticity_snapshot()['GA_weights']
    # The seven composite CRITERIA weights sum to 1.0; the two penalty
    # fields (overlap_penalty, bottleneck_penalty) are a separate, stricter
    # scale and are excluded.
    criteria = ['traffic', 'cross_merch', 'impulse', 'flow',
                'revenue_placement', 'section_compliance', 'accessibility']
    assert set(w) == set(criteria) | {'overlap_penalty', 'bottleneck_penalty'}
    assert abs(sum(w[k] for k in criteria) - 1.0) < 1e-6


def test_ga_weight_values():
    # Five spatial criteria share 0.80 equally; section compliance and
    # accessibility keep fixed shares. A dwell criterion would be constant
    # across layouts, so there is none.
    spatial = (RL.GA_W_TRAFFIC, RL.GA_W_CROSS_MERCH, RL.GA_W_IMPULSE,
               RL.GA_W_FLOW, RL.GA_W_REVENUE_PLACEMENT)
    assert all(v == pytest.approx(0.16) for v in spatial)
    assert RL.GA_W_SECTION_COMPLIANCE == pytest.approx(0.16)
    assert RL.GA_W_ACCESSIBILITY == pytest.approx(0.04)
    assert not hasattr(RL, 'GA_W_DWELL')
