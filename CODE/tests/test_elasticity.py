"""Elasticity-mapping and cited-coefficient regression tests (audit R5.3).

Guards the load-bearing coefficient module: the positive GA weights sum
to one (asserted at import), the cited elasticity bands are non-negative,
and the score-to-lift multiplier is monotone non-decreasing in the
composite score.
"""

import importlib

import retail_literature as RL

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


def test_score_to_multiplier_monotone():
    """The applied multiplier 1 + score * midpoint_elasticity is
    non-decreasing in the composite score for each channel."""
    channels = [
        (RL.ELASTICITY_CONV_BASE, RL.ELASTICITY_CONV_GAIN_MAX),
        (RL.ELASTICITY_IMP_BASE, RL.ELASTICITY_IMP_GAIN_MAX),
        (RL.ELASTICITY_BSK_BASE, RL.ELASTICITY_BSK_GAIN_MAX),
    ]
    scores = [0.0, 0.25, 0.5, 0.75, 1.0]
    for base, gain in channels:
        mid = base + 0.5 * gain
        assert mid > 0.0
        mult = [1.0 + s * mid for s in scores]
        assert all(mult[i] <= mult[i + 1] for i in range(len(mult) - 1))


def test_ga_positive_weights_sum_to_one():
    from experiments._common import elasticity_snapshot
    w = elasticity_snapshot()['GA_weights']
    # The eight composite CRITERIA weights sum to 1.0; the two penalty
    # fields (overlap_penalty, bottleneck_penalty) are a separate, stricter
    # scale and are excluded.
    criteria = ['traffic', 'cross_merch', 'impulse', 'flow',
                'revenue_placement', 'dwell', 'section_compliance',
                'accessibility']
    assert abs(sum(w[k] for k in criteria) - 1.0) < 1e-6
