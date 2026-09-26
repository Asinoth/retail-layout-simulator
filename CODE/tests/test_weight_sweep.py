"""The composite-weight sweep in ``run_elasticity_lhs``.

The sweep re-weights the five spatial criteria of layouts that were
already scored, by recomposing the composite from the per-criterion
breakdown ``_ga_compute_layout_score`` returns. That is only a test of the
weights if the recomposition IS the composite: at the default weights it
must give back the score the layout function computed, including the
penalty terms. The draws must lie on the simplex scaled to the spatial
total, and the sign-flip count must count a pair once, whichever way its
lift crosses zero.
"""

import numpy as np
import pytest

import retail_literature as RL
from synthetic_shops import generate_synthetic_shop
from baselines import random_valid, popularity_rank
from experiments._common import (build_headless_shop, base_params_for,
                                 feasible_layout, layout_to_chromosome)
from layout_objective import with_zero_anchor
from experiments.run_elasticity_lhs import (
    COMPARATORS, DEFAULT_SPATIAL_WEIGHTS, ELASTICITY_MIDPOINTS,
    RECOMPOSE_TOL, SPATIAL_CRITERIA, SPATIAL_TOTAL, WEIGHT_COLUMNS,
    check_recomposition, draw_spatial_weights, horizon_revenue,
    recompose_score, weight_sweep_lifts, weight_sweep_summary)


def _scored_layouts(seed=10_000, n_items=10):
    """(score, breakdown) of several layouts of one synthetic shop: the
    as-built layout, repaired random and popularity layouts, and a raw
    stacked layout whose overlap penalty is nonzero."""
    ss = generate_synthetic_shop(name='weights', seed=seed, n_items=n_items)
    shop = build_headless_shop(ss)
    names = [it.name for it in ss.items]
    # The zero anchor, asked for explicitly: these tests pin the
    # recomposition itself, which is the same under any anchor (the
    # anchored sweeps are tested in tests/test_effective_weights.py).
    bp = with_zero_anchor(base_params_for(ss))
    as_built = {n: tuple(shop.floors[1]['items'][n]['position'])
                for n in names}
    layouts = [as_built,
               feasible_layout(shop, names, random_valid(ss, seed=1)),
               feasible_layout(shop, names, random_valid(ss, seed=2)),
               feasible_layout(shop, names, popularity_rank(ss, seed=0))]
    x0, y0 = as_built[names[0]]
    layouts.append({n: (x0, y0) for n in names})
    return shop, names, bp, [
        shop._ga_compute_layout_score(layout_to_chromosome(lay, names),
                                      names, bp)
        for lay in layouts]


def test_default_weights_are_the_literature_weights():
    assert DEFAULT_SPATIAL_WEIGHTS == (
        RL.GA_W_TRAFFIC, RL.GA_W_CROSS_MERCH, RL.GA_W_IMPULSE,
        RL.GA_W_FLOW, RL.GA_W_REVENUE_PLACEMENT)
    assert SPATIAL_TOTAL == pytest.approx(0.80)


def test_recomposition_reproduces_layout_score_at_default_weights():
    shop, names, bp, scored = _scored_layouts()
    assert any(b['overlap_penalty'] > 0 for _, b in scored)
    for score, bkd in scored:
        assert set(SPATIAL_CRITERIA) <= set(bkd)
        assert abs(recompose_score(bkd, DEFAULT_SPATIAL_WEIGHTS)
                   - score) <= RECOMPOSE_TOL
        check_recomposition(score, bkd)


def test_recomposition_reproduces_bottleneck_penalty():
    # The synthetic shop records no congestion, so the bottleneck term is
    # zero above; give it some so that term is exercised too.
    ss = generate_synthetic_shop(name='weights_bn', seed=10_003, n_items=10)
    shop = build_headless_shop(ss)
    names = [it.name for it in ss.items]
    bp = base_params_for(ss)
    lay = feasible_layout(shop, names, random_valid(ss, seed=4))
    cells = {}
    for n in names:
        x, y = lay[n]
        w, h = shop.floors[1]['items'][n]['size']
        cells[f'{int(x + w / 2)},{int(y + h / 2)}'] = 1 + len(cells)
    shop.customer_simulation.analytics['bottlenecks'] = cells
    score, bkd = shop._ga_compute_layout_score(
        layout_to_chromosome(lay, names), names, bp)
    assert bkd['bottleneck_penalty'] > 0
    assert abs(recompose_score(bkd, DEFAULT_SPATIAL_WEIGHTS)
               - score) <= RECOMPOSE_TOL


def test_check_recomposition_rejects_a_mismatch():
    _, _, _, scored = _scored_layouts()
    score, bkd = scored[0]
    with pytest.raises(AssertionError):
        check_recomposition(score + 1e-6, bkd)


def test_weight_draws_on_the_scaled_simplex():
    w = draw_spatial_weights(512, seed=7)
    assert w.shape == (512, len(SPATIAL_CRITERIA))
    assert (w >= 0).all()
    np.testing.assert_allclose(w.sum(axis=1), 0.80, atol=1e-12)
    # Seeded: the same seed gives the same draws, another seed others.
    np.testing.assert_array_equal(w, draw_spatial_weights(512, seed=7))
    assert not np.array_equal(w, draw_spatial_weights(512, seed=8))
    # Uniform on the simplex: each coordinate has mean total / 5.
    np.testing.assert_allclose(w.mean(axis=0), 0.16, atol=0.02)


def test_hand_made_breakdown():
    bkd = {'traffic': 0.5, 'cross_merch': 0.25, 'impulse': 1.0,
           'flow': 0.0, 'revenue_placement': 0.75,
           'section_compliance': 1.0, 'accessibility': 0.5,
           'overlap_penalty': 1.0, 'bottleneck_penalty': 0.5}
    fixed = (1.0 * RL.GA_W_SECTION_COMPLIANCE + 0.5 * RL.GA_W_ACCESSIBILITY
             - 1.0 * RL.GA_PEN_OVERLAP - 0.5 * RL.GA_PEN_BOTTLENECK)
    # All of the spatial total on traffic.
    assert recompose_score(bkd, (0.8, 0.0, 0.0, 0.0, 0.0)) == \
        pytest.approx(0.8 * 0.5 + fixed, abs=1e-12)
    # An uneven split.
    w = (0.1, 0.2, 0.3, 0.0, 0.2)
    assert recompose_score(bkd, w) == pytest.approx(
        0.1 * 0.5 + 0.2 * 0.25 + 0.3 * 1.0 + 0.0 * 0.0 + 0.2 * 0.75 + fixed,
        abs=1e-12)
    # Equal weights: the composite _ga_compute_layout_score would return.
    assert recompose_score(bkd, DEFAULT_SPATIAL_WEIGHTS) == pytest.approx(
        0.16 * (0.5 + 0.25 + 1.0 + 0.0 + 0.75) + fixed, abs=1e-12)


def test_sweep_lifts_at_default_weights_match_projection():
    shop, names, bp, scored = _scored_layouts()
    scores = dict(zip(('GA',) + COMPARATORS, scored))
    draws = np.array([DEFAULT_SPATIAL_WEIGHTS,
                      (0.8, 0.0, 0.0, 0.0, 0.0),
                      (0.0, 0.0, 0.8, 0.0, 0.0)])
    lifts = weight_sweep_lifts(scores, draws, bp)
    rev = {k: horizon_revenue(s, b, bp, dict(ELASTICITY_MIDPOINTS))
           for k, (s, b) in scores.items()}
    # Off the default weights, each lift is the projection of the
    # RECOMPOSED composite with the breakdown untouched, so impulse and
    # abandonment still read their own sub-scores. These parameters carry
    # the explicit zero anchor, which stays zero under every weighting.
    rev_w = [{k: horizon_revenue(recompose_score(b, w), b, bp)
              for k, (_, b) in scores.items()} for w in draws]
    for m in COMPARATORS:
        at_default, swept = lifts[m]
        assert swept.shape == (3,)
        assert at_default == pytest.approx(rev['GA'] - rev[m], abs=1e-6)
        assert swept[0] == pytest.approx(at_default, abs=1e-9)
        for i in (1, 2):
            assert swept[i] == pytest.approx(rev_w[i]['GA'] - rev_w[i][m],
                                             rel=1e-12, abs=1e-9)
    # The weights reach the revenue: a sweep that ignored them would give
    # the default lift at every draw.
    assert any(abs(lifts[m][1][i] - lifts[m][0]) > 1e-6
               for m in COMPARATORS for i in (1, 2))


def test_sweep_refuses_a_breakdown_that_does_not_recompose():
    shop, names, bp, scored = _scored_layouts()
    scores = dict(zip(('GA',) + COMPARATORS, scored))
    s, b = scores['oracle']
    scores['oracle'] = (s + 0.01, b)
    with pytest.raises(AssertionError):
        weight_sweep_lifts(scores, np.array([DEFAULT_SPATIAL_WEIGHTS]), bp)


def test_sign_flip_count():
    w = (0.16,) * 5

    def rows(sc, seed, m, default, lifts):
        return [(sc, seed, d, m, *w, lift, default)
                for d, lift in enumerate(lifts)]

    assert WEIGHT_COLUMNS[-2:] == ['lift30', 'lift30_default']
    data = []
    for m in COMPARATORS:
        data += rows(0, 0, m, 5.0, [4.0, 6.0, -1.0])   # flips below zero
        data += rows(0, 1, m, 5.0, [4.0, 6.0, 1.0])    # never flips
        data += rows(1, 0, m, -2.0, [-1.0, -3.0, -4.0])  # never flips
        data += rows(1, 1, m, -2.0, [-1.0, 3.0, -4.0])   # flips above zero
        data += rows(2, 0, m, 0.0, [0.0, 0.0, 0.0])    # identical layouts
        # Every draw negative, the default weights positive: the default
        # is a point of the simplex, so this pair flips too.
        data += rows(2, 1, m, 1.0, [-1.0, -2.0, -3.0])
    out = weight_sweep_summary(data)
    assert set(out) == set(COMPARATORS)
    for st in out.values():
        assert st['n_pairs'] == 6
        assert st['n_values'] == 18
        assert st['n_pairs_sign_flip'] == 3
        assert st['n_pairs_positive_at_default'] == 3
        assert st['frac_positive'] == pytest.approx(6 / 18)
        assert set(st) >= {'frac_positive', 'median', 'p5', 'p95'}
