"""Effective weights, the standardized weight sweep and the LHS reporting
additions (review R09, R37, R38).

Equal nominal weights are equal effective weights only for standardized
criteria (Dawes 1979). ``criterion_spread`` measures each criterion's SD
over repaired random layouts and its effective weight; the elasticity
runner sweeps the weights on the raw and on the standardized scale, anchors
every re-weighted score at the as-built layout re-scored under the same
weights, and reports the pair counts and within-pair ranges behind its
fractions. These tests pin those pieces on small inputs.
"""

import numpy as np
import pytest

import retail_literature as RL
from baselines import random_valid, popularity_rank
from experiments._common import (SCORE_CRITERIA, anchor_base_params,
                                 base_params_for, build_headless_shop,
                                 criterion_spread, feasible_layout,
                                 layout_score)
from experiments.run_elasticity_lhs import (
    COLUMNS, COMPARATORS, DEFAULT_SPATIAL_WEIGHTS, SPATIAL_CRITERIA,
    STANDARDIZED_SCALES, _anchor_under, _weight_design,
    closed_form_extras_summary, horizon_revenue, pair_breakdown,
    recompose_score, recompose_standardized, standardization,
    weight_sweep_lifts)
from layout_objective import SCORE_ANCHOR_KEY
from synthetic_shops import generate_synthetic_shop


@pytest.fixture(scope='module')
def store():
    ss = generate_synthetic_shop(name='eff', seed=10_000, n_items=10)
    shop = build_headless_shop(ss)
    names = [it.name for it in ss.items]
    as_built = {n: tuple(shop.floors[1]['items'][n]['position'])
                for n in names}
    bp = anchor_base_params(shop, names, base_params_for(ss), as_built)
    return ss, shop, names, as_built, bp


def test_criterion_spread_is_seeded_and_consistent(store):
    ss, shop, names, _, _ = store
    state = np.random.get_state()
    a = criterion_spread(shop, names, n_layouts=12, seed=3)
    # Only its own generator is drawn from.
    assert np.array_equal(np.random.get_state()[1], state[1])
    b = criterion_spread(shop, names, n_layouts=12, seed=3)
    c = criterion_spread(shop, names, n_layouts=12, seed=4)
    assert a == b
    assert a['sd'] != c['sd']
    assert set(a['sd']) == set(SCORE_CRITERIA)
    # The repair holds every item inside its zone: section compliance is
    # constant and carries no effective weight.
    assert a['sd']['section_compliance'] == 0.0
    assert a['effective_weight']['section_compliance'] == 0.0
    for k in SCORE_CRITERIA:
        assert a['effective_weight'][k] == pytest.approx(
            abs(a['weight'][k]) * a['sd'][k])
    assert sum(a['effective_share'].values()) == pytest.approx(1.0)
    # The nominal weights are the registry's, signed as they enter.
    assert a['weight']['traffic'] == RL.GA_W_TRAFFIC
    assert a['weight']['overlap_penalty'] == -RL.GA_PEN_OVERLAP
    # Unequal effective weights: cross-merchandising moves the composite
    # far more than traffic does.
    assert a['effective_weight']['cross_merch'] > \
        5 * a['effective_weight']['traffic']


def test_standardization_equalizes_the_spread():
    spread = {'sd': {'traffic': 0.001, 'cross_merch': 0.03, 'impulse': 0.0,
                     'flow': 0.01, 'revenue_placement': 0.0001}}
    factors, kappa = standardization(spread)
    varying = [0.001, 0.03, 0.01, 0.0001]
    assert kappa == pytest.approx(np.mean(varying))
    sd = [spread['sd'][c] for c in SPATIAL_CRITERIA]
    for f, s in zip(factors, sd):
        if s > 0:
            assert f * s == pytest.approx(kappa)   # equal spread afterwards
        else:
            assert f == 0.0                         # constant: drops out


def test_standardized_recomposition_with_unit_factors_is_the_raw_one():
    bkd = {'traffic': 0.5, 'cross_merch': 0.25, 'impulse': 1.0, 'flow': 0.1,
           'revenue_placement': 0.75, 'section_compliance': 1.0,
           'accessibility': 0.5, 'overlap_penalty': 0.0,
           'bottleneck_penalty': 0.2}
    w = (0.1, 0.2, 0.3, 0.1, 0.1)
    assert recompose_standardized(bkd, w, (1.0,) * 5) == \
        pytest.approx(recompose_score(bkd, w), abs=1e-12)
    doubled = recompose_standardized(bkd, w, (2.0,) * 5)
    spatial = sum(bkd[c] * wk for c, wk in zip(SPATIAL_CRITERIA, w))
    assert doubled - recompose_score(bkd, w) == pytest.approx(spatial)


def test_anchor_is_rescored_under_the_same_weights(store):
    """Under any weighting the anchor layout keeps reproducing the
    calibrated revenue: its re-weighted score is its own anchor."""
    ss, shop, names, as_built, bp = store
    s0, b0 = bp[SCORE_ANCHOR_KEY]['score'], bp[SCORE_ANCHOR_KEY]['breakdown']
    base = horizon_revenue(s0, b0, bp)
    for w in [(0.8, 0, 0, 0, 0), (0, 0, 0.8, 0, 0), DEFAULT_SPATIAL_WEIGHTS]:
        anchor = _anchor_under(bp, lambda b: recompose_score(b, w))
        assert anchor['score'] == pytest.approx(recompose_score(b0, w))
        assert horizon_revenue(recompose_score(b0, w), b0, bp,
                               anchor=anchor) == pytest.approx(base, rel=1e-12)


def test_weight_sweeps_run_on_an_anchored_store(store):
    ss, shop, names, as_built, bp = store
    layouts = {'GA': feasible_layout(shop, names, random_valid(ss, seed=1)),
               'popularity_rank': feasible_layout(
                   shop, names, popularity_rank(ss, seed=0)),
               'oracle': feasible_layout(shop, names, random_valid(ss, seed=2)),
               'random_search': feasible_layout(shop, names, as_built),
               'simulated_annealing': feasible_layout(
                   shop, names, random_valid(ss, seed=3))}
    scores = {k: layout_score(shop, names, v) for k, v in layouts.items()}
    draws = np.array([DEFAULT_SPATIAL_WEIGHTS, (0.8, 0, 0, 0, 0)])
    raw = weight_sweep_lifts(scores, draws, bp)
    # At the default weights the raw sweep is the midpoint projection.
    for m in COMPARATORS:
        want = (horizon_revenue(*scores['GA'], bp)
                - horizon_revenue(*scores[m], bp))
        assert raw[m][0] == pytest.approx(want, rel=1e-12, abs=1e-9)
        assert raw[m][1][0] == pytest.approx(raw[m][0], rel=1e-12, abs=1e-9)
    spread = criterion_spread(shop, names, n_layouts=20, seed=1)
    factors, _ = standardization(spread)
    std = weight_sweep_lifts(scores, draws, bp, factors=factors)
    assert set(std) == set(COMPARATORS)
    # The standardized scale changes the lifts.
    assert any(abs(std[m][0] - raw[m][0]) > 1e-6 for m in COMPARATORS)


def _row(sc, seed, draw, lifts, revs):
    """A results.csv row with the given comparator lifts and revenues."""
    rec = dict(zip(COLUMNS, [0.0] * len(COLUMNS)))
    rec.update(scenario=sc, seed=seed, draw=draw)
    rec['lift30'] = lifts[0]
    rec['rev30_baseline'] = revs[0]
    for m, l, r in zip(COMPARATORS[1:], lifts[1:], revs[1:]):
        rec[f'lift30_{m}'] = l
        rec[f'rev30_{m}'] = r
    return tuple(rec[c] for c in COLUMNS)


def test_pair_breakdown_counts_and_ranges():
    rows = []
    for d, (a, b) in enumerate([(5.0, -1.0), (7.0, 2.0), (6.0, -3.0)]):
        rows.append(_row(0, 0, d, [a] * 4, [100.0] * 4))
        rows.append(_row(0, 1, d, [b] * 4, [100.0] * 4))
    per_pair, agg = pair_breakdown(rows, 'popularity_rank')
    assert [p['n_positive'] for p in per_pair] == [3, 1]
    assert per_pair[0]['lift_range'] == pytest.approx(2.0)
    assert per_pair[1]['lift_min'] == -3.0 and per_pair[1]['lift_max'] == 2.0
    assert per_pair[0]['pct_max'] == pytest.approx(7.0)
    assert agg['n_pairs'] == 2
    assert agg['n_pairs_all_draws_positive'] == 1
    assert agg['n_pairs_mixed_sign'] == 1
    assert agg['n_positive'] == 4 and agg['n_draws_total'] == 6
    assert agg['within_pair_range_max'] == pytest.approx(5.0)


def test_closed_form_extras_summary_counts():
    def pair(sc, seed, v):
        lifts = {m: v for m in COMPARATORS}
        return {'scenario': sc, 'seed': seed, 'midpoint': lifts,
                'basket_corners': {'0.02': lifts, '0': lifts},
                'short_path_variant': {
                    'midpoint': lifts,
                    'lhs': {m: {'n': 4, 'n_positive': 4 if v > 0 else 1,
                                'frac_positive': 1.0, 'median': v,
                                'min': v - 1, 'max': v + 1}
                            for m in COMPARATORS}}}
    extras = [{'scenario': 0, 'pairs': [pair(0, 0, 2.0), pair(0, 1, -1.0)]}]
    out = closed_form_extras_summary(extras)
    assert out['n_pairs'] == 2
    st = out['basket_corners']['0.02']['comparisons']['oracle']
    assert st['n_positive'] == 1 and st['n'] == 2
    assert out['basket_corners']['0']['elasticities']['bsk'] == 0.0
    lhs = out['short_path_variant']['lhs']['oracle']
    assert lhs['n_positive'] == 5 and lhs['n'] == 8
    assert lhs['n_pairs_all_positive'] == 1
    assert lhs['frac_positive'] == pytest.approx(5 / 8)
    assert out['short_path_variant']['basket_exclude'] == \
        ['flow', 'accessibility']


def test_unit_scale_is_the_criteria_divided_by_their_sd():
    """Design decision D3's standardized scale: kappa = 1."""
    spread = {'sd': {'traffic': 0.001, 'cross_merch': 0.03, 'impulse': 0.0,
                     'flow': 0.01, 'revenue_placement': 0.0001}}
    factors, kappa = standardization(spread, 'unit')
    assert kappa == 1.0
    for c, f in zip(SPATIAL_CRITERIA, factors):
        sd = spread['sd'][c]
        assert f == (1.0 / sd if sd > 0 else 0.0)
    with pytest.raises(ValueError):
        standardization(spread, 'bogus')


def test_kappa_is_not_a_neutral_factor(store):
    """kappa scales the standardized spatial terms only; accessibility, the
    penalties and the raw sub-scores revenue reads outside the composite do
    not scale with it, so the two standardized designs give different lifts
    -- which is why both are reported."""
    ss, shop, names, as_built, bp = store
    layouts = {'GA': feasible_layout(shop, names, random_valid(ss, seed=1)),
               'popularity_rank': feasible_layout(
                   shop, names, popularity_rank(ss, seed=0)),
               'oracle': feasible_layout(shop, names, random_valid(ss, seed=2)),
               'random_search': feasible_layout(shop, names, as_built),
               'simulated_annealing': feasible_layout(
                   shop, names, random_valid(ss, seed=3))}
    scores = {k: layout_score(shop, names, v) for k, v in layouts.items()}
    spread = criterion_spread(shop, names, n_layouts=20, seed=1)
    draws = np.array([DEFAULT_SPATIAL_WEIGHTS])
    mean_sd = weight_sweep_lifts(scores, draws, bp,
                                 factors=standardization(spread)[0])
    unit = weight_sweep_lifts(scores, draws, bp,
                              factors=standardization(spread, 'unit')[0])
    # The same layouts and the same (equal) weights: only kappa differs.
    assert all(abs(unit[m][0] - mean_sd[m][0])
               > 1e-6 * max(abs(mean_sd[m][0]), 1.0) for m in COMPARATORS)


def test_standardized_design_records_kappa_and_its_role():
    raw = _weight_design(8, 1)
    assert 'scale' not in raw
    for name, (key, fname, what) in STANDARDIZED_SCALES.items():
        d = _weight_design(8, 1, scale=name)
        assert d['scale']['name'] == name
        assert d['scale']['kappa'] == what
        assert 'not a neutral common factor' in d['scale']['kappa_role']
        assert key.startswith('weight_sweep_standardized')
        assert fname == key + '.csv'
    assert set(STANDARDIZED_SCALES) == {'mean_sd', 'unit'}
