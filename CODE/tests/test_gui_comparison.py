"""The interactive layout comparisons (review R42, R43).

The Optimize pipeline's projection, its re-scored comparison after the POST
window and the What-If tab all compare two layouts through
``layout_comparison.paired_comparison``. These tests pin, on a headless
synthetic store and without Tk:

  * the two arms really are paired: both run ``mc_engine`` under one seed,
    so their totals are strongly correlated and the paired SD is a small
    fraction of either arm's (the old hand-rolled loop gave a correlation
    near zero), identical layouts differ by exactly zero, and a seed
    reproduces the comparison;
  * the reported interval is a t interval over the pairs that covers the
    exact lift, and a direction is named only when it excludes zero;
  * the GUI projection and the headless closed form agree on the same
    layout pair: the Optimize path, the What-If path and
    ``experiments.closed_form.expected_revenue`` give the same exact lift,
    and the Monte Carlo estimate sits on it;
  * the pipeline's tornado has no basket bar and no longer feeds the
    optimizer, whose drivers are the band midpoints;
  * neither tornado report names a "top" driver: the Sensitivity tab
    prints the same product-form caveat as the Optimize report;
  * a run that optimized no layout (fewer than two movable items) prints
    no lift at all -- in particular none against the raw Phase-2a run
    labelled paired -- and says why in each section that would carry one.
"""

import copy
import inspect

import numpy as np
import pytest
from scipy import stats

from baselines import random_valid
from experiments._common import (HeadlessShop, anchor_base_params,
                                 base_params_for, build_headless_shop,
                                 feasible_layout, layout_score)
from experiments.closed_form import expected_revenue, mc_expected_total
from layout_comparison import direction, paired_comparison, summarize_pairs
from layout_objective import (ELASTICITY_MIDPOINTS, layout_drivers,
                              make_anchor)
from synthetic_shops import generate_synthetic_shop
from viz_optimize import OptimizeMixin
from viz_whatif import WhatIfMixin

HORIZON = 30


class _Shop(HeadlessShop, OptimizeMixin, WhatIfMixin):
    """A headless store with the Optimize and What-If mixins: everything
    the comparisons read, and no Tk."""


@pytest.fixture(scope='module')
def store():
    ss = generate_synthetic_shop(name='gui_cmp', seed=10_003, n_items=10)
    shop = build_headless_shop(ss)
    shop.__class__ = _Shop                  # add the two GUI mixins
    names = [it.name for it in ss.items]
    as_built = {n: tuple(shop.floors[1]['items'][n]['position'])
                for n in names}
    bp = anchor_base_params(shop, names, base_params_for(ss), as_built)
    # The repaired as-built layout: the one bp is anchored at.
    rep = feasible_layout(shop, names, as_built)
    other = feasible_layout(shop, names, random_valid(ss, seed=4))
    return ss, shop, names, rep, other, bp


def _kwargs(shop, names, layout, bp, n_iter, anchor=None):
    s, b = layout_score(shop, names, layout)
    return OptimizeMixin._opt_layout_mc_kwargs(
        shop, s, b, bp, n_days=HORIZON, n_iter=n_iter, anchor=anchor)


def test_arms_share_their_random_numbers(store):
    ss, shop, names, rep, other, bp = store
    kw_a = _kwargs(shop, names, rep, bp, 600)
    kw_b = _kwargs(shop, names, other, bp, 600)
    res = paired_comparison(kw_a, kw_b, seed=123)
    a, b, d = res['A']['totals'], res['B']['totals'], res['paired']['diffs']
    assert res['paired']['n_pairs'] == 600
    assert res['paired']['pair_correlation'] > 0.95
    # The paired SD is a small fraction of an arm's spread: the pairing,
    # not the iteration count, is what makes the lift resolvable.
    assert d.std() < 0.2 * a.std()
    np.testing.assert_array_equal(d, b - a)

    # Identical layouts differ by exactly zero, and no direction is named.
    same = paired_comparison(kw_a, dict(kw_a), seed=7)
    assert np.all(same['paired']['diffs'] == 0.0)
    assert same['paired']['lift_ci'] == (0.0, 0.0)
    assert same['direction'] is None

    # The seed reproduces the comparison; another seed does not.
    again = paired_comparison(kw_a, kw_b, seed=123)
    np.testing.assert_array_equal(again['paired']['diffs'], d)
    moved = paired_comparison(kw_a, kw_b, seed=124)
    assert not np.array_equal(moved['paired']['diffs'], d)


def test_chunks_are_paired_too(store):
    """The pipeline runs the projection in chunks to keep the GUI alive;
    each chunk replays one seed for both arms, so the pairs stay pairs."""
    ss, shop, names, rep, other, bp = store
    kw_a = _kwargs(shop, names, rep, bp, 100)
    kw_b = _kwargs(shop, names, other, bp, 100)
    calls = []
    res = paired_comparison(kw_a, kw_b, seed=55, n_chunks=4,
                            between_chunks=calls.append)
    assert calls == [0, 1, 2, 3]
    assert res['n_iter'] == 400 and res['paired']['n_pairs'] == 400
    assert res['paired']['pair_correlation'] > 0.95


def test_interval_is_a_t_interval_that_covers_the_exact_lift(store):
    ss, shop, names, rep, other, bp = store
    kw_a = _kwargs(shop, names, rep, bp, 2000)
    kw_b = _kwargs(shop, names, other, bp, 2000)
    res = paired_comparison(kw_a, kw_b, seed=2024)
    p = res['paired']
    d = p['diffs']
    se = d.std(ddof=1) / np.sqrt(d.size)
    assert p['lift_se'] == pytest.approx(se)
    half = p['lift_ci'][1] - p['lift_mean']
    assert half == pytest.approx(p['lift_mean'] - p['lift_ci'][0])
    assert half == pytest.approx(stats.t.ppf(0.975, d.size - 1) * se)
    # The exact lift is the expectation the estimate converges to.
    exact = mc_expected_total(**kw_b) - mc_expected_total(**kw_a)
    assert res['exact']['lift'] == pytest.approx(exact, rel=1e-12)
    assert abs(p['lift_mean'] - exact) < 4.0 * se
    # The relative interval brackets the relative estimate.
    lo, hi = p['lift_pct_ci']
    assert lo < p['lift_pct'] < hi
    assert p['lift_pct'] == pytest.approx(
        p['lift_mean'] / res['A']['totals'].mean() * 100)


def test_direction_needs_the_interval_to_exclude_zero():
    assert direction((1.0, 2.0)) == 'B'
    assert direction((-2.0, -1.0)) == 'A'
    assert direction((-1.0, 2.0)) is None
    rng = np.random.default_rng(0)
    a = rng.normal(100.0, 10.0, 50)
    s = summarize_pairs(a, a + rng.normal(0.0, 1.0, 50))
    assert s['lift_ci'][0] < 0.0 < s['lift_ci'][1]
    with pytest.raises(ValueError):
        summarize_pairs(a, a[:-1])


def test_gui_projection_agrees_with_the_headless_closed_form(store):
    """One layout pair, three paths, one exact lift: the Optimize
    pipeline's projection kwargs, the What-If tab, and the closed form the
    headless experiments score with (R42: the pair once projected 4.0% in
    the Optimize report and 5.1% in What-If)."""
    ss, shop, names, rep, other, bp = store
    s_a, b_a = layout_score(shop, names, rep)
    s_b, b_b = layout_score(shop, names, other)
    headless = (expected_revenue(bp, HORIZON, score=s_b, breakdown=b_b)
                - expected_revenue(bp, HORIZON, score=s_a, breakdown=b_a))
    assert headless != 0.0

    # The Optimize pipeline's projection path (its fitness is
    # expected_revenue itself, the closed form above).
    kw_a = _kwargs(shop, names, rep, bp, 3000)
    kw_b = _kwargs(shop, names, other, bp, 3000)
    res = paired_comparison(kw_a, kw_b, seed=99)
    assert res['exact']['lift'] == pytest.approx(headless, rel=1e-12)
    assert abs(res['paired']['lift_mean'] - headless) \
        < 4.0 * res['paired']['lift_se']

    # The What-If path: snapshots of the two layouts, anchored at the
    # layout on the floor, projected under snapshot A's inputs.
    def snapshot(layout):
        floors = copy.deepcopy(shop.floors)
        for n, pos in layout.items():
            floors[1]['items'][n]['position'] = tuple(pos)
        return {'floors': floors, 'connectors': {}, 'current_floor': 1,
                'items': floors[1]['items'], 'params': bp}

    for n, pos in rep.items():               # the floor holds layout A
        shop.floors[1]['items'][n]['position'] = tuple(pos)
    shop._snapshots = {'A': snapshot(rep), 'B': snapshot(other)}
    w = shop._whatif_comparison(n_iter=3000, n_days=HORIZON, seed=99)
    # The floor held the layout bp is anchored at, so the What-If anchor is
    # that anchor, arm A reproduces the calibrated inputs, and the lift is
    # the headless one.
    assert w['anchor']['score'] == pytest.approx(bp['score_anchor']['score'])
    d_a = w['drivers']['A']
    assert d_a['conv'] == pytest.approx(bp['conversion_rate'])
    assert d_a['rev_mult'] == pytest.approx(1.0)
    assert d_a['imp_rate'] == pytest.approx(bp['impulse_rate'])
    assert w['exact']['lift'] == pytest.approx(headless, rel=1e-9)
    assert w['scores']['B'] == pytest.approx(s_b)


def test_gui_drivers_are_the_band_midpoints(store):
    """No swing weights: the GUI transform takes no elasticities and
    returns the fitness's drivers at the band midpoints."""
    ss, shop, names, rep, other, bp = store
    params = inspect.signature(OptimizeMixin._opt_layout_drivers).parameters
    assert list(params) == ['self', 'score', 'bkd', 'p', 'anchor']
    s, b = layout_score(shop, names, other)
    got = OptimizeMixin._opt_layout_drivers(None, s, b, bp)
    assert got == layout_drivers(s, b, bp, elasticities=ELASTICITY_MIDPOINTS)


def test_pipeline_tornado_has_no_basket_bar(store):
    ss, shop, names, rep, other, bp = store
    steps = []
    t = shop._opt_sensitivity_tornado(bp, n_days=7, n_iter=50, seed=3,
                                      on_step=lambda *a: steps.append(a))
    assert set(t) == {'Customers/hr', 'Conversion rate', 'Revenue/customer',
                      'Impulse rate', 'Impulse value'}
    assert len(steps) == 2 * len(t)
    for row in t.values():
        assert row['lo_val'] < row['baseline'] < row['hi_val']
        assert row['swing'] == pytest.approx(abs(row['hi_rev']
                                                 - row['lo_rev']))
    # Common random numbers: the same seed gives the same swings.
    assert shop._opt_sensitivity_tornado(bp, n_days=7, n_iter=50,
                                         seed=3) == t


class _FakeWidget:
    """Stands in for every Tk widget the report renderers create; a Text
    keeps what is inserted into it."""

    def __init__(self, *args, **kwargs):
        self.text = []

    def __getattr__(self, name):          # pack, config, tag_configure, ...
        return lambda *a, **k: 0

    def insert(self, _index, text, *tags):
        self.text.append(text)

    def get(self, *args):
        return ''.join(self.text)


class _FakeTk:
    """The names of ``tkinter`` the renderers use, without a display."""
    END, NORMAL, DISABLED = 'end', 'normal', 'disabled'
    BOTH = RIGHT = LEFT = X = Y = WORD = ''
    Toplevel = Frame = Scrollbar = Text = Button = _FakeWidget

    def __init__(self):
        self.texts = []

    def Text(self, *a, **k):              # noqa: N802 -- mirrors tkinter
        w = _FakeWidget()
        self.texts.append(w)
        return w


def _pipeline_record(shop, names, rep, other, bp):
    """What the Optimize pipeline leaves in ``_opt_pipeline_data`` by the
    time the report is drawn, built from the same pieces it uses."""
    s_a, b_a = layout_score(shop, names, rep)
    s_b, b_b = layout_score(shop, names, other)
    projection = paired_comparison(
        _kwargs(shop, names, rep, bp, 100), _kwargs(shop, names, other, bp, 100),
        seed=1, n_chunks=4)
    ab = paired_comparison(
        _kwargs(shop, names, rep, bp, 200), _kwargs(shop, names, other, bp, 200),
        seed=2)
    ab['scores'] = {'A': s_a, 'B': s_b}
    ab['drivers'] = {'A': layout_drivers(s_a, b_a, bp),
                     'B': layout_drivers(s_b, b_b, bp)}
    return {
        'pre_snapshot': {'params': bp},
        'tornado': shop._opt_sensitivity_tornado(bp, n_days=7, n_iter=50,
                                                 seed=3),
        'markov_absorb': {'p_purchase_from_entering': 0.3,
                          'p_abandon_from_entering': 0.1,
                          'expected_steps_from_entering': 4.0,
                          'time_in_each_state': {'moving': 2.0}},
        'markov_steady': {'moving': 0.5},
        'ga_best_fit': expected_revenue(bp, HORIZON, score=s_b, breakdown=b_b),
        'ga_current_fit': expected_revenue(bp, HORIZON, score=s_a,
                                           breakdown=b_a),
        'ga_best_breakdown': b_b, 'ga_current_breakdown': b_a,
        'ga_history_best': [1.0, 2.0], 'ga_pop_size': 40, 'ga_n_gens': 20,
        'ga_fitness_kind': 'closed_form', 'ga_horizon_days': HORIZON,
        'ga_item_names': names, 'score_anchor': bp['score_anchor'],
        'mc_calibrated_baseline': projection['A'],
        'projection': projection,
        'mc_baseline': projection['A'], 'mc_optimized': projection['B'],
        'ab_comparison': ab, 'ab_score_anchor': bp['score_anchor'],
        'real_pre_rev': 123.0, 'real_post_rev': 150.0,
    }


def test_optimize_report_carries_no_test_verdicts(store, monkeypatch,
                                                  tmp_path):
    """The report renders headlessly and states the paired lift with its
    interval; the live-window lift, alpha, significance, Welch, KS and
    Cohen's d are gone, and nothing is labelled PRE vs POST."""
    import viz_optimize_results as vor
    ss, shop, names, rep, other, bp = store
    fake = _FakeTk()
    monkeypatch.setattr(vor, 'tk', fake)
    monkeypatch.chdir(tmp_path)              # the report saves a copy
    A = shop.customer_simulation.analytics
    monkeypatch.setitem(A, '_opt_pipeline_data',
                        _pipeline_record(shop, names, rep, other, bp))
    shop.tk_root = None
    vor.OptimizeResultsMixin._show_optimization_results(
        shop, ['moved x'], {'conversion_rate': 0.3, 'total_revenue': 10.0})
    text = ''.join(t.get() for t in fake.texts)
    for gone in ('Welch', 'Kolmogorov', "Cohen", 'alpha=0', 'Significant',
                 'p < ', 'Real-time revenue lift', 'PRE vs POST',
                 'sensitivity weights dynamically', 'Real-time lift'):
        assert gone not in text, gone
    assert 'Paired lift (optimized - current)' in text
    assert 'Paired lift (optimized - as-built)' in text
    assert '95% interval' in text and 'Exact (model)' in text
    assert 'No lift is computed from these two windows' in text
    assert 'Basket size' not in text
    assert list(tmp_path.glob('optimization_report_*.txt'))


def _raw_phase_2a_baseline(bp):
    """The Phase-2a Monte Carlo run on the RAW calibrated parameters, which
    the pipeline stores under ``pipe['mc_baseline']`` before the GA and
    overwrites with the projection's current-layout arm only if the GA
    runs."""
    from retail_literature import (DEFAULT_OP_HOURS_PER_DAY,
                                   DEFAULT_WEEKEND_MULTIPLIER)
    from sim_calibration import mc_engine
    return mc_engine(
        cph=bp['customers_per_hour'], conv=bp['conversion_rate'],
        rev_mean=bp['rev_per_converting_customer'], rev_std=bp['rev_std'],
        imp_rate=bp['impulse_rate'], imp_val=bp['avg_impulse_value'],
        avg_bsk=bp['avg_basket_size'], std_bsk=bp['std_basket_size'],
        observed_baskets=bp['basket_sizes_observed'], n_days=HORIZON,
        n_iter=100, op_hours=DEFAULT_OP_HOURS_PER_DAY,
        wknd_mult=DEFAULT_WEEKEND_MULTIPLIER, monthly_growth=0.0,
        rng=np.random.default_rng(0))


@pytest.mark.parametrize('recorded_reason', [True, False])
def test_optimize_report_without_a_comparison_prints_no_lift(
        store, monkeypatch, tmp_path, recorded_reason):
    """A run with fewer than two movable items skips the GA and goes
    straight to the POST window. Its record holds the Phase-2a run on the
    RAW calibrated parameters under 'mc_baseline', the tornado and the
    Markov results, and no projection or re-scored comparison. The report
    says why in every section that would carry a lift and prints none:
    not the GA's absent fitness ("+0.00%, $0 -> $0"), not a lift taken
    against that raw run and labelled paired (it once printed "-100.00%"
    as the "Monte Carlo check (paired draws)"), and no layout
    recommendation drawn from an optimized breakdown that does not
    exist."""
    import re
    import viz_optimize_results as vor
    ss, shop, names, rep, other, bp = store
    fake = _FakeTk()
    monkeypatch.setattr(vor, 'tk', fake)
    monkeypatch.chdir(tmp_path)
    full = _pipeline_record(shop, names, rep, other, bp)
    raw = _raw_phase_2a_baseline(bp)
    assert raw['mean'] > 0
    pipe = {k: full[k] for k in ('pre_snapshot', 'tornado',
                                 'markov_absorb', 'markov_steady')}
    pipe.update({'mc_baseline': raw, 'ab_comparison': None,
                 'real_pre_rev': 123.0, 'real_post_rev': 150.0})
    if recorded_reason:
        # What the pipeline's no-GA branch records.
        pipe['not_optimized_reason'] = OptimizeMixin._OPT_TOO_FEW_ITEMS
        reason = ('not run: fewer than two movable items, no layout was '
                  'optimized')
    else:
        reason = 'not run: no layout was optimized'
    monkeypatch.setitem(shop.customer_simulation.analytics,
                        '_opt_pipeline_data', pipe)
    shop.tk_root = None
    vor.OptimizeResultsMixin._show_optimization_results(
        shop, [], {'conversion_rate': 0.3, 'total_revenue': 10.0})
    text = ''.join(t.get() for t in fake.texts)

    for said in (f'Projected {HORIZON}-day lift:      {reason}',   # Sec. 1
                 f'GA search {reason}.',                            # Sec. 5
                 f'No items were moved: the GA search was {reason}.',
                 f'Projection {reason}.',                           # Sec. 7
                 f'Re-scored comparison {reason}.',                 # Sec. 8
                 f'No layout recommendation: the GA search was {reason}.',
                 f'No layout effect was estimated: the GA search was '
                 f'{reason}.',                                      # Sec. 9
                 f'No improvement is expected: the GA search was {reason}.'):
        assert said in text, said
    # No signed percentage anywhere: every lift prints as one.
    signed = re.search(r'.*[+-]\d+\.\d+%.*', text)
    assert signed is None, signed.group(0)
    for gone in ('paired draws', 'Paired lift', 'Exact (model)', '$0 -> $0',
                 f'Mean {HORIZON}-day revenue', 'insufficient snapshot data',
                 'layout already optimal', 'Improve section compliance',
                 'Strengthen cross-merchandising', 'Reposition impulse items',
                 'appears well-optimized', 'Composite layout quality',
                 'The layout effect is the paired comparison'):
        assert gone not in text, gone
    # The raw run is still reported where it belongs, as Sec. 2.2's
    # baseline on the raw calibrated parameters.
    assert f"${raw['mean']:>12,.2f}" in text


def test_whatif_report_carries_no_test_verdicts(store):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure
    ss, shop, names, rep, other, bp = store
    for n, pos in rep.items():
        shop.floors[1]['items'][n]['position'] = tuple(pos)
    floors_b = copy.deepcopy(shop.floors)
    for n, pos in other.items():
        floors_b[1]['items'][n]['position'] = tuple(pos)
    shop._snapshots = {
        'A': {'floors': copy.deepcopy(shop.floors), 'connectors': {},
              'current_floor': 1, 'params': bp},
        'B': {'floors': floors_b, 'connectors': {}, 'current_floor': 1,
              'params': bp}}
    shop._ab_text = _FakeWidget()
    shop._ab_fig = Figure()
    shop._ab_canvas_fig = _FakeWidget()
    shop._display_ab_results(shop._whatif_comparison(500, HORIZON, seed=8))
    text = shop._ab_text.get()
    for gone in ('Welch', 'KS ', "Cohen", 'alpha', 'p-value', 'P(B > A)'):
        assert gone not in text, gone
    assert 'PAIRED LIFT' in text and '95% interval' in text
    assert len(shop._ab_fig.axes) == 4


def test_an_explicit_anchor_reaches_the_projection(store):
    """Anchored at layout B, B reproduces the calibrated inputs."""
    ss, shop, names, rep, other, bp = store
    s_b, b_b = layout_score(shop, names, other)
    anc = make_anchor(s_b, b_b, source='test')
    kw_b = _kwargs(shop, names, other, bp, 10, anchor=anc)
    assert kw_b['conv'] == pytest.approx(bp['conversion_rate'])
    assert kw_b['rev_mean'] == pytest.approx(
        bp['rev_per_converting_customer'])


def test_sensitivity_tab_names_no_top_driver(monkeypatch):
    """The Sensitivity tab's interpretation carries the product-form caveat
    the Optimize report prints and names no "largest" or "top 3" driver:
    customers/hr, conversion, revenue per customer and operating hours
    each move revenue by about the perturbation, so their order is not a
    finding about the store."""
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure
    import viz_sensitivity as vs

    class _Tab(vs.SensitivityMixin):
        pass

    tab = _Tab()
    tab._sa_text = _FakeWidget()
    tab._sa_fig = Figure()
    tab._sa_canvas = _FakeWidget()
    monkeypatch.setattr(vs, 'tk', _FakeTk())
    base = 1000.0
    tornado = {
        label: {'baseline': 1.0, 'lo_val': 0.8, 'hi_val': 1.2,
                'lo_rev': base - sw / 2, 'hi_rev': base + sw / 2,
                'swing': sw}
        for label, sw in (('Customers/hr', 401.0), ('Conversion rate', 399.0),
                          ('Revenue/customer', 398.0), ('Op. hours', 400.0),
                          ('Impulse rate', 12.0))}
    spider = {label: ([-20.0, 0.0, 20.0], [d['lo_rev'], base, d['hi_rev']])
              for label, d in tornado.items()}
    tab._display_sensitivity_results(tornado, spider, base, 0.2, 7, 100)
    text = tab._sa_text.get()
    assert 'largest impact' not in text
    assert 'Top 3 drivers' not in text
    assert 'not a finding about this store' in text
    assert 'buyers / assumed conversion' in text
    assert text.count('INTERPRETATION') == 1
    assert vs.interpretation_lines(0.2) == \
        text[text.index('INTERPRETATION'):].split('\n')
