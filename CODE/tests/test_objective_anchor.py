"""The layout objective is anchored at the store as built (review R03).

The elasticities turn a layout's score into conversion, basket and impulse
lifts. They act on the score DIFFERENCE from the anchor layout -- the
as-built layout after the shared repair -- so that layout reproduces the
calibrated inputs exactly. These tests pin, through the production
``_ga_fitness`` rather than ``mc_engine`` directly:

  * on a calibrated store, the as-built layout hands the engine the
    calibrated conversion, basket, spend and impulse rate, and the engine
    run through the fitness reproduces the observed invoices and spend per
    calendar day;
  * a criterion that is constant over the repaired layouts (section
    compliance) cancels exactly: re-weighting it moves every absolute score
    and no driver;
  * the GUI's transform (Optimize pipeline, What-If tab) is the fitness's
    own at the same elasticities, and both scale the spend SD with the
    spend (R62); the Optimize pipeline's PRE/POST A/B anchors both arms at
    the PRE layout scored under the same analytics as they are;
  * every runner that builds base parameters anchors them, and every entry
    point refuses parameters without an anchor unless the zero anchor is
    asked for explicitly.
"""

import inspect
import os

import numpy as np
import pandas as pd
import pytest

import retail_literature as RL
from baselines import random_valid
from dataset_calibration import calibrate_transactional
from experiments._common import (anchor_base_params, base_params_for,
                                 base_params_for_calibration,
                                 base_params_record, build_headless_shop,
                                 build_headless_shop_from_calibration,
                                 feasible_layout, layout_score,
                                 layout_to_chromosome)
from layout_objective import (ELASTICITY_MIDPOINTS, SCORE_ANCHOR_KEY,
                              MissingAnchorError, layout_drivers,
                              layout_mc_kwargs, make_anchor, with_zero_anchor,
                              zero_anchor)
from synthetic_shops import generate_synthetic_shop

WEEK_MULT = (5.0 + 2.0 * RL.DEFAULT_WEEKEND_MULTIPLIER) / 7.0


def _invoices(n_invoices=600, seed=3):
    """Invoices over four categories and 15 calendar days, 1-4 products each."""
    rng = np.random.default_rng(seed)
    products = [(f'{cat[:3].upper()}{i}', cat, 1.5 + i)
                for cat in ('Bakery', 'Dairy', 'Garden', 'Kitchen')
                for i in range(4)]
    day0 = pd.Timestamp('2010-12-01 09:00')
    rows = []
    for k in range(n_invoices):
        ts = day0 + pd.Timedelta(days=k // 40,
                                 minutes=int(rng.integers(0, 480)))
        for j in rng.choice(len(products), size=int(rng.integers(1, 5)),
                            replace=False):
            pid, cat, price = products[j]
            rows.append((f'I{k:05d}', pid, f'ITEM {pid}',
                         int(rng.integers(1, 4)), ts, price, cat))
    return pd.DataFrame(rows, columns=['invoice_id', 'product_id',
                                       'product_name', 'quantity',
                                       'timestamp', 'unit_price', 'category'])


@pytest.fixture(scope='module')
def calibrated():
    df = _invoices()
    params = calibrate_transactional(df, currency='GBP')
    shop = build_headless_shop_from_calibration(
        params, max_items_per_category=4, naive=True)
    names = list(shop.floors[1]['items'])
    as_built = {n: tuple(shop.floors[1]['items'][n]['position'])
                for n in names}
    bp = anchor_base_params(shop, names, base_params_for_calibration(params),
                            as_built)
    dates = df['timestamp'].dt.normalize()
    days = int((dates.max() - dates.min()).days) + 1
    return params, shop, names, as_built, bp, days


@pytest.fixture(scope='module')
def synthetic():
    ss = generate_synthetic_shop(name='anchor', seed=10_000, n_items=10)
    shop = build_headless_shop(ss)
    names = [it.name for it in ss.items]
    as_built = {n: tuple(shop.floors[1]['items'][n]['position'])
                for n in names}
    bp = anchor_base_params(shop, names, base_params_for(ss), as_built)
    return ss, shop, names, as_built, bp


def _spy(shop):
    """Record what ``_ga_fitness`` hands the engine, and what it returns."""
    real = shop._mc_engine
    seen = {}

    def spy(**kw):
        res = real(**kw)
        seen['kw'], seen['res'] = kw, res
        return res
    shop._mc_engine = spy
    return seen


def test_anchor_is_the_repaired_as_built_layout(synthetic):
    ss, shop, names, as_built, bp = synthetic
    anc = bp[SCORE_ANCHOR_KEY]
    s, b = layout_score(shop, names, feasible_layout(shop, names, as_built))
    assert anc['score'] == s
    assert anc['breakdown']['flow'] == b['flow']
    assert anc['source'] == 'as_built_repaired'
    # The input parameters are not modified.
    assert SCORE_ANCHOR_KEY not in base_params_for(ss)


def test_anchor_layout_reproduces_the_calibrated_drivers_exactly(synthetic):
    _, shop, names, as_built, bp = synthetic
    anc = bp[SCORE_ANCHOR_KEY]
    d = layout_drivers(anc['score'], anc['breakdown'], bp)
    assert d['conv'] == bp['conversion_rate']
    assert d['imp_rate'] == bp['impulse_rate']
    assert d['avg_bsk'] == bp['avg_basket_size']
    assert d['rev_mult'] == 1.0


def test_as_built_calibrated_store_reproduces_its_inputs_through_the_fitness(
        calibrated):
    params, shop, names, as_built, bp, _ = calibrated
    lay = feasible_layout(shop, names, as_built)
    seen = _spy(shop)
    try:
        shop._ga_fitness(layout_to_chromosome(lay, names), names, bp,
                         mc_days=7, mc_iters=10)
    finally:
        del shop._mc_engine
    kw = seen['kw']
    assert kw['conv'] == pytest.approx(params.assumed_conversion_rate,
                                       rel=0, abs=1e-15)
    assert kw['avg_bsk'] == pytest.approx(float(np.mean(params.basket_sizes)),
                                          rel=1e-15)
    assert kw['rev_mean'] == pytest.approx(bp['rev_per_converting_customer'],
                                           rel=1e-15)
    assert kw['rev_std'] == pytest.approx(bp['rev_std'], rel=1e-15)
    assert kw['imp_rate'] == pytest.approx(RL.IMPULSE_RATE_STANDIN, rel=1e-15)
    # Net base spend plus the stand-in impulse spend is the observed gross
    # spend per invoice.
    gross = float(np.mean(params.invoice_revenues))
    assert (kw['rev_mean'] + kw['imp_rate'] * kw['imp_val']
            == pytest.approx(gross, rel=1e-12))


def test_as_built_calibrated_store_reproduces_observed_volume_and_spend(
        calibrated):
    """End to end through ``_ga_fitness``: the engine it runs gives back the
    dataset's invoices and spend per calendar day on the as-built layout
    (four whole weeks, so the weekend lift averages out as in the data)."""
    params, shop, names, as_built, bp, days = calibrated
    lay = feasible_layout(shop, names, as_built)
    seen = _spy(shop)
    np.random.seed(11)
    try:
        mean = shop._ga_fitness(layout_to_chromosome(lay, names), names, bp,
                                mc_days=28, mc_iters=3000)
    finally:
        del shop._mc_engine
    raw = seen['res']['raw']
    observed_invoices = params.n_invoices / days
    observed_spend = float(np.sum(params.invoice_revenues)) / days
    simulated_invoices = raw['daily_converting'].mean()
    assert simulated_invoices == pytest.approx(observed_invoices, rel=0.02)
    assert mean / 28.0 == pytest.approx(observed_spend, rel=0.02)


def test_the_old_zero_anchor_lifted_the_as_built_store(calibrated):
    """What the anchoring fixes: under the zero anchor the same store's own
    layout is scored as a lift over the data that describe it."""
    params, shop, names, as_built, bp, _ = calibrated
    s0 = bp[SCORE_ANCHOR_KEY]['score']
    assert s0 > 0.1
    bp_zero = with_zero_anchor(bp)
    b0 = bp[SCORE_ANCHOR_KEY]['breakdown']
    d_zero = layout_drivers(s0, b0, bp_zero)
    assert d_zero['conv_lifted'] > 1.05 * params.assumed_conversion_rate
    assert layout_drivers(s0, b0, bp)['conv_lifted'] == \
        bp['conversion_rate']


def test_constant_section_compliance_cancels(synthetic, monkeypatch):
    """Section compliance is 1 on every repaired layout, so its weight shifts
    every absolute score by the same amount. Anchored, the drivers do not
    move when that weight changes; with the zero anchor they did."""
    ss, shop, names, as_built, bp = synthetic
    layouts = [feasible_layout(shop, names, random_valid(ss, seed=k))
               for k in range(4)]
    base = base_params_for(ss)

    def drivers_under(weight):
        monkeypatch.setattr(RL, 'GA_W_SECTION_COMPLIANCE', weight)
        anchored = anchor_base_params(shop, names, base, as_built)
        out = []
        for lay in layouts:
            s, b = layout_score(shop, names, lay)
            assert b['section_compliance'] == 1.0
            out.append((s, layout_drivers(s, b, anchored),
                        layout_drivers(s, b, base, allow_zero_anchor=True)))
        return out

    nominal = drivers_under(RL.GA_W_SECTION_COMPLIANCE)
    shifted = drivers_under(RL.GA_W_SECTION_COMPLIANCE + 0.5)
    for (s_a, anc_a, zero_a), (s_b, anc_b, zero_b) in zip(nominal, shifted):
        assert s_b == pytest.approx(s_a + 0.5, abs=1e-12)
        for key in ('conv', 'imp_rate', 'avg_bsk', 'rev_mult'):
            assert anc_b[key] == pytest.approx(anc_a[key], rel=0, abs=1e-12)
        assert zero_b['conv'] > zero_a['conv']


def test_gui_transform_is_the_fitness_transform(synthetic):
    """The Optimize pipeline and the What-If tab call
    ``_opt_layout_drivers``: the drivers ``_ga_fitness`` uses, at the band
    midpoints, anchor and queue channel included."""
    from viz_optimize import OptimizeMixin
    ss, shop, names, as_built, bp = synthetic
    lay = feasible_layout(shop, names, random_valid(ss, seed=9))
    s, b = layout_score(shop, names, lay)
    b = dict(b, bottleneck_penalty=0.3)          # exercise the queue channel
    got = OptimizeMixin._opt_layout_drivers(None, s, b, bp)
    d = layout_drivers(s, b, bp)
    for key in ('conv', 'imp_rate', 'avg_bsk', 'rev_mult'):
        assert got[key] == pytest.approx(d[key], rel=1e-14)
    assert d['queue_penalty'] < 1.0
    # An explicit anchor overrides the one the parameters carry.
    other = make_anchor(s, b)
    got_o = OptimizeMixin._opt_layout_drivers(None, s, b, bp, anchor=other)
    assert got_o['conv'] == pytest.approx(bp['conversion_rate'])
    assert got_o['rev_mult'] == pytest.approx(1.0)


def test_spend_sd_scales_with_spend_on_every_path(synthetic):
    ss, shop, names, as_built, bp = synthetic
    lay = feasible_layout(shop, names, random_valid(ss, seed=5))
    s, b = layout_score(shop, names, lay)
    d = layout_drivers(s, b, bp)
    kw = layout_mc_kwargs(d, bp, n_days=7, n_iter=3)
    assert kw['rev_mean'] / bp['rev_per_converting_customer'] == \
        pytest.approx(d['rev_mult'])
    assert kw['rev_std'] / bp['rev_std'] == pytest.approx(d['rev_mult'])
    assert kw['op_hours'] == RL.DEFAULT_OP_HOURS_PER_DAY
    assert kw['wknd_mult'] == RL.DEFAULT_WEEKEND_MULTIPLIER


def test_midpoints_are_the_band_midpoints():
    for key, (lo, hi) in (('conv', (RL.ELASTICITY_CONV_BASE,
                                    RL.ELASTICITY_CONV_BASE
                                    + RL.ELASTICITY_CONV_GAIN_MAX)),
                          ('imp', (RL.ELASTICITY_IMP_BASE,
                                   RL.ELASTICITY_IMP_BASE
                                   + RL.ELASTICITY_IMP_GAIN_MAX)),
                          ('bsk', (RL.ELASTICITY_BSK_BASE,
                                   RL.ELASTICITY_BSK_BASE
                                   + RL.ELASTICITY_BSK_GAIN_MAX))):
        assert ELASTICITY_MIDPOINTS[key] == pytest.approx(0.5 * (lo + hi))


def test_basket_exclusion_touches_only_the_basket(synthetic):
    ss, shop, names, as_built, bp = synthetic
    lay = feasible_layout(shop, names, random_valid(ss, seed=6))
    s, b = layout_score(shop, names, lay)
    full = layout_drivers(s, b, bp)
    short = layout_drivers(s, b, bp, basket_exclude=('flow', 'accessibility'))
    assert short['conv'] == full['conv']
    assert short['imp_rate'] == full['imp_rate']
    b0 = bp[SCORE_ANCHOR_KEY]['breakdown']
    want = full['score_delta'] - (
        RL.GA_W_FLOW * (b['flow'] - b0['flow'])
        + RL.GA_W_ACCESSIBILITY * (b['accessibility'] - b0['accessibility']))
    assert short['basket_score_delta'] == pytest.approx(want, abs=1e-12)


def test_base_params_record_is_compact_and_keeps_the_anchor(calibrated):
    params, shop, names, as_built, bp, _ = calibrated
    rec = base_params_record(bp)
    assert rec[SCORE_ANCHOR_KEY] == bp[SCORE_ANCHOR_KEY]
    obs = rec['basket_sizes_observed']
    assert obs['n'] == len(bp['basket_sizes_observed'])
    assert obs['mean'] == pytest.approx(np.mean(bp['basket_sizes_observed']))
    assert len(obs['sha256']) == 64
    assert rec['conversion_rate'] == bp['conversion_rate']


RUNNERS_THAT_BUILD_BASE_PARAMS = (
    'run_synthetic_gt.py', 'run_baseline_comparison.py',
    'run_mc_groundtruth.py', 'run_ga_sensitivity.py',
    'run_elasticity_lhs.py', 'run_real_data_example.py',
    'run_objective_alignment.py', 'make_paper_figures.py')


@pytest.mark.parametrize('script', RUNNERS_THAT_BUILD_BASE_PARAMS)
def test_every_runner_anchors_its_base_params(script):
    import experiments
    path = os.path.join(os.path.dirname(experiments.__file__), script)
    src = open(path, encoding='utf-8').read()
    assert 'anchor_base_params(' in src, script


def test_fitness_reads_the_shared_transform():
    from viz_ga_run import GARunMixin
    src = inspect.getsource(GARunMixin._ga_fitness)
    assert 'layout_drivers' in src and 'layout_mc_kwargs' in src
    assert 'score *' not in src


# --- no silent zero anchor ------------------------------------------------

def test_every_entry_point_refuses_parameters_without_an_anchor(synthetic):
    """Parameters rebuilt by ``base_params_for`` carry no anchor. Re-scoring
    a layout with them must fail loudly, not fall back to the absolute
    (zero-anchor) objective."""
    from experiments._common import paired_mc_revenue, run_ga_headless
    from experiments.closed_form import expected_revenue
    ss, shop, names, as_built, bp = synthetic
    bare = base_params_for(ss)
    lay = feasible_layout(shop, names, as_built)
    s, b = layout_score(shop, names, lay)
    chrom = layout_to_chromosome(lay, names)
    with pytest.raises(MissingAnchorError):
        layout_drivers(s, b, bare)
    with pytest.raises(MissingAnchorError):
        expected_revenue(bare, 7, score=s, breakdown=b)
    with pytest.raises(MissingAnchorError):
        shop._ga_fitness(chrom, names, bare, mc_days=2, mc_iters=2)
    before = {n: shop.floors[1]['items'][n]['position'] for n in names}
    with pytest.raises(MissingAnchorError):
        paired_mc_revenue(shop, names, lay, bare, seed=0, mc_iters=2,
                          mc_days=2)
    # Refused before the shop was touched.
    assert {n: shop.floors[1]['items'][n]['position']
            for n in names} == before
    with pytest.raises(MissingAnchorError):
        run_ga_headless(shop, names, bare, pop_size=2, n_gens=1,
                        mc_iters=2, mc_days=2)
    # A ValueError, so callers that guard numeric input keep working.
    assert issubclass(MissingAnchorError, ValueError)


def test_the_zero_anchor_is_an_explicit_opt_in(synthetic):
    """Three explicit ways into the zero anchor, all the same objective."""
    from experiments.closed_form import expected_revenue
    ss, shop, names, as_built, bp = synthetic
    bare = base_params_for(ss)
    s, b = layout_score(shop, names,
                        feasible_layout(shop, names, random_valid(ss, seed=2)))
    by_flag = layout_drivers(s, b, bare, allow_zero_anchor=True)
    by_record = layout_drivers(s, b, with_zero_anchor(bare))
    by_arg = layout_drivers(s, b, bare, anchor=zero_anchor())
    assert by_flag == by_record == by_arg
    assert by_flag['anchor_score'] == 0.0
    assert by_flag['score_delta'] == s
    assert with_zero_anchor(bare)[SCORE_ANCHOR_KEY]['source'] == 'zero'
    assert SCORE_ANCHOR_KEY not in bare             # input untouched
    assert expected_revenue(bare, 7, score=s, breakdown=b,
                            allow_zero_anchor=True) == \
        expected_revenue(with_zero_anchor(bare), 7, score=s, breakdown=b)


# --- the Optimize pipeline's PRE/POST A/B ---------------------------------

class _DriftingAnalytics:
    """Stands in for the visualizer in ``OptimizeMixin._ab_arm_scores``: each
    call of ``_ab_layout_score`` sees the analytics a little further on, as
    the live POST window keeps adding to the heat map and the bottleneck
    counts."""

    def __init__(self, layouts):
        self.layouts = layouts            # snapshot label -> (score, bkd)
        self.drift = 0.0

    def _ab_layout_score(self, snap, with_breakdown=False):
        self.drift += 0.01
        s, b = self.layouts[snap['label']]
        b = dict(b, traffic=b['traffic'] + self.drift,
                 bottleneck_penalty=b['bottleneck_penalty'] + self.drift)
        s = s + self.drift * (RL.GA_W_TRAFFIC - RL.GA_PEN_BOTTLENECK)
        return (s, b) if with_breakdown else s


def test_ab_arms_are_anchored_at_pre_under_the_same_analytics(synthetic):
    """Phase 5 scores PRE and POST after the POST window. Anchored at the
    optimize-start score, PRE would no longer reproduce the calibrated
    inputs (its heat-map and bottleneck criteria have drifted); anchored at
    PRE re-scored with the arms, arm A reproduces them exactly."""
    from viz_optimize import OptimizeMixin
    ss, shop, names, as_built, bp = synthetic
    pre = layout_score(shop, names, feasible_layout(shop, names, as_built))
    post = layout_score(shop, names,
                        feasible_layout(shop, names, random_valid(ss, seed=3)))
    fake = _DriftingAnalytics({'pre': pre, 'post': post})
    start_anchor = make_anchor(*pre, source='floor_at_optimize_start')
    pipe = {'score_anchor': start_anchor,
            'pre_snapshot': {'label': 'pre', 'params': bp},
            'post_snapshot': {'label': 'post', 'params': bp}}

    scores, anchor = OptimizeMixin._ab_arm_scores(fake, pipe)
    assert set(scores) == {'A', 'B'}
    assert anchor['score'] == scores['A'][0]
    assert anchor['source'] == 'pre_snapshot_at_ab'
    d_a = OptimizeMixin._opt_layout_drivers(None, *scores['A'], bp,
                                            anchor=anchor)
    assert d_a['conv'] == bp['conversion_rate']
    assert d_a['imp_rate'] == bp['impulse_rate']
    assert d_a['avg_bsk'] == bp['avg_basket_size']
    assert d_a['rev_mult'] == 1.0
    # What the optimize-start anchor would have done: arm A moved off the
    # calibrated inputs by the analytics drift alone.
    d_old = OptimizeMixin._opt_layout_drivers(None, *scores['A'], bp,
                                              anchor=start_anchor)
    assert (d_old['conv'], d_old['rev_mult']) != (bp['conversion_rate'], 1.0)
    # The pipeline's own anchor is left for the GA and the projection.
    assert pipe['score_anchor'] is start_anchor


def test_ab_anchor_falls_back_without_a_pre_snapshot():
    from viz_optimize import OptimizeMixin
    fake = _DriftingAnalytics({'post': (0.5, {'traffic': 0.1,
                                              'bottleneck_penalty': 0.0})})
    start = make_anchor(0.4, {'traffic': 0.1})
    scores, anchor = OptimizeMixin._ab_arm_scores(
        fake, {'score_anchor': start,
               'post_snapshot': {'label': 'post', 'params': {}}})
    # A snapshot without parameters is not an arm.
    assert scores == {} and anchor is start
