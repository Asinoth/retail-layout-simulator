"""The Monte Carlo engine's day-spend law at Online Retail II's spread
(review R27).

A day's spend over its converters is one non-negative lognormal total with
the exact mean and SD of a sum of independent per-converter spends
(``sim_calibration.lognormal_spend_total``, retail_literature
MC_SPEND_LAW). Online Retail II's invoices have a coefficient of variation
of about 2.3 after the reversal pairs are removed and 3.4 before, where the
former law -- a normal floored at zero -- was biased on thin days and fell
as a converter was added. These tests hold the new law to what it claims
at that spread: exact moments, an unbiased engine, and common random
numbers that respond to a layout in the direction it moves the inputs.
"""

import numpy as np
import pytest

import retail_literature as RL
import sim_calibration as SC
from experiments.closed_form import legacy_floor_bias, mc_expected_total
from sim_calibration import lognormal_spend_total, mc_engine

UCI_CVS = (2.3, 3.0, 3.4)

# Gauss-Hermite rule for expectations over the standard normal z.
_GH_X, _GH_W = np.polynomial.hermite.hermgauss(160)
_Z, _W = np.sqrt(2.0) * _GH_X, _GH_W / np.sqrt(np.pi)


def _kw(**over):
    kw = dict(cph=1.2, conv=0.3, rev_mean=480.0, rev_std=3.0 * 480.0,
              imp_rate=0.0, imp_val=75.0, avg_bsk=3.0, std_bsk=1.0,
              observed_baskets=None, n_days=30,
              op_hours=RL.DEFAULT_OP_HOURS_PER_DAY,
              wknd_mult=RL.DEFAULT_WEEKEND_MULTIPLIER, monthly_growth=0.0)
    kw.update(over)
    return kw


def test_the_registry_names_the_law_the_engine_draws():
    assert RL.MC_SPEND_LAW == 'lognormal_moment_matched'
    assert SC._SPEND_TOTALS[RL.MC_SPEND_LAW] is lognormal_spend_total
    assert RL.MC_SPEND_LAW_CITE in RL.CITATIONS


@pytest.mark.parametrize('cv', UCI_CVS + (0.35,))
@pytest.mark.parametrize('n', (1, 2, 5, 40, 400))
def test_total_has_the_exact_mean_and_sd_of_the_sum(n, cv):
    mean, sd = 480.0, cv * 480.0
    x = lognormal_spend_total(np.full(_Z.size, n), mean, sd, _Z)
    m1 = float(_W @ x)
    m2 = float(_W @ (x * x))
    assert m1 == pytest.approx(n * mean, rel=1e-9)
    assert np.sqrt(m2 - m1 * m1) == pytest.approx(np.sqrt(n) * sd, rel=1e-6)
    assert (x >= 0.0).all()


def test_no_spenders_or_no_spend_gives_zero():
    z = np.linspace(-5, 5, 11)
    assert (lognormal_spend_total(np.zeros(11), 10.0, 30.0, z) == 0.0).all()
    assert (lognormal_spend_total(np.full(11, 3), 0.0, 30.0, z) == 0.0).all()
    assert (lognormal_spend_total(np.full(11, 3), -1.0, 30.0, z) == 0.0).all()


@pytest.mark.parametrize('cv', UCI_CVS)
def test_total_rises_with_the_converter_count(cv):
    """Within |z| <= 4 (all but 6e-5 of draws) one more converter never
    lowers the day's spend; the former normal fell for z < -2 sqrt(n)/cv."""
    z = np.linspace(-4.0, 4.0, 161)
    n = np.arange(0, 301, dtype=float)
    x = lognormal_spend_total(n[:, None], 480.0, cv * 480.0, z[None, :])
    assert (np.diff(x, axis=0) >= 0.0).all()
    old = n[:, None] * 480.0 + np.sqrt(n[:, None]) * cv * 480.0 * z[None, :]
    assert (np.diff(old, axis=0) < 0.0).any()


@pytest.mark.parametrize('cv', UCI_CVS)
def test_engine_mean_is_exact_where_the_floor_biased_it(cv):
    kw = _kw(rev_std=cv * 480.0, imp_rate=0.2)
    n = 60_000
    res = mc_engine(**kw, n_iter=n, rng=np.random.default_rng(31))
    exact = mc_expected_total(**kw)
    se = res['std'] / np.sqrt(n)
    assert abs(res['mean'] - exact) <= 4.0 * se, (res['mean'], exact, se)
    # The test can see the bias it rules out: the former law sat far
    # outside this tolerance.
    assert legacy_floor_bias(**kw) > 10.0 * se
    assert (res['daily_revenue'] >= 0.0).all()


def test_engine_day_spend_has_the_sum_variance():
    """With no day noise and no impulse, a day's spend is a compound sum
    over n ~ Poisson(lam p) converters: variance lam p (sd^2 + mean^2)."""
    kw = _kw(cph=0.5, rev_std=2.3 * 480.0, day_noise_std=0.0, wknd_mult=1.0)
    res = mc_engine(**kw, n_iter=40_000, rng=np.random.default_rng(5))
    lam_p = kw['cph'] * kw['op_hours'] * kw['conv']
    day = res['daily_revenue'].ravel()
    assert day.mean() == pytest.approx(lam_p * kw['rev_mean'], rel=0.02)
    assert day.var() == pytest.approx(
        lam_p * (kw['rev_std'] ** 2 + kw['rev_mean'] ** 2), rel=0.06)


def test_a_layout_factor_scales_every_draw_exactly():
    """A layout moves spend per converter and its SD by one factor
    (layout_mc_kwargs); under the same seed every day's spend moves by
    exactly that factor, so the paired difference has the sign of the
    change on every iteration."""
    base = mc_engine(**_kw(), n_iter=3000, rng=np.random.default_rng(2))
    kw = _kw()
    up = mc_engine(**_kw(rev_mean=kw['rev_mean'] * 1.07,
                         rev_std=kw['rev_std'] * 1.07),
                   n_iter=3000, rng=np.random.default_rng(2))
    np.testing.assert_allclose(up['daily_revenue'],
                               1.07 * base['daily_revenue'], rtol=1e-12)
    assert (up['totals'] > base['totals'])[base['totals'] > 0].all()


@pytest.mark.parametrize('cv', UCI_CVS)
def test_more_conversion_lowers_a_day_only_in_the_laws_rare_tail(cv):
    """Under common numbers a higher conversion rate gives every day at
    least as many converters (inverse-CDF counts) and raises the mean. A
    day's spend can still fall when a converter is added: the law's draw
    decreases in n where z > 4.14 on days of a few dozen spenders or fewer
    (``lognormal_spend_total``), about one day in a million here. The test
    therefore pools several seeds and bounds that share, and checks that
    only thin days fall -- properties of the law -- instead of asserting
    every day of one pinned seed, which fails for about one seed in ten."""
    falls, days, fall_n = 0, 0, []
    for seed in range(6):
        low = mc_engine(**_kw(rev_std=cv * 480.0), n_iter=4000,
                        rng=np.random.default_rng(900 + seed))
        high = mc_engine(**_kw(rev_std=cv * 480.0, conv=0.31), n_iter=4000,
                         rng=np.random.default_rng(900 + seed))
        assert (high['daily_converting'] >= low['daily_converting']).all()
        assert high['mean'] > low['mean']
        fell = high['daily_revenue'] < low['daily_revenue']
        # A day whose converter count did not move keeps its draw exactly.
        same = high['daily_converting'] == low['daily_converting']
        assert (high['daily_revenue'][same] == low['daily_revenue'][same]).all()
        falls += int(fell.sum())
        days += fell.size
        fall_n.extend(low['daily_converting'][fell].tolist())
    assert falls / days < 1e-4
    # For z <= 6 (all but 1e-9 of draws) no day of more than 72 spenders
    # (cv 3.4; fewer at smaller cv) falls.
    assert all(n <= 72 for n in fall_n)


@pytest.mark.parametrize('cv', UCI_CVS)
def test_the_draw_falls_in_n_only_above_z_4_14_on_thin_days(cv):
    """Where ``lognormal_spend_total`` is not increasing in n: only for
    z > 4.14 (probability 1.7e-5), and for z <= 6 only from n <= 72 spenders
    at cv 3.4 (56 at cv 3, 33 at cv 2.3). The bound on n grows with z (about
    150 at z = 8), which no Monte Carlo run reaches."""
    n = np.arange(0, 401, dtype=float)
    z = np.linspace(-8.0, 4.14, 1215)
    x = lognormal_spend_total(n[:, None], 480.0, cv * 480.0, z[None, :])
    assert (np.diff(x, axis=0) >= 0.0).all()
    z_hi = np.linspace(4.14, 6.0, 187)
    x_hi = lognormal_spend_total(n[:, None], 480.0, cv * 480.0,
                                 z_hi[None, :])
    falls = np.diff(x_hi, axis=0) < 0.0
    assert falls.any()
    assert np.flatnonzero(falls.any(axis=1)).max() <= 72


def test_paired_noise_stays_small_at_the_uci_spread():
    """Paired differences between nearby conversion rates are dominated by
    the change, not by the sampler, at CV 3 as at the synthetic spread."""
    diffs, levels = [], []
    for s in range(16):
        a = mc_engine(**_kw(cph=4.0), n_iter=1000,
                      rng=np.random.default_rng(100 + s))
        b = mc_engine(**_kw(cph=4.0, conv=0.3 + 1e-3), n_iter=1000,
                      rng=np.random.default_rng(100 + s))
        diffs.append(b['mean'] - a['mean'])
        levels.append(a['mean'])
    diffs = np.asarray(diffs)
    expected = (mc_expected_total(**_kw(cph=4.0, conv=0.3 + 1e-3))
                - mc_expected_total(**_kw(cph=4.0)))
    assert np.std(levels) > 20.0 * np.std(diffs)
    assert abs(diffs.mean() - expected) < 0.25 * expected


# --- The GUI's A/B comparisons run the engine itself ------------------------

def test_gui_ab_comparisons_draw_the_engines_spend_law():
    """The Optimize pipeline (its projection and the re-scored as-built vs
    applied comparison) and the What-If tab compare two layouts through
    ``layout_comparison.paired_comparison``, which runs ``mc_engine`` itself
    on both arms: their day spend is the engine's moment-matched lognormal
    (MC_SPEND_LAW), not the former hand-rolled loop's normal floored at
    zero. At a CV of 3 on thin days each arm's mean is the exact
    expectation, where the floored normal sat far above it."""
    import inspect

    import layout_comparison
    import viz_optimize
    import viz_whatif

    assert layout_comparison.mc_engine is SC.mc_engine
    for mod in (viz_optimize, viz_whatif):
        src = inspect.getsource(mod)
        assert 'paired_comparison(' in src, mod.__name__
        assert 'MC_SPEND_LAW' in src, mod.__name__
        for floored in ('np.maximum(base', 'np.maximum(imp',
                        'rs.normal(', 'np.random.normal(nc'):
            assert floored not in src, (mod.__name__, floored)

    kw_a = _kw(cph=0.4, rev_std=3.0 * 480.0, imp_rate=0.2, imp_val=20.0,
               n_days=14, n_iter=20_000)
    kw_b = dict(kw_a, conv=0.31)
    res = layout_comparison.paired_comparison(kw_a, kw_b, seed=11)
    for label, kw in (('A', kw_a), ('B', kw_b)):
        arm = res[label]
        se = arm['std'] / np.sqrt(res['n_iter'])
        assert arm['exact_mean'] == pytest.approx(mc_expected_total(**kw))
        assert abs(arm['mean'] - arm['exact_mean']) <= 4.0 * se, label
        assert (arm['totals'] >= 0.0).all()
        # The test can see the bias the former law carried.
        assert legacy_floor_bias(**kw) > 10.0 * se, label
