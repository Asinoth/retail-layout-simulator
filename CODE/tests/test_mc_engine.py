"""Monte Carlo engine regression tests (audit R5.3).

Encodes the paper's Verification-section claims for the MC engine as
runnable assertions: (1) with day noise disabled the empirical mean
matches the closed-form expectation, (2) basket size is *tracked*, not
monetized, so it does not move revenue, and (3) two parameter sets scored
under the same seed walk the same sample path, which is what makes the
paired (common-random-number) comparisons in the experiment runners
measure layouts rather than sampler noise.
"""

import numpy as np

from retail_literature import DEFAULT_WEEKEND_MULTIPLIER
import sim_calibration
from sim_calibration import (_binomial_cdf_table, _binomial_icdf,
                             mc_engine)

_PARAMS = dict(
    cph=20.0, op_hours=10.0, conv=0.30,
    rev_mean=25.0, rev_std=2.0, imp_rate=0.40, imp_val=5.0,
    avg_bsk=4.0, std_bsk=1.0, n_days=30,
)


def _run(n_iter, avg_bsk=None, seed=0, day_noise_std=0.0, wknd_mult=1.0):
    p = dict(_PARAMS)
    if avg_bsk is not None:
        p['avg_bsk'] = avg_bsk
    return mc_engine(
        cph=p['cph'], conv=p['conv'], rev_mean=p['rev_mean'],
        rev_std=p['rev_std'], imp_rate=p['imp_rate'], imp_val=p['imp_val'],
        avg_bsk=p['avg_bsk'], std_bsk=p['std_bsk'], observed_baskets=None,
        n_days=p['n_days'], n_iter=n_iter, op_hours=p['op_hours'],
        wknd_mult=wknd_mult, monthly_growth=0.0, day_noise_std=day_noise_std,
        rng=np.random.default_rng(seed))


def test_mc_mean_matches_closed_form():
    """E[total] = n_days * (cph*op_hours) * conv * (rev_mean + imp_rate*imp_val)
    with weekend multiplier 1 and day noise off."""
    p = _PARAMS
    expected = (p['n_days'] * p['cph'] * p['op_hours'] * p['conv']
                * (p['rev_mean'] + p['imp_rate'] * p['imp_val']))
    res = _run(n_iter=40000, seed=0)
    rel_err = abs(res['mean'] - expected) / expected
    assert rel_err < 0.01, f"MC mean {res['mean']:.1f} vs closed-form {expected:.1f} (rel err {rel_err:.4f})"


def test_basket_size_does_not_move_revenue():
    """Basket statistics are tracked but not double-monetized: changing
    avg_bsk must leave the revenue totals identical (the basket draws come
    from their own stream, so nothing else can see them)."""
    small = _run(n_iter=5000, avg_bsk=4.0, seed=1)
    large = _run(n_iter=5000, avg_bsk=40.0, seed=1)
    assert small['mean'] == large['mean']
    assert np.array_equal(small['totals'], large['totals'])


def test_day_noise_preserves_mean():
    """With the default day noise (sigma 0.08) and a weekend multiplier the
    mean still matches the closed form to 0.5%: the per-day factor has
    mean exactly 1."""
    p = _PARAMS
    wknd = DEFAULT_WEEKEND_MULTIPLIER
    day_mult = [wknd if d % 7 in (5, 6) else 1.0 for d in range(p['n_days'])]
    expected = (sum(day_mult) * p['cph'] * p['op_hours'] * p['conv']
                * (p['rev_mean'] + p['imp_rate'] * p['imp_val']))
    res = _run(n_iter=40000, seed=2, day_noise_std=0.08, wknd_mult=wknd)
    rel_err = abs(res['mean'] - expected) / expected
    assert rel_err < 0.005, f"MC mean {res['mean']:.1f} vs closed-form {expected:.1f} (rel err {rel_err:.4f})"


def test_day_noise_is_independent_across_days():
    """Day factors are drawn fresh each day, so the customer-count spread
    does not grow over the horizon and consecutive days are uncorrelated."""
    res = _run(n_iter=40000, seed=3, day_noise_std=0.08)
    cust = res['daily_customers']
    cv = cust.std(axis=0) / cust.mean(axis=0)
    assert abs(cv[-1] - cv[0]) < 0.01, f"CV day 1 {cv[0]:.3f} vs last day {cv[-1]:.3f}"
    lag1 = np.corrcoef(cust[:, :-1].ravel(), cust[:, 1:].ravel())[0, 1]
    assert abs(lag1) < 0.02, f"lag-1 correlation {lag1:.3f}"


def test_mc_seed_reproducible():
    """Same seed -> identical totals (determinism of the engine itself)."""
    a = _run(n_iter=3000, seed=5)
    b = _run(n_iter=3000, seed=5)
    assert np.array_equal(a['totals'], b['totals'])


def test_global_seed_determines_output():
    """The engine draws its own generators from the caller's stream, so
    seeding the global stream still pins the result exactly -- which is
    how every headless runner asks for a specific sample path."""
    def once(seed):
        np.random.seed(seed)
        return mc_engine(observed_baskets=None, n_iter=2000, wknd_mult=1.2,
                         monthly_growth=0.0, day_noise_std=0.08,
                         **_PARAMS)['totals']

    assert np.array_equal(once(17), once(17))
    assert not np.array_equal(once(17), once(18))


# --- common random numbers ----------------------------------------------

def _legacy_single_stream(conv, n_iter, rng):
    """The engine's earlier draw order: every family off one stream, with
    the counts from numpy's binomial sampler. Kept here as the reference
    the coupling below is measured against."""
    p = _PARAMS
    day_mult = np.ones(7)
    totals = np.zeros(n_iter)
    for d in range(p['n_days']):
        lam = p['cph'] * p['op_hours'] * day_mult[d % 7]
        half_var = 0.5 * 0.08 ** 2
        lam_vec = lam * np.exp(rng.normal(0.0, 0.08, n_iter) - half_var)
        n_cust = rng.poisson(lam_vec, n_iter)
        n_conv = rng.binomial(n_cust, conv)
        nc = n_conv.astype(np.float64)
        rng.normal(nc * p['avg_bsk'], np.sqrt(nc) * p['std_bsk'], n_iter)
        base = np.maximum(rng.normal(nc * p['rev_mean'],
                                     np.sqrt(nc) * p['rev_std'], n_iter), 0.0)
        n_imp = rng.binomial(n_conv, p['imp_rate'])
        ni = n_imp.astype(np.float64)
        imp = np.maximum(rng.normal(ni * p['imp_val'],
                                    np.sqrt(ni) * 1.5, n_iter), 0.0)
        totals += base + imp
    return totals


def _paired_diffs(engine, delta_conv, n_seeds=16, n_iter=1000):
    """Mean-revenue difference between two nearly identical parameter sets
    scored under the same seed, once per seed."""
    out = []
    for s in range(n_seeds):
        a = engine(_PARAMS['conv'], n_iter, np.random.default_rng(100 + s))
        b = engine(_PARAMS['conv'] + delta_conv, n_iter,
                   np.random.default_rng(100 + s))
        out.append(float(np.mean(b) - np.mean(a)))
    return np.asarray(out)


def _current(conv, n_iter, rng):
    p = dict(_PARAMS, conv=conv)
    return mc_engine(
        cph=p['cph'], conv=p['conv'], rev_mean=p['rev_mean'],
        rev_std=p['rev_std'], imp_rate=p['imp_rate'], imp_val=p['imp_val'],
        avg_bsk=p['avg_bsk'], std_bsk=p['std_bsk'], observed_baskets=None,
        n_days=p['n_days'], n_iter=n_iter, op_hours=p['op_hours'],
        wknd_mult=1.0, monthly_growth=0.0, impulse_value_std=1.5,
        day_noise_std=0.08, rng=rng)['totals']


def test_paired_difference_noise_is_an_order_of_magnitude_smaller():
    """Two layouts differing by a hair in conversion must give a paired
    difference dominated by that difference, not by the sampler. Drawing
    the counts by inverse CDF from per-family streams is what buys this:
    numpy's binomial sampler is neither monotone in p nor constant in how
    many uniforms it consumes, so under the old draw order a 1e-4 change
    in conversion redrew everything downstream."""
    delta = 1e-4
    coupled = _paired_diffs(_current, delta)
    single_stream = _paired_diffs(_legacy_single_stream, delta)
    assert coupled.std() < single_stream.std() / 10.0, (
        f"paired noise SD {coupled.std():.2f} vs single-stream "
        f"{single_stream.std():.2f}")

    # And the difference it does report is the real one: the closed-form
    # derivative of expected revenue with respect to conversion.
    p = _PARAMS
    expected = (delta * p['n_days'] * p['cph'] * p['op_hours']
                * (p['rev_mean'] + p['imp_rate'] * p['imp_val']))
    assert abs(coupled.mean() - expected) < 0.25 * expected, (
        f"paired mean {coupled.mean():.2f} vs closed-form {expected:.2f}")


def test_conversion_counts_are_monotone_in_the_rate():
    """Same seed, higher conversion: every iteration converts at least as
    many customers. Without that monotonicity a paired comparison is not
    measuring the parameter it varies."""
    low = _run(n_iter=4000, seed=9)['daily_converting']
    p = dict(_PARAMS, conv=_PARAMS['conv'] + 0.01)
    high = mc_engine(
        cph=p['cph'], conv=p['conv'], rev_mean=p['rev_mean'],
        rev_std=p['rev_std'], imp_rate=p['imp_rate'], imp_val=p['imp_val'],
        avg_bsk=p['avg_bsk'], std_bsk=p['std_bsk'], observed_baskets=None,
        n_days=p['n_days'], n_iter=4000, op_hours=p['op_hours'],
        wknd_mult=1.0, monthly_growth=0.0, day_noise_std=0.0,
        rng=np.random.default_rng(9))['daily_converting']
    assert (high >= low).all()
    assert high.sum() > low.sum()


def test_block_size_does_not_change_the_draws(monkeypatch):
    """Days are generated a block at a time. Every draw family has its own
    stream, so one day per block must reproduce the batched run exactly --
    otherwise the horizon or n_iter would quietly change the sample path."""
    kw = dict(cph=60.0, conv=0.35, rev_mean=25.0, rev_std=2.0, imp_rate=0.2,
              imp_val=5.0, avg_bsk=4.0, std_bsk=1.0, observed_baskets=None,
              n_days=10, n_iter=300, op_hours=10.0,
              wknd_mult=DEFAULT_WEEKEND_MULTIPLIER, monthly_growth=0.02,
              day_noise_std=0.08)
    batched = mc_engine(rng=np.random.default_rng(4), **kw)
    monkeypatch.setattr(sim_calibration, '_MC_BLOCK_DRAWS', kw['n_iter'])
    per_day = mc_engine(rng=np.random.default_rng(4), **kw)
    for key in ('daily_revenue', 'daily_customers', 'daily_converting',
                'daily_impulse_rev', 'daily_baskets'):
        assert np.array_equal(batched[key], per_day[key]), key


def test_table_and_direct_quantile_routes_agree():
    """Counts shared by few draws skip the cumulative table and use scipy's
    binomial quantile. The two routes must return the same count for the
    same uniform, or the route choice would leak into paired comparisons."""
    rng = np.random.default_rng(8)
    n = rng.integers(0, 5000, 4000)          # mostly one draw per count
    u = rng.random(4000)
    p = 0.37
    direct = _binomial_icdf(u, n, p)
    log_ratio = np.log(p) - np.log1p(-p)
    tables = {int(v): _binomial_cdf_table(int(v), p, log_ratio)
              for v in np.unique(n) if v > 0}
    tabulated = _binomial_icdf(u, n, p, tables)
    assert np.array_equal(direct, tabulated)
