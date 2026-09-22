"""Calibrated arrival rates, inter-arrival sample and co-purchase pairs.

The arrival rate must be in the units the Monte Carlo engine consumes.
That engine projects over consecutive CALENDAR days and lifts two days in
seven by the weekend multiplier, so ``arrivals_per_hour`` is buyers
(invoices) per open hour of an AVERAGE day, and ``visitors_per_hour``
divides out the assumed conversion. ``cph * op_hours * conversion``,
averaged over the engine's weekly pattern, then reproduces the observed
invoices per calendar day -- including the days the shop did not trade.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from dataset_calibration import calibrate_omnichannel, calibrate_transactional
from retail_literature import (DEFAULT_OP_HOURS_PER_DAY,
                               DEFAULT_WEEKEND_MULTIPLIER)
from sim_calibration import (DEFAULT_CALIBRATED_IMPULSE_RATE,
                             extract_simulation_parameters, mc_engine)

# Mean of the engine's weekly day-multiplier pattern (five ordinary days,
# two weekend days), stated here independently of the calibration code.
WEEK_MULT = (5.0 + 2.0 * DEFAULT_WEEKEND_MULTIPLIER) / 7.0


def _poisson_invoices(rate_per_hour, n_days, seed=0, closed_weekdays=()):
    """One-line invoices from a Poisson process at ``rate_per_hour`` over
    ``DEFAULT_OP_HOURS_PER_DAY`` open hours per trading day, timestamps
    floored to the minute as in Online Retail II. ``closed_weekdays`` holds
    Monday=0-style indices the shop does not trade on."""
    rng = np.random.default_rng(seed)
    open_s = DEFAULT_OP_HOURS_PER_DAY * 3600.0
    day0 = pd.Timestamp('2010-01-04 09:00')          # a Monday
    stamps = []
    for d in range(n_days):
        day = day0 + pd.Timedelta(days=d)
        if day.dayofweek in closed_weekdays:
            continue
        n = rng.poisson(rate_per_hour * DEFAULT_OP_HOURS_PER_DAY)
        offsets = np.sort(rng.uniform(0.0, open_s, n))
        stamps.extend(day + pd.to_timedelta(offsets, unit='s'))
    ts = pd.DatetimeIndex(stamps).floor('min')
    n = len(ts)
    return pd.DataFrame({
        'invoice_id': [f'INV{i:07d}' for i in range(n)],
        'product_id': [f'P{i % 7}' for i in range(n)],
        'product_name': 'THING',
        'quantity': 1,
        'timestamp': ts,
        'unit_price': 2.0,
        'category': 'General',
    })


def _calendar_days(df):
    dates = df['timestamp'].dt.normalize()
    return int((dates.max() - dates.min()).days) + 1


def test_arrival_rate_is_on_the_average_calendar_day_basis():
    for rate in (2.0, 6.0, 20.0, 60.0):
        n_days = max(21, int(np.ceil(8000 / (rate * DEFAULT_OP_HOURS_PER_DAY))))
        n_days -= n_days % 7                         # whole weeks
        df = _poisson_invoices(rate, n_days, seed=int(rate))
        params = calibrate_transactional(df)
        per_day = params.n_invoices / _calendar_days(df)
        recovered = (params.arrivals_per_hour * DEFAULT_OP_HOURS_PER_DAY
                     * WEEK_MULT)
        rel = recovered / per_day - 1.0
        assert abs(rel) < 0.02, (f"true {per_day:.2f} invoices/calendar day -> "
                                 f"engine would produce {recovered:.2f}")


def test_closed_days_lower_the_rate():
    """A shop trading five days a week must not be projected as if it
    traded every day: the calibrated rate carries the closed days."""
    df = _poisson_invoices(6.0, 70, seed=21, closed_weekdays=(5, 6))
    params = calibrate_transactional(df)
    per_trading_day = params.n_invoices / df['timestamp'].dt.normalize().nunique()
    per_calendar_day = params.n_invoices / _calendar_days(df)
    recovered = (params.arrivals_per_hour * DEFAULT_OP_HOURS_PER_DAY
                 * WEEK_MULT)
    assert abs(recovered / per_calendar_day - 1.0) < 0.02
    assert recovered < 0.8 * per_trading_day


def test_visitor_rate_reproduces_daily_invoices():
    df = _poisson_invoices(6.0, 63, seed=11)
    conv = 0.30
    params = calibrate_transactional(df, assumed_conversion_rate=conv)
    assert np.isclose(params.visitors_per_hour, params.arrivals_per_hour / conv)
    daily_purchases = (params.visitors_per_hour * DEFAULT_OP_HOURS_PER_DAY
                       * conv * WEEK_MULT)
    assert np.isclose(daily_purchases, params.n_invoices / _calendar_days(df),
                      rtol=0.02)


def test_mc_horizon_reproduces_observed_volume():
    """The contract end to end: run the engine the way the projection
    paths do (whole weeks, weekend lift, calibrated visitor rate) and the
    purchases per day come back at the dataset's own calendar-day rate."""
    df = _poisson_invoices(6.0, 63, seed=12)
    conv = 0.30
    params = calibrate_transactional(df, assumed_conversion_rate=conv)
    observed_per_day = params.n_invoices / _calendar_days(df)
    res = mc_engine(
        cph=params.visitors_per_hour, conv=conv, rev_mean=20.0, rev_std=5.0,
        imp_rate=0.2, imp_val=3.0, avg_bsk=3.0, std_bsk=1.0,
        observed_baskets=None, n_days=28, n_iter=2000,
        op_hours=DEFAULT_OP_HOURS_PER_DAY,
        wknd_mult=DEFAULT_WEEKEND_MULTIPLIER, monthly_growth=0.0,
        day_noise_std=0.0, rng=np.random.default_rng(3))
    simulated_per_day = res['daily_converting'].sum(axis=1).mean() / 28.0
    assert np.isclose(simulated_per_day, observed_per_day, rtol=0.03), (
        f"MC {simulated_per_day:.2f} vs observed {observed_per_day:.2f} "
        "purchases per calendar day")


def test_seeded_calibration_feeds_visitor_rate_and_resets_basket_cache():
    params = calibrate_transactional(_poisson_invoices(6.0, 28, seed=5),
                                     assumed_conversion_rate=0.25)
    sim = SimpleNamespace(analytics={}, run_time=0.0, sim_time=0.0,
                          simulation_speed=1.0, shop=None,
                          _basket_struct_cache=('stale', None, None, None))
    params.seed_into(sim)
    assert sim._basket_struct_cache is None
    cal = sim.analytics['calibration']
    assert np.isclose(cal['arrivals_per_hour'], params.arrivals_per_hour)
    assert np.isclose(cal['visitors_per_hour'], params.visitors_per_hour)
    cph = extract_simulation_parameters(sim)['customers_per_hour']
    assert np.isclose(cph, params.visitors_per_hour)


def _sim_with_live_impulse(calibration=None):
    analytics = {
        'total_customers': 200, 'completed_purchases': 100,
        'abandoned_carts': 100, 'impulse_purchases': 30,
        'impulse_item_sales': {'Gum': 30}, 'total_revenue': 2000.0,
        'basket_sizes': [3] * 20, 'customer_revenues': [20.0] * 20,
    }
    if calibration is not None:
        analytics['calibration'] = calibration
    shop = SimpleNamespace(prices={'Gum': 2.0},
                           floors={1: {'prices': {'Gum': 2.0}}})
    return SimpleNamespace(analytics=analytics, run_time=3600.0,
                           sim_time=3600.0, simulation_speed=1.0, shop=shop)


def test_impulse_rate_source_follows_what_the_dataset_can_observe():
    """Transaction records carry no impulse label, so a purchase-side
    calibration hands the impulse share to the literature stand-in rather
    than to a live counter that depends on session length. A trajectory
    calibration observes no purchases at all, so the live counter stays."""
    live = extract_simulation_parameters(_sim_with_live_impulse())
    assert live['impulse_rate_source'] == 'live_counter'
    assert np.isclose(live['impulse_rate'], 30 / 100)

    transactional = extract_simulation_parameters(_sim_with_live_impulse(
        {'visitors_per_hour': 15.8, 'conversion_rate': 0.30}))
    assert transactional['impulse_rate_source'] == 'literature_default'
    assert np.isclose(transactional['impulse_rate'],
                      DEFAULT_CALIBRATED_IMPULSE_RATE)

    trajectory = extract_simulation_parameters(_sim_with_live_impulse(
        {'spatial_source': 'trajectory_dataset', 'n_tracks': 360}))
    assert trajectory['impulse_rate_source'] == 'live_counter'
    assert np.isclose(trajectory['impulse_rate'], 30 / 100)

    measured = extract_simulation_parameters(_sim_with_live_impulse(
        {'visitors_per_hour': 298.0, 'conversion_rate': 0.99,
         'mean_impulse_rate': 0.11}))
    assert measured['impulse_rate_source'] == 'dataset'
    assert np.isclose(measured['impulse_rate'], 0.11)


def test_hourly_profile_integrates_to_the_operating_day():
    """The live spawn loop multiplies its base rate by the profile, so the
    profile has to integrate to the operating hours the rate is defined on
    -- otherwise a live day driven at the calibrated rate delivers a
    different volume than the daily Monte Carlo."""
    df = _poisson_invoices(6.0, 28, seed=7)
    # A stray out-of-hours invoice: it must widen the shape without
    # inflating the day's total.
    stray = df.iloc[[0]].copy()
    stray['invoice_id'] = 'INV_STRAY'
    stray['timestamp'] = pd.Timestamp('2010-01-06 06:00')
    params = calibrate_transactional(pd.concat([df, stray], ignore_index=True))
    sim = SimpleNamespace(analytics={}, run_time=0.0, sim_time=0.0,
                          simulation_speed=1.0, shop=None,
                          _basket_struct_cache=None)
    params.seed_into(sim)
    profile = np.asarray(sim.hourly_profile, dtype=np.float64)
    assert np.isclose(profile.sum(), DEFAULT_OP_HOURS_PER_DAY)
    open_hours = int((profile > 0).sum())
    assert open_hours >= DEFAULT_OP_HOURS_PER_DAY      # the stray hour is kept
    # The clock still starts inside the busy part of the day, not on the
    # 06:00 outlier.
    assert profile[int(sim.sim_clock_start_hour)] >= 0.5 * profile[profile > 0].mean()


def test_distinct_basket_sample_is_kept():
    """Validation compares distinct items per visit and spend at one unit
    per item, which need the per-invoice distinct-product samples, not just
    their medians."""
    rows = [('A', 'P1'), ('A', 'P1'), ('A', 'P2'),      # 3 units, 2 products
            ('B', 'P3'),                                # 1 unit,  1 product
            ('C', 'P1'), ('C', 'P2'), ('C', 'P3')]      # 3 units, 3 products
    df = pd.DataFrame(rows, columns=['invoice_id', 'product_id'])
    df['product_name'] = 'THING'
    df['quantity'] = 1
    df['unit_price'] = df['product_id'].map({'P1': 1.0, 'P2': 2.0, 'P3': 4.0})
    df['category'] = 'General'
    df['timestamp'] = pd.to_datetime(['2010-12-01 10:00'] * 3
                                     + ['2010-12-01 10:05']
                                     + ['2010-12-01 10:10'] * 3)
    params = calibrate_transactional(df)
    assert sorted(params.basket_distinct_sizes.tolist()) == [1.0, 2.0, 3.0]
    assert sorted(params.basket_sizes.tolist()) == [1.0, 3.0, 3.0]
    # A's repeated P1 line counts once; invoice order A, B, C throughout.
    assert params.invoice_distinct_revenues.tolist() == [3.0, 4.0, 7.0]
    assert params.basket_distinct_sizes.tolist() == [2.0, 1.0, 3.0]


def test_inter_arrival_keeps_same_minute_and_drops_cross_date_gaps():
    stamps = ['2010-12-01 10:00', '2010-12-01 10:00', '2010-12-01 10:05',
              '2010-12-01 23:50',                   # 13h45m gap, same date
              '2010-12-02 00:10',                   # 20 min, crosses date
              '2010-12-02 00:12']
    df = pd.DataFrame({
        'invoice_id': [f'I{i}' for i in range(len(stamps))],
        'product_id': 'P0', 'product_name': 'THING', 'quantity': 1,
        'timestamp': pd.to_datetime(stamps), 'unit_price': 1.0,
        'category': 'General',
    })
    params = calibrate_transactional(df)
    got = sorted(params.inter_arrival_seconds.tolist())
    assert got == sorted([0.0, 300.0, 13 * 3600.0 + 45 * 60.0, 120.0])


def test_pair_cap_keeps_most_purchased_products():
    # One wholesale invoice with 60 distinct products. 'ZZZ' sorts last by
    # code but is the most purchased product overall, so it must survive
    # the 50-product cap; the least purchased codes (ties broken by id)
    # are the ones cut.
    rows = []
    big = [f'P{i:03d}' for i in range(59)] + ['ZZZ']
    for pid in big:
        rows.append(('BIG', pid))
    for k in range(5):
        rows.append((f'S{k}', 'ZZZ'))
    df = pd.DataFrame(rows, columns=['invoice_id', 'product_id'])
    df['product_name'] = 'THING'
    df['quantity'] = 1
    df['unit_price'] = 1.0
    df['category'] = 'General'
    df['timestamp'] = pd.Timestamp('2010-12-01 10:00')
    params = calibrate_transactional(df, top_pairs_n=5000)
    ids = {x for a, b, _ in params.top_pairs for x in (a, b)}
    assert 'ZZZ' in ids
    assert ids == set(big[:49]) | {'ZZZ'}
    assert all(a < b for a, b, _ in params.top_pairs)


def _omni_families(n_fam=20):
    return pd.DataFrame({
        'aisle_id': np.arange(1, n_fam + 1),
        'product_family': [f'Family {i}' for i in range(n_fam)],
        'zone_id': [i % 4 for i in range(n_fam)],
        'avg_price': 3.0,
        'purchase_pct_instore': 0.6,       # sums far above 1 -> saturates
        'daily_demand_instore': 10.0,
        'dwell_s': 30.0,
        'impulse_rate': 0.1,
    })


def test_omnichannel_saturated_conversion_is_an_assumption():
    # Seventeen open hours, every weekday: the store's measured day is
    # longer than the operating day the projection engine runs, so the rate
    # must be re-expressed on that basis rather than copied per hour.
    weekdays = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday',
                'saturday', 'sunday']
    hours = list(range(6, 23))
    arrivals = pd.DataFrame([
        {'weekday': wd, 'from_hour': h, 'instore_arrivals': 30.0 + h}
        for wd in weekdays for h in hours
    ])
    params = calibrate_omnichannel(_omni_families(), arrivals)
    assert params.conversion_rate_source == 'assumption'
    assert params.calibration_extra['arrival_rate_source'] == 'arrivals_table'

    weekly = float(arrivals['instore_arrivals'].sum())
    daily_visitors = (params.visitors_per_hour * DEFAULT_OP_HOURS_PER_DAY
                      * WEEK_MULT)
    assert np.isclose(daily_visitors, weekly / 7.0)
    assert np.isclose(params.arrivals_per_hour,
                      params.visitors_per_hour * params.assumed_conversion_rate)


def test_omnichannel_without_arrivals_converts_demand_to_visitors():
    """Daily demand counts product families bought, not shoppers. A trip
    takes about a dozen families, so the fallback has to divide by the
    trip size instead of treating every unit as a visitor."""
    families = _omni_families()
    params = calibrate_omnichannel(families, None)
    assert params.calibration_extra['arrival_rate_source'] == 'derived_from_demand'

    demand = float(families['daily_demand_instore'].sum())
    trip_families = float(np.mean(params.basket_sizes))
    expected_visitors = demand / trip_families / params.assumed_conversion_rate
    got_visitors = (params.visitors_per_hour * DEFAULT_OP_HOURS_PER_DAY
                    * WEEK_MULT)
    assert np.isclose(got_visitors, expected_visitors, rtol=1e-6)
    # Sanity: a dozen families per trip means far fewer visitors than units.
    assert got_visitors < 0.2 * demand
