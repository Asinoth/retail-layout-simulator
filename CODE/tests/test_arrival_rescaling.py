"""The arrival row tests time-rescaled gaps against Exp(1) (review R47).

The former row set the dataset's gaps against a constant-rate Poisson
reference, so a store whose arrivals are exactly Poisson at an hourly rate
-- the live simulator's own law -- was rejected for its hour-of-day profile
alone. The row now divides each gap by the local mean gap the calibrated
hour-of-day rate implies (the rate integrated over the gap), which is
Exp(1) under a non-homogeneous Poisson process with that profile and the
same volume every trading day, and it keeps the source's 60 s timestamp
resolution on both sides.
"""
import numpy as np
import pandas as pd
import pytest

import dataset_validation as DV
from dataset_calibration import calibrate_transactional

# Invoices per hour of an average trading day, opening 09:00-18:00: a
# strong hour-of-day profile, so a constant-rate reference is badly wrong.
PROFILE = {9: 4.0, 10: 12.0, 11: 22.0, 12: 30.0, 13: 26.0, 14: 16.0,
           15: 10.0, 16: 6.0, 17: 3.0}


def _invoices(n_days=90, day_sd=0.0, seed=0):
    """One single-line invoice per arrival of a Poisson process at the
    hourly PROFILE, stamped to the minute as Online Retail II is. With
    ``day_sd`` each day's volume is scaled by a lognormal factor, which the
    null leaves out."""
    rng = np.random.default_rng(seed)
    day0 = pd.Timestamp('2010-01-04')
    stamps = []
    for d in range(n_days):
        mult = float(np.exp(rng.normal(-0.5 * day_sd ** 2, day_sd))) if day_sd else 1.0
        for h, per_hour in PROFILE.items():
            n = rng.poisson(per_hour * mult)
            secs = h * 3600.0 + rng.random(n) * 3600.0
            stamps.extend(day0 + pd.to_timedelta(d, 'D')
                          + pd.to_timedelta(np.floor(secs / 60.0) * 60.0, 's'))
    n = len(stamps)
    return pd.DataFrame({
        'invoice_id': [f'I{k:06d}' for k in range(n)],
        'product_id': [f'P{k % 7}' for k in range(n)],
        'product_name': [f'ITEM {k % 7}' for k in range(n)],
        'quantity': 1, 'timestamp': stamps, 'unit_price': 2.5,
        'category': 'Kitchen'})


@pytest.fixture(scope='module')
def nhpp_params():
    return calibrate_transactional(_invoices(), currency='GBP')


def test_gap_start_times_line_up_with_the_gaps(nhpp_params):
    p = nhpp_params
    starts = np.asarray(p.inter_arrival_start_s)
    gaps = np.asarray(p.inter_arrival_seconds)
    assert starts.size == gaps.size > 1000
    # Same-day gaps between minute stamps inside the opening hours.
    assert starts.min() >= 9 * 3600 and (starts + gaps).max() < 18 * 3600
    assert np.all(np.mod(starts, 60.0) == 0.0)


def test_rescaling_integrates_the_hourly_rate():
    rates = np.zeros(24)
    rates[9], rates[10] = 1.0 / 60.0, 1.0 / 30.0      # per second
    # A minute inside the 09:00 hour is one expected arrival, a minute
    # inside the 10:00 hour two, and a gap from 09:59 to 10:01 takes one
    # from each: three.
    got = DV.time_rescaled_gaps([9 * 3600.0, 10 * 3600.0, 10 * 3600 - 60.0],
                                [60.0, 60.0, 120.0], rates)
    assert got == pytest.approx([1.0, 2.0, 3.0])


def test_the_hourly_rates_carry_the_calibrated_volume(nhpp_params):
    rates = DV.hourly_arrival_rates(nhpp_params)
    days = nhpp_params.calibration_extra['n_trading_days']
    assert rates.sum() * 3600.0 * days == pytest.approx(nhpp_params.n_invoices)
    assert np.argmax(rates) == 12


def test_a_poisson_store_at_an_hourly_rate_passes(nhpp_params):
    """Arrivals drawn exactly from the null: the rescaled gaps are Exp(1)
    in mean and spread, and the row does not reject, where the former
    constant-rate reference rejects outright."""
    p = nhpp_params
    row = DV._arrival_row(p, np.asarray(p.inter_arrival_seconds))
    assert row.name == DV.ARRIVAL_TEST_NAME
    assert 'inter-arrival' in row.name.lower()       # the macro tag
    assert row.extra['rescaled'] is True
    assert row.extra['timestamp_step_s'] == 60.0
    assert row.extra['null'] == DV.ARRIVAL_NULL
    assert row.extra['rescaled_mean'] == pytest.approx(1.0, abs=0.05)
    assert row.extra['rescaled_cv'] == pytest.approx(1.0, abs=0.08)
    assert row.p_value > 0.001
    assert row.extra['homogeneous_p_value'] < 1e-6


def test_day_to_day_variation_shows_as_overdispersion():
    """A volume that changes from day to day is what the null leaves out:
    the rescaled gaps spread wider than Exp(1)."""
    p = calibrate_transactional(_invoices(day_sd=0.6, seed=1), currency='GBP')
    row = DV._arrival_row(p, np.asarray(p.inter_arrival_seconds))
    assert row.extra['rescaled_cv'] > 1.1
    assert row.extra['rescaled_cv'] > row.extra['reference_rescaled_cv']


def test_the_reference_is_fixed(nhpp_params):
    p = nhpp_params
    a = DV._arrival_row(p, np.asarray(p.inter_arrival_seconds))
    b = DV._arrival_row(p, np.asarray(p.inter_arrival_seconds))
    assert (a.statistic, a.p_value) == (b.statistic, b.p_value)


def test_without_start_times_the_row_says_it_is_unrescaled(nhpp_params):
    from dataclasses import replace
    bare = replace(nhpp_params, inter_arrival_start_s=np.zeros(0))
    row = DV._arrival_row(bare, np.asarray(bare.inter_arrival_seconds))
    assert row.extra['rescaled'] is False
    assert 'constant-rate' in row.note
