"""The rules of the live-protocol study (review R19).

``run_live_protocol_study`` derives the live runners' shared protocol from
seeded runs. These tests pin each rule on inputs whose answer is known:
the analytic warm-up on exponential visit lengths (E[(S-t)+]/E[S] =
exp(-t/mu), so the 1% point is mu ln 100), the nominal rate as the highest
rate below the cap-share threshold with every lower rate below it too, the
window as the shortest whole minute with enough completed cohort visits,
the drift's equivalence test, and seeds kept away from the runners'.
"""
import math

import numpy as np
import pytest

from experiments import run_live_protocol_study as P


def _run(visits, open_spawns=(), end=3600.0, occ=None, cap=45, serving=None):
    times = np.arange(1.0, end + 1.0)
    return {'times': times,
            'occ': (np.full(times.size, 30.0) if occ is None
                    else np.asarray(occ, dtype=float)),
            'serving': (np.full(times.size, 0.3) if serving is None
                        else np.asarray(serving, dtype=float)),
            'visits': np.asarray(visits, dtype=float).reshape(-1, 2),
            'open_spawns': np.asarray(open_spawns, dtype=float),
            'end': end, 'cap': cap, 'arrivals': len(visits), 'balked': 0}


def _exponential_visits(rng, n, mean, start=400.0, span=2000.0):
    spawn = start + rng.random(n) * span
    return np.column_stack([spawn, spawn + rng.exponential(mean, n)])


def test_analytic_warmup_on_exponential_visits():
    rng = np.random.default_rng(0)
    runs = [_run(_exponential_visits(rng, 20000, 100.0), end=1e6)]
    uniq, surv, n, cens = P.km_survival(runs)
    assert n == 20000 and cens == 0
    t, mean = P.fill_time(uniq, surv, 0.01)
    assert mean == pytest.approx(100.0, rel=0.03)
    assert t == pytest.approx(100.0 * math.log(100.0), rel=0.05)
    assert P.whole_minutes(t) % 60 == 0 and P.whole_minutes(t) >= t
    assert P.km_median(uniq, surv) == pytest.approx(100 * math.log(2), rel=0.05)


def test_censoring_is_kept_out_of_the_events():
    """Visits still open at the end are censored, not dropped: dropping
    them would shorten the visit-length distribution."""
    rng = np.random.default_rng(1)
    v = _exponential_visits(rng, 5000, 200.0)
    end = 1500.0
    done = v[v[:, 1] <= end]
    still = v[v[:, 1] > end][:, 0]
    runs = [_run(done, still, end=end)]
    uniq, surv, n, cens = P.km_survival(runs)
    assert cens == still[still > P.KM_DROP_BEFORE_S].size > 0
    _, mean = P.fill_time(uniq, surv)
    naive = float((done[:, 1] - done[:, 0]).mean())
    assert mean > naive                      # censoring accounted for


def test_nominal_rule_stops_at_the_first_rate_over_the_threshold():
    table = {0.25: {'cap_share_after_warmup': 0.002},
             0.26: {'cap_share_after_warmup': 0.009},
             0.27: {'cap_share_after_warmup': 0.012},
             0.28: {'cap_share_after_warmup': 0.005}}
    assert P.nominal_rule(table) == 0.26
    assert P.nominal_rule({0.25: {'cap_share_after_warmup': 0.02}}) is None


def test_window_rule_needs_enough_cohort_visits_in_every_rep():
    # One completed 60 s visit arriving every 4 s from t = 1000.
    spawns = np.arange(1001.0, 3500.0, 4.0)
    visits = np.column_stack([spawns, spawns + 60.0])
    runs = [_run(visits), _run(visits)]
    W, counts = P.window_rule(runs, 1000.0, 60.0, 3600.0)
    # Arrivals in (1000, 1000 + W] that left by 1000 + W: (W - 60) / 4.
    assert W == min(w for w in np.arange(600.0, 2601.0, 60.0)
                    if (w - 60.0) // 4 + 1 >= P.MIN_COHORT)
    assert counts[f'{W:.0f}']['min'] >= P.MIN_COHORT
    # Ten medians of 60 s is 600 s, already inside every candidate; ten of
    # 200 s push the window to the first whole minute past 2,000 s, and
    # ten of 400 s do not fit before the run ends.
    W2, _ = P.window_rule(runs, 1000.0, 200.0, 3600.0)
    assert W2 == 2040.0
    W3, _ = P.window_rule(runs, 1000.0, 400.0, 3600.0)
    assert W3 is None


def test_drift_equivalence_against_the_warmup_level():
    rng = np.random.default_rng(2)
    t = np.arange(1.0, 3601.0)
    flat = [_run([], occ=30 + rng.normal(0, 0.05, t.size)) for _ in range(12)]
    d = P.drift(flat, 1000.0, 1800.0)
    assert d['equivalent_within_margin'] is True
    assert abs(d['relative_change_over_window_mean']) < 0.01
    # A 5% rise across the window is not equivalent to "within 1%".
    rising = [_run([], occ=30 * (1 + 0.05 * np.clip((t - 1000) / 1800, 0, 1))
                   + rng.normal(0, 0.05, t.size)) for _ in range(12)]
    d = P.drift(rising, 1000.0, 1800.0)
    assert d['equivalent_within_margin'] is False
    assert d['relative_change_over_window_mean'] == pytest.approx(0.05, abs=0.005)
    assert d['slope_per_1000s_mean'] == pytest.approx(30 * 0.05 / 1.8, rel=0.05)


def test_cap_share_and_utilisation_count_only_after_the_warmup():
    occ = np.r_[np.full(1000, 45.0), np.full(2600, 40.0)]
    serving = np.r_[np.zeros(1000), np.ones(2600)]
    r = _run([], occ=occ, serving=serving)
    assert P.cap_share([r], 45, 1001.0) == 0.0
    assert P.cap_share([r], 45, 0.0) == pytest.approx(1000 / 3600)
    assert P.utilisation([r], 1001.0) == 1.0


def test_double_warmup_pairs_the_two_windows():
    spawns = np.arange(10.0, 3890.0, 5.0)
    visits = np.column_stack([spawns, spawns + 50.0])
    runs = [_run(visits, end=3900.0) for _ in range(3)]
    out = P.double_warmup(runs, 1020.0, 1860.0)
    assert out['mean_occupancy']['diff_mean'] == pytest.approx(0.0)
    assert out['cohort_completed']['after_1x_warmup'] > 0


def test_study_seeds_avoid_the_runners():
    P.check_seed_ranges(P.SEED_BASE, 12, 16)
    with pytest.raises(ValueError):
        P.check_seed_ranges(900, 6, 16)


def test_differences_name_the_runner_and_the_constant():
    derived = {'NOMINAL_SPAWN': 0.27, 'WARMUP_S': 1080.0}
    used = {'run_abm_diagnostics': {'NOMINAL_SPAWN': 0.27, 'WARMUP_S': 1020.0}}
    diffs = P.compare(derived, used)
    assert diffs == [{'runner': 'run_abm_diagnostics', 'constant': 'WARMUP_S',
                      'runner_value': 1020.0, 'study_value': 1080.0}]
    assert set(P.runner_constants()) == set(P.LIVE_RUNNERS)


def test_window_stats_report_the_runners_outcomes():
    """Outcomes over the whole cohort of the window (arrivals followed to
    their exit), queue waits of the services starting in it."""
    spawns = np.array([1000.5, 1100.0, 1200.0, 2900.0])
    exits = np.array([1100.0, 1300.0, 3200.0, 3000.0])
    r = _run(np.column_stack([spawns, exits]), end=3600.0)
    # paid, basket, revenue per visit, in visit order.
    r['outcomes'] = np.array([[1, 2, 10.0], [0, 0, 0.0], [1, 4, 30.0],
                              [1, 1, 5.0]])
    # Two services start by t = 1000, three more by 2000, one after.
    r['services'] = np.where(r['times'] <= 1000, 2,
                             np.where(r['times'] <= 2000, 5, 6))
    r['waits'] = np.array([9.0, 9.0, 1.0, 2.0, 3.0, 50.0])
    r['perimeter_ratio'] = {'1000': 1.7}
    w = P.window_stats(r, 1000.0, 1000.0)
    # Cohort: the first three arrivals (the fourth arrives after 2000).
    assert w['conversion'] == pytest.approx(2 / 3)
    assert w['revenue_per_visit'] == pytest.approx(40.0 / 3)
    assert w['mean_basket_paying'] == pytest.approx(3.0)
    assert w['revenue_per_paying_visit'] == pytest.approx(20.0)
    assert w['cohort_mean_visit_all_s'] == pytest.approx(
        (99.5 + 200.0 + 2000.0) / 3)
    assert w['cohort_censored'] == 0
    assert w['mean_queue_wait_s'] == pytest.approx(2.0)
    assert w['perimeter_ratio'] == 1.7
    # Only the visit that also left inside the window counts as completed.
    assert w['cohort_completed'] == 2


def test_protocol_status_separates_reproduced_from_validated():
    assert P.protocol_status(True, True) == 'reproduced_drift_equivalent'
    assert P.protocol_status(True, False) == 'reproduced_drift_not_equivalent'
    assert P.protocol_status(False, None) == \
        'differs_from_runners_drift_not_tested'


def _synthetic_params():
    import pandas as pd
    from dataset_calibration import calibrate_transactional
    rng = np.random.default_rng(0)
    products = [(f'{c[:3].upper()}{i}', c, 1.5 + i)
                for c in ('Bakery', 'Dairy', 'Garden') for i in range(4)]
    day0 = pd.Timestamp('2010-12-01 09:00')
    rows = []
    for k in range(400):
        ts = day0 + pd.Timedelta(days=k // 40,
                                 minutes=int(rng.integers(0, 480)))
        for j in rng.choice(len(products), size=int(rng.integers(1, 5)),
                            replace=False):
            pid, cat, price = products[j]
            rows.append((f'I{k:05d}', pid, f'ITEM {pid}', 1, ts, price, cat))
    df = pd.DataFrame(rows, columns=['invoice_id', 'product_id',
                                     'product_name', 'quantity', 'timestamp',
                                     'unit_price', 'category'])
    return calibrate_transactional(df, currency='GBP')


def test_probe_drains_its_cohort_and_records_outcomes():
    r = P.probe(_synthetic_params(), 0.4, 30, 240.0, seed=P.SEED_BASE,
                freeze_profile=True, heat_windows=[(60.0, 180.0)],
                drain_after=240.0)
    assert len(r['outcomes']) == len(r['visits']) > 0
    # Every agent that arrived by the end of the run has left.
    assert not np.any(r['open_spawns'] <= 240.0)
    assert r['drained_s'] > 0 and r['end'] > 240.0
    assert set(np.unique(r['outcomes'][:, 0])) <= {0.0, 1.0}
    assert np.all(np.diff(r['services']) >= 0)
    assert r['services'][-1] <= len(r['waits'])
    assert r['perimeter_ratio']['60'] > 0
    w = P.window_stats(r, 60.0, 120.0)
    assert w['cohort_censored'] == 0 and 0 < w['conversion'] <= 1
