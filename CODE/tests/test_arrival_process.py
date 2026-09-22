"""The live arrival stream must be Poisson at the stated rate.

Section 3.2 of the paper states live arrivals follow a non-homogeneous
Poisson process. These tests drive the simulator's own arrival scheduler
(``_process_arrivals``) tick by tick with ``_spawn_customer`` stubbed, so
they check what the live loop delivers rather than numpy's exponential
sampler. Two failure modes are pinned: a fixed 1/rate cadence (right mean,
zero inter-arrival variance, which understates queueing) and firing on the
first tick after each gap while restarting the next gap from that tick
(every interval padded by the tick overshoot, so the realized rate falls
short of spawn_rate).
"""
import numpy as np
import pytest
from scipy import stats

from simulation import CustomerFlowSimulation


TICK_S = 0.04          # one live tick at 25 fps
RATE = 2.5             # arrivals per simulated second
HOUR_S = 3600.0
SEEDS = (11, 22, 33, 44, 55)


class _StubShop:
    """Minimum surface CustomerFlowSimulation touches at construction."""
    width = 20.0
    height = 15.0
    items = {}
    walls = {}
    prices = {}
    floors = {1: {'items': {}, 'walls': {}, 'prices': {}}}
    current_floor = 1


@pytest.fixture
def sim():
    return CustomerFlowSimulation(_StubShop())


def _drive_arrivals(sim, seconds, dt=TICK_S):
    """Step the scheduler the way step() does; return the exact simulated
    time of every arrival."""
    times = []

    def _spawn():
        # The scheduler has already consumed the gap when it spawns, so the
        # arrival instant is the clock minus the time carried forward.
        times.append(sim.sim_time - sim._since_last_arrival)

    sim._spawn_customer = _spawn
    for _ in range(int(round(seconds / dt))):
        sim.sim_time += dt
        sim._process_arrivals(dt)
    return np.asarray(times)


@pytest.fixture(scope='module')
def hour_runs():
    """One simulated hour at RATE per seed, each on a fresh simulation."""
    runs = []
    for seed in SEEDS:
        s = CustomerFlowSimulation(_StubShop())
        s.spawn_rate = RATE
        s.arrival_rng = np.random.default_rng(seed)
        runs.append(_drive_arrivals(s, HOUR_S))
    return runs


def test_arrival_rng_is_wired(sim):
    """The simulation owns a dedicated arrival stream and starts unprimed."""
    assert hasattr(sim, 'arrival_rng')
    assert sim._next_spawn_gap is None
    assert sim._since_last_arrival == 0.0


def test_loop_realizes_nominal_rate_with_poisson_counts(hour_runs):
    """dt 0.04 s, 2.5 arrivals/s, one simulated hour per seed.

    The realized rate is pooled over the seeds (about 45,000 arrivals,
    standard error about 0.5%), so the 2% tolerance is several standard
    errors wide yet still catches tick snapping, which at this rate and
    tick loses about 5% of arrivals. Counts in one-second windows must have
    variance close to their mean, as Poisson counts do; a fixed cadence
    gives a ratio near zero.
    """
    n_arrivals = sum(len(t) for t in hour_runs)
    realized = n_arrivals / (HOUR_S * len(hour_runs))
    assert realized == pytest.approx(RATE, rel=0.02), \
        f"realized rate {realized:.4f}/s against nominal {RATE}/s"

    edges = np.arange(0.0, HOUR_S + 0.5, 1.0)
    counts = np.concatenate([np.histogram(t, bins=edges)[0] for t in hour_runs])
    ratio = counts.var(ddof=1) / counts.mean()
    assert ratio == pytest.approx(1.0, abs=0.1), \
        f"variance/mean={ratio:.3f}; a fixed cadence would give ~0"


def test_inter_arrival_gaps_are_exponential(hour_runs):
    """Gaps between the scheduled arrival instants are Exp(1/rate).

    The goodness-of-fit is pooled over several seeds on purpose. A KS test
    at thousands of samples is knife-edge: even a correct generator fails
    at the nominal rate, so a single fixed seed makes a flaky test rather
    than a strict one. A genuinely wrong distribution (gaps snapped to the
    tick lattice, or constant gaps) drives every p-value to zero, which the
    median still catches.
    """
    scale = 1.0 / RATE
    pvalues, cvs = [], []
    for t in hour_runs:
        gaps = np.diff(t)
        assert gaps.mean() == pytest.approx(scale, rel=0.05)
        cvs.append(gaps.std(ddof=1) / gaps.mean())
        pvalues.append(stats.kstest(gaps, 'expon', args=(0, scale)).pvalue)

    # The property a fixed cadence lacks: unit coefficient of variation.
    assert np.mean(cvs) == pytest.approx(1.0, abs=0.05), \
        f"mean CV={np.mean(cvs):.3f}; a deterministic cadence would give 0.0"

    assert np.median(pvalues) > 0.05, \
        f"gaps not exponential (median KS p={np.median(pvalues):.4f})"


def test_closed_hour_releases_no_burst(sim):
    """A zero-rate hour drops the pending gap, so reopening does not fire the
    arrivals that would have fallen due while the store was closed."""
    sim.spawn_rate = RATE
    sim.arrival_rng = np.random.default_rng(3)
    sim.hourly_profile = np.ones(24)
    sim.hourly_profile[9] = 0.0
    sim.sim_clock_start_hour = 8.99      # 36 s open, then the closed hour

    first = _drive_arrivals(sim, HOUR_S)
    assert np.all(first < 36.0 + 2 * TICK_S), "arrivals during a closed hour"
    assert sim._next_spawn_gap is None

    # Reopens at 3636 s; one second of trading at RATE.
    reopened = _drive_arrivals(sim, 37.0)
    assert np.all(reopened >= 3636.0 - 2 * TICK_S)
    assert len(reopened) < 15, f"{len(reopened)} arrivals in the first second"


def test_full_store_consumes_arrivals_as_balks(sim):
    """At capacity an arrival is counted as a balk and dropped, not deferred."""
    sim.spawn_rate = RATE
    sim.arrival_rng = np.random.default_rng(5)
    sim.max_customers = 1
    sim.customers = [object()]
    spawned = []
    sim._spawn_customer = lambda: spawned.append(sim.sim_time)

    n_arrivals = 0
    for _ in range(int(round(100.0 / TICK_S))):
        sim.sim_time += TICK_S
        n_arrivals += sim._process_arrivals(TICK_S)

    assert not spawned
    assert n_arrivals > 0
    assert sim.analytics['balked_arrivals'] == n_arrivals


def test_hourly_profile_preserves_mean_rate(sim):
    """The simulator's default hour-of-day profile has unit mean (Eq. 5)."""
    assert sim.hourly_profile.shape == (24,)
    assert float(sim.hourly_profile.mean()) == pytest.approx(1.0, abs=1e-9)


def test_literature_hourly_profile_is_normalized():
    """The shipped non-uniform profile satisfies Eq. 5 over OPEN hours.

    The simulator's default is uniform, so the assertion above holds
    trivially and says nothing about the literature-derived profile --
    which was in fact shipped un-normalized (mean 1.29 over open hours)
    until this test was added. Closed hours are excluded because the
    equation normalizes over the open-hours set, not all 24.
    """
    from retail_literature import DEFAULT_HOURLY_PROFILE

    prof = np.asarray(DEFAULT_HOURLY_PROFILE, dtype=float)
    assert prof.shape == (24,)

    open_hours = prof[prof > 0.0]
    assert open_hours.size > 0
    assert float(open_hours.mean()) == pytest.approx(1.0, abs=1e-9)

    # Normalizing must not flatten the shape it exists to represent.
    assert open_hours.max() / open_hours.min() > 2.0
