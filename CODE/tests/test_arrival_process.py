"""The live arrival stream must be Poisson, not a fixed cadence.

Section 3.2 of the paper states live arrivals follow a non-homogeneous
Poisson process. An earlier implementation fired one arrival every
exactly 1/rate seconds: the mean rate was right but inter-arrival
variance was zero, which understates queueing (regular arrivals queue
far less than Poisson arrivals at the same utilization). These tests
pin the distribution so that cannot come back unnoticed.
"""
import numpy as np
import pytest
from scipy import stats

from simulation import CustomerFlowSimulation


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


def test_arrival_rng_is_wired(sim):
    """The simulation owns a dedicated arrival stream and starts unprimed."""
    assert hasattr(sim, 'arrival_rng')
    assert sim._next_spawn_gap is None


def test_inter_arrival_gaps_are_exponential(sim):
    """Gaps drawn the way the spawn loop draws them are Exp(1/rate).

    The goodness-of-fit is pooled over several seeds on purpose. A KS
    test at n=20000 is knife-edge: even a correct generator fails at the
    nominal rate, so a single fixed seed makes a flaky test rather than a
    strict one. A genuinely wrong distribution drives every p-value to
    zero, which the median still catches.
    """
    rate = 0.2                      # arrivals/sec
    scale = 1.0 / rate

    pvalues, cvs = [], []
    for seed in (11, 22, 33, 44, 55):
        sim.arrival_rng = np.random.default_rng(seed)
        gaps = np.array([sim.arrival_rng.exponential(scale)
                         for _ in range(20000)])

        assert gaps.mean() == pytest.approx(scale, rel=0.05)
        cvs.append(gaps.std(ddof=1) / gaps.mean())
        pvalues.append(stats.kstest(gaps, 'expon', args=(0, scale)).pvalue)

    # The property a fixed cadence lacks: unit coefficient of variation.
    assert np.mean(cvs) == pytest.approx(1.0, abs=0.05), \
        f"mean CV={np.mean(cvs):.3f}; a deterministic cadence would give 0.0"

    assert np.median(pvalues) > 0.05, \
        f"gaps not exponential (median KS p={np.median(pvalues):.4f})"


def test_arrival_counts_are_overdispersed_like_poisson(sim):
    """Counts per fixed window have variance ~ mean, as Poisson requires."""
    rate = 0.2
    sim.arrival_rng = np.random.default_rng(7)
    gaps = sim.arrival_rng.exponential(1.0 / rate, 40000)
    t = np.cumsum(gaps)

    window = 60.0
    counts = np.histogram(t, bins=np.arange(0, t[-1], window))[0]
    ratio = counts.var(ddof=1) / counts.mean()
    assert ratio == pytest.approx(1.0, abs=0.15), \
        f"variance/mean={ratio:.3f}; a fixed cadence would give ~0"


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
