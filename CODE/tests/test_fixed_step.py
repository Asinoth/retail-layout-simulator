"""Fixed-step headless runs and the simulated clock.

The live diagnostics run the agent model through
``CustomerFlowSimulation.run_headless``: the same ``step(dt)`` the threaded
GUI loop calls, advanced in fixed ticks with no sleeping. These tests pin
what that mode promises. A seeded run is reproducible (state histories,
heat map and lane-queue records), a different seed gives a different run,
every agent and analytics timing reads simulated rather than wall-clock
time, and the threaded loop still drives the model through ``step``.

Shops come from the architecture engine (grid archetype, two checkout
lanes) mounted on the Tk-free ``HeadlessShop``, so walls, A* paths, lane
choice and FIFO queues are all exercised.
"""
import random
import time

import numpy as np
import pytest

from customer import Customer
from experiments._common import HeadlessShop
from shop_architecture import generate_architecture, scale_catalog


W, H = 14.0, 11.0
SHOP_TYPE = 'Grocery Store'
RUN_S = 400.0          # long enough for dozens of visits and lane queues


@pytest.fixture(scope='module')
def plan():
    catalog = scale_catalog(SHOP_TYPE, W, H)
    return generate_architecture(W, H, SHOP_TYPE, catalog,
                                 rng=random.Random(3), vary=False)


def _build_sim(plan, spawn_rate=0.25, cap=20):
    shop = HeadlessShop(W, H)
    f1 = shop.floors[1]
    f1['walls'] = {k: dict(v) for k, v in plan['walls'].items()}
    f1['items'] = {k: dict(v) for k, v in plan['items'].items()}
    f1['prices'] = {k: 2.95 + (i % 7) * 1.35
                    for i, k in enumerate(plan['items'])}
    shop.door_position = tuple(plan['door_position'])
    shop.door_side = plan['door_side']
    f1['door_position'], f1['door_side'] = shop.door_position, shop.door_side
    sim = shop.customer_simulation
    sim.spawn_rate, sim.max_customers = spawn_rate, cap
    sim.record_state_histories = True
    return sim


def _seeded_run(plan, seed, duration=RUN_S):
    """One headless run; returns the simulation and a per-second record of
    every lane queue's contents."""
    sim = _build_sim(plan)
    queue_records = []

    def _record(s):
        queue_records.append(
            (s.sim_time, sorted((lane, list(q)) for lane, q in s.lane_queues.items())))

    sim.run_headless(duration, seed=seed, callback=_record, callback_every_s=1.0)
    return sim, queue_records


@pytest.fixture(scope='module')
def runs(plan):
    return {'a': _seeded_run(plan, seed=7),
            'b': _seeded_run(plan, seed=7),
            'other': _seeded_run(plan, seed=8)}


def _frozen_wall_clock(monkeypatch):
    """Stop the wall clock and forbid sleeping for the rest of the test."""
    monkeypatch.setattr(time, 'time', lambda: 1.0e9)
    monkeypatch.setattr(time, 'perf_counter', lambda: 1.0e4)
    monkeypatch.setattr(time, 'monotonic', lambda: 1.0e4)

    def _no_sleep(_seconds):
        raise AssertionError('the simulation slept')

    monkeypatch.setattr(time, 'sleep', _no_sleep)


# --- determinism ----------------------------------------------------------------

def test_same_seed_reproduces_histories_heat_map_and_queues(runs):
    (a, qa), (b, qb) = runs['a'], runs['b']

    # The comparison only means something if the run has content: completed
    # visits, samples in the heat map, and lanes that actually queued.
    assert len(a.completed_state_histories) >= 20
    assert a.heat_raw.sum() > 0
    assert max(len(q) for _, rec in qa for _, q in rec) >= 2
    assert max(a.analytics['queue_wait_times']) > 0.0
    assert not a.suppressed_errors and not b.suppressed_errors

    assert a.completed_state_histories == b.completed_state_histories
    assert np.array_equal(a.heat_raw, b.heat_raw)
    assert sorted(a._floor_heat_raw) == sorted(b._floor_heat_raw)
    for fid in a._floor_heat_raw:
        assert np.array_equal(a._floor_heat_raw[fid], b._floor_heat_raw[fid])
    assert qa == qb
    assert a.analytics['queue_wait_times'] == b.analytics['queue_wait_times']
    assert a.analytics['total_revenue'] == b.analytics['total_revenue']
    assert a.sim_time == b.sim_time


def test_different_seed_gives_a_different_run(runs):
    (a, qa), (other, q_other) = runs['a'], runs['other']
    assert a.completed_state_histories != other.completed_state_histories
    assert not np.array_equal(a.heat_raw, other.heat_raw)
    assert qa != q_other


def test_callbacks_follow_simulated_time(plan):
    """Periodic calls land on the first tick at or past each multiple of the
    period; the closing call is not repeated when one already ran there."""
    for duration, expected in ((10.0, [2.52, 5.0, 7.52, 10.0]),
                               (11.0, [2.52, 5.0, 7.52, 10.0, 11.0])):
        sim = _build_sim(plan, spawn_rate=0.0)
        seen = []
        sim.run_headless(duration, dt=0.04, callback_every_s=2.5,
                         callback=lambda s: seen.append(s.sim_time))
        assert seen == pytest.approx(expected)
        assert sim.sim_time == pytest.approx(duration)


def test_heat_map_samples_once_per_simulated_second(plan):
    """At dt = 0.04 s a 1 s heat-map cadence is every 25th tick, not every
    26th once round-off in the summed dt accumulates."""
    sim = _build_sim(plan)
    tick = [0]
    sampled_at = []
    step, sample = sim.step, sim._update_floor_heatmaps

    def _counting_step(dt):
        tick[0] += 1
        return step(dt)

    def _recording_sample():
        sampled_at.append(tick[0])
        return sample()

    sim.step, sim._update_floor_heatmaps = _counting_step, _recording_sample
    sim.run_headless(60.0, dt=0.04, seed=5)
    assert sampled_at == list(range(25, 1501, 25))


# --- one clock ------------------------------------------------------------------

def test_visits_complete_with_the_wall_clock_stopped(plan, monkeypatch):
    """Entering delays, dwell, service and time in shop all elapse in
    simulated time: with the wall clock frozen, visits still run to the
    door and their durations are simulated seconds."""
    sim = _build_sim(plan)
    _frozen_wall_clock(monkeypatch)
    sim.run_headless(300.0, seed=11)

    recs = sim.completed_state_histories
    assert sum(1 for r in recs if r['states'][-1] == 'purchased') >= 10
    assert all(r['states'][:2] == ['entering', 'moving'] for r in recs)
    assert any('shopping' in r['states'] for r in recs)
    assert sim.analytics['markov_state_occupancy']['shopping'] > 0.0

    durations = [r['exit_t'] - r['spawn_t'] for r in recs]
    assert min(durations) > 0.0
    assert sim.analytics['average_time_in_shop'] == pytest.approx(np.mean(durations))


def test_agent_without_a_simulation_runs_on_the_dt_it_is_given(plan, monkeypatch):
    _frozen_wall_clock(monkeypatch)
    walls = plan['walls']
    items = {k: dict(v, floor=1) for k, v in plan['items'].items()}
    c = Customer(0, [W / 2, 0.5], items, (W, H),
                 door_position=tuple(plan['door_position']),
                 door_side=plan['door_side'], simulation_ref=None)
    c.entering_time = 0.3

    dt = 0.125                                    # exact in binary
    c.update(dt, items, walls, [c])
    c.update(dt, items, walls, [c])
    assert c.state == 'entering'                  # 0.25 s < 0.3 s
    c.update(dt, items, walls, [c])
    assert c.state == 'moving'                    # 0.375 s

    c._change_state('shopping')
    c.dwell_time = 0.5
    for _ in range(3):
        c.update(dt, items, walls, [c])
    assert c.state == 'shopping'
    c.update(dt, items, walls, [c])
    assert c.state != 'shopping'


def test_attached_agent_reads_the_simulation_clock_not_its_ticks(plan, monkeypatch):
    """An agent in a simulation times its entering delay on sim_time, so
    ticks that do not advance the simulated clock do not advance it."""
    _frozen_wall_clock(monkeypatch)
    sim = _build_sim(plan, spawn_rate=0.0)
    sim.step(0.04)                                # builds the geometry caches
    items = sim.shop.all_items_across_floors()
    walls = sim.shop.walls
    sim._spawn_customer()
    c = sim.customers[-1]
    c.entering_time = 0.3

    for _ in range(50):
        c.update(0.04, items, walls, sim.customers)
    assert c.state == 'entering'

    sim.sim_time += 0.4
    c.update(0.04, items, walls, sim.customers)
    assert c.state == 'moving'


def test_clock_restart_keeps_agents_elapsed_times(plan):
    """Clearing data restarts the simulated clock mid-run; agents still in
    the store keep their elapsed times instead of stalling at the door."""
    sim = _build_sim(plan, spawn_rate=0.0)
    sim.step(0.04)
    sim.sim_time = 500.0
    sim._spawn_customer()
    newcomer = sim.customers[-1]
    newcomer.entering_time = 0.3
    sim.run_headless(120.0, dt=0.04)              # older agents mid-visit
    sim.sim_time += 0.1
    sim._spawn_customer()
    entering = sim.customers[-1]
    entering.entering_time = 0.3
    sim.analytics['pre_window_start'] = sim.sim_time - 30.0

    before = {c.id: (sim.sim_time - c.state_start_time,
                     sim.sim_time - c.spawn_sim_time,
                     sim.sim_time - c.start_shop_time)
              for c in sim.customers}
    sim._restart_sim_clock()

    assert sim.sim_time == 0.0
    assert sim.analytics['pre_window_start'] == pytest.approx(-30.0)
    for c in sim.customers:
        assert (-c.state_start_time, -c.spawn_sim_time,
                -c.start_shop_time) == pytest.approx(before[c.id])

    sim.run_headless(0.4, dt=0.04)
    assert entering.state != 'entering'
    assert not sim.suppressed_errors


def test_a_seeded_run_refuses_a_store_that_already_holds_agents(plan):
    """An agent admitted before the streams are seeded would shape every
    later draw, so the seeded run would not be the one its seed names."""
    sim = _build_sim(plan)
    sim._spawn_customer()
    with pytest.raises(ValueError, match='empty store'):
        sim.run_headless(1.0, seed=3)
    sim.run_headless(0.4)                  # an unseeded run may carry on
    assert sim.sim_time > 0.0


# --- the threaded loop ------------------------------------------------------------

def test_threaded_loop_advances_the_model_through_step(plan):
    sim = _build_sim(plan, spawn_rate=0.5)
    calls = []
    original = sim.step

    def _spy(dt):
        calls.append(dt)
        return original(dt)

    sim.step = _spy
    sim.start_simulation()
    try:
        deadline = time.perf_counter() + 10.0
        while len(calls) < 5 and time.perf_counter() < deadline:
            time.sleep(0.02)
        with pytest.raises(RuntimeError):
            sim.run_headless(1.0)
    finally:
        sim.hard_stop()

    assert len(calls) >= 5
    assert all(dt >= 0.0 for dt in calls)
    assert sim.sim_time == pytest.approx(sum(calls))
    assert not sim.suppressed_errors
