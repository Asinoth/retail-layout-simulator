"""The separation push is a rate, so the tick length is not part of the
model (review R41).

Two agents closer than ``SEPARATION_MIN_DIST_M`` are each pushed apart by
a share of their overlap. As a fixed share per tick, the push acted twice
as often per simulated second at a 0.02 s tick as at 0.04 s, and a head-on
pair of slow walkers then locked at a stable gap instead of passing each
other. As a rate, ``1 - exp(-k dt)`` with k set so the reference tick
reproduces the old share exactly, the pair passes at either tick, and every
fixed-step run at 0.04 s is unchanged.
"""
import math

import pytest

import retail_literature as RL
import simulation
from customer import Customer
from experiments._common import HeadlessShop
from simulation import separation_fraction


def test_rate_reproduces_the_old_share_at_the_reference_tick():
    s = RL.SEPARATION_STRENGTH
    # Exactly, not approximately: every run at 0.04 s must stay bit for bit.
    assert separation_fraction(s, RL.SEPARATION_REFERENCE_DT_S) == s
    assert RL.SEPARATION_REFERENCE_DT_S == simulation.MAX_STEP_S
    assert RL.SEPARATION_RATE_PER_S == pytest.approx(
        -math.log(1.0 - s) / RL.SEPARATION_REFERENCE_DT_S)
    for dt in (0.005, 0.01, 0.02, 0.03, 0.05, 0.1):
        assert separation_fraction(s, dt) == pytest.approx(
            1.0 - math.exp(-RL.SEPARATION_RATE_PER_S * dt))
    # Two half ticks keep the same share of the overlap as one full tick.
    half = separation_fraction(s, 0.02)
    assert (1.0 - half) ** 2 == pytest.approx(1.0 - s)


def test_share_is_monotone_and_bounded():
    s = RL.SEPARATION_STRENGTH
    shares = [separation_fraction(s, dt) for dt in (0.001, 0.01, 0.04, 0.4)]
    assert shares == sorted(shares)
    assert all(0.0 < f < 1.0 for f in shares)
    assert separation_fraction(0.0, 0.02) == 0.0
    assert separation_fraction(s, 0.0) == 0.0
    assert separation_fraction(1.0, 0.02) == 1.0


def _open_floor_sim():
    """A shop with nothing in it, geometry caches built without arrivals."""
    shop = HeadlessShop(10.0, 6.0)
    shop.door_position, shop.door_side = (5.0, 0.0), 'bottom'
    f1 = shop.floors[1]
    f1['door_position'], f1['door_side'] = shop.door_position, shop.door_side
    f1['items'], f1['walls'] = {}, {}
    sim = shop.customer_simulation
    sim.door_position, sim.door_side = shop.door_position, shop.door_side
    sim.spawn_rate = 0.0
    sim.step(0.04)
    assert not sim.customers
    return sim


def _head_on(sim, dt, seconds=15.0, speed=0.8):
    """Two walkers on one line, each heading for the other's start, moved
    and separated tick by tick. The slowest walking speed is the hardest
    case: the less an agent advances per tick, the easier the push holds it
    back. Returns their final x positions."""
    walls = sim.shop.floors[1]['walls']
    a = Customer(0, [2.0, 3.0], {}, (10.0, 6.0), simulation_ref=sim)
    b = Customer(1, [8.0, 3.0], {}, (10.0, 6.0), simulation_ref=sim)
    for agent, goal in ((a, [8.0, 3.0]), (b, [2.0, 3.0])):
        agent.speed = speed
        agent.state = 'moving'
        agent.target_position = goal
    for _ in range(int(round(seconds / dt))):
        a._move_towards_target(dt, walls)
        b._move_towards_target(dt, walls)
        sim._apply_separation([a, b], dt)
    return a.position[0], b.position[0]


@pytest.mark.parametrize('dt', [0.02, 0.04])
def test_a_head_on_pair_passes_at_either_tick(dt):
    sim = _open_floor_sim()
    xa, xb = _head_on(sim, dt)
    assert xa == pytest.approx(8.0, abs=1e-6)
    assert xb == pytest.approx(2.0, abs=1e-6)


def test_the_per_tick_rule_locked_the_pair_at_the_finer_tick(monkeypatch):
    """The regression the rate removes: the old fixed share per tick holds
    the same pair apart at 0.02 s, where it passes at 0.04 s."""
    monkeypatch.setattr(simulation, 'separation_fraction',
                        lambda strength, dt, ref_dt=None: strength)
    sim = _open_floor_sim()
    xa, xb = _head_on(sim, 0.02)
    assert xa < 5.0 < xb            # never got past each other
    sim = _open_floor_sim()
    xa, xb = _head_on(sim, 0.04)
    assert xa == pytest.approx(8.0, abs=1e-6)


def test_step_applies_the_push_at_its_own_tick(monkeypatch):
    """step() hands its dt to the separation rule, so the GUI's fixed tick
    and any headless tick get the push their length calls for."""
    seen = []
    sim = _open_floor_sim()
    monkeypatch.setattr(sim, '_apply_separation',
                        lambda customers, dt: seen.append(dt))
    sim.step(0.02)
    sim.step(0.04)
    assert seen == [0.02, 0.04]
