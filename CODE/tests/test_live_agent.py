"""Live agent behaviour the live diagnostics depend on.

Covers the stuck detector (net displacement over a simulated-time window),
abandonment on every route out of the shopping phase, single-server FIFO
checkout lanes, the simulation reference being available while the
shopping list is drawn, and the state histories that feed the empirical
Markov estimator. Everything runs on a Tk-free stub shop, driving
``Customer.update`` tick by tick without the simulation thread.
"""
import numpy as np
import pytest

from customer import Customer
from retail_literature import STUCK_PROGRESS_M, STUCK_WINDOW_MOVING_S
from simulation import CustomerFlowSimulation


W, H = 12.0, 10.0
DOOR = (6.0, 0.0)

ITEMS = {
    'Near': {'position': (1.0, 8.0), 'size': (0.6, 0.6),
             'category': 'Dairy', 'price': 2.0},
    'Far':  {'position': (10.5, 8.5), 'size': (0.6, 0.6),
             'category': 'Bakery', 'price': 3.0},
}
LANES = {
    'Checkout':       {'position': (8.0, 1.0), 'size': (1.5, 0.6)},
    'Checkout_Lane2': {'position': (2.0, 1.0), 'size': (1.5, 0.6)},
    'WC':             {'position': (0.5, 4.0), 'size': (1.0, 1.0)},
}


class _StubShop:
    """Minimum shop surface the simulation and its agents read."""

    def __init__(self, walls):
        self.width, self.height = W, H
        self.current_floor = 1
        self.num_floors = 1
        self.connectors = {}
        self.door_position, self.door_side = DOOR, 'bottom'
        self.canvas = None
        self.current_tab = None
        self.floors = {1: {
            'items': {k: dict(v) for k, v in ITEMS.items()},
            'walls': walls,
            'prices': {k: v['price'] for k, v in ITEMS.items()},
            'door_position': DOOR, 'door_side': 'bottom',
        }}

    @property
    def items(self):
        return self.floors[1]['items']

    @property
    def walls(self):
        return self.floors[1]['walls']

    @property
    def prices(self):
        return self.floors[1]['prices']

    def all_items_across_floors(self):
        out = {}
        for fid, fdata in self.floors.items():
            for name, data in fdata['items'].items():
                d = dict(data)
                d['floor'] = fid
                out[name] = d
        return out

    def _assign_default_prices(self):
        pass


def _make_sim(walls):
    np.random.seed(1234)
    sim = CustomerFlowSimulation(_StubShop(walls))
    sim.door_position, sim.door_side = DOOR, 'bottom'
    sim._rebuild_geometry_caches()
    sim.geometry_dirty = False
    return sim


@pytest.fixture
def sim():
    return _make_sim({k: dict(v) for k, v in LANES.items()})


def _agent(sim, pos):
    c = Customer(sim.customer_counter, pos, sim.shop.all_items_across_floors(),
                 (W, H), door_position=DOOR, door_side='bottom',
                 simulation_ref=sim)
    sim.customer_counter += 1
    c.needs_wc = False
    sim.customers.append(c)
    return c


# --- stuck detection ---------------------------------------------------------

def test_open_floor_walk_is_not_flagged_stuck(sim):
    """A slow walker on an open floor keeps its target past the window.

    At 0.9 m/s and a 0.04 s tick each step is 0.036 m, so a detector that
    looks for progress within a single tick never sees any.
    """
    items, walls = sim.shop.all_items_across_floors(), sim.shop.walls
    c = _agent(sim, [1.0, 1.0])
    c.speed = 0.9
    c.shopping_list = ['Far']
    c._change_state('moving')
    c._choose_next_target(items, walls)
    start = list(c.position)

    for _ in range(int(round(4.0 / 0.04))):   # longer than the moving window
        c.update(0.04, items, walls, sim.customers)

    assert c.state == 'moving'
    assert c.current_target_item == 'Far'
    assert 'Far' not in c.abandoned_items
    assert np.hypot(c.position[0] - start[0], c.position[1] - start[1]) > 2.0


def test_stall_window_is_simulated_time_without_net_progress(sim):
    c = _agent(sim, [5.0, 5.0])
    c._reset_progress()
    for _ in range(70):                       # 2.8 s standing still
        assert not c._stalled(0.04, STUCK_WINDOW_MOVING_S)

    c.position[0] += 0.8 * STUCK_PROGRESS_M   # creeping is not progress
    stalled = [c._stalled(0.04, STUCK_WINDOW_MOVING_S) for _ in range(10)]
    assert stalled[-1]                        # 3.2 s without net progress

    c.position[0] += 0.4 * STUCK_PROGRESS_M   # now past the anchor distance
    assert not c._stalled(0.04, STUCK_WINDOW_MOVING_S)
    assert c._no_progress_s == 0.0


def test_state_change_and_new_target_restart_the_window(sim):
    walls = sim.shop.walls
    c = _agent(sim, [5.0, 5.0])
    c._reset_progress()

    c._no_progress_s = 2.9
    c._change_state('shopping')
    assert c._no_progress_s == 0.0

    c._no_progress_s = 2.9
    c.position = [5.1, 5.0]
    c.target_position = [9.0, 8.0]
    c._initialize_path(walls)
    assert c._no_progress_s == 0.0
    assert c._last_progress_pos == (5.1, 5.0)


def test_boxed_in_exiting_agent_teleports_only_after_the_window():
    walls = {k: dict(v) for k, v in LANES.items()}
    walls['Pillar'] = {'position': (4.5, 4.5), 'size': (1.0, 1.0)}
    sim = _make_sim(walls)
    items = sim.shop.all_items_across_floors()
    c = _agent(sim, [5.0, 5.0])               # inside the pillar: cannot move
    c._change_state('exiting')
    c._set_exit_target(walls)

    for _ in range(120):                      # 4.8 s: still inside the window
        c.update(0.04, items, walls, sim.customers)
    assert c.position == [5.0, 5.0]

    for _ in range(15):                       # 5.4 s
        c.update(0.04, items, walls, sim.customers)
    assert c.position == list(DOOR)
    assert c.current_target_type == 'exit'


# --- abandonment ---------------------------------------------------------------

@pytest.mark.parametrize('abandon', [True, False])
def test_wc_trip_after_last_item_honours_abandonment(sim, abandon):
    """The WC can come after the last item; the agent must still abandon."""
    items, walls = sim.shop.all_items_across_floors(), sim.shop.walls
    c = _agent(sim, [1.3, 7.5])
    c.abandon_cart = abandon
    c.needs_wc, c.visited_wc = True, False
    c.shopping_list = ['Near']
    c.current_target_item, c.current_target_type = 'Near', 'item'
    c._change_state('shopping')
    c.dwell_time = 0.0

    c.update(0.04, items, walls, sim.customers)   # item done -> WC
    assert (c.state, c.current_target_type) == ('moving', 'wc')

    c._change_state('shopping')                   # at the WC, dwell over
    c.visited_wc, c.dwell_time = True, 0.0
    c.update(0.04, items, walls, sim.customers)

    if abandon:
        assert (c.state, c.current_target_type) == ('exiting', 'exit')
    else:
        assert (c.state, c.current_target_type) == ('moving', 'checkout')
        assert c.checkout_lane in ('Checkout', 'Checkout_Lane2')
    assert not c.has_checked_out


@pytest.mark.parametrize('abandon', [True, False])
def test_exhausted_list_honours_abandonment(sim, abandon):
    """Nothing left to visit (e.g. the stuck handler dropped the rest)."""
    items, walls = sim.shop.all_items_across_floors(), sim.shop.walls
    c = _agent(sim, [5.0, 5.0])
    c.abandon_cart = abandon
    c.shopping_list = []
    c._change_state('moving')
    c._choose_next_target(items, walls)
    expected = ('exiting', 'exit') if abandon else ('moving', 'checkout')
    assert (c.state, c.current_target_type) == expected


def test_paid_agent_is_not_turned_into_an_abandonment(sim):
    c = _agent(sim, [5.0, 5.0])
    c.abandon_cart = True
    c.has_checked_out = True
    c._finish_shopping(sim.shop.walls)
    assert c.current_target_type == 'checkout'


# --- checkout lanes --------------------------------------------------------------

@pytest.mark.parametrize('reverse_order', [False, True])
def test_lane_serves_one_customer_at_a_time(sim, reverse_order):
    """Three customers reach one lane together, each needing 3 s of service.

    A single server finishes them at 3, 6 and 9 s with waits of 0, 3 and
    6 s; concurrent service would finish all three at 3 s. The order in
    which agents update within a tick must not change the result.
    """
    items, walls = sim.shop.all_items_across_floors(), sim.shop.walls
    dt = 0.125                                    # exact in binary
    agents = []
    for _ in range(3):
        c = _agent(sim, [8.75, 1.3])
        c.checkout_lane = 'Checkout'
        c._change_state('checking_out')
        c.dwell_time = 3.0
        c._join_lane_queue()
        agents.append(c)
    q = sim.lane_queues['Checkout']
    assert q == [c.id for c in agents]

    finished = {}
    order = agents[::-1] if reverse_order else agents
    for step in range(1, 200):
        sim.sim_time = step * dt
        for c in order:
            if c.id not in finished:
                c.update(dt, items, walls, sim.customers)
                if c.has_checked_out:
                    finished[c.id] = sim.sim_time
        if step * dt == 1.0:
            assert max(len(q) - 1, 0) == 2      # one in service, two waiting
        if len(finished) == 3:
            break

    assert [finished[c.id] for c in agents] == pytest.approx([3.0, 6.0, 9.0])
    assert sim.analytics['queue_wait_times'] == pytest.approx([0.0, 3.0, 6.0])
    assert q == []


def test_lane_choice_counts_queue_length(sim):
    walls = sim.shop.walls
    c = _agent(sim, [8.75, 2.0])                  # right next to 'Checkout'
    sim.lane_queues['Checkout'] = [900, 901]
    assert c._pick_checkout_lane(walls)[0] == 'Checkout_Lane2'
    sim.lane_queues['Checkout'] = []
    assert c._pick_checkout_lane(walls)[0] == 'Checkout'


def test_lane_queues_forget_customers_that_are_gone(sim):
    c = _agent(sim, [8.75, 1.3])
    c.checkout_lane = 'Checkout'
    c._change_state('checking_out')
    c._join_lane_queue()
    sim.lane_queues['Checkout'].insert(0, 999)    # cleared from the store
    sim._prune_lane_queues()
    assert sim.lane_queues['Checkout'] == [c.id]

    sim._drop_from_lane_queues(c.id)
    assert sim.lane_queues['Checkout'] == []

    c._join_lane_queue()
    sim.hard_stop()
    assert sim.lane_queues == {}
    assert sim.customers == []


# --- spawn wiring and state histories --------------------------------------------

def test_simulation_reference_is_set_before_the_list_is_drawn(sim, monkeypatch):
    seen = []
    original = Customer._generate_shopping_list

    def spy(self, shop_items):
        seen.append(getattr(self, '_simulation_ref', None))
        return original(self, shop_items)

    monkeypatch.setattr(Customer, '_generate_shopping_list', spy)
    sim.sim_time = 12.5
    sim._spawn_customer()

    assert len(seen) == 1 and seen[0] is sim
    assert sim.customers[-1]._simulation_ref is sim
    assert sim.customers[-1].spawn_sim_time == 12.5


def test_departure_records_complete_history_and_markov_counts():
    sim = _make_sim({})                           # no checkout lane at all
    sim.record_state_histories = True
    c = _agent(sim, [5.0, 5.0])
    assert c.state_history == ['entering']
    assert c.spawn_sim_time == 0.0

    c._change_state('moving')
    c._change_state('shopping')
    sim._update_analytics_realtime(c, 0.5)        # roll-up mid-visit
    c._change_state('moving')
    c._set_checkout_target({})                    # no lane: pays and leaves
    assert c.state == 'exiting' and c.has_checked_out
    sim._update_analytics_realtime(c, 0.25)
    sim.sim_time = 42.0
    sim._process_customer_exit(c)

    rec = sim.completed_state_histories[-1]
    assert rec['states'] == ['entering', 'moving', 'shopping', 'moving',
                             'exiting', 'purchased']
    assert rec['spawn_t'] == 0.0
    assert rec['exit_t'] == 42.0

    # Each jump is counted once, whether it was logged mid-visit or at exit.
    assert dict(sim.analytics['markov_transition_counts']) == {
        ('entering', 'moving'): 1,
        ('moving', 'shopping'): 1,
        ('shopping', 'moving'): 1,
        ('moving', 'exiting'): 1,
        ('exiting', 'purchased'): 1,
    }
    occ = sim.analytics['markov_state_occupancy']
    assert occ['shopping'] == 0.5
    assert occ['exiting'] == 0.25


# --- exit routes ---------------------------------------------------------------

def test_exits_are_tallied_by_the_route_that_sent_them_out(sim):
    """Every exit is tallied by how it came about (sim_analytics'
    ``exit_routes``): paid, unpaid through the spawn-time abandonment
    draw, unpaid with no route left to walk. The structural sweep reads
    these to tell a movement-dependent unpaid exit from the draw."""
    walls = sim.shop.walls
    drawn = _agent(sim, [5.0, 5.0])
    drawn.abandon_cart = True
    drawn.shopping_list = []
    drawn._finish_shopping(walls)
    assert drawn.exit_route == 'abandon_draw'
    stranded = _agent(sim, [5.0, 5.0])
    stranded.abandon_cart = False
    stranded._leave_through_door()
    assert stranded.exit_route == 'no_route'
    paid = _agent(sim, [5.0, 5.0])
    paid.abandon_cart = False
    paid.has_checked_out = True
    paid._leave_through_door()        # a paid agent is never 'unpaid'
    assert paid.exit_route is None
    paid.moving_stalls = 2
    for c in (drawn, stranded, paid):
        sim._process_customer_exit(c)
    ex = sim.analytics['exit_routes']
    assert ex['paid'] == 1
    assert ex['unpaid_abandon_draw'] == 1 and ex['unpaid_no_route'] == 1
    assert ex['unpaid_abandon_drawn'] == 1 and ex['abandon_drawn'] == 1
    assert ex['moving_stalls'] == 2 and ex['exits_after_moving_stall'] == 1
    assert sim.analytics['abandoned_carts'] == 2
    assert sim.analytics['completed_purchases'] == 1


def test_the_exit_record_draws_nothing(sim):
    """Recording an exit route touches no random stream: an agent drawn
    after one that took each route gets the same draws as without."""
    walls = sim.shop.walls
    np.random.seed(7)
    c = _agent(sim, [5.0, 5.0])
    c.abandon_cart = True
    c._finish_shopping(walls)
    c._leave_through_door()
    after = np.random.random()
    np.random.seed(7)
    _agent(sim, [5.0, 5.0])
    assert np.random.random() == after


def test_structural_runner_reports_post_warmup_route_increments():
    from experiments.run_structural_sensitivity import _route_increments
    start = {'paid': 10, 'unpaid_abandon_draw': 1}
    end = {'paid': 40, 'unpaid_abandon_draw': 3, 'unpaid_no_route': 1}
    assert _route_increments(start, end) == {
        'paid': 30, 'unpaid_abandon_draw': 2, 'unpaid_no_route': 1}
