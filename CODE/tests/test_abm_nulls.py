"""The ABM diagnostics' extra baselines (review R13).

The perimeter ratio and the second-order Markov gain could both be
produced by construction: the endpoints of every visit (door, lanes,
washroom) sit in the perimeter band, the heat map counts agents standing
still, and the state machine's bookkeeping makes the next state depend on
the previous one. ``run_abm_diagnostics`` therefore reports a routing null
(shortest paths between the endpoints), a moving-only ratio, and a
bookkeeping-only Markov null. These tests pin the bookkeeping generator to
the real automaton, and the routing null to the agents' own planner.
"""
import itertools

import numpy as np
import pandas as pd
import pytest

from dataset_calibration import calibrate_transactional
from experiments import run_abm_diagnostics as ABM
from experiments._common import (HeadlessShop,
                                 build_headless_shop_from_calibration)


# --- the bookkeeping generator against the real automaton -------------------

def _open_store():
    """A store with room to walk round everything: five small shelves, a
    lane and a washroom, no impulse display."""
    shop = HeadlessShop(12.0, 10.0)
    shop.door_position, shop.door_side = (6.0, 0.0), 'bottom'
    f1 = shop.floors[1]
    f1['door_position'], f1['door_side'] = shop.door_position, shop.door_side
    f1['items'] = {
        name: {'position': pos, 'size': (0.6, 0.4), 'category': 'Kitchen',
               'price': 2.0}
        for name, pos in (('A', (2.0, 7.0)), ('B', (6.0, 7.0)),
                          ('C', (9.0, 7.0)), ('D', (3.5, 4.5)),
                          ('E', (8.5, 4.5)))}
    f1['prices'] = {k: 2.0 for k in f1['items']}
    f1['walls'] = {'Checkout': {'position': (5.0, 1.2), 'size': (1.5, 0.6)},
                   'WC': {'position': (0.5, 8.5), 'size': (1.0, 1.0)}}
    sim = shop.customer_simulation
    sim.door_position, sim.door_side = shop.door_position, shop.door_side
    sim.spawn_rate = 0.0
    sim.record_state_histories = True
    sim.step(0.04)
    return sim


def _real_sequence(n_items, needs_wc, abandon):
    sim = _open_store()
    sim._spawn_customer()
    cust = sim.customers[-1]
    cust.shopping_list = ['A', 'B', 'C', 'D', 'E'][:n_items]
    cust.needs_wc, cust.visited_wc = needs_wc, False
    cust.abandon_cart = abandon
    for _ in range(int(600 / 0.04)):
        sim.step(0.04)
        if not sim.customers:
            break
    assert not sim.customers, 'the visit did not finish'
    assert not sim.suppressed_errors
    (rec,) = sim.completed_state_histories
    return rec['states']


@pytest.mark.parametrize('n_items, needs_wc, abandon', list(itertools.product(
    (1, 2, 3, 5), (False, True), (False, True))))
def test_bookkeeping_sequence_is_the_automatons(n_items, needs_wc, abandon):
    """With nothing in the way the agents' state machine produces exactly
    the sequence the generator writes down for the same draws."""
    assert ABM.bookkeeping_sequence(n_items, needs_wc, abandon) == \
        _real_sequence(n_items, needs_wc, abandon)


def test_bookkeeping_sequences_follow_their_draws():
    ctx = {'list_lengths': [1, 2, 7], 'has_wc': True,
           'n_impulse_candidates': 0}
    a = ABM.bookkeeping_sequences(300, ctx, np.random.default_rng(5))
    b = ABM.bookkeeping_sequences(300, ctx, np.random.default_rng(5))
    assert a == b
    assert {s[-1] for s in a} == {'purchased', 'abandoned'}
    shelf_visits = {sum(1 for x in s if x == 'shopping') for s in a}
    # One shopping state per item, plus one for the washroom.
    assert shelf_visits <= {1, 2, 3, 7, 8}
    with pytest.raises(ValueError):
        ABM.bookkeeping_sequences(1, {'list_lengths': []},
                                  np.random.default_rng(0))


def test_the_bookkeeping_alone_carries_second_order_memory():
    """The outcome closing each sequence depends on the state before
    'exiting', so the generator already shows a positive information gain
    beyond its first-order permutation null."""
    ctx = {'list_lengths': list(range(1, 12)), 'has_wc': True,
           'n_impulse_candidates': 0}
    seqs = ABM.bookkeeping_sequences(3000, ctx, np.random.default_rng(1))
    mk = ABM.markov_order_analysis(seqs, n_perm=50)
    assert mk['info_gain_excess_bits'] > 0.0
    assert mk['info_gain_perm_p'] < 0.05


# --- the routing null ---------------------------------------------------------

def _invoices(n_invoices=400, seed=0):
    rng = np.random.default_rng(seed)
    products = [(f'{cat[:3].upper()}{i}', cat, 1.5 + i)
                for cat in ('Bakery', 'Dairy', 'Garden') for i in range(4)]
    day0 = pd.Timestamp('2010-12-01 09:00')
    rows = []
    for k in range(n_invoices):
        ts = day0 + pd.Timedelta(days=k // 40,
                                 minutes=int(rng.integers(0, 480)))
        for j in rng.choice(len(products), size=int(rng.integers(1, 5)),
                            replace=False):
            pid, cat, price = products[j]
            rows.append((f'I{k:05d}', pid, f'ITEM {pid}', 1, ts, price, cat))
    return pd.DataFrame(rows, columns=['invoice_id', 'product_id',
                                       'product_name', 'quantity',
                                       'timestamp', 'unit_price', 'category'])


@pytest.fixture(scope='module')
def store():
    params = calibrate_transactional(_invoices(), currency='GBP')
    return build_headless_shop_from_calibration(params,
                                                max_items_per_category=4,
                                                naive=True)


def test_routing_null_walks_whole_visits(store):
    """One tour per visit: every drawn list item walked once, one exit leg
    per tour and a lane leg for every paid one, and the map's mass is the
    metres the planner's paths cover."""
    n = 300
    heat, info = ABM.routing_null_heat(store, n_tours=n, seed=11)
    sim = store.customer_simulation
    assert info['construction'] == 'whole_visit_tours'
    assert info['list_source'] == 'stocked_invoices'
    assert info['n_tours'] == n and info['seed'] == 11
    assert info['unroutable_legs'] == 0
    # The same draws, replayed: each tour walks its list once.
    ctx = ABM.routing_context(store)
    rng = np.random.default_rng(11)
    draws = [ABM.routing_tour_draw(ctx, rng) for _ in range(n)]
    assert info['n_list_items_walked'] == sum(len(d[0]) for d in draws)
    assert info['abandonment']['n_abandoned'] == sum(d[2] for d in draws)
    assert info['n_exit_legs'] == n
    assert info['n_lane_legs'] == n - info['abandonment']['n_abandoned']
    # Rasterised per metre walked: the map's mass is the tours' length.
    assert heat.sum() == pytest.approx(info['mean_tour_m'] * n, rel=1e-6)
    walkable = ABM.walkable_heat_cells(
        sim.path_blocked_grid_by_floor[1], sim.path_grid_resolution,
        heat.shape, sim.heat_map_resolution)
    assert heat[walkable].sum() >= 0.99 * heat.sum()
    stats = ABM.emergence_stats(heat, walkable, store.width, store.height,
                                routing=heat)
    assert stats['ratio_to_routing_null'] == pytest.approx(1.0)
    # The per-tour ratio estimator is the map's own ratio.
    assert info['ratio_by_band']['2.5'] == pytest.approx(
        stats['perimeter_interior_ratio_routing_null'], abs=6e-4)
    assert 0 < info['ratio_mc_se_by_band']['2.5'] <         0.5 * info['ratio_by_band']['2.5']
    again, _ = ABM.routing_null_heat(store, n_tours=n, seed=11)
    assert np.array_equal(heat, again)
    other, _ = ABM.routing_null_heat(store, n_tours=n, seed=12)
    assert not np.array_equal(heat, other)


def _states_of(stops):
    """The jump sequence an agent with no geometry makes through
    ``stops``: every stop is reached by a 'moving' leg, except the door,
    which an agent walks to in 'exiting'; after the washroom the agent is
    'moving' again."""
    seq = ['entering']

    def go(state):
        if seq[-1] != state:
            seq.append(state)

    paid = False
    for kind, _ in stops:
        if kind == 'exit':
            go('exiting')
            continue
        go('moving')
        go({'item': 'shopping', 'wc': 'shopping',
            'checkout': 'checking_out'}[kind])
        if kind == 'wc':
            go('moving')
        paid = paid or kind == 'checkout'
    return seq + ['purchased' if paid else 'abandoned']


@pytest.mark.parametrize('n_items, needs_wc, abandon', list(itertools.product(
    range(1, 9), (False, True), (False, True))))
def test_tour_stops_follow_the_automaton(n_items, needs_wc, abandon):
    """The tour's stops make the jump sequence the bookkeeping generator
    (pinned to the real automaton above) makes for the same draws, and the
    washroom comes after the first shelf visit at which half the list,
    rounded down, is done -- ``Customer.update``'s rule."""
    items = [f'i{k}' for k in range(n_items)]
    stops = ABM.routing_tour_stops(items, needs_wc, abandon)
    assert _states_of(stops) == ABM.bookkeeping_sequence(n_items, needs_wc,
                                                         abandon)
    kinds = [k for k, _ in stops]
    assert [key for k, key in stops if k == 'item'] == items
    assert kinds[-1] == 'exit' and kinds.count('exit') == 1
    assert kinds.count('checkout') == (0 if abandon else 1)
    assert kinds.count('wc') == (1 if needs_wc else 0)
    if needs_wc:
        assert kinds.index('wc') == max(1, n_items // 2)
    no_wc = ABM.routing_tour_stops(items, needs_wc, abandon, has_wc=False)
    assert 'wc' not in [k for k, _ in no_wc]


def test_tour_draws_take_whole_invoices_uniformly():
    ctx = {'invoice_keys': ['a', 'b', 'c', 'd'],
           'invoice_ptr': np.array([0, 1, 3, 6]),
           'invoice_items': np.array([0, 1, 2, 0, 2, 3]),
           'regular_items': ['a', 'b', 'c', 'd'], 'has_wc': True,
           'list_source': 'stocked_invoices'}
    rng = np.random.default_rng(3)
    seen = {}
    for _ in range(6000):
        lst, _, _ = ABM.routing_tour_draw(ctx, rng)
        key = tuple(sorted(lst))
        seen[key] = seen.get(key, 0) + 1
    assert set(seen) == {('a',), ('b', 'c'), ('a', 'c', 'd')}
    for n in seen.values():
        assert n / 6000 == pytest.approx(1 / 3, abs=0.03)
    # Without stored invoices the type law's lengths, uniform items.
    bare = dict(ctx, invoice_keys=[], invoice_ptr=np.zeros(1, dtype=int),
                invoice_items=np.zeros(0, dtype=int))
    lo = min(v[0] for v in ABM.LIST_LENGTH_BY_TYPE.values())
    for _ in range(50):
        lst, _, _ = ABM.routing_tour_draw(bare, rng)
        assert min(lo, 4) <= len(lst) <= 4 and len(set(lst)) == len(lst)


def test_moving_block_reads_its_own_map():
    W, H = 6.0, 4.0
    shape = (int(W * 20) + 1, int(H * 20) + 1)
    walkable = np.ones(shape, dtype=bool)
    heat = np.ones(shape)
    moving = np.zeros(shape)
    moving[:20, :] = 1.0                   # walkers along one wall only
    s = ABM.emergence_stats(heat, walkable, W, H, band_m=1.0, routing=heat,
                            moving=moving)
    assert s['perimeter_interior_ratio'] == 1.0
    assert s['moving']['perimeter_interior_ratio'] == \
        ABM.perimeter_ratio(moving, W, H, 1.0)['perimeter_interior_ratio']
    assert s['moving']['ratio_to_routing_null'] == \
        s['moving']['perimeter_interior_ratio']
    assert set(s['moving']['band_sweep']) == set(s['band_sweep'])
