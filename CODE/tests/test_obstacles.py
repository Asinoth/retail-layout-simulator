"""Fixtures are obstacles, and the live rules that go with them.

Shelving, gondolas and display tables are furniture: agents walk around
them and shop from the aisle beside them, which is what the floor plan's
aisles exist for. These tests pin that on both shops the project builds --
one from the architecture engine and one laid out from a calibration, plus
the UCI-calibrated store of the live diagnostics when the workbook is
available: nothing walks through a fixture over a seeded run, shoppers
stand beside the shelving rather than in it, every fixture has a standing
spot beside it that is reachable from the door, agents get round shelf
corners instead of stalling on them, visits still complete and the run is
still reproducible. The driver rules that depend
on the same geometry follow: a long frame is split into model-sized ticks
so walking speed does not follow the host's frame rate, an agent a layout
change buried inside a fixture is lifted out, an agent stranded on a floor
with no connector leaves instead of asking routing again, and a pending
arrival gap is redrawn when the arrival rate changes.
"""
import math
import os
import random

import numpy as np
import pandas as pd
import pytest

import dataset_adapters as DA
from customer import Customer
from customer_pathfinding import AGENT_RADIUS_M
from dataset_calibration import calibrate_transactional
from experiments._common import (HeadlessShop,
                                 build_headless_shop_from_calibration)
from shop_architecture import generate_architecture, scale_catalog
from simulation import MAX_STEP_S, MAX_SUBSTEPS, _substeps_for


W, H = 16.0, 12.0
SHOP_TYPE = 'Grocery Store'
RUN_S = 240.0            # long enough for dozens of visits on this floor
SEED = 4
IMPULSE = 'Gum'          # a low rack at the lane: reached over, not around
INSET = 0.2              # how far inside a fixture counts as "through" it


@pytest.fixture(scope='module')
def plan():
    catalog = scale_catalog(SHOP_TYPE, W, H)
    return generate_architecture(W, H, SHOP_TYPE, catalog,
                                 rng=random.Random(3), vary=False)


def _build_sim(plan, spawn_rate=0.3, cap=20):
    shop = HeadlessShop(W, H)
    f1 = shop.floors[1]
    f1['walls'] = {k: dict(v) for k, v in plan['walls'].items()}
    f1['items'] = {k: dict(v) for k, v in plan['items'].items()}
    lane = plan['walls']['Checkout']
    f1['items'][IMPULSE] = {
        'position': (lane['position'][0],
                     lane['position'][1] + lane['size'][1] + 0.1),
        'size': (0.4, 0.3), 'category': 'Impulse',
    }
    f1['prices'] = {k: 2.95 + (i % 7) * 1.35 for i, k in enumerate(f1['items'])}
    shop.door_position = tuple(plan['door_position'])
    shop.door_side = plan['door_side']
    f1['door_position'], f1['door_side'] = shop.door_position, shop.door_side
    sim = shop.customer_simulation
    sim.door_position, sim.door_side = shop.door_position, shop.door_side
    sim.spawn_rate, sim.max_customers = spawn_rate, cap
    sim.record_state_histories = True
    return sim


def _fixtures(sim):
    """The item rectangles agents must walk around."""
    return [(nm, d['position'][0], d['position'][1], d['size'][0], d['size'][1])
            for nm, d in sim.shop.floors[1]['items'].items()
            if nm != IMPULSE]


def _cell(sim, x, y):
    res = sim.path_grid_resolution
    return int(round(x / res)), int(round(y / res))


def _reachable_cells(blocked, start):
    """Cells a 4-connected walk from `start` can get to."""
    nx, ny = blocked.shape
    seen = np.zeros_like(blocked)
    seen[start] = True
    stack = [start]
    while stack:
        x, y = stack.pop()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = x + dx, y + dy
            if 0 <= a < nx and 0 <= b < ny and not blocked[a, b] and not seen[a, b]:
                seen[a, b] = True
                stack.append((a, b))
    return seen


def _seeded_run(plan, seed=SEED, duration=RUN_S):
    """One headless run, sampling every agent's position on every tick.

    Returns the simulation and, for each sample of a walking agent, whether
    it was inside a fixture other than the one it is shopping.
    """
    sim = _build_sim(plan)
    sim.step(0.04)                       # build the geometry caches
    fixtures = _fixtures(sim)
    samples = {'walking': 0, 'through': 0, 'where': [],
               'shopping': 0, 'standing_inside': 0}

    def _sample(s):
        for cust in s.customers:
            if cust.state == 'shopping':
                samples['shopping'] += 1
                x, y = cust.position
                if any(ix < x < ix + iw and iy < y < iy + ih
                       for _, ix, iy, iw, ih in fixtures):
                    samples['standing_inside'] += 1
                continue
            if cust.state not in ('entering', 'moving', 'exiting'):
                continue
            samples['walking'] += 1
            x, y = cust.position
            for nm, ix, iy, iw, ih in fixtures:
                if (ix + INSET <= x <= ix + iw - INSET
                        and iy + INSET <= y <= iy + ih - INSET):
                    if nm != getattr(cust, 'current_target_item', None):
                        samples['through'] += 1
                        if len(samples['where']) < 5:
                            samples['where'].append(
                                (round(s.sim_time, 2), cust.id, nm,
                                 round(x, 2), round(y, 2)))
                    break

    sim.run_headless(duration, dt=0.04, seed=seed,
                     callback=_sample, callback_every_s=0.04)
    return sim, samples


@pytest.fixture(scope='module')
def run(plan):
    return _seeded_run(plan)


# --- the obstacle set -------------------------------------------------------

def test_a_fixture_blocks_where_it_stands(run):
    """Its cells are off the grid and its body refuses a step; the impulse
    rack at the lane does neither."""
    sim, _ = run
    walls = sim.shop.walls
    blocked = sim.path_blocked_grid_by_floor[1]
    agent = Customer(0, list(sim.door_position), sim.shop.all_items_across_floors(),
                     (W, H), door_position=sim.door_position,
                     door_side=sim.door_side, simulation_ref=sim)

    for nm, ix, iy, iw, ih in _fixtures(sim):
        cx, cy = ix + iw / 2, iy + ih / 2
        assert blocked[_cell(sim, cx, cy)], f'{nm} is not on the blocked grid'
        assert agent._collides_with_interior(cx, cy, walls), \
            f'a step into {nm} was allowed'

    rack = sim.shop.floors[1]['items'][IMPULSE]
    rx = rack['position'][0] + rack['size'][0] / 2
    ry = rack['position'][1] + rack['size'][1] / 2
    assert not blocked[_cell(sim, rx, ry)]
    assert not agent._collides_with_interior(rx, ry, walls)


def test_access_point_is_a_free_spot_beside_the_fixture(run):
    """Agents shop from the aisle, so every fixture resolves to a standing
    spot that is free, and close enough that arriving there counts as
    reaching the fixture."""
    sim, _ = run
    blocked = sim.path_blocked_grid_by_floor[1]
    margin = AGENT_RADIUS_M + sim.path_grid_resolution + 0.05
    items = sim.shop.floors[1]['items']
    moved_off_centre = 0

    for nm, ix, iy, iw, ih in _fixtures(sim):
        ax, ay = sim.item_access_point(nm, items[nm], 1)
        assert not blocked[_cell(sim, ax, ay)], f'{nm} stands inside a fixture'
        dx = max(ix - ax, 0.0, ax - (ix + iw))
        dy = max(iy - ay, 0.0, ay - (iy + ih))
        assert max(dx, dy) <= margin + 1e-9, \
            f'{nm} access point {max(dx, dy):.3f} m away, margin {margin:.3f} m'
        if (ax, ay) != (ix + iw / 2, iy + ih / 2):
            moved_off_centre += 1

    assert moved_off_centre == len(_fixtures(sim))


def test_every_fixture_can_be_reached_from_the_door(run):
    sim, _ = run
    blocked = sim.path_blocked_grid_by_floor[1]
    inside = sim.nearest_free_position(sim.door_position[0],
                                       sim.door_position[1] + 0.4, 1)
    reach = _reachable_cells(blocked, _cell(sim, *inside))
    items = sim.shop.floors[1]['items']
    for nm, *_ in _fixtures(sim):
        ax, ay = sim.item_access_point(nm, items[nm], 1)
        assert reach[_cell(sim, ax, ay)], f'{nm} is walled off from the door'


# --- what the agents do with it ---------------------------------------------

def test_no_agent_walks_through_a_fixture(run):
    sim, samples = run
    assert samples['walking'] > 10_000, 'too few samples to mean anything'
    assert samples['through'] == 0, samples['where']


def test_shoppers_stand_in_the_aisle_not_inside_the_shelving(run):
    """An agent shopping a fixture stands at its access point, so it is
    never inside any fixture's body -- its own included."""
    _, samples = run
    assert samples['shopping'] > 1_000, 'too few samples to mean anything'
    assert samples['standing_inside'] == 0


def test_an_agent_that_rounds_a_shelf_corner_early_gets_past_it():
    """Layouts put fixture edges on round numbers, so a fixture's inflated
    footprint can end a hair short of a grid line and leave the aisle
    column beside it with no clearance at all. An agent that turns into
    that column a little early is in the fixture's shadow: it has to cover
    the sliver sideways before it can walk down the aisle, and must not
    stall there."""
    r = AGENT_RADIUS_M
    ox, oy, ow, oh = 2.25 - r - 1.0 - 1e-12, 1.0, 1.0, 2.0
    shop = HeadlessShop(6.0, 6.0)
    shop.door_position, shop.door_side = (3.0, 0.0), 'bottom'
    f1 = shop.floors[1]
    f1['door_position'], f1['door_side'] = shop.door_position, shop.door_side
    f1['items'] = {'Shelf': {'position': (ox, oy), 'size': (ow, oh),
                             'category': 'Pantry & Dry Goods', 'price': 2.0}}
    f1['walls'] = {}
    sim = shop.customer_simulation
    sim.door_position, sim.door_side = shop.door_position, shop.door_side
    sim.step(0.04)
    assert not sim.path_blocked_grid_by_floor[1][9, 8]   # (2.25, 2.0) is aisle

    sim._spawn_customer()
    cust = sim.customers[-1]
    cust.speed = 1.2
    cust.position = [2.0, oy + oh + r + 0.02]     # just above the corner
    cust.target_position = [2.25, 1.5]            # down the aisle beside it
    for _ in range(50):                           # 2 s
        cust._move_towards_target(0.04, f1['walls'])
        assert not cust._collides_with_interior(cust.position[0],
                                                cust.position[1], f1['walls'])
    assert cust.position == pytest.approx([2.25, 1.5])


def test_visits_still_complete(run):
    """Walking round the shelving must not leave agents stranded: they
    reach items, pay and leave, and nothing was swallowed on the way."""
    sim, _ = run
    recs = sim.completed_state_histories
    assert len(recs) >= 10
    assert sum(1 for r in recs if r['states'][-1] == 'purchased') >= 8
    assert any('shopping' in r['states'] for r in recs)
    assert sum(sim.analytics['popular_items'].values()) >= 20
    assert not sim.suppressed_errors


def test_same_seed_still_reproduces_the_run(plan, run):
    sim, _ = run
    again, _ = _seeded_run(plan)
    assert again.completed_state_histories == sim.completed_state_histories
    assert np.array_equal(again.heat_raw, sim.heat_raw)
    assert again.analytics['total_revenue'] == sim.analytics['total_revenue']


def test_a_layout_change_lifts_an_agent_out_of_a_fixture(plan):
    """Optimize and drag can drop shelving on top of somebody. An agent
    inside a fixture is blocked in every direction, so the rebuild moves it
    to the nearest spot it can stand on."""
    sim = _build_sim(plan, spawn_rate=0.0)
    sim.step(0.04)
    sim._spawn_customer()
    cust = sim.customers[-1]
    cust.position = [W / 2, H / 2]

    sim.shop.floors[1]['items']['Pallet'] = {
        'position': (W / 2 - 1.0, H / 2 - 1.0), 'size': (2.0, 2.0),
        'category': 'Pantry & Dry Goods',
    }
    sim.geometry_dirty = True
    sim.step(0.04)

    assert cust.position != [W / 2, H / 2]
    assert not cust._collides_with_interior(cust.position[0], cust.position[1],
                                            sim.shop.walls)
    assert not sim.suppressed_errors


# --- a store laid out from a calibration ------------------------------------

def _invoices(n_invoices=400, seed=0):
    """Invoices over four categories, enough for the layout builder to lay
    out a small store the way a loaded dataset does."""
    rng = np.random.default_rng(seed)
    products = [(f'{cat[:3].upper()}{i}', cat, 1.5 + i)
                for cat in ('Bakery', 'Dairy', 'Garden', 'Kitchen')
                for i in range(4)]
    day0 = pd.Timestamp('2010-12-01 09:00')
    rows = []
    for k in range(n_invoices):
        ts = day0 + pd.Timedelta(days=k // 40, minutes=int(rng.integers(0, 480)))
        for j in rng.choice(len(products), size=int(rng.integers(1, 5)),
                            replace=False):
            pid, cat, price = products[j]
            rows.append((f'I{k:05d}', pid, f'ITEM {pid}',
                         int(rng.integers(1, 4)), ts, price, cat))
    return pd.DataFrame(rows, columns=['invoice_id', 'product_id',
                                       'product_name', 'quantity',
                                       'timestamp', 'unit_price', 'category'])


@pytest.fixture(scope='module')
def dataset_shop():
    """The shop the dataset pipeline builds: shelving segments sized from
    the calibrated assortment, on a floor sized to fit them."""
    params = calibrate_transactional(_invoices(), currency='GBP')
    shop = build_headless_shop_from_calibration(params,
                                                max_items_per_category=4,
                                                naive=True)
    sim = shop.customer_simulation
    sim.spawn_rate, sim.max_customers = 0.25, 20
    sim.record_state_histories = True
    sim.step(0.04)
    return sim


def test_a_dataset_built_store_is_walkable(dataset_shop):
    """Same checks on the layout the Dataset button produces: every
    product's shelving has a standing spot beside it, reachable from the
    door, and a seeded run keeps agents out of the shelving."""
    sim = dataset_shop
    blocked = sim.path_blocked_grid_by_floor[1]
    margin = AGENT_RADIUS_M + sim.path_grid_resolution + 0.05
    items = sim.shop.floors[1]['items']
    inside = sim.nearest_free_position(sim.door_position[0],
                                       sim.door_position[1] + 0.4, 1)
    reach = _reachable_cells(blocked, _cell(sim, *inside))

    assert len(items) >= 8
    for nm, data in items.items():
        if 'impulse' in str(data.get('category', '')).lower():
            continue
        ax, ay = sim.item_access_point(nm, data, 1)
        cell = _cell(sim, ax, ay)
        assert not blocked[cell], f'{nm} stands inside a fixture'
        assert reach[cell], f'{nm} is walled off from the door'
        ix, iy = data['position']
        iw, ih = data['size']
        assert max(max(ix - ax, 0.0, ax - (ix + iw)),
                   max(iy - ay, 0.0, ay - (iy + ih))) <= margin + 1e-9

    fixtures = [(nm, d['position'][0], d['position'][1],
                 d['size'][0], d['size'][1]) for nm, d in items.items()
                if 'impulse' not in str(d.get('category', '')).lower()]
    through = []

    def _sample(s):
        for cust in s.customers:
            if cust.state not in ('entering', 'moving', 'exiting'):
                continue
            x, y = cust.position
            for nm, ix, iy, iw, ih in fixtures:
                if (ix + INSET <= x <= ix + iw - INSET
                        and iy + INSET <= y <= iy + ih - INSET
                        and nm != getattr(cust, 'current_target_item', None)):
                    through.append((round(s.sim_time, 2), cust.id, nm))
                    break

    sim.run_headless(180.0, dt=0.04, seed=6,
                     callback=_sample, callback_every_s=0.04)
    assert not through
    assert len(sim.completed_state_histories) >= 5
    assert not sim.suppressed_errors


# --- the store the live diagnostics use --------------------------------------

def _uci_workbook():
    """The UCI Online Retail II workbook under DATASETS/, or None."""
    code = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for root in (code, os.path.dirname(code)):
        path = os.path.join(root, 'DATASETS', 'UCI Online Retail II .xlsx.xlsx')
        if os.path.exists(path):
            return path
    return None


@pytest.fixture(scope='module')
def uci_shop():
    """The UCI-calibrated store the live diagnostics run on: a 60k-row
    sample of the last sheet, eight products per category, the naive
    layout, at the nominal load. Its fixtures sit on round-number
    coordinates, so aisle columns with no clearance to spare occur here
    as they do in the paper's runs."""
    path = _uci_workbook()
    if path is None:
        pytest.skip('UCI Online Retail II workbook not under DATASETS/')
    pytest.importorskip('openpyxl')
    df, _ = DA.read_excel_sheets(path, [DA.list_excel_sheets(path)[-1][0]])
    df = df.sample(n=60000, random_state=0).reset_index(drop=True)
    norm, _ = DA.OnlineRetailIIAdapter().adapt(df)
    params = calibrate_transactional(norm, currency='GBP')
    shop = build_headless_shop_from_calibration(params,
                                                max_items_per_category=8,
                                                naive=True)
    sim = shop.customer_simulation
    sim.spawn_rate, sim.max_customers = 0.17, 45
    sim.record_state_histories = True
    sim.step(0.04)
    return sim


def test_the_diagnostics_store_is_walkable_and_its_visits_complete(uci_shop):
    """Every product is shoppable from a free spot reachable from the door;
    over a seeded run no agent, in any state, is ever inside a fixture,
    and no agent gives up on a product because it could not get to it."""
    sim = uci_shop
    blocked = sim.path_blocked_grid_by_floor[1]
    margin = AGENT_RADIUS_M + sim.path_grid_resolution + 0.05
    items = sim.shop.floors[1]['items']
    inside = sim.nearest_free_position(sim.door_position[0],
                                       sim.door_position[1] + 0.4, 1)
    reach = _reachable_cells(blocked, _cell(sim, *inside))
    fixtures = []
    for nm, data in items.items():
        if 'impulse' in str(data.get('category', '')).lower():
            continue
        ax, ay = sim.item_access_point(nm, data, 1)
        cell = _cell(sim, ax, ay)
        assert not blocked[cell], f'{nm} stands inside a fixture'
        assert reach[cell], f'{nm} is walled off from the door'
        ix, iy = data['position']
        iw, ih = data['size']
        assert max(max(ix - ax, 0.0, ax - (ix + iw)),
                   max(iy - ay, 0.0, ay - (iy + ih))) <= margin + 1e-9
        fixtures.append((nm, ix, iy, iw, ih))

    seen = {'samples': 0, 'inside': [], 'abandoned': set()}

    def _sample(s):
        for cust in s.customers:
            seen['samples'] += 1
            x, y = cust.position
            for nm, ix, iy, iw, ih in fixtures:
                if ix < x < ix + iw and iy < y < iy + ih:
                    if len(seen['inside']) < 5:
                        seen['inside'].append((round(s.sim_time, 2), cust.id,
                                               cust.state, nm))
                    break
            seen['abandoned'].update((cust.id, nm) for nm in cust.abandoned_items)

    sim.run_headless(600.0, dt=0.04, seed=11,
                     callback=_sample, callback_every_s=0.04)
    recs = sim.completed_state_histories
    assert seen['samples'] > 100_000, 'too few samples to mean anything'
    assert not seen['inside']
    assert not seen['abandoned']
    assert len(recs) >= 20
    assert sum(1 for r in recs if r['states'][-1] == 'purchased') >= 0.9 * len(recs)
    assert not sim.suppressed_errors


# --- the driver rules -------------------------------------------------------

def test_a_long_frame_is_split_into_model_ticks():
    """The threaded driver's frame length follows host load and the speed
    slider; the model must always see ticks of at most MAX_STEP_S."""
    assert _substeps_for(MAX_STEP_S) == (1, MAX_STEP_S)
    assert _substeps_for(0.01) == (1, 0.01)
    n, sub = _substeps_for(0.2)
    assert (n, sub) == pytest.approx((5, 0.04))
    n, sub = _substeps_for(0.13)
    assert n == 4 and sub <= MAX_STEP_S and n * sub == pytest.approx(0.13)
    # Past the cap the excess simulated time is dropped, so a frame can
    # never ask for unbounded work.
    assert _substeps_for(5.0) == (MAX_SUBSTEPS, MAX_STEP_S)


def test_walking_speed_does_not_depend_on_the_frame_length(plan):
    """An agent follows its path one waypoint per tick, so the same walk
    over one long tick falls behind the sub-stepped one."""
    def _walk(ticks, dt):
        sim = _build_sim(plan, spawn_rate=0.0)
        sim.step(0.04)
        sim._spawn_customer()
        cust = sim.customers[-1]
        cust.speed = 1.2
        cust.position = list(sim.nearest_free_position(1.0, H - 1.0, 1))
        cust._change_state('moving')
        cust.current_target_type = 'exit'
        cust.target_position = list(sim.door_position)
        cust._initialize_path(sim.shop.walls)
        start = list(cust.position)
        walls, items = sim.shop.walls, sim.shop.all_items_across_floors()
        for _ in range(ticks):
            cust.update(dt, items, walls, sim.customers)
        return math.hypot(cust.position[0] - start[0],
                          cust.position[1] - start[1])

    fine = _walk(int(round(0.4 / MAX_STEP_S)) * 5, MAX_STEP_S)   # 2.0 s
    coarse = _walk(5, 0.4)                                       # 2.0 s
    assert fine > 1.5          # ~1.2 m/s over 2 s, less the path's corners
    assert coarse < 0.6 * fine


def test_a_pending_arrival_gap_follows_the_rate(plan):
    """A gap drawn in a quiet hour must not hold up a busy one: the profile
    is piecewise constant and exponential gaps are memoryless, so the gap
    is redrawn when the rate changes."""
    sim = _build_sim(plan, spawn_rate=0.05)
    sim.hourly_profile = np.ones(24)
    sim.hourly_profile[9] = 0.002        # an all-but-closed hour, then a busy one
    sim.sim_clock_start_hour = 9.0
    sim.arrival_rng = np.random.default_rng(17)
    arrivals = []
    sim._spawn_customer = lambda: arrivals.append(sim.sim_time)

    for _ in range(int(round(7200.0 / 0.04))):
        sim.sim_time += 0.04
        sim._process_arrivals(0.04)

    busy = [t for t in arrivals if t >= 3600.0]
    assert len(busy) == pytest.approx(180, rel=0.25)    # 0.05/s over an hour
    assert min(busy) - 3600.0 < 120.0                   # not waiting out the old gap


def test_a_stranded_agent_leaves_instead_of_asking_routing_again():
    """Removing the only stair while somebody is upstairs leaves no route
    to satisfy: not an item, not the checkout, not the door."""
    shop = HeadlessShop(10.0, 8.0)
    shop.door_position, shop.door_side = (5.0, 0.0), 'bottom'
    f1 = shop.floors[1]
    f1['door_position'], f1['door_side'] = shop.door_position, shop.door_side
    f1['items'] = {'Tea': {'position': (1.0, 6.0), 'size': (0.6, 0.6),
                           'category': 'Beverages', 'price': 2.0}}
    f1['walls'] = {'Checkout': {'position': (7.0, 1.0), 'size': (1.5, 0.6)}}
    shop.floors[2] = {'items': {}, 'walls': {}, 'prices': {},
                      'is_main_floor': False}
    shop.num_floors = 2
    sim = shop.customer_simulation
    sim.door_position, sim.door_side = shop.door_position, shop.door_side
    sim.step(0.04)

    sim._spawn_customer()
    cust = sim.customers[-1]
    cust.floor = 2                       # upstairs, with the stair now gone
    cust.position = [5.0, 5.0]
    cust._change_state('moving')
    cust._set_exit_target(shop.floors[2]['walls'])

    assert cust.floor == 1
    assert cust.state == 'exiting'
    assert cust.current_target_type == 'exit'
    assert cust.position == list(sim.door_position)

    sim.step(0.04)
    assert cust not in sim.customers     # despawned at the door
    assert not sim.suppressed_errors
