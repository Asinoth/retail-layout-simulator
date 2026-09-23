"""Per-invoice product sets, the in-store portion of an invoice, and the
shopping-list law it calibrates.

A dataset-built shop stocks only the top products of each category, so a
simulated shopper can only reproduce the part of a real invoice that the
shop carries. The calibration keeps each invoice's distinct product set
(CSR arrays), ``placed_invoice_sample`` cuts every invoice to a given
assortment, ``seed_into`` stores those cut invoices over the shop's item
keys, each spawning agent takes one of them as its shopping list, and the
basket-size / per-visit-revenue goodness-of-fit rows use the same sample as
their reference.
"""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sp_stats

import customer as customer_mod
from customer import Customer
from dataset_calibration import (calibrate_omnichannel,
                                 calibrate_transactional,
                                 placed_invoice_lists,
                                 placed_invoice_sample,
                                 regular_item_keys)
from dataset_validation import (observed_category_counts,
                                validate_against_simulation)
from retail_literature import LIST_LENGTH_BY_TYPE

PRICES = {'P1': 1.0, 'P2': 2.0, 'P3': 4.0, 'P4': 8.0}


def _invoices_df(invoices, prices):
    """One line per (invoice, product) entry, a minute apart per invoice,
    in the shape ``calibrate_transactional`` takes. A product listed twice
    on an invoice is a repeated line, as in Online Retail II."""
    rows = []
    t0 = pd.Timestamp('2010-12-01 10:00')
    for i, (inv, pids) in enumerate(invoices):
        for pid in pids:
            rows.append({'invoice_id': inv, 'product_id': pid,
                         'product_name': f'NAME {pid}', 'quantity': 3,
                         'timestamp': t0 + pd.Timedelta(minutes=i),
                         'unit_price': prices[pid], 'category': 'General'})
    return pd.DataFrame(rows)


# A: P1 twice (one product), P2.  B: P3.  C: P1, P2, P3.  D: P4, P2 (no
# stocked product under {P1, P3}).  E: P1 again -- P1 is on three invoices.
HAND_INVOICES = [('A', ['P1', 'P1', 'P2']),
                 ('B', ['P3']),
                 ('C', ['P1', 'P2', 'P3']),
                 ('D', ['P4', 'P2']),
                 ('E', ['P1'])]
HAND_SETS = {'A': {'P1', 'P2'}, 'B': {'P3'}, 'C': {'P1', 'P2', 'P3'},
             'D': {'P2', 'P4'}, 'E': {'P1'}}


@pytest.fixture
def hand_params():
    return calibrate_transactional(_invoices_df(HAND_INVOICES, PRICES))


def _stub_sim(analytics=None, shop=None):
    return SimpleNamespace(analytics={} if analytics is None else analytics,
                           run_time=0.0, sim_time=0.0, simulation_speed=1.0,
                           shop=shop)


def _shop(floor_items):
    """A shop stand-in: ``{floor: {item name: product_id}}``, or
    ``(product_id, category)`` for an item outside 'General'."""
    def item(spec):
        pid, cat = spec if isinstance(spec, tuple) else (spec, 'General')
        return {'product_id': pid, 'category': cat}
    return SimpleNamespace(floors={
        fid: {'items': {name: item(spec) for name, spec in items.items()}}
        for fid, items in floor_items.items()})


def _invoice_sets(params):
    idx, ptr, items = (params.invoice_product_index, params.invoice_ptr,
                       params.invoice_items)
    return [{idx[j] for j in items[ptr[i]:ptr[i + 1]]}
            for i in range(len(ptr) - 1)]


# --- CSR structure ------------------------------------------------------------

def test_csr_structure_matches_the_invoices(hand_params):
    p = hand_params
    assert p.invoice_ptr.dtype == np.int64
    assert p.invoice_items.dtype == np.int32
    assert len(p.invoice_ptr) == p.n_invoices + 1 == 5 + 1
    assert p.invoice_ptr[0] == 0 and p.invoice_ptr[-1] == len(p.invoice_items)
    # Same invoice order as the other per-invoice samples (A..E), and a
    # repeated line counted once.
    assert _invoice_sets(p) == [HAND_SETS[k] for k in 'ABCDE']
    assert np.diff(p.invoice_ptr).tolist() == p.basket_distinct_sizes.tolist()
    assert sorted(p.invoice_product_index) == ['P1', 'P2', 'P3', 'P4']
    assert p.has_invoice_structure


def test_aggregate_source_has_no_invoice_structure():
    fams = pd.DataFrame({
        'aisle_id': np.arange(1, 9),
        'product_family': [f'Family {i}' for i in range(8)],
        'zone_id': [i % 3 for i in range(8)],
        'avg_price': 3.0, 'purchase_pct_instore': 0.3,
        'daily_demand_instore': 10.0, 'dwell_s': 30.0, 'impulse_rate': 0.1,
    })
    params = calibrate_omnichannel(fams, None)
    assert not params.has_invoice_structure
    assert params.invoice_product_index == []
    assert params.invoice_ptr.size == 0 and params.invoice_items.size == 0
    out = placed_invoice_sample(params, [str(k) for k in params.item_prices])
    assert out['sizes'].size == 0 and out['revenues'].size == 0
    assert out['n_invoices'] == 0 and out['n_with_placed'] == 0


# --- placed_invoice_sample ----------------------------------------------------

def test_placed_sample_on_a_hand_computed_example(hand_params):
    # Stocked: P1, P3 (and an id the dataset never sold).
    out = placed_invoice_sample(hand_params, iter(['P1', 'P3', 'NOT_SOLD']))
    # A -> {P1}: 1 item, 1.0; B -> {P3}: 1, 4.0; C -> {P1, P3}: 2, 5.0;
    # D holds neither and is left out; E -> {P1}: 1, 1.0. One unit of each
    # stocked product, at the calibrated price -- not the 3 units a line
    # carries.
    assert out['sizes'].tolist() == [1, 1, 2, 1]
    assert np.allclose(out['revenues'], [1.0, 4.0, 5.0, 1.0])
    assert out['n_invoices'] == 5
    assert out['n_with_placed'] == 4
    assert out['sizes'].dtype.kind == 'i'


def test_placed_sample_with_the_whole_catalogue_is_the_distinct_sample(
        hand_params):
    out = placed_invoice_sample(hand_params, PRICES.keys())
    assert out['sizes'].tolist() == hand_params.basket_distinct_sizes.tolist()
    assert np.allclose(out['revenues'], hand_params.invoice_distinct_revenues)


def test_csr_order_survives_shuffled_rows():
    """Row order in the source does not matter: invoice i of the CSR arrays
    is invoice i of the other per-invoice samples."""
    invoices, prices = _larger_example()
    df = _invoices_df(invoices, prices).sample(frac=1.0, random_state=3)
    params = calibrate_transactional(df)
    by_id = dict(invoices)
    assert _invoice_sets(params) == [set(by_id[k]) for k in sorted(by_id)]
    out = placed_invoice_sample(params, prices.keys())
    assert out['sizes'].tolist() == params.basket_distinct_sizes.tolist()
    assert np.allclose(out['revenues'], params.invoice_distinct_revenues)


def test_placed_sample_prices_non_string_product_ids():
    """A source whose product ids are not strings still prices the placed
    products: the ids are matched, and priced, by their string form."""
    df = _invoices_df(HAND_INVOICES, PRICES)
    df['product_id'] = df['product_id'].str[1:].astype(int)
    params = calibrate_transactional(df)
    out = placed_invoice_sample(params, ['1', '3'])
    assert out['sizes'].tolist() == [1, 1, 2, 1]
    assert np.allclose(out['revenues'], [1.0, 4.0, 5.0, 1.0])


def test_placed_sample_with_nothing_stocked_is_empty(hand_params):
    for ids in ([], None, ['NOT_SOLD']):
        out = placed_invoice_sample(hand_params, ids)
        assert out['sizes'].size == 0 and out['revenues'].size == 0
        assert out['n_invoices'] == 5 and out['n_with_placed'] == 0


# --- seed_into ----------------------------------------------------------------

LIST_KEYS = ('list_invoice_keys', 'list_invoice_ptr', 'list_invoice_items',
             'list_length_sample', 'list_length_source',
             'n_invoices_with_placed')


def _stored_invoices(cal):
    """The stored invoices, decoded to lists of item keys."""
    keys, ptr, items = (cal['list_invoice_keys'], cal['list_invoice_ptr'],
                        cal['list_invoice_items'])
    return [[keys[j] for j in items[ptr[i]:ptr[i + 1]]]
            for i in range(len(ptr) - 1)]


def test_seed_into_writes_the_invoices_only_with_a_shop(hand_params):
    sim = _stub_sim()
    hand_params.seed_into(sim)
    for key in LIST_KEYS:
        assert key not in sim.analytics['calibration']

    # P3 is stocked on the upper floor only; it still counts.
    shop = _shop({1: {'Mug': 'P1', 'Sign': None}, 2: {'Lamp': 'P3'}})
    hand_params.seed_into(sim, shop=shop)
    cal = sim.analytics['calibration']
    # A -> [Mug], B -> [Lamp], C -> [Mug, Lamp], D holds neither and is
    # left out, E -> [Mug]; stored order is the calibration's own.
    assert cal['list_invoice_keys'] == ['Mug', 'Lamp']
    assert cal['list_invoice_ptr'].tolist() == [0, 1, 2, 4, 5]
    assert cal['list_invoice_items'].tolist() == [0, 1, 0, 1, 0]
    assert cal['list_invoice_ptr'].dtype == np.int64
    assert cal['list_invoice_items'].dtype == np.int32
    assert _stored_invoices(cal) == [['Mug'], ['Lamp'], ['Mug', 'Lamp'],
                                     ['Mug']]
    assert cal['list_length_sample'] == [1, 1, 2, 1]
    assert all(type(v) is int for v in cal['list_length_sample'])
    assert cal['list_length_source'] == 'placed-invoice empirical'
    assert cal['n_invoices_with_placed'] == 4
    for key in LIST_KEYS:
        assert key in sim._calibration_seeded_keys


def test_reseeding_pops_the_stored_invoices(hand_params):
    sim = _stub_sim()
    shop = _shop({1: {'Mug': 'P1', 'Lamp': 'P3'}})
    hand_params.seed_into(sim, shop=shop)
    assert sim.analytics['calibration']['list_length_sample']
    assert len(sim.analytics['calibration']['list_invoice_keys'])

    hand_params.seed_into(sim)                     # no shop: no invoices
    cal = sim.analytics['calibration']
    for key in LIST_KEYS:
        assert key not in cal

    # A source without invoices, onto a shop: nothing to cut, nothing left.
    hand_params.seed_into(sim, shop=shop)
    no_invoices = replace(hand_params, invoice_product_index=[],
                          invoice_ptr=np.zeros(0, dtype=np.int64),
                          invoice_items=np.zeros(0, dtype=np.int32))
    no_invoices.seed_into(sim, shop=shop)
    for key in LIST_KEYS:
        assert key not in sim.analytics['calibration']

    # A shop none of whose regular items the dataset sells: likewise.
    hand_params.seed_into(sim, shop=shop)
    hand_params.seed_into(sim, shop=_shop({1: {'Mug': 'NOT_SOLD'}}))
    for key in LIST_KEYS:
        assert key not in sim.analytics['calibration']


def test_seed_into_rekeys_as_before_with_the_shop(hand_params):
    """Collecting product ids over every floor leaves the floor-1 re-keying
    of the per-item dicts as it was."""
    sim = _stub_sim()
    shop = _shop({1: {'Mug': 'P1', 'Lamp': 'P3'}, 2: {'Vase': 'P2'}})
    hand_params.seed_into(sim, shop=shop)
    cal = sim.analytics['calibration']
    assert set(cal['popular_items']) == {'Mug', 'Lamp'}
    assert cal['popular_items']['Mug'] == 3        # P1 is on A, C, E


def test_stored_invoices_hold_only_regular_items(hand_params):
    """Impulse displays, the checkout and the WC are never on a list, so
    their products are not part of any stored invoice, even when they carry
    a product id; ``list_length_sample`` is the stored invoices' sizes and
    matches ``placed_invoice_sample`` over the regular products."""
    sim = _stub_sim()
    shop = _shop({1: {'Mug': 'P1', 'Gum': ('P2', 'Impulse'),
                      'Checkout': 'P4', 'WC': 'P4'},
                  2: {'Lamp': 'P3'}})
    assert regular_item_keys(shop) == {'P1': 'Mug', 'P3': 'Lamp'}
    hand_params.seed_into(sim, shop=shop)
    cal = sim.analytics['calibration']
    assert set(cal['list_invoice_keys']) == {'Mug', 'Lamp'}
    assert _stored_invoices(cal) == [['Mug'], ['Lamp'], ['Mug', 'Lamp'],
                                     ['Mug']]
    sizes = np.diff(cal['list_invoice_ptr']).tolist()
    assert cal['list_length_sample'] == sizes
    assert sizes == placed_invoice_sample(hand_params,
                                          ['P1', 'P3'])['sizes'].tolist()
    assert cal['n_invoices_with_placed'] == len(sizes)


def test_invoice_keys_are_the_keys_agents_receive(hand_params):
    """Agents are handed ``shop.all_items_across_floors()``, so the stored
    invoices use that view's keys -- including the 'F<floor>:' spelling a
    name repeated on a higher floor gets there."""
    class _FlatShop:
        floors = {1: {'items': {'Mug': {'product_id': 'P1'}}},
                  2: {'items': {'Mug': {'product_id': 'P3'}}}}

        def all_items_across_floors(self):
            return {'Mug': dict(self.floors[1]['items']['Mug'], floor=1),
                    'F2:Mug': dict(self.floors[2]['items']['Mug'], floor=2)}

    sim = _stub_sim()
    hand_params.seed_into(sim, shop=_FlatShop())
    cal = sim.analytics['calibration']
    assert _stored_invoices(cal) == [['Mug'], ['F2:Mug'], ['Mug', 'F2:Mug'],
                                     ['Mug']]


def test_placed_invoice_lists_match_the_placed_sample():
    """Same invoices, same order, same sizes as ``placed_invoice_sample``,
    whatever the source row order."""
    invoices, prices = _larger_example()
    df = _invoices_df(invoices, prices).sample(frac=1.0, random_state=5)
    params = calibrate_transactional(df)
    stocked = {'P0': 'Item P0', 'P3': 'Item P3', 'P7': 'Item P7'}
    lists = placed_invoice_lists(params, stocked)
    sample = placed_invoice_sample(params, stocked)
    assert np.diff(lists['ptr']).tolist() == sample['sizes'].tolist()
    assert lists['n_with_placed'] == sample['n_with_placed']
    assert lists['n_invoices'] == sample['n_invoices'] == len(invoices)
    by_id = dict(invoices)
    expected = [{stocked[p] for p in by_id[k] if p in stocked}
                for k in sorted(by_id)]
    expected = [s for s in expected if s]
    got = [{lists['keys'][j] for j in lists['items'][a:b]}
           for a, b in zip(lists['ptr'][:-1], lists['ptr'][1:])]
    assert got == expected
    empty = placed_invoice_lists(params, {})
    assert empty['n_with_placed'] == 0 and empty['keys'] == []


# --- the agent's shopping list ------------------------------------------------

def _regular_items(n):
    items = {f'Item{i}': {'category': 'General', 'price': 1.0 + i}
             for i in range(n)}
    items['Gum'] = {'category': 'Impulse', 'price': 0.5}
    items['Checkout'] = {'category': 'Checkout', 'price': 0.0}
    return items


def _hand_store_items():
    """Agent-side items for ``_shop({1: {'Mug': 'P1', 'Lamp': 'P3'}})``,
    plus an item the dataset never sold and the non-list items."""
    return {'Mug': {'category': 'General', 'price': 1.0, 'product_id': 'P1'},
            'Lamp': {'category': 'General', 'price': 4.0, 'product_id': 'P3'},
            'Poster': {'category': 'General', 'price': 9.0},
            'Gum': {'category': 'Impulse', 'price': 0.5},
            'Checkout': {'category': 'Checkout', 'price': 0.0}}


def _spawn(sim, items, k):
    return Customer(k, [1.0, 1.0], items, (10.0, 10.0),
                    door_position=(5.0, 0.0), door_side='bottom',
                    simulation_ref=sim)


def _seeded_hand_store(hand_params):
    sim = _stub_sim()
    hand_params.seed_into(sim, shop=_shop({1: {'Mug': 'P1', 'Lamp': 'P3'}}))
    return sim


def test_calibrated_list_is_one_stored_invoice(hand_params, monkeypatch):
    """The list is the invoice the one ``randint`` picks, in stored order."""
    sim = _seeded_hand_store(hand_params)
    stored = _stored_invoices(sim.analytics['calibration'])
    items = _hand_store_items()
    a = _spawn(sim, items, 0)
    for i, invoice in enumerate(stored):
        with monkeypatch.context() as m:
            m.setattr(customer_mod.np.random, 'randint',
                      lambda n, *rest, _i=i: _i)
            a._generate_shopping_list(items)
        assert a.shopping_list == invoice
        assert a.basket_value == sum(items[k]['price'] for k in invoice)


def test_calibrated_lists_cover_the_stored_invoices_for_every_type(
        hand_params):
    np.random.seed(7)
    sim = _seeded_hand_store(hand_params)
    stored = _stored_invoices(sim.analytics['calibration'])
    items = _hand_store_items()
    agents = [_spawn(sim, items, k) for k in range(80)]
    assert {a.customer_type for a in agents} == set(LIST_LENGTH_BY_TYPE)
    assert all(a.shopping_list in stored for a in agents)
    # Every type draws from the same invoices -- 'thorough' included, whose
    # type law would ask for 6-12 items.
    for t in LIST_LENGTH_BY_TYPE:
        assert {len(a.shopping_list) for a in agents
                if a.customer_type == t} <= {1, 2}
    assert {tuple(a.shopping_list) for a in agents} == {
        tuple(s) for s in stored}
    # Items the dataset never sold, and impulse / checkout items, are never
    # on a calibrated list.
    assert all(i in ('Mug', 'Lamp') for a in agents for i in a.shopping_list)


def test_items_off_the_floor_are_dropped_from_the_invoice(hand_params,
                                                          monkeypatch):
    """An invoice item no longer among the agent's regular items (deleted,
    renamed, on an unreachable floor) is left out; the rest of the invoice
    stays."""
    sim = _seeded_hand_store(hand_params)
    items = _hand_store_items()
    del items['Lamp']
    a = _spawn(sim, items, 0)
    with monkeypatch.context() as m:
        m.setattr(customer_mod.np.random, 'randint',
                  lambda n, *rest: 2)                    # invoice C
        a._generate_shopping_list(items)
    assert a.shopping_list == ['Mug']


_COUNTED = ('randint', 'choice', 'random', 'uniform', 'normal', 'lognormal',
            'exponential', 'shuffle', 'permutation')


def _calls_for(monkeypatch, sim, items, seed=11):
    """The global-stream calls one ``_generate_shopping_list`` makes, in
    order, for a fresh agent on ``sim``."""
    np.random.seed(seed)
    a = _spawn(sim, items, 0)
    calls = []
    with monkeypatch.context() as m:
        for name in _COUNTED:
            real = getattr(np.random, name)

            def counting(*args, _name=name, _real=real, **kwargs):
                calls.append(_name)
                return _real(*args, **kwargs)
            m.setattr(customer_mod.np.random, name, counting)
        a._generate_shopping_list(items)
    return calls, a


def test_one_global_call_picks_the_list_in_either_branch(hand_params,
                                                         monkeypatch):
    """The invoice draw is one ``randint`` in the place the type law's
    length ``randint`` has, and a calibrated list needs no other draw: the
    spawn's calls up to the list are the same with or without a
    calibration."""
    calibrated, a = _calls_for(monkeypatch, _seeded_hand_store(hand_params),
                               _hand_store_items())
    assert a.shopping_list
    uncalibrated, b = _calls_for(monkeypatch, _stub_sim(), _regular_items(8))
    assert b.shopping_list
    assert calibrated == ['random', 'randint']
    assert uncalibrated == ['random', 'randint', 'choice']
    assert calibrated.count('randint') == uncalibrated.count('randint') == 1


def test_empty_intersection_falls_back_to_the_type_law(hand_params,
                                                       monkeypatch):
    """GUI-edit edge case: every item of the drawn invoice is gone. The
    agent falls back to the uncalibrated law; the spawn then makes the
    invoice ``randint`` it already made plus the type law's ``randint`` and
    ``choice``, and draws no second invoice."""
    sim = _seeded_hand_store(hand_params)
    items = _regular_items(20)         # neither Mug nor Lamp is on the floor
    calls, a = _calls_for(monkeypatch, sim, items)
    assert calls == ['random', 'randint', 'randint', 'choice']
    lo, hi = LIST_LENGTH_BY_TYPE[a.customer_type]
    assert lo <= len(a.shopping_list) <= hi
    assert len(set(a.shopping_list)) == len(a.shopping_list)
    assert all(i.startswith('Item') for i in a.shopping_list)


def test_without_stored_invoices_the_type_law_applies():
    np.random.seed(10)
    items = _regular_items(20)
    for analytics in ({}, {'calibration': {'popular_items': {'Item1': 50}}},
                      # A size sample alone does not make a list.
                      {'calibration': {'list_length_sample': [3] * 40}},
                      {'calibration': {'list_invoice_keys': [],
                                       'list_invoice_ptr': [0],
                                       'list_invoice_items': []}},
                      # Live counters are never read for the list.
                      {'list_invoice_keys': ['Item1'],
                       'list_invoice_ptr': [0, 1],
                       'list_invoice_items': [0]}):
        sim = _stub_sim(dict(analytics))
        for k in range(60):
            a = _spawn(sim, items, k)
            lo, hi = LIST_LENGTH_BY_TYPE[a.customer_type]
            assert lo <= len(a.shopping_list) <= hi
    # And it is clipped to the assortment as before.
    small = _regular_items(3)
    sim = _stub_sim()
    assert all(len(_spawn(sim, small, k).shopping_list) <= 3
               for k in range(30))


def _category_example():
    """Three categories whose products co-occur within an invoice far more
    than across them, the way a trip clusters in a few departments."""
    rng = np.random.default_rng(21)
    cats = {'Kitchen': ['K0', 'K1', 'K2'], 'Garden': ['G0', 'G1', 'G2'],
            'Toys': ['T0', 'T1', 'T2']}
    cat_of = {p: c for c, ps in cats.items() for p in ps}
    rows = []
    t0 = pd.Timestamp('2010-12-01 09:00')
    for i in range(400):
        home = ['Kitchen', 'Garden', 'Toys'][rng.choice(3, p=[0.6, 0.3, 0.1])]
        n = int(rng.integers(1, 5))
        pids = set(rng.choice(cats[home], size=min(n, 3), replace=False))
        if rng.random() < 0.3:                      # an occasional side trip
            pids.add(str(rng.choice(sorted(cat_of))))
        for pid in sorted(pids):
            rows.append({'invoice_id': f'INV{i:04d}', 'product_id': pid,
                         'product_name': f'NAME {pid}', 'quantity': 1,
                         'timestamp': t0 + pd.Timedelta(minutes=i),
                         'unit_price': 2.0, 'category': cat_of[pid]})
    return pd.DataFrame(rows), cat_of


def test_calibrated_lists_reproduce_the_reference_category_shares():
    """Drawing whole invoices reproduces the reference's category shares in
    expectation: the stored invoices hold exactly the product-invoice
    touches the category test counts, and the drawn lists match them up to
    sampling noise (judged per category against the invoice-level spread,
    since items cluster within a list). The within-invoice co-purchase rate
    of a pair comes along with no constant."""
    df, cat_of = _category_example()
    params = calibrate_transactional(df)
    stocked = ['K0', 'K1', 'G0', 'G1', 'T0', 'T1']      # K2, G2, T2 unplaced
    shop = _shop({1: {f'Item {p}': (p, cat_of[p]) for p in stocked}})
    sim = _stub_sim()
    params.seed_into(sim, shop=shop)
    stored = _stored_invoices(sim.analytics['calibration'])
    cats = sorted(set(cat_of.values()))

    def cat_counts(lists):
        return np.array([[sum(cat_of[k.split()[1]] == c for k in lst)
                          for c in cats] for lst in lists], dtype=float)

    # The stored invoices carry the reference's touches exactly.
    ref = observed_category_counts(params, stocked)
    stored_counts = cat_counts(stored)
    assert np.allclose(stored_counts.sum(axis=0), [ref[c] for c in cats])

    items = {f'Item {p}': {'category': cat_of[p], 'price': 2.0,
                           'product_id': p} for p in stocked}
    items['Gum'] = {'category': 'Impulse', 'price': 0.5}
    np.random.seed(2024)
    n_draws = 4000
    drawn = [_spawn(sim, items, k).shopping_list for k in range(n_draws)]
    drawn_counts = cat_counts(drawn)

    se = stored_counts.std(axis=0) / np.sqrt(n_draws)
    gap = drawn_counts.mean(axis=0) - stored_counts.mean(axis=0)
    assert np.all(np.abs(gap) < 4.0 * se), (gap, se)
    ref_share = stored_counts.sum(axis=0) / stored_counts.sum()
    drawn_share = drawn_counts.sum(axis=0) / drawn_counts.sum()
    assert np.max(np.abs(drawn_share - ref_share)) < 0.03

    pair = ('Item K0', 'Item K1')

    def both(lists):
        return np.array([pair[0] in lst and pair[1] in lst for lst in lists],
                        dtype=float)
    p_ref, p_drawn = both(stored).mean(), both(drawn).mean()
    assert p_ref > 0
    assert abs(p_drawn - p_ref) < 4.0 * np.sqrt(p_ref * (1 - p_ref) / n_draws)


# --- goodness-of-fit reference ------------------------------------------------

def _larger_example():
    prices = {f'P{k}': float(k + 1) for k in range(10)}
    invoices = []
    for i in range(80):
        pids = sorted({f'P{i % 10}', f'P{(3 * i) % 10}', f'P{(7 * i + 1) % 10}'})
        invoices.append((f'INV{i:03d}', pids))
    return invoices, prices


def test_validation_compares_against_the_placed_reference():
    invoices, prices = _larger_example()
    params = calibrate_transactional(_invoices_df(invoices, prices))
    stocked = {'P0', 'P1', 'P2', 'P3'}
    shop = _shop({1: {f'Item {p}': p for p in sorted(stocked)}})

    # Reference computed directly from the invoice lists.
    ref_sizes, ref_revs = [], []
    for _, pids in invoices:
        on_shelf = set(pids) & stocked
        if on_shelf:
            ref_sizes.append(len(on_shelf))
            ref_revs.append(sum(prices[p] for p in on_shelf))
    assert len(ref_sizes) < len(invoices)       # some invoices drop out

    sim_baskets = [1, 2, 2, 1, 3, 1, 2, 1, 1, 2, 1, 1]
    sim_revs = [1.0, 3.0, 5.0, 2.0, 6.0, 4.0, 3.0, 1.0, 2.0, 7.0, 3.0, 4.0]
    sim = _stub_sim({'basket_sizes': list(sim_baskets),
                     'customer_revenues': list(sim_revs)}, shop=shop)
    res = validate_against_simulation(params, sim)
    rows = {t.name: t for t in res.tests}

    basket = rows['Basket size (items/visit)']
    D, p = sp_stats.ks_2samp(ref_sizes, sim_baskets)
    assert basket.n_observed == len(ref_sizes)
    assert np.isclose(basket.statistic, D) and np.isclose(basket.p_value, p)
    assert 'distinct stocked products per invoice' in basket.note
    # Not the whole-catalogue sample.
    D_full, _ = sp_stats.ks_2samp(params.basket_distinct_sizes, sim_baskets)
    assert not np.isclose(basket.statistic, D_full)

    revenue = rows['Per-visit revenue']
    D, p = sp_stats.ks_2samp(ref_revs, sim_revs)
    assert revenue.n_observed == len(ref_revs)
    assert np.isclose(revenue.statistic, D) and np.isclose(revenue.p_value, p)
    assert 'one unit of each stocked product' in revenue.note


def test_validation_without_invoice_structure_reports_not_run():
    invoices, prices = _larger_example()
    params = calibrate_transactional(_invoices_df(invoices, prices))
    params = replace(params, invoice_product_index=[],
                     invoice_ptr=np.zeros(0, dtype=np.int64),
                     invoice_items=np.zeros(0, dtype=np.int32))
    shop = _shop({1: {'Item P0': 'P0', 'Item P1': 'P1'}})
    sim = _stub_sim({'basket_sizes': [1, 2, 1, 2, 1, 1],
                     'customer_revenues': [1.0, 3.0, 1.0, 2.0, 1.0, 2.0]},
                    shop=shop)
    res = validate_against_simulation(params, sim)
    rows = {t.name: t for t in res.tests}
    for name in ('Basket size (items/visit)', 'Per-visit revenue'):
        assert not rows[name].applicable()
        assert rows[name].verdict() == 'N/A'
        assert 'per-invoice product sets' in rows[name].note
