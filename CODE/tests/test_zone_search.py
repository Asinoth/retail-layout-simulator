"""Zone-bounded search moves on a calibrated store, and portable run paths.

Random search and simulated annealing draw layouts from a SyntheticShop on
the synthetic scenarios. The calibrated store has none, so on it they take
``zone_sampler`` / ``zone_neighbor`` from ``experiments._common``: draws
and one-item moves inside the zone the GA constrains each item to. These
tests pin that, on a small store laid out from a synthetic invoice frame
the way a loaded dataset is: every sampled item lies inside its zone, a
move changes exactly one item and keeps it in its zone, both are fixed by
the generator handed to them and neither touches the global numpy RNG
the Monte Carlo evaluations are seeded through.

The last tests pin how run records name paths: inside the repository they
are written relative to its root with forward slashes, outside it they
stay absolute, and the provenance git fields are left alone.
"""
import json
import os

import numpy as np
import pandas as pd
import pytest

from dataset_calibration import calibrate_transactional
from dataset_paths import REPO_ROOT
from experiments._common import (_zone_position_bounds,
                                 build_headless_shop_from_calibration,
                                 feasible_layout, portable_paths,
                                 write_sidecar, zone_neighbor, zone_sampler)
from viz_ga_run import item_zone_name

EPS = 1e-9
CODE_DIR = os.path.join(REPO_ROOT, 'CODE')


def _invoices(n_invoices=400, seed=0):
    """Invoices over four categories: enough for the layout builder to
    split two departments over two gondolas, so stamped zones
    (``Section_<cat>_<i>``) occur as they do in the UCI store."""
    rng = np.random.default_rng(seed)
    products = [(f'{cat[:3].upper()}{i}', cat, 1.5 + i)
                for cat in ('Bakery', 'Dairy', 'Garden', 'Kitchen')
                for i in range(4)]
    day0 = pd.Timestamp('2010-12-01 09:00')
    rows = []
    for k in range(n_invoices):
        ts = day0 + pd.Timedelta(days=k // 40,
                                 minutes=int(rng.integers(0, 480)))
        for j in rng.choice(len(products), size=int(rng.integers(1, 5)),
                            replace=False):
            pid, cat, price = products[j]
            rows.append((f'I{k:05d}', pid, f'ITEM {pid}',
                         int(rng.integers(1, 4)), ts, price, cat))
    return pd.DataFrame(rows, columns=['invoice_id', 'product_id',
                                       'product_name', 'quantity',
                                       'timestamp', 'unit_price', 'category'])


def _build_store():
    params = calibrate_transactional(_invoices(), currency='GBP')
    shop = build_headless_shop_from_calibration(
        params, max_items_per_category=4, naive=True)
    return shop, list(shop.floors[1]['items'].keys())


@pytest.fixture(scope='module')
def store():
    return _build_store()


def _zone_rect(shop, name):
    f1 = shop.floors[1]
    zone = item_zone_name(f1['items'][name], f1['walls'])
    assert zone is not None, f"{name} resolves to no zone"
    (sx, sy), (sw, sh) = f1['walls'][zone]['position'], f1['walls'][zone]['size']
    return zone, sx, sy, sw, sh


def _assert_in_zone(shop, name, pos):
    zone, sx, sy, sw, sh = _zone_rect(shop, name)
    w, h = shop.floors[1]['items'][name]['size']
    x, y = pos
    assert sx - EPS <= x and x + w <= sx + sw + EPS, (name, zone, pos)
    assert sy - EPS <= y and y + h <= sy + sh + EPS, (name, zone, pos)


def _zone_centred(shop, names):
    """Every item centred in the room its zone leaves it once the aisles to
    the zones the as-built store keeps apart are taken out
    (``_zone_position_bounds``): a neighbour step cannot be clipped back
    onto the starting position from here. The zone's own centre can lie in
    such an aisle, where every move would first clip the item out of it."""
    lo, hi = _zone_position_bounds(shop, names)
    return {n: (float((lo[i, 0] + hi[i, 0]) / 2.0),
                float((lo[i, 1] + hi[i, 1]) / 2.0))
            for i, n in enumerate(names)}


def test_store_uses_split_zones(store):
    """The fixture exercises the stamped-zone path, not only
    ``Section_<category>``."""
    shop, names = store
    zones = {_zone_rect(shop, n)[0] for n in names}
    assert any(z.rsplit('_', 1)[-1].isdigit() for z in zones), zones


def test_sampler_puts_every_item_inside_its_zone(store):
    shop, names = store
    sample = zone_sampler(shop, names)
    draws = [sample(np.random.default_rng(s)) for s in range(40)]
    for lay in draws:
        assert list(lay) == names
        for n in names:
            _assert_in_zone(shop, n, lay[n])
        # The repair the searches apply keeps them there.
        repaired = feasible_layout(shop, names, lay)
        for n in names:
            _assert_in_zone(shop, n, repaired[n])
    # The draws actually spread over the zones rather than repeat a point.
    for n in names:
        xs = {round(lay[n][0], 9) for lay in draws}
        assert len(xs) > 1, n


def test_neighbor_moves_exactly_one_item_and_keeps_it_in_its_zone(store):
    shop, names = store
    step_frac = 0.25
    move = zone_neighbor(shop, names, step_frac=step_frac)
    start = _zone_centred(shop, names)
    lo, hi = _zone_position_bounds(shop, names)
    moved_items = set()
    for s in range(60):
        after = move(start, np.random.default_rng(s))
        assert list(after) == names
        changed = [n for n in names if after[n] != start[n]]
        assert len(changed) == 1, changed
        (n,) = changed
        moved_items.add(n)
        _assert_in_zone(shop, n, after[n])
        i = names.index(n)
        # The step is scaled by the room the item has, never more than its
        # zone's own.
        _, sx, sy, sw, sh = _zone_rect(shop, n)
        w, h = shop.floors[1]['items'][n]['size']
        room_x = max(hi[i, 0] - lo[i, 0], 1e-3)
        room_y = max(hi[i, 1] - lo[i, 1], 1e-3)
        assert room_x <= (sw - w) + EPS and room_y <= (sh - h) + EPS
        assert abs(after[n][0] - start[n][0]) <= step_frac * room_x + EPS
        assert abs(after[n][1] - start[n][1]) <= step_frac * room_y + EPS
    # The item is chosen uniformly, so 60 moves reach most of 16 items.
    assert len(moved_items) >= len(names) // 2


def test_k_item_neighbor_moves_that_many_items(store):
    """A move of k items changes exactly k distinct items, each inside its
    zone; with k = 1 it is the one-item move, draw for draw."""
    shop, names = store
    one = zone_neighbor(shop, names)
    one_again = zone_neighbor(shop, names, n_move=1)
    start = _zone_centred(shop, names)
    for s in range(20):
        assert (one(start, np.random.default_rng(s))
                == one_again(start, np.random.default_rng(s)))
    k = 3
    move = zone_neighbor(shop, names, n_move=k)
    for s in range(30):
        after = move(start, np.random.default_rng(s))
        changed = [n for n in names if after[n] != start[n]]
        assert len(changed) == k, changed
        for n in changed:
            _assert_in_zone(shop, n, after[n])
    with pytest.raises(ValueError):
        zone_neighbor(shop, names, n_move=0)
    with pytest.raises(ValueError):
        zone_neighbor(shop, names, n_move=len(names) + 1)


def test_neighbor_chain_stays_inside_zones_from_the_as_built_layout(store):
    """A walk the way the annealer takes it: from the repaired as-built
    layout, where fixtures sit flush on zone edges and steps get clipped,
    each move changes at most one item and nothing ever leaves its zone."""
    shop, names = store
    move = zone_neighbor(shop, names)
    as_built = {n: tuple(shop.floors[1]['items'][n]['position'])
                for n in names}
    cur = feasible_layout(shop, names, as_built)
    rng = np.random.default_rng(11)
    for _ in range(200):
        nxt = move(cur, rng)
        assert sum(nxt[n] != cur[n] for n in names) <= 1
        for n in names:
            _assert_in_zone(shop, n, nxt[n])
        cur = nxt


def test_sampler_and_neighbor_are_fixed_by_the_generator(store):
    shop, names = store
    sample = zone_sampler(shop, names)
    move = zone_neighbor(shop, names)
    start = _zone_centred(shop, names)

    np.random.seed(123)
    global_before = np.random.get_state()
    a1, a2 = sample(np.random.default_rng(5)), sample(np.random.default_rng(5))
    b1 = move(start, np.random.default_rng(5))
    b2 = move(start, np.random.default_rng(5))
    global_after = np.random.get_state()

    assert a1 == a2
    assert b1 == b2
    assert sample(np.random.default_rng(6)) != a1
    # Neither draws from the global stream the MC evaluations are seeded on.
    assert global_before[0] == global_after[0]
    assert np.array_equal(global_before[1], global_after[1])
    assert global_before[2:] == global_after[2:]

    # One generator threaded through several calls reproduces as a sequence.
    r1, r2 = np.random.default_rng(9), np.random.default_rng(9)
    assert ([sample(r1) for _ in range(3)] == [sample(r2) for _ in range(3)])


def test_zoneless_item_falls_back_to_the_floor_box_the_ga_uses():
    """An item that resolves to no zone is searched over the box
    ``_ga_repair`` clips it to, so the search space stays the GA's."""
    shop, names = _build_store()
    lone = names[0]
    shop.floors[1]['items'][lone]['category'] = 'NoSuchDepartment'
    shop.floors[1]['items'][lone]['zone'] = None
    assert shop._ga_get_section_bounds(lone) is None
    w, h = shop.floors[1]['items'][lone]['size']
    sample = zone_sampler(shop, names)
    for s in range(20):
        x, y = sample(np.random.default_rng(s))[lone]
        assert 0.1 - EPS <= x <= shop.width - w - 0.1 + EPS
        assert 0.1 - EPS <= y <= shop.height - h - 0.1 + EPS


def test_hooks_reject_degenerate_arguments(store):
    shop, names = store
    with pytest.raises(ValueError):
        zone_sampler(shop, [])
    with pytest.raises(ValueError):
        zone_neighbor(shop, names, step_frac=0.0)


# --- portable run paths ------------------------------------------------------

def _payload(outside):
    return {
        'args': {
            'out_root': os.path.join(CODE_DIR, 'experiments', 'results'),
            # Typed relative to the working directory the run started in.
            'retail_path': os.path.join('..', 'DATASETS', 'workbook.xlsx'),
            'sheets': 'Year 2009-2010,Year 2010-2011',
            'figs_dir': outside,
        },
        'figs_dir': os.path.join(REPO_ROOT, 'figs'),
        # A folder handed on outside the argparse namespace, as typed.
        'plots_dir': os.path.join('..', 'figs'),
        # Recorded relative to the run directory by the runner itself.
        'csv_path': 'results.csv',
        'provenance': {'source_path': os.path.join('..', 'DATASETS',
                                                   'workbook.xlsx'),
                       'extra': {'outputs': [os.path.join(REPO_ROOT, 'figs',
                                                          'a.png'),
                                             outside]}},
    }


def test_portable_paths_relativizes_inside_and_keeps_outside(tmp_path,
                                                             monkeypatch):
    monkeypatch.chdir(CODE_DIR)
    outside = str(tmp_path)
    # Outside the checkout. os.path.relpath raises on Windows when the
    # temporary directory is on another drive, which is outside too.
    if (os.path.splitdrive(outside)[0].lower()
            == os.path.splitdrive(REPO_ROOT)[0].lower()):
        assert os.path.relpath(outside, REPO_ROOT).startswith(os.pardir)
    payload = _payload(outside)
    before = json.dumps(payload, sort_keys=True)

    out = portable_paths(payload)

    assert out['args']['out_root'] == 'CODE/experiments/results'
    assert out['args']['retail_path'] == 'DATASETS/workbook.xlsx'
    assert out['args']['sheets'] == 'Year 2009-2010,Year 2010-2011'
    assert out['args']['figs_dir'] == outside
    assert out['figs_dir'] == 'figs'
    assert out['plots_dir'] == 'figs'
    assert out['csv_path'] == 'results.csv'
    assert out['provenance']['source_path'] == 'DATASETS/workbook.xlsx'
    assert out['provenance']['extra']['outputs'] == ['figs/a.png', outside]
    # The caller's dict (often ``vars(args)``) is left as it was.
    assert json.dumps(payload, sort_keys=True) == before


def test_write_sidecar_records_portable_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(CODE_DIR)
    outside = str(tmp_path / 'elsewhere')
    in_repo = os.path.join(CODE_DIR, 'experiments', 'results')
    prov = {'git_sha': 'deadbeef', 'git_dirty': False,
            'git_diff_sha256': None, 'git_modified': [in_repo]}
    path = write_sidecar(str(tmp_path / 'run'), _payload(outside),
                         provenance=prov)
    with open(path, encoding='utf-8') as f:
        side = json.load(f)
    assert side['args']['out_root'] == 'CODE/experiments/results'
    assert side['args']['retail_path'] == 'DATASETS/workbook.xlsx'
    assert side['args']['figs_dir'] == outside
    assert side['figs_dir'] == 'figs'
    assert side['provenance']['source_path'] == 'DATASETS/workbook.xlsx'
    # provenance_snapshot's git fields are written as they were handed in.
    assert side['git_modified'] == [in_repo]
    assert side['git_sha'] == 'deadbeef'
    # No in-repo absolute path survives anywhere in the payload part.
    text = json.dumps({k: side[k] for k in ('args', 'figs_dir', 'csv_path',
                                            'provenance')})
    # Checked in both separator spellings a path can be recorded in.
    for form in (REPO_ROOT, REPO_ROOT.replace('\\', '/')):
        assert json.dumps(form)[1:-1] not in text, form
