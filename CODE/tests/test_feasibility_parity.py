"""Feasibility parity between the GA and every other layout method.

Every method's layouts are mapped through ``feasible_layout`` (the GA's
own repair chain) before MC scoring. For that to put all methods on the
same feasible set, the map must be idempotent (a repaired layout is a
fixed point) and its output must carry no overlap under the simulator
fitness's own predicate.

It must also keep the floor-plan engine's invariants
(``experiments._feasibility``): the aisles the as-built store keeps between
zones stay at least ``MIN_AISLE`` wide, and every fixture's access point
can be reached from the door. The later tests pin that on a store the size
of Figure C's (108 fixtures, 15 zones), where the zones include the aisles
and the rule binds, and on the synthetic template, where it takes nothing
and the repair is the one it always was.
"""

import numpy as np
import pandas as pd
import pytest

from synthetic_shops import generate_synthetic_shop
from baselines import (random_valid, perimeter_only, popularity_rank,
                       greedy_swap, assert_layout_valid,
                       assert_no_strict_overlap, _section_inner_bounds)
from dataset_calibration import calibrate_transactional
from experiments._common import (build_headless_shop, base_params_for,
                                 build_headless_shop_from_calibration,
                                 check_layout_invariants,
                                 feasible_layout, layout_to_chromosome,
                                 zone_neighbor, zone_sampler)
from experiments import _feasibility as F
from shop_architecture import MIN_AISLE
from sim_geometry import _nearest_free_cell


def _raw_random_layout(ss, rng):
    """Uniform positions inside section bounds with no overlap repair, so
    the repair chain has overlaps to resolve."""
    out = {}
    for it in ss.items:
        lo_x, lo_y, hi_x, hi_y = _section_inner_bounds(ss, it)
        out[it.name] = (float(rng.uniform(lo_x, hi_x)),
                        float(rng.uniform(lo_y, hi_y)))
    return out


def test_feasible_layout_idempotent_and_overlap_free():
    rng = np.random.default_rng(0)
    for idx in (0, 3, 12, 25):
        for n_items in (10, 12):
            ss = generate_synthetic_shop(name=f'parity_{idx}_{n_items}',
                                         seed=10_000 + idx, n_items=n_items,
                                         width=12.0, height=10.0)
            shop = build_headless_shop(ss)
            names = [it.name for it in ss.items]
            bp = base_params_for(ss)
            layouts = [fn(ss, seed=idx) for fn in (random_valid, perimeter_only,
                                                  popularity_rank, greedy_swap)]
            layouts += [_raw_random_layout(ss, rng) for _ in range(8)]
            for lay in layouts:
                once = feasible_layout(shop, names, lay)
                assert feasible_layout(shop, names, once) == once
                assert_layout_valid(ss, once)
                assert_no_strict_overlap(ss, once)
                _, breakdown = shop._ga_compute_layout_score(
                    layout_to_chromosome(once, names), names, bp)
                assert breakdown['overlap_penalty'] == 0


# --- The floor-plan invariants -----------------------------------------------

def _uci_sized_invoices(n_invoices=1500, seed=0):
    """Invoices over nine categories of twelve products each: the
    assortment of Figure C's store (108 fixtures), so the layout builder
    lays out the same plan -- three perimeter departments and six split
    over two gondolas each, fifteen zones."""
    rng = np.random.default_rng(seed)
    cats = ('Bags', 'Baking', 'Cards', 'General', 'Gifts', 'Home',
            'Kitchen', 'Seasonal', 'Toys')
    products = [(f'{c[:3].upper()}{i:02d}', c, 1.0 + 0.5 * i)
                for c in cats for i in range(12)]
    day0 = pd.Timestamp('2010-12-01 09:00')
    rows = []
    for k in range(n_invoices):
        ts = day0 + pd.Timedelta(days=k // 60,
                                 minutes=int(rng.integers(0, 480)))
        for j in rng.choice(len(products), size=int(rng.integers(1, 8)),
                            replace=False):
            pid, cat, price = products[j]
            rows.append((f'I{k:05d}', pid, f'ITEM {pid}',
                         int(rng.integers(1, 4)), ts, price, cat))
    return pd.DataFrame(rows, columns=['invoice_id', 'product_id',
                                       'product_name', 'quantity',
                                       'timestamp', 'unit_price', 'category'])


@pytest.fixture(scope='module')
def uci_store():
    params = calibrate_transactional(_uci_sized_invoices(), currency='GBP')
    shop = build_headless_shop_from_calibration(
        params, max_items_per_category=12, naive=True)
    names = list(shop.floors[1]['items'])
    return shop, names


def _clean(v):
    return {k: x for k, x in v.items() if x}


def test_uci_sized_store_has_figure_cs_plan(uci_store):
    shop, names = uci_store
    zones = [w for w in shop.floors[1]['walls'] if w.startswith('Section_')]
    assert len(names) == 108 and len(zones) == 15
    plan = F.aisle_plan(shop, names)
    # The aisles between gondolas and to the perimeter runs are kept apart;
    # the regions are tighter than the zones there.
    assert len(plan.kept_apart) > 50
    assert any(any(e > 0 for e in ex) for ex in plan.extras.values())


def _uniform_over_zones(shop, names, rng):
    """Positions uniform over each item's whole zone less the pad -- what
    the searches drew from before the aisle rule, aisles included."""
    plan = F.aisle_plan(shop, names)
    out = {}
    for i, n in enumerate(names):
        sx, sy, sw, sh = plan.zone_rect[plan.zone[i]]
        w, h = plan.sizes[i]
        out[n] = (float(rng.uniform(sx + 0.05, max(sx + 0.05, sx + sw - w - 0.05))),
                  float(rng.uniform(sy + 0.05, max(sy + 0.05, sy + sh - h - 0.05))))
    return out


def test_repair_keeps_aisles_and_reach_on_a_uci_sized_store(uci_store):
    """Uniform draws over every whole zone, uniform draws over the regions,
    one-item and ten-item annealing walks and GA-style perturbations from
    the as-built layout: every repaired layout keeps the as-built aisles at
    MIN_AISLE or more and every fixture shoppable, and is a fixed point of
    the repair. A draw over the whole zones puts fixtures into the aisles,
    so the rule is doing work there (the region clip binds in every call)."""
    shop, names = uci_store
    as_built = {n: tuple(shop.floors[1]['items'][n]['position'])
                for n in names}
    start = feasible_layout(shop, names, as_built)
    assert not _clean(F.layout_violations(shop, names, start))
    rng = np.random.default_rng(1)
    raw = [_uniform_over_zones(shop, names, rng) for _ in range(25)]
    F.reset_repair_stats(shop)
    for lay in raw:
        feasible_layout(shop, names, lay)
    st = F.repair_stats(shop)
    assert st['aisle_clip_calls'] == len(raw)
    assert st['aisle_clip_max_m'] > 0.5          # well past the pad
    sample = zone_sampler(shop, names)
    layouts = raw + [sample(rng) for _ in range(25)]
    for n_move in (1, 10):
        move = zone_neighbor(shop, names, n_move=n_move)
        cur = start
        for _ in range(25):
            cur = feasible_layout(shop, names, move(cur, rng))
            layouts.append(cur)
    for _ in range(10):
        layouts.append({n: (x + rng.normal(0, 1.0), y + rng.normal(0, 3.0))
                        for n, (x, y) in start.items()})
    for lay in layouts:
        once = feasible_layout(shop, names, lay)
        assert not _clean(F.layout_violations(shop, names, once))
        assert feasible_layout(shop, names, once) == once
        check_layout_invariants(shop, names, once)


def test_every_search_operator_stays_inside_the_regions(uci_store):
    """Random search's draws, the annealer's one- and ten-item moves and the
    GA's mutation all propose positions inside each item's region, so the
    region clip never moves one: the searches search the same space, and
    none spends proposals the repair would clip onto an aisle edge. The
    GA's move box is the zone less the aisle rule, and with the repair's
    pad it is the region exactly."""
    shop, names = uci_store
    plan = F.aisle_plan(shop, names)
    boxes = shop._ga_move_boxes(names)
    narrowed = 0
    for i, n in enumerate(names):
        z = plan.zone[i]
        assert boxes[i] == plan.move_box(z)
        bx, by, bw, bh = boxes[i]
        x0, y0, x1, y1 = plan.region(z)
        assert abs(bx + 0.05 - x0) < 1e-12 and abs(by + 0.05 - y0) < 1e-12
        assert abs(bx + bw - 0.05 - x1) < 1e-12
        assert abs(by + bh - 0.05 - y1) < 1e-12
        narrowed += boxes[i] != shop._ga_get_section_bounds(n)
    assert narrowed > 0

    start = feasible_layout(shop, names,
                            {n: tuple(shop.floors[1]['items'][n]['position'])
                             for n in names})
    rng = np.random.default_rng(4)
    sample = zone_sampler(shop, names)
    F.reset_repair_stats(shop)
    for _ in range(30):
        feasible_layout(shop, names, sample(rng))
    for n_move in (1, 10):
        move = zone_neighbor(shop, names, n_move=n_move)
        cur = start
        for _ in range(30):
            cur = feasible_layout(shop, names, move(cur, rng))
    np.random.seed(4)
    chrom = layout_to_chromosome(start, names)
    for _ in range(30):
        child = shop._ga_mutate(chrom.copy(), names, 1.0)
        feasible_layout(shop, names, {n: tuple(child[i])
                                      for i, n in enumerate(names)})
    st = F.repair_stats(shop)
    assert st['calls'] >= 120
    assert st['aisle_clip_calls'] == 0 and st['aisle_clip_max_m'] == 0.0


def test_aisle_rule_summary(uci_store):
    """What the rule takes from the search space: on the UCI-sized store it
    narrows many fixtures on one axis and keeps a share of the zone's room
    between zero and one; on the synthetic template it takes nothing."""
    shop, names = uci_store
    rec = F.aisle_rule_summary(shop, names)
    assert rec['n_items'] == 108 and rec['n_zones'] == 15
    assert rec['n_items_narrowed'] > 0 and rec['n_zones_narrowed'] > 0
    assert 0.0 < rec['area_share_kept_geomean_unpinned'] < 1.0
    assert 0 <= rec['n_items_pinned'] < rec['n_items']
    for ax in rec['per_axis'].values():
        assert 0.0 <= ax['share_kept_min'] <= ax['share_kept_median'] <= 1.0
    ss = generate_synthetic_shop(name='aisle_sum', seed=10_003, n_items=10,
                                 width=12.0, height=10.0)
    sshop = build_headless_shop(ss)
    snames = [it.name for it in ss.items]
    srec = F.aisle_rule_summary(sshop, snames)
    assert srec['n_items_narrowed'] == 0 and srec['n_zones_narrowed'] == 0
    assert srec['area_share_kept_geomean_unpinned'] == pytest.approx(1.0)
    assert srec['n_items_pinned'] == 0
    # ...and there the GA moves in the zone itself, as it always did.
    assert sshop._ga_move_boxes(snames) == [
        sshop._ga_get_section_bounds(n) for n in snames]


def test_the_check_detects_a_closed_aisle_and_a_walled_in_fixture(uci_store):
    """The invariant check is not vacuous: a gondola pushed to its zone
    edge, 0.05 m from the next zone, closes the aisle it faces; a perimeter
    fixture boxed in by its neighbours cannot be shopped."""
    shop, names = uci_store
    plan = F.aisle_plan(shop, names)
    start = feasible_layout(shop, names,
                            {n: tuple(shop.floors[1]['items'][n]['position'])
                             for n in names})
    # A gondola zone kept apart from a neighbour on its right.
    za, zb, _ = next((a, b, need) for a, b, need in plan.kept_apart
                     if plan.run_axis[a] == 1 and plan.run_axis[b] == 1
                     and plan.zone_rect[a][0] < plan.zone_rect[b][0]
                     and abs(plan.zone_rect[b][0] - (plan.zone_rect[a][0]
                                                     + plan.zone_rect[a][2]))
                     < 1e-6)
    i = plan.members[za][0]
    sx, sy, sw, sh = plan.zone_rect[za]
    w = plan.sizes[i][0]
    bad = dict(start)
    bad[names[i]] = (sx + sw - w - 0.05, start[names[i]][1])
    v = F.layout_violations(shop, names, bad)
    assert v['clearance'] and all(g < MIN_AISLE for _, _, g, _ in
                                  v['clearance'])
    with pytest.raises(F.LayoutInvariantError):
        check_layout_invariants(shop, names, bad)
    # ...and the repair puts it back inside its region.
    fixed = feasible_layout(shop, names, bad)
    assert not _clean(F.layout_violations(shop, names, fixed))


def test_the_rule_takes_nothing_on_the_synthetic_template():
    """Sections 1.5 m apart, zones without aisles: every region is the zone
    less the repair's pad, and every repaired layout keeps the invariants."""
    rng = np.random.default_rng(3)
    for idx in (0, 7, 19):
        ss = generate_synthetic_shop(name=f'aisle_{idx}', seed=10_000 + idx,
                                     n_items=10, width=12.0, height=10.0)
        shop = build_headless_shop(ss)
        names = [it.name for it in ss.items]
        plan = F.aisle_plan(shop, names)
        assert all(ex == (0.0, 0.0, 0.0, 0.0) for ex in plan.extras.values())
        F.reset_repair_stats(shop)
        for _ in range(20):
            once = feasible_layout(shop, names, _raw_random_layout(ss, rng))
            assert not _clean(F.layout_violations(shop, names, once))
        assert F.repair_stats(shop)['aisle_clip_calls'] == 0
        assert F.repair_stats(shop)['reach_single_file'] == 0


def test_access_cells_are_the_simulators():
    """The vectorized access-point search returns the simulator's own
    standing spot, cell for cell, ties included."""
    rng = np.random.default_rng(5)
    for density in (0.2, 0.6, 0.95):
        blocked = rng.random((40, 30)) < density
        cx = rng.uniform(-2, 42, 200)
        cy = rng.uniform(-2, 32, 200)
        got = F._access_cells(blocked, cx, cy)
        want = [_nearest_free_cell(blocked, x, y) for x, y in zip(cx, cy)]
        assert got == want
    full = np.ones((5, 5), dtype=bool)
    assert F._access_cells(full, np.array([2.0]), np.array([2.0])) == [None]


def test_single_file_fallback_puts_a_run_back_on_its_line(uci_store):
    """The first reachability fallback: a zone's fixtures back on their
    as-built line across the run, in the layout's order along it, without
    overlap and inside the region."""
    shop, names = uci_store
    plan = F.aisle_plan(shop, names)
    z = next(z for z, idx in plan.members.items() if len(idx) >= 6)
    idx = plan.members[z]
    out = plan.as_built.copy()
    rng = np.random.default_rng(2)
    x0, y0, x1, y1 = plan.region(z)
    for i in idx:
        out[i] = (rng.uniform(x0, x1 - plan.sizes[i][0]),
                  rng.uniform(y0, y1 - plan.sizes[i][1]))
    order_before = sorted(idx, key=lambda i: (out[i, plan.run_axis[z]], i))
    assert F._single_file(plan, out, z)
    a, lat = plan.run_axis[z], 1 - plan.run_axis[z]
    assert sorted(idx, key=lambda i: out[i, a]) == order_before
    for i in idx:
        assert out[i, lat] == plan.as_built[i, lat]
        assert x0 - 1e-9 <= out[i, 0] and out[i, 0] + plan.sizes[i][0] <= x1 + 1e-9
        assert y0 - 1e-9 <= out[i, 1] and out[i, 1] + plan.sizes[i][1] <= y1 + 1e-9
    assert not F._zone_overlaps(out, plan.sizes, idx)


def test_repair_stats_combine_counts_and_distances():
    """Records from several runs combine as counts summed and the farthest
    clip kept, never a sum of distances."""
    a = {'calls': 3, 'aisle_clip_calls': 1, 'aisle_clip_max_m': 0.4}
    b = {'calls': 2, 'aisle_clip_calls': 2, 'aisle_clip_max_m': 0.7}
    assert F.combine_repair_stats([a, b]) == {
        'calls': 5, 'aisle_clip_calls': 3, 'aisle_clip_max_m': 0.7}


# --- The GUI's GA searches the same feasible set (R06) -----------------------

def test_gui_ga_repair_is_the_shared_repair(uci_store, monkeypatch):
    """The GA tab and the Optimize pipeline repair every candidate with
    ``GARunMixin._ga_resolve_overlaps`` and scale their moves to
    ``_ga_move_boxes``. On a store whose zones include the aisles both are
    the shared repair and its move boxes, read as the GUI reads them -- the
    layout on the floor as the store as built: every repaired layout equals
    the headless repair's bit for bit and keeps the aisles and the reach,
    where the zone-only snap the GUI used before closes aisles."""
    from viz_ga_run import GARunMixin
    from experiments._common import _repair_chrom_overlaps
    shop, names = uci_store
    headless_plan = F.aisle_plan(shop, names)
    # The GUI keeps no builder record: its plan comes from the floor.
    monkeypatch.setattr(shop, 'as_built_layout', None)
    plan = shop._ga_aisle_plan(names)
    assert plan is not None and shop._ga_plan_matches_zones(plan, names)
    assert plan.extras == headless_plan.extras
    assert plan.kept_apart == headless_plan.kept_apart
    assert GARunMixin._ga_move_boxes(shop, names) == [
        plan.move_box(z) for z in plan.zone]
    rng = np.random.default_rng(7)
    closed = 0
    for _ in range(12):
        raw = _uniform_over_zones(shop, names, rng)
        chrom = shop._ga_repair(layout_to_chromosome(raw, names), names)
        gui = GARunMixin._ga_resolve_overlaps(shop, chrom.copy(), names)
        np.testing.assert_array_equal(
            gui, _repair_chrom_overlaps(shop, chrom.copy(), names))
        lay = {n: (float(gui[i, 0]), float(gui[i, 1]))
               for i, n in enumerate(names)}
        assert not _clean(F.layout_violations(shop, names, lay))
        old = GARunMixin._ga_zone_snap(shop, chrom.copy(), names)
        closed += bool(F.layout_violations(
            shop, names, {n: (float(old[i, 0]), float(old[i, 1]))
                          for i, n in enumerate(names)})['clearance'])
    # The test can see what the shared repair prevents.
    assert closed > 0


def test_gui_ga_keeps_the_zone_snap_where_the_plan_does_not_apply():
    """A store with items off the ground floor (``F<n>:`` keys) has no
    aisle plan, and the GUI repairs it zone by zone as before; on the
    synthetic template, where the plan applies and takes nothing, the shared
    repair and the zone snap agree exactly."""
    from viz_ga_run import GARunMixin
    ss = generate_synthetic_shop(name='gui_plan', seed=10_004, n_items=10,
                                 width=12.0, height=10.0)
    shop = build_headless_shop(ss)
    names = [it.name for it in ss.items]
    assert shop._ga_aisle_plan(names) is not None
    assert shop._ga_aisle_plan(names + ['F2:not_here']) is None
    rng = np.random.default_rng(5)
    for _ in range(10):
        chrom = shop._ga_repair(
            layout_to_chromosome(_raw_random_layout(ss, rng), names), names)
        np.testing.assert_array_equal(
            GARunMixin._ga_resolve_overlaps(shop, chrom.copy(), names),
            GARunMixin._ga_zone_snap(shop, chrom.copy(), names))


class _Var:
    def __init__(self, v=None):
        self.v = v

    def get(self):
        return self.v

    def set(self, v):
        self.v = v


class _Root:
    def update_idletasks(self):
        pass


def test_gui_ga_tab_run_returns_and_scores_only_feasible_layouts(
        uci_store, monkeypatch):
    """The GA tab's own loop (``_run_ga_optimization``), run headlessly on
    the UCI-sized store with its smallest settings: every layout it repairs
    -- the seeded population and every child -- and the layout it returns
    keep the floor-plan invariants."""
    from viz_ga_run import GARunMixin
    shop, names = uci_store
    monkeypatch.setattr(shop, 'as_built_layout', None)
    for attr, v in (('_ga_pop', 10), ('_ga_gens', 5), ('_ga_mut', 18),
                    ('_ga_elite', 20), ('_ga_mc_iters', 100),
                    ('_ga_mc_days', 1), ('_ga_progress', '')):
        monkeypatch.setattr(shop, attr, _Var(v), raising=False)
    monkeypatch.setattr(shop, 'tk_root', _Root())
    repaired = []

    def _spy(chrom, item_names):
        out = GARunMixin._ga_resolve_overlaps(shop, chrom, item_names)
        repaired.append((list(item_names), out.copy()))
        return out
    monkeypatch.setattr(shop, '_ga_resolve_overlaps', _spy, raising=False)
    shown = {}
    monkeypatch.setattr(shop, '_display_ga_results',
                        lambda item_names, base_params, best_chrom, *a, **k:
                        shown.update(names=item_names, best=best_chrom),
                        raising=False)
    np.random.seed(11)
    GARunMixin._run_ga_optimization(shop)
    assert sorted(shown['names']) == sorted(names)
    # The seeded population, then eight children a generation (two
    # elites of ten carried over).
    assert len(repaired) == 10 + 5 * 8
    for item_names, chrom in repaired + [(shown['names'], shown['best'])]:
        lay = {n: (float(chrom[i, 0]), float(chrom[i, 1]))
               for i, n in enumerate(item_names)}
        assert not _clean(F.layout_violations(shop, item_names, lay))
    # The engine built this store, so the shared rules held throughout.
    assert shop._ga_repair_notice is None
    for attr in ('_ga_best_chromosome', '_ga_best_layout', '_ga_item_names',
                 '_ga_repair_notice'):
        shop.__dict__.pop(attr, None)


def _walled_in_store():
    """A synthetic store read the way the GUI reads it (no builder record:
    the layout on the floor is the store as built) with one fixture walled
    into a pocket 0.6 m around it, so it cannot be shopped from the door as
    the store stands."""
    ss = generate_synthetic_shop(name='gui_pocket', seed=10_004, n_items=10,
                                 width=12.0, height=10.0)
    shop = build_headless_shop(ss)
    shop.as_built_layout = None
    names = [it.name for it in ss.items]
    f1 = shop.floors[1]
    boxed = next(n for n in names
                 if 'impulse' not in f1['items'][n]['category'].lower())
    x, y = f1['items'][boxed]['position']
    w, h = f1['items'][boxed]['size']
    m, t = 0.6, 0.2
    f1['walls']['Pocket_L'] = {'position': [x - m - t, y - m - t],
                               'size': [t, h + 2 * m + 2 * t]}
    f1['walls']['Pocket_R'] = {'position': [x + w + m, y - m - t],
                               'size': [t, h + 2 * m + 2 * t]}
    f1['walls']['Pocket_B'] = {'position': [x - m - t, y - m - t],
                               'size': [w + 2 * m + 2 * t, t]}
    f1['walls']['Pocket_T'] = {'position': [x - m - t, y + h + m],
                               'size': [w + 2 * m + 2 * t, t]}
    return ss, shop, names, boxed


def test_gui_ga_keeps_searching_a_floor_with_an_unshoppable_fixture():
    """A GUI floor with a fixture that cannot be shopped as built can never
    pass the shared repair's reach check, so that repair would end every
    candidate at the whole as-built store and the GA would score its start
    alone. The GUI then repairs zone by zone, as before the shared repair,
    scales its moves to the zones, and says why."""
    from viz_ga_run import GARunMixin
    ss, shop, names, boxed = _walled_in_store()
    plan = shop._ga_aisle_plan(names)
    assert plan is not None and shop._ga_plan_matches_zones(plan, names)
    assert F.as_built_unshoppable(plan) == (boxed,)
    # The shared repair on its own returns the as-built store every time.
    chrom0 = np.array([shop.floors[1]['items'][n]['position'] for n in names],
                      dtype=float)
    np.random.seed(3)
    children = [shop._ga_mutate(chrom0.copy(), names, 0.5)
                for _ in range(12)]
    F.reset_repair_stats(shop)
    for child in children:
        np.testing.assert_array_equal(
            F.repair_positions(shop, child.copy(), names), plan.as_built)
    assert F.repair_stats(shop)['reach_as_built_store'] == len(children)
    # The GUI falls back to the zone snap, and says so.
    assert shop._ga_shared_plan(names) is None
    note = shop._ga_repair_note(names)
    assert note and 'cannot be shopped' in note and boxed in note
    assert GARunMixin._ga_move_boxes(shop, names) == [
        shop._ga_get_section_bounds(n) for n in names]
    F.reset_repair_stats(shop)
    distinct = 0
    for child in children:
        out = GARunMixin._ga_resolve_overlaps(shop, child.copy(), names)
        np.testing.assert_array_equal(
            out, GARunMixin._ga_zone_snap(shop, child.copy(), names))
        distinct += not np.array_equal(out, plan.as_built)
    assert distinct == len(children)
    assert F.repair_stats(shop)['calls'] == 0


def test_gui_ga_tab_run_moves_items_on_a_floor_with_an_unshoppable_fixture(
        monkeypatch):
    """The GA tab's own loop on that floor searches layouts other than the
    one it started from, and reports the fallback."""
    from viz_ga_run import GARunMixin
    ss, shop, names, boxed = _walled_in_store()
    for attr, v in (('_ga_pop', 10), ('_ga_gens', 5), ('_ga_mut', 18),
                    ('_ga_elite', 20), ('_ga_mc_iters', 100),
                    ('_ga_mc_days', 1), ('_ga_progress', '')):
        monkeypatch.setattr(shop, attr, _Var(v), raising=False)
    monkeypatch.setattr(shop, 'tk_root', _Root())
    repaired = []

    def _spy(chrom, item_names):
        out = GARunMixin._ga_resolve_overlaps(shop, chrom, item_names)
        repaired.append(out.copy())
        return out
    monkeypatch.setattr(shop, '_ga_resolve_overlaps', _spy, raising=False)
    monkeypatch.setattr(shop, '_display_ga_results',
                        lambda *a, **k: None, raising=False)
    np.random.seed(11)
    GARunMixin._run_ga_optimization(shop)
    as_built = shop._ga_aisle_plan(names).as_built
    moved = sum(not np.array_equal(c, as_built) for c in repaired)
    assert moved >= len(repaired) - 1        # all but the seeded start
    assert 'cannot be shopped' in shop._ga_repair_notice
    assert 'not applied' in shop._ga_progress.get()


def test_gui_ga_holds_an_engine_built_floor_to_the_shared_repair(uci_store,
                                                                 monkeypatch):
    """On a store the engine built, every fixture is shoppable as built, so
    the fallback never applies and the GUI says nothing."""
    shop, names = uci_store
    monkeypatch.setattr(shop, 'as_built_layout', None)
    plan = shop._ga_aisle_plan(names)
    assert F.as_built_unshoppable(plan) == ()
    assert shop._ga_shared_plan(names) is plan
    assert shop._ga_repair_note(names) is None
