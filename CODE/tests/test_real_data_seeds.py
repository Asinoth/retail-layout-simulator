"""Figure C over several search seeds: seeds, aggregation, worker count.

``experiments.run_real_data_seeds`` repeats Figure C's three equal-budget
searches (GA, random search, simulated annealing) over a range of search
seeds and scores every layout under Figure C's held-out evaluation seeds,
so the across-seed interval measures how much a different search seed
changes the GA's lead. These tests pin, without the workbook:

  * every search seed of the range keeps its own search and selection
    seeds inside its block of 1000 and below the evaluation seeds, and the
    command line refuses a range that would not -- before it looks for the
    workbook -- while taking every other option, default included, from
    Figure C's runner;
  * the aggregation on a hand-made table: per-seed paired means and
    percentages, the across-seed t-interval, range and lead counts, and the
    checks that refuse results from unequal budgets, another start or a
    different store;
  * on a small store calibrated from a synthetic invoice frame, the way
    tests/test_zone_search.py builds one: a seed run in a worker process
    returns exactly what Figure C's own steps return for that seed in this
    one, and the results and their summary do not depend on the worker
    count;
  * the paper's across-seed macros only quote a run at Figure C's own
    store, budget and evaluation panel.
"""
import argparse
import csv
import json
import math

import numpy as np
import pandas as pd
import pytest

import make_results_macros as MRM
from dataset_calibration import calibrate_transactional
from experiments import run_real_data_seeds as RDS
from experiments._common import GA_N_FINAL_SEEDS
from experiments.run_real_data_example import (EVAL_SEED_BASE, add_arguments,
                                               build_store, run_searches,
                                               score_layouts)

#: Student t quantile t_{0.975, 2} from its closed form at two degrees of
#: freedom, t_p = (2p - 1) / sqrt(2p(1 - p)), so the interval is checked
#: against a number the code under test did not compute.
T975_DF2 = (2 * 0.975 - 1) / math.sqrt(2 * 0.975 * (1 - 0.975))


# --- seeds -------------------------------------------------------------------

@pytest.mark.parametrize('first, k, n_gens, pop_size', [
    (0, 10, 25, 30),                        # the paper design
    (990, 10, 25, 30),                      # last seed at the ceiling
    (0, 4, 1000 - 1 - GA_N_FINAL_SEEDS, 2),  # search ranges fill their block
    (500, 3, 1, 2),
])
def test_search_seed_ranges_are_disjoint_and_below_the_evaluation_seeds(
        first, k, n_gens, pop_size):
    assert RDS.seed_plan_error(first, k, n_gens, pop_size) is None
    plan = RDS.seed_plan(first, k, n_gens, pop_size, n_replicates=30)
    assert plan['search_seeds'] == list(range(first, first + k))
    assert plan['paired_eval_seeds'] == [EVAL_SEED_BASE, EVAL_SEED_BASE + 30]
    owner = {}
    for s in plan['search_seeds']:
        ranges = plan['search_seed_ranges'][str(s)]
        assert set(ranges) == set(RDS.SEARCHES)
        for lo, hi in ranges.values():
            # Inside the seed's own block of 1000 ...
            assert s * 1000 <= lo < hi <= (s + 1) * 1000
            # ... and below every evaluation seed.
            assert hi <= EVAL_SEED_BASE
            for mc_seed in range(lo, hi):
                # No Monte Carlo seed is searched under by two search seeds
                # (the three searches of one seed share theirs by design).
                assert owner.setdefault(mc_seed, s) == s


@pytest.mark.parametrize('first, k, n_gens', [
    (991, 10, 25),     # last search seed 1000 reaches the evaluation seeds
    (995, 10, 25),
    (-1, 10, 25),
    (0, 10, 1000 - GA_N_FINAL_SEEDS),  # final seeds spill out of the block
])
def test_seed_plan_refuses_ranges_that_leave_their_block(first, k, n_gens):
    assert RDS.seed_plan_error(first, k, n_gens, 30) is not None


def _no_workbook(monkeypatch, found=None):
    """Stand in for the workbook lookup: record that it ran, or fail if a
    test expects the command line to stop before it."""
    def _resolve(p, args):
        if found is None:
            raise AssertionError("the workbook was looked up before the "
                                 "arguments were checked")
        found.append(True)
        args.retail_path = 'workbook.xlsx'
    monkeypatch.setattr(RDS, 'resolve_workbook', _resolve)


@pytest.mark.parametrize('argv', [
    ['--ga-seed', '995', '--n-search-seeds', '10'],
    ['--ga-seed', '1000'],
    ['--n-search-seeds', '1'],
    ['--workers', '0'],
    ['--pop-size', '1'],
])
def test_command_line_refuses_bad_designs_before_the_workbook(monkeypatch,
                                                              argv):
    _no_workbook(monkeypatch)
    with pytest.raises(SystemExit):
        RDS.parse_args(argv)


def test_command_line_takes_figure_cs_options_and_defaults(monkeypatch):
    found = []
    _no_workbook(monkeypatch, found)
    args = RDS.parse_args([])
    assert found == [True]
    p = argparse.ArgumentParser()
    add_arguments(p)
    figc = vars(p.parse_args([]))
    ours = vars(args)
    for key, value in figc.items():
        if key != 'retail_path':
            assert ours[key] == value, key
    assert args.n_search_seeds == RDS.DEFAULT_N_SEARCH_SEEDS
    assert args.workers == 1


# --- aggregation on a hand-made table ------------------------------------------

BASE = [100.0, 110.0, 90.0, 100.0]


def _plus(offsets):
    if np.isscalar(offsets):
        offsets = [offsets] * len(BASE)
    return [b + o for b, o in zip(BASE, offsets)]


def _table():
    """Three search seeds, four replicates. Per seed, the paired means are
    lift 20 / 22 / 24, GA minus random search 15 / 15 / 19 and GA minus
    annealing -2 / 6 / 14."""
    return [
        {'search_seed': 7, 'revenues': {
            'baseline': list(BASE), 'optimized': _plus([19, 21, 20, 20]),
            'rs': _plus([4, 6, 5, 5]), 'sa': _plus(22)}},
        {'search_seed': 8, 'revenues': {
            'baseline': list(BASE), 'optimized': _plus(22),
            'rs': _plus(7), 'sa': _plus(16)}},
        {'search_seed': 9, 'revenues': {
            'baseline': list(BASE), 'optimized': _plus([23, 25, 24, 24]),
            'rs': _plus(5), 'sa': _plus(10)}},
    ]


def _expected_across(per_seed, other_means):
    x = np.asarray(per_seed, dtype=float)
    mean, sd = x.mean(), x.std(ddof=1)
    half = T975_DF2 * sd / math.sqrt(3)
    denom = float(np.mean(other_means))
    return {'mean': mean, 'sd': sd, 'ci_lo': mean - half,
            'ci_hi': mean + half, 'min': x.min(), 'max': x.max(),
            'n_ga_leads': int((x > 0).sum()), 'pct': mean / denom * 100.0,
            'pct_ci': [(mean - half) / denom * 100.0,
                       (mean + half) / denom * 100.0]}


def test_per_seed_paired_means_and_percentages():
    rows = RDS.summarize_seeds(_table())['per_seed']
    assert [r['search_seed'] for r in rows] == [7, 8, 9]
    assert all(r['n_replicates'] == 4 for r in rows)
    got = {key: [r[key]['paired_mean_diff_GA_minus_X'] for r in rows]
           for key in ('lift_over_baseline', 'random_search',
                       'simulated_annealing')}
    assert got['lift_over_baseline'] == pytest.approx([20, 22, 24])
    assert got['random_search'] == pytest.approx([15, 15, 19])
    assert got['simulated_annealing'] == pytest.approx([-2, 6, 14])
    # Percentages of the other layout's mean revenue in that seed.
    assert rows[0]['lift_over_baseline']['pct_diff'] == pytest.approx(20.0)
    assert rows[0]['random_search']['pct_diff'] == \
        pytest.approx(15 / 105 * 100)
    assert rows[2]['simulated_annealing']['pct_diff'] == \
        pytest.approx(14 / 110 * 100)
    assert [r['simulated_annealing']['ga_leads'] for r in rows] == \
        [False, True, True]
    # Evaluation noise inside a per-seed mean: seed 7's lift varies by
    # (-1, +1, 0, 0) around 20 over its replicates.
    assert rows[0]['lift_over_baseline']['replicate_se'] == \
        pytest.approx(math.sqrt(2 / 3) / 2)
    assert rows[1]['lift_over_baseline']['replicate_se'] == \
        pytest.approx(0.0)


@pytest.mark.parametrize('key, per_seed, other_means, excludes_zero', [
    ('lift_over_baseline', [20, 22, 24], [100, 100, 100], True),
    ('random_search', [15, 15, 19], [105, 107, 105], True),
    ('simulated_annealing', [-2, 6, 14], [122, 116, 110], False),
])
def test_across_seed_interval_range_and_leads(key, per_seed, other_means,
                                              excludes_zero):
    a = RDS.summarize_seeds(_table())['across_seeds'][key]
    want = _expected_across(per_seed, other_means)
    assert a['n_seeds'] == 3
    assert a['ci_level'] == 0.95
    for field in ('mean', 'sd', 'min', 'max', 'pct'):
        assert a[field] == pytest.approx(want[field], rel=1e-12), field
    # The interval goes through a numerical t quantile, good to about 1e-11.
    for field in ('ci_lo', 'ci_hi'):
        assert a[field] == pytest.approx(want[field], rel=1e-9), field
    assert a['pct_ci'] == pytest.approx(want['pct_ci'], rel=1e-9)
    assert a['n_ga_leads'] == want['n_ga_leads']
    assert a['excludes_zero'] is excludes_zero


def test_t_interval_matches_the_textbook_formula():
    mean, lo, hi, sd = RDS.t_interval([-2.0, 6.0, 14.0])
    assert (mean, sd) == pytest.approx((6.0, 8.0))
    assert (lo, hi) == pytest.approx((6.0 - T975_DF2 * 8 / math.sqrt(3),
                                      6.0 + T975_DF2 * 8 / math.sqrt(3)))
    assert all(math.isnan(v) for v in RDS.t_interval([3.0])[1:])


def test_results_csv_holds_one_row_per_seed_and_replicate(tmp_path):
    table = _table()
    path = RDS.write_results_csv(str(tmp_path), table)
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3 * 4
    assert [(int(r['search_seed']), int(r['replicate'])) for r in rows] == \
        [(s, r) for s in (7, 8, 9) for r in range(4)]
    assert {int(r['mc_seed']) for r in rows} == \
        set(range(EVAL_SEED_BASE, EVAL_SEED_BASE + 4))
    # The table carries the per-seed means the summary reports.
    for s, lift in ((7, 20.0), (8, 22.0), (9, 24.0)):
        diffs = [float(r['diff']) for r in rows if int(r['search_seed']) == s]
        assert np.mean(diffs) == pytest.approx(lift)
    assert float(rows[0]['diff_sa']) == pytest.approx(19 - 22)


def _checked(**change):
    rec = {'search_seed': 1,
           'revenues': {'baseline': [1.0, 2.0]},
           'evaluation_counts': {m: 12 for m in RDS.SEARCHES},
           'search_start': {m: 'asbuilt' for m in RDS.SEARCHES}}
    rec.update(change)
    return rec


def test_seed_results_are_checked_before_they_are_aggregated():
    good = [_checked(search_seed=1), _checked(search_seed=2)]
    RDS.check_seed_results(good, [1, 2], budget=12)
    with pytest.raises(AssertionError):      # out of order
        RDS.check_seed_results(good[::-1], [1, 2], budget=12)
    with pytest.raises(AssertionError):      # unequal search budgets
        RDS.check_seed_results(
            [good[0], _checked(search_seed=2, evaluation_counts={
                'GA': 12, 'random_search': 12, 'simulated_annealing': 11})],
            [1, 2], budget=12)
    with pytest.raises(AssertionError):      # a search not from as-built
        RDS.check_seed_results(
            [good[0], _checked(search_seed=2, search_start={
                'GA': 'asbuilt', 'random_search': 'asbuilt',
                'simulated_annealing': 'popularity'})], [1, 2], budget=12)
    with pytest.raises(AssertionError):      # a different store
        RDS.check_seed_results(
            [good[0], _checked(search_seed=2,
                               revenues={'baseline': [1.0, 2.5]})],
            [1, 2], budget=12)


def test_search_start_joins_starts_that_differ_between_seeds():
    recs = [_checked(), _checked(search_start={
        'GA': 'asbuilt', 'random_search': 'cold',
        'simulated_annealing': 'asbuilt'})]
    assert RDS.search_start(recs) == {'GA': 'asbuilt',
                                      'random_search': 'asbuilt/cold',
                                      'simulated_annealing': 'asbuilt'}


# --- a small calibrated store, run in one process and in two -----------------

TINY = dict(max_items_per_category=4, n_gens=2, pop_size=4, mc_iters=20,
            mc_days=3, sa_initial_accept=0.8, n_mc_replicates=3)
SEEDS = (3, 4, 5)


def _invoices(n_invoices=400, seed=0):
    """Invoices over four categories, as tests/test_zone_search.py builds
    them: enough for the layout builder to lay out a small store."""
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


@pytest.fixture(scope='module')
def params():
    return calibrate_transactional(_invoices(), currency='GBP')


@pytest.fixture(scope='module')
def design(params):
    store = build_store(params, TINY['max_items_per_category'], verbose=False)
    return RDS.SeedDesign(**TINY,
                          store_fingerprint=RDS.store_fingerprint(store))


@pytest.fixture(scope='module')
def runs(params, design):
    """The same seeds in this process and spread over two workers."""
    return {workers: RDS.map_search_seeds(params, SEEDS, design, workers)
            for workers in (1, 2)}


def _without_walls(per_seed):
    return [{k: v for k, v in r.items() if k != 'wall_seconds'}
            for r in per_seed]


def test_results_do_not_depend_on_the_worker_count(runs):
    serial, pooled = runs[1], runs[2]
    assert [r['search_seed'] for r in pooled] == list(SEEDS)
    assert _without_walls(serial) == _without_walls(pooled)
    assert RDS.summarize_seeds(serial) == RDS.summarize_seeds(pooled)
    RDS.check_seed_results(pooled, SEEDS,
                           budget=TINY['n_gens'] * TINY['pop_size'])


def test_a_search_seed_runs_figure_cs_steps_at_that_seed(params, runs):
    """A seed run in a worker process gives what Figure C's own store,
    searches and scoring give for that seed here."""
    seed = SEEDS[1]
    got = runs[2][SEEDS.index(seed)]
    store = build_store(params, TINY['max_items_per_category'], verbose=False)
    searches = run_searches(
        store, seed=seed, n_gens=TINY['n_gens'], pop_size=TINY['pop_size'],
        mc_iters=TINY['mc_iters'], mc_days=TINY['mc_days'],
        sa_initial_accept=TINY['sa_initial_accept'], verbose=False)
    revs = score_layouts(
        store, (('baseline', store.baseline_layout),
                ('optimized', searches['optimized']),
                ('rs', searches['rs']), ('sa', searches['sa'])),
        n_replicates=TINY['n_mc_replicates'], mc_iters=TINY['mc_iters'],
        mc_days=TINY['mc_days'])
    assert got['revenues'] == {k: v.tolist() for k, v in revs.items()}
    assert got['evaluation_counts'] == searches['eval_counts']
    assert got['final_evaluation_counts'] == searches['final_eval_counts']
    assert got['search_start'] == searches['starts']


def test_seed_macros_need_figure_cs_own_design(tmp_path):
    """The across-seed macros describe the spread of Figure C's comparison,
    so a run that searched another store or at another budget is not
    quoted, however many seeds and replicates it has."""
    figc_args = {'sheets': 'Year 2009-2010,Year 2010-2011',
                 'max_items_per_category': 12, 'assumed_conversion': 0.30,
                 'n_gens': 25, 'pop_size': 30, 'mc_iters': 2000,
                 'mc_days': 30, 'n_mc_replicates': 30,
                 'sa_initial_accept': 0.8, 'exclude_anonymous': False}
    figc = tmp_path / 'real_data_uci_x'
    seeds = tmp_path / 'real_data_seeds_x'
    figc.mkdir()
    seeds.mkdir()
    (figc / 'sidecar.json').write_text(json.dumps({'args': figc_args}))

    def summary(**change):
        design = {k: figc_args[k] for k in MRM.FIGC_SEEDS_SHARED_DESIGN}
        design.update(change)
        (seeds / 'summary.json').write_text(json.dumps({
            'design': design,
            'sa_schedule': {'sa_initial_accept': 0.8}}))
        return MRM._figc_seeds_match_figc(str(seeds), str(figc))

    assert summary()
    assert not summary(n_gens=2)                       # another budget
    assert not summary(max_items_per_category=8)       # another store
    assert not summary(sheets='Year 2010-2011')
    assert not summary(mc_days=5)
    assert not summary(exclude_anonymous=True)          # a sensitivity run
    assert summary() and not MRM._figc_seeds_match_figc(str(seeds), None)


def test_a_rebuilt_store_that_differs_is_refused(params, design):
    other = RDS.SeedDesign(**{**TINY, 'store_fingerprint': '0' * 64})
    with pytest.raises(RuntimeError):
        RDS.run_search_seed(params, SEEDS[0], other)
    assert design.store_fingerprint != other.store_fingerprint
