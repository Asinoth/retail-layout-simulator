"""Search parity: the equal-budget searches share one starting point.

Figure B compares the GA, random search and simulated annealing at one
search budget. That comparison is about the searches only if none of them
is handed a better starting point than the others, so all three start
from the as-built layout: the GA seeds its population from it, random
search scores it as its first candidate and SA anneals from it. SA's
former warm start from ``popularity_rank`` survives as a named
sensitivity variant (``start='popularity'``). These tests pin the shared
start, that variant, the hooks that replace the default draw and move,
and the search budget in every configuration.
"""

import numpy as np
import pytest

import experiments.metaheuristics as mh
from synthetic_shops import generate_synthetic_shop
from baselines import popularity_rank, perimeter_only, random_valid
from experiments._common import (build_headless_shop, base_params_for,
                                 anchor_base_params,
                                 feasible_layout, run_ga_headless,
                                 chromosome_to_layout)
from experiments.metaheuristics import (random_search, simulated_annealing,
                                        _neighbor)

MC = dict(mc_iters=20, mc_days=5)
SEED = 3


def _setup(seed=0):
    ss = generate_synthetic_shop(name=f'parity_{seed}', seed=41_000 + seed,
                                 n_items=8, width=12.0, height=10.0)
    shop = build_headless_shop(ss)
    names = [it.name for it in ss.items]
    init = {n: tuple(shop.floors[1]['items'][n]['position']) for n in names}
    bp = anchor_base_params(shop, names, base_params_for(ss), init)
    return ss, shop, names, bp, init


@pytest.fixture
def fit_log(monkeypatch):
    """Every fitness evaluation a search makes, as (layout, MC seed) in
    call order. The real objective still scores each one."""
    log = []
    real = mh._fit

    def _logged(shop, item_names, layout, base_params, seed, mc_iters,
                mc_days):
        log.append((dict(layout), seed))
        return real(shop, item_names, layout, base_params, seed, mc_iters,
                    mc_days)

    monkeypatch.setattr(mh, '_fit', _logged)
    return log


def _search_calls(log, seed, budget, block):
    """The evaluations made under search seeds, i.e. before final
    selection, whose seeds start at ``seed*1000 + n_blocks + 1``."""
    first_final = seed * 1000 + mh._n_blocks(budget, block) + 1
    return [(lay, s) for lay, s in log if seed * 1000 <= s < first_final]


def test_every_search_starts_from_the_as_built_layout():
    """The first layout the GA, random search and SA each evaluate is the
    as-built layout mapped onto the feasible set."""
    ss, shop, names, bp, init = _setup()
    start = feasible_layout(shop, names, init)
    # Otherwise the check could not tell the two starts apart.
    assert start != feasible_layout(shop, names,
                                    popularity_rank(ss, seed=SEED))

    evaluated = []
    real = shop._ga_fitness

    def _logged(chrom, *args, **kwargs):
        evaluated.append(chromosome_to_layout(chrom, names))
        return real(chrom, *args, **kwargs)

    shop._ga_fitness = _logged
    runs = (
        lambda: run_ga_headless(shop, names, bp, pop_size=4, n_gens=2,
                                rng_seed=SEED, init_layout=init, **MC),
        lambda: random_search(shop, ss, names, bp, seed=SEED, budget=8,
                              block=4, init_layout=init, **MC),
        lambda: simulated_annealing(shop, ss, names, bp, seed=SEED,
                                    budget=8, block=4, init_layout=init,
                                    **MC),
    )
    firsts = []
    for run in runs:
        mark = len(evaluated)
        run()
        firsts.append(evaluated[mark])
    assert firsts == [start, start, start]


def test_random_search_scores_init_layout_first(fit_log):
    """With ``init_layout`` the as-built layout is candidate 0, charged to
    the budget, and the ``budget - 1`` draws that follow are the cold
    search's own first draws."""
    ss, shop, names, bp, init = _setup()
    budget, block = 10, 4
    cold_stats = {}
    random_search(shop, ss, names, bp, seed=SEED, budget=budget, block=block,
                  stats=cold_stats, **MC)
    cold = _search_calls(fit_log, SEED, budget, block)
    del fit_log[:]

    stats = {}
    random_search(shop, ss, names, bp, seed=SEED, budget=budget, block=block,
                  stats=stats, init_layout=init, **MC)
    warm = _search_calls(fit_log, SEED, budget, block)

    assert warm[0] == (feasible_layout(shop, names, init), SEED * 1000)
    assert len(warm) == budget == stats['n_search_evals']
    assert [lay for lay, _ in warm[1:]] == [lay for lay, _ in cold[:-1]]
    # Candidate k is scored under its block's seed whatever it is.
    assert [s for _, s in warm] == [s for _, s in cold]
    assert stats['rs_start'] == 'asbuilt'
    assert cold_stats['rs_start'] == 'cold'


def test_random_search_sampler_hook(fit_log):
    """A sampler replaces the default draw and is given the search's own
    generator; with one, ``shop_synth`` is not needed. A sampler making the
    default draw reproduces the default run exactly."""
    ss, shop, names, bp, init = _setup()
    kw = dict(seed=SEED, budget=10, block=4, init_layout=init, **MC)
    ref_stats = {}
    ref = random_search(shop, ss, names, bp, stats=ref_stats, **kw)
    ref_log = list(fit_log)
    del fit_log[:]

    seen = []

    def default_draw(rng):
        assert isinstance(rng, np.random.Generator)
        seen.append(1)
        return random_valid(ss, seed=int(rng.integers(0, 2 ** 31 - 1)))

    stats = {}
    out = random_search(shop, None, names, bp, sampler=default_draw,
                        stats=stats, **kw)
    assert len(seen) == kw['budget'] - 1
    assert out == ref and fit_log == ref_log and stats == ref_stats

    # What the sampler returns is what gets evaluated, after the repair.
    del fit_log[:]
    fixed = perimeter_only(ss, seed=0)
    random_search(shop, None, names, bp, sampler=lambda rng: fixed, **kw)
    search = _search_calls(fit_log, SEED, kw['budget'], kw['block'])
    assert all(lay == feasible_layout(shop, names, fixed)
               for lay, _ in search[1:])

    with pytest.raises(ValueError):
        random_search(shop, None, names, bp, **kw)


def test_sa_asbuilt_starts_at_init_layout(fit_log):
    ss, shop, names, bp, init = _setup()
    stats = {}
    simulated_annealing(shop, ss, names, bp, seed=SEED, budget=12, block=4,
                        stats=stats, init_layout=init, **MC)
    assert fit_log[0] == (feasible_layout(shop, names, init), SEED * 1000)
    assert stats['sa_start'] == 'asbuilt'
    assert stats['n_search_evals'] == 12


def test_sa_start_arguments_are_checked():
    ss, shop, names, bp, init = _setup()
    kw = dict(seed=SEED, budget=8, block=4, **MC)
    with pytest.raises(ValueError):          # as-built needs the layout
        simulated_annealing(shop, ss, names, bp, **kw)
    with pytest.raises(ValueError):
        simulated_annealing(shop, ss, names, bp, init_layout=init,
                            start='greedy', **kw)
    with pytest.raises(ValueError):          # popularity needs the shop
        simulated_annealing(shop, None, names, bp, start='popularity',
                            neighbor=lambda lay, rng: lay, **kw)
    with pytest.raises(ValueError):          # the default move needs it too
        simulated_annealing(shop, None, names, bp, init_layout=init, **kw)


def test_sa_popularity_start_reproduces_previous_behaviour(fit_log):
    """``start='popularity'`` is the former warm start: the same run as an
    as-built start handed the popularity_rank layout explicitly, and it
    does not read ``init_layout``."""
    ss, shop, names, bp, init = _setup()
    kw = dict(seed=SEED, budget=12, block=4, **MC)
    pop_stats = {}
    out = simulated_annealing(shop, ss, names, bp, start='popularity',
                              stats=pop_stats, **kw)
    pop_log = list(fit_log)
    del fit_log[:]

    explicit = popularity_rank(ss, seed=SEED)
    exp_stats = {}
    ref = simulated_annealing(shop, ss, names, bp, start='asbuilt',
                              init_layout=explicit, stats=exp_stats, **kw)
    assert out == ref and fit_log == pop_log
    assert pop_log[0] == (feasible_layout(shop, names, explicit), SEED * 1000)
    assert pop_stats.pop('sa_start') == 'popularity'
    assert exp_stats.pop('sa_start') == 'asbuilt'
    assert pop_stats == exp_stats

    assert simulated_annealing(shop, ss, names, bp, start='popularity',
                               init_layout=init, **kw) == out


def test_sa_neighbor_hook(fit_log):
    """A neighbour function replaces the default move and is given the
    annealer's own generator; with one and an as-built start,
    ``shop_synth`` is not needed."""
    ss, shop, names, bp, init = _setup()
    budget, block = 12, 4
    kw = dict(seed=SEED, budget=budget, block=block, init_layout=init, **MC)
    ref_stats = {}
    ref = simulated_annealing(shop, ss, names, bp, stats=ref_stats, **kw)
    ref_log = list(fit_log)
    del fit_log[:]

    calls = []

    def default_move(layout, rng):
        assert isinstance(rng, np.random.Generator)
        calls.append(1)
        return _neighbor(ss, layout, rng)

    stats = {}
    out = simulated_annealing(shop, None, names, bp, neighbor=default_move,
                              stats=stats, **kw)
    # One proposal per search evaluation, except the start and the
    # incumbent re-score that opens every later block.
    assert len(calls) == budget - mh._n_blocks(budget, block)
    assert out == ref and fit_log == ref_log and stats == ref_stats

    # What the neighbour returns is what gets evaluated: a move that never
    # moves leaves the annealer at its start.
    del fit_log[:]
    start = feasible_layout(shop, names, init)
    out = simulated_annealing(shop, None, names, bp,
                              neighbor=lambda lay, rng: dict(lay), **kw)
    assert out == start
    assert all(lay == start for lay, _ in fit_log)


def _run(config, ss, shop, names, bp, init, budget, block, stats):
    """One search in one configuration of start and hooks."""
    kw = dict(seed=SEED, budget=budget, block=block, stats=stats, **MC)
    if config == 'rs_cold':
        random_search(shop, ss, names, bp, **kw)
    elif config == 'rs_asbuilt':
        random_search(shop, ss, names, bp, init_layout=init, **kw)
    elif config == 'rs_sampler_cold':
        random_search(shop, None, names, bp, **kw,
                      sampler=lambda rng: random_valid(
                          ss, seed=int(rng.integers(0, 2 ** 31 - 1))))
    elif config == 'rs_sampler_asbuilt':
        random_search(shop, None, names, bp, init_layout=init, **kw,
                      sampler=lambda rng: perimeter_only(ss, seed=0))
    elif config == 'sa_asbuilt':
        simulated_annealing(shop, ss, names, bp, init_layout=init, **kw)
    elif config == 'sa_popularity':
        simulated_annealing(shop, ss, names, bp, start='popularity', **kw)
    elif config == 'sa_neighbor_asbuilt':
        simulated_annealing(shop, None, names, bp, init_layout=init, **kw,
                            neighbor=lambda lay, rng: _neighbor(ss, lay, rng))
    else:
        raise AssertionError(config)


@pytest.mark.parametrize('budget,block', [(12, 4), (10, 4)])
@pytest.mark.parametrize('config', [
    'rs_cold', 'rs_asbuilt', 'rs_sampler_cold', 'rs_sampler_asbuilt',
    'sa_asbuilt', 'sa_popularity', 'sa_neighbor_asbuilt'])
def test_search_evaluations_equal_budget(fit_log, config, budget, block):
    """Whatever the start and hooks, a search spends exactly ``budget``
    search evaluations, and the counts it reports are the calls made --
    including a budget that does not fill its last block."""
    ss, shop, names, bp, init = _setup()
    stats = {}
    _run(config, ss, shop, names, bp, init, budget, block, stats)
    assert len(_search_calls(fit_log, SEED, budget, block)) == budget
    assert stats['n_search_evals'] == budget
    assert stats['n_evals'] == len(fit_log)
    assert stats['n_evals'] == stats['n_search_evals'] + stats['n_final_evals']


@pytest.mark.parametrize('config', ['rs_asbuilt', 'sa_asbuilt',
                                    'sa_popularity'])
def test_final_selection_scores_as_many_candidates_as_the_ga(fit_log,
                                                             config):
    """Every search re-scores exactly ``block`` candidates under the
    selection seeds, as the GA re-scores its population of ``block``."""
    ss, shop, names, bp, init = _setup()
    budget, block = 12, 4
    stats = {}
    _run(config, ss, shop, names, bp, init, budget, block, stats)
    assert stats['n_final_evals'] == block * 5
    ga = run_ga_headless(shop, names, bp, pop_size=block, n_gens=3,
                         rng_seed=SEED, init_layout=init, **MC)
    assert ga['n_final_evals'] == stats['n_final_evals']


def test_block_rank_pool_ignores_what_each_block_seed_did():
    """The annealer's pool compares states only within their block: any
    strictly increasing change of one block's scores -- a kinder or harsher
    seed -- leaves it unchanged. It takes every block's best first, the
    latest block first."""
    rng = np.random.default_rng(0)
    entries = []
    for k in range(40):
        b = k // 8
        key = (int(rng.integers(0, 15)),)
        entries.append((b, key, float(rng.normal(100, 5)), {'k': key}, k))
    pool, src = mh._pool_by_block_rank(entries, 10)
    shift = {b: (3.0 * b + 1.0, 50.0 * b) for b in range(5)}
    moved = [(b, key, f * shift[b][0] + shift[b][1], lay, k)
             for b, key, f, lay, k in entries]
    pool2, src2 = mh._pool_by_block_rank(moved, 10)
    assert pool == pool2 and src == src2
    assert src[:5] == [4, 3, 2, 1, 0]
    assert len({tuple(p['k']) for p in pool}) == len(pool)


def test_random_search_pool_ignores_a_multiplicative_block_seed(monkeypatch):
    """Random search ranks a candidate by its score over its block's median
    under the block's seed, so a seed that scales every score in its block
    -- the way the day-level noise acts on revenue -- cannot promote or
    demote a candidate: the search returns the same layout."""
    ss, shop, names, bp, init = _setup()
    real = mh._fit
    runs = {}
    for label, factor in (('plain', lambda s: 1.0),
                          ('scaled', lambda s: 1.0 + 0.05 * (s % 7))):
        def _fit(shop_, item_names, layout, base_params, seed, mc_iters,
                 mc_days, _factor=factor):
            v = real(shop_, item_names, layout, base_params, 5, mc_iters,
                     mc_days)
            # search seeds are scaled; the final-selection seeds are not
            first_final = SEED * 1000 + mh._n_blocks(12, 4) + 1
            return v * (_factor(seed) if seed < first_final else 1.0)
        monkeypatch.setattr(mh, '_fit', _fit)
        runs[label] = random_search(shop, ss, names, bp, seed=SEED, budget=12,
                                    block=4, init_layout=init, **MC)
    assert runs['plain'] == runs['scaled']


def test_searches_report_their_traces():
    ss, shop, names, bp, init = _setup()
    rs, sa = {}, {}
    random_search(shop, ss, names, bp, seed=SEED, budget=10, block=4,
                  init_layout=init, stats=rs, **MC)
    simulated_annealing(shop, ss, names, bp, seed=SEED, budget=10, block=4,
                        init_layout=init, stats=sa, **MC)
    ga = run_ga_headless(shop, names, bp, pop_size=4, n_gens=3,
                         rng_seed=SEED, init_layout=init, **MC)
    assert [t['block'] for t in rs['trace']] == [0, 1, 2]
    assert sum(t['n'] for t in rs['trace']) == 10
    assert [t['block'] for t in sa['trace']] == [0, 1, 2]
    assert sum(t['n'] for t in sa['trace']) == 10
    assert sum(t['proposals'] for t in sa['trace']) == 10 - 3
    assert [t['gen'] for t in ga['trace']] == [0, 1, 2]
    assert [t['best'] for t in ga['trace']] == ga['history_best']


def test_sa_schedule_arguments():
    """The final temperature and the default move's step are settable (the
    annealer sweep varies them) and checked; the defaults are the run the
    comparison always made."""
    ss, shop, names, bp, init = _setup()
    kw = dict(seed=SEED, budget=12, block=4, init_layout=init, **MC)
    ref = simulated_annealing(shop, ss, names, bp, **kw)
    same = simulated_annealing(shop, ss, names, bp,
                               final_temp_frac=mh.DEFAULT_FINAL_TEMP_FRAC,
                               step_frac=mh.DEFAULT_STEP_FRAC, **kw)
    assert ref == same
    for bad in (dict(final_temp_frac=0.0), dict(final_temp_frac=1.5),
                dict(step_frac=0.0)):
        with pytest.raises(ValueError):
            simulated_annealing(shop, ss, names, bp, **bad, **kw)


def test_popstart_row_is_outside_the_family_and_budget_set():
    """Figure B scores the popularity-started annealer beside the
    comparison, never inside it: it must not enlarge the family the
    headline intervals are corrected over, nor join the searches held to
    one budget and one start."""
    from experiments import run_baseline_comparison as rbc
    assert 'simulated_annealing_popstart' in rbc.METHODS
    assert rbc.SENSITIVITY_METHODS == ['simulated_annealing_popstart']
    assert 'simulated_annealing_popstart' not in rbc.COMPARISON_FAMILY
    assert 'GA' not in rbc.COMPARISON_FAMILY
    assert set(rbc.COMPARISON_FAMILY) == {
        'random_valid', 'perimeter_only', 'popularity_rank', 'greedy_swap',
        'random_search', 'simulated_annealing', 'oracle'}
    assert rbc.EQUAL_BUDGET_METHODS == ('GA', 'random_search',
                                        'simulated_annealing')
