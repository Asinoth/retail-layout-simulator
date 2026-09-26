"""The structural sweep's exit-route check (run_structural_exit_check), on
hand-made exit logs.

The check re-runs replications of the committed sweep through the runner's
own ``run_one`` with an observe-only hook, so the parts tested here are the
hook, the choice of the window's exits, the classification of an exit, the
agreement with the simulation's own tally and the matching of agents across
settings: two settings' agents are the same arrival when their spawn times
agree (as a multiset: two arrivals can share a tick) and carry the same
draws only when their abandonment probability, a continuous draw, agrees
too.
"""
import types

import pytest

from experiments import run_structural_exit_check as X
from experiments import run_structural_sensitivity as RS


def _exit(t, paid=True, drawn=False, p=0.03, route=None, stalls=0,
          teleports=0):
    return {'paid': paid, 'route': route, 'abandon_drawn': drawn, 'p': p,
            'spawn_t': t, 'exit_t': t + 100.0, 'moving_stalls': stalls,
            'exit_stall_teleports': teleports, 'list_len': 3, 'visited': 3,
            'dropped': 0}


def test_window_exits_are_the_last_counted_exits():
    log = [_exit(1.0), _exit(2.0, paid=False, drawn=True,
                             route='abandon_draw'),
           _exit(3.0), _exit(4.0, paid=False, drawn=True,
                             route='abandon_draw'), _exit(5.0)]
    win = X.window_exits(log, completed=2, abandoned=1)
    assert [e['spawn_t'] for e in win] == [3.0, 4.0, 5.0]
    assert X.window_exits(log, 0, 0) == []
    with pytest.raises(RuntimeError):      # more counted than logged
        X.window_exits(log, completed=5, abandoned=1)
    with pytest.raises(RuntimeError):      # the split does not match
        X.window_exits(log, completed=1, abandoned=2)


def test_exits_by_route_draw_and_stall():
    win = [_exit(1.0, p=0.02),
           _exit(2.0, paid=False, drawn=True, p=0.04, route='abandon_draw'),
           _exit(3.0, paid=False, drawn=True, p=0.05, route='abandon_draw',
                 stalls=2),
           _exit(4.0, paid=False, p=0.01, route='no_route'),
           _exit(5.0, paid=False, p=0.01),                  # no route set
           _exit(6.0, drawn=True, p=0.05, stalls=1, teleports=1)]
    s = X.summarise_exits(win)
    assert (s['n_exits'], s['paid'], s['unpaid']) == (6, 2, 4)
    assert s['unpaid_by_route'] == {'abandon_draw': 2, 'no_route': 1,
                                    'other': 1}
    assert s['abandon_drawn'] == 3
    assert s['unpaid_without_draw'] == 2
    assert s['drawn_but_paid'] == 1
    assert (s['moving_stalls'], s['stalled_exits'], s['stalled_unpaid'],
            s['stalled_unpaid_drawn'], s['exit_stall_teleports']) == \
        (3, 2, 1, 1, 1)
    assert s['expected_unpaid'] == pytest.approx(0.18)
    assert s['var_unpaid'] == pytest.approx(
        sum(p * (1 - p) for p in (0.02, 0.04, 0.05, 0.01, 0.01, 0.05)))


def test_tally_agrees_with_the_log():
    win = [_exit(1.0), _exit(2.0, paid=False, drawn=True,
                             route='abandon_draw', stalls=1)]
    s = X.summarise_exits(win)
    routes = {'paid': 1, 'unpaid_abandon_draw': 1, 'unpaid_abandon_drawn': 1,
              'abandon_drawn': 1, 'moving_stalls': 1,
              'exits_after_moving_stall': 1, 'exit_stall_teleports': 0}
    assert X.tally_agrees(routes, s)
    assert not X.tally_agrees(dict(routes, unpaid_no_route=1), s)
    assert not X.tally_agrees(dict(routes, paid=2), s)


def test_agents_are_matched_by_spawn_time_and_draws_by_probability():
    ref = [_exit(1.0, p=0.011), _exit(2.0, p=0.022),
           _exit(3.0, paid=False, drawn=True, p=0.033, route='abandon_draw'),
           _exit(4.0, p=0.044), _exit(4.0, p=0.045)]      # a shared tick
    other = [_exit(1.0, p=0.011),                          # same draws
             _exit(2.0, paid=False, drawn=True, p=0.029,
                   route='abandon_draw'),                  # other draws
             _exit(3.0, paid=False, drawn=True, p=0.047,
                   route='abandon_draw'),                  # both abandon
             _exit(4.0, p=0.045),                          # one of the tick
             _exit(9.0, p=0.02)]                           # not in ref
    s = X.shared_draws(other, ref)
    assert (s['n_exits'], s['n_exits_reference']) == (5, 5)
    assert s['n_common'] == 4
    assert s['n_same_draw'] == 2                   # t = 1.0 and 4.0 / 0.045
    assert s['n_same_draw_abandoning'] == 0
    assert s['n_both_abandon'] == 1
    assert s['n_either_abandon'] == 2
    assert s['n_spawn_time_ties'] == 0
    assert X.shared_draws(ref, other)['n_spawn_time_ties'] == 1


def test_summary_pools_runs_and_pairs_against_the_default():
    struct = {'strengths': [0.0, 0.08], 'default_strength': 0.08}
    wins = {(0.0, 7): [_exit(1.0, p=0.01),
                       _exit(2.0, paid=False, drawn=True, p=0.2,
                             route='abandon_draw')],
            (0.08, 7): [_exit(1.0, p=0.01), _exit(2.0, p=0.3)]}
    runs = []
    for (s, seed), win in wins.items():
        rec = {'strength': s, 'seed': seed, 'relocations': 0,
               'matches_artifact': True, **X.summarise_exits(win)}
        rec['tally_agrees'] = True
        runs.append(rec)
    out = X.summarise(struct, runs, wins)
    t = out['total']
    assert (t['n_runs'], t['n_exits'], t['unpaid']) == (2, 4, 1)
    assert t['all_reproduce'] and t['all_tallies_agree']
    assert t['expected_unpaid'] == pytest.approx(0.52)
    assert t['z_unpaid'] == pytest.approx(
        (1 - 0.52) / (0.01 * 0.99 * 2 + 0.2 * 0.8 + 0.3 * 0.7) ** 0.5)
    assert set(out['per_setting']) == {'0.0', '0.08'}
    assert out['per_setting']['0.0']['unpaid_by_route']['abandon_draw'] == 1
    assert [(p['strength'], p['seed']) for p in out['sharing']['pairs']] == \
        [(0.0, 7)]
    assert out['sharing']['per_setting']['0.0']['n_common'] == 2
    assert out['sharing']['total']['n_same_draw'] == 1


class _FakeSim:
    def __init__(self):
        self.sim_time = 12.5
        self.customers = [types.SimpleNamespace(position=[1.0, 1.0])]
        self.processed = []

    def _process_customer_exit(self, cust):
        self.processed.append(cust)
        return 'done'

    def _free_trapped_customers(self):
        self.customers[0].position = [2.0, 1.0]


def test_the_hook_only_observes():
    sim, log, reloc = _FakeSim(), [], [0]
    X.instrument(log, reloc)(sim)
    cust = types.SimpleNamespace(
        has_checked_out=False, exit_route='abandon_draw', abandon_cart=True,
        abandon_probability=0.04, spawn_sim_time=3.0, moving_stalls=0,
        exit_stall_teleports=0, shopping_list=['a'], visited_items={'a'},
        abandoned_items=set())
    assert sim._process_customer_exit(cust) == 'done'   # still processed
    assert sim.processed == [cust]
    assert log == [{'paid': False, 'route': 'abandon_draw',
                    'abandon_drawn': True, 'p': 0.04, 'spawn_t': 3.0,
                    'exit_t': 12.5, 'moving_stalls': 0,
                    'exit_stall_teleports': 0, 'list_len': 1, 'visited': 1,
                    'dropped': 0}]
    sim._free_trapped_customers()
    assert reloc == [1]


def test_run_one_hands_the_hook_the_simulation_before_it_runs(monkeypatch):
    """The sweep's own run function calls the hook once, with the
    simulation it is about to run, after the setting is applied."""
    seen = []

    class Sim:
        analytics = {}
        suppressed_errors = {}

        def run_headless(self, duration, dt, seed, callback,
                         callback_every_s):
            seen.append(('run', self.separation_strength))
            callback(self)

    shop = types.SimpleNamespace(customer_simulation=Sim())
    monkeypatch.setattr(RS.LS, 'build_live_store', lambda params: shop)
    out = RS.run_one(None, 0.16, 10.0, 0.27, 45, warmup=5.0, dt=0.04,
                     seed=1, instrument=lambda sim: seen.append(
                         ('hook', sim.separation_strength)))
    assert seen == [('hook', 0.16), ('run', 0.16)]
    assert out['completed'] == 0 and out['strength'] == 0.16
    seen.clear()
    RS.run_one(None, 0.08, 10.0, 0.27, 45, warmup=5.0, seed=1)
    assert seen == [('run', 0.08)]
