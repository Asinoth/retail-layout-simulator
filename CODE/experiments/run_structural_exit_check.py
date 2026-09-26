"""How the structural sweep's unpaid exits come about, checked on
replications of the committed sweep.

The structural sweep (``run_structural_sensitivity``) counts every unpaid
exit of its window as an abandoned cart, so its conversion would depend on
the separation rule it sweeps if an agent could leave unpaid for a reason
that depends on how agents move: a stranded agent sent to the door
(``no_route``), an exiting-state stall that puts it there, a relocation out
of a fixture. An agent also draws, at spawn, whether it will abandon
(``Customer.abandon_cart``), independently of how it moves. The sweep's
artifact predates the exit-route tally, so this re-runs replications of it
-- the same store, seeds, protocol and tick, through the runner's own
``run_one`` -- with every exit logged, and requires each run to reproduce
the artifact's counts and revenue exactly, so what it finds describes
those very exits. Per run it records:

* the window's exits by route: paid, unpaid through the spawn-time draw,
  unpaid with no route left, unpaid otherwise; agents that drew
  abandonment but paid and agents that left unpaid without drawing it;
* the stall detector: moving-state stalls (they drop a list item; the
  agent goes on), exits after one and whether those paid, exiting-state
  door placements, and relocations out of a fixture;
* the unpaid count the exiting agents' own drawn probabilities predict
  (sum of p, variance sum of p(1 - p));
* whether the run's exit-route tally (``sim_analytics``) agrees with the
  per-exit log.

Then, per checked replication, every other setting against the default:
the agents that left in both windows -- matched by spawn time, a multiset
since two arrivals can share a tick -- and how many of them carried the
SAME spawn-time draws in both (an identical abandonment probability, a
continuous draw, and the same abandonment outcome). Common random numbers
synchronize the arrival stream, which the separation rule never draws
from; the agents' own draws come from the global stream, which falls out
of step once the settings' paths diverge. Few identical draws means each
setting's abandoned count is an independent binomial draw.

What this shows is bounded by what it re-ran: ``--reps`` replications per
setting (default the first two of the sweep's five), so a route that never
fired here could still fire in the others.

    python -m experiments.run_structural_exit_check
    python -m experiments.run_structural_exit_check --reps 0,1,2,3,4

Single process; each run takes about as long as one of the sweep's runs
(about half a minute here).
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from experiments import _live_store as LS  # noqa: E402
from experiments import run_structural_sensitivity as RS  # noqa: E402
from experiments._common import (make_run_dir, portable_paths,  # noqa: E402
                                 provenance_snapshot, write_sidecar)

EXPERIMENT = 'structural_exit_check'
STRUCT_PREFIX = 'structural_sensitivity_'
#: Replications of the sweep re-run by default (its first two).
DEFAULT_REPS = (0, 1)
#: A re-run's revenue must equal the artifact's to this (currency units).
REVENUE_TOL = 1e-6
#: Spawn times and drawn probabilities are matched at this resolution: the
#: tick is 0.04 s and the probability a continuous draw.
TIME_DIGITS = 6
P_DIGITS = 12


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


def default_struct_dir(results: str) -> Optional[str]:
    """The newest finished structural sweep under ``results``."""
    for d in sorted(glob.glob(os.path.join(results, STRUCT_PREFIX + '*')),
                    reverse=True):
        if (os.path.isfile(os.path.join(d, 'summary.json'))
                and os.path.isfile(os.path.join(d, 'sidecar.json'))):
            return d
    return None


def exit_record(cust, sim) -> Dict[str, Any]:
    """What the check keeps about one exit."""
    return {'paid': bool(cust.has_checked_out),
            'route': cust.exit_route,
            'abandon_drawn': bool(cust.abandon_cart),
            'p': float(cust.abandon_probability),
            'spawn_t': float(cust.spawn_sim_time),
            'exit_t': float(sim.sim_time),
            'moving_stalls': int(cust.moving_stalls),
            'exit_stall_teleports': int(cust.exit_stall_teleports),
            'list_len': len(cust.shopping_list),
            'visited': len(cust.visited_items),
            'dropped': len(cust.abandoned_items)}


def instrument(log: List[Dict[str, Any]], relocations: List[int]):
    """An observer for ``run_one``: logs every exit as the simulation
    processes it and counts agents lifted out of a fixture. It draws no
    random number and changes nothing the run reads."""
    def attach(sim):
        process_exit = sim._process_customer_exit
        free_trapped = sim._free_trapped_customers

        def logged_exit(cust):
            log.append(exit_record(cust, sim))
            return process_exit(cust)

        def counted_free():
            before = [tuple(c.position) for c in sim.customers]
            free_trapped()
            after = [tuple(c.position) for c in sim.customers]
            relocations[0] += sum(a != b for a, b in zip(before, after))

        sim._process_customer_exit = logged_exit
        sim._free_trapped_customers = counted_free
    return attach


def window_exits(log: Sequence[Dict[str, Any]], completed: int,
                 abandoned: int) -> List[Dict[str, Any]]:
    """The exits of the measurement window: the run counts every exit after
    its warm-up snapshot as completed or abandoned, and exits are logged in
    order, so they are the last ``completed + abandoned`` of the log."""
    n = int(completed) + int(abandoned)
    if n > len(log):
        raise RuntimeError(f"the run counted {n} window exits but logged "
                           f"{len(log)}")
    window = list(log[len(log) - n:]) if n else []
    if sum(e['paid'] for e in window) != int(completed):
        raise RuntimeError("the logged window exits do not split into the "
                           "run's completed and abandoned counts")
    return window


def summarise_exits(window: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """One run's window exits by route, the stall detector's actions, and
    the unpaid count the exiting agents' drawn probabilities predict."""
    unpaid = [e for e in window if not e['paid']]
    by_route = Counter(e['route'] or 'other' for e in unpaid)
    stalled = [e for e in window if e['moving_stalls'] > 0]
    return {
        'n_exits': len(window),
        'paid': len(window) - len(unpaid),
        'unpaid': len(unpaid),
        'unpaid_by_route': {'abandon_draw': by_route.get('abandon_draw', 0),
                            'no_route': by_route.get('no_route', 0),
                            'other': sum(v for k, v in by_route.items()
                                         if k not in ('abandon_draw',
                                                      'no_route'))},
        'abandon_drawn': sum(e['abandon_drawn'] for e in window),
        'unpaid_without_draw': sum(not e['abandon_drawn'] for e in unpaid),
        'drawn_but_paid': sum(e['abandon_drawn'] and e['paid']
                              for e in window),
        'moving_stalls': sum(e['moving_stalls'] for e in window),
        'stalled_exits': len(stalled),
        'stalled_unpaid': sum(not e['paid'] for e in stalled),
        'stalled_unpaid_drawn': sum(not e['paid'] and e['abandon_drawn']
                                    for e in stalled),
        'exit_stall_teleports': sum(e['exit_stall_teleports']
                                    for e in window),
        'expected_unpaid': float(sum(e['p'] for e in window)),
        'var_unpaid': float(sum(e['p'] * (1 - e['p']) for e in window)),
    }


def tally_agrees(routes: Dict[str, int], s: Dict[str, Any]) -> bool:
    """The run's own exit-route tally (``sim_analytics``, post-warm-up)
    against the per-exit log."""
    return (int(routes.get('paid', 0)) == s['paid']
            and int(routes.get('unpaid_abandon_draw', 0))
            == s['unpaid_by_route']['abandon_draw']
            and int(routes.get('unpaid_no_route', 0))
            == s['unpaid_by_route']['no_route']
            and int(routes.get('unpaid_other', 0))
            == s['unpaid_by_route']['other']
            and int(routes.get('abandon_drawn', 0)) == s['abandon_drawn']
            and int(routes.get('moving_stalls', 0)) == s['moving_stalls']
            and int(routes.get('exits_after_moving_stall', 0))
            == s['stalled_exits']
            and int(routes.get('exit_stall_teleports', 0))
            == s['exit_stall_teleports'])


def _time_key(e):
    return round(float(e['spawn_t']), TIME_DIGITS)


def _draw_key(e):
    return (_time_key(e), round(float(e['p']), P_DIGITS),
            bool(e['abandon_drawn']))


def shared_draws(window: Sequence[Dict[str, Any]],
                 reference: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    """Two settings' window exits under one seed: how many agents left in
    both windows (spawn times matched as a multiset), how many of those
    carried the same spawn-time draws in both (the same abandonment
    probability and outcome), how many of those drew abandonment, and how
    many of the common agents drew abandonment in both or in either."""
    ta, tb = Counter(map(_time_key, window)), Counter(map(_time_key,
                                                          reference))
    common = sum((ta & tb).values())
    same = Counter(map(_draw_key, window)) & Counter(map(_draw_key,
                                                         reference))
    ab_a = Counter(_time_key(e) for e in window if e['abandon_drawn'])
    ab_b = Counter(_time_key(e) for e in reference if e['abandon_drawn'])
    both = sum((ab_a & ab_b).values())
    either = (sum((ab_a & tb).values()) + sum((ab_b & ta).values())
              - both)
    return {'n_exits': len(window), 'n_exits_reference': len(reference),
            'n_common': common,
            'n_same_draw': sum(same.values()),
            'n_same_draw_abandoning': sum(v for k, v in same.items()
                                          if k[2]),
            'n_both_abandon': both,
            'n_either_abandon': either,
            'n_spawn_time_ties': sum(v - 1 for v in ta.values() if v > 1)}


def _totals(runs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    keys = ('n_exits', 'paid', 'unpaid', 'abandon_drawn',
            'unpaid_without_draw', 'drawn_but_paid', 'moving_stalls',
            'stalled_exits', 'stalled_unpaid', 'stalled_unpaid_drawn',
            'exit_stall_teleports', 'relocations')
    out = {k: int(sum(r[k] for r in runs)) for k in keys}
    out['unpaid_by_route'] = {
        k: int(sum(r['unpaid_by_route'][k] for r in runs))
        for k in ('abandon_draw', 'no_route', 'other')}
    exp = float(sum(r['expected_unpaid'] for r in runs))
    var = float(sum(r['var_unpaid'] for r in runs))
    out.update({'expected_unpaid': exp, 'sd_unpaid': var ** 0.5,
                'z_unpaid': ((out['unpaid'] - exp) / var ** 0.5
                             if var > 0 else None)})
    return out


def summarise(struct: Dict[str, Any], runs: Sequence[Dict[str, Any]],
              windows: Dict[tuple, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """The check's summary from its per-run records and window exits."""
    default = float(struct['default_strength'])
    strengths = [float(s) for s in struct['strengths']]
    per_setting = {}
    for s in strengths:
        mine = [r for r in runs if float(r['strength']) == s]
        if mine:
            per_setting[str(s)] = _totals(mine)
    pairs = []
    seeds = sorted({int(r['seed']) for r in runs})
    for s in strengths:
        if s == default:
            continue
        for seed in seeds:
            a, b = windows.get((s, seed)), windows.get((default, seed))
            if a is None or b is None:
                continue
            pairs.append({'strength': s, 'seed': seed,
                          **shared_draws(a, b)})
    by_setting = {}
    for s in strengths:
        mine = [p for p in pairs if p['strength'] == s]
        if mine:
            by_setting[str(s)] = {k: int(sum(p[k] for p in mine))
                                  for k in mine[0]
                                  if k not in ('strength', 'seed')}
    total = _totals(runs)
    total.update({'n_runs': len(runs),
                  'all_reproduce': all(r['matches_artifact'] for r in runs),
                  'all_tallies_agree': all(r['tally_agrees'] for r in runs)})
    return {'per_setting': per_setting, 'total': total,
            'sharing': {'reference_strength': default, 'pairs': pairs,
                        'per_setting': by_setting,
                        'total': {k: int(sum(p[k] for p in pairs))
                                  for k in (pairs[0] if pairs else {})
                                  if k not in ('strength', 'seed')}}}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--struct-dir', default=None,
                   help='the structural sweep to check (default: the newest '
                        'finished structural_sensitivity_*)')
    p.add_argument('--reps', default=','.join(map(str, DEFAULT_REPS)),
                   help='comma-separated replications of the sweep to re-run '
                        '(default: %(default)s)')
    p.add_argument('--results', default=os.path.join(_HERE, 'results'))
    p.add_argument('--retail-path', default=None,
                   help='the UCI workbook (default: the sweep\'s own, else '
                        'dataset_paths)')
    p.add_argument('--out-root', default=os.path.join(_HERE, 'results'))
    args = p.parse_args(argv)
    if args.struct_dir is None:
        args.struct_dir = default_struct_dir(args.results)
        if args.struct_dir is None:
            p.error(f'no finished structural sweep under {args.results}')
    try:
        args.reps = sorted({int(x) for x in str(args.reps).split(',')
                            if x.strip()})
    except ValueError:
        p.error('--reps takes comma-separated integers')
    if not args.reps or min(args.reps) < 0:
        p.error('--reps needs at least one replication index >= 0')
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    prov = provenance_snapshot()
    t0 = time.perf_counter()
    struct = json.load(open(os.path.join(args.struct_dir, 'summary.json')))
    side = json.load(open(os.path.join(args.struct_dir, 'sidecar.json')))
    proto = struct['protocol']
    n_reps = int(struct['n_reps'])
    if max(args.reps) >= n_reps:
        raise SystemExit(f"--reps {args.reps}: the sweep has {n_reps} "
                         f"replications")
    ref = {(float(r['strength']), int(run['seed'])): run
           for r in struct['rows'] for run in r['runs']}
    retail = args.retail_path or side.get('args', {}).get('retail_path')
    print('[exit check] calibrating the live store (current period) once...',
          flush=True)
    params = LS.calibrate_live_store(retail, 'current')

    runs, windows, exits = [], {}, {}
    for s in struct['strengths']:
        for rep in args.reps:
            seed = int(proto['seed_base']) + rep
            log, reloc = [], [0]
            t_run = time.perf_counter()
            out = RS.run_one(params, s, float(proto['collect_s']),
                             float(proto['spawn']), int(proto['cap']),
                             warmup=float(proto['warmup_s']),
                             dt=float(proto['dt']), seed=seed,
                             instrument=instrument(log, reloc))
            win = window_exits(log, out['completed'], out['abandoned'])
            a = ref[(float(s), seed)]
            rec = {'strength': float(s), 'rep': rep, 'seed': seed,
                   'completed': out['completed'],
                   'abandoned': out['abandoned'],
                   'customers': out['customers'], 'balked': out['balked'],
                   'revenue': out['revenue'],
                   'matches_artifact': (
                       all(out[k] == a[k] for k in ('completed', 'abandoned',
                                                    'customers', 'balked'))
                       and abs(out['revenue'] - a['revenue']) <= REVENUE_TOL),
                   'exit_routes': out['exit_routes'],
                   'relocations': reloc[0],
                   **summarise_exits(win)}
            rec['tally_agrees'] = tally_agrees(out['exit_routes'], rec)
            runs.append(rec)
            windows[(float(s), seed)] = win
            exits[f'{float(s)}|{seed}'] = win
            print(f"[exit check] s={s} seed={seed}: {rec['completed']} paid, "
                  f"{rec['abandoned']} unpaid (artifact {a['completed']}/"
                  f"{a['abandoned']}), reproduces {rec['matches_artifact']}, "
                  f"unpaid by route {rec['unpaid_by_route']}, stalled exits "
                  f"{rec['stalled_exits']} "
                  f"[{time.perf_counter() - t_run:.0f}s]", flush=True)

    unpaid_sweep = int(sum(int(r['abandoned']) for r in struct['rows']))
    exits_sweep = int(sum(int(r['completed']) + int(r['abandoned'])
                          for r in struct['rows']))
    summary = {
        'experiment': EXPERIMENT,
        'struct': {'run_name': os.path.basename(
                       os.path.normpath(args.struct_dir)),
                   'summary_sha256': _sha256(os.path.join(args.struct_dir,
                                                          'summary.json')),
                   'git_sha': side.get('git_sha'),
                   'n_reps': n_reps,
                   'strengths': [float(s) for s in struct['strengths']],
                   'default_strength': float(struct['default_strength']),
                   'unpaid_total': unpaid_sweep,
                   'exits_total': exits_sweep},
        'protocol': {k: proto[k] for k in ('mode', 'dt', 'seed_base',
                                           'warmup_s', 'collect_s', 'spawn',
                                           'cap', 'design')},
        'reps_checked': list(args.reps),
        'seeds_checked': [int(proto['seed_base']) + r for r in args.reps],
        'runs': runs,
        **summarise(struct, runs, windows),
    }
    out_dir = make_run_dir(args.out_root, EXPERIMENT)
    with open(os.path.join(out_dir, 'exits.json'), 'w', encoding='utf-8',
              newline='\n') as f:
        json.dump(exits, f)
    wall = time.perf_counter() - t0
    write_sidecar(out_dir, {'experiment': EXPERIMENT, 'args': vars(args),
                            'wall_seconds': wall,
                            'struct_dir': args.struct_dir,
                            'summary': summary}, provenance=prov)
    # summary.json last: it marks the check finished.
    with open(os.path.join(out_dir, 'summary.json'), 'w', encoding='utf-8',
              newline='\n') as f:
        json.dump(portable_paths(summary), f, indent=1)
    t = summary['total']
    print(f"[exit check] {t['n_runs']} runs, all reproduce the sweep: "
          f"{t['all_reproduce']}; {t['unpaid']} unpaid exits of the sweep's "
          f"{unpaid_sweep}, by route {t['unpaid_by_route']}; unpaid without "
          f"a draw {t['unpaid_without_draw']}, drawn but paid "
          f"{t['drawn_but_paid']}; expected {t['expected_unpaid']:.1f} "
          f"(z {t['z_unpaid']:+.2f}); same draws vs default "
          f"{summary['sharing']['total'].get('n_same_draw')}/"
          f"{summary['sharing']['total'].get('n_common')} common agents")
    print(f"[exit check] wrote {out_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
