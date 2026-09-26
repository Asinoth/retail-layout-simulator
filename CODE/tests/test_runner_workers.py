"""The non-live runners write the same results for any worker count (R52).

``--workers`` spreads scenarios over processes. Each scenario builds its
own shop and seeds every stream it draws from, and the runner puts the
results back in scenario order before aggregating, so one process and two
must produce the same numbers. ``tests/test_live_runner_workers.py`` pins
this for the live diagnostics; this file pins it for Figure A
(``run_synthetic_gt``), Figure B (``run_baseline_comparison``), the
elasticity and weight sweeps (``run_elasticity_lhs``), the MC ground
truth (``run_mc_groundtruth``) and the objective-alignment diagnostic
(``run_objective_alignment``), at tiny settings and at most two worker
processes. Figure C across search seeds is pinned in
``tests/test_real_data_seeds.py``.

What may differ between the two runs, and is removed before comparing:
the worker count itself, the output root, wall-clock times, and the
provenance a sidecar records (checkout state, interpreter, packages,
timestamp). Everything else -- every CSV cell, the summary, and the
sidecar's design record (budgets, seeds, schedules, base parameters) -- is
compared as text.
"""

import csv
import glob
import json
import sys

import pytest

from experiments import (run_baseline_comparison, run_elasticity_lhs,
                         run_mc_groundtruth, run_objective_alignment,
                         run_synthetic_gt)

WORKER_COUNTS = (1, 2)

#: Sidecar and summary keys that describe the run rather than its results.
VOLATILE = {'git_diff_sha256', 'git_dirty', 'git_modified', 'git_sha',
            'git_state_changed_during_run', 'git_untracked_py', 'iso_time',
            'packages', 'platform', 'python', 'wall_seconds', 'out_root'}

#: CSV columns holding wall-clock times.
TIMING_COLUMNS = {'wall_seconds'}


def _strip(obj, workers):
    """``obj`` without the volatile keys; every ``workers`` entry must name
    the worker count the run was given, and is then dropped."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in VOLATILE:
                continue
            if k == 'workers':
                assert v == workers
                continue
            out[k] = _strip(v, workers)
        return out
    if isinstance(obj, list):
        return [_strip(v, workers) for v in obj]
    return obj


def _json_text(path, workers):
    with open(path, encoding='utf-8') as f:
        return json.dumps(_strip(json.load(f), workers), sort_keys=True)


def _csv_text(path):
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    return json.dumps([{k: v for k, v in r.items()
                        if k not in TIMING_COLUMNS} for r in rows])


def _run(monkeypatch, module, argv, root):
    monkeypatch.setattr(sys, 'argv', [module.__name__] + argv
                        + ['--out-root', str(root)])
    assert module.main() in (0, None)
    runs = glob.glob(str(root / '*'))
    assert len(runs) == 1, runs
    return runs[0]


def _outputs(run_dir, workers, csv_names, json_names):
    texts = {}
    for name in csv_names:
        texts[name] = _csv_text(f'{run_dir}/{name}')
    for name in json_names:
        texts[name] = _json_text(f'{run_dir}/{name}', workers)
    return texts


def _assert_invariant(monkeypatch, tmp_path, module, argv, csv_names,
                      json_names):
    per_count = []
    for workers in WORKER_COUNTS:
        run_dir = _run(monkeypatch, module,
                       argv + ['--workers', str(workers)],
                       tmp_path / f'workers{workers}')
        per_count.append(_outputs(run_dir, workers, csv_names, json_names))
    for name in csv_names + json_names:
        assert per_count[0][name] == per_count[1][name], name
    return per_count[0]


@pytest.mark.parametrize('module, argv, csv_names, json_names', [
    pytest.param(
        run_synthetic_gt,
        ['--n-scenarios', '2', '--n-seeds', '2', '--mc-iters', '40',
         '--n-gens', '2', '--pop-size', '6', '--n-spearman-samples', '10'],
        ['results.csv', 'spearman.csv'], ['sidecar.json'],
        id='synthetic_gt'),
    pytest.param(
        run_baseline_comparison,
        ['--n-scenarios', '2', '--n-seeds', '2', '--mc-iters', '40',
         '--n-gens', '2', '--pop-size', '6'],
        ['results.csv'], ['summary.json', 'sidecar.json'],
        id='baseline_comparison'),
    pytest.param(
        run_elasticity_lhs,
        ['--n-scenarios', '2', '--n-seeds', '1', '--n-draws', '8',
         '--n-gens', '2', '--pop-size', '6', '--mc-iters', '40',
         '--mc-days', '7', '--n-weight-draws', '4',
         '--n-spread-layouts', '4'],
        ['results.csv', 'weight_sweep.csv', 'weight_sweep_standardized.csv',
         'weight_sweep_standardized_unit.csv'],
        ['summary.json', 'sidecar.json'],
        id='elasticity_lhs'),
    pytest.param(
        run_mc_groundtruth,
        ['--n-scenarios', '2', '--n-items', '6', '--normal-budget', '30',
         '--big-budget', '60', '--mc-iters', '40', '--mc-days', '7',
         '--confirm-seeds', '2'],
        ['results.csv'], ['summary.json', 'sidecar.json'],
        id='mc_groundtruth'),
    pytest.param(
        run_objective_alignment,
        ['--n-scenarios', '2', '--n-samples', '5', '--n-gens', '2',
         '--pop-size', '6', '--n-restarts', '2'],
        ['results.csv'], ['summary.json', 'sidecar.json'],
        id='objective_alignment'),
])
def test_runner_output_does_not_depend_on_worker_count(
        monkeypatch, tmp_path, module, argv, csv_names, json_names):
    out = _assert_invariant(monkeypatch, tmp_path, module, argv, csv_names,
                            json_names)
    # The comparison is not vacuous: both scenarios produced results.
    rows = json.loads(out['results.csv'])
    assert {r['scenario'] for r in rows} == {'0', '1'}
