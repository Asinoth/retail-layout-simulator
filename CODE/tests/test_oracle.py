"""The analytical reference solver: independence, failures, convergence.

``oracle.py`` must stay independent of the simulator and of the GA -- the
point of Figure A is that the GA is judged by a model it never saw -- so the
import check runs in a fresh interpreter, where no other test can have
imported the banned modules first. A restart that fails is recorded with
its reason, a solve in which every restart fails raises instead of handing
back a fallback layout, and the best value's growth with the number of
restarts is kept so a reader can see whether the reference had converged.
"""

import os
import subprocess
import sys

import numpy as np
import pytest

import oracle
from synthetic_shops import analytical_revenue, generate_synthetic_shop

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: What the reference solver must never load, directly or transitively.
BANNED = ('viz_ga', 'viz_ga_run', 'viz_optimize', 'viz_optimize_helpers',
          'viz_optimize_results', 'simulation', 'customer',
          'customer_pathfinding', 'sim_calibration', 'layout_objective')


def test_oracle_imports_neither_the_simulator_nor_the_ga():
    code = ("import sys; sys.path.insert(0, sys.argv[1]); import oracle; "
            f"banned = set({BANNED!r}); "
            "leaked = sorted(banned & set(sys.modules)); "
            "print(','.join(leaked)); sys.exit(1 if leaked else 0)")
    res = subprocess.run([sys.executable, '-c', code, CODE_DIR],
                         capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, (f"oracle.py transitively imports: "
                                 f"{res.stdout.strip()} {res.stderr[-500:]}")


def _shop(seed=10_003):
    return generate_synthetic_shop('oracle_test', seed=seed, n_items=10,
                                   width=12.0, height=10.0)


def test_best_curve_is_the_running_best_of_the_restarts():
    shop = _shop()
    res = oracle.solve_oracle(shop, n_restarts=8)
    assert res.n_restarts == 8 and res.n_failed == 0 and not res.failures
    assert len(res.restart_values) == len(res.best_curve) == 8
    run = np.maximum.accumulate(np.asarray(res.restart_values, dtype=float))
    assert np.allclose(res.best_curve, run)
    assert res.best_curve[-1] == max(res.restart_values)
    # The layout's revenue is the best restart's, after contact separation
    # (which moves flush neighbours apart and costs a hair of revenue).
    assert abs(res.true_revenue - analytical_revenue(shop, res.layout)) < 1e-9
    assert res.true_revenue == pytest.approx(res.best_curve[-1], rel=1e-4)


def test_failed_restarts_are_recorded_not_skipped(monkeypatch):
    real = oracle.minimize
    calls = {'n': 0}

    def flaky(*args, **kwargs):
        calls['n'] += 1
        if calls['n'] in (1, 3):
            raise FloatingPointError('boom')
        return real(*args, **kwargs)

    monkeypatch.setattr(oracle, 'minimize', flaky)
    res = oracle.solve_oracle(_shop(), n_restarts=5)
    assert res.n_failed == 2
    assert [k for k, _ in res.failures] == [0, 2]
    assert all('FloatingPointError: boom' in r for _, r in res.failures)
    assert res.restart_values[0] is None and res.restart_values[2] is None
    assert res.best_curve[0] is None and res.best_curve[1] is not None


def test_a_solve_whose_every_restart_fails_raises(monkeypatch):
    def broken(*args, **kwargs):
        raise ValueError('no')

    monkeypatch.setattr(oracle, 'minimize', broken)
    with pytest.raises(oracle.OracleFailure):
        oracle.solve_oracle(_shop(), n_restarts=3)
    with pytest.raises(ValueError):
        oracle.solve_oracle(_shop(), n_restarts=0)


@pytest.mark.parametrize('module', ['run_synthetic_gt',
                                    'run_baseline_comparison',
                                    'run_elasticity_lhs',
                                    'run_objective_alignment'])
def test_every_runner_gives_the_reference_the_same_restarts(module,
                                                            monkeypatch):
    """No comparison is made against a weaker reference than another:
    Figures A and B, the elasticity LHS and the objective alignment all
    default to ``ORACLE_RESTARTS``."""
    import importlib
    from experiments._common import ORACLE_RESTARTS
    mod = importlib.import_module(f'experiments.{module}')
    monkeypatch.setattr(sys, 'argv', [module])
    assert mod.parse_args().oracle_restarts == ORACLE_RESTARTS == 96


def test_oracle_record_and_summary(monkeypatch):
    from experiments._common import oracle_record, oracle_summary
    real = oracle.minimize
    calls = {'n': 0}

    def flaky(*args, **kwargs):
        calls['n'] += 1
        if calls['n'] == 2:
            raise FloatingPointError('boom')
        return real(*args, **kwargs)

    monkeypatch.setattr(oracle, 'minimize', flaky)
    rec = oracle_record(oracle.solve_oracle(_shop(), n_restarts=3))
    assert rec['n_restarts'] == 3 and rec['n_failed_restarts'] == 1
    assert rec['failures'][0][0] == 1
    summ = oracle_summary([rec, dict(rec, n_failed_restarts=0,
                                     converged=False)], 3)
    assert summ == {'n_restarts': 3, 'n_failed_restarts': 1,
                    'n_not_converged': 1 + (not rec['converged']),
                    'n_scenarios': 2}


def test_lhs_solves_and_records_the_reference_at_figure_as_restarts(
        monkeypatch, tmp_path):
    """The elasticity LHS runner, run end to end at its defaults, hands the
    analytical reference exactly Figure A's restart count (96, the
    ``run_synthetic_gt`` default) for every scenario and records it in its
    summary, so its 'oracle' comparisons are made against the same
    reference as Figures A and B."""
    import glob
    import json
    import importlib
    from experiments._common import ORACLE_RESTARTS
    lhs = importlib.import_module('experiments.run_elasticity_lhs')
    gt = importlib.import_module('experiments.run_synthetic_gt')
    monkeypatch.setattr(sys, 'argv', ['run_synthetic_gt'])
    figure_a_restarts = gt.parse_args().oracle_restarts
    seen = []
    real = lhs.solve_oracle

    def _spy(shop, n_restarts):
        seen.append(n_restarts)
        return real(shop, n_restarts=n_restarts)
    monkeypatch.setattr(lhs, 'solve_oracle', _spy)
    monkeypatch.setattr(sys, 'argv', [
        'run_elasticity_lhs', '--n-scenarios', '2', '--n-seeds', '1',
        '--n-draws', '4', '--n-gens', '2', '--pop-size', '4',
        '--mc-iters', '20', '--mc-days', '3', '--n-weight-draws', '2',
        '--n-spread-layouts', '2', '--out-root', str(tmp_path)])
    lhs.main()
    assert seen == [figure_a_restarts] * 2
    assert figure_a_restarts == ORACLE_RESTARTS == 96
    (path,) = glob.glob(str(tmp_path / 'elasticity_lhs_*' / 'summary.json'))
    with open(path) as f:
        summary = json.load(f)
    assert summary['oracle']['n_restarts'] == figure_a_restarts
    assert summary['oracle']['n_scenarios'] == 2
