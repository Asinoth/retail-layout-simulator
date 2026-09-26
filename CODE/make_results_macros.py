"""Generate paper_results_macros.tex and paper_table_budgets.tex from the
newest valid experiment artifacts.

Scans CODE/experiments/results/ for the latest run of each kind and emits
LaTeX \\newcommand macros consumed by TOMACS_submission.tex, so the paper
recompiles with fresh numbers after every re-run -- no hand-editing -- and
the experiment-budget table (Table 4) the manuscript inputs.

Every artifact is checked against the design its numbers are quoted at
before it can set a macro:

  * the size of the experiment, and the budgets its runner defines (read
    from the runner's own argparse defaults and module constants, so a
    retuned design moves the check with it);
  * a clean sidecar: one that names the checkout the run started from, on
    a tree with no uncommitted code, that did not move during the run;
  * the model the numbers were made under. Every family that scores
    layouts must record the anchored objective (``base_params`` carrying
    the repaired as-built ``score_anchor``) and the objective constants of
    today's registry, the Monte Carlo spend law included, so no run made
    under an older objective can set a number. Every live family must
    record the stage-2 live protocol (``framing``, the drained cohort, the
    routing null, common random numbers ...) and a store calibrated by
    today's adapter with the anonymous invoices kept.

A family with no run that passes is reported on stderr and left out,
rather than filled in from a run that does not support it; its Table 4 row
reads TBD. A valid run whose macros or Table 4 row cannot be read is a
bug: both files are still written, and the generator exits 1.

Three groups of macros need no artifact: every constant of the registry
(``Reg<Name>``, read from ``retail_literature``'s source, for the
coefficient tables), the release the paper cites (``Release*``, from
CITATION.cff), and the operating constants. The compute statement
(``Hardware*``, ``<Stem>WallHours``, ``<Stem>Workers``, ``WallHoursTotal``)
comes from the sidecars of the runs that set the macros.

    python make_results_macros.py            # writes ../paper_results_macros.tex
                                             # and ../paper_table_budgets.tex
    python make_results_macros.py --out X.tex --table-out Y.tex \\
        --results DIR --allow-missing-core   # a check against another tree
"""

from __future__ import annotations

import argparse
import ast
import csv
import functools
import glob
import hashlib
import json
import math
import operator
import os
import re
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'experiments', 'results')
ROOT = os.path.dirname(HERE) if os.path.basename(HERE).upper() == 'CODE' else HERE
OUT = os.path.join(ROOT, 'paper_results_macros.tex')
TABLE_OUT = os.path.join(ROOT, 'paper_table_budgets.tex')


# Minimum size for an artifact to be allowed to set a paper number.
# Selecting purely by newest timestamp meant any smoke run -- a reviewer
# sanity-checking the code, a debugging pass -- silently replaced the
# paper-grade numbers with a toy on the next regeneration. This has bitten
# twice (audit R25, R39), so the size bar is now enforced rather than
# remembered. The runners' own defaults (``_argparse_defaults``) are the
# paper design; these floors stay as a second line.
MIN_SCENARIOS = 10
MIN_SEEDS = 5

# Tolerance the regret distribution is summarised against, and the level
# the per-scenario rank correlations are called significant at. Both are
# quoted in the paper through macros, so the text cannot state one
# threshold while the number beside it was computed at another.
REGRET_TOL_PCT = 2.0
SPEARMAN_ALPHA = 0.05

# Operating constants the projection engine runs at, read from the
# constants module so the horizon arithmetic in the paper cannot drift
# from the engine's own defaults.
LIT_SCRIPT = 'retail_literature.py'
LIT_CONSTANTS = ('DEFAULT_OP_HOURS_PER_DAY', 'DEFAULT_WEEKEND_MULTIPLIER')

# The runner of each family, read for its argparse defaults and constants.
COMMON_SCRIPT = 'experiments/_common.py'
FIGA_SCRIPT = 'experiments/run_synthetic_gt.py'
FIGB_SCRIPT = 'experiments/run_baseline_comparison.py'
MCGT_SCRIPT = 'experiments/run_mc_groundtruth.py'
GASENS_SCRIPT = 'experiments/run_ga_sensitivity.py'
LHS_SCRIPT = 'experiments/run_elasticity_lhs.py'
ALIGN_SCRIPT = 'experiments/run_objective_alignment.py'
FIGC_SCRIPT = 'experiments/run_real_data_example.py'
SEEDS_SCRIPT = 'experiments/run_real_data_seeds.py'
BUDGET_SCRIPT = 'experiments/run_real_data_budget.py'
INPUT_SCRIPT = 'experiments/run_input_uncertainty.py'
TRANSFER_SCRIPT = 'experiments/run_heldout_transfer.py'
PROTO_SCRIPT = 'experiments/run_live_protocol_study.py'
HEATMAP_SCRIPT = 'experiments/make_heatmap_figure.py'
ADAPTER_SCRIPT = 'dataset_adapters.py'

# The remaining artifact families are held to the design each paper number
# is quoted at. A completion check alone lets a smoke run, or a runner's
# lighter default, replace those macros in the same way.
MIN_FIGC_REPLICATES = 30
MIN_FIGC_MC_ITERS = 2000
MIN_FIGC_SHEETS = 2          # both Online Retail II workbook years
MIN_LHS_POINTS = 256
MIN_LHS_SCENARIOS = 12
# The GA budget behind every LHS layout. The runner's argparse defaults
# are a lighter budget than the published sweep, so a run at the default
# budget would otherwise pass on its point and scenario counts alone.
MIN_LHS_DESIGN = {'n_seeds': 3, 'n_gens': 15, 'pop_size': 24,
                  'mc_iters': 500}
# Spatial weight vectors behind the weight-sweep numbers (the runner's
# default, which the Makefile runs the sweep at).
MIN_LHS_WEIGHT_DRAWS = 256
MIN_MCGT_SCENARIOS = 6
MIN_MCGT_BUDGET_RATIO = 10
MIN_MCGT_NORMAL_BUDGET = 750
MIN_MCGT_MC_ITERS = 1000
MIN_ABM_REPS = 10
MIN_STRUCT_REPS = 5
# Replications per setting the structural sweep's exit-route check must
# re-run (run_structural_exit_check's default: the sweep's first two).
MIN_STRUCT_CHECK_REPS = 2
MIN_QUEUE_REPS = 5
MIN_GOF_REPS = 5

# The equal-budget searches are compared from one starting layout, the
# as-built one. Runs from before the searches shared that start handed the
# annealer the popularity ranking instead, so a difference between the
# searches measured the start as much as the search; those runs do not
# record 'search_start' and cannot set these numbers.
ASBUILT = 'asbuilt'
FIGB_ASBUILT_SEARCHES = ('random_search', 'simulated_annealing')
MCGT_ASBUILT_SEARCHES = ('random_search_big', 'simulated_annealing_big',
                         'GA_big')
# The ground truth's best-known reference also anneals from the popularity
# ranking (review R50): a reference that lacks it is weaker, and its
# regrets smaller, than the one the paper describes.
MCGT_POPSTART = 'simulated_annealing_big_popstart'
POPULARITY = 'popularity'

# Figure B's comparator family and the short names its macros use. The
# Bonferroni divisor counts the members an artifact contains, so a run
# whose recorded family holds a method without a name here would be
# corrected for fewer comparisons than it made, and is refused.
FIGB_SHORT_NAMES = {'oracle': 'Oracle', 'popularity_rank': 'Pop',
                    'perimeter_only': 'Perim', 'random_valid': 'Random',
                    'greedy_swap': 'Greedy', 'random_search': 'RandSearch',
                    'simulated_annealing': 'SA'}
# Rows scored beside the family but outside it: reported on the family's
# estimator and level, never counted in NComparators or NSigComparisons.
FIGB_SENSITIVITY = {'simulated_annealing_popstart': 'SAPopStart'}
# The interval schemes Figure B's runner reports beside the manuscript's
# scenario cluster bootstrap (review R28, R63), and their macro infixes.
FIGB_SCHEMES = (('crossed_bootstrap', 'Crossed'),
                ('twoway_cluster_t', 'TwoWay'),
                ('g1_t', 'GOne'))

# Equivalence margin of the GA-vs-SA comparison, as a fraction of the
# annealer's mean paired revenue. Fixed before the re-run; the reasoning is
# at the computation in figb_macros().
EQUIV_MARGIN_FRAC = 0.001

# Figure C's equal-budget comparators: short name, the paired-difference
# column and the comparator's own revenue column in results.csv.
FIGC_COMPARATORS = {'random_search': ('RandSearch', 'diff_rs', 'rs_revenue'),
                    'simulated_annealing': ('SA', 'diff_sa', 'sa_revenue')}
FIGC_COMPARATOR_COLUMNS = tuple(c for _, diff, rev in FIGC_COMPARATORS.values()
                                for c in (rev, diff))
# Figure C's store and search options, compared exactly with the runner's
# defaults (the number of evaluation replicates may only be larger).
FIGC_DESIGN_ARGS = ('sheets', 'max_items_per_category', 'assumed_conversion',
                    'mc_iters', 'mc_days', 'n_gens', 'pop_size', 'ga_seed',
                    'sa_initial_accept')

# Figure C over several search seeds (run_real_data_seeds). Figure C's own
# intervals re-score one search per method, so they say nothing about how
# much a second search would differ; the across-seed interval does, and
# with fewer than five seeds its t quantile is too wide to say much.
MIN_FIGC_SEEDS = 5
FIGC_SEED_SEARCHES = ('GA', 'random_search', 'simulated_annealing')
# The across-seed comparisons and the stems their macros use.
FIGC_SEED_STEMS = {'random_search': 'FigCSeedsGAvsRandSearch',
                   'simulated_annealing': 'FigCSeedsGAvsSA',
                   'lift_over_baseline': 'FigCSeedsLift'}

# Figures A and B: the options that fix the search and evaluation budget,
# held to the runner's defaults exactly (review R50), and the ones that may
# only be larger.
FIGAB_BUDGET_ARGS = ('n_items', 'mc_iters', 'mc_days', 'n_gens', 'pop_size')

# Where the anchored objective puts the as-built layout's score, and the
# source the headless runners give it (``_common.anchor_base_params``).
ANCHOR_SOURCE = 'as_built_repaired'
# The elasticity constants and GA weights every sidecar snapshots
# (``_common.elasticity_snapshot``), each against its registry name.
ELASTICITY_CONSTANTS = ('ELASTICITY_CONV_BASE', 'ELASTICITY_CONV_GAIN_MAX',
                        'ELASTICITY_IMP_BASE', 'ELASTICITY_IMP_GAIN_MAX',
                        'ELASTICITY_BSK_BASE', 'ELASTICITY_BSK_GAIN_MAX')
GA_WEIGHT_CONSTANTS = {'traffic': 'GA_W_TRAFFIC',
                       'cross_merch': 'GA_W_CROSS_MERCH',
                       'impulse': 'GA_W_IMPULSE', 'flow': 'GA_W_FLOW',
                       'revenue_placement': 'GA_W_REVENUE_PLACEMENT',
                       'section_compliance': 'GA_W_SECTION_COMPLIANCE',
                       'accessibility': 'GA_W_ACCESSIBILITY',
                       'overlap_penalty': 'GA_PEN_OVERLAP',
                       'bottleneck_penalty': 'GA_PEN_BOTTLENECK'}

# How every live runner labels its analysis since the stage-2 live package
# (review R19): the rate-based separation rule, the drained cohorts and the
# protocol study came with it, and runs made before it say 'terminating'.
LIVE_FRAMING = 'steady_state_replication_deletion'
# The ABM runner's routing null over whole visits (review R13); the former
# per-fixture null counted each visit's door and lane legs once per item.
ROUTING_NULL_CONSTRUCTION = 'whole_visit_tours'
# The structural sweep's design since it moved to common random numbers.
STRUCT_DESIGN = 'common_random_numbers'


# --- Reading the code the artifacts are held to ----------------------------

_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub,
           ast.Mult: operator.mul, ast.Div: operator.truediv,
           ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
           ast.Pow: operator.pow}
_CALLS = {'list': list, 'tuple': tuple, 'float': float, 'int': int,
          'sorted': sorted, 'round': round}
# Pure one-argument ``math`` functions a registry constant may be defined
# with (``SEPARATION_RATE_PER_S = -math.log1p(-s) / dt``).
_MATH_CALLS = {'log1p': math.log1p, 'log': math.log, 'exp': math.exp,
               'expm1': math.expm1, 'sqrt': math.sqrt}


class _Unresolved(Exception):
    """A value this reader cannot evaluate without running the module."""


def _eval_node(node, env):
    """Value of a constant expression: literals, containers, arithmetic,
    names bound earlier in ``env`` and a few pure builtins."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        raise _Unresolved(node.id)
    if isinstance(node, ast.Tuple):
        return tuple(_eval_node(e, env) for e in node.elts)
    if isinstance(node, ast.List):
        return [_eval_node(e, env) for e in node.elts]
    if isinstance(node, ast.Dict):
        if any(k is None for k in node.keys):
            raise _Unresolved('**')
        return {_eval_node(k, env): _eval_node(v, env)
                for k, v in zip(node.keys, node.values)}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op,
                                                    (ast.USub, ast.UAdd)):
        v = _eval_node(node.operand, env)
        return -v if isinstance(node.op, ast.USub) else +v
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_eval_node(node.left, env),
                                      _eval_node(node.right, env))
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in _CALLS and not node.keywords
            and len(node.args) == 1):
        return _CALLS[node.func.id](_eval_node(node.args[0], env))
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == 'math'
            and node.func.attr in _MATH_CALLS and not node.keywords
            and len(node.args) == 1):
        return _MATH_CALLS[node.func.attr](_eval_node(node.args[0], env))
    raise _Unresolved(type(node).__name__)


def _parse(relpath):
    path = os.path.join(HERE, *relpath.split('/'))
    with open(path, encoding='utf-8') as f:
        return ast.parse(f.read(), filename=path)


@functools.lru_cache(maxsize=None)
def _module_values(relpath):
    """Every module-level constant of a script under CODE/ that can be
    evaluated from its source alone, in assignment order (read, never
    imported: importing the runners would pull the simulator and dataset
    stack into this script, which restyle_figures imports)."""
    env = {}
    for node in _parse(relpath).body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            name, value = node.targets[0].id, node.value
        elif (isinstance(node, ast.AnnAssign)
              and isinstance(node.target, ast.Name)
              and node.value is not None):
            name, value = node.target.id, node.value
        else:
            continue
        try:
            env[name] = _eval_node(value, env)
        except Exception:
            env.pop(name, None)
    return env


@functools.lru_cache(maxsize=None)
def _script_constants(relpath, names):
    """Module-level constants of a script under CODE/, read from its
    source rather than imported.

    The validators below hold a run to the design its script defines --
    the live runners' protocol, the paper figures' scenario -- so reading
    those values from the script itself means a retuned protocol moves the
    check with it instead of leaving a stale copy here."""
    values = _module_values(relpath)
    absent = [n for n in names if n not in values]
    if absent:
        raise RuntimeError(f"{relpath} no longer defines {', '.join(absent)}, "
                           "which the artifact checks here are held to")
    return {n: values[n] for n in names}


@functools.lru_cache(maxsize=None)
def _class_constant(relpath, cls, name):
    """A literal class attribute (``version = "1.2"``) of a script."""
    for node in _parse(relpath).body:
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for stmt in node.body:
                if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                        and isinstance(stmt.targets[0], ast.Name)
                        and stmt.targets[0].id == name):
                    return ast.literal_eval(stmt.value)
    raise RuntimeError(f"{relpath} no longer defines {cls}.{name}, which "
                       "the artifact checks here are held to")


def _adapter_version():
    """The version of the Online Retail II adapter every UCI calibration
    must have been cleaned by (review R27: 1.2 pairs each cancellation
    with the purchase it reverses)."""
    return _class_constant(ADAPTER_SCRIPT, 'OnlineRetailIIAdapter', 'version')


# Modules a runner's defaults are imported from.
_DEFAULT_SOURCES = (COMMON_SCRIPT, LIT_SCRIPT,
                    'experiments/metaheuristics.py')


@functools.lru_cache(maxsize=None)
def _argparse_defaults(relpath, extra_sources=()):
    """``{dest: default}`` of every ``add_argument`` call in a script whose
    default is a constant expression, names resolved in the script, then
    in ``extra_sources`` and the modules runners import defaults from. The
    paper runs the runners at these defaults (the Makefile passes no other
    budget), so they are the design the macros are quoted at."""
    env = {}
    for src in reversed(tuple(extra_sources) + _DEFAULT_SOURCES):
        env.update(_module_values(src))
    env.update(_module_values(relpath))
    out = {}
    for node in ast.walk(_parse(relpath)):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'add_argument'):
            continue
        opts = [a.value for a in node.args
                if isinstance(a, ast.Constant) and isinstance(a.value, str)]
        kw = {k.arg: k.value for k in node.keywords if k.arg}
        if isinstance(kw.get('dest'), ast.Constant):
            dest = kw['dest'].value
        else:
            longs = [o for o in opts if o.startswith('--')]
            if not longs:
                continue
            dest = longs[0][2:].replace('-', '_')
        action = kw.get('action')
        if ('default' not in kw and isinstance(action, ast.Constant)
                and action.value in ('store_true', 'store_false')):
            out[dest] = action.value == 'store_false'
            continue
        if 'default' not in kw:
            continue
        try:
            out[dest] = _eval_node(kw['default'], env)
        except Exception:
            continue
    return out


def _design_default(relpath, key, extra_sources=()):
    d = _argparse_defaults(relpath, extra_sources)
    if key not in d:
        raise RuntimeError(f"{relpath} has no constant default for "
                           f"--{key.replace('_', '-')}, which the artifact "
                           "checks here are held to")
    return d[key]


def _oracle_restarts():
    """Restarts of the analytical reference every runner that uses it must
    give it (``_common.ORACLE_RESTARTS``, review R36)."""
    return int(_script_constants(COMMON_SCRIPT,
                                 ('ORACLE_RESTARTS',))['ORACLE_RESTARTS'])


def _final_seeds():
    return int(_script_constants(COMMON_SCRIPT,
                                 ('GA_N_FINAL_SEEDS',))['GA_N_FINAL_SEEDS'])


# --- Generic artifact checks ------------------------------------------------

def _read_json(path):
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def _side(d):
    return _read_json(os.path.join(d, 'sidecar.json'))


def _summ(d):
    return _read_json(os.path.join(d, 'summary.json'))


def _get(obj, *path, default=None):
    """``obj[path[0]][path[1]]...`` or ``default`` when any step is absent."""
    for key in path:
        if isinstance(obj, dict) and key in obj:
            obj = obj[key]
        elif isinstance(obj, list) and isinstance(key, int) \
                and -len(obj) <= key < len(obj):
            obj = obj[key]
        else:
            return default
    return obj


def _same(a, b):
    """Recorded value ``a`` equal to the code's ``b`` (tuples and lists
    alike, floats to 1e-12)."""
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-15)
    return a == b


def _args_match(args, relpath, exact=(), at_least=(), extra_sources=()):
    """True if the recorded ``args`` equal the runner's defaults on
    ``exact`` and reach them on ``at_least``."""
    if not isinstance(args, dict):
        return False
    for k in exact:
        if not _same(args.get(k), _design_default(relpath, k, extra_sources)):
            return False
    for k in at_least:
        v = args.get(k)
        if v is None or float(v) < float(_design_default(relpath, k,
                                                         extra_sources)):
            return False
    return True


def _traceable(side):
    """True if a sidecar names the checkout the run started from.

    Runners snapshot the checkout at start-up and record whether it moved
    before the sidecar was written. Sidecars without that record read the
    tree only when the run ended, hours later for a paper-grade run, so
    their commit and diff hash need not describe the code that produced
    the numbers, and such a run cannot back a paper number."""
    return 'git_state_changed_during_run' in side


def _sidecar_clean(side):
    """A sidecar that pins the code: traceable, made from a tree with no
    uncommitted code (``git_dirty`` false) that did not move while the run
    was in flight. Only such a run's commit is the code behind its
    numbers."""
    return (isinstance(side, dict) and _traceable(side)
            and side.get('git_state_changed_during_run') is False
            and side.get('git_dirty') is False)


def _anchor_records(bp):
    """Every score anchor in a ``base_params`` record, however it is
    nested (per scenario, per prior variant, per period). A record that
    carries parameters but no anchor -- a mapping with a non-mapping value
    and no ``score_anchor``, or an empty one -- contributes None, so one
    unanchored scenario or variant among anchored ones is not missed."""
    if not isinstance(bp, dict):
        return []
    if 'score_anchor' in bp:
        return [bp['score_anchor']]
    if not bp or any(not isinstance(v, dict) for v in bp.values()):
        return [None]
    return [a for v in bp.values() for a in _anchor_records(v)]


def _model_current(rec):
    """True if ``rec`` (a sidecar, or a record stamped like one) was made
    under today's layout objective.

    Stage 1 anchored the elasticities at the repaired as-built layout
    (review R03); every run before it scored layouts by their absolute
    score, about a third above the calibrated level. Such a run records no
    ``score_anchor`` in its ``base_params`` and no ``anchoring`` in its
    elasticity snapshot. The snapshot must also hold every objective
    constant ``_common.OBJECTIVE_CONSTANTS`` names, at today's registry
    value -- so a run under the former floored spend law (no
    ``MC_SPEND_LAW``) or under any constant since changed is refused --
    and today's elasticity bands and GA weights."""
    el = rec.get('elasticities') if isinstance(rec, dict) else None
    if not isinstance(el, dict) or 'anchoring' not in el:
        return False
    reg = _module_values(LIT_SCRIPT)
    names = _script_constants(COMMON_SCRIPT, ('OBJECTIVE_CONSTANTS',))[
        'OBJECTIVE_CONSTANTS']
    consts = el.get('objective_constants')
    if not isinstance(consts, dict):
        return False
    for n in names:
        if n not in reg:
            raise RuntimeError(f"{LIT_SCRIPT} has no constant value for {n}, "
                               "which the artifact checks here are held to")
        if n not in consts or not _same(consts[n], reg[n]):
            return False
    if not all(n in el and _same(el[n], reg[n]) for n in ELASTICITY_CONSTANTS):
        return False
    w = el.get('GA_weights')
    if not (isinstance(w, dict) and all(
            f in w and _same(w[f], reg[c])
            for f, c in GA_WEIGHT_CONSTANTS.items())):
        return False
    anchors = _anchor_records(rec.get('base_params'))
    return bool(anchors) and all(
        isinstance(a, dict) and a.get('source') == ANCHOR_SOURCE
        and a.get('score') is not None for a in anchors)


def _sidecar_big_enough(d):
    """True if the run's sidecar reports a paper-scale design."""
    j = _side(d)
    if not isinstance(j, dict):
        return False
    try:
        return (int(j.get('n_scenarios', 0)) >= MIN_SCENARIOS
                and int(j.get('n_seeds_per_scenario', 0)) >= MIN_SEEDS
                and _traceable(j))
    except Exception:
        return False


def _has_summary(d):
    """True if the run finished: the runner writes summary.json last.

    Without this, an in-progress run is newer than the completed one and
    silently blanks its macros to TBD -- observed live when a paper-grade
    diagnostic was running during a regeneration (audit R68).
    """
    return os.path.exists(os.path.join(d, 'summary.json'))


def _has_results_csv(d):
    """True if the run got as far as writing its result table.

    The real-data runner writes results.csv at the end, so an in-progress
    directory would otherwise be selected and then fail to read (audit
    R68, completing the sweep of unguarded lookups).
    """
    return os.path.exists(os.path.join(d, 'results.csv'))


def _csv_header(d):
    try:
        with open(os.path.join(d, 'results.csv'), newline='') as f:
            return next(csv.reader(f))
    except Exception:
        return []


def _dir_summary_ok(d, pred):
    """Apply ``pred`` to a finished run's summary.json."""
    if not _has_summary(d):
        return False
    try:
        return bool(pred(_summ(d)))
    except Exception:
        return False


def _checked_json(path, pred):
    """A figs/ record (the headless figures' realized scores), or None when
    it fails ``pred``."""
    if not os.path.exists(path):
        return None
    try:
        s = _read_json(path)
        return s if pred(s) else None
    except Exception:
        return None


def _data_ok(data, exclude_anonymous=False):
    """A UCI calibration record (Figure C's ``data`` block, a live period
    record) made by today's adapter with the anonymous invoices kept. A
    no-anonymous run is a sensitivity analysis (review R27) and is never a
    headline number; its own family (``exclude_anonymous=True``) asks for
    the opposite."""
    return (isinstance(data, dict)
            and data.get('adapter_version') == _adapter_version()
            and data.get('exclude_anonymous') is bool(exclude_anonymous))


def _starts_asbuilt(record, searches):
    """True if ``record`` names the start of every one of ``searches`` and
    each of them started from the as-built layout. A search that started
    from different layouts in different runs is recorded as their
    '/'-joined names, which fails the check."""
    starts = record.get('search_start')
    return (isinstance(starts, dict)
            and all(starts.get(m) == ASBUILT for m in searches))


# --- Non-live families ------------------------------------------------------

def _figa_big_enough(d):
    """Paper-scale synthetic run under today's objective: the runner's own
    budget (generations, population, MC iterations, horizon, items; review
    R50), at least its scenarios, seeds and held-out Spearman layouts, the
    analytical reference at the full restart count (R36) with its restart
    curve, the reference mapped onto the GA's feasible set (the
    ``oracle_R_feasible`` column), the moved-reference count per scenario
    (R64), the closed-form re-scoring (R02) and a clean sidecar."""
    if not (_sidecar_big_enough(d) and _has_results_csv(d)
            and _has_summary(d)):
        return False
    try:
        side, s = _side(d), _summ(d)
        return ({'oracle_R_feasible', 'ga_cf_R'} <= set(_csv_header(d))
                and _sidecar_clean(side) and _model_current(side)
                and _args_match(side.get('args'), FIGA_SCRIPT,
                                exact=FIGAB_BUDGET_ARGS,
                                at_least=('n_scenarios', 'n_seeds',
                                          'n_spearman_samples',
                                          'oracle_restarts'))
                and int(_get(s, 'oracle', 'n_restarts', default=0))
                >= _oracle_restarts()
                and isinstance(_get(s, 'oracle',
                                    'gain_after_checkpoint_pct'), dict)
                and isinstance(_get(s, 'reference_moved_by_repair',
                                    'n_scenarios'), int)
                and isinstance(_get(s, 'closed_form',
                                    'conclusions_unchanged'), dict))
    except Exception:
        return False


def _figb_big_enough(d):
    """Paper-scale, finished comparison run under today's objective at the
    runner's own budget (review R50), with the analytical reference at the
    full restart count; the annealing schedule its comparator ran under;
    the comparator family it corrects for, with the sensitivity rows
    outside it; an as-built start for random search and the annealer; the
    equal budget as recorded (every equal-budget search at exactly the
    search and final-selection evaluations its budget block states); the
    runner's interval schemes and closed-form re-scoring (R02, R28, R63);
    and a clean sidecar."""
    if not (_sidecar_big_enough(d) and _has_results_csv(d)
            and _has_summary(d)):
        return False
    try:
        j, s = _side(d), _summ(d)
        family = j.get('comparison_family')
        a = j.get('args') or {}
        search = int(a['pop_size']) * int(a['n_gens'])
        final = int(a['pop_size']) * _final_seeds()
        budget = s.get('budget') or {}
        equal = j.get('equal_budget_methods') or []
        inf = s.get('inference') or {}
        return (isinstance(j.get('sa_schedule'), dict)
                and isinstance(family, list) and bool(family)
                and all(m in FIGB_SHORT_NAMES for m in family)
                and _starts_asbuilt(j, FIGB_ASBUILT_SEARCHES)
                and _sidecar_clean(j) and _model_current(j)
                and _args_match(a, FIGB_SCRIPT, exact=FIGAB_BUDGET_ARGS,
                                at_least=('n_scenarios', 'n_seeds',
                                          'oracle_restarts'))
                and int(s.get('oracle_restarts', 0)) >= _oracle_restarts()
                and int(budget.get('search_evals', -1)) == search
                and int(budget.get('final_evals', -1)) == final
                and bool(equal)
                and all(list(j['evaluation_counts'][m]) == [search]
                        for m in equal)
                and all(list(j['final_evaluation_counts'][m]) == [final]
                        for m in equal)
                and isinstance(_get(inf, 'mc', 'comparisons'), dict)
                and isinstance(_get(inf, 'closed_form', 'comparisons'), dict)
                and isinstance(s.get('closed_form_conclusions_unchanged'),
                               dict)
                and isinstance(_get(s, 'reference_moved_by_repair',
                                    'n_scenarios'), int))
    except Exception:
        return False


def _figc_comparators_recorded(d):
    """True if a real-data run scored the equal-budget searches beside the
    GA: their columns in results.csv, or their paired differences in the
    run's summary."""
    if all(c in _csv_header(d) for c in FIGC_COMPARATOR_COLUMNS):
        return True
    comps = _get(_summ(d), 'results', 'comparators') or {}
    return all(isinstance(comps.get(m), dict) for m in FIGC_COMPARATORS)


def _figc_big_enough(d):
    """True if a finished real-data run used Figure C's design -- the
    runner's own store and search options, at least its evaluation
    replicates, both sheets -- under today's objective and today's
    cleaning with the anonymous invoices kept; records what the workbook
    reader did (rows read per sheet and the cross-sheet repeats dropped);
    scored the equal-budget searches under the same paired replicates;
    carries the closed-form scan and check (R02, R25), the search
    operators and the aisle rule's record (R06); and has a clean
    sidecar."""
    return _figc_valid(d, exclude_anonymous=False)


# Run-directory prefix of Figure C's no-anonymous sensitivity family
# (``run_real_data_example.NOANON_DIR_PREFIX``, checked in
# ``_check_sources``): its runs never share the headline's prefix, so the
# headline glob cannot pick one up.
NOANON_PREFIX = 'noanon_'


def _figc_noanon_big_enough(d):
    """Figure C's sensitivity run without the anonymous invoices (review
    R27): every check of the headline, at the same design, with
    ``exclude_anonymous`` true in the arguments and the data record, and
    written under the no-anonymous prefix."""
    return (os.path.basename(d).startswith(NOANON_PREFIX)
            and _figc_valid(d, exclude_anonymous=True))


def _figc_valid(d, exclude_anonymous):
    if not (_has_results_csv(d) and _has_summary(d)):
        return False
    try:
        j, s = _side(d), _summ(d)
        a = j.get('args', {})
        sheets = [x for x in str(a.get('sheets', '')).split(',') if x.strip()]
        res = s.get('results') or {}
        return (int(a.get('n_mc_replicates', 0)) >= MIN_FIGC_REPLICATES
                and int(a.get('mc_iters', 0)) >= MIN_FIGC_MC_ITERS
                and len(sheets) >= MIN_FIGC_SHEETS
                and isinstance(j.get('reader'), dict)
                and _sidecar_clean(j) and _model_current(j)
                and _args_match(a, FIGC_SCRIPT, exact=FIGC_DESIGN_ARGS,
                                at_least=('n_mc_replicates',))
                and a.get('exclude_anonymous') is bool(exclude_anonymous)
                and _data_ok(s.get('data'), exclude_anonymous)
                and _get(j, 'provenance', 'adapter_version')
                == _adapter_version()
                and isinstance(res.get('closed_form'), dict)
                and isinstance(res.get('closed_form_check'), dict)
                and isinstance(_get(s, 'feasibility', 'aisle_rule'), dict)
                and isinstance(_get(s, 'search_design', 'operators'), dict)
                and _figc_comparators_recorded(d))
    except Exception:
        return False


def _figc_seeds_big_enough(d):
    """True if a finished multi-seed real-data run has the design its
    numbers are quoted at: at least the runner's default number of search
    seeds, each with a recorded result; Figure C's evaluation panel,
    precision and sheets; every search of every seed from the as-built
    layout and at the run's one search budget; today's objective and
    cleaning with the anonymous invoices kept; the closed-form re-scoring,
    the search operators and the aisle rule's record; and a clean
    sidecar. A run short of any of these measures a different comparison
    from the one the across-seed macros describe."""
    if not (_has_summary(d) and _has_results_csv(d)):
        return False
    try:
        side, s = _side(d), _summ(d)
        design = s['design']
        seeds = [int(x) for x in design['search_seeds']]
        k = int(design['n_search_seeds'])
        budget = int(design['budget_search_evals'])
        counts = s['evaluation_counts']
        results = s['results']
        per_seed = [int(r['search_seed']) for r in results['per_seed']]
        across = results['across_seeds']
        sheets = [x for x in str(design.get('sheets', '')).split(',')
                  if x.strip()]
        return (k >= max(MIN_FIGC_SEEDS,
                         int(_design_default(SEEDS_SCRIPT,
                                             'n_search_seeds')))
                and seeds == per_seed and len(set(seeds)) == k == len(seeds)
                and int(design['n_mc_replicates']) >= MIN_FIGC_REPLICATES
                and int(design['mc_iters']) >= MIN_FIGC_MC_ITERS
                and len(sheets) >= MIN_FIGC_SHEETS
                and _starts_asbuilt(s, FIGC_SEED_SEARCHES)
                and budget > 0
                and set(counts) == {str(x) for x in seeds}
                and all(int(counts[str(x)][m]) == budget
                        for x in seeds for m in FIGC_SEED_SEARCHES)
                and all(int(across[key]['n_seeds']) == k
                        for key in FIGC_SEED_STEMS)
                and design.get('exclude_anonymous') is False
                and isinstance(design.get('operators'), dict)
                and isinstance(results.get('closed_form'), dict)
                and isinstance(_get(s, 'feasibility', 'aisle_rule'), dict)
                and _get(side, 'provenance', 'adapter_version')
                == _adapter_version()
                and _sidecar_clean(side) and _model_current(side))
    except Exception:
        return False


# The options a multi-seed real-data run shares with Figure C: the store
# (sheets, items per category, assumed conversion, the anonymous invoices
# in or out), the search budget and annealing schedule, and the Monte Carlo
# precision and evaluation panel. Keyed by the name in the seeds run's
# design block; the value is the name in Figure C's recorded args.
FIGC_SEEDS_SHARED_DESIGN = {
    'sheets': 'sheets',
    'max_items_per_category': 'max_items_per_category',
    'assumed_conversion': 'assumed_conversion',
    'exclude_anonymous': 'exclude_anonymous',
    'n_gens': 'n_gens',
    'pop_size': 'pop_size',
    'mc_iters': 'mc_iters',
    'mc_days': 'mc_days',
    'n_mc_replicates': 'n_mc_replicates',
}


def _figc_seeds_match_figc(d, figc_dir):
    """True if the multi-seed run ``d`` searched and scored Figure C's own
    design (``figc_dir``'s recorded args): the same store, budget, schedule
    and evaluation panel. The across-seed macros are quoted as the
    search-to-search spread OF Figure C's comparison; a run at another
    budget or on another store would pass the size checks and still
    describe a different comparison."""
    if not figc_dir:
        return False
    try:
        a = _side(figc_dir)['args']
        s = _summ(d)
        design = s['design']
        same = all(design[mine] == a[theirs]
                   for mine, theirs in FIGC_SEEDS_SHARED_DESIGN.items())
        return (same and float(s['sa_schedule']['sa_initial_accept'])
                == float(a['sa_initial_accept']))
    except Exception:
        return False


RESCORE_SCRIPT = 'experiments/run_figc_rescore.py'
RESCORE_LAYOUTS = ('optimized', 'rs', 'sa')


def _rescore_big_enough(d):
    """The closed-form re-scoring of Figure C's saved layouts
    (``run_figc_rescore``): a clean sidecar under today's model (it scores
    with Figure C's base parameters, which it records), the Figure C run's
    own closed-form values reproduced, the Omnichannel conversion bound
    with the bundle's digest, the lift at that bound for every searched
    layout, and the decomposition of every searched layout's lift."""
    if not _has_summary(d):
        return False
    try:
        s, side = _summ(d), _side(d)
        per = s['decomposition']['per_layout']
        at = s['lift_at_conversion_bound']
        return (_sidecar_clean(side) and _model_current(side)
                and s['figc']['values_reproduced'] is True
                and 0.0 < float(s['conversion_bound']['value']) <= 1.0
                and bool(s['conversion_bound']['provenance']['source_sha256'])
                and all(k in at['lift_pct'] and k in per
                        and per[k].get('abandonment_share') is not None
                        and per[k].get('flow_share_of_score_gain') is not None
                        for k in RESCORE_LAYOUTS))
    except Exception:
        return False


def _summary_digests(path):
    """SHA-256 digests of a run's summary file as it is on disk and with its
    line endings made all-LF or all-CRLF. A recorded digest names the bytes
    the run wrote -- CRLF when a runner wrote text on Windows -- while git
    may check the same file out with either ending (LF in its object store
    and on most platforms, CRLF under core.autocrlf), so a match on any of
    the three identifies the same content."""
    with open(path, 'rb') as fh:
        raw = fh.read()
    lf = raw.replace(b'\r\n', b'\n')
    return {hashlib.sha256(b).hexdigest()
            for b in (raw, lf, lf.replace(b'\n', b'\r\n'))}


def _rescore_matches_figc(d, figc_dir):
    """True if re-scoring ``d`` read the Figure C run that sets Figure C's
    macros, as it is now: the same directory and the same summary file
    (its SHA-256, line endings aside), so a re-run of Figure C cannot leave the
    re-scoring describing the layouts of an earlier run."""
    if not figc_dir:
        return False
    try:
        f = _summ(d)['figc']
        digests = _summary_digests(os.path.join(figc_dir, 'summary.json'))
        return (os.path.basename(os.path.normpath(f['run_name']))
                == os.path.basename(os.path.normpath(figc_dir))
                and f['summary_sha256'] in digests)
    except Exception:
        return False


def _lhs_big_enough(d):
    """True if a finished LHS run covers the paper's design resolution and
    sweeps every headline comparison rather than only the GA against the
    popularity baseline, which is what the ``comparisons`` block records,
    and carries the weight sweep over the same layouts at the paper's
    number of weight vectors; under today's objective and spend law, with
    the analytical reference at the full restart count (R36), the search
    horizon and effective-weight sample of the runner's defaults, the
    effective weights and both standardized sweeps (R09), the pair counts
    and within-pair ranges (R38), the basket corners and the short-path
    variant (R37, R38); and a clean sidecar."""
    if not _has_summary(d):
        return False
    try:
        s, side = _summ(d), _side(d)
        cfg = s.get('config', {})
        ws = s.get('weight_sweep')
        comps = s.get('comparisons')
        return (int(s.get('n_lhs_points', 0)) >= MIN_LHS_POINTS
                and int(s.get('n_scenarios', 0)) >= MIN_LHS_SCENARIOS
                and all(int(cfg.get(k, 0)) >= v
                        for k, v in MIN_LHS_DESIGN.items())
                and isinstance(comps, dict)
                and isinstance(ws, dict)
                and isinstance(ws.get('comparisons'), dict)
                and bool(ws['comparisons'])
                and isinstance(ws.get('design'), dict)
                and int(ws['design'].get('n_weight_draws', 0))
                >= MIN_LHS_WEIGHT_DRAWS
                and _args_match(cfg, LHS_SCRIPT, exact=('n_items', 'mc_days'),
                                at_least=('n_weight_draws',
                                          'n_spread_layouts'))
                and int(_get(s, 'oracle', 'n_restarts', default=0))
                >= _oracle_restarts()
                and _get(s, 'objective', 'mc_spend_law')
                == _script_constants(LIT_SCRIPT,
                                     ('MC_SPEND_LAW',))['MC_SPEND_LAW']
                and all(isinstance(s.get(k), dict) for k in (
                    'effective_weights', 'weight_sweep_standardized',
                    'weight_sweep_standardized_unit', 'closed_form_extras'))
                and all(isinstance(c.get('pairs'), dict)
                        for c in comps.values())
                and _sidecar_clean(side) and _model_current(side))
    except Exception:
        return False


def _mcgt_big_enough(d):
    """True if a finished MC ground-truth run used the runner's budgets --
    items, normal and big budget, MC iterations, horizon, at least its
    scenarios and confirmation seeds (review R50, R54) -- reports the
    smallest regret with the count that came out below zero, without which
    the confirmation-noise tail cannot be quoted, started every big-budget
    search from the as-built layout and added the popularity-started
    annealer to the best-known reference (R50); under today's objective
    with the closed-form re-scoring (R02) and a clean sidecar."""
    if not _has_summary(d):
        return False
    try:
        s, side = _summ(d), _side(d)
        normal = float(s.get('normal_budget', 0))
        return (int(s.get('n_scenarios', 0)) >= MIN_MCGT_SCENARIOS
                and normal >= MIN_MCGT_NORMAL_BUDGET
                and float(s.get('big_budget', 0))
                >= MIN_MCGT_BUDGET_RATIO * normal
                and int(s.get('mc_iters', 0)) >= MIN_MCGT_MC_ITERS
                and s.get('mc_regret_min_pct') is not None
                and s.get('n_negative_regret') is not None
                and _starts_asbuilt(s, MCGT_ASBUILT_SEARCHES)
                and _get(s, 'search_start', MCGT_POPSTART) == POPULARITY
                and isinstance(_get(s, 'closed_form',
                                    'conclusions_unchanged'), dict)
                and _args_match(side.get('args'), MCGT_SCRIPT,
                                exact=('n_items', 'normal_budget',
                                       'big_budget', 'mc_iters', 'mc_days',
                                       'sa_initial_accept'),
                                at_least=('n_scenarios', 'confirm_seeds'))
                and _sidecar_clean(side) and _model_current(side))
    except Exception:
        return False


def _ga_budget(pop_size, n_gens):
    return int(pop_size) * (int(n_gens) + _final_seeds())


def _sa_sweep_size():
    c = _script_constants(GASENS_SCRIPT, ('SA_STEP_FRACS',
                                          'SA_FINAL_TEMP_FRACS',
                                          'SA_INITIAL_ACCEPTS'))
    return (len(c['SA_STEP_FRACS']) * len(c['SA_FINAL_TEMP_FRACS'])
            + len(c['SA_INITIAL_ACCEPTS']) - 1)


def _gasens_ok(s):
    """The sweep in its fixed-budget form (review R17) at the runner's own
    design: every GA setting at the default setting's budget, population x
    (generations + final seeds); the settings compared on re-scored
    held-out seeds; the seed-blocked null and ANOVA; the annealer sweep
    over all its settings with the default annealer as power reference;
    and the checkout that produced it; and the diversity figure (Fig. 11)
    drawn at its print width, with no text below 7 pt there (review R56)."""
    cfg = s.get('config', {})
    budget = _ga_budget(cfg['pop_size'], cfg['n_gens'])
    gs, sa = s.get('ga_sweep') or {}, s.get('sa_sweep') or {}
    return (isinstance(s.get('provenance'), dict)
            and _args_match(cfg, GASENS_SCRIPT,
                            exact=('n_items', 'n_gens', 'pop_size',
                                   'mc_iters', 'mc_days', 'mode'),
                            at_least=('n_scenarios', 'n_seeds',
                                      'n_heldout_seeds'))
            and s.get('fitness_statistic') == 'heldout_mean'
            and int(s.get('budget_total_evals', -1)) == budget
            and bool(s.get('settings'))
            and all(int(x['total_evals']) == budget for x in s['settings'])
            and bool(gs.get('null_design'))
            and isinstance(_get(gs, 'anova_setting',
                                'setting_vs_interaction'), dict)
            and isinstance(sa.get('power_reference'), dict)
            and int(sa.get('n_settings', -1)) == _sa_sweep_size()
            and float(_get(s, 'diversity_figure', 'min_font_pt_printed',
                           default=0)) >= 7.0)


def _gasens_big_enough(d):
    if not _has_summary(d):
        return False
    try:
        side = _side(d)
        return (_gasens_ok(_summ(d)) and _sidecar_clean(side)
                and _model_current(side))
    except Exception:
        return False


def _align_big_enough(d):
    """The objective-alignment diagnostic (review R08) at the runner's own
    design: its scenarios, held-out layouts, GA budget, horizon, wall
    coefficient and reference restarts, both priors reported over every
    scenario, under today's objective, with a clean sidecar."""
    if not _has_summary(d):
        return False
    try:
        s, side = _summ(d), _side(d)
        cfg = s.get('config') or {}
        v = s.get('variants') or {}
        n = int(cfg['n_scenarios'])
        return (_args_match(cfg, ALIGN_SCRIPT,
                            exact=('n_items', 'horizon_days', 'n_gens',
                                   'pop_size', 'ga_seed', 'wall_coef'),
                            at_least=('n_scenarios', 'n_samples',
                                      'oracle_restarts'))
                and all(int(_get(v, k, 'n_scenarios', default=-1)) == n
                        for k in ('as_now', 'wall_aligned'))
                and isinstance(s.get('change_as_now_to_wall_aligned'), dict)
                and int(_get(s, 'oracle', 'n_restarts', default=0))
                >= _oracle_restarts()
                and _sidecar_clean(side) and _model_current(side))
    except Exception:
        return False


def _uci_design_ok(design, *, gens_key='n_gens', seed_key='ga_seed',
                   sheets=True, replicates=True):
    """A real-data runner's design block against Figure C's defaults: the
    store, the Monte Carlo precision, the annealer, the first search seed
    and the evaluation replicates."""
    keys = ['max_items_per_category', 'assumed_conversion', 'mc_iters',
            'mc_days', 'pop_size', 'sa_initial_accept']
    if sheets:
        keys.append('sheets')
    rec = {k: design.get(k) for k in keys}
    ok = _args_match(rec, FIGC_SCRIPT, exact=keys)
    if gens_key:
        ok = ok and _same(design.get(gens_key),
                          _design_default(FIGC_SCRIPT, 'n_gens'))
    if seed_key:
        ok = ok and _same(design.get(seed_key),
                          _design_default(FIGC_SCRIPT, 'ga_seed'))
    if replicates:
        ok = ok and (int(design.get('n_mc_replicates', 0))
                     >= int(_design_default(FIGC_SCRIPT, 'n_mc_replicates')))
    return ok and design.get('exclude_anonymous') is False


def _budget_big_enough(d):
    """The real-data budget sweep (review R05) at Figure C's store and
    search design, every one of the runner's default budget multipliers
    (1x being Figure C's own) and at least its search seeds; today's
    objective and cleaning with the anonymous invoices kept; the search
    operators, the aisle rule's record, the closed-form check of every
    conclusion (R02) and every search from the as-built layout; and a
    clean sidecar."""
    if not _has_summary(d):
        return False
    try:
        s, side = _summ(d), _side(d)
        des = s['design']
        mults = [int(m) for m in des['budget_multipliers']]
        gens = int(_design_default(FIGC_SCRIPT, 'n_gens'))
        want = [int(m) for m in _design_default(BUDGET_SCRIPT,
                                                'budget_multipliers')]
        res = s.get('results') or {}
        return (_uci_design_ok(des, gens_key=None,
                               seed_key='first_search_seed')
                and set(want) <= set(mults) and 1 in mults
                and all(int(des['n_gens_per_multiplier'][str(m)])
                        == gens * m for m in mults)
                and int(des['n_search_seeds'])
                >= int(_design_default(BUDGET_SCRIPT, 'n_search_seeds'))
                and int(des.get('sa_k_move', 0)) > 0
                and isinstance(des.get('operators'), dict)
                and _data_ok(s.get('data'))
                and isinstance(_get(s, 'feasibility', 'aisle_rule'), dict)
                and 'all_unchanged' in (res.get(
                    'closed_form_conclusions_unchanged') or {})
                and isinstance(res.get('reference'), dict)
                and isinstance(res.get('per_method'), dict)
                and set((s.get('search_start') or {}).values()) == {ASBUILT}
                and _get(side, 'provenance', 'adapter_version')
                == _adapter_version()
                and _sidecar_clean(side) and _model_current(side))
    except Exception:
        return False


def _input_big_enough(d):
    """The input-uncertainty bootstrap (review R11) at Figure C's design
    and at least the runner's replicate count, under today's objective and
    cleaning with the anonymous invoices kept: the three searches from the
    as-built layout at one budget, the GA's margins over random search and
    annealing re-scored in every replicate, the former spend law's bias at
    adapter 1.1 inputs, the aisle rule's record; and a clean sidecar."""
    if not _has_summary(d):
        return False
    try:
        s, side = _summ(d), _side(d)
        des = s['design']
        ga, sr = des['ga'], des['searches']
        rec = {**{k: des.get(k) for k in ('sheets', 'max_items_per_category',
                                          'assumed_conversion',
                                          'exclude_anonymous')},
               'mc_iters': ga['mc_iters'], 'mc_days': ga['mc_days'],
               'pop_size': ga['pop_size'], 'n_gens': ga['n_gens'],
               'ga_seed': ga['seed'],
               'sa_initial_accept': sr['sa_initial_accept']}
        iu = s.get('input_uncertainty') or {}
        return (_uci_design_ok(rec, replicates=False)
                and des.get('adapter_version') == _adapter_version()
                and int(des['n_boot']) >= int(_script_constants(
                    INPUT_SCRIPT, ('DEFAULT_N_BOOT',))['DEFAULT_N_BOOT'])
                and set(sr['starts'].values()) == {ASBUILT}
                and len({int(v) for v in sr['evaluation_counts'].values()})
                == 1
                and len({int(v) for v in
                         sr['final_evaluation_counts'].values()}) == 1
                and isinstance(sr.get('operators'), dict)
                and all(isinstance(_get(iu, sc, f'ga_minus_{c}'), dict)
                        for sc in ('whole', 'stocked') for c in ('rs', 'sa'))
                and isinstance(_get(s, 'legacy_floor_bias',
                                    'adapter_1_1_inputs'), dict)
                and isinstance(_get(s, 'feasibility', 'aisle_rule'), dict)
                and _sidecar_clean(side) and _model_current(side))
    except Exception:
        return False


def _transfer_big_enough(d):
    """The held-out transfer runner (review R12) at Figure C's search
    design and at least the runner's search seeds, under today's objective
    and cleaning with the anonymous invoices kept: both periods from
    today's adapter, the prior one cut on the raw rows before cleaning (so
    no later cancellation reaches into it) and sharing no invoice with the
    current one; every search from the as-built layout at one budget; the
    paired transfer gap; the aisle rule's record; and a clean sidecar."""
    if not _has_summary(d):
        return False
    try:
        s, side = _summ(d), _side(d)
        des = s['design']
        per = des['periods']
        prior, current = per['prior'], per['current']
        budget = int(des['budget_search_evals'])
        seeds = [int(x) for x in des['search_seeds']]
        methods = _get(s, 'across_seeds', 'methods') or {}
        return (_uci_design_ok(des, sheets=False,
                               seed_key=None)
                and seeds[0] == int(_design_default(FIGC_SCRIPT, 'ga_seed'))
                and int(des['n_search_seeds']) >= int(_script_constants(
                    TRANSFER_SCRIPT, ('DEFAULT_N_SEARCH_SEEDS',))[
                        'DEFAULT_N_SEARCH_SEEDS'])
                and des.get('search_start') == ASBUILT
                and isinstance(des.get('operators'), dict)
                and all(p.get('adapter_version') == _adapter_version()
                        for p in (prior, current))
                and isinstance(prior.get('reversals_across_cut'), dict)
                and bool(prior.get('cut_applied_to'))
                and int(prior.get('n_invoices_shared_with_current', -1)) == 0
                and all(int(v) == budget
                        for c in s['evaluation_counts'].values()
                        for v in c.values())
                and all(isinstance(_get(methods, m, 'transfer_gap'), dict)
                        for m in ('GA', 'random_search',
                                  'simulated_annealing'))
                and isinstance(_get(s, 'feasibility', 'aisle_rule'), dict)
                and _sidecar_clean(side) and _model_current(side))
    except Exception:
        return False


# --- Live families ------------------------------------------------------------

# The live diagnostics -- ABM, structural sweep, queueing, goodness of fit
# -- share one measurement protocol: a fixed-step run, a deleted warm-up,
# then a collection window at the nominal load. A paper number is taken
# only from a run that used it: the replication count alone cannot tell a
# paper-grade run from a quick one, because the runners' defaults are
# exactly the minimums above, so a shorter window or a different arrival
# rate would pass unnoticed. Each family is checked against the constants
# of the runner that produced it.
LIVE_MODE = 'fixed_step'
# The simulator's fixed tick (run_headless's default, the GUI loop's frame
# ceiling). A coarser tick moves agents in longer jumps, which is a
# different model, not a faster run of the same one.
LIVE_DT = 0.04
_LIVE_NAMES = ('NOMINAL_SPAWN', 'NOMINAL_CAP', 'WARMUP_S', 'COLLECT_S')
LIVE_RUNNERS = {'abm': 'experiments/run_abm_diagnostics.py',
                'structural': 'experiments/run_structural_sensitivity.py',
                'gof': 'experiments/run_validation_gof.py',
                'queue': 'experiments/measure_queueing.py'}


def _live_design(family):
    names = _LIVE_NAMES + (('STRESS_SPAWN', 'STRESS_CAP')
                           if family == 'queue' else ())
    return _script_constants(LIVE_RUNNERS[family], names)


def _live_protocol_ok(proto, spawn, cap, design, level='NOMINAL'):
    """True if a live run used its runner's protocol at the ``level`` load."""
    try:
        return (proto.get('mode') == LIVE_MODE
                and bool(np.isclose(float(proto.get('dt', 0)), LIVE_DT))
                and float(proto.get('collect_s', 0)) >= design['COLLECT_S']
                and float(proto.get('warmup_s', 0)) >= design['WARMUP_S']
                and bool(np.isclose(float(spawn), design[f'{level}_SPAWN']))
                and int(cap) == design[f'{level}_CAP'])
    except Exception:
        return False


def balked_pct(load):
    """Share of would-be arrivals the door turned away, as a percentage.

    The live runners cap the number of agents in the store; an arrival
    that meets a full store balks and is never simulated. How much of the
    offered load that removed belongs next to any number measured under
    the cap, since a cap that bites hard makes two loads less comparable
    than their arrival rates suggest. Some runners store the fraction,
    others only the two counts. Returns None when the run recorded
    neither."""
    if not isinstance(load, dict):
        return None
    if load.get('balked_frac') is not None:
        return 100.0 * float(load['balked_frac'])
    arrivals = float(load.get('arrivals') or 0.0)
    if arrivals <= 0:
        return None
    return 100.0 * float(load.get('balked') or 0.0) / arrivals


LIVE_STORE_SCRIPT = 'experiments/_live_store.py'


def _list_law_ok(data):
    """The run's store drew its shoppers' lists under the current list
    law (read from the live-store module). Runs from before each list
    became one stocked invoice carry no ``list_law`` and describe a
    different agent model."""
    want = _script_constants(LIVE_STORE_SCRIPT, ('LIST_LAW',))['LIST_LAW']
    return isinstance(data, dict) and data.get('list_law') == want


def _live_data_ok(data):
    """A live store's period record: today's list law, today's adapter
    (review R27, R51), the anonymous invoices kept, and the workbook's
    digest."""
    return (_list_law_ok(data) and _data_ok(data)
            and bool(data.get('source_sha256')))


def _live_ok(s, proto):
    """What every live family shares since the stage-2 live package: the
    steady-state framing its runners record (and with it the rate-based
    separation rule, review R41) and a store calibrated under today's
    list law and cleaning."""
    return (isinstance(proto, dict) and proto.get('framing') == LIVE_FRAMING
            and _live_data_ok(s.get('data')))


def _abm_ok(s):
    """The paper's replications, the shared protocol, no censored visit
    and finite t-intervals. A single-replication run stores NaN
    half-widths, which would otherwise be printed into the paper as
    'nan'; a censored agent contributes a truncated state sequence. The
    perimeter ratio must come with its geometric null -- the ratio the
    floor plan gives traffic spread evenly over the walkable cells -- in
    the summary and in every replication, since the ratio alone mixes the
    shoppers' routes with how much of each band is walkable; and with the
    whole-visit routing null at the runner's tour count, the moving-only
    ratio and the bookkeeping-only Markov null (review R13)."""
    mk, em = s.get('markov_order', {}), s.get('emergence', {})
    cis = (mk.get('info_gain_ci95'), mk.get('tv_ci95'),
           em.get('perimeter_ratio_ci95'),
           em.get('perimeter_interior_ratio_geometric_null'),
           em.get('ratio_to_geometric_null'),
           em.get('ratio_to_geometric_null_ci95'))
    proto = s.get('protocol', {})
    reps = s.get('per_rep') or []
    rn = em.get('routing_null') or {}
    tours = _script_constants(LIVE_RUNNERS['abm'], ('ROUTING_NULL_TOURS',))[
        'ROUTING_NULL_TOURS']
    return (_list_law_ok(s.get('data'))
            and _live_ok(s, proto)
            and int(proto.get('reps', 0)) >= MIN_ABM_REPS
            and _live_protocol_ok(proto, proto.get('spawn'), proto.get('cap'),
                                  _live_design('abm'))
            and int(mk.get('n_censored_final', -1)) == 0
            and all(c is not None and np.isfinite(float(c)) for c in cis)
            and bool(reps)
            and all('perimeter_interior_ratio_geometric_null'
                    in (r.get('emergence') or {})
                    and isinstance((r.get('emergence') or {})
                                   .get('band_sweep_geometric_null'), dict)
                    for r in reps)
            and rn.get('construction') == ROUTING_NULL_CONSTRUCTION
            and int(rn.get('n_tours', 0)) >= int(tours)
            and em.get('ratio_to_routing_null_ci95') is not None
            and isinstance(em.get('moving_only'), dict)
            and isinstance(em.get('headline'), dict)
            and isinstance(mk.get('bookkeeping_null'), dict))


def _abm_big_enough(d):
    return _dir_summary_ok(d, _abm_ok) and _sidecar_clean(_side(d))


def _struct_ok(s):
    """Replicated enough to judge between-setting spread against
    within-setting noise, run under the shared protocol in the common-
    random-numbers design, and carrying the per-replication tests the
    sweep is now read from: the block ANOVA, the paired contrasts against
    the default and their minimum detectable effects (review R40)."""
    proto = s.get('protocol', {})
    return (int(s.get('n_reps', 0)) >= MIN_STRUCT_REPS
            and _list_law_ok(s.get('data'))
            and _live_ok(s, proto)
            and _live_protocol_ok(proto, proto.get('spawn'), proto.get('cap'),
                                  _live_design('structural'))
            and s.get('rev_per_cust_anova_p') is not None
            and s.get('completions_anova_p') is not None
            and s.get('design') == STRUCT_DESIGN
            and isinstance(s.get('paired_contrasts'), dict)
            and isinstance(s.get('block_anova'), dict)
            and isinstance(s.get('mde'), dict))


def _struct_big_enough(d):
    return _dir_summary_ok(d, _struct_ok) and _sidecar_clean(_side(d))


def _struct_check_big_enough(d):
    """The structural sweep's exit-route check
    (``run_structural_exit_check``): a clean sidecar; at least
    ``MIN_STRUCT_CHECK_REPS`` replications of every setting re-run; every
    re-run reproducing the sweep's counts and revenue exactly, so what it
    logged are the sweep's own exits; and every run's exit-route tally
    agreeing with its per-exit log."""
    if not _has_summary(d):
        return False
    try:
        s, side = _summ(d), _side(d)
        runs, reps = s['runs'], s['reps_checked']
        t = s['total']
        return (_sidecar_clean(side) and bool(side.get('git_sha'))
                and s.get('experiment') == 'structural_exit_check'
                and len(set(reps)) >= MIN_STRUCT_CHECK_REPS
                and len(runs) == len(s['struct']['strengths']) * len(reps)
                and int(t['n_runs']) == len(runs)
                and t['all_reproduce'] is True
                and t['all_tallies_agree'] is True
                and all(r['matches_artifact'] is True
                        and r['tally_agrees'] is True for r in runs)
                and isinstance(s['sharing']['per_setting'], dict))
    except Exception:
        return False


def _struct_check_matches(d, struct_dir):
    """True if the check re-ran the structural sweep that sets the Struct
    macros, as it is now: the same directory and the same summary file
    (its SHA-256, line endings aside)."""
    if not struct_dir:
        return False
    try:
        rec = _summ(d)['struct']
        digests = _summary_digests(os.path.join(struct_dir, 'summary.json'))
        return (os.path.basename(os.path.normpath(rec['run_name']))
                == os.path.basename(os.path.normpath(struct_dir))
                and rec['summary_sha256'] in digests)
    except Exception:
        return False


def _queue_ok(s):
    """Replicated queue measurement at both the nominal and the stress
    load of the shared protocol, with the checkout that produced it. The
    busy fraction swings widely between windows, so a single short run is
    not a number the paper can quote."""
    proto = s.get('protocol', {})
    spawn, cap = proto.get('spawn'), proto.get('cap')
    if not isinstance(spawn, dict) or not isinstance(cap, dict):
        return False
    design = _live_design('queue')
    return (isinstance(s.get('provenance'), dict)
            and _list_law_ok(s.get('data'))
            and _live_ok(s, proto)
            and int(proto.get('n_reps', 0)) >= MIN_QUEUE_REPS
            and _live_protocol_ok(proto, spawn.get('nominal'),
                                  cap.get('nominal'), design)
            and _live_protocol_ok(proto, spawn.get('stress'),
                                  cap.get('stress'), design, level='STRESS'))


def _queue_big_enough(d):
    return _dir_summary_ok(d, _queue_ok) and _sidecar_clean(_side(d))


GOF_DESIGNS = ('in_sample', 'held_out')
GOF_PERIODS = ('current', 'prior')
GOF_SCRIPT = LIVE_RUNNERS['gof']

# The category row's test, read from the module that runs it. Runs from
# before it moved whole baskets compared item purchases as if each were an
# independent observation, which rejects a model that reproduces the data
# exactly far more often than alpha; they carry neither this label nor a
# permutation count, and cannot set the category numbers.
GOF_CATEGORY_SCRIPT = 'dataset_validation.py'
_GOF_CATEGORY_NAMES = ('CATEGORY_N_PERM', 'CATEGORY_TEST_KIND')
_GOF_RUNNER_NAMES = ('COHORT', 'LIST_CHAIN_DRAWN', 'LIST_CHAIN_SELECTION',
                     'VISITS_FILE', 'ANONYMOUS_LIST_KEYS')


def _gof_category_row(block):
    """The pooled category-shares row of one design, or None."""
    return next((t for t in block.get('pooled') or []
                 if 'categor' in str(t.get('test', '')).lower()), None)


def _gof_category_ok(block):
    """The pooled category row ran as the basket-level permutation test at
    its full permutation count, with the naive item-level p beside it and
    the effect sizes (review R29)."""
    want = _script_constants(GOF_CATEGORY_SCRIPT, _GOF_CATEGORY_NAMES)
    row = _gof_category_row(block)
    try:
        return (row is not None
                and row.get('kind') == want['CATEGORY_TEST_KIND']
                and int(row.get('n_permutations') or 0)
                >= want['CATEGORY_N_PERM']
                and row.get('p_value') is not None
                and 'naive_p_value' in row
                and 'share_tv_distance' in row)
    except Exception:
        return False


def _gof_block_ok(b):
    """One design's replicated tests under the shared protocol, with the
    pooled tests present and the category row tested on whole baskets; the
    window's arrivals followed until they left (review R46) with the
    realised load recorded; and the list chain along its two tested links
    (review R29)."""
    proto = b.get('protocol', {})
    g = _script_constants(GOF_SCRIPT, _GOF_RUNNER_NAMES)
    chain = [t.get('test') for t in _get(b, 'list_chain', 'tests',
                                         default=[])]
    return (int(b.get('n_reps', 0)) >= MIN_GOF_REPS
            and _live_protocol_ok(proto, proto.get('spawn'), proto.get('cap'),
                                  _live_design('gof'))
            and proto.get('framing') == LIVE_FRAMING
            and proto.get('cohort') == g['COHORT']
            and _get(b, 'load', 'door_rate_per_s') is not None
            and chain == [g['LIST_CHAIN_DRAWN'], g['LIST_CHAIN_SELECTION']]
            and bool(b.get('pooled'))
            and _gof_category_ok(b))


GOF_REPLICA_ROWS = ('basket', 'revenue', 'category')


def _gof_ok(s):
    """Replicated goodness-of-fit run under the shared protocol, with the
    pooled tests present for the in-sample design (the top level) and for
    the held-out one, the category row tested on whole baskets in both,
    and the record of what each design's store was built from and tested
    against: both periods (today's list law and cleaning, the prior one
    cut before cleaning), and per design the products it placed and the
    reference invoices holding one of them, the anonymous list shares, and
    for the held-out design the category test between the two periods' own
    invoices and the year shift on the fixed assortment (review R16).
    Each design must also carry its size-matched replicas of the store
    period for all three rows: the only yardstick on the simulator's own
    sample size, so a run without them cannot set the held-out comparison.
    And every cohort visit written to the visits file."""
    design = s.get('design')
    if not (isinstance(design, dict) and isinstance(s.get('held_out'), dict)):
        return False
    periods = design.get('periods')
    ho = design.get('held_out') or {}
    shift = ho.get('year_shift') or {}
    anon_keys = _script_constants(GOF_SCRIPT, ('ANONYMOUS_LIST_KEYS',))[
        'ANONYMOUS_LIST_KEYS']
    vf = s.get('visits_file') or {}
    n_cohort = sum(int(_get(b, 'load', 'cohort_visits', default=0))
                   for b in (s, s['held_out']))
    return (_gof_block_ok(s) and _gof_block_ok(s['held_out'])
            and all(isinstance(design.get(k), dict) for k in GOF_DESIGNS)
            and isinstance(periods, dict)
            and all(isinstance(periods.get(p), dict) for p in GOF_PERIODS)
            and all(_live_data_ok(periods.get(p)) for p in GOF_PERIODS)
            and isinstance(periods['prior'].get('reversals_across_cut'), dict)
            and isinstance(shift.get('category'), dict)
            and isinstance(ho.get('year_shift_fixed_assortment'), dict)
            and all(k in design[dd] for dd in GOF_DESIGNS
                    for k in anon_keys)
            and all(isinstance(((design.get(dd) or {}).get('replica')
                                or {}).get(row), dict)
                    for dd in GOF_DESIGNS for row in GOF_REPLICA_ROWS)
            and int(vf.get('n_records', -1)) == n_cohort)


def _gof_big_enough(d):
    if not (_dir_summary_ok(d, _gof_ok) and _sidecar_clean(_side(d))):
        return False
    try:
        s = _summ(d)
        vf = s['visits_file']
        with open(os.path.join(d, vf['path']), encoding='utf-8') as f:
            lines = sum(1 for line in f if line.strip())
        return lines == int(vf['n_records'])
    except Exception:
        return False


# The protocol study's outcomes compared after one and two warm-ups, and
# their macro tags.
PROTO_OUTCOMES = (('revenue_per_visit', 'RevPerVisit'),
                  ('conversion', 'Conversion'),
                  ('mean_basket_paying', 'Basket'),
                  ('revenue_per_paying_visit', 'RevPerPaying'),
                  ('cohort_mean_visit_all_s', 'VisitLength'),
                  ('mean_queue_wait_s', 'QueueWait'),
                  ('perimeter_ratio', 'PerimRatio'))
_PROTO_NAMES = ('NOMINAL_GRID', 'STRESS_GRID', 'CAP_GRID', 'N_REPS',
                'STRESS_REPS', 'CAP_REPS', 'RUN_S', 'STRESS_RUN_S',
                'NOMINAL_CAP', 'HEADLINE_OUTCOMES')
# The live runners the study compares its derived protocol with (its
# ``runner_constants`` keys, the runner modules' names) and the family
# whose constants each one is, as the live validators read them.
PROTO_RUNNERS = {'run_abm_diagnostics': 'abm',
                 'run_structural_sensitivity': 'structural',
                 'measure_queueing': 'queue',
                 'run_validation_gof': 'gof'}


def _proto_runner_constants_current(s):
    """The study compared its derived protocol with the constants the live
    runners use today: every runner's recorded ``runner_constants`` equal
    to the constants in its source, the ones each live family is held to
    (``_live_design``). The study's ``matches``, ``differences`` and
    ``protocol_status`` describe the runners as they were when it ran; a
    study made against other constants -- before a retune of the runners,
    or after the runners were aligned with it -- would otherwise keep
    reporting a comparison with code that no longer exists (gotcha 14:
    changing a protocol constant retires every older artifact)."""
    rc = s.get('runner_constants')
    if not isinstance(rc, dict) or set(rc) != set(PROTO_RUNNERS):
        return False
    for runner, fam in PROTO_RUNNERS.items():
        got, want = rc.get(runner), _live_design(fam)
        if not (isinstance(got, dict) and set(got) == set(want)
                and all(_same(got[k], want[k]) for k in want)):
            return False
    return True


def _proto_ok(s):
    """The live protocol study (review R19) at the runner's full design --
    its rate, stress and cap grids, replications and run lengths -- with
    every protocol constant derived, the drift test run, the double warm-up
    drained like the runners' cohorts and compared on the outcomes they
    report, the status label, compared with the live runners' constants as
    they stand today, and today's store."""
    c = _script_constants(PROTO_SCRIPT, _PROTO_NAMES)
    des = s.get('design') or {}
    nom, st = des.get('nominal') or {}, des.get('stress') or {}
    dbl = s.get('double_warmup') or {}
    return (des.get('mode') == LIVE_MODE
            and bool(np.isclose(float(des.get('dt', 0)), LIVE_DT))
            and des.get('framing') == LIVE_FRAMING
            and _same(nom.get('grid'), list(c['NOMINAL_GRID']))
            and int(nom.get('cap', -1)) == c['NOMINAL_CAP']
            and int(nom.get('reps', 0)) >= c['N_REPS']
            and float(nom.get('run_s', 0)) >= c['RUN_S']
            and _same(st.get('grid'), list(c['STRESS_GRID']))
            and _same(st.get('cap_grid'), list(c['CAP_GRID']))
            and int(st.get('rate_reps', 0)) >= c['STRESS_REPS']
            and int(st.get('cap_reps', 0)) >= c['CAP_REPS']
            and float(st.get('run_s', 0)) >= c['STRESS_RUN_S']
            and all(v is not None for v in (s.get('derived') or {}).values())
            and len(s.get('derived') or {}) == 6
            and int(_get(s, 'drift', 'n_reps', default=0)) > 1
            and dbl.get('cohort') == 'window_arrivals_drained'
            and all(k in (dbl.get('stats') or {})
                    for k in c['HEADLINE_OUTCOMES'])
            and bool(s.get('protocol_status'))
            and 'matches' in s
            and _proto_runner_constants_current(s)
            and _live_data_ok(s.get('data')))


def _proto_big_enough(d):
    return _dir_summary_ok(d, _proto_ok) and _sidecar_clean(_side(d))


def _heatmap_ok(s):
    """The seeded heat-map run at the nominal protocol and the figure's
    own seed, on today's store, drawn with text no smaller than the print
    minimum (review R52, R56)."""
    proto = s.get('protocol') or {}
    seed = _script_constants(HEATMAP_SCRIPT, ('FIGURE_SEED',))['FIGURE_SEED']
    return (_live_protocol_ok(proto, proto.get('spawn'), proto.get('cap'),
                              _live_design('abm'))
            and int(proto.get('seed', -1)) == int(seed)
            and _live_data_ok(s.get('data'))
            and float(_get(s, 'figure', 'min_font_pt_printed',
                           default=0)) >= 7.0
            and s.get('perimeter_interior_ratio_geometric_null') is not None)


def _heatmap_big_enough(d):
    return _dir_summary_ok(d, _heatmap_ok) and _sidecar_clean(_side(d))


def _profile_ok(s):
    """The UCI calendar profile, read as Figure C reads the workbook: both
    sheets, with the weekday and October-December records it quotes."""
    return (set(_get(s, 'data', 'sheets', default=[]))
            == {'Year 2009-2010', 'Year 2010-2011'}
            and len(s.get('weekday') or {}) == 7
            and s.get('fourth_quarter', {}).get('revenue_share') is not None)


def _profile_big_enough(d):
    return _dir_summary_ok(d, _profile_ok) and _sidecar_clean(_side(d))


def _profile_matches_figc(d, figc_dir):
    """True if the profile read the same workbook, byte for byte, as the
    Figure C run that sets Figure C's macros."""
    if not figc_dir:
        return False
    try:
        sha = _summ(d)['data']['source_sha256']
        with open(os.path.join(figc_dir, 'sidecar.json'), encoding='utf-8') as f:
            return f'"source_sha256": "{sha}"' in f.read()
    except Exception:
        return False


FIGURES_SCRIPT = 'experiments/make_paper_figures.py'
# The headless figure run's design, held to the script's argparse defaults
# exactly (review R50): the scenario, and the GA and Monte Carlo design the
# Fig. 9 caption quotes (seeds, population, generations, iterations, days).
FIGURES_DESIGN_ARGS = ('scenario_seed', 'n_items', 'n_seeds', 'n_gens',
                       'pop_size', 'mc_iters', 'mc_days')


def _figures_ok(s):
    """The headless figure run's design and checkout, stamped next to the
    realized scores it reports. The realized scores belong to one fixed
    scenario and one GA design -- the script's defaults, exactly, since a
    larger or smaller run is a different number, not a better version of
    the same one; a file the script stopped writing half-way has no Monte
    Carlo precision figure yet; and the scores and the precision figure
    depend on the objective, so the record must carry today's (the
    anchored drivers, the spend law) from a clean tree that did not move
    while the run was in flight (``_sidecar_clean`` on its provenance)."""
    return (isinstance(s.get('provenance'), dict)
            and _args_match(s.get('config'), FIGURES_SCRIPT,
                            exact=FIGURES_DESIGN_ARGS)
            and s.get('mc_rse_pct') is not None
            and _sidecar_clean(s['provenance'])
            and _model_current(s))


def latest(prefix, validator=None):
    """Newest artifact for ``prefix``; with ``validator``, newest that passes.

    Falling back to the newest *valid* run (rather than erroring on a fresh
    toy directory) keeps a smoke run from clobbering the paper while still
    letting a genuine re-run take effect.
    """
    dirs = sorted(glob.glob(os.path.join(RES, prefix + '*')))
    if validator is not None:
        dirs = [d for d in dirs if validator(d)]
    return dirs[-1] if dirs else None


def read_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


# --- Statistics ---------------------------------------------------------------

N_BOOT = 4000


def boot_ci(x, n=N_BOOT, seed=0, alpha=0.05):
    """Percentile bootstrap CI (i.i.d. resample). ``alpha`` sets the
    two-sided level (0.05 -> 95%; 0.01 -> 99% for Bonferroni)."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    m = np.array([rng.choice(x, x.size, replace=True).mean() for _ in range(n)])
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return float(np.percentile(m, lo)), float(np.percentile(m, hi))


def cluster_boot_ci(clustered, n=N_BOOT, seed=0, alpha=0.05):
    """Hierarchical (cluster) bootstrap over scenarios (audit R3.1):
    resample scenarios with replacement, carry ALL their seed-level
    paired differences, take the grand mean. This respects the fact
    that seeds within a scenario are not independent replicates of the
    shop population. ``clustered`` is a list of arrays (one per
    scenario). Returns (mean, lo, hi) at the requested two-sided level."""
    scen = [np.asarray(c, dtype=float) for c in clustered if len(c)]
    rng = np.random.default_rng(seed)
    k = len(scen)
    grand = np.concatenate(scen).mean()
    boots = np.empty(n)
    for b in range(n):
        idx = rng.integers(0, k, k)
        boots[b] = np.concatenate([scen[i] for i in idx]).mean()
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return float(grand), float(np.percentile(boots, lo)), float(np.percentile(boots, hi))


def runner_boot_ci(x, n=2000, seed=0, alpha=0.05):
    """The percentile bootstrap the experiment runners report
    (``experiments._common.bootstrap_ci``: 2000 resamples of integer
    indices under seed 0), reproduced here so a number the runner reports
    can be recomputed from its rows without importing the experiment
    stack. Returns (mean, lo, hi)."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    rng = np.random.default_rng(seed)
    means = np.empty(n)
    for k in range(n):
        means[k] = x[rng.integers(0, x.size, x.size)].mean()
    return (float(x.mean()), float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


def paired_clusters(by, a, b):
    """Paired differences ``a - b`` grouped by scenario, each carried with
    ``b``'s own revenue so a margin can also be read relative to what it is
    a margin over. ``by`` maps (scenario, seed) -> {method: revenue}; only
    the runs that scored both methods enter, in ``by``'s order."""
    clusters = defaultdict(list)
    for (scen, seed), v in by.items():
        if a in v and b in v:
            clusters[scen].append((v[a] - v[b], v[b]))
    return clusters


def paired_macros(stem, clusters, alpha):
    """The paired-difference macros of one Figure B row: the cluster
    bootstrap mean and interval at two-sided ``alpha``, whether the interval
    excludes zero, the mean as a share of the second method's revenue, and
    the win structure behind the average -- a mean difference says nothing
    about how often it went the other way. A run is one (scenario, seed)
    pair; a scenario is won when its own mean difference is positive.
    Returns (macros, significant, per-scenario differences)."""
    clustered = [[d for d, _ in c] for c in clusters.values()]
    comp_rev = np.array([b for c in clusters.values() for _, b in c])
    mean, lo, hi = cluster_boot_ci(clustered, alpha=alpha)
    sig = (lo > 0 or hi < 0)
    diffs = np.concatenate([np.asarray(c, dtype=float) for c in clustered])
    out = {
        stem: money(mean),
        f'{stem}CI': f"[{money(lo)}, {money(hi)}]",
        f'{stem}Sig': 'yes' if sig else 'no',
        f'{stem}Pct': f"{mean / max(comp_rev.mean(), 1e-9) * 100:+.2f}\\%",
        f'{stem}WinRuns': str(int((diffs > 0).sum())),
        f'{stem}WinScen': str(int(sum(np.mean(c) > 0 for c in clustered))),
    }
    return out, sig, clustered


def figc_comparator_stats(d, rows=None):
    """GA minus each equal-budget search on a real-data run, as the runner
    reports it: {method: (mean, lo, hi, pct)}, the 95% percentile bootstrap
    over the paired replicates and the mean as a share of the comparator's
    mean revenue.

    Read from the run's summary (or its sidecar, which carries the same
    results), so the macros quote the runner's own figures. A run whose
    record lacks them but whose results.csv has the comparator columns is
    recomputed from ``rows`` with the runner's bootstrap."""
    recorded = {}
    for name in ('summary.json', 'sidecar.json'):
        got = _get(_read_json(os.path.join(d, name)), 'results',
                   'comparators')
        if isinstance(got, dict) and got:
            recorded = got
            break
    out = {}
    for meth, (_, diff_col, rev_col) in FIGC_COMPARATORS.items():
        c = recorded.get(meth)
        if (isinstance(c, dict)
                and c.get('paired_mean_diff_GA_minus_X') is not None):
            out[meth] = (float(c['paired_mean_diff_GA_minus_X']),
                         float(c['ci_lo']), float(c['ci_hi']),
                         float(c['pct_diff']))
        elif rows and diff_col in rows[0] and rev_col in rows[0]:
            mean, lo, hi = runner_boot_ci([float(r[diff_col]) for r in rows])
            other = np.array([float(r[rev_col]) for r in rows])
            out[meth] = (mean, lo, hi, mean / max(float(other.mean()), 1e-6)
                         * 100.0)
    return out


def fisher_mean_rho(rhos):
    """Average correlations via the Fisher z-transform (audit R3.4)."""
    r = np.clip(np.asarray(rhos, dtype=float), -0.999, 0.999)
    z = np.arctanh(r)
    return float(np.tanh(z.mean()))


def holm_reject(pvals, alpha):
    """Holm's step-down procedure: which of the hypotheses with p-values
    ``pvals`` are rejected at family-wise level ``alpha``. The i-th
    smallest p (i from 0) is compared with alpha / (m - i), stopping at the
    first that fails."""
    p = np.asarray(pvals, dtype=float)
    m = p.size
    out = np.zeros(m, dtype=bool)
    for i, k in enumerate(np.argsort(p, kind='stable')):
        if p[k] > alpha / (m - i):
            break
        out[k] = True
    return out


def cochran_q_fisher_z(rhos, n):
    """Cochran's Q for heterogeneity of rank correlations, each from ``n``
    pairs, on the Fisher-z scale with the Spearman variance 1.06/(n - 3)
    (Fieller et al. 1957) for every one: {'Q', 'df', 'p', 'I2'}, with p
    from the chi-square on k - 1 df and I^2 = max(0, (Q - df) / Q)."""
    from scipy.stats import chi2
    z = np.arctanh(np.clip(np.asarray(rhos, dtype=float), -0.999, 0.999))
    w = (n - 3) / 1.06
    q = float(w * np.sum((z - z.mean()) ** 2))
    df = int(z.size - 1)
    return {'Q': q, 'df': df, 'p': float(chi2.sf(q, df)),
            'I2': max(0.0, (q - df) / q) if q > 0 else 0.0}


def _grid_regret_feasible(rows):
    """Figure A's as-built start, per scenario, against the reference
    mapped onto the shared feasible set (``oracle_R_feasible``), in
    percent. The start is one layout per scenario, so each scenario counts
    once."""
    out = {}
    for r in rows:
        ref_f = float(r['oracle_R_feasible'])
        out[r['scenario']] = ((ref_f - float(r['grid_R']))
                              / max(ref_f, 1e-9) * 100)
    return out


# Expected range of k independent normal samples in SD units (the
# control-chart d2 constants): what spread between setting means a sweep
# with no real effect shows anyway.
D2_RANGE = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534, 7: 2.704,
            8: 2.847, 9: 2.970, 10: 3.078}

N_POWER_SIMS = 2000


def gamma_poisson(rng, mean, phi, size):
    """Counts with mean ``mean`` and variance ``phi * mean``: Poisson when
    the replications are no wider than Poisson, otherwise a gamma mixture
    of Poissons at the dispersion they show."""
    if phi <= 1.0:
        return rng.poisson(mean, size).astype(float)
    scale = phi - 1.0
    lam = rng.gamma(shape=max(mean, 1e-9) / scale, scale=scale, size=size)
    return rng.poisson(lam).astype(float)


# --- Formatting ---------------------------------------------------------------

def money(v):
    s = f"{abs(v):,.0f}"
    return ("-" if v < 0 else "+") + "\\$" + s


def amount(v):
    """Unsigned currency amount (a level, not a difference)."""
    return "\\$" + f"{v:,.0f}"


def pounds(v):
    """A signed GBP difference (the real-data runs are in pounds)."""
    return money(v).replace('\\$', '\\pounds ')


def pounds_amount(v):
    """An unsigned GBP level."""
    return amount(v).replace('\\$', '\\pounds ')


def pounds2(v, signed=True):
    """A GBP amount to the penny, for quantities too small to round to
    whole pounds."""
    body = f"\\pounds {abs(float(v)):,.2f}"
    return (('-' if v < 0 else '+') + body) if signed else body


def pfmt(p):
    """A p-value at the precision the paper reports, without rounding a
    small one to 0.000. The bound is wrapped in \\ensuremath so the macro
    also works inside math, where the paper puts its p-values
    (``$p=\\StructChiP{}$``); a literal ``$...$`` would close that math.
    Every p-value macro goes through here (review R57)."""
    p = float(p)
    return f"{p:.3f}" if p >= 0.001 else "\\ensuremath{<0.001}"


def p_bound(p):
    """A p-value with its order of magnitude when it is below 0.001 --
    ``\\ensuremath{<10^{-23}}`` for 6e-24 -- where ``pfmt`` would only say
    <0.001; as ``pfmt`` otherwise."""
    p = float(p)
    if p >= 0.001:
        return pfmt(p)
    if p <= 0.0:
        return "\\ensuremath{<10^{-300}}"
    e = int(math.ceil(math.log10(p)))
    if 10.0 ** e == p:          # an exact power of ten is not below itself
        e += 1
    return f"\\ensuremath{{<10^{{{e}}}}}"


def level_pct(level):
    """A confidence level in percent, whole when it is whole (95\\%) and to
    one decimal otherwise (99.3\\%)."""
    return (f"{level:.0f}\\%" if abs(level - round(level)) < 0.05
            else f"{level:.1f}\\%")


def spct(v, nd=2):
    """A signed percentage."""
    return f"{float(v):+.{nd}f}\\%"


def upct(v, nd=1):
    """An unsigned percentage."""
    return f"{float(v):.{nd}f}\\%"


def pp(v, nd=2):
    """A signed difference of two percentages in percentage points
    (``+0.63\\,pp``), so it cannot be read as a percentage of anything."""
    return f"{float(v):+.{nd}f}\\,pp"


def count_of(k, n):
    """A share as the counts behind it (``199/200``): a rounded percentage
    of a count can overstate it -- 199 of 200 is not '100%' and 97.5% is
    not 'above 98%' (review R57)."""
    return f"{int(k):,}/{int(n):,}"


def share_count(frac, n):
    """The count behind a recorded share of ``n`` (a replica rejection
    rate, a percentile among replicas), as ``count_of``."""
    return count_of(int(round(float(frac) * int(n))), n)


def yesno(b):
    return 'yes' if b else 'no'


def tex_text(s):
    """Plain text for a macro body: TeX specials escaped."""
    out = str(s)
    for a, b in (('\\', '\\textbackslash{}'), ('_', '\\_'), ('%', '\\%'),
                 ('&', '\\&'), ('#', '\\#'), ('$', '\\$')):
        out = out.replace(a, b)
    return out


_UNITS = ('Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven',
          'Eight', 'Nine', 'Ten', 'Eleven', 'Twelve', 'Thirteen', 'Fourteen',
          'Fifteen', 'Sixteen', 'Seventeen', 'Eighteen', 'Nineteen')
_TENS = ('', '', 'Twenty', 'Thirty', 'Forty', 'Fifty', 'Sixty', 'Seventy',
         'Eighty', 'Ninety')


def num_word(n):
    """A small integer as a macro-safe word (LaTeX macro names hold no
    digits): 3 -> Three, 24 -> TwentyFour, 100 -> OneHundred."""
    n = int(n)
    if n < 0:
        return 'Minus' + num_word(-n)
    if n < 20:
        return _UNITS[n]
    if n < 100:
        return _TENS[n // 10] + (_UNITS[n % 10] if n % 10 else '')
    if n < 1000:
        return (_UNITS[n // 100] + 'Hundred'
                + (num_word(n % 100) if n % 100 else ''))
    raise ValueError(f'no macro word for {n}')


def _num_list(xs, fmt='{}'):
    xs = [fmt.format(x) for x in xs]
    return xs[0] if len(xs) == 1 else ', '.join(xs[:-1]) + ' and ' + xs[-1]


# --- Family macros --------------------------------------------------------------

def _wall_hours(seconds):
    """A recorded wall time in hours as a macro value (two decimals below
    an hour, one above), or None when the run recorded none."""
    try:
        v = float(seconds)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v) or v < 0:
        return None
    h = v / 3600.0
    return f"{h:.1f}" if h >= 1.0 else f"{h:.2f}"


def dir_macros(stem, d, side=None):
    """Which run set a family's numbers (review R50): its directory and
    the commit its sidecar names; and what it cost (review R53): the wall
    time the sidecar records, in hours, and the worker processes it ran
    with (the runners give the same numbers at any worker count, so the
    count only explains the wall time)."""
    side = side if side is not None else (_side(d) or {})
    out = {f'{stem}Dir': tex_text(os.path.basename(d))}
    sha = side.get('git_sha')
    if sha:
        out[f'{stem}Commit'] = tex_text(str(sha)[:7])
    wall = _wall_hours(side.get('wall_seconds'))
    if wall is not None:
        out[f'{stem}WallHours'] = wall
    workers = _get(side, 'args', 'workers',
                   default=side.get('workers'))
    if workers is not None:
        out[f'{stem}Workers'] = str(int(workers))
    return out


# The machine a run records (``_common.hardware_record``): the fields that
# must agree for one hardware statement to cover every family.
HARDWARE_FIELDS = ('processor', 'cpu_count', 'ram_gib', 'machine')


def hardware_macros(by_stem):
    """The compute statement (review R53) from the hardware every run that
    set a macro recorded, ``{stem: hardware record or None}``: one
    ``HardwareCPU`` / ``HardwareCPUCount`` / ``HardwareRAM`` /
    ``HardwareMachine`` when every such run was made on the same machine,
    else the same per family (``<Stem>HardwareCPU`` ...), so a mixed set is
    never summarised as one machine. ``HardwareSame`` says which;
    ``HardwareNRuns`` counts the runs behind it, and a run that recorded no
    hardware is counted in ``HardwareNUnrecorded``."""
    recs = {k: v for k, v in by_stem.items() if isinstance(v, dict)}
    out = {}
    unrecorded = len(by_stem) - len(recs)
    if unrecorded:
        out['HardwareNUnrecorded'] = str(unrecorded)
    if not recs:
        return out

    def _fields(prefix, h):
        f = {}
        if h.get('processor'):
            f[f'{prefix}CPU'] = tex_text(str(h['processor']))
        if h.get('cpu_count') is not None:
            f[f'{prefix}CPUCount'] = str(int(h['cpu_count']))
        if h.get('ram_gib') is not None:
            f[f'{prefix}RAM'] = f"{float(h['ram_gib']):.1f}"
        if h.get('machine'):
            f[f'{prefix}Machine'] = tex_text(str(h['machine']))
        return f

    distinct = {tuple(h.get(k) for k in HARDWARE_FIELDS)
                for h in recs.values()}
    out['HardwareNRuns'] = str(len(recs))
    out['HardwareSame'] = yesno(len(distinct) == 1)
    if len(distinct) == 1:
        out.update(_fields('Hardware', next(iter(recs.values()))))
    else:
        for stem, h in recs.items():
            out.update(_fields(f'{stem}Hardware', h))
    return out


def figa_macros(d):
    """Figure A: regret against the analytical reference, the Spearman
    agreement of the two objectives, the reference's restarts and restart
    curve (R36), the moved references per scenario (R64) and the
    closed-form re-scoring (R02)."""
    macros = {}
    rows = read_csv(os.path.join(d, 'results.csv'))
    side, s = _side(d), _summ(d)
    reg = np.array([float(r['regret_pct']) for r in rows])
    macros.update({
        'RegretMedian': f"{np.median(reg):.2f}\\%",
        'RegretMean': f"{np.mean(reg):.2f}\\%",
        'RegretMin': f"{np.min(reg):.2f}\\%",
        'RegretPFive': f"{np.percentile(reg, 5):.2f}\\%",
        'RegretPNinetyFive': f"{np.percentile(reg, 95):.2f}\\%",
        'RegretNScen': str(len({r['scenario'] for r in rows})),
        'RegretNSeeds': str(len({r['seed'] for r in rows})),
        'OptimumRecovered': f"{100 - np.mean(reg):.1f}\\%",
    })
    # The shape of the regret distribution, not only its middle: the
    # worst run of the design, and how much of the design sat inside
    # the tolerance the text quotes. The threshold travels with the
    # numbers, so a reworded sentence cannot quote a share that was
    # computed at a different one.
    inside = reg < REGRET_TOL_PCT
    macros.update({
        'RegretMax': f"{np.max(reg):.2f}\\%",
        'RegretTol': f"{REGRET_TOL_PCT:g}\\%",
        'RegretInsideTolRuns': str(int(inside.sum())),
        'RegretInsideTolFrac': f"{inside.mean() * 100:.1f}\\%",
        'RegretInsideTolCount': count_of(inside.sum(), inside.size),
    })
    # Reference-solver convergence diagnostic (audit R3.3): fraction of
    # runs / scenarios where the GA's analytical revenue exceeds the
    # analytical reference's own analytical revenue (i.e. the reference
    # under-converged on its own objective).
    ga = np.array([float(r['ga_true_R']) for r in rows])
    ref = np.array([float(r['oracle_R']) for r in rows])
    beat = ga > ref
    by_sc = defaultdict(list)
    for r, b in zip(rows, beat):
        by_sc[r['scenario']].append(bool(b))
    macros['RefBeatFrac'] = f"{beat.mean()*100:.1f}\\%"
    macros['RefBeatScenFrac'] = \
        f"{np.mean([any(v) for v in by_sc.values()])*100:.1f}\\%"
    macros['RefBeatRuns'] = count_of(beat.sum(), beat.size)
    macros['RefBeatScen'] = count_of(sum(any(v) for v in by_sc.values()),
                                     len(by_sc))
    # Where the search starts. The GA is initialized from the shelf-order
    # grid layout, whose analytical regret is measured against the same
    # reference and on the same scale as the GA's own, so the two can be
    # read next to each other. That layout does not depend on the seed,
    # so it is summarised over scenarios -- averaging over runs would
    # count each scenario's single starting layout once per seed.
    per_scen = {}
    for r in rows:
        r_ref = float(r['oracle_R'])
        per_scen[r['scenario']] = ((r_ref - float(r['grid_R']))
                                   / max(r_ref, 1e-9) * 100)
    grid_reg = np.array(list(per_scen.values()))
    macros['GridRegretMean'] = f"{grid_reg.mean():.2f}\\%"
    macros['GridRegretMedian'] = f"{np.median(grid_reg):.2f}\\%"
    macros['GridRegretNScen'] = str(len(grid_reg))
    # The same start against the reference mapped onto the shared feasible
    # set -- the basis of RegretFeas* and of Figure B's per-method
    # FigBAnalyticalRegret* -- so the start and the methods compared with
    # it are read on one basis (GridRegret* above is against the unrepaired
    # reference, the basis of Regret*). The start needs no mapping: it is
    # the repaired as-built layout already.
    grid_reg_f = np.array(list(_grid_regret_feasible(rows).values()))
    macros['GridRegretFeasMean'] = f"{grid_reg_f.mean():.2f}\\%"
    macros['GridRegretFeasMedian'] = f"{np.median(grid_reg_f):.2f}\\%"
    # The share of the start-to-reference gap each run closes (review
    # R07): most of the analytical revenue does not depend on the layout,
    # so the start already recovers nearly all of it and a regret quoted
    # as 'revenue recovered' reads as near-perfect whatever the search
    # did. The normalized share, (GA - start) / (reference - start), says
    # how much of what the layout can still earn the GA found; runs whose
    # start already is the reference have no gap and are left out (and
    # counted). The runs that end below their own start are counted too.
    start = np.array([float(r['grid_R']) for r in rows])
    gap = ref - start
    has_gap = gap > 1e-9 * np.maximum(np.abs(ref), 1.0)
    share = (ga[has_gap] - start[has_gap]) / gap[has_gap] * 100
    below = ga < start
    if share.size:
        macros.update({
            'FigAGapClosedMean': f"{share.mean():.1f}\\%",
            'FigAGapClosedMedian': f"{np.median(share):.1f}\\%",
            'FigAGapClosedPFive': f"{np.percentile(share, 5):.1f}\\%",
            'FigAGapClosedPNinetyFive':
                f"{np.percentile(share, 95):.1f}\\%",
            'FigAGapClosedMin': f"{share.min():.1f}\\%",
        })
    # The runs the share leaves out (no gap to close). The count of runs
    # it keeps is not emitted: the reference solver runs one restart from
    # this very layout, so its value is at least the start's and 'every
    # run has a gap' holds by construction -- a count that reads like
    # evidence and is none. The exclusions are the informative side.
    macros['FigAGapClosedExcludedRuns'] = count_of((~has_gap).sum(),
                                                   has_gap.size)
    macros['FigABelowStartRuns'] = count_of(below.sum(), below.size)
    # Regret against the reference mapped onto the GA's feasible set. The
    # headline regret is taken against the unrepaired reference, which
    # the GA's constraints can put out of reach; this is the part of the
    # same gap that the GA could actually have closed.
    ref_f = np.array([float(r['oracle_R_feasible']) for r in rows])
    reg_f = (ref_f - ga) / np.maximum(ref_f, 1e-9) * 100
    macros.update({
        'RegretFeasMean': f"{reg_f.mean():.2f}\\%",
        'RegretFeasMedian': f"{np.median(reg_f):.2f}\\%",
        'RegretFeasMax': f"{reg_f.max():.2f}\\%",
        # Runs whose reference layout the repair moved at all: the rest
        # have the two regrets equal by construction.
        'RegretFeasMovedRuns': str(int((ref_f < ref).sum())),
    })
    # The same as scenarios (review R64): the reference is one layout per
    # scenario, so counting runs counts each moved scenario once per seed.
    mv = s['reference_moved_by_repair']
    macros['RegretFeasMovedScen'] = str(int(mv['n_scenarios']))
    macros['RegretFeasMovedOfScen'] = str(int(mv['of_scenarios']))
    sp = os.path.join(d, 'spearman.csv')
    if os.path.exists(sp):
        sp_rows = read_csv(sp)
        rho = np.array([float(r['spearman_rho']) for r in sp_rows])
        # Fisher-z averaged mean (audit R3.4); report distribution, not
        # just a point, and the detectable |rho| at the number of layouts
        # the run actually sampled per scenario. The z-scale SE for a
        # Spearman coefficient is sqrt(1.06/(n-3)) (Fieller et al. 1957),
        # wider than the Pearson 1/sqrt(n-3).
        n_lay = int(side['args']['n_spearman_samples'])
        se_z = np.sqrt(1.06 / (n_lay - 3))
        # two-sided 0.05 detectable |rho| at 80% power:
        z_det = (1.96 + 0.84) * se_z
        rho_det = np.tanh(z_det)
        macros.update({
            'SpearmanMean': f"{fisher_mean_rho(rho):+.3f}",
            'SpearmanMedian': f"{np.median(rho):+.3f}",
            'SpearmanMax': f"{np.max(rho):+.3f}",
            'SpearmanMin': f"{np.min(rho):+.3f}",
            'SpearmanDetect': f"{rho_det:.2f}",
            'SpearmanNLayouts': str(n_lay),
        })
        # How the scenarios split, rather than only where their average
        # landed: a Fisher-z mean near zero is produced both by scenarios
        # that all agree on no relationship and by scenarios that disagree
        # in sign. The quartiles carry the spread the mean hides.
        # Significance is the per-scenario test the runner stored, at the
        # level quoted here.
        if 'spearman_p' in sp_rows[0]:
            pv = np.array([float(r['spearman_p']) for r in sp_rows])
            sig = pv < SPEARMAN_ALPHA
            macros.update({
                'SpearmanNScen': str(len(rho)),
                'SpearmanAlpha': f"{SPEARMAN_ALPHA:g}",
                'SpearmanNPos': str(int((sig & (rho > 0)).sum())),
                'SpearmanNNeg': str(int((sig & (rho < 0)).sum())),
                'SpearmanNNull': str(int((~sig).sum())),
            })
            # The same split with the per-scenario tests Holm-corrected
            # for the number of scenarios (review R48): family-wise level
            # SPEARMAN_ALPHA over one test per scenario.
            holm = holm_reject(pv, SPEARMAN_ALPHA)
            macros.update({
                'SpearmanHolmNPos': str(int((holm & (rho > 0)).sum())),
                'SpearmanHolmNNeg': str(int((holm & (rho < 0)).sum())),
                'SpearmanHolmNNull': str(int((~holm).sum())),
            })
        # Whether the agreement depends on the scenario (review R48):
        # Cochran's Q on the Fisher-z correlations, each weighted by the
        # inverse of the Spearman z-variance 1.06/(n-3) used above, and I^2,
        # the share of their spread beyond sampling error.
        q = cochran_q_fisher_z(rho, n_lay)
        macros.update({
            'SpearmanCochranQ': f"{q['Q']:.1f}",
            'SpearmanCochranQDf': str(q['df']),
            'SpearmanCochranQP': pfmt(q['p']),
            'SpearmanCochranQPBound': p_bound(q['p']),
            'SpearmanISquared': f"{100 * q['I2']:.0f}\\%",
        })
        macros['SpearmanQOne'] = f"{np.percentile(rho, 25):+.3f}"
        macros['SpearmanQThree'] = f"{np.percentile(rho, 75):+.3f}"
    # The same agreement with the GA's fitness at its exact mean (R02):
    # Figure A's regret involves no Monte Carlo, the Spearman sample does.
    cf = s['closed_form']
    n_scen = len({r['scenario'] for r in rows})
    macros.update({
        'SpearmanMeanCF': f"{cf['spearman_fisher_mean_cf']:+.3f}",
        'SpearmanMedianCF': f"{cf['spearman_median_cf']:+.3f}",
        'SpearmanMinCF': f"{cf['spearman_min_cf']:+.3f}",
        'SpearmanMaxCF': f"{cf['spearman_max_cf']:+.3f}",
        'SpearmanNPosCF': count_of(cf['spearman_n_positive_cf'], n_scen),
        'FigAFitMinusCFMedian':
            spct(cf['ga_mc_fit_minus_cf_pct']['median'], 3),
        'FigAFitMinusCFMax': spct(cf['ga_mc_fit_minus_cf_pct']['max'], 3),
        'FigACFUnchanged':
            yesno(cf['conclusions_unchanged']['all_unchanged']),
    })
    # The analytical reference (R36): its restarts, failures, and how much
    # the restarts past each checkpoint still raised its best value.
    o = s['oracle']
    macros['OracleRestarts'] = str(int(o['n_restarts']))
    macros['OracleFailedRestarts'] = str(int(o['n_failed_restarts']))
    macros['OracleNotConverged'] = str(int(o['n_not_converged']))
    for c, g in (o.get('gain_after_checkpoint_pct') or {}).items():
        # A checkpoint at (or past) the last restart has no restarts after
        # it, so its gain is zero by construction; emitted, it would read
        # as evidence that the reference had converged.
        if not g or int(c) >= int(o['n_restarts']):
            continue
        w = num_word(int(c))
        macros[f'OracleGainAfter{w}Median'] = f"{float(g['median']):.3f}\\%"
        macros[f'OracleGainAfter{w}Max'] = f"{float(g['max']):.3f}\\%"
        macros[f'OracleGainAfter{w}Improved'] = count_of(
            g['n_improved'], g['n_scenarios'])
    a = side['args']
    macros.update({'FigAGens': str(int(a['n_gens'])),
                   'FigAPop': str(int(a['pop_size'])),
                   'FigAItems': str(int(a['n_items'])),
                   'FigAMcIters': f"{int(a['mc_iters']):,}",
                   'FigAMcDays': str(int(a['mc_days']))})
    return macros


def figb_macros(d):
    """Figure B: every paired comparison under the manuscript's scenario
    cluster bootstrap with Bonferroni correction (audits R3.1, R3.2), the
    crossed, two-way and G-1 intervals beside it (R28, R63), the closed-
    form re-scoring (R02), the GA-SA equivalence and the smallest margin
    it would hold at, the popstart sensitivity row, the annealing schedule
    and the equal budget as recorded."""
    macros = {}
    rows = read_csv(os.path.join(d, 'results.csv'))
    side, s = _side(d), _summ(d)
    by = defaultdict(dict)      # (scenario, seed) -> {method: revenue}
    for r in rows:
        by[(r['scenario'], r['seed'])][r['method']] = float(r['mc_revenue'])
    # The family is the one the run records, and within it only the
    # comparators the artifact actually contains; a run without some
    # methods must not be corrected for them, and a sensitivity row the
    # run scored beside the family is never one of them. The divisor is
    # derived from the data, not hardcoded -- it was 5 before the two
    # equal-budget metaheuristics were added and is 7 now.
    family = set(side['comparison_family'])
    present = [m for m in FIGB_SHORT_NAMES if m in family
               and any('GA' in v and m in v for v in by.values())]
    n_comparisons = len(present)
    alpha_bonf = 0.05 / max(n_comparisons, 1)
    npairs, nscen, n_sig = 0, 0, 0
    for meth in present:
        # Paired differences grouped BY SCENARIO for the cluster
        # bootstrap; the share of the comparator's revenue lets the size
        # of a margin be read without knowing the scale of the scenarios.
        block, sig, clustered = paired_macros(
            f'GAvs{FIGB_SHORT_NAMES[meth]}',
            paired_clusters(by, 'GA', meth), alpha_bonf)
        macros.update(block)
        npairs = max(npairs, sum(len(c) for c in clustered))
        nscen = max(nscen, len(clustered))
        n_sig += int(sig)
    per_comp_level = 100 * (1 - alpha_bonf)
    macros['PairedN'] = str(npairs)
    macros['PairedNScen'] = str(nscen)
    macros['NComparators'] = str(n_comparisons)
    # How much of the comparator family the design separated at the
    # corrected level, so the text does not have to count the table.
    macros['NSigComparisons'] = str(n_sig)
    macros['BonfLevel'] = level_pct(per_comp_level)
    macros['CIscheme'] = (f"scenario-level cluster bootstrap "
                          f"({N_BOOT} resamples), Bonferroni-corrected "
                          f"{macros['BonfLevel']} per-comparison")

    # Sensitivity rows, outside the family: the annealer warm-started
    # from the popularity ranking, at the same budget, block and seeds.
    # They are reported on the family's estimator and level, so they
    # read like the table's rows, but they are neither counted in
    # NComparators nor in NSigComparisons.
    for meth, short in FIGB_SENSITIVITY.items():
        clusters = paired_clusters(by, 'GA', meth)
        if clusters:
            macros.update(paired_macros(f'GAvs{short}', clusters,
                                        alpha_bonf)[0])
    # What the warm start was worth to the annealer: the as-built SA
    # minus the popularity-started one, paired on the same runs. A
    # negative value means the popularity start helped.
    clusters = paired_clusters(by, 'simulated_annealing',
                               'simulated_annealing_popstart')
    if clusters:
        macros.update(paired_macros('SAStartEffect', clusters,
                                    alpha_bonf)[0])

    # Equivalence of the GA and the annealer (the family's SA row, from
    # the as-built start). The margin was fixed after a pilot, before the
    # re-run, at 0.1% of the annealer's mean paired revenue: about a third
    # of the smallest effect this design resolves (GA minus random search,
    # near 0.34% of the comparator's revenue). Two one-sided tests, each
    # at the family's Bonferroni level alpha' = 0.05 / NComparators,
    # reject non-equivalence exactly when the (1 - 2 alpha') cluster-
    # bootstrap percentile interval of the paired difference lies inside
    # [-margin, +margin]. The smallest margin at which it would still
    # hold is reported beside it (R28).
    clusters = paired_clusters(by, 'GA', 'simulated_annealing')
    if 'simulated_annealing' in present and clusters:
        clustered = [[x for x, _ in c] for c in clusters.values()]
        sa_rev = float(np.mean([b for c in clusters.values()
                                for _, b in c]))
        margin = EQUIV_MARGIN_FRAC * sa_rev
        _, lo, hi = cluster_boot_ci(clustered, alpha=2 * alpha_bonf)
        macros.update({
            'GAvsSAEquivMargin': amount(margin),
            'GAvsSAEquivMarginPct': f"{EQUIV_MARGIN_FRAC * 100:.1f}\\%",
            'GAvsSAEquivLevel': level_pct(100 * (1 - 2 * alpha_bonf)),
            'GAvsSAEquivCI': f"[{money(lo)}, {money(hi)}]",
            'GAvsSAEquivCIPct': (f"[{lo / max(sa_rev, 1e-9) * 100:+.2f}\\%, "
                                 f"{hi / max(sa_rev, 1e-9) * 100:+.2f}\\%]"),
            'GAvsSAEquiv': 'yes' if -margin < lo and hi < margin else 'no',
        })

    # The runner's other interval schemes (R28, R63): search seeds are
    # shared across scenarios, which the scenario bootstrap ignores. Each
    # comparison's crossed-bootstrap, two-way cluster-robust and G-1 t
    # intervals at the same Bonferroni level, and how many of the family
    # each separates.
    inf = s['inference']
    mc_comp = inf['mc']['comparisons']
    for scheme, tag in FIGB_SCHEMES:
        sig_n, tested = 0, 0
        for meth, rec in mc_comp.items():
            short = FIGB_SHORT_NAMES.get(meth)
            iv = rec.get(scheme) or {}
            if not short or iv.get('lo') is None:
                continue
            macros[f'GAvs{short}{tag}CI'] = \
                f"[{money(iv['lo'])}, {money(iv['hi'])}]"
            macros[f'GAvs{short}{tag}Sig'] = yesno(iv['excludes_zero'])
            sig_n += int(bool(iv['excludes_zero']))
            tested += 1
        if tested:
            macros[f'NSigComparisons{tag}'] = str(sig_n)
    # The same comparisons with every layout at its exact mean (R02).
    for meth, rec in inf['closed_form']['comparisons'].items():
        short = FIGB_SHORT_NAMES.get(meth)
        if not short:
            continue
        macros[f'GAvs{short}CF'] = money(rec['mean'])
        macros[f'GAvs{short}CFSig'] = \
            yesno(rec['cluster_bootstrap']['excludes_zero'])
    macros['FigBCFUnchanged'] = yesno(
        s['closed_form_conclusions_unchanged']['all_unchanged'])
    # The equivalence under each scheme, and the smallest margin at which
    # each would declare it, in currency and as a share of the annealer's
    # revenue.
    eq = inf['mc'].get('equivalence') or {}
    for scheme, tag in (('cluster_bootstrap', ''),) + FIGB_SCHEMES:
        e = (eq.get('schemes') or {}).get(scheme)
        if not e:
            continue
        macros[f'GAvsSAEquivMinMargin{tag}'] = amount(e['smallest_margin'])
        macros[f'GAvsSAEquivMinMargin{tag}Pct'] = \
            f"{100 * e['smallest_margin_frac']:.3f}\\%"
        if tag:
            macros[f'GAvsSAEquiv{tag}CI'] = \
                f"[{money(e['lo'])}, {money(e['hi'])}]"
            macros[f'GAvsSAEquiv{tag}'] = yesno(e['equivalent_at_margin'])
    eq_cf = _get(inf, 'closed_form', 'equivalence', 'schemes',
                 'cluster_bootstrap')
    if eq_cf:
        macros['GAvsSAEquivCF'] = yesno(eq_cf['equivalent_at_margin'])
        macros['GAvsSAEquivMinMarginCFPct'] = \
            f"{100 * eq_cf['smallest_margin_frac']:.3f}\\%"
    se_cf = s.get('sa_start_effect_closed_form') or {}
    if se_cf.get('mean_diff_popstart_minus_asbuilt') is not None:
        macros['SAStartEffectCF'] = \
            money(-float(se_cf['mean_diff_popstart_minus_asbuilt']))
    # Moved references per scenario (R64) and the reference's restarts.
    mv = s['reference_moved_by_repair']
    macros['FigBRefMovedScen'] = str(int(mv['n_scenarios']))
    macros['FigBRefMovedOfScen'] = str(int(mv['of_scenarios']))
    macros['FigBOracleRestarts'] = str(int(s['oracle_restarts']))
    macros['FigBOracleFailedRestarts'] = \
        str(int(s.get('oracle_failed_restarts', 0)))

    # Annealing schedule of the equal-budget comparator. The starting
    # temperature is calibrated per run from that run's own first
    # block, so the paper quotes the acceptance probability it was
    # calibrated to and the spread of the temperatures it produced.
    sched = side.get('sa_schedule', {})
    t0s = [float(x['sa_T0']) for x in sched.get('sa_T0', [])
           if x.get('sa_T0') is not None]
    if t0s:
        p0 = float(sched['sa_initial_accept'])
        macros['SAInitialAccept'] = f"{p0:.2f}"
        macros['SATZeroMedian'] = amount(float(np.median(t0s)))
        macros['SATZeroRange'] = (f"[{amount(min(t0s))}, "
                                  f"{amount(max(t0s))}]")
    # The equal-budget claim, as the run recorded it: the evaluations
    # each search method spent searching and then selecting its final
    # layout. Quoted only when every method and every run spent the
    # same number, which is what the claim says.
    for key, macro in (('evaluation_counts', 'SearchEvals'),
                       ('final_evaluation_counts', 'FinalSelectEvals')):
        spent = {int(n) for ns in (side.get(key) or {}).values()
                 for n in ns}
        if len(spent) == 1 and len(side[key]) > 1:
            macros[macro] = f"{spent.pop():,}"
    # Every method's analytical regret (review R07): its analytical revenue
    # against the reference's, per run, on the feasible set every method is
    # scored on (the reference row is the reference mapped through the
    # shared repair), mean over runs. Figure B's inference is on the MC
    # objective; these say where each method's layouts stand on the
    # synthetic scenarios' own ground truth, where the start already
    # recovers most of the revenue.
    an = defaultdict(dict)
    for r in rows:
        an[(r['scenario'], r['seed'])][r['method']] = \
            float(r['analytical_R'])
    for meth, short in (('GA', 'GA'),) + tuple(
            (m, n) for m, n in {**FIGB_SHORT_NAMES,
                                **FIGB_SENSITIVITY}.items()
            if m != 'oracle'):
        regs = [(v['oracle'] - v[meth]) / max(v['oracle'], 1e-9) * 100
                for v in an.values() if meth in v and 'oracle' in v]
        if regs:
            macros[f'FigBAnalyticalRegret{short}'] = \
                f"{np.mean(regs):.2f}\\%"
    # The order of those means is a claim only where a paired interval
    # backs it: per run, the GA's analytical regret minus each annealer's
    # (percentage points), its 95% scenario cluster bootstrap, and the runs
    # in which the GA's regret is the lower.
    for other, short in (('simulated_annealing', 'SA'),
                         ('simulated_annealing_popstart', 'SAPopStart')):
        by_scen = defaultdict(list)
        for (sc, _), v in sorted(an.items()):
            if {'GA', other, 'oracle'} <= set(v):
                ref = max(v['oracle'], 1e-9)
                by_scen[sc].append((v[other] - v['GA']) / ref * 100)
        if not by_scen:
            continue
        diffs = [by_scen[sc] for sc in sorted(by_scen)]
        mean, lo, hi = cluster_boot_ci(diffs)
        stem = f'FigBAnalyticalRegretGAvs{short}'
        macros[stem] = pp(mean)
        macros[f'{stem}CI'] = f"[{lo:+.2f}, {hi:+.2f}]\\,pp"
        macros[f'{stem}Sig'] = yesno(lo > 0 or hi < 0)
        macros[f'{stem}GALower'] = count_of(
            sum(x < 0 for c in diffs for x in c), sum(len(c) for c in diffs))
    a = side['args']
    macros.update({'FigBGens': str(int(a['n_gens'])),
                   'FigBPop': str(int(a['pop_size'])),
                   'FigBMcIters': f"{int(a['mc_iters']):,}",
                   'FigBMcDays': str(int(a['mc_days'])),
                   'FigBNMethods': str(len({r['method'] for r in rows}))})
    return macros


def asbuilt_regret_macros(figa_dir, figb_dir, rel_tol=1e-6):
    """The as-built start's analytical regret on Figure B's basis
    (review R07): per Figure B run, the start's analytical revenue against
    the run's reference row (the reference mapped through the shared
    repair), mean over runs -- exactly how FigBAnalyticalRegret* is taken
    for every method, so the start can be read beside them. Figure B does
    not score the start itself; Figure A records its analytical revenue
    (``grid_R``, the repaired as-built layout) for the same scenarios. The
    two are joined only when every scenario's Figure B reference equals
    Figure A's mapped reference (``oracle_R_feasible``) to ``rel_tol``,
    i.e. both runs built the same scenarios; otherwise ValueError."""
    rows_a = read_csv(os.path.join(figa_dir, 'results.csv'))
    start = {r['scenario']: float(r['grid_R']) for r in rows_a}
    ref_a = {r['scenario']: float(r['oracle_R_feasible']) for r in rows_a}
    regs = []
    for r in read_csv(os.path.join(figb_dir, 'results.csv')):
        if r['method'] != 'oracle':
            continue
        sc, ref_b = r['scenario'], float(r['analytical_R'])
        if sc not in start:
            raise ValueError(f"Figure B scenario {sc} is not in Figure A")
        if abs(ref_b - ref_a[sc]) > rel_tol * max(abs(ref_a[sc]), 1.0):
            raise ValueError(f"scenario {sc}: Figure B's reference "
                             f"{ref_b:.4f} is not Figure A's mapped "
                             f"reference {ref_a[sc]:.4f}")
        regs.append((ref_b - start[sc]) / max(ref_b, 1e-9) * 100)
    if not regs:
        raise ValueError("Figure B has no reference rows")
    return {'FigBAnalyticalRegretAsBuilt': f"{np.mean(regs):.2f}\\%"}


# Short names of the LHS comparisons, the same as Figure B's so a reader
# can put \LhsPopMedian next to \GAvsPop.
LHS_NAMES = {'popularity_rank': 'Pop', 'oracle': 'Oracle',
             'random_search': 'RandSearch', 'simulated_annealing': 'SA'}
# The composite's criteria and their macro tags (effective weights, R09).
CRITERION_TAGS = (('traffic', 'Traffic'), ('cross_merch', 'CrossMerch'),
                  ('impulse', 'Impulse'), ('flow', 'Flow'),
                  ('revenue_placement', 'RevPlace'),
                  ('section_compliance', 'Section'),
                  ('accessibility', 'Access'))


def _weight_sweep_macros(prefix, ws):
    out = {}
    for meth, st in ws['comparisons'].items():
        short = LHS_NAMES.get(meth)
        if not short:
            continue
        out.update({
            f'{prefix}{short}FracPos': f"{st['frac_positive'] * 100:.1f}\\%",
            f'{prefix}{short}PosCount': count_of(st['n_positive'],
                                                 st['n_values']),
            f'{prefix}{short}Median': money(st['median']),
            f'{prefix}{short}PFive': money(st['p5']),
            f'{prefix}{short}PNinetyFive': money(st['p95']),
            f'{prefix}{short}SignFlips':
                f"{int(st['n_pairs_sign_flip'])}/{int(st['n_pairs'])}",
            f'{prefix}{short}AlwaysPos':
                f"{int(st['n_pairs_always_positive'])}/{int(st['n_pairs'])}",
        })
    return out


def lhs_macros(d):
    """The elasticity-band LHS and the weight sweeps: every comparison's
    share of positive draws with the counts and the pair structure behind
    it (R38), the basket corners and the short-path variant (R37, R38),
    the effective weights and both standardized sweeps (R09), and the
    design (R54: the search horizon beside the projection horizon)."""
    macros = {}
    s = _summ(d)
    macros.update({
        'LhsFracPos': f"{s['frac_positive'] * 100:.1f}\\%",
        'LhsMedian': money(s['lift_median']),
        'LhsPFive': money(s['lift_p5']),
        'LhsPNinetyFive': money(s['lift_p95']),
        'LhsDraws': str(s['n_draws_total']),
    })
    # Design resolution vs evaluation count (audit R67): the LHS covers
    # the 3-D elasticity box with n_lhs_points; scenarios x seeds
    # replicate each point and must not be quoted as coverage.
    macros['LhsPoints'] = str(int(s['n_lhs_points']))
    macros['LhsNScen'] = str(int(s['n_scenarios']))
    macros['LhsNSeeds'] = str(int(s['n_seeds']))
    # The budget behind every layout in the sweep. The sweep re-scores
    # fixed layouts analytically rather than re-running the search per
    # draw, so this is the budget the layouts came from, and it is
    # lighter than the headline runs' -- stating it keeps the two from
    # being read as the same experiment. The search horizon (mc_days) is
    # shorter than the projection horizon (R54).
    cfg = s.get('config', {})
    macros['LhsGens'] = str(int(cfg['n_gens']))
    macros['LhsPop'] = str(int(cfg['pop_size']))
    macros['LhsIters'] = f"{int(cfg['mc_iters']):,}"
    macros['LhsSearchDays'] = str(int(cfg['mc_days']))
    macros['LhsItems'] = str(int(cfg['n_items']))
    # The projection horizon the lifts are summed over, counted the way
    # the MC engine counts it (weekend days by their position in the
    # week), so the text and the closed form cannot disagree.
    if s.get('horizon_days') is not None:
        macros['LhsHorizonDays'] = str(int(s['horizon_days']))
        macros['LhsHorizonCustHours'] = \
            f"{float(s['horizon_customer_hours']):.0f}"
    macros['LhsOracleRestarts'] = str(int(s['oracle']['n_restarts']))
    # The same sweep over every headline comparison, not just the GA
    # against the popularity baseline: the elasticity bands move all of
    # the reported differences, so each one gets its own share of
    # positive draws and its own spread -- as the counts behind it, and
    # per (scenario, seed) pair: how many pairs keep one sign across the
    # whole design and how much the size of the lift moves inside a pair
    # (R38; the sign is near-structural, the size is not).
    for meth, st in (s.get('comparisons') or {}).items():
        short = LHS_NAMES.get(meth)
        if not short:
            continue
        p = st['pairs']
        macros.update({
            f'Lhs{short}FracPos': f"{st['frac_positive'] * 100:.1f}\\%",
            f'Lhs{short}PosCount': count_of(st['n_positive'],
                                            st['n_draws_total']),
            f'Lhs{short}Median': money(st['lift_median']),
            f'Lhs{short}PFive': money(st['lift_p5']),
            f'Lhs{short}PNinetyFive': money(st['lift_p95']),
            f'Lhs{short}PairsAllPos':
                count_of(p['n_pairs_all_draws_positive'], p['n_pairs']),
            f'Lhs{short}PairsMixed':
                count_of(p['n_pairs_mixed_sign'], p['n_pairs']),
            f'Lhs{short}PairsAllNeg':
                count_of(p['n_pairs_all_draws_nonpositive'], p['n_pairs']),
            f'Lhs{short}PairRangeMedian':
                amount(p['within_pair_range_median']),
            f'Lhs{short}PairRangeMax': amount(p['within_pair_range_max']),
            f'Lhs{short}PairRangeRelMedian':
                upct(100 * p['within_pair_range_rel_median'], 0),
            f'Lhs{short}PairPctRangeMedian':
                f"{p['within_pair_pct_range_median']:.2f}",
            f'Lhs{short}PairPctRangeMax':
                f"{p['within_pair_pct_range_max']:.2f}",
        })
    macros['LhsNPairs'] = str(int(s.get('n_pairs', 0)))
    # Basket corners below the band (R38): the cited per-metre figure alone
    # (0.02) and no basket effect (0), conversion and impulse at their
    # midpoints; and the variant without the short-path criteria in the
    # basket driver (R37), at the midpoints and over the LHS draws.
    ce = s['closed_form_extras']
    n_pairs = int(ce['n_pairs'])
    for key, tag in (('0.02', 'BskTwo'), ('0', 'BskZero')):
        corner = (ce.get('basket_corners') or {}).get(key)
        if not corner:
            continue
        for meth, st in corner['comparisons'].items():
            short = LHS_NAMES.get(meth)
            if not short:
                continue
            macros.update({
                f'Lhs{tag}{short}Pos': count_of(st['n_positive'], st['n']),
                f'Lhs{tag}{short}Median': money(st['median']),
                f'Lhs{tag}{short}Min': money(st['min']),
                f'Lhs{tag}{short}Max': money(st['max'])})
    for meth, st in (ce.get('midpoint') or {}).items():
        short = LHS_NAMES.get(meth)
        if short:
            macros[f'LhsMid{short}Pos'] = count_of(st['n_positive'], st['n'])
            macros[f'LhsMid{short}Median'] = money(st['median'])
    sp = ce['short_path_variant']
    for meth, st in sp['midpoint'].items():
        short = LHS_NAMES.get(meth)
        if short:
            macros[f'LhsNoFlow{short}Pos'] = count_of(st['n_positive'],
                                                      st['n'])
            macros[f'LhsNoFlow{short}Median'] = money(st['median'])
    for meth, st in sp['lhs'].items():
        short = LHS_NAMES.get(meth)
        if short:
            macros[f'LhsNoFlow{short}LhsPos'] = count_of(st['n_positive'],
                                                         st['n'])
            macros[f'LhsNoFlow{short}LhsFracPos'] = \
                upct(100 * st['frac_positive'], 1)
            macros[f'LhsNoFlow{short}PairsAllPos'] = \
                count_of(st['n_pairs_all_positive'], n_pairs)
            macros[f'LhsNoFlow{short}Min'] = money(st['min'])
            macros[f'LhsNoFlow{short}Max'] = money(st['max'])
    # The effective weights (R09): nominal weight x the criterion's SD over
    # repaired random layouts, as a share of the positive criteria's
    # total, averaged over scenarios; and the SDs themselves.
    ew = s['effective_weights']
    for crit, tag in CRITERION_TAGS:
        share = (ew.get('mean_share') or {}).get(crit)
        if share is not None and np.isfinite(float(share)):
            macros[f'EffShare{tag}'] = upct(100 * float(share), 1)
        sd = (ew.get('mean_sd') or {}).get(crit)
        if sd is not None:
            macros[f'EffSD{tag}'] = f"{float(sd):.4f}"
    per = ew.get('per_scenario') or []
    if per:
        macros['EffWeightNLayouts'] = str(int(per[0]['n_layouts']))
    # The weight sweep: the same layouts re-scored under spatial weight
    # vectors drawn over the simplex, with everything else held. Each
    # comparison gets the same summary as under the elasticity bands,
    # plus how many (scenario, seed) pairs saw their lift take both
    # signs -- a lower bound, since the draws only sample the simplex --
    # and the same on both standardized scales (R09).
    ws = s['weight_sweep']
    macros.update(_weight_sweep_macros('LhsWeight', ws))
    for key, tag in (('weight_sweep_standardized', 'Std'),
                     ('weight_sweep_standardized_unit', 'StdUnit')):
        macros.update(_weight_sweep_macros(f'LhsWeight{tag}', s[key]))
    # The design, and what it holds fixed: the five spatial weights
    # share their total, while section compliance, accessibility and
    # both penalties keep their values and the elasticities sit at
    # their band midpoints.
    wd = ws['design']
    fixed = wd.get('fixed') or {}
    macros['LhsWeightDraws'] = str(int(wd['n_weight_draws']))
    macros['LhsWeightPairs'] = str(int(ws['n_pairs']))
    n_values = {int(st['n_values']) for st in ws['comparisons'].values()
                if st.get('n_values') is not None}
    if len(n_values) == 1:
        macros['LhsWeightValues'] = f"{n_values.pop():,}"
    macros['LhsWeightSpatialTotal'] = f"{float(wd['spatial_total']):.2f}"
    if fixed.get('section_compliance') is not None:
        macros['LhsWeightFixedSection'] = \
            f"{float(fixed['section_compliance']):.2f}"
    if fixed.get('accessibility') is not None:
        macros['LhsWeightFixedAccess'] = \
            f"{float(fixed['accessibility']):.2f}"
    if fixed.get('overlap_penalty') is not None:
        macros['LhsWeightFixedPenOverlap'] = \
            f"{float(fixed['overlap_penalty']):g}"
    if fixed.get('bottleneck_penalty') is not None:
        macros['LhsWeightFixedPenBottleneck'] = \
            f"{float(fixed['bottleneck_penalty']):g}"
    return macros


def figc_macros(d):
    """Figure C: the lift and the GA's margins over the equal-budget
    searches on held-out paired replicates, their closed-form values (R02),
    the band-box range and the conversion scan with its clamp threshold
    (R25), the aisle rule's record (R06), the cleaning counts and the
    anonymous invoices' shares (R27), and the calibration descriptors."""
    macros = {}
    rows = read_csv(os.path.join(d, 'results.csv'))
    side, s = _side(d), _summ(d)
    diffs = np.array([float(r['diff']) for r in rows])
    base = np.array([float(r['baseline_revenue']) for r in rows])
    opt = np.array([float(r['optimized_revenue']) for r in rows])
    lo, hi = boot_ci(diffs)
    pct = diffs.mean() / max(base.mean(), 1e-9) * 100
    # Achieved relative standard error of the mean (audit R1.6): the MC
    # estimator noise on the reported optimized mean.
    rse = (opt.std(ddof=1) / np.sqrt(len(opt))) / max(opt.mean(), 1e-9) * 100
    # RSE of the LIFT itself (audit R21.2b). The level RSE above is much
    # smaller because the paired difference cancels the shared revenue
    # base; the lift is the estimand the sentence is about, so report its
    # own precision next to it rather than letting the reader borrow the
    # level's.
    lift_rse = (diffs.std(ddof=1) / np.sqrt(len(diffs))) \
        / max(abs(diffs.mean()), 1e-9) * 100
    macros.update({
        'FigCLift': pounds(float(diffs.mean())),
        'FigCLiftCI': f"[{pounds(lo)}, {pounds(hi)}]",
        'FigCLiftPct': f"{pct:+.2f}\\%",
        'FigCReps': str(len(rows)),
        'FigCRSE': f"{rse:.2f}\\%",
        'FigCLiftRSE': f"{lift_rse:.2f}\\%",
    })
    # The GA against the equal-budget searches on the same store, paired
    # on the same held-out replicates: the runner's 95% percentile
    # bootstrap per comparison (uncorrected), and the difference as a
    # share of the comparator's mean revenue. They say whether the lift
    # over the as-built store needs the GA or only needs search.
    for meth, (mean_c, lo_c, hi_c, pct_c) in \
            figc_comparator_stats(d, rows).items():
        short = FIGC_COMPARATORS[meth][0]
        macros.update({
            f'FigCGAvs{short}': pounds(mean_c),
            f'FigCGAvs{short}CI': f"[{pounds(lo_c)}, {pounds(hi_c)}]",
            f'FigCGAvs{short}Pct': f"{pct_c:+.2f}\\%",
            f'FigCGAvs{short}Sig': 'yes' if lo_c > 0 or hi_c < 0 else 'no',
        })
    # Every conclusion with the layouts at their exact means (R02).
    res = s['results']
    cf, chk = res['closed_form'], res['closed_form_check']
    macros['FigCLiftCF'] = pounds(chk['lift']['cf'])
    macros['FigCLiftCFPct'] = spct(cf['lift_pct']['optimized'])
    macros['FigCLiftCFInsideCI'] = yesno(chk['lift']['cf_inside_mc_ci'])
    for key, short in (('random_search', 'RandSearch'),
                       ('simulated_annealing', 'SA')):
        macros[f'FigCGAvs{short}CF'] = pounds(chk[key]['cf'])
    macros['FigCCFSignsUnchanged'] = yesno(chk['all_signs_unchanged'])
    # The lift re-scored where the paper's inputs are assumptions (R25):
    # over the elasticity band box (every corner, the band's own corners,
    # and the basket elasticity at the cited 0.02 and at 0), and over the
    # assumed conversion rate, which the lift does not depend on only
    # below the rate at which the 0.99 clamps start to bind.
    bb = cf['band_box']
    for key, tag in (('range_all', 'BandBox'),
                     ('range_band_corners', 'BandCorners'),
                     ('range_bsk_0.02', 'BskTwo'), ('range_bsk_0', 'BskZero')):
        r = (bb.get(key) or {}).get('optimized')
        if r:
            macros[f'FigC{tag}Min'] = spct(r['min'])
            macros[f'FigC{tag}Max'] = spct(r['max'])
    cs = cf['conversion_scan']
    onset = cs.get('clamp_onset')
    macros['FigCClamp'] = f"{float(cs['clamp']):.2f}"
    macros['FigCClampOnset'] = (f"{float(onset):.3f}" if onset is not None
                                else 'none')
    macros['FigCConvRef'] = f"{float(cs['reference_rate']):.2f}"
    change = (cs.get('max_abs_change_below_onset_pp') or {}).get('optimized')
    if change is not None:
        macros['FigCConvScanMaxChange'] = f"{float(change):.2f}"
    scan = list(zip(cs['grid'], cs['lift_pct']['optimized']))
    below = [(p, v) for p, v in scan if onset is None or p < onset]
    above = [(p, v) for p, v in scan if onset is not None and p >= onset]
    if below:
        macros['FigCConvScanRateLo'] = f"{below[0][0]:.2f}"
        macros['FigCConvScanLiftLo'] = spct(below[0][1])
        macros['FigCConvScanRateHi'] = f"{below[-1][0]:.2f}"
        macros['FigCConvScanLiftHi'] = spct(below[-1][1])
    if above:
        macros['FigCConvScanRateAbove'] = f"{above[-1][0]:.2f}"
        macros['FigCConvScanLiftAbove'] = spct(above[-1][1])
    # The impulse channel cannot move on a store without impulse fixtures.
    imp = [float((v.get('breakdown') or {}).get('impulse', 0.0))
           for v in (cf.get('scores') or {}).values()]
    if imp:
        macros['FigCImpulseInactive'] = yesno(max(abs(x) for x in imp) == 0.0)
    # What the aisle rule takes from the search space (R06).
    ar = s['feasibility']['aisle_rule']
    macros.update({
        'FigCAisleZonesNarrowed': count_of(ar['n_zones_narrowed'],
                                           ar['n_zones']),
        'FigCAislePairsKept': str(int(ar['n_pairs_kept_apart'])),
        'FigCAisleItemsNarrowed': count_of(ar['n_items_narrowed'],
                                           ar['n_items']),
        'FigCAisleItemsPinned': count_of(ar['n_items_pinned'],
                                         ar['n_items']),
        'FigCAisleAreaKept':
            upct(100 * float(ar['area_share_kept_geomean_unpinned']), 0),
        'FigCAisleMinM': f"{float(ar['min_aisle_m']):.1f}",
    })
    # The equal-budget claim as the run recorded it: search evaluations
    # (held equal by the runner) and the final-selection evaluations each
    # search spent on top of them.
    spent = side.get('evaluation_counts') or {}
    if spent and len({int(n) for n in spent.values()}) == 1:
        macros['FigCSearchEvals'] = f"{int(next(iter(spent.values()))):,}"
    final = side.get('final_evaluation_counts') or {}
    for meth, short in (('GA', 'GA'), ('random_search', 'RandSearch'),
                        ('simulated_annealing', 'SA')):
        if final.get(meth) is not None:
            macros[f'FigCFinalEvals{short}'] = f"{int(final[meth]):,}"
    if final and len({int(n) for n in final.values()}) == 1:
        macros['FigCFinalSelectEvals'] = \
            f"{int(next(iter(final.values()))):,}"
    a = side['args']
    macros.update({'FigCGens': str(int(a['n_gens'])),
                   'FigCPop': str(int(a['pop_size'])),
                   'FigCMcIters': f"{int(a['mc_iters']):,}",
                   'FigCMcDays': str(int(a['mc_days'])),
                   'FigCItemsPerCategory':
                       str(int(a['max_items_per_category']))})
    # Data-quality descriptors from the calibration sidecar (audit
    # R7.3/R7.5). What the workbook reader handed the adapter: the sheets
    # of this workbook overlap in time, so the rows the adapter saw are
    # fewer than the rows read; both are reported, since only the first
    # can be checked against the file itself.
    cs_ = side.get('calibration_summary', {})
    rd = side.get('reader', {})
    if rd.get('rows_read'):
        macros['FigCRowsRead'] = f"{int(rd['rows_read']):,}"
        macros['FigCRowsUsed'] = f"{int(rd['rows_after_dedup']):,}"
        macros['FigCRowsDropped'] = \
            f"{int(rd['cross_sheet_duplicates_dropped']):,}"
    # Rows left after the adapter's cleaning (cancellations with the
    # purchases they reverse, non-merchandise lines, missing ids,
    # non-positive quantity or price), which is what calibration ran on.
    prov = side.get('provenance', {})
    if prov.get('rows_kept') is not None:
        macros['FigCRowsKept'] = f"{int(prov['rows_kept']):,}"
    # The cleaning (R27): cancellation lines, the reversal pairs removed
    # with them, anonymous and unmatched cancellations; and the anonymous
    # invoices kept in the calibration.
    cl = _get(prov, 'extra', 'cleaning') or {}
    for key, macro in (('cancellation_lines', 'FigCCancelLines'),
                       ('reversal_pairs', 'FigCReversalPairs'),
                       ('cancellation_lines_anonymous', 'FigCCancelAnon'),
                       ('cancellation_lines_unmatched', 'FigCCancelUnmatched')):
        if cl.get(key) is not None:
            macros[macro] = f"{int(cl[key]):,}"
    if cl.get('reversed_revenue_removed') is not None:
        macros['FigCReversedRevenue'] = \
            pounds_amount(float(cl['reversed_revenue_removed']))
    an = _get(prov, 'extra', 'anonymous') or {}
    if an.get('anonymous_invoice_frac') is not None:
        macros['FigCAnonInvoiceFrac'] = \
            upct(100 * float(an['anonymous_invoice_frac']), 1)
        macros['FigCAnonRevenueFrac'] = \
            upct(100 * float(an['anonymous_revenue_frac']), 1)
    if 'basket_units_median' in cs_:
        macros['BasketUnitsMed'] = f"{cs_['basket_units_median']:.0f}"
        macros['BasketDistinctMed'] = f"{cs_['basket_distinct_median']:.0f}"
    if 'category_fallback_frac' in cs_:
        macros['CatFallbackFrac'] = \
            f"{cs_['category_fallback_frac']*100:.0f}\\%"
    if 'return_customer_rate' in cs_:
        macros['ReturnRate'] = f"{cs_['return_customer_rate']*100:.0f}\\%"
    # The store the calibration produced. Its size follows from the
    # assortment the data supports, so it is a result of the load, not a
    # setting, and the text should not restate it by hand.
    shop = side.get('shop_summary', {})
    if shop.get('sections') is not None:
        macros['FigCSections'] = str(int(shop['sections']))
        macros['FigCItems'] = str(int(shop['items_placed']))
    # The assortment the sections were built from. This is not the
    # section count: the engine also lays out service zones, so the store
    # has more sections than the data has categories.
    if cs_.get('n_unique_categories'):
        macros['FigCCategories'] = str(int(cs_['n_unique_categories']))
    # The two arrival rates the projection engine is driven at. Buyers
    # per open hour are observed in the invoices; visitors per open hour
    # are that rate divided by the assumed conversion rate, which is an
    # assumption and is flagged as one wherever the visitor figure is
    # used.
    if cs_.get('arrivals_per_hour') and cs_.get('visitors_per_hour'):
        macros['FigCBuyersPerHour'] = f"{cs_['arrivals_per_hour']:.1f}"
        macros['FigCVisitorsPerHour'] = f"{cs_['visitors_per_hour']:.1f}"
    if cs_.get('assumed_conversion_rate') is not None:
        macros['FigCAssumedConv'] = f"{cs_['assumed_conversion_rate']:.2f}"
    # The span the invoices cover, which is what the calibrated daily
    # rates are an average over.
    if cs_.get('span_seconds'):
        macros['FigCSpanDays'] = \
            f"{float(cs_['span_seconds']) / 86400.0:,.0f}"
    return macros


# Figure C's macros its no-anonymous sensitivity run (review R27) reports,
# under the FigCNoAnon stem in place of FigC: the lift and the GA's margins
# over the equal-budget searches, as Monte Carlo on held-out replicates and
# in closed form, and the store and rates the calibration without the
# anonymous invoices produced.
FIGC_NOANON_KEYS = ('Lift', 'LiftCI', 'LiftPct', 'LiftRSE', 'LiftCF',
                    'LiftCFPct', 'Reps',
                    'GAvsRandSearch', 'GAvsRandSearchCI', 'GAvsRandSearchPct',
                    'GAvsRandSearchSig', 'GAvsRandSearchCF',
                    'GAvsSA', 'GAvsSACI', 'GAvsSAPct', 'GAvsSASig',
                    'GAvsSACF', 'CFSignsUnchanged',
                    'Items', 'Sections', 'Categories', 'RowsKept',
                    'BuyersPerHour', 'VisitorsPerHour')


def figc_noanon_macros(d):
    """Figure C without the anonymous invoices (review R27): the headline's
    numbers re-made on a store calibrated without them, under the
    ``FigCNoAnon`` stem, plus the invoices, rows and revenue the
    calibration left out."""
    full = figc_macros(d)
    out = {f'FigCNoAnon{k}': full[f'FigC{k}'] for k in FIGC_NOANON_KEYS
           if f'FigC{k}' in full}
    an = _get(_side(d), 'provenance', 'extra', 'anonymous') or {}
    if an.get('n_anonymous_invoices_excluded') is not None:
        out['FigCNoAnonInvoicesExcluded'] = \
            f"{int(an['n_anonymous_invoices_excluded']):,}"
    if an.get('n_anonymous_rows_excluded') is not None:
        out['FigCNoAnonRowsExcluded'] = \
            f"{int(an['n_anonymous_rows_excluded']):,}"
    if an.get('anonymous_revenue_excluded') is not None:
        out['FigCNoAnonRevenueExcluded'] = \
            pounds_amount(float(an['anonymous_revenue_excluded']))
    return out


def seeds_macros(d):
    """Figure C over several search seeds: the runner's across-seed means,
    t-intervals over the seeds, ranges and win counts, under Monte Carlo
    and in closed form (R02), with the band box of every seed's lift and
    the lowest clamp onset (R25)."""
    macros = {}
    s = _summ(d)
    k = int(s['design']['n_search_seeds'])
    across = s['results']['across_seeds']
    macros['FigCSeedsN'] = str(k)
    macros['FigCSeedsReps'] = str(int(s['design']['n_mc_replicates']))
    macros['FigCSeedsSearchEvals'] = \
        f"{int(s['design']['budget_search_evals']):,}"
    macros['FigCSeedsLevel'] = level_pct(
        100.0 * float(across['lift_over_baseline']['ci_level']))
    for key, stem in FIGC_SEED_STEMS.items():
        a = across[key]
        lo, hi = float(a['ci_lo']), float(a['ci_hi'])
        macros.update({
            stem: pounds(float(a['mean'])),
            f'{stem}CI': f"[{pounds(lo)}, {pounds(hi)}]",
            f'{stem}Pct': f"{float(a['pct']):+.2f}\\%",
            f'{stem}PctCI': f"[{float(a['pct_ci'][0]):+.2f}\\%, "
                            f"{float(a['pct_ci'][1]):+.2f}\\%]",
            f'{stem}Sig': 'yes' if lo > 0 or hi < 0 else 'no',
            f'{stem}Wins': f"{int(a['n_ga_leads'])}/{k}",
            f'{stem}Min': pounds(float(a['min'])),
            f'{stem}Max': pounds(float(a['max'])),
        })
    cfa = s['results']['closed_form']
    for key, stem in FIGC_SEED_STEMS.items():
        a = cfa['across_seeds'][key]
        macros.update({
            f'{stem}CF': pounds(float(a['mean'])),
            f'{stem}CFPct': f"{float(a['pct']):+.2f}\\%",
            f'{stem}CFWins': f"{int(a['n_ga_leads'])}/{k}",
            f'{stem}CFSig': yesno(a['excludes_zero']),
        })
    macros['FigCSeedsCFUnchanged'] = \
        yesno(cfa['conclusions_unchanged']['all_unchanged'])
    bb = cfa.get('lift_band_box_pct')
    if bb:
        macros['FigCSeedsBandBoxMin'] = spct(bb['min'])
        macros['FigCSeedsBandBoxMax'] = spct(bb['max'])
    if cfa.get('clamp_onset_min') is not None:
        macros['FigCSeedsClampOnset'] = f"{float(cfa['clamp_onset_min']):.3f}"
    return macros


LAYOUT_FIGURE_STEM = 'figc_layouts'


#: What a layout figure record must carry for today's caption: the kinds of
#: the long moves and the flow tour, checked against the GA's flow score.
LAYOUT_MOVE_KEYS = ('n_moved_at_least_aisle', 'n_slot_exchange',
                    'n_along_run', 'slot_tol_m', 'along_run_tol_m')


def _layout_figure_ok(rec, figc_dir):
    """The layout figure's record (``make_layout_figure``): drawn from a
    clean checkout that did not move while it ran, from the Figure C run
    that sets Figure C's macros as that run is now (its directory and its
    summary's SHA-256), with both drawn layouts passing every floor-plan
    invariant. A record from before the figure counted slot exchanges and
    moves along a run, and drew the flow tour checked against the GA's own
    flow score, is refused: its caption numbers are not today's."""
    if not figc_dir or not isinstance(rec, dict):
        return False
    try:
        src = rec['source']
        digests = _summary_digests(os.path.join(figc_dir, 'summary.json'))
        tour = rec['flow_tour']
        return (rec['figure'] == LAYOUT_FIGURE_STEM
                and _sidecar_clean(rec['provenance'])
                and bool(rec['provenance'].get('git_sha'))
                and os.path.basename(os.path.normpath(src['figc_dir']))
                == os.path.basename(os.path.normpath(figc_dir))
                and src['summary_sha256'] in digests
                and rec['invariants_hold'] is True
                and all(isinstance(rec['invariants'].get(k), dict)
                        for k in ('baseline', 'optimized'))
                and all(k in rec['moves'] for k in LAYOUT_MOVE_KEYS)
                and tour.get('matches_ga_flow_score') is True
                and all(tour['length_m'].get(k) is not None
                        for k in ('baseline', 'optimized')))
    except Exception:
        return False


def layout_figure_macros(rec):
    """What the layout figure's caption quotes (review R06): how many
    fixtures the GA moved and how many by at least the minimum aisle --
    and, of those long moves, how many took another fixture's as-built
    slot (FigCLayoutMovedSlotExchange, within FigCLayoutSlotTol metres of
    its centre) and how many stayed in their own run of shelving
    (FigCLayoutMovedAlongRun, displaced across the long axis by less than
    FigCLayoutAlongRunTol metres), each over the long moves: on this store
    a long move is mostly an exchange of products within a run, not a move
    across an aisle. Then the moves' median and largest; the flow tour the
    figure draws (entrance, the FigCLayoutTourItems most-bought products,
    checkout) in both layouts, in metres; and the invariant check of both
    drawn layouts -- the smallest clearance between zones the store keeps
    apart against the required aisle, and the unshoppable (unreachable)
    fixtures."""
    m, inv, tour = rec['moves'], rec['invariants'], rec['flow_tour']
    n = int(m['n_fixtures'])
    n_long = int(m['n_moved_at_least_aisle'])
    return {
        'FigCLayoutMoved': count_of(m['n_moved'], n),
        'FigCLayoutMovedAtLeastAisle': count_of(n_long, n),
        'FigCLayoutMovedSlotExchange': count_of(m['n_slot_exchange'],
                                                n_long),
        'FigCLayoutMovedAlongRun': count_of(m['n_along_run'], n_long),
        'FigCLayoutSlotTol': f"{float(m['slot_tol_m']):g}",
        'FigCLayoutAlongRunTol': f"{float(m['along_run_tol_m']):g}",
        'FigCLayoutTourItems': str(int(tour['n_items'])),
        'FigCLayoutTourAsBuilt': f"{float(tour['length_m']['baseline']):.0f}",
        'FigCLayoutTourGA': f"{float(tour['length_m']['optimized']):.0f}",
        'FigCLayoutHighlight': f"{float(m['highlight_m']):g}",
        'FigCLayoutMoveMedian': f"{float(m['median_m']):.1f}",
        'FigCLayoutMoveMax': f"{float(m['max_m']):.1f}",
        'FigCLayoutZones': str(int(rec['store']['n_zones'])),
        'FigCLayoutMinClearance': f"{float(rec['min_clearance_m']):.2f}",
        'FigCLayoutRequiredClearance':
            f"{float(rec['required_clearance_m']):.1f}",
        'FigCLayoutUnshoppable': str(sum(int(v.get('n_unshoppable', 0))
                                         for v in inv.values())),
        'FigCLayoutInvariantsHold': yesno(rec['invariants_hold']),
        'FigCLayoutCommit': tex_text(str(rec['provenance']['git_sha'])[:7]),
    }


def rescore_macros(d):
    """Figure C's saved layouts re-scored in closed form
    (``run_figc_rescore``): the Omnichannel bundle's lower bound on visit
    conversion -- the only data-derived value against Figure C's assumed
    rate -- and every searched layout's lift at it, with the conversion
    scan's grid rates either side; and where the lift comes from: the
    share carried by the tuned abandonment channel (lift with it switched
    off), by the conversion and basket elasticities, and the flow
    criterion's share of the composite-score gain, for the GA's layout and
    as a range over the three searched layouts."""
    macros = {}
    s = _summ(d)
    b = s['conversion_bound']
    macros['FigCConvBound'] = f"{float(b['value']):.3f}"
    macros['FigCConvBoundFamily'] = tex_text(b['family'])
    macros['FigCConvBoundNFamilies'] = str(int(b['n_families']))
    macros['FigCConvBoundSHA'] = tex_text(
        str(b['provenance']['source_sha256'])[:12])
    # The families' purchase probabilities combined as if independent
    # (at least one family bought), which is why the bound is the largest
    # single family's instead: the combination reaches the calibrator's
    # 0.99 clamp (``dataset_calibration.calibrate_omnichannel``).
    if b.get('independence_estimate') is not None:
        ind = float(b['independence_estimate'])
        macros['FigCConvBoundIndependence'] = f"{ind:.2f}"
        macros['FigCConvBoundIndependenceSaturates'] = yesno(ind >= 0.99)
    at = s['lift_at_conversion_bound']
    for k, short in zip(RESCORE_LAYOUTS, ('', 'RandSearch', 'SA')):
        macros[f'FigCLiftAtConvBound{short}'] = spct(at['lift_pct'][k])
    macros['FigCLiftAtConvBoundChange'] = pp(
        float(at['lift_pct']['optimized'])
        - float(at['lift_pct_at_assumed']['optimized']))
    macros['FigCConvBoundClampsBind'] = yesno(at['clamps_bind'])
    for side, tag in (('below', 'Below'), ('above', 'Above')):
        br = (at.get('scan_bracket') or {}).get(side)
        if br:
            macros[f'FigCConvBoundScanRate{tag}'] = f"{float(br['rate']):.2f}"
            macros[f'FigCConvBoundScanLift{tag}'] = \
                spct(br['lift_pct']['optimized'])
    per = s['decomposition']['per_layout']
    ga = per['optimized']
    macros.update({
        'FigCLiftNoAbandon': spct(ga['lift_pct_no_abandonment_channel']),
        'FigCLiftAbandonShare': upct(100 * ga['abandonment_share'], 0),
        'FigCLiftAbandonPP': pp(ga['abandonment_pp']),
        'FigCLiftNoConvElasticity':
            spct(ga['lift_pct_no_conversion_elasticity']),
        'FigCLiftConvElasticityShare':
            upct(100 * ga['conversion_elasticity_share'], 0),
        'FigCLiftNoBskElasticity': spct(ga['lift_pct_no_basket_elasticity']),
        'FigCLiftBskElasticityShare':
            upct(100 * ga['basket_elasticity_share'], 0),
        'FigCScoreGain': f"{float(ga['score_gain']):.4f}",
        'FigCScoreGainFlowShare':
            upct(100 * ga['flow_share_of_score_gain'], 0),
        'FigCScoreGainCrossMerchShare': upct(
            100 * ga['criterion_contributions']['cross_merch']
            / ga['score_gain'], 0),
    })
    for key, stem in (('abandonment_share', 'FigCLiftAbandonShare'),
                      ('flow_share_of_score_gain', 'FigCScoreGainFlowShare')):
        vals = [float(per[k][key]) for k in RESCORE_LAYOUTS]
        macros[f'{stem}Min'] = upct(100 * min(vals), 0)
        macros[f'{stem}Max'] = upct(100 * max(vals), 0)
    return macros


def figures_macros(rs):
    """The headless figure run's realized elasticities at realized scores
    (audit R2.3) and the MC estimator's precision at the iteration count
    the experiments use."""
    macros = {
        'RealizedScoreBaseline': f"{rs['score_baseline']:.3f}",
        'RealizedScoreGA': f"{rs['score_ga']:.3f}",
        'RealizedConvLift': f"{rs['conv_lift_pct']:+.1f}\\%",
        'RealizedImpLift': f"{rs['impulse_lift_pct']:+.1f}\\%",
        'RealizedBskLift': f"{rs['basket_lift_pct']:+.1f}\\%",
    }
    if rs.get('score_anchor') is not None:
        macros['RealizedScoreAnchor'] = f"{float(rs['score_anchor']):.3f}"
    if 'mc_rse_pct' in rs:
        macros['McRSEatUsed'] = f"{rs['mc_rse_pct']:.2f}\\%"
        macros['McRSEIters'] = str(int(rs['mc_rse_iters']))
    macros['FiguresDir'] = 'figs/realized\\_scores.json'
    sha = _get(rs, 'provenance', 'git_sha')
    if sha:
        macros['FiguresCommit'] = tex_text(str(sha)[:7])
    # The design behind Figs. 8-10 and the realized scores (review R50):
    # what the Fig. 9 caption states, from the run itself.
    cfg = rs['config']
    macros.update({
        'FiguresSeeds': str(int(cfg['n_seeds'])),
        'FiguresGens': str(int(cfg['n_gens'])),
        'FiguresPop': str(int(cfg['pop_size'])),
        'FiguresMcIters': f"{int(cfg['mc_iters']):,}",
        'FiguresMcDays': str(int(cfg['mc_days'])),
        'FiguresItems': str(int(cfg['n_items'])),
        'FiguresScenarioSeed': str(int(cfg['scenario_seed'])),
    })
    wall = _wall_hours(rs.get('wall_seconds'))
    if wall is not None:
        macros['FiguresWallHours'] = wall
    return macros


def queue_macros(d):
    """Queue measurement (audit R1.7) at the nominal and stress loads,
    with the replication spread, and the shared live protocol."""
    macros = {}
    q = _summ(d)
    macros.update({
        'QueuePeakAgents': str(q['nominal']['peak_agents']),
        'QueueMaxOccNom': str(q['nominal']['max_lane_occupancy']),
        'QueueMaxOccStress': str(q['stress']['max_lane_occupancy']),
        'QueueBusyNom': f"{q['nominal']['busy_frac']*100:.0f}\\%",
        'QueueBusyStress': f"{q['stress']['busy_frac']*100:.0f}\\%",
        'QueueStressCap': str(q['stress']['cap']),
    })
    # Replication spread (audit R74). The nominal busy fraction varies
    # enormously run-to-run -- quoting it as a point estimate, which we
    # did, was not defensible; the nominal-vs-stress contrast is what
    # replicates.
    if q['nominal'].get('n_reps'):
        macros['QueueReps'] = str(int(q['nominal']['n_reps']))
        macros['QueueBusyNomSD'] = f"{q['nominal']['busy_frac_sd']*100:.0f}"
        macros['QueueBusyStressSD'] = \
            f"{q['stress']['busy_frac_sd']*100:.0f}"
    # Waiting-line statistics of the single-server FIFO lanes: mean
    # number waiting, server occupancy and mean wait to service start.
    if 'mean_waiting' in q['nominal']:
        for level, tag in (('nominal', 'Nom'), ('stress', 'Stress')):
            ql = q[level]
            macros[f'QueueMeanWaiting{tag}'] = f"{ql['mean_waiting']:.2f}"
            macros[f'QueueOcc{tag}'] = f"{ql['occupancy']*100:.0f}\\%"
            if ql.get('mean_wait_s') is not None:
                macros[f'QueueMeanWait{tag}'] = f"{ql['mean_wait_s']:.1f}"
    # On how many of the five outcomes the two levels' replications
    # overlap: the nominal level's largest replication at or above the
    # stress level's smallest.
    pn, ps = q['nominal'].get('per_rep'), q['stress'].get('per_rep')
    if pn and ps:
        keys = ('busy_frac', 'occupancy', 'mean_waiting', 'mean_wait_s',
                'max_lane_occupancy')
        overlap = sum(max(float(r[k]) for r in pn)
                      >= min(float(r[k]) for r in ps) for k in keys)
        macros['QueueLevelsOverlap'] = count_of(overlap, len(keys))
    proto = q.get('protocol', {})
    if proto.get('warmup_s') is not None:
        macros['QueueWarmup'] = f"{proto['warmup_s']:.0f}"
    if proto.get('collect_s') is not None:
        macros['QueueCollect'] = f"{proto['collect_s']:.0f}"
    # The protocol the live diagnostics share, read from the one run that
    # carries both loads: the fixed tick, and the arrival rate and
    # occupancy cap at each level. The stress cap is already reported
    # above as QueueStressCap and is not repeated under a second name.
    spawn, cap = proto.get('spawn') or {}, proto.get('cap') or {}
    if spawn and cap:
        macros['LiveDt'] = f"{float(proto['dt']):g}"
        macros['LiveSpawnNom'] = f"{float(spawn['nominal']):.2f}"
        macros['LiveSpawnStress'] = f"{float(spawn['stress']):.2f}"
        macros['LiveCapNom'] = str(int(cap['nominal']))
    # What each level turned away at the door. The stress level is the
    # one whose cap could bind, so the two shares belong next to the
    # rates rather than being left to the reader to assume away.
    for level, tag in (('nominal', 'Nom'), ('stress', 'Stress')):
        b = balked_pct(q.get(level, {}).get('load'))
        if b is not None:
            macros[f'QueueBalked{tag}'] = f"{b:.1f}\\%"
    return macros


def mcgt_macros(d):
    """The MC-objective ground truth (audit R4.3): the normal-budget GA's
    regret against a 10x-budget best-known optimum, its closed-form re-
    scoring (R02), the popularity-start reference's share (R50) and the
    design (R54: its own 8-item template and seed blocks)."""
    macros = {}
    s, side = _summ(d), _side(d)
    ratio = int(round(s['big_budget'] / max(s['normal_budget'], 1)))
    macros.update({
        'McRegretMedian': f"{s['mc_regret_median_pct']:.2f}\\%",
        'McRegretMean': f"{s['mc_regret_mean_pct']:.2f}\\%",
        'McRegretMax': f"{s['mc_regret_max_pct']:.2f}\\%",
        'McGTBudgetRatio': f"{ratio}\\times",
        'McGTNScen': str(s['n_scenarios']),
    })
    # The reference layout is chosen on one block of seeds and both
    # layouts are then re-estimated on a disjoint block, so a regret can
    # come out below zero on confirmation noise. Report the smallest one
    # and how many scenarios did, rather than describing the design as
    # non-negative.
    macros['McRegretMin'] = f"{s['mc_regret_min_pct']:.2f}\\%"
    macros['McRegretNegCount'] = str(int(s['n_negative_regret']))
    # The normal-budget GA layout is itself a candidate for the
    # best-known reference; where no larger search beat it, its regret is
    # zero by construction, so the count belongs next to the median.
    if s.get('n_best_known_is_ga_normal') is not None:
        macros['McGTBestIsGANormal'] = \
            str(int(s['n_best_known_is_ga_normal']))
    # Which search supplied the reference; the popularity-started annealer
    # is part of it by design (R50).
    rows = read_csv(os.path.join(d, 'results.csv')) \
        if _has_results_csv(d) else []
    if rows:
        macros['McGTBestFromPopStart'] = str(sum(
            r['best_known_source'] == MCGT_POPSTART for r in rows))
        # Every search that can supply it, so 'mostly annealers' is a
        # count: the as-built-started annealer at the big budget, both
        # annealers together, and the big-budget GA.
        src = [r['best_known_source'] for r in rows]
        macros['McGTBestFromSABig'] = str(sum(
            x == 'simulated_annealing_big' for x in src))
        macros['McGTBestFromAnnealing'] = str(sum(
            x.startswith('simulated_annealing') for x in src))
        macros['McGTBestFromGABig'] = str(sum(x == 'GA_big' for x in src))
    # The estimator behind the comparison: the iterations each evaluation
    # averaged over, and the size of the disjoint seed block both layouts
    # were re-estimated on before their regret was taken.
    macros['McGTIters'] = f"{int(s['mc_iters']):,}"
    a = side.get('args', {})
    if a.get('confirm_seeds'):
        macros['McGTConfirmSeeds'] = str(int(a['confirm_seeds']))
    macros['McGTItems'] = str(int(a['n_items']))
    macros['McGTNormalBudget'] = f"{int(s['normal_budget']):,}"
    macros['McGTBigBudget'] = f"{int(s['big_budget']):,}"
    macros['McGTMcDays'] = str(int(a['mc_days']))
    # The same regret with every candidate at its exact mean (R02).
    cf = s['closed_form']
    macros.update({
        'McRegretCFMedian': f"{cf['regret_pct']['median']:.2f}\\%",
        'McRegretCFMax': f"{cf['regret_pct']['max']:.2f}\\%",
        'McRegretVsCFBestMedian':
            f"{cf['regret_vs_cf_best_pct']['median']:.2f}\\%",
        'McRegretVsCFBestMax': f"{cf['regret_vs_cf_best_pct']['max']:.2f}\\%",
        'McGTRefIsCFBest': count_of(cf['n_reference_is_cf_best'],
                                    s['n_scenarios']),
        'McGTCFTolerance': f"{cf['tolerance_pct']:.3f}\\%",
        'McGTCFUnchanged': yesno(cf['all_unchanged']),
    })
    return macros


def _sa_family_macros(d, s):
    """How many annealer settings beat the GA, and how many lose to it,
    counted across the sweep. The runner's per-setting GA-vs-annealer
    intervals are at Figure B's per-comparison level, the level one
    annealer put in Figure B's place would get; a count over the sweep's
    settings is a family of its own and needs the Bonferroni level over
    them. Recomputed from ``settings.csv`` with the same scenario cluster
    bootstrap, after checking that it reproduces every recorded mean and
    interval at the runner's level (ValueError otherwise)."""
    path = os.path.join(d, 'settings.csv')
    if not os.path.exists(path):
        return {}
    tabs = {'ga': defaultdict(lambda: defaultdict(dict)),
            'sa': defaultdict(lambda: defaultdict(dict))}
    for r in read_csv(path):
        tabs[r['search']][r['setting']][int(r['scenario'])][int(r['seed'])] = \
            float(r['heldout_fitness'])
    ga, sa = tabs['ga'], tabs['sa']
    sweep, cfg = s['sa_sweep'], s['config']
    rec = sweep['settings']

    def clusters(gk, sk):
        return [[ga[gk][sc][j] - sa[sk][sc][j] for j in sorted(sa[sk][sc])]
                for sc in sorted(sa[sk])]

    def reproduces_means(gk):
        return all(abs(float(np.mean(np.concatenate(clusters(gk, sk))))
                       - float(v['mean_ga_minus_sa'])) < 1e-3
                   for sk, v in rec.items())
    tail = f"|{int(cfg['pop_size'])}|{int(cfg['n_gens'])}"
    gks = [k for k in ga if k.endswith(tail) and reproduces_means(k)]
    if len(gks) != 1:
        raise ValueError(f"{d}: settings.csv does not single out the default "
                         f"GA the sweep compared against ({gks})")
    a_run = float(sweep['alpha_per_comparison'])
    a_fam = 0.05 / len(rec)
    beat, lose = set(), set()
    for sk, v in rec.items():
        cl = clusters(gks[0], sk)
        _, lo, hi = cluster_boot_ci(cl, alpha=a_run)
        if abs(lo - float(v['ci_lo'])) > 1e-3 or abs(hi - float(v['ci_hi'])) > 1e-3:
            raise ValueError(f"{d}: settings.csv does not reproduce the "
                             f"recorded interval of annealer {sk}")
        _, lo, hi = cluster_boot_ci(cl, alpha=a_fam)
        if hi < 0:
            beat.add(sk)
        elif lo > 0:
            lose.add(sk)
    big = max(float(v['step_frac']) for v in rec.values())
    at_big = {k for k, v in rec.items() if float(v['step_frac']) == big}
    return {'SASensFamilyLevel': level_pct(100 * (1 - a_fam)),
            'SASensNBeatGA': count_of(len(beat), len(rec)),
            'SASensNLoseToGA': count_of(len(lose), len(rec)),
            'SASensLargestStep': f"{big:g}",
            'SASensLargestStepNBeatGA': count_of(len(beat & at_big),
                                                 len(at_big)),
            'SASensBeatGAAllLargestStep': yesno(bool(beat)
                                                and beat <= at_big)}


def gasens_macros(d):
    """The GA operator sweep in its fixed-budget form and the annealer
    sweep (R17): the spread across settings against the seed-blocked null,
    both ANOVA readings, and each annealer setting against the GA and
    against the default annealer, with the default as power reference."""
    macros = {}
    s = _summ(d)
    macros.update({
        'GAHyperSpread': f"{s['hyper_spread_median_pct']:.2f}\\%",
        'GAHyperSpreadMax': f"{s['hyper_spread_max_pct']:.2f}\\%",
        'GADefaultGap': f"{s['default_gap_median_pct']:.2f}\\%",
        'GANSettings': str(s['n_settings']),
        'GADivStart': f"{s['diversity_start']*100:.1f}\\%",
        'GADivEnd': f"{s['diversity_end']*100:.1f}\\%",
        # Settings are compared on seeds no run searched or selected
        # under, so the spread measures the layouts and not the upward
        # bias of picking a maximum of noisy estimates.
        'GAHeldoutSeeds': str(int(s.get('heldout_seeds', 0))),
    })
    cfg = s['config']
    macros['GASensBudget'] = f"{int(s['budget_total_evals']):,}"
    macros['GASensNScen'] = str(int(cfg['n_scenarios']))
    macros['GASensNSeeds'] = str(int(cfg['n_seeds']))
    macros['GASensItems'] = str(int(cfg['n_items']))
    macros['GASensMcIters'] = f"{int(cfg['mc_iters']):,}"
    shapes = sorted({(int(x['pop_size']), int(x['n_gens']))
                     for x in s['settings']})
    macros['GASensShapes'] = _num_list(
        [f"{p}\\,$\\times$\\,{g}" for p, g in shapes])
    macros['GASensMutRates'] = _num_list(
        sorted({float(x['mut_rate']) for x in s['settings']}), '{:.2f}')
    gs = s['ga_sweep']
    per = gs.get('per_scenario') or []
    macros['GASensNullRange'] = f"{gs['null_range_median_pct']:.2f}\\%"
    macros['GASensNullRangeIndep'] = \
        f"{gs['null_range_median_pct_independent']:.2f}\\%"
    macros['GASensSpreadOverNull'] = f"{gs['spread_over_null_median']:.2f}"
    macros['GASensNScenAboveNull'] = count_of(
        gs['n_scenarios_spread_above_null'], len(per) or cfg['n_scenarios'])
    macros['GASensSeedBlockShare'] = \
        upct(100 * float(gs['seed_block_share_median']), 0)
    an = gs['anova_setting']
    for key, tag in (('setting_vs_residual', 'Fixed'),
                     ('setting_vs_interaction', 'Random'),
                     ('interaction', 'Interaction'),
                     ('seed_blocks_vs_residual', 'SeedBlocks')):
        t = an.get(key) or {}
        if t.get('p') is not None:
            macros[f'GASensAnova{tag}P'] = pfmt(t['p'])
            macros[f'GASensAnova{tag}F'] = f"{float(t['F']):.2f}"
            macros[f'GASensAnova{tag}DF'] = \
                f"{int(t['df'])},\\,{int(t['df_error'])}"
    sa = s['sa_sweep']
    n_set = int(sa['n_settings'])
    macros.update({
        'SASensNSettings': str(n_set),
        'SASensNEquivalent': count_of(sa['n_equivalent'], n_set),
        'SASensNPairs': str(int(sa['n_pairs'])),
        'SASensFigBPairs': str(int(sa['figb_design_pairs'])),
        'SASensNDifferFromDefault':
            count_of(sa['n_settings_differ_from_default'], n_set - 1),
        'SASensVsDefaultLevel': level_pct(100 * float(sa['vs_default_sa_level'])),
        'SASensEquivLevel': level_pct(100 * float(sa['equivalence_level'])),
        'SASensBudget': f"{int(s['sa_budget_search_evals']):,}",
    })
    pr = sa['power_reference']
    macros['SASensPowerRefEquivalent'] = yesno(pr['equivalent'])
    macros['SASensPowerRefMinMarginPct'] = \
        f"{100 * float(pr['smallest_margin_frac']):.3f}\\%"
    sets = list(sa['settings'].values())
    pcts = [float(x['pct']) for x in sets]
    margins = [100 * float(x['equivalence']['smallest_margin_frac'])
               for x in sets]
    vd = [float(x['vs_default_sa']['pct']) for x in sets
          if not x.get('is_default') and x.get('vs_default_sa')]
    macros['SASensGAvsSAMinPct'] = spct(min(pcts), 3)
    macros['SASensGAvsSAMaxPct'] = spct(max(pcts), 3)
    macros['SASensMinMarginMinPct'] = f"{min(margins):.3f}\\%"
    macros['SASensMinMarginMaxPct'] = f"{max(margins):.3f}\\%"
    if vd:
        macros['SASensVsDefaultMinPct'] = spct(min(vd), 3)
        macros['SASensVsDefaultMaxPct'] = spct(max(vd), 3)
    macros.update(_sa_family_macros(d, s))
    grid = s.get('sa_settings') or []
    for key, macro, fmt in (('step_frac', 'SASensStepFracs', '{:g}'),
                            ('final_temp_frac', 'SASensFinalTemps', '{:g}'),
                            ('initial_accept', 'SASensAccepts', '{:g}')):
        vals = sorted({float(x[key]) for x in grid})
        if vals:
            macros[macro] = _num_list(vals, fmt)
    return macros


def abm_macros(d):
    """ABM diagnostics (audits R6.2, R6.4): the Markov-order test against
    its permutation null and the bookkeeping-only null, the perimeter
    ratio against the geometric and the whole-visit routing nulls, overall
    and on walking agents only (R13), and the protocol."""
    macros = {}
    s = _summ(d)
    mk, em = s.get('markov_order', {}), s.get('emergence', {})
    macros.update({
        'MarkovNSeq': str(mk['n_sequences']),
        'MarkovInfoGain': f"{mk['info_gain_second_order_bits']:.3f}",
        'MarkovTV': f"{mk['mean_tv_first_vs_second']:.3f}",
        'MarkovInfoGainCI': f"{mk['info_gain_ci95']:.3f}",
        'MarkovTVCI': f"{mk['tv_ci95']:.3f}",
        'MarkovHOne': f"{mk['H_next_given_cur_bits']:.2f}",
    })
    # The estimator is positive even for a first-order chain, so the
    # memory is read against its permutation null.
    if 'info_gain_null_mean_bits' in mk:
        macros.update({
            'MarkovInfoGainNull': f"{mk['info_gain_null_mean_bits']:.3f}",
            'MarkovInfoGainExcess': f"{mk['info_gain_excess_bits']:.3f}",
            'MarkovInfoGainExcessCI': f"{mk['info_gain_excess_ci95']:.3f}",
            'MarkovPermP': pfmt(mk['info_gain_perm_p']),
            'MarkovTVNull': f"{mk['tv_null_mean']:.3f}",
            'MarkovTVExcess': f"{mk['tv_excess']:.3f}",
            'MarkovNPerm': str(int(mk['n_permutations'])),
        })
    # ... and against what the state machine's bookkeeping alone produces
    # with no geometry (R13): the same estimator on generated sequences.
    bk = mk['bookkeeping_null']
    macros.update({
        'MarkovBookkeepingNSeq': f"{int(bk['n_sequences']):,}",
        'MarkovBookkeepingGain': f"{bk['info_gain_second_order_bits']:.3f}",
        'MarkovBookkeepingGainCI': f"{bk['info_gain_ci95']:.3f}",
        'MarkovBookkeepingP': pfmt(bk['info_gain_perm_p']),
        'MarkovBookkeepingTV': f"{bk['mean_tv_first_vs_second']:.3f}",
        'MarkovObsMinusBookkeeping':
            f"{bk['observed_minus_bookkeeping_bits']:+.3f}",
        'MarkovObsMinusBookkeepingCI':
            f"{bk['observed_minus_bookkeeping_ci95']:.3f}",
        'MarkovBookkeepingSupport':
            upct(100 * float(bk['share_observed_in_bookkeeping_support']), 1),
    })
    macros['PerimRatio'] = f"{em['perimeter_interior_ratio']:.2f}"
    macros['PerimRatioCI'] = f"{em['perimeter_ratio_ci95']:.2f}"
    # The ratio's geometric null: what the floor plan gives traffic
    # spread evenly over the walkable cells, fixtures excluded. The
    # observed-to-null quotient is the part of the ratio the shoppers'
    # routes add; both are replication means with t half-widths, like
    # the ratio itself, and the quotient's interval is read against 1.
    to_null = float(em['ratio_to_geometric_null'])
    hw = float(em['ratio_to_geometric_null_ci95'])
    macros.update({
        'PerimRatioNull':
            f"{em['perimeter_interior_ratio_geometric_null']:.2f}",
        'PerimRatioToNull': f"{to_null:.2f}",
        'PerimRatioToNullCI': f"{hw:.2f}",
        'PerimRatioToNullExcludesOne':
            'yes' if to_null - hw > 1 or to_null + hw < 1 else 'no',
    })
    if em.get('geometric_null_ci95') is not None:
        macros['PerimRatioNullCI'] = f"{float(em['geometric_null_ci95']):.2f}"
    # The routing null (R13): shortest-path tours over whole visits --
    # door, the drawn invoice's fixtures in random order, the washroom when
    # due, the nearest lane, the door -- which put the door, lane and
    # washroom legs in the perimeter band by construction. The like-for-
    # like comparison is the walking agents' ratio over it.
    rnull = float(em['perimeter_interior_ratio_routing_null'])
    macros['PerimRatioRoutingNull'] = f"{rnull:.2f}"
    if em.get('routing_null_mc_se') is not None:
        macros['PerimRatioRoutingNullSE'] = f"{float(em['routing_null_mc_se']):.3f}"
    rr, rr_hw = float(em['ratio_to_routing_null']), \
        float(em['ratio_to_routing_null_ci95'])
    macros['PerimRatioToRoutingNull'] = f"{rr:.2f}"
    macros['PerimRatioToRoutingNullCI'] = f"{rr_hw:.2f}"
    rn = em['routing_null']
    macros['RoutingNullTours'] = f"{int(rn['n_tours']):,}"
    wc = rn.get('wc_leg') or {}
    if wc.get('n_tours_with_wc') is not None and rn.get('n_tours'):
        macros['RoutingNullWCShare'] = \
            upct(100 * wc['n_tours_with_wc'] / rn['n_tours'], 1)
    mo = em['moving_only']
    mvr, mvr_hw = float(mo['ratio_to_routing_null']), \
        float(mo['ratio_to_routing_null_ci95'])
    macros.update({
        'PerimRatioMoving': f"{mo['perimeter_interior_ratio']:.2f}",
        'PerimRatioMovingCI': f"{mo['perimeter_ratio_ci95']:.2f}",
        'PerimRatioMovingToNull': f"{mo['ratio_to_geometric_null']:.2f}",
        'PerimRatioMovingToNullCI':
            f"{mo['ratio_to_geometric_null_ci95']:.2f}",
        'PerimRatioMovingToRoutingNull': f"{mvr:.2f}",
        'PerimRatioMovingToRoutingNullCI': f"{mvr_hw:.2f}",
        'PerimRatioMovingToRoutingNullExcludesOne':
            yesno(mvr - mvr_hw > 1 or mvr + mvr_hw < 1),
        'PerimRatioMovingShare': upct(100 * float(mo['share_of_samples']), 0),
    })
    # The band-width sweep, replication by replication: the smallest ratio
    # to the geometric null over every width and replication, and the
    # walking agents' ratio to the routing null at the narrowest and the
    # widest band (replication means).
    reps = [r['emergence'] for r in (s.get('per_rep') or [])
            if r.get('emergence', {}).get('band_sweep_ratio_to_geometric_null')]
    if reps:
        geo = [float(v) for e in reps
               for v in e['band_sweep_ratio_to_geometric_null'].values()]
        macros['PerimRatioToNullMinRep'] = f"{min(geo):.2f}"
        macros['PerimRatioToNullAllAboveOne'] = yesno(min(geo) > 1)
        widths = sorted(reps[0]['moving']['band_sweep_ratio_to_routing_null'],
                        key=float)
        for w, tag in ((widths[0], 'Narrow'), (widths[-1], 'Wide')):
            vals = [float(e['moving']['band_sweep_ratio_to_routing_null'][w])
                    for e in reps]
            macros[f'PerimRatioMovingToRoutingNull{tag}'] = \
                f"{np.mean(vals):.2f}"
            macros[f'PerimBand{tag}'] = f"{float(w):g}"
    proto = s.get('protocol', {})
    macros['AbmReps'] = str(proto['reps'])
    macros['AbmWarmup'] = f"{proto['warmup_s']:.0f}"
    macros['AbmCollect'] = f"{proto['collect_s']:.0f}"
    # Every agent that arrived inside the window was followed until it
    # left, so no visit is missing for being long; the drain is how far
    # past the window that took.
    macros['AbmCensored'] = str(int(mk['n_censored_final']))
    if mk.get('mean_drain_seconds') is not None:
        macros['AbmDrain'] = f"{float(mk['mean_drain_seconds']):.0f}"
    b = balked_pct(s.get('load'))
    if b is not None:
        macros['AbmBalked'] = f"{b:.1f}\\%"
    return macros


# Separation strengths of the structural sweep, named by their ratio to
# the shipped default for the paired-contrast macros.
_STRENGTH_WORDS = {0.0: 'Off', 0.5: 'Half', 2.0: 'Double', 4.0: 'Quad'}


def _strength_word(strength, default):
    ratio = round(float(strength) / float(default), 6) if default else None
    if ratio in _STRENGTH_WORDS:
        return _STRENGTH_WORDS[ratio]
    return 'At' + num_word(round(float(strength) * 100))


def struct_macros(d):
    """The structural sweep (audit R6.5): throughput and revenue per
    customer across separation strengths, now read from the common-
    random-numbers design -- the block ANOVA, each setting's paired
    contrast against the default and the minimum detectable effects at
    80% power (R40) -- beside the one-way tests it used before."""
    macros = {}
    s = _summ(d)
    if 'throughput_range_pct' in s:
        macros['StructRangePct'] = f"{s['throughput_range_pct']:.1f}\\%"
        macros['StructChiP'] = pfmt(s['chi2_p'])
    # The pooled chi-square treats the completions as Poisson. The
    # replications say how wide they really are, so the one-way ANOVA on
    # the per-replication counts is reported, and the chi-square is also
    # reported corrected by the dispersion they show.
    if s.get('completions_anova_p') is not None:
        macros['StructCompAnovaP'] = pfmt(s['completions_anova_p'])
    if s.get('dispersion_phi') is not None:
        macros['StructDispersion'] = f"{float(s['dispersion_phi']):.2f}"
    if s.get('quasi_poisson_p') is not None:
        macros['StructQuasiP'] = pfmt(s['quasi_poisson_p'])

    # Power of the one-way test at the observed counts, as before: a null
    # result is only informative if the design could have seen the
    # effect. The replications are the unit of the test and they are
    # wider than Poisson, so the counts are simulated as gamma-Poisson at
    # the dispersion the sweep measured. (The minimum detectable effects
    # below are the design's own reading.)
    comp_reps = [list(map(float, c))
                 for c in s.get('completed_per_rep', []) if c]
    if comp_reps:
        from scipy.stats import f_oneway
        phi = float(s.get('dispersion_phi') or 1.0)
        k_set = len(comp_reps)
        n_rep = min(len(c) for c in comp_reps)
        base = float(np.mean([np.mean(c) for c in comp_reps]))
        rng_p = np.random.default_rng(0)
        for eff, key in ((0.30, 'StructPowerThirty'),
                         (0.40, 'StructPowerForty')):
            means = np.full(k_set, base)
            means[-1] *= (1.0 - eff)
            hits = 0
            for _ in range(N_POWER_SIMS):
                groups = [gamma_poisson(rng_p, mu, phi, n_rep)
                          for mu in means]
                pv = f_oneway(*groups).pvalue
                hits += int(np.isfinite(pv) and pv < 0.05)
            macros[key] = f"{100.0 * hits / N_POWER_SIMS:.0f}\\%"

    # Per-customer revenue across settings, judged against replication
    # noise. An earlier UNREPLICATED sweep appeared to show a large
    # monotone decline; it did not survive a second run (audit R73).
    m = np.array(s['rev_per_cust_mean'], dtype=float)
    sd = np.array(s['rev_per_cust_sd'], dtype=float)
    n_reps = int(s['n_reps'])
    macros['StructReps'] = str(n_reps)
    macros['StructNSettings'] = str(len(m))
    macros['StructRevRange'] = f"{m.max() - m.min():.2f}"
    # Pooled within-setting SD, i.e. the square root of the mean
    # variance. Averaging the SDs themselves runs low.
    pooled_sd = float(np.sqrt(np.mean(sd ** 2)))
    macros['StructRevNoiseSD'] = f"{pooled_sd:.2f}"
    # The range is taken over setting MEANS, so what it should be set
    # against is the standard error of a mean and the range that a sweep
    # with no effect at all would still show: the expected range of k
    # normal draws is d2(k) standard errors.
    se = pooled_sd / np.sqrt(n_reps)
    macros['StructRevSE'] = f"{se:.2f}"
    if len(m) in D2_RANGE:
        macros['StructRevNullRange'] = f"{D2_RANGE[len(m)] * se:.2f}"
    if s.get('rev_per_cust_anova_p') is not None:
        macros['StructRevAnovaP'] = pfmt(s['rev_per_cust_anova_p'])
    macros['StructNCompletions'] = str(
        int(sum(s.get('completed_per_setting', []))))
    # The design's own tests (R40): the block ANOVA over settings x
    # replication blocks, and every setting against the default within
    # blocks, with the minimum detectable effect at 80% power.
    macros['StructDesign'] = 'common random numbers'
    for metric, tag in (('completed', 'Comp'), ('rev_per_cust', 'Rev'),
                        ('conversion', 'Conv'), ('revenue', 'Revenue')):
        b = (s.get('block_anova') or {}).get(metric) or {}
        if b.get('p') is not None:
            macros[f'StructBlock{tag}P'] = pfmt(b['p'])
            macros[f'StructBlock{tag}F'] = f"{float(b['F']):.2f}"
    mde = s['mde']
    macros['StructMDEPower'] = upct(100 * float(mde['power']), 0)
    for metric, tag in (('completed', 'Comp'), ('rev_per_cust', 'Rev'),
                        ('conversion', 'Conv'), ('revenue', 'Revenue')):
        v = (mde.get('max_pct_of_default') or {}).get(metric)
        if v is not None:
            macros[f'StructMDE{tag}'] = upct(v, 1)
    default = float(s['default_strength'])
    k0 = int(s['default_index'])
    for metric, tag in (('completed', 'Comp'), ('rev_per_cust', 'Rev')):
        contrasts = (s.get('paired_contrasts') or {}).get(metric) or {}
        if not contrasts:
            continue
        base_mean = float(np.mean(s['per_rep_values'][metric][k0]))
        ps = []
        for strength, c in contrasts.items():
            stem = f'StructPair{_strength_word(strength, default)}{tag}'
            if c.get('mean_diff_pct_of_default') is not None:
                macros[f'{stem}Pct'] = spct(c['mean_diff_pct_of_default'], 1)
            if c.get('ci95_half_width') is not None and base_mean:
                macros[f'{stem}CIPct'] = \
                    upct(100 * float(c['ci95_half_width']) / base_mean, 1)
            if c.get('p') is not None:
                macros[f'{stem}P'] = pfmt(c['p'])
                ps.append(float(c['p']))
            if c.get('mde80_pct_of_default') is not None:
                macros[f'{stem}MDE'] = upct(c['mde80_pct_of_default'], 1)
        if ps:
            macros[f'StructPair{tag}MinP'] = pfmt(min(ps))
            macros[f'StructPair{tag}NSig'] = count_of(
                sum(p < 0.05 for p in ps), len(ps))
    # Pooled over the sweep: every setting ran at the same load, so a
    # large balked share would mean the settings were compared under a
    # cap rather than under the rule being swept.
    b = balked_pct(s.get('load'))
    if b is not None:
        macros['StructBalked'] = f"{b:.1f}\\%"
    macros.update(struct_conversion_macros(s))
    return macros


def _setting_word(strength, default):
    """A sweep setting's macro word: 'Default' for the shipped strength,
    else the paired contrasts' word (Off, Half, Double, Quad)."""
    if default and abs(float(strength) - float(default)) < 1e-12:
        return 'Default'
    return _strength_word(strength, default)


def struct_conversion_macros(s):
    """The structural sweep's conversion, per setting, beside what the
    spawn-time abandonment draw alone predicts for it.

    Conversion is paid exits over all exits of the window, and every
    unpaid exit is counted as abandoned. An agent draws its abandonment
    at spawn (probability uniform on AGENT_ABANDON_PROB_RANGE, so a mean
    p-bar), independently of how it moves; were that draw the only route
    to an unpaid exit, a setting's abandoned count would be about
    Binomial(exits, p-bar): mean p-bar x exits, SD sqrt(exits p-bar
    (1 - p-bar)). Per setting: conversion (StructConv<W>), abandoned and
    exit counts (StructAbandoned<W>, StructExits<W>) and that prediction
    (StructAbandonedExpected<W>, ...SD<W>, ...Z<W>); pooled: the observed
    abandoned share against p-bar, and Pearson's chi-square of the
    settings' paid / unpaid split (homogeneity across settings, K - 1 df).
    The contrasts' conversion differences are in percentage points. When
    the run recorded its exits by route (``exit_routes``), each setting's
    unpaid exits by route and the stall counts are reported too.

    What these numbers can and cannot carry: in the runs the exit-route
    check re-ran (``run_structural_exit_check``, StructCheck*), the draw
    was the only route by which any agent left unpaid, and the draw does
    not depend on movement. Conversion in this sweep is therefore
    insensitive to separation by construction (up to which agents finish
    inside the window), so neither the conversion contrasts nor
    StructMDEConv can detect a separation effect; they are no robustness
    result and are reported to explain the homogeneity test's borderline p
    only. Common random numbers line up the arrivals but hardly any agent
    draws (StructCheckSameDraw*), so the settings' abandoned counts are
    independent binomials and the homogeneity test holds its level at its
    own alpha. The Bonferroni alpha (StructPairConvBonfAlpha) covers the
    four paired contrasts, not that test."""
    from scipy.stats import chi2 as _chi2
    out = {}
    rows = s.get('rows') or []
    if not rows:
        return out
    lo, hi = _script_constants(LIT_SCRIPT, ('AGENT_ABANDON_PROB_RANGE',))[
        'AGENT_ABANDON_PROB_RANGE']
    pbar = 0.5 * (float(lo) + float(hi))
    out['StructAbandonDrawMean'] = upct(100 * pbar, 1)
    default = float(s['default_strength'])
    paid = np.array([int(r['completed']) for r in rows], dtype=float)
    unpaid = np.array([int(r['abandoned']) for r in rows], dtype=float)
    exits = paid + unpaid
    for r, n_paid, n_ab, n_ex in zip(rows, paid, unpaid, exits):
        w = _setting_word(r['strength'], default)
        exp, sd = pbar * n_ex, math.sqrt(n_ex * pbar * (1 - pbar))
        out[f'StructConv{w}'] = upct(100 * n_paid / max(n_ex, 1), 1)
        out[f'StructAbandoned{w}'] = str(int(n_ab))
        out[f'StructExits{w}'] = f"{int(n_ex):,}"
        out[f'StructAbandonedExpected{w}'] = f"{exp:.0f}"
        out[f'StructAbandonedSD{w}'] = f"{sd:.0f}"
        out[f'StructAbandonedZ{w}'] = f"{(n_ab - exp) / sd:+.1f}"
        routes = r.get('exit_routes')
        if isinstance(routes, dict) and routes:
            for key, tag in (('unpaid_abandon_draw', 'Draw'),
                             ('unpaid_no_route', 'NoRoute'),
                             ('unpaid_other', 'Other'),
                             ('moving_stalls', 'MovingStalls'),
                             ('exit_stall_teleports', 'ExitTeleports')):
                out[f'StructUnpaid{tag}{w}' if key.startswith('unpaid')
                    else f'Struct{tag}{w}'] = str(int(routes.get(key, 0)))
    tot_ab, tot_ex = unpaid.sum(), exits.sum()
    out['StructAbandonedPooled'] = upct(100 * tot_ab / tot_ex, 2)
    out['StructAbandonedPooledCount'] = count_of(tot_ab, tot_ex)
    out['StructAbandonedPooledZ'] = (
        f"{(tot_ab - pbar * tot_ex) / math.sqrt(tot_ex * pbar * (1 - pbar)):+.1f}")
    pooled = tot_ab / tot_ex
    e_ab, e_paid = exits * pooled, exits * (1 - pooled)
    x2 = float(np.sum((unpaid - e_ab) ** 2 / e_ab)
               + np.sum((paid - e_paid) ** 2 / e_paid))
    df = len(rows) - 1
    out['StructAbandonedChi'] = f"{x2:.1f}"
    out['StructAbandonedChiDf'] = str(df)
    out['StructAbandonedChiP'] = pfmt(_chi2.sf(x2, df))
    # The conversion contrasts against the default, in percentage points.
    for strength, c in ((s.get('paired_contrasts') or {})
                        .get('conversion') or {}).items():
        stem = f'StructPair{_strength_word(strength, default)}Conv'
        if c.get('mean_diff') is not None:
            out[f'{stem}PP'] = pp(100 * float(c['mean_diff']), 1)
        if c.get('ci95_half_width') is not None:
            out[f'{stem}CIPP'] = f"{100 * float(c['ci95_half_width']):.1f}\\,pp"
        if c.get('p') is not None:
            out[f'{stem}P'] = pfmt(c['p'])
    ps = [float(c['p']) for c in ((s.get('paired_contrasts') or {})
                                  .get('conversion') or {}).values()
          if c.get('p') is not None]
    if ps:
        out['StructPairConvMinP'] = pfmt(min(ps))
        out['StructPairConvNSig'] = count_of(sum(p < 0.05 for p in ps),
                                             len(ps))
        # Bonferroni over the contrasts against the default.
        out['StructPairConvBonfAlpha'] = f"{0.05 / len(ps):.4f}"
    return out


def struct_check_macros(d):
    """What the structural sweep's exit-route check found, on the
    replications it re-ran (StructCheckReps per setting, StructCheckRuns
    runs, StructCheckReproduced of them reproducing the sweep exactly):
    the window's unpaid exits (StructCheckUnpaid, of the sweep's
    StructCheckUnpaidOfSweep) by route -- the spawn-time abandonment draw
    (StructCheckUnpaidDraw, over the unpaid exits), no route left, any
    other -- agents that left unpaid without drawing abandonment and that
    drew it but paid; the stall detector's exits (StructCheckStalledExits,
    per setting too; unpaid among them and whether they had drawn),
    exiting-state door placements and relocations; the unpaid count the
    exiting agents' drawn probabilities predict and the z of the observed
    count against it. Then how far common random numbers carry the agents'
    own draws: per non-default setting, the agents that left in both its
    window and the default's (StructCheckCommon<W>, pooled over the
    checked replications) and how many of them carried the same spawn-time
    draws in both (StructCheckSameDraw<W>); few means every setting's
    abandoned count is an independent binomial draw.

    What this covers is only what was re-run: a route that never fired in
    these runs is not shown never to fire in the sweep's others."""
    s = _summ(d)
    t, sh = s['total'], s['sharing']
    default = float(s['struct']['default_strength'])
    routes = t['unpaid_by_route']
    out = {
        'StructCheckRuns': str(int(t['n_runs'])),
        'StructCheckReps': str(len(s['reps_checked'])),
        'StructCheckReproduced': count_of(
            sum(bool(r['matches_artifact']) for r in s['runs']),
            len(s['runs'])),
        'StructCheckExits': f"{int(t['n_exits']):,}",
        'StructCheckUnpaid': str(int(t['unpaid'])),
        'StructCheckUnpaidOfSweep': count_of(t['unpaid'],
                                             s['struct']['unpaid_total']),
        'StructCheckUnpaidDraw': count_of(routes['abandon_draw'],
                                          t['unpaid']),
        'StructCheckUnpaidNoRoute': str(int(routes['no_route'])),
        'StructCheckUnpaidOther': str(int(routes['other'])),
        'StructCheckUnpaidWithoutDraw': str(int(t['unpaid_without_draw'])),
        'StructCheckDrawnPaid': str(int(t['drawn_but_paid'])),
        'StructCheckStalledExits': str(int(t['stalled_exits'])),
        'StructCheckStalledUnpaid': str(int(t['stalled_unpaid'])),
        'StructCheckStalledUnpaidDrawn': str(int(t['stalled_unpaid_drawn'])),
        'StructCheckExitTeleports': str(int(t['exit_stall_teleports'])),
        'StructCheckRelocations': str(int(t['relocations'])),
        'StructCheckExpectedUnpaid': f"{float(t['expected_unpaid']):.1f}",
    }
    if t.get('z_unpaid') is not None:
        out['StructCheckUnpaidZ'] = f"{float(t['z_unpaid']):+.1f}"
    for strength, rec in (s.get('per_setting') or {}).items():
        w = _setting_word(strength, default)
        out[f'StructCheckStalledExits{w}'] = str(int(rec['stalled_exits']))
    common = same = same_ab = 0
    for strength, rec in (sh.get('per_setting') or {}).items():
        w = _setting_word(strength, default)
        out[f'StructCheckCommon{w}'] = f"{int(rec['n_common']):,}"
        out[f'StructCheckSameDraw{w}'] = count_of(rec['n_same_draw'],
                                                  rec['n_common'])
        common += int(rec['n_common'])
        same += int(rec['n_same_draw'])
        same_ab += int(rec['n_same_draw_abandoning'])
    if common:
        out['StructCheckSameDrawTotal'] = count_of(same, common)
        out['StructCheckSameDrawAbandoning'] = str(same_ab)
    return out


# The goodness-of-fit tests, matched on what each is about rather than on
# its exact label, which is written for the Validation tab and reads better
# when it is free to change.
GOF_TAGS = (('basket', 'Basket'), ('revenue', 'Revenue'),
            ('categor', 'Category'), ('inter-arrival', 'Arrival'))


def gof_macros(block, prefix):
    """The pooled tests of one goodness-of-fit design under ``prefix``:
    statistic, p-value, decision and both sample sizes per test, how many
    single windows reached the pooled test's verdict on their own, the
    replication count and the share of arrivals turned away at the cap;
    the category row's effect sizes and the arrival row's time-rescaled
    record (R29, R47); the realised load (R46); and the list chain (R29).

    The category row's p-value is its permutation p, and its sample sizes
    count baskets (reference invoices, simulated visits). The item-level
    p of the same statistic, which treats every purchase as independent,
    goes beside it as ``CategoryNaiveP``."""
    out = {}
    split = block.get('per_test_decisions', {})
    for t in block.get('pooled', []):
        name = str(t['test']).lower()
        tag = next((v for k, v in GOF_TAGS if k in name), None)
        if not tag or t.get('p_value') is None:
            continue
        out.update({
            # KS distances to three places; the category chi-square to one.
            f'{prefix}{tag}Stat': format(t['statistic'],
                                         '.1f' if tag == 'Category' else '.3f'),
            f'{prefix}{tag}P': pfmt(t['p_value']),
            f'{prefix}{tag}Decision': t['decision'],
            f'{prefix}{tag}NObs': f"{int(t['n_observed']):,}",
            f'{prefix}{tag}NSim': f"{int(t['n_simulated']):,}",
        })
        if tag == 'Category':
            if t.get('naive_p_value') is not None:
                out[f'{prefix}CategoryNaiveP'] = pfmt(t['naive_p_value'])
            # Effect sizes: the share distance is the one to read; V depends
            # on how the purchases split between the two samples.
            for key, macro, fmt in (
                    ('share_tv_distance', 'ShareTV', '.3f'),
                    ('cramers_v', 'CramersV', '.3f'),
                    ('cramers_v_balanced', 'CramersVBalanced', '.3f')):
                if t.get(key) is not None:
                    out[f'{prefix}Category{macro}'] = format(t[key], fmt)
            if t.get('sim_share_of_items') is not None:
                out[f'{prefix}CategorySimShare'] = \
                    upct(100 * float(t['sim_share_of_items']), 1)
        if tag == 'Arrival':
            # Time-rescaled gaps against Exp(1) under the NHPP null (R47),
            # with the homogeneous test beside it.
            for key, macro, fmt in (
                    ('rescaled_cv', 'RescaledCV', '.2f'),
                    ('reference_rescaled_cv', 'RefRescaledCV', '.2f'),
                    ('rescaled_mean', 'RescaledMean', '.2f'),
                    ('homogeneous_statistic', 'HomogStat', '.3f')):
                if t.get(key) is not None:
                    out[f'{prefix}Arrival{macro}'] = format(t[key], fmt)
            if t.get('homogeneous_p_value') is not None:
                out[f'{prefix}ArrivalHomogP'] = pfmt(t['homogeneous_p_value'])
            if t.get('n_trading_days') is not None:
                out[f'{prefix}ArrivalTradingDays'] = \
                    str(int(t['n_trading_days']))
        # How many single windows reached the same verdict on their own:
        # a pooled decision that rests on the pooled sample size shows up
        # as a split here.
        if t['test'] in split:
            out[f'{prefix}{tag}RepPass'] = str(int(split[t['test']]['PASS']))
    out[f'{prefix}Reps'] = str(int(block['n_reps']))
    ld = block.get('load') or {}
    b = balked_pct(ld)
    if b is not None:
        out[f'{prefix}Balked'] = f"{b:.1f}\\%"
    # The realised load (R46): the door rate, the rate the protocol
    # offered, the hour's profile multiplier, the cohort followed and the
    # drain; the held-out store runs at another hour's multiplier.
    if ld.get('arrivals') is not None:
        out[f'{prefix}BalkedCount'] = count_of(ld.get('balked', 0),
                                               ld['arrivals'])
        out[f'{prefix}Arrivals'] = f"{int(ld['arrivals']):,}"
    for key, macro, fmt in (('door_rate_per_s', 'DoorRate', '{:.3f}'),
                            ('offered_rate_per_s', 'OfferedRate', '{:.3f}'),
                            ('profile_multiplier', 'ProfileMult', '{:.3f}'),
                            ('cohort_visits', 'CohortVisits', '{:,}'),
                            ('mean_drain_s', 'Drain', '{:.0f}')):
        if ld.get(key) is not None:
            out[f'{prefix}{macro}'] = fmt.format(ld[key])
    # The list chain (R29): the drawn lists against the reference (the
    # list law) and the paying visits' drawn lists against the unpaid
    # ones' (selection), each with the category row's own test; and the
    # completion, counted.
    lc = block.get('list_chain') or {}
    g = _script_constants(GOF_SCRIPT, ('LIST_CHAIN_DRAWN',
                                       'LIST_CHAIN_SELECTION'))
    chain_split = lc.get('per_test_decisions') or {}
    for name, tag in ((g['LIST_CHAIN_DRAWN'], 'ChainDrawn'),
                      (g['LIST_CHAIN_SELECTION'], 'ChainSelect')):
        t = next((x for x in lc.get('tests') or [] if x.get('test') == name),
                 None)
        if not t or t.get('p_value') is None:
            continue
        out.update({
            f'{prefix}{tag}Stat': format(t['statistic'], '.1f'),
            f'{prefix}{tag}P': pfmt(t['p_value']),
            f'{prefix}{tag}Decision': t['decision'],
            f'{prefix}{tag}NObs': f"{int(t['n_observed']):,}",
            f'{prefix}{tag}NSim': f"{int(t['n_simulated']):,}",
        })
        if t.get('share_tv_distance') is not None:
            out[f'{prefix}{tag}ShareTV'] = f"{t['share_tv_distance']:.3f}"
        if name in chain_split:
            out[f'{prefix}{tag}RepPass'] = str(int(chain_split[name]['PASS']))
    c = lc.get('counts') or {}
    if c.get('n_paying') is not None:
        out[f'{prefix}ChainVisits'] = f"{int(c['n_visits']):,}"
        out[f'{prefix}ChainPaying'] = f"{int(c['n_paying']):,}"
        out[f'{prefix}ChainUnpaid'] = f"{int(c['n_visits'] - c['n_paying']):,}"
        out[f'{prefix}ChainWholeList'] = count_of(c['n_paying_whole_list'],
                                                  c['n_paying'])
        out[f'{prefix}ChainItemsBought'] = count_of(
            c['items_bought_from_list'], c['items_drawn_by_paying'])
    comp = lc.get('completion') or {}
    if comp.get('share_tv_distance_drawn_vs_bought') is not None:
        out[f'{prefix}ChainCompletionTV'] = \
            f"{comp['share_tv_distance_drawn_vs_bought']:.3f}"
    return out


def _replica_macros(prefix, replica):
    """Size-matched replicas of a store's calibration period: what a
    perfect transfer of that period scores at the simulator's own sample
    size (median and central 95% range), how many the test rejects, and
    how many score no more than the simulator -- as the counts behind
    each share (R57) -- which side of the replicas' median the simulator
    falls on (R16), the direction of its mean error, and for the category
    row the replicas' effect sizes."""
    out = {}
    for tag, key in (('Basket', 'basket'), ('Revenue', 'revenue'),
                     ('Category', 'category')):
        r = replica.get(key)
        if not r:
            continue
        n = int(r['n_replicas'])
        fmt = '.1f' if key == 'category' else '.3f'
        stem = f'{prefix}{tag}Replica'
        out[f'{stem}Median'] = format(r['median'], fmt)
        out[f'{stem}Lo'] = format(r['lo'], fmt)
        out[f'{stem}Hi'] = format(r['hi'], fmt)
        out[f'{stem}Reject'] = share_count(r['reject_rate'], n)
        out[f'{stem}RejectPct'] = upct(100 * float(r['reject_rate']), 1)
        if r.get('simulator_percentile') is not None:
            out[f'{stem}Below'] = share_count(r['simulator_percentile'], n)
            out[f'{stem}BelowPct'] = \
                upct(100 * float(r['simulator_percentile']), 1)
        if r.get('simulator_vs_median'):
            out[f'{stem}Side'] = str(r['simulator_vs_median'])
        if r.get('simulator_mean_minus_reference') is not None:
            # Items per visit for the basket row, pounds per visit for the
            # revenue row, both to two decimals.
            v = float(r['simulator_mean_minus_reference'])
            out[f'{stem}SimMeanMinusRef'] = (f"{v:+.2f}" if key == 'basket'
                                             else pounds2(v))
        if r.get('simulator_mean_percentile') is not None:
            out[f'{stem}SimMeanBelow'] = share_count(
                r['simulator_mean_percentile'], n)
        out['GofReplicaN'] = str(n)
        eff = (r.get('effect_sizes') or {}).get('share_tv_distance')
        if eff and eff.get('median') is not None:
            out[f'{stem}ShareTVMedian'] = f"{eff['median']:.3f}"
            out[f'{stem}ShareTVLo'] = f"{eff['lo']:.3f}"
            out[f'{stem}ShareTVHi'] = f"{eff['hi']:.3f}"
            if eff.get('simulator_percentile') is not None:
                out[f'{stem}ShareTVBelow'] = share_count(
                    eff['simulator_percentile'], n)
    return out


def live_load_macros(period, load, counts):
    """The live store's calibrated rates beside the load the live runs put
    on it (review R04), from the in-sample goodness-of-fit run: its period
    record (the store every live family measures), its realised load and
    its cohort's list-chain counts.

    ``LiveStoreBuyersPerHour`` and ``LiveStoreVisitorsPerHour`` are the
    calibration's rates per open hour (the Monte Carlo engine's units),
    the visitors being the buyers over ``LiveStoreAssumedConversion``.
    ``LiveLoadMultiple`` is the rate the protocol offered at the door in
    the run's hour (nominal spawn rate x that hour's profile multiplier,
    per hour) over the calibrated visitor rate of a mean open hour;
    ``LiveLoadMultipleSameHour`` compares it with the calibrated rate of
    the same hour instead (the calibrated rate x the same multiplier),
    which leaves the spawn rate against the calibrated visitor rate.
    ``LiveConversionPct`` / ``LiveConversionCount`` are the share of the
    window's visitors who paid. ``LiveStoreItemsPerCategory`` is the
    assortment rule (the top products of each category, review R34)."""
    out = {}
    buyers = period.get('arrivals_per_hour')
    visitors = period.get('visitors_per_hour')
    if buyers is not None:
        out['LiveStoreBuyersPerHour'] = f"{float(buyers):.1f}"
    if visitors is not None:
        out['LiveStoreVisitorsPerHour'] = f"{float(visitors):.1f}"
    if period.get('assumed_conversion_rate') is not None:
        out['LiveStoreAssumedConversion'] = \
            f"{float(period['assumed_conversion_rate']):.2f}"
    offered = load.get('offered_rate_per_s')
    mult = load.get('profile_multiplier')
    if visitors and offered is not None:
        out['LiveOfferedPerHour'] = f"{3600.0 * float(offered):,.0f}"
        out['LiveLoadMultiple'] = \
            f"{3600.0 * float(offered) / float(visitors):.0f}"
        if mult:
            out['LiveLoadMultipleSameHour'] = \
                f"{3600.0 * float(offered) / (float(visitors) * float(mult)):.0f}"
    if counts.get('n_visits'):
        out['LiveConversionPct'] = \
            upct(100.0 * counts['n_paying'] / counts['n_visits'], 1)
        out['LiveConversionCount'] = count_of(counts['n_paying'],
                                              counts['n_visits'])
    if period.get('max_items_per_category') is not None:
        out['LiveStoreItemsPerCategory'] = \
            str(int(period['max_items_per_category']))
    return out


def gof_family_macros(d):
    """Goodness of fit: the live model against its calibration, in sample
    (the top level, under the macro names it has always had) and held out
    (a store calibrated on the prior period, tested against the current
    one), with what each design's store was built from and tested against,
    the year shift on the fixed assortment and the popularity turnover
    (R16), and the replicas."""
    macros = {}
    s = _summ(d)
    macros.update(gof_macros(s, 'Gof'))
    macros['GofAlpha'] = f"{float(s['alpha']):.2f}"
    # Relabellings behind the category p-values (both designs run the
    # same count; the validator holds each to the module's).
    macros['GofCategoryNPerm'] = \
        f"{int(_gof_category_row(s)['n_permutations']):,}"
    # The held-out design: a store calibrated on the prior period and
    # tested against the current one, whose invoices the model never saw.
    # Same tests, level and replication count, under a HeldOut infix. Its
    # inter-arrival row describes the current period's own arrivals and
    # is the in-sample one again.
    macros.update(gof_macros(s['held_out'], 'GofHeldOut'))
    design = s['design']
    for period, tag in (('current', 'Current'), ('prior', 'Prior')):
        p = design['periods'][period]
        macros[f'GofPeriod{tag}From'] = str(p['first_date'])
        macros[f'GofPeriod{tag}To'] = str(p['last_date'])
        macros[f'GofPeriod{tag}Invoices'] = f"{int(p['n_invoices']):,}"
    # The prior period is cut on the raw rows before cleaning; the
    # purchases a cancellation after the cut reverses stay in it.
    rac = design['periods']['prior'].get('reversals_across_cut') or {}
    if rac.get('purchase_lines') is not None:
        macros['GofReversalsAcrossCut'] = f"{int(rac['purchase_lines']):,}"
        macros['GofReversalsAcrossCutRevenue'] = \
            pounds_amount(float(rac['revenue']))
    for name, prefix in (('in_sample', 'Gof'), ('held_out', 'GofHeldOut')):
        facts = design[name]
        macros[f'{prefix}PlacedProducts'] = \
            str(int(facts['n_placed_products']))
        macros[f'{prefix}PlacedProductsSold'] = \
            str(int(facts['n_placed_products_in_reference']))
        macros[f'{prefix}RefInvoices'] = \
            f"{int(facts['n_reference_invoices']):,}"
        macros[f'{prefix}RefInvoicesPlaced'] = \
            f"{int(facts['n_reference_invoices_with_placed']):,}"
        # The anonymous invoices among the store's lists and among the
        # reference invoices on its products (R27).
        for key, macro in (('anonymous_list_share', 'AnonListShare'),
                           ('anonymous_list_item_share',
                            'AnonListItemShare')):
            if facts.get(key) is not None:
                macros[f'{prefix}{macro}'] = upct(100 * float(facts[key]), 1)
            ref = (facts.get('anonymous_lists_reference') or {}).get(key)
            if ref is not None:
                macros[f'{prefix}Ref{macro}'] = upct(100 * float(ref), 1)
        macros.update(_replica_macros(prefix, facts.get('replica') or {}))
    # The live store every live diagnostic measures is the in-sample
    # design's: the current period, laid out by the same builder. Its
    # shoppers draw their list lengths from the current invoices cut to
    # the products it stocks, one value per invoice holding at least one
    # of them, so the sample's size is that invoice count.
    ins = design['in_sample']
    macros['LiveStoreProducts'] = str(int(ins['n_placed_products']))
    macros['LiveStoreListSampleN'] = \
        f"{int(ins['n_reference_invoices_with_placed']):,}"
    macros.update(live_load_macros(design['periods']['current'],
                                   s.get('load') or {},
                                   (s.get('list_chain') or {}).get('counts')
                                   or {}))

    def _list_stats(prefix, stats):
        if not stats or not stats.get('n'):
            return
        macros[f'{prefix}Median'] = f"{stats['median']:.0f}"
        macros[f'{prefix}Mean'] = f"{stats['mean']:.1f}"
        macros[f'{prefix}PNinety'] = f"{stats['p90']:.0f}"
    _list_stats('LiveStoreList', ins.get('list_length_agents'))
    _list_stats('HeldOutStoreList', design['held_out'].get('list_length_agents'))
    _list_stats('HeldOutRefList',
                design['held_out'].get('list_length_reference'))
    # How far the two periods' own invoices differ on the held-out
    # store's products: the yardstick for the held-out distances. The
    # category one is the category row's own test with the prior period's
    # invoices in the simulated visits' place. Then the same on the
    # products both stores stock -- the fixed assortment -- and the
    # remainder, popularity turnover (R16).
    ho = design['held_out']
    for block, stem in (('year_shift', 'GofYearShift'),
                        ('year_shift_fixed_assortment', 'GofYearShiftFixed')):
        for tag, key in (('Basket', 'basket'), ('Revenue', 'revenue'),
                         ('Category', 'category')):
            ys = (ho.get(block) or {}).get(key)
            if ys:
                fmt = '.1f' if key == 'category' else '.3f'
                macros[f'{stem}{tag}Stat'] = format(ys['statistic'], fmt)
                macros[f'{stem}{tag}P'] = pfmt(ys['p_value'])
                if key == 'category' and ys.get('share_tv_distance') is not None:
                    macros[f'{stem}CategoryShareTV'] = \
                        f"{ys['share_tv_distance']:.3f}"
    fa = ho.get('fixed_assortment') or {}
    if fa.get('n_products') is not None:
        macros['GofFixedAssortmentN'] = str(int(fa['n_products']))
    for tag, key in (('Basket', 'basket'), ('Revenue', 'revenue'),
                     ('Category', 'category')):
        t = (ho.get('popularity_turnover') or {}).get(key)
        if t:
            macros[f'GofTurnover{tag}'] = f"{t['popularity_turnover']:+.3f}"
    # Stocked products per invoice of both periods on each assortment:
    # the held-out store's (prior-selected), the in-sample store's
    # (current-selected) and the shared one. Held out, the store period is
    # the prior one and the reference period the current one.
    spi = ho.get('stocked_per_invoice') or {}
    for key, tag in (('store', 'PriorSet'), ('other_design_store',
                                             'CurrentSet'),
                     ('fixed', 'Shared')):
        rec = spi.get(key) or {}
        for period, ptag in (('store_period', 'Prior'),
                             ('reference_period', 'Current')):
            if rec.get(period) is not None:
                macros[f'GofStockedPerInv{tag}{ptag}'] = \
                    f"{float(rec[period]):.2f}"
    return macros


def pooled_balk_macros(abm, struct, queue, gof):
    """One pooled balk share for the nominal configuration (review R66),
    over the four live families that report one -- ABM, structural sweep,
    queueing at nominal load, in-sample goodness of fit -- with a
    percentile interval from a bootstrap of replications, stratified by
    family (each family's replications resampled within it). The
    structural sweep's settings share their replication seeds (common
    random numbers), so each of its replication blocks is one unit."""
    strata = []
    strata.append([(int(r['load']['arrivals']), int(r['load']['balked']))
                   for r in abm['per_rep']])
    blocks = defaultdict(lambda: [0, 0])
    for row in struct['rows']:
        for run in row['runs']:
            key = run.get('rep', run.get('seed'))
            blocks[key][0] += int(run['customers']) + int(run['balked'])
            blocks[key][1] += int(run['balked'])
    strata.append([tuple(v) for _, v in sorted(blocks.items())])
    strata.append([(int(r['arrivals']), int(r['balked']))
                   for r in queue['nominal']['per_rep']])
    strata.append([(int(r['load']['arrivals']), int(r['load']['balked']))
                   for r in gof['per_rep']])
    arr = sum(a for st in strata for a, _ in st)
    bal = sum(b for st in strata for _, b in st)
    if arr <= 0 or any(not st for st in strata):
        return {}

    def _boot_ci(sel):
        rng = np.random.default_rng(0)
        boots = np.empty(N_BOOT)
        arrays = [np.asarray(st, dtype=float) for st in sel]
        for i in range(N_BOOT):
            a = b = 0.0
            for x in arrays:
                pick = x[rng.integers(0, len(x), len(x))]
                a += pick[:, 0].sum()
                b += pick[:, 1].sum()
            boots[i] = b / a if a > 0 else 0.0
        return np.percentile(boots, [2.5, 97.5])

    lo, hi = _boot_ci(strata)
    out = {'LiveBalkedPooled': f"{100 * bal / arr:.2f}\\%",
           'LiveBalkedPooledCI': f"[{100 * lo:.2f}\\%, {100 * hi:.2f}\\%]",
           'LiveBalkedPooledCount': count_of(bal, arr),
           'LiveBalkedPooledReps': str(sum(len(st) for st in strata)),
           # The pooled share is balks over arrivals summed across the
           # families, i.e. weighted by each family's arrivals -- which the
           # structural sweep (five settings per replication) dominates.
           'LiveBalkedPooledWeighting': 'arrival-weighted'}
    # Each family's own share, at the pooled share's precision and with
    # its counts, so the weighting is visible.
    for tag, st in zip(POOLED_BALK_TAGS, strata):
        a_f = sum(a for a, _ in st)
        b_f = sum(b for _, b in st)
        out[f'LiveBalked{tag}'] = f"{100 * b_f / max(a_f, 1):.2f}\\%"
        out[f'LiveBalked{tag}Count'] = count_of(b_f, a_f)
    # The three families other than the structural sweep, pooled the same
    # way: the nominal configuration without the sweep's weight.
    rest = [st for tag, st in zip(POOLED_BALK_TAGS, strata)
            if tag != 'Struct']
    a_r = sum(a for st in rest for a, _ in st)
    b_r = sum(b for st in rest for _, b in st)
    lo_r, hi_r = _boot_ci(rest)
    out.update({
        'LiveBalkedNonStruct': f"{100 * b_r / a_r:.2f}\\%",
        'LiveBalkedNonStructCI':
            f"[{100 * lo_r:.2f}\\%, {100 * hi_r:.2f}\\%]",
        'LiveBalkedNonStructCount': count_of(b_r, a_r),
        'LiveBalkedNonStructReps': str(sum(len(st) for st in rest))})
    return out


# The families pooled_balk_macros pools, in its strata order.
POOLED_BALK_TAGS = ('Abm', 'Struct', 'QueueNom', 'Gof')


def align_macros(d):
    """The objective-alignment diagnostic (review R08): the rank agreement
    of the GA's exact fitness and the analytical revenue, and the cross-
    objective reversal, with the synthetic traffic prior as every
    experiment uses it and with the calibrated prior's wall term added."""
    macros = {}
    s = _summ(d)
    v = s['variants']
    ch = s['change_as_now_to_wall_aligned']
    n = int(v['as_now']['n_scenarios'])
    for key, tag in (('as_now', 'Now'), ('wall_aligned', 'Wall')):
        st = v[key]
        macros.update({
            f'Align{tag}SpearmanMean': f"{st['spearman_fisher_mean']:+.3f}",
            f'Align{tag}SpearmanMedian': f"{st['spearman_median']:+.3f}",
            f'Align{tag}SpearmanMin': f"{st['spearman_min']:+.3f}",
            f'Align{tag}SpearmanMax': f"{st['spearman_max']:+.3f}",
            f'Align{tag}Reversals': count_of(st['n_reversal'], n),
            f'Align{tag}FitnessPrefersGA':
                count_of(st['n_fitness_prefers_ga'], n),
            f'Align{tag}AnalyticalPrefersRef':
                count_of(st['n_analytical_prefers_reference'], n),
            f'Align{tag}GapFitness': spct(st['gap_fitness_pct_median'], 2),
            f'Align{tag}GapAnalytical':
                spct(st['gap_analytical_pct_median'], 2),
        })
    macros['AlignWallCoef'] = f"{float(v['wall_aligned']['wall_coef']):g}"
    macros['AlignSpearmanDiffMean'] = f"{ch['spearman_diff_mean']:+.3f}"
    macros['AlignSpearmanDiffMedian'] = f"{ch['spearman_diff_median']:+.3f}"
    macros['AlignRhoUp'] = count_of(ch['n_scenarios_rho_up'], n)
    macros['AlignNScen'] = str(n)
    cfg = s['config']
    macros['AlignNSamples'] = str(int(cfg['n_samples']))
    macros['AlignGens'] = str(int(cfg['n_gens']))
    macros['AlignPop'] = str(int(cfg['pop_size']))
    macros['AlignOracleRestarts'] = str(int(s['oracle']['n_restarts']))
    return macros


# The real-data budget sweep's searches and their macro names.
BUDGET_METHODS = {'GA': 'GA', 'random_search': 'RandSearch',
                  'simulated_annealing': 'SA',
                  'simulated_annealing_k': 'SAK'}
_METHOD_TEXT = {'GA': 'the GA', 'random_search': 'random search',
                'simulated_annealing': 'annealing',
                'simulated_annealing_k': 'k-item annealing'}


def budget_macros(d):
    """The real-data budget sweep (review R05): every search at 1x, 3x and
    10x Figure C's budget over several search seeds, its gap to the best-
    known layout as a share of Figure C's lift (closed form and held-out
    Monte Carlo), its lift, the GA's margin over each other search per
    budget, and the closed-form check (R02)."""
    macros = {}
    s = _summ(d)
    r, des = s['results'], s['design']
    mults = [int(m) for m in des['budget_multipliers']]
    macros['BudgetMultipliers'] = _num_list(mults, '{}$\\times$')
    macros['BudgetNSeeds'] = str(int(des['n_search_seeds']))
    macros['BudgetSAKMove'] = str(int(des['sa_k_move']))
    macros['BudgetFinalEvals'] = f"{int(des['final_evals']):,}"
    for m in mults:
        macros[f'BudgetSearchEvals{num_word(m)}'] = \
            f"{int(des['search_evals_per_multiplier'][str(m)]):,}"
    ref = r['reference']
    macros['BudgetRefMethod'] = _METHOD_TEXT.get(ref['method'],
                                                 tex_text(ref['method']))
    macros['BudgetRefMultiplier'] = f"{int(ref['multiplier'])}$\\times$"
    macros['BudgetRefLift'] = pounds(ref['lift_over_asbuilt_cf'])
    macros['BudgetRefLiftPct'] = spct(ref['lift_over_asbuilt_pct'])
    fl = r['figc_lift']
    macros['BudgetFigCLift'] = pounds(fl['closed_form'])
    macros['BudgetFigCLiftPct'] = spct(fl['pct_cf'])
    for meth, rows in r['per_method'].items():
        short = BUDGET_METHODS.get(meth)
        if not short:
            continue
        for mult, row in rows.items():
            w = num_word(int(mult))
            stem = f'BudgetGap{short}{w}'
            macros[stem] = pounds(row['gap_cf']['mean'])
            share = row['gap_cf_share_of_figc_lift']
            if share.get('mean') is not None:
                macros[f'{stem}Share'] = upct(100 * share['mean'], 1)
                macros[f'{stem}ShareRange'] = \
                    (f"[{100 * share['min']:.1f}\\%, "
                     f"{100 * share['max']:.1f}\\%]")
            if row.get('gap_mc_share_of_figc_lift') is not None:
                macros[f'{stem}MCShare'] = \
                    upct(100 * row['gap_mc_share_of_figc_lift'], 1)
            macros[f'BudgetLift{short}{w}Pct'] = \
                spct(row['lift_cf']['pct_mean'])
    cfc = r['closed_form_conclusions_unchanged']
    macros['BudgetCFUnchanged'] = yesno(cfc['all_unchanged'])
    # Every held-out MC difference also as a share of the as-built layout's
    # held-out MC revenue -- the scale the lifts are quoted on -- so a GA-
    # minus-X margin can be set beside a lift without dividing macros. The
    # as-built revenue is one number for every seed and budget (the same
    # layout under the same evaluation seeds), so the interval scales with
    # the difference.
    base_mc = float(r['baseline']['held_out_mc_mean'])
    for mult, blk in (cfc.get('per_multiplier') or {}).items():
        w = num_word(int(mult))
        for label, rec in (blk.get('ga_minus') or {}).items():
            tag = 'Lift' if label == 'lift_over_asbuilt' \
                else 'GAvs' + BUDGET_METHODS.get(label, '')
            if tag == 'GAvs':
                continue
            stem = f'Budget{tag}{w}MC'
            mc = rec['mc']
            macros[stem] = pounds(mc['mean'])
            macros[f'{stem}Pct'] = spct(100 * float(mc['mean']) / base_mc)
            if mc.get('ci_lo') is not None and np.isfinite(mc['ci_lo']):
                macros[f'{stem}CI'] = \
                    f"[{pounds(mc['ci_lo'])}, {pounds(mc['ci_hi'])}]"
                macros[f'{stem}PctCI'] = (
                    f"[{spct(100 * float(mc['ci_lo']) / base_mc)}, "
                    f"{spct(100 * float(mc['ci_hi']) / base_mc)}]")
            if mc.get('excludes_zero') is not None:
                macros[f'{stem}Sig'] = yesno(mc['excludes_zero'])
            macros[f'{stem}Wins'] = count_of(mc['seeds_ga_ahead'],
                                             rec['n_seeds'])
    return macros


def input_macros(d):
    """Input uncertainty of Figure C (review R11): the percentage lift's
    customer-clustered bootstrap interval, delta-method intervals for the
    GBP level, lift and the GA's margins, on the whole-business and the
    stocked scale; the anonymous invoices' shares (R27); and the former
    spend law's bias at both sets of inputs."""
    macros = {}
    s = _summ(d)
    iu, pt, des = s['input_uncertainty'], s['point'], s['design']
    n_boot = int(iu['n_boot'])
    macros['InputNBoot'] = str(n_boot)
    macros['InputCILevel'] = level_pct(100 * float(iu['level']))
    cl = des['clusters']
    macros['InputNClusters'] = f"{int(cl['n_clusters']):,}"
    macros['InputNCustomers'] = f"{int(cl['n_customers']):,}"
    macros['InputNAnonInvoices'] = f"{int(cl['n_anonymous_invoices']):,}"
    macros['InputScaleRatio'] = upct(100 * float(iu['scale_ratio']), 2)
    macros['InputStockedInvoiceShare'] = \
        upct(100 * float(pt['stocked_invoice_share']), 1)
    macros['InputHorizonDays'] = str(int(des['horizon_days']))
    # The percentage is the same on both scales by construction.
    pl = iu['whole']['pct_lift']
    macros.update({
        'InputPctLift': spct(pl['estimate'], 2),
        'InputPctLiftCI': f"[{pl['ci'][0]:+.2f}\\%, {pl['ci'][1]:+.2f}\\%]",
        'InputPctLiftBootMean': spct(pl['boot_mean'], 2),
        'InputPctLiftBootBias': f"{pl['boot_bias']:+.2f}",
        'InputPctLiftBootSE': f"{pl['boot_se']:.2f}",
    })
    for scale, tag in (('whole', 'Whole'), ('stocked', 'Stocked')):
        b = iu[scale]
        lv, lf = b['level_gbp'], b['lift_gbp']
        macros.update({
            f'InputLevel{tag}': pounds_amount(lv['estimate']),
            f'InputLevel{tag}CI': (f"[{pounds_amount(lv['ci'][0])}, "
                                   f"{pounds_amount(lv['ci'][1])}]"),
            f'InputLevel{tag}RSE': upct(100 * float(lv['rse']), 1),
            f'InputLift{tag}': pounds(lf['estimate']),
            f'InputLift{tag}CI': (f"[{pounds(lf['ci'][0])}, "
                                  f"{pounds(lf['ci'][1])}]"),
        })
        for cmp_, short in (('rs', 'RandSearch'), ('sa', 'SA')):
            g = b[f'ga_minus_{cmp_}']
            macros[f'InputGAvs{short}{tag}'] = pounds(g['gbp']['estimate'])
            macros[f'InputGAvs{short}{tag}CI'] = \
                f"[{pounds(g['gbp']['ci'][0])}, {pounds(g['gbp']['ci'][1])}]"
            if scale == 'whole':
                p = g['pct']
                macros[f'InputGAvs{short}Pct'] = spct(p['estimate'], 2)
                macros[f'InputGAvs{short}PctCI'] = \
                    f"[{p['ci'][0]:+.2f}\\%, {p['ci'][1]:+.2f}\\%]"
                macros[f'InputGAvs{short}BootAhead'] = \
                    count_of(g['n_boot_ga_ahead'], n_boot)
    an = s.get('anonymous') or {}
    for key, macro in (('invoice_share', 'AnonInvoiceShare'),
                       ('purchase_share', 'AnonPurchaseShare'),
                       ('revenue_share', 'AnonRevenueShare')):
        if an.get(key) is not None:
            macros[macro] = upct(100 * float(an[key]), 1)
    for key, macro in (('mean_size_anonymous', 'AnonMeanSize'),
                       ('mean_size_identified', 'IdentMeanSize')):
        if an.get(key) is not None:
            macros[macro] = f"{float(an[key]):.1f}"
    for key, macro in (('mean_revenue_anonymous', 'AnonMeanRevenue'),
                       ('mean_revenue_identified', 'IdentMeanRevenue')):
        if an.get(key) is not None:
            macros[macro] = '\\pounds ' + f"{float(an[key]):.2f}"
    lfb = s['legacy_floor_bias']
    macros['LegacyFloorBiasHorizonDays'] = str(int(lfb['horizon_days']))
    for key, tag in (('current_inputs', 'Current'),
                     ('adapter_1_1_inputs', 'OneOne')):
        rec = lfb.get(key) or {}
        if rec.get('asbuilt') is None:
            continue
        macros.update({
            f'LegacyFloorBias{tag}Level': pounds2(rec['asbuilt'],
                                                  signed=False),
            f'LegacyFloorBias{tag}LevelPct':
                upct(100 * float(rec['frac_of_asbuilt_level']), 3),
            f'LegacyFloorBias{tag}Lift': pounds2(rec['lift']),
            f'LegacyFloorBias{tag}LiftPct':
                spct(100 * float(rec['frac_of_lift']), 2),
            f'LegacyFloorBias{tag}SpendCV': f"{float(rec['spend_cv']):.2f}",
        })
    macros['InputAbsentFixturesMax'] = str(int(s['absent_fixtures']['max']))
    return macros


def transfer_macros(d):
    """Held-out transfer (review R12): per search, the lift promised on
    the prior period, transferred to the current one and found in
    hindsight on it, the paired transfer gap with its interval over the
    search seeds, the ratio of the means, and the GA's transferred lift
    against the comparators'; the fixtures and the periods."""
    macros = {}
    s = _summ(d)
    a, des = s['across_seeds'], s['design']
    n = int(a['n_seeds'])
    macros['TransferNSeeds'] = str(n)
    macros['TransferCILevel'] = level_pct(100 * float(a['level']))
    for meth, short in (('GA', 'GA'), ('random_search', 'RandSearch'),
                        ('simulated_annealing', 'SA')):
        b = a['methods'][meth]
        stem = f'Transfer{short}'
        for lift, tag in (('promised', 'Promised'),
                          ('transferred', 'Transferred'),
                          ('hindsight', 'Hindsight')):
            macros[f'{stem}{tag}'] = pounds(b[lift]['gbp']['mean'])
            macros[f'{stem}{tag}Pct'] = spct(b[lift]['pct']['mean'], 2)
            lo, hi = b[lift]['pct']['ci']
            if lo is not None and np.isfinite(lo):
                macros[f'{stem}{tag}PctCI'] = \
                    f"[{lo:+.2f}\\%, {hi:+.2f}\\%]"
        for lift, tag in (('transferred', 'Transferred'),
                          ('hindsight', 'Hindsight')):
            macros[f'{stem}{tag}MC'] = pounds(b[lift]['mc_gbp']['mean'])
        g = b['transfer_gap']
        lo, hi = g['gbp']['ci']
        macros[f'{stem}Gap'] = pounds(g['gbp']['mean'])
        if lo is not None and np.isfinite(lo):
            macros[f'{stem}GapCI'] = f"[{pounds(lo)}, {pounds(hi)}]"
            macros[f'{stem}GapSig'] = yesno(lo > 0 or hi < 0)
        # The gap between two percentage lifts is in percentage points,
        # and says so: a bare '+0.63' beside the lifts' '%' reads as either.
        macros[f'{stem}GapPct'] = pp(g['pct']['mean'])
        plo, phi = g['pct']['ci']
        if plo is not None and np.isfinite(plo):
            macros[f'{stem}GapPctCI'] = f"[{plo:+.2f}, {phi:+.2f}]\\,pp"
        macros[f'{stem}GapMC'] = pounds(g['mc_gbp']['mean'])
        macros[f'{stem}GapShort'] = \
            count_of(g['n_seeds_transferred_below_hindsight'], n)
        r = b.get('transfer_ratio_of_means')
        macros[f'{stem}Ratio'] = f"{float(r):.2f}" if r is not None else 'n/a'
        # The share of the projected lift drift costs, against each
        # yardstick: one minus the transferred mean percentage lift over the
        # hindsight one (same period) or the promised one (prior period).
        tp = float(b['transferred']['pct']['mean'])
        for lift, tag in (('hindsight', 'Hindsight'), ('promised', 'Promised')):
            ref = float(b[lift]['pct']['mean'])
            if ref > 0:
                macros[f'{stem}ShareLost{tag}'] = upct(100 * (1 - tp / ref), 0)
        macros[f'{stem}Positive'] = \
            count_of(b['n_seeds_transferred_positive'], n)
    for key, short in (('GA_minus_random_search', 'RandSearch'),
                       ('GA_minus_simulated_annealing', 'SA')):
        v = a['transferred_ga_vs'][key]
        macros[f'TransferGAvs{short}'] = pounds(v['mean'])
        lo, hi = v['ci']
        if lo is not None and np.isfinite(lo):
            macros[f'TransferGAvs{short}CI'] = f"[{pounds(lo)}, {pounds(hi)}]"
            macros[f'TransferGAvs{short}Sig'] = yesno(lo > 0 or hi < 0)
        macros[f'TransferGAvs{short}Leads'] = count_of(v['n_ga_leads'], n)
    fx = s['fixtures']
    macros['TransferFixtures'] = str(int(fx['n_fixtures']))
    macros['TransferAbsent'] = str(int(fx['n_absent_in_current']))
    macros['TransferTopSize'] = str(int(fx['current_top_n_size']))
    macros['TransferTopNotStocked'] = str(int(fx['current_top_n_not_stocked']))
    per = des['periods']
    for period, tag in (('prior', 'Prior'), ('current', 'Current')):
        macros[f'Transfer{tag}From'] = str(per[period]['first_date'])
        macros[f'Transfer{tag}To'] = str(per[period]['last_date'])
        macros[f'Transfer{tag}Invoices'] = \
            f"{int(des['n_invoices_calibrated'][period]):,}"
    bl = s['baseline']['closed_form']
    macros['TransferBaselinePrior'] = pounds_amount(bl['prior'])
    macros['TransferBaselineCurrent'] = pounds_amount(bl['current'])
    return macros


def _status_text(label):
    """The protocol study's status label as words: 'reproduced; drift not
    equivalent'."""
    head, sep, tail = str(label).partition('_drift_')
    words = head.replace('_', ' ')
    return f"{words}; drift {tail.replace('_', ' ')}" if sep else words


def proto_macros(d):
    """The live protocol study (review R19): the protocol it derives,
    whether that reproduces the runners' constants, the drift after the
    warm-up with its equivalence test, the double warm-up on the outcomes
    the runners report, and the stress level."""
    macros = {}
    s = _summ(d)
    dv = s['derived']
    macros.update({
        'ProtoNominalSpawn': f"{float(dv['NOMINAL_SPAWN']):.2f}",
        'ProtoNominalCap': str(int(dv['NOMINAL_CAP'])),
        'ProtoWarmup': f"{float(dv['WARMUP_S']):,.0f}",
        'ProtoCollect': f"{float(dv['COLLECT_S']):,.0f}",
        'ProtoStressSpawn': f"{float(dv['STRESS_SPAWN']):.2f}",
        'ProtoStressCap': str(int(dv['STRESS_CAP'])),
        'ProtoMatches': yesno(s['matches']),
        'ProtoNDifferences': str(len(s.get('differences') or [])),
        'ProtoStatusLabel': tex_text(s['protocol_status']),
        'ProtoStatus': tex_text(_status_text(s['protocol_status'])),
    })
    de = s.get('drift_equivalent')
    macros['ProtoDriftEquivalent'] = ('not tested' if de is None
                                      else yesno(de))
    des = s['design']
    nom, st = des['nominal'], des['stress']
    macros.update({
        'ProtoReps': str(int(nom['reps'])),
        'ProtoRunS': f"{float(nom['run_s']):,.0f}",
        'ProtoGridLo': f"{min(nom['grid']):.2f}",
        'ProtoGridHi': f"{max(nom['grid']):.2f}",
        'ProtoGridN': str(len(nom['grid'])),
        'ProtoStressGridN': str(len(st['grid'])),
        'ProtoCapGridN': str(len(st['cap_grid'])),
        'ProtoStressReps': str(int(st['rate_reps'])),
        'ProtoCapReps': str(int(st['cap_reps'])),
        'ProtoStressRunS': f"{float(st['run_s']):,.0f}",
        'ProtoNRuns': f"{int(des['n_runs']):,}",
        'ProtoProfileMult': f"{float(des['profile_multiplier']):.3f}",
        'ProtoStartHour': str(des['start_hour']),
    })
    th = _get(s, 'rules', 'thresholds') or {}
    if th.get('CAP_SHARE_MAX') is not None:
        macros['ProtoCapShareMax'] = upct(100 * float(th['CAP_SHARE_MAX']), 0)
    if th.get('MIN_COHORT') is not None:
        macros['ProtoMinCohort'] = str(int(th['MIN_COHORT']))
    dr = s['drift']
    macros.update({
        'ProtoDriftReps': str(int(dr['n_reps'])),
        'ProtoDriftSlope': f"{float(dr['slope_per_1000s_mean']):.2f}",
        'ProtoDriftSlopeCI': f"{float(dr['slope_per_1000s_ci95']):.2f}",
        'ProtoDriftSlopeP': pfmt(dr['slope_p']),
        'ProtoDriftChangePct':
            spct(100 * float(dr['relative_change_over_window_mean']), 1),
        'ProtoDriftChangeCI':
            upct(100 * float(dr['relative_change_ci95']), 1),
        'ProtoDriftChangeCINinety':
            upct(100 * float(dr['relative_change_ci90']), 1),
        'ProtoDriftMargin': upct(100 * float(dr['tost_margin']), 0),
        'ProtoDriftTostP': pfmt(dr['tost_p']),
        'ProtoDriftEarlyLateP': pfmt(dr['early_minus_late_p']),
    })
    w = s.get('window') or {}
    if w.get('km_median_visit_s') is not None:
        macros['ProtoKMMedianVisit'] = f"{float(w['km_median_visit_s']):.0f}"
    cm = w.get('cohort_min_at_window')
    if isinstance(cm, dict) and cm.get('min') is not None:
        macros['ProtoCohortMin'] = str(int(cm['min']))
    for key, macro in (('km_mean_visit_s_rule_pool', 'ProtoKMMeanVisitRule'),
                       ('km_mean_visit_s_after_warmup',
                        'ProtoKMMeanVisitAfter')):
        if w.get(key) is not None:
            macros[macro] = f"{float(w[key]):.0f}"
    grid = s.get('nominal_grid') or {}
    nominal = float(dv['NOMINAL_SPAWN'])
    row = grid.get(f'{nominal:.2f}') or {}
    if row.get('cap_share_after_warmup') is not None:
        macros['ProtoCapShareNominal'] = \
            upct(100 * float(row['cap_share_after_warmup']), 2)
    above = sorted(float(k) for k in grid if float(k) > nominal + 1e-9)
    if above:
        nxt = grid[f'{above[0]:.2f}']
        macros['ProtoRateAbove'] = f"{above[0]:.2f}"
        if nxt.get('cap_share_after_warmup') is not None:
            macros['ProtoCapShareAbove'] = \
                upct(100 * float(nxt['cap_share_after_warmup']), 2)
    dbl = s['double_warmup']
    stats = dbl.get('stats') or {}
    ps = []
    for key, tag in PROTO_OUTCOMES:
        r = stats.get(key)
        if not isinstance(r, dict):
            continue
        if r.get('diff_pct') is not None:
            macros[f'ProtoDbl{tag}DiffPct'] = spct(r['diff_pct'], 1)
        if r.get('diff_ci95') is not None and r.get('after_1x_warmup'):
            macros[f'ProtoDbl{tag}CIPct'] = upct(
                100 * float(r['diff_ci95']) / abs(float(r['after_1x_warmup'])),
                1)
        if r.get('p') is not None:
            macros[f'ProtoDbl{tag}P'] = pfmt(r['p'])
            ps.append(float(r['p']))
    # The live store's conversion itself, after the runners' warm-up
    # (review R04): the share of the window's visitors who pay, at the
    # nominal load -- far above the 0.30 the Monte Carlo engine assumes,
    # since a live agent's list is a whole invoice of a buyer.
    conv = stats.get('conversion')
    if isinstance(conv, dict) and conv.get('after_1x_warmup') is not None:
        macros['ProtoDblConversionLevel'] = \
            upct(100 * float(conv['after_1x_warmup']), 1)
    if ps:
        macros['ProtoDblMinP'] = pfmt(min(ps))
        macros['ProtoDblNSig'] = count_of(sum(p < 0.05 for p in ps), len(ps))
    if stats.get('cohort_censored') is not None:
        macros['ProtoDblCensored'] = str(int(stats['cohort_censored']))
    if dbl.get('mean_drain_s') is not None:
        macros['ProtoDblDrain'] = f"{float(dbl['mean_drain_s']):.0f}"
    sc = _get(s, 'stress', 'cap_scan', str(int(dv['STRESS_CAP']))) or {}
    if sc.get('utilisation') is not None:
        macros['ProtoStressUtil'] = upct(100 * float(sc['utilisation']), 1)
    if sc.get('cap_share') is not None:
        macros['ProtoStressCapShare'] = upct(100 * float(sc['cap_share']), 2)
    aw = _get(s, 'stress', 'analytic_warmup_at_stress') or {}
    if aw.get('warmup_s') is not None:
        macros['ProtoStressWarmup'] = f"{float(aw['warmup_s']):,.0f}"
    return macros


def profile_macros(d):
    """What the paper says about UCI's calendar, from the profile run:
    Saturday invoices, the Saturdays that traded out of those in the span,
    and October-December's share of revenue against its share of days."""
    s = _summ(d)
    sat = s['weekday']['Saturday']
    q4 = s['fourth_quarter']
    return {
        'UCISaturdayInvoices': f"{int(sat['invoices']):,}",
        'UCISaturdayInvoiceShare': upct(100 * float(sat['invoice_share']), 2),
        'UCISaturdaysTraded': count_of(sat['trading_days'],
                                       sat['days_in_span']),
        'UCIQFourRevenueShare': upct(100 * float(q4['revenue_share']), 1),
        'UCIQFourDayShare': upct(100 * float(q4['day_share']), 1),
    }


def heatmap_macros(d):
    """The seeded heat-map run behind the emergent heat map figure."""
    s = _summ(d)
    out = {'HeatmapSeed': str(int(s['protocol']['seed'])),
           'HeatmapPerimRatioNull':
               f"{float(s['perimeter_interior_ratio_geometric_null']):.2f}"}
    pr = _get(s, 'perimeter_ratio', 'perimeter_interior_ratio')
    if pr is not None:
        out['HeatmapPerimRatio'] = f"{float(pr):.2f}"
    if _get(s, 'load', 'arrivals') is not None:
        out['HeatmapArrivals'] = f"{int(s['load']['arrivals']):,}"
    return out


# --- Table 4: experiment budgets ------------------------------------------------

# --- The constants registry and the release (no artifact behind them) ----------

def _reg_name(name):
    """``SYNTH_DETOUR_ELASTICITY_RANGE`` -> ``RegSynthDetourElasticityRange``;
    a run of digits is spelled out (TeX control words are letters only)."""
    out = 'Reg'
    for part in name.lower().split('_'):
        if not part:
            continue
        for tok in re.findall(r'[a-z]+|\d+', part):
            out += num_word(int(tok)) if tok.isdigit() else tok.capitalize()
    return out


def _reg_number(v):
    """A registry number as macro text: integers (and integral floats
    below 10^5) plainly, other floats to four significant digits, powers of
    ten in math mode."""
    if isinstance(v, bool):
        return yesno(v)
    if isinstance(v, int) or (isinstance(v, float) and v.is_integer()
                              and abs(v) < 1e5):
        return f"{int(v):,}"
    s = f"{float(v):.4g}"
    if 'e' in s:
        mant, exp = s.split('e')
        base = f"10^{{{int(exp)}}}"
        return (f"\\ensuremath{{{base}}}" if mant in ('1', '1.0')
                else f"\\ensuremath{{{mant}\\times {base}}}")
    return s


def _is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def registry_macros():
    """Every constant of the registry (``retail_literature``) as a macro
    (review R18, R41), read from its source like every design check here,
    so Tables 6-7 quote the values the code runs with -- and a changed
    constant, which already retires every artifact made under the old one
    (``_model_current``), changes the table with it.

    ``Reg<Name>`` per constant; a two-element range as ``...Lo`` / ``...Hi``;
    a longer tuple as its comma-separated values; a mapping per key
    (``RegWcProbabilityByTypeQuick``; ``RegListLengthByTypeQuickLo``); text
    escaped. Private names, the citation table and values that cannot be
    read from the source are left out. Derived: ``RegWcShareOfAgents``, the
    share of agents who visit the restroom under the type mix."""
    vals = _module_values(LIT_SCRIPT)
    out = {}

    def put(name, text):
        if name in out:
            raise RuntimeError(f"registry macro {name} defined twice")
        out[name] = text

    for name, v in vals.items():
        if not re.fullmatch(r'[A-Z][A-Z0-9_]*', name) or name == 'CITATIONS':
            continue
        m = _reg_name(name)
        if _is_number(v):
            put(m, _reg_number(v))
        elif isinstance(v, str):
            put(m, tex_text(v))
        elif isinstance(v, (tuple, list)) and v:
            if len(v) == 2 and all(_is_number(x) for x in v):
                put(m + 'Lo', _reg_number(v[0]))
                put(m + 'Hi', _reg_number(v[1]))
            elif all(_is_number(x) for x in v):
                put(m, ', '.join(_reg_number(x) for x in v))
            elif all(isinstance(x, str) for x in v):
                put(m, tex_text(', '.join(v)))
        elif isinstance(v, dict) and v and all(isinstance(k, str) for k in v):
            for k, x in v.items():
                km = m + _reg_name(k)[len('Reg'):]
                if _is_number(x):
                    put(km, _reg_number(x))
                elif (isinstance(x, (tuple, list)) and len(x) == 2
                      and all(_is_number(y) for y in x)):
                    put(km + 'Lo', _reg_number(x[0]))
                    put(km + 'Hi', _reg_number(x[1]))
    types, probs = vals.get('CUSTOMER_TYPES'), vals.get('CUSTOMER_TYPE_PROBS')
    wc = vals.get('WC_PROBABILITY_BY_TYPE')
    if types and probs and isinstance(wc, dict):
        share = sum(p * wc[t] for t, p in zip(types, probs))
        put('RegWcShareOfAgents', upct(100 * share, 1))
    return out


# Where the release is described: the citation file the archive (Zenodo)
# and GitHub read, kept at the repository root.
CITATION_FILE = os.path.join(ROOT, 'CITATION.cff')


def read_citation(path=CITATION_FILE):
    """The top-level scalar fields of a CITATION.cff (``version``,
    ``date-released``, ``doi``, ...) and, as ``doi`` when there is no
    top-level one, the value of the first ``identifiers`` entry of type
    doi. A reader for the file's simple layout, not a YAML parser; None
    when the file is absent."""
    if not os.path.exists(path):
        return None
    fields, in_ids, id_type = {}, False, None
    with open(path, encoding='utf-8') as f:
        for raw in f:
            line = raw.rstrip('\n')
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            if not line[0].isspace():
                key, sep, val = line.partition(':')
                in_ids = key.strip() == 'identifiers'
                if sep and val.strip():
                    fields[key.strip()] = val.strip().strip('"\'')
                continue
            if in_ids:
                item = line.strip().lstrip('-').strip()
                key, sep, val = item.partition(':')
                val = val.strip().strip('"\'')
                if key.strip() == 'type':
                    id_type = val
                elif (key.strip() == 'value' and id_type == 'doi'
                      and 'doi' not in fields):
                    fields['doi'] = val
    return fields


def release_macros(path=CITATION_FILE):
    """The software release the paper cites (review R23), from the same
    CITATION.cff the archive reads: ``ReleaseVersion`` (``1.0.2``),
    ``ReleaseTag`` (``v1.0.2``), ``ReleaseDate`` and, once the archive has
    minted one, ``ReleaseDOI`` -- so the manuscript cannot cite a release
    other than the one the citation file names."""
    c = read_citation(path)
    if not c or not c.get('version'):
        return {}
    v = str(c['version'])
    out = {'ReleaseVersion': tex_text(v),
           'ReleaseTag': tex_text(v if v.startswith('v') else 'v' + v)}
    if c.get('date-released'):
        out['ReleaseDate'] = tex_text(str(c['date-released']))
    if c.get('doi'):
        out['ReleaseDOI'] = tex_text(str(c['doi']))
    return out


TBD = '\\textit{TBD}'


def _x():
    return '$\\times$'


def _n(n, word):
    """'1 rate', '6 rates'."""
    return f"{int(n):,} {word}{'' if int(n) == 1 else 's'}"


def _row_figa(d):
    a, s = _side(d)['args'], _summ(d)
    g, p, f = int(a['n_gens']), int(a['pop_size']), _final_seeds()
    return [f"{int(a['n_scenarios'])} scenarios {_x()} {int(a['n_seeds'])} "
            f"GA seeds",
            f"GA {g}{_x()}{p} = {g * p:,} + {p * f} final",
            f"MC {int(a['mc_iters']):,} iter.\\ {_x()} {int(a['mc_days'])} d; "
            f"reference {int(s['oracle']['n_restarts'])} restarts; "
            f"{int(a['n_spearman_samples'])} held-out layouts per scenario",
            str(int(a['n_items']))]


def _row_figb(d):
    side = _side(d)
    a, s = side['args'], _summ(d)
    n_methods = len({r['method'] for r in read_csv(os.path.join(
        d, 'results.csv'))})
    b = s['budget']
    return [f"{int(a['n_scenarios'])} {_x()} {int(a['n_seeds'])}; "
            f"{n_methods} layouts per run",
            f"GA, RS, SA each {int(b['search_evals']):,} + "
            f"{int(b['final_evals'])} final",
            f"MC {int(a['mc_iters']):,} iter.\\ {_x()} {int(a['mc_days'])} d "
            f"on held-out seeds; reference {int(s['oracle_restarts'])} "
            f"restarts",
            str(int(a['n_items']))]


def _row_mcgt(d):
    s, a = _summ(d), _side(d)['args']
    ratio = int(round(s['big_budget'] / max(s['normal_budget'], 1)))
    c = int(a['confirm_seeds'])
    return [f"{int(s['n_scenarios'])} scenarios of its own "
            f"({int(a['n_items'])} items; not those of Fig.~\\ref{{fig:regret}})",
            f"GA at {int(s['normal_budget']):,} vs.\\ GA, RS, SA and "
            f"popularity-start SA at {int(s['big_budget']):,} "
            f"({ratio}{_x()})",
            f"MC {int(s['mc_iters']):,} iter.\\ {_x()} {int(a['mc_days'])} d; "
            f"{c} selection + {c} confirmation seeds",
            str(int(a['n_items']))]


def _row_gasens(d):
    s = _summ(d)
    cfg = s['config']
    return [f"{int(cfg['n_scenarios'])} {_x()} {int(cfg['n_seeds'])}; "
            f"{int(s['n_settings'])} GA and {int(s['sa_sweep']['n_settings'])} "
            f"annealer settings",
            f"{int(s['budget_total_evals']):,} per GA setting "
            f"(pop.\\ {_x()} (gen.\\ + {_final_seeds()})); annealer "
            f"{int(s['sa_budget_search_evals']):,}",
            f"MC {int(cfg['mc_iters']):,} iter.\\ {_x()} {int(cfg['mc_days'])} "
            f"d; {int(s['heldout_seeds'])} held-out seeds",
            str(int(cfg['n_items']))]


def _row_lhs(d):
    s = _summ(d)
    cfg = s['config']
    return [f"{int(s['n_scenarios'])} {_x()} {int(s['n_seeds'])}; "
            f"{int(s['n_lhs_points'])} LHS points; "
            f"{int(s['weight_sweep']['design']['n_weight_draws'])} weight "
            f"vectors",
            f"GA {int(cfg['n_gens'])}{_x()}{int(cfg['pop_size'])}, RS and SA "
            f"equal; search horizon {int(cfg['mc_days'])} d",
            f"closed form over {int(s['horizon_days'])} d; MC "
            f"{int(cfg['mc_iters']):,} iter.\\ in the searches",
            str(int(cfg['n_items']))]


def _row_align(d):
    s = _summ(d)
    cfg = s['config']
    return [f"{int(cfg['n_scenarios'])} scenarios {_x()} 2 traffic priors; "
            f"{int(cfg['n_samples'])} layouts each",
            f"GA {int(cfg['n_gens'])}{_x()}{int(cfg['pop_size'])} on the "
            f"exact fitness",
            f"closed form over {int(cfg['horizon_days'])} d; reference "
            f"{int(s['oracle']['n_restarts'])} restarts",
            str(int(cfg['n_items']))]


def _figc_budget_cell(search, final):
    return f"GA, RS, SA each {int(search):,} + {int(final)} final"


def _row_figc(d):
    side = _side(d)
    a = side['args']
    s = _summ(d)
    sd = s['search_design']
    n_sheets = len([x for x in str(a['sheets']).split(',') if x.strip()])
    sheets = 'both sheets' if n_sheets == 2 else _n(n_sheets, 'sheet')
    return [f"UCI, {sheets}; one search per method (seed "
            f"{int(a['ga_seed'])})",
            _figc_budget_cell(sd['budget_search_evals'],
                              sd['budget_final_evals']),
            f"{int(a['n_mc_replicates'])} held-out paired replicates {_x()} "
            f"MC {int(a['mc_iters']):,} iter.\\ {_x()} {int(a['mc_days'])} d",
            str(int(side['shop_summary']['items_placed']))]


def _row_seeds(d):
    s = _summ(d)
    des = s['design']
    return [f"{int(des['n_search_seeds'])} search seeds from "
            f"{int(des['first_search_seed'])}",
            _figc_budget_cell(des['budget_search_evals'],
                              int(des['pop_size']) * int(des['n_final_seeds'])),
            f"{int(des['n_mc_replicates'])} replicates per seed, as Fig.~\\ref{{fig:figc}}",
            str(int(s['store']['items_placed']))]


def _row_budget(d):
    s = _summ(d)
    des = s['design']
    mults = [int(m) for m in des['budget_multipliers']]
    ev = des['search_evals_per_multiplier']
    return [f"{int(des['n_search_seeds'])} seeds {_x()} budgets "
            f"{', '.join(str(m) for m in mults)}{_x()}; GA, RS, SA and "
            f"{int(des['sa_k_move'])}-item SA",
            f"{int(ev[str(mults[0])]):,} to {int(ev[str(mults[-1])]):,} + "
            f"{int(des['final_evals'])} final",
            f"closed form; {int(des['n_mc_replicates'])} held-out MC "
            f"replicates",
            str(int(s['store']['items_placed']))]


def _row_input(d):
    s = _summ(d)
    des = s['design']
    return [f"B = {int(des['n_boot'])} customer-clustered resamples "
            f"({int(des['clusters']['n_clusters']):,} clusters)",
            f"the searches of Fig.~\\ref{{fig:figc}}, once ("
            f"{int(des['searches']['budget_search_evals']):,} each)",
            f"closed form over {int(des['horizon_days'])} d, two scales",
            str(int(s['store']['items_placed']))]


def _row_transfer(d):
    s = _summ(d)
    des = s['design']
    return [f"{int(des['n_search_seeds'])} seeds {_x()} 2 periods {_x()} 3 "
            f"searches",
            _figc_budget_cell(des['budget_search_evals'],
                              int(des['pop_size']) * int(des['n_final_seeds'])),
            f"closed form; {int(des['n_mc_replicates'])} MC replicates on "
            f"the current period",
            str(int(s['store']['items_placed']))]


def _row_proto(d, store_items=None):
    s = _summ(d)
    des = s['design']
    nom, st = des['nominal'], des['stress']
    dbl = s.get('double_warmup') or {}
    cells = [f"{_n(len(nom['grid']), 'rate')} {_x()} {_n(nom['reps'], 'rep')} "
             f"{_x()} {float(nom['run_s']):,.0f} s"]
    if dbl.get('reps'):
        cells.append(f"double warm-up {_n(dbl['reps'], 'rep')} {_x()} "
                     f"{float(dbl['run_s']):,.0f} s")
    stress = (f"stress {_n(len(st['grid']), 'rate')} {_x()} "
              f"{_n(st['rate_reps'], 'rep')}")
    if (s.get('stress') or {}).get('cap_scan'):
        stress += (f" and {_n(len(st['cap_grid']), 'cap')} {_x()} "
                   f"{_n(st['cap_reps'], 'rep')}")
    cells.append(f"{stress}, {float(st['run_s']):,.0f} s each")
    return ['; '.join(cells),
            '--',
            f"fixed step {float(des['dt']):g} s; {int(des['n_runs']):,} runs",
            _live_items_cell(s, store_items)]


def _live_window(proto):
    return (f"warm-up {float(proto['warmup_s']):,.0f} s + window "
            f"{float(proto['collect_s']):,.0f} s")


# What makes two live runs' stores the same store: the workbook, the period
# and how it was cleaned and calibrated, and how the store is laid out from
# it (``_live_store.period_record``).
LIVE_STORE_FIELDS = ('source_sha256', 'period', 'adapter_version',
                     'reader_version', 'n_invoices', 'list_law',
                     'exclude_anonymous', 'max_items_per_category',
                     'naive_layout')


def _live_store_key(data):
    """A live run's store, from its period record (None without one)."""
    if not isinstance(data, dict) or not data.get('source_sha256'):
        return None
    return tuple(json.dumps(data.get(k), sort_keys=True)
                 for k in LIVE_STORE_FIELDS)


def live_store_items(found):
    """Items placed in each live store, ``{store key: n}``, from the live
    runs that record the count: the goodness-of-fit run (its in-sample
    store) and the heat-map run. The agent-model, structural, queueing and
    protocol-study runners build the same store (``_live_store``) but do
    not record its size, so their Table 4 rows read it here, from a run
    whose period record matches theirs field for field. Two runs that
    record different counts for one store drop it (the rows then read --)."""
    got = defaultdict(set)
    d = found.get('validation_gof')
    if d:
        try:
            des = _summ(d)['design']
            ins = des['in_sample']
            key = _live_store_key(des['periods'][ins['store_period']])
            if key is not None:
                got[key].add(int(ins['n_placed_products']))
        except (KeyError, TypeError, ValueError):
            pass
    d = found.get('heatmap_figure')
    if d:
        try:
            s = _summ(d)
            key = _live_store_key(s.get('data'))
            if key is not None and _get(s, 'store', 'n_items') is not None:
                got[key].add(int(s['store']['n_items']))
        except (KeyError, TypeError, ValueError):
            pass
    return {k: next(iter(v)) for k, v in got.items() if len(v) == 1}


def _live_items_cell(s, store_items):
    """A live run's Items cell: its own recorded count, else the count
    another live run recorded for the identical store, else --."""
    n = _get(s, 'store', 'n_items')
    if n is None:
        n = (store_items or {}).get(_live_store_key(s.get('data')))
    return str(int(n)) if n is not None else '--'


def _row_abm(d, store_items=None):
    s = _summ(d)
    p = s['protocol']
    return [f"{int(p['reps'])} replications; {_live_window(p)}",
            '--',
            f"routing null {int(s['emergence']['routing_null']['n_tours']):,} "
            f"tours; {int(s['markov_order']['n_permutations'])} permutations",
            _live_items_cell(s, store_items)]


def _row_struct(d, store_items=None):
    s = _summ(d)
    return [f"{len(s['strengths'])} settings {_x()} {int(s['n_reps'])} reps "
            f"(common random numbers)",
            '--', _live_window(s['protocol']),
            _live_items_cell(s, store_items)]


def _row_queue(d, store_items=None):
    s = _summ(d)
    return [f"{int(s['nominal']['n_reps'])} reps at each of 2 loads",
            '--', _live_window(s['protocol']),
            _live_items_cell(s, store_items)]


def _row_gof(d):
    s = _summ(d)
    n_rep = int(_get(s, 'design', 'in_sample', 'replica', 'basket',
                     'n_replicas', default=0))
    return [f"{int(s['n_reps'])} in-sample + "
            f"{_n(s['held_out']['n_reps'], 'held-out rep')}",
            '--',
            f"{_live_window(s['protocol'])}; "
            f"{int(_gof_category_row(s)['n_permutations']):,} basket "
            f"permutations; {n_rep} replicas",
            str(int(s['design']['in_sample']['n_placed_products']))]


def _row_heatmap(d):
    s = _summ(d)
    p = s['protocol']
    n = _get(s, 'store', 'n_items')
    return [f"1 seeded run (seed {int(p['seed'])})", '--', _live_window(p),
            str(int(n)) if n is not None else '--']


def _row_figures(path):
    """The headless paper-figures run, from its realized-scores record (the
    file, not a run directory)."""
    rs = _read_json(path)
    c = rs['config']
    g, p, f = int(c['n_gens']), int(c['pop_size']), _final_seeds()
    return [f"1 scenario (seed {int(c['scenario_seed'])}) {_x()} "
            f"{int(c['n_seeds'])} GA seeds",
            f"GA {g}{_x()}{p} = {g * p:,} + {p * f} final",
            f"MC {int(c['mc_iters']):,} iter.\\ {_x()} {int(c['mc_days'])} d; "
            f"trace to {int(rs.get('mc_trace_iters', 20000)):,} iter.",
            str(int(c['n_items']))]


# (key, label, row builder) in table order.
TABLE_ROWS = (
    ('synthetic_gt', 'Recovery (Fig.~\\ref{fig:regret})', _row_figa),
    ('baseline_comparison', 'Methods (Fig.~\\ref{fig:methods})', _row_figb),
    ('mc_groundtruth', 'MC ground truth', _row_mcgt),
    ('paper_figures', 'Paper figures (Figs.~\\ref{fig:mcconv}--'
     '\\ref{fig:components})', _row_figures),
    ('ga_sensitivity', 'Operator sweep (Fig.~\\ref{fig:diversity})',
     _row_gasens),
    ('elasticity_lhs', 'Robustness (Fig.~\\ref{fig:lhs})', _row_lhs),
    ('objective_alignment', 'Objective alignment', _row_align),
    ('real_data_uci', 'UCI example (Fig.~\\ref{fig:figc})', _row_figc),
    ('noanon_real_data_uci', 'UCI example, no anonymous invoices',
     _row_figc),
    ('real_data_seeds', 'UCI, search seeds', _row_seeds),
    ('real_data_budget', 'UCI, budget sweep', _row_budget),
    ('input_uncertainty', 'UCI, input uncertainty', _row_input),
    ('heldout_transfer', 'UCI, held-out transfer', _row_transfer),
    ('live_protocol_study', 'Live protocol study', _row_proto),
    ('abm_diagnostics', 'Agent-model diagnostics', _row_abm),
    ('structural_sensitivity', 'Structural sweep', _row_struct),
    ('measure_queueing', 'Queueing', _row_queue),
    ('validation_gof', 'Goodness of fit', _row_gof),
    ('heatmap_figure', 'Heat map (Fig.~\\ref{fig:heatmap})', _row_heatmap),
)


# Table 4 rows of live runs that read their Items cell through
# ``live_store_items``.
LIVE_STORE_ROWS = ('live_protocol_study', 'abm_diagnostics',
                   'structural_sensitivity', 'measure_queueing')


def budget_table(found, errors=None):
    """Table 4 (review R50, R54) from the runs that set the macros: every
    experiment's design, search budget, evaluation and items, each cell
    read from the run's own sidecar and summary. A family with no valid
    run reads TBD. ``found`` maps each family key to its run directory
    (or None; the paper figures' key maps to their realized-scores file).
    A row that cannot be read from a run that set the family's macros is
    a bug, not a missing run: it reads TBD too, and ``(row, run, error)``
    is appended to ``errors`` so that the caller fails."""
    lines = ['% AUTO-GENERATED by CODE/make_results_macros.py -- do not edit.',
             '% Table 4 (experiment budgets): every cell read from the run '
             'that sets the',
             '% paper\'s macros for that family; TBD where no valid run '
             'exists yet.',
             '{\\footnotesize\\setlength{\\tabcolsep}{3pt}%',
             '\\begin{tabular}{@{}p{0.17\\textwidth}p{0.24\\textwidth}'
             'p{0.22\\textwidth}p{0.26\\textwidth}r@{}}',
             '\\toprule',
             'Experiment & Design & Search budget (evaluations) & '
             'Evaluation & Items \\\\',
             '\\midrule']
    store_items = live_store_items(found)
    for key, label, builder in TABLE_ROWS:
        d = found.get(key)
        cells = [TBD] * 4
        if d:
            try:
                cells = (builder(d, store_items) if key in LIVE_STORE_ROWS
                         else builder(d))
            except Exception as exc:
                sys.stderr.write(f"make_results_macros: Table 4 row {key} "
                                 f"could not be read from {d}: {exc!r}\n")
                if errors is not None:
                    errors.append((f'Table 4 row {key}', d, exc))
        lines.append(f"{label} & " + ' & '.join(cells) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}}', '']
    return '\n'.join(lines)


# --- Main ------------------------------------------------------------------------

def _check_sources():
    """Read every design the validators check against up front: inside a
    validator a failure would only look like a family with no valid run."""
    for family in LIVE_RUNNERS:
        _live_design(family)
    for k in FIGURES_DESIGN_ARGS:
        _design_default(FIGURES_SCRIPT, k)
    _script_constants(LIT_SCRIPT, LIT_CONSTANTS + ('MC_SPEND_LAW',
                                             'AGENT_ABANDON_PROB_RANGE'))
    _script_constants(GOF_CATEGORY_SCRIPT, _GOF_CATEGORY_NAMES)
    _script_constants(GOF_SCRIPT, _GOF_RUNNER_NAMES)
    _script_constants(PROTO_SCRIPT, _PROTO_NAMES)
    noanon = _script_constants(FIGC_SCRIPT,
                               ('NOANON_DIR_PREFIX',))['NOANON_DIR_PREFIX']
    if noanon != NOANON_PREFIX:
        raise RuntimeError(f"{FIGC_SCRIPT} writes its no-anonymous runs "
                           f"under {noanon!r}, not {NOANON_PREFIX!r}")
    study_runners = _script_constants(PROTO_SCRIPT,
                                      ('LIVE_RUNNERS',))['LIVE_RUNNERS']
    if set(study_runners) != set(PROTO_RUNNERS):
        raise RuntimeError(f"{PROTO_SCRIPT} compares its protocol with "
                           f"{sorted(study_runners)}, not the live runners "
                           f"the checks here map ({sorted(PROTO_RUNNERS)})")
    _script_constants(HEATMAP_SCRIPT, ('FIGURE_SEED',))
    _script_constants(LIVE_RUNNERS['abm'], ('ROUTING_NULL_TOURS',))
    _script_constants(LIVE_STORE_SCRIPT, ('LIST_LAW',))
    _script_constants(INPUT_SCRIPT, ('DEFAULT_N_BOOT',))
    _script_constants(TRANSFER_SCRIPT, ('DEFAULT_N_SEARCH_SEEDS',))
    _sa_sweep_size()
    _oracle_restarts()
    _final_seeds()
    _adapter_version()
    reg = _module_values(LIT_SCRIPT)
    names = _script_constants(COMMON_SCRIPT, ('OBJECTIVE_CONSTANTS',))[
        'OBJECTIVE_CONSTANTS']
    missing = [n for n in tuple(names) + ELASTICITY_CONSTANTS
               + tuple(GA_WEIGHT_CONSTANTS.values()) if n not in reg]
    if missing:
        raise RuntimeError(f"{LIT_SCRIPT} has no constant value for "
                           f"{', '.join(missing)}")
    for script, keys in (
            (FIGA_SCRIPT, FIGAB_BUDGET_ARGS + ('n_scenarios', 'n_seeds',
                                               'n_spearman_samples',
                                               'oracle_restarts')),
            (FIGB_SCRIPT, FIGAB_BUDGET_ARGS + ('n_scenarios', 'n_seeds',
                                               'oracle_restarts')),
            (MCGT_SCRIPT, ('n_items', 'normal_budget', 'big_budget',
                           'mc_iters', 'mc_days', 'sa_initial_accept',
                           'n_scenarios', 'confirm_seeds')),
            (GASENS_SCRIPT, ('n_items', 'n_gens', 'pop_size', 'mc_iters',
                             'mc_days', 'mode', 'n_scenarios', 'n_seeds',
                             'n_heldout_seeds')),
            (LHS_SCRIPT, ('n_items', 'mc_days', 'n_weight_draws',
                          'n_spread_layouts')),
            (ALIGN_SCRIPT, ('n_items', 'horizon_days', 'n_gens', 'pop_size',
                            'ga_seed', 'wall_coef', 'n_scenarios',
                            'n_samples', 'oracle_restarts')),
            (FIGC_SCRIPT, FIGC_DESIGN_ARGS + ('n_mc_replicates',)),
            (SEEDS_SCRIPT, ('n_search_seeds',)),
            (BUDGET_SCRIPT, ('budget_multipliers', 'n_search_seeds'))):
        for k in keys:
            _design_default(script, k)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--out', default=OUT,
                   help='macros file to write (default: the repository\'s '
                        'paper_results_macros.tex)')
    p.add_argument('--table-out', default=TABLE_OUT,
                   help='Table 4 file to write (default: the repository\'s '
                        'paper_table_budgets.tex)')
    p.add_argument('--results', default=None,
                   help='artifact tree to read (default: '
                        'CODE/experiments/results)')
    p.add_argument('--realized-scores',
                   default=os.path.join(ROOT, 'figs', 'realized_scores.json'),
                   help='the headless paper figures\' record (default: the '
                        'repository\'s figs/realized_scores.json)')
    p.add_argument('--layout-figure',
                   default=os.path.join(ROOT, 'figs',
                                        LAYOUT_FIGURE_STEM + '.json'),
                   help='the layout figure\'s record (default: the '
                        'repository\'s figs/figc_layouts.json)')
    p.add_argument('--citation', default=CITATION_FILE,
                   help='citation file the release macros are read from '
                        '(default: the repository\'s CITATION.cff)')
    p.add_argument('--allow-missing-core', action='store_true',
                   help='write even when Figures A and B have no valid run '
                        '(for checking a tree; never for the paper\'s file)')
    return p.parse_args(argv)


def main(argv=None):
    global RES
    args = parse_args(argv)
    if args.results:
        RES = os.path.abspath(args.results)
    _check_sources()
    macros = {}
    # Families with no artifact that passes its design check. Their macros
    # are left out rather than filled from a run that does not support
    # them, and the paper falls back to its TBD placeholders -- which is
    # only useful if the reason is visible, so it is reported at the end.
    missing = []
    # Families whose valid run could not be read: a bug to fix, reported
    # and turned into a non-zero exit after the files are written.
    failed = []
    found = {}

    # Refuse to overwrite a good macros file from an artifact-less tree
    # (audit R15.1): on a clean checkout with experiments/results/ empty,
    # regenerating would silently replace the shipped numbers with TBD
    # fallbacks. The core experiment artifacts must be present.
    if not args.allow_missing_core and (
            latest('synthetic_gt_', _figa_big_enough) is None
            or latest('baseline_comparison_', _figb_big_enough) is None):
        sys.stderr.write(
            "make_results_macros: no synthetic_gt_*/baseline_comparison_* "
            "artifacts under experiments/results/ that pass the design "
            "check -- refusing to overwrite "
            f"{args.out}.\nEither run the experiments (make experiments-paper) "
            "or keep the shipped paper_results_macros.tex, which was "
            "generated from the released artifact files.\n")
        sys.exit(1)

    def family(key, validator, builder, stem, prefix=None, extra=None):
        d = latest(prefix or key + '_',
                   validator if extra is None
                   else (lambda x: validator(x) and extra(x)))
        if not d:
            found[key] = None
            missing.append(key)
            return None
        try:
            block = builder(d)
            block.update(dir_macros(stem, d))
        except Exception as exc:
            found[key] = None
            failed.append((key, d, exc))
            return None
        macros.update(block)
        found[key] = d
        side = _side(d) or {}
        hardware[stem] = side.get('hardware')
        walls[stem] = side.get('wall_seconds')
        return d

    # What each run that sets macros was made on and how long it took
    # (review R53), per family stem.
    hardware, walls = {}, {}

    figa_dir = family('synthetic_gt', _figa_big_enough, figa_macros, 'FigA')
    figb_dir = family('baseline_comparison', _figb_big_enough, figb_macros,
                      'FigB')
    # The as-built start on Figure B's analytical basis (R07), from both.
    # Runs that did not build the same scenarios leave it out (reported).
    if figa_dir and figb_dir:
        try:
            macros.update(asbuilt_regret_macros(figa_dir, figb_dir))
        except ValueError as exc:
            missing.append(f'as-built regret on Figure B\'s basis ({exc})')
        except Exception as exc:
            failed.append(('asbuilt_regret', f'{figa_dir}, {figb_dir}', exc))
    family('elasticity_lhs', _lhs_big_enough, lhs_macros, 'Lhs')
    figc_dir = family('real_data_uci', _figc_big_enough, figc_macros, 'FigC')
    # Figure C without the anonymous invoices, a sensitivity analysis
    # (review R27) with a directory prefix of its own.
    family('noanon_real_data_uci', _figc_noanon_big_enough,
           figc_noanon_macros, 'FigCNoAnon')
    # Figure C searches once per method, so its intervals cover evaluation
    # noise only; the seeds runner repeats the searches. Only a run at the
    # design of the Figure C artifact quoted above counts.
    family('real_data_seeds', _figc_seeds_big_enough, seeds_macros,
           'FigCSeeds', extra=lambda x: _figc_seeds_match_figc(x, figc_dir))
    # Figure C's saved layouts re-scored in closed form: the conversion
    # bound and the lift's decomposition. Only a re-scoring of the Figure C
    # run quoted above counts.
    family('figc_rescore', _rescore_big_enough, rescore_macros, 'FigCRescore',
           extra=lambda x: _rescore_matches_figc(x, figc_dir))
    # The layout figure's caption numbers (R06), from its record in figs/.
    lf = _checked_json(args.layout_figure,
                       lambda r: _layout_figure_ok(r, figc_dir))
    if lf is None:
        missing.append(f'layout figure record ({args.layout_figure})')
    else:
        try:
            macros.update(layout_figure_macros(lf))
        except Exception as exc:
            failed.append(('layout_figure', args.layout_figure, exc))
    # Realized elasticities at realized scores (audit R2.3).
    rs_path = args.realized_scores
    rs = _checked_json(rs_path, _figures_ok)
    found['paper_figures'] = None
    if rs is None:
        missing.append('realized_scores (make_paper_figures)')
    else:
        try:
            macros.update(figures_macros(rs))
            found['paper_figures'] = rs_path
            hardware['Figures'] = rs.get('hardware')
            walls['Figures'] = rs.get('wall_seconds')
        except Exception as exc:
            failed.append(('realized_scores', rs_path, exc))
    family('measure_queueing', _queue_big_enough, queue_macros, 'Queue')
    family('mc_groundtruth', _mcgt_big_enough, mcgt_macros, 'McGT')
    family('ga_sensitivity', _gasens_big_enough, gasens_macros, 'GASens')
    family('abm_diagnostics', _abm_big_enough, abm_macros, 'Abm')
    struct_dir = family('structural_sensitivity', _struct_big_enough,
                        struct_macros, 'Struct')
    # How the sweep's unpaid exits came about, on re-runs of its own
    # replications. Only a check of the sweep quoted above counts.
    family('structural_exit_check', _struct_check_big_enough,
           struct_check_macros, 'StructCheck',
           extra=lambda x: _struct_check_matches(x, struct_dir))
    family('validation_gof', _gof_big_enough, gof_family_macros, 'Gof')
    # One pooled balk share over the four nominal-load families (R66).
    live = [found.get(k) for k in ('abm_diagnostics',
                                   'structural_sensitivity',
                                   'measure_queueing', 'validation_gof')]
    if all(live):
        try:
            macros.update(pooled_balk_macros(*(_summ(x) for x in live)))
        except Exception as exc:
            failed.append(('pooled_balk', ', '.join(live), exc))
    family('objective_alignment', _align_big_enough, align_macros, 'Align')
    family('real_data_budget', _budget_big_enough, budget_macros, 'Budget')
    family('input_uncertainty', _input_big_enough, input_macros, 'Input')
    family('heldout_transfer', _transfer_big_enough, transfer_macros,
           'Transfer')
    family('live_protocol_study', _proto_big_enough, proto_macros, 'Proto')
    family('heatmap_figure', _heatmap_big_enough, heatmap_macros, 'Heatmap')
    # The workbook's calendar (weekday mix, October-December), read from the
    # same file as the Figure C run quoted above.
    family('profile_uci', _profile_big_enough, profile_macros, 'Profile',
           extra=lambda x: _profile_matches_figc(x, figc_dir))

    # -- Operating constants the projection engine runs at -------------------
    # Read from the constants module rather than restated in the text, so
    # the horizon arithmetic in the paper (days, open hours, weekday /
    # weekend mix) cannot drift from the engine that produced the numbers.
    lit = _script_constants(LIT_SCRIPT, LIT_CONSTANTS)
    macros['OpHoursPerDay'] = f"{lit['DEFAULT_OP_HOURS_PER_DAY']:g}"
    macros['WeekendMultiplier'] = f"{lit['DEFAULT_WEEKEND_MULTIPLIER']:g}"
    # Every registry constant, for the coefficient tables (R18, R41).
    macros.update(registry_macros())
    # The release the paper cites (R23), from the citation file.
    rel = release_macros(args.citation)
    if rel:
        macros.update(rel)
    else:
        missing.append(f'release version ({args.citation})')

    # The compute statement (R53): the machine the runs behind the macros
    # were made on, and their total wall time.
    if hardware:
        macros.update(hardware_macros(hardware))
    total = [float(w) for w in walls.values()
             if _wall_hours(w) is not None]
    if total:
        macros['WallHoursTotal'] = _wall_hours(sum(total))
        macros['WallHoursNRuns'] = str(len(total))

    macros.setdefault('ResultsGrade', 'current artifacts')

    table = budget_table(found, failed)
    with open(args.out, 'w', encoding='utf-8', newline='\n') as f:
        f.write('% AUTO-GENERATED by CODE/make_results_macros.py — do not edit.\n')
        for k, v in sorted(macros.items()):
            f.write(f'\\newcommand{{\\{k}}}{{{v}}}\n')
    with open(args.table_out, 'w', encoding='utf-8', newline='\n') as f:
        f.write(table)
    print(f'wrote {args.out} with {len(macros)} macros')
    print(f'wrote {args.table_out}')
    for k in sorted(macros):
        print(f'  \\{k} = {macros[k]}')
    if missing:
        # Say which numbers are absent and why. A silently shorter macros
        # file turns into TBD placeholders in the PDF, which is easy to
        # miss until a reader finds them.
        sys.stderr.write(
            'make_results_macros: no artifact passed the design check for: '
            + ', '.join(missing) + '.\nThe macros those runs feed are left '
            'out, so the paper falls back to its TBD placeholders; re-run '
            'them at the paper design (make experiments-paper) to restore '
            'the numbers.\n')
    if failed:
        for key, where, exc in failed:
            sys.stderr.write(f'make_results_macros: FAILED to read {key} '
                             f'from {where}: {exc!r}\n')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
