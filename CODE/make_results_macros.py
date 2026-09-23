"""Generate paper_results_macros.tex from the newest experiment artifacts.

Scans CODE/experiments/results/ for the latest run of each kind and emits
LaTeX \\newcommand macros consumed by TOMACS_submission.tex, so the paper
recompiles with fresh numbers after every re-run — no hand-editing.

Every artifact is checked against the design its numbers are quoted at
before it can set a macro: the size of the experiment, and for the live
diagnostics the measurement protocol as well. A family with no run that
passes is reported on stderr and left out, rather than filled in from a
run that does not support it.

    python make_results_macros.py            # writes ../paper_results_macros.tex
"""

from __future__ import annotations

import ast
import csv
import functools
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'experiments', 'results')
ROOT = os.path.dirname(HERE) if os.path.basename(HERE).upper() == 'CODE' else HERE
OUT = os.path.join(ROOT, 'paper_results_macros.tex')


# Minimum size for an artifact to be allowed to set a paper number.
# Selecting purely by newest timestamp meant any smoke run -- a reviewer
# sanity-checking the code, a debugging pass -- silently replaced the
# paper-grade numbers with a toy on the next regeneration. This has bitten
# twice (audit R25, R39), so the size bar is now enforced rather than
# remembered.
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


def _traceable(side):
    """True if a sidecar names the checkout the run started from.

    Runners snapshot the checkout at start-up and record whether it moved
    before the sidecar was written. Sidecars without that record read the
    tree only when the run ended, hours later for a paper-grade run, so
    their commit and diff hash need not describe the code that produced
    the numbers, and such a run cannot back a paper number."""
    return 'git_state_changed_during_run' in side


def _sidecar_big_enough(d):
    """True if the run's sidecar reports a paper-scale design."""
    p = os.path.join(d, 'sidecar.json')
    if not os.path.exists(p):
        return False
    try:
        j = json.load(open(p))
    except Exception:
        return False
    return (int(j.get('n_scenarios', 0)) >= MIN_SCENARIOS
            and int(j.get('n_seeds_per_scenario', 0)) >= MIN_SEEDS
            and _traceable(j))


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
MIN_QUEUE_REPS = 5
MIN_GOF_REPS = 5
# The GA-sensitivity sweep and the headless paper figures are quoted at
# the design the Makefile runs them at.
MIN_GASENS = {'n_scenarios': 3, 'n_seeds': 2, 'n_gens': 25, 'mc_iters': 800,
              'n_heldout_seeds': 10}
MIN_FIGURES = {'n_seeds': 5, 'n_gens': 25, 'pop_size': 30, 'mc_iters': 2000}

# The equal-budget searches are compared from one starting layout, the
# as-built one. Runs from before the searches shared that start handed the
# annealer the popularity ranking instead, so a difference between the
# searches measured the start as much as the search; those runs do not
# record 'search_start' and cannot set these numbers.
ASBUILT = 'asbuilt'
FIGB_ASBUILT_SEARCHES = ('random_search', 'simulated_annealing')
MCGT_ASBUILT_SEARCHES = ('random_search_big', 'simulated_annealing_big',
                         'GA_big')

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

# Equivalence margin of the GA-vs-SA comparison, as a fraction of the
# annealer's mean paired revenue. Fixed before the re-run; the reasoning is
# at the computation in main().
EQUIV_MARGIN_FRAC = 0.001

# Figure C's equal-budget comparators: short name, the paired-difference
# column and the comparator's own revenue column in results.csv.
FIGC_COMPARATORS = {'random_search': ('RandSearch', 'diff_rs', 'rs_revenue'),
                    'simulated_annealing': ('SA', 'diff_sa', 'sa_revenue')}
FIGC_COMPARATOR_COLUMNS = tuple(c for _, diff, rev in FIGC_COMPARATORS.values()
                                for c in (rev, diff))

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


@functools.lru_cache(maxsize=None)
def _script_constants(relpath, names):
    """Module-level literal constants of a script under CODE/, read from its
    source rather than imported.

    The validators below hold a run to the design its script defines --
    the live runners' protocol, the paper figures' scenario -- so reading
    those values from the script itself means a retuned protocol moves the
    check with it instead of leaving a stale copy here. Importing the
    scripts would pull the simulator and dataset stack into this one,
    which restyle_figures imports.
    """
    path = os.path.join(HERE, *relpath.split('/'))
    with open(path, encoding='utf-8') as f:
        tree = ast.parse(f.read(), filename=path)
    found = {}
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in names):
            found[node.targets[0].id] = ast.literal_eval(node.value)
    absent = [n for n in names if n not in found]
    if absent:
        raise RuntimeError(f"{relpath} no longer defines {', '.join(absent)}, "
                           "which the artifact checks here are held to")
    return found


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


def _dir_summary_ok(d, pred):
    """Apply ``pred`` to a finished run's summary.json."""
    if not _has_summary(d):
        return False
    try:
        return bool(pred(json.load(open(os.path.join(d, 'summary.json')))))
    except Exception:
        return False


def _checked_json(path, pred):
    """A figs/ copy of a run summary, or None when it fails ``pred``.

    The queue runner and the GA-sensitivity sweep drop a copy of their
    summary next to the figure they draw, which is where these numbers
    used to be read from with no check at all. The copy goes through the
    same check as a run directory, so a quick run that overwrote it
    cannot set a macro.
    """
    if not os.path.exists(path):
        return None
    try:
        s = json.load(open(path))
        return s if pred(s) else None
    except Exception:
        return None


def _figa_big_enough(d):
    """Paper-scale synthetic run that also reports the reference solution
    mapped onto the GA's feasible set. The regret is still taken against
    the unrepaired reference; the mapped value is recorded beside it, and
    runs from before the feasibility repair was shared do not carry it."""
    if not (_sidecar_big_enough(d) and _has_results_csv(d)):
        return False
    try:
        with open(os.path.join(d, 'results.csv'), newline='') as f:
            return 'oracle_R_feasible' in next(csv.reader(f))
    except Exception:
        return False


def _starts_asbuilt(record, searches):
    """True if ``record`` names the start of every one of ``searches`` and
    each of them started from the as-built layout. A search that started
    from different layouts in different runs is recorded as their
    '/'-joined names, which fails the check."""
    starts = record.get('search_start')
    return (isinstance(starts, dict)
            and all(starts.get(m) == ASBUILT for m in searches))


def _figb_big_enough(d):
    """Paper-scale, finished comparison run that records the annealing
    schedule its simulated-annealing comparator ran under, so the
    equal-budget claim rests on the run's own record; the comparator family
    it corrects for, with the sensitivity rows outside it; and an as-built
    start for random search and the annealer."""
    if not (_sidecar_big_enough(d) and _has_results_csv(d)
            and _has_summary(d)):
        return False
    try:
        j = json.load(open(os.path.join(d, 'sidecar.json')))
        family = j.get('comparison_family')
        return (isinstance(j.get('sa_schedule'), dict)
                and isinstance(family, list) and bool(family)
                and all(m in FIGB_SHORT_NAMES for m in family)
                and _starts_asbuilt(j, FIGB_ASBUILT_SEARCHES))
    except Exception:
        return False


def _figc_comparators_recorded(d):
    """True if a real-data run scored the equal-budget searches beside the
    GA: their columns in results.csv, or their paired differences in the
    run's summary."""
    try:
        with open(os.path.join(d, 'results.csv'), newline='') as f:
            header = next(csv.reader(f))
        if all(c in header for c in FIGC_COMPARATOR_COLUMNS):
            return True
    except Exception:
        pass
    try:
        s = json.load(open(os.path.join(d, 'summary.json')))
        comps = (s.get('results') or {}).get('comparators') or {}
        return all(isinstance(comps.get(m), dict) for m in FIGC_COMPARATORS)
    except Exception:
        return False


def _figc_big_enough(d):
    """True if a finished real-data run used the paper-scale design and
    records what the workbook reader did -- rows read per sheet and the
    cross-sheet repeats dropped before the adapter saw the frame, without
    which the row counts cannot be checked against the workbook -- and
    scored the equal-budget searches under the same paired replicates."""
    if not _has_results_csv(d):
        return False
    try:
        j = json.load(open(os.path.join(d, 'sidecar.json')))
        a = j.get('args', {})
        sheets = [s for s in str(a.get('sheets', '')).split(',') if s.strip()]
        return (int(a.get('n_mc_replicates', 0)) >= MIN_FIGC_REPLICATES
                and int(a.get('mc_iters', 0)) >= MIN_FIGC_MC_ITERS
                and len(sheets) >= MIN_FIGC_SHEETS
                and isinstance(j.get('reader'), dict)
                and _traceable(j)
                and _figc_comparators_recorded(d))
    except Exception:
        return False


def _figc_seeds_big_enough(d):
    """True if a finished multi-seed real-data run has the design its
    numbers are quoted at: at least ``MIN_FIGC_SEEDS`` search seeds, each
    with a recorded result; Figure C's evaluation panel, precision and
    sheets; every search of every seed from the as-built layout and at the
    run's one search budget; and a sidecar that names the checkout it
    started from. A run short of any of these measures a different
    comparison from the one the across-seed macros describe."""
    if not (_has_summary(d) and _has_results_csv(d)):
        return False
    try:
        side = json.load(open(os.path.join(d, 'sidecar.json')))
        s = json.load(open(os.path.join(d, 'summary.json')))
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
        return (k >= MIN_FIGC_SEEDS
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
                and _traceable(side))
    except Exception:
        return False


# The options a multi-seed real-data run shares with Figure C: the store
# (sheets, items per category, assumed conversion), the search budget and
# annealing schedule, and the Monte Carlo precision and evaluation panel.
# Keyed by the name in the seeds run's design block; the value is the name
# in Figure C's recorded args.
FIGC_SEEDS_SHARED_DESIGN = {
    'sheets': 'sheets',
    'max_items_per_category': 'max_items_per_category',
    'assumed_conversion': 'assumed_conversion',
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
        a = json.load(open(os.path.join(figc_dir, 'sidecar.json')))['args']
        s = json.load(open(os.path.join(d, 'summary.json')))
        design = s['design']
        same = all(design[mine] == a[theirs]
                   for mine, theirs in FIGC_SEEDS_SHARED_DESIGN.items())
        return (same and float(s['sa_schedule']['sa_initial_accept'])
                == float(a['sa_initial_accept']))
    except Exception:
        return False


def _lhs_big_enough(d):
    """True if a finished LHS run covers the paper's design resolution and
    sweeps every headline comparison rather than only the GA against the
    popularity baseline, which is what the ``comparisons`` block records,
    and carries the weight sweep over the same layouts at the paper's
    number of weight vectors."""
    if not _has_summary(d):
        return False
    try:
        s = json.load(open(os.path.join(d, 'summary.json')))
        cfg = s.get('config', {})
        ws = s.get('weight_sweep')
        return (int(s.get('n_lhs_points', 0)) >= MIN_LHS_POINTS
                and int(s.get('n_scenarios', 0)) >= MIN_LHS_SCENARIOS
                and all(int(cfg.get(k, 0)) >= v
                        for k, v in MIN_LHS_DESIGN.items())
                and isinstance(s.get('comparisons'), dict)
                and isinstance(ws, dict)
                and isinstance(ws.get('comparisons'), dict)
                and bool(ws['comparisons'])
                and isinstance(ws.get('design'), dict)
                and int(ws['design'].get('n_weight_draws', 0))
                >= MIN_LHS_WEIGHT_DRAWS)
    except Exception:
        return False


def _mcgt_big_enough(d):
    """True if a finished MC ground-truth run used the paper's budgets,
    reports the smallest regret with the count that came out below zero,
    without which the confirmation-noise tail cannot be quoted, and started
    every big-budget search from the as-built layout."""
    if not _has_summary(d):
        return False
    try:
        s = json.load(open(os.path.join(d, 'summary.json')))
        normal = float(s.get('normal_budget', 0))
        return (int(s.get('n_scenarios', 0)) >= MIN_MCGT_SCENARIOS
                and normal >= MIN_MCGT_NORMAL_BUDGET
                and float(s.get('big_budget', 0))
                >= MIN_MCGT_BUDGET_RATIO * normal
                and int(s.get('mc_iters', 0)) >= MIN_MCGT_MC_ITERS
                and s.get('mc_regret_min_pct') is not None
                and s.get('n_negative_regret') is not None
                and _starts_asbuilt(s, MCGT_ASBUILT_SEARCHES))
    except Exception:
        return False


def _abm_ok(s):
    """The paper's replications, the shared protocol, no censored visit
    and finite t-intervals. A single-replication run stores NaN
    half-widths, which would otherwise be printed into the paper as
    'nan'; a censored agent contributes a truncated state sequence. The
    perimeter ratio must come with its geometric null -- the ratio the
    floor plan gives traffic spread evenly over the walkable cells -- in
    the summary and in every replication, since the ratio alone mixes the
    shoppers' routes with how much of each band is walkable."""
    mk, em = s.get('markov_order', {}), s.get('emergence', {})
    cis = (mk.get('info_gain_ci95'), mk.get('tv_ci95'),
           em.get('perimeter_ratio_ci95'),
           em.get('perimeter_interior_ratio_geometric_null'),
           em.get('ratio_to_geometric_null'),
           em.get('ratio_to_geometric_null_ci95'))
    proto = s.get('protocol', {})
    reps = s.get('per_rep') or []
    return (_list_law_ok(s.get('data'))
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
                    for r in reps))


LIVE_STORE_SCRIPT = 'experiments/_live_store.py'


def _list_law_ok(data):
    """The run's store drew its shoppers' lists under the current list
    law (read from the live-store module). Runs from before each list
    became one stocked invoice carry no ``list_law`` and describe a
    different agent model."""
    want = _script_constants(LIVE_STORE_SCRIPT, ('LIST_LAW',))['LIST_LAW']
    return isinstance(data, dict) and data.get('list_law') == want


def _abm_big_enough(d):
    return _dir_summary_ok(d, _abm_ok)


def _struct_ok(s):
    """Replicated enough to judge between-setting spread against
    within-setting noise, run under the shared protocol, and carrying the
    per-replication tests the sweep is now read from."""
    proto = s.get('protocol', {})
    return (int(s.get('n_reps', 0)) >= MIN_STRUCT_REPS
            and _list_law_ok(s.get('data'))
            and _live_protocol_ok(proto, proto.get('spawn'), proto.get('cap'),
                                  _live_design('structural'))
            and s.get('rev_per_cust_anova_p') is not None
            and s.get('completions_anova_p') is not None)


def _struct_big_enough(d):
    return _dir_summary_ok(d, _struct_ok)


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
            and int(proto.get('n_reps', 0)) >= MIN_QUEUE_REPS
            and _live_protocol_ok(proto, spawn.get('nominal'),
                                  cap.get('nominal'), design)
            and _live_protocol_ok(proto, spawn.get('stress'),
                                  cap.get('stress'), design, level='STRESS'))


def _queue_big_enough(d):
    return _dir_summary_ok(d, _queue_ok)


GOF_DESIGNS = ('in_sample', 'held_out')
GOF_PERIODS = ('current', 'prior')

# The category row's test, read from the module that runs it. Runs from
# before it moved whole baskets compared item purchases as if each were an
# independent observation, which rejects a model that reproduces the data
# exactly far more often than alpha; they carry neither this label nor a
# permutation count, and cannot set the category numbers.
GOF_CATEGORY_SCRIPT = 'dataset_validation.py'
_GOF_CATEGORY_NAMES = ('CATEGORY_N_PERM', 'CATEGORY_TEST_KIND')


def _gof_category_row(block):
    """The pooled category-shares row of one design, or None."""
    return next((t for t in block.get('pooled') or []
                 if 'categor' in str(t.get('test', '')).lower()), None)


def _gof_category_ok(block):
    """The pooled category row ran as the basket-level permutation test at
    its full permutation count, with the naive item-level p beside it."""
    want = _script_constants(GOF_CATEGORY_SCRIPT, _GOF_CATEGORY_NAMES)
    row = _gof_category_row(block)
    try:
        return (row is not None
                and row.get('kind') == want['CATEGORY_TEST_KIND']
                and int(row.get('n_permutations') or 0)
                >= want['CATEGORY_N_PERM']
                and row.get('p_value') is not None
                and 'naive_p_value' in row)
    except Exception:
        return False


def _gof_block_ok(b):
    """One design's replicated tests under the shared protocol, with the
    pooled tests present and the category row tested on whole baskets."""
    proto = b.get('protocol', {})
    return (int(b.get('n_reps', 0)) >= MIN_GOF_REPS
            and _live_protocol_ok(proto, proto.get('spawn'), proto.get('cap'),
                                  _live_design('gof'))
            and bool(b.get('pooled'))
            and _gof_category_ok(b))


def _gof_ok(s):
    """Replicated goodness-of-fit run under the shared protocol, with the
    pooled tests present for the in-sample design (the top level) and for
    the held-out one, the category row tested on whole baskets in both,
    and the record of what each design's store was built from and tested
    against: both periods, and per design the products it placed and the
    reference invoices holding one of them, and for the held-out design
    the category test between the two periods' own invoices. Without the
    held-out block the rows are transfer checks only. Each design must
    also carry its size-matched replicas of the store period for all
    three rows: the only yardstick on the simulator's own sample size, so
    a run without them cannot set the held-out comparison."""
    design = s.get('design')
    if not (isinstance(design, dict) and isinstance(s.get('held_out'), dict)):
        return False
    periods = design.get('periods')
    shift = (design.get('held_out') or {}).get('year_shift') or {}
    return (_gof_block_ok(s) and _gof_block_ok(s['held_out'])
            and all(isinstance(design.get(k), dict) for k in GOF_DESIGNS)
            and isinstance(periods, dict)
            and all(isinstance(periods.get(p), dict) for p in GOF_PERIODS)
            and all(_list_law_ok(periods.get(p)) for p in GOF_PERIODS)
            and isinstance(shift.get('category'), dict)
            and all(isinstance(((design.get(d) or {}).get('replica')
                                or {}).get(row), dict)
                    for d in GOF_DESIGNS for row in GOF_REPLICA_ROWS))


GOF_REPLICA_ROWS = ('basket', 'revenue', 'category')


def _gof_big_enough(d):
    return _dir_summary_ok(d, _gof_ok)


def _gasens_ok(s):
    """The Makefile's sweep design, with the settings compared on
    re-scored held-out seeds rather than on the seeds they searched
    under, and the checkout that produced it."""
    cfg = s.get('config', {})
    return (isinstance(s.get('provenance'), dict)
            and all(int(cfg.get(k, 0)) >= v for k, v in MIN_GASENS.items())
            and s.get('fitness_statistic') == 'heldout_mean')


def _gasens_big_enough(d):
    return _dir_summary_ok(d, _gasens_ok)


FIGURES_SCRIPT = 'experiments/make_paper_figures.py'
_FIGURE_SCENARIO = {'SCENARIO_SEED': 'scenario_seed', 'N_ITEMS': 'n_items',
                    'MC_DAYS': 'mc_days'}


def _figures_ok(s):
    """The headless figure run's design and checkout, stamped next to the
    realized scores it reports. The realized scores belong to one fixed
    scenario, so a run on another one is a different number, not a
    smaller version of the same one; and a file the script stopped
    writing half-way has no Monte Carlo precision figure yet."""
    cfg = s.get('config', {})
    scenario = _script_constants(FIGURES_SCRIPT, tuple(_FIGURE_SCENARIO))
    return (isinstance(s.get('provenance'), dict)
            and all(int(cfg.get(k, 0)) >= v for k, v in MIN_FIGURES.items())
            and all(cfg.get(key) == scenario[const]
                    for const, key in _FIGURE_SCENARIO.items())
            and s.get('mc_rse_pct') is not None)


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
        try:
            got = (json.load(open(os.path.join(d, name))).get('results')
                   or {}).get('comparators')
        except Exception:
            got = None
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


def money(v):
    s = f"{abs(v):,.0f}"
    return ("-" if v < 0 else "+") + "\\$" + s


def amount(v):
    """Unsigned currency amount (a level, not a difference)."""
    return "\\$" + f"{v:,.0f}"


def pfmt(p):
    """A p-value at the precision the paper reports, without rounding a
    small one to 0.000. The bound is wrapped in \\ensuremath so the macro
    also works inside math, where the paper puts its p-values
    (``$p=\\StructChiP{}$``); a literal ``$...$`` would close that math."""
    p = float(p)
    return f"{p:.3f}" if p >= 0.001 else "\\ensuremath{<0.001}"


def level_pct(level):
    """A confidence level in percent, whole when it is whole (95\\%) and to
    one decimal otherwise (99.3\\%)."""
    return (f"{level:.0f}\\%" if abs(level - round(level)) < 0.05
            else f"{level:.1f}\\%")


# The goodness-of-fit tests, matched on what each is about rather than on
# its exact label, which is written for the Validation tab and reads better
# when it is free to change.
GOF_TAGS = (('basket', 'Basket'), ('revenue', 'Revenue'),
            ('categor', 'Category'), ('inter-arrival', 'Arrival'))


def gof_macros(block, prefix):
    """The pooled tests of one goodness-of-fit design under ``prefix``:
    statistic, p-value, decision and both sample sizes per test, how many
    single windows reached the pooled test's verdict on their own, the
    replication count and the share of arrivals turned away at the cap.

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
        if tag == 'Category' and t.get('naive_p_value') is not None:
            out[f'{prefix}CategoryNaiveP'] = pfmt(t['naive_p_value'])
        # How many single windows reached the same verdict on their
        # own: a pooled decision that rests on the pooled sample size
        # shows up as a split here.
        if t['test'] in split:
            out[f'{prefix}{tag}RepPass'] = str(int(split[t['test']]['PASS']))
    out[f'{prefix}Reps'] = str(int(block['n_reps']))
    b = balked_pct(block.get('load'))
    if b is not None:
        out[f'{prefix}Balked'] = f"{b:.1f}\\%"
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


def main():
    macros = {}
    # Families with no artifact that passes its design check. Their macros
    # are left out rather than filled from a run that does not support
    # them, and the paper falls back to its TBD placeholders -- which is
    # only useful if the reason is visible, so it is reported at the end.
    missing = []
    # Read the designs the validators check against up front: inside a
    # validator a failure would only look like a family with no valid run.
    for family in LIVE_RUNNERS:
        _live_design(family)
    _script_constants(FIGURES_SCRIPT, tuple(_FIGURE_SCENARIO))
    _script_constants(LIT_SCRIPT, LIT_CONSTANTS)
    _script_constants(GOF_CATEGORY_SCRIPT, _GOF_CATEGORY_NAMES)

    # Refuse to overwrite a good macros file from an artifact-less tree
    # (audit R15.1): on a clean checkout with experiments/results/ empty,
    # regenerating would silently replace the shipped numbers with TBD
    # fallbacks. The core experiment artifacts must be present.
    if (latest('synthetic_gt_', _figa_big_enough) is None
            or latest('baseline_comparison_', _figb_big_enough) is None):
        sys.stderr.write(
            "make_results_macros: no synthetic_gt_*/baseline_comparison_* "
            "artifacts under experiments/results/ that pass the design "
            "check -- refusing to overwrite "
            f"{OUT}.\nEither run the experiments (make experiments-paper) or "
            "keep the shipped paper_results_macros.tex, which was generated "
            "from the released artifact files.\n")
        sys.exit(1)

    # -- Figure A: synthetic ground truth -------------------------------------
    d = latest('synthetic_gt_', _figa_big_enough)
    if d:
        rows = read_csv(os.path.join(d, 'results.csv'))
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
        })
        # Reference-solver convergence diagnostic (audit R3.3): fraction of
        # runs / scenarios where the GA's analytical revenue exceeds the
        # analytical reference's own analytical revenue (i.e. the reference
        # under-converged on its own objective).
        try:
            ga = np.array([float(r['ga_true_R']) for r in rows])
            ref = np.array([float(r['oracle_R']) for r in rows])
            beat = ga > ref
            by_sc = defaultdict(list)
            for r, b in zip(rows, beat):
                by_sc[r['scenario']].append(bool(b))
            macros['RefBeatFrac'] = f"{beat.mean()*100:.1f}\\%"
            macros['RefBeatScenFrac'] = \
                f"{np.mean([any(v) for v in by_sc.values()])*100:.1f}\\%"
        except Exception:
            pass
        # Where the search starts. The GA is initialized from the
        # shelf-order grid layout, whose analytical regret is measured
        # against the same reference and on the same scale as the GA's
        # own, so the two can be read next to each other. That layout does
        # not depend on the seed, so it is summarised over scenarios --
        # averaging over runs would count each scenario's single starting
        # layout once per seed.
        try:
            per_scen = {}
            for r in rows:
                r_ref = float(r['oracle_R'])
                per_scen[r['scenario']] = ((r_ref - float(r['grid_R']))
                                           / max(r_ref, 1e-9) * 100)
            grid_reg = np.array(list(per_scen.values()))
            macros['GridRegretMean'] = f"{grid_reg.mean():.2f}\\%"
            macros['GridRegretMedian'] = f"{np.median(grid_reg):.2f}\\%"
            macros['GridRegretNScen'] = str(len(grid_reg))
        except Exception:
            pass
        # Regret against the reference mapped onto the GA's feasible set.
        # The headline regret is taken against the unrepaired reference,
        # which the GA's constraints can put out of reach; this is the
        # part of the same gap that the GA could actually have closed.
        # Runs made before the repair was shared do not carry the column.
        if 'oracle_R_feasible' in rows[0]:
            ref_f = np.array([float(r['oracle_R_feasible']) for r in rows])
            ref_u = np.array([float(r['oracle_R']) for r in rows])
            ga_R = np.array([float(r['ga_true_R']) for r in rows])
            reg_f = (ref_f - ga_R) / np.maximum(ref_f, 1e-9) * 100
            macros.update({
                'RegretFeasMean': f"{reg_f.mean():.2f}\\%",
                'RegretFeasMedian': f"{np.median(reg_f):.2f}\\%",
                'RegretFeasMax': f"{reg_f.max():.2f}\\%",
                # Runs whose reference layout the repair moved at all: the
                # rest have the two regrets equal by construction.
                'RegretFeasMovedRuns': str(int((ref_f < ref_u).sum())),
            })
        sp = os.path.join(d, 'spearman.csv')
        if os.path.exists(sp):
            sp_rows = read_csv(sp)
            rho = np.array([float(r['spearman_rho']) for r in sp_rows])
            # Fisher-z averaged mean (audit R3.4); report distribution, not
            # just a point, and the detectable |rho| at the number of layouts
            # the run actually sampled per scenario. The z-scale SE for a
            # Spearman coefficient is sqrt(1.06/(n-3)) (Fieller et al. 1957),
            # wider than the Pearson 1/sqrt(n-3).
            n_lay = int(json.load(open(os.path.join(d, 'sidecar.json')))
                        ['args']['n_spearman_samples'])
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
            # How the scenarios split, rather than only where their
            # average landed: a Fisher-z mean near zero is produced both
            # by scenarios that all agree on no relationship and by
            # scenarios that disagree in sign. The quartiles carry the
            # spread the mean hides. Significance is the per-scenario test
            # the runner stored, at the level quoted here.
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
            macros['SpearmanQOne'] = f"{np.percentile(rho, 25):+.3f}"
            macros['SpearmanQThree'] = f"{np.percentile(rho, 75):+.3f}"
        macros['FigADir'] = os.path.basename(d).replace('_', '\\_')

    # -- Figure B: baseline comparison (paired differences) -------------------
    # Cluster (scenario-level) bootstrap + Bonferroni over the comparator
    # family (audits R3.1, R3.2). The divisor is derived from the data, not
    # hardcoded -- it was 5 before the two equal-budget metaheuristics were
    # added and is 7 now; ``NComparators`` records whatever it actually was.
    d = latest('baseline_comparison_', _figb_big_enough)
    if d:
        rows = read_csv(os.path.join(d, 'results.csv'))
        side = json.load(open(os.path.join(d, 'sidecar.json')))
        by = defaultdict(dict)      # (scenario, seed) -> {method: revenue}
        for r in rows:
            by[(r['scenario'], r['seed'])][r['method']] = float(r['mc_revenue'])
        # The family is the one the run records, and within it only the
        # comparators the artifact actually contains; a run without some
        # methods must not be corrected for them, and a sensitivity row the
        # run scored beside the family is never one of them.
        family = set(side['comparison_family'])
        present = [m for m in FIGB_SHORT_NAMES if m in family
                   and any('GA' in v and m in v for v in by.values())]
        n_comparisons = len(present)
        alpha_bonf = 0.05 / max(n_comparisons, 1)
        npairs, nscen, n_sig = 0, 0, 0
        for meth in present:
            # Paired differences grouped BY SCENARIO for the cluster
            # bootstrap; the share of the comparator's revenue lets the size
            # of a margin be read without knowing the scale of the
            # scenarios.
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
        macros['FigBDir'] = os.path.basename(d).replace('_', '\\_')

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
        # the as-built start). The margin was fixed before the re-run, at
        # 0.1% of the annealer's mean paired revenue: about a third of the
        # smallest effect this design resolves (GA minus random search, near
        # 0.34% of the comparator's revenue), so a difference inside it is
        # smaller than anything the comparison could otherwise tell apart.
        # Two one-sided tests, each at the family's Bonferroni level
        # alpha' = 0.05 / NComparators, reject non-equivalence exactly when
        # the (1 - 2 alpha') cluster-bootstrap percentile interval of the
        # paired difference lies inside [-margin, +margin].
        clusters = paired_clusters(by, 'GA', 'simulated_annealing')
        if 'simulated_annealing' in present and clusters:
            clustered = [[d for d, _ in c] for c in clusters.values()]
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

    # -- LHS elasticity robustness --------------------------------------------
    d = latest('elasticity_lhs_', _lhs_big_enough)
    if d and os.path.exists(os.path.join(d, 'summary.json')):
        s = json.load(open(os.path.join(d, 'summary.json')))
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
        # The budget behind every layout in the sweep. The sweep re-scores
        # fixed layouts analytically rather than re-running the search per
        # draw, so this is the budget the layouts came from, and it is
        # lighter than the headline runs' -- stating it keeps the two from
        # being read as the same experiment.
        cfg = s.get('config', {})
        if cfg.get('n_gens'):
            macros['LhsGens'] = str(int(cfg['n_gens']))
            macros['LhsPop'] = str(int(cfg['pop_size']))
            macros['LhsIters'] = f"{int(cfg['mc_iters']):,}"
        # The projection horizon the lifts are summed over, counted the
        # way the MC engine counts it (weekend days by their position in
        # the week), so the text and the closed form cannot disagree.
        if s.get('horizon_days') is not None:
            macros['LhsHorizonDays'] = str(int(s['horizon_days']))
            macros['LhsHorizonCustHours'] = \
                f"{float(s['horizon_customer_hours']):.0f}"
        # The same sweep over every headline comparison, not just the GA
        # against the popularity baseline: the elasticity bands move all
        # of the reported differences, so each one gets its own share of
        # positive draws and its own spread.
        # Same short names as the paired comparisons above, so a reader
        # can put \LhsPopMedian next to \GAvsPop.
        lhs_names = {'popularity_rank': 'Pop', 'oracle': 'Oracle',
                     'random_search': 'RandSearch',
                     'simulated_annealing': 'SA'}
        for meth, st in (s.get('comparisons') or {}).items():
            short = lhs_names.get(meth)
            if not short:
                continue
            macros.update({
                f'Lhs{short}FracPos': f"{st['frac_positive'] * 100:.1f}\\%",
                f'Lhs{short}Median': money(st['lift_median']),
                f'Lhs{short}PFive': money(st['lift_p5']),
                f'Lhs{short}PNinetyFive': money(st['lift_p95']),
            })
        # The weight sweep: the same layouts re-scored under spatial weight
        # vectors drawn over the simplex, with everything else held. Each
        # comparison gets the same summary as under the elasticity bands,
        # plus how many (scenario, seed) pairs saw their lift take both
        # signs -- a lower bound, since the draws only sample the simplex.
        ws = s['weight_sweep']
        for meth, st in ws['comparisons'].items():
            short = lhs_names.get(meth)
            if not short:
                continue
            macros.update({
                f'LhsWeight{short}FracPos':
                    f"{st['frac_positive'] * 100:.1f}\\%",
                f'LhsWeight{short}Median': money(st['median']),
                f'LhsWeight{short}PFive': money(st['p5']),
                f'LhsWeight{short}PNinetyFive': money(st['p95']),
                f'LhsWeight{short}SignFlips':
                    f"{int(st['n_pairs_sign_flip'])}/{int(st['n_pairs'])}",
            })
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
    else:
        missing.append('elasticity_lhs')

    # -- Figure C: real-data worked example ----------------------------------
    d = latest('real_data_uci_', _figc_big_enough)
    if d:
        rows = read_csv(os.path.join(d, 'results.csv'))
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
            'FigCLift': money(float(diffs.mean())).replace('\\$', '\\pounds '),
            'FigCLiftCI': f"[{money(lo)}, {money(hi)}]".replace('\\$', '\\pounds '),
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
                f'FigCGAvs{short}':
                    money(mean_c).replace('\\$', '\\pounds '),
                f'FigCGAvs{short}CI':
                    f"[{money(lo_c)}, {money(hi_c)}]".replace('\\$',
                                                             '\\pounds '),
                f'FigCGAvs{short}Pct': f"{pct_c:+.2f}\\%",
                f'FigCGAvs{short}Sig': 'yes' if lo_c > 0 or hi_c < 0 else 'no',
            })
        # Data-quality descriptors from the calibration sidecar (audit R7.3/R7.5).
        sc = os.path.join(d, 'sidecar.json')
        if os.path.exists(sc):
            side = json.load(open(sc))
            # The equal-budget claim as the run recorded it: search
            # evaluations (held equal by the runner) and the final-selection
            # evaluations each search spent on top of them, which need not
            # be equal -- the annealer ranks distinct archived states only.
            spent = side.get('evaluation_counts') or {}
            if spent and len({int(n) for n in spent.values()}) == 1:
                macros['FigCSearchEvals'] = \
                    f"{int(next(iter(spent.values()))):,}"
            final = side.get('final_evaluation_counts') or {}
            for meth, short in (('GA', 'GA'),
                                ('random_search', 'RandSearch'),
                                ('simulated_annealing', 'SA')):
                if final.get(meth) is not None:
                    macros[f'FigCFinalEvals{short}'] = \
                        f"{int(final[meth]):,}"
            if final and len({int(n) for n in final.values()}) == 1:
                macros['FigCFinalSelectEvals'] = \
                    f"{int(next(iter(final.values()))):,}"
            cs = side.get('calibration_summary', {})
            # What the workbook reader handed the adapter. The sheets of
            # this workbook overlap in time, so the rows the adapter saw
            # are fewer than the rows read; both are reported, since only
            # the first can be checked against the file itself.
            rd = side.get('reader', {})
            if rd.get('rows_read'):
                macros['FigCRowsRead'] = f"{int(rd['rows_read']):,}"
                macros['FigCRowsUsed'] = f"{int(rd['rows_after_dedup']):,}"
                macros['FigCRowsDropped'] = \
                    f"{int(rd['cross_sheet_duplicates_dropped']):,}"
            # Rows left after the adapter's cleaning (cancellations,
            # non-merchandise lines, missing ids, non-positive quantity or
            # price), which is what calibration ran on.
            kept = side.get('provenance', {}).get('rows_kept')
            if kept is not None:
                macros['FigCRowsKept'] = f"{int(kept):,}"
            if 'basket_units_median' in cs:
                macros['BasketUnitsMed'] = f"{cs['basket_units_median']:.0f}"
                macros['BasketDistinctMed'] = f"{cs['basket_distinct_median']:.0f}"
            if 'category_fallback_frac' in cs:
                macros['CatFallbackFrac'] = f"{cs['category_fallback_frac']*100:.0f}\\%"
            if 'return_customer_rate' in cs:
                macros['ReturnRate'] = f"{cs['return_customer_rate']*100:.0f}\\%"
            # The store the calibration produced. Its size follows from the
            # assortment the data supports, so it is a result of the load,
            # not a setting, and the text should not restate it by hand.
            shop = side.get('shop_summary', {})
            if shop.get('sections') is not None:
                macros['FigCSections'] = str(int(shop['sections']))
                macros['FigCItems'] = str(int(shop['items_placed']))
            # The assortment the sections were built from. This is not the
            # section count: the engine also lays out service zones, so the
            # store has more sections than the data has categories.
            if cs.get('n_unique_categories'):
                macros['FigCCategories'] = str(int(cs['n_unique_categories']))
            # The two arrival rates the projection engine is driven at.
            # Buyers per open hour are observed in the invoices; visitors
            # per open hour are that rate divided by the assumed conversion
            # rate, which is an assumption and is flagged as one wherever
            # the visitor figure is used.
            if cs.get('arrivals_per_hour') and cs.get('visitors_per_hour'):
                macros['FigCBuyersPerHour'] = f"{cs['arrivals_per_hour']:.1f}"
                macros['FigCVisitorsPerHour'] = \
                    f"{cs['visitors_per_hour']:.1f}"
            if cs.get('assumed_conversion_rate') is not None:
                macros['FigCAssumedConv'] = \
                    f"{cs['assumed_conversion_rate']:.2f}"
            # The span the invoices cover, which is what the calibrated
            # daily rates are an average over.
            if cs.get('span_seconds'):
                macros['FigCSpanDays'] = \
                    f"{float(cs['span_seconds']) / 86400.0:,.0f}"
    else:
        missing.append('real_data_uci')

    # -- Figure C over several search seeds -----------------------------------
    # Figure C searches once per method, so its intervals cover evaluation
    # noise only. Here each search seed runs the three searches again and
    # scores them on Figure C's evaluation seeds; the macros quote the
    # runner's own across-seed figures: the mean of the per-seed paired
    # means, its t-interval over the seeds (the search-to-search
    # uncertainty), the range, and in how many seeds the GA came out ahead.
    # Only a run at the design of the Figure C artifact quoted above counts.
    figc_dir = d
    d = latest('real_data_seeds_',
               lambda x: (_figc_seeds_big_enough(x)
                          and _figc_seeds_match_figc(x, figc_dir)))
    if d:
        s = json.load(open(os.path.join(d, 'summary.json')))
        k = int(s['design']['n_search_seeds'])
        across = s['results']['across_seeds']

        def pounds(v):
            return money(v).replace('\\$', '\\pounds ')

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
    else:
        missing.append('real_data_seeds')

    # -- Realized elasticities at realized scores (audit R2.3) ---------------
    rs = _checked_json(os.path.join(ROOT, 'figs', 'realized_scores.json'),
                       _figures_ok)
    if rs is None:
        missing.append('realized_scores (make_paper_figures)')
    else:
        macros.update({
            'RealizedScoreBaseline': f"{rs['score_baseline']:.3f}",
            'RealizedScoreGA': f"{rs['score_ga']:.3f}",
            'RealizedConvLift': f"{rs['conv_lift_pct']:+.1f}\\%",
            'RealizedImpLift': f"{rs['impulse_lift_pct']:+.1f}\\%",
            'RealizedBskLift': f"{rs['basket_lift_pct']:+.1f}\\%",
        })
        # MC estimator convergence: relative SE of the mean at the
        # iteration count the experiments actually use, and that count, so
        # the two cannot drift apart in the text.
        if 'mc_rse_pct' in rs:
            macros['McRSEatUsed'] = f"{rs['mc_rse_pct']:.2f}\\%"
            macros['McRSEIters'] = str(int(rs['mc_rse_iters']))

    # -- Queue measurement (audit R1.7) --------------------------------------
    # From the newest validated run directory; the copy the runner leaves
    # next to the figure is used only when it passes the same check.
    d = latest('measure_queueing_', _queue_big_enough)
    q = (json.load(open(os.path.join(d, 'summary.json'))) if d
         else _checked_json(os.path.join(ROOT, 'figs', 'queue_summary.json'),
                            _queue_ok))
    if q is None:
        missing.append('measure_queueing')
    else:
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
            macros['QueueBusyNomSD'] = \
                f"{q['nominal']['busy_frac_sd']*100:.0f}"
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
        proto = q.get('protocol', {})
        if proto.get('warmup_s') is not None:
            macros['QueueWarmup'] = f"{proto['warmup_s']:.0f}"
        if proto.get('collect_s') is not None:
            macros['QueueCollect'] = f"{proto['collect_s']:.0f}"
        # The protocol the live diagnostics share, read from the one run
        # that carries both loads: the fixed tick, and the arrival rate and
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

    # -- MC-objective ground truth (audit R4.3) ------------------------------
    d = latest('mc_groundtruth_', _mcgt_big_enough)
    if d and os.path.exists(os.path.join(d, 'summary.json')):
        s = json.load(open(os.path.join(d, 'summary.json')))
        ratio = int(round(s['big_budget'] / max(s['normal_budget'], 1)))
        macros.update({
            'McRegretMedian': f"{s['mc_regret_median_pct']:.2f}\\%",
            'McRegretMean': f"{s['mc_regret_mean_pct']:.2f}\\%",
            'McRegretMax': f"{s['mc_regret_max_pct']:.2f}\\%",
            'McGTBudgetRatio': f"{ratio}\\times",
            'McGTNScen': str(s['n_scenarios']),
        })
        # The reference layout is chosen on one block of seeds and both
        # layouts are then re-estimated on a disjoint block, so a regret
        # can come out below zero on confirmation noise. Report the
        # smallest one and how many scenarios did, rather than describing
        # the design as non-negative.
        if 'mc_regret_min_pct' in s:
            macros['McRegretMin'] = f"{s['mc_regret_min_pct']:.2f}\\%"
            macros['McRegretNegCount'] = str(int(s['n_negative_regret']))
        # The normal-budget GA layout is itself a candidate for the
        # best-known reference; where no larger search beat it, its regret
        # is zero by construction, so the count belongs next to the median.
        if s.get('n_best_known_is_ga_normal') is not None:
            macros['McGTBestIsGANormal'] = \
                str(int(s['n_best_known_is_ga_normal']))
        # The estimator behind the comparison: the iterations each
        # evaluation averaged over, and the size of the disjoint seed block
        # both layouts were re-estimated on before their regret was taken.
        macros['McGTIters'] = f"{int(s['mc_iters']):,}"
        sc = os.path.join(d, 'sidecar.json')
        if os.path.exists(sc):
            a = json.load(open(sc)).get('args', {})
            if a.get('confirm_seeds'):
                macros['McGTConfirmSeeds'] = str(int(a['confirm_seeds']))
    else:
        missing.append('mc_groundtruth')

    # -- GA hyperparameter sensitivity + diversity (audit R4.5) --------------
    d = latest('ga_sensitivity_', _gasens_big_enough)
    s = (json.load(open(os.path.join(d, 'summary.json'))) if d
         else _checked_json(os.path.join(ROOT, 'figs', 'ga_sensitivity.json'),
                            _gasens_ok))
    if s is None:
        missing.append('ga_sensitivity')
    else:
        macros.update({
            'GAHyperSpread': f"{s['hyper_spread_median_pct']:.2f}\\%",
            'GAHyperSpreadMax': f"{s['hyper_spread_max_pct']:.2f}\\%",
            'GADefaultGap': f"{s['default_gap_median_pct']:.2f}\\%",
            'GANSettings': str(s['n_settings']),
            'GADivStart': f"{s['diversity_start']*100:.1f}\\%",
            'GADivEnd': f"{s['diversity_end']*100:.1f}\\%",
            # Settings are compared on seeds no run searched or selected
            # under, so the spread measures the layouts and not the
            # upward bias of picking a maximum of noisy estimates.
            'GAHeldoutSeeds': str(int(s.get('heldout_seeds', 0))),
        })

    # -- ABM diagnostics: Markov order + emergence (audits R6.2, R6.4) -------
    d = latest('abm_diagnostics_', _abm_big_enough)
    if d and os.path.exists(os.path.join(d, 'summary.json')):
        s = json.load(open(os.path.join(d, 'summary.json')))
        mk, em = s.get('markov_order', {}), s.get('emergence', {})
        if mk:
            macros.update({
                'MarkovNSeq': str(mk['n_sequences']),
                'MarkovInfoGain': f"{mk['info_gain_second_order_bits']:.3f}",
                'MarkovTV': f"{mk['mean_tv_first_vs_second']:.3f}",
            })
            # Replication half-widths (audit R11.5), present in the
            # replicated-protocol schema only.
            if 'info_gain_ci95' in mk:
                macros['MarkovInfoGainCI'] = f"{mk['info_gain_ci95']:.3f}"
                macros['MarkovTVCI'] = f"{mk['tv_ci95']:.3f}"
            macros['MarkovHOne'] = f"{mk['H_next_given_cur_bits']:.2f}"
            # The estimator is positive even for a first-order chain, so the
            # memory is read against its permutation null.
            if 'info_gain_null_mean_bits' in mk:
                macros.update({
                    'MarkovInfoGainNull': f"{mk['info_gain_null_mean_bits']:.3f}",
                    'MarkovInfoGainExcess': f"{mk['info_gain_excess_bits']:.3f}",
                    'MarkovInfoGainExcessCI': f"{mk['info_gain_excess_ci95']:.3f}",
                    'MarkovPermP': f"{mk['info_gain_perm_p']:.3f}",
                    'MarkovTVNull': f"{mk['tv_null_mean']:.3f}",
                    'MarkovTVExcess': f"{mk['tv_excess']:.3f}",
                    'MarkovNPerm': str(int(mk['n_permutations'])),
                })
        if em.get('perimeter_interior_ratio') is not None:
            macros['PerimRatio'] = f"{em['perimeter_interior_ratio']:.2f}"
            if 'perimeter_ratio_ci95' in em:
                macros['PerimRatioCI'] = f"{em['perimeter_ratio_ci95']:.2f}"
        # The ratio's geometric null: what the floor plan gives traffic
        # spread evenly over the walkable cells, fixtures excluded. The
        # observed-to-null quotient is the part of the ratio the shoppers'
        # routes add; both are replication means with t half-widths, like
        # the ratio itself, and the quotient's interval is read against 1.
        if em.get('perimeter_interior_ratio_geometric_null') is not None:
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
                macros['PerimRatioNullCI'] = \
                    f"{float(em['geometric_null_ci95']):.2f}"
        proto = s.get('protocol', {})
        if proto:
            macros['AbmReps'] = str(proto['reps'])
            macros['AbmWarmup'] = f"{proto['warmup_s']:.0f}"
            macros['AbmCollect'] = f"{proto['collect_s']:.0f}"
        # Every agent that arrived inside the window was followed until it
        # left, so no visit is missing for being long; the drain is how
        # far past the window that took.
        if 'n_censored_final' in s.get('markov_order', {}):
            macros['AbmCensored'] = str(int(mk['n_censored_final']))
        if mk.get('mean_drain_seconds') is not None:
            macros['AbmDrain'] = f"{float(mk['mean_drain_seconds']):.0f}"
        b = balked_pct(s.get('load'))
        if b is not None:
            macros['AbmBalked'] = f"{b:.1f}\\%"
    else:
        missing.append('abm_diagnostics')

    # -- Structural (micro-rule) sensitivity (audit R6.5) --------------------
    d = latest('structural_sensitivity_', _struct_big_enough)
    if d and os.path.exists(os.path.join(d, 'summary.json')):
        s = json.load(open(os.path.join(d, 'summary.json')))
        if 'throughput_range_pct' in s:
            macros['StructRangePct'] = f"{s['throughput_range_pct']:.1f}\\%"
            macros['StructChiP'] = f"{s['chi2_p']:.2f}"
        elif 'conversion_range_pp' in s:   # older artifact schema
            macros['StructRangePP'] = f"{s['conversion_range_pp']:.1f}"
            macros['StructChiP'] = f"{s['chi2_p']:.2f}"
        # The pooled chi-square treats the completions as Poisson. The
        # replications say how wide they really are, so the primary test
        # is the one-way ANOVA on the per-replication counts, and the
        # chi-square is also reported corrected by the dispersion they
        # show.
        if s.get('completions_anova_p') is not None:
            macros['StructCompAnovaP'] = pfmt(s['completions_anova_p'])
        if s.get('dispersion_phi') is not None:
            macros['StructDispersion'] = f"{float(s['dispersion_phi']):.2f}"
        if s.get('quasi_poisson_p') is not None:
            macros['StructQuasiP'] = pfmt(s['quasi_poisson_p'])

        # Power of the test at the observed counts. A null
        # result is only informative if the design could have seen the
        # effect. The replications are the unit of the test and they are
        # wider than Poisson, so the counts are simulated as gamma-Poisson
        # at the dispersion the sweep measured -- a multinomial split of
        # the pooled total would credit the design with precision the
        # replications do not show.
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
        # monotone decline; it did not survive a second run (audit R73), so
        # what is reported now is the between-setting spread next to the
        # within-setting spread, plus a one-way ANOVA.
        if s.get('n_reps'):
            m = np.array(s['rev_per_cust_mean'], dtype=float)
            sd = np.array(s['rev_per_cust_sd'], dtype=float)
            n_reps = int(s['n_reps'])
            macros['StructReps'] = str(n_reps)
            macros['StructRevRange'] = f"{m.max() - m.min():.2f}"
            # Pooled within-setting SD, i.e. the square root of the mean
            # variance. Averaging the SDs themselves runs low.
            pooled_sd = float(np.sqrt(np.mean(sd ** 2)))
            macros['StructRevNoiseSD'] = f"{pooled_sd:.2f}"
            # The range is taken over setting MEANS, so what it should be
            # set against is the standard error of a mean and the range
            # that a sweep with no effect at all would still show: the
            # expected range of k normal draws is d2(k) standard errors.
            se = pooled_sd / np.sqrt(n_reps)
            macros['StructRevSE'] = f"{se:.2f}"
            if len(m) in D2_RANGE:
                macros['StructRevNullRange'] = f"{D2_RANGE[len(m)] * se:.2f}"
            if s.get('rev_per_cust_anova_p') is not None:
                macros['StructRevAnovaP'] = f"{float(s['rev_per_cust_anova_p']):.2f}"
            macros['StructNCompletions'] = str(
                int(sum(s.get('completed_per_setting', []))))
        # Pooled over the sweep: every setting ran at the same load, so a
        # large balked share would mean the settings were compared under a
        # cap rather than under the rule being swept.
        b = balked_pct(s.get('load'))
        if b is not None:
            macros['StructBalked'] = f"{b:.1f}\\%"
    else:
        missing.append('structural_sensitivity')

    # -- Goodness of fit: live model vs its calibration ----------------------
    # In-sample transfer checks: the parameters under test were estimated
    # from the same distributions, so a pass says the calibration survives
    # the pipeline, not that the model predicts unseen data.
    d = latest('validation_gof_', _gof_big_enough)
    if d:
        s = json.load(open(os.path.join(d, 'summary.json')))
        # The top level is the in-sample design, under the macro names it
        # has always had.
        macros.update(gof_macros(s, 'Gof'))
        macros['GofAlpha'] = f"{float(s['alpha']):.2f}"
        # Relabellings behind the category p-values (both designs run the
        # same count; the validator holds each to the module's).
        macros['GofCategoryNPerm'] = \
            f"{int(_gof_category_row(s)['n_permutations']):,}"
        # The held-out design: a store calibrated on the prior period and
        # tested against the current one, whose invoices the model never
        # saw. Same tests, level and replication count, under a HeldOut
        # infix. Its inter-arrival row describes the current period's own
        # arrivals and is the in-sample one again.
        macros.update(gof_macros(s['held_out'], 'GofHeldOut'))
        # What each design's store was built from and tested against: the
        # two periods' date ranges and invoice counts, and per design the
        # products the store placed, how many of them the reference period
        # sells, and the reference invoices holding at least one of them --
        # the invoices the basket and revenue references are cut from.
        design = s['design']
        for period, tag in (('current', 'Current'), ('prior', 'Prior')):
            p = design['periods'][period]
            macros[f'GofPeriod{tag}From'] = str(p['first_date'])
            macros[f'GofPeriod{tag}To'] = str(p['last_date'])
            macros[f'GofPeriod{tag}Invoices'] = f"{int(p['n_invoices']):,}"
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
        # The live store every live diagnostic measures is the in-sample
        # design's: the current period, laid out by the same builder. Its
        # shoppers draw their list lengths from the current invoices cut to
        # the products it stocks, one value per invoice holding at least
        # one of them, so the sample's size is that invoice count.
        ins = design['in_sample']
        macros['LiveStoreProducts'] = str(int(ins['n_placed_products']))
        macros['LiveStoreListSampleN'] = \
            f"{int(ins['n_reference_invoices_with_placed']):,}"
        # Shopping-list length: what the store's shoppers draw from (the
        # stocked part of each invoice of the period the store was
        # calibrated on). The held-out store draws from the prior period,
        # so its sample differs from the current-period reference it is
        # tested against.
        def _list_stats(prefix, stats):
            if not stats or not stats.get('n'):
                return
            macros[f'{prefix}Median'] = f"{stats['median']:.0f}"
            macros[f'{prefix}Mean'] = f"{stats['mean']:.1f}"
            macros[f'{prefix}PNinety'] = f"{stats['p90']:.0f}"
        _list_stats('LiveStoreList', ins.get('list_length_agents'))
        _list_stats('HeldOutStoreList',
                    design['held_out'].get('list_length_agents'))
        _list_stats('HeldOutRefList',
                    design['held_out'].get('list_length_reference'))
        # How far the two periods' own invoices differ on the held-out
        # store's products: the yardstick for the held-out distances. The
        # category one is the category row's own test with the prior
        # period's invoices in the simulated visits' place.
        for tag, key in (('Basket', 'basket'), ('Revenue', 'revenue'),
                         ('Category', 'category')):
            ys = (design['held_out'].get('year_shift') or {}).get(key)
            if ys:
                # KS distances to three places; the category row's
                # chi-square to one.
                fmt = '.1f' if key == 'category' else '.3f'
                macros[f'GofYearShift{tag}Stat'] = format(ys['statistic'], fmt)
                macros[f'GofYearShift{tag}P'] = pfmt(ys['p_value'])
        # Size-matched replicas of each store's calibration period: what a
        # perfect transfer of that period scores at the simulator's own
        # sample size (median and central 95% range), how often the test
        # rejects one, and the share of replicas scoring no more than the
        # simulator. In sample the replicas are the test's null.
        for name, prefix in (('in_sample', 'Gof'), ('held_out', 'GofHeldOut')):
            replica = design[name].get('replica') or {}
            for tag, key in (('Basket', 'basket'), ('Revenue', 'revenue'),
                             ('Category', 'category')):
                r = replica.get(key)
                if not r:
                    continue
                fmt = '.1f' if key == 'category' else '.3f'
                macros[f'{prefix}{tag}ReplicaMedian'] = format(r['median'], fmt)
                macros[f'{prefix}{tag}ReplicaLo'] = format(r['lo'], fmt)
                macros[f'{prefix}{tag}ReplicaHi'] = format(r['hi'], fmt)
                macros[f'{prefix}{tag}ReplicaReject'] = \
                    f"{100 * r['reject_rate']:.0f}\\%"
                if r.get('simulator_percentile') is not None:
                    macros[f'{prefix}{tag}ReplicaBelow'] = \
                        f"{100 * r['simulator_percentile']:.0f}\\%"
                macros['GofReplicaN'] = str(int(r['n_replicas']))
    else:
        missing.append('validation_gof')

    # -- Operating constants the projection engine runs at -------------------
    # Read from the constants module rather than restated in the text, so
    # the horizon arithmetic in the paper (days, open hours, weekday /
    # weekend mix) cannot drift from the engine that produced the numbers.
    lit = _script_constants(LIT_SCRIPT, LIT_CONSTANTS)
    macros['OpHoursPerDay'] = f"{lit['DEFAULT_OP_HOURS_PER_DAY']:g}"
    macros['WeekendMultiplier'] = f"{lit['DEFAULT_WEEKEND_MULTIPLIER']:g}"

    macros.setdefault('ResultsGrade', 'current artifacts')

    with open(OUT, 'w', encoding='utf-8') as f:
        f.write('% AUTO-GENERATED by CODE/make_results_macros.py — do not edit.\n')
        for k, v in sorted(macros.items()):
            f.write(f'\\newcommand{{\\{k}}}{{{v}}}\n')
    print(f'wrote {OUT} with {len(macros)} macros')
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


if __name__ == '__main__':
    main()
