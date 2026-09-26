"""The macro generator's artifact checks, on small hand-made artifacts.

``make_results_macros`` sets a paper number only from a run made at the
paper's design, from a clean checkout, under today's model. These tests
build minimal artifacts the way the runners write them and pin that:

  * a run under today's model and design is accepted, family by family,
    including the stage-2 families (objective alignment, the real-data
    budget sweep, input uncertainty, held-out transfer, the live protocol
    study, the heat map);
  * a run made under the old model is refused: no score anchor or a zero
    one, no ``MC_SPEND_LAW`` or any objective constant since changed; a
    live run from before the stage-2 live package (framing 'terminating',
    the per-fixture routing null, one-way structural sweep, the old GoF
    cohort); a store calibrated by adapter 1.1 or without the anonymous
    invoices; and a run whose sidecar is dirty or moved mid-run;
  * a run at another budget than the runner's defaults is refused (review
    R50), and so is a ground truth without the popularity-start reference;
  * the defaults read from the runners' source equal their parsers';
  * shares are printed as the counts behind them (review R57), Table 4
    falls back to TBD, and the generator runs end to end on a tree holding
    only Figures A and B.
"""

import copy
import csv
import json
import os
import sys

import numpy as np
import pytest

import make_results_macros as MRM
from experiments import _common as C
from experiments._inference import paired_family_inference
from experiments.run_baseline_comparison import (COMPARISON_FAMILY,
                                                 conclusions_unchanged)

ANCHOR = {'score': 0.41, 'breakdown': {'traffic': 0.2, 'flow': 0.5},
          'source': MRM.ANCHOR_SOURCE}
SHA = '0123456789abcdef0123456789abcdef01234567'


# --- building artifacts ---------------------------------------------------------

def _write(d, name, obj):
    with open(os.path.join(d, name), 'w', encoding='utf-8') as f:
        json.dump(obj, f)


def _csv(d, header, rows, name='results.csv'):
    with open(os.path.join(d, name), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _clean(**extra):
    """A sidecar as write_sidecar stamps it, from a clean checkout."""
    side = {'git_sha': SHA, 'git_dirty': False, 'git_diff_sha256': 'e3b0',
            'git_modified': [], 'git_untracked_py': [],
            'git_state_changed_during_run': False,
            'elasticities': json.loads(json.dumps(C.elasticity_snapshot()))}
    side.update(extra)
    return side


def _bp():
    return {'0': {'customers_per_hour': 20.0, 'score_anchor': dict(ANCHOR)}}


def _defaults(module, argv=None):
    """A runner's parsed defaults, from its own parser."""
    old = sys.argv
    try:
        sys.argv = ['runner']
        args = module.parse_args() if argv is None else module.parse_args(argv)
    finally:
        sys.argv = old
    return {k: v for k, v in vars(args).items()
            if not isinstance(v, str) or k in ('sheets', 'mode')}


def _figc_defaults():
    import argparse
    from experiments.run_real_data_example import add_arguments
    p = argparse.ArgumentParser()
    add_arguments(p)
    return vars(p.parse_args([]))


class Tree:
    """A results directory the artifacts are written into, one run
    directory per call, named so that later calls are newer."""

    def __init__(self, root):
        self.root = str(root)
        self.n = 0

    def run(self, prefix):
        self.n += 1
        d = os.path.join(self.root, f'{prefix}_20990101-{self.n:06d}')
        os.makedirs(d)
        return d


def live_data(**change):
    data = {'period': 'current', 'list_law': 'stocked-invoice',
            'adapter_version': MRM._adapter_version(),
            'exclude_anonymous': False, 'source_sha256': 'ab' * 32,
            'first_date': '2010-12-01', 'last_date': '2011-12-09',
            'n_invoices': 19000}
    data.update(change)
    return data


def live_protocol(family='abm', **change):
    des = MRM._live_design(family)
    proto = {'mode': 'fixed_step', 'dt': 0.04, 'reps': 10,
             'warmup_s': des['WARMUP_S'], 'collect_s': des['COLLECT_S'],
             'spawn': des['NOMINAL_SPAWN'], 'cap': des['NOMINAL_CAP'],
             'framing': MRM.LIVE_FRAMING}
    proto.update(change)
    return proto


# --- Figure A -------------------------------------------------------------------

def figa_parts():
    import experiments.run_synthetic_gt as RSG
    args = _defaults(RSG)
    side = _clean(args=args, n_scenarios=args['n_scenarios'],
                  n_seeds_per_scenario=args['n_seeds'], base_params=_bp())
    gains = {str(c): {'median': 0.01, 'max': 0.05, 'n_improved': 2,
                      'n_scenarios': 3} for c in (6, 12, 24, 48, 96)}
    summary = {
        'oracle': {'n_restarts': args['oracle_restarts'],
                   'n_failed_restarts': 0, 'n_not_converged': 0,
                   'gain_after_checkpoint_pct': gains},
        'reference_moved_by_repair': {'n_scenarios': 1, 'of_scenarios': 3},
        'closed_form': {
            'spearman_fisher_mean_cf': 0.5, 'spearman_median_cf': 0.5,
            'spearman_min_cf': 0.1, 'spearman_max_cf': 0.9,
            'spearman_n_positive_cf': 3,
            'ga_mc_fit_minus_cf_pct': {'mean': 0.1, 'median': 0.1,
                                       'min': 0.0, 'max': 0.2},
            'conclusions_unchanged': {'all_unchanged': True}}}
    header = ['scenario', 'seed', 'oracle_R', 'oracle_R_feasible', 'grid_R',
              'ga_true_R', 'ga_mc_fit', 'ga_cf_R', 'regret_pct',
              'wall_seconds']
    rows = [[sc, sd, 100.0, 99.5, 90.0, 99.0 - sd, 99.2, 99.1,
             (100.0 - (99.0 - sd)) / 100.0 * 100, 1.0]
            for sc in range(3) for sd in range(2)]
    spearman = [[sc, 0.5, 0.01, 0.5, 0.01] for sc in range(3)]
    return side, summary, header, rows, spearman


def write_figa(tree, mutate=None):
    side, summary, header, rows, spearman = figa_parts()
    if mutate:
        mutate(side, summary, header)
    d = tree.run('synthetic_gt')
    _write(d, 'sidecar.json', side)
    _write(d, 'summary.json', summary)
    _csv(d, header, [r[:len(header)] for r in rows])
    _csv(d, ['scenario', 'spearman_rho', 'spearman_p', 'spearman_rho_cf',
             'spearman_p_cf'], spearman, name='spearman.csv')
    return d


def _drop_anchor(side, summary, header):
    side['base_params'] = {'0': {'customers_per_hour': 20.0}}


def _zero_anchor(side, summary, header):
    """Every anchor of the record replaced by the explicit zero anchor."""
    def walk(bp):
        if isinstance(bp, dict):
            if 'score_anchor' in bp:
                bp['score_anchor'] = {'score': 0.0, 'breakdown': {},
                                      'source': 'zero'}
            else:
                for v in bp.values():
                    walk(v)
    walk(side['base_params'])


def _old_spend_law(side, summary, header):
    del side['elasticities']['objective_constants']['MC_SPEND_LAW']


def _changed_constant(side, summary, header):
    side['elasticities']['objective_constants']['CONV_CLAMP_HI'] = 0.98


def _no_anchoring(side, summary, header):
    del side['elasticities']['anchoring']


def _changed_weight(side, summary, header):
    side['elasticities']['GA_weights']['traffic'] = 0.2


def _dirty(side, summary, header):
    side['git_dirty'] = True


def _moved(side, summary, header):
    side['git_state_changed_during_run'] = True


def _untraceable(side, summary, header):
    del side['git_state_changed_during_run']


OLD_MODEL = [_drop_anchor, _zero_anchor, _old_spend_law, _changed_constant,
             _no_anchoring, _changed_weight]
UNCLEAN = [_dirty, _moved, _untraceable]


def test_figure_a_accepts_a_run_under_todays_model(tmp_path):
    assert MRM._figa_big_enough(write_figa(Tree(tmp_path)))


@pytest.mark.parametrize('mutate', OLD_MODEL + UNCLEAN + [
    lambda s, m, h: m['oracle'].update(n_restarts=24),            # R36
    lambda s, m, h: s['args'].update(n_gens=20),                  # R50
    lambda s, m, h: s['args'].update(mc_iters=1000),              # R50
    lambda s, m, h: s['args'].update(n_scenarios=10),
    lambda s, m, h: m.pop('closed_form'),                         # R02
    lambda s, m, h: m.pop('reference_moved_by_repair'),           # R64
    lambda s, m, h: h.remove('ga_cf_R'),
])
def test_figure_a_refuses_old_model_design_or_tree(tmp_path, mutate):
    assert not MRM._figa_big_enough(write_figa(Tree(tmp_path), mutate))


# --- Figure B -------------------------------------------------------------------

def figb_parts():
    import experiments.run_baseline_comparison as RBC
    args = _defaults(RBC)
    methods = ['GA'] + list(COMPARISON_FAMILY) + ['simulated_annealing_popstart']
    rng = np.random.default_rng(3)
    rows = []
    for sc in range(3):
        for sd in range(2):
            for k, m in enumerate(methods):
                v = 1000.0 - 5 * k + rng.normal(0, 1)
                rows.append([sc, sd, m, v, v + 0.5, 100.0])
    by = {}
    by_cf = {}
    for sc, sd, m, v, cf, _ in rows:
        by.setdefault((sc, sd), {})[m] = v
        by_cf.setdefault((sc, sd), {})[m] = cf
    inf_mc = paired_family_inference(by, COMPARISON_FAMILY)
    inf_cf = paired_family_inference(by_cf, COMPARISON_FAMILY)
    search = args['pop_size'] * args['n_gens']
    final = args['pop_size'] * C.GA_N_FINAL_SEEDS
    equal = ['GA', 'random_search', 'simulated_annealing']
    side = _clean(
        args=args, n_scenarios=args['n_scenarios'],
        n_seeds_per_scenario=args['n_seeds'], base_params=_bp(),
        comparison_family=list(COMPARISON_FAMILY),
        equal_budget_methods=equal,
        search_start={m: 'asbuilt' for m in equal},
        evaluation_counts={m: [search] for m in equal},
        final_evaluation_counts={m: [final] for m in equal},
        sa_schedule={'sa_initial_accept': 0.8, 'sa_start': 'asbuilt',
                     'sa_T0': [{'scenario': 0, 'seed': 0, 'sa_T0': 12.0}]})
    summary = {
        'comparison_family': list(COMPARISON_FAMILY),
        'search_start': side['search_start'],
        'inference': {'mc': inf_mc, 'closed_form': inf_cf},
        'closed_form_conclusions_unchanged': conclusions_unchanged(inf_mc,
                                                                   inf_cf),
        'reference_moved_by_repair': {'n_scenarios': 0, 'of_scenarios': 3},
        'oracle_restarts': args['oracle_restarts'],
        'oracle_failed_restarts': 0,
        'budget': {'search_evals': search, 'final_evals': final,
                   'block': args['pop_size']},
        'sa_start_effect_closed_form': {}}
    return side, summary, rows


def write_figb(tree, mutate=None):
    side, summary, rows = figb_parts()
    if mutate:
        mutate(side, summary, None)
    d = tree.run('baseline_comparison')
    _write(d, 'sidecar.json', side)
    _write(d, 'summary.json', summary)
    _csv(d, ['scenario', 'seed', 'method', 'mc_revenue', 'cf_revenue',
             'analytical_R'], rows)
    return d


def test_figure_b_accepts_a_run_under_todays_model(tmp_path):
    assert MRM._figb_big_enough(write_figb(Tree(tmp_path)))


def test_figure_a_reports_the_share_of_the_gap_the_ga_closes(tmp_path):
    """R07: beside the regret, the share of the start-to-reference gap each
    run closes, (GA - start) / (reference - start), and the runs that end
    below their own start, as counts. figa_parts: reference 100, start 90,
    GA 99 (seed 0) and 98 (seed 1), so 90% and 80% of the gap."""
    out = MRM.figa_macros(write_figa(Tree(tmp_path)))
    assert out['FigAGapClosedMean'] == '85.0\\%'
    assert out['FigAGapClosedMedian'] == '85.0\\%'
    assert out['FigAGapClosedMin'] == '80.0\\%'
    # Every run having a gap is true by construction (the reference starts
    # a restart from the as-built layout): the runs left out are counted
    # instead, and the tautological count is not emitted.
    assert out['FigAGapClosedExcludedRuns'] == '0/6'
    assert 'FigAGapClosedRuns' not in out
    assert out['FigABelowStartRuns'] == '0/6'

    side, summary, header, rows, spearman = figa_parts()
    rows[0][5] = 85.0                   # GA below its start of 90
    rows[1][4] = rows[1][5] = rows[1][2]    # the start is the reference
    d = Tree(tmp_path / 'b').run('synthetic_gt')
    _write(d, 'sidecar.json', side)
    _write(d, 'summary.json', summary)
    _csv(d, header, rows)
    _csv(d, ['scenario', 'spearman_rho', 'spearman_p', 'spearman_rho_cf',
             'spearman_p_cf'], spearman, name='spearman.csv')
    out = MRM.figa_macros(d)
    assert out['FigABelowStartRuns'] == '1/6'
    assert out['FigAGapClosedExcludedRuns'] == '1/6'
    assert out['FigAGapClosedMin'] == '-50.0\\%'


def test_figure_b_reports_every_methods_analytical_regret(tmp_path):
    """R07: each method's mean analytical regret against the reference row
    of the same run, the popularity-started annealer included."""
    side, summary, rows = figb_parts()
    loss = {'GA': 1.0, 'random_valid': 2.0, 'perimeter_only': 2.5,
            'popularity_rank': 1.5, 'greedy_swap': 1.25,
            'random_search': 1.2, 'simulated_annealing': 1.05,
            'simulated_annealing_popstart': 0.9, 'oracle': 0.0}
    for r in rows:
        r[5] = 200.0 * (1 - loss[r[2]] / 100.0)
    d = Tree(tmp_path).run('baseline_comparison')
    _write(d, 'sidecar.json', side)
    _write(d, 'summary.json', summary)
    _csv(d, ['scenario', 'seed', 'method', 'mc_revenue', 'cf_revenue',
             'analytical_R'], rows)
    out = MRM.figb_macros(d)
    for meth, short in (('GA', 'GA'), ('random_valid', 'Random'),
                        ('perimeter_only', 'Perim'),
                        ('popularity_rank', 'Pop'), ('greedy_swap', 'Greedy'),
                        ('random_search', 'RandSearch'),
                        ('simulated_annealing', 'SA'),
                        ('simulated_annealing_popstart', 'SAPopStart')):
        assert out[f'FigBAnalyticalRegret{short}'] == \
            f"{loss[meth]:.2f}\\%", short
    assert 'FigBAnalyticalRegretOracle' not in out


@pytest.mark.parametrize('mutate', OLD_MODEL + UNCLEAN + [
    lambda s, m, h: s['args'].update(pop_size=24),                # R50
    lambda s, m, h: s['args'].update(mc_days=14),                 # R50
    lambda s, m, h: m.pop('inference'),                           # R28
    lambda s, m, h: m['budget'].update(final_evals=120),
    lambda s, m, h: s['evaluation_counts'].update(random_search=[749]),
    lambda s, m, h: m.update(oracle_restarts=24),                 # R36
    lambda s, m, h: s['search_start'].update(simulated_annealing='popularity'),
])
def test_figure_b_refuses_old_model_design_or_tree(tmp_path, mutate):
    assert not MRM._figb_big_enough(write_figb(Tree(tmp_path), mutate))


# --- MC ground truth, GA sweep, LHS, alignment -------------------------------------

def write_mcgt(tree, mutate=None):
    import experiments.run_mc_groundtruth as RMG
    args = _defaults(RMG)
    starts = {m: 'asbuilt' for m in MRM.MCGT_ASBUILT_SEARCHES}
    starts['GA_normal'] = 'asbuilt'
    starts[MRM.MCGT_POPSTART] = MRM.POPULARITY
    summary = {'n_scenarios': args['n_scenarios'],
               'normal_budget': args['normal_budget'],
               'big_budget': args['big_budget'], 'mc_iters': args['mc_iters'],
               'mc_regret_median_pct': 0.2, 'mc_regret_mean_pct': 0.2,
               'mc_regret_max_pct': 0.4, 'mc_regret_min_pct': -0.01,
               'n_negative_regret': 1, 'n_best_known_is_ga_normal': 1,
               'search_start': starts,
               'closed_form': {'regret_pct': {'median': 0.2, 'max': 0.4},
                               'regret_vs_cf_best_pct': {'median': 0.2,
                                                         'max': 0.4},
                               'n_reference_is_cf_best': 5,
                               'tolerance_pct': 0.01,
                               'conclusions_unchanged': {},
                               'all_unchanged': True}}
    side = _clean(args=args, base_params=_bp())
    if mutate:
        mutate(side, summary, None)
    d = tree.run('mc_groundtruth')
    _write(d, 'sidecar.json', side)
    _write(d, 'summary.json', summary)
    return d


def test_ground_truth_accepts_a_run_under_todays_model(tmp_path):
    assert MRM._mcgt_big_enough(write_mcgt(Tree(tmp_path)))


@pytest.mark.parametrize('mutate', OLD_MODEL + UNCLEAN + [
    # R50: the reference without the popularity-started annealer
    lambda s, m, h: m['search_start'].pop(MRM.MCGT_POPSTART),
    lambda s, m, h: m['search_start'].update(
        {MRM.MCGT_POPSTART: 'asbuilt'}),
    lambda s, m, h: s['args'].update(n_items=10),                 # R54
    lambda s, m, h: s['args'].update(mc_iters=2000),
    lambda s, m, h: m.pop('closed_form'),                         # R02
])
def test_ground_truth_refuses_old_model_design_or_reference(tmp_path, mutate):
    assert not MRM._mcgt_big_enough(write_mcgt(Tree(tmp_path), mutate))


def write_gasens(tree, mutate=None):
    import experiments.run_ga_sensitivity as RGS
    cfg = _defaults(RGS, argv=[])
    budget = cfg['pop_size'] * (cfg['n_gens'] + C.GA_N_FINAL_SEEDS)
    settings = [{'mut_rate': m, 'pop_size': p, 'n_gens': budget // p - 5,
                 'total_evals': budget}
                for m in RGS.MUT_RATES for p in RGS.POP_SIZES]
    summary = {
        'config': cfg, 'provenance': {'git_sha': SHA},
        'fitness_statistic': 'heldout_mean', 'budget_total_evals': budget,
        'settings': settings, 'n_settings': len(settings),
        'ga_sweep': {'null_design': 'blocked by (scenario, seed)',
                     'anova_setting': {'setting_vs_interaction':
                                       {'F': 1.0, 'df': 8, 'df_error': 72,
                                        'p': 0.4}}},
        'sa_sweep': {'power_reference': {'setting': 'default',
                                         'equivalent': True,
                                         'smallest_margin_frac': 0.0005},
                     'n_settings': MRM._sa_sweep_size()},
        'diversity_figure': {'saved_width_in': 4.382,
                             'include_width_in': 4.382,
                             'min_font_pt': 7.5,
                             'min_font_pt_printed': 7.5}}
    side = _clean(args=cfg, base_params=_bp())
    if mutate:
        mutate(side, summary, None)
    d = tree.run('ga_sensitivity')
    _write(d, 'sidecar.json', side)
    _write(d, 'summary.json', summary)
    return d


def test_operator_sweep_accepts_the_fixed_budget_form(tmp_path):
    assert MRM._gasens_big_enough(write_gasens(Tree(tmp_path)))


@pytest.mark.parametrize('mutate', OLD_MODEL + [
    _dirty,
    # the old sweep: population varied at fixed generations (R17)
    lambda s, m, h: m['settings'][0].update(total_evals=500),
    lambda s, m, h: m['ga_sweep'].pop('null_design'),
    lambda s, m, h: m['ga_sweep']['anova_setting'].pop(
        'setting_vs_interaction'),
    lambda s, m, h: m['sa_sweep'].pop('power_reference'),
    lambda s, m, h: m['config'].update(n_scenarios=3, n_seeds=2),
    lambda s, m, h: m.update(fitness_statistic='search_best'),
    # the diversity figure at screen size (R56: 10 pt at 7.07 in, about
    # 6.2 pt at the width the paper prints it), or not recorded
    lambda s, m, h: m['diversity_figure'].update(min_font_pt_printed=6.2),
    lambda s, m, h: m.pop('diversity_figure'),
])
def test_operator_sweep_refuses_the_old_sweep(tmp_path, mutate):
    assert not MRM._gasens_big_enough(write_gasens(Tree(tmp_path), mutate))


def write_lhs(tree, mutate=None):
    import experiments.run_elasticity_lhs as LHS
    cfg = _defaults(LHS)
    cfg.update(n_scenarios=12, n_seeds=3, n_draws=256, n_gens=15,
               pop_size=24, mc_iters=500)
    comps = {m: {'frac_positive': 1.0, 'pairs': {'n_pairs': 36}}
             for m in LHS.COMPARATORS}
    ws = {'comparisons': {m: {} for m in LHS.COMPARATORS},
          'design': {'n_weight_draws': 256}, 'n_pairs': 36}
    summary = {'n_lhs_points': 256, 'n_scenarios': 12, 'n_seeds': 3,
               'config': cfg, 'comparisons': comps, 'weight_sweep': ws,
               'weight_sweep_standardized': copy.deepcopy(ws),
               'weight_sweep_standardized_unit': copy.deepcopy(ws),
               'effective_weights': {'mean_share': {}},
               'closed_form_extras': {'n_pairs': 36},
               'oracle': {'n_restarts': MRM._oracle_restarts()},
               'objective': {'mc_spend_law': 'lognormal_moment_matched'}}
    side = _clean(base_params=_bp())
    if mutate:
        mutate(side, summary, None)
    d = tree.run('elasticity_lhs')
    _write(d, 'sidecar.json', side)
    _write(d, 'summary.json', summary)
    return d


def test_lhs_accepts_a_run_under_todays_model(tmp_path):
    assert MRM._lhs_big_enough(write_lhs(Tree(tmp_path)))


@pytest.mark.parametrize('mutate', OLD_MODEL + [
    _moved,
    lambda s, m, h: m['oracle'].update(n_restarts=24),            # R36
    lambda s, m, h: m.pop('weight_sweep_standardized'),           # R09
    lambda s, m, h: m.pop('effective_weights'),                   # R09
    lambda s, m, h: m.pop('closed_form_extras'),                  # R37/R38
    lambda s, m, h: m['comparisons']['oracle'].pop('pairs'),      # R38
    lambda s, m, h: m['objective'].update(mc_spend_law='floored_normal'),
    lambda s, m, h: m['config'].update(mc_days=30),               # R54
])
def test_lhs_refuses_old_model_or_missing_analyses(tmp_path, mutate):
    assert not MRM._lhs_big_enough(write_lhs(Tree(tmp_path), mutate))


def write_align(tree, mutate=None):
    import experiments.run_objective_alignment as ROA
    cfg = _defaults(ROA)
    n = cfg['n_scenarios']
    summary = {'variants': {'as_now': {'n_scenarios': n},
                            'wall_aligned': {'n_scenarios': n}},
               'change_as_now_to_wall_aligned': {},
               'oracle': {'n_restarts': cfg['oracle_restarts']},
               'config': cfg}
    side = _clean(args=cfg, base_params={
        '0': {'as_now': {'score_anchor': dict(ANCHOR)},
              'wall_aligned': {'score_anchor': dict(ANCHOR)}}})
    if mutate:
        mutate(side, summary, None)
    d = tree.run('objective_alignment')
    _write(d, 'sidecar.json', side)
    _write(d, 'summary.json', summary)
    return d


def test_alignment_accepts_the_runners_design(tmp_path):
    assert MRM._align_big_enough(write_align(Tree(tmp_path)))


@pytest.mark.parametrize('mutate', OLD_MODEL + [
    _dirty,
    lambda s, m, h: m['config'].update(n_scenarios=2),
    lambda s, m, h: m['config'].update(oracle_restarts=24),
    lambda s, m, h: m['oracle'].update(n_restarts=24),
    lambda s, m, h: m['config'].update(wall_coef=3.0),
    lambda s, m, h: m['variants'].pop('wall_aligned'),
    # one variant's record without an anchor, deep in the per-variant record
    lambda s, m, h: s['base_params']['0'].update(
        wall_aligned={'customers_per_hour': 20.0}),
])
def test_alignment_refuses_smaller_or_older_runs(tmp_path, mutate):
    assert not MRM._align_big_enough(write_align(Tree(tmp_path), mutate))


# --- Figure C and the real-data families ---------------------------------------------

def _figc_design(**change):
    a = _figc_defaults()
    design = {k: a[k] for k in ('sheets', 'max_items_per_category',
                                'assumed_conversion', 'mc_iters', 'mc_days',
                                'pop_size', 'sa_initial_accept', 'n_gens',
                                'n_mc_replicates')}
    design['exclude_anonymous'] = False
    design.update(change)
    return design


def _aisle():
    return {'n_zones': 15, 'n_zones_narrowed': 15, 'n_pairs_kept_apart': 105,
            'n_items': 108, 'n_items_narrowed': 108, 'n_items_pinned': 12,
            'area_share_kept_geomean_unpinned': 0.29, 'min_aisle_m': 1.6}


def write_figc(tree, mutate=None, prefix='real_data_uci'):
    a = _figc_defaults()
    a['exclude_anonymous'] = False
    side = _clean(args=a, reader={'rows_read': 10}, base_params=_bp(),
                  provenance={'adapter_version': MRM._adapter_version()})
    summary = {'data': {'sheets': a['sheets'],
                        'exclude_anonymous': False,
                        'adapter_version': MRM._adapter_version()},
               'results': {'closed_form': {}, 'closed_form_check': {},
                           'comparators': {}},
               'feasibility': {'aisle_rule': _aisle()},
               'search_design': {'operators': {'GA': 'region'}}}
    header = ['replicate', 'mc_seed', 'baseline_revenue', 'optimized_revenue',
              'diff', 'rs_revenue', 'sa_revenue', 'diff_rs', 'diff_sa']
    if mutate:
        mutate(side, summary, header)
    d = tree.run(prefix)
    _write(d, 'sidecar.json', side)
    _write(d, 'summary.json', summary)
    _csv(d, header, [[0, 1000000, 1.0, 2.0, 1.0, 1.5, 1.5, 0.5, 0.5]])
    return d


def test_figure_c_accepts_a_run_under_todays_model(tmp_path):
    assert MRM._figc_big_enough(write_figc(Tree(tmp_path)))


@pytest.mark.parametrize('mutate', OLD_MODEL + UNCLEAN + [
    # the no-anonymous sensitivity run (R27)
    lambda s, m, h: (s['args'].update(exclude_anonymous=True),
                     m['data'].update(exclude_anonymous=True)),
    # adapter 1.1 cleaning (cancellations without their reversed purchases)
    lambda s, m, h: m['data'].update(adapter_version='1.1'),
    lambda s, m, h: s['provenance'].update(adapter_version='1.1'),
    lambda s, m, h: m['results'].pop('closed_form'),              # R02/R25
    lambda s, m, h: m['feasibility'].pop('aisle_rule'),           # R06
    lambda s, m, h: m['search_design'].pop('operators'),
    lambda s, m, h: s['args'].update(n_gens=20),
    lambda s, m, h: s['args'].update(max_items_per_category=8),
])
def test_figure_c_refuses_old_model_cleaning_or_design(tmp_path, mutate):
    assert not MRM._figc_big_enough(write_figc(Tree(tmp_path), mutate))


def _no_anonymous(s, m, h):
    s['args'].update(exclude_anonymous=True)
    m['data'].update(exclude_anonymous=True)


def test_figure_c_without_anonymous_invoices_is_its_own_family(tmp_path):
    """R27: the sensitivity run without the anonymous invoices has a
    validator of its own -- Figure C's design and checks, with the
    invoices excluded, under the no-anonymous prefix -- and never counts
    as the headline, nor the headline as it."""
    tree = Tree(tmp_path)
    noanon = write_figc(tree, _no_anonymous,
                        prefix=MRM.NOANON_PREFIX + 'real_data_uci')
    assert MRM._figc_noanon_big_enough(noanon)
    assert not MRM._figc_big_enough(noanon)
    headline = write_figc(tree)
    assert MRM._figc_big_enough(headline)
    assert not MRM._figc_noanon_big_enough(headline)
    # the right flags under the headline's prefix, or the design changed
    assert not MRM._figc_noanon_big_enough(write_figc(tree, _no_anonymous))
    assert not MRM._figc_noanon_big_enough(write_figc(
        tree, lambda s, m, h: (_no_anonymous(s, m, h),
                               s['args'].update(n_gens=20)),
        prefix=MRM.NOANON_PREFIX + 'real_data_uci'))
    # The headline glob never reaches the sensitivity family's runs, and
    # each family picks its own newest valid run.
    old = MRM.RES
    try:
        MRM.RES = tree.root
        assert MRM.latest('real_data_uci_', MRM._figc_big_enough) == headline
        assert MRM.latest('noanon_real_data_uci_',
                          MRM._figc_noanon_big_enough) == noanon
        import glob as _glob
        assert not any(os.path.basename(x).startswith(MRM.NOANON_PREFIX)
                       for x in _glob.glob(os.path.join(tree.root,
                                                        'real_data_uci_*')))
    finally:
        MRM.RES = old


def test_figure_c_no_anonymous_macros_carry_their_own_stem(monkeypatch):
    full = {'FigCLift': '+\\pounds 900', 'FigCLiftPct': '+1.00\\%',
            'FigCGAvsRandSearchPct': '+0.30\\%', 'FigCGAvsSAPct': '+0.90\\%',
            'FigCAnonInvoiceFrac': '7.4\\%', 'FigCGens': '25'}
    monkeypatch.setattr(MRM, 'figc_macros', lambda d: dict(full))
    monkeypatch.setattr(MRM, '_side', lambda d: {'provenance': {'extra': {
        'anonymous': {'n_anonymous_invoices_excluded': 1234,
                      'n_anonymous_rows_excluded': 56789,
                      'anonymous_revenue_excluded': 1.5e6}}}})
    out = MRM.figc_noanon_macros('noanon_real_data_uci_x')
    assert out['FigCNoAnonLift'] == '+\\pounds 900'
    assert out['FigCNoAnonLiftPct'] == '+1.00\\%'
    assert out['FigCNoAnonGAvsRandSearchPct'] == '+0.30\\%'
    assert out['FigCNoAnonGAvsSAPct'] == '+0.90\\%'
    assert out['FigCNoAnonInvoicesExcluded'] == '1,234'
    # nothing leaks out under the headline's stem
    assert not any(k.startswith('FigC') and not k.startswith('FigCNoAnon')
                   for k in out)
    assert 'FigCNoAnonGens' not in out


def write_budget(tree, mutate=None):
    import experiments.run_real_data_budget as RDB
    a = _figc_defaults()
    mults = list(RDB.DEFAULT_BUDGET_MULTIPLIERS)
    design = _figc_design(
        budget_multipliers=mults,
        n_gens_per_multiplier={str(m): a['n_gens'] * m for m in mults},
        n_search_seeds=RDB.DEFAULT_N_SEARCH_SEEDS,
        first_search_seed=a['ga_seed'], sa_k_move=10,
        operators={'GA': 'region'})
    design.pop('n_gens')
    summary = {'design': design,
               'data': {'exclude_anonymous': False,
                        'adapter_version': MRM._adapter_version()},
               'search_start': {m: 'asbuilt' for m in RDB.METHODS},
               'feasibility': {'aisle_rule': _aisle()},
               'results': {'closed_form_conclusions_unchanged':
                           {'all_unchanged': True},
                           'reference': {}, 'per_method': {}}}
    side = _clean(args=a, base_params=_bp(),
                  provenance={'adapter_version': MRM._adapter_version()})
    if mutate:
        mutate(side, summary, None)
    d = tree.run('real_data_budget')
    _write(d, 'sidecar.json', side)
    _write(d, 'summary.json', summary)
    return d


def test_budget_sweep_accepts_the_runners_design(tmp_path):
    assert MRM._budget_big_enough(write_budget(Tree(tmp_path)))


@pytest.mark.parametrize('mutate', OLD_MODEL + [
    _moved,
    lambda s, m, h: m['design'].update(budget_multipliers=[1, 2],
                                       n_gens_per_multiplier={'1': 25,
                                                              '2': 50}),
    lambda s, m, h: m['design'].update(n_search_seeds=2),
    lambda s, m, h: m['design'].update(exclude_anonymous=True),
    lambda s, m, h: m['data'].update(adapter_version='1.1'),
    lambda s, m, h: m['results'].pop('closed_form_conclusions_unchanged'),
    lambda s, m, h: m['design'].pop('operators'),
    lambda s, m, h: m['search_start'].update(GA='popularity'),
    lambda s, m, h: m['design'].update(max_items_per_category=8),
])
def test_budget_sweep_refuses_other_designs(tmp_path, mutate):
    assert not MRM._budget_big_enough(write_budget(Tree(tmp_path), mutate))


def write_input(tree, mutate=None):
    import experiments.run_input_uncertainty as RIU
    a = _figc_defaults()
    budget = a['n_gens'] * a['pop_size']
    iu = {sc: {'ga_minus_rs': {}, 'ga_minus_sa': {}}
          for sc in ('whole', 'stocked')}
    design = {'sheets': a['sheets'],
              'max_items_per_category': a['max_items_per_category'],
              'assumed_conversion': a['assumed_conversion'],
              'exclude_anonymous': False,
              'adapter_version': MRM._adapter_version(),
              'n_boot': RIU.DEFAULT_N_BOOT,
              'ga': {'seed': a['ga_seed'], 'n_gens': a['n_gens'],
                     'pop_size': a['pop_size'], 'mc_iters': a['mc_iters'],
                     'mc_days': a['mc_days']},
              'searches': {'sa_initial_accept': a['sa_initial_accept'],
                           'starts': {m: 'asbuilt' for m in
                                      ('GA', 'random_search',
                                       'simulated_annealing')},
                           'evaluation_counts': {'GA': budget,
                                                 'random_search': budget,
                                                 'simulated_annealing':
                                                     budget},
                           'final_evaluation_counts': {'GA': 150,
                                                       'random_search': 150,
                                                       'simulated_annealing':
                                                           150},
                           'operators': {'GA': 'region'}}}
    summary = {'design': design, 'input_uncertainty': iu,
               'legacy_floor_bias': {'adapter_1_1_inputs': {}},
               'feasibility': {'aisle_rule': _aisle()}}
    side = _clean(args=a, base_params=_bp())
    if mutate:
        mutate(side, summary, None)
    d = tree.run('input_uncertainty')
    _write(d, 'sidecar.json', side)
    _write(d, 'summary.json', summary)
    return d


def test_input_uncertainty_accepts_figure_cs_design(tmp_path):
    assert MRM._input_big_enough(write_input(Tree(tmp_path)))


@pytest.mark.parametrize('mutate', OLD_MODEL + [
    _dirty,
    lambda s, m, h: m['design'].update(n_boot=3),
    lambda s, m, h: m['design']['ga'].update(n_gens=2),
    lambda s, m, h: m['design'].update(exclude_anonymous=True),
    lambda s, m, h: m['design'].update(adapter_version='1.1'),
    # runs from before the comparators' margins were re-scored (R11)
    lambda s, m, h: m['input_uncertainty']['whole'].pop('ga_minus_rs'),
    lambda s, m, h: m['legacy_floor_bias'].pop('adapter_1_1_inputs'),
    lambda s, m, h: m['design']['searches']['starts'].update(
        simulated_annealing='popularity'),
    lambda s, m, h: m['design']['searches']['evaluation_counts'].update(
        random_search=12),
])
def test_input_uncertainty_refuses_other_runs(tmp_path, mutate):
    assert not MRM._input_big_enough(write_input(Tree(tmp_path), mutate))


def write_transfer(tree, mutate=None):
    import experiments.run_heldout_transfer as RHT
    a = _figc_defaults()
    budget = a['n_gens'] * a['pop_size']
    seeds = list(range(RHT.DEFAULT_N_SEARCH_SEEDS))
    design = _figc_design(
        search_seeds=seeds, n_search_seeds=len(seeds),
        budget_search_evals=budget, search_start='asbuilt',
        operators={'GA': 'region'},
        periods={'prior': {'adapter_version': MRM._adapter_version(),
                           'cut_applied_to': 'raw rows',
                           'reversals_across_cut': {'purchase_lines': 102},
                           'n_invoices_shared_with_current': 0},
                 'current': {'adapter_version': MRM._adapter_version()}})
    design.pop('sheets')
    methods = {m: {'transfer_gap': {}} for m in RHT.METHODS}
    summary = {'design': design,
               'across_seeds': {'methods': methods},
               'evaluation_counts': {f'{p}|{s}': {'GA': budget,
                                                  'random_search': budget,
                                                  'simulated_annealing':
                                                      budget}
                                     for p in ('prior', 'current')
                                     for s in seeds},
               'feasibility': {'aisle_rule': _aisle()}}
    side = _clean(args=a, base_params={'prior': {'score_anchor':
                                                 dict(ANCHOR)},
                                       'current': {'score_anchor':
                                                   dict(ANCHOR)}})
    if mutate:
        mutate(side, summary, None)
    d = tree.run('heldout_transfer')
    _write(d, 'sidecar.json', side)
    _write(d, 'summary.json', summary)
    return d


def test_heldout_transfer_accepts_figure_cs_design(tmp_path):
    assert MRM._transfer_big_enough(write_transfer(Tree(tmp_path)))


@pytest.mark.parametrize('mutate', OLD_MODEL + [
    _moved,
    # the prior period cleaned before it was cut (the leak review found)
    lambda s, m, h: m['design']['periods']['prior'].pop(
        'reversals_across_cut'),
    lambda s, m, h: m['design']['periods']['current'].update(
        adapter_version='1.1'),
    lambda s, m, h: m['design'].update(n_search_seeds=2,
                                       search_seeds=[0, 1]),
    lambda s, m, h: m['design'].update(exclude_anonymous=True),
    # runs from before the paired transfer gap (R12)
    lambda s, m, h: m['across_seeds']['methods']['GA'].pop('transfer_gap'),
    lambda s, m, h: m['design'].update(mc_iters=100),
])
def test_heldout_transfer_refuses_other_runs(tmp_path, mutate):
    assert not MRM._transfer_big_enough(write_transfer(Tree(tmp_path),
                                                       mutate))


# --- Live families -------------------------------------------------------------------

def write_abm(tree, mutate=None):
    tours = MRM._script_constants(MRM.LIVE_RUNNERS['abm'],
                                  ('ROUTING_NULL_TOURS',))['ROUTING_NULL_TOURS']
    rep = {'emergence': {'perimeter_interior_ratio_geometric_null': 0.9,
                         'band_sweep_geometric_null': {}},
           'load': {'arrivals': 100, 'balked': 1}}
    summary = {
        'data': live_data(), 'protocol': live_protocol('abm'),
        'markov_order': {'info_gain_ci95': 0.01, 'tv_ci95': 0.01,
                         'n_censored_final': 0, 'bookkeeping_null': {}},
        'emergence': {'perimeter_ratio_ci95': 0.1,
                      'perimeter_interior_ratio_geometric_null': 0.9,
                      'ratio_to_geometric_null': 2.0,
                      'ratio_to_geometric_null_ci95': 0.1,
                      'ratio_to_routing_null_ci95': 0.05,
                      'routing_null': {'construction':
                                       MRM.ROUTING_NULL_CONSTRUCTION,
                                       'n_tours': tours},
                      'moving_only': {}, 'headline': {}},
        'per_rep': [rep] * 10}
    side = _clean()
    if mutate:
        mutate(side, summary, None)
    d = tree.run('abm_diagnostics')
    _write(d, 'sidecar.json', side)
    _write(d, 'summary.json', summary)
    return d


def test_abm_accepts_the_stage2_protocol(tmp_path):
    assert MRM._abm_big_enough(write_abm(Tree(tmp_path)))


@pytest.mark.parametrize('mutate', UNCLEAN + [
    lambda s, m, h: m['protocol'].update(framing='terminating'),
    # the per-fixture routing null (R13)
    lambda s, m, h: m['emergence']['routing_null'].update(
        construction='per_fixture_trips'),
    lambda s, m, h: m['emergence']['routing_null'].update(n_tours=500),
    lambda s, m, h: m['emergence'].pop('moving_only'),
    lambda s, m, h: m['markov_order'].pop('bookkeeping_null'),
    lambda s, m, h: m['data'].update(adapter_version='1.1'),
    lambda s, m, h: m['data'].pop('exclude_anonymous'),
    lambda s, m, h: m['data'].update(list_law='popularity-weighted'),
    lambda s, m, h: m['protocol'].update(collect_s=900),
])
def test_abm_refuses_runs_before_the_live_package(tmp_path, mutate):
    assert not MRM._abm_big_enough(write_abm(Tree(tmp_path), mutate))


def write_struct(tree, mutate=None):
    summary = {'n_reps': 5, 'data': live_data(),
               'protocol': live_protocol('structural'),
               'rev_per_cust_anova_p': 0.5, 'completions_anova_p': 0.5,
               'design': MRM.STRUCT_DESIGN, 'paired_contrasts': {},
               'block_anova': {}, 'mde': {}}
    side = _clean()
    if mutate:
        mutate(side, summary, None)
    d = tree.run('structural_sensitivity')
    _write(d, 'sidecar.json', side)
    _write(d, 'summary.json', summary)
    return d


def test_structural_sweep_accepts_common_random_numbers(tmp_path):
    assert MRM._struct_big_enough(write_struct(Tree(tmp_path)))


@pytest.mark.parametrize('mutate', [
    _dirty,
    lambda s, m, h: m.pop('design'),                  # separate seeds (R40)
    lambda s, m, h: m.pop('paired_contrasts'),
    lambda s, m, h: m.pop('mde'),
    lambda s, m, h: m['protocol'].update(framing='terminating'),
])
def test_structural_sweep_refuses_the_old_design(tmp_path, mutate):
    assert not MRM._struct_big_enough(write_struct(Tree(tmp_path), mutate))


def write_queue(tree, mutate=None):
    des = MRM._live_design('queue')
    proto = live_protocol('queue', n_reps=5,
                          spawn={'nominal': des['NOMINAL_SPAWN'],
                                 'stress': des['STRESS_SPAWN']},
                          cap={'nominal': des['NOMINAL_CAP'],
                               'stress': des['STRESS_CAP']})
    summary = {'protocol': proto, 'data': live_data(),
               'provenance': {'git_sha': SHA}}
    side = _clean()
    if mutate:
        mutate(side, summary, None)
    d = tree.run('measure_queueing')
    _write(d, 'sidecar.json', side)
    _write(d, 'summary.json', summary)
    return d


def test_queueing_accepts_and_refuses(tmp_path):
    tree = Tree(tmp_path)
    assert MRM._queue_big_enough(write_queue(tree))
    assert not MRM._queue_big_enough(write_queue(
        tree, lambda s, m, h: m['protocol'].update(framing='terminating')))
    assert not MRM._queue_big_enough(write_queue(
        tree, lambda s, m, h: m['data'].update(adapter_version='1.1')))
    assert not MRM._queue_big_enough(write_queue(tree, _dirty))


def _gof_block():
    g = MRM._script_constants(MRM.GOF_SCRIPT, MRM._GOF_RUNNER_NAMES)
    cat = MRM._script_constants(MRM.GOF_CATEGORY_SCRIPT,
                                MRM._GOF_CATEGORY_NAMES)
    return {'n_reps': 5,
            'protocol': live_protocol('gof', cohort=g['COHORT']),
            'load': {'arrivals': 100, 'balked': 0, 'door_rate_per_s': 0.2,
                     'cohort_visits': 3},
            'list_chain': {'tests': [{'test': g['LIST_CHAIN_DRAWN']},
                                     {'test': g['LIST_CHAIN_SELECTION']}]},
            'pooled': [{'test': 'Category purchase shares',
                        'kind': cat['CATEGORY_TEST_KIND'],
                        'n_permutations': cat['CATEGORY_N_PERM'],
                        'p_value': 0.3, 'naive_p_value': 0.1,
                        'share_tv_distance': 0.05}]}


def write_gof(tree, mutate=None, n_lines=None):
    keys = MRM._script_constants(MRM.GOF_SCRIPT, ('ANONYMOUS_LIST_KEYS',))[
        'ANONYMOUS_LIST_KEYS']
    replica = {row: {} for row in MRM.GOF_REPLICA_ROWS}
    design = {d: {**{k: 0.06 for k in keys}, 'replica': replica}
              for d in MRM.GOF_DESIGNS}
    design['held_out'].update(year_shift={'category': {}},
                              year_shift_fixed_assortment={})
    design['periods'] = {
        'current': live_data(),
        'prior': live_data(period='prior',
                           reversals_across_cut={'purchase_lines': 102})}
    summary = {**_gof_block(), 'held_out': _gof_block(), 'design': design,
               'visits_file': {'path': 'visits.jsonl', 'n_records': 6}}
    side = _clean()
    if mutate:
        mutate(side, summary, None)
    d = tree.run('validation_gof')
    _write(d, 'sidecar.json', side)
    _write(d, 'summary.json', summary)
    with open(os.path.join(d, 'visits.jsonl'), 'w') as f:
        for _ in range(6 if n_lines is None else n_lines):
            f.write('{}\n')
    return d


def test_gof_accepts_the_drained_cohort(tmp_path):
    assert MRM._gof_big_enough(write_gof(Tree(tmp_path)))


@pytest.mark.parametrize('mutate', UNCLEAN + [
    # the old cohort: every exit in the window (R46)
    lambda s, m, h: m['protocol'].update(cohort='window_exits'),
    lambda s, m, h: m['held_out']['protocol'].pop('cohort'),
    lambda s, m, h: m['protocol'].update(framing='terminating'),
    lambda s, m, h: m['load'].pop('door_rate_per_s'),
    # the former overlapping chain test (R29)
    lambda s, m, h: m['list_chain']['tests'][1].update(
        test='Purchases vs drawn lists'),
    # a prior period cleaned across the cut
    lambda s, m, h: m['design']['periods']['prior'].pop(
        'reversals_across_cut'),
    lambda s, m, h: m['design']['held_out'].pop(
        'year_shift_fixed_assortment'),                          # R16
    lambda s, m, h: m['design']['in_sample'].pop('anonymous_list_share'),
    lambda s, m, h: m['visits_file'].update(n_records=5),
    lambda s, m, h: m['design']['periods']['current'].update(
        adapter_version='1.1'),
])
def test_gof_refuses_runs_before_the_live_package(tmp_path, mutate):
    assert not MRM._gof_big_enough(write_gof(Tree(tmp_path), mutate))


def test_gof_refuses_a_visits_file_short_of_its_record(tmp_path):
    assert not MRM._gof_big_enough(write_gof(Tree(tmp_path), n_lines=5))


def write_proto(tree, mutate=None):
    c = MRM._script_constants(MRM.PROTO_SCRIPT, MRM._PROTO_NAMES)
    summary = {
        'derived': {'NOMINAL_SPAWN': 0.27, 'NOMINAL_CAP': 45,
                    'WARMUP_S': 1020.0, 'COLLECT_S': 1860.0,
                    'STRESS_SPAWN': 0.7, 'STRESS_CAP': 100},
        'matches': True, 'protocol_status': 'reproduced_drift_not_equivalent',
        # The runners' constants as the study imported them: today's.
        'runner_constants': {runner: dict(MRM._live_design(fam))
                             for runner, fam in MRM.PROTO_RUNNERS.items()},
        'drift': {'n_reps': c['N_REPS']},
        'double_warmup': {'cohort': 'window_arrivals_drained',
                          'stats': {k: {} for k in c['HEADLINE_OUTCOMES']}},
        'design': {'mode': 'fixed_step', 'dt': 0.04,
                   'framing': MRM.LIVE_FRAMING,
                   'nominal': {'grid': list(c['NOMINAL_GRID']),
                               'cap': c['NOMINAL_CAP'], 'reps': c['N_REPS'],
                               'run_s': c['RUN_S']},
                   'stress': {'grid': list(c['STRESS_GRID']),
                              'cap_grid': list(c['CAP_GRID']),
                              'rate_reps': c['STRESS_REPS'],
                              'cap_reps': c['CAP_REPS'],
                              'run_s': c['STRESS_RUN_S']}},
        'data': live_data()}
    side = _clean()
    if mutate:
        mutate(side, summary, None)
    d = tree.run('live_protocol_study')
    _write(d, 'sidecar.json', side)
    _write(d, 'summary.json', summary)
    return d


def test_protocol_study_accepts_the_full_design(tmp_path):
    assert MRM._proto_big_enough(write_proto(Tree(tmp_path)))


@pytest.mark.parametrize('mutate', [
    _moved,
    lambda s, m, h: m['design']['nominal'].update(reps=2),
    lambda s, m, h: m['design']['nominal'].update(grid=[0.27]),
    lambda s, m, h: m['design']['stress'].update(run_s=600.0),
    lambda s, m, h: m['derived'].update(COLLECT_S=None),
    # a double warm-up before it followed the window's arrivals out
    lambda s, m, h: m['double_warmup'].pop('cohort'),
    lambda s, m, h: m['double_warmup']['stats'].pop('revenue_per_visit'),
    lambda s, m, h: m.pop('protocol_status'),
    lambda s, m, h: m['data'].update(adapter_version='1.1'),
    # compared with runner constants the runners no longer use: its
    # 'reproduced' / 'differs' would describe code that is gone
    lambda s, m, h: m['runner_constants']['run_abm_diagnostics'].update(
        WARMUP_S=900.0),
    lambda s, m, h: m['runner_constants']['measure_queueing'].update(
        STRESS_SPAWN=0.6),
    lambda s, m, h: m['runner_constants'].pop('run_validation_gof'),
    lambda s, m, h: m.pop('runner_constants'),
])
def test_protocol_study_refuses_smaller_or_older_runs(tmp_path, mutate):
    assert not MRM._proto_big_enough(write_proto(Tree(tmp_path), mutate))


def write_heatmap(tree, mutate=None):
    seed = MRM._script_constants(MRM.HEATMAP_SCRIPT,
                                 ('FIGURE_SEED',))['FIGURE_SEED']
    proto = live_protocol('abm', seed=seed)
    summary = {'protocol': proto, 'data': live_data(),
               'figure': {'min_font_pt_printed': 7.0},
               'perimeter_interior_ratio_geometric_null': 0.92}
    side = _clean()
    if mutate:
        mutate(side, summary, None)
    d = tree.run('heatmap_figure')
    _write(d, 'sidecar.json', side)
    _write(d, 'summary.json', summary)
    return d


def test_heatmap_accepts_and_refuses(tmp_path):
    tree = Tree(tmp_path)
    assert MRM._heatmap_big_enough(write_heatmap(tree))
    for mutate in (lambda s, m, h: m['protocol'].update(seed=1),
                   lambda s, m, h: m['protocol'].update(warmup_s=600.0),
                   lambda s, m, h: m['figure'].update(min_font_pt_printed=4.3),
                   lambda s, m, h: m['data'].update(adapter_version='1.1'),
                   _dirty):
        assert not MRM._heatmap_big_enough(write_heatmap(tree, mutate))


# --- The code the checks read -----------------------------------------------------------

def test_defaults_read_from_source_are_the_parsers():
    """The validators hold runs to the runners' defaults read from their
    source; those must be what the runners' own parsers produce."""
    import experiments.run_synthetic_gt as RSG
    import experiments.run_baseline_comparison as RBC
    import experiments.run_mc_groundtruth as RMG
    import experiments.run_ga_sensitivity as RGS
    import experiments.run_elasticity_lhs as LHS
    import experiments.run_objective_alignment as ROA
    import experiments.run_real_data_budget as RDB
    import experiments.run_real_data_seeds as RDS
    import experiments.make_paper_figures as MPF
    cases = [(MRM.FIGURES_SCRIPT, _defaults(MPF, argv=[])),
             (MRM.FIGA_SCRIPT, _defaults(RSG)),
             (MRM.FIGB_SCRIPT, _defaults(RBC)),
             (MRM.MCGT_SCRIPT, _defaults(RMG)),
             (MRM.GASENS_SCRIPT, _defaults(RGS, argv=[])),
             (MRM.LHS_SCRIPT, _defaults(LHS)),
             (MRM.ALIGN_SCRIPT, _defaults(ROA)),
             (MRM.FIGC_SCRIPT, _figc_defaults())]
    for script, parsed in cases:
        read = MRM._argparse_defaults(script)
        for key, value in read.items():
            assert key in parsed, (script, key)
            assert MRM._same(value, parsed[key]), (script, key, value,
                                                   parsed[key])
    assert tuple(MRM._design_default(MRM.BUDGET_SCRIPT,
                                     'budget_multipliers')) \
        == tuple(RDB.DEFAULT_BUDGET_MULTIPLIERS)
    assert MRM._design_default(MRM.SEEDS_SCRIPT, 'n_search_seeds') \
        == RDS.DEFAULT_N_SEARCH_SEEDS
    assert MRM._oracle_restarts() == C.ORACLE_RESTARTS
    assert MRM._final_seeds() == C.GA_N_FINAL_SEEDS
    MRM._check_sources()


def test_todays_runner_stamp_is_todays_model():
    """What a runner stamps today -- the elasticity snapshot and the
    anchored base parameters of a real store -- passes the model check,
    and the anchor's source is the one the check requires."""
    from synthetic_shops import generate_synthetic_shop
    shop_synth = generate_synthetic_shop(name='mrm', seed=10_001, n_items=10)
    shop = C.build_headless_shop(shop_synth)
    names = [it.name for it in shop_synth.items]
    init = {n: tuple(shop.floors[1]['items'][n]['position']) for n in names}
    bp = C.anchor_base_params(shop, names, C.base_params_for(shop_synth), init)
    assert bp['score_anchor']['source'] == MRM.ANCHOR_SOURCE
    rec = json.loads(json.dumps({
        'elasticities': C.elasticity_snapshot(),
        'base_params': {'0': C.base_params_record(bp)}}, default=str))
    assert MRM._model_current(rec)
    unanchored = copy.deepcopy(rec)
    unanchored['base_params']['0'].pop('score_anchor')
    assert not MRM._model_current(unanchored)


def realized_scores(**config_change):
    """A realized-scores record as ``make_paper_figures`` writes it, at the
    script's own defaults."""
    import experiments.make_paper_figures as MPF
    cfg = _defaults(MPF, argv=[])
    cfg.update(config_change)
    return {'config': cfg,
            'provenance': {'git_sha': SHA, 'git_dirty': False,
                           'git_state_changed_during_run': False},
            'mc_rse_pct': 0.2, 'mc_rse_iters': cfg['mc_iters'],
            'mc_trace_iters': 20000,
            'score_baseline': 0.4, 'score_ga': 0.5, 'score_anchor': 0.41,
            'conv_lift_pct': 2.0, 'impulse_lift_pct': 1.0,
            'basket_lift_pct': 0.5,
            'hardware': {'cpu_count': 16, 'processor': 'CPU X',
                         'machine': 'AMD64', 'ram_gib': 15.9},
            'wall_seconds': 5400.0,
            'elasticities': json.loads(json.dumps(C.elasticity_snapshot())),
            'base_params': {'score_anchor': dict(ANCHOR)}}


def test_realized_scores_need_todays_model(tmp_path):
    rs = realized_scores()
    assert MRM._figures_ok(rs)
    clean = rs['provenance']
    for change in ({'base_params': {}}, {'elasticities': {}},
                   {'provenance': {**clean, 'git_dirty': True}},
                   # the checkout moved while the run was in flight, or
                   # the record cannot say (R50: every family clean)
                   {'provenance': {**clean,
                                   'git_state_changed_during_run': True}},
                   {'provenance': {k: v for k, v in clean.items()
                                   if k != 'git_state_changed_during_run'}}):
        assert not MRM._figures_ok({**rs, **change})


@pytest.mark.parametrize('change', [
    {'n_seeds': 10}, {'pop_size': 60}, {'n_gens': 20}, {'mc_iters': 4000},
    {'mc_days': 14}, {'n_items': 12}, {'scenario_seed': 1}])
def test_realized_scores_are_held_to_the_scripts_design(change):
    """R50: the figures' design is the script's defaults exactly -- a
    larger run is refused as surely as a smaller one, since the Fig. 9
    caption quotes the design."""
    assert not MRM._figures_ok(realized_scores(**change))


def test_realized_scores_design_macros_and_table_row(tmp_path):
    rs = realized_scores()
    out = MRM.figures_macros(rs)
    c = rs['config']
    assert out['FiguresSeeds'] == str(c['n_seeds'])
    assert out['FiguresPop'] == str(c['pop_size'])
    assert out['FiguresGens'] == str(c['n_gens'])
    assert out['FiguresMcIters'] == f"{c['mc_iters']:,}"
    assert out['FiguresWallHours'] == '1.5'
    path = tmp_path / 'realized_scores.json'
    path.write_text(json.dumps(rs))
    table = MRM.budget_table({'paper_figures': str(path)})
    row = next(line for line in table.splitlines()
               if line.startswith('Paper figures'))
    assert 'TBD' not in row
    assert f"{c['n_seeds']} GA seeds" in row and '20,000' in row


# --- Formatting ------------------------------------------------------------------------

def test_replica_shares_print_as_counts_not_rounded_percentages():
    """R57: 199 of 200 is not '100%', and 97.5% is not 'above 98%'."""
    out = MRM._replica_macros('GofHeldOut', {'basket': {
        'n_replicas': 200, 'median': 0.02, 'lo': 0.01, 'hi': 0.03,
        'reject_rate': 0.995, 'simulator_percentile': 0.975,
        'simulator_vs_median': 'above',
        'simulator_mean_minus_reference': -0.123,
        'simulator_mean_percentile': 0.2}})
    assert out['GofHeldOutBasketReplicaReject'] == '199/200'
    assert out['GofHeldOutBasketReplicaRejectPct'] == '99.5\\%'
    assert out['GofHeldOutBasketReplicaBelow'] == '195/200'
    assert out['GofHeldOutBasketReplicaBelowPct'] == '97.5\\%'
    assert out['GofHeldOutBasketReplicaSide'] == 'above'
    assert out['GofHeldOutBasketReplicaSimMeanMinusRef'] == '-0.12'
    assert out['GofReplicaN'] == '200'


def test_p_values_and_words():
    assert MRM.pfmt(0.0004) == '\\ensuremath{<0.001}'
    assert MRM.pfmt(0.21) == '0.210'
    assert MRM.count_of(1234, 5000) == '1,234/5,000'
    assert [MRM.num_word(n) for n in (1, 3, 10, 24, 96, 100)] == \
        ['One', 'Three', 'Ten', 'TwentyFour', 'NinetySix', 'OneHundred']
    assert MRM._status_text('reproduced_drift_not_equivalent') == \
        'reproduced; drift not equivalent'


def test_structural_and_markov_p_values_go_through_pfmt(tmp_path):
    """R57: the p-values that bypassed pfmt (the chi-square, the revenue
    ANOVA, the Markov permutation p) now go through it."""
    s = {'throughput_range_pct': 3.0, 'chi2_p': 0.0002,
         'completions_anova_p': 0.5, 'rev_per_cust_anova_p': 0.2,
         'rev_per_cust_mean': [1.0, 1.1], 'rev_per_cust_sd': [0.1, 0.1],
         'n_reps': 5, 'completed_per_setting': [10, 11],
         'strengths': [0.0, 0.08], 'default_strength': 0.08,
         'default_index': 1, 'block_anova': {}, 'mde': {'power': 0.8},
         'paired_contrasts': {'completed': {'0.0': {'p': 0.04,
                                                    'mean_diff_pct_of_default':
                                                        -2.0,
                                                    'ci95_half_width': 1.0,
                                                    'mde80_pct_of_default':
                                                        5.0}}},
         'per_rep_values': {'completed': [[10] * 5, [11] * 5]}}
    d = tmp_path / 'structural_sensitivity_x'
    d.mkdir()
    (d / 'summary.json').write_text(json.dumps(s))
    out = MRM.struct_macros(str(d))
    assert out['StructChiP'] == '\\ensuremath{<0.001}'
    assert out['StructRevAnovaP'] == '0.200'
    assert out['StructPairOffCompP'] == '0.040'
    assert out['StructPairOffCompPct'] == '-2.0\\%'
    assert out['StructPairCompNSig'] == '1/1'


def test_pooled_balk_share_counts_every_nominal_replication():
    """R66: one pooled share over the four nominal-load families. The
    structural sweep's settings share their replication seeds, so each
    replication block counts once, over all settings."""
    abm = {'per_rep': [{'load': {'arrivals': a, 'balked': b}}
                       for a, b in ((100, 1), (120, 0), (80, 2))]}
    struct = {'rows': [{'runs': [{'rep': r, 'customers': 90, 'balked': r}
                                 for r in range(2)]} for _ in range(2)]}
    queue = {'nominal': {'per_rep': [{'arrivals': 50, 'balked': 0},
                                     {'arrivals': 60, 'balked': 1}]}}
    gof = {'per_rep': [{'load': {'arrivals': 80, 'balked': 4}},
                       {'load': {'arrivals': 70, 'balked': 0}}]}
    out = MRM.pooled_balk_macros(abm, struct, queue, gof)
    arr = 300 + (2 * 90 + 0) + (2 * 91) + 110 + 150
    bal = 3 + (0 + 2) + 1 + 4
    assert out['LiveBalkedPooledCount'] == MRM.count_of(bal, arr)
    assert out['LiveBalkedPooled'] == f"{100 * bal / arr:.2f}\\%"
    assert out['LiveBalkedPooledReps'] == str(3 + 2 + 2 + 2)
    lo, hi = (float(x.strip(' \\%')) for x in
              out['LiveBalkedPooledCI'].strip('[]').split(','))
    assert lo < round(100 * bal / arr, 2) < hi
    # The pooled share is weighted by arrivals, so each family's own share
    # is given beside it, and the three families other than the structural
    # sweep (which contributes most of the arrivals) are pooled apart.
    assert out['LiveBalkedPooledWeighting'] == 'arrival-weighted'
    assert out['LiveBalkedAbmCount'] == MRM.count_of(3, 300)
    assert out['LiveBalkedStruct'] == f"{100 * 2 / 362:.2f}\\%"
    assert out['LiveBalkedQueueNomCount'] == MRM.count_of(1, 110)
    assert out['LiveBalkedGofCount'] == MRM.count_of(4, 150)
    assert out['LiveBalkedNonStructCount'] == MRM.count_of(8, 560)
    assert out['LiveBalkedNonStruct'] == f"{100 * 8 / 560:.2f}\\%"
    assert out['LiveBalkedNonStructReps'] == str(3 + 2 + 2)


def test_live_load_is_reported_beside_the_calibrated_rate():
    """R04: the live store's calibrated buyer and visitor rates, the load
    the protocol put on it as a multiple of the calibrated visitor rate
    (the review's '46x': 0.27/s x 0.744 against 15.8 visitors an hour),
    the live conversion as a count, and the assortment rule (R34)."""
    period = {'arrivals_per_hour': 4.74, 'visitors_per_hour': 15.8,
              'assumed_conversion_rate': 0.30, 'max_items_per_category': 8}
    load = {'offered_rate_per_s': 0.27 * 0.744, 'profile_multiplier': 0.744}
    counts = {'n_visits': 3000, 'n_paying': 2910}
    out = MRM.live_load_macros(period, load, counts)
    assert out['LiveStoreBuyersPerHour'] == '4.7'
    assert out['LiveStoreVisitorsPerHour'] == '15.8'
    assert out['LiveStoreAssumedConversion'] == '0.30'
    assert out['LiveOfferedPerHour'] == '723'
    assert out['LiveLoadMultiple'] == '46'
    assert out['LiveLoadMultipleSameHour'] == '62'
    assert out['LiveConversionCount'] == '2,910/3,000'
    assert out['LiveConversionPct'] == '97.0\\%'
    assert out['LiveStoreItemsPerCategory'] == '8'
    # A period record from before the rates were recorded sets none of them.
    assert MRM.live_load_macros({}, load, {}) == {}


def test_protocol_study_reports_the_live_conversion_level(tmp_path):
    """R04: the live store's conversion after the runners' warm-up, beside
    the double warm-up's difference."""
    def fill(s, m, h):
        m['drift'].update(slope_per_1000s_mean=0.9, slope_per_1000s_ci95=1.9,
                          slope_p=0.09, relative_change_over_window_mean=0.01,
                          relative_change_ci95=0.02,
                          relative_change_ci90=0.015, tost_margin=0.01,
                          tost_p=0.2, early_minus_late_p=0.4)
        m['design'].update(n_runs=100, profile_multiplier=0.744,
                           start_hour=9)
        m['double_warmup']['stats']['conversion'] = {
            'after_1x_warmup': 0.9646, 'after_2x_warmup': 0.9697,
            'diff_pct': 0.53, 'diff_ci95': 0.04, 'p': 0.35}
    d = write_proto(Tree(tmp_path), fill)
    assert MRM._proto_big_enough(d)
    out = MRM.proto_macros(d)
    assert out['ProtoDblConversionLevel'] == '96.5\\%'
    assert out['ProtoDblConversionDiffPct'] == '+0.5\\%'
    assert out['ProtoMatches'] == 'yes'


def test_the_study_compares_with_the_constants_the_check_reads():
    """The protocol study imports the live runners' constants; the check
    reads them from source. They must name the same runners and values."""
    import experiments.run_live_protocol_study as P
    rc = P.runner_constants()
    assert set(rc) == set(MRM.PROTO_RUNNERS)
    assert MRM._proto_runner_constants_current({'runner_constants': rc})


def test_wall_time_workers_and_hardware(tmp_path):
    """R53: each family's wall time and worker count from its sidecar, and
    one hardware statement when every run was made on the same machine --
    per family otherwise, never one machine for a mixed set."""
    side = _clean(wall_seconds=5400.0, args={'workers': 4},
                  hardware={'cpu_count': 16, 'processor': 'CPU X',
                            'machine': 'AMD64', 'ram_gib': 15.9})
    out = MRM.dir_macros('FigA', str(tmp_path / 'synthetic_gt_x'), side)
    assert out['FigAWallHours'] == '1.5'
    assert out['FigAWorkers'] == '4'
    assert MRM.dir_macros('Lhs', 'd', {'workers': 2, 'wall_seconds': 90}) \
        == {'LhsDir': 'd', 'LhsWallHours': '0.03', 'LhsWorkers': '2'}
    hw = side['hardware']
    same = MRM.hardware_macros({'FigA': hw, 'FigB': dict(hw)})
    assert same['HardwareSame'] == 'yes' and same['HardwareNRuns'] == '2'
    assert same['HardwareCPU'] == 'CPU X' and same['HardwareRAM'] == '15.9'
    assert same['HardwareCPUCount'] == '16'
    mixed = MRM.hardware_macros({'FigA': hw, 'FigB': {**hw, 'ram_gib': 64.0},
                                 'Abm': None})
    assert mixed['HardwareSame'] == 'no' and 'HardwareCPU' not in mixed
    assert mixed['FigBHardwareRAM'] == '64.0'
    assert mixed['FigAHardwareRAM'] == '15.9'
    assert mixed['HardwareNUnrecorded'] == '1'


def test_release_macros_follow_the_citation_file(tmp_path):
    """R23: the release the paper cites is the one CITATION.cff names, with
    the DOI once the archive has minted one."""
    cff = tmp_path / 'CITATION.cff'
    cff.write_text('cff-version: 1.2.0\n'
                   'title: "x"\n'
                   'version: 1.0.3\n'
                   'date-released: 2026-10-01\n'
                   'identifiers:\n'
                   '  - type: url\n'
                   '    value: "https://example.org"\n'
                   '  - type: doi\n'
                   '    value: 10.5281/zenodo.1234567\n'
                   'authors:\n'
                   '  - family-names: A\n', encoding='utf-8')
    out = MRM.release_macros(str(cff))
    assert out == {'ReleaseVersion': '1.0.3', 'ReleaseTag': 'v1.0.3',
                   'ReleaseDate': '2026-10-01',
                   'ReleaseDOI': '10.5281/zenodo.1234567'}
    assert MRM.release_macros(str(tmp_path / 'absent.cff')) == {}
    # The repository's own file names the release the paper must cite.
    rel = MRM.release_macros()
    assert rel['ReleaseVersion'] == MRM.read_citation()['version']


def test_every_registry_constant_the_tables_quote_has_a_macro():
    """R18, R41: every objective constant, the synthetic bands and the
    separation rate reach the coefficient tables as macros, at the values
    the code runs with."""
    import retail_literature as RL
    out = MRM.registry_macros()
    assert all(k.isalpha() for k in out)
    for name in C.OBJECTIVE_CONSTANTS:
        m = MRM._reg_name(name)
        assert m in out or (m + 'Lo' in out and m + 'Hi' in out), name
    assert out['RegSeparationRatePerS'] == \
        f"{RL.SEPARATION_RATE_PER_S:.4g}"
    assert out['RegSeparationReferenceDtS'] == '0.04'
    assert out['RegSynthDetourElasticityRangeLo'] == \
        f"{RL.SYNTH_DETOUR_ELASTICITY_RANGE[0]:g}"
    assert out['RegSynthAffinityDensity'] == f"{RL.SYNTH_AFFINITY_DENSITY:g}"
    assert out['RegMcLambdaCap'] == '\\ensuremath{10^{6}}'
    assert out['RegMcSpendLaw'] == 'lognormal\\_moment\\_matched'
    assert out['RegWcShareOfAgents'] == '33.5\\%'
    assert out['RegAgentSpeedRangeMpsLo'] == '0.8'
    assert out['RegWcProbabilityByTypeBrowser'] == '0.35'
    assert MRM._reg_name('HEAT_PRIOR_2D') == 'RegHeatPriorTwoD'


# --- Table 4 and the whole run ------------------------------------------------------------

def test_budget_table_reads_tbd_without_runs_and_fills_a_found_row(tmp_path):
    empty = MRM.budget_table({})
    assert empty.count('\\textit{TBD}') == 4 * len(MRM.TABLE_ROWS)
    assert '\\begin{tabular}' in empty and '\\end{tabular}' in empty
    d = write_figa(Tree(tmp_path))
    filled = MRM.budget_table({'synthetic_gt': d})
    row = next(line for line in filled.splitlines()
               if line.startswith('Recovery'))
    assert 'TBD' not in row and '30 scenarios' in row and '96 restarts' in row
    assert filled.count('\\textit{TBD}') == 4 * (len(MRM.TABLE_ROWS) - 1)


def test_the_generator_runs_on_a_tree_with_only_figures_a_and_b(
        tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(MRM, 'RES', MRM.RES)      # main() repoints it
    tree = Tree(tmp_path / 'results')
    os.makedirs(tree.root)
    write_figa(tree)
    write_figb(tree)
    # An older, unanchored run beside them is never picked, even if newer.
    write_figa(tree, _drop_anchor)
    out, table = tmp_path / 'macros.tex', tmp_path / 'table.tex'
    rc = MRM.main(['--results', tree.root, '--out', str(out),
                   '--table-out', str(table)])
    assert rc == 0
    text = out.read_text(encoding='utf-8')
    for macro in ('FigADir', 'FigACommit', 'RegretMedian', 'OracleRestarts',
                  'RegretFeasMovedScen', 'SpearmanMeanCF', 'FigBDir',
                  'GAvsSA', 'GAvsSACrossedCI', 'GAvsSAGOneCI',
                  'GAvsSAEquivMinMarginPct', 'FigBCFUnchanged',
                  'NComparators', 'FigAGapClosedMean', 'FigABelowStartRuns',
                  'FigBAnalyticalRegretSAPopStart', 'ReleaseVersion',
                  'RegSeparationRatePerS', 'RegConvClampHi'):
        assert f'\\newcommand{{\\{macro}}}' in text, macro
    assert 'synthetic\\_gt\\_20990101-000001' in text
    err = capsys.readouterr().err
    assert 'elasticity_lhs' in err and 'objective_alignment' in err
    assert 'Recovery' in table.read_text(encoding='utf-8')


def test_the_generator_fails_when_a_valid_run_has_an_unreadable_row(
        tmp_path, capsys, monkeypatch):
    """A Table 4 row that cannot be read from a run that set its family's
    macros is a bug: the row reads TBD, and the generator exits 1 after
    writing, as it does for a macro builder that fails."""
    monkeypatch.setattr(MRM, 'RES', MRM.RES)
    tree = Tree(tmp_path / 'results')
    os.makedirs(tree.root)
    write_figa(tree)
    write_figb(tree)

    def _broken(d):
        raise KeyError('renamed_summary_key')
    monkeypatch.setattr(MRM, 'TABLE_ROWS', tuple(
        (k, label, _broken if k == 'synthetic_gt' else b)
        for k, label, b in MRM.TABLE_ROWS))
    out, table = tmp_path / 'macros.tex', tmp_path / 'table.tex'
    rc = MRM.main(['--results', tree.root, '--out', str(out),
                   '--table-out', str(table)])
    assert rc == 1
    assert out.exists() and table.exists()
    row = next(line for line in table.read_text(encoding='utf-8').splitlines()
               if line.startswith('Recovery'))
    assert 'TBD' in row
    assert 'Table 4 row synthetic_gt' in capsys.readouterr().err
    # The same tree without the broken row passes.
    monkeypatch.undo()
    monkeypatch.setattr(MRM, 'RES', MRM.RES)
    assert MRM.main(['--results', tree.root, '--out', str(out),
                     '--table-out', str(table)]) == 0


def test_the_generator_refuses_a_tree_without_figures_a_and_b(
        tmp_path, monkeypatch):
    monkeypatch.setattr(MRM, 'RES', MRM.RES)
    os.makedirs(tmp_path / 'results')
    with pytest.raises(SystemExit):
        MRM.main(['--results', str(tmp_path / 'results'),
                  '--out', str(tmp_path / 'm.tex'),
                  '--table-out', str(tmp_path / 't.tex')])
    assert not (tmp_path / 'm.tex').exists()


# --- Holm counts, heterogeneity, the start on Figure B's basis (R48, R07) ------

def test_holm_rejections_stop_at_the_first_failure():
    # Sorted p 0.01, 0.03, 0.04 against 0.05/3, 0.05/2, 0.05/1: the second
    # fails, so the third is not rejected although it is below 0.05.
    assert MRM.holm_reject([0.01, 0.03, 0.04], 0.05).tolist() == \
        [True, False, False]
    assert MRM.holm_reject([0.04, 0.01, 0.02], 0.05).tolist() == \
        [True, True, True]
    assert not MRM.holm_reject([0.2, 0.3], 0.05).any()


def test_cochran_q_on_fisher_z():
    same = MRM.cochran_q_fisher_z([0.4] * 5, 100)
    assert same['Q'] == pytest.approx(0.0, abs=1e-12)
    assert same['df'] == 4 and same['p'] == pytest.approx(1.0)
    assert same['I2'] == 0.0
    # Two correlations 0.2 apart on the z scale, Spearman variance 1.06/97.
    q = MRM.cochran_q_fisher_z([0.0, np.tanh(0.2)], 100)
    assert q['Q'] == pytest.approx(97 / 1.06 * 0.02)
    assert q['df'] == 1
    assert MRM.p_bound(6.4e-24) == '\\ensuremath{<10^{-23}}'
    assert MRM.p_bound(1e-5) == '\\ensuremath{<10^{-4}}'
    assert MRM.p_bound(0.2) == MRM.pfmt(0.2)


def test_figure_a_reports_holm_counts_and_heterogeneity(tmp_path):
    """R48: the per-scenario Spearman tests Holm-corrected beside the
    uncorrected split, and Cochran's Q with its df, p and I^2."""
    side, summary, header, rows, _ = figa_parts()
    spearman = [[0, 0.5, 0.001, 0.5, 0.001], [1, 0.2, 0.03, 0.2, 0.03],
                [2, -0.2, 0.04, -0.2, 0.04]]
    d = Tree(tmp_path).run('synthetic_gt')
    _write(d, 'sidecar.json', side)
    _write(d, 'summary.json', summary)
    _csv(d, header, rows)
    _csv(d, ['scenario', 'spearman_rho', 'spearman_p', 'spearman_rho_cf',
             'spearman_p_cf'], spearman, name='spearman.csv')
    out = MRM.figa_macros(d)
    assert (out['SpearmanNPos'], out['SpearmanNNeg'], out['SpearmanNNull']) \
        == ('2', '1', '0')
    assert (out['SpearmanHolmNPos'], out['SpearmanHolmNNeg'],
            out['SpearmanHolmNNull']) == ('1', '0', '2')
    q = MRM.cochran_q_fisher_z([0.5, 0.2, -0.2],
                               int(side['args']['n_spearman_samples']))
    assert out['SpearmanCochranQ'] == f"{q['Q']:.1f}"
    assert out['SpearmanCochranQDf'] == '2'
    assert out['SpearmanCochranQP'] == MRM.pfmt(q['p'])
    assert out['SpearmanISquared'] == f"{100 * q['I2']:.0f}\\%"


def test_figure_a_start_regret_on_the_repaired_basis_and_no_tautologies(
        tmp_path):
    """R07: the as-built start against the reference mapped onto the
    feasible set (figa_parts: 99.5 against a start of 90); R36: no gain is
    reported after the last restart checkpoint, where it is zero by
    construction."""
    out = MRM.figa_macros(write_figa(Tree(tmp_path)))
    assert out['GridRegretFeasMean'] == f"{(99.5 - 90) / 99.5 * 100:.2f}\\%"
    assert out['GridRegretFeasMedian'] == out['GridRegretFeasMean']
    assert out['GridRegretMean'] == f"{(100 - 90) / 100 * 100:.2f}\\%"
    assert 'OracleGainAfterFortyEightMedian' in out
    assert not any(k.startswith('OracleGainAfterNinetySix') for k in out)


def test_as_built_regret_on_figure_bs_basis(tmp_path):
    """The start's analytical regret against each Figure B run's reference
    row, joined through Figure A's record of the start, only when both
    runs built the same scenarios."""
    tree = Tree(tmp_path)
    figa = write_figa(tree)
    side, summary, rows = figb_parts()
    for r in rows:
        if r[2] == 'oracle':
            r[5] = 99.5                      # Figure A's mapped reference
    figb = tree.run('baseline_comparison')
    _write(figb, 'sidecar.json', side)
    _write(figb, 'summary.json', summary)
    _csv(figb, ['scenario', 'seed', 'method', 'mc_revenue', 'cf_revenue',
                'analytical_R'], rows)
    out = MRM.asbuilt_regret_macros(figa, figb)
    assert out == {'FigBAnalyticalRegretAsBuilt':
                   f"{(99.5 - 90) / 99.5 * 100:.2f}\\%"}
    # write_figb's reference rows (100.0) are not Figure A's (99.5).
    with pytest.raises(ValueError):
        MRM.asbuilt_regret_macros(figa, write_figb(tree))


# --- the budget sweep in percent, transfer in points (R05, R12) -----------------

def test_budget_sweep_differences_in_percent_of_the_as_built_revenue(
        tmp_path):
    def mc(mean, lo, hi, ahead):
        return {'mc': {'mean': mean, 'ci_lo': lo, 'ci_hi': hi,
                       'excludes_zero': lo > 0 or hi < 0,
                       'seeds_ga_ahead': ahead}, 'n_seeds': 5}
    summary = {
        'design': {'budget_multipliers': [1], 'n_search_seeds': 5,
                   'sa_k_move': 10, 'final_evals': 150,
                   'search_evals_per_multiplier': {'1': 750}},
        'results': {
            'reference': {'method': 'simulated_annealing_k', 'multiplier': 1,
                          'lift_over_asbuilt_cf': 1.0,
                          'lift_over_asbuilt_pct': 1.0},
            'figc_lift': {'closed_form': 1.0, 'pct_cf': 1.0},
            'baseline': {'closed_form': 800_100.0,
                         'held_out_mc_mean': 800_000.0},
            'per_method': {},
            'closed_form_conclusions_unchanged': {
                'all_unchanged': True,
                'per_multiplier': {'1': {'ga_minus': {
                    'lift_over_asbuilt': mc(12_000.0, 11_000.0, 13_000.0, 5),
                    'random_search': mc(2_000.0, 640.0, 3_360.0, 5),
                    'simulated_annealing_k': mc(-3_040.0, -3_840.0,
                                                -2_240.0, 0)}}}}}}
    d = Tree(tmp_path).run('real_data_budget')
    _write(d, 'summary.json', summary)
    out = MRM.budget_macros(d)
    assert out['BudgetGAvsRandSearchOneMC'] == '+\\pounds 2,000'
    assert out['BudgetGAvsRandSearchOneMCPct'] == '+0.25\\%'
    assert out['BudgetGAvsRandSearchOneMCPctCI'] == '[+0.08\\%, +0.42\\%]'
    assert out['BudgetGAvsSAKOneMCPct'] == '-0.38\\%'
    assert out['BudgetGAvsSAKOneMCPctCI'] == '[-0.48\\%, -0.28\\%]'
    assert out['BudgetLiftOneMCPct'] == '+1.50\\%'


def _transfer_summary():
    def lift(g, p):
        return {'gbp': {'mean': g}, 'pct': {'mean': p, 'ci': [p - 0.1,
                                                               p + 0.1]},
                'mc_gbp': {'mean': g}}
    methods = {m: {'promised': lift(3.0, 2.7), 'transferred': lift(2.0, 1.9),
                   'hindsight': lift(2.5, 2.5),
                   'transfer_gap': {'gbp': {'mean': 500.0,
                                            'ci': [300.0, 700.0]},
                                    'pct': {'mean': 0.63,
                                            'ci': [0.39, 0.87]},
                                    'mc_gbp': {'mean': 500.0},
                                    'n_seeds_transferred_below_hindsight': 5},
                   'transfer_ratio_of_means': 0.8,
                   'n_seeds_transferred_positive': 5}
               for m in ('GA', 'random_search', 'simulated_annealing')}
    vs = {'mean': 100.0, 'ci': [50.0, 150.0], 'n_ga_leads': 4}
    period = {'first_date': '2010-01-01', 'last_date': '2010-12-01'}
    return {'across_seeds': {'n_seeds': 5, 'level': 0.95, 'methods': methods,
                             'transferred_ga_vs': {
                                 'GA_minus_random_search': dict(vs),
                                 'GA_minus_simulated_annealing': dict(vs)}},
            'fixtures': {'n_fixtures': 108, 'n_absent_in_current': 3,
                         'current_top_n_size': 108,
                         'current_top_n_not_stocked': 20},
            'design': {'periods': {'prior': period, 'current': period},
                       'n_invoices_calibrated': {'prior': 10, 'current': 20}},
            'baseline': {'closed_form': {'prior': 1.0, 'current': 2.0}}}


def test_transfer_gap_between_percentages_is_in_points(tmp_path):
    d = Tree(tmp_path).run('heldout_transfer')
    _write(d, 'summary.json', _transfer_summary())
    out = MRM.transfer_macros(d)
    assert out['TransferGAGapPct'] == '+0.63\\,pp'
    assert out['TransferGAGapPctCI'] == '[+0.39, +0.87]\\,pp'
    # The lifts themselves stay percentages.
    assert out['TransferGAPromisedPct'] == '+2.70\\%'
    assert MRM.pp(-1.849, 1) == '-1.8\\,pp'


# --- Table 4: one store, one item count (R50) ------------------------------------

def _live_store_data(**change):
    base = {'max_items_per_category': 8, 'naive_layout': True,
            'reader_version': '1.1'}
    base.update(change)
    return live_data(**base)


def _write_summary(tree, prefix, summary):
    d = tree.run(prefix)
    _write(d, 'summary.json', summary)
    return d


def _proto_summary():
    return {'design': {'nominal': {'grid': [0.25, 0.26], 'reps': 16,
                                   'run_s': 3600.0},
                       'stress': {'grid': [0.3, 0.7], 'cap_grid': [90, 100],
                                  'rate_reps': 6, 'cap_reps': 8,
                                  'run_s': 1800.0},
                       'dt': 0.04, 'n_runs': 32 + 16 + 12 + 16},
            'double_warmup': {'reps': 16, 'run_s': 3900.0},
            'stress': {'cap_scan': {'90': {}, '100': {}}},
            'data': _live_store_data()}


def test_live_rows_read_the_item_count_of_their_store(tmp_path):
    """The live runners share one store; those that do not record its size
    read it from a run that does, for the same store field for field."""
    tree = Tree(tmp_path)
    gof = _write_summary(tree, 'validation_gof', {'design': {
        'in_sample': {'store_period': 'current', 'n_placed_products': 72},
        'periods': {'current': _live_store_data()}}})
    heat = _write_summary(tree, 'heatmap_figure', {
        'store': {'n_items': 72}, 'data': _live_store_data()})
    items = MRM.live_store_items({'validation_gof': gof,
                                  'heatmap_figure': heat})
    assert list(items.values()) == [72]
    struct = _write_summary(tree, 'structural_sensitivity', {
        'strengths': [0.0, 0.08], 'n_reps': 5,
        'protocol': {'warmup_s': 1020.0, 'collect_s': 1860.0},
        'data': _live_store_data()})
    assert MRM._row_struct(struct, items)[3] == '72'
    other = _write_summary(tree, 'structural_sensitivity', {
        'strengths': [0.0, 0.08], 'n_reps': 5,
        'protocol': {'warmup_s': 1020.0, 'collect_s': 1860.0},
        'data': _live_store_data(n_invoices=123)})
    assert MRM._row_struct(other, items)[3] == '--'
    # Two runs that disagree on one store's size settle nothing.
    heat2 = _write_summary(tree, 'heatmap_figure', {
        'store': {'n_items': 71}, 'data': _live_store_data()})
    assert MRM.live_store_items({'validation_gof': gof,
                                 'heatmap_figure': heat2}) == {}
    proto = _write_summary(tree, 'live_protocol_study', _proto_summary())
    row = MRM._row_proto(proto, items)
    assert row[0] == ('2 rates $\\times$ 16 reps $\\times$ 3,600 s; double '
                      'warm-up 16 reps $\\times$ 3,900 s; stress 2 rates '
                      '$\\times$ 6 reps and 2 caps $\\times$ 8 reps, 1,800 s '
                      'each')
    assert row[3] == '72'
    table = MRM.budget_table({'validation_gof': None, 'heatmap_figure': heat,
                              'structural_sensitivity': struct,
                              'live_protocol_study': proto})
    for label in ('Structural sweep', 'Live protocol study'):
        line = next(x for x in table.splitlines() if x.startswith(label))
        assert line.endswith('& 72 \\\\'), line


# --- the structural sweep's conversion and its abandonment draw -------------------

def test_structural_conversion_against_the_abandonment_draw():
    """Per setting: conversion, the abandoned and exit counts, and what the
    spawn-time draw alone predicts (mean of AGENT_ABANDON_PROB_RANGE times
    the exits); the settings' paid / unpaid split tested for homogeneity;
    the conversion contrasts in points; the exit routes when recorded."""
    lo, hi = MRM._script_constants(MRM.LIT_SCRIPT,
                                   ('AGENT_ABANDON_PROB_RANGE',))[
        'AGENT_ABANDON_PROB_RANGE']
    pbar = (lo + hi) / 2
    routes = {'unpaid_abandon_draw': 30, 'unpaid_no_route': 0,
              'moving_stalls': 4, 'exit_stall_teleports': 0}
    s = {'default_strength': 0.08,
         'rows': [{'strength': 0.0, 'completed': 970, 'abandoned': 30,
                   'exit_routes': routes},
                  {'strength': 0.08, 'completed': 980, 'abandoned': 20}],
         'paired_contrasts': {'conversion': {'0.0': {
             'mean_diff': -0.01, 'ci95_half_width': 0.005, 'p': 0.03}}}}
    out = MRM.struct_conversion_macros(s)
    sd = np.sqrt(1000 * pbar * (1 - pbar))
    assert out['StructAbandonDrawMean'] == f"{100 * pbar:.1f}\\%"
    assert out['StructConvOff'] == '97.0\\%'
    assert out['StructConvDefault'] == '98.0\\%'
    assert out['StructAbandonedOff'] == '30'
    assert out['StructExitsDefault'] == '1,000'
    assert out['StructAbandonedExpectedOff'] == f"{1000 * pbar:.0f}"
    assert out['StructAbandonedSDOff'] == f"{sd:.0f}"
    assert out['StructAbandonedZDefault'] == \
        f"{(20 - 1000 * pbar) / sd:+.1f}"
    assert out['StructAbandonedPooledCount'] == '50/2,000'
    x2 = 2 * (5 ** 2 / 25) + 2 * (5 ** 2 / 975)
    assert out['StructAbandonedChi'] == f"{x2:.1f}"
    assert out['StructAbandonedChiDf'] == '1'
    assert out['StructPairOffConvPP'] == '-1.0\\,pp'
    assert out['StructPairOffConvCIPP'] == '0.5\\,pp'
    assert out['StructPairConvNSig'] == '1/1'
    assert out['StructUnpaidDrawOff'] == '30'
    assert out['StructUnpaidNoRouteOff'] == '0'
    assert out['StructMovingStallsOff'] == '4'
    assert 'StructUnpaidDrawDefault' not in out     # not recorded there


def _check_run(strength, seed, rep, unpaid, stalled=0):
    return {'strength': strength, 'rep': rep, 'seed': seed,
            'matches_artifact': True, 'tally_agrees': True,
            'n_exits': 350, 'paid': 350 - unpaid, 'unpaid': unpaid,
            'unpaid_by_route': {'abandon_draw': unpaid, 'no_route': 0,
                                'other': 0},
            'abandon_drawn': unpaid, 'unpaid_without_draw': 0,
            'drawn_but_paid': 0, 'moving_stalls': stalled,
            'stalled_exits': stalled, 'stalled_unpaid': 0,
            'stalled_unpaid_drawn': 0, 'exit_stall_teleports': 0,
            'relocations': 0, 'expected_unpaid': 10.5, 'var_unpaid': 10.2}


def write_struct_check(tree, struct_dir, mutate=None):
    """A structural exit-route check as run_structural_exit_check writes
    it: two replications of two settings, re-run from ``struct_dir``."""
    import hashlib
    from experiments import run_structural_exit_check as X
    with open(os.path.join(struct_dir, 'summary.json'), 'rb') as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    runs = [_check_run(0.0, 2000, 0, 13), _check_run(0.0, 2001, 1, 16),
            _check_run(0.08, 2000, 0, 8), _check_run(0.08, 2001, 1, 6, 2)]
    per = {'0.0': {'n_exits': 700, 'n_exits_reference': 700,
                   'n_common': 676, 'n_same_draw': 6,
                   'n_same_draw_abandoning': 0, 'n_both_abandon': 1,
                   'n_either_abandon': 39, 'n_spawn_time_ties': 0}}
    summary = {'experiment': X.EXPERIMENT,
               'struct': {'run_name': os.path.basename(struct_dir),
                          'summary_sha256': digest, 'n_reps': 5,
                          'strengths': [0.0, 0.08], 'default_strength': 0.08,
                          'unpaid_total': 114, 'exits_total': 3500},
               'reps_checked': [0, 1], 'seeds_checked': [2000, 2001],
               'runs': runs,
               'per_setting': {'0.0': {'stalled_exits': 0},
                               '0.08': {'stalled_exits': 2}},
               'total': {'n_runs': 4, 'n_exits': 1400, 'paid': 1357,
                         'unpaid': 43, 'abandon_drawn': 43,
                         'unpaid_without_draw': 0, 'drawn_but_paid': 0,
                         'moving_stalls': 2, 'stalled_exits': 2,
                         'stalled_unpaid': 0, 'stalled_unpaid_drawn': 0,
                         'exit_stall_teleports': 0, 'relocations': 0,
                         'unpaid_by_route': {'abandon_draw': 43,
                                             'no_route': 0, 'other': 0},
                         'expected_unpaid': 42.0, 'sd_unpaid': 6.4,
                         'z_unpaid': 1 / 6.4, 'all_reproduce': True,
                         'all_tallies_agree': True},
               'sharing': {'reference_strength': 0.08, 'pairs': [],
                           'per_setting': per,
                           'total': {'n_common': 676, 'n_same_draw': 6}}}
    side = _clean()
    if mutate:
        mutate(side, summary, None)
    d = tree.run('structural_exit_check')
    _write(d, 'sidecar.json', side)
    _write(d, 'summary.json', summary)
    return d


def test_structural_exit_check_sets_its_numbers(tmp_path):
    tree = Tree(tmp_path)
    struct = write_struct(tree)
    d = write_struct_check(tree, struct)
    assert MRM._struct_check_big_enough(d)
    assert MRM._struct_check_matches(d, struct)
    out = MRM.struct_check_macros(d)
    assert out['StructCheckRuns'] == '4'
    assert out['StructCheckReps'] == '2'
    assert out['StructCheckReproduced'] == '4/4'
    assert out['StructCheckExits'] == '1,400'
    # Of the sweep's unpaid exits, and by route over the checked ones.
    assert out['StructCheckUnpaidOfSweep'] == '43/114'
    assert out['StructCheckUnpaidDraw'] == '43/43'
    assert out['StructCheckUnpaidNoRoute'] == '0'
    assert out['StructCheckUnpaidWithoutDraw'] == '0'
    assert out['StructCheckDrawnPaid'] == '0'
    assert out['StructCheckStalledExits'] == '2'
    assert out['StructCheckStalledExitsDefault'] == '2'
    assert out['StructCheckStalledExitsOff'] == '0'
    assert out['StructCheckExpectedUnpaid'] == '42.0'
    assert out['StructCheckUnpaidZ'] == '+0.2'
    assert out['StructCheckCommonOff'] == '676'
    assert out['StructCheckSameDrawOff'] == '6/676'
    assert out['StructCheckSameDrawTotal'] == '6/676'
    assert out['StructCheckSameDrawAbandoning'] == '0'
    # A check of another sweep, or of this one before it changed, does not
    # describe the sweep quoted.
    assert not MRM._struct_check_matches(d, None)
    assert not MRM._struct_check_matches(d, write_struct(tree))
    with open(os.path.join(struct, 'summary.json'), 'a') as f:
        f.write(' ')
    assert not MRM._struct_check_matches(d, struct)


@pytest.mark.parametrize('mutate', [
    _dirty,
    lambda s, m, h: m['runs'][0].update(matches_artifact=False),
    lambda s, m, h: m['total'].update(all_reproduce=False),
    lambda s, m, h: m['runs'][1].update(tally_agrees=False),
    lambda s, m, h: m.update(reps_checked=[0],
                             runs=m['runs'][::2]),     # one replication
    lambda s, m, h: m.update(runs=m['runs'][:3]),       # a run missing
    lambda s, m, h: m.update(experiment='other'),
])
def test_structural_exit_check_refuses_an_incomplete_check(tmp_path, mutate):
    tree = Tree(tmp_path)
    d = write_struct_check(tree, write_struct(tree), mutate)
    assert not MRM._struct_check_big_enough(d)


# --- Figure C re-scored in closed form; the layout figure (R06) --------------------

def _summary_digest(figc_dir):
    import hashlib
    with open(os.path.join(figc_dir, 'summary.json'), 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def _rescore_parts(figc_dir):
    lifts = {'optimized': 1.68, 'rs': 1.39, 'sa': 1.19}
    per = {k: {'lift_pct': v + 0.13, 'lift_pct_no_abandonment_channel': v,
               'abandonment_share': 0.2 + 0.01 * i, 'abandonment_pp': 0.36,
               'lift_pct_no_conversion_elasticity': 0.87,
               'conversion_elasticity_share': 0.52,
               'lift_pct_no_basket_elasticity': 1.29,
               'basket_elasticity_share': 0.29, 'score_gain': 0.0266,
               'criterion_contributions': {'flow': 0.0189,
                                           'cross_merch': 0.0085},
               'flow_share_of_score_gain': 0.71 + 0.08 * i}
           for i, (k, v) in enumerate(lifts.items())}
    summary = {
        'figc': {'run_name': os.path.basename(figc_dir),
                 'summary_sha256': _summary_digest(figc_dir),
                 'values_reproduced': True},
        'conversion_bound': {'value': 0.538, 'family': 'Fresh fruits',
                             'n_families': 134,
                             'provenance': {'source_sha256': 'ab' * 32}},
        'lift_at_conversion_bound': {
            'lift_pct': lifts, 'clamps_bind': False,
            'lift_pct_at_assumed': {'optimized': 1.81, 'rs': 1.52,
                                    'sa': 1.28},
            'scan_bracket': {'below': {'rate': 0.5, 'lift_pct':
                                       {'optimized': 1.70}},
                             'above': {'rate': 0.55, 'lift_pct':
                                       {'optimized': 1.678}}}},
        'decomposition': {'per_layout': per}}
    return _clean(base_params=_bp()), summary


def test_figure_c_rescoring_is_held_to_the_figure_c_run(tmp_path):
    tree = Tree(tmp_path)
    figc = write_figc(tree)
    side, summary = _rescore_parts(figc)
    d = tree.run('figc_rescore')
    _write(d, 'sidecar.json', side)
    _write(d, 'summary.json', summary)
    assert MRM._rescore_big_enough(d)
    assert MRM._rescore_matches_figc(d, figc)
    out = MRM.rescore_macros(d)
    assert out['FigCConvBound'] == '0.538'
    assert out['FigCConvBoundFamily'] == 'Fresh fruits'
    assert out['FigCLiftAtConvBound'] == '+1.68\\%'
    assert out['FigCLiftAtConvBoundSA'] == '+1.19\\%'
    assert out['FigCLiftAtConvBoundChange'] == '-0.13\\,pp'
    assert out['FigCConvBoundScanRateAbove'] == '0.55'
    assert out['FigCConvBoundScanLiftAbove'] == '+1.68\\%'
    assert out['FigCLiftAbandonShare'] == '20\\%'
    assert out['FigCLiftAbandonShareMax'] == '22\\%'
    assert out['FigCScoreGainFlowShare'] == '71\\%'
    assert out['FigCScoreGainFlowShareMax'] == '87\\%'
    assert out['FigCLiftNoAbandon'] == '+1.68\\%'
    # A re-run of Figure C (its summary changed) orphans the re-scoring.
    with open(os.path.join(figc, 'summary.json'), 'a') as f:
        f.write(' ')
    assert not MRM._rescore_matches_figc(d, figc)
    assert not MRM._rescore_matches_figc(d, None)
    for mutate in (_dirty, _drop_anchor):
        s2, m2 = _rescore_parts(figc)
        mutate(s2, m2, None)
        d2 = tree.run('figc_rescore')
        _write(d2, 'sidecar.json', s2)
        _write(d2, 'summary.json', m2)
        assert not MRM._rescore_big_enough(d2)


def _layout_record(figc_dir, **change):
    inv = {'n_outside_zone': 0, 'n_overlap': 0, 'n_clearance': 0,
           'n_unshoppable': 0, 'min_clearance_kept_apart_m': 1.6,
           'required_clearance_m': 1.6}
    rec = {'figure': MRM.LAYOUT_FIGURE_STEM,
           'source': {'figc_dir': 'CODE/experiments/results/'
                                  + os.path.basename(figc_dir),
                      'summary_sha256': _summary_digest(figc_dir)},
           'store': {'n_items': 108, 'n_zones': 15},
           'moves': {'n_fixtures': 108, 'n_moved': 107,
                     'n_moved_at_least_aisle': 53, 'n_slot_exchange': 41,
                     'n_along_run': 43, 'slot_tol_m': 0.3,
                     'along_run_tol_m': 0.5, 'highlight_m': 1.6,
                     'median_m': 0.8, 'max_m': 22.57},
           'flow_tour': {'n_items': 5,
                         'length_m': {'baseline': 128.34,
                                      'optimized': 85.49},
                         'matches_ga_flow_score': True},
           'invariants': {'baseline': dict(inv), 'optimized': dict(inv)},
           'invariants_hold': True, 'min_clearance_m': 1.6,
           'required_clearance_m': 1.6,
           'provenance': {'git_sha': SHA, 'git_dirty': False,
                          'git_state_changed_during_run': False}}
    rec.update(change)
    return rec


def test_layout_figure_record_sets_its_caption_numbers(tmp_path):
    figc = write_figc(Tree(tmp_path))
    rec = _layout_record(figc)
    assert MRM._layout_figure_ok(rec, figc)
    out = MRM.layout_figure_macros(rec)
    assert out['FigCLayoutMoved'] == '107/108'
    assert out['FigCLayoutMovedAtLeastAisle'] == '53/108'
    assert 'FigCLayoutMovedPastAisle' not in out
    # The long moves' kinds are shares of the long moves, not of the store.
    assert out['FigCLayoutMovedSlotExchange'] == '41/53'
    assert out['FigCLayoutMovedAlongRun'] == '43/53'
    assert out['FigCLayoutSlotTol'] == '0.3'
    assert out['FigCLayoutAlongRunTol'] == '0.5'
    assert out['FigCLayoutTourItems'] == '5'
    assert out['FigCLayoutTourAsBuilt'] == '128'
    assert out['FigCLayoutTourGA'] == '85'
    assert out['FigCLayoutHighlight'] == '1.6'
    assert out['FigCLayoutMinClearance'] == '1.60'
    assert out['FigCLayoutUnshoppable'] == '0'
    assert out['FigCLayoutInvariantsHold'] == 'yes'
    assert out['FigCLayoutMoveMax'] == '22.6'
    assert not MRM._layout_figure_ok(rec, None)
    assert not MRM._layout_figure_ok(
        _layout_record(figc, invariants_hold=False), figc)
    assert not MRM._layout_figure_ok(_layout_record(
        figc, provenance={'git_sha': SHA, 'git_dirty': True,
                          'git_state_changed_during_run': False}), figc)
    other = _layout_record(figc)
    other['source']['summary_sha256'] = '0' * 64
    assert not MRM._layout_figure_ok(other, figc)
    # A record from before the long moves were classified, or whose tour
    # was not checked against the GA's flow score, is not today's figure.
    old = _layout_record(figc)
    old['moves'] = {k: v for k, v in old['moves'].items()
                    if k not in ('n_slot_exchange', 'n_along_run')}
    assert not MRM._layout_figure_ok(old, figc)
    unchecked = _layout_record(figc)
    unchecked['flow_tour']['matches_ga_flow_score'] = False
    assert not MRM._layout_figure_ok(unchecked, figc)
    assert not MRM._layout_figure_ok(
        {k: v for k, v in _layout_record(figc).items() if k != 'flow_tour'},
        figc)


def test_summary_digest_survives_line_ending_conversion(tmp_path):
    """A recorded digest of a summary a runner wrote with CRLF still
    matches after git checks the file out with LF, and the reverse."""
    import hashlib
    body = b'{\n  "a": 1,\n  "b": [2, 3]\n}\n'
    crlf = body.replace(b'\n', b'\r\n')
    for written, checked_out in ((crlf, body), (body, crlf), (body, body)):
        p = tmp_path / 'summary.json'
        p.write_bytes(checked_out)
        recorded = hashlib.sha256(written).hexdigest()
        assert recorded in MRM._summary_digests(str(p))
    p.write_bytes(body.replace(b'1', b'9'))
    assert hashlib.sha256(body).hexdigest() not in MRM._summary_digests(str(p))
