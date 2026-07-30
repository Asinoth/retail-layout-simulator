"""Generate paper_results_macros.tex from the newest experiment artifacts.

Scans CODE/experiments/results/ for the latest run of each kind and emits
LaTeX \\newcommand macros consumed by TOMACS_submission.tex, so the paper
recompiles with fresh numbers after every re-run -- no hand-editing.

    python make_results_macros.py            # writes ../paper_results_macros.tex
"""

from __future__ import annotations

import csv
import glob
import json
import os
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
            and int(j.get('n_seeds_per_scenario', 0)) >= MIN_SEEDS)


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


def fisher_mean_rho(rhos):
    """Average correlations via the Fisher z-transform (audit R3.4)."""
    r = np.clip(np.asarray(rhos, dtype=float), -0.999, 0.999)
    z = np.arctanh(r)
    return float(np.tanh(z.mean()))


def money(v):
    s = f"{abs(v):,.0f}"
    return ("-" if v < 0 else "+") + "\\$" + s


macros = {}

# Refuse to overwrite a good macros file from an artifact-less tree
# (audit R15.1): on a clean checkout with experiments/results/ empty,
# regenerating would silently replace the shipped numbers with TBD
# fallbacks. The core experiment artifacts must be present.
if (latest('synthetic_gt_', _sidecar_big_enough) is None
        or latest('baseline_comparison_', _sidecar_big_enough) is None):
    import sys
    sys.stderr.write(
        "make_results_macros: no synthetic_gt_*/baseline_comparison_* "
        "artifacts under experiments/results/ -- refusing to overwrite "
        f"{OUT}.\nEither run the experiments (make experiments-paper) or "
        "keep the shipped paper_results_macros.tex, which was generated "
        "from the released artifact files.\n")
    sys.exit(1)

# -- Figure A: synthetic ground truth -------------------------------------
d = latest('synthetic_gt_', _sidecar_big_enough)
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
    sp = os.path.join(d, 'spearman.csv')
    if os.path.exists(sp):
        rho = np.array([float(r['spearman_rho']) for r in read_csv(sp)])
        # Fisher-z averaged mean (audit R3.4); report distribution, not
        # just a point, and the power to detect rho=0.3 at n=100 layouts.
        n_lay = 100
        se_z = 1.0 / np.sqrt(n_lay - 3)
        # two-sided 0.05 detectable |rho| at 80% power:
        z_det = (1.96 + 0.84) * se_z
        rho_det = np.tanh(z_det)
        macros.update({
            'SpearmanMean': f"{fisher_mean_rho(rho):+.3f}",
            'SpearmanMedian': f"{np.median(rho):+.3f}",
            'SpearmanMax': f"{np.max(rho):+.3f}",
            'SpearmanMin': f"{np.min(rho):+.3f}",
            'SpearmanDetect': f"{rho_det:.2f}",
        })
    macros['FigADir'] = os.path.basename(d).replace('_', '\\_')

# -- Figure B: baseline comparison (paired differences) -------------------
# Cluster (scenario-level) bootstrap + Bonferroni over the comparator
# family (audits R3.1, R3.2). The divisor is derived from the data, not
# hardcoded -- it was 5 before the two equal-budget metaheuristics were
# added and is 7 now; ``NComparators`` records whatever it actually was.
d = latest('baseline_comparison_', _sidecar_big_enough)
if d:
    rows = read_csv(os.path.join(d, 'results.csv'))
    by = defaultdict(dict)      # (scenario, seed) -> {method: revenue}
    for r in rows:
        by[(r['scenario'], r['seed'])][r['method']] = float(r['mc_revenue'])
    names = {'oracle': 'Oracle', 'popularity_rank': 'Pop',
             'perimeter_only': 'Perim', 'random_valid': 'Random',
             'greedy_swap': 'Greedy', 'random_search': 'RandSearch',
             'simulated_annealing': 'SA'}
    n_comparisons = len(names)
    alpha_bonf = 0.05 / n_comparisons
    npairs, nscen = 0, 0
    for meth, short in names.items():
        # group paired differences BY SCENARIO for the cluster bootstrap
        clusters = defaultdict(list)
        for (scen, seed), v in by.items():
            if 'GA' in v and meth in v:
                clusters[scen].append(v['GA'] - v[meth])
        clustered = list(clusters.values())
        if not clustered:
            continue
        npairs = max(npairs, sum(len(c) for c in clustered))
        nscen = max(nscen, len(clustered))
        mean, lo, hi = cluster_boot_ci(clustered, alpha=alpha_bonf)
        macros[f'GAvs{short}'] = money(mean)
        macros[f'GAvs{short}CI'] = f"[{money(lo)}, {money(hi)}]"
        macros[f'GAvs{short}Sig'] = 'yes' if (lo > 0 or hi < 0) else 'no'
    per_comp_level = 100 * (1 - alpha_bonf)
    macros['PairedN'] = str(npairs)
    macros['PairedNScen'] = str(nscen)
    macros['NComparators'] = str(n_comparisons)
    macros['BonfLevel'] = (f"{per_comp_level:.0f}\\%"
                           if abs(per_comp_level - round(per_comp_level)) < 0.05
                           else f"{per_comp_level:.1f}\\%")
    macros['CIscheme'] = (f"scenario-level cluster bootstrap "
                          f"({N_BOOT} resamples), Bonferroni-corrected "
                          f"{macros['BonfLevel']} per-comparison")
    macros['FigBDir'] = os.path.basename(d).replace('_', '\\_')

# -- LHS elasticity robustness --------------------------------------------
d = latest('elasticity_lhs_', _has_summary)
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
    # replicate each point and must not be quoted as coverage. Older
    # artifacts predate the field, so derive it when absent.
    n_pts = s.get('n_lhs_points')
    if n_pts is None and s.get('n_scenarios') and s.get('n_seeds'):
        n_pts = int(s['n_draws_total']) // (int(s['n_scenarios'])
                                            * int(s['n_seeds']))
    if n_pts is None:
        n_pts = int(s['n_draws_total']) // 36   # shipped 12 x 3 design
    macros['LhsPoints'] = str(int(n_pts))

# -- Figure C: real-data worked example ----------------------------------
d = latest('real_data_uci_', _has_results_csv)
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
    # Data-quality descriptors from the calibration sidecar (audit R7.3/R7.5).
    sc = os.path.join(d, 'sidecar.json')
    if os.path.exists(sc):
        cs = json.load(open(sc)).get('calibration_summary', {})
        if 'basket_units_median' in cs:
            macros['BasketUnitsMed'] = f"{cs['basket_units_median']:.0f}"
            macros['BasketDistinctMed'] = f"{cs['basket_distinct_median']:.0f}"
        if 'category_fallback_frac' in cs:
            macros['CatFallbackFrac'] = f"{cs['category_fallback_frac']*100:.0f}\\%"
        if 'return_customer_rate' in cs:
            macros['ReturnRate'] = f"{cs['return_customer_rate']*100:.0f}\\%"

# -- Realized elasticities at realized scores (audit R2.3) ---------------
rpath = os.path.join(ROOT, 'figs', 'realized_scores.json')
if os.path.exists(rpath):
    rs = json.load(open(rpath))
    macros.update({
        'RealizedScoreBaseline': f"{rs['score_baseline']:.3f}",
        'RealizedScoreGA': f"{rs['score_ga']:.3f}",
        'RealizedConvLift': f"{rs['conv_lift_pct']:+.1f}\\%",
        'RealizedImpLift': f"{rs['impulse_lift_pct']:+.1f}\\%",
        'RealizedBskLift': f"{rs['basket_lift_pct']:+.1f}\\%",
    })
    # MC estimator convergence (audit R3.8): relative SE of the mean at the
    # 2000 iterations actually used.
    if 'mc_rse_at_2000_pct' in rs:
        macros['McRSEatUsed'] = f"{rs['mc_rse_at_2000_pct']:.2f}\\%"

# -- Queue measurement (audit R1.7) --------------------------------------
qpath = os.path.join(ROOT, 'figs', 'queue_summary.json')
if os.path.exists(qpath):
    q = json.load(open(qpath))
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

# -- MC-objective ground truth (audit R4.3) ------------------------------
d = latest('mc_groundtruth_', _has_summary)
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

# -- GA hyperparameter sensitivity + diversity (audit R4.5) --------------
sp = os.path.join(ROOT, 'figs', 'ga_sensitivity.json')
if os.path.exists(sp):
    s = json.load(open(sp))
    macros.update({
        'GAHyperSpread': f"{s['hyper_spread_median_pct']:.2f}\\%",
        'GAHyperSpreadMax': f"{s['hyper_spread_max_pct']:.2f}\\%",
        'GADefaultGap': f"{s['default_gap_median_pct']:.2f}\\%",
        'GANSettings': str(s['n_settings']),
        'GADivStart': f"{s['diversity_start']*100:.1f}\\%",
        'GADivEnd': f"{s['diversity_end']*100:.1f}\\%",
    })

# -- ABM diagnostics: Markov order + emergence (audits R6.2, R6.4) -------
d = latest('abm_diagnostics_', _has_summary)
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
    if em.get('perimeter_interior_ratio') is not None:
        macros['PerimRatio'] = f"{em['perimeter_interior_ratio']:.2f}"
        if 'perimeter_ratio_ci95' in em:
            macros['PerimRatioCI'] = f"{em['perimeter_ratio_ci95']:.2f}"
    proto = s.get('protocol', {})
    if proto:
        macros['AbmReps'] = str(proto['reps'])
        macros['AbmWarmup'] = f"{proto['warmup_s']:.0f}"

# -- Structural (micro-rule) sensitivity (audit R6.5) --------------------
d = latest('structural_sensitivity_', _has_summary)
if d and os.path.exists(os.path.join(d, 'summary.json')):
    s = json.load(open(os.path.join(d, 'summary.json')))
    if 'throughput_range_pct' in s:
        macros['StructRangePct'] = f"{s['throughput_range_pct']:.1f}\\%"
        macros['StructChiP'] = f"{s['chi2_p']:.2f}"
    elif 'conversion_range_pp' in s:   # older artifact schema
        macros['StructRangePP'] = f"{s['conversion_range_pp']:.1f}"
        macros['StructChiP'] = f"{s['chi2_p']:.2f}"
    # Basket-value response (audit R27). Throughput is flat across the
    # sweep, but revenue per completing customer is not: strong
    # separation pushes agents off the shelves, so they reach fewer
    # items. Report it rather than let the throughput null stand in for
    # every outcome.
    # Power of the homogeneity test at the observed totals (audit R30).
    # A null result is only informative if the design could have seen the
    # effect; computed here from the stored counts so it tracks re-runs.
    comp_counts = [int(r['completed']) for r in s.get('rows', [])]
    if comp_counts:
        from scipy.stats import chisquare as _chi2
        n_tot, k_set = sum(comp_counts), len(comp_counts)
        rng_p = np.random.default_rng(0)
        for eff, key in ((0.30, 'StructPowerThirty'),
                         (0.40, 'StructPowerForty')):
            p_vec = np.ones(k_set) / k_set
            p_vec[-1] *= (1.0 - eff)
            p_vec = p_vec / p_vec.sum()
            hits = sum(
                1 for _ in range(2000)
                if _chi2(rng_p.multinomial(n_tot, p_vec)).pvalue < 0.05)
            macros[key] = f"{100.0 * hits / 2000:.0f}\\%"

    # Per-customer revenue across settings, judged against replication
    # noise. An earlier UNREPLICATED sweep appeared to show a large
    # monotone decline; it did not survive a second run (audit R73), so
    # what is reported now is the between-setting spread next to the
    # within-setting spread, plus a one-way ANOVA.
    if s.get('n_reps'):
        m = np.array(s['rev_per_cust_mean'], dtype=float)
        sd = np.array(s['rev_per_cust_sd'], dtype=float)
        macros['StructReps'] = str(int(s['n_reps']))
        macros['StructRevRange'] = f"{m.max() - m.min():.2f}"
        macros['StructRevNoiseSD'] = f"{sd.mean():.2f}"
        if s.get('rev_per_cust_anova_p') is not None:
            macros['StructRevAnovaP'] = f"{float(s['rev_per_cust_anova_p']):.2f}"
        macros['StructNCompletions'] = str(
            int(sum(s.get('completed_per_setting', []))))

macros.setdefault('ResultsGrade', 'current artifacts')

with open(OUT, 'w', encoding='utf-8') as f:
    f.write('% AUTO-GENERATED by CODE/make_results_macros.py — do not edit.\n')
    for k, v in sorted(macros.items()):
        f.write(f'\\newcommand{{\\{k}}}{{{v}}}\n')
print(f'wrote {OUT} with {len(macros)} macros')
for k in sorted(macros):
    print(f'  \\{k} = {macros[k]}')
