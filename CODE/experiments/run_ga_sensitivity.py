"""Operator sensitivity of the GA and of the annealer, at a fixed budget.

The GA's operator settings need justifying rather than asserting, and the
GA-SA equivalence of Figure B holds for the annealer as it was configured,
not for annealing in general. Three artifacts answer this:

  1. GA sweep: mutation rate x (population, generations). The population
     and the generations move together so that every setting spends the
     same number of fitness evaluations -- ``--pop-size`` x (``--n-gens`` +
     ``GA_N_FINAL_SEEDS``), the default setting's search plus final
     selection -- and a setting cannot look better only for having been
     given more of them. Each run's returned layout is re-scored on a block
     of held-out MC seeds shared by every setting of a scenario (paired).
     Every per-setting mean and the SD within each setting (over seeds) are
     archived. Every setting of a (scenario, seed) runs from the same search
     seed and is re-scored on the same held-out block -- common random
     numbers, so the settings' means move together -- and whether they
     differ at all is judged against the noise left once those seed blocks
     are removed: the spread of the k setting means is set beside the range
     k means would show from that noise alone (the expected range of k
     standard normals times the blocked residual SD over sqrt(seeds); the
     independent-draws range is kept beside it), and tested by an ANOVA with
     the seeds of each scenario as blocks, the setting effect against the
     residual (scenarios fixed) and against the scenario x setting
     interaction (scenarios random).

  2. Annealer sweep: step fraction x final-temperature fraction, plus the
     initial acceptance target at the default step and final temperature,
     at the budget and on the scenarios and seeds of the GA sweep, from the
     as-built start. Each setting's layout is re-scored on the same held-out
     seeds as the GA's default setting. Reported per setting: the
     annealer's own sensitivity, that setting minus the default annealer
     (paired, with its interval); and the GA-minus-SA paired difference with
     the Figure B equivalence test (the margin of
     ``experiments._inference.EQUIV_MARGIN_FRAC`` of SA's mean, two
     one-sided tests at Figure B's per-comparison level) and the smallest
     margin at which it would hold. The sweep has far fewer (scenario, seed)
     pairs than Figure B, so the default annealer's own result is recorded
     as the power reference: a setting without established equivalence may
     be one this sample cannot resolve.

  3. A population-diversity trace of the default GA setting (mean
     per-coordinate spread, normalized by the shop diagonal, per
     generation). A nonzero plateau -- rather than a collapse to ~0 --
     shows the search retains exploratory spread, so the fitness plateau
     reflects convergence, not diversity loss.

Writes ../figs/ga_diversity.{pdf,png} and ../figs/ga_sensitivity.json (both
under ``--figs-dir``), a run directory under ``--out-root`` with
``settings.csv`` (every per-(scenario, setting, seed) held-out fitness), the
sidecar and ``summary.json``, and prints macro-ready summary numbers.

    python -m experiments.run_ga_sensitivity --n-scenarios 2 --n-seeds 2 \
        --n-gens 4 --pop-size 8 --mc-iters 100 --n-heldout-seeds 2

``--workers N`` spreads the scenarios over N processes; the output does not
depend on N.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import integrate, stats as sps

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

import figstyle  # noqa: E402  (shared style)
figstyle.apply()

from synthetic_shops import generate_synthetic_shop        # noqa: E402
from experiments._common import (GA_N_FINAL_SEEDS,          # noqa: E402
                                 build_headless_shop,
                                 base_params_for, anchor_base_params,
                                 base_params_record, chromosome_to_layout,
                                 checked_layout, layout_to_chromosome,
                                 make_run_dir, package_versions,
                                 portable_paths, provenance_snapshot,
                                 write_json, write_sidecar)
from experiments._inference import (EQUIV_MARGIN_FRAC,       # noqa: E402
                                    cluster_boot_ci, g1_t_interval,
                                    smallest_equivalence_margin)
from experiments.metaheuristics import (                   # noqa: E402
    DEFAULT_FINAL_TEMP_FRAC, DEFAULT_INITIAL_ACCEPT, DEFAULT_STEP_FRAC,
    simulated_annealing)

MUT_RATES = [0.10, 0.18, 0.30]
#: Populations swept; the generations of each follow from the fixed budget
#: (``setting_gens``). 30 is the default.
POP_SIZES = [20, 30, 45]
DEFAULT = (0.18, 30)

#: Annealer settings: step fraction x final-temperature fraction at the
#: default initial acceptance, then the initial acceptance at the default
#: step and final temperature.
SA_STEP_FRACS = [0.10, 0.25, 0.50]
SA_FINAL_TEMP_FRACS = [0.001, 0.01, 0.10]
SA_INITIAL_ACCEPTS = [0.5, 0.8, 0.95]
SA_DEFAULT = (DEFAULT_STEP_FRAC, DEFAULT_FINAL_TEMP_FRAC,
              DEFAULT_INITIAL_ACCEPT)

#: Figure B's comparison family size, whose Bonferroni level the annealer
#: sweep's equivalence tests use (``run_baseline_comparison``).
FIGB_N_COMPARATORS = 7

#: (scenario, seed) pairs of Figure B's paper-grade design (30 x 10), which
#: the annealer sweep's sample is recorded against: its equivalence test has
#: the power of its own, smaller, sample.
FIGB_DESIGN_PAIRS = 300

# Base of the held-out seed block the settings are re-scored on. Every run
# of scenario ``sc``, seed ``sd`` searches and selects under
# ``(1000*sc + sd) * 1000 + [0, n_gens_max + 6)``; ``parse_args`` keeps that
# range below this base so no setting is judged on a seed it optimized
# against.
HELDOUT_SEED_BASE = 100_000_000


def _figs_dir():
    root = os.path.dirname(os.path.dirname(_HERE))
    d = os.path.join(root, 'figs')
    os.makedirs(d, exist_ok=True)
    return d


def total_budget(pop_size: int, n_gens: int) -> int:
    """Evaluations of the default setting: its search plus its final
    selection, the budget every setting is held to."""
    return pop_size * (n_gens + GA_N_FINAL_SEEDS)


def setting_gens(pop: int, budget: int) -> int:
    """Generations a population of ``pop`` gets inside ``budget`` total
    evaluations (``pop`` x (gens + final seeds) <= budget)."""
    return max(1, budget // pop - GA_N_FINAL_SEEDS)


def ga_settings(pop_size: int, n_gens: int
                ) -> List[Tuple[float, int, int]]:
    """``(mutation rate, population, generations)`` of every GA setting,
    each at the default setting's total budget. The default population's
    entry is ``(pop_size, n_gens)`` itself; the others scale ``POP_SIZES``
    by ``pop_size / 30`` (so a smoke run can shrink them)."""
    budget = total_budget(pop_size, n_gens)
    scale = pop_size / DEFAULT[1]
    pops = []
    for p in POP_SIZES:
        q = pop_size if p == DEFAULT[1] else max(4, int(round(p * scale)))
        # A population too large for one generation inside the budget cannot
        # be held to it, and a shrunken grid can repeat a population.
        if q not in pops and budget // q - GA_N_FINAL_SEEDS >= 1:
            pops.append(q)
    out = []
    for mr in MUT_RATES:
        for p in pops:
            g = n_gens if p == pop_size else setting_gens(p, budget)
            out.append((mr, p, g))
    return out


def sa_settings() -> List[Tuple[float, float, float]]:
    """``(step fraction, final-temperature fraction, initial acceptance)``
    of every annealer setting."""
    out = [(st, ft, DEFAULT_INITIAL_ACCEPT)
           for st in SA_STEP_FRACS for ft in SA_FINAL_TEMP_FRACS]
    out += [(DEFAULT_STEP_FRAC, DEFAULT_FINAL_TEMP_FRAC, ia)
            for ia in SA_INITIAL_ACCEPTS if ia != DEFAULT_INITIAL_ACCEPT]
    return out


def expected_range_of_normals(k: int) -> float:
    """E[max - min] of k independent standard normals:
    the integral of 1 - Phi(x)^k - (1 - Phi(x))^k over the real line."""
    if k < 2:
        return 0.0
    f = lambda x: 1.0 - sps.norm.cdf(x) ** k - sps.norm.sf(x) ** k
    val, _ = integrate.quad(f, -np.inf, np.inf, limit=200)
    return float(val)


def _paired_best_gain(shop, names, bp, ref_chrom, history_best, rng_seed,
                      mc_days, mc_iters):
    """Per-generation population-best fitness minus the fitness of a fixed
    reference layout scored under the same MC seed.

    run_ga_headless scores generation ``gen`` under seed
    ``rng_seed * 1000 + gen``, so raw ``history_best`` values carry a
    seed-level shift shared by every layout, and a running max over them
    ratchets up on that noise. Differencing against the reference under
    the identical seed cancels the shared shift."""
    gains = np.empty(len(history_best), dtype=np.float64)
    for gen, best in enumerate(history_best):
        np.random.seed(rng_seed * 1000 + gen)
        ref = shop._ga_fitness(ref_chrom, names, bp, mc_days, mc_iters)
        gains[gen] = best - ref
    return gains


def _heldout_fitness(shop, names, bp, chrom, sc, args):
    """Mean fitness of one chromosome over the scenario's held-out seeds.

    ``best_fit`` is the largest of the final population's multi-seed means,
    taken over the very seeds that picked it, so it carries the upward bias
    of a maximum of noisy estimates -- and that bias grows with the
    population size, one of the axes this sweep varies. Re-scoring the
    returned layout on seeds no run searched or selected under, the same
    block for every setting of a scenario so the comparison stays paired,
    measures the layouts rather than the selection."""
    vals = []
    for k in range(args.n_heldout_seeds):
        np.random.seed(HELDOUT_SEED_BASE + 1000 * sc + k)
        vals.append(shop._ga_fitness(chrom, names, bp,
                                     args.mc_days, args.mc_iters))
    return float(np.mean(vals))


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--n-scenarios', type=int, default=10)
    p.add_argument('--n-seeds', type=int, default=3)
    p.add_argument('--n-items', type=int, default=10)
    p.add_argument('--n-gens', type=int, default=25,
                   help="Generations of the default setting; with "
                        "--pop-size it fixes every setting's budget")
    p.add_argument('--pop-size', type=int, default=30,
                   help="Population of the default setting")
    p.add_argument('--mc-iters', type=int, default=800)
    p.add_argument('--mc-days', type=int, default=30)
    p.add_argument('--n-heldout-seeds', type=int, default=10,
                   help="Seeds each setting's result is re-scored on, "
                        "outside every seed any run searched under")
    p.add_argument('--mode', choices=('ga', 'sa', 'both'), default='both',
                   help="Which sweep to run; the annealer sweep also runs the "
                        "GA's default setting it is compared with")
    p.add_argument('--workers', type=int, default=1,
                   help="Processes the scenarios are spread over")
    p.add_argument('--figs-dir', type=str, default=None,
                   help="Directory for ga_diversity.{pdf,png} and "
                        "ga_sensitivity.json (default: repo figs/)")
    p.add_argument('--out-root', type=str,
                   default=os.path.join(_HERE, 'results'),
                   help="Parent of the run directory holding the sidecar")
    args = p.parse_args(argv)
    if args.n_heldout_seeds < 1:
        p.error("--n-heldout-seeds must be >= 1")
    if args.n_seeds < 2:
        p.error("--n-seeds must be >= 2: the within-setting spread needs it")
    if args.workers < 1:
        p.error("--workers must be >= 1")
    if args.pop_size < 2 or args.n_gens < 1:
        p.error("--pop-size must be >= 2 and --n-gens >= 1")
    max_gens = max(g for _, _, g in ga_settings(args.pop_size, args.n_gens))
    top_search_seed = ((1000 * (args.n_scenarios - 1) + args.n_seeds - 1)
                       * 1000 + max_gens + GA_N_FINAL_SEEDS + 1)
    if (top_search_seed >= HELDOUT_SEED_BASE or args.n_heldout_seeds > 1000
            or max_gens + GA_N_FINAL_SEEDS + 1 > 1000):
        p.error("the sweep's search seeds must stay below the held-out "
                f"block at {HELDOUT_SEED_BASE} and inside each run's block "
                "of 1000, and --n-heldout-seeds must be <= 1000")
    return args


# --- One scenario ------------------------------------------------------------

def run_scenario(sc: int, args) -> Dict:
    """Every GA and annealer setting on scenario ``sc``, each seed's layout
    re-scored on the scenario's held-out seeds. Draws only from generators
    seeded here, so the result is a function of ``sc`` and ``args``."""
    shop_synth = generate_synthetic_shop(
        name=f"sens_{sc:03d}", seed=30_000 + sc,
        n_items=args.n_items, width=12.0, height=10.0)
    shop = build_headless_shop(shop_synth)
    names = [it.name for it in shop_synth.items]
    init_layout = {n: tuple(shop.floors[1]['items'][n]['position'])
                   for n in names}
    bp = anchor_base_params(shop, names, base_params_for(shop_synth),
                            init_layout)
    init_chrom = layout_to_chromosome(init_layout, names)
    out: Dict = {'scenario': sc, 'base_params': base_params_record(bp),
                 'ga': {}, 'sa': {}, 'div_traces': [], 'best_traces': [],
                 'evals': {}}
    default_key = f"{DEFAULT[0]}|{args.pop_size}|{args.n_gens}"
    ga_list = ga_settings(args.pop_size, args.n_gens)
    if args.mode == 'sa':
        ga_list = [s for s in ga_list
                   if s[0] == DEFAULT[0] and s[1] == args.pop_size]
    from experiments._common import run_ga_headless
    for mr, ps, ng in ga_list:
        key = f"{mr}|{ps}|{ng}"
        fits = []
        for sd in range(args.n_seeds):
            res = run_ga_headless(
                shop, names, bp, pop_size=ps, n_gens=ng,
                mut_rate=mr, elite_frac=0.20,
                mc_iters=args.mc_iters, mc_days=args.mc_days,
                rng_seed=1000 * sc + sd, init_layout=init_layout)
            checked_layout(shop, names,
                           chromosome_to_layout(res['best_chrom'], names),
                           f"sens scenario {sc} GA {key} seed {sd}")
            fits.append(_heldout_fitness(shop, names, bp, res['best_chrom'],
                                         sc, args))
            out['evals'][f"GA|{key}|{sd}"] = res['n_evals']
            if key == default_key:
                out['div_traces'].append(
                    [float(v) for v in res['history_diversity']])
                out['best_traces'].append(_paired_best_gain(
                    shop, names, bp, init_chrom, res['history_best'],
                    1000 * sc + sd, args.mc_days, args.mc_iters).tolist())
        out['ga'][key] = fits

    if args.mode in ('sa', 'both'):
        budget = args.pop_size * args.n_gens
        for st, ft, ia in sa_settings():
            key = f"{st}|{ft}|{ia}"
            fits = []
            for sd in range(args.n_seeds):
                stats: Dict = {}
                lay = simulated_annealing(
                    shop, shop_synth, names, bp, seed=1000 * sc + sd,
                    budget=budget, block=args.pop_size,
                    mc_iters=args.mc_iters, mc_days=args.mc_days,
                    n_final_seeds=GA_N_FINAL_SEEDS,
                    initial_accept=ia, final_temp_frac=ft, step_frac=st,
                    stats=stats, init_layout=init_layout, start='asbuilt')
                checked_layout(shop, names, lay,
                               f"sens scenario {sc} SA {key} seed {sd}")
                fits.append(_heldout_fitness(
                    shop, names, bp, layout_to_chromosome(lay, names),
                    sc, args))
                out['evals'][f"SA|{key}|{sd}"] = stats['n_evals']
            out['sa'][key] = fits
    print(f"[sens {sc}] done", flush=True)
    return out


def _map(fn, n: int, workers: int, *fn_args) -> list:
    if workers <= 1:
        return [fn(s, *fn_args) for s in range(n)]
    done = {}
    pool = ProcessPoolExecutor(max_workers=min(workers, n))
    try:
        futures = {pool.submit(fn, s, *fn_args): s for s in range(n)}
        for fut in as_completed(futures):
            done[futures[fut]] = fut.result()
    except BaseException:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown(wait=True)
    return [done[s] for s in sorted(done)]


# --- Aggregation -------------------------------------------------------------

#: What the GA sweep's null assumes, as its summary records it.
GA_NULL_DESIGN = (
    "blocked by (scenario, seed): every setting of scenario sc, seed sd runs "
    "from rng_seed 1000*sc + sd and is re-scored on the scenario's one "
    "held-out block (common random numbers), so the setting means are "
    "correlated; the null range and the ANOVA take the noise from the "
    "residual after the seed blocks are removed")


def _blocked_residual_sd(vals: np.ndarray) -> float:
    """SD of the residual of the additive setting + seed model on a
    (settings, seeds) table: the noise in one value once the effect every
    setting of a seed shares is removed, with (k - 1)(n - 1) degrees of
    freedom. Under common random numbers this, not the SD within a setting,
    is the noise the setting means differ by."""
    k, n = vals.shape
    if k < 2 or n < 2:
        return float('nan')
    resid = (vals - vals.mean(axis=1, keepdims=True)
             - vals.mean(axis=0, keepdims=True) + vals.mean())
    return float(np.sqrt((resid ** 2).sum() / ((k - 1) * (n - 1))))


def ga_sweep_summary(results: Sequence[Dict], default_key: str) -> Dict:
    """Per scenario: every setting's mean held-out fitness and its SD over
    seeds, the relative spread of the setting means, the default's gap to
    the best setting, and the spread the noise alone would give.

    Every setting of a (scenario, seed) shares its search seed and its
    held-out evaluation block (``GA_NULL_DESIGN``), so the k setting means
    of a scenario move together and are not k independent draws. The null
    range is therefore the expected range of k standard normals times the
    BLOCKED residual SD (``_blocked_residual_sd``) over sqrt(seeds)
    (``null_expected_range_pct``); the independent-draws null, with the
    pooled SD within a setting, is kept beside it for reference
    (``null_expected_range_pct_independent``) and is too wide by the share
    of that SD the seed blocks explain (``seed_block_share``). Across
    scenarios: their medians, the ratio of the observed to the blocked null
    range, and the ANOVA of ``_blocked_anova`` on the fitness relative to
    its scenario mean."""
    per = []
    rel = []
    for r in results:
        keys = sorted(r['ga'])
        vals = np.array([r['ga'][k] for k in keys])        # (k, n_seeds)
        means = vals.mean(axis=1)
        sds = vals.std(axis=1, ddof=1)
        center = float(abs(means.mean())) or 1e-9
        spread = (means.max() - means.min()) / center * 100
        best = means.max()
        gap = (best - means[keys.index(default_key)]) / max(abs(best), 1e-9) \
            * 100 if default_key in keys else float('nan')
        k, n = vals.shape
        pooled_sd = float(np.sqrt(np.mean(sds ** 2)))
        block_sd = _blocked_residual_sd(vals)
        er = expected_range_of_normals(k)
        null = er * block_sd / math.sqrt(n) / center * 100
        null_ind = er * pooled_sd / math.sqrt(n) / center * 100
        per.append({'scenario': r['scenario'], 'settings': keys,
                    'mean': means.tolist(), 'sd_within': sds.tolist(),
                    'spread_pct': float(spread), 'default_gap_pct': float(gap),
                    'pooled_sd_within': pooled_sd,
                    'blocked_residual_sd': block_sd,
                    'seed_block_share': (float(1.0 - block_sd ** 2
                                               / pooled_sd ** 2)
                                         if pooled_sd > 0 else None),
                    'null_expected_range_pct': float(null),
                    'null_expected_range_pct_independent': float(null_ind),
                    'spread_over_null': float(spread / null) if null > 0
                    else None})
        rel.append(vals / center)
    spreads = np.array([p['spread_pct'] for p in per])
    nulls = np.array([p['null_expected_range_pct'] for p in per])
    nulls_ind = np.array([p['null_expected_range_pct_independent']
                          for p in per])
    n_min = min(v.shape[1] for v in rel)
    anova = _blocked_anova(np.stack([v[:, :n_min] for v in rel], axis=0))
    shares = [p['seed_block_share'] for p in per
              if p['seed_block_share'] is not None]
    return {
        'per_scenario': per,
        'null_design': GA_NULL_DESIGN,
        'hyper_spread_median_pct': float(np.median(spreads)),
        'hyper_spread_max_pct': float(np.max(spreads)),
        'default_gap_median_pct': float(np.median([p['default_gap_pct']
                                                   for p in per])),
        'null_range_median_pct': float(np.median(nulls)),
        'null_range_median_pct_independent': float(np.median(nulls_ind)),
        'spread_over_null_median': float(np.median(spreads / np.maximum(
            nulls, 1e-12))),
        'n_scenarios_spread_above_null': int(np.sum(spreads > nulls)),
        'seed_block_share_median': (float(np.median(shares)) if shares
                                    else None),
        'anova_setting': anova,
    }


def _f_test(ss: float, df: int, ss_den: float, df_den: int) -> Dict:
    if df <= 0 or df_den <= 0 or ss_den <= 0:
        return {'F': None, 'df': int(df), 'df_error': int(df_den), 'p': None}
    F = (ss / df) / (ss_den / df_den)
    return {'F': float(F), 'df': int(df), 'df_error': int(df_den),
            'p': float(sps.f.sf(F, df, df_den))}


def _blocked_anova(y: np.ndarray) -> Dict:
    """ANOVA of the setting effect on ``y`` of shape (scenarios a, settings
    b, seeds n), with each scenario's seeds as blocks.

    Every setting of scenario s, seed k shares that seed's search and
    evaluation draws, so (s, k) is a block and the seeds are not
    replicates inside a (scenario, setting) cell. The model is scenario +
    seed-within-scenario block + setting + scenario x setting + residual,
    the residual being the setting x block interaction, on
    a (b - 1)(n - 1) degrees of freedom. Two tests of the setting main
    effect:

      * ``setting_vs_residual``: against the residual -- the scenarios taken
        as fixed, the seed blocks removed from the error;
      * ``setting_vs_interaction``: against the scenario x setting
        interaction -- the scenarios taken as a random sample of the
        template family, which a claim about operator robustness in general
        needs, on (b - 1)(a - 1) degrees of freedom.

    ``interaction`` tests the scenario x setting interaction against the
    residual (do the settings rank differently by scenario?), and
    ``seed_blocks_vs_residual`` how much the shared seeds move every
    setting together."""
    a, b, n = y.shape
    grand = y.mean()
    m_s = y.mean(axis=(1, 2))                    # scenario
    m_t = y.mean(axis=(0, 2))                    # setting
    m_st = y.mean(axis=2)                        # scenario x setting
    m_sk = y.mean(axis=1)                        # scenario x seed (block)
    ss_t = a * n * ((m_t - grand) ** 2).sum()
    ss_st = n * ((m_st - m_s[:, None] - m_t[None, :] + grand) ** 2).sum()
    ss_blk = b * ((m_sk - m_s[:, None]) ** 2).sum()
    ss_e = ((y - m_st[:, :, None] - m_sk[:, None, :]
             + m_s[:, None, None]) ** 2).sum()
    df_t, df_st = b - 1, (a - 1) * (b - 1)
    df_blk, df_e = a * (n - 1), a * (b - 1) * (n - 1)
    return {
        'design': 'scenario + seed-within-scenario block + setting + '
                  'scenario x setting; residual = setting x block',
        'n_scenarios': int(a), 'n_settings': int(b), 'n_seeds': int(n),
        'setting_vs_residual': _f_test(ss_t, df_t, ss_e, df_e),
        'setting_vs_interaction': _f_test(ss_t, df_t, ss_st, df_st),
        'interaction': _f_test(ss_st, df_st, ss_e, df_e),
        'seed_blocks_vs_residual': _f_test(ss_blk, df_blk, ss_e, df_e),
    }


def sa_sweep_summary(results: Sequence[Dict], default_key: str) -> Dict:
    """The annealer sweep, in two parts.

    The annealer's OWN sensitivity: per setting, that annealer minus the
    default annealer, paired per (scenario, seed) on the same held-out
    seeds (every setting of a (scenario, seed) shares its search seed), with
    its scenario cluster-bootstrap interval at a Bonferroni level over the
    non-default settings (``vs_default_sa``). This is what says whether the
    annealer's configuration matters.

    The GA against each annealer: the GA (default setting) minus that
    annealer, paired the same way; its cluster-bootstrap and G - 1 t
    intervals at Figure B's per-comparison level; and the equivalence test
    at Figure B's margin (the (1 - 2 alpha') interval inside +/-
    ``EQUIV_MARGIN_FRAC`` of the annealer's mean), with the smallest margin
    at which it would hold.

    This sweep runs on far fewer (scenario, seed) pairs than Figure B, so
    its equivalence test has less power: a setting that is not declared
    equivalent may only be one this sample cannot resolve. The default
    annealer's own result in the sweep is the POWER REFERENCE
    (``power_reference``): each setting's smallest margin is reported beside
    it (``margin_ratio_to_default``), and ``n_equivalent`` counts the
    settings for which equivalence could be established at this sample
    size, not the configurations for which it fails to hold."""
    alpha = 0.05 / FIGB_N_COMPARATORS
    keys = sorted({k for r in results for k in r['sa']})
    default_sa = next((k for k in keys
                       if tuple(float(v) for v in k.split('|'))
                       == tuple(map(float, SA_DEFAULT))), None)
    n_other = max(1, len(keys) - (1 if default_sa is not None else 0))
    alpha_sa = 0.05 / n_other
    out: Dict[str, Dict] = {}
    for key in keys:
        clusters = []
        sa_vals = []
        for r in results:
            ga = r['ga'][default_key]
            sa = r['sa'][key]
            clusters.append([g - s for g, s in zip(ga, sa)])
            sa_vals.extend(sa)
        mean, lo, hi = cluster_boot_ci(clusters, alpha=alpha)
        _, elo, ehi = cluster_boot_ci(clusters, alpha=2 * alpha)
        g1 = g1_t_interval(clusters, alpha=2 * alpha)
        base = float(np.mean(sa_vals))
        margin = EQUIV_MARGIN_FRAC * base
        st, ft, ia = (float(v) for v in key.split('|'))
        smallest = smallest_equivalence_margin(elo, ehi)
        rec = {
            'step_frac': st, 'final_temp_frac': ft, 'initial_accept': ia,
            'is_default': key == default_sa,
            'mean_ga_minus_sa': mean, 'ci_lo': lo, 'ci_hi': hi,
            'excludes_zero': bool(lo > 0 or hi < 0),
            'sa_mean': base, 'pct': mean / max(abs(base), 1e-9) * 100,
            'equivalence': {
                'margin': margin, 'lo': elo, 'hi': ehi,
                'equivalent': bool(-margin < elo and ehi < margin),
                'smallest_margin': smallest,
                'smallest_margin_frac': smallest / max(abs(base), 1e-9),
                'g1_t': {'lo': g1['lo'], 'hi': g1['hi'],
                         'equivalent': bool(-margin < g1['lo']
                                            and g1['hi'] < margin)}},
            'win_runs_ga': int(sum(d > 0 for c in clusters for d in c)),
            'n_runs': int(sum(len(c) for c in clusters)),
        }
        if default_sa is not None and key != default_sa:
            diff = [[s - d for s, d in zip(r['sa'][key], r['sa'][default_sa])]
                    for r in results]
            dm, dlo, dhi = cluster_boot_ci(diff, alpha=alpha_sa)
            dbase = float(np.mean([v for r in results
                                   for v in r['sa'][default_sa]]))
            rec['vs_default_sa'] = {
                'mean_sa_minus_default': dm, 'ci_lo': dlo, 'ci_hi': dhi,
                'excludes_zero': bool(dlo > 0 or dhi < 0),
                'pct': dm / max(abs(dbase), 1e-9) * 100,
                'ci_lo_pct': dlo / max(abs(dbase), 1e-9) * 100,
                'ci_hi_pct': dhi / max(abs(dbase), 1e-9) * 100}
        out[key] = rec
    ref = out.get(default_sa) if default_sa is not None else None
    if ref is not None:
        ref_frac = ref['equivalence']['smallest_margin_frac']
        for rec in out.values():
            rec['margin_ratio_to_default'] = (
                rec['equivalence']['smallest_margin_frac'] / ref_frac
                if ref_frac > 0 else None)
    n_pairs = int(sum(len(r['sa'][keys[0]]) for r in results)) if keys else 0
    return {'alpha_per_comparison': alpha,
            'level': 1 - alpha, 'equivalence_level': 1 - 2 * alpha,
            'margin_frac': EQUIV_MARGIN_FRAC, 'settings': out,
            'n_equivalent': int(sum(v['equivalence']['equivalent']
                                    for v in out.values())),
            'n_settings': len(out),
            'n_scenarios': len(results), 'n_pairs': n_pairs,
            'figb_design_pairs': FIGB_DESIGN_PAIRS,
            'power_reference': (None if ref is None else {
                'setting': default_sa,
                'equivalent': ref['equivalence']['equivalent'],
                'smallest_margin_frac':
                    ref['equivalence']['smallest_margin_frac']}),
            'vs_default_sa_alpha': alpha_sa,
            'vs_default_sa_level': 1 - alpha_sa,
            'n_settings_differ_from_default': int(sum(
                v.get('vs_default_sa', {}).get('excludes_zero', False)
                for v in out.values())),
            'interpretation': (
                "n_equivalent counts the settings for which GA-SA "
                "equivalence at the Figure B margin could be established on "
                "this sweep's (scenario, seed) pairs; a setting without it "
                "may be one this sample cannot resolve -- compare its "
                "smallest margin with the default's (power_reference). "
                "vs_default_sa is the annealer's own sensitivity.")}


# --- Main --------------------------------------------------------------------

#: Height of the diversity figure (in); its width is its print width,
#: ``figstyle.PRINT_FRAC['ga_diversity']`` of the text block.
DIVERSITY_FIG_HEIGHT_IN = 2.7


def plot_diversity(div_traces, best_traces, *, mut_rate, pop_size, figs):
    """The GA-convergence-versus-diversity figure (``ga_diversity``), drawn
    at the width it prints at (``figstyle.print_figure``) with the print
    type sizes (``figstyle.apply_print``, inside an rc context so the rest
    of the run keeps its style), and saved through ``figstyle.save``,
    which refuses it unless its width is its print width and its smallest
    text is at least ``figstyle.MIN_FONT_PT`` (review R56). Each series has
    its own line style as well as its colour. Returns the summary's
    diversity fields and the figure's print record."""
    G = min(len(t) for t in div_traces)
    div = np.stack([np.asarray(t, float)[:G] for t in div_traces], 0)
    bst = np.stack([np.asarray(t, float)[:G] for t in best_traces], 0)
    gens = np.arange(1, G + 1)
    div_m = div.mean(0)
    bst_m = bst.mean(0)
    bst_norm = (bst_m - bst_m.min()) / max(bst_m.max() - bst_m.min(), 1e-9)
    with matplotlib.rc_context():
        figstyle.apply_print()
        fig, ax1 = figstyle.print_figure('ga_diversity',
                                         height_in=DIVERSITY_FIG_HEIGHT_IN)
        best_style = figstyle.SERIES[0]
        div_style = figstyle.SERIES[3]
        ax1.plot(gens, bst_norm, color=best_style['color'],
                 linestyle=best_style['linestyle'],
                 label='best gain (normalized)')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('best gain over start (normalized)',
                       color=best_style['color'])
        ax1.tick_params(axis='y', labelcolor=best_style['color'])
        ax2 = ax1.twinx()
        ax2.spines['right'].set_visible(True)
        ax2.grid(False)
        ax2.plot(gens, div_m * 100, color=div_style['color'],
                 linestyle=div_style['linestyle'],
                 label='population diversity')
        ax2.set_ylabel('diversity (% of diagonal)', color=div_style['color'])
        ax2.tick_params(axis='y', labelcolor=div_style['color'])
        ax2.set_ylim(0, max(div_m.max() * 100 * 1.25, 1e-3))
        ax1.set_title(f'Mutation {mut_rate:g}, population {pop_size}; '
                      f'{len(div_traces)} runs')
        smallest = figstyle.check_print(fig, 'ga_diversity')
        pdf = figstyle.save(fig, 'ga_diversity', out_dir=figs)
        plt.close(fig)
    return {
        'diversity_start': float(div_m[0]),
        'diversity_end': float(div_m[-1]),
        'diversity_retained_pct': float(div_m[-1] / max(div_m[0], 1e-9)
                                        * 100),
        'diversity_figure': figstyle.print_record(
            pdf, smallest, figstyle.print_width('ga_diversity')),
    }


def main(argv=None):
    args = parse_args(argv)
    figs = args.figs_dir or _figs_dir()
    os.makedirs(figs, exist_ok=True)
    out_dir = make_run_dir(args.out_root, 'ga_sensitivity')
    prov = provenance_snapshot()
    # The arguments as run: the figures directory is the one actually
    # written to, not None when --figs-dir was left at its default, and
    # both output locations are absolute here so they mean the same thing
    # whatever the working directory. The JSON files below name them
    # relative to the repository when they lie inside it
    # (``portable_paths``; ``write_sidecar`` applies the same rule), so a
    # shipped file does not carry this checkout's absolute path.
    run_args = {**vars(args), 'figs_dir': os.path.abspath(figs),
                'out_root': os.path.abspath(args.out_root)}

    t0 = time.perf_counter()
    results = _map(run_scenario, args.n_scenarios, args.workers, args)
    default_key = f"{DEFAULT[0]}|{args.pop_size}|{args.n_gens}"
    budget = total_budget(args.pop_size, args.n_gens)

    # Every per-(scenario, setting, seed) held-out fitness.
    with open(os.path.join(out_dir, 'settings.csv'), 'w', newline='',
              encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['scenario', 'search', 'setting', 'seed',
                    'heldout_fitness'])
        for r in results:
            for tag in ('ga', 'sa'):
                for key, fits in sorted(r[tag].items()):
                    for sd, v in enumerate(fits):
                        w.writerow([r['scenario'], tag, key, sd, f"{v:.6f}"])

    summary: Dict = {
        'fitness_statistic': 'heldout_mean',
        'heldout_seeds': int(args.n_heldout_seeds),
        'heldout_seed_rule': f'{HELDOUT_SEED_BASE} + 1000*scenario + k',
        'budget_total_evals': budget,
        'budget_rule': 'pop x (gens + final seeds) = default setting total',
        'mode': args.mode,
        'n_scenarios': args.n_scenarios, 'n_seeds': args.n_seeds,
    }
    evals = sorted({v for r in results for k, v in r['evals'].items()
                    if k.startswith('GA|')})
    summary['ga_total_evals_per_run'] = evals

    if args.mode in ('ga', 'both'):
        sweep = ga_sweep_summary(results, default_key)
        settings = ga_settings(args.pop_size, args.n_gens)
        summary.update({
            'hyper_spread_median_pct': sweep['hyper_spread_median_pct'],
            'hyper_spread_max_pct': sweep['hyper_spread_max_pct'],
            'default_gap_median_pct': sweep['default_gap_median_pct'],
            'n_settings': len(settings),
            'mut_rates': MUT_RATES,
            'pop_sizes': sorted({p for _, p, _ in settings}),
            'settings': [{'mut_rate': mr, 'pop_size': p, 'n_gens': g,
                          'total_evals': p * (g + GA_N_FINAL_SEEDS)}
                         for mr, p, g in settings],
            'ga_sweep': sweep,
        })
    if args.mode in ('sa', 'both'):
        summary['sa_sweep'] = sa_sweep_summary(results, default_key)
        summary['sa_settings'] = [{'step_frac': st, 'final_temp_frac': ft,
                                   'initial_accept': ia}
                                  for st, ft, ia in sa_settings()]
        summary['sa_budget_search_evals'] = args.pop_size * args.n_gens

    div_traces = [np.asarray(t, float) for r in results
                  for t in r['div_traces']]
    best_traces = [np.asarray(t, float) for r in results
                   for t in r['best_traces']]
    if div_traces:
        summary.update(plot_diversity(div_traces, best_traces,
                                      mut_rate=DEFAULT[0],
                                      pop_size=args.pop_size, figs=figs))
    summary['config'] = run_args
    summary['provenance'] = {**prov, 'python': sys.version.split()[0],
                             'packages': package_versions()}

    with open(os.path.join(figs, 'ga_sensitivity.json'), 'w',
              encoding='utf-8', newline='\n') as f:
        json.dump(portable_paths(summary), f, indent=2)
    write_sidecar(out_dir, {'experiment': 'ga_sensitivity',
                            'args': run_args,
                            'figs_dir': run_args['figs_dir'],
                            'wall_seconds': time.perf_counter() - t0,
                            'base_params': {str(r['scenario']):
                                            r['base_params'] for r in results},
                            'evaluation_counts': {str(r['scenario']):
                                                  r['evals'] for r in results},
                            'summary': summary},
                  provenance=prov)
    # Written last: its presence marks a finished run.
    write_json(out_dir, 'summary.json', portable_paths(summary))
    if 'hyper_spread_median_pct' in summary:
        gs = summary['ga_sweep']
        an = gs['anova_setting']
        print(f"\nGA settings: spread (median across scenarios) "
              f"{summary['hyper_spread_median_pct']:.3f}% against "
              f"{gs['null_range_median_pct']:.3f}% from the blocked noise "
              f"alone ({gs['null_range_median_pct_independent']:.3f}% if the "
              f"settings were independent); setting p = "
              f"{an['setting_vs_residual']['p']} (scenarios fixed), "
              f"{an['setting_vs_interaction']['p']} (scenarios random); "
              f"default {summary['default_gap_median_pct']:.3f}% below best")
    if 'sa_sweep' in summary:
        ss = summary['sa_sweep']
        pr = ss['power_reference'] or {}
        print(f"Annealer settings: GA-SA equivalence at the Figure B margin "
              f"established for {ss['n_equivalent']} of {ss['n_settings']} on "
              f"{ss['n_pairs']} (scenario, seed) pairs; the default "
              f"annealer's smallest margin is "
              f"{pr.get('smallest_margin_frac', float('nan')):.4%}")
        for key, v in ss['settings'].items():
            vd = v.get('vs_default_sa')
            print(f"   step {v['step_frac']:<5g} final T {v['final_temp_frac']:<6g}"
                  f" accept {v['initial_accept']:<5g}: GA-SA "
                  f"{v['mean_ga_minus_sa']:+9.2f} [{v['ci_lo']:+.2f}, "
                  f"{v['ci_hi']:+.2f}]  equivalent={v['equivalence']['equivalent']}"
                  + ('' if vd is None else
                     f"  SA-default {vd['pct']:+.3f}% "
                     f"[{vd['ci_lo_pct']:+.3f}, {vd['ci_hi_pct']:+.3f}]"))
    print(f"wall {time.perf_counter() - t0:.0f}s; run record in {out_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
