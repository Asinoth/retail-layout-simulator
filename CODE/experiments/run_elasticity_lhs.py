"""Elasticity- and weight-uncertainty robustness sweeps.

The GA composite layout score is ELASTICITY-INDEPENDENT; the cited
elasticity coefficients only map score -> (conversion, impulse, basket)
lifts -> projected revenue. The headline "GA beats baseline" claim is
therefore robust iff the PROJECTED LIFT stays positive across the
literature-implied coefficient bands -- which we can evaluate in closed
form without re-running the GA per draw.

Design:
  * N scenarios x S seeds: run the (moderate-budget) GA once per case and
    build, on the same scenario and the same feasible set, the comparators
    the paper's headline comparisons use -- the popularity_rank baseline,
    the analytical reference solution, and the two equal-budget
    metaheuristics (random search and simulated annealing). Both searches
    start from the as-built layout the GA's population is seeded from, so
    the three searches leave from the same point.
  * Score every layout once (elasticity-free composite + breakdown).
  * Latin Hypercube (scipy.stats.qmc) over the three total elasticities,
    each within its cited band [BASE, BASE + GAIN_MAX]:
        conversion  (Hui 2009),
        impulse     (Hui, Inman 2013),
        basket size (conservative band; flagged in retail_literature).
  * For each draw, the exact expected horizon revenue of every layout,
    ``experiments.closed_form.expected_revenue`` -- the mean the GA's Monte
    Carlo fitness converges to, anchored at the scenario's repaired
    as-built layout (``layout_objective``) -> paired lift distribution
    across the band, per comparison.

What the sign of a lift can and cannot say: conversion and basket both
rise with the composite score, so the sign of a lift is, to first order,
the sign of the two layouts' score difference, whatever the elasticities.
The sweep therefore reports, beside the pooled fractions, the (scenario,
seed) pair counts behind each fraction and the range of each pair's lift
across the design -- how much the SIZE of the effect moves inside the
bands, which the sign cannot show.

Basket corners and a short-path variant (same layouts, closed form):
  * the basket elasticity at 0.02, a fifth of the band's floor, and at 0,
    no basket effect, with conversion and impulse at their midpoints;
  * the basket driver without the flow and entrance-proximity criteria,
    which reward SHORTER paths although the basket elasticity is anchored
    on 'more walking, more unplanned spending' (``basket_exclude``); at the
    midpoints and over the LHS draws.

Weight sweep -- a second design over the same scored layouts:
  The composite's seven weights are fixed in every experiment: the five
  spatial criteria (traffic, cross-merchandising, impulse, flow, revenue
  placement) share 0.80 equally, section compliance holds 0.16 and
  accessibility 0.04 (retail_literature). The equal split of the five
  rests on Dawes (1979), an argument for improper linear models, not a
  measurement. This design tests it without any extra GA or search run:
  * W spatial weight vectors drawn uniformly on the simplex scaled to the
    spatial total, Dirichlet(1,1,1,1,1) x 0.80, from a dedicated generator
    (``--weight-seed``). Section compliance, accessibility and both
    penalties keep their values.
  * Each layout's composite is recomposed from the per-criterion
    breakdown ``_ga_compute_layout_score`` already returned. Before any
    draw is evaluated, recomposing at the default weights must reproduce
    the returned score to 1e-9, so the sweep measures the weights and not
    the recomposer. The anchor layout is recomposed under the same weights,
    so it reproduces the calibrated inputs under every weighting.
  * The recomposed composite goes through the same closed-form horizon
    revenue, at the elasticity band midpoints. Only the composite moves:
    the impulse elasticity still reads the impulse sub-score, and
    abandonment the flow, section and bottleneck sub-scores.
  What it does and does not test: every layout was SEARCHED under the
  default weights. The sweep asks whether the ranking of the layouts in
  hand survives when they are SCORED under other weightings of the
  criteria. It does not ask whether the GA would still win if it searched
  under other weights; that needs one search per weight vector, which is
  a different experiment. A (scenario, seed) pair counts as a sign flip
  when its lift takes both signs over the default weights and the W
  draws. W points only sample the simplex, so the count is a lower bound
  on the pairs whose lift changes sign somewhere in it.

Effective weights and the standardized weight sweeps:
  Dawes' unit weights are for STANDARDIZED predictors. The criteria here
  are raw [0, 1] scores whose spread over feasible layouts differs by two
  orders of magnitude, so a nominal weight is not an effective one. Per
  scenario, ``_common.criterion_spread`` draws a seeded reference sample of
  repaired random layouts and records each criterion's SD and effective
  weight (weight x SD). Two more designs repeat the weight sweep on a
  standardized scale: spatial criterion k enters the composite as
  x_k * kappa / SD_k, SD_k its reference SD, so the five have equal spread
  and equal weights on them are Dawes' unit weights (the default point).
  The weights are the same Dirichlet draws. Section compliance,
  accessibility and the penalties stay fixed on their raw scale. A
  criterion with no spread is constant over the layouts and cancels
  against the anchor, so it contributes nothing. The two designs differ in
  kappa (``STANDARDIZED_SCALES``):
  * 'weight_sweep_standardized_unit', kappa = 1: the criteria divided by
    their reference SD and nothing else, as the review's design decision
    D3 specified. A score difference is then counted in SDs, far outside
    the range of raw [0, 1] score changes the elasticity bands were built
    for, so its lifts run to tens of percent;
  * 'weight_sweep_standardized', kappa = the mean reference SD of the
    spatial criteria that vary: a deliberate addition to D3 that keeps the
    standardized composite at the raw composite's magnitude, the range the
    bands were built for.
  kappa is NOT a neutral common factor. It scales the five standardized
  spatial terms only, while accessibility, section compliance and the
  penalties stay raw and revenue also reads raw sub-scores outside the
  composite -- the impulse sub-score through the impulse elasticity, the
  flow, section and bottleneck sub-scores through abandonment. kappa
  therefore sets how much the composite channel weighs against those, and
  it moves the size of the lifts, their signs and the sign-flip counts
  (even the ranking under the composite alone, through accessibility).
  Hence both designs, side by side. Re-scored, not re-searched, and
  reported beside the raw sweep with the same statistics.

Outputs (experiments/results/elasticity_lhs_<ts>/):
    results.csv       one row per (scenario, seed, draw)
    weight_sweep.csv  one row per (scenario, seed, weight draw,
                      comparator): the five spatial weights, the
                      GA-minus-comparator lift, and the pair's lift at the
                      default weights
    weight_sweep_standardized.csv, weight_sweep_standardized_unit.csv
                      the same for the two standardized designs
    summary.json      fraction of positive-lift draws and lift percentiles
                      for the popularity baseline, and the same per
                      comparison under 'comparisons' (with pair counts and
                      within-pair ranges); the basket corners and the
                      short-path variant; the effective weights per
                      scenario; the three weight designs and their
                      per-comparison results under 'weight_sweep',
                      'weight_sweep_standardized' and
                      'weight_sweep_standardized_unit'
    lhs_hist.png      lift histogram across all draws
    sidecar.json      git SHA, seeds, elasticity snapshot, base parameters

Smoke:   python -m experiments.run_elasticity_lhs --n-scenarios 3 --n-seeds 1 --n-draws 64
Paper:   python -m experiments.run_elasticity_lhs --n-scenarios 12 --n-seeds 3 --n-draws 256 \
             --n-gens 15 --pop-size 24 --mc-iters 500
         (the GA budget flags are required: the argparse defaults are a
         smaller budget than the published run; each scenario-seed now also
         runs the two equal-budget searches, so it costs about three GA runs)
         ``--workers N`` runs scenarios in N processes; the CSVs are unchanged.
         ``--n-weight-draws`` (default 256) and ``--weight-seed`` set the
         weight designs; they re-score the layouts above and add no search.
         ``--n-spread-layouts`` (default 150) sets the reference sample of
         the effective weights.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from scipy.stats import qmc

from synthetic_shops import generate_synthetic_shop
from baselines import popularity_rank, assert_no_strict_overlap
from oracle import solve_oracle
from experiments._common import (ORACLE_RESTARTS, oracle_record,
                                 oracle_summary,
                                 build_headless_shop, base_params_for,
                                 anchor_base_params, base_params_record,
                                 checked_layout,
                                 criterion_spread, SPREAD_N_LAYOUTS,
                                 SPREAD_SEED,
                                 run_ga_headless, feasible_layout,
                                 chromosome_to_layout, layout_to_chromosome,
                                 write_sidecar, make_run_dir, portable_paths)
from experiments.closed_form import expected_revenue
from experiments.metaheuristics import random_search, simulated_annealing
from layout_objective import (ELASTICITY_BANDS, ELASTICITY_MIDPOINTS,
                              SHORT_PATH_CRITERIA, anchor_of, make_anchor)
from retail_literature import (
    DEFAULT_OP_HOURS_PER_DAY, DEFAULT_WEEKEND_MULTIPLIER,
    GA_W_TRAFFIC, GA_W_CROSS_MERCH, GA_W_IMPULSE, GA_W_FLOW,
    GA_W_REVENUE_PLACEMENT, GA_W_SECTION_COMPLIANCE, GA_W_ACCESSIBILITY,
    GA_PEN_OVERLAP, GA_PEN_BOTTLENECK, MC_SPEND_LAW,
)

BANDS = dict(ELASTICITY_BANDS)

# The comparators the GA is differenced against, in report order. The first
# is the popularity-rank baseline this sweep has always used; the other
# three are the comparisons the paper's headline table reports, so the
# elasticity bands are swept over all of them rather than one.
COMPARATORS = ('popularity_rank', 'oracle', 'random_search',
               'simulated_annealing')

# Length of the projected horizon, and its customer-hours counted the way
# mc_engine counts them: it takes day d of the horizon as day d % 7 of the
# week, so a 30-day horizon holds 8 weekend days, not 30 * 2/7 of them.
# expected_revenue counts the days the same way.
MC_DAYS = 30
HORIZON_CUST_HOURS = DEFAULT_OP_HOURS_PER_DAY * sum(
    DEFAULT_WEEKEND_MULTIPLIER if d % 7 in (5, 6) else 1.0
    for d in range(MC_DAYS))

# results.csv layout. ``rev30_baseline``, ``lift30`` and ``lift_pct`` keep
# their names for the popularity baseline and the GA-minus-popularity
# difference the lift figure is drawn from; the other comparators follow.
COLUMNS = (['scenario', 'seed', 'draw', 'conv_e', 'imp_e', 'bsk_e',
            'rev30_ga', 'rev30_baseline']
           + [f'rev30_{m}' for m in COMPARATORS[1:]]
           + ['lift30', 'lift_pct']
           + [f'lift30_{m}' for m in COMPARATORS[1:]])

# The basket-elasticity corners below the band: 0.02, a fifth of the band's
# floor, and no basket effect at all. Conversion and impulse stay at their
# midpoints.
BASKET_CORNERS = (0.02, 0.0)

# --- weight sweep ---------------------------------------------------------
# The five spatial criteria, in the order their weights are drawn and
# written, with the weights the composite uses today. They share
# SPATIAL_TOTAL (0.80); section compliance, accessibility and the two
# penalties are held at their retail_literature values throughout.
SPATIAL_CRITERIA = ('traffic', 'cross_merch', 'impulse', 'flow',
                    'revenue_placement')
DEFAULT_SPATIAL_WEIGHTS = (GA_W_TRAFFIC, GA_W_CROSS_MERCH, GA_W_IMPULSE,
                           GA_W_FLOW, GA_W_REVENUE_PLACEMENT)
SPATIAL_TOTAL = float(sum(DEFAULT_SPATIAL_WEIGHTS))

# How closely the recomposed default-weight composite must match the score
# _ga_compute_layout_score returned before the sweep may run.
RECOMPOSE_TOL = 1e-9

# A reference SD at or below this counts as no spread: the criterion is
# constant over the reference layouts and drops out of the standardized
# composite.
SD_EPS = 1e-12

# The standardized scales, name -> (summary key, CSV, what kappa is). Spatial
# criterion k enters the standardized composite as x_k * kappa / SD_k. kappa
# is not a neutral common factor (module docstring): it weighs the composite
# channel against the raw accessibility, impulse and abandonment terms, so
# the sweep is reported at both values.
STANDARDIZED_SCALES = {
    'mean_sd': ('weight_sweep_standardized',
                'weight_sweep_standardized.csv',
                'the mean reference SD of the spatial criteria that vary, '
                'which keeps the standardized composite at the raw '
                'composite\'s magnitude, the range of score changes the '
                'elasticity bands were built for (a deliberate addition to '
                'the review\'s design decision D3)'),
    'unit':    ('weight_sweep_standardized_unit',
                'weight_sweep_standardized_unit.csv',
                '1: the criteria divided by their reference SD and nothing '
                'else, as design decision D3 specified; score differences '
                'are then counted in SDs, beyond the range the elasticity '
                'bands were built for'),
}

WEIGHT_SEED = 1979

# weight_sweep.csv layout (and weight_sweep_standardized.csv's).
# ``lift30_default`` repeats the pair's lift at the default weights on every
# row, so the sign-flip count in summary.json can be recomputed from the
# file alone.
WEIGHT_COLUMNS = (['scenario', 'seed', 'draw', 'comparator']
                  + [f'w_{c}' for c in SPATIAL_CRITERIA]
                  + ['lift30', 'lift30_default'])


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-scenarios', type=int, default=12)
    ap.add_argument('--n-seeds', type=int, default=3)
    ap.add_argument('--n-draws', type=int, default=256)
    ap.add_argument('--n-items', type=int, default=10)
    ap.add_argument('--n-gens', type=int, default=12)
    ap.add_argument('--pop-size', type=int, default=20)
    ap.add_argument('--mc-iters', type=int, default=300)
    ap.add_argument('--mc-days', type=int, default=14)
    ap.add_argument('--workers', type=int, default=1,
                    help='Processes running scenarios in parallel')
    ap.add_argument('--n-weight-draws', type=int, default=256,
                    help='Spatial weight vectors drawn for the weight sweeps')
    ap.add_argument('--weight-seed', type=int, default=WEIGHT_SEED,
                    help='Seed of the weight-sweep generator')
    ap.add_argument('--n-spread-layouts', type=int, default=SPREAD_N_LAYOUTS,
                    help='Repaired random layouts behind each scenario\'s '
                         'criterion SDs and effective weights')
    ap.add_argument('--oracle-restarts', type=int,
                    default=ORACLE_RESTARTS,
                    help='L-BFGS-B restarts of the analytical reference '
                         '(oracle.solve_oracle); Figures A and B\'s default')
    ap.add_argument('--out-root', type=str,
                    default=os.path.join(_HERE, 'results'))
    args = ap.parse_args()
    if args.workers < 1:
        ap.error('--workers must be >= 1')
    if args.n_weight_draws < 1:
        ap.error('--n-weight-draws must be >= 1')
    if args.n_spread_layouts < 2:
        ap.error('--n-spread-layouts must be >= 2')
    if args.oracle_restarts < 1:
        ap.error('--oracle-restarts must be >= 1')
    return args


def _elasticities(ce, ie, be):
    return {'conv': ce, 'imp': ie, 'bsk': be}


def horizon_revenue(score, bkd, bp, elasticities=None, anchor=None,
                    basket_exclude=()):
    """Exact expected revenue over the projection horizon (MC_DAYS), under
    the given elasticities (default: the band midpoints). One call of
    ``closed_form.expected_revenue``, the function every re-scoring uses."""
    return expected_revenue(bp, MC_DAYS, score=score, breakdown=bkd,
                            elasticities=elasticities, anchor=anchor,
                            basket_exclude=basket_exclude)


def recompose_score(bkd, spatial_weights):
    """Composite layout score of a breakdown under other spatial weights.

    ``bkd`` is the per-criterion breakdown ``_ga_compute_layout_score``
    returns and ``spatial_weights`` the five weights in SPATIAL_CRITERIA
    order. Section compliance, accessibility and both penalties keep their
    retail_literature weights. The terms are summed in the order
    ``_ga_compute_layout_score`` sums them, so at the default weights the
    result is that function's own score, not an approximation of it."""
    w = spatial_weights
    return float(bkd['traffic']            * w[0]
                 + bkd['cross_merch']        * w[1]
                 + bkd['impulse']            * w[2]
                 + bkd['flow']               * w[3]
                 + bkd['revenue_placement']  * w[4]
                 + bkd['section_compliance'] * GA_W_SECTION_COMPLIANCE
                 + bkd['accessibility']      * GA_W_ACCESSIBILITY
                 - bkd['overlap_penalty']    * GA_PEN_OVERLAP
                 - bkd['bottleneck_penalty'] * GA_PEN_BOTTLENECK)


def standardization(spread, scale='mean_sd'):
    """``(factors, kappa)`` of a standardized scale from a
    ``criterion_spread`` record: each spatial criterion is multiplied by
    kappa / SD_k. With ``scale='mean_sd'`` kappa is the mean reference SD of
    the spatial criteria that vary; with ``scale='unit'`` it is 1 (see
    STANDARDIZED_SCALES). A criterion with no spread gets factor 0 (it is
    constant over the layouts and cancels against the anchor). kappa moves
    the lifts, not only the composite's magnitude: the terms it does not
    scale -- accessibility, the penalties, and the raw sub-scores revenue
    reads outside the composite -- keep their size."""
    if scale not in STANDARDIZED_SCALES:
        raise ValueError(f'unknown standardized scale {scale!r}')
    sds = [float(spread['sd'][c]) for c in SPATIAL_CRITERIA]
    varying = [sd for sd in sds if sd > SD_EPS]
    if scale == 'unit':
        kappa = 1.0 if varying else 0.0
    else:
        kappa = float(np.mean(varying)) if varying else 0.0
    factors = tuple((kappa / sd) if sd > SD_EPS else 0.0 for sd in sds)
    return factors, kappa


def recompose_standardized(bkd, spatial_weights, factors):
    """Composite of a breakdown on the standardized scale: spatial criterion
    k enters as ``bkd[k] * factors[k]`` with weight ``spatial_weights[k]``;
    section compliance, accessibility and the penalties as in
    ``recompose_score``."""
    w = spatial_weights
    return float(sum(bkd[c] * f * wk
                     for c, f, wk in zip(SPATIAL_CRITERIA, factors, w))
                 + bkd['section_compliance'] * GA_W_SECTION_COMPLIANCE
                 + bkd['accessibility']      * GA_W_ACCESSIBILITY
                 - bkd['overlap_penalty']    * GA_PEN_OVERLAP
                 - bkd['bottleneck_penalty'] * GA_PEN_BOTTLENECK)


def check_recomposition(score, bkd, tol=RECOMPOSE_TOL):
    """Raise AssertionError unless recomposing ``bkd`` at the default
    weights gives back ``score``.

    A recomposer that drifted from the composite (a criterion added,
    renamed or re-weighted in viz_ga) would otherwise turn the weight sweep
    into a measurement of the drift. Raised explicitly so ``python -O``
    cannot strip the check."""
    got = recompose_score(bkd, DEFAULT_SPATIAL_WEIGHTS)
    if not abs(got - float(score)) <= tol:
        raise AssertionError(
            f'recomposed composite {got!r} differs from the layout score '
            f'{float(score)!r} by more than {tol:g}; the weight sweep would '
            f'measure the recomposer rather than the weights')


def draw_spatial_weights(n_draws, seed):
    """``n_draws`` x 5 spatial weight vectors, uniform on the simplex scaled
    to SPATIAL_TOTAL: Dirichlet(1,1,1,1,1) x 0.80, from a generator of its
    own so the draws depend on ``seed`` alone."""
    rng = np.random.default_rng(seed)
    return rng.dirichlet(np.ones(len(SPATIAL_CRITERIA)),
                         size=n_draws) * SPATIAL_TOTAL


def _anchor_under(bp, recompose):
    """The run's anchor re-scored by ``recompose`` (a breakdown -> score
    map). An explicit zero anchor (``layout_objective.zero_anchor``), which
    has no breakdown, stays zero; parameters with no anchor at all raise
    ``MissingAnchorError``."""
    anc = anchor_of(bp)
    b0 = anc.get('breakdown') or {}
    if not b0:
        return anc
    return make_anchor(recompose(b0), b0, anc.get('source'))


def weight_sweep_lifts(scores, weight_draws, bp, factors=None):
    """GA-minus-comparator horizon lifts under each spatial weight vector.

    ``scores`` maps layout name -> ``(score, breakdown)`` as
    ``_ga_compute_layout_score`` returned them, 'GA' and every entry of
    COMPARATORS among them. Every layout's recomposition (and the anchor's)
    is checked before any draw is evaluated. Revenue is
    ``horizon_revenue`` at the elasticity midpoints with the recomposed
    composite, the breakdown unchanged and the anchor recomposed under the
    same weights, so impulse and abandonment read their own sub-scores
    exactly as in the elasticity design and the anchor layout reproduces
    the calibrated inputs under every weighting.

    With ``factors`` (from ``standardization``) the composite is the
    standardized one and the default point is equal weights on it.

    Returns ``{comparator: (lift at the default weights, array of lifts,
    one per row of weight_draws)}``."""
    for s, b in scores.values():
        check_recomposition(s, b)
    b0 = anchor_of(bp).get('breakdown') or {}
    if b0:
        check_recomposition(anchor_of(bp)['score'], b0)

    if factors is None:
        def recompose(b, w):
            return recompose_score(b, w)
    else:
        def recompose(b, w):
            return recompose_standardized(b, w, factors)

    def _rev(b, w, anchor):
        return horizon_revenue(recompose(b, w), b, bp, anchor=anchor)

    def _all(w):
        anchor = _anchor_under(bp, lambda b: recompose(b, w))
        return {name: _rev(b, w, anchor) for name, (_, b) in scores.items()}

    at_default = _all(DEFAULT_SPATIAL_WEIGHTS)
    per_draw = [_all(w) for w in weight_draws]
    swept = {name: np.array([r[name] for r in per_draw], dtype=float)
             for name in scores}
    return {m: (float(at_default['GA'] - at_default[m]),
                swept['GA'] - swept[m])
            for m in COMPARATORS}


def weight_sweep_summary(weight_rows):
    """Per-comparator results of a weight sweep from its CSV rows
    (WEIGHT_COLUMNS order).

    The lift percentiles pool every (scenario, seed, draw). A (scenario,
    seed) pair counts in ``n_pairs_sign_flip`` when its lift is strictly
    positive somewhere and strictly negative somewhere over the default
    weights and the drawn ones, i.e. the ranking of that pair's two
    layouts depends on how the criteria are weighted. The counts behind
    the fraction (``n_positive`` of ``n_values``) and the pairs whose lift
    keeps one sign throughout are reported beside it."""
    col = {c: i for i, c in enumerate(WEIGHT_COLUMNS)}
    out = {}
    for m in COMPARATORS:
        mine = [r for r in weight_rows if r[col['comparator']] == m]
        lifts = np.array([r[col['lift30']] for r in mine], dtype=float)
        by_pair = {}
        for r in mine:
            key = (r[col['scenario']], r[col['seed']])
            by_pair.setdefault(key, [r[col['lift30_default']]]).append(
                r[col['lift30']])
        out[m] = {
            'frac_positive': float((lifts > 0).mean()),
            'n_positive': int((lifts > 0).sum()),
            'median': float(np.median(lifts)),
            'p5': float(np.percentile(lifts, 5)),
            'p95': float(np.percentile(lifts, 95)),
            'n_pairs_sign_flip': int(sum(1 for v in by_pair.values()
                                         if max(v) > 0 and min(v) < 0)),
            'n_pairs_positive_at_default': int(sum(
                1 for v in by_pair.values() if v[0] > 0)),
            'n_pairs_always_positive': int(sum(
                1 for v in by_pair.values() if min(v) > 0)),
            'n_pairs_always_negative': int(sum(
                1 for v in by_pair.values() if max(v) < 0)),
            'n_pairs': len(by_pair),
            'n_values': int(lifts.size),
        }
    return out


def _stats(values):
    v = np.asarray(values, dtype=float)
    return {'n': int(v.size), 'n_positive': int((v > 0).sum()),
            'frac_positive': float((v > 0).mean()) if v.size else float('nan'),
            'median': float(np.median(v)) if v.size else float('nan'),
            'min': float(v.min()) if v.size else float('nan'),
            'max': float(v.max()) if v.size else float('nan')}


def pair_breakdown(rows, comparator):
    """Per (scenario, seed) pair: how many LHS draws put the GA ahead of
    ``comparator`` and the range of the lift across the design, in money
    and in percent of the comparator's revenue. The pooled fraction in
    'comparisons' is the sum of ``n_positive`` over the sum of ``n_draws``.
    Returns ``(per_pair, aggregate)``."""
    i_sc, i_seed = COLUMNS.index('scenario'), COLUMNS.index('seed')
    lift_col = COLUMNS.index('lift30' if comparator == COMPARATORS[0]
                             else f'lift30_{comparator}')
    rev_col = COLUMNS.index('rev30_baseline' if comparator == COMPARATORS[0]
                            else f'rev30_{comparator}')
    by_pair = {}
    for r in rows:
        by_pair.setdefault((r[i_sc], r[i_seed]), []).append(
            (float(r[lift_col]), float(r[rev_col])))
    per_pair = []
    for (sc, seed), vals in sorted(by_pair.items()):
        lifts = np.array([v[0] for v in vals])
        pct = np.array([v[0] / max(abs(v[1]), 1e-9) * 100.0 for v in vals])
        per_pair.append({
            'scenario': int(sc), 'seed': int(seed),
            'n_draws': int(lifts.size),
            'n_positive': int((lifts > 0).sum()),
            'lift_min': float(lifts.min()),
            'lift_median': float(np.median(lifts)),
            'lift_max': float(lifts.max()),
            'lift_range': float(lifts.max() - lifts.min()),
            'pct_min': float(pct.min()),
            'pct_median': float(np.median(pct)),
            'pct_max': float(pct.max()),
        })
    ranges = np.array([p['lift_range'] for p in per_pair])
    rel = np.array([p['lift_range'] / max(abs(p['lift_median']), 1e-9)
                    for p in per_pair])
    pct_ranges = np.array([p['pct_max'] - p['pct_min'] for p in per_pair])
    aggregate = {
        'n_pairs': len(per_pair),
        'n_pairs_all_draws_positive': int(sum(
            1 for p in per_pair if p['n_positive'] == p['n_draws'])),
        'n_pairs_all_draws_nonpositive': int(sum(
            1 for p in per_pair if p['n_positive'] == 0)),
        'n_pairs_mixed_sign': int(sum(
            1 for p in per_pair if 0 < p['n_positive'] < p['n_draws'])),
        'n_positive': int(sum(p['n_positive'] for p in per_pair)),
        'n_draws_total': int(sum(p['n_draws'] for p in per_pair)),
        # Within-pair spread of the lift across the elasticity design:
        # what the bands do to the size of the effect.
        'within_pair_range_median': float(np.median(ranges)) if ranges.size else float('nan'),
        'within_pair_range_max': float(ranges.max()) if ranges.size else float('nan'),
        'within_pair_range_rel_median': float(np.median(rel)) if rel.size else float('nan'),
        'within_pair_pct_range_median': float(np.median(pct_ranges)) if pct_ranges.size else float('nan'),
        'within_pair_pct_range_max': float(pct_ranges.max()) if pct_ranges.size else float('nan'),
    }
    return per_pair, aggregate


def lhs_scenario(sc, args, draws, weight_draws, t0):
    """Results of one scenario: the elasticity rows (every (seed, draw)
    pair), the weight-sweep rows (every (seed, weight draw, comparator)),
    the standardized weight-sweep rows, and a record of the scenario's
    effective weights, base parameters and per-pair closed-form extras
    (basket corners, short-path variant). The standardized rows are a dict,
    one list per STANDARDIZED_SCALES entry."""
    rows = []
    weight_rows = []
    std_weight_rows = {name: [] for name in STANDARDIZED_SCALES}
    scenario = generate_synthetic_shop(
        name=f'lhs_{sc:03d}', seed=10_000 + sc, n_items=args.n_items)
    # The analytical reference depends on the scenario alone (its solver
    # starts are seeded from it), so it is solved once for every seed, with
    # as many restarts as Figures A and B give it.
    oracle_result = solve_oracle(scenario, n_restarts=args.oracle_restarts)
    oracle_layout = oracle_result.layout
    item_names = [it.name for it in scenario.items]
    # The reference spread of each criterion, over repaired random layouts
    # of this scenario (the shop is the same for every seed).
    spread = criterion_spread(build_headless_shop(scenario), item_names,
                              n_layouts=args.n_spread_layouts,
                              seed=SPREAD_SEED + sc)
    scales = {name: standardization(spread, name)
              for name in STANDARDIZED_SCALES}
    extra = {'scenario': sc,
             'oracle': oracle_record(oracle_result),
             'effective_weights': spread,
             'standardization': {
                 name: {'factors': dict(zip(SPATIAL_CRITERIA, factors)),
                        'kappa': kappa}
                 for name, (factors, kappa) in scales.items()},
             'pairs': []}
    for seed in range(args.n_seeds):
        shop = build_headless_shop(scenario)
        init_layout = {n: tuple(shop.floors[1]['items'][n]['position'])
                       for n in item_names}
        # Anchored at the as-built layout after the shared repair: the
        # layout every search starts from reproduces the calibrated inputs.
        bp = anchor_base_params(shop, item_names, base_params_for(scenario),
                                init_layout)
        if seed == 0:
            extra['base_params'] = base_params_record(bp)

        ga_out = run_ga_headless(
            shop, item_names, bp,
            n_gens=args.n_gens, pop_size=args.pop_size,
            mc_iters=args.mc_iters, mc_days=args.mc_days,
            rng_seed=seed, init_layout=init_layout)

        # Every comparator is scored on the GA's feasible set, like every
        # GA candidate, so a lift is not a by-product of the repair itself.
        # The two searches get the GA's own evaluation budget and seed
        # block, as in the paired-MC comparison they come from, and start
        # from the as-built snapshot the GA is seeded from: random search
        # spends its first evaluation on it and SA anneals away from it, so
        # no search is handed a better starting point than the others.
        budget = args.pop_size * args.n_gens
        layouts = {
            'GA': chromosome_to_layout(ga_out['best_chrom'], item_names),
            'popularity_rank': popularity_rank(scenario, seed=seed),
            'oracle': oracle_layout,
            'random_search': random_search(
                shop, scenario, item_names, bp, seed=seed, budget=budget,
                block=args.pop_size, mc_iters=args.mc_iters,
                mc_days=args.mc_days, init_layout=init_layout),
            'simulated_annealing': simulated_annealing(
                shop, scenario, item_names, bp, seed=seed, budget=budget,
                block=args.pop_size, mc_iters=args.mc_iters,
                mc_days=args.mc_days, init_layout=init_layout,
                start='asbuilt'),
        }
        scores = {}
        for name, lay in layouts.items():
            lay = feasible_layout(shop, item_names, lay)
            # Every reported layout keeps the floor-plan invariants.
            checked_layout(shop, item_names, lay,
                           f"lhs scenario {sc} seed {seed} {name}")
            if name == 'oracle':
                # Held to zero overlap as in the paired-MC comparison: the
                # layout score penalizes any positive overlap, so solver
                # contact residue would sink the reference on a technicality.
                assert_no_strict_overlap(scenario, lay)
            chrom = layout_to_chromosome(lay, item_names)
            scores[name] = shop._ga_compute_layout_score(chrom, item_names, bp)

        for di, (ce, ie, be) in enumerate(draws):
            e = _elasticities(ce, ie, be)
            rev = {name: horizon_revenue(s, b, bp, e)
                   for name, (s, b) in scores.items()}
            r_ga = rev['GA']
            r_bl = rev['popularity_rank']
            pct = (r_ga / max(r_bl, 1e-9) - 1.0) * 100.0
            rows.append((sc, seed, di, ce, ie, be, r_ga,
                         *(rev[m] for m in COMPARATORS),
                         r_ga - r_bl, pct,
                         *(r_ga - rev[m] for m in COMPARATORS[1:])))

        # Basket corners and the short-path variant, in closed form on the
        # same layouts.
        mid = dict(ELASTICITY_MIDPOINTS)

        def _lifts(**kw):
            rev = {name: horizon_revenue(s, b, bp, **kw)
                   for name, (s, b) in scores.items()}
            return {m: float(rev['GA'] - rev[m]) for m in COMPARATORS}

        pair = {'scenario': sc, 'seed': seed,
                'midpoint': _lifts(elasticities=mid),
                'basket_corners': {
                    f'{e_b:g}': _lifts(elasticities=dict(mid, bsk=e_b))
                    for e_b in BASKET_CORNERS},
                'short_path_variant': {
                    'midpoint': _lifts(elasticities=mid,
                                       basket_exclude=SHORT_PATH_CRITERIA)}}
        lhs_short = {m: [] for m in COMPARATORS}
        for ce, ie, be in draws:
            for m, v in _lifts(elasticities=_elasticities(ce, ie, be),
                               basket_exclude=SHORT_PATH_CRITERIA).items():
                lhs_short[m].append(v)
        pair['short_path_variant']['lhs'] = {m: _stats(v)
                                             for m, v in lhs_short.items()}
        extra['pairs'].append(pair)

        # Weight sweeps: the same scored layouts, re-weighted. No layout is
        # searched or scored again; weight_sweep_lifts checks the
        # recomposition against these scores before it evaluates a draw.
        designs = [(weight_rows, None)] + [
            (std_weight_rows[name], factors)
            for name, (factors, _) in scales.items()]
        for target, fac in designs:
            sweep = weight_sweep_lifts(scores, weight_draws, bp, factors=fac)
            for wi, w in enumerate(weight_draws):
                for m in COMPARATORS:
                    at_default, lifts = sweep[m]
                    target.append((sc, seed, wi, m, *(float(x) for x in w),
                                   float(lifts[wi]), at_default))
    print(f'[lhs] scenario {sc + 1}/{args.n_scenarios} done '
          f'({time.time() - t0:.1f}s)', flush=True)
    return rows, weight_rows, std_weight_rows, extra


def _map_scenarios(fn, n_scenarios, workers, *fn_args):
    """``[fn(s, *fn_args) for s in range(n_scenarios)]``, in scenario order.

    With ``workers`` > 1 the scenarios run in separate processes. They are
    independent (each builds its own shop and reseeds every RNG it draws
    from), and results are put back in scenario order, so the output is
    identical to the serial run."""
    if workers <= 1:
        return [fn(s, *fn_args) for s in range(n_scenarios)]
    done = {}
    pool = ProcessPoolExecutor(max_workers=workers)
    try:
        futures = {pool.submit(fn, s, *fn_args): s for s in range(n_scenarios)}
        for fut in as_completed(futures):
            done[futures[fut]] = fut.result()
    except BaseException:
        # Leaving the pool's context manager would wait for every queued
        # scenario first, so a scenario that fails minutes into a run of
        # hours would only report at the end. Drop what has not started and
        # let the error out now.
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown(wait=True)
    return [done[s] for s in sorted(done)]


def effective_weight_summary(extras):
    """Per-scenario effective weights and their mean over scenarios."""
    per_scenario = [{'scenario': x['scenario'],
                     'n_layouts': x['effective_weights']['n_layouts'],
                     'seed': x['effective_weights']['seed'],
                     'sd': x['effective_weights']['sd'],
                     'effective_weight': x['effective_weights']['effective_weight'],
                     'effective_share': x['effective_weights']['effective_share'],
                     'standardization': x['standardization']}
                    for x in extras]
    crit = list(per_scenario[0]['effective_share']) if per_scenario else []
    return {
        'definition': ('effective weight = |nominal weight| x SD of the '
                       'criterion over repaired random layouts of the '
                       'scenario; share = effective weight over the sum '
                       'of the positive criteria\'s'),
        'sampler': 'zone_sampler + feasible_layout',
        'nominal_weight': (dict(extras[0]['effective_weights']['weight'])
                           if extras else {}),
        'mean_share': {c: float(np.mean([p['effective_share'][c]
                                         for p in per_scenario]))
                       for c in crit},
        'mean_sd': {c: float(np.mean([p['sd'][c] for p in per_scenario]))
                    for c in (per_scenario[0]['sd'] if per_scenario else {})},
        'per_scenario': per_scenario,
    }


def closed_form_extras_summary(extras):
    """The basket corners and the short-path variant, per comparator:
    counts, fractions and ranges over the (scenario, seed) pairs (and over
    the LHS draws for the variant), with each pair's values."""
    pairs = [p for x in extras for p in x['pairs']]
    out = {'n_pairs': len(pairs), 'basket_corners': {},
           'short_path_variant': {}}
    for e_b in BASKET_CORNERS:
        key = f'{e_b:g}'
        out['basket_corners'][key] = {
            'elasticities': dict(ELASTICITY_MIDPOINTS, bsk=e_b),
            'comparisons': {m: _stats([p['basket_corners'][key][m]
                                       for p in pairs])
                            for m in COMPARATORS}}
    out['midpoint'] = {m: _stats([p['midpoint'][m] for p in pairs])
                       for m in COMPARATORS}
    out['short_path_variant'] = {
        'basket_exclude': list(SHORT_PATH_CRITERIA),
        'note': ('flow and entrance proximity taken out of the basket '
                 'driver b(s) only; conversion, impulse and abandonment '
                 'unchanged'),
        'midpoint': {m: _stats([p['short_path_variant']['midpoint'][m]
                                for p in pairs]) for m in COMPARATORS},
        'lhs': {m: {'n_positive': int(sum(
                        p['short_path_variant']['lhs'][m]['n_positive']
                        for p in pairs)),
                    'n': int(sum(p['short_path_variant']['lhs'][m]['n']
                                 for p in pairs)),
                    'n_pairs_all_positive': int(sum(
                        1 for p in pairs
                        if p['short_path_variant']['lhs'][m]['n_positive']
                        == p['short_path_variant']['lhs'][m]['n'])),
                    'min': float(min(p['short_path_variant']['lhs'][m]['min']
                                     for p in pairs)),
                    'max': float(max(p['short_path_variant']['lhs'][m]['max']
                                     for p in pairs))}
                for m in COMPARATORS},
    }
    for m, st in out['short_path_variant']['lhs'].items():
        st['frac_positive'] = st['n_positive'] / st['n'] if st['n'] else float('nan')
    out['per_pair'] = pairs
    return out


def _weight_design(n_draws, seed, scale=None):
    """The design record of a weight sweep: raw (``scale`` None) or on the
    standardized scale ``scale`` (a STANDARDIZED_SCALES key)."""
    design = {
        'n_weight_draws': int(n_draws),
        'seed': int(seed),
        'distribution': ('Dirichlet(1,1,1,1,1) x spatial_total over '
                         'the spatial criteria'),
        'spatial_criteria': list(SPATIAL_CRITERIA),
        'spatial_total': SPATIAL_TOTAL,
        'default_spatial_weights': dict(zip(SPATIAL_CRITERIA,
                                            DEFAULT_SPATIAL_WEIGHTS)),
        'fixed': {
            'section_compliance': GA_W_SECTION_COMPLIANCE,
            'accessibility': GA_W_ACCESSIBILITY,
            'overlap_penalty': GA_PEN_OVERLAP,
            'bottleneck_penalty': GA_PEN_BOTTLENECK,
            'elasticities': dict(ELASTICITY_MIDPOINTS),
            'layouts': ('searched under the default weights and '
                        're-scored from their breakdowns; no '
                        'search is re-run'),
        },
        'anchor': ('the scenario\'s repaired as-built layout, re-scored '
                   'under the same weights as the layout'),
        'recompose_tolerance': RECOMPOSE_TOL,
        'sign_flip': ('lift strictly positive and strictly negative '
                      'somewhere over the default weights and the '
                      'drawn ones'),
    }
    if scale is not None:
        design['scale'] = {
            'name': scale,
            'transform': ('spatial criterion k enters as x_k * kappa / SD_k, '
                          'SD_k its reference SD over repaired random '
                          'layouts of the scenario; a criterion with no '
                          'spread contributes nothing'),
            'kappa': STANDARDIZED_SCALES[scale][2],
            'kappa_per_scenario': ('effective_weights.per_scenario[]'
                                   f'.standardization.{scale}.kappa'),
            'kappa_role': ('not a neutral common factor: kappa scales the '
                           'five standardized spatial terms only, while '
                           'accessibility, section compliance and the '
                           'penalties stay raw and revenue also reads the raw '
                           'impulse, flow, section and bottleneck sub-scores '
                           'outside the composite; it therefore changes the '
                           'size and the sign of the lifts and the sign-flip '
                           'counts, which is why the sweep is reported at '
                           'both kappas'),
        }
        design['default_point'] = 'equal weights on the standardized criteria'
    return design


def main():
    args = parse_args()
    t0 = time.time()
    out_dir = make_run_dir(args.out_root, 'elasticity_lhs')

    sampler = qmc.LatinHypercube(d=3, seed=123)
    unit = sampler.random(args.n_draws)
    lo = np.array([BANDS['conv'][0], BANDS['imp'][0], BANDS['bsk'][0]])
    hi = np.array([BANDS['conv'][1], BANDS['imp'][1], BANDS['bsk'][1]])
    draws = qmc.scale(unit, lo, hi)
    # Drawn once, here, and shared by every scenario and seed, so each
    # (scenario, seed) pair is re-weighted by the same vectors and the
    # draws do not depend on --workers.
    weight_draws = draw_spatial_weights(args.n_weight_draws, args.weight_seed)

    per_scenario = _map_scenarios(lhs_scenario, args.n_scenarios,
                                  args.workers, args, draws, weight_draws, t0)
    rows = [row for r in per_scenario for row in r[0]]
    weight_rows = [row for r in per_scenario for row in r[1]]
    std_weight_rows = {name: [row for r in per_scenario for row in r[2][name]]
                       for name in STANDARDIZED_SCALES}
    extras = [r[3] for r in per_scenario]

    def _lift_stats(values):
        v = np.asarray(values, dtype=float)
        return {'frac_positive': float((v > 0).mean()),
                'n_positive': int((v > 0).sum()),
                'lift_median': float(np.median(v)),
                'lift_p5': float(np.percentile(v, 5)),
                'lift_p95': float(np.percentile(v, 95)),
                'n_draws_total': int(v.size)}

    def _column(name):
        i = COLUMNS.index(name)
        return [row[i] for row in rows]

    # One entry per headline comparison, so the bands are swept over the
    # differences the paper reports and not only over GA-minus-popularity.
    # Each carries the pair counts behind its fraction and the within-pair
    # ranges of the lift across the design.
    comparisons = {}
    for m in COMPARATORS:
        st = _lift_stats(_column('lift30' if m == COMPARATORS[0]
                                 else f'lift30_{m}'))
        per_pair, agg = pair_breakdown(rows, m)
        st['pairs'] = agg
        st['per_pair'] = per_pair
        comparisons[m] = st
    lifts = np.array(_column('lift30'), dtype=float)
    summary = {
        'n_draws_total': int(lifts.size),
        # Resolution of the LHS design in the 3-D elasticity box. The
        # total above multiplies this by scenarios x seeds, which is
        # replication, not additional coverage of the coefficient space
        # -- keep them distinct so the paper can quote the right one.
        'n_lhs_points': int(args.n_draws),
        'n_scenarios': int(args.n_scenarios),
        'n_seeds': int(args.n_seeds),
        'n_pairs': int(args.n_scenarios * args.n_seeds),
        # Top-level figures stay the GA-minus-popularity comparison this
        # sweep has always reported; the rest are under 'comparisons'.
        'frac_positive': float((lifts > 0).mean()),
        'n_positive': int((lifts > 0).sum()),
        'lift_median': float(np.median(lifts)),
        'lift_p5': float(np.percentile(lifts, 5)),
        'lift_p95': float(np.percentile(lifts, 95)),
        'comparisons': comparisons,
        'horizon_days': MC_DAYS,
        'horizon_customer_hours': float(HORIZON_CUST_HOURS),
        'bands': {k: list(v) for k, v in BANDS.items()},
        'objective': {
            'revenue': 'experiments.closed_form.expected_revenue (exact '
                       'mean of the MC fitness; the engine\'s spend law '
                       'keeps the mean, so there is no floor term)',
            'mc_spend_law': MC_SPEND_LAW,
            'anchoring': ('elasticities act on the score difference from '
                          'the scenario\'s repaired as-built layout'),
            'elasticity_midpoints': dict(ELASTICITY_MIDPOINTS),
        },
        'sign_argument': ('conversion and basket both rise with the '
                          'composite score, so the sign of a lift is to '
                          'first order the sign of the score difference; '
                          'the per-pair counts and within-pair ranges show '
                          'what the bands do to its size'),
        'closed_form_extras': closed_form_extras_summary(extras),
        'effective_weights': effective_weight_summary(extras),
        # The analytical reference the 'oracle' comparisons are against:
        # its restarts (Figures A and B's default) and any that failed.
        'oracle': oracle_summary([x['oracle'] for x in extras],
                                 args.oracle_restarts),
        'config': vars(args),
        # The second design: the same layouts under other weightings of the
        # five spatial criteria. The layouts were searched under the
        # default weights, so this tests whether their ranking survives
        # re-weighting when scored, not whether a search under other
        # weights would rank the methods the same way.
        'weight_sweep': {
            'design': _weight_design(args.n_weight_draws, args.weight_seed),
            'n_pairs': int(args.n_scenarios * args.n_seeds),
            'comparisons': weight_sweep_summary(weight_rows),
        },
    }
    # The same weights on the standardized criteria, where equal weights are
    # Dawes' unit weights, at both kappas (STANDARDIZED_SCALES).
    for name, (key, _, _) in STANDARDIZED_SCALES.items():
        summary[key] = {
            'design': _weight_design(args.n_weight_draws, args.weight_seed,
                                     scale=name),
            'n_pairs': int(args.n_scenarios * args.n_seeds),
            'comparisons': weight_sweep_summary(std_weight_rows[name]),
        }

    with open(os.path.join(out_dir, 'results.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        w.writerows(rows)
    csvs = [('weight_sweep.csv', weight_rows)] + [
        (fname, std_weight_rows[name])
        for name, (_, fname, _) in STANDARDIZED_SCALES.items()]
    for name, data in csvs:
        with open(os.path.join(out_dir, name), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(WEIGHT_COLUMNS)
            w.writerows(data)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.hist(lifts, bins=50, color='#4E79A7', edgecolor='white')
        ax.axvline(0, color='crimson', lw=1.2)
        ax.set_xlabel('30-day projected lift, GA − popularity baseline ($)')
        ax.set_ylabel('LHS draws')
        ax.set_title(f'Elasticity-band robustness: {summary["frac_positive"]*100:.1f}% '
                     f'of draws positive (n={lifts.size})')
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'lhs_hist.png'), dpi=130)
    except Exception as e:
        print('[lhs] plot skipped:', e)

    write_sidecar(out_dir, {'kind': 'elasticity_lhs', 'summary': summary,
                            'workers': args.workers,
                            'wall_seconds': time.time() - t0,
                            'base_params': {str(x['scenario']): x['base_params']
                                            for x in extras}})
    # Written last: its presence marks a finished run.
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(portable_paths(summary), f, indent=2)
    for m, st in comparisons.items():
        print(f'[lhs] GA - {m:20s} positive in {st["frac_positive"]*100:5.1f}% '
              f'of draws ({st["n_positive"]}/{st["n_draws_total"]}; '
              f'{st["pairs"]["n_pairs_all_draws_positive"]}/'
              f'{st["pairs"]["n_pairs"]} pairs always ahead)  '
              f'median ${st["lift_median"]:,.0f}  '
              f'p5..p95 = [${st["lift_p5"]:,.0f}, ${st["lift_p95"]:,.0f}]')
    for key in ['weight_sweep'] + [k for k, _, _ in
                                   STANDARDIZED_SCALES.values()]:
        for m, st in summary[key]['comparisons'].items():
            print(f'[{key}] GA - {m:20s} positive in '
                  f'{st["frac_positive"]*100:5.1f}% of draws  '
                  f'median ${st["median"]:,.0f}  '
                  f'p5..p95 = [${st["p5"]:,.0f}, ${st["p95"]:,.0f}]  '
                  f'sign flips in {st["n_pairs_sign_flip"]}/{st["n_pairs"]} pairs')
    print('Artifacts in:', out_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())
