"""Input uncertainty of Figure C's lift (review R11).

Figure C's intervals cover Monte Carlo evaluation noise, and
``run_real_data_seeds`` covers the search-to-search spread. Neither covers
the calibration's own sampling error: the store's arrival rate, spend,
co-purchase pairs, popularity and category revenue are estimates from one
sample of invoices, and the lift a layout earns is a function of them.
This runner measures that layer.

Design:
  * Figure C's store, built as Figure C builds it (the same options and
    defaults: ``run_real_data_example.add_arguments`` and ``build_store``),
    and Figure C's three equal-budget searches run once, at Figure C's
    settings and seed (``run_real_data_example.run_searches``: the GA,
    random search and simulated annealing, all from the as-built layout;
    the GA's layout is the one Figure C's GA finds). The four layouts --
    the repaired as-built one, the GA's, random search's and annealing's
    -- are then FIXED: the bootstrap re-scores them, it does not re-search.
  * A customer-clustered bootstrap of the invoices: B replicates, each
    drawing the data's customers with replacement (an anonymous invoice,
    which names no customer, is a cluster of its own), keeping all of a
    drawn customer's invoice lines, and giving each copy of a customer
    fresh invoice and customer ids so copies stay distinct invoices.
    Invoices of one customer are not independent -- the same buyer's
    habits recur -- so resampling invoices one by one would understate the
    spread (the review's estimate: 1.6% of the level iid against 4.5%
    clustered). Replicate b draws only from a generator seeded with
    ``[--boot-seed, b]``, so the replicates do not depend on each other or
    on ``--workers``.
  * Each replicate is recalibrated by the same ``calibrate_transactional``
    call, re-keyed onto the SAME fixtures (``experiments._rekey``: the
    store's per-item statistics and base parameters come from the
    replicate, the anchor is the as-built layout re-scored under them) and
    every layout is scored in closed form
    (``experiments.closed_form.expected_revenue``, the exact mean of the
    Monte Carlo fitness), so no Monte Carlo noise enters.

Reported (``summary.json``):
  * ``point``: the full-data values -- the as-built revenue level, each
    searched layout's, the GA's lift and percentage lift, and the GA's
    margins over random search and annealing -- on two scales: WHOLE
    (Figure C's: whole invoices across the wholesaler's catalogue, every
    invoice a buyer) and STOCKED (``_rekey.stocked_scale``: each invoice
    cut to the store's products, buyers the invoices holding one). Every
    percentage is the same on both by construction; the pounds differ by
    the ratio ``scale_ratio``.
  * ``input_uncertainty``: per scale, the percentage lift's percentile
    bootstrap interval (with the bootstrap SE and bias), and delta-method
    intervals for the GBP level and lift. The level is linear in the
    data's total revenue over the calendar span (the as-built layout
    reproduces the calibration exactly), so its SE is the customer-
    clustered SE of that total; the lift is level x fraction, and its SE
    combines the level's with the bootstrap variance of the fraction and
    their bootstrap covariance. The bootstrap's own percentile intervals
    for the GBP figures are recorded beside them. The GA's margins over
    the comparators (``ga_minus_rs``, ``ga_minus_sa``) get the same two
    intervals: the percentage (of the comparator's revenue, Figure C's
    ``pct_diff`` convention) by percentile bootstrap, the GBP margin by
    the delta method.
  * ``uncertainty_layers``: which uncertainty each interval of the paper
    covers -- input (this runner), Monte Carlo (Figure C's paired
    replicates; none here), search seed (``run_real_data_seeds``; not
    here: the layouts are fixed).
  * ``anonymous``: the anonymous invoices' shares of the store's stocked
    invoices, purchases and spend (review R27).
  * ``legacy_floor_bias``: how far the former floored spend law raised
    each layout's expected revenue and the lift, at two sets of inputs:
    ``current_inputs`` (this run's calibration, adapter 1.2 cleaning) and
    ``adapter_1_1_inputs`` (the same rows cleaned as adapter 1.1 cleaned
    them -- cancellation lines dropped, the purchases they reverse kept --
    re-keyed onto this run's fixtures). The published Figure C numbers
    were made under 1.1 inputs, where the invoice spend is more dispersed
    and the floor bit harder; their own layouts were not stored, so the
    second block scores this run's layouts, a proxy for theirs.

Outputs (under ``--out-root``, one ``input_uncertainty_<time>`` directory;
``noanon_input_uncertainty_<time>`` for an ``--exclude-anonymous``
sensitivity run, so it cannot stand in for the headline): results.csv (one
row per replicate), sidecar.json (provenance with the workbook's SHA-256,
base parameters, timings), summary.json (written last).

Smoke (one sheet, tiny searches, three replicates; about a minute after
the workbook read):
    python -m experiments.run_input_uncertainty ^
        --sheets "Year 2010-2011" --mc-iters 100 --mc-days 7 ^
        --n-gens 2 --pop-size 6 --n-boot 3 --out-root <scratch folder>

Paper-grade (Figure C's design, B = 200: Figure C's three searches once,
then about 7 s of calibration per replicate on both sheets):
    python -m experiments.run_input_uncertainty --workers 4
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from dataset_adapters import OnlineRetailIIAdapter, anonymous_customer_mask  # noqa: E402
from dataset_calibration import (CalibratedParams,  # noqa: E402
                                 anonymous_placed_shares,
                                 calibrate_transactional)
from dataset_provenance import stamp as stamp_provenance  # noqa: E402
from experiments import _rekey as RK  # noqa: E402
from experiments._common import (aisle_rule_summary,  # noqa: E402
                                 base_params_record, layout_score,
                                 make_run_dir, provenance_snapshot,
                                 repair_stats, write_sidecar)
from experiments.closed_form import legacy_floor_bias  # noqa: E402
from experiments.run_real_data_example import (  # noqa: E402
    GA_ELITE_FRAC, GA_MUT_RATE, SA_STEP_FRAC, adapt_rows, add_arguments,
    build_store, calibrate_normalized, check_reported, check_search_arguments,
    data_provenance_extra, experiment_name, read_workbook, reported_layouts,
    resolve_workbook, run_dir_prefix, run_searches, search_operators)
from experiments.run_real_data_seeds import store_fingerprint  # noqa: E402
from layout_objective import layout_drivers, layout_mc_kwargs  # noqa: E402

#: Bootstrap replicates unless told otherwise.
DEFAULT_N_BOOT = 200
#: Root of every replicate's generator ([boot_seed, b]).
DEFAULT_BOOT_SEED = 20_251
#: Two-sided level of every interval reported.
CI_LEVEL = 0.95
#: Most worker processes the runner accepts (the workstation's thermal
#: limit; the output does not depend on the count).
MAX_WORKERS = 4

SCALES = ('whole', 'stocked')

#: The searched layouts, keyed as in ``point`` and the replicates: the GA
#: (``run_searches``' 'optimized'), random search and annealing.
LAYOUTS = ('ga', 'rs', 'sa')
#: ``run_searches``' key for each.
SEARCH_KEY = {'ga': 'optimized', 'rs': 'rs', 'sa': 'sa'}
#: The equal-budget comparators the GA's margins are taken over.
COMPARATORS = ('rs', 'sa')

UNCERTAINTY_LAYERS = {
    'input': ('this runner: customer-clustered bootstrap of the invoices, '
              'each replicate recalibrated and re-keyed onto the same '
              'fixtures, the layouts held fixed and scored in closed '
              'form'),
    'monte_carlo': ('not here (closed-form scoring has none); Figure C\'s '
                    'paired held-out replicates cover the evaluation noise '
                    'of its Monte Carlo lift and margins'),
    'search_seed': ('not here (the layouts are fixed, from one search per '
                    'method at --ga-seed); run_real_data_seeds covers the '
                    'search-to-search spread'),
}


# --- Arguments ---------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Customer-clustered bootstrap of Figure C's lift: the "
                    "calibration's sampling error, with the layouts fixed.")
    add_arguments(p)
    p.add_argument('--n-boot', type=int, default=DEFAULT_N_BOOT,
                   help="Bootstrap replicates B")
    p.add_argument('--boot-seed', type=int, default=DEFAULT_BOOT_SEED,
                   help="Replicate b draws from default_rng([boot_seed, b])")
    p.add_argument('--workers', type=int, default=1,
                   help=f"Processes the replicates are spread over (at most "
                        f"{MAX_WORKERS}); the output is the same for any value")
    args = p.parse_args(argv)
    if args.n_boot < 2:
        p.error("--n-boot must be >= 2")
    if not 1 <= args.workers <= MAX_WORKERS:
        p.error(f"--workers must be in [1, {MAX_WORKERS}]")
    if args.boot_seed < 0:
        p.error("--boot-seed must be >= 0")
    check_search_arguments(p, args)
    resolve_workbook(p, args)
    return args


# --- Clusters and resampling ---------------------------------------------------

@dataclass(frozen=True)
class Clusters:
    """Rows grouped by cluster: cluster c holds rows
    ``order[starts[c]:starts[c + 1]]`` (positions in the frame)."""
    order: np.ndarray
    starts: np.ndarray
    n_customers: int
    n_anonymous_invoices: int

    @property
    def n(self) -> int:
        return int(self.starts.size - 1)


def customer_clusters(df: pd.DataFrame) -> Clusters:
    """One cluster per identified customer, and one per anonymous invoice
    (no customer to group it with). Without a customer column every
    invoice is its own cluster. Deterministic in the frame's row order."""
    inv = df['invoice_id'].astype(str).reset_index(drop=True)
    if 'customer_id' in df.columns:
        anon = pd.Series(anonymous_customer_mask(df['customer_id'])
                         .to_numpy())
        cust = df['customer_id'].astype(str).reset_index(drop=True)
        key = ('customer:' + cust).where(~anon, 'invoice:' + inv)
        n_customers = int(cust[~anon].nunique())
        n_anon = int(inv[anon].nunique())
    else:
        key = 'invoice:' + inv
        n_customers, n_anon = 0, int(inv.nunique())
    codes, uniques = pd.factorize(key.to_numpy(), sort=True)
    order = np.argsort(codes, kind='stable')
    starts = np.zeros(len(uniques) + 1, dtype=np.int64)
    np.cumsum(np.bincount(codes, minlength=len(uniques)), out=starts[1:])
    return Clusters(order=order, starts=starts, n_customers=n_customers,
                    n_anonymous_invoices=n_anon)


def resample(df: pd.DataFrame, clusters: Clusters,
             rng: np.random.Generator) -> pd.DataFrame:
    """One bootstrap replicate: ``clusters.n`` clusters drawn with
    replacement, each drawn cluster's rows kept whole. A cluster drawn k
    times appears k times; copies after the first get '#<copy>' appended to
    their invoice ids and, for an identified customer, to the customer id,
    so a copy is a distinct invoice of a distinct customer rather than a
    doubled one. Anonymous rows keep their anonymous customer cell."""
    C = clusters.n
    counts = np.bincount(rng.integers(0, C, C), minlength=C)
    sel = np.repeat(np.arange(C), counts)
    first = np.cumsum(counts) - counts
    copy = np.arange(sel.size) - np.repeat(first, counts)
    starts, sizes = clusters.starts[sel], np.diff(clusters.starts)[sel]
    offset = np.arange(int(sizes.sum())) - np.repeat(np.cumsum(sizes) - sizes,
                                                     sizes)
    rows = clusters.order[np.repeat(starts, sizes) + offset]
    copy_row = np.repeat(copy, sizes)
    out = df.iloc[rows].reset_index(drop=True)
    dup = pd.Series(copy_row > 0)
    if dup.any():
        tag = '#' + pd.Series(copy_row).astype(str)
        inv = out['invoice_id'].astype(str)
        out['invoice_id'] = inv.where(~dup, inv + tag)
        if 'customer_id' in out.columns:
            anon = pd.Series(anonymous_customer_mask(out['customer_id'])
                             .to_numpy())
            cust = out['customer_id'].astype(str)
            out['customer_id'] = cust.where(~(dup & ~anon), cust + tag)
    return out


# --- Scoring ---------------------------------------------------------------------

def score_fixed_layouts(store, params: CalibratedParams,
                        layouts: Mapping[str, Any], product_ids,
                        horizon_days: int) -> Dict[str, Any]:
    """The as-built layout's and each of ``layouts``' exact revenue on
    ``store`` (already carrying ``params``) on both scales, with the anchor
    score.

    ``layouts`` maps 'ga' (required) and optionally 'rs' / 'sa' to a
    layout. Per scale: ``level_asbuilt``, ``level_<key>`` for each layout,
    the GA's ``lift`` and ``pct_lift`` (of the as-built level), and for
    each comparator present ``ga_minus_<key>``: ``gbp`` (GA minus it),
    ``frac_of_asbuilt`` (that over the as-built level, the fraction the
    delta method scales) and ``pct`` (of the comparator's own level,
    Figure C's ``pct_diff`` convention)."""
    out: Dict[str, Any] = {
        'anchor_score': float(store.base_params['score_anchor']['score'])}
    scales = {'whole': store.base_params,
              'stocked': RK.stocked_scale(store.base_params, params,
                                          product_ids)}
    for scale, bp in scales.items():
        base = RK.layout_value(store, store.baseline_layout, horizon_days, bp)
        rec: Dict[str, Any] = {'level_asbuilt': base}
        for key in LAYOUTS:
            if key in layouts:
                rec[f'level_{key}'] = RK.layout_value(store, layouts[key],
                                                      horizon_days, bp)
        ga = rec['level_ga']
        rec['lift'] = ga - base
        rec['pct_lift'] = (ga - base) / base * 100.0
        for key in COMPARATORS:
            if f'level_{key}' in rec:
                other = rec[f'level_{key}']
                rec[f'ga_minus_{key}'] = {
                    'gbp': ga - other,
                    'frac_of_asbuilt': (ga - other) / base,
                    'pct': (ga - other) / other * 100.0}
        out[scale] = rec
    out['stocked_invoice_share'] = float(scales['stocked'][
        'stocked_invoice_share'])
    return out


@dataclass(frozen=True)
class BootContext:
    """What each replicate needs: the adapted rows, their clusters, the
    calibration that built the fixtures, the searched layouts (``LAYOUTS``
    keys), and the options."""
    df: pd.DataFrame
    clusters: Clusters
    fixture_params: CalibratedParams
    layouts: Dict[str, Dict[str, Any]]
    boot_seed: int
    assumed_conversion: float
    currency: str
    max_items_per_category: int
    horizon_days: int
    store_fingerprint: str


def _replicate_store(ctx: BootContext):
    """A fresh store with the fixtures, checked against the parent's."""
    store = build_store(ctx.fixture_params, ctx.max_items_per_category,
                        verbose=False)
    fp = store_fingerprint(store)
    if fp != ctx.store_fingerprint:
        raise RuntimeError(f"rebuilt store differs from the parent's "
                           f"({fp[:12]} != {ctx.store_fingerprint[:12]})")
    return store


def run_replicate(ctx: BootContext, store, b: int) -> Dict[str, Any]:
    """Replicate ``b``: resample, recalibrate, re-key onto ``store`` (in
    place), score every fixed layout. A function of ``b`` and ``ctx``
    alone: re-keying replaces everything the previous replicate wrote."""
    t0 = time.perf_counter()
    df_b = resample(ctx.df, ctx.clusters,
                    np.random.default_rng([ctx.boot_seed, b]))
    t1 = time.perf_counter()
    params_b = calibrate_transactional(
        df_b, assumed_conversion_rate=ctx.assumed_conversion,
        currency=ctx.currency)
    t2 = time.perf_counter()
    rekeyed = RK.reseed_store(store, params_b)
    scores = score_fixed_layouts(rekeyed, params_b, ctx.layouts,
                                 RK.fixture_products(rekeyed),
                                 ctx.horizon_days)
    t3 = time.perf_counter()
    return {'replicate': int(b), 'n_invoices': int(params_b.n_invoices),
            'n_rows': int(len(df_b)),
            'n_absent_fixtures': len(RK.absent_fixtures(rekeyed, params_b)),
            **scores,
            'seconds': {'resample': t1 - t0, 'calibrate': t2 - t1,
                        'rekey_and_score': t3 - t2}}


_WORKER: Dict[str, Any] = {}


def _init_worker(ctx: BootContext) -> None:
    _WORKER['ctx'] = ctx
    _WORKER['store'] = _replicate_store(ctx)


def _replicate_task(b: int) -> Dict[str, Any]:
    return run_replicate(_WORKER['ctx'], _WORKER['store'], b)


def map_replicates(ctx: BootContext, n_boot: int,
                   workers: int) -> List[Dict[str, Any]]:
    """Every replicate, in replicate order, whatever ``workers`` is."""
    reps = list(range(n_boot))
    if workers <= 1:
        store = _replicate_store(ctx)
        out = []
        for b in reps:
            out.append(run_replicate(ctx, store, b))
            _progress(out[-1], n_boot)
        return out
    done: Dict[int, Dict[str, Any]] = {}
    pool = ProcessPoolExecutor(max_workers=min(workers, n_boot),
                               initializer=_init_worker, initargs=(ctx,))
    try:
        futures = {pool.submit(_replicate_task, b): b for b in reps}
        for fut in as_completed(futures):
            done[futures[fut]] = fut.result()
            _progress(done[futures[fut]], n_boot)
    except BaseException:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown(wait=True)
    return [done[b] for b in reps]


def _progress(rec: Mapping[str, Any], n_boot: int) -> None:
    w = rec['whole']
    print(f"[input] replicate {rec['replicate'] + 1}/{n_boot}: "
          f"lift {w['lift']:+.2f} ({w['pct_lift']:+.3f}%)  "
          f"calibrated in {rec['seconds']['calibrate']:.1f}s", flush=True)


# --- Aggregation ---------------------------------------------------------------

def cluster_total_se(values: np.ndarray) -> float:
    """SE of a sum over C clusters drawn with replacement: sqrt(C/(C-1) *
    sum (x_c - mean)^2), the clustered sampling SE of a total."""
    x = np.asarray(values, dtype=np.float64)
    if x.size < 2:
        return float('nan')
    return float(np.sqrt(x.size / (x.size - 1.0)
                         * np.sum((x - x.mean()) ** 2)))


def cluster_totals(df: pd.DataFrame, clusters: Clusters,
                   params: CalibratedParams, product_ids) -> Dict[str, Any]:
    """Per-cluster totals behind each scale's level: whole-invoice line
    revenue, and stocked spend (one unit of each distinct stocked product
    per invoice at the calibrated prices, ``placed_invoice_sample``'s
    unit), with the clustered SE of each total."""
    C = clusters.n
    cluster_of_row = np.empty(len(df), dtype=np.int64)
    cluster_of_row[clusters.order] = np.repeat(np.arange(C),
                                               np.diff(clusters.starts))
    line = (pd.to_numeric(df['quantity'], errors='coerce')
            * pd.to_numeric(df['unit_price'], errors='coerce')).to_numpy()
    whole = np.bincount(cluster_of_row, weights=np.nan_to_num(line),
                        minlength=C)
    pairs = pd.DataFrame({'inv': df['invoice_id'].astype(str).to_numpy(),
                          'pid': df['product_id'].astype(str).to_numpy(),
                          'c': cluster_of_row}).drop_duplicates(['inv', 'pid'])
    price = {str(k): float(v) for k, v in params.item_prices.items()}
    wanted = {str(p) for p in product_ids}
    stocked_price = np.where(pairs['pid'].isin(wanted),
                             pairs['pid'].map(price).fillna(0.0), 0.0)
    stocked = np.bincount(pairs['c'].to_numpy(), weights=stocked_price,
                          minlength=C)
    return {'whole': {'total': float(whole.sum()),
                      'se': cluster_total_se(whole)},
            'stocked': {'total': float(stocked.sum()),
                        'se': cluster_total_se(stocked)}}


def _delta_gbp(estimate: float, L: float, frac: float, se_L: float,
               lev_boot: np.ndarray, frac_boot: np.ndarray,
               gbp_boot: np.ndarray, z: float, q: Sequence[float],
               method: str) -> Dict[str, Any]:
    """Delta-method interval for a GBP quantity written as L x f -- the
    as-built level times a fraction of it -- with var(L) from the clustered
    SE of the total and var(f), cov(L, f) from the bootstrap; the
    bootstrap's own percentile interval and SD beside it."""
    var_f = float(np.var(frac_boot, ddof=1))
    cov = float(np.cov(lev_boot, frac_boot, ddof=1)[0, 1])
    var = frac * frac * se_L ** 2 + L * L * var_f + 2.0 * L * frac * cov
    se = float(np.sqrt(max(var, 0.0)))
    return {'estimate': estimate,
            'se': se,
            'ci': [estimate - z * se, estimate + z * se],
            'method': method,
            'boot_ci': [float(v) for v in np.percentile(gbp_boot, q)],
            'boot_se': float(gbp_boot.std(ddof=1)),
            'covers': 'input'}


def _percentile(estimate: float, boot: np.ndarray, q: Sequence[float],
                **extra: Any) -> Dict[str, Any]:
    return {'estimate': estimate,
            'boot_mean': float(boot.mean()),
            'boot_bias': float(boot.mean() - estimate),
            'boot_se': float(boot.std(ddof=1)),
            'ci': [float(v) for v in np.percentile(boot, q)],
            'method': 'percentile, customer-clustered bootstrap',
            **extra,
            'covers': 'input'}


def summarize(point: Mapping[str, Any], reps: Sequence[Mapping[str, Any]],
              totals: Mapping[str, Any],
              level: float = CI_LEVEL) -> Dict[str, Any]:
    """The intervals of ``summary['input_uncertainty']`` (see the module
    docstring)."""
    z = float(sps.norm.ppf(0.5 + level / 2.0))
    q = [100.0 * (0.5 - level / 2.0), 100.0 * (0.5 + level / 2.0)]
    out: Dict[str, Any] = {'level': level, 'n_boot': len(reps)}
    for scale in SCALES:
        pct = np.asarray([r[scale]['pct_lift'] for r in reps])
        lev = np.asarray([r[scale]['level_asbuilt'] for r in reps])
        lift = np.asarray([r[scale]['lift'] for r in reps])
        p0 = point[scale]
        L = p0['level_asbuilt']
        se_T = totals[scale]['se'] / totals[scale]['total']
        se_L = L * se_T
        block: Dict[str, Any] = {
            'pct_lift': _percentile(p0['pct_lift'], pct, q,
                                    pct_of='asbuilt_level'),
            'level_gbp': {
                'estimate': L,
                'se': se_L,
                'rse': float(se_T),
                'ci': [L - z * se_L, L + z * se_L],
                'method': ('delta: the as-built level is proportional to '
                           'the total over the calendar span, SE of the '
                           'total clustered by customer'),
                'boot_ci': [float(v) for v in np.percentile(lev, q)],
                'boot_se': float(lev.std(ddof=1)),
                'covers': 'input'},
            'lift_gbp': _delta_gbp(
                p0['lift'], L, p0['pct_lift'] / 100.0, se_L, lev,
                pct / 100.0, lift, z, q,
                method=('delta: lift = level x fraction; var = '
                        'f^2 var(L) + L^2 var_boot(f) + '
                        '2 L f cov_boot(L, f)')),
        }
        # The GA's margins over the equal-budget comparators (review R11:
        # "the GBP lift and margins").
        for key in COMPARATORS:
            name = f'ga_minus_{key}'
            if name not in p0:
                continue
            m0 = p0[name]
            m_pct = np.asarray([r[scale][name]['pct'] for r in reps])
            m_frac = np.asarray([r[scale][name]['frac_of_asbuilt']
                                 for r in reps])
            m_gbp = np.asarray([r[scale][name]['gbp'] for r in reps])
            block[name] = {
                'pct': _percentile(m0['pct'], m_pct, q,
                                   pct_of='comparator_level'),
                'gbp': _delta_gbp(
                    m0['gbp'], L, m0['frac_of_asbuilt'], se_L, lev, m_frac,
                    m_gbp, z, q,
                    method=('delta: margin = level x fraction of the '
                            'as-built level; var = f^2 var(L) + '
                            'L^2 var_boot(f) + 2 L f cov_boot(L, f)')),
                'n_boot_ga_ahead': int((m_gbp > 0.0).sum()),
            }
        out[scale] = block
    out['scale_ratio'] = (point['stocked']['level_asbuilt']
                          / point['whole']['level_asbuilt'])
    return out


def floor_bias(store, layouts: Mapping[str, Any],
               horizon_days: int) -> Dict[str, Any]:
    """``closed_form.legacy_floor_bias`` -- how far the former floored
    spend law raised the Monte Carlo mean above the exact one -- for the
    as-built layout and each of ``layouts`` ('ga', 'rs', 'sa') on ``store``
    at its base parameters, over ``horizon_days``. Also the lift's bias
    (GA minus as-built), each as a share of the quantity it biased, and the
    per-converter spend CV the as-built layout draws with (the floor bites
    harder the more dispersed the spend)."""
    out: Dict[str, Any] = {}
    exact: Dict[str, float] = {}
    for name, lay in (('asbuilt', store.baseline_layout),
                      *((k, layouts[k]) for k in LAYOUTS if k in layouts)):
        s, b = layout_score(store.shop, store.item_names, lay)
        kw = layout_mc_kwargs(layout_drivers(s, b, store.base_params),
                              store.base_params, n_days=horizon_days,
                              n_iter=1)
        out[name] = legacy_floor_bias(**kw)
        exact[name] = RK.layout_value(store, lay, horizon_days)
        if name == 'asbuilt':
            out['spend_cv'] = float(kw['rev_std'] / kw['rev_mean'])
    out['lift'] = out['ga'] - out['asbuilt']
    lift = exact['ga'] - exact['asbuilt']
    out['frac_of_asbuilt_level'] = out['asbuilt'] / exact['asbuilt']
    out['frac_of_lift'] = out['lift'] / lift if lift != 0.0 else None
    return out


# --- Main ----------------------------------------------------------------------

def write_results_csv(out_dir: str, reps: Sequence[Mapping[str, Any]]) -> str:
    """One row per replicate: every layout's level on both scales, the
    GA's lift and its margins over the comparators."""
    keys = [k for k in LAYOUTS if f'level_{k}' in reps[0]['whole']]
    margins = [f'ga_minus_{k}' for k in COMPARATORS
               if f'ga_minus_{k}' in reps[0]['whole']]
    cols = (['level_asbuilt'] + [f'level_{k}' for k in keys]
            + ['lift', 'pct_lift'])
    path = os.path.join(out_dir, 'results.csv')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['replicate', 'n_invoices', 'n_absent_fixtures',
                    'anchor_score', 'stocked_invoice_share',
                    *(f'{s}_{c}' for s in SCALES for c in cols),
                    *(f'{s}_{m}_{v}' for s in SCALES for m in margins
                      for v in ('gbp', 'pct'))])
        for r in reps:
            w.writerow([r['replicate'], r['n_invoices'],
                        r['n_absent_fixtures'], f"{r['anchor_score']:.10f}",
                        f"{r['stocked_invoice_share']:.6f}",
                        *(f"{r[s][c]:.6f}" for s in SCALES for c in cols),
                        *(f"{r[s][m][v]:.6f}" for s in SCALES
                          for m in margins for v in ('gbp', 'pct'))])
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    out_dir = make_run_dir(args.out_root,
                           run_dir_prefix('input_uncertainty', args))
    experiment = experiment_name('input_uncertainty_uci', args)
    prov_run = provenance_snapshot()
    print(f"output dir: {out_dir}", flush=True)
    wall_t0 = time.perf_counter()

    # -- 1. Figure C's data, calibration and store ------------------------
    raw, reader = read_workbook(args)
    df, report = adapt_rows(raw)
    t0 = time.perf_counter()
    # On every row, so that an exclusion of the anonymous invoices is
    # counted in the calibration's record.
    params = calibrate_normalized(df, report, args)
    calib_seconds = time.perf_counter() - t0
    # The same rows under adapter 1.1's cleaning (cancellation lines only),
    # for the former spend law's bias at the inputs the published numbers
    # were made under (``legacy_floor_bias``). The raw rows are dropped as
    # soon as this is done.
    df_11, report_11 = adapt_rows(raw, match_reversals=False)
    params_11 = calibrate_normalized(df_11, report_11, args)
    cleaning_11 = dict(report_11.extra.get('cleaning') or {})
    del raw, df_11
    if args.exclude_anonymous:
        # The bootstrap then resamples the identified customers only, the
        # rows the calibration above kept.
        df = df[~anonymous_customer_mask(df['customer_id']).to_numpy()]
        df = df.reset_index(drop=True)
    store = build_store(params, args.max_items_per_category)
    fingerprint = store_fingerprint(store)
    pids = RK.fixture_products(store)

    # -- 2. Figure C's three searches, once -----------------------------------
    print(f"[input] Figure C's searches (seed {args.ga_seed}, pop "
          f"{args.pop_size}, gens {args.n_gens}, mc_iters {args.mc_iters}, "
          f"mc_days {args.mc_days})...", flush=True)
    t0 = time.perf_counter()
    searches = run_searches(store, seed=args.ga_seed, n_gens=args.n_gens,
                            pop_size=args.pop_size, mc_iters=args.mc_iters,
                            mc_days=args.mc_days,
                            sa_initial_accept=args.sa_initial_accept)
    search_seconds = time.perf_counter() - t0
    # Every layout the run re-scores keeps the floor-plan invariants (the
    # aisles the as-built store keeps, every fixture shoppable); a layout
    # that breaks one stops the run (LayoutInvariantError).
    invariants = check_reported(store, reported_layouts(store, searches))
    layouts = {k: searches[SEARCH_KEY[k]] for k in LAYOUTS}

    # -- 3. Full-data values, and the bootstrap -----------------------------
    point = score_fixed_layouts(store, params, layouts, pids, args.mc_days)
    clusters = customer_clusters(df)
    ctx = BootContext(df=df, clusters=clusters, fixture_params=params,
                      layouts=layouts, boot_seed=args.boot_seed,
                      assumed_conversion=args.assumed_conversion,
                      currency=report.extra.get('currency', 'GBP'),
                      max_items_per_category=args.max_items_per_category,
                      horizon_days=args.mc_days,
                      store_fingerprint=fingerprint)
    print(f"[input] full data: lift {point['whole']['lift']:+.2f} "
          f"({point['whole']['pct_lift']:+.3f}%); {args.n_boot} replicates "
          f"over {clusters.n:,} clusters ({clusters.n_customers:,} "
          f"customers, {clusters.n_anonymous_invoices:,} anonymous "
          f"invoices) on {args.workers} worker(s)...", flush=True)
    t0 = time.perf_counter()
    reps = map_replicates(ctx, args.n_boot, args.workers)
    boot_seconds = time.perf_counter() - t0
    totals = cluster_totals(df, clusters, params, pids)
    intervals = summarize(point, reps, totals)

    # -- 4. The former spend law's bias, at both sets of inputs ---------------
    store_11 = RK.reseed_store(_replicate_store(ctx), params_11)
    legacy = {
        'meaning': ("how far the former floored-normal spend law raised "
                    "each layout's expected revenue over horizon_days above "
                    "the exact mean, and the lift's bias (GA minus "
                    "as-built); frac_of_* divide by the exact quantity"),
        'horizon_days': args.mc_days,
        'current_inputs': {
            'inputs': ('this run\'s calibration (adapter '
                       f'{OnlineRetailIIAdapter.version} cleaning: '
                       'reversal pairs removed)'),
            **floor_bias(store, layouts, args.mc_days)},
        'adapter_1_1_inputs': {
            'inputs': ('the same rows cleaned as adapter 1.1 cleaned them '
                       '(cancellation lines dropped, the purchases they '
                       'reverse kept) -- the inputs of the numbers published '
                       'before adapter 1.2 -- re-keyed onto this run\'s '
                       'fixtures and scored at this run\'s layouts (the '
                       'published layouts were not stored)'),
            'cleaning': cleaning_11,
            'n_absent_fixtures': len(RK.absent_fixtures(store_11,
                                                        params_11)),
            **floor_bias(store_11, layouts, args.mc_days)},
    }

    ga_out = searches['ga_out']
    design = {
        'sheets':                 args.sheets,
        'max_items_per_category': args.max_items_per_category,
        'assumed_conversion':     args.assumed_conversion,
        'exclude_anonymous':      bool(args.exclude_anonymous),
        'adapter_version':        OnlineRetailIIAdapter.version,
        'horizon_days':           args.mc_days,
        'ga': {'seed': args.ga_seed, 'n_gens': args.n_gens,
               'pop_size': args.pop_size, 'mc_iters': args.mc_iters,
               'mc_days': args.mc_days, 'mut_rate': GA_MUT_RATE,
               'elite_frac': GA_ELITE_FRAC, 'start': 'asbuilt',
               'n_search_evals': ga_out['n_search_evals'],
               'n_final_evals': ga_out['n_final_evals']},
        # Figure C's equal-budget comparators, searched once at the GA's
        # seed and budget (run_real_data_example.run_searches).
        'searches': {
            'seed': args.ga_seed,
            'budget_search_evals': searches['budget'],
            'evaluation_counts': searches['eval_counts'],
            'final_evaluation_counts': searches['final_eval_counts'],
            'starts': searches['starts'],
            'sa_initial_accept': args.sa_initial_accept,
            'sa_T0': searches['sa_stats'].get('sa_T0'),
            'rs_sampler': 'zone_sampler',
            'sa_neighbor': f'zone_neighbor(step_frac={SA_STEP_FRAC})',
            # What each search's operators draw from: the same space for
            # all three (each item's zone less the aisles the repair keeps).
            'operators': search_operators()},
        'n_boot':                 args.n_boot,
        'boot_seed':              args.boot_seed,
        'replicate_rng':          'numpy default_rng([boot_seed, b])',
        'clusters': {'rule': ('one per identified customer, one per '
                              'anonymous invoice; copies relabelled as '
                              'distinct invoices and customers'),
                     'n_clusters': clusters.n,
                     'n_customers': clusters.n_customers,
                     'n_anonymous_invoices': clusters.n_anonymous_invoices},
        'layouts': ('fixed: the repaired as-built layout and the GA, '
                    'random-search and annealing layouts of the full-data '
                    'searches'),
        'rekey': ('same fixtures; each replicate calibration re-seeded onto '
                  'them and the base parameters re-anchored at the '
                  'as-built layout'),
        'scoring': 'experiments.closed_form.expected_revenue',
        'scales': {'whole': 'whole invoices, every invoice a buyer '
                            '(Figure C)',
                   'stocked': 'each invoice cut to the stocked products, '
                              'buyers the invoices holding one '
                              '(placed_invoice_sample)'},
        'ci_level': CI_LEVEL,
    }
    n_absent = [r['n_absent_fixtures'] for r in reps]
    summary = {
        'experiment': experiment,
        'design': design,
        'point': point,
        'input_uncertainty': intervals,
        'uncertainty_layers': UNCERTAINTY_LAYERS,
        'totals': totals,
        'absent_fixtures': {'max': int(max(n_absent)),
                            'mean': float(np.mean(n_absent)),
                            'full_data': len(RK.absent_fixtures(store,
                                                                params))},
        'anonymous': anonymous_placed_shares(params, pids),
        'legacy_floor_bias': legacy,
        # The floor-plan invariants of every reported layout, what the
        # shared repair did during the searches, and what the aisle rule
        # takes from the search space (as Figure C records them).
        'feasibility': {'invariants': invariants,
                        'repair_stats': repair_stats(store.shop),
                        'aisle_rule': aisle_rule_summary(store.shop,
                                                         store.item_names)},
        'store': {'fingerprint': fingerprint,
                  'width_m': store.shop.width, 'height_m': store.shop.height,
                  'sections': store.n_sections,
                  'items_placed': len(store.item_names)},
    }

    prov = stamp_provenance(
        source_path=args.retail_path,
        adapter_name=OnlineRetailIIAdapter.name,
        adapter_version=OnlineRetailIIAdapter.version,
        rows_in=report.rows_in, rows_kept=report.rows_kept,
        currency=report.extra.get('currency', 'GBP'), seed=args.boot_seed,
        extra={'sheets': args.sheets, 'reader': reader,
               **data_provenance_extra(report, params, args)})
    csv_path = write_results_csv(out_dir, reps)
    write_sidecar(out_dir, {
        'experiment':   experiment,
        'args':         vars(args),
        'wall_seconds': time.perf_counter() - wall_t0,
        'timing': {'calibration_full_seconds': calib_seconds,
                   'search_seconds': search_seconds,
                   'search_walls': searches['walls'],
                   'bootstrap_seconds': boot_seconds,
                   'replicate_seconds_mean': {
                       k: float(np.mean([r['seconds'][k] for r in reps]))
                       for k in ('resample', 'calibrate',
                                 'rekey_and_score')}},
        'csv_path':     os.path.relpath(csv_path, out_dir),
        'provenance':   prov.to_dict(),
        'reader':       reader,
        'base_params':  base_params_record(store.base_params),
        'summary':      summary,
    }, provenance=prov_run)
    with open(os.path.join(out_dir, 'summary.json'), 'w',
              encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    for scale in SCALES:
        s = intervals[scale]
        print(f"[input] {scale:>7s}: level {s['level_gbp']['estimate']:,.2f} "
              f"(delta CI {s['level_gbp']['ci'][0]:,.2f} .. "
              f"{s['level_gbp']['ci'][1]:,.2f}); lift "
              f"{s['lift_gbp']['estimate']:+,.2f} (delta CI "
              f"{s['lift_gbp']['ci'][0]:+,.2f} .. "
              f"{s['lift_gbp']['ci'][1]:+,.2f}); pct "
              f"{s['pct_lift']['estimate']:+.3f}% (boot CI "
              f"{s['pct_lift']['ci'][0]:+.3f} .. "
              f"{s['pct_lift']['ci'][1]:+.3f})", flush=True)
        for key in COMPARATORS:
            m = s[f'ga_minus_{key}']
            print(f"[input] {scale:>7s}: GA - {key}: "
                  f"{m['gbp']['estimate']:+,.2f} (delta CI "
                  f"{m['gbp']['ci'][0]:+,.2f} .. {m['gbp']['ci'][1]:+,.2f}); "
                  f"{m['pct']['estimate']:+.3f}% (boot CI "
                  f"{m['pct']['ci'][0]:+.3f} .. {m['pct']['ci'][1]:+.3f})",
                  flush=True)
    print(f"\nArtifacts in: {out_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
