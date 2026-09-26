"""Figure C: paper's real-data worked example.

Loads UCI Online Retail II (transactional, ~1M invoices 2009-2011) end-
to-end through the calibration pipeline, materializes the resulting
multi-section shop on a headless instance, runs the GA optimizer, and
reports the paired-MC projected revenue lift over the as-calibrated
baseline layout (mapped onto the GA's feasible set, like every GA
candidate), evaluated on seeds the GA never searched under.

The as-built layout is not the GA's only comparator. Random search and
simulated annealing (``experiments.metaheuristics``) run on the same store
at the GA's search budget -- ``pop_size * n_gens`` evaluations with one
shared seed per ``pop_size`` block, then the GA's final selection over
``GA_N_FINAL_SEEDS`` seeds -- both starting from the naive as-built layout
the GA's population is seeded from, and seeded from ``--ga-seed`` as the
GA is. The calibrated store has no SyntheticShop to draw layouts from, so
their draws and moves come from ``zone_sampler`` / ``zone_neighbor``,
inside each item's zone less the aisles the shared repair keeps (its
region) -- the space the GA's initial noise and mutation are scaled to and
clipped into as well (``search_operators``) -- and every candidate goes
through ``feasible_layout`` like the GA's. Their layouts are scored under
the same held-out paired replicates as the GA and the baseline, which says
whether the lift over the as-built store needs the GA or only needs
search.

Each method searches once, from ``--ga-seed``, so the paired replicates
cover evaluation noise only. ``experiments.run_real_data_seeds`` repeats the
three searches over a range of search seeds to measure the search-to-search
spread; it builds the store with ``build_store``, searches with
``run_searches`` and scores with ``score_layouts`` from this module, so the
two runners cannot drift apart on the store, the budget or the evaluation
seeds.

Every reported layout (the as-built baseline and the three searches') is
checked against the floor-plan engine's invariants before it is scored --
the aisles the as-built store keeps between zones stay at least 1.6 m wide
and every fixture stays shoppable (``experiments._feasibility``); the run
stops on a violation. Each layout is also scored in closed form, at the
exact mean of its Monte Carlo fitness (``closed_form_scan``): its score
breakdown, its lift over the baseline at the corners of the three
elasticity bands and at the basket elasticities 0.02 and 0, and at assumed
conversion rates up to the 0.99 clamp, with the rate at which the clamp
starts to bind; ``closed_form_check`` says whether each conclusion holds
in closed form. The final layouts and each search's convergence trace are
saved (``layouts.json``, ``traces.json``).

This is the script that reproduces the paper's real-data application
figure. Output sidecar.json records dataset SHA-256, git SHA, elasticity
snapshot, and the conversion-rate assumption marker -- everything needed
for end-to-end reproducibility.

``--retail-path`` is optional: without it the workbook is found by
``dataset_paths.uci_workbook()`` (``$UCI_RETAIL_XLSX``, then the workbook
under ``DATASETS/``).

Smoke (one sheet + small GA budget; a few minutes):
    python -m experiments.run_real_data_example ^
        --sheets "Year 2010-2011" ^
        --max-items-per-category 8 ^
        --mc-iters 200 --n-gens 5 --pop-size 10 --n-mc-replicates 5

Paper-grade (both sheets + Tier-1 GA budget; the two comparators each
spend the GA's budget, so about three times the GA's own wall time):
    python -m experiments.run_real_data_example ^
        --sheets "Year 2009-2010,Year 2010-2011" ^
        --max-items-per-category 12 ^
        --mc-iters 2000 --mc-days 30 ^
        --n-gens 25 --pop-size 30 --n-mc-replicates 30

Why no ATC spatial component: the user does not have the ATC Shopping
Mall trajectory dataset locally. The spatial-calibration code path in
``dataset_adapters.ATCShoppingMallAdapter`` + ``calibrate_trajectory``
exists and is tested via the GUI; integrating ATC into this script is
a small follow-on once the dataset is on disk.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# Make sibling project modules importable when run as ``python -m ...``.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from dataset_adapters import (
    READER_VERSION,
    OnlineRetailIIAdapter,
    list_excel_sheets,
    read_excel_sheets,
)
from dataset_calibration import calibrate_transactional, CalibratedParams
from dataset_paths import uci_workbook
from dataset_provenance import stamp as stamp_provenance

from experiments._common import (
    GA_N_FINAL_SEEDS,
    build_headless_shop_from_calibration,
    base_params_for_calibration,
    anchor_base_params,
    base_params_record,
    checked_layout,
    run_ga_headless,
    paired_mc_revenue,
    bootstrap_ci,
    chromosome_to_layout,
    feasible_layout,
    layout_json,
    layout_score,
    aisle_rule_summary,
    repair_stats,
    zone_sampler,
    zone_neighbor,
    make_run_dir,
    write_json,
    write_sidecar,
)
from experiments.closed_form import expected_revenue
from experiments.metaheuristics import (
    DEFAULT_INITIAL_ACCEPT,
    random_search,
    simulated_annealing,
)
from experiments.run_elasticity_lhs import BASKET_CORNERS
from layout_objective import ELASTICITY_BANDS, layout_drivers
from retail_literature import (ABANDON_FRAC_OF_NONCONVERTERS,
                               ASSUMED_CONVERSION_RATE, CONV_CLAMP_HI)


# Paired-evaluation seeds are EVAL_SEED_BASE + replicate. Every search --
# the GA, random search and simulated annealing -- scores and selects under
# ga_seed*1000 + [0, n_gens + 1 + GA_N_FINAL_SEEDS) (generation or block
# seeds, a gap, then the final-selection seeds; ``search_seed_ranges``);
# ``parse_args`` keeps that range below the base.
EVAL_SEED_BASE = 1_000_000

#: Step of the annealer's one-item move, as a fraction of the room the
#: item's zone gives it on each axis (``zone_neighbor``). The same fraction
#: ``metaheuristics._neighbor`` scales its moves by on the synthetic shops.
SA_STEP_FRAC = 0.25

#: Run-directory prefix and experiment-name suffix of a run calibrated
#: without the anonymous invoices (``--exclude-anonymous``). Such a run is
#: a sensitivity analysis. The macro generator takes, per family, the
#: newest valid run whose directory name starts with the family's prefix
#: (``real_data_uci_``, ``real_data_seeds_``, ...), so a sensitivity run
#: written under the headline prefix could silently replace the headline;
#: it gets a prefix of its own instead.
NOANON_DIR_PREFIX = 'noanon_'
NOANON_EXPERIMENT_SUFFIX = '_noanon'


def run_dir_prefix(base: str, args: argparse.Namespace) -> str:
    """``make_run_dir`` prefix of ``base``'s family: the headline family,
    or its no-anonymous sensitivity family (``NOANON_DIR_PREFIX``)."""
    return (NOANON_DIR_PREFIX + base
            if getattr(args, 'exclude_anonymous', False) else base)


def experiment_name(base: str, args: argparse.Namespace) -> str:
    """The ``experiment`` field of ``base``'s summary and sidecar, suffixed
    for a no-anonymous sensitivity run."""
    return (base + NOANON_EXPERIMENT_SUFFIX
            if getattr(args, 'exclude_anonymous', False) else base)


def search_operators(sa_k_move: Optional[int] = None) -> Dict[str, str]:
    """What each search's operators draw from, as the runs record it
    (``search_design['operators']``). All three work inside the same space,
    each item's zone less the aisles the shared repair keeps (its region,
    ``experiments._feasibility``): random search draws uniformly over it,
    the annealer's step is a fraction of the room it leaves, and the GA's
    initial noise and mutation are scaled to it and clipped into it. None of
    them proposes a position the repair would clip onto an aisle edge."""
    out = {
        'space': ("region: each item's zone less the aisles the shared "
                  "repair keeps (_feasibility.AislePlan.move_box / "
                  "position_bounds)"),
        'GA': ("initial noise N(0, max(0.15 x side, 0.3 m)) and mutation "
               "N(0, max(0.12 x side, 0.2 m)) per axis, side = the side of "
               "the item's move box (its zone less the aisle rule), clipped "
               "0.05 m inside that box, i.e. into the region "
               "(HeadlessShop._ga_move_boxes)"),
        'random_search': "zone_sampler: uniform over each item's region",
        'simulated_annealing': f'zone_neighbor(step_frac={SA_STEP_FRAC}): '
                               f'one item, step uniform in +/- '
                               f'{SA_STEP_FRAC} x its room in its region',
    }
    if sa_k_move is not None:
        out['simulated_annealing_k'] = (
            f'zone_neighbor(step_frac={SA_STEP_FRAC}, n_move={int(sa_k_move)})'
            f': {int(sa_k_move)} distinct items per move, each as above')
    return out


def search_seed_ranges(ga_seed: int, n_gens: int,
                       pop_size: int) -> Dict[str, List[int]]:
    """Half-open range ``[lo, hi)`` of the Monte Carlo seeds each search
    scores or selects under.

    The GA evaluates generation ``g`` under ``ga_seed*1000 + g`` and runs
    its final selection under ``ga_seed*1000 + n_gens + 1 + s``. Random
    search and SA, given ``budget = pop_size * n_gens`` and ``block =
    pop_size``, use one seed per block, ``ga_seed*1000 + [0, n_blocks)``,
    and the same final-selection rule with ``n_blocks`` in place of
    ``n_gens``; the two counts coincide."""
    base = ga_seed * 1000
    n_blocks = -(-(pop_size * n_gens) // pop_size)
    meta = [base, base + n_blocks + 1 + GA_N_FINAL_SEEDS]
    return {'GA': [base, base + n_gens + 1 + GA_N_FINAL_SEEDS],
            'random_search': list(meta),
            'simulated_annealing': list(meta)}


def search_seeds_fit(ga_seed: int, n_gens: int, pop_size: int) -> bool:
    """True if every seed the searches of run seed ``ga_seed`` score or
    select under stays inside that seed's own block of 1000 and below the
    held-out evaluation seeds.

    Inside its own block, no method is finally compared under a noise draw
    it searched or selected under; and when several run seeds are searched
    (``run_real_data_seeds``), no search reuses the draws of another."""
    ceiling = min((ga_seed + 1) * 1000, EVAL_SEED_BASE)
    ranges = search_seed_ranges(ga_seed, n_gens, pop_size)
    return ga_seed >= 0 and all(hi <= ceiling for _, hi in ranges.values())


_GA_SEED_HELP = ("RNG seed for the GA initialization; random search "
                 "and simulated annealing are seeded from it too")


def add_arguments(p: argparse.ArgumentParser,
                  ga_seed_help: str = _GA_SEED_HELP) -> None:
    """The store and search options, defaults included.

    ``run_real_data_seeds`` takes them from here: it has to build this
    example's store and search it at this example's budget, and a copy of
    the defaults could drift from them unnoticed. Only the meaning of
    ``--ga-seed`` differs there (the first of its search seeds)."""
    p.add_argument('--retail-path', type=str, default=None,
                   help="Path to the UCI Online Retail II workbook. Default: "
                        "found by dataset_paths.uci_workbook() "
                        "($UCI_RETAIL_XLSX, then DATASETS/)")
    p.add_argument('--sheets', type=str,
                   default="Year 2009-2010,Year 2010-2011",
                   help="Comma-separated Excel sheet names to concatenate")
    p.add_argument('--max-items-per-category', type=int, default=12,
                   help="Cap on items placed per inferred category")
    p.add_argument('--assumed-conversion', type=float,
                   default=ASSUMED_CONVERSION_RATE,
                   help="Conversion rate assumption (transactional data has no buy/no-buy split)")
    p.add_argument('--exclude-anonymous', action='store_true',
                   help="Calibrate without the invoices that carry no "
                        "customer id (a sensitivity run; they are kept by "
                        "default and their share is recorded)")
    p.add_argument('--mc-iters', type=int, default=2000)
    p.add_argument('--mc-days', type=int, default=30)
    p.add_argument('--n-gens', type=int, default=25)
    p.add_argument('--pop-size', type=int, default=30)
    p.add_argument('--n-mc-replicates', type=int, default=30,
                   help="Number of paired-MC replicate seeds used to estimate the lift CI")
    p.add_argument('--ga-seed', type=int, default=0, help=ga_seed_help)
    p.add_argument('--sa-initial-accept', type=float,
                   default=DEFAULT_INITIAL_ACCEPT,
                   help="Acceptance probability of a median worsening move "
                        "at the annealer's starting temperature")
    p.add_argument('--out-root', type=str,
                   default=os.path.join(_HERE, 'results'))


def check_search_arguments(p: argparse.ArgumentParser,
                           args: argparse.Namespace) -> None:
    """Refuse a search design the three searches cannot run as specified.

    Kept apart from the workbook lookup so that a bad design is reported
    before, and without, the dataset being found."""
    if args.pop_size < 2 or args.n_gens < 1:
        p.error("--pop-size must be >= 2 (simulated annealing re-scores its "
                "incumbent at the start of every pop_size block) and "
                "--n-gens >= 1")
    if not 0.0 < args.sa_initial_accept < 1.0:
        p.error("--sa-initial-accept must lie strictly between 0 and 1")
    # Each search's seeds must stay inside the run seed's own block of 1000
    # and below the held-out evaluation seeds, so no method is finally
    # compared under a noise draw it searched or selected under.
    if not search_seeds_fit(args.ga_seed, args.n_gens, args.pop_size):
        p.error(f"--ga-seed must be in [0, 999] and --n-gens + "
                f"{1 + GA_N_FINAL_SEEDS} <= 1000 so the search seeds of the "
                f"GA, random search and simulated annealing stay below the "
                f"evaluation seeds")


def resolve_workbook(p: argparse.ArgumentParser,
                     args: argparse.Namespace) -> None:
    """Replace ``args.retail_path`` by the workbook ``dataset_paths`` finds,
    or stop with the paths it tried."""
    try:
        args.retail_path = uci_workbook(args.retail_path)
    except FileNotFoundError as exc:
        p.error(str(exc))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    add_arguments(p)
    args = p.parse_args()
    check_search_arguments(p, args)
    resolve_workbook(p, args)
    return args


# Per-sheet row counts as ``read_excel_sheets`` reports them in its notes.
_READ_ROWS_RE = re.compile(r"^Read sheet '.*' \(([\d,]+) rows\)\.$")


def reader_facts(read_notes: List[str], rows_after: int) -> Dict:
    """What the workbook reader did, for the provenance record.

    Sheets of this workbook overlap in time, so the reader drops rows of a
    later sheet that repeat an earlier one before the adapter sees the
    frame. ``report.rows_in`` is therefore the post-de-duplication count and
    cannot be checked against the workbook on its own; the rows read and the
    number dropped are what tie the record back to the file's SHA-256."""
    per_sheet = [int(m.group(1).replace(',', ''))
                 for m in (_READ_ROWS_RE.match(n) for n in read_notes) if m]
    rows_read = sum(per_sheet) if per_sheet else None
    return {
        'rows_read': rows_read,
        'rows_per_sheet': per_sheet,
        'rows_after_dedup': int(rows_after),
        'cross_sheet_duplicates_dropped': (
            None if rows_read is None else rows_read - int(rows_after)),
        'read_notes': list(read_notes),
    }


def calibrate_from_file(args: argparse.Namespace) -> tuple:
    """Returns (params, report, reader_facts)."""
    normalized, report, reader = read_normalized(args)
    return calibrate_normalized(normalized, report, args), report, reader


def calibrate_normalized(normalized, report,
                         args: argparse.Namespace) -> CalibratedParams:
    """``calibrate_transactional`` on the adapted rows with this example's
    options (assumed conversion, the report's currency, the anonymous
    invoices in or out)."""
    t0 = time.perf_counter()
    params = calibrate_transactional(
        normalized,
        assumed_conversion_rate=args.assumed_conversion,
        currency=report.extra.get('currency', 'GBP'),
        exclude_anonymous=bool(getattr(args, 'exclude_anonymous', False)),
    )
    print(f"[real] calibrated in {time.perf_counter()-t0:.1f}s: "
          f"{params.n_invoices:,} invoices, "
          f"{params.n_unique_products:,} unique products, "
          f"{params.n_unique_categories} categories, "
          f"{params.arrivals_per_hour:.1f} invoices/hr (avg)",
          flush=True)
    return params


def read_normalized(args: argparse.Namespace) -> tuple:
    """The requested sheets through the reader and the UCI adapter:
    ``(normalized rows, adapter report, reader facts)``."""
    df, reader = read_workbook(args)
    normalized, report = adapt_rows(df)
    return normalized, report, reader


def read_workbook(args: argparse.Namespace) -> tuple:
    """The requested sheets as the reader returns them (cross-sheet
    repeats dropped): ``(raw rows, reader facts)``."""
    if not os.path.isfile(args.retail_path):
        raise SystemExit(f"--retail-path does not exist: {args.retail_path}")
    sheets_avail = [name for name, _ in list_excel_sheets(args.retail_path)]
    sheets_requested = [s.strip() for s in args.sheets.split(',') if s.strip()]
    missing = [s for s in sheets_requested if s not in sheets_avail]
    if missing:
        raise SystemExit(
            f"--sheets references sheets not in workbook: {missing}. "
            f"Available: {sheets_avail}"
        )
    print(f"[real] reading {len(sheets_requested)} sheet(s) from "
          f"{os.path.basename(args.retail_path)}: {sheets_requested}",
          flush=True)
    t0 = time.perf_counter()
    df, read_notes = read_excel_sheets(args.retail_path, sheets_requested)
    reader = reader_facts(read_notes, len(df))
    print(f"[real] read {len(df):,} rows in {time.perf_counter()-t0:.1f}s",
          flush=True)
    for note in read_notes:
        print(f"[real]   {note}", flush=True)
    return df, reader


def adapt_rows(df, match_reversals: bool = True) -> tuple:
    """Raw workbook rows through the UCI adapter: ``(normalized rows,
    adapter report)``. ``match_reversals=False`` is adapter 1.1's cleaning
    (``OnlineRetailIIAdapter``), used only to measure what results made
    under it carried."""
    adapter = OnlineRetailIIAdapter(match_reversals=match_reversals)
    normalized, report = adapter.adapt(df)
    if report.is_blocking():
        raise SystemExit(
            "UCI adapter validation is blocking; see report:\n\n"
            + report.to_text()
        )
    print(f"[real] adapter kept {report.rows_kept:,}/{report.rows_in:,} rows "
          f"({adapter.name} v{adapter.version}"
          f"{'' if match_reversals else ', 1.1 cleaning'})",
          flush=True)
    return normalized, report


def data_provenance_extra(report, params: CalibratedParams,
                          args: argparse.Namespace) -> Dict[str, Any]:
    """What the adapter and the calibration did to the workbook's rows,
    for the ``extra`` of a real-data run's provenance record: the reader's
    version, the adapter's cleaning counts (cancellation lines and the
    purchase lines they reverse), whether anonymous invoices were
    calibrated, and their counts and shares. Every real-data runner stamps
    these, so the families identify their data the same way the live
    runners' period record does."""
    return {
        'reader_version':    READER_VERSION,
        'cleaning':          dict(report.extra.get('cleaning') or {}),
        'exclude_anonymous': bool(getattr(args, 'exclude_anonymous', False)),
        'anonymous': {
            'calibration':            params.calibration_extra.get(
                'anonymous_invoices'),
            'n_anonymous_invoices':   int(params.n_anonymous_invoices),
            'anonymous_invoice_frac': float(params.anonymous_invoice_frac),
            'anonymous_revenue_frac': float(params.anonymous_revenue_frac),
            **{k: params.calibration_extra[k]
               for k in ('n_anonymous_invoices_excluded',
                         'n_anonymous_rows_excluded',
                         'anonymous_revenue_excluded')
               if k in params.calibration_extra},
        },
    }


def comparator_summary(optimized: np.ndarray, other: np.ndarray,
                       baseline: np.ndarray) -> Dict:
    """The GA against one comparator over the held-out paired replicates.

    ``paired_mean_diff_GA_minus_X`` and its interval use the bootstrap the
    lift over the as-built layout uses (percentile, 2000 resamples, 95%);
    the percentage is taken of the comparator's mean revenue, as the lift's
    is taken of the baseline's. ``lift_over_baseline`` is the comparator's
    own paired lift over the as-built layout, on the lift's conventions, so
    the headroom each search recovered can be read beside the GA's."""
    diff = optimized - other
    mean_d, lo, hi = bootstrap_ci(diff, alpha=0.05, n_boot=2000)
    denom = max(float(other.mean()), 1e-6)
    lift = other - baseline
    mean_l, lo_l, hi_l = bootstrap_ci(lift, alpha=0.05, n_boot=2000)
    denom_b = max(float(baseline.mean()), 1e-6)
    return {
        'mean':                        float(other.mean()),
        'paired_mean_diff_GA_minus_X': mean_d,
        'ci_lo':                       lo,
        'ci_hi':                       hi,
        'pct_diff':                    mean_d / denom * 100.0,
        'pct_diff_ci':                 [lo / denom * 100.0,
                                        hi / denom * 100.0],
        'pct_of':                      'comparator_mean',
        'lift_over_baseline': {
            'paired_mean_diff': mean_l,
            'ci_lo':            lo_l,
            'ci_hi':            hi_l,
            'pct_lift':         mean_l / denom_b * 100.0,
            'pct_lift_ci':      [lo_l / denom_b * 100.0,
                                 hi_l / denom_b * 100.0],
        },
    }


@dataclass
class CalibratedStore:
    """The calibrated store as the searches and the scoring see it.

    ``init_layout`` is the as-calibrated layout every search starts from;
    ``baseline_layout`` is its image under the GA's repair chain, so the lift
    is measured between two layouts of the same feasible set."""
    shop: Any
    item_names: List[str]
    init_layout: Dict[str, Tuple[float, float]]
    baseline_layout: Dict[str, Tuple[float, float]]
    base_params: Dict[str, Any]

    @property
    def n_sections(self) -> int:
        return sum(1 for w in self.shop.floors[1]['walls']
                   if w.startswith('Section_'))


def build_store(params: CalibratedParams, max_items_per_category: int,
                verbose: bool = True) -> CalibratedStore:
    """Lay the calibrated store out on a headless shop and snapshot its
    starting layout. Deterministic in ``params``: the layout engine runs
    with ``vary=False`` and a fixed seed, so a rebuild -- in another process
    too -- gives the same store."""
    if verbose:
        print("[real] building headless shop from calibration…", flush=True)
    # Figure C uses the naive baseline (the GUI default): the starting
    # layout groups items by category but does NOT pre-cluster by traffic
    # or impulse-near-checkout. This is the user-facing thesis test -- the
    # GA should measurably improve over a realistic un-optimized store.
    shop = build_headless_shop_from_calibration(
        params, max_items_per_category=max_items_per_category,
        naive=True,
    )
    item_names = list(shop.floors[1]['items'].keys())
    if verbose:
        print(f"[real] shop dims {shop.width:.1f}x{shop.height:.1f} m  "
              f"sections={sum(1 for w in shop.floors[1]['walls'] if w.startswith('Section_'))}, "
              f"items_placed={len(item_names)}",
              flush=True)
    if len(item_names) < 2:
        raise SystemExit("Too few items placed to run a GA (need >= 2).")
    # Snapshot the as-calibrated layout. The GA starts from it; the
    # baseline is its image under the GA's repair chain, so the lift is
    # measured between two layouts of the same feasible set.
    init_layout = {
        name: tuple(shop.floors[1]['items'][name]['position'])
        for name in item_names
    }
    baseline_layout = feasible_layout(shop, item_names, init_layout)

    # Anchored at the baseline layout (the as-built one after the shared
    # repair): the elasticities act on score differences from it, so the
    # baseline reproduces the calibrated conversion, basket, spend and
    # daily invoice volume, and the lift is measured from the store the
    # data describe.
    base_params = anchor_base_params(shop, item_names,
                                     base_params_for_calibration(params),
                                     init_layout)
    if verbose:
        print(f"[real] MC base_params: cph={base_params['customers_per_hour']:.1f}, "
              f"conv={base_params['conversion_rate']:.2f} (assumed), "
              f"avg_basket={base_params['avg_basket_size']:.2f}, "
              f"rev_mean={base_params['rev_per_converting_customer']:.2f}",
              flush=True)
    return CalibratedStore(shop=shop, item_names=item_names,
                           init_layout=init_layout,
                           baseline_layout=baseline_layout,
                           base_params=base_params)


#: The GA's operator settings on the calibrated store (``run_ga_search``).
GA_MUT_RATE = 0.18
GA_ELITE_FRAC = 0.20


def run_ga_search(store: CalibratedStore, *, seed: int, n_gens: int,
                  pop_size: int, mc_iters: int,
                  mc_days: int) -> Dict[str, Any]:
    """Figure C's GA search on ``store`` from its as-built layout, seeded
    from ``seed``: ``run_ga_headless``'s output. ``run_searches`` runs the
    GA through here, as would any runner that needs Figure C's optimized
    layout without its comparators, so the two cannot drift apart on the
    operator settings."""
    return run_ga_headless(
        store.shop, store.item_names, store.base_params,
        pop_size=pop_size,
        n_gens=n_gens,
        mut_rate=GA_MUT_RATE,
        elite_frac=GA_ELITE_FRAC,
        mc_iters=mc_iters,
        mc_days=mc_days,
        rng_seed=seed,
        init_layout=store.init_layout,
    )


def run_searches(store: CalibratedStore, *, seed: int, n_gens: int,
                 pop_size: int, mc_iters: int, mc_days: int,
                 sa_initial_accept: float,
                 verbose: bool = True,
                 sa_k_move: Optional[int] = None) -> Dict[str, Any]:
    """The GA, random search and simulated annealing at one budget, from
    the as-built layout, each seeded from ``seed``.

    Returns the three layouts (``'optimized'``, ``'rs'``, ``'sa'``), each on
    the GA's feasible set, with the search statistics, each search's
    convergence trace (``'traces'``), the evaluation counts and the starts
    -- after checking that the searches spent equal search and
    final-selection budgets and all started as-built, since a comparison
    that fails either is not the one the paper reports.

    ``sa_k_move`` adds a fourth search, ``'sa_k'``: the same annealer, seed
    and budget, with moves of ``sa_k_move`` items at a time
    (``zone_neighbor(n_move=...)``) instead of one; it is held to the same
    budgets and start."""
    shop, item_names = store.shop, store.item_names
    base_params, init_layout = store.base_params, store.init_layout

    # -- 3. Run the GA on the calibrated shop ----------------------
    if verbose:
        print(f"[real] running GA (pop={pop_size}, gens={n_gens}, "
              f"mc_iters={mc_iters}, mc_days={mc_days})…",
              flush=True)
    ga_t0 = time.perf_counter()
    ga_out = run_ga_search(store, seed=seed, n_gens=n_gens,
                           pop_size=pop_size, mc_iters=mc_iters,
                           mc_days=mc_days)
    ga_wall = time.perf_counter() - ga_t0
    if verbose:
        print(f"[real] GA done in {ga_wall:.1f}s; best fitness "
              f"(MC mean revenue) = {ga_out['best_fit']:.2f}",
              flush=True)
    optimized_layout = chromosome_to_layout(ga_out['best_chrom'], item_names)

    # -- 4. Equal-budget comparators on the same store ---------------
    # Random search and simulated annealing search the SAME MC objective
    # under the GA's search budget and seeds, starting from the as-built
    # layout the GA's population is seeded from. Their draws and moves stay
    # inside the zone the GA constrains each item to, and every candidate
    # is mapped through feasible_layout before it is scored.
    budget = pop_size * n_gens
    if verbose:
        print(f"[real] running random search and simulated annealing "
              f"(budget={budget} search evaluations, block={pop_size}, "
              f"{GA_N_FINAL_SEEDS} final-selection seeds)…", flush=True)
    rs_stats: Dict = {}
    rs_t0 = time.perf_counter()
    rs_layout = random_search(
        shop, None, item_names, base_params,
        seed=seed, budget=budget, block=pop_size,
        n_final_seeds=GA_N_FINAL_SEEDS,
        mc_iters=mc_iters, mc_days=mc_days, stats=rs_stats,
        init_layout=init_layout,
        sampler=zone_sampler(shop, item_names))
    rs_wall = time.perf_counter() - rs_t0
    sa_stats: Dict = {}
    sa_t0 = time.perf_counter()
    sa_layout = simulated_annealing(
        shop, None, item_names, base_params,
        seed=seed, budget=budget, block=pop_size,
        n_final_seeds=GA_N_FINAL_SEEDS,
        mc_iters=mc_iters, mc_days=mc_days,
        initial_accept=sa_initial_accept, stats=sa_stats,
        init_layout=init_layout, start='asbuilt',
        neighbor=zone_neighbor(shop, item_names, step_frac=SA_STEP_FRAC))
    sa_wall = time.perf_counter() - sa_t0
    if verbose:
        print(f"[real] random search done in {rs_wall:.1f}s; simulated "
              f"annealing done in {sa_wall:.1f}s (T0={sa_stats.get('sa_T0')})",
              flush=True)
    sak_stats: Dict = {}
    sak_layout = None
    sak_wall = 0.0
    if sa_k_move is not None:
        sak_t0 = time.perf_counter()
        sak_layout = simulated_annealing(
            shop, None, item_names, base_params,
            seed=seed, budget=budget, block=pop_size,
            n_final_seeds=GA_N_FINAL_SEEDS,
            mc_iters=mc_iters, mc_days=mc_days,
            initial_accept=sa_initial_accept, stats=sak_stats,
            init_layout=init_layout, start='asbuilt',
            neighbor=zone_neighbor(shop, item_names, step_frac=SA_STEP_FRAC,
                                   n_move=sa_k_move))
        sak_wall = time.perf_counter() - sak_t0
        if verbose:
            print(f"[real] {sa_k_move}-item annealing done in {sak_wall:.1f}s",
                  flush=True)

    # The equal-budget check covers the SEARCH evaluations, which is what
    # the budget buys, and the final selection, where each search re-scores
    # pop_size candidates under the GA's selection seeds.
    eval_counts = {'GA': ga_out['n_search_evals'],
                   'random_search': rs_stats['n_search_evals'],
                   'simulated_annealing': sa_stats['n_search_evals']}
    final_eval_counts = {'GA': ga_out['n_final_evals'],
                         'random_search': rs_stats['n_final_evals'],
                         'simulated_annealing': sa_stats['n_final_evals']}
    starts = {'GA': 'asbuilt',
              'random_search': rs_stats['rs_start'],
              'simulated_annealing': sa_stats['sa_start']}
    if sa_k_move is not None:
        eval_counts['simulated_annealing_k'] = sak_stats['n_search_evals']
        final_eval_counts['simulated_annealing_k'] = sak_stats['n_final_evals']
        starts['simulated_annealing_k'] = sak_stats['sa_start']
    if len(set(eval_counts.values())) != 1:
        raise AssertionError(f"search methods spent unequal "
                             f"search-evaluation budgets {eval_counts}")
    if len(set(final_eval_counts.values())) != 1:
        raise AssertionError(f"search methods spent unequal "
                             f"final-selection budgets {final_eval_counts}")
    if set(starts.values()) != {'asbuilt'}:
        raise AssertionError(f"searches did not share the as-built start: "
                             f"{starts}")
    # Both searchers only evaluate repaired candidates, so these are fixed
    # points of the repair; mapping them keeps the rule uniform.
    rs_layout = feasible_layout(shop, item_names, rs_layout)
    sa_layout = feasible_layout(shop, item_names, sa_layout)
    out = {
        'optimized':         optimized_layout,
        'rs':                rs_layout,
        'sa':                sa_layout,
        'ga_out':            ga_out,
        'rs_stats':          rs_stats,
        'sa_stats':          sa_stats,
        'budget':            budget,
        'eval_counts':       eval_counts,
        'final_eval_counts': final_eval_counts,
        'starts':            starts,
        'traces':            {'GA': ga_out['trace'],
                              'random_search': rs_stats.get('trace'),
                              'simulated_annealing': sa_stats.get('trace')},
        'walls':             {'GA': ga_wall, 'random_search': rs_wall,
                              'simulated_annealing': sa_wall},
    }
    if sa_k_move is not None:
        out['sa_k'] = feasible_layout(shop, item_names, sak_layout)
        out['sak_stats'] = sak_stats
        out['traces']['simulated_annealing_k'] = sak_stats.get('trace')
        out['walls']['simulated_annealing_k'] = sak_wall
        out['sa_k_move'] = int(sa_k_move)
    return out


#: The layouts ``run_searches`` reports, by the names the runs save them
#: under, beside the as-built baseline.
REPORTED_LAYOUTS = ('optimized', 'rs', 'sa', 'sa_k')


def reported_layouts(store: CalibratedStore,
                     searches: Mapping[str, Any]) -> Dict[str, Any]:
    """``{'baseline': ..., 'optimized': ..., 'rs': ..., 'sa': ...[, 'sa_k']}``:
    every layout a run reports, in that order."""
    out = {'baseline': store.baseline_layout}
    for k in REPORTED_LAYOUTS:
        if searches.get(k) is not None:
            out[k] = searches[k]
    return out


def check_reported(store: CalibratedStore,
                   layouts: Mapping[str, Mapping[str, Tuple[float, float]]],
                   label: str = '') -> Dict[str, Dict[str, Any]]:
    """Check every reported layout against the floor-plan engine's
    invariants (aisles the as-built store keeps, every fixture shoppable,
    inside its zone, no overlap); ``LayoutInvariantError`` stops the run on
    the first that breaks one. Returns each layout's invariant record."""
    return {k: checked_layout(store.shop, store.item_names, lay,
                              f"{label}{k}")
            for k, lay in layouts.items()}


def score_layouts(store: CalibratedStore,
                  layouts: Sequence[Tuple[str, Dict[str, Tuple[float, float]]]],
                  *, n_replicates: int, mc_iters: int, mc_days: int,
                  on_replicate: Optional[Callable[[int, Dict[str, float]],
                                                  None]] = None
                  ) -> Dict[str, np.ndarray]:
    """Monte Carlo revenue of each ``(name, layout)`` under the held-out
    evaluation seeds ``EVAL_SEED_BASE + r``, r in [0, ``n_replicates``).

    Every layout is scored under the same seed in a replicate, so the
    differences between them are paired; ``on_replicate(r, {name: revenue})``
    sees each replicate as it completes. Returns ``{name: revenues}`` in
    replicate order."""
    out: Dict[str, List[float]] = {name: [] for name, _ in layouts}
    for r in range(n_replicates):
        mc_seed = EVAL_SEED_BASE + r
        rev = {
            name: paired_mc_revenue(
                store.shop, store.item_names, lay, store.base_params,
                seed=mc_seed, mc_iters=mc_iters, mc_days=mc_days,
            )
            for name, lay in layouts
        }
        for name in out:
            out[name].append(rev[name])
        if on_replicate is not None:
            on_replicate(r, rev)
    return {name: np.asarray(v) for name, v in out.items()}


#: Conversion rates the closed-form scan re-scores the saved layouts at.
#: The lift's claimed invariance to the assumed conversion rate holds only
#: below the rate at which the 0.99 conversion clamps start to bind, which
#: the scan also locates exactly (``clamp_onset``).
CONVERSION_GRID = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
                   0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90,
                   0.95, 0.99)

#: Basket elasticities scored beside the band's own corners: 0.02, a fifth
#: of the band's floor, and no basket effect -- the corners the LHS runner
#: adds below the band (``run_elasticity_lhs.BASKET_CORNERS``).
BASKET_ELASTICITY_POINTS = BASKET_CORNERS


def _at_conversion(base_params: Dict[str, Any], p: float) -> Dict[str, Any]:
    """``base_params`` recalibrated to an assumed conversion rate ``p``,
    as ``base_params_for_calibration`` would build them: the observed
    buyers stay fixed, so the visitor rate is the buyer rate over ``p``,
    and the baseline abandonment is ``ABANDON_FRAC_OF_NONCONVERTERS`` of
    the ``1 - p`` who do not buy. The anchor and every spend input stay."""
    p0 = float(base_params['conversion_rate'])
    out = dict(base_params)
    out['conversion_rate'] = float(p)
    out['customers_per_hour'] = float(base_params['customers_per_hour']) \
        * p0 / float(p)
    out['abandonment_rate'] = (max(0.0, 1.0 - float(p))
                               * ABANDON_FRAC_OF_NONCONVERTERS)
    return out


def _clamps_bind(score: float, breakdown: Mapping[str, Any],
                 base_params: Mapping[str, Any]) -> bool:
    """True when the 0.99 clamp on the lifted or the abandonment-adjusted
    conversion binds for a layout scored ``(score, breakdown)``."""
    d = layout_drivers(score, breakdown, base_params)
    return (d['conv_lifted'] >= CONV_CLAMP_HI
            or d['conv'] >= CONV_CLAMP_HI)


def closed_form_scan(store: CalibratedStore,
                     layouts: Mapping[str, Mapping[str, Tuple[float, float]]],
                     horizon_days: int,
                     baseline: str = 'baseline') -> Dict[str, Any]:
    """Every saved layout scored in closed form (the exact mean of its Monte
    Carlo fitness, ``experiments.closed_form``), and its percentage lift
    over ``baseline`` re-scored where the paper's inputs are assumptions.

    Returns, per layout: its closed-form value, score and per-criterion
    breakdown; its lift over the baseline at the band midpoints the search
    used; the BAND BOX -- the lift at every corner of the three elasticity
    bands (conversion x impulse x basket), and with the basket elasticity at
    each of ``BASKET_ELASTICITY_POINTS`` instead of the band's corners --
    with its range; and the CONVERSION SCAN -- the lift at every assumed
    conversion rate of ``CONVERSION_GRID`` (the store recalibrated to it,
    ``_at_conversion``), the rate above which the 0.99 conversion clamps bind
    for any saved layout (``clamp_onset``, by bisection), and how far the
    lift moves below that rate. The impulse channel cannot move on a store
    without impulse fixtures (the UCI store has none), so its band leaves
    the lift where it is; the scan shows that rather than assuming it."""
    shop, names, bp = store.shop, store.item_names, store.base_params
    scored = {k: layout_score(shop, names, lay) for k, lay in layouts.items()}

    def cf(k, params=bp, **kw):
        s, b = scored[k]
        return float(expected_revenue(params, int(horizon_days), score=s,
                                      breakdown=b, **kw))

    others = [k for k in layouts if k != baseline]
    values = {k: cf(k) for k in layouts}
    lift = {k: (values[k] - values[baseline]) / max(values[baseline], 1e-9)
            * 100.0 for k in others}

    # Band box.
    points = []
    bands = ELASTICITY_BANDS
    bsk_values = tuple(bands['bsk']) + tuple(BASKET_ELASTICITY_POINTS)
    for ec in bands['conv']:
        for ei in bands['imp']:
            for eb in bsk_values:
                e = {'conv': float(ec), 'imp': float(ei), 'bsk': float(eb)}
                base_v = cf(baseline, elasticities=e)
                points.append({
                    **e, 'basket_point': ('band' if eb in bands['bsk']
                                          else 'fixed'),
                    'lift_pct': {k: (cf(k, elasticities=e) - base_v)
                                 / max(base_v, 1e-9) * 100.0 for k in others}})

    def _range(sel):
        return {k: {'min': float(min(p['lift_pct'][k] for p in sel)),
                    'max': float(max(p['lift_pct'][k] for p in sel))}
                for k in others} if sel else {}

    band_box = {
        'bands': {k: list(v) for k, v in bands.items()},
        'basket_points': list(BASKET_ELASTICITY_POINTS),
        'points': points,
        'range_all': _range(points),
        'range_band_corners': _range([p for p in points
                                      if p['basket_point'] == 'band']),
        **{f'range_bsk_{eb:g}': _range([p for p in points
                                        if p['basket_point'] == 'fixed'
                                        and p['bsk'] == eb])
           for eb in BASKET_ELASTICITY_POINTS},
    }

    # Conversion scan.
    p0 = float(bp['conversion_rate'])
    grid = sorted(set(CONVERSION_GRID) | {p0})
    scan_lift = {k: [] for k in others}
    binds = []
    for p in grid:
        bpp = _at_conversion(bp, p)
        base_v = cf(baseline, params=bpp)
        for k in others:
            scan_lift[k].append((cf(k, params=bpp) - base_v)
                                / max(base_v, 1e-9) * 100.0)
        binds.append(any(_clamps_bind(*scored[k], bpp) for k in layouts))

    def _any_bind(p):
        bpp = _at_conversion(bp, p)
        return any(_clamps_bind(*scored[k], bpp) for k in layouts)

    onset = None
    if _any_bind(0.999):
        lo, hi = 1e-4, 0.999
        if _any_bind(lo):
            onset = lo
        else:
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if _any_bind(mid):
                    hi = mid
                else:
                    lo = mid
            onset = hi
    at_p0 = {k: scan_lift[k][grid.index(p0)] for k in others}
    below = [i for i, p in enumerate(grid) if onset is None or p < onset]
    dev = {k: (float(max(abs(scan_lift[k][i] - at_p0[k]) for i in below))
               if below else None) for k in others}
    return {
        'horizon_days': int(horizon_days),
        'values': values,
        'scores': {k: {'score': float(s), 'breakdown': {
            c: float(v) for c, v in b.items() if isinstance(v, (int, float))}}
            for k, (s, b) in scored.items()},
        'lift_pct': lift,
        'band_box': band_box,
        'conversion_scan': {
            'grid': grid, 'reference_rate': p0, 'lift_pct': scan_lift,
            'clamps_bind': binds, 'clamp_onset': onset,
            'clamp': CONV_CLAMP_HI,
            'lift_at_reference_pct': at_p0,
            'max_abs_change_below_onset_pp': dev,
        },
    }


def closed_form_check(cf_scan: Dict[str, Any], results: Dict[str, Any]
                      ) -> Dict[str, Any]:
    """Whether Figure C's conclusions hold with every layout at its exact
    mean: the sign of the lift and of each GA-minus-comparator difference,
    and whether each closed-form difference lies inside the Monte Carlo
    interval the run reports for it (an unbiased estimate's interval
    should hold its mean)."""
    v = cf_scan['values']
    out: Dict[str, Any] = {}
    lift = v['optimized'] - v['baseline']
    out['lift'] = {'cf': lift, 'cf_pct': cf_scan['lift_pct']['optimized'],
                   'mc': results['paired_mean_diff'],
                   'sign_unchanged': bool(np.sign(lift)
                                          == np.sign(results['paired_mean_diff'])),
                   'cf_inside_mc_ci': bool(results['ci_lo'] <= lift
                                           <= results['ci_hi'])}
    for key, lay in (('random_search', 'rs'), ('simulated_annealing', 'sa')):
        c = results['comparators'][key]
        d = v['optimized'] - v[lay]
        out[key] = {'cf': d, 'mc': c['paired_mean_diff_GA_minus_X'],
                    'sign_unchanged': bool(np.sign(d) == np.sign(
                        c['paired_mean_diff_GA_minus_X'])),
                    'mc_excludes_zero': bool(c['ci_lo'] > 0 or c['ci_hi'] < 0),
                    'cf_inside_mc_ci': bool(c['ci_lo'] <= d <= c['ci_hi'])}
    out['all_signs_unchanged'] = bool(all(
        out[k]['sign_unchanged']
        for k in ('lift', 'random_search', 'simulated_annealing')))
    return out


def main() -> int:
    args = parse_args()
    out_dir = make_run_dir(args.out_root,
                           run_dir_prefix('real_data_uci', args))
    experiment = experiment_name('real_data_uci_figure_c', args)
    print(f"output dir: {out_dir}", flush=True)

    wall_t0 = time.perf_counter()

    # -- 1. Calibrate from the UCI dataset --------------------------
    params, report, reader = calibrate_from_file(args)

    # -- 2. Build the headless shop --------------------------------
    store = build_store(params, args.max_items_per_category)
    shop, item_names = store.shop, store.item_names

    # -- 3 + 4. The GA and the equal-budget comparators ---------------
    searches = run_searches(
        store, seed=args.ga_seed, n_gens=args.n_gens,
        pop_size=args.pop_size, mc_iters=args.mc_iters,
        mc_days=args.mc_days, sa_initial_accept=args.sa_initial_accept)
    rs_stats, sa_stats = searches['rs_stats'], searches['sa_stats']
    budget = searches['budget']
    eval_counts = searches['eval_counts']
    final_eval_counts = searches['final_eval_counts']
    starts = searches['starts']
    ga_wall = searches['walls']['GA']
    rs_wall = searches['walls']['random_search']
    sa_wall = searches['walls']['simulated_annealing']
    reported = reported_layouts(store, searches)
    invariants = check_reported(store, reported)

    # -- 5. Paired-MC: every layout under the same replicate seeds ----
    print(f"[real] paired-MC over {args.n_mc_replicates} replicates "
          f"(same RNG seed per pair)…", flush=True)

    def _print_replicate(r: int, rev: Dict[str, float]) -> None:
        print(f"  rep {r:>3d}: baseline={rev['baseline']:>12.2f}  "
              f"optimized={rev['optimized']:>12.2f}  "
              f"diff={rev['optimized'] - rev['baseline']:+10.2f}  "
              f"GA-RS={rev['optimized'] - rev['rs']:+10.2f}  "
              f"GA-SA={rev['optimized'] - rev['sa']:+10.2f}",
              flush=True)

    revs = score_layouts(
        store, (('baseline', store.baseline_layout),
                ('optimized', searches['optimized']),
                ('rs', searches['rs']),
                ('sa', searches['sa'])),
        n_replicates=args.n_mc_replicates, mc_iters=args.mc_iters,
        mc_days=args.mc_days, on_replicate=_print_replicate)

    baseline_revs_arr = revs['baseline']
    optimized_revs_arr = revs['optimized']
    rs_revs_arr = revs['rs']
    sa_revs_arr = revs['sa']
    diffs_arr = optimized_revs_arr - baseline_revs_arr
    mean_diff, ci_lo, ci_hi = bootstrap_ci(diffs_arr, alpha=0.05, n_boot=2000)
    mean_base = float(baseline_revs_arr.mean())
    mean_opt = float(optimized_revs_arr.mean())
    pct_lift = mean_diff / max(mean_base, 1e-6) * 100.0
    pct_lift_lo = ci_lo / max(mean_base, 1e-6) * 100.0
    pct_lift_hi = ci_hi / max(mean_base, 1e-6) * 100.0

    comparators = {
        'random_search': {
            **comparator_summary(optimized_revs_arr, rs_revs_arr,
                                 baseline_revs_arr),
            'n_search_evals': rs_stats['n_search_evals'],
            'n_final_evals':  rs_stats['n_final_evals'],
            'start':          rs_stats['rs_start'],
            'wall_seconds':   rs_wall,
        },
        'simulated_annealing': {
            **comparator_summary(optimized_revs_arr, sa_revs_arr,
                                 baseline_revs_arr),
            'n_search_evals':    sa_stats['n_search_evals'],
            'n_final_evals':     sa_stats['n_final_evals'],
            'start':             sa_stats['sa_start'],
            'sa_T0':             sa_stats.get('sa_T0'),
            'sa_initial_accept': sa_stats.get('sa_initial_accept'),
            'wall_seconds':      sa_wall,
        },
    }

    print()
    print(f"[real] PAPER FIGURE C HEADLINE:")
    print(f"       baseline mean MC revenue:  {mean_base:>14.2f}")
    print(f"       optimized mean MC revenue: {mean_opt:>14.2f}")
    print(f"       paired mean lift:          {mean_diff:>+14.2f}  "
          f"({pct_lift:+.2f}%)")
    print(f"       95% CI on lift:            [{ci_lo:+.2f}, {ci_hi:+.2f}]  "
          f"([{pct_lift_lo:+.2f}%, {pct_lift_hi:+.2f}%])")
    for label, key in (('random search', 'random_search'),
                       ('sim. annealing', 'simulated_annealing')):
        c = comparators[key]
        print(f"       GA - {label + ':':<22s}"
              f"{c['paired_mean_diff_GA_minus_X']:>+14.2f}  "
              f"({c['pct_diff']:+.2f}%)  95% CI "
              f"[{c['ci_lo']:+.2f}, {c['ci_hi']:+.2f}]")
    print(f"       search evaluations (GA / RS / SA): "
          f"{eval_counts['GA']} / {eval_counts['random_search']} / "
          f"{eval_counts['simulated_annealing']}  "
          f"(final: {final_eval_counts['GA']} / "
          f"{final_eval_counts['random_search']} / "
          f"{final_eval_counts['simulated_annealing']})")

    # Every layout at the exact mean of its fitness, its score breakdown,
    # and its lift re-scored over the elasticity bands and the assumed
    # conversion rate; and whether the headline holds in closed form.
    cf_scan = closed_form_scan(store, reported, args.mc_days)
    results_mc = {'paired_mean_diff': mean_diff, 'ci_lo': ci_lo,
                  'ci_hi': ci_hi, 'comparators': comparators}
    cf_check = closed_form_check(cf_scan, results_mc)
    bb = cf_scan['band_box']['range_all']['optimized']
    print(f"       closed form: lift {cf_scan['lift_pct']['optimized']:+.2f}%"
          f"  (band box {bb['min']:+.2f}% .. {bb['max']:+.2f}%; clamps bind "
          f"above p_c = {cf_scan['conversion_scan']['clamp_onset']})")

    # -- 6. Stamp provenance ---------------------------------------
    prov = stamp_provenance(
        source_path=args.retail_path,
        adapter_name=OnlineRetailIIAdapter.name,
        adapter_version=OnlineRetailIIAdapter.version,
        rows_in=report.rows_in,
        rows_kept=report.rows_kept,
        currency=report.extra.get('currency', 'GBP'),
        seed=args.ga_seed,
        extra={
            'sheets':                  args.sheets,
            # rows_in above counts the frame the adapter received, which the
            # reader has already de-duplicated across sheets; these fields
            # carry the workbook's own row count and what was dropped.
            'reader':                  reader,
            'max_items_per_category':  args.max_items_per_category,
            'mc_iters':                args.mc_iters,
            'mc_days':                 args.mc_days,
            'n_gens':                  args.n_gens,
            'pop_size':                args.pop_size,
            'n_mc_replicates':         args.n_mc_replicates,
            'shop_dims_m':             [shop.width, shop.height],
            'sections':                sum(1 for w in shop.floors[1]['walls']
                                           if w.startswith('Section_')),
            'items_placed':            len(item_names),
            # Reader version, cleaning counts, anonymous invoices.
            **data_provenance_extra(report, params, args),
        },
    )

    # -- 7. Write CSV + figure + sidecar + summary ---------------------
    csv_path = os.path.join(out_dir, 'results.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['replicate', 'mc_seed', 'baseline_revenue',
                    'optimized_revenue', 'diff',
                    'rs_revenue', 'sa_revenue', 'diff_rs', 'diff_sa'])
        for r, (b, o, d, rs, sa) in enumerate(zip(
                baseline_revs_arr, optimized_revs_arr, diffs_arr,
                rs_revs_arr, sa_revs_arr)):
            w.writerow([r, EVAL_SEED_BASE + r,
                        f"{b:.4f}", f"{o:.4f}", f"{d:.4f}",
                        f"{rs:.4f}", f"{sa:.4f}",
                        f"{o - rs:.4f}", f"{o - sa:.4f}"])

    png_path = make_figure_c(
        out_dir, baseline_revs_arr, optimized_revs_arr,
        diffs_arr, ci_lo, ci_hi, params,
        comparators=[
            (label,
             comparators[key]['paired_mean_diff_GA_minus_X'],
             comparators[key]['ci_lo'],
             comparators[key]['ci_hi'])
            for label, key in (('random\nsearch', 'random_search'),
                               ('simulated\nannealing',
                                'simulated_annealing'))
        ],
    )

    results = {
        'baseline_mean':   mean_base,
        'optimized_mean':  mean_opt,
        'paired_mean_diff': mean_diff,
        'ci_lo':           ci_lo,
        'ci_hi':           ci_hi,
        'pct_lift':        pct_lift,
        'pct_lift_ci':     [pct_lift_lo, pct_lift_hi],
        'n_replicates':    args.n_mc_replicates,
        # The GA against each equal-budget search, paired on the same
        # held-out replicates as the lift above (GA minus X).
        'comparators':     comparators,
        # Every layout at the exact mean of its fitness, its score
        # breakdown, the band box and the conversion scan
        # (``closed_form_scan``), and whether each conclusion above holds in
        # closed form (``closed_form_check``).
        'closed_form':     cf_scan,
        'closed_form_check': cf_check,
    }
    # The floor-plan invariants of every reported layout; how much of the
    # zones the aisle rule takes from the searches (``aisle_rule``); and how
    # often the region clip and the reachability fallbacks bound during them
    # (the searches' operators stay inside the regions, so the clip is a
    # safety net expected at zero).
    feasibility = {'invariants': invariants,
                   'repair_stats': repair_stats(shop),
                   'aisle_rule': aisle_rule_summary(shop, item_names)}
    write_json(out_dir, 'layouts.json', {
        k: layout_json(lay) for k, lay in reported.items()})
    write_json(out_dir, 'traces.json', searches['traces'])
    # How the three searches were held level: one budget, one block, one
    # number of final-selection seeds, one start, one seed family.
    search_design = {
        'budget_search_evals':  budget,
        'budget_final_evals':   args.pop_size * GA_N_FINAL_SEEDS,
        'final_pool':           {'GA': 'final population (elites + '
                                       'unscored children)',
                                 'random_search': rs_stats.get('pool_rule'),
                                 'simulated_annealing':
                                     sa_stats.get('pool_rule')},
        'block':                args.pop_size,
        'n_final_seeds':        GA_N_FINAL_SEEDS,
        'seed':                 args.ga_seed,
        'starts':               starts,
        'search_seed_ranges':   search_seed_ranges(args.ga_seed, args.n_gens,
                                                   args.pop_size),
        'paired_eval_seeds':    [EVAL_SEED_BASE,
                                 EVAL_SEED_BASE + args.n_mc_replicates],
        'rs_sampler':           'zone_sampler',
        'sa_neighbor':          f'zone_neighbor(step_frac={SA_STEP_FRAC})',
        'operators':            search_operators(),
        'ci':                   'percentile bootstrap, 2000 resamples, 95%, '
                                'per comparison (uncorrected)',
    }
    sa_schedule = {
        'sa_initial_accept': sa_stats.get('sa_initial_accept'),
        'sa_T0':             sa_stats.get('sa_T0'),
        'sa_start':          sa_stats.get('sa_start'),
    }

    # What the store was calibrated from, in the summary itself: a
    # validator of the headline must refuse a no-anonymous sensitivity run
    # (``exclude_anonymous``) and a run under older cleaning
    # (``adapter_version``).
    data_design = {
        'sheets':                 args.sheets,
        'max_items_per_category': args.max_items_per_category,
        'assumed_conversion':     args.assumed_conversion,
        'exclude_anonymous':      bool(args.exclude_anonymous),
        'adapter_version':        OnlineRetailIIAdapter.version,
    }

    write_sidecar(out_dir, {
        'experiment':        experiment,
        'args':              vars(args),
        'wall_seconds':      time.perf_counter() - wall_t0,
        'ga_wall_seconds':   ga_wall,
        'rs_wall_seconds':   rs_wall,
        'sa_wall_seconds':   sa_wall,
        'csv_path':          os.path.relpath(csv_path, out_dir),
        'figure_path':       os.path.relpath(png_path, out_dir),
        'provenance':        prov.to_dict(),
        'reader':            reader,
        'calibration_summary': {
            'n_invoices':              params.n_invoices,
            'n_unique_products':       params.n_unique_products,
            'n_unique_customers':      params.n_unique_customers,
            'n_unique_categories':     params.n_unique_categories,
            'span_seconds':            params.span_seconds,
            'arrivals_per_hour':       params.arrivals_per_hour,
            'visitors_per_hour':       params.visitors_per_hour,
            'assumed_conversion_rate': params.assumed_conversion_rate,
            'conversion_rate_source':  'assumption',  # explicit marker
            'return_customer_rate':    params.return_customer_rate,
            'currency':                params.currency,
            # Data-quality descriptors.
            'basket_units_median':     params.basket_units_median,
            'basket_distinct_median':  params.basket_distinct_median,
            'category_fallback_frac':  params.category_fallback_frac,
        },
        'shop_summary': {
            'width_m':           shop.width,
            'height_m':          shop.height,
            'sections':          sum(1 for w in shop.floors[1]['walls']
                                     if w.startswith('Section_')),
            'items_placed':      len(item_names),
        },
        # The inputs every layout was scored with, the as-built anchor
        # included (lists as length, moments and a hash).
        'base_params': base_params_record(store.base_params),
        'results': results,
        # Search-evaluation totals (checked equal) and the final-selection
        # evaluations each search spent on top of them.
        'evaluation_counts':       eval_counts,
        'final_evaluation_counts': final_eval_counts,
        'sa_schedule':             sa_schedule,
        'search_design':           search_design,
        'feasibility':             feasibility,
        'layouts_path':            'layouts.json',
        'traces_path':             'traces.json',
        'data':                    data_design,
    })

    # Written last: its presence marks a finished run.
    with open(os.path.join(out_dir, 'summary.json'), 'w',
              encoding='utf-8') as f:
        json.dump({
            'experiment':              experiment,
            'data':                    data_design,
            'results':                 results,
            'evaluation_counts':       eval_counts,
            'final_evaluation_counts': final_eval_counts,
            'sa_schedule':             sa_schedule,
            'search_design':           search_design,
            'feasibility':             feasibility,
            'layouts_path':            'layouts.json',
            'traces_path':             'traces.json',
        }, f, indent=2)

    print(f"\nArtifacts in: {out_dir}")
    return 0


def make_figure_c(out_dir: str,
                  baseline: np.ndarray,
                  optimized: np.ndarray,
                  diffs: np.ndarray,
                  ci_lo: float,
                  ci_hi: float,
                  params: CalibratedParams,
                  comparators: Optional[Sequence[Tuple[str, float, float,
                                                       float]]] = None
                  ) -> str:
    """Three panels: per-replicate revenue of the as-built and GA layouts,
    their paired scatter, and the paired differences with bootstrap CIs.

    ``comparators`` adds a bar per ``(label, GA-minus-X mean, ci_lo,
    ci_hi)`` to the right panel beside the lift over the as-built layout,
    so every bar reads the same way: above zero, the GA's layout earns
    more."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    fig = plt.figure(figsize=(13, 4.8))
    # No explicit wspace: tight_layout gives up on a gridspec that fixes its
    # own spacing, and the panel titles then run into the suptitle.
    gs = fig.add_gridspec(1, 3, width_ratios=[3, 3, 3])

    # Left: overlapping histograms of MC revenue
    ax_h = fig.add_subplot(gs[0, 0])
    all_vals = np.concatenate([baseline, optimized])
    lo = float(np.percentile(all_vals, 1))
    hi = float(np.percentile(all_vals, 99))
    bins = np.linspace(lo, hi, 25)
    ax_h.hist(baseline, bins=bins, alpha=0.55, color='#4ECDC4',
              label='As-calibrated layout', edgecolor='#2a8a85')
    ax_h.hist(optimized, bins=bins, alpha=0.55, color='#FF6B6B',
              label='GA-optimized layout', edgecolor='#a83434')
    ax_h.set_xlabel(f'MC mean revenue ({params.currency})')
    ax_h.set_ylabel('Replicate count')
    ax_h.set_title(f'Per-replicate revenue\n({len(baseline)} paired MC seeds)')
    ax_h.legend(fontsize=8, framealpha=0.9)
    # Revenue levels are large and close together; fewer ticks keep their
    # labels from running into each other.
    ax_h.xaxis.set_major_locator(MaxNLocator(nbins=4))

    # Middle: scatter of paired (baseline, optimized) with y=x reference
    ax_s = fig.add_subplot(gs[0, 1])
    ax_s.scatter(baseline, optimized, s=22, alpha=0.7, color='#FF6B6B',
                 edgecolor='#a83434', zorder=3)
    lo_s = float(min(baseline.min(), optimized.min()))
    hi_s = float(max(baseline.max(), optimized.max()))
    ax_s.plot([lo_s, hi_s], [lo_s, hi_s], '--', color='#888',
              linewidth=1.0, label='y = x (no lift)')
    ax_s.set_xlabel(f'Baseline revenue ({params.currency})')
    ax_s.set_ylabel(f'Optimized revenue ({params.currency})')
    ax_s.set_title('Paired-MC: optimized vs. baseline\n(points above diagonal = GA wins)')
    ax_s.legend(fontsize=8, loc='upper left')
    ax_s.xaxis.set_major_locator(MaxNLocator(nbins=4))

    # Right: GA minus each layout, paired, with bootstrap CIs. The first
    # bar is the lift over the as-built store; the rest are the
    # equal-budget searches.
    ax_b = fig.add_subplot(gs[0, 2])
    bars = [('as-built\n(lift)', float(diffs.mean()), float(ci_lo),
             float(ci_hi), '#FF6B6B')]
    bars += [(label, float(m), float(c_lo), float(c_hi), '#B8C4CE')
             for label, m, c_lo, c_hi in (comparators or [])]
    xs = np.arange(len(bars))
    means = [b[1] for b in bars]
    err = [[b[1] - b[2] for b in bars], [b[3] - b[1] for b in bars]]
    ax_b.bar(xs, means, yerr=err, width=0.6, color=[b[4] for b in bars],
             alpha=0.85, capsize=8, edgecolor='black')
    ax_b.axhline(0, color='black', linewidth=0.6)
    ax_b.set_xticks(xs)
    ax_b.set_xticklabels([b[0] for b in bars], fontsize=8)
    ax_b.set_ylabel(f'GA minus layout, paired mean ({params.currency})')
    for x, (_, m, c_lo, c_hi, _) in zip(xs, bars):
        sign = '+' if c_lo > 0 else ('-' if c_hi < 0 else '~')
        above = m >= 0
        ax_b.annotate(f'[{c_lo:+.0f}, {c_hi:+.0f}]\n({sign})',
                      xy=(x, c_hi if above else c_lo),
                      xytext=(0, 4 if above else -4),
                      textcoords='offset points', ha='center',
                      va='bottom' if above else 'top', fontsize=7)
    ax_b.margins(y=0.25)
    ax_b.set_title('Paired difference, 95% bootstrap CI\n'
                   '(+ / -: CI excludes 0; ~: it does not)')

    fig.suptitle(
        f"Figure C: UCI Online Retail II "
        f"({params.n_invoices:,} invoices, {params.n_unique_categories} categories)",
        fontsize=11, fontweight='bold'
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95], w_pad=2.0)
    png_path = os.path.join(out_dir, 'figure_c.png')
    fig.savefig(png_path, dpi=150)
    pdf_path = os.path.join(out_dir, 'figure_c.pdf')
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path


if __name__ == '__main__':
    sys.exit(main())
