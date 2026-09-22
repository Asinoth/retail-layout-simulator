"""Figure C: paper's real-data worked example.

Loads UCI Online Retail II (transactional, ~1M invoices 2009-2011) end-
to-end through the calibration pipeline, materializes the resulting
multi-section shop on a headless instance, runs the GA optimizer, and
reports the paired-MC projected revenue lift over the as-calibrated
baseline layout (mapped onto the GA's feasible set, like every GA
candidate), evaluated on seeds the GA never searched under.

This is the script a TOMACS reviewer should be able to run to reproduce
the paper's real-data application figure. Output sidecar.json records dataset
SHA-256, git SHA, elasticity snapshot, and the conversion-rate
assumption marker -- everything needed for end-to-end reproducibility.

Smoke (small dataset slice + small GA budget; <5 min):
    python -m experiments.run_real_data_example ^
        --retail-path C:\\path\\to\\online_retail_II.xlsx ^
        --sheets "Year 2010-2011" ^
        --max-items-per-category 8 ^
        --mc-iters 200 --n-gens 5 --pop-size 10 --n-mc-replicates 5

Paper-grade (both sheets + Tier-1 GA budget; ~15-30 min):
    python -m experiments.run_real_data_example ^
        --retail-path C:\\path\\to\\online_retail_II.xlsx ^
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
import os
import re
import sys
import time
from typing import Dict, List

import numpy as np

# Make sibling project modules importable when run as ``python -m ...``.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from dataset_adapters import (
    OnlineRetailIIAdapter,
    list_excel_sheets,
    read_excel_sheets,
)
from dataset_calibration import calibrate_transactional, CalibratedParams
from dataset_provenance import stamp as stamp_provenance

from experiments._common import (
    build_headless_shop_from_calibration,
    base_params_for_calibration,
    run_ga_headless,
    paired_mc_revenue,
    bootstrap_ci,
    apply_layout,
    layout_to_chromosome,
    chromosome_to_layout,
    feasible_layout,
    make_run_dir,
    write_sidecar,
)


# Paired-evaluation seeds are EVAL_SEED_BASE + replicate. The GA searches
# under ga_seed*1000 + [0, n_gens + 6) (generation seeds, a gap, then 5
# final-selection seeds); ``parse_args`` keeps that range below the base.
EVAL_SEED_BASE = 1_000_000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--retail-path', type=str, required=True,
                   help="Path to online_retail_II.xlsx (the UCI dataset)")
    p.add_argument('--sheets', type=str,
                   default="Year 2009-2010,Year 2010-2011",
                   help="Comma-separated Excel sheet names to concatenate")
    p.add_argument('--max-items-per-category', type=int, default=12,
                   help="Cap on items placed per inferred category")
    p.add_argument('--assumed-conversion', type=float, default=0.30,
                   help="Conversion rate assumption (transactional data has no buy/no-buy split)")
    p.add_argument('--mc-iters', type=int, default=2000)
    p.add_argument('--mc-days', type=int, default=30)
    p.add_argument('--n-gens', type=int, default=25)
    p.add_argument('--pop-size', type=int, default=30)
    p.add_argument('--n-mc-replicates', type=int, default=30,
                   help="Number of paired-MC replicate seeds used to estimate the lift CI")
    p.add_argument('--ga-seed', type=int, default=0,
                   help="RNG seed for the GA initialization")
    p.add_argument('--out-root', type=str,
                   default=os.path.join(_HERE, 'results'))
    args = p.parse_args()
    if (args.ga_seed < 0 or args.n_gens + 6 > 1000
            or (args.ga_seed + 1) * 1000 > EVAL_SEED_BASE):
        p.error("--ga-seed must be in [0, 999] and --n-gens + 6 <= 1000 so "
                "the GA's search seeds stay below the evaluation seeds")
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

    adapter = OnlineRetailIIAdapter()
    normalized, report = adapter.adapt(df)
    if report.is_blocking():
        raise SystemExit(
            "UCI adapter validation is blocking; see report:\n\n"
            + report.to_text()
        )
    print(f"[real] adapter kept {report.rows_kept:,}/{report.rows_in:,} rows "
          f"({adapter.name} v{adapter.version})",
          flush=True)

    t0 = time.perf_counter()
    params = calibrate_transactional(
        normalized,
        assumed_conversion_rate=args.assumed_conversion,
        currency=report.extra.get('currency', 'GBP'),
    )
    print(f"[real] calibrated in {time.perf_counter()-t0:.1f}s: "
          f"{params.n_invoices:,} invoices, "
          f"{params.n_unique_products:,} unique products, "
          f"{params.n_unique_categories} categories, "
          f"{params.arrivals_per_hour:.1f} invoices/hr (avg)",
          flush=True)
    return params, report, reader


def main() -> int:
    args = parse_args()
    out_dir = make_run_dir(args.out_root, 'real_data_uci')
    print(f"output dir: {out_dir}", flush=True)

    wall_t0 = time.perf_counter()

    # -- 1. Calibrate from the UCI dataset --------------------------
    params, report, reader = calibrate_from_file(args)

    # -- 2. Build the headless shop --------------------------------
    print("[real] building headless shop from calibration…", flush=True)
    # Figure C uses the naive baseline (the GUI default): the starting
    # layout groups items by category but does NOT pre-cluster by traffic
    # or impulse-near-checkout. This is the user-facing thesis test -- the
    # GA should measurably improve over a realistic un-optimized store.
    shop = build_headless_shop_from_calibration(
        params, max_items_per_category=args.max_items_per_category,
        naive=True,
    )
    item_names = list(shop.floors[1]['items'].keys())
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

    base_params = base_params_for_calibration(params)
    print(f"[real] MC base_params: cph={base_params['customers_per_hour']:.1f}, "
          f"conv={base_params['conversion_rate']:.2f} (assumed), "
          f"avg_basket={base_params['avg_basket_size']:.2f}, "
          f"rev_mean={base_params['rev_per_converting_customer']:.2f}",
          flush=True)

    # -- 3. Run the GA on the calibrated shop ----------------------
    print(f"[real] running GA (pop={args.pop_size}, gens={args.n_gens}, "
          f"mc_iters={args.mc_iters}, mc_days={args.mc_days})…",
          flush=True)
    ga_t0 = time.perf_counter()
    ga_out = run_ga_headless(
        shop, item_names, base_params,
        pop_size=args.pop_size,
        n_gens=args.n_gens,
        mut_rate=0.18,
        elite_frac=0.20,
        mc_iters=args.mc_iters,
        mc_days=args.mc_days,
        rng_seed=args.ga_seed,
        init_layout=init_layout,
    )
    ga_wall = time.perf_counter() - ga_t0
    print(f"[real] GA done in {ga_wall:.1f}s; best fitness "
          f"(MC mean revenue) = {ga_out['best_fit']:.2f}",
          flush=True)
    optimized_layout = chromosome_to_layout(ga_out['best_chrom'], item_names)

    # -- 4. Paired-MC: baseline vs. optimized across replicate seeds --
    print(f"[real] paired-MC over {args.n_mc_replicates} replicates "
          f"(same RNG seed per pair)…", flush=True)
    baseline_revs: List[float] = []
    optimized_revs: List[float] = []
    diffs: List[float] = []
    for r in range(args.n_mc_replicates):
        mc_seed = EVAL_SEED_BASE + r
        rev_base = paired_mc_revenue(
            shop, item_names, baseline_layout, base_params,
            seed=mc_seed, mc_iters=args.mc_iters, mc_days=args.mc_days,
        )
        rev_opt = paired_mc_revenue(
            shop, item_names, optimized_layout, base_params,
            seed=mc_seed, mc_iters=args.mc_iters, mc_days=args.mc_days,
        )
        baseline_revs.append(rev_base)
        optimized_revs.append(rev_opt)
        diffs.append(rev_opt - rev_base)
        print(f"  rep {r:>3d}: baseline={rev_base:>12.2f}  "
              f"optimized={rev_opt:>12.2f}  diff={rev_opt - rev_base:+10.2f}",
              flush=True)

    baseline_revs_arr = np.asarray(baseline_revs)
    optimized_revs_arr = np.asarray(optimized_revs)
    diffs_arr = np.asarray(diffs)
    mean_diff, ci_lo, ci_hi = bootstrap_ci(diffs_arr, alpha=0.05, n_boot=2000)
    mean_base = float(baseline_revs_arr.mean())
    mean_opt = float(optimized_revs_arr.mean())
    pct_lift = mean_diff / max(mean_base, 1e-6) * 100.0
    pct_lift_lo = ci_lo / max(mean_base, 1e-6) * 100.0
    pct_lift_hi = ci_hi / max(mean_base, 1e-6) * 100.0

    print()
    print(f"[real] PAPER FIGURE C HEADLINE:")
    print(f"       baseline mean MC revenue:  {mean_base:>14.2f}")
    print(f"       optimized mean MC revenue: {mean_opt:>14.2f}")
    print(f"       paired mean lift:          {mean_diff:>+14.2f}  "
          f"({pct_lift:+.2f}%)")
    print(f"       95% CI on lift:            [{ci_lo:+.2f}, {ci_hi:+.2f}]  "
          f"([{pct_lift_lo:+.2f}%, {pct_lift_hi:+.2f}%])")

    # -- 5. Stamp provenance ---------------------------------------
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
        },
    )

    # -- 6. Write CSV + sidecar + figure ---------------------------
    csv_path = os.path.join(out_dir, 'results.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['replicate', 'mc_seed', 'baseline_revenue',
                    'optimized_revenue', 'diff'])
        for r, (b, o, d) in enumerate(zip(baseline_revs, optimized_revs, diffs)):
            w.writerow([r, EVAL_SEED_BASE + r,
                        f"{b:.4f}", f"{o:.4f}", f"{d:.4f}"])

    png_path = make_figure_c(
        out_dir, baseline_revs_arr, optimized_revs_arr,
        diffs_arr, ci_lo, ci_hi, params,
    )

    write_sidecar(out_dir, {
        'experiment':        'real_data_uci_figure_c',
        'args':              vars(args),
        'wall_seconds':      time.perf_counter() - wall_t0,
        'ga_wall_seconds':   ga_wall,
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
            # Data-quality descriptors (audit R7.3/R7.5).
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
        'results': {
            'baseline_mean':   mean_base,
            'optimized_mean':  mean_opt,
            'paired_mean_diff': mean_diff,
            'ci_lo':           ci_lo,
            'ci_hi':           ci_hi,
            'pct_lift':        pct_lift,
            'pct_lift_ci':     [pct_lift_lo, pct_lift_hi],
            'n_replicates':    args.n_mc_replicates,
        },
    })

    print(f"\nArtifacts in: {out_dir}")
    return 0


def make_figure_c(out_dir: str,
                  baseline: np.ndarray,
                  optimized: np.ndarray,
                  diffs: np.ndarray,
                  ci_lo: float,
                  ci_hi: float,
                  params: CalibratedParams) -> str:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11, 4.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[3, 3, 2], wspace=0.35)

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

    # Right: lift bar with bootstrap CI
    ax_b = fig.add_subplot(gs[0, 2])
    mean_diff = float(diffs.mean())
    err = [[mean_diff - ci_lo], [ci_hi - mean_diff]]
    ax_b.bar(['GA lift'], [mean_diff], yerr=err,
             color='#FF6B6B', alpha=0.85, capsize=8, edgecolor='black')
    ax_b.axhline(0, color='black', linewidth=0.6)
    ax_b.set_ylabel(f'Paired mean lift ({params.currency})')
    sign = '+' if ci_lo > 0 else ('-' if ci_hi < 0 else '~')
    ax_b.set_title(f'95% bootstrap CI\n[{ci_lo:+.0f}, {ci_hi:+.0f}]  ({sign} significant)')

    fig.suptitle(
        f"Figure C: UCI Online Retail II "
        f"({params.n_invoices:,} invoices, {params.n_unique_categories} categories)",
        fontsize=11, fontweight='bold'
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    png_path = os.path.join(out_dir, 'figure_c.png')
    fig.savefig(png_path, dpi=150)
    pdf_path = os.path.join(out_dir, 'figure_c.pdf')
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path


if __name__ == '__main__':
    sys.exit(main())
