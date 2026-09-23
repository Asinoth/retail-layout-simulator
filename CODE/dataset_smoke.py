"""Headless end-to-end smoke for the dataset ingestion pipeline.

Drives every real dataset through the full pipeline (adapter ->
calibration -> layout -> seed) without Tk, asserting that each stage
produces sane, non-empty output. The files are found by ``dataset_paths``
-- under ``DATASETS/``, or wherever ``UCI_RETAIL_XLSX``,
``OMNICHANNEL_DIR``, ``OPENTRAJ_ETH_OBSMAT`` or ``RETAIL_DATASETS_DIR``
point -- the same search the GUI chooser and the runners use. Run:

    python dataset_smoke.py

Exits 0 if all datasets pass, 1 otherwise; a missing dataset fails its
check with the list of paths tried. This is the project's idiomatic
``__main__`` smoke (cf. oracle.py / baselines.py), a standalone script
rather than part of the pytest suite.

What it guards:
  * OpenTraj ETH obsmat -> OpenTrajAdapter -> calibrate_trajectory -> seed
  * Omnichannel bundle  -> load_omnichannel_bundle -> calibrate_omnichannel
                           -> build_layout_from_calibration -> seed
  * UCI Online Retail II -> OnlineRetailIIAdapter -> calibrate_transactional
                           -> build_layout_from_calibration -> seed
  * The cosmetic layout convergence (section colors, WC).
"""

from __future__ import annotations

import os
import sys
import traceback

import numpy as np

import dataset_adapters as DA
import dataset_calibration as DC
import dataset_paths
from dataset_layout import build_layout_from_calibration

# Each check resolves its files when it runs, so one missing dataset fails
# that check -- with the paths tried -- rather than the whole script.


class StubSim:
    """Tk-free stand-in exposing the attributes that ``seed_into`` and
    ``build_layout_from_calibration`` read/write."""
    def __init__(self):
        self.analytics = {}
        self.heat_map_resolution = 20
        self.run_time = 0.0
        self.sim_time = 0.0
        self.hourly_profile = None
        self.sim_clock_start_hour = 9.0
        self.heat_raw = np.zeros((16, 16), dtype=np.float32)
        self.heat_map_data = np.zeros((16, 16))
        self._floor_heat_raw = {1: self.heat_raw}
        self.door_position = None
        self.door_side = None
        self.geometry_dirty = False
        self.seed = 0

    def invalidate_zones_cache(self):
        pass


class StubShop:
    """Minimal multi-floor shop for ``build_layout_from_calibration``."""
    def __init__(self):
        self.floors = {1: {}}
        self.connectors = {}
        self.num_floors = 1
        self.current_floor = 1
        self.width = 20.0
        self.height = 15.0
        self.door_position = None
        self.door_side = None
        self.customer_simulation = StubSim()


def _assert(cond, msg):
    if not cond:
        raise AssertionError(msg)


def _check_cosmetics(shop, expect_alphabetical=False):
    """Visual convergence assertions: colored sections + a WC, and (when
    ``expect_alphabetical``, the GUI naive default) departments assembled
    in alphabetical category order."""
    walls = shop.floors[1]["walls"]
    section_walls = [(n, w) for n, w in walls.items() if n.startswith("Section_")]
    _assert(section_walls, "no Section_ walls produced")
    _assert(all("color" in w for _, w in section_walls),
            "section walls missing 'color' (convergence regression)")
    _assert("WC" in walls, "WC wall not placed (convergence regression)")
    if expect_alphabetical:
        # Verify departments were assembled in alphabetical category order
        # (naive=True path). Zone walls may carry a numeric `_N` suffix
        # when a department splits across two gondolas -- compare on the
        # BASE category name, which must be non-decreasing.
        def _base(nm):
            s = nm.replace("Section_", "", 1)
            head, _, tail = s.rpartition("_")
            return head if (head and tail.isdigit()) else s
        cats_placed = [_base(n) for n, _ in section_walls]
        _assert(cats_placed == sorted(cats_placed),
                f"sections not in alphabetical order under naive=True: "
                f"{cats_placed[:6]}")
    return len(section_walls)


def smoke_opentraj():
    print("\n[OpenTraj / ETH]")
    eth_obsmat = dataset_paths.opentraj_eth_obsmat()
    df, notes = DA.read_any(eth_obsmat)
    kind = DA.detect_schema_kind(df, eth_obsmat)
    _assert(kind == "trajectory", f"detected {kind}, expected trajectory")
    adapter = DA.best_trajectory_adapter(df, eth_obsmat)
    _assert(adapter.name == "opentraj", f"chose {adapter.name}, expected opentraj")
    out, report = adapter.adapt(df)
    _assert(len(out) > 0 and not report.is_blocking(), "empty/blocking adapt")
    sp = DC.calibrate_trajectory(out, target_width_m=20.0, target_height_m=15.0,
                                 heat_resolution=20)
    _assert(sp.n_tracks > 0 and sp.speeds_m_s.size > 0, "no tracks/speeds")
    sim = StubSim()
    sp.seed_into(sim)
    _assert(sim.analytics["calibration"]["spatial_source"] == "trajectory_dataset",
            "spatial calibration not seeded")
    print(f"  adapter={adapter.name} rows={len(out):,} tracks={sp.n_tracks:,} "
          f"speed_mean={sp.speeds_m_s.mean():.2f} m/s span={sp.span_seconds:.0f}s")
    print("  PASS")


def smoke_omnichannel():
    print("\n[Omnichannel]")
    omni_dir = dataset_paths.omnichannel_dir()
    # Detection should fire on the demand CSV (by name and by columns).
    demand_csv = os.path.join(omni_dir, "Demand and Shopping Behavior.csv")
    ddf, _ = DA.read_any(demand_csv)
    _assert(DA.looks_like_omnichannel(ddf, demand_csv),
            "looks_like_omnichannel did not detect the demand CSV")
    families, arrivals, notes = DA.load_omnichannel_bundle(omni_dir)
    _assert(len(families) > 0, "empty families frame")
    params = DC.calibrate_omnichannel(families, arrivals)
    _assert(params.n_unique_products > 0 and params.n_unique_categories > 0,
            "no products/categories")
    _assert(params.conversion_rate_source in ("estimated_from_purchase_probabilities",
                                              "assumption"),
            "conversion source not tagged")
    shop = StubShop()
    stats = build_layout_from_calibration(shop, params)
    _assert(stats["sections"] > 0 and stats["items"] > 0, "empty layout")
    params.seed_into(shop.customer_simulation)
    cal = shop.customer_simulation.analytics["calibration"]
    _assert(cal.get("source_kind") == "aggregate_retail_omnichannel",
            "aggregate source_kind not seeded")
    # Naive baseline by default (GUI path). Sections alphabetical, no
    # impulse-shelf relocation -- so the GA has measurable room to improve.
    n_sec = _check_cosmetics(shop, expect_alphabetical=True)
    print(f"  families={params.n_unique_products} zones={params.n_unique_categories} "
          f"arr/hr={params.arrivals_per_hour:.1f} conv={params.assumed_conversion_rate:.2f}")
    print(f"  layout: sections={stats['sections']} items={stats['items']} "
          f"shop={stats['width_m']:.0f}x{stats['height_m']:.0f}m "
          f"colored_sections={n_sec} WC=yes")
    print("  PASS")


def smoke_uci(sample_rows=60000):
    print("\n[UCI Online Retail II]")
    uci_xlsx = dataset_paths.uci_workbook()
    sheets = DA.list_excel_sheets(uci_xlsx)
    sheet = sheets[-1][0]            # 'Year 2010-2011'
    df, notes = DA.read_excel_sheets(uci_xlsx, [sheet])
    if sample_rows and len(df) > sample_rows:
        df = df.sample(n=sample_rows, random_state=0).reset_index(drop=True)
        print(f"  (smoke sample of {sample_rows:,} rows from '{sheet}')")
    adapter = DA.best_transactional_adapter(df, uci_xlsx)
    _assert(adapter.name == "uci_online_retail_ii",
            f"chose {adapter.name}, expected uci_online_retail_ii")
    normalized, report = adapter.adapt(df)
    _assert(len(normalized) > 0 and not report.is_blocking(), "empty/blocking adapt")
    params = DC.calibrate_transactional(
        normalized, assumed_conversion_rate=0.30,
        currency=report.extra.get("currency", "GBP"))
    _assert(params.n_invoices > 0, "no invoices")
    shop = StubShop()
    stats = build_layout_from_calibration(shop, params)
    _assert(stats["sections"] > 0 and stats["items"] > 0, "empty layout")
    params.seed_into(shop.customer_simulation)
    # Naive baseline (GUI default).
    n_sec = _check_cosmetics(shop, expect_alphabetical=True)
    # B.1 invariant: top-level live counters stay clean; calibration block exists.
    A = shop.customer_simulation.analytics
    _assert("calibration" in A and A["calibration"].get("n_invoices"),
            "calibration namespace not populated by seed_into")
    _assert(A.get("total_customers", 0) == 0,
            "seed_into leaked into live total_customers (should stay 0 until Start)")
    _assert(shop.customer_simulation.run_time == 0.0,
            "seed_into leaked into sim.run_time (should stay 0 until Start)")
    print(f"  invoices={params.n_invoices:,} products={params.n_unique_products:,} "
          f"categories={params.n_unique_categories} currency={params.currency}")
    print(f"  layout: sections={stats['sections']} items={stats['items']} "
          f"colored_sections={n_sec} WC=yes")
    print("  PASS")


def smoke_live_sim(sample_rows=60000):
    """End-to-end LIVE simulation on a dataset-built shop: the real
    CustomerFlowSimulation spawn loop must actually produce customers.

    Guards the three regressions the GUI hit:
      * UCI 'no customers': the NHPP clock used to start at the earliest
        hour that ever saw one invoice (multiplier ~0.02) -> effectively
        zero spawn rate. Now it starts at the first core-open hour.
      * Omnichannel provenance crash: stamp() on a directory bundle.
      * Trajectory calibration was display-only: agents now draw their
        walking speed from the empirical distribution.
    """
    print("\n[Live sim on dataset shop]")
    import time as _time
    from dataset_provenance import stamp
    from experiments._common import build_headless_shop_from_calibration

    uci_xlsx = dataset_paths.uci_workbook()
    omni_dir = dataset_paths.omnichannel_dir()
    eth_obsmat = dataset_paths.opentraj_eth_obsmat()

    # -- Build the UCI shop with a REAL CustomerFlowSimulation --
    sheets = DA.list_excel_sheets(uci_xlsx)
    df, _ = DA.read_excel_sheets(uci_xlsx, [sheets[-1][0]])
    df = df.sample(n=sample_rows, random_state=0).reset_index(drop=True)
    normalized, report = DA.OnlineRetailIIAdapter().adapt(df)
    params = DC.calibrate_transactional(normalized, currency="GBP")
    shop = build_headless_shop_from_calibration(params,
                                                max_items_per_category=8,
                                                naive=True)
    sim = shop.customer_simulation

    # 1) NHPP start-hour fix: the sim clock must start in a CORE open hour.
    start_h = int(sim.sim_clock_start_hour)
    mult = float(sim.hourly_profile[start_h])
    _assert(mult >= 0.5,
            f"sim starts at hour {start_h} with multiplier {mult:.3f} "
            f"(<0.5) — dead-hour regression")
    print(f"  NHPP start hour={start_h} multiplier={mult:.2f} (>=0.5) OK")

    # 2) Live spawn loop: run the real worker thread for a few seconds.
    # Arrivals are Poisson, so a low rate over a 4 s window sometimes
    # produces one customer with perfectly healthy code. A seeded arrival
    # stream, and a rate that puts the expected count (this rate times the
    # hour-of-day multiplier above) well clear of the threshold, keep the
    # check on the threaded loop without a knife-edge on the draw.
    sim.arrival_rng = np.random.default_rng(0)
    sim.spawn_rate = 5.0
    sim.start_simulation()
    _time.sleep(4.0)
    n_cust = sim.analytics.get('total_customers', 0)
    moved = any((abs(c.position[0] - shop.door_position[0]) > 0.5
                 or abs(c.position[1] - shop.door_position[1]) > 0.5)
                for c in list(sim.customers))
    sim.hard_stop()
    _assert(n_cust >= 2, f"live sim spawned only {n_cust} customers in 4s")
    print(f"  live spawn OK: {n_cust} customers in 4s, movement={moved}")

    # 3) Directory provenance (the Omnichannel bundle crash).
    prov = stamp(source_path=omni_dir,
                 adapter_name=DA.OmnichannelRetailAdapter.name,
                 adapter_version=DA.OmnichannelRetailAdapter.version,
                 rows_in=134, rows_kept=134)
    _assert(len(prov.source_sha256) == 64 and prov.source_bytes > 0,
            "directory provenance hash failed")
    print(f"  dir provenance OK: sha={prov.source_sha256[:12]}… "
          f"bytes={prov.source_bytes:,}")

    # 4) Trajectory speed seeding shapes AGENTS (not just validation).
    tdf, _ = DA.read_any(eth_obsmat)
    out, _ = DA.OpenTrajAdapter().adapt(tdf)
    sp = DC.calibrate_trajectory(out, target_width_m=shop.width,
                                 target_height_m=shop.height)
    profile_before = sim.hourly_profile.copy()
    sp.seed_into(sim)
    _assert((sim.hourly_profile == profile_before).all(),
            "spatial seeding must not touch the NHPP hourly profile")
    # Agents draw Normal(mean, std) clipped to [max(0.3, p5), min(2.5, p95)]
    # (simulation._spawn_customer), so the target is that clipped mean, not
    # the raw empirical mean. With 400 draws the sampling error of the mean
    # is ~0.02 m/s, far inside the 4-standard-error tolerance (~0.1), which
    # still catches a fall-back to the uniform(0.8, 1.5) default (mean 1.15).
    n_probe = 400
    speeds = []
    for _ in range(n_probe):
        sim._spawn_customer()
        speeds.append(sim.customers[-1].speed)
        sim.customers.pop()
    mean_s = sum(speeds) / len(speeds)
    emp = float(sp.speeds_m_s.mean())
    cal = sim.analytics["calibration"]
    mu = cal["empirical_speed_mean"]
    sd = cal["empirical_speed_std"] or 0.2
    lo = max(0.3, cal["empirical_speed_p5"])
    hi = min(2.5, cal["empirical_speed_p95"])
    target = float(np.clip(np.random.default_rng(0).normal(mu, sd, 1_000_000),
                           lo, hi).mean())
    tol = 4.0 * sd / np.sqrt(n_probe)
    _assert(abs(mean_s - target) < tol,
            f"agent speeds mean {mean_s:.3f} != clipped empirical mean "
            f"{target:.3f} (tolerance {tol:.3f})")
    print(f"  agent speed calibration OK: mean {mean_s:.2f} m/s "
          f"(clipped empirical {target:.2f}, raw empirical {emp:.2f})")
    print("  PASS")


def main():
    checks = [("OpenTraj", smoke_opentraj),
              ("Omnichannel", smoke_omnichannel),
              ("UCI", smoke_uci),
              ("LiveSim", smoke_live_sim)]
    failed = []
    for name, fn in checks:
        try:
            fn()
        except Exception as e:
            failed.append(name)
            print(f"  FAIL: {e}")
            traceback.print_exc()
    print("\n" + "=" * 56)
    if failed:
        print(f"SMOKE FAILED: {', '.join(failed)}")
        return 1
    print("ALL DATASET SMOKES PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
