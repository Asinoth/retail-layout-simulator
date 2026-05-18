"""Empirical calibration of simulation parameters from a normalized
transactional DataFrame (as produced by ``dataset_adapters``).

The output of ``calibrate_transactional`` is two things:

  * a ``CalibratedParams`` record (numbers, distributions, top pairs) —
    used by the Validation tab and the optimization report.

  * a side-effect: ``CalibratedParams.seed_into(sim)`` writes the empirical
    distributions directly into ``simulation.analytics`` so the Monte Carlo
    / Markov / GA pipelines see real values from t=0 instead of priors.

Conversion rate handling
------------------------
A purchase dataset only sees customers who bought *something*. The
conversion rate (visitors → buyers) is fundamentally unobservable from
transactions alone. We expose ``CalibratedParams.assumed_conversion_rate``
as an explicit knob (default 0.30, configurable per UI), and store
``A['conversion_rate_source'] = 'assumption'`` in provenance so a TOMACS
reviewer can immediately see which numbers are calibrated vs. asserted.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd


@dataclass
class CalibratedParams:
    # Aggregates
    n_invoices: int
    n_unique_products: int
    n_unique_customers: int
    n_unique_categories: int
    span_seconds: float

    # Rates
    arrivals_per_hour: float          # invoices / observation hours
    assumed_conversion_rate: float    # see module docstring
    return_customer_rate: float       # share of customers with >1 invoice

    # Empirical distributions (raw samples, not summaries — let downstream
    # callers compute KS goodness-of-fit, percentiles, plots).
    basket_sizes: np.ndarray              # items per invoice
    invoice_revenues: np.ndarray          # GBP per invoice
    inter_arrival_seconds: np.ndarray     # gaps between successive invoices
    dwell_hour_distribution: np.ndarray   # shape (24,), share per hour
    dow_distribution: np.ndarray          # shape (7,), share per weekday

    # Per-item / per-category
    item_prices: Dict[str, float]               # product_id -> median price
    item_names: Dict[str, str]                  # product_id -> display name
    item_categories: Dict[str, str]             # product_id -> category
    item_visit_counts: Dict[str, int]           # product_id -> # invoices it appears in
    category_revenue_share: Dict[str, float]    # category -> share of total rev (sums to 1)
    top_pairs: List[Tuple[str, str, int]]       # (product_id_a, product_id_b, co_count)

    # Currency tag (for labels only)
    currency: str = "GBP"

    # ---------------------------------------------------------------------
    # Seeding the simulation
    # ---------------------------------------------------------------------
    def seed_into(self, sim) -> None:
        """Populate ``sim.analytics`` with the empirical values so the
        downstream optimize / Markov / MC engines reflect real data
        immediately instead of waiting for in-sim customers to accumulate.
        """
        A = sim.analytics

        # Counts the optimize pipeline reads via ``A['total_customers']`` etc.
        # Use n_invoices as the synthetic 'customer' count — one basket = one
        # observed customer. Conversion is the assumed rate.
        n = int(self.n_invoices)
        completed = int(round(n * 1.0))   # every invoice is by definition a purchase
        A['total_customers'] = max(A.get('total_customers', 0), n)
        A['completed_purchases'] = max(A.get('completed_purchases', 0), completed)
        # Abandoned carts are implied by the (1 - conversion) assumption.
        implied_visitors = int(round(n / max(self.assumed_conversion_rate, 1e-6)))
        A['abandoned_carts'] = max(0, implied_visitors - n)

        # Empirical distributions (the optimize MC reads these as
        # 'basket_sizes_observed' via sim_calibration.extract_simulation_parameters).
        A['basket_sizes'] = list(map(int, self.basket_sizes.astype(int)))
        # Stored for variance estimation in extract_simulation_parameters
        A['customer_revenues'] = list(map(float, self.invoice_revenues))
        A['impulse_revenues'] = []   # transactional data doesn't distinguish impulse

        # Per-item conversion proxy: every invoice that contains an item
        # counts as a 'visit' AND a 'purchase' (it's already in the basket).
        A['item_conversion_rates'] = {
            pid: {'visits': cnt, 'purchases': cnt}
            for pid, cnt in self.item_visit_counts.items()
        }
        A['popular_items'] = dict(self.item_visit_counts)

        # Category-level revenue
        # The optimization report reads revenue_by_area for the report
        # and the GA reads it for scoring. Use absolute GBP, not shares.
        total_rev = float(self.invoice_revenues.sum())
        A['revenue_by_area'] = {cat: share * total_rev
                                for cat, share in self.category_revenue_share.items()}
        A['total_revenue'] = max(A.get('total_revenue', 0.0), total_rev)

        # Cross-merchandising — top co-purchase pairs as a 'A|B' counter
        A['cross_merchandising'] = {f"{a}|{b}": c for a, b, c in self.top_pairs}

        # Run-time scaffolding so customers_per_hour comes out right when
        # extract_simulation_parameters calls sim_hours(sim).
        # span_seconds is the observation window of the source dataset.
        sim.run_time = max(getattr(sim, 'run_time', 0.0), float(self.span_seconds))
        # Tell sim_hours we have real seconds of "simulation" time.
        sim.sim_time = max(getattr(sim, 'sim_time', 0.0), float(self.span_seconds))

        # Provenance: mark which numbers were assumed vs measured.
        A.setdefault('calibration', {})
        A['calibration'].update({
            'conversion_rate_source': 'assumption',
            'assumed_conversion_rate': self.assumed_conversion_rate,
            'span_seconds': self.span_seconds,
            'arrivals_per_hour': self.arrivals_per_hour,
            'return_customer_rate': self.return_customer_rate,
            'currency': self.currency,
            'n_invoices': self.n_invoices,
            'n_unique_products': self.n_unique_products,
            'n_unique_categories': self.n_unique_categories,
        })

        # Non-homogeneous Poisson hourly profile -- normalize the
        # empirical hour-of-day share so its mean over operating hours
        # (non-zero entries) equals 1.0. This way the live simulator's
        # *average* spawn rate matches ``spawn_rate`` exactly while peak
        # hours produce proportionally more arrivals.
        try:
            hod = np.asarray(self.dwell_hour_distribution, dtype=np.float64).copy()
            open_mask = hod > 0
            if open_mask.any():
                mean_open = hod[open_mask].mean()
                if mean_open > 0:
                    profile = np.zeros(24, dtype=np.float64)
                    profile[open_mask] = hod[open_mask] / mean_open
                    sim.hourly_profile = profile
                    # Set start-of-day to the first open hour
                    first_open = int(np.argmax(open_mask))
                    sim.sim_clock_start_hour = float(first_open)
                    A['calibration']['hourly_profile_source'] = 'dataset'
                    A['calibration']['hourly_profile'] = profile.tolist()
                    A['calibration']['operating_hours_start'] = first_open
                    A['calibration']['operating_hours_count'] = int(open_mask.sum())
        except Exception:
            pass


# ─── Calibration ──────────────────────────────────────────────────────────

def calibrate_transactional(df: pd.DataFrame,
                            assumed_conversion_rate: float = 0.30,
                            top_pairs_n: int = 50,
                            currency: str = "GBP") -> CalibratedParams:
    """Compute empirical simulation parameters from a normalized transactional df.

    ``df`` must have columns: invoice_id, product_id, product_name, quantity,
    timestamp (datetime64), unit_price, category. customer_id is optional.
    """
    required = {"invoice_id", "product_id", "product_name", "quantity",
                "timestamp", "unit_price", "category"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"calibrate_transactional missing columns: {missing}")
    if len(df) == 0:
        raise ValueError("calibrate_transactional: empty dataframe")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["line_revenue"] = df["quantity"] * df["unit_price"]

    # Per-invoice rollups
    inv_g = df.groupby("invoice_id")
    invoice_revs = inv_g["line_revenue"].sum().to_numpy(dtype=np.float64)
    basket_sizes = inv_g["quantity"].sum().to_numpy(dtype=np.float64)
    invoice_times = inv_g["timestamp"].min().sort_values().to_numpy()

    n_invoices = int(invoice_times.size)
    if n_invoices < 2:
        raise ValueError("calibrate_transactional: need at least 2 invoices")

    # Inter-arrival gaps (seconds). Filter overnight gaps > 12h so the
    # arrival-rate estimate reflects business hours, not calendar time.
    deltas_ns = np.diff(invoice_times.astype("datetime64[ns]").astype(np.int64))
    deltas_s = deltas_ns / 1e9
    inter_arrival = deltas_s[(deltas_s > 0) & (deltas_s < 12 * 3600)]

    # Open hours: 60-second buckets that had at least one invoice.
    ts_minute = pd.to_datetime(invoice_times).floor("min").unique()
    open_hours = max(len(ts_minute) / 60.0, 1e-6)
    arrivals_per_hour = n_invoices / open_hours

    # Hour-of-day + day-of-week distributions
    ts = pd.to_datetime(invoice_times)
    hod = np.bincount(ts.hour, minlength=24).astype(np.float64)
    hod = hod / hod.sum() if hod.sum() > 0 else hod
    dow = np.bincount(ts.dayofweek, minlength=7).astype(np.float64)
    dow = dow / dow.sum() if dow.sum() > 0 else dow

    # Per-item
    prod_g = df.groupby("product_id")
    item_prices = prod_g["unit_price"].median().to_dict()
    # First non-null name per product
    item_names = prod_g["product_name"].agg(
        lambda s: next((x for x in s if isinstance(x, str) and x.strip()), "")
    ).to_dict()
    item_categories = prod_g["category"].agg(
        lambda s: next((x for x in s if isinstance(x, str) and x.strip()),
                       "General")
    ).to_dict()
    item_visit_counts = prod_g["invoice_id"].nunique().astype(int).to_dict()

    # Category-level revenue share
    cat_rev = df.groupby("category")["line_revenue"].sum()
    total_rev = float(cat_rev.sum())
    category_revenue_share = (cat_rev / total_rev).to_dict() if total_rev > 0 else {}

    # Top co-purchase pairs
    pair_counter: Counter = Counter()
    # Group product_ids per invoice, generate unordered pairs.
    inv_products = df.groupby("invoice_id")["product_id"].apply(
        lambda s: sorted(set(s))
    )
    for ids in inv_products:
        if len(ids) < 2:
            continue
        # Cap per-invoice combinatorial explosion for very large baskets.
        if len(ids) > 30:
            ids = ids[:30]
        for i in range(len(ids)):
            a = ids[i]
            for j in range(i + 1, len(ids)):
                b = ids[j]
                pair_counter[(a, b)] += 1
    top_pairs = [(a, b, c) for (a, b), c in pair_counter.most_common(top_pairs_n)]

    # Return-customer rate
    if "customer_id" in df.columns:
        cust_inv = df.groupby("customer_id")["invoice_id"].nunique()
        return_rate = float((cust_inv > 1).sum() / max(len(cust_inv), 1))
        n_unique_customers = int(len(cust_inv))
    else:
        return_rate = 0.0
        n_unique_customers = 0

    span_seconds = float(
        (invoice_times[-1] - invoice_times[0]).astype("timedelta64[s]").astype(np.int64)
    )

    return CalibratedParams(
        n_invoices=n_invoices,
        n_unique_products=int(df["product_id"].nunique()),
        n_unique_customers=n_unique_customers,
        n_unique_categories=int(df["category"].nunique()),
        span_seconds=span_seconds,
        arrivals_per_hour=float(arrivals_per_hour),
        assumed_conversion_rate=float(assumed_conversion_rate),
        return_customer_rate=return_rate,
        basket_sizes=basket_sizes,
        invoice_revenues=invoice_revs,
        inter_arrival_seconds=inter_arrival,
        dwell_hour_distribution=hod,
        dow_distribution=dow,
        item_prices={str(k): float(v) for k, v in item_prices.items()},
        item_names={str(k): str(v) for k, v in item_names.items()},
        item_categories={str(k): str(v) for k, v in item_categories.items()},
        item_visit_counts={str(k): int(v) for k, v in item_visit_counts.items()},
        category_revenue_share={str(k): float(v) for k, v in category_revenue_share.items()},
        top_pairs=[(str(a), str(b), int(c)) for a, b, c in top_pairs],
        currency=currency,
    )


# ─── Spatial calibration (trajectory data) ────────────────────────────────

@dataclass
class SpatialParams:
    """Empirical spatial-behavior parameters extracted from pedestrian
    trajectories (e.g., ATC Shopping Mall). Seed into the simulation so
    customer speeds, zone-dwell distributions, and zone-transition
    probabilities match real observed pedestrian behavior.
    """
    n_tracks: int
    span_seconds: float
    bbox_m: Tuple[float, float, float, float]   # (x_min, y_min, x_max, y_max)
    speeds_m_s: np.ndarray                       # one sample per (track, second)
    track_lengths_s: np.ndarray                  # seconds per track
    track_lengths_m: np.ndarray                  # total path length per track
    n_unique_zones: int                          # observed zones (heat-map cells)
    zone_traffic_raw: np.ndarray                 # 2D (heat map) of visit counts
    heat_resolution: int                         # cells per metre

    def seed_into(self, sim) -> None:
        """Populate ``sim.analytics`` and the heatmap buffers with the
        spatial empirical distributions."""
        A = sim.analytics

        # Empirical speed distribution
        A.setdefault('calibration', {})
        if self.speeds_m_s.size > 0:
            A['calibration']['empirical_speed_mean'] = float(self.speeds_m_s.mean())
            A['calibration']['empirical_speed_std']  = float(self.speeds_m_s.std())
            A['calibration']['empirical_speed_p5']   = float(np.percentile(self.speeds_m_s, 5))
            A['calibration']['empirical_speed_p95']  = float(np.percentile(self.speeds_m_s, 95))

        # Track-length distribution
        if self.track_lengths_s.size > 0:
            A['calibration']['empirical_track_dwell_mean_s'] = float(
                self.track_lengths_s.mean()
            )
            A['calibration']['empirical_track_length_mean_m'] = float(
                self.track_lengths_m.mean()
            )

        # Heat map: scale traffic map to sim's heat_raw buffer.
        # ``zone_traffic_raw`` is in trajectory's native cell space; the
        # simulator owns its own heat_map_resolution and may have a
        # different shop bbox, so we rasterize the traffic map into the
        # sim's existing buffer via simple bilinear-ish nearest-neighbour
        # scaling.
        try:
            from scipy.ndimage import zoom as _zoom
        except Exception:
            _zoom = None
        if _zoom is not None and self.zone_traffic_raw.size:
            target = sim.heat_raw
            src = self.zone_traffic_raw.astype(np.float32)
            if src.shape != target.shape and target.size > 0:
                zx = target.shape[0] / max(src.shape[0], 1)
                zy = target.shape[1] / max(src.shape[1], 1)
                scaled = _zoom(src, (zx, zy), order=1)
                # Add to existing heat_raw without zeroing it (caller may
                # have transactional data + sim-derived heat already).
                slc = (slice(0, min(scaled.shape[0], target.shape[0])),
                       slice(0, min(scaled.shape[1], target.shape[1])))
                target[slc] = target[slc] + scaled[slc]
                if hasattr(sim, '_floor_heat_raw'):
                    sim._floor_heat_raw[1] = target

        A['calibration']['spatial_source'] = 'trajectory_dataset'
        A['calibration']['n_tracks'] = self.n_tracks
        A['calibration']['observation_span_seconds'] = self.span_seconds


def calibrate_trajectory(df: pd.DataFrame,
                         target_width_m: float,
                         target_height_m: float,
                         heat_resolution: int = 20) -> SpatialParams:
    """Compute empirical spatial parameters from a normalized trajectory df.

    The trajectory's bbox (x_min, y_min, x_max, y_max) is linearly
    rescaled to fit the simulator's shop dimensions so the heat map
    overlays correctly. This is acceptable for VALIDATION (showing that
    the simulator reproduces real-world traffic patterns) even though the
    real space and the simulated space have different layouts; what we're
    calibrating is the *distribution* of dwell + speed + traffic density,
    not the exact spatial locations.
    """
    required = {"track_id", "time_s", "x_m", "y_m"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"calibrate_trajectory missing columns: {missing}")
    if len(df) == 0:
        raise ValueError("calibrate_trajectory: empty dataframe")

    df = df.copy().sort_values(["track_id", "time_s"]).reset_index(drop=True)

    # Rescale to target bbox
    x_min, x_max = float(df["x_m"].min()), float(df["x_m"].max())
    y_min, y_max = float(df["y_m"].min()), float(df["y_m"].max())
    src_w = max(x_max - x_min, 1e-6)
    src_h = max(y_max - y_min, 1e-6)
    df["x_sim"] = (df["x_m"] - x_min) / src_w * target_width_m
    df["y_sim"] = (df["y_m"] - y_min) / src_h * target_height_m

    # Per-track speed (m/s) -- use velocity column if provided, else from
    # consecutive positions within each track.
    if "velocity_m_s" in df.columns:
        speeds = df["velocity_m_s"].to_numpy(dtype=np.float64)
        speeds = speeds[np.isfinite(speeds) & (speeds >= 0)]
    else:
        # Compute speed from successive samples per track. This is the
        # robust fallback when the source doesn't carry velocity.
        speeds_list: List[float] = []
        for tid, g in df.groupby("track_id"):
            xs = g["x_sim"].to_numpy()
            ys = g["y_sim"].to_numpy()
            ts = g["time_s"].to_numpy()
            if len(ts) < 2:
                continue
            dx = np.diff(xs); dy = np.diff(ys)
            dt = np.diff(ts)
            dt = np.where(dt > 0, dt, np.nan)
            sp = np.sqrt(dx*dx + dy*dy) / dt
            sp = sp[np.isfinite(sp) & (sp < 5.0)]   # cap walking-runs
            speeds_list.extend(sp.tolist())
        speeds = np.asarray(speeds_list, dtype=np.float64)

    # Per-track length + duration
    track_dur: List[float] = []
    track_len: List[float] = []
    for tid, g in df.groupby("track_id"):
        ts = g["time_s"].to_numpy()
        xs = g["x_sim"].to_numpy()
        ys = g["y_sim"].to_numpy()
        if len(ts) < 2:
            continue
        track_dur.append(float(ts.max() - ts.min()))
        dx = np.diff(xs); dy = np.diff(ys)
        track_len.append(float(np.sqrt(dx*dx + dy*dy).sum()))

    track_lengths_s = np.asarray(track_dur, dtype=np.float64)
    track_lengths_m = np.asarray(track_len, dtype=np.float64)

    # Heat map: rasterize positions into a grid.
    wc = int(target_width_m * heat_resolution) + 1
    hc = int(target_height_m * heat_resolution) + 1
    heat = np.zeros((wc, hc), dtype=np.float64)
    xi = np.clip((df["x_sim"].to_numpy() * heat_resolution).astype(int), 0, wc - 1)
    yi = np.clip((df["y_sim"].to_numpy() * heat_resolution).astype(int), 0, hc - 1)
    np.add.at(heat, (xi, yi), 1.0)
    n_unique_zones = int((heat > 0).sum())

    span_seconds = float(df["time_s"].max() - df["time_s"].min())

    return SpatialParams(
        n_tracks=int(df["track_id"].nunique()),
        span_seconds=span_seconds,
        bbox_m=(x_min, y_min, x_max, y_max),
        speeds_m_s=speeds,
        track_lengths_s=track_lengths_s,
        track_lengths_m=track_lengths_m,
        n_unique_zones=n_unique_zones,
        zone_traffic_raw=heat,
        heat_resolution=heat_resolution,
    )
