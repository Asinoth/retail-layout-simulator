"""Empirical calibration of simulation parameters from a normalized
transactional DataFrame (as produced by ``dataset_adapters``).

The output of ``calibrate_transactional`` is two things:

  * a ``CalibratedParams`` record (numbers, distributions, top pairs) --
    used by the Validation tab and the optimization report.

  * a side-effect: ``CalibratedParams.seed_into(sim)`` writes the empirical
    distributions directly into ``simulation.analytics`` so the Monte Carlo
    / Markov / GA pipelines see real values from t=0 instead of priors.

Conversion rate handling
------------------------
A purchase dataset only sees customers who bought *something*. The
conversion rate (visitors -> buyers) is fundamentally unobservable from
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

    # Empirical distributions (raw samples, not summaries -- let downstream
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

    # Provenance flags. Defaults preserve the transactional/UCI behavior;
    # aggregate sources (Omnichannel) override these so the Validation tab
    # and reports show which numbers are measured vs. parametric/assumed.
    conversion_rate_source: str = "assumption"
    calibration_extra: Dict[str, Any] = field(default_factory=dict)

    # Data-quality descriptors (audit R7.3/R7.5). ``basket_sizes`` is
    # units-per-invoice; the distinct-item count is reported alongside so a
    # wholesale bulk line (200 identical units) is not mistaken for a 200-
    # item shopper trip. ``category_fallback_frac`` is the share of revenue
    # in the inferred 'General Merchandise' bucket.
    basket_units_median: float = 0.0
    basket_distinct_median: float = 0.0
    category_fallback_frac: float = 0.0

    # ---------------------------------------------------------------------
    # Seeding the simulation
    # ---------------------------------------------------------------------
    def seed_into(self, sim) -> None:
        """Populate ``sim.analytics['calibration']`` with the empirical values
        so the optimize / Markov / MC / GA / Sensitivity engines see real
        data immediately without requiring a prior live simulation run.

        IMPORTANT -- calibration namespace separation:
          * Dataset-derived inputs (rates, distributions, per-item dicts) all
            live under ``A['calibration']`` only.
          * The LIVE display counters at the top of ``A`` (``total_customers``,
            ``completed_purchases``, ``abandoned_carts``, ``total_revenue``,
            ``basket_sizes``, ``customer_revenues``, ``revenue_by_area``,
            ``popular_items``, ``cross_merchandising``, ``item_conversion_rates``)
            are NOT seeded -- they start empty and are filled by the live
            simulation, so the Customer Flow panel reads clean zeros and live
            increments are not double-counted on top of dataset totals.
          * ``sim.run_time`` / ``sim.sim_time`` are NOT touched -- the NHPP
            spawn clock advances naturally from ``sim_clock_start_hour``.
          * ``sim.hourly_profile`` + ``sim.sim_clock_start_hour`` ARE set
            because they shape live spawns; with ``run_time`` starting at 0
            the clock progresses correctly through the day.

        Consumers (``sim_calibration.extract_simulation_parameters``,
        ``viz_optimize_helpers``, ``viz_ga``) read from
        ``analytics['calibration']`` via the ``_calib`` accessor in
        ``sim_calibration`` and fall back to the top-level live keys when no
        dataset is loaded.
        """
        A = sim.analytics
        A.setdefault('calibration', {})
        cal = A['calibration']

        # Counts and rates (would-be live values, stored in calibration only)
        n = int(self.n_invoices)
        implied_visitors = int(round(n / max(self.assumed_conversion_rate, 1e-6)))
        cal['n_invoices'] = n
        cal['total_customers'] = implied_visitors    # visitors, not buyers
        cal['completed_purchases'] = n
        cal['abandoned_carts'] = max(0, implied_visitors - n)
        cal['conversion_rate'] = float(self.assumed_conversion_rate)
        cal['assumed_conversion_rate'] = float(self.assumed_conversion_rate)
        cal['conversion_rate_source'] = getattr(self, 'conversion_rate_source',
                                                'assumption')
        cal['return_customer_rate'] = float(self.return_customer_rate)
        cal['arrivals_per_hour'] = float(self.arrivals_per_hour)
        cal['span_seconds'] = float(self.span_seconds)
        cal['currency'] = self.currency
        cal['n_unique_products'] = int(self.n_unique_products)
        cal['n_unique_categories'] = int(self.n_unique_categories)

        # Empirical distribution samples (MC + validation inputs).
        cal['basket_sizes'] = list(map(int, self.basket_sizes.astype(int)))
        cal['customer_revenues'] = list(map(float, self.invoice_revenues))
        cal['impulse_revenues'] = []
        try:
            cal['inter_arrival_seconds'] = list(map(float, self.inter_arrival_seconds))
        except Exception:
            cal['inter_arrival_seconds'] = []

        # Per-item conversion proxy: every invoice with an item is a visit AND
        # a purchase for that item (already in the observed basket).
        cal['item_conversion_rates'] = {
            pid: {'visits': int(cnt), 'purchases': int(cnt)}
            for pid, cnt in self.item_visit_counts.items()
        }
        cal['popular_items'] = dict(self.item_visit_counts)

        # Category revenue (absolute, in dataset currency -- for GA scoring).
        total_rev = float(self.invoice_revenues.sum()) if len(self.invoice_revenues) else 0.0
        cal['total_revenue'] = total_rev
        cal['revenue_by_area'] = {
            cat: float(share * total_rev)
            for cat, share in self.category_revenue_share.items()
        }

        # Cross-merchandising -- top co-purchase pairs as 'A|B' string keys.
        cal['cross_merchandising'] = {f"{a}|{b}": int(c) for a, b, c in self.top_pairs}

        # Non-homogeneous Poisson hourly profile -- normalize the empirical
        # hour-of-day share so its mean over OPEN hours = 1.0. This shapes
        # live spawns (sim.hourly_profile + sim_clock_start_hour); we do NOT
        # set sim.run_time or sim.sim_time, so the spawn loop starts the
        # clock at the first open hour and advances naturally.
        try:
            hod = np.asarray(self.dwell_hour_distribution, dtype=np.float64).copy()
            open_mask = hod > 0
            if open_mask.any():
                mean_open = hod[open_mask].mean()
                if mean_open > 0:
                    profile = np.zeros(24, dtype=np.float64)
                    profile[open_mask] = hod[open_mask] / mean_open
                    sim.hourly_profile = profile
                    first_open = int(np.argmax(open_mask))
                    # Start the sim clock at the first CORE-open hour (at
                    # least half the average open-hour rate), not at the
                    # earliest hour that ever saw a single transaction.
                    # A year of data usually has a handful of 6 a.m.
                    # outliers whose multiplier is ~0.02 -- starting there
                    # made the live sim look dead (effective spawn rate
                    # ~0) for the whole session, since at speed 1 one sim
                    # hour takes a real hour.
                    core = np.where(profile >= 0.5)[0]
                    start_hour = int(core[0]) if core.size else int(np.argmax(profile))
                    sim.sim_clock_start_hour = float(start_hour)
                    cal['hourly_profile_source'] = 'dataset'
                    cal['hourly_profile'] = profile.tolist()
                    cal['operating_hours_start'] = first_open
                    cal['sim_start_hour'] = start_hour
                    cal['operating_hours_count'] = int(open_mask.sum())
        except Exception:
            pass

        # Extra provenance (aggregate-source / parametric tags, measured
        # dwell + impulse means from omnichannel, etc.). Merged last.
        extra = getattr(self, 'calibration_extra', None)
        if extra:
            cal.update(extra)


# --- Calibration ----------------------------------------------------------

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
    # Distinct products per invoice (audit R7.3): basket_sizes counts UNITS,
    # which a wholesale bulk line inflates; distinct-item count is the
    # trip-size companion.
    distinct_per_invoice = inv_g["product_id"].nunique().to_numpy(dtype=np.float64)
    basket_units_median = float(np.median(basket_sizes)) if basket_sizes.size else 0.0
    basket_distinct_median = (float(np.median(distinct_per_invoice))
                              if distinct_per_invoice.size else 0.0)

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

    # Per-item. Use vectorized ``groupby.first()`` for the "first non-empty
    # name / category" reductions instead of an agg(lambda next(...)) chain:
    # blank strings become NaN, groupby.first() ignores NaN, which is the
    # same semantics but 3-10x faster on 500k+ rows.
    prod_g = df.groupby("product_id")
    item_prices = prod_g["unit_price"].median().to_dict()

    name_str = df["product_name"].astype(str)
    name_clean = df["product_name"].where(name_str.str.strip() != "", other=pd.NA)
    cat_str = df["category"].astype(str)
    cat_clean = df["category"].where(cat_str.str.strip() != "", other=pd.NA)

    item_names = (df.assign(_pn=name_clean)
                  .groupby("product_id")["_pn"]
                  .first()
                  .fillna("").astype(str).to_dict())
    item_categories = (df.assign(_cc=cat_clean)
                       .groupby("product_id")["_cc"]
                       .first()
                       .fillna("General").astype(str).to_dict())
    item_visit_counts = prod_g["invoice_id"].nunique().astype(int).to_dict()

    # Category-level revenue share
    cat_rev = df.groupby("category")["line_revenue"].sum()
    total_rev = float(cat_rev.sum())
    category_revenue_share = (cat_rev / total_rev).to_dict() if total_rev > 0 else {}
    # Share of revenue in the keyword-inference fallback bucket (audit R7.5):
    # a large value means the section structure is largely un-categorized.
    category_fallback_frac = float(category_revenue_share.get(
        "General Merchandise", 0.0))

    # Top co-purchase pairs. Use ``itertools.combinations`` + ``Counter.update``
    # (compiled-C inner loop) instead of nested Python for-loops; raise the
    # per-invoice cap from 30 to 50 to recover pairs that wholesale baskets
    # previously truncated. The combinatorial growth at k=50 (1225 pairs) is
    # still bounded.
    from itertools import combinations as _combinations
    pair_counter: Counter = Counter()
    inv_products = df.groupby("invoice_id")["product_id"].apply(
        lambda s: sorted(set(s))
    )
    PAIRS_BASKET_CAP = 50
    for ids in inv_products:
        if len(ids) < 2:
            continue
        if len(ids) > PAIRS_BASKET_CAP:
            ids = ids[:PAIRS_BASKET_CAP]
        pair_counter.update(_combinations(ids, 2))
    top_pairs = [(a, b, c) for (a, b), c in pair_counter.most_common(top_pairs_n)]

    # Return-customer rate. Exclude anonymous invoices (missing Customer ID
    # is ~25% of Online Retail II); attributing them all to one "nan"
    # pseudo-customer would corrupt the repeat-purchase estimate (audit R7.2).
    return_rate = 0.0
    n_unique_customers = 0
    if "customer_id" in df.columns:
        cid = df["customer_id"].astype(str).str.strip().str.lower()
        known = df[~cid.isin({"", "nan", "none", "na", "<na>"})]
        if len(known):
            cust_inv = known.groupby("customer_id")["invoice_id"].nunique()
            return_rate = float((cust_inv > 1).sum() / max(len(cust_inv), 1))
            n_unique_customers = int(len(cust_inv))

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
        basket_units_median=basket_units_median,
        basket_distinct_median=basket_distinct_median,
        category_fallback_frac=category_fallback_frac,
    )


# --- Aggregate retail calibration (Omnichannel) ---------------------------

def calibrate_omnichannel(families: pd.DataFrame,
                          arrivals: Optional[pd.DataFrame] = None,
                          max_sections: int = 14,
                          n_synth: int = 3000,
                          currency: str = "USD",
                          rng_seed: int = 0) -> CalibratedParams:
    """Direct aggregate calibration for the Omnichannel Retail dataset.

    The dataset has NO transactions; it reports, per product family: in-store
    purchase probability, average daily demand, dwell time, impulse rate,
    average price, and an in-store Zone ID, plus an hour x weekday arrival
    table. We map those *measured* aggregates straight into CalibratedParams:

      * item_prices            <- Avg price (USD)
      * item_categories        <- 'Zone <id>' (real in-store zone)
      * item_visit_counts      <- Average daily demand (popularity weight)
      * category_revenue_share <- demand*price aggregated per zone
      * hourly/day-of-week profile + arrival rate <- arrivals table
      * conversion rate        <- 1 - prod(1 - purchase_pct) (independence est.)
      * dwell / impulse        <- demand-weighted means (-> calibration meta)

    The basket-size and per-visit-revenue arrays are NOT observed; they are a
    parametric realization of the measured per-family purchase probabilities
    (one Bernoulli draw per family per synthetic visitor), tagged as such via
    ``calibration_extra`` so the Validation tab does not present them as
    data-validated. ``families`` / ``arrivals`` come from
    ``dataset_adapters.load_omnichannel_bundle``.
    """
    if families is None or len(families) == 0:
        raise ValueError("calibrate_omnichannel: empty families frame")

    df = families.copy()
    df = df[df["product_family"].astype(str).str.strip() != ""]
    if len(df) == 0:
        raise ValueError("calibrate_omnichannel: no product families after cleaning")

    rng = np.random.default_rng(rng_seed)

    # Prices: fill gaps with the median so every fixture has a price.
    price = pd.to_numeric(df["avg_price"], errors="coerce")
    med_price = float(np.nanmedian(price)) if np.isfinite(price).any() else 5.0
    df["avg_price"] = price.fillna(med_price).clip(lower=0.01)

    df["purchase_pct_instore"] = (pd.to_numeric(df["purchase_pct_instore"],
                                                errors="coerce")
                                  .fillna(0.0).clip(0.0, 1.0))
    df["daily_demand_instore"] = (pd.to_numeric(df["daily_demand_instore"],
                                                errors="coerce")
                                  .fillna(0.0).clip(lower=0.0))
    df["dwell_s"] = pd.to_numeric(df["dwell_s"], errors="coerce")
    df["impulse_rate"] = pd.to_numeric(df["impulse_rate"],
                                       errors="coerce").clip(0.0, 1.0)

    # Category = in-store zone; merge low-demand zones so the shop has a
    # manageable number of sections (display aggregation only).
    if "zone_id" in df.columns and df["zone_id"].notna().any():
        df["category"] = df["zone_id"].apply(
            lambda z: f"Zone {int(z)}" if pd.notna(z) else "Zone NA")
    else:
        df["category"] = df["product_family"].astype(str)
    if df["category"].nunique() > max_sections:
        zone_demand = (df.groupby("category")["daily_demand_instore"].sum()
                       .sort_values(ascending=False))
        keep = set(zone_demand.head(max_sections).index)
        df["category"] = df["category"].where(df["category"].isin(keep),
                                              "Other Zones")

    pids = df["aisle_id"].astype(int).astype(str).tolist()
    item_names = dict(zip(pids, df["product_family"].astype(str)))
    item_prices = dict(zip(pids, df["avg_price"].astype(float)))
    item_categories = dict(zip(pids, df["category"].astype(str)))
    vc = np.maximum(1, np.round(df["daily_demand_instore"].to_numpy())).astype(int)
    item_visit_counts = dict(zip(pids, [int(v) for v in vc]))

    df["rev"] = df["daily_demand_instore"] * df["avg_price"]
    cat_rev = df.groupby("category")["rev"].sum()
    tot_rev = float(cat_rev.sum())
    if tot_rev > 0:
        category_revenue_share = {str(k): float(v / tot_rev)
                                  for k, v in cat_rev.items()}
    else:
        n_cat = max(1, len(cat_rev))
        category_revenue_share = {str(k): 1.0 / n_cat for k in cat_rev.index}

    # Parametric basket / revenue realization from per-family purchase prob.
    p = df["purchase_pct_instore"].to_numpy(dtype=np.float64)
    prices = df["avg_price"].to_numpy(dtype=np.float64)
    n_fam = len(df)
    buys = rng.random((n_synth, n_fam)) < p[None, :]
    basket_counts = buys.sum(axis=1).astype(np.float64)
    basket_rev = (buys * prices[None, :]).sum(axis=1)
    buyer = basket_counts >= 1
    basket_sizes = basket_counts[buyer]
    invoice_revenues = basket_rev[buyer]
    if basket_sizes.size < 5:
        basket_sizes = np.array([max(1.0, float(p.sum()))] * 10)
        invoice_revenues = np.array([float((p * prices).sum())] * 10)
    conv = float(buyer.mean()) if buyer.size else 0.30
    conv = min(max(conv, 0.05), 0.99)

    # Arrivals -> hour-of-day + day-of-week + rate + span.
    hod = np.zeros(24, dtype=np.float64)
    dow = np.zeros(7, dtype=np.float64)
    arrivals_per_hour = 0.0
    weekly_customers = float(df["daily_demand_instore"].sum() * 7.0)
    if arrivals is not None and len(arrivals):
        a = arrivals.dropna(subset=["from_hour", "instore_arrivals"]).copy()
        if len(a):
            by_hour = a.groupby(a["from_hour"].astype(int))["instore_arrivals"].mean()
            for h, v in by_hour.items():
                if 0 <= int(h) < 24:
                    hod[int(h)] = float(v)
            wd_order = ["monday", "tuesday", "wednesday", "thursday",
                        "friday", "saturday", "sunday"]
            wd_tot = (a.groupby(a["weekday"].astype(str).str.lower())
                      ["instore_arrivals"].sum())
            for i, wd in enumerate(wd_order):
                if wd in wd_tot.index:
                    dow[i] = float(wd_tot[wd])
            nz = hod[hod > 0]
            arrivals_per_hour = float(nz.mean()) if nz.size else 0.0
            weekly_customers = float(a["instore_arrivals"].sum())

    hod_share = hod / hod.sum() if hod.sum() > 0 else hod
    dow_share = dow / dow.sum() if dow.sum() > 0 else dow
    if arrivals_per_hour <= 0:
        open_h = int((hod > 0).sum()) or 10
        arrivals_per_hour = max(1e-3, weekly_customers / (7.0 * open_h))
    if weekly_customers <= 0:
        weekly_customers = float(n_synth)

    n_invoices = int(max(2, round(weekly_customers)))
    span_seconds = 7.0 * 24.0 * 3600.0

    lam_per_s = max(arrivals_per_hour, 1e-6) / 3600.0
    inter_arrival = rng.exponential(
        1.0 / lam_per_s, size=int(min(5000, max(10, n_invoices)))
    ).astype(np.float64)

    # Demand-weighted dwell + impulse (metadata).
    w = df["daily_demand_instore"].to_numpy(dtype=np.float64)
    wsum = float(w.sum()) if w.sum() > 0 else float(len(df))
    if df["dwell_s"].notna().any():
        dv = np.nan_to_num(df["dwell_s"].to_numpy(dtype=np.float64))
        dwell_mean = float((dv * w).sum() / wsum) if wsum > 0 else float(np.nanmean(dv))
    else:
        dwell_mean = float("nan")
    if df["impulse_rate"].notna().any():
        iv = np.nan_to_num(df["impulse_rate"].to_numpy(dtype=np.float64))
        impulse_mean = float((iv * w).sum() / wsum) if wsum > 0 else float(np.nanmean(iv))
    else:
        impulse_mean = float("nan")

    calibration_extra = {
        "source_kind": "aggregate_retail_omnichannel",
        "basket_size_source": "parametric_from_purchase_probabilities",
        "revenue_source": "parametric_from_purchase_probabilities",
        "empirical_dwell_mean_s": dwell_mean,
        "mean_impulse_rate": impulse_mean,
        "n_product_families": int(len(df)),
        "n_zones": int(df["category"].nunique()),
        "weekly_instore_customers": weekly_customers,
    }

    return CalibratedParams(
        n_invoices=n_invoices,
        n_unique_products=int(len(df)),
        n_unique_customers=0,
        n_unique_categories=int(df["category"].nunique()),
        span_seconds=span_seconds,
        arrivals_per_hour=float(arrivals_per_hour),
        assumed_conversion_rate=conv,
        return_customer_rate=0.0,
        basket_sizes=basket_sizes.astype(np.float64),
        invoice_revenues=invoice_revenues.astype(np.float64),
        inter_arrival_seconds=inter_arrival,
        dwell_hour_distribution=hod_share,
        dow_distribution=dow_share,
        item_prices={str(k): float(v) for k, v in item_prices.items()},
        item_names={str(k): str(v) for k, v in item_names.items()},
        item_categories={str(k): str(v) for k, v in item_categories.items()},
        item_visit_counts={str(k): int(v) for k, v in item_visit_counts.items()},
        category_revenue_share=category_revenue_share,
        top_pairs=[],
        currency=currency,
        conversion_rate_source="estimated_from_purchase_probabilities",
        calibration_extra=calibration_extra,
    )


# --- Spatial calibration (trajectory data) --------------------------------

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
