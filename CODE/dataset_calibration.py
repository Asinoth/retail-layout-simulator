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

from customer import is_regular_item
from dataset_adapters import anonymous_customer_mask
from retail_literature import (ASSUMED_CONVERSION_RATE,
                               DEFAULT_OP_HOURS_PER_DAY,
                               DEFAULT_WEEKEND_MULTIPLIER)

# The Monte Carlo engine projects over consecutive CALENDAR days and lifts
# two days in seven by the weekend multiplier, so the rate it consumes has
# to be an average-calendar-day rate. This is the mean of that weekly
# pattern; dividing by it turns observed volume per calendar day into the
# engine's base rate, so ``cph * op_hours * conv`` averaged over a week
# reproduces the data.
_MC_MEAN_DAY_MULTIPLIER = (5.0 + 2.0 * DEFAULT_WEEKEND_MULTIPLIER) / 7.0

# Calibration keys holding the live agents' shopping-list law: the stored
# invoices the agents draw from (read by ``Customer._draw_invoice``) and
# their sizes with the counts describing them. Written only by
# ``CalibratedParams.seed_into`` when it is given the shop.
_SHOPPING_LIST_KEYS = ('list_invoice_keys', 'list_invoice_ptr',
                       'list_invoice_items', 'list_length_sample',
                       'list_length_source', 'n_invoices_with_placed',
                       'anonymous_list_share', 'anonymous_list_item_share')


@dataclass
class CalibratedParams:
    # Aggregates
    n_invoices: int
    n_unique_products: int
    n_unique_customers: int
    n_unique_categories: int
    span_seconds: float

    # Rates
    arrivals_per_hour: float          # BUYERS (invoices) per open hour, average day
    assumed_conversion_rate: float    # see module docstring
    return_customer_rate: float       # share of customers with >1 invoice

    # Empirical distributions (raw samples, not summaries -- let downstream
    # callers compute KS goodness-of-fit, percentiles, plots).
    basket_sizes: np.ndarray              # units per invoice
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

    # Distinct products per invoice -- the trip-size sample the validation
    # goodness-of-fit tests compare with the simulator's items per visit
    # (``basket_sizes`` counts units, which a wholesale line inflates).
    basket_distinct_sizes: np.ndarray = field(
        default_factory=lambda: np.zeros(0))

    # Spend per invoice at one unit of each distinct product, each at the
    # product's calibrated price (``item_prices``, the price the dataset-
    # built shop charges), in the same invoice order as
    # ``basket_distinct_sizes``. A simulated visit buys one unit of each
    # item it picks, so this -- not ``invoice_revenues``, which is
    # quantity x price -- is the per-visit revenue the validation compares.
    invoice_distinct_revenues: np.ndarray = field(
        default_factory=lambda: np.zeros(0))

    # Each invoice's distinct product set, compressed-sparse-row style and
    # in the same invoice order as ``basket_distinct_sizes``: invoice i
    # holds the products
    # ``invoice_product_index[j] for j in invoice_items[invoice_ptr[i]:
    # invoice_ptr[i + 1]]``. A dataset-built shop stocks only the top
    # products of each category, so the per-invoice totals above describe
    # a catalogue its shoppers cannot buy from; this structure lets
    # ``placed_invoice_sample`` and ``placed_invoice_lists`` cut every
    # invoice down to the products a given shop carries. Empty for sources
    # without invoices (aggregate calibrations), which
    # ``has_invoice_structure`` reports.
    invoice_product_index: List[str] = field(default_factory=list)
    invoice_ptr: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int64))
    invoice_items: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int32))

    # Time of day (seconds after midnight) at which each gap in
    # ``inter_arrival_seconds`` starts, in the same order. The gaps alone
    # cannot be set against an hour-of-day rate; with their start times the
    # validation can rescale each one by the rate integrated over it
    # (time-rescaling), which is Exp(1) under a non-homogeneous Poisson
    # process with that rate. Empty for sources without invoice times.
    inter_arrival_start_s: np.ndarray = field(
        default_factory=lambda: np.zeros(0))
    # Anonymous invoices -- those whose customer id is missing (about 7% of
    # Online Retail II's invoices, at roughly twice the stocked spend of an
    # identified one). They are kept by default, and disclosed here:
    # ``invoice_anonymous`` flags each invoice in the order of
    # ``basket_distinct_sizes``; the counts and shares describe the whole
    # calibration, and ``anonymous_placed_shares`` gives the shares among
    # the invoices a given shop stocks. ``calibrate_transactional(...,
    # exclude_anonymous=True)`` calibrates without them, for a sensitivity
    # run. ``customer_id_available`` is False when the source has no
    # customer column, and then nothing is known about anonymity.
    invoice_anonymous: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=bool))
    n_anonymous_invoices: int = 0
    anonymous_invoice_frac: float = 0.0
    anonymous_revenue_frac: float = 0.0
    customer_id_available: bool = False

    @property
    def has_invoice_structure(self) -> bool:
        """True when the per-invoice product sets were kept (transactional
        sources), i.e. when ``placed_invoice_sample`` has invoices to cut."""
        ptr = getattr(self, 'invoice_ptr', None)
        index = getattr(self, 'invoice_product_index', None)
        return (ptr is not None and np.asarray(ptr).size >= 2
                and index is not None and len(index) > 0)

    @property
    def visitors_per_hour(self) -> float:
        """Visitors per open hour: buyers divided by the conversion rate.

        This is the rate the Monte Carlo engine and the live spawn loop
        need, because both draw visitors and apply conversion themselves.
        ``cph * op_hours * conversion``, averaged over the engine's weekly
        multiplier pattern, then reproduces the observed invoices per
        CALENDAR day."""
        return float(self.arrivals_per_hour) / max(
            float(self.assumed_conversion_rate), 1e-6)

    # ---------------------------------------------------------------------
    # Seeding the simulation
    # ---------------------------------------------------------------------
    def seed_into(self, sim, shop=None) -> None:
        """Populate ``sim.analytics['calibration']`` with the empirical values
        so the optimize / Markov / MC / GA / Sensitivity engines see real
        data immediately without requiring a prior live simulation run.

        ``shop``: the shop ``build_layout_from_calibration`` was run on. When
        given, the per-item dicts (``popular_items``,
        ``item_conversion_rates``, ``cross_merchandising``) are re-keyed from
        ``product_id`` to the shop's item keys, which is how the GA score and
        the optimize helpers look items up. When omitted they stay keyed by
        ``product_id``.

        With a shop, and when the params carry per-invoice product sets,
        the agents' shopping-list law is written too: the in-store portion
        of every invoice holding at least one of the shop's regular items
        (on any floor; impulse displays, checkout and WC excluded by
        ``customer.is_regular_item``), which is all a simulated shopper can
        buy. Each spawning agent takes one of these invoices as its list
        (``Customer._draw_invoice``). They are stored as compressed sparse
        rows over the shop's item keys -- ``list_invoice_keys`` (the keys
        the invoices reference), ``list_invoice_ptr`` (int64 offsets) and
        ``list_invoice_items`` (int32 positions in ``list_invoice_keys``)
        -- in the invoice order of the params' own CSR arrays, because a
        spawn draws one row and reads a few entries, and the full UCI sheet
        holds tens of thousands of rows. ``list_length_sample`` is the size
        of each stored invoice, so the list-length statistics a run reports
        describe exactly what the agents draw; ``list_length_source`` and
        ``n_invoices_with_placed`` (the number of stored invoices) sit
        beside it, and, when the source names customers,
        ``anonymous_list_share`` / ``anonymous_list_item_share`` -- the
        anonymous invoices' share of the stored lists and of the items on
        them. Without a shop there is no assortment to cut the
        invoices to, so none of these is written and the agents keep the
        type-conditional law.

        Reseeding replaces the keys the previous call wrote, so a second
        dataset does not inherit the first one's keys; keys written by
        ``SpatialParams.seed_into`` are kept.

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

        # Drop what the previous seeding wrote before writing this dataset's
        # values. Only some keys are rewritten on every call (the
        # ``calibration_extra`` tags and the hourly-profile keys are
        # source-dependent), and a stale ``mean_impulse_rate`` or
        # ``source_kind`` from an earlier dataset would change the MC impulse
        # rate and which Validation tests run. The record lives on ``sim``
        # so the calibration dict itself only carries data. The
        # shopping-list keys are dropped whatever the record says: only this
        # method writes them, and invoices left behind would keep driving
        # the agents after a reseed that writes none (another source, or no
        # shop). Popped here, they are registered below like every other key
        # written.
        for k in (*getattr(sim, '_calibration_seeded_keys', ()),
                  *_SHOPPING_LIST_KEYS):
            cal.pop(k, None)
        keys_before = set(cal)

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
        # Buyers and visitors are stored under separate names: consumers
        # that feed a customers-per-hour rate into the MC engine read
        # ``visitors_per_hour``.
        cal['arrivals_per_hour'] = float(self.arrivals_per_hour)
        cal['visitors_per_hour'] = float(self.visitors_per_hour)
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

        # Non-homogeneous Poisson hourly profile -- scale the empirical
        # hour-of-day share so the multipliers SUM to the operating hours
        # the rates are defined on. A live day driven at
        # ``visitors_per_hour`` then delivers the same expected volume as
        # the Monte Carlo's ``cph * op_hours``. Normalizing to mean 1 over
        # however many hours ever saw a transaction (15 on the full UCI
        # sheets, a few of them 6 a.m. outliers) would instead run the live
        # day 40-50% hot, and would move with the outlier count of whatever
        # sample was loaded. This shapes live spawns (sim.hourly_profile +
        # sim_clock_start_hour); we do NOT set sim.run_time or sim.sim_time,
        # so the spawn loop starts the clock at the first open hour and
        # advances naturally.
        try:
            hod = np.asarray(self.dwell_hour_distribution, dtype=np.float64).copy()
            open_mask = hod > 0
            if open_mask.any():
                open_total = hod[open_mask].sum()
                if open_total > 0:
                    profile = np.zeros(24, dtype=np.float64)
                    profile[open_mask] = (hod[open_mask] / open_total
                                          * DEFAULT_OP_HOURS_PER_DAY)
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
                    # The threshold is relative to the average open-hour
                    # multiplier, so it means the same thing whatever the
                    # profile is scaled to.
                    core = np.where(profile >= 0.5 * profile[open_mask].mean())[0]
                    start_hour = int(core[0]) if core.size else int(np.argmax(profile))
                    sim.sim_clock_start_hour = float(start_hour)
                    cal['hourly_profile_source'] = 'dataset'
                    cal['hourly_profile'] = profile.tolist()
                    cal['hourly_profile_basis_hours'] = float(
                        DEFAULT_OP_HOURS_PER_DAY)
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

        # Re-key the per-item dicts onto the shop's item keys (display names,
        # "(product_id)" suffix on collision). The first floor-1 item
        # carrying a product_id wins; pairs with an unplaced item are
        # dropped.
        if shop is not None:
            pid_to_key: Dict[str, str] = {}
            floors = getattr(shop, 'floors', None) or {}
            for fid in sorted(floors):
                for key, idata in ((floors[fid] or {}).get('items')
                                   or {}).items():
                    pid = (idata or {}).get('product_id')
                    if pid is not None and fid == 1:
                        pid_to_key.setdefault(str(pid), key)
            for name in ('popular_items', 'item_conversion_rates'):
                cal[name] = {pid_to_key[str(k)]: v
                             for k, v in cal[name].items()
                             if str(k) in pid_to_key}
            cross = {}
            for pair, count in cal['cross_merchandising'].items():
                pa, pb = pair.split('|', 1)
                if pa in pid_to_key and pb in pid_to_key:
                    cross[f"{pid_to_key[pa]}|{pid_to_key[pb]}"] = count
            cal['cross_merchandising'] = cross

            # The live agents' shopping lists: the stocked part of each
            # invoice. Whole invoices cannot be used -- a shop stocking ~100
            # of ~4,000 products would send agents out for items it does not
            # carry -- and invoices touching none of the stocked products
            # are trips this shop never sees.
            if self.has_invoice_structure:
                key_of = regular_item_keys(shop)
                lists = placed_invoice_lists(self, key_of)
                if lists['n_with_placed']:
                    cal['list_invoice_keys'] = lists['keys']
                    cal['list_invoice_ptr'] = lists['ptr']
                    cal['list_invoice_items'] = lists['items']
                    cal['list_length_sample'] = (
                        np.diff(lists['ptr']).astype(np.int64).tolist())
                    cal['list_length_source'] = 'placed-invoice empirical'
                    cal['n_invoices_with_placed'] = int(lists['n_with_placed'])
                    # How much of the list law anonymous invoices make up:
                    # their share of the stored lists and of the listed
                    # items (review R27). Absent without a customer column.
                    if getattr(self, 'customer_id_available', False):
                        shares = anonymous_placed_shares(self, key_of)
                        cal['anonymous_list_share'] = shares['invoice_share']
                        cal['anonymous_list_item_share'] = \
                            shares['purchase_share']

        sim._calibration_seeded_keys = sorted(set(cal) - keys_before)


# --- In-store portion of each invoice -------------------------------------

def placed_invoice_sample(params: CalibratedParams,
                          product_ids) -> Dict[str, Any]:
    """Each invoice cut down to the products a shop carries.

    A dataset-built shop stocks only the top products of each category
    (about 100 of roughly 4,000 on Online Retail II), and a simulated
    shopper can only buy what is stocked. The trip-level quantity the
    simulator can reproduce is therefore the in-store portion of an
    invoice, not the whole invoice:

      * ``'sizes'`` -- distinct placed products on each invoice that holds
        at least one (np.ndarray of int, invoice order);
      * ``'revenues'`` -- the same invoices priced at one unit of each of
        those products, at ``params.item_prices`` (the price the
        dataset-built shop charges), which is the unit a simulated visit
        spends in;
      * ``'n_invoices'`` -- every invoice in the structure;
      * ``'n_with_placed'`` -- how many of them hold a placed product.

    Invoices touching none of the placed products are left out: they are
    trips this shop never sees, and a simulated visit that reaches the
    checkout has a non-empty list. ``product_ids`` is any iterable of
    product id strings. The arrays are empty, and both counts zero, when
    the params carry no per-invoice product sets (aggregate sources).

    Vectorised over the CSR arrays: membership is tested once per distinct
    product, then summed per invoice with ``bincount``, so the cost stays
    linear in the invoice lines of the full UCI sheets.
    """
    empty = {'sizes': np.zeros(0, dtype=np.int64),
             'revenues': np.zeros(0, dtype=np.float64),
             'n_invoices': 0, 'n_with_placed': 0}
    per_invoice = _placed_per_invoice(params, product_ids)
    if per_invoice is None:
        return empty
    sizes, revenues = per_invoice
    empty['n_invoices'] = int(sizes.size)
    keep = sizes > 0
    if not keep.any():
        return empty
    return {'sizes': sizes[keep].astype(np.int64),
            'revenues': revenues[keep].astype(np.float64),
            'n_invoices': int(sizes.size),
            'n_with_placed': int(keep.sum())}


def _placed_per_invoice(params: CalibratedParams, product_ids):
    """``(sizes, revenues)`` over EVERY invoice of the CSR structure, in
    invoice order: the distinct stocked products on each invoice and their
    spend at one unit each, at ``params.item_prices`` (zero for an invoice
    holding none). None when the params carry no per-invoice product sets.
    ``placed_invoice_sample`` keeps the invoices with a stocked product;
    ``anonymous_placed_shares`` splits them by anonymity."""
    if not getattr(params, 'has_invoice_structure', False):
        return None
    ptr = np.asarray(params.invoice_ptr, dtype=np.int64)
    items = np.asarray(params.invoice_items, dtype=np.int64)
    index = np.asarray([str(p) for p in params.invoice_product_index],
                       dtype=str)
    n_inv = int(ptr.size) - 1
    zeros = (np.zeros(n_inv, dtype=np.int64), np.zeros(n_inv, dtype=np.float64))

    wanted = np.asarray(
        sorted({str(p) for p in (() if product_ids is None else product_ids)}),
        dtype=str)
    if wanted.size == 0:
        return zeros
    stocked = np.isin(index, wanted)
    if not stocked.any():
        return zeros

    # Invoice row of every stored (invoice, product) entry.
    row = np.repeat(np.arange(n_inv, dtype=np.int64), np.diff(ptr))
    hit = stocked[items]
    hit_rows = row[hit]
    sizes = np.bincount(hit_rows, minlength=n_inv)

    # ``invoice_product_index`` holds ids as strings; ``item_prices`` keeps
    # the source's own id type, so it is looked up by the string form too.
    price_of = {str(k): v for k, v in (params.item_prices or {}).items()}
    prices = (pd.Series(params.invoice_product_index, dtype=object)
              .astype(str).map(price_of)
              .fillna(0.0).to_numpy(dtype=np.float64))
    revenues = np.bincount(hit_rows, weights=prices[items[hit]],
                           minlength=n_inv)
    return sizes.astype(np.int64), revenues.astype(np.float64)


def anonymous_placed_shares(params: CalibratedParams,
                            product_ids) -> Dict[str, Any]:
    """How much of a shop's stocked demand comes from anonymous invoices.

    Over the invoices holding at least one of ``product_ids`` (the trips
    the shop sees, as ``placed_invoice_sample`` defines them): the number
    that are anonymous and their share (``invoice_share``), their share of
    the stocked purchases -- distinct stocked products, summed over
    invoices (``purchase_share``) -- and of the stocked spend at one unit
    each at the calibrated prices (``revenue_share``), with the mean
    stocked size and spend of an anonymous and of an identified invoice
    side by side. The whole calibration's shares sit beside them
    (``anonymous_invoice_frac`` of the invoices, ``anonymous_revenue_frac``
    of the line revenue). Every share is NaN when the source has no
    customer column (``customer_id_available``) or no invoice structure."""
    nan = float('nan')
    out = {'customer_id_available': bool(getattr(params,
                                                 'customer_id_available',
                                                 False)),
           'anonymous_invoice_frac_all': float(getattr(
               params, 'anonymous_invoice_frac', nan)),
           'anonymous_revenue_frac_all': float(getattr(
               params, 'anonymous_revenue_frac', nan)),
           'n_with_placed': 0, 'n_anonymous_with_placed': 0,
           'invoice_share': nan, 'purchase_share': nan, 'revenue_share': nan,
           'mean_size_anonymous': nan, 'mean_size_identified': nan,
           'mean_revenue_anonymous': nan, 'mean_revenue_identified': nan}
    per_invoice = _placed_per_invoice(params, product_ids)
    anon = np.asarray(getattr(params, 'invoice_anonymous', ()), dtype=bool)
    if per_invoice is None:
        return out
    sizes, revenues = per_invoice
    keep = sizes > 0
    out['n_with_placed'] = int(keep.sum())
    if not out['customer_id_available'] or anon.size != sizes.size:
        return out
    a, s, r = anon[keep], sizes[keep], revenues[keep]
    out['n_anonymous_with_placed'] = int(a.sum())
    if keep.any():
        out['invoice_share'] = float(a.mean())
        out['purchase_share'] = float(s[a].sum() / max(s.sum(), 1))
        out['revenue_share'] = (float(r[a].sum() / r.sum())
                                if r.sum() > 0 else nan)
    if a.any():
        out['mean_size_anonymous'] = float(s[a].mean())
        out['mean_revenue_anonymous'] = float(r[a].mean())
    if (~a).any():
        out['mean_size_identified'] = float(s[~a].mean())
        out['mean_revenue_identified'] = float(r[~a].mean())
    return out


def regular_item_keys(shop) -> Dict[str, str]:
    """product_id -> the key a spawning agent knows the shop's item by, for
    every product on a regular item (``customer.is_regular_item``: not an
    impulse display, not the checkout or the WC), on any floor.

    Agents receive the shop's items as ``shop.all_items_across_floors()``
    (the simulation's spawn does exactly this), so the keys are read from
    the same view whenever the shop provides it; a shop without it (a
    stand-in holding only ``floors``) is walked floor by floor, the lowest
    floor keeping a name two floors share. A product carried by several
    regular items maps to the first of them in that order, and a product
    carried only by impulse displays maps to none -- it can never be on a
    shopping list.
    """
    view = None
    flatten = getattr(shop, 'all_items_across_floors', None)
    if callable(flatten):
        view = flatten()
    if view is None:
        view = {}
        floors = getattr(shop, 'floors', None) or {}
        for fid in sorted(floors):
            for name, data in ((floors[fid] or {}).get('items')
                               or {}).items():
                view.setdefault(name, data)
    out: Dict[str, str] = {}
    for key, data in view.items():
        pid = (data or {}).get('product_id')
        if pid is not None and is_regular_item(key, data):
            out.setdefault(str(pid), str(key))
    return out


def placed_invoice_lists(params: CalibratedParams,
                         key_of: Dict[str, str]) -> Dict[str, Any]:
    """Each invoice cut to the products a shop carries, as shopping lists.

    ``key_of`` maps product id (string form) to the shop's item key, one
    key per product, as ``regular_item_keys`` gives it (an item carries a
    single product id, so no key is shared). Every invoice holding at
    least one mapped product is kept, in invoice order, as the keys of its
    mapped products in their stored order:

      * ``'keys'`` -- the item keys the kept invoices reference (list of
        str, in the order of ``params.invoice_product_index``);
      * ``'ptr'`` -- CSR offsets, np.int64, ``n_with_placed + 1`` long;
      * ``'items'`` -- positions in ``'keys'``, np.int32; kept invoice i is
        ``[keys[j] for j in items[ptr[i]:ptr[i + 1]]]``;
      * ``'n_invoices'`` / ``'n_with_placed'`` -- as in
        ``placed_invoice_sample``.

    The kept invoices and their sizes are those ``placed_invoice_sample``
    gives for the mapped product ids, so the sizes are that function's
    ``'sizes'``. Vectorised over the CSR arrays the same way.
    """
    out = {'keys': [], 'ptr': np.zeros(1, dtype=np.int64),
           'items': np.zeros(0, dtype=np.int32),
           'n_invoices': 0, 'n_with_placed': 0}
    if not getattr(params, 'has_invoice_structure', False):
        return out
    ptr = np.asarray(params.invoice_ptr, dtype=np.int64)
    entries = np.asarray(params.invoice_items, dtype=np.int64)
    n_inv = int(ptr.size) - 1
    out['n_invoices'] = n_inv

    # Column of each dataset product in the output key list, -1 when the
    # shop does not carry it on a regular item.
    key_of = {str(k): str(v) for k, v in (key_of or {}).items()}
    column = np.full(len(params.invoice_product_index), -1, dtype=np.int64)
    keys: List[str] = []
    for j, pid in enumerate(params.invoice_product_index):
        key = key_of.get(str(pid))
        if key is not None:
            column[j] = len(keys)
            keys.append(key)
    if not keys:
        return out

    # Entries on a carried product, kept in place: they already run
    # invoice by invoice in stored order, and an invoice with none of them
    # contributes nothing, so they are the kept invoices' rows end to end.
    col = column[entries]
    hit = col >= 0
    row = np.repeat(np.arange(n_inv, dtype=np.int64), np.diff(ptr))
    sizes = np.bincount(row[hit], minlength=n_inv)
    kept = sizes[sizes > 0]
    new_ptr = np.zeros(kept.size + 1, dtype=np.int64)
    np.cumsum(kept, out=new_ptr[1:])
    out.update(keys=keys, ptr=new_ptr, items=col[hit].astype(np.int32),
               n_with_placed=int(kept.size))
    return out


# --- Calibration ----------------------------------------------------------

def calibrate_transactional(df: pd.DataFrame,
                            assumed_conversion_rate: float = ASSUMED_CONVERSION_RATE,
                            top_pairs_n: int = 50,
                            currency: str = "GBP",
                            exclude_anonymous: bool = False) -> CalibratedParams:
    """Compute empirical simulation parameters from a normalized transactional df.

    ``df`` must have columns: invoice_id, product_id, product_name, quantity,
    timestamp (datetime64), unit_price, category. customer_id is optional.

    Anonymous invoices (no customer id) are kept and disclosed
    (``CalibratedParams.invoice_anonymous`` and the counts beside it).
    ``exclude_anonymous=True`` calibrates without them -- every rate,
    distribution and per-item count then describes the identified
    customers only -- for a sensitivity run; it needs a customer column,
    since without one no invoice can be told anonymous.
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
    # Timestamps carrying a UTC offset (ISO-8601 '...+01:00') parse as
    # tz-aware, or as an object column when offsets are mixed; the numpy
    # datetime arithmetic below needs naive datetime64. Drop the zone and
    # keep the local wall-clock time, which is what the hour-of-day profile
    # should reflect.
    if isinstance(df["timestamp"].dtype, pd.DatetimeTZDtype):
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    elif df["timestamp"].dtype == object:
        df["timestamp"] = pd.to_datetime(
            df["timestamp"].map(lambda t: t.replace(tzinfo=None)
                                if isinstance(t, pd.Timestamp) else pd.NaT),
            errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["line_revenue"] = df["quantity"] * df["unit_price"]

    # Anonymous lines: the customer column names nobody. Excluded on
    # request (sensitivity run), otherwise flagged per invoice below.
    customer_id_available = "customer_id" in df.columns
    if customer_id_available:
        df["_anonymous"] = anonymous_customer_mask(df["customer_id"]).to_numpy()
    elif exclude_anonymous:
        raise ValueError("calibrate_transactional: exclude_anonymous needs a "
                         "customer_id column to tell anonymous invoices "
                         "apart")
    else:
        df["_anonymous"] = False
    anonymous_excluded = {}
    if exclude_anonymous:
        anon_rows = df["_anonymous"].to_numpy(dtype=bool)
        anonymous_excluded = {
            "anonymous_invoices": "excluded",
            "n_anonymous_invoices_excluded": int(
                df.loc[anon_rows, "invoice_id"].nunique()),
            "n_anonymous_rows_excluded": int(anon_rows.sum()),
            "anonymous_revenue_excluded": float(
                df.loc[anon_rows, "line_revenue"].sum()),
        }
        df = df[~anon_rows]
        if len(df) == 0:
            raise ValueError("calibrate_transactional: no identified "
                             "customer's invoice left after excluding the "
                             "anonymous ones")

    # Per-invoice rollups
    inv_g = df.groupby("invoice_id")
    invoice_revs = inv_g["line_revenue"].sum().to_numpy(dtype=np.float64)
    basket_sizes = inv_g["quantity"].sum().to_numpy(dtype=np.float64)
    invoice_times = inv_g["timestamp"].min().sort_values().to_numpy()
    # Distinct products per invoice (audit R7.3): basket_sizes counts UNITS,
    # which a wholesale bulk line inflates; distinct-item count is the
    # trip-size companion.
    distinct_per_invoice = inv_g["product_id"].nunique().to_numpy(dtype=np.float64)
    # An invoice is anonymous when its lines name no customer (an invoice
    # carries one customer; ``any`` guards a malformed one).
    invoice_anonymous = inv_g["_anonymous"].any().to_numpy(dtype=bool)
    basket_units_median = float(np.median(basket_sizes)) if basket_sizes.size else 0.0
    basket_distinct_median = (float(np.median(distinct_per_invoice))
                              if distinct_per_invoice.size else 0.0)

    n_invoices = int(invoice_times.size)
    if n_invoices < 2:
        raise ValueError("calibrate_transactional: need at least 2 invoices")

    # Inter-arrival gaps (seconds) between successive invoices on the same
    # trading day. Timestamps in Online Retail II have one-minute
    # resolution, so invoices placed within the same minute give a gap of
    # exactly 0 s; those are real arrivals and stay in the sample (a KS test
    # against a continuous exponential will show the discretization). Only
    # the gap from one calendar date's last invoice to the next date's first
    # is dropped, since it spans the closed hours.
    invoice_ns = invoice_times.astype("datetime64[ns]")
    deltas_s = np.diff(invoice_ns.astype(np.int64)) / 1e9
    invoice_dates = invoice_ns.astype("datetime64[D]")
    same_day = invoice_dates[1:] == invoice_dates[:-1]
    inter_arrival = deltas_s[same_day]
    # Where in the day each kept gap starts, for the time-rescaled arrival
    # test (``dataset_validation``): seconds after the gap's first invoice's
    # midnight.
    time_of_day_s = (invoice_ns.astype(np.int64)
                     - invoice_dates.astype("datetime64[ns]").astype(np.int64)
                     ) / 1e9
    inter_arrival_start = time_of_day_s[:-1][same_day]

    # Buyers per open hour on the projection engine's own day model: the
    # engine runs every CALENDAR day in the horizon and lifts weekends, so
    # the basis is invoices per calendar day of the observed span, spread
    # over the operating hours it uses for its daily lambda and divided by
    # its average weekly multiplier. Spreading over trading days only (a
    # calendar date with at least one invoice) would over-produce the
    # dataset's volume on every closed day the engine still trades through:
    # Online Retail II trades about 5.7 days a week. Counting only minutes
    # that contain an invoice is not open time either -- each such minute
    # holds at least one invoice, which pins the estimate near 60/h
    # whatever the true rate.
    n_trading_days = max(int(np.unique(invoice_dates).size), 1)
    n_calendar_days = max(
        int((invoice_dates[-1] - invoice_dates[0]).astype(np.int64)) + 1, 1)
    arrivals_per_hour = ((n_invoices / n_calendar_days)
                         / DEFAULT_OP_HOURS_PER_DAY / _MC_MEAN_DAY_MULTIPLIER)

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

    # Grouped by invoice id like ``inv_g``, so the order matches the other
    # per-invoice samples.
    inv_prod = df[["invoice_id", "product_id"]].drop_duplicates()
    invoice_distinct_revs = (inv_prod["product_id"].map(item_prices)
                             .groupby(inv_prod["invoice_id"]).sum()
                             .to_numpy(dtype=np.float64))

    # Each invoice's distinct product set, as CSR arrays in ``inv_g``'s
    # invoice order (so row i lines up with ``distinct_per_invoice[i]``).
    # The invoice position is looked up in ``inv_g``'s own key index rather
    # than re-derived by sorting, so the two orders cannot disagree. Missing
    # product ids are left out, as ``nunique`` leaves them out of the count.
    inv_keys = inv_g.size().index
    pairs = inv_prod[inv_prod["product_id"].notna()]
    row_of = inv_keys.get_indexer(pairs["invoice_id"])
    pairs = pairs[row_of >= 0]
    row_of = row_of[row_of >= 0].astype(np.int64)
    prod_codes, prod_uniques = pd.factorize(pairs["product_id"], sort=False)
    order = np.lexsort((prod_codes, row_of))
    invoice_items = prod_codes[order].astype(np.int32)
    invoice_ptr = np.zeros(len(inv_keys) + 1, dtype=np.int64)
    np.cumsum(np.bincount(row_of, minlength=len(inv_keys)),
              out=invoice_ptr[1:])
    invoice_product_index = [str(p) for p in prod_uniques]

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
    #
    # An invoice over the cap keeps its most frequently purchased products
    # (invoice count across the dataset, product id as tie-break). Cutting
    # the sorted id list would keep the lexicographically smallest stock
    # codes, so which pairs survive would depend on code spelling. The kept
    # ids are re-sorted so each pair is counted under one (a, b) key.
    from itertools import combinations as _combinations
    pair_counter: Counter = Counter()
    inv_products = df.groupby("invoice_id")["product_id"].apply(
        lambda s: sorted(set(s))
    )
    PAIRS_BASKET_CAP = 50

    def _popularity_key(pid):
        return (-item_visit_counts.get(pid, 0), str(pid))

    for ids in inv_products:
        if len(ids) < 2:
            continue
        if len(ids) > PAIRS_BASKET_CAP:
            ids = sorted(sorted(ids, key=_popularity_key)[:PAIRS_BASKET_CAP])
        pair_counter.update(_combinations(ids, 2))
    top_pairs = [(a, b, c) for (a, b), c in pair_counter.most_common(top_pairs_n)]

    # Return-customer rate. Exclude anonymous invoices (missing Customer ID
    # is ~25% of Online Retail II); attributing them all to one "nan"
    # pseudo-customer would corrupt the repeat-purchase estimate (audit R7.2).
    return_rate = 0.0
    n_unique_customers = 0
    if customer_id_available:
        known = df[~df["_anonymous"].to_numpy(dtype=bool)]
        if len(known):
            cust_inv = known.groupby("customer_id")["invoice_id"].nunique()
            return_rate = float((cust_inv > 1).sum() / max(len(cust_inv), 1))
            n_unique_customers = int(len(cust_inv))

    span_seconds = float(
        (invoice_times[-1] - invoice_times[0]).astype("timedelta64[s]").astype(np.int64)
    )

    total_line_revenue = float(df["line_revenue"].sum())
    anonymous_revenue_frac = (
        float(df.loc[df["_anonymous"].to_numpy(dtype=bool),
                     "line_revenue"].sum() / total_line_revenue)
        if customer_id_available and total_line_revenue > 0 else 0.0)
    n_anonymous_invoices = int(invoice_anonymous.sum())

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
        basket_distinct_sizes=distinct_per_invoice,
        invoice_distinct_revenues=invoice_distinct_revs,
        invoice_product_index=invoice_product_index,
        invoice_ptr=invoice_ptr,
        invoice_items=invoice_items,
        inter_arrival_start_s=inter_arrival_start,
        invoice_anonymous=invoice_anonymous,
        n_anonymous_invoices=n_anonymous_invoices,
        anonymous_invoice_frac=(n_anonymous_invoices / max(n_invoices, 1)),
        anonymous_revenue_frac=anonymous_revenue_frac,
        customer_id_available=customer_id_available,
        # How the rate was put on the engine's day model, and how far the
        # two day counts differ (UCI: 305 trading days over 374 calendar
        # days), so a reader can see what the divisor was. Whether the
        # anonymous invoices are in the calibration is recorded beside it.
        calibration_extra={
            "arrival_rate_source": "invoices_per_calendar_day",
            "n_trading_days": n_trading_days,
            "n_calendar_days": n_calendar_days,
            **(anonymous_excluded
               or {"anonymous_invoices": ("included" if customer_id_available
                                          else "unknown")}),
        },
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
      * hourly/day-of-week profile + arrival rate <- arrivals table (in-store
                                  visitors; stored as buyers, see below)
      * conversion rate        <- 1 - prod(1 - purchase_pct) (independence est.;
                                  tagged 'assumption' when it saturates)
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
    conv_raw = float(buyer.mean()) if buyer.size else ASSUMED_CONVERSION_RATE
    conv = min(max(conv_raw, 0.05), 0.99)
    # Independent per-family draws make at least one purchase near-certain
    # once the marginal probabilities sum well above 1 (the shipped bundle
    # sums to ~12.7), so the estimate saturates at the clamp and carries no
    # information from the data. When the clamp binds the value is the
    # clamp bound itself, i.e. an assumption, and is flagged as one; the
    # data only bounds visit conversion from below, by the largest
    # single-family purchase probability (recorded in calibration_extra).
    conv_source = ("estimated_from_purchase_probabilities" if conv == conv_raw
                   else "assumption")

    # Arrivals -> hour-of-day + day-of-week + rate + span.
    hod = np.zeros(24, dtype=np.float64)
    dow = np.zeros(7, dtype=np.float64)
    arrivals_per_hour = 0.0
    weekly_customers = 0.0
    arrival_rate_source = "derived_from_demand"
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
            # The table's own open-hour count (06:00-22:00 in the shipped
            # bundle) is not the operating day the projection engine runs,
            # so the measured weekly total is put on the same basis as the
            # transactional path: visitors per average calendar day spread
            # over DEFAULT_OP_HOURS_PER_DAY. Using the table's per-hour mean
            # directly made the engine draw a 10-hour day's worth of a
            # 17-hour store.
            weekly_customers = float(a["instore_arrivals"].sum())
            if weekly_customers > 0:
                arrivals_per_hour = (weekly_customers / 7.0
                                     / DEFAULT_OP_HOURS_PER_DAY
                                     / _MC_MEAN_DAY_MULTIPLIER)
                arrival_rate_source = "arrivals_table"

    hod_share = hod / hod.sum() if hod.sum() > 0 else hod
    dow_share = dow / dow.sum() if dow.sum() > 0 else dow
    if arrivals_per_hour <= 0:
        # No arrivals table (missing file, or unrecognised columns). The
        # only volume signal left is per-family daily demand, which counts
        # FAMILY purchases: a trip buys about a dozen families, so feeding
        # that sum in as visitors overstates traffic by an order of
        # magnitude. Convert with the parametric trip size and the
        # conversion estimate (buyers -> visitors) instead.
        demand_units = float(df["daily_demand_instore"].sum())
        trip_families = float(basket_sizes.mean()) if basket_sizes.size else 1.0
        daily_visitors = (demand_units / max(trip_families, 1.0)
                          / max(conv, 1e-6))
        weekly_customers = 7.0 * daily_visitors
        arrivals_per_hour = max(1e-3, daily_visitors / DEFAULT_OP_HOURS_PER_DAY
                                / _MC_MEAN_DAY_MULTIPLIER)
        arrival_rate_source = "derived_from_demand"
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
        "conversion_rate_lower_bound": float(p.max()),
        "conversion_rate_independence_estimate": conv_raw,
        "arrival_rate_source": arrival_rate_source,
    }

    return CalibratedParams(
        n_invoices=n_invoices,
        n_unique_products=int(len(df)),
        n_unique_customers=0,
        n_unique_categories=int(df["category"].nunique()),
        span_seconds=span_seconds,
        # The arrivals table counts in-store VISITORS, while
        # ``arrivals_per_hour`` means buyers throughout; storing visitors x
        # conversion keeps ``visitors_per_hour`` equal to the measured rate.
        arrivals_per_hour=float(arrivals_per_hour) * conv,
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
        conversion_rate_source=conv_source,
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

        # Heat map: add the traffic counts to FLOOR 1's accumulator, without
        # zeroing it (caller may have sim-derived heat already). The traffic
        # is ground-floor data, and ``sim.heat_raw`` is only the display copy
        # of whichever floor is on screen -- adding there while floor 2 is
        # shown would hand floor 2's history to floor 1 and lose floor 1's.
        # ``calibrate_trajectory`` is normally called with the sim's shop
        # size and heat_map_resolution, so the grids match cell for cell and
        # the counts are added directly. Otherwise the grid is resampled
        # with bilinear zoom, which interpolates values rather than
        # conserving them, so the result is rescaled to the source total.
        if self.zone_traffic_raw.size and sim.heat_raw.size > 0:
            get_floor_raw = getattr(sim, '_get_floor_heat_raw', None)
            target = get_floor_raw(1) if callable(get_floor_raw) else sim.heat_raw
            src = self.zone_traffic_raw.astype(np.float32)
            if src.shape == target.shape:
                target += src
            else:
                try:
                    from scipy.ndimage import zoom as _zoom
                except Exception:
                    _zoom = None
                if _zoom is not None:
                    zx = target.shape[0] / max(src.shape[0], 1)
                    zy = target.shape[1] / max(src.shape[1], 1)
                    scaled = _zoom(src, (zx, zy), order=1)
                    slc = (slice(0, min(scaled.shape[0], target.shape[0])),
                           slice(0, min(scaled.shape[1], target.shape[1])))
                    part = scaled[slc]
                    mass = float(part.sum())
                    if mass > 0:
                        part = part * (float(src.sum()) / mass)
                    target[slc] = target[slc] + part
            # Mirror into the display buffer only while floor 1 is the one
            # being shown; the buffers stay separate objects so a later
            # floor switch cannot overwrite the accumulator. The smoothed
            # map is rebuilt from the display buffer, so it is refreshed
            # whenever that buffer changed -- otherwise the heat view keeps
            # showing the pre-load traffic until the next simulation tick.
            floor1_shown = int(getattr(getattr(sim, 'shop', None),
                                       'current_floor', 1) or 1) == 1
            if target is not sim.heat_raw and floor1_shown:
                w = min(target.shape[0], sim.heat_raw.shape[0])
                h = min(target.shape[1], sim.heat_raw.shape[1])
                sim.heat_raw[:w, :h] = target[:w, :h]
            if target is sim.heat_raw or floor1_shown:
                refresh = getattr(sim, '_refresh_smoothed_heatmap', None)
                if callable(refresh):
                    refresh()

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
    # consecutive positions within each track. Positions come from the
    # source coordinates in metres: x_sim/y_sim are stretched
    # (anisotropically) to the shop size and are only for the heat map, so
    # speeds or path lengths taken from them would scale with the shop.
    if "velocity_m_s" in df.columns:
        speeds = df["velocity_m_s"].to_numpy(dtype=np.float64)
        speeds = speeds[np.isfinite(speeds) & (speeds >= 0)]
    else:
        # Compute speed from successive samples per track. This is the
        # robust fallback when the source doesn't carry velocity.
        speeds_list: List[float] = []
        for tid, g in df.groupby("track_id"):
            xs = g["x_m"].to_numpy()
            ys = g["y_m"].to_numpy()
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
        xs = g["x_m"].to_numpy()
        ys = g["y_m"].to_numpy()
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
