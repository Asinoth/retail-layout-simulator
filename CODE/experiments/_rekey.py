"""A calibration re-keyed onto the fixtures of an existing store.

Figure C's store is built from one calibration: which products stand on the
floor, where, and in what zones (``run_real_data_example.build_store``).
The layout score then reads that calibration's per-item statistics --
co-purchase pairs, popularity, category revenue -- and the Monte Carlo
objective its arrival rate and spend. Two runners need the SAME fixtures
scored under ANOTHER calibration:

  * ``run_input_uncertainty`` re-scores fixed layouts under bootstrap
    re-calibrations of the same data (input uncertainty, review R11);
  * ``run_heldout_transfer`` scores layouts found on one period under the
    next period's calibration (held-out transfer, review R12).

Re-keying keeps the geometry and the assortment and replaces everything
the calibration supplies: ``CalibratedParams.seed_into(sim, shop=shop)``
re-writes the per-item statistics under the shop's item keys (by product
id), and the base parameters are rebuilt from the new calibration and
anchored at the store's own as-built layout re-scored under the new
statistics, so the as-built store reproduces the new calibration exactly.
A fixture whose product the new calibration never saw keeps its place on
the floor and carries no demand: it drops out of the co-purchase pairs,
the popularity waypoints and the revenue and accessibility weights
(``absent_fixtures`` lists them).

``stocked_scale`` puts base parameters on the stocked-invoice scale: each
invoice cut to the products the store carries (``placed_invoice_sample``),
instead of whole invoices across the wholesaler's ~4,000 products.
"""

from __future__ import annotations

import dataclasses
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from dataset_calibration import (CalibratedParams,  # noqa: E402
                                 placed_invoice_sample, regular_item_keys)
from experiments._common import (anchor_base_params,  # noqa: E402
                                 base_params_for_calibration,
                                 layout_to_chromosome)
from experiments.closed_form import expected_revenue  # noqa: E402
from experiments.run_real_data_example import (CalibratedStore,  # noqa: E402
                                               build_store)


def fixture_products(store: CalibratedStore) -> List[str]:
    """Product ids on the store's regular items (not impulse displays, the
    checkout or the WC), in the shop's item order."""
    return list(regular_item_keys(store.shop))


def reseed_store(store: CalibratedStore,
                 params: CalibratedParams) -> CalibratedStore:
    """``store`` with ``params`` re-keyed onto its fixtures, IN PLACE: the
    shop's calibration analytics are replaced (``seed_into`` drops what the
    previous seeding wrote) and the returned store carries base parameters
    from ``params``, anchored at the store's as-built layout re-scored under
    them. The geometry, the item keys and both layouts are unchanged."""
    params.seed_into(store.shop.customer_simulation, shop=store.shop)
    bp = anchor_base_params(store.shop, store.item_names,
                            base_params_for_calibration(params),
                            store.init_layout)
    return dataclasses.replace(store, base_params=bp)


def store_on_fixtures(fixture_params: CalibratedParams,
                      params: CalibratedParams,
                      max_items_per_category: int) -> CalibratedStore:
    """A fresh store with ``fixture_params``' fixtures (``build_store``,
    deterministic) and ``params`` re-keyed onto them."""
    store = build_store(fixture_params, max_items_per_category,
                        verbose=False)
    return reseed_store(store, params)


def absent_fixtures(store: CalibratedStore,
                    params: CalibratedParams) -> List[Dict[str, Any]]:
    """The store's regular fixtures whose product ``params`` holds no
    invoice for: product id, the item key, category and zone. Such a
    fixture stays on the floor with no observed demand."""
    seen = {str(k) for k, v in (params.item_visit_counts or {}).items()
            if int(v) > 0}
    items = store.shop.floors[1]['items']
    out = []
    for pid, key in regular_item_keys(store.shop).items():
        if pid not in seen:
            d = items.get(key, {})
            out.append({'product_id': pid, 'item': key,
                        'category': d.get('category'),
                        'zone': d.get('zone')})
    return out


def stocked_scale(base_params: Dict[str, Any], params: CalibratedParams,
                  product_ids) -> Dict[str, Any]:
    """``base_params`` (anchored, whole-invoice scale, from ``params``) put
    on the stocked-invoice scale of ``product_ids``.

    The buyers are the invoices holding a stocked product, so the visitor
    rate is scaled by their share of all invoices; the spend per converter
    is the stocked part of an invoice at one unit of each stocked product,
    at the calibrated prices (``placed_invoice_sample``'s revenues), and its
    SD theirs. The net spend and the impulse value are rescaled by the
    ratio of the two gross spends, so every spend quantity keeps its share
    of gross: the percentage lift of any layout over another is the same on
    both scales, and only the currency amounts change. The anchor is kept
    (it is a layout score, the same on both scales)."""
    placed = placed_invoice_sample(params, product_ids)
    n_inv, n_with = int(placed['n_invoices']), int(placed['n_with_placed'])
    if n_with < 2 or n_inv < 1:
        raise ValueError(f"stocked_scale: {n_with} of {n_inv} invoices hold "
                         f"a stocked product; need at least 2")
    revs = np.asarray(placed['revenues'], dtype=np.float64)
    gross = float(revs.mean())
    gross_whole = float(base_params['rev_per_customer_gross'])
    if not gross_whole > 0.0:
        raise ValueError("stocked_scale: the whole-invoice gross spend must "
                         "be positive")
    ratio = gross / gross_whole
    out = dict(base_params)
    out.update(
        customers_per_hour=float(base_params['customers_per_hour'])
        * n_with / n_inv,
        rev_per_converting_customer=float(
            base_params['rev_per_converting_customer']) * ratio,
        rev_per_customer_gross=gross,
        rev_std=float(revs.std(ddof=1)),
        avg_impulse_value=float(base_params['avg_impulse_value']) * ratio,
        revenue_scale='stocked_invoice',
        stocked_invoice_share=n_with / n_inv,
    )
    return out


def layout_value(store: CalibratedStore, layout, horizon_days: int,
                 base_params: Optional[Dict[str, Any]] = None) -> float:
    """Exact expected revenue of ``layout`` on ``store`` over the horizon
    (``experiments.closed_form.expected_revenue``), under the store's base
    parameters or the ``base_params`` given (e.g. ``stocked_scale``'s)."""
    return expected_revenue(
        store.base_params if base_params is None else base_params,
        int(horizon_days), shop=store.shop,
        chromosome=layout_to_chromosome(layout, store.item_names),
        item_names=store.item_names)
