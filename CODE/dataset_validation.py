"""Goodness-of-fit tests between simulated and empirical distributions.

For a TOMACS submission the calibration story isn't complete without
showing how close the simulator's output matches the dataset it was
calibrated against. We compare three primary distributions:

  * basket size (distinct items per visit)
  * per-visit revenue (one unit of each distinct item)
  * inter-arrival time

Test choice:
  - **Kolmogorov-Smirnov 2-sample** for continuous-ish distributions
    (basket size, revenue, inter-arrival). KS makes no distributional
    assumption -- appropriate when the simulator's distribution is itself
    empirical and may not match any known family.
  - **Chi-square** for the categorical distribution of per-category
    purchases.

Matching units is what makes these tests meaningful, and it is the part
that is easy to get wrong: the dataset counts units per invoice and prices
them at wholesale volumes, while a simulated visit picks up one unit of
each item on its list; the dataset's catalogue holds every product ever
sold, while the shop carries only the products that were laid out. Each
test below therefore states, in its note, the quantity both sides count.

We do NOT use t-tests on means: TOMACS reviewers will rightly note that
a t-test only compares first moments. KS compares the entire CDF.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
from scipy import stats as sp_stats

from dataset_calibration import CalibratedParams


@dataclass
class GoodnessOfFit:
    name: str
    test: str
    statistic: float
    p_value: float
    n_observed: int
    n_simulated: int
    note: str = ""

    def pass_at(self, alpha: float = 0.05) -> bool:
        """KS/chi-square: p > alpha means we cannot reject equality of
        distributions, i.e. the simulator matches the data."""
        return self.p_value > alpha

    def applicable(self) -> bool:
        """False when the test could not be run (too few samples, an empty
        side, ...). Those rows carry a NaN p-value, which would otherwise
        compare as a rejection."""
        return bool(np.isfinite(self.p_value))

    def verdict(self, alpha: float = 0.05) -> str:
        if not self.applicable():
            return "N/A"
        return "PASS" if self.pass_at(alpha) else "FAIL"


@dataclass
class ValidationResult:
    alpha: float
    tests: List[GoodnessOfFit] = field(default_factory=list)
    summary_lines: List[str] = field(default_factory=list)

    def all_pass(self) -> Optional[bool]:
        """True when every test that ran passed; None when none could run."""
        ran = [t for t in self.tests if t.applicable()]
        if not ran:
            return None
        return all(t.pass_at(self.alpha) for t in ran)

    def to_text(self) -> str:
        lines = [
            f"VALIDATION REPORT  (alpha = {self.alpha})",
            "=" * 60,
            "",
        ]
        for t in self.tests:
            verdict = t.verdict(self.alpha)
            # KS reports the largest CDF gap D; chi-square its chi2 sum.
            stat = "D" if t.test.startswith("KS") else "chi2"
            lines.append(f"  [{verdict}] {t.name}")
            lines.append(f"          {t.test}  {stat}={t.statistic:.4f}  p={t.p_value:.4g}")
            lines.append(f"          n_obs={t.n_observed}  n_sim={t.n_simulated}")
            if t.note:
                lines.append(f"          {t.note}")
            lines.append("")
        if self.summary_lines:
            lines.append("NOTES")
            lines.append("-" * 60)
            for s in self.summary_lines:
                lines.append(f"  • {s}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alpha": self.alpha,
            "all_pass": self.all_pass(),
            "tests": [dict(asdict(t), verdict=t.verdict(self.alpha))
                      for t in self.tests],
            "notes": list(self.summary_lines),
        }


# --- Test runner ----------------------------------------------------------

def _ks_safe(observed: np.ndarray, simulated: np.ndarray,
             name: str, note: str = "") -> Optional[GoodnessOfFit]:
    """KS 2-sample with safety rails -- returns None instead of crashing if
    either sample is too small or all-constant."""
    if observed is None or simulated is None:
        return None
    obs = np.asarray(observed, dtype=np.float64)
    sim = np.asarray(simulated, dtype=np.float64)
    obs = obs[np.isfinite(obs)]
    sim = sim[np.isfinite(sim)]
    if obs.size < 5 or sim.size < 5:
        return GoodnessOfFit(name=name, test="KS-2sample",
                             statistic=float('nan'), p_value=float('nan'),
                             n_observed=int(obs.size), n_simulated=int(sim.size),
                             note="Insufficient samples (need >= 5 each). " + note)
    if obs.std() == 0 and sim.std() == 0 and obs[0] == sim[0]:
        # Both samples are the same constant -- distributions identical.
        return GoodnessOfFit(name=name, test="KS-2sample",
                             statistic=0.0, p_value=1.0,
                             n_observed=int(obs.size), n_simulated=int(sim.size),
                             note="Both samples constant and equal. " + note)
    try:
        D, p = sp_stats.ks_2samp(obs, sim)
    except Exception as e:
        return GoodnessOfFit(name=name, test="KS-2sample",
                             statistic=float('nan'), p_value=float('nan'),
                             n_observed=int(obs.size), n_simulated=int(sim.size),
                             note=f"KS failed: {e}. " + note)
    return GoodnessOfFit(name=name, test="KS-2sample",
                         statistic=float(D), p_value=float(p),
                         n_observed=int(obs.size), n_simulated=int(sim.size),
                         note=note)


def _chi2_categorical(observed: Dict[str, float],
                      simulated: Dict[str, int],
                      name: str, note: str = "") -> Optional[GoodnessOfFit]:
    """Chi-square goodness of fit of the raw simulated counts to the
    observed category distribution.

    The dataset side is large (tens of thousands of invoices) while a live
    run records a few hundred visits, so the observed shares are the
    reference and the simulated counts are the sample. Rescaling the
    simulated shares up to the observed total instead treats a small
    sample's proportions as known and rejects almost every time under a
    true null.
    """
    obs_total = float(sum(observed.values())) if observed else 0.0
    n_obs = int(round(obs_total))
    n_sim = int(sum(simulated.values())) if simulated else 0
    if obs_total <= 0 or n_sim <= 0:
        return GoodnessOfFit(name=name, test="Chi-square",
                             statistic=float('nan'), p_value=float('nan'),
                             n_observed=n_obs, n_simulated=n_sim,
                             note="One side empty. " + note)
    keys = sorted(set(observed) | set(simulated))
    obs = np.array([observed.get(k, 0) for k in keys], dtype=np.float64)
    sim = np.array([simulated.get(k, 0) for k in keys], dtype=np.float64)

    # Expected simulated counts under the observed shares.
    exp = obs / obs.sum() * sim.sum()
    # Combine sparse bins (<5 expected) into 'Other' so the chi-square
    # approximation holds.
    keep = exp >= 5
    if keep.sum() < 2:
        return GoodnessOfFit(name=name, test="Chi-square",
                             statistic=float('nan'), p_value=float('nan'),
                             n_observed=n_obs, n_simulated=n_sim,
                             note="Too few categories with expected count >= 5. " + note)
    sim_kept = sim[keep]
    exp_kept = exp[keep]
    other_sim = sim[~keep].sum()
    other_exp = exp[~keep].sum()
    if other_exp >= 5:
        sim_kept = np.append(sim_kept, other_sim)
        exp_kept = np.append(exp_kept, other_exp)
    elif other_sim > 0 or other_exp > 0:
        # The pooled bin is itself below 5 expected; fold it into the
        # smallest kept bin rather than test a cell that breaks the rule.
        j = int(np.argmin(exp_kept))
        sim_kept[j] += other_sim
        exp_kept[j] += other_exp
    try:
        chi2, p = sp_stats.chisquare(sim_kept, f_exp=exp_kept)
    except Exception as e:
        return GoodnessOfFit(name=name, test="Chi-square",
                             statistic=float('nan'), p_value=float('nan'),
                             n_observed=n_obs, n_simulated=n_sim,
                             note=f"Chi-square failed: {e}. " + note)
    return GoodnessOfFit(name=name, test="Chi-square",
                         statistic=float(chi2), p_value=float(p),
                         n_observed=n_obs, n_simulated=n_sim, note=note)


# Live zone keys are Section wall names minus 'Section_'; the architecture
# engine names a second zone for the same category '<category>_<i>'.
_SPLIT_ZONE_RE = re.compile(r'^(.+)_(\d+)$')
_ZONE_WALL_PREFIX = 'Section_'


def _iter_shop_items(shop):
    """(item key, item dict) for every item on every floor."""
    if shop is None:
        return
    floors = getattr(shop, 'floors', None)
    if floors:
        for fid in sorted(floors):
            for name, data in ((floors[fid] or {}).get('items') or {}).items():
                yield str(name), (data or {})
    else:
        for name, data in (getattr(shop, 'items', None) or {}).items():
            yield str(name), (data or {})


def placed_product_ids(shop) -> set:
    """product_ids of the products the shop actually carries.

    A dataset-built layout keeps only the top products of each category, so
    the full catalogue is a different population from the one agents can
    buy from."""
    return {str(data['product_id']) for _, data in _iter_shop_items(shop)
            if data.get('product_id') is not None}


def _item_field_map(shop, field: str) -> Dict[str, str]:
    """Item key -> the item's ``field`` over every floor.

    The live counters key items the way the flattened item view does: a
    plain name for the lowest floor holding it, 'F<floor>:<name>' for a
    duplicate on a higher floor, so both spellings are listed."""
    out: Dict[str, str] = {}
    floors = getattr(shop, 'floors', None) or {}
    for name, data in _iter_shop_items(shop):
        value = data.get(field)
        if value is not None:
            out.setdefault(name, str(value))
    for fid in sorted(floors):
        for name, data in ((floors[fid] or {}).get('items') or {}).items():
            value = (data or {}).get(field)
            if value is not None:
                out.setdefault(f"F{fid}:{name}", str(value))
    return out


def item_category_map(shop) -> Dict[str, str]:
    """Item key -> category over every floor, keyed like the live counters
    (see ``_item_field_map``)."""
    return _item_field_map(shop, 'category')


def zone_category_map(shop) -> Dict[str, str]:
    """Live zone key -> category, read off the items standing in the zone.

    Items carry the exact name of the Section wall they sit in, and the
    live zone key is that name without the 'Section_' prefix. Reading the
    mapping off the items is the only reliable route: a department split
    over two gondolas gets a '<name>_<i>' zone, which can spell another
    category's name."""
    out: Dict[str, str] = {}
    for _, data in _iter_shop_items(shop):
        zone, cat = data.get('zone'), data.get('category')
        if zone and cat:
            z = str(zone)
            if z.startswith(_ZONE_WALL_PREFIX):
                z = z[len(_ZONE_WALL_PREFIX):]
            out.setdefault(z, str(cat))
    return out


def observed_category_counts(params: CalibratedParams,
                             product_ids=None) -> Dict[str, float]:
    """Product-invoice touches per category: each product's invoice count
    summed over its category. This counts item-transactions, the same
    quantity as the simulated per-item purchases; revenue shares would
    weight categories by price instead.

    ``product_ids`` restricts the reference to the products the shop
    carries, so both sides describe the same assortment. Falls back to the
    revenue shares when no per-product counts are available."""
    keep = None if product_ids is None else {str(p) for p in product_ids}
    counts: Dict[str, float] = {}
    for pid, n in (params.item_visit_counts or {}).items():
        if keep is not None and str(pid) not in keep:
            continue
        cat = params.item_categories.get(pid)
        if cat is not None:
            counts[cat] = counts.get(cat, 0.0) + float(n)
    if sum(counts.values()) > 0:
        return counts
    return {cat: share * params.n_invoices
            for cat, share in params.category_revenue_share.items()}


def simulated_category_purchases(sim, product_ids=None
                                 ) -> Tuple[Dict[str, int], int]:
    """Per-category item purchases recorded by the live simulation.

    ``analytics['item_conversion_rates'][item]['purchases']`` counts one
    purchase per item an agent shopped for and took to the checkout, and
    ``analytics['impulse_item_sales']`` the pick-ups made at the till,
    which are a separate set of items. Together they are every item that
    left the shop -- the unit the dataset's product-invoice touches are
    in.

    ``product_ids`` limits the count to items carrying one of those product
    ids, so the sample covers the same products as a reference restricted
    to them; an item added to the floor by hand carries none, and the
    dataset says nothing about how often it sells. Returns the counts and
    the number of purchases left out: items no longer on any floor and,
    with ``product_ids``, items outside that set.
    """
    A = sim.analytics
    shop = getattr(sim, 'shop', None)
    cats = item_category_map(shop)
    carried = None
    if product_ids is not None:
        wanted = {str(p) for p in product_ids}
        carried = {item for item, pid
                   in _item_field_map(shop, 'product_id').items()
                   if pid in wanted}
    sold: Dict[str, int] = {}
    for item, data in dict(A.get('item_conversion_rates', {})).items():
        n = int((data or {}).get('purchases', 0) or 0)
        if n > 0:
            sold[str(item)] = sold.get(str(item), 0) + n
    for item, n in dict(A.get('impulse_item_sales', {})).items():
        n = int(n or 0)
        if n > 0:
            sold[str(item)] = sold.get(str(item), 0) + n

    counts: Dict[str, int] = {}
    left_out = 0
    for item, n in sold.items():
        cat = cats.get(item)
        if cat is None or (carried is not None and item not in carried):
            left_out += n
        else:
            counts[cat] = counts.get(cat, 0) + n
    return counts, left_out


def category_visit_counts(zone_visits: Dict[str, int],
                          categories,
                          zone_categories: Optional[Dict[str, str]] = None
                          ) -> Tuple[Dict[str, int], int]:
    """Aggregate live zone visits onto dataset categories.

    ``zone_categories`` (from ``zone_category_map``) is used first because
    it is exact. Zones with no items fall back to the zone name and then to
    stripping a '<category>_<i>' split-section suffix. Zones that are not a
    category ('General Area', 'Checkout', 'WC') are excluded; their visit
    total is returned so callers can report it.
    """
    cats = set(categories)
    zmap = {str(k): str(v) for k, v in (zone_categories or {}).items()}
    counts = {c: 0 for c in cats}
    dropped = 0
    for zone, n in dict(zone_visits).items():
        zone, n = str(zone), int(n)
        cat = zmap.get(zone)
        if cat is None:
            if zone in cats:
                cat = zone
            else:
                m = _SPLIT_ZONE_RE.match(zone)
                cat = m.group(1) if m else None
        if cat in cats:
            counts[cat] += n
        else:
            dropped += n
    return counts, dropped


def observed_distinct_baskets(params: CalibratedParams) -> Optional[np.ndarray]:
    """Distinct products per invoice, when the calibration kept that sample.

    ``params.basket_sizes`` is units per invoice, which cannot be compared
    with the simulator's distinct items per visit."""
    arr = getattr(params, 'basket_distinct_sizes', None)
    if arr is None:
        return None
    arr = np.asarray(arr, dtype=np.float64)
    return arr if arr.size else None


def observed_distinct_revenues(params: CalibratedParams
                               ) -> Tuple[Optional[np.ndarray], str]:
    """Per-invoice revenue in the units a simulated visit spends in.

    ``params.invoice_revenues`` is sum(quantity x price), which a wholesale
    line of 200 identical units inflates, while a simulated visit buys one
    unit of each item it picks up. The comparable dataset quantity is the
    sum of unit prices over an invoice's distinct products. Forming it
    needs each invoice's lines, which only the calibration sees, so it is
    read from ``params.invoice_distinct_revenues``.

    The per-invoice totals cannot be converted instead. revenue / units is
    the invoice's quantity-weighted mean price, which bulk lines of cheap
    products pull down, so scaling it by the distinct-product count
    understates the one-unit-per-product sum -- by a fifth to a third at
    the median on Online Retail II, a gap a KS test of a few hundred visits
    detects on its own.

    Returns ``(sample, note)``; the sample is None when the calibration
    does not carry it, and the note then says why.
    """
    exact = getattr(params, 'invoice_distinct_revenues', None)
    if exact is not None:
        arr = np.asarray(exact, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            return arr, ("Observed = sum of unit prices over an invoice's "
                         "distinct products; simulated = sum of prices over "
                         "a visit's distinct items, one unit each. The "
                         "dataset side covers the whole catalogue, the "
                         "simulated side only the products placed in the "
                         "shop.")
    return None, ("the calibration carries no per-invoice sum of one unit "
                  "per distinct product; its per-invoice totals are "
                  "sum(quantity x price), which a simulated visit (one unit "
                  "of each item) cannot be compared with.")


def _timestamp_resolution_s(gaps: np.ndarray) -> float:
    """Smallest time step the source's timestamps can express.

    Invoice stamps in Online Retail II carry the minute only, so every gap
    is a multiple of 60 s and 7.6% of them are exactly zero. The greatest
    common divisor of the positive gaps recovers that step. Whole-second
    stamps (step 1 s) round just as coarsely once arrivals are a few
    seconds apart, so they count too; gaps that are not whole seconds mean
    the stamps are finer than that, and 0.0 asks for a continuous
    reference.
    """
    pos = np.asarray(gaps, dtype=np.float64)
    pos = pos[np.isfinite(pos) & (pos > 0)]
    if pos.size == 0 or not np.allclose(pos, np.rint(pos), atol=1e-6):
        return 0.0
    return float(np.gcd.reduce(np.rint(pos).astype(np.int64)))


def validate_against_simulation(params: CalibratedParams,
                                sim,
                                alpha: float = 0.05) -> ValidationResult:
    """Compare the just-run simulation's analytics against the empirical
    distributions in ``params``. Returns a structured ValidationResult.
    """
    A = sim.analytics
    res = ValidationResult(alpha=alpha)

    # Aggregate sources (e.g. Omnichannel) have no observed per-visit baskets
    # or revenue -- those arrays are a parametric realization of the measured
    # purchase probabilities, so a KS test against them would be circular.
    # We skip them and validate only what the source actually measures
    # (category shares; dwell / arrival metadata). The tag is read from the
    # params under test: the live calibration block can be cleared or can
    # hold another dataset's keys.
    extra = getattr(params, 'calibration_extra', None) or {}
    parametric = (str(extra.get('basket_size_source', '')).startswith('parametric')
                  or extra.get('source_kind') == 'aggregate_retail_omnichannel')

    if parametric:
        res.summary_lines.append(
            "Aggregate source: basket-size and per-visit-revenue distributions "
            "are parametric (not directly observed); KS tests on them are "
            "omitted as not applicable. Category shares are still validated.")
    else:
        # -- Basket size --
        # The simulator counts distinct items per visit, while
        # params.basket_sizes is units per invoice (a bulk line of 200
        # identical units is one item), so the test needs the distinct sample.
        sim_baskets = np.array(A.get('basket_sizes', []), dtype=np.float64)
        obs_distinct = observed_distinct_baskets(params)
        if obs_distinct is not None:
            t = _ks_safe(obs_distinct, sim_baskets,
                         name="Basket size (items/visit)",
                         note="Observed = distinct products per invoice; "
                              "simulated = distinct items per visit.")
        else:
            t = GoodnessOfFit(name="Basket size (items/visit)", test="KS-2sample",
                              statistic=float('nan'), p_value=float('nan'),
                              n_observed=int(np.asarray(params.basket_sizes).size),
                              n_simulated=int(sim_baskets.size),
                              note="Not run: the dataset basket sample counts units "
                                   "per invoice, the simulator counts distinct "
                                   "items per visit.")
        if t: res.tests.append(t)

        # -- Per-visit revenue --
        # A simulated visit buys one unit of each item on its list, so the
        # comparable dataset quantity is an invoice priced one unit per
        # product, not its wholesale total.
        sim_revs = np.array(A.get('customer_revenues', []), dtype=np.float64)
        obs_revs, rev_note = observed_distinct_revenues(params)
        if obs_revs is None:
            # Keep the row so the report shows revenue was not validated.
            t = GoodnessOfFit(name="Per-visit revenue", test="KS-2sample",
                              statistic=float('nan'), p_value=float('nan'),
                              n_observed=int(np.asarray(params.invoice_revenues).size),
                              n_simulated=int(sim_revs.size),
                              note="Not run: " + rev_note)
        else:
            t = _ks_safe(obs_revs, sim_revs,
                         name="Per-visit revenue", note=rev_note)
        if t: res.tests.append(t)

    # -- Category purchase shares (chi-square) --
    # Both sides count item purchases: the dataset's product-invoice touches
    # restricted to the products the shop carries, against the live per-item
    # purchase counters of those products mapped onto their category. Zone
    # entries are not comparable -- an agent crossing a department on its
    # way elsewhere is counted there too, and the dataset has no such notion.
    pids = placed_product_ids(getattr(sim, 'shop', None))
    cat_observed = observed_category_counts(params,
                                            product_ids=pids or None)
    cat_sim, left_out = simulated_category_purchases(
        sim, product_ids=pids or None)
    outside_set = " or are not among those products" if pids else ""
    if sum(cat_sim.values()) > 0:
        if pids:
            note = (f"Item purchases per category against the dataset's "
                    f"product-invoice touches, both over the {len(pids)} "
                    f"products placed in the shop.")
        else:
            note = ("Item purchases per category against the dataset's "
                    "product-invoice touches over the whole catalogue: the "
                    "shop's items carry no product ids to narrow it to the "
                    "assortment on the floor.")
        if left_out:
            note += (f" {left_out} purchases of items that are no longer on a "
                     f"floor{outside_set} were excluded.")
        t = _chi2_categorical(cat_observed, cat_sim,
                              name="Category purchase shares", note=note)
        if t: res.tests.append(t)
    elif left_out:
        res.summary_lines.append(
            f"{left_out} purchases were recorded, all of items that are no "
            f"longer on a floor{outside_set}; category shares were not tested."
        )
    else:
        res.summary_lines.append(
            "No simulated purchases yet -- run the simulation until customers "
            "reach the checkout before validating category shares."
        )

    # Zone entries measure traffic, including agents passing through a
    # department on the way somewhere else, so they are reported alongside
    # the test rather than tested.
    zone_visits = dict(A.get('area_visits', {}))
    if zone_visits:
        traffic, outside = category_visit_counts(
            zone_visits, cat_observed,
            zone_categories=zone_category_map(getattr(sim, 'shop', None)))
        total = sum(int(v) for v in zone_visits.values())
        top = ", ".join(
            f"{c} {n / max(total - outside, 1) * 100:.0f}%"
            for c, n in sorted(traffic.items(), key=lambda kv: -kv[1])[:3] if n)
        res.summary_lines.append(
            f"Traffic (not a test): {total - outside} of {total} zone entries "
            f"fell in a category section"
            + (f"; busiest {top}." if top else "."))

    # -- Inter-arrival time (currently informational only) --
    # The simulator drives spawn via a Poisson process with an hour-of-day
    # rate, not the dataset's empirical inter-arrival, so this test measures
    # how well a homogeneous Poisson approximates the data, not the
    # simulator itself. We report it for transparency.
    obs_ia = params.inter_arrival_seconds
    if (not parametric) and obs_ia.size > 100:
        # The reference is stamped at the source's own resolution. Invoice
        # times in Online Retail II carry the minute only, so a share of the
        # observed gaps is exactly zero; a continuous exponential reference
        # has no mass there, and the KS distance would report the timestamp
        # resolution instead of the arrival process.
        step = _timestamp_resolution_s(obs_ia)
        mean_gap = max(float(obs_ia.mean()), 1e-6)
        times = np.cumsum(np.random.default_rng(0).exponential(
            mean_gap, size=obs_ia.size + 1))
        if step > 0:
            times = np.floor(times / step) * step
        sim_ia = np.diff(times)
        note = ("Informational: tests whether the dataset's inter-arrival "
                "times are approximately Poisson (the family the simulator "
                "assumes)")
        note += (f", against a reference stamped at the source's {step:g} s "
                 f"resolution." if step > 0 else ".")
        note += (" The reference rate is constant, while the data's rate (and "
                 "the simulator's) varies by hour of day, so a rejection can "
                 "come from that variation too.")
        t = _ks_safe(obs_ia, sim_ia,
                     name="Inter-arrival vs. Poisson reference", note=note)
        if t: res.tests.append(t)

    return res
