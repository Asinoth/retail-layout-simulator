"""Goodness-of-fit tests between simulated and empirical distributions.

For a TOMACS submission the calibration story isn't complete without
showing how close the simulator's output matches the dataset it was
calibrated against. We compare three primary distributions:

  * basket size (distinct stocked products per invoice vs. distinct items
    per visit)
  * per-visit revenue (one unit of each stocked product on an invoice vs.
    one unit of each item a visit bought)
  * inter-arrival time -- informational, a test of the data: the
    dataset's gaps, time-rescaled by the calibrated hour-of-day rate,
    against Exp(1) (``_arrival_row``)

Test choice:
  - **Kolmogorov-Smirnov 2-sample** for continuous-ish distributions
    (basket size, revenue, rescaled inter-arrival). KS makes no
    distributional assumption -- appropriate when the simulator's
    distribution is itself empirical and may not match any known family.
  - **Chi-square with a cluster permutation null** for the per-category
    purchases. Items bought on one trip are not independent draws -- a
    shopper who came for one department buys several of its products --
    so the chi-square distribution, which assumes every purchase is a
    separate observation, rejects a model that reproduces the data
    exactly far more often than alpha. The p-value instead comes from
    reassigning whole baskets (simulated visits and reference invoices)
    between the two sides at random, which keeps every basket's purchases
    together; the chi-square-distribution p is reported beside it as the
    naive item-level figure. Sources without invoices (aggregate data)
    have no baskets to reassign and fall back to the item-level test,
    labelled as such.

Matching units is what makes these tests meaningful, and it is the part
that is easy to get wrong: the dataset counts units per invoice and prices
them at wholesale volumes, while a simulated visit picks up one unit of
each item on its list; the dataset's catalogue holds every product ever
sold, while the shop carries only the products that were laid out. The
basket and revenue references are therefore each invoice cut down to the
products the simulated shop stocks (``placed_invoice_sample``): a shopper
there cannot buy the rest. Each test below states, in its note, the
quantity both sides count.

We do NOT use t-tests on means: TOMACS reviewers will rightly note that
a t-test only compares first moments. KS compares the entire CDF.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
from scipy import stats as sp_stats

from dataset_calibration import CalibratedParams, placed_invoice_sample


@dataclass
class GoodnessOfFit:
    name: str
    test: str
    statistic: float
    p_value: float
    n_observed: int
    n_simulated: int
    note: str = ""
    # Test-specific record beside the common fields -- for the category
    # test the item-level p, the permutation count and seed, the categories
    # compared. Plain JSON values, so a runner can write it as it stands.
    extra: Dict[str, Any] = field(default_factory=dict)

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
            naive = t.extra.get('naive_p_value')
            if naive is not None and np.isfinite(naive):
                lines.append(f"          naive item-level p={naive:.4g} "
                             f"(treats purchases as independent)")
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
    """Item-level chi-square goodness of fit of the raw simulated counts to
    the observed category distribution.

    Only used where there are no baskets to reassign (aggregate sources):
    it counts every purchase as an independent observation, which purchases
    made on one visit are not, so it rejects a correct model more often
    than alpha. ``cluster_permutation_chi2`` is the test wherever invoices
    exist.

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
        return GoodnessOfFit(name=name, test=CATEGORY_ITEM_TEST_KIND,
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
        return GoodnessOfFit(name=name, test=CATEGORY_ITEM_TEST_KIND,
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
        return GoodnessOfFit(name=name, test=CATEGORY_ITEM_TEST_KIND,
                             statistic=float('nan'), p_value=float('nan'),
                             n_observed=n_obs, n_simulated=n_sim,
                             note=f"Chi-square failed: {e}. " + note)
    return GoodnessOfFit(name=name, test=CATEGORY_ITEM_TEST_KIND,
                         statistic=float(chi2), p_value=float(p),
                         n_observed=n_obs, n_simulated=n_sim, note=note)


# --- Category shares on whole baskets -------------------------------------

# Random relabellings behind the category test's p-value. Fixed with the
# test's design, before any result under it was seen. At 2000 the Monte
# Carlo standard error of a p-value near 0.05 is about 0.005.
CATEGORY_N_PERM = 2000
# Seed of those relabellings: a given pair of samples always gets the same
# p-value, so a re-run, a worker count or a GUI click cannot move it.
CATEGORY_PERM_SEED = 20260924
# The row's name, and the test's label, which the paper-macro validator
# also checks for.
CATEGORY_TEST_NAME = "Category purchase shares"
CATEGORY_TEST_KIND = "Chi-square, cluster permutation"
# Label of the fallback for sources with no invoices to reassign.
CATEGORY_ITEM_TEST_KIND = "Chi-square, item-level"
# Column holding every category no reference invoice touches.
UNREFERENCED_CATEGORY = "(not in reference)"
# Fewest baskets on either side the test is run with, the same floor as
# the KS rows' five samples.
MIN_CATEGORY_CLUSTERS = 5
# Uniform keys drawn per block of relabellings (about 32 MB as float64).
# Blocks only bound memory: the keys are drawn in one sequence, so the
# p-value does not depend on the block size.
_PERM_BLOCK_VALUES = 4_000_000


def pearson_chi2_two_rows(first, col_totals) -> np.ndarray:
    """Pearson chi-square of 2 x K tables given one row and the column
    totals.

    ``first`` is (..., K), one row of each table; the other row is the
    column totals minus it. In a two-row table the rows' deviations from
    their expected counts are equal and opposite, so the statistic is
    sum_k dev_k^2 (1/E1_k + 1/E2_k) with dev the first row's deviation,
    and is the same whichever row is called first. Written this way, the
    statistics of thousands of relabelled tables are one array expression.
    Every column total and both row totals must be positive.
    """
    col = np.asarray(col_totals, dtype=np.float64)
    first = np.asarray(first, dtype=np.float64)
    total = col.sum()
    r1 = first.sum(axis=-1, keepdims=True)
    r2 = total - r1
    e1 = r1 * col / total
    e2 = r2 * col / total
    dev = first - e1
    return (dev * dev * (1.0 / e1 + 1.0 / e2)).sum(axis=-1)


def cluster_permutation_chi2(sim_counts, ref_counts,
                             n_perm: int = CATEGORY_N_PERM,
                             seed: int = CATEGORY_PERM_SEED
                             ) -> Dict[str, Any]:
    """Two-sample test of category shares that moves whole baskets.

    ``sim_counts`` and ``ref_counts`` hold one row per basket (a simulated
    visit, a reference invoice) and one column per category: how many of
    the basket's purchases fall in each. The statistic is the Pearson
    chi-square of the 2 x K table of category totals, simulated against
    reference. Its null distribution comes from pooling the baskets and
    handing the two labels out again at random, ``n_perm`` times, keeping
    how many baskets each side has; p = (1 + #{relabelled >= observed}) /
    (1 + n_perm). Under the null that the simulated visits and the
    reference invoices are draws from one distribution of baskets -- which
    is what a model drawing each shopping list from the data's invoices
    claims -- the labels are exchangeable and this p-value is exact at any
    sample size, and because a basket moves as a unit, purchases made
    together stay together: the within-visit clustering that inflates the
    item-level chi-square is in the null distribution by construction.

    Each relabelling draws one uniform key per basket and gives the
    smaller side's label to the baskets with the smallest keys, a
    uniformly random split with both counts kept. The statistic is
    symmetric in the two rows, so the smaller side's totals are all it
    needs. A relabelled statistic within 1e-9 (relative) of the observed
    one counts as reaching it, so floating-point noise in a tie cannot
    shrink p.

    Baskets with no purchase in any column, and columns with no purchase
    on either side, carry nothing and are dropped first. Returns the
    statistic, the permutation p, ``naive_p_value`` (the chi-square(K - 1)
    tail of the same statistic, i.e. the item-level test, which treats
    every purchase as independent), the sizes of both sides in baskets and
    items, and ``clustering_inflation``: the relabelled statistics' mean
    over K - 1. It is about 1 when purchases are independent, above 1 when
    they cluster within baskets -- the factor by which that clustering
    inflates the item-level statistic -- and below 1 when a basket's
    purchases tend to avoid sharing a category. When the test cannot run,
    the numbers are NaN and ``reason`` says why.

    Effect sizes go with the p-value, since a chi-square grows with the
    sample and a tiny departure is significant on enough purchases. The
    one to read is ``share_tv_distance``, the total-variation distance
    between the two sides' category shares -- half the summed absolute
    share differences, the share of one side's purchases that would have
    to change category to match the other -- which does not depend on how
    many purchases either side holds. ``cramers_v`` is sqrt(chi2 / N) over
    the N purchases of the 2 x K table (min(2, K) - 1 = 1). In a two-sample
    table it depends on the split as well as on the shares: with p the
    first side's share of the purchases, chi2 / N = p (1 - p) sum_k
    (s1_k - s2_k)^2 / m_k (m_k the pooled share), so the same share
    difference gives a smaller V the more one side outweighs the other,
    and V is not to be read against conventional benchmarks.
    ``cramers_v_balanced`` = sqrt(chi2 / (4 N p (1 - p))) removes that
    factor: it equals V at an even split and changes with the split only
    through the pooled shares. ``sim_share_of_items`` is p.
    """
    if int(n_perm) < 1:
        raise ValueError("n_perm must be at least 1")
    sim = np.asarray(sim_counts, dtype=np.float64)
    ref = np.asarray(ref_counts, dtype=np.float64)
    width = max(sim.shape[-1] if sim.ndim == 2 else 0,
                ref.shape[-1] if ref.ndim == 2 else 0)
    sim = sim.reshape(-1, width) if sim.size else np.zeros((0, width))
    ref = ref.reshape(-1, width) if ref.size else np.zeros((0, width))
    sim = sim[sim.sum(axis=1) > 0]
    ref = ref[ref.sum(axis=1) > 0]
    n_perm = int(n_perm)
    out = {'statistic': float('nan'), 'p_value': float('nan'),
           'naive_p_value': float('nan'), 'df': 0, 'n_categories': 0,
           'n_permutations': n_perm, 'permutation_seed': int(seed),
           'n_sim_clusters': int(sim.shape[0]),
           'n_ref_clusters': int(ref.shape[0]),
           'n_items_sim': int(round(sim.sum())),
           'n_items_ref': int(round(ref.sum())),
           'clustering_inflation': float('nan'),
           'cramers_v': float('nan'), 'share_tv_distance': float('nan'),
           'cramers_v_balanced': float('nan'),
           'sim_share_of_items': float('nan'),
           'reason': ''}
    if min(sim.shape[0], ref.shape[0]) < MIN_CATEGORY_CLUSTERS:
        out['reason'] = (f"Insufficient baskets (need >= "
                         f"{MIN_CATEGORY_CLUSTERS} on each side).")
        return out
    pooled = np.vstack([sim, ref])
    col = pooled.sum(axis=0)
    used = col > 0
    pooled, col = pooled[:, used], col[used]
    k = int(pooled.shape[1])
    out['n_categories'] = k
    if k < 2:
        out['reason'] = "Fewer than two categories with a purchase."
        return out

    observed = float(pearson_chi2_two_rows(sim[:, used].sum(axis=0), col))
    df = k - 1
    n = int(pooled.shape[0])
    m = int(min(sim.shape[0], ref.shape[0]))
    rng = np.random.default_rng(seed)
    block = max(1, min(n_perm, _PERM_BLOCK_VALUES // n))
    null = np.empty(n_perm, dtype=np.float64)
    for start in range(0, n_perm, block):
        b = min(block, n_perm - start)
        keys = rng.random((b, n))
        cut = np.partition(keys, m - 1, axis=1)[:, m - 1:m]
        chosen = (keys <= cut).astype(np.float64)
        null[start:start + b] = pearson_chi2_two_rows(chosen @ pooled, col)
    tol = 1e-9 * max(abs(observed), 1.0)
    reached = int(np.count_nonzero(null >= observed - tol))
    sim_tot = sim[:, used].sum(axis=0)
    ref_tot = ref[:, used].sum(axis=0)
    share = float(sim_tot.sum() / col.sum())
    out.update({
        'statistic': observed,
        'p_value': (1.0 + reached) / (1.0 + n_perm),
        'naive_p_value': float(sp_stats.chi2.sf(observed, df)),
        'df': df,
        'clustering_inflation': float(null.mean() / df),
        'cramers_v': float(np.sqrt(observed / col.sum())),
        'cramers_v_balanced': float(np.sqrt(
            observed / (4.0 * col.sum() * share * (1.0 - share)))),
        'sim_share_of_items': share,
        'share_tv_distance': float(0.5 * np.abs(
            sim_tot / sim_tot.sum() - ref_tot / ref_tot.sum()).sum()),
    })
    return out


def placed_product_categories(params: CalibratedParams, shop
                              ) -> Dict[str, str]:
    """product_id -> category for every product the shop carries.

    Both sides of the category test put a product in the category the
    reference calibration (``params``) gives it, so a product counts in
    the same column whether a simulated visit or a reference invoice
    bought it. A store built from another period's calibration can carry
    a product under that period's label; the reference's decides. A
    product the reference does not know keeps the label on its fixture.
    """
    ref_cat = {str(k): str(v)
               for k, v in (getattr(params, 'item_categories', None)
                            or {}).items()}
    out: Dict[str, str] = {}
    for _, data in _iter_shop_items(shop):
        pid = data.get('product_id')
        if pid is None or str(pid) in out:
            continue
        cat = ref_cat.get(str(pid), data.get('category'))
        out[str(pid)] = 'Unknown' if cat is None else str(cat)
    return out


def invoice_category_clusters(params: CalibratedParams,
                              product_category: Dict[str, str],
                              categories: List[str]) -> np.ndarray:
    """One row per invoice holding at least one product of
    ``product_category``: how many of those products it holds in each of
    ``categories``. int64 (n, K), invoice order.

    Invoices store distinct products, so a row sums to the invoice's
    ``placed_invoice_sample`` size -- the unit a simulated visit buys in
    (one of each item). Vectorised over the CSR arrays like
    ``placed_invoice_sample``. Empty when the params carry no per-invoice
    product sets.
    """
    k = len(categories)
    if not getattr(params, 'has_invoice_structure', False) or k == 0:
        return np.zeros((0, k), dtype=np.int64)
    col_of = {c: j for j, c in enumerate(categories)}
    code = np.asarray([col_of.get(product_category.get(str(p)), -1)
                       for p in params.invoice_product_index],
                      dtype=np.int64)
    ptr = np.asarray(params.invoice_ptr, dtype=np.int64)
    items = np.asarray(params.invoice_items, dtype=np.int64)
    n_inv = int(ptr.size) - 1
    row = np.repeat(np.arange(n_inv, dtype=np.int64), np.diff(ptr))
    c = code[items]
    hit = c >= 0
    counts = np.bincount(row[hit] * k + c[hit],
                         minlength=n_inv * k).reshape(n_inv, k)
    return counts[counts.sum(axis=1) > 0].astype(np.int64)


def visit_category_clusters(visit_purchases, item_product: Dict[str, str],
                            product_category: Dict[str, str],
                            categories: List[str]
                            ) -> Tuple[np.ndarray, int, int]:
    """One row per simulated visit that bought at least one product of
    ``product_category``: its purchases in each of ``categories``.

    ``visit_purchases`` is ``analytics['visit_purchases']``, the item keys
    each paying visit took home; ``item_product`` maps an item key to its
    product id (``_item_field_map(shop, 'product_id')``). Returns the
    int64 (n, K) matrix, the purchases left out (items no longer on a
    floor, or carrying no product id) and the visits left with nothing,
    which are dropped as the reference drops invoices holding no placed
    product.
    """
    k = len(categories)
    col_of = {c: j for j, c in enumerate(categories)}
    rows: List[np.ndarray] = []
    left_out = 0
    empty = 0
    for basket in visit_purchases or ():
        v = np.zeros(k, dtype=np.int64)
        for item in basket or ():
            pid = item_product.get(str(item))
            j = (None if pid is None
                 else col_of.get(product_category.get(str(pid))))
            if j is None:
                left_out += 1
            else:
                v[j] += 1
        if v.any():
            rows.append(v)
        else:
            empty += 1
    matrix = np.vstack(rows) if rows else np.zeros((0, k), dtype=np.int64)
    return matrix, left_out, empty


def merge_unreferenced_categories(ref: np.ndarray, other: np.ndarray,
                                  categories: List[str]
                                  ) -> Tuple[np.ndarray, np.ndarray,
                                             List[str], List[str]]:
    """Pool the categories no reference basket touches into one column.

    The only merging the category test does. The permutation null needs no
    minimum expected count, so sparse categories stay as they are; but a
    category the reference never buys is pooled, not dropped -- a
    simulated purchase there is one the data never makes, which is
    evidence against the model -- and pooling keeps that evidence in one
    column instead of spreading it over several whose reference row is
    all zero. When the other side has not bought them either, they carry
    nothing and no column is kept for them. Returns both matrices, the
    column labels and the categories the reference never touches.
    """
    ref = np.asarray(ref, dtype=np.int64).reshape(-1, len(categories))
    other = np.asarray(other, dtype=np.int64).reshape(-1, len(categories))
    touched = ref.sum(axis=0) > 0
    if touched.all():
        return ref, other, list(categories), []
    unreferenced = [c for c, t in zip(categories, touched) if not t]
    labels = [c for c, t in zip(categories, touched) if t]
    if not other[:, ~touched].any():
        return ref[:, touched], other[:, touched], labels, unreferenced
    labels.append(UNREFERENCED_CATEGORY)

    def _merge(mat):
        return np.hstack([mat[:, touched],
                          mat[:, ~touched].sum(axis=1, keepdims=True)])
    return _merge(ref), _merge(other), labels, unreferenced


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


def item_product_map(shop) -> Dict[str, str]:
    """Item key -> product id over every floor, keyed like the live
    counters (see ``_item_field_map``)."""
    return _item_field_map(shop, 'product_id')


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


def observed_placed_invoices(params: CalibratedParams, product_ids
                             ) -> Tuple[Optional[Dict[str, Any]], str]:
    """The basket-size and per-visit-revenue reference for a shop carrying
    ``product_ids``: ``placed_invoice_sample`` -- distinct stocked products
    per invoice, and the same invoices at one unit of each stocked product
    -- over the invoices holding at least one of those products.

    The whole-catalogue samples (``observed_distinct_baskets`` /
    ``observed_distinct_revenues``) count products a shopper in the
    dataset-built shop could never buy; on Online Retail II that shop
    stocks about 100 of roughly 4,000 products.

    Returns ``(sample, reason)``: the sample dict, or None when no
    comparable reference can be formed, with ``reason`` saying why.
    """
    if not getattr(params, 'has_invoice_structure', False):
        return None, ("the calibration carries no per-invoice product sets, "
                      "so an invoice cannot be cut to the products the shop "
                      "stocks.")
    pids = {str(p) for p in (() if product_ids is None else product_ids)}
    if not pids:
        return None, ("the shop's items carry no product ids, so the stocked "
                      "portion of an invoice cannot be formed.")
    sample = placed_invoice_sample(params, pids)
    if not sample['n_with_placed']:
        return None, (f"none of the {len(pids)} products placed in the shop "
                      f"appears on an invoice.")
    return sample, ""


def _placed_note(sample: Dict[str, Any], n_pids: int) -> str:
    """Which invoices the placed reference keeps, for the row notes."""
    return (f" {sample['n_with_placed']:,} of {sample['n_invoices']:,} "
            f"invoices hold at least one of the {n_pids} products placed in "
            f"the shop; the rest are left out.")


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

    # The products the simulated shop carries. Every dataset-side sample
    # below is narrowed to them, since a simulated shopper can buy nothing
    # else.
    pids = placed_product_ids(getattr(sim, 'shop', None))

    if parametric:
        res.summary_lines.append(
            "Aggregate source: basket-size and per-visit-revenue distributions "
            "are parametric (not directly observed); KS tests on them are "
            "omitted as not applicable. Category shares are still validated.")
    else:
        # Both rows compare against each invoice cut down to the products
        # the simulated shop stocks: the whole invoice is not a quantity a
        # simulated shopper can reproduce.
        placed, why_not = observed_placed_invoices(params, pids)

        # -- Basket size --
        # The simulator counts distinct items per visit, while
        # params.basket_sizes is units per invoice (a bulk line of 200
        # identical units is one item), so the test needs distinct counts.
        sim_baskets = np.array(A.get('basket_sizes', []), dtype=np.float64)
        if placed is not None:
            t = _ks_safe(placed['sizes'], sim_baskets,
                         name="Basket size (items/visit)",
                         note="Observed = distinct stocked products per "
                              "invoice; simulated = distinct items per "
                              "visit." + _placed_note(placed, len(pids)))
        else:
            t = GoodnessOfFit(name="Basket size (items/visit)", test="KS-2sample",
                              statistic=float('nan'), p_value=float('nan'),
                              n_observed=0,
                              n_simulated=int(sim_baskets.size),
                              note="Not run: " + why_not
                                   + " Distinct stocked products per invoice "
                                   "is the only dataset count comparable "
                                   "with distinct items per visit.")
        if t: res.tests.append(t)

        # -- Per-visit revenue --
        # A simulated visit buys one unit of each item on its list, so the
        # comparable dataset quantity is an invoice priced one unit per
        # stocked product, not its wholesale total.
        sim_revs = np.array(A.get('customer_revenues', []), dtype=np.float64)
        if placed is not None:
            t = _ks_safe(placed['revenues'], sim_revs,
                         name="Per-visit revenue",
                         note="Observed = one unit of each stocked product "
                              "on an invoice, at the shop's calibrated "
                              "prices; simulated = one unit of each item a "
                              "visit bought." + _placed_note(
                                  placed, len(pids)))
        else:
            # Keep the row so the report shows revenue was not validated.
            t = GoodnessOfFit(name="Per-visit revenue", test="KS-2sample",
                              statistic=float('nan'), p_value=float('nan'),
                              n_observed=0,
                              n_simulated=int(sim_revs.size),
                              note="Not run: " + why_not
                                   + " One unit of each stocked product per "
                                   "invoice is the only dataset spend "
                                   "comparable with a simulated visit's.")
        if t: res.tests.append(t)

    # -- Category purchase shares --
    # Both sides count item purchases: the dataset's product-invoice touches
    # restricted to the products the shop carries, against the live
    # purchases of those products, per category. Zone entries are not
    # comparable -- an agent crossing a department on its way elsewhere is
    # counted there too, and the dataset has no such notion. Where the
    # calibration kept its invoices, the test moves whole baskets
    # (``_category_basket_test``); otherwise only per-item totals exist and
    # the item-level test below is all that can run.
    cat_observed = observed_category_counts(params,
                                            product_ids=pids or None)
    if getattr(params, 'has_invoice_structure', False):
        t, line = _category_basket_test(params, sim, pids)
        if t is not None:
            res.tests.append(t)
        if line:
            res.summary_lines.append(line)
    else:
        _category_item_test(params, sim, pids, cat_observed, res)

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

    # -- Inter-arrival time (informational: a test of the data, not the
    # simulator) --
    # The simulator draws its arrivals from a Poisson process whose rate
    # follows the hour-of-day profile, not from the dataset's empirical
    # gaps, so this row asks whether the dataset's own arrivals are
    # consistent with that process. It says nothing about the simulator's
    # output.
    obs_ia = np.asarray(params.inter_arrival_seconds, dtype=np.float64)
    if (not parametric) and obs_ia.size > 100:
        t = _arrival_row(params, obs_ia)
        if t: res.tests.append(t)

    return res


# --- The arrival row: time-rescaled gaps ----------------------------------

# Name of the arrival row. The paper-macro generator finds it by the
# 'inter-arrival' in it.
ARRIVAL_TEST_NAME = "Inter-arrival (time-rescaled) vs. Exp(1)"
# Seed of the arrival row's reference realisation: fixed, so a given
# calibration always gets the same p-value.
ARRIVAL_REFERENCE_SEED = 0
# The null the arrival row tests, recorded with it.
ARRIVAL_NULL = ("non-homogeneous Poisson process whose rate follows the "
                "calibrated hour-of-day profile and is the same on every "
                "trading day (the live simulator's arrival law)")


def hourly_arrival_rates(params: CalibratedParams) -> Optional[np.ndarray]:
    """Invoices per second in each hour of a trading day, from the
    calibrated hour-of-day profile: n_invoices x share_h / (trading days x
    3600). The rate a non-homogeneous Poisson process with that profile and
    the same volume on every trading day runs at. None when the params
    carry no profile or no trading-day count."""
    hod = np.asarray(getattr(params, 'dwell_hour_distribution', ()),
                     dtype=np.float64)
    extra = getattr(params, 'calibration_extra', None) or {}
    days = int(extra.get('n_trading_days') or 0)
    if hod.size != 24 or hod.sum() <= 0 or days <= 0 or params.n_invoices <= 0:
        return None
    return float(params.n_invoices) * hod / hod.sum() / (days * 3600.0)


def cumulative_intensity(t, rates: np.ndarray) -> np.ndarray:
    """Integrated rate from midnight to time of day ``t`` (seconds), for a
    rate that is constant within each hour (``rates``, 24 values per
    second)."""
    rates = np.asarray(rates, dtype=np.float64)
    t = np.clip(np.asarray(t, dtype=np.float64), 0.0, 86400.0)
    h = np.minimum((t // 3600.0).astype(np.int64), 23)
    cum = np.concatenate([[0.0], np.cumsum(rates * 3600.0)])
    return cum[h] + rates[h] * (t - 3600.0 * h)


def time_rescaled_gaps(starts, gaps, rates: np.ndarray) -> np.ndarray:
    """Each gap mapped through the integrated rate over it (the
    time-rescaling theorem): the expected number of arrivals the hourly
    rate puts inside the gap. Under a Poisson process with that rate the
    rescaled gaps are independent Exp(1) -- a gap in a busy hour is divided
    by a short local mean gap, one in a quiet hour by a long one, and a gap
    that spans an hour boundary by the rates of both hours."""
    starts = np.asarray(starts, dtype=np.float64)
    gaps = np.asarray(gaps, dtype=np.float64)
    return (cumulative_intensity(starts + gaps, rates)
            - cumulative_intensity(starts, rates))


def nhpp_reference_gaps(rates: np.ndarray, n_days: int, step: float,
                        rng: np.random.Generator):
    """Same-day gaps, and their start times, of a Poisson process running
    at ``rates`` (per second, per hour of day) for ``n_days`` days, with
    every arrival stamped down to ``step`` seconds as the source's
    timestamps are. Within an hour a Poisson count of arrivals is placed
    uniformly, which is exactly a Poisson process at that hour's rate."""
    rates = np.asarray(rates, dtype=np.float64)
    counts = rng.poisson(rates * 3600.0, size=(int(n_days), 24)).ravel()
    day = np.repeat(np.repeat(np.arange(int(n_days)), 24), counts)
    hour = np.repeat(np.tile(np.arange(24), int(n_days)), counts)
    t = hour * 3600.0 + rng.random(int(counts.sum())) * 3600.0
    if step > 0:
        t = np.floor(t / step) * step
    order = np.lexsort((t, day))
    day, t = day[order], t[order]
    same = day[1:] == day[:-1]
    return t[:-1][same], np.diff(t)[same]


def _homogeneous_reference_gaps(obs_ia: np.ndarray, step: float
                                ) -> np.ndarray:
    """Gaps of a constant-rate Poisson process at the data's mean gap,
    stamped at ``step``: the reference the arrival row used before it
    rescaled by the hourly rate, kept as a record beside it."""
    mean_gap = max(float(obs_ia.mean()), 1e-6)
    times = np.cumsum(np.random.default_rng(ARRIVAL_REFERENCE_SEED)
                      .exponential(mean_gap, size=obs_ia.size + 1))
    if step > 0:
        times = np.floor(times / step) * step
    return np.diff(times)


def _cv(x: np.ndarray) -> Optional[float]:
    x = np.asarray(x, dtype=np.float64)
    if x.size < 2 or x.mean() <= 0:
        return None
    return float(x.std(ddof=1) / x.mean())


def _arrival_row(params: CalibratedParams, obs_ia: np.ndarray
                 ) -> Optional[GoodnessOfFit]:
    """The arrival row: the dataset's same-day gaps, time-rescaled by the
    calibrated hour-of-day rate, against an equally rescaled Poisson
    reference at the same rates and timestamp resolution (review R47).

    A gap measured in seconds mixes the process's variation over the day
    with its randomness: a constant-rate reference then rejects a process
    that is exactly Poisson at an hourly rate. Rescaling each gap by the
    rate integrated over it removes the hour-of-day variation, so under the
    null of ``ARRIVAL_NULL`` the rescaled gaps are Exp(1). The reference is
    a realisation of that null -- an NHPP at the calibrated hourly rates
    over as many days as the data trade on -- with each arrival stamped
    down to the source's resolution (60 s on Online Retail II, where 7.6%
    of the gaps are zero) and rescaled the same way, so the KS distance
    does not report the timestamp resolution. What is left for the test to
    see is what the null leaves out: a rate that differs between days
    (weekdays, seasons), which makes the rescaled gaps over-dispersed
    (coefficient of variation above 1), and arrivals that come in bursts.

    Falls back to the former constant-rate reference, labelled as such,
    when the params carry no gap start times or no hourly profile."""
    step = _timestamp_resolution_s(obs_ia)
    homogeneous = _ks_safe(obs_ia, _homogeneous_reference_gaps(obs_ia, step),
                           name="Inter-arrival vs. Poisson reference")
    starts = np.asarray(getattr(params, 'inter_arrival_start_s', ()),
                        dtype=np.float64)
    rates = hourly_arrival_rates(params)
    extra_days = int((getattr(params, 'calibration_extra', None) or {})
                     .get('n_trading_days') or 0)
    if rates is None or starts.size != obs_ia.size:
        if homogeneous is not None:
            homogeneous.note = (
                "Informational: tests whether the dataset's inter-arrival "
                "times are those of a constant-rate Poisson process"
                + (f", against a reference stamped at the source's {step:g} s "
                   f"resolution." if step > 0 else ".")
                + " The calibration carries no gap start times or no "
                "hour-of-day profile, so the gaps could not be rescaled by "
                "the hourly rate; a rejection can come from the rate's "
                "variation over the day alone.")
            homogeneous.extra = {'null': 'homogeneous Poisson process',
                                 'rescaled': False,
                                 'timestamp_step_s': float(step)}
        return homogeneous

    obs_rescaled = time_rescaled_gaps(starts, obs_ia, rates)
    rng = np.random.default_rng(ARRIVAL_REFERENCE_SEED)
    ref_starts, ref_gaps = nhpp_reference_gaps(rates, extra_days, step, rng)
    ref_rescaled = time_rescaled_gaps(ref_starts, ref_gaps, rates)
    note = ("Informational, a test of the data rather than the simulator: "
            "the dataset's same-day inter-arrival gaps, each divided by the "
            "local mean gap the calibrated hour-of-day rate implies (the "
            "rate integrated over the gap), against a Poisson reference at "
            "the same hourly rates over the same number of trading days. "
            f"Null: {ARRIVAL_NULL}; under it the rescaled gaps are Exp(1).")
    note += (f" Both sides are stamped at the source's {step:g} s resolution "
             f"before rescaling." if step > 0 else "")
    note += (f" Rescaled data: mean {obs_rescaled.mean():.3f}, coefficient "
             f"of variation {_cv(obs_rescaled) or float('nan'):.3f} (Exp(1): "
             f"1 and 1); a CV above 1 points to a rate that varies between "
             f"days or to bursts, which the null leaves out.")
    t = _ks_safe(obs_rescaled, ref_rescaled, name=ARRIVAL_TEST_NAME, note=note)
    if t is None:
        return None
    t.extra = {
        'null': ARRIVAL_NULL,
        'rescaled': True,
        'rescaling': 'rate integrated over each gap (time-rescaling)',
        'timestamp_step_s': float(step),
        'n_trading_days': int(extra_days),
        'reference_seed': int(ARRIVAL_REFERENCE_SEED),
        'n_reference_gaps': int(ref_rescaled.size),
        'rescaled_mean': _json_float(obs_rescaled.mean()),
        'rescaled_cv': _cv(obs_rescaled),
        'reference_rescaled_mean': _json_float(ref_rescaled.mean()),
        'reference_rescaled_cv': _cv(ref_rescaled),
        'homogeneous_statistic': (None if homogeneous is None
                                  else _json_float(homogeneous.statistic)),
        'homogeneous_p_value': (None if homogeneous is None
                                else _json_float(homogeneous.p_value)),
    }
    return t



def _no_purchases_line(left_out: int, outside_set: str) -> str:
    """The report line for a run whose purchases give the category test
    nothing to count."""
    if left_out:
        return (f"{left_out} purchases were recorded, all of items that are "
                f"no longer on a floor{outside_set}; category shares were "
                f"not tested.")
    return ("No simulated purchases yet -- run the simulation until customers "
            "reach the checkout before validating category shares.")


def _json_float(x) -> Optional[float]:
    x = float(x)
    return x if np.isfinite(x) else None


def _category_basket_test(params: CalibratedParams, sim, pids
                          ) -> Tuple[Optional[GoodnessOfFit], str]:
    """The category row as a test on whole baskets: simulated visits
    against reference invoices, both cut to the products placed in the
    shop, compared by ``cluster_permutation_chi2``.

    Returns the row (None when there is nothing to test yet) and a report
    line saying why a row is missing, if one is."""
    A = sim.analytics
    shop = getattr(sim, 'shop', None)
    name = CATEGORY_TEST_NAME

    def _not_run(why, n_obs=0, n_sim=0):
        return GoodnessOfFit(name=name, test=CATEGORY_TEST_KIND,
                             statistic=float('nan'), p_value=float('nan'),
                             n_observed=int(n_obs), n_simulated=int(n_sim),
                             note="Not run: " + why)

    visits = A.get('visit_purchases')
    if visits is None:
        # Per-item totals alone cannot be split back into visits. A record
        # that has none of either is a run nobody has paid in yet.
        cat_sim, left_out = simulated_category_purchases(
            sim, product_ids=pids or None)
        if not cat_sim and not left_out:
            return None, _no_purchases_line(0, "")
        return _not_run(
            "the analytics record carries no per-visit purchases "
            "('visit_purchases'), only per-item totals, and the category "
            "test compares whole visits with whole invoices. Re-run the "
            "simulation to record them."), ""
    if not pids:
        return _not_run(
            "the shop's items carry no product ids, so neither a visit nor "
            "an invoice can be cut to the products placed in the shop."), ""

    product_category = placed_product_categories(params, shop)
    categories = sorted(set(product_category.values()))
    ref = invoice_category_clusters(params, product_category, categories)
    simm, left_out, empty = visit_category_clusters(
        visits, _item_field_map(shop, 'product_id'), product_category,
        categories)
    if simm.shape[0] == 0:
        return None, _no_purchases_line(
            left_out, " or carry no product id")
    if ref.shape[0] == 0:
        return _not_run(f"none of the {len(pids)} products placed in the "
                        f"shop appears on an invoice.",
                        n_sim=simm.shape[0]), ""
    ref, simm, labels, unreferenced = merge_unreferenced_categories(
        ref, simm, categories)
    r = cluster_permutation_chi2(simm, ref)

    note = (f"Whole baskets: {r['n_ref_clusters']:,} invoices holding at "
            f"least one of the {len(pids)} products placed in the shop "
            f"against {r['n_sim_clusters']:,} simulated visits that bought "
            f"one, each reduced to its purchases of those products per "
            f"category ({r['n_categories']} categories; "
            f"{r['n_items_ref']:,} reference and {r['n_items_sim']:,} "
            f"simulated purchases). "
            f"Statistic = Pearson chi-square of the 2 x K table of category "
            f"totals; p from {r['n_permutations']:,} random reassignments of "
            f"the visit / invoice labels (seed {r['permutation_seed']}), "
            f"which move each basket whole.")
    if np.isfinite(r['naive_p_value']):
        note += (f" Naive item-level p = {r['naive_p_value']:.3g} (the "
                 f"chi-square({r['df']}) tail, which treats every purchase as "
                 f"independent; the reassigned statistics average "
                 f"{r['clustering_inflation']:.2f} x its null mean).")
    if np.isfinite(r['share_tv_distance']):
        note += (f" Effect size: the category shares differ by a "
                 f"total-variation distance of "
                 f"{r['share_tv_distance']:.3f} (Cramer's V = "
                 f"{r['cramers_v']:.3f} at a {100 * r['sim_share_of_items']:.0f}"
                 f"% simulated share of the purchases, "
                 f"{r['cramers_v_balanced']:.3f} balanced; V shrinks as "
                 f"the split grows uneven, so it is not read against "
                 f"conventional benchmarks).")
    if UNREFERENCED_CATEGORY in labels:
        note += (f" Categories no reference invoice touches are pooled into "
                 f"one column: {', '.join(unreferenced)}.")
    elif unreferenced:
        note += (f" Categories neither side bought are left out: "
                 f"{', '.join(unreferenced)}.")
    if left_out:
        note += (f" {left_out} purchases of items that are no longer on a "
                 f"floor or carry no product id were excluded.")
    if empty:
        note += (f" {empty} paying visits bought none of the placed products "
                 f"and are left out, as invoices holding none are.")
    if r['reason']:
        note = "Not run: " + r['reason'] + " " + note
    extra = {'naive_p_value': _json_float(r['naive_p_value']),
             'n_permutations': int(r['n_permutations']),
             'permutation_seed': int(r['permutation_seed']),
             'df': int(r['df']),
             'n_categories': int(r['n_categories']),
             'categories': list(labels),
             'unreferenced_categories': list(unreferenced),
             'n_items_observed': int(r['n_items_ref']),
             'n_items_simulated': int(r['n_items_sim']),
             'clustering_inflation': _json_float(r['clustering_inflation']),
             # Effect sizes beside the p-value (review R29): a chi-square
             # grows with the purchases counted. The share distance is the
             # one to read; V also depends on the split between the sides.
             'share_tv_distance': _json_float(r['share_tv_distance']),
             'effect_size_primary': 'share_tv_distance',
             'cramers_v': _json_float(r['cramers_v']),
             'cramers_v_balanced': _json_float(r['cramers_v_balanced']),
             'sim_share_of_items': _json_float(r['sim_share_of_items']),
             'items_left_out': int(left_out),
             'visits_without_placed_item': int(empty)}
    return GoodnessOfFit(name=name, test=CATEGORY_TEST_KIND,
                         statistic=float(r['statistic']),
                         p_value=float(r['p_value']),
                         n_observed=int(r['n_ref_clusters']),
                         n_simulated=int(r['n_sim_clusters']),
                         note=note, extra=extra), ""


def _category_item_test(params: CalibratedParams, sim, pids,
                        cat_observed: Dict[str, float],
                        res: ValidationResult) -> None:
    """The category row for a calibration with no invoices to reassign
    (aggregate sources): per-item totals against the reference shares,
    the item-level test, labelled as such."""
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
                              name=CATEGORY_TEST_NAME, note=note)
        if t:
            t.note += (" Item-level test: the source has no invoices whose "
                       "baskets could be reassigned, so every purchase counts "
                       "as an independent observation. Purchases made on one "
                       "visit are not independent, so this test rejects a "
                       "correct model more often than alpha.")
            res.tests.append(t)
    else:
        res.summary_lines.append(_no_purchases_line(left_out, outside_set))
