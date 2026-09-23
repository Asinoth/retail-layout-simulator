"""Cited constants for the retail-layout simulator.

Every magic number that used to sit inline in viz_ga / viz_optimize /
viz_whatif lives here instead, next to the source it came from. Pair this
file with the paper's Methods section: a reviewer can read it against the
cited papers and confirm nothing was conjured from thin air.

A few conventions worth knowing before you touch anything:

Scoring weights are equal across the five SPATIAL criteria -- we don't
claim the optimal weights are known, and equal weighting is just the
"no informative prior" choice (Dawes 1979 on improper linear models).
The two non-spatial criteria get fixed shares instead: section
compliance keeps 0.16 as a soft structural constraint (the same size as
one spatial share), and entrance proximity is down-weighted to 0.04
(tie-breaker, largely redundant with traffic + flow). Robustness is left
to the Sensitivity tab to demonstrate.

Elasticities (layout-score -> conversion / impulse / basket-size lift) are
bounded ranges from the cited papers, not point estimates. The simulator
uses each range's midpoint; the half-width lives in ELASTICITY_*_UNCERTAINTY
so the paper can show the headline lift sits inside the literature band.

Penalties (overlap, bottleneck) are deliberately on a stricter scale than
the positive criteria: one overlapping pair (-0.80) outweighs almost any
positive-score gain, so a colliding layout is strongly dominated -- a soft
penalty, not a hard lexicographic constraint. Feasibility itself is
guaranteed downstream by the repair steps (GUI _repair_item_overlaps after
apply; headless _repair_chrom_overlaps before MC scoring).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


# --- citations (informal, for the paper appendix) ---

CITATIONS: Dict[str, str] = {
    "larson2005":  ("Larson, J.S., Bradlow, E.T., Fader, P.S. (2005). "
                    "An exploratory look at supermarket shopping paths. "
                    "International Journal of Research in Marketing, 22(4), 395-414. "
                    "(Originally Marketing Science working paper, used RFID tracking of "
                    "~27,000 shopping trips; established perimeter-dominant traffic "
                    "and 14 path archetypes.)"),

    "hui2009":     ("Hui, S.K., Bradlow, E.T., Fader, P.S. (2009). "
                    "Testing behavioral hypotheses using an integrated model of "
                    "grocery store shopping path and purchase behavior. "
                    "Journal of Consumer Research, 36(3), 478-493. "
                    "(Joint path-and-purchase model; reports ~10-15%% conversion "
                    "drop per additional meter of detour from category zones.)"),

    "hui2013":     ("Hui, S.K., Inman, J.J., Huang, Y., Suher, J. (2013). "
                    "The effect of in-store travel distance on unplanned spending: "
                    "applications to mobile promotion strategies. "
                    "Journal of Marketing, 77(2), 1-16. "
                    "(Path-tracking field study; ~0.4%% unplanned-spending lift "
                    "per additional meter walked; 50-70%% impulse-rate gap "
                    "between checkout-adjacent (<3m) and remote (>10m) impulse SKUs.)"),

    "sorensen2009": ("Sorensen, H. (2009). Inside the Mind of the Shopper: "
                     "The Science of Retailing. Pearson FT Press. "
                     "(Practitioner reference for hot-zone front-of-store traffic "
                     "and eye-level shelf-placement lift estimates.)"),

    "botsali2005": ("Botsali, A.R., Peters, B.A. (2005). A network based layout "
                    "design model for retail stores. Proceedings of the 2005 "
                    "Industrial Engineering Research Conference, Atlanta, GA. "
                    "(Network model of a SERPENTINE store maximizing expected "
                    "impulse-purchase revenue via a product-visibility factor "
                    "-- exposure count along the shopper's tour. Analytical, "
                    "not simulated; closest prior work to our impulse and "
                    "traffic-exposure criteria.)"),

    "brscic2013":  ("Brscic, D., Kanda, T., Ikeda, T., Miyashita, T. (2013). "
                    "Person tracking in large public spaces using 3-D range sensors. "
                    "IEEE Transactions on Human-Machine Systems, 43(6), 522-534. "
                    "(ATC Shopping Mall pedestrian-tracking dataset; ~92 days of "
                    "person-level trajectories in an Osaka mall; reference dataset "
                    "for indoor pedestrian flow validation.)"),

    "dawes1979":   ("Dawes, R.M. (1979). The robust beauty of improper linear "
                    "models in decision making. American Psychologist, 34(7), "
                    "571-582. (Foundation for equal-weighting when component "
                    "weights are uncertain.)"),
}


# --- GA composite scoring weights ---
# The five spatial criteria are equal-weighted (Dawes 1979). Section
# compliance is a soft structural constraint with its own fixed share --
# a 'shoe in lingerie' violation should cost as much as a whole spatial
# criterion. Penalties sit on a strictly higher scale so a colliding layout
# can never beat a compliant one.
#
# Seven positive criteria: five spatial ones share 0.80 of the [0,1] budget
# (0.80 / 5 = 0.16 each), section compliance gets 0.16, and accessibility
# gets 0.04 (it's revenue-weighted and already overlaps traffic + flow).
#
# There is no dwell-time criterion. A per-category dwell average does not
# depend on where items are placed, so it would add the same constant to
# every layout -- no search signal, yet it would still shift the absolute
# score that the elasticities turn into revenue.

GA_W_TRAFFIC            = 0.80 / 5   # 0.16  (Larson 2005 perimeter)
GA_W_CROSS_MERCH        = 0.80 / 5   # 0.16  (Hui 2009 co-purchase distance)
GA_W_IMPULSE            = 0.80 / 5   # 0.16  (Hui 2013 in-zone impulse lift)
GA_W_FLOW               = 0.80 / 5   # 0.16  (Larson 2005 path efficiency)
GA_W_REVENUE_PLACEMENT  = 0.80 / 5   # 0.16  (Sorensen 2009 hot zones)
GA_W_SECTION_COMPLIANCE = 0.16       # structural (Ozgormus & Smith 2020, block/department layout)
GA_W_ACCESSIBILITY      = 0.04       # tie-breaker (Hui 2013 entrance distance)

# positive weights must sum to 1.0
assert abs(GA_W_TRAFFIC + GA_W_CROSS_MERCH + GA_W_IMPULSE + GA_W_FLOW
           + GA_W_REVENUE_PLACEMENT
           + GA_W_SECTION_COMPLIANCE + GA_W_ACCESSIBILITY - 1.0) < 1e-9

# penalties, on a strictly higher scale than the positives
GA_PEN_OVERLAP    = 0.80   # any item-on-item or item-on-wall overlap is a deal-breaker
GA_PEN_BOTTLENECK = 0.20   # measured congestion at the item's cell


# --- elasticities: layout-score -> behavioral parameter lift ---
# Each elasticity is a "% change in the parameter per unit change in the
# corresponding 0..1 score". Midpoints come from the cited papers; the
# *_UNCERTAINTY half-width lets a sensitivity analysis show the headline
# lift stays inside the literature band.

# Conversion. This band is OUR construction, anchored by Hui 2009, not a
# number that appears in the paper. The chain, stated explicitly so the
# assumptions are auditable:
#   (a) Hui 2009 reports ~10-15% purchase-probability drop per meter of
#       detour from a category zone (a category-level, nonlinear effect);
#   (b) we LINEARIZE that marginal effect over an ASSUMED ~5m detour gap
#       between a random and an optimal layout (the 5m is our modeling
#       choice, not from the paper);
#   (c) we treat the aggregated category-level effect as a store-level
#       conversion multiplier (a composition Hui does not make).
# The product gives a 50-75% ceiling at score=1.0; we apply 20% base +
# up to 30% SA-modulated. The comparative results are invariant to this
# choice (common factor across methods) and the LHS sweep spans the band.
ELASTICITY_CONV_BASE     = 0.20    # base (conservative end of the construction)
ELASTICITY_CONV_GAIN_MAX = 0.30    # additional, scaled by SA conv-weight
ELASTICITY_CONV_CITE     = "hui2009"

# Impulse. Hui Inman 2013 report a 50-70% impulse-rate gap between
# checkout-adjacent (<3m) and remote (>10m) SKUs -- that gap is the one
# band here taken DIRECTLY from a cited result. Our applied band is 30%
# (base) + up to 40% (SA-modulated) = 30-70%: the 70% ceiling matches the
# cited gap; the 30% floor sits BELOW the cited 50% lower bound as
# deliberate conservatism (a layout rarely moves every impulse SKU from
# 'remote' to 'checkout-adjacent', so applying the full cited floor to
# the whole assortment would overstate the effect).
ELASTICITY_IMP_BASE      = 0.30    # deliberately below the cited 50% floor
ELASTICITY_IMP_GAIN_MAX  = 0.40    # additional, scaled by SA impulse-weight
ELASTICITY_IMP_CITE     = "hui2013"

# Basket size. The weakest translation of the three, and labeled as a
# cited-ANCHORED ASSUMPTION, not a cited band. What the citation gives:
# Hui Inman 2013 report +0.4% unplanned spending per meter walked, which
# over our assumed ~5m gap is only ~2%. The applied 10-30% band inflates
# that by invoking planned-list completion effects (a better-organized
# layout helps shoppers find and complete planned purchases) -- a
# mechanism with face validity but NO quantified citation. The paper
# names this the weakest coefficient; the LHS sweep bounds its influence,
# and the comparative findings do not depend on it (common factor).
ELASTICITY_BSK_BASE      = 0.10
ELASTICITY_BSK_GAIN_MAX  = 0.20
ELASTICITY_BSK_CITE      = "hui2013"

# Abandonment recovery: flow + section + bottleneck additively reduce
# abandonment. Coefficients are set so a fully-compliant optimal layout
# (flow=1, section=1, bottleneck=0) recovers ~60% of baseline abandonment,
# while a bad one (flow=0, section=0, bottleneck=1) inflates it ~50%.
# Sources: Larson 2005 (flow), Ozgormus & Smith 2020 (department/section
# organization), Hui 2013 (congestion).
ABANDON_FLOW_COEF        = 0.40
ABANDON_SECTION_COEF     = 0.20
ABANDON_BOTTLENECK_COEF  = 0.50
ABANDON_FLOOR_FRAC       = 0.30    # never recover more than 70% of abandonment


# --- impulse sub-model (live agent) ---
# These used to sit inline in customer.py, which quietly broke the
# "every cited coefficient lives here" rule the paper leans on (audit
# R36). Values are unchanged from the inline originals.
#
# The catchment radius is the checkout-adjacent band of Hui Inman 2013
# (<3 m converts markedly better than remote impulse facings). The
# per-type propensities and the two situational bonuses are operational
# defaults, not cited figures -- an agent that is browsing, or that has
# already been in the store a while, is more impulse-prone.

IMPULSE_CHECKOUT_RADIUS_M = 3.0      # Hui Inman 2013 checkout-adjacent band
IMPULSE_DECAY_PER_ITEM    = 0.6      # each extra impulse item is 0.6x as likely
IMPULSE_BASE_PROPENSITY   = {        # by customer type (operational)
    'quick':    0.15,
    'browser':  0.35,
    'thorough': 0.25,
}
IMPULSE_BROWSER_BONUS     = 0.10     # added for 'browser' agents
IMPULSE_LONG_DWELL_BONUS  = 0.15     # added after this many seconds in store
IMPULSE_LONG_DWELL_SECS   = 120.0


# --- basket composition ---
# Shopping lists used to be drawn uniformly over the assortment, which
# meant the calibrated co-purchase structure reached the layout score
# and the MC parameters but never an agent. Co-locating two frequently
# co-purchased products therefore could not change anyone's behaviour
# inside the ABM, so that criterion was carried by the model rather
# than exhibited by it.
#
# Lists are now drawn with popularity weighting, and each item after
# the first is taken from the co-purchase partners of something already
# in the basket with probability BASKET_AFFINITY_PROB. Both fall back
# to the old uniform draw when no calibration is loaded, so generated
# and synthetic shops behave exactly as before.

BASKET_AFFINITY_PROB = 0.35   # chance the next item comes from an affinity pair
BASKET_POP_SMOOTHING = 1.0    # additive smoothing on popularity counts


# --- shopping-list length (uncalibrated shops only) ---
# How many regular items an agent sets out for, by customer type: an
# inclusive (min, max) range, drawn uniformly. This is an OPERATIONAL
# ASSUMPTION carried over from the original simulator, not a literature
# value -- nothing cited in this module measures list length per shopper
# type, and a transaction log cannot label its buyers 'quick' or
# 'thorough' either.
#
# It applies only while no calibration supplies a list-length sample. A
# transactional calibration seeded onto its own shop replaces it with the
# distribution of distinct STOCKED products per invoice
# (``analytics['calibration']['list_length_sample']``), which the data do
# observe, so a calibrated shop's list length no longer depends on type.

LIST_LENGTH_BY_TYPE = {
    'quick':    (2, 4),
    'browser':  (4, 7),
    'thorough': (6, 12),
}


# --- checkout service time ---
# Service was a flat U(3,8) s draw, independent of what the agent was
# carrying. Two things were wrong with that. Real checkout service is
# right-skewed, not uniform; and it scales with basket contents, so a
# layout that enlarges baskets lengthens its own queues. Omitting that
# feedback understated congestion under exactly the layouts the
# optimizer prefers.
#
# The replacement is the standard decomposition -- a fixed overhead
# (greeting, payment, bagging) plus a per-item handling time -- carried
# through a lognormal multiplier for the right tail. Constants are
# operational, chosen so a typical basket still services in a few
# seconds and the mean is close to the previous draw's, which keeps the
# nominal load regime comparable to earlier measurements.

CHECKOUT_BASE_SECS      = 2.0    # fixed overhead per transaction
CHECKOUT_PER_ITEM_SECS  = 0.5    # marginal handling cost per basket item
CHECKOUT_LOGNORM_SIGMA  = 0.25   # right-skew; median multiplier 1.0
CHECKOUT_MIN_SECS       = 1.0    # floor, so the draw cannot vanish


# --- agent progress monitoring ---
# Implementation parameters of the live agent, not literature coefficients.
# An agent counts as making progress once its net displacement from the
# anchor position reaches STUCK_PROGRESS_M; if that does not happen within
# the window (simulated seconds, per state) it is treated as stuck.

STUCK_PROGRESS_M       = 0.25   # net displacement that counts as progress
STUCK_WINDOW_MOVING_S  = 3.0    # no-progress window while moving to a target
STUCK_WINDOW_EXITING_S = 5.0    # no-progress window while heading for the exit


# --- operational defaults ---
# Fallbacks for the MC engine and reports before any dataset is loaded.
# Derived from UCI Online Retail II averages and Sorensen (2009).

DEFAULT_OP_HOURS_PER_DAY     = 10.0
DEFAULT_WEEKEND_MULTIPLIER   = 1.4   # Sorensen 2009 reports weekend lift 1.3-1.6
DEFAULT_DAY_NOISE_STD        = 0.08  # lognormal day-level traffic noise

# Hour-of-day arrival multipliers. Shape from UCI Online Retail II (UK
# retailer, 09:00-18:00 with a lunchtime peak).
#
# The RAW shape below is a relative profile; it is normalized at import
# so the mean over OPEN (non-zero) hours is exactly 1.0, which is the
# property the paper's arrival equation states. The raw numbers summed
# to 14.2 over 11 open hours (mean 1.29), so using them unnormalized
# would have run arrivals ~29% hot -- a live trap for anyone wiring this
# in, since nothing currently reads it (audit R64).
_RAW_HOURLY_SHAPE = (
    #   00    01    02    03    04    05    06    07    08    09
      0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.30, 0.95,
    #   10    11    12    13    14    15    16    17    18    19
      1.55, 1.85, 2.10, 1.95, 1.60, 1.40, 1.20, 0.90, 0.40, 0.00,
    #   20    21    22    23
      0.00, 0.00, 0.00, 0.00,
)

_open_hours = [h for h in _RAW_HOURLY_SHAPE if h > 0.0]
_open_mean = sum(_open_hours) / len(_open_hours)
DEFAULT_HOURLY_PROFILE = tuple(h / _open_mean for h in _RAW_HOURLY_SHAPE)

# The normalization the paper's arrival equation asserts, checked here so
# it cannot drift: mean over open hours == 1.
assert abs(sum(h for h in DEFAULT_HOURLY_PROFILE if h > 0.0)
           / len([h for h in DEFAULT_HOURLY_PROFILE if h > 0.0]) - 1.0) < 1e-9


def cite(*keys: str) -> str:
    """Multi-line citation block for the given keys, ready to paste into the
    optimization report or the paper appendix."""
    return "\n".join(f"  [{k}]  {CITATIONS[k]}" for k in keys if k in CITATIONS)


@dataclass(frozen=True)
class GaWeights:
    """Snapshot of the active GA weights -- handy for the methods table and
    for sensitivity sweeps."""
    traffic: float = GA_W_TRAFFIC
    cross_merch: float = GA_W_CROSS_MERCH
    impulse: float = GA_W_IMPULSE
    flow: float = GA_W_FLOW
    revenue_placement: float = GA_W_REVENUE_PLACEMENT
    section_compliance: float = GA_W_SECTION_COMPLIANCE
    accessibility: float = GA_W_ACCESSIBILITY
    overlap_penalty: float = GA_PEN_OVERLAP
    bottleneck_penalty: float = GA_PEN_BOTTLENECK
