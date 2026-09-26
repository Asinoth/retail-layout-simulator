"""Constants for the retail-layout simulator, each with its source.

Every number that shapes a result -- the layout score, the score-to-revenue
transform and the Monte Carlo engine's guards, the stand-in inputs the
headless runners use where a dataset measures nothing, the cold-start
traffic priors, the synthetic scenarios' analytical ground truth, and the
live agent's behaviour -- lives here, next to where it came from. Pair this file with
the paper's Methods section and its coefficient tables: a reviewer can read
it against the cited papers. They give the direction of an effect at most;
no number here is read from one.

Every constant carries one of these labels:

  ASSUMPTION         a modelling choice whose direction, at most, is
                     cited; its size is ours;
  TUNED              set to hit a stated target, not measured;
  OPERATIONAL ASSUMPTION
                     a modelling choice nothing cited measures;
  STAND-IN           a placeholder for an input the data do not observe
                     (it applies only where no dataset supplies the value);
  NUMERICAL GUARD    a clamp or floor that keeps a formula finite; it binds
                     only at extreme inputs.

A few conventions worth knowing before you touch anything:

Scoring weights are equal across the five SPATIAL criteria -- we don't
claim the optimal weights are known, and equal weighting is the "no
informative prior" choice (Dawes 1979 on improper linear models). Dawes'
argument is about STANDARDIZED predictors; the criteria here are raw
[0, 1] scores whose spread over feasible layouts differs by two orders of
magnitude, so the nominal weights are not the effective ones.
``experiments._common.criterion_spread`` reports each criterion's spread
and effective weight (weight x SD), and ``run_elasticity_lhs`` sweeps the
weights on both the raw and the standardized scale. The two non-spatial
criteria get fixed shares instead: section compliance keeps 0.16 as a soft
structural constraint (the same size as one spatial share), and entrance
proximity is down-weighted to 0.04 (tie-breaker, largely redundant with
traffic + flow).

Elasticities (layout score -> conversion / impulse / basket-size lift) are
bounded ranges, not point estimates: each band is [BASE, BASE + GAIN_MAX].
The GA fitness uses each band's midpoint, and run_elasticity_lhs sweeps the
whole box. They act on the score DIFFERENCE from an anchor layout -- the
store as built, after the shared feasibility repair -- so the anchor layout
reproduces the calibrated conversion, basket and spend exactly and a layout
only moves them by how much better or worse it scores (``layout_objective``).

Penalties (overlap, bottleneck) are deliberately on a stricter scale than
the positive criteria: one overlapping pair (-0.80) outweighs almost any
positive-score gain, so a colliding layout is strongly dominated -- a soft
penalty, not a hard lexicographic constraint. Feasibility itself is
guaranteed downstream by the repair steps (GUI _repair_item_overlaps after
apply; headless _repair_chrom_overlaps before MC scoring).
"""

from __future__ import annotations

import math
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
                    "(Joint path-and-purchase model of grocery trips; cited for "
                    "the direction of the conversion driver only.)"),

    "hui2013":     ("Hui, S.K., Inman, J.J., Huang, Y., Suher, J. (2013). "
                    "The effect of in-store travel distance on unplanned spending: "
                    "applications to mobile promotion strategies. "
                    "Journal of Marketing, 77(2), 1-16. "
                    "(Path-tracking field study: unplanned spending rises with "
                    "the distance a shopper walks in the store. Cited for the "
                    "direction of the basket and impulse drivers only.)"),

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

    "fenton1960":  ("Fenton, L. (1960). The sum of log-normal probability "
                    "distributions in scatter transmission systems. IRE "
                    "Transactions on Communications Systems, 8(1), 57-67. "
                    "(Approximates a sum of lognormal variables by the "
                    "lognormal with the sum's mean and variance; the Monte "
                    "Carlo engine's day-spend law, MC_SPEND_LAW.)"),
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
# Each elasticity is a "fractional change in the parameter per unit change
# in the corresponding 0..1 score", applied to the change from the anchor
# layout: p = p_calibrated * (1 + (s - s_anchor) * e). Each band is
# [*_BASE, *_BASE + *_GAIN_MAX]; the GA fitness uses the midpoint and
# run_elasticity_lhs Latin-hypercube-samples the whole box, so the paper can
# show how far the headline lift moves inside the literature band.
#
# The elasticities are NOT a common factor that cancels between methods.
# The sign of a lift between two layouts is, to first order, the sign of
# their score difference (conversion and basket both rise with the score),
# but its size scales with e_conv + e_bsk, and the score channel trades off
# against the impulse and abandonment channels, which read sub-scores. The
# LHS runner reports the per-pair counts and magnitude ranges behind that.

# Conversion. ASSUMPTION. Hui 2009 give the direction -- purchase
# probability falls with the detour a category demands -- and no size.
# The band, 20-50% (BASE 0.20, width GAIN_MAX 0.30), its linearity in the
# score and its aggregation from category zones to one store-level
# conversion multiplier (a composition Hui does not make) are ours. Every
# path -- the GUI pipeline, What-If and every headless experiment -- uses
# the midpoint, 0.35; every lift scales with this choice (see the note
# above), so the LHS sweep spans the band.
ELASTICITY_CONV_BASE     = 0.20    # band floor
ELASTICITY_CONV_GAIN_MAX = 0.30    # band width above BASE: fitness uses BASE + 0.5*GAIN_MAX, LHS samples the band
ELASTICITY_CONV_CITE     = "hui2009"

# Impulse. ASSUMPTION. Hui, Inman 2013 give the direction -- unplanned
# spending rises with what a shopper's path passes -- and placing impulse
# items where every paying shopper passes, at the checkout, is our
# application of it. The band, 30-70% (BASE 0.30, width GAIN_MAX 0.40;
# every path uses the midpoint, 0.50, and the LHS sweep spans the band), is
# ours; its floor is low because a layout moves only part of an assortment
# into checkout adjacency.
ELASTICITY_IMP_BASE      = 0.30    # band floor
ELASTICITY_IMP_GAIN_MAX  = 0.40    # band width above BASE: fitness uses BASE + 0.5*GAIN_MAX, LHS samples the band
ELASTICITY_IMP_CITE     = "hui2013"

# Basket size. ASSUMPTION, and the least grounded of the three. Hui,
# Inman 2013 give the direction -- unplanned spending rises with the
# distance walked -- and no size this module uses. The 10-30% band rests on
# planned-list completion (a better-organized layout helps shoppers find
# and complete planned purchases) -- a mechanism with face validity but NO
# quantified citation. The paper names this the weakest coefficient. Every
# lift scales with it, so the LHS sweep spans the band and adds two corners
# below it: 0.02 (a fifth of the band's floor) and 0 (no basket effect).
#
# A caveat on direction: the basket driver reads the whole composite, which
# includes flow efficiency and entrance proximity -- criteria that reward
# SHORTER paths -- while the cited mechanism is 'more walking, more
# unplanned spending'. run_elasticity_lhs reports a variant with those two
# criteria taken out of the basket driver only.
ELASTICITY_BSK_BASE      = 0.10    # band floor
ELASTICITY_BSK_GAIN_MAX  = 0.20    # band width above BASE: fitness uses BASE + 0.5*GAIN_MAX, LHS samples the band
ELASTICITY_BSK_CITE      = "hui2013"

# Abandonment recovery: flow + section + bottleneck additively change
# abandonment, relative to the anchor layout (d = layout minus anchor):
#   a(s) = a0 * max(FLOOR, 1 - dflow*FLOW - dsection*SECTION
#                          + dbottleneck*BOTTLENECK)
# TUNED, not cited: the coefficients were set so that a full-unit gain in
# flow and in section compliance removes 60% of the baseline abandonment,
# and a full-unit rise in the bottleneck penalty inflates it by 50%. The
# cited works give only the DIRECTION of each effect -- Larson 2005 (flow),
# Ozgormus & Smith 2020 (department/section organization), Hui 2013
# (congestion) -- not these sizes.
ABANDON_FLOW_COEF        = 0.40    # TUNED
ABANDON_SECTION_COEF     = 0.20    # TUNED
ABANDON_BOTTLENECK_COEF  = 0.50    # TUNED
ABANDON_FLOOR_FRAC       = 0.30    # TUNED: never recover more than 70% of abandonment


# --- clamps on the transformed drivers (NUMERICAL GUARDS) ---
# The score -> driver transform can in principle push a rate outside the
# range the Monte Carlo engine accepts; these keep it inside. They bind
# only at extreme scores or base rates (a conversion above 0.99, say).
CONV_CLAMP_LO            = 0.01    # conversion after the score lift and abandonment
CONV_CLAMP_HI            = 0.99
IMPULSE_RATE_CLAMP_HI    = 0.99    # impulse share of converters
ABANDON_RATE_CLAMP_HI    = 0.50    # layout-adjusted abandonment rate
ABANDON_BASE_CAP         = 0.95    # baseline abandonment in the relative-abandonment denominator
# Net base revenue per converter (gross minus the expected impulse spend,
# ``sim_calibration.net_base_revenue``) is floored at this share of gross.
NET_BASE_REVENUE_MIN_FRAC = 0.05


# --- the Monte Carlo engine's own guards (NUMERICAL GUARDS) ---
# ``sim_calibration.mc_engine`` guards its inputs once more before it draws,
# and the closed form of its mean (``experiments.closed_form``) must apply
# the same guards to converge to the same value, so both read them here.
# The conversion clamp is wider than CONV_CLAMP_* above: it is the last
# guard before the binomial draw, and the parameter extraction
# (``sim_calibration.extract_simulation_parameters``) applies it to the
# conversion it hands the objective. None of them binds at any rate the
# runners use.
MC_LAMBDA_FLOOR          = 0.1     # floor on a day's expected arrivals
MC_LAMBDA_CAP            = 1.0e6   # cap on a day's expected arrivals
MC_CONV_CLAMP_LO         = 0.001   # conversion probability the engine draws with
MC_CONV_CLAMP_HI         = 0.999
MC_SD_FLOOR              = 0.01    # floor on a spend SD (per converter, per impulse buy)


# --- the Monte Carlo engine's day-spend law (OPERATIONAL ASSUMPTION) ---
# A day's spend over its C converters is drawn as ONE total, lognormal with
# the exact mean C * mu and SD sqrt(C) * sigma of a sum of C independent
# spends of mean mu and SD sigma (Fenton 1960's moment matching;
# ``sim_calibration.lognormal_spend_total``); impulse spend likewise. Only
# the first two moments of per-converter spend are calibrated, so the law's
# shape beyond them is an assumption; it is non-negative and keeps the mean
# exactly, so the closed form (``experiments.closed_form``) needs no floor
# term. It replaced a normal N(C mu, sqrt(C) sigma) floored at zero, whose
# floor raised the mean on thin, heavy-tailed days -- Online Retail II's
# invoices have a coefficient of variation above 2 -- and whose lower tail
# made a day's spend fall as a converter was added. Recorded in every
# sidecar (``experiments._common.OBJECTIVE_CONSTANTS``), so artifacts made
# with the floored law can be told apart.
MC_SPEND_LAW             = 'lognormal_moment_matched'
MC_SPEND_LAW_CITE        = 'fenton1960'


# --- checkout-queue channel of the objective (OPERATIONAL ASSUMPTION) ---
# A layout whose items sit in congested cells lengthens the checkout queue
# and loses some spend to it:
#   queue factor  q(s) = 1 + (bottleneck - bottleneck_anchor) * QUEUE_BOTTLENECK_FACTOR
#   spend factor      = max(QUEUE_PENALTY_FLOOR, 1 - (q - 1) * QUEUE_PENALTY_SLOPE)
# Nothing cited sizes either coefficient. The bottleneck criterion is fed by
# congestion the live simulation records; the headless runners record
# none, so on every synthetic and calibrated store this channel is off.
QUEUE_BOTTLENECK_FACTOR  = 0.30
QUEUE_PENALTY_SLOPE      = 0.15
QUEUE_PENALTY_FLOOR      = 0.85
DEFAULT_QUEUE_TIME_S     = 5.0     # STAND-IN baseline queue time; only q's ratio reaches revenue


# --- stand-in inputs where the data observe nothing ---
# The headless runners and the GUI's parameter extraction need a few inputs
# no dataset in the paper measures. Each is a STAND-IN (or an explicit
# ASSUMPTION) and is recorded, with every other base parameter, in the
# sidecar of each runner that uses it.
#
# Conversion: transaction logs record buyers only, so the share of visitors
# who buy is unobservable. ASSUMPTION, flagged as such wherever a
# calibration records it (``conversion_rate_source``).
ASSUMED_CONVERSION_RATE      = 0.30
# Impulse share of converting customers and the value of one impulse
# purchase. STAND-IN: UCI Online Retail II does not label impulse buys and
# its stores carry no impulse fixtures, so these only set the size of the
# additive impulse term; the synthetic scenarios use the impulse items'
# own mean price as the value when they have impulse items.
IMPULSE_RATE_STANDIN         = 0.20
IMPULSE_VALUE_FRAC_OF_GROSS  = 0.15    # impulse value = 0.15 x gross spend per converter
# Spread of one impulse purchase's value, as a fraction of its mean.
# STAND-IN (the Monte Carlo engine's default; affects only noise).
IMPULSE_VALUE_CV             = 0.30
# Spread of per-converter spend where no invoice data supply it (synthetic
# scenarios; the GUI before any purchase is observed). STAND-IN; it moves
# only the Monte Carlo noise, not the mean (the day-spend law keeps the mean
# exactly, MC_SPEND_LAW).
REV_STD_FRAC_OF_MEAN         = 0.35
# Baseline cart-abandonment rate. STAND-IN on the synthetic scenarios; on a
# calibrated store ASSUMPTION: 10% of the non-converting visitors.
ABANDON_RATE_SYNTHETIC       = 0.05
ABANDON_FRAC_OF_NONCONVERTERS = 0.10
# Basket size where no basket data exist (the synthetic scenarios; the GUI
# before any purchase). STAND-IN: basket size reaches revenue only as the
# ratio b(s)/b0, so its level cancels.
STANDIN_AVG_BASKET           = 3.0
STANDIN_STD_BASKET           = 1.0


# --- cold-start traffic priors (OPERATIONAL ASSUMPTION) ---
# The traffic and revenue-placement criteria read a heat map. The live
# simulator builds one from agent paths; the headless runners paint a
# prior instead, as a sum of inverse-distance terms, heat = c / (1 + d):
#   synthetic scenarios   5/(1 + d_entrance) + 3/(1 + d_checkout)
#   calibrated store      3/(1 + d_entrance) + 2/(1 + d_checkout)
#                         + 1.5/(1 + max(d_wall, 0.1))
# The shapes follow Sorensen 2009 (front-of-store hot zone) and, for the
# wall term, Larson 2005 (perimeter-dominant traffic); the coefficients are
# ours and nothing cited measures them. Note the asymmetry: only the
# calibrated prior rewards wall proximity, while the synthetic scenarios'
# analytical objective does (``synthetic_shops.analytical_revenue``);
# ``experiments/run_objective_alignment.py`` measures what aligning them
# changes.
HEAT_PRIOR_SYNTH_ENTRANCE    = 5.0
HEAT_PRIOR_SYNTH_CHECKOUT    = 3.0
HEAT_PRIOR_CAL_ENTRANCE      = 3.0
HEAT_PRIOR_CAL_CHECKOUT      = 2.0
HEAT_PRIOR_CAL_WALL          = 1.5
HEAT_PRIOR_WALL_MIN_D        = 0.1     # NUMERICAL GUARD on d_wall (m)


# --- the synthetic scenarios' analytical objective (synthetic_shops) ---
# The analytical revenue behind Figure A's regret and Spearman, the LHS
# runner's 'oracle' comparator and run_objective_alignment reads four
# per-item coefficients, each drawn uniformly from a band when a scenario is
# generated (``synthetic_shops.generate_synthetic_shop``); the *_DEFAULT
# values are what a hand-built SyntheticItem gets. They are the synthetic
# scenarios' ground truth, not simulator coefficients: the Monte Carlo
# objective reads none of them. Each term acts on a distance normalised by
# the store (``synthetic_shops.analytical_revenue``), so no band is a
# per-metre figure and none is read off a cited table.
#
# Detour from the entrance: an item's revenue falls by e * d_entrance /
# diagonal. OPERATIONAL ASSUMPTION: the direction is Hui 2009's (a detour
# lowers purchase probability); the band, 0.40 +- 25%, is ours.
SYNTH_DETOUR_ELASTICITY_RANGE      = (0.30, 0.50)
SYNTH_DETOUR_ELASTICITY_DEFAULT    = 0.40
# Impulse items near the checkout: revenue rises by e * (1 - d_checkout /
# diagonal). OPERATIONAL ASSUMPTION: the direction is Hui, Inman 2013's
# (unplanned spending rises with what a shopper passes), as with
# ELASTICITY_IMP_BASE; the band is ours.
SYNTH_IMPULSE_ELASTICITY_RANGE     = (0.40, 0.70)
SYNTH_IMPULSE_ELASTICITY_DEFAULT   = 0.50
# Perimeter proximity: revenue rises by e * (1 - 2 d_wall / min(W, H)),
# floored at zero. OPERATIONAL ASSUMPTION: the direction is Larson 2005's
# (perimeter-dominant traffic); the size is ours. This is the wall term the
# synthetic Monte Carlo traffic prior lacks (heat priors above).
SYNTH_PERIMETER_ELASTICITY_RANGE   = (0.15, 0.30)
SYNTH_PERIMETER_ELASTICITY_DEFAULT = 0.20
# Cross-merchandising: each affinity pair adds a * (1 - d_ij / diagonal) to
# both items' revenue, and this share of item pairs carries one.
# OPERATIONAL ASSUMPTION: 'modest' affinity; the direction is Hui 2009's
# (co-purchased items gain from proximity), nothing cited sizes it.
SYNTH_AFFINITY_RANGE               = (0.10, 0.30)
SYNTH_AFFINITY_DENSITY             = 0.15
# Visitors per day of a synthetic scenario. STAND-IN: it sets the revenue
# level of every synthetic result (analytical and Monte Carlo alike).
SYNTH_DAILY_CUSTOMERS              = 200.0


# --- impulse sub-model (live agent) ---
# These used to sit inline in customer.py, which quietly broke the
# "every cited coefficient lives here" rule the paper leans on (audit
# R36). Values are unchanged from the inline originals.
#
# The catchment radius, the per-type propensities and the two situational
# bonuses are operational defaults, not cited figures: within 3 m of a lane
# counts as checkout-adjacent, and an agent that is browsing, or that has
# already been in the store a while, is more impulse-prone.

IMPULSE_CHECKOUT_RADIUS_M = 3.0      # operational: the checkout-adjacent band
IMPULSE_DECAY_PER_ITEM    = 0.6      # each extra impulse item is 0.6x as likely
IMPULSE_BASE_PROPENSITY   = {        # by customer type (operational)
    'quick':    0.15,
    'browser':  0.35,
    'thorough': 0.25,
}
IMPULSE_BROWSER_BONUS     = 0.10     # added for 'browser' agents
IMPULSE_LONG_DWELL_BONUS  = 0.15     # added after this many seconds in store
IMPULSE_LONG_DWELL_SECS   = 120.0


# --- basket composition (no constants) ---
# On a calibrated shop an agent's shopping list is the stocked part of ONE
# empirical invoice, drawn uniformly from the invoices holding at least one
# of the shop's regular items (``Customer._draw_invoice``; the invoices are
# stored by ``CalibratedParams.seed_into``). The invoice is what the data
# observe: its size, its contents, its category mix and its co-purchases
# come jointly, so the draw carries all four to the agents and leaves no
# coefficient to set. Co-locating two products that real invoices hold
# together therefore changes agent behaviour through the data alone.
#
# The earlier law -- popularity-weighted items plus a fixed chance of
# pulling a co-purchase partner of an item already chosen -- had two
# free constants (pull probability, popularity smoothing) that nothing
# cited fixed, and it counted each co-purchase twice: a product's
# popularity already counts every invoice it shares with its partners.
#
# Without a calibration the list is a uniform draw over the regular items,
# its length set by LIST_LENGTH_BY_TYPE below.


# --- shopping-list length (uncalibrated shops only) ---
# How many regular items an agent sets out for, by customer type: an
# inclusive (min, max) range, drawn uniformly. This is an OPERATIONAL
# ASSUMPTION carried over from the original simulator, not a literature
# value -- nothing cited in this module measures list length per shopper
# type, and a transaction log cannot label its buyers 'quick' or
# 'thorough' either.
#
# It applies only while no calibration supplies stored invoices. A
# transactional calibration seeded onto its own shop replaces it: each
# agent's list is then one invoice's stocked part, so its length follows
# the distribution of distinct STOCKED products per invoice
# (``analytics['calibration']['list_length_sample']``), which the data do
# observe, and no longer depends on type.

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


# --- live agent behaviour (OPERATIONAL ASSUMPTIONS unless noted) ---
# These used to sit inline in customer.py and simulation.py. Values are
# unchanged, and the agents draw them in the same calls and the same order
# as before, so every seeded live run reproduces bit for bit. None is
# calibrated from the paper's datasets: the live store's walking speed and
# dwell times are not fitted to UCI (which has no paths) and only the
# walking speed is replaced by the ETH distribution when a trajectory
# calibration is loaded (``simulation.py``, spawn). They shape the live
# diagnostics and the Validation tab, not the Monte Carlo objective, which
# runs no agents.
#
# Walking speed, uniform over this range (m/s). The range brackets typical
# free walking speeds; in-store browsing is slower, which is why a
# trajectory calibration overrides it.
AGENT_SPEED_RANGE_MPS         = (0.8, 1.5)
# Guard band applied to a speed drawn from a trajectory calibration's
# normal fit, and the SD used when the calibration records none.
AGENT_SPEED_CLIP_MPS          = (0.3, 2.5)   # NUMERICAL GUARD
AGENT_SPEED_FALLBACK_SD       = 0.2
# Cart abandonment: each agent's probability is uniform over this range and
# decided once, at spawn (so it does not depend on the layout).
AGENT_ABANDON_PROB_RANGE      = (0.01, 0.05)
# Seconds an agent spends in the 'entering' state before it starts moving.
AGENT_ENTERING_TIME_RANGE_S   = (0.25, 0.5)
# Upper bound on impulse items per agent: numpy randint(lo, hi), so 1 to 3.
AGENT_MAX_IMPULSE_ITEMS_RANGE = (1, 4)
# Recorded per agent for the analytics; no decision reads it.
AGENT_PURCHASE_INTENT_RANGE   = (0.7, 0.95)
# Customer-type mix. Type sets the impulse propensity and the restroom
# probability (and the list length on an uncalibrated shop).
CUSTOMER_TYPES                = ('quick', 'browser', 'thorough')
CUSTOMER_TYPE_PROBS           = (0.3, 0.4, 0.3)
# Loyalty mix; it only weights the customer-lifetime-value report.
LOYALTY_LEVELS                = ('new', 'regular', 'vip')
LOYALTY_PROBS                 = (0.5, 0.35, 0.15)
# Probability an agent visits the restroom, by type. With the type mix
# above, 33.5% of agents do (0.3*0.15 + 0.4*0.35 + 0.3*0.50).
WC_PROBABILITY_BY_TYPE        = {'quick': 0.15, 'browser': 0.35,
                                 'thorough': 0.50}
# Dwell times (s), uniform over each range: at a shelf fixture (also the
# pause of an agent that finds no checkout lane), at an impulse display,
# and in the restroom.
DWELL_ITEM_RANGE_S            = (5, 15)
DWELL_IMPULSE_RANGE_S         = (2, 5)
DWELL_WC_RANGE_S              = (2, 8)
# Anti-stacking separation: two agents closer than SEPARATION_MIN_DIST_M
# are each pushed apart by a fraction of their overlap. The push is a RATE,
# not a per-tick step: in a tick of dt simulated seconds each agent moves by
# (1 - exp(-SEPARATION_RATE_PER_S * dt)) of the overlap. A fixed fraction
# per tick would act dt-times more often per second at a finer tick -- at
# 0.02 s a head-on pair of slow walkers then locks at a stable gap instead
# of passing -- so the tick length would be part of the model (review R41).
# SEPARATION_STRENGTH is the fraction at the reference tick
# SEPARATION_REFERENCE_DT_S, the value the per-tick rule always had, and
# the rate is defined from it so that tick reproduces the old push exactly:
# every fixed-step run at dt 0.04 is unchanged. OPERATIONAL ASSUMPTION:
# nothing cited sizes either constant. experiments/
# run_structural_sensitivity.py sweeps the strength (as the fraction at
# the reference tick) and finds no effect on revenue or throughput.
SEPARATION_STRENGTH           = 0.08   # displacement fraction at the reference tick
SEPARATION_REFERENCE_DT_S     = 0.04   # the tick SEPARATION_STRENGTH is quoted at (s)
SEPARATION_RATE_PER_S         = (-math.log1p(-SEPARATION_STRENGTH)
                                 / SEPARATION_REFERENCE_DT_S)   # ~2.0845 / s
SEPARATION_MIN_DIST_M         = 0.35
# An agent counts as having reached a path waypoint or target within this
# distance (m).
WAYPOINT_RADIUS_M             = 0.5


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
