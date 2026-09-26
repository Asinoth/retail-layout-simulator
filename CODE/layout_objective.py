"""The layout objective: how a layout's score becomes revenue drivers.

One definition, read by every path that turns a layout into money:

  * ``viz_ga_run._ga_fitness`` -- the GA tab and, through
    ``experiments._common.run_ga_headless`` / ``paired_mc_revenue``, every
    headless runner and search comparator;
  * ``viz_optimize._opt_layout_drivers`` -- the Optimize pipeline's
    fitness, its Monte Carlo projection and its re-scored comparison after
    the POST window, and the What-If tab, all at the band midpoints;
  * ``experiments.closed_form.expected_revenue`` -- the exact mean of the
    Monte Carlo fitness, which the elasticity sweep and the re-scoring of
    final layouts use.

Anchoring. The calibrated inputs (conversion p_c, basket b0, impulse rate
p_imp, abandonment a0) describe the store as it stands. A layout's score
therefore moves them by its DIFFERENCE from the score of that store, the
anchor layout, never by its absolute score:

    conversion   p_c * (1 + (s - s0) * e_conv)
    basket       b0  * (1 + (s - s0) * e_bsk)
    impulse      p_imp * (1 + (c_imp - c_imp0) * e_imp)
    abandonment  a0 * max(FLOOR, 1 - (flow - flow0) * FLOW
                                 - (section - section0) * SECTION
                                 + (bottleneck - bottleneck0) * BOTTLENECK)
    queue        1 + (bottleneck - bottleneck0) * QUEUE_BOTTLENECK_FACTOR

so the anchor layout reproduces the calibrated conversion, basket, spend
and daily volume exactly, and a criterion that is constant over the layouts
compared (section compliance on repaired layouts, for one) cancels instead
of shifting every layout's level. With the old absolute form every real
layout, the baseline included, was lifted about a third above the
calibrated level, i.e. the calibration described a store scoring zero.

The anchor lives in ``base_params['score_anchor']`` as
``{'score': s0, 'breakdown': {...}, 'source': ...}``. The headless runners
set it to the store's as-built layout after the shared feasibility repair
(``experiments._common.anchor_base_params``); the GUI sets it to the layout
on the floor when an optimization or projection starts. Because the
heat-map and bottleneck criteria read live analytics, an anchor holds only
while those analytics do: the Optimize pipeline's re-scored comparison,
made after the live POST window, re-scores the PRE layout together with
both arms and anchors them there (``viz_optimize._ab_arm_scores``).

There is no silent default. Parameters without an anchor are refused
(``MissingAnchorError``) by ``layout_drivers`` and therefore by every path
above: re-scoring a layout from parameters rebuilt by
``base_params_for*``, which carry none, would otherwise bring back the
absolute form and its one-third lift without any error. The zero anchor,
the old convention, is an explicit opt-in for the callers that genuinely
have no as-built layout (unit tests of the transform, mostly): put
``zero_anchor()`` in the parameters (``with_zero_anchor``), pass it as
``anchor``, or pass ``allow_zero_anchor=True``.

Clamps are those of retail_literature and act on the transformed values.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional

from retail_literature import (
    ABANDON_BASE_CAP, ABANDON_BOTTLENECK_COEF, ABANDON_FLOOR_FRAC,
    ABANDON_FLOW_COEF, ABANDON_RATE_CLAMP_HI, ABANDON_SECTION_COEF,
    CONV_CLAMP_HI, CONV_CLAMP_LO, DEFAULT_OP_HOURS_PER_DAY,
    DEFAULT_QUEUE_TIME_S, DEFAULT_WEEKEND_MULTIPLIER,
    ELASTICITY_BSK_BASE, ELASTICITY_BSK_GAIN_MAX,
    ELASTICITY_CONV_BASE, ELASTICITY_CONV_GAIN_MAX,
    ELASTICITY_IMP_BASE, ELASTICITY_IMP_GAIN_MAX,
    GA_PEN_BOTTLENECK, GA_PEN_OVERLAP, GA_W_ACCESSIBILITY, GA_W_CROSS_MERCH,
    GA_W_FLOW, GA_W_IMPULSE, GA_W_REVENUE_PLACEMENT, GA_W_SECTION_COMPLIANCE,
    GA_W_TRAFFIC, IMPULSE_RATE_CLAMP_HI, QUEUE_BOTTLENECK_FACTOR,
    QUEUE_PENALTY_FLOOR, QUEUE_PENALTY_SLOPE,
)

#: Where ``base_params`` carries the anchor.
SCORE_ANCHOR_KEY = 'score_anchor'

#: Each elasticity band as [BASE, BASE + GAIN_MAX] (retail_literature).
ELASTICITY_BANDS = {
    'conv': (ELASTICITY_CONV_BASE, ELASTICITY_CONV_BASE + ELASTICITY_CONV_GAIN_MAX),
    'imp':  (ELASTICITY_IMP_BASE,  ELASTICITY_IMP_BASE + ELASTICITY_IMP_GAIN_MAX),
    'bsk':  (ELASTICITY_BSK_BASE,  ELASTICITY_BSK_BASE + ELASTICITY_BSK_GAIN_MAX),
}

#: The band midpoints: the elasticities the GA fitness searches under.
#: Written as BASE + 0.5 * GAIN_MAX, the form the fitness always used, so
#: the values are the same floats.
ELASTICITY_MIDPOINTS = {
    'conv': ELASTICITY_CONV_BASE + 0.5 * ELASTICITY_CONV_GAIN_MAX,
    'imp':  ELASTICITY_IMP_BASE + 0.5 * ELASTICITY_IMP_GAIN_MAX,
    'bsk':  ELASTICITY_BSK_BASE + 0.5 * ELASTICITY_BSK_GAIN_MAX,
}

#: Composite weight of each criterion, signed as it enters the score.
CRITERION_WEIGHTS = {
    'traffic':            GA_W_TRAFFIC,
    'cross_merch':        GA_W_CROSS_MERCH,
    'impulse':            GA_W_IMPULSE,
    'flow':               GA_W_FLOW,
    'revenue_placement':  GA_W_REVENUE_PLACEMENT,
    'section_compliance': GA_W_SECTION_COMPLIANCE,
    'accessibility':      GA_W_ACCESSIBILITY,
    'overlap_penalty':    -GA_PEN_OVERLAP,
    'bottleneck_penalty': -GA_PEN_BOTTLENECK,
}

#: The criteria that reward SHORTER paths (flow efficiency, entrance
#: proximity). The basket elasticity is anchored on 'more walking, more
#: unplanned spending', so a sensitivity variant takes these out of the
#: basket driver (``basket_exclude``).
SHORT_PATH_CRITERIA = ('flow', 'accessibility')

#: ``source`` of the explicit zero anchor.
ZERO_ANCHOR_SOURCE = 'zero'


class MissingAnchorError(ValueError):
    """Base parameters reached the objective without a score anchor."""


def zero_anchor() -> Dict[str, Any]:
    """The zero anchor, as an explicit record: the elasticities then act on
    the absolute score, i.e. the calibration is taken to describe a store
    scoring zero. Only for callers with no as-built layout to anchor at."""
    return {'score': 0.0, 'breakdown': {}, 'source': ZERO_ANCHOR_SOURCE}


def with_zero_anchor(base_params: Mapping[str, Any]) -> Dict[str, Any]:
    """A copy of ``base_params`` carrying the explicit zero anchor."""
    out = dict(base_params)
    out[SCORE_ANCHOR_KEY] = zero_anchor()
    return out


def require_anchor(base_params: Mapping[str, Any],
                   where: str = 'the layout objective') -> None:
    """Raise ``MissingAnchorError`` unless ``base_params`` carries an
    anchor. The entry points that score many layouts (the headless GA, the
    paired Monte Carlo evaluator) call it up front, so a missing anchor
    fails before any work rather than on the first evaluation."""
    if base_params.get(SCORE_ANCHOR_KEY) is None:
        raise MissingAnchorError(
            f'{where} got base parameters without {SCORE_ANCHOR_KEY!r}. '
            f'Anchor them at the store\'s as-built layout '
            f'(experiments._common.anchor_base_params, or the GUI\'s '
            f'_ga_score_anchor); only a caller with no as-built layout may '
            f'opt into the zero anchor (layout_objective.with_zero_anchor, '
            f'or allow_zero_anchor=True).')


def make_anchor(score: float, breakdown: Mapping[str, Any],
                source: Optional[str] = None) -> Dict[str, Any]:
    """The anchor record for a layout scored ``(score, breakdown)``.

    Only numeric breakdown entries are kept (plain floats, so the record
    goes through JSON into the sidecars unchanged)."""
    return {'score': float(score),
            'breakdown': {k: float(v) for k, v in breakdown.items()
                          if isinstance(v, (int, float))},
            'source': source}


def anchor_of(base_params: Mapping[str, Any],
              allow_zero_anchor: bool = False) -> Dict[str, Any]:
    """The anchor ``base_params`` carries. Without one, the zero anchor if
    ``allow_zero_anchor``, else ``MissingAnchorError``."""
    anchor = base_params.get(SCORE_ANCHOR_KEY)
    if anchor is None:
        if not allow_zero_anchor:
            require_anchor(base_params)      # raises
        return zero_anchor()
    return anchor


def with_anchor(base_params: Mapping[str, Any], score: float,
                breakdown: Mapping[str, Any],
                source: Optional[str] = None) -> Dict[str, Any]:
    """A copy of ``base_params`` anchored at the layout scored
    ``(score, breakdown)``. The input is not modified."""
    out = dict(base_params)
    out[SCORE_ANCHOR_KEY] = make_anchor(score, breakdown, source)
    return out


def basket_score(score: float, breakdown: Mapping[str, Any],
                 exclude: Iterable[str] = ()) -> float:
    """The composite with the criteria in ``exclude`` taken out, at their
    composite weights: what the basket driver reads in the
    ``basket_exclude`` sensitivity variant."""
    return float(score) - sum(CRITERION_WEIGHTS[k] * float(breakdown.get(k, 0.0))
                              for k in exclude)


def _clamp(v: float, lo: float, hi: float) -> float:
    return min(max(v, lo), hi)


def layout_drivers(score: float, breakdown: Mapping[str, Any],
                   base_params: Mapping[str, Any],
                   elasticities: Optional[Mapping[str, float]] = None,
                   anchor: Optional[Mapping[str, Any]] = None,
                   basket_exclude: Iterable[str] = (),
                   allow_zero_anchor: bool = False) -> Dict[str, float]:
    """Revenue drivers of a layout scored ``(score, breakdown)``.

    ``elasticities`` overrides any of ``'conv'``, ``'imp'``, ``'bsk'``
    (default: the band midpoints). ``anchor`` overrides the one in
    ``base_params`` -- the weight sweeps pass the anchor layout re-scored
    under the same weights as the layout. With neither, the call raises
    ``MissingAnchorError`` unless ``allow_zero_anchor`` opts into the zero
    anchor. ``basket_exclude`` names criteria to take out of the basket
    driver only (both the layout's and the anchor's), e.g.
    ``SHORT_PATH_CRITERIA``.

    Returns the conversion after the layout-induced abandonment change
    (``conv``), the impulse rate (``imp_rate``), the basket size
    (``avg_bsk``), the basket ratio b(s)/b0 (``bsk_mult``), the queue spend
    factor (``queue_penalty``) and their product (``rev_mult``), which
    scales both the mean and the SD of spend per converter, plus the
    intermediate values and the score differences the drivers read.
    """
    e = dict(ELASTICITY_MIDPOINTS)
    if elasticities:
        e.update(elasticities)
    anc = (anchor if anchor is not None
           else anchor_of(base_params, allow_zero_anchor=allow_zero_anchor))
    s0 = float(anc['score'])
    b0 = anc.get('breakdown') or {}

    def delta(key):
        return float(breakdown.get(key, 0.0)) - float(b0.get(key, 0.0))

    ds = float(score) - s0
    exclude = tuple(basket_exclude)
    ds_bsk = (basket_score(score, breakdown, exclude)
              - basket_score(s0, b0, exclude)) if exclude else ds

    conv_base = base_params['conversion_rate']
    conv_lifted = _clamp(conv_base * (1.0 + ds * e['conv']),
                         CONV_CLAMP_LO, CONV_CLAMP_HI)

    imp_rate = _clamp(base_params['impulse_rate']
                      * (1.0 + delta('impulse') * e['imp']),
                      0.0, IMPULSE_RATE_CLAMP_HI)

    bsk_base = base_params['avg_basket_size']
    avg_bsk = bsk_base * (1.0 + ds_bsk * e['bsk'])

    # Abandonment enters as a RELATIVE factor against the baseline rate:
    # the calibrated conversion is already net of baseline abandonment
    # (completed / total), so only the layout-induced CHANGE may move it.
    abandon_base = base_params.get('abandonment_rate', 0.0)
    abandon = abandon_base * max(
        ABANDON_FLOOR_FRAC,
        1.0 - delta('flow') * ABANDON_FLOW_COEF
            - delta('section_compliance') * ABANDON_SECTION_COEF
            + delta('bottleneck_penalty') * ABANDON_BOTTLENECK_COEF)
    abandon = _clamp(abandon, 0.0, ABANDON_RATE_CLAMP_HI)
    conv = conv_lifted * (1.0 - abandon) \
        / max(1.0 - min(abandon_base, ABANDON_BASE_CAP), 1e-6)
    conv = _clamp(conv, CONV_CLAMP_LO, CONV_CLAMP_HI)

    queue_base = base_params.get('avg_queue_time', DEFAULT_QUEUE_TIME_S)
    queue_adj = queue_base * (1.0 + delta('bottleneck_penalty')
                              * QUEUE_BOTTLENECK_FACTOR)
    queue_penalty = 1.0
    if queue_adj > 0:
        queue_penalty = max(QUEUE_PENALTY_FLOOR,
                            1.0 - (queue_adj - queue_base)
                            / max(queue_base, 1e-6) * QUEUE_PENALTY_SLOPE)

    # Basket size reaches revenue multiplicatively: more items at the same
    # average item price is proportionally more spend per converter.
    bsk_mult = avg_bsk / max(bsk_base, 1e-9)
    return {
        'conv': conv,
        'conv_lifted': conv_lifted,
        'abandonment': abandon,
        'imp_rate': imp_rate,
        'avg_bsk': avg_bsk,
        'bsk_mult': bsk_mult,
        'queue_penalty': queue_penalty,
        'rev_mult': bsk_mult * queue_penalty,
        'score_delta': ds,
        'basket_score_delta': ds_bsk,
        'anchor_score': s0,
    }


def layout_mc_kwargs(drivers: Mapping[str, float],
                     base_params: Mapping[str, Any],
                     n_days: int, n_iter: int) -> Dict[str, Any]:
    """``mc_engine`` keyword arguments for a layout with ``drivers``.

    Spend per converter is the net base revenue times ``rev_mult``, and its
    SD is scaled by the same factor, so a basket or queue effect keeps the
    spend's coefficient of variation; the GUI and headless paths both go
    through here, so they treat ``rev_std`` the same way. The day is
    ``DEFAULT_OP_HOURS_PER_DAY`` long with the cited weekend multiplier."""
    rev_mult = drivers['rev_mult']
    return dict(
        cph=base_params['customers_per_hour'],
        conv=drivers['conv'],
        rev_mean=base_params['rev_per_converting_customer'] * rev_mult,
        rev_std=base_params['rev_std'] * rev_mult,
        imp_rate=drivers['imp_rate'],
        imp_val=base_params['avg_impulse_value'],
        avg_bsk=drivers['avg_bsk'],
        std_bsk=base_params['std_basket_size'],
        observed_baskets=base_params['basket_sizes_observed'],
        n_days=n_days, n_iter=n_iter,
        op_hours=DEFAULT_OP_HOURS_PER_DAY,
        wknd_mult=DEFAULT_WEEKEND_MULTIPLIER,
        monthly_growth=0.0,
        impulse_value_std=base_params.get('impulse_value_std'),
    )
