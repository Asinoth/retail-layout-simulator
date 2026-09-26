"""Closed-form re-scoring of Figure C's saved layouts (no search, no Monte
Carlo, no re-calibration of the UCI store).

Figure C's run records, for every layout it reports (the repaired as-built
``baseline``, the GA's ``optimized``, random search's ``rs`` and
annealing's ``sa``), the composite score and its per-criterion breakdown,
and its sidecar records the store's base parameters with their anchor.
That is all ``experiments.closed_form.expected_revenue`` needs, so the
questions below are answered exactly from the saved run, and the run's own
closed-form values are reproduced first (to ``REPRODUCE_RTOL``) to show the
model is the one the run used.

1. The data-derived conversion bound (review: the conversion-invariance
   claim). The UCI log has no non-buyers, so Figure C assumes a visit
   conversion (``assumed_conversion``, 0.30). The only data that bound it
   is the Omnichannel Retail bundle: ``calibrate_omnichannel`` records
   ``conversion_rate_lower_bound``, the largest single-family in-store
   purchase probability -- a visitor buys at least that family with that
   probability, so visit conversion is at least that. The bound is read
   through the adapter's own loader and calibration, with the bundle's
   SHA-256 (``dataset_provenance.stamp`` over the directory).
2. The lift at that bound: every saved layout's closed-form lift over the
   baseline with the store recalibrated to the bound exactly as Figure C's
   conversion scan recalibrates it (``run_real_data_example._at_conversion``:
   the observed buyers held fixed, the visitor rate buyers / p, baseline
   abandonment ``ABANDON_FRAC_OF_NONCONVERTERS`` of the 1 - p who do not
   buy), whether the 0.99 conversion clamps bind there, and the scan's own
   grid rates on either side of the bound with the lifts Figure C recorded
   at them.
3. Where the lift comes from, per layout:
   * the ABANDONMENT channel: the layout-induced change in abandonment
     (``layout_objective.layout_drivers``: the flow and section-compliance
     deltas times ``ABANDON_FLOW_COEF`` / ``ABANDON_SECTION_COEF``, both
     TUNED coefficients the elasticity bands do not cover). Switching it
     off is exact: with a zero baseline abandonment rate the abandonment
     factor is one for every layout, and the baseline -- the anchor layout
     -- keeps its revenue, because its factor was one already. The share
     it carries is 1 - (lift without it) / (lift with it);
   * the conversion and basket elasticities the same way (each set to
     zero), for scale -- the channels interact, so the shares need not add
     to one;
   * the composite-score gain over the baseline split by criterion: weight
     times the change in the criterion (the penalties with a minus), the
     weights retail_literature's GA_W_* / GA_PEN_*, checked to sum to the
     recorded score gain to ``RECOMPOSE_ATOL``; and the flow criterion's
     share of it.

    python -m experiments.run_figc_rescore
    python -m experiments.run_figc_rescore --figc-dir experiments/results/real_data_uci_20260926-111819

Writes ``figc_rescore_<timestamp>/summary.json`` and ``sidecar.json``.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time
from typing import Any, Dict, Mapping, Optional, Sequence

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import dataset_adapters as DA  # noqa: E402
import dataset_calibration as DC  # noqa: E402
import dataset_paths  # noqa: E402
from dataset_provenance import stamp as stamp_provenance  # noqa: E402
from experiments._common import (make_run_dir, portable_paths,  # noqa: E402
                                 provenance_snapshot, write_sidecar)
from experiments.closed_form import expected_revenue  # noqa: E402
from experiments.run_real_data_example import (_at_conversion,  # noqa: E402
                                               _clamps_bind)
from retail_literature import (ABANDON_FLOW_COEF,  # noqa: E402
                               ABANDON_SECTION_COEF, GA_PEN_BOTTLENECK,
                               GA_PEN_OVERLAP, GA_W_ACCESSIBILITY,
                               GA_W_CROSS_MERCH, GA_W_FLOW, GA_W_IMPULSE,
                               GA_W_REVENUE_PLACEMENT,
                               GA_W_SECTION_COMPLIANCE, GA_W_TRAFFIC)

#: Figure C's reported layouts: the baseline and the three searches'.
BASELINE = 'baseline'
SEARCHED = ('optimized', 'rs', 'sa')

#: Each criterion's signed weight in the composite score, in the order
#: ``_ga_compute_layout_score`` sums them.
CRITERION_WEIGHTS = (('traffic', GA_W_TRAFFIC),
                     ('cross_merch', GA_W_CROSS_MERCH),
                     ('impulse', GA_W_IMPULSE),
                     ('flow', GA_W_FLOW),
                     ('revenue_placement', GA_W_REVENUE_PLACEMENT),
                     ('section_compliance', GA_W_SECTION_COMPLIANCE),
                     ('accessibility', GA_W_ACCESSIBILITY),
                     ('overlap_penalty', -GA_PEN_OVERLAP),
                     ('bottleneck_penalty', -GA_PEN_BOTTLENECK))

#: How closely the run's recorded closed-form values must be reproduced,
#: and the recorded score gain by its criterion split.
REPRODUCE_RTOL = 1e-9
RECOMPOSE_ATOL = 1e-9

#: The Figure C run directories this reads by default (the newest one);
#: the no-anonymous sensitivity run has a prefix of its own.
FIGC_PREFIX = 'real_data_uci_'


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


def default_figc_dir(results: str) -> Optional[str]:
    """The newest Figure C run under ``results`` whose summary records the
    closed-form scores and whose sidecar records the base parameters."""
    for d in sorted(glob.glob(os.path.join(results, FIGC_PREFIX + '*')),
                    reverse=True):
        try:
            s = json.load(open(os.path.join(d, 'summary.json')))
            side = json.load(open(os.path.join(d, 'sidecar.json')))
        except (OSError, ValueError):
            continue
        if (isinstance(((s.get('results') or {}).get('closed_form') or {})
                       .get('scores'), dict)
                and isinstance(side.get('base_params'), dict)):
            return d
    return None


def conversion_bound(omni_dir: str) -> Dict[str, Any]:
    """The Omnichannel bundle's lower bound on visit conversion, as its
    calibration records it, with the family that attains it and the
    bundle's provenance."""
    families, arrivals, notes = DA.load_omnichannel_bundle(omni_dir)
    params = DC.calibrate_omnichannel(families, arrivals)
    bound = float(params.calibration_extra['conversion_rate_lower_bound'])
    # The family behind it, cleaned as calibrate_omnichannel cleans it.
    fam = families[families['product_family'].astype(str).str.strip() != '']
    p = (pd.to_numeric(fam['purchase_pct_instore'], errors='coerce')
         .fillna(0.0).clip(0.0, 1.0))
    if abs(float(p.max()) - bound) > 1e-12:
        raise RuntimeError('the recorded bound is not the largest family '
                           'purchase probability')
    prov = stamp_provenance(source_path=omni_dir,
                            adapter_name=DA.OmnichannelRetailAdapter.name,
                            adapter_version=DA.OmnichannelRetailAdapter.version,
                            rows_in=int(len(families)),
                            rows_kept=int(len(fam)))
    return {'value': bound,
            'family': str(fam.loc[p.idxmax(), 'product_family']),
            'n_families': int(len(fam)),
            'independence_estimate': float(
                params.calibration_extra['conversion_rate_independence_estimate']),
            'calibration_conversion_source': params.conversion_rate_source,
            'definition': ('calibrate_omnichannel conversion_rate_lower_bound:'
                           ' the largest single-family in-store purchase '
                           'probability (a visitor buys at least that family '
                           'with that probability)'),
            'provenance': prov.to_dict(),
            'loader_notes': list(notes or [])}


def _revenue(bp: Mapping[str, Any], horizon: int, rec: Mapping[str, Any],
             **kw) -> float:
    return float(expected_revenue(bp, horizon, score=rec['score'],
                                  breakdown=rec['breakdown'], **kw))


def _lifts(bp, horizon, scores, **kw) -> Dict[str, float]:
    base = _revenue(bp, horizon, scores[BASELINE], **kw)
    return {k: (_revenue(bp, horizon, scores[k], **kw) - base)
            / max(base, 1e-9) * 100.0 for k in SEARCHED if k in scores}


def lift_at_rate(bp, horizon, scores, rate) -> Dict[str, Any]:
    """Every searched layout's lift with the store recalibrated to
    conversion ``rate`` (Figure C's conversion-scan recalibration), and
    whether a 0.99 conversion clamp binds for any layout there."""
    bpp = _at_conversion(bp, rate)
    return {'rate': float(rate),
            'lift_pct': _lifts(bpp, horizon, scores),
            'clamps_bind': bool(any(
                _clamps_bind(r['score'], r['breakdown'], bpp)
                for r in scores.values()))}


def scan_bracket(scan: Mapping[str, Any], rate: float) -> Dict[str, Any]:
    """The conversion scan's grid rates on either side of ``rate`` and the
    lifts Figure C recorded at them."""
    grid = [float(g) for g in scan['grid']]
    out = {}
    below = [i for i, g in enumerate(grid) if g <= rate]
    above = [i for i, g in enumerate(grid) if g >= rate]
    for side, idx in (('below', below[-1] if below else None),
                      ('above', above[0] if above else None)):
        if idx is not None:
            out[side] = {'rate': grid[idx],
                         'lift_pct': {k: float(v[idx])
                                      for k, v in scan['lift_pct'].items()}}
    return out


def decomposition(bp, horizon, scores) -> Dict[str, Any]:
    """Per searched layout: the lift with each channel switched off and the
    share the abandonment channel carries; the composite-score gain split
    by criterion and the flow criterion's share."""
    full = _lifts(bp, horizon, scores)
    no_ab_bp = dict(bp)
    no_ab_bp['abandonment_rate'] = 0.0
    no_ab = _lifts(no_ab_bp, horizon, scores)
    base_same = abs(_revenue(no_ab_bp, horizon, scores[BASELINE])
                    - _revenue(bp, horizon, scores[BASELINE]))
    if base_same > 1e-9 * abs(_revenue(bp, horizon, scores[BASELINE])):
        raise RuntimeError('switching the abandonment channel off moved the '
                           'baseline; the baseline is not the anchor layout')
    no_conv = _lifts(bp, horizon, scores, elasticities={'conv': 0.0})
    no_bsk = _lifts(bp, horizon, scores, elasticities={'bsk': 0.0})
    b0 = scores[BASELINE]
    per = {}
    for k in full:
        b = scores[k]
        contrib = {c: w * (float(b['breakdown'].get(c, 0.0))
                           - float(b0['breakdown'].get(c, 0.0)))
                   for c, w in CRITERION_WEIGHTS}
        gain = float(b['score']) - float(b0['score'])
        if abs(sum(contrib.values()) - gain) > RECOMPOSE_ATOL:
            raise RuntimeError(f'{k}: the criteria do not recompose the '
                               f'recorded score gain')
        per[k] = {
            'lift_pct': full[k],
            'lift_pct_no_abandonment_channel': no_ab[k],
            'abandonment_share': (1.0 - no_ab[k] / full[k]
                                  if full[k] else None),
            'abandonment_pp': full[k] - no_ab[k],
            'lift_pct_no_conversion_elasticity': no_conv[k],
            'conversion_elasticity_share': (1.0 - no_conv[k] / full[k]
                                            if full[k] else None),
            'lift_pct_no_basket_elasticity': no_bsk[k],
            'basket_elasticity_share': (1.0 - no_bsk[k] / full[k]
                                        if full[k] else None),
            'score_gain': gain,
            'criterion_contributions': contrib,
            'flow_share_of_score_gain': (contrib['flow'] / gain
                                         if gain else None),
        }
    return {'per_layout': per,
            'abandonment_coefficients': {
                'ABANDON_FLOW_COEF': ABANDON_FLOW_COEF,
                'ABANDON_SECTION_COEF': ABANDON_SECTION_COEF},
            'method': ('abandonment channel off: base_params abandonment_rate '
                       '0 (abandonment factor 1 for every layout; the anchor '
                       'baseline unchanged); elasticity channels off: that '
                       'elasticity 0; share = 1 - lift_off / lift. Score gain: '
                       'weight x criterion change over the baseline, '
                       'penalties negative')}


def rescore(figc_dir: str, omni_dir: str) -> tuple:
    """``(summary, base_params)``: the re-scoring of the Figure C run in
    ``figc_dir`` and the base parameters it was made with."""
    s = json.load(open(os.path.join(figc_dir, 'summary.json')))
    side = json.load(open(os.path.join(figc_dir, 'sidecar.json')))
    cf = s['results']['closed_form']
    horizon = int(cf['horizon_days'])
    scores = cf['scores']
    bp = side['base_params']
    anchor = bp['score_anchor']
    if abs(float(anchor['score']) - float(scores[BASELINE]['score'])) > 1e-12:
        raise RuntimeError("the baseline's score is not the run's anchor")
    worst = max(abs(_revenue(bp, horizon, rec) - float(cf['values'][k]))
                / max(abs(float(cf['values'][k])), 1e-9)
                for k, rec in scores.items())
    if worst > REPRODUCE_RTOL:
        raise RuntimeError(f"the run's closed-form values are not "
                           f"reproduced (relative error {worst:.2e})")
    bound = conversion_bound(omni_dir)
    scan = cf['conversion_scan']
    at_bound = lift_at_rate(bp, horizon, scores, bound['value'])
    at_bound['scan_bracket'] = scan_bracket(scan, bound['value'])
    at_bound['assumed_rate'] = float(bp['conversion_rate'])
    at_bound['lift_pct_at_assumed'] = _lifts(bp, horizon, scores)
    at_bound['clamp_onset'] = scan.get('clamp_onset')
    at_bound['method'] = ('closed form (experiments.closed_form.'
                          'expected_revenue) of the saved scores and '
                          'breakdowns, store recalibrated with '
                          'run_real_data_example._at_conversion, as the '
                          'conversion scan does')
    return {
        'experiment': 'figc_rescore',
        # The run's name, not a path: a key ending in 'dir' would be read
        # as a path by portable_paths and rewritten against the cwd.
        'figc': {'run_name': os.path.basename(os.path.normpath(figc_dir)),
                 'summary_sha256': _sha256(os.path.join(figc_dir,
                                                        'summary.json')),
                 'git_sha': side.get('git_sha'),
                 'horizon_days': horizon,
                 'assumed_conversion': float(bp['conversion_rate']),
                 'values_reproduced': True,
                 'max_rel_error': worst},
        'conversion_bound': bound,
        'lift_at_conversion_bound': at_bound,
        'decomposition': decomposition(bp, horizon, scores),
    }, bp


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--figc-dir', default=None,
                   help='Figure C run to re-score (default: the newest '
                        'real_data_uci_* under --results)')
    p.add_argument('--results', default=os.path.join(_HERE, 'results'),
                   help='where Figure C runs are found')
    p.add_argument('--omnichannel-dir', default=None,
                   help='the Omnichannel Retail bundle (default: found by '
                        'dataset_paths.omnichannel_dir)')
    p.add_argument('--out-root', default=os.path.join(_HERE, 'results'))
    args = p.parse_args(argv)
    if args.figc_dir is None:
        args.figc_dir = default_figc_dir(args.results)
        if args.figc_dir is None:
            p.error(f'no Figure C run with closed-form scores under '
                    f'{args.results}')
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    prov = provenance_snapshot()
    t0 = time.perf_counter()
    omni_dir = dataset_paths.omnichannel_dir(args.omnichannel_dir)
    summary, bp = rescore(args.figc_dir, omni_dir)
    out_dir = make_run_dir(args.out_root, 'figc_rescore')
    # Paths inside the repository are recorded relative to its root. The
    # sidecar is handed the summary as it came (write_sidecar makes it
    # portable itself): a path made relative once would be read as
    # relative to the working directory and rewritten a second time.
    with open(os.path.join(out_dir, 'summary.json'), 'w',
              encoding='utf-8', newline='\n') as f:
        json.dump(portable_paths(summary), f, indent=1)
    write_sidecar(out_dir, {'experiment': 'figc_rescore',
                            'args': vars(args),
                            'wall_seconds': time.perf_counter() - t0,
                            'figc_dir': args.figc_dir,
                            'base_params': bp,
                            'summary': summary}, provenance=prov)
    b = summary['conversion_bound']
    a = summary['lift_at_conversion_bound']
    d = summary['decomposition']['per_layout']['optimized']
    print(f"[rescore] {summary['figc']['run_name']}: conversion bound "
          f"{b['value']:.3f} ({b['family']}); GA lift at it "
          f"{a['lift_pct']['optimized']:+.3f}% (at the assumed "
          f"{a['assumed_rate']:.2f}: {a['lift_pct_at_assumed']['optimized']:+.3f}%)")
    print(f"[rescore] abandonment channel {100 * d['abandonment_share']:.1f}% "
          f"of the GA's lift ({d['abandonment_pp']:+.3f} pp); flow "
          f"{100 * d['flow_share_of_score_gain']:.1f}% of its score gain")
    print(f"[rescore] wrote {out_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
