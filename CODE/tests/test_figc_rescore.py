"""Figure C's saved layouts re-scored in closed form (run_figc_rescore) and
drawn (make_layout_figure).

The re-scoring reads only what a Figure C run records -- each layout's
score and per-criterion breakdown, and the base parameters with their
anchor -- so it is tested on hand-made records: the run's closed-form
values must be reproduced before anything is reported; switching the
abandonment channel off leaves the anchor baseline where it is and takes
away exactly the lift that flows through it; the criterion split adds up
to the score gain; the lift at the assumed conversion rate is the lift the
run reports. The Omnichannel bound itself is checked against the bundle
when the bundle is present. The layout figure is drawn from a synthetic
store at print width.
"""
import json
import os
import types

import pytest

import figstyle
from experiments import make_layout_figure as MLF
from experiments import run_figc_rescore as RFR
from experiments.closed_form import expected_revenue

H = 30
WEIGHTS = dict(RFR.CRITERION_WEIGHTS)
CRITERIA = [c for c, _ in RFR.CRITERION_WEIGHTS]


def _breakdown(**delta):
    b = {c: 0.0 for c in CRITERIA}
    b.update({'traffic': 0.14, 'cross_merch': 0.79, 'flow': 0.65,
              'revenue_placement': 0.03, 'section_compliance': 1.0,
              'accessibility': 0.54})
    for k, v in delta.items():
        b[k] += v
    return b


def _record(b):
    return {'score': float(sum(WEIGHTS[c] * b[c] for c in CRITERIA)),
            'breakdown': b}


def _scores():
    base = _record(_breakdown())
    return {'baseline': base,
            'optimized': _record(_breakdown(flow=0.10, cross_merch=0.05)),
            'rs': _record(_breakdown(cross_merch=0.05)),
            'sa': _record(_breakdown(flow=0.05))}


def _base_params(scores):
    anchor = dict(scores['baseline'], source='as_built_repaired')
    return {'customers_per_hour': 16.0, 'conversion_rate': 0.3,
            'rev_per_converting_customer': 470.0, 'rev_std': 1000.0,
            'impulse_rate': 0.2, 'avg_impulse_value': 73.0,
            'avg_basket_size': 275.0, 'std_basket_size': 900.0,
            'basket_sizes_observed': [], 'abandonment_rate': 0.07,
            'avg_queue_time': 5.0, 'score_anchor': anchor}


def test_the_abandonment_channel_and_the_flow_share():
    scores = _scores()
    bp = _base_params(scores)
    dec = RFR.decomposition(bp, H, scores)['per_layout']
    # No flow change: nothing moves the abandonment, so none of the lift
    # comes through it and none of the score gain is flow.
    assert dec['rs']['abandonment_share'] == pytest.approx(0.0, abs=1e-12)
    assert dec['rs']['flow_share_of_score_gain'] == pytest.approx(0.0)
    # Flow 0.10 and cross-merchandising 0.05 at equal weights: two thirds.
    assert dec['optimized']['flow_share_of_score_gain'] == \
        pytest.approx(2 / 3)
    assert 0.0 < dec['optimized']['abandonment_share'] < 1.0
    assert dec['sa']['flow_share_of_score_gain'] == pytest.approx(1.0)
    # The lift without the channel is the lift at zero baseline
    # abandonment, from the same (unchanged) baseline.
    no_ab = dict(bp, abandonment_rate=0.0)
    base = expected_revenue(bp, H, **_kw(scores['baseline']))
    assert expected_revenue(no_ab, H, **_kw(scores['baseline'])) == \
        pytest.approx(base, rel=1e-12)
    lift = (expected_revenue(no_ab, H, **_kw(scores['optimized'])) - base) \
        / base * 100
    assert dec['optimized']['lift_pct_no_abandonment_channel'] == \
        pytest.approx(lift)


def _kw(rec):
    return {'score': rec['score'], 'breakdown': rec['breakdown']}


def test_a_breakdown_that_does_not_recompose_its_score_is_refused():
    scores = _scores()
    scores['optimized'] = dict(scores['optimized'],
                               score=scores['optimized']['score'] + 1e-3)
    with pytest.raises(RuntimeError):
        RFR.decomposition(_base_params(_scores()), H, scores)


def test_lift_at_the_assumed_rate_is_the_runs_lift_and_the_bracket():
    scores = _scores()
    bp = _base_params(scores)
    at = RFR.lift_at_rate(bp, H, scores, 0.3)
    assert at['lift_pct'] == pytest.approx(RFR._lifts(bp, H, scores))
    assert not at['clamps_bind']
    lower = RFR.lift_at_rate(bp, H, scores, 0.54)['lift_pct']['optimized']
    assert lower < at['lift_pct']['optimized']
    scan = {'grid': [0.05, 0.5, 0.55, 0.99],
            'lift_pct': {'optimized': [2.0, 1.7, 1.68, 0.5]}}
    br = RFR.scan_bracket(scan, 0.538)
    assert br['below'] == {'rate': 0.5, 'lift_pct': {'optimized': 1.7}}
    assert br['above'] == {'rate': 0.55, 'lift_pct': {'optimized': 1.68}}


def _figc_run(tmp_path, scores, bp, values=None):
    d = tmp_path / 'real_data_uci_20990101-000001'
    d.mkdir()
    vals = values or {k: expected_revenue(bp, H, **_kw(r))
                      for k, r in scores.items()}
    summary = {'results': {'closed_form': {
        'horizon_days': H, 'values': vals, 'scores': scores,
        'conversion_scan': {'grid': [0.3, 0.55],
                            'lift_pct': {'optimized': [1.8, 1.7]},
                            'clamp_onset': 0.98}}}}
    (d / 'summary.json').write_text(json.dumps(summary))
    (d / 'sidecar.json').write_text(json.dumps({'base_params': bp,
                                                'git_sha': 'abc'}))
    return str(d)


def test_rescore_reproduces_the_run_before_it_reports(tmp_path, monkeypatch):
    scores = _scores()
    bp = _base_params(scores)
    monkeypatch.setattr(RFR, 'conversion_bound', lambda _d: {
        'value': 0.538, 'family': 'Fresh fruits'})
    d = _figc_run(tmp_path, scores, bp)
    summary, used = RFR.rescore(d, 'unused')
    assert used is not None and summary['figc']['values_reproduced']
    assert summary['figc']['run_name'] == os.path.basename(d)
    # Paths made portable leave the run's name alone.
    from experiments._common import portable_paths
    portable = portable_paths(summary)
    assert portable['figc']['run_name'] == os.path.basename(d)
    at = summary['lift_at_conversion_bound']
    assert at['scan_bracket']['above']['rate'] == 0.55
    assert at['lift_pct']['optimized'] < at['lift_pct_at_assumed']['optimized']
    assert RFR.default_figc_dir(str(tmp_path)) == d
    # A record whose closed-form values are not today's model's is refused.
    bad = tmp_path / 'other'
    bad.mkdir()
    d2 = _figc_run(bad, scores, bp,
                   values={k: 1.0 for k in scores})
    with pytest.raises(RuntimeError):
        RFR.rescore(d2, 'unused')


def test_omnichannel_conversion_bound():
    import dataset_paths
    try:
        omni = dataset_paths.omnichannel_dir()
    except FileNotFoundError:
        pytest.skip('Omnichannel bundle not present')
    b = RFR.conversion_bound(omni)
    assert b['value'] == pytest.approx(0.538, abs=1e-9)
    assert b['family'] == 'Fresh fruits'
    assert len(b['provenance']['source_sha256']) == 64


# --- the layout figure ------------------------------------------------------------

def test_moves_and_their_summary():
    base = {'a': (0.0, 0.0), 'b': (1.0, 1.0), 'c': (2.0, 2.0)}
    opt = {'a': (0.0, 0.0), 'b': (1.0, 1.5), 'c': (2.0, 5.0)}
    d = MLF.moves(base, opt)
    assert d == {'a': 0.0, 'b': 0.5, 'c': 3.0}
    sizes = {n: (1.0, 3.2) for n in base}
    m = MLF.move_stats(base, opt, sizes)
    assert (m['n_fixtures'], m['n_moved'], m['n_moved_at_least_aisle']) ==         (3, 2, 1)
    assert 'n_moved_past_aisle' not in m
    assert m['max_m'] == 3.0 and m['highlight_m'] == MLF.MOVE_HIGHLIGHT_M
    assert (m['slot_tol_m'], m['along_run_tol_m']) ==         (MLF.SLOT_TOL_M, MLF.ALONG_RUN_TOL_M)
    json.dumps(m)   # plain numbers, so the record can be written


def test_long_moves_split_into_slot_exchanges_and_runs():
    """A run of four gondolas (1 x 3.2 m, long axis vertical) and a wall
    row (3 x 0.7 m, long axis horizontal): two gondolas swap slots along
    the run, one slides along it into a gap, one is pushed a metre across
    it; one wall fixture swaps along the row, one leaves the row."""
    g, w = (1.0, 3.2), (3.0, 0.7)
    base = {'g1': (0.0, 0.0), 'g2': (0.0, 4.0), 'g3': (0.0, 8.0),
            'g4': (0.0, 12.0), 'w1': (5.0, 20.0), 'w2': (8.5, 20.0),
            'w3': (12.0, 20.0)}
    opt = dict(base)
    opt.update({'g1': (0.0, 4.0), 'g2': (0.0, 0.0),     # exchange, in run
                'g3': (0.1, 10.0),                      # along, into a gap
                'g4': (1.2, 14.0),                      # across the run
                'w1': (8.5, 20.0), 'w2': (5.0, 20.0),   # exchange, in row
                'w3': (12.0, 17.0)})                    # out of its row
    sizes = {n: (g if n.startswith('g') else w) for n in base}
    kinds = MLF.move_kinds(base, opt, sizes)
    assert set(kinds) == set(base)          # every move here is >= 1.6 m
    assert {n for n, k in kinds.items() if k['slot_exchange']} ==         {'g1', 'g2', 'w1', 'w2'}
    assert {n for n, k in kinds.items() if k['along_run']} ==         {'g1', 'g2', 'g3', 'w1', 'w2'}
    m = MLF.move_stats(base, opt, sizes)
    assert (m['n_moved_at_least_aisle'], m['n_slot_exchange'],
            m['n_along_run'], m['n_slot_exchange_along_run']) == (7, 4, 5, 4)
    # A short move is neither: it is not counted among the long ones.
    short = dict(base, g1=(0.0, 1.0))
    assert MLF.move_kinds(base, short, sizes) == {}


def test_popularity_ranks_and_their_classes():
    pop = {'a': 5, 'b': 9, 'c': 9, 'd': 1, 'e': 7}
    # Most bought first, ties kept in the calibration's order -- the GA's
    # own sort for the flow tour.
    assert MLF.popularity_order(pop) == ['b', 'c', 'e', 'a', 'd']
    assert MLF.item_ranks(pop, ['a', 'c', 'd', 'e']) ==         {'c': 1, 'e': 2, 'a': 3, 'd': 4}
    assert MLF.rank_classes(108) == [5, 27, 54, 108]
    assert MLF.rank_classes(12) == [5, 6, 12]
    assert MLF.rank_classes(4) == [5]


def test_drawn_flow_tour_is_the_gas_flow_criterion():
    """The tour the figure draws scores what the GA's flow criterion gives
    the same layout, on the as-built layout and on one with the most-bought
    product moved; and it runs entrance, products in rank order,
    checkout."""
    from experiments._common import build_headless_shop
    from synthetic_shops import generate_synthetic_shop
    s = generate_synthetic_shop(name='t', seed=10_000, n_items=10,
                                width=12.0, height=10.0)
    shop = build_headless_shop(s)
    names = list(shop.floors[1]['items'])
    base = {n: tuple(shop.floors[1]['items'][n]['position']) for n in names}
    tour = MLF.checked_tour(shop, base, names)
    assert tour['length_m'] is not None and tour['length_m'] > 0
    assert 0 < tour['ga_flow_score'] < 1
    assert len(tour['points']) == len(tour['items']) + 2
    pop = shop.customer_simulation.analytics['popular_items']
    assert tour['items'] == MLF.popularity_order(pop)[:MLF.FLOW_TOUR_ITEMS]
    moved = dict(base)
    top = tour['items'][0]
    moved[top] = (base[top][0] + 0.5, base[top][1] + 0.5)
    t2 = MLF.checked_tour(shop, moved, names)
    assert t2['length_m'] != pytest.approx(tour['length_m'])


def test_layout_figure_draws_at_print_width(tmp_path):
    from experiments._common import build_headless_shop
    from synthetic_shops import generate_synthetic_shop
    s = generate_synthetic_shop(name='t', seed=10_000, n_items=10,
                                width=12.0, height=10.0)
    shop = build_headless_shop(s)
    names = list(shop.floors[1]['items'])
    base = {n: tuple(shop.floors[1]['items'][n]['position']) for n in names}
    opt = dict(base)
    opt[names[0]] = (base[names[0]][0], base[names[0]][1] + 2.0)
    for kw in ({},
               {'ranks': MLF.item_ranks(
                   shop.customer_simulation.analytics['popular_items'],
                   names),
                'tours': {k: MLF.flow_tour(shop, lay, names)
                          for k, lay in (('baseline', base),
                                         ('optimized', opt))}}):
        pdf, smallest = MLF.draw(types.SimpleNamespace(shop=shop),
                                 {'baseline': base, 'optimized': opt},
                                 MLF.moves(base, opt), out_dir=str(tmp_path),
                                 **kw)
        assert smallest >= figstyle.MIN_FONT_PT
        w, _h = figstyle.pdf_size_in(pdf)
        assert w == pytest.approx(figstyle.print_width(MLF.STEM), abs=0.01)
        assert os.path.exists(os.path.join(tmp_path, MLF.STEM + '.png'))


def test_summary_and_sidecar_record_the_same_portable_paths(tmp_path,
                                                             monkeypatch):
    """A dataset path inside the repository is recorded relative to its
    root once, the same in summary.json and in the sidecar's copy."""
    import dataset_paths
    scores = _scores()
    bp = _base_params(scores)
    inside = os.path.join(dataset_paths.REPO_ROOT, 'DATASETS', 'bundle')
    monkeypatch.setattr(RFR, 'conversion_bound', lambda _d: {
        'value': 0.538, 'family': 'Fresh fruits',
        'provenance': {'source_path': inside}})
    monkeypatch.setattr(RFR.dataset_paths, 'omnichannel_dir',
                        lambda _p=None: inside)
    d = _figc_run(tmp_path, scores, bp)
    out = tmp_path / 'out'
    assert RFR.main(['--figc-dir', d, '--out-root', str(out)]) == 0
    run = next(out.iterdir())
    summary = json.loads((run / 'summary.json').read_text())
    side = json.loads((run / 'sidecar.json').read_text())
    got = summary['conversion_bound']['provenance']['source_path']
    assert got == 'DATASETS/bundle'
    assert side['summary']['conversion_bound']['provenance'][
        'source_path'] == got
