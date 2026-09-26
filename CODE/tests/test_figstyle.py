"""Paper figures are drawn at the width they print at (review R56).

``figstyle.save`` refuses a figure listed in ``PRINT_FRAC`` unless it is
exactly as wide as it prints, none of its text is below ``MIN_FONT_PT``
there and nothing runs past its edge. These tests pin that guard and run
the display-free figure code that does not need an experiment artifact:
the three ``make_paper_figures`` plots from a small synthetic data record
and from a real run of the script at tiny GA settings, and the archetype
previews from the architecture engine, whose panel counts Fig. 4's
caption quotes.
"""

import json

import numpy as np
import pytest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

import figstyle  # noqa: E402


@pytest.fixture(autouse=True)
def _print_style():
    with matplotlib.rc_context():
        figstyle.apply_print()
        yield
    plt.close('all')


def test_text_width_is_the_acm_small_block():
    # acmart.cls, acmsmall: 6.75 in paper, 46 pt inner and outer margins.
    assert figstyle.TEXTWIDTH_IN == pytest.approx(5.477, abs=1e-3)
    assert all(0 < f <= 1 for f in figstyle.PRINT_FRAC.values())


def test_guard_accepts_a_figure_at_print_size(tmp_path):
    fig, ax = figstyle.print_figure('lhs_hist', height_in=2.4)
    ax.plot([0, 1], [0, 1], label='series')
    ax.set_xlabel('x label')
    ax.legend()
    assert figstyle.check_print(fig, 'lhs_hist') >= figstyle.MIN_FONT_PT
    path = figstyle.save(fig, 'lhs_hist', out_dir=str(tmp_path))
    assert path.endswith('lhs_hist.pdf')
    assert (tmp_path / 'lhs_hist.png').exists()


def test_guard_refuses_the_wrong_width():
    fig, _ = plt.subplots(figsize=(7.2, 4.0))
    with pytest.raises(ValueError, match='prints at'):
        figstyle.check_print(fig, 'lhs_hist')


def test_guard_refuses_small_type():
    fig, ax = figstyle.print_figure('lhs_hist', height_in=2.4)
    ax.set_title('too small', fontsize=6)
    with pytest.raises(ValueError, match='below'):
        figstyle.check_print(fig, 'lhs_hist')


def test_guard_refuses_text_past_the_edge():
    fig, ax = figstyle.print_figure('lhs_hist', height_in=2.4)
    ax.set_xlabel('a label far too long for a figure this narrow ' * 3)
    with pytest.raises(ValueError, match='past the'):
        figstyle.check_print(fig, 'lhs_hist')


def test_series_differ_by_more_than_hue():
    styles = [(s['linestyle'], s['marker']) for s in figstyle.SERIES]
    assert len(set(styles)) == len(styles)
    assert len(set(figstyle.HATCHES)) == len(figstyle.HATCHES)


def test_paper_figures_redraw_from_their_data_record(tmp_path):
    from experiments import make_paper_figures as mpf
    rng = np.random.default_rng(0)
    best = np.maximum.accumulate(
        64_000 + np.cumsum(rng.uniform(0, 50, (3, 12)), axis=1), axis=1)
    m = np.unique(np.geomspace(200, 20_000, 60).astype(int))
    data = {
        'ga_convergence': {'best_so_far': best.tolist(),
                           'pop_mean': (best - 200).tolist(),
                           'n_seeds': 3, 'pop_size': 30, 'mc_iters': 2000,
                           'mc_days': 30},
        'score_components': {
            'baseline': {k: 0.4 for k in mpf.SCORE_KEYS},
            'ga': {k: 0.5 for k in mpf.SCORE_KEYS},
            'composite_baseline': 0.55, 'composite_ga': 0.57},
        'mc_convergence': {'m': m.tolist(),
                           'run_mean': (64_200 + 300 / np.sqrt(m)).tolist(),
                           'se': (1500 / np.sqrt(m)).tolist(),
                           'n_used': 2000, 'n_big': 20_000, 'mc_days': 30},
    }
    (tmp_path / mpf.DATA_FILE).write_text(json.dumps(data))
    assert mpf.main(['--replot', '--figs-dir', str(tmp_path)]) == 0
    for stem in ('ga_convergence', 'score_components', 'mc_convergence'):
        assert (tmp_path / f'{stem}.pdf').exists(), stem


def test_archetype_previews_print_legibly(tmp_path):
    from experiments import make_layout_previews as mlp
    out = mlp.archetype_previews(str(tmp_path))
    assert out.endswith('layout_previews.pdf')
    assert (tmp_path / 'layout_previews.png').exists()


def _pdf_width_in(path):
    """Page width of a one-page matplotlib PDF, from its MediaBox."""
    import re
    box = re.search(rb'/MediaBox\s*\[\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)',
                    path.read_bytes())
    return (float(box.group(3)) - float(box.group(1))) / 72.0


def test_paper_figures_pass_the_guard_on_a_real_run(tmp_path):
    """The three ``make_paper_figures`` plots drawn from a real run of the
    script -- GA seeds on the fixed scenario, the score decomposition and
    the 20,000-iteration Monte Carlo trace -- at tiny GA settings, not from
    a made-up record: ``figstyle.save`` raises inside ``main`` if a figure
    fails the print guard, so a guard failure shows up here and not at the
    end of the paper-grade run. Each PDF is exactly as wide as it prints
    (the stale drawings in figs/ were 7-8 in wide, i.e. scaled to about
    4.3-5.4 pt type). The data record is written before anything is
    drawn, so ``--replot`` redraws the same figures without re-running the
    GA -- the recovery path if the paper-grade run's figures are refused."""
    from experiments import make_paper_figures as mpf
    stems = ('ga_convergence', 'score_components', 'mc_convergence')
    assert mpf.main(['--n-seeds', '2', '--pop-size', '4', '--n-gens', '3',
                     '--mc-iters', '100', '--figs-dir', str(tmp_path)]) == 0
    for stem in stems:
        pdf = tmp_path / f'{stem}.pdf'
        assert _pdf_width_in(pdf) == pytest.approx(
            figstyle.PRINT_FRAC[stem] * figstyle.TEXTWIDTH_IN, abs=0.01), stem
    data = json.loads((tmp_path / mpf.DATA_FILE).read_text())
    assert np.asarray(data['ga_convergence']['best_so_far']).shape == (2, 3)
    assert data['mc_convergence']['n_used'] == 100
    assert (tmp_path / 'realized_scores.json').exists()
    # The record carries the objective it was made under, which the macro
    # generator's model check reads before it quotes the realized scores.
    import make_results_macros as MRM
    rs = json.loads((tmp_path / 'realized_scores.json').read_text())
    assert MRM._model_current(rs)
    # Like a sidecar: whether the checkout moved during the run, and what
    # the run cost on what machine (review R50, R53).
    assert isinstance(rs['provenance']['git_state_changed_during_run'], bool)
    assert rs['wall_seconds'] > 0 and rs['hardware']['cpu_count']
    assert rs['mc_trace_iters'] == data['mc_convergence']['n_big']

    for stem in stems:
        (tmp_path / f'{stem}.pdf').unlink()
    assert mpf.main(['--replot', '--figs-dir', str(tmp_path)]) == 0
    for stem in stems:
        assert (tmp_path / f'{stem}.pdf').exists(), stem


def test_ga_diversity_figure_prints_at_its_width(tmp_path):
    """Fig. 11 (``run_ga_sensitivity``'s diversity figure) is drawn at its
    print width with print-size type, through the guard, and leaves the
    runner's own style as it found it (review R56: it was 7.07 in wide at
    10 pt, about 6.2 pt at 0.8 of the text width)."""
    import matplotlib
    from experiments import run_ga_sensitivity as rgs
    before = dict(matplotlib.rcParams)
    rng = np.random.default_rng(0)
    div = [np.linspace(0.2, 0.08, 25) + rng.normal(0, 0.005, 25)
           for _ in range(6)]
    best = [np.cumsum(np.abs(rng.normal(0, 10, 25))) + 5000 for _ in range(6)]
    out = rgs.plot_diversity(div, best, mut_rate=0.18, pop_size=30,
                             figs=str(tmp_path))
    pdf = tmp_path / 'ga_diversity.pdf'
    assert _pdf_width_in(pdf) == pytest.approx(
        figstyle.print_width('ga_diversity'), abs=0.01)
    rec = out['diversity_figure']
    assert rec['min_font_pt_printed'] >= figstyle.MIN_FONT_PT
    assert rec['include_width_in'] == pytest.approx(
        figstyle.print_width('ga_diversity'), abs=0.001)
    assert out['diversity_start'] > out['diversity_end'] > 0
    assert dict(matplotlib.rcParams) == before


def test_archetype_preview_counts_are_the_captions(tmp_path):
    """Fig. 4's caption quotes the counts the panels report: the 18x13 m
    stores have 7 sections, 28 items and 2 lanes, the 32x22 m stores 10
    sections, 80 items and 3 lanes -- every one of them (the caption once
    said '60--80 items'). If the engine, the catalog scaling or the preview
    seeds change, this fails, and the caption must change with it."""
    from experiments import make_layout_previews as mlp
    counts = {}
    for p in mlp.archetype_panels():
        counts.setdefault((p['width'], p['height']), set()).add(
            (p['n_sections'], p['n_items'], p['n_lanes']))
    assert counts == {(18.0, 13.0): {(7, 28, 2)},
                      (32.0, 22.0): {(10, 80, 3)}}
