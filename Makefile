# Reproduction targets for the retail-layout simulator (audit R5.2).
#
# Unix / macOS:  make <target>
# Windows:       use reproduce.ps1 (identical steps) — `make` is usually
#                absent on Windows.
#
# Quick start (minutes):   make deps test smoke
# Rebuild the PDF:         make macros paper      (needs figs/ present)
# Full paper-grade regen:  make paper-grade       (HOURS; see below)

PY    ?= python
CODE   = CODE
PAPER  = TOMACS_submission
RETAIL ?= DATASETS/UCI Online Retail II .xlsx.xlsx

.PHONY: help deps test verify smoke figures-headless macros paper all \
        experiments-paper paper-grade clean

help:
	@echo "Targets:"
	@echo "  deps              pip install -r requirements.txt"
	@echo "  test              run the pytest numeric-core suite (~2 min)"
	@echo "  verify            architecture invariants + dataset pipeline smokes"
	@echo "  smoke             fast end-to-end experiment smokes (~mins)"
	@echo "  figures-headless  regen the display-free paper figures"
	@echo "                    (plots + the floor-plan previews)"
	@echo "  macros            regen paper_results_macros.tex from artefacts"
	@echo "  paper             pdflatex->bibtex->pdflatex x2 (needs figs/)"
	@echo "  experiments-paper paper-grade experiment runs (HOURS)"
	@echo "  paper-grade       experiments-paper + figures + macros + paper"
	@echo "  all               deps test smoke"

deps:
	$(PY) -m pip install -r requirements.txt

test:
	cd $(CODE) && $(PY) -m pytest -q tests/

# The two standalone verification scripts cited in the paper's
# Verification section. They are not pytest tests (one needs the
# datasets, the other sweeps 105 generated floor plans), so they get
# their own target rather than slowing `make test` down.
verify:
	cd $(CODE) && $(PY) architecture_smoke.py
	cd $(CODE) && $(PY) dataset_smoke.py

smoke:
	cd $(CODE) && $(PY) -m experiments.run_synthetic_gt --n-scenarios 3 \
	    --n-seeds 2 --mc-iters 200 --n-gens 8 --pop-size 16 --n-spearman-samples 40
	cd $(CODE) && $(PY) -m experiments.run_baseline_comparison --n-scenarios 3 \
	    --n-seeds 2 --mc-iters 200 --n-gens 8 --pop-size 16

figures-headless:
	cd $(CODE) && $(PY) -m experiments.make_paper_figures
	cd $(CODE) && $(PY) -m experiments.restyle_figures
	cd $(CODE) && $(PY) -m experiments.make_layout_previews

macros:
	cd $(CODE) && $(PY) make_results_macros.py

paper:
	pdflatex -interaction=nonstopmode -halt-on-error $(PAPER).tex
	bibtex $(PAPER)
	pdflatex -interaction=nonstopmode -halt-on-error $(PAPER).tex
	pdflatex -interaction=nonstopmode -halt-on-error $(PAPER).tex

# ── Paper-grade experiment regeneration (HOURS of compute) ──────────────
# Reproduces every artefact the macros consume. Figure C additionally
# needs the UCI Online Retail II workbook under DATASETS/ (RETAIL var).
experiments-paper:
	cd $(CODE) && $(PY) -m experiments.run_synthetic_gt --n-scenarios 30 \
	    --n-seeds 10 --mc-iters 2000 --n-gens 25 --pop-size 30
	cd $(CODE) && $(PY) -m experiments.run_baseline_comparison --n-scenarios 30 \
	    --n-seeds 10 --mc-iters 2000 --n-gens 25 --pop-size 30
	cd $(CODE) && $(PY) -m experiments.run_mc_groundtruth --n-scenarios 6 \
	    --normal-budget 750 --big-budget 7500 --mc-iters 1000
	cd $(CODE) && $(PY) -m experiments.run_ga_sensitivity --n-scenarios 3 \
	    --n-seeds 2 --n-gens 25 --mc-iters 800
	cd $(CODE) && $(PY) -m experiments.run_elasticity_lhs --n-scenarios 12 \
	    --n-seeds 3 --n-draws 256 --n-gens 15 --pop-size 24 --mc-iters 500
	cd $(CODE) && $(PY) -m experiments.make_paper_figures
	cd $(CODE) && $(PY) -m experiments.run_abm_diagnostics --reps 10
	cd $(CODE) && $(PY) -m experiments.run_structural_sensitivity
	cd $(CODE) && $(PY) -m experiments.measure_queueing
	cd $(CODE) && $(PY) -m experiments.run_validation_gof
	cd $(CODE) && $(PY) -m experiments.run_real_data_example \
	    --retail-path "../$(RETAIL)"

# Ordered recipe lines rather than prerequisites: `make -j` runs
# prerequisites concurrently, but restyle_figures, macros and the paper
# all read what the experiment runs write. The figures
# CODE/make_gui_figures.py draws (the GUI screenshots and the emergent
# heat map) are the only ones not regenerated here: they need a display.
paper-grade:
	$(MAKE) experiments-paper
	$(MAKE) figures-headless
	$(MAKE) macros
	$(MAKE) paper

all: deps test smoke

clean:
	rm -f $(PAPER).aux $(PAPER).log $(PAPER).out $(PAPER).bbl $(PAPER).blg
