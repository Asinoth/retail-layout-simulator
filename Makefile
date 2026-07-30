# Reproduction targets for the retail-layout simulator (audit R5.2).
#
# Unix / macOS:  make <target>
# Windows:       use reproduce.ps1 (identical steps) — `make` is usually
#                absent on Windows.
#
# Quick start (minutes):   make deps test smoke
# Regen result macros:     make macros            (from shipped artefacts)
# Full paper-grade regen:  make paper-grade       (HOURS; see below)
#
# NOTE: the manuscript is not shipped in this repository while the
# paper is under submission, so there is no `paper` (pdflatex) target.

PY    ?= python
CODE   = CODE
RETAIL ?= DATASETS/UCI Online Retail II .xlsx.xlsx

.PHONY: help deps test verify smoke figures-headless macros all \
        experiments-paper paper-grade clean

help:
	@echo "Targets:"
	@echo "  deps              pip install -r requirements.txt"
	@echo "  test              run the pytest numeric-core suite (~5s)"
	@echo "  verify            architecture invariants + dataset pipeline smokes"
	@echo "  smoke             fast end-to-end experiment smokes (~mins)"
	@echo "  figures-headless  regen the display-free figures"
	@echo "  macros            regen paper_results_macros.tex from artefacts"
	@echo "  experiments-paper paper-grade experiment runs (HOURS)"
	@echo "  paper-grade       experiments-paper + figures + macros"
	@echo "  all               deps test smoke"

deps:
	$(PY) -m pip install -r requirements.txt

test:
	cd $(CODE) && $(PY) -m pytest -q tests/

# The two standalone verification scripts cited in the paper's
# Verification section. They are not pytest tests (one needs the
# datasets, the other sweeps 84 generated floor plans), so they get
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

macros:
	cd $(CODE) && $(PY) make_results_macros.py

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
	cd $(CODE) && $(PY) -m experiments.run_elasticity_lhs
	cd $(CODE) && $(PY) -m experiments.make_paper_figures
	cd $(CODE) && $(PY) -m experiments.run_abm_diagnostics --reps 3 \
	    --seconds 120 --warmup 30
	cd $(CODE) && $(PY) -m experiments.run_structural_sensitivity \
	    --seconds 90 --warmup 30
	cd $(CODE) && $(PY) -m experiments.run_real_data_example \
	    --retail-path "../$(RETAIL)"

paper-grade: experiments-paper macros

all: deps test smoke

clean:
	rm -f *.aux *.log *.out *.bbl *.blg *.toc *.synctex.gz
