# Reproduction targets for the retail-layout simulator.
#
# Unix / macOS:  make <target>
# Windows:       use reproduce.ps1 (identical steps) — `make` is usually
#                absent on Windows.
#
# Quick start (minutes):   make deps test smoke
# Rebuild the PDF:         make macros paper      (needs figs/ present)
# Full paper-grade regen:  make paper-grade       (HOURS; see below)
#
# Variables:
#   RETAIL   path to the UCI Online Retail II workbook, absolute or
#            relative to CODE/ (the runners run there). Empty (the
#            default): the runners find it themselves, under DATASETS/ or
#            wherever UCI_RETAIL_XLSX points; see CODE/dataset_paths.py.
#   WORKERS  worker processes for every runner that takes --workers.
#            Outputs are identical for any value; only wall time changes.

PY      ?= python
CODE     = CODE
PAPER    = TOMACS_submission
RETAIL  ?=
WORKERS ?= 1

# `--retail-path "<RETAIL>"` when RETAIL is set, nothing otherwise.
RETAIL_ARG = $(if $(RETAIL),--retail-path "$(RETAIL)",)

.PHONY: help deps deps-lock test verify smoke figures-headless macros paper \
        all experiments-paper paper-grade clean lock

help:
	@echo "Targets:"
	@echo "  deps              pip install -r requirements.txt"
	@echo "  deps-lock         pip install -r requirements-lock.txt (the"
	@echo "                    experiments' exact environment)"
	@echo "  lock              regen requirements-lock.txt from this environment"
	@echo "  test              run the pytest numeric-core suite (~2 min)"
	@echo "  verify            architecture invariants + dataset pipeline smokes"
	@echo "  smoke             fast end-to-end experiment smokes (~mins)"
	@echo "  figures-headless  regen the display-free paper figures"
	@echo "                    (plots + the floor-plan previews)"
	@echo "  macros            regen paper_results_macros.tex + paper_table_budgets.tex"
	@echo "  paper             pdflatex->bibtex->pdflatex x2 (needs figs/)"
	@echo "  experiments-paper paper-grade experiment runs (HOURS)"
	@echo "  paper-grade       experiments-paper + figures + macros + paper"
	@echo "  all               deps test smoke"
	@echo "Variables: RETAIL=<workbook path> (default: discovered),"
	@echo "           WORKERS=<n> (default: 1)"

deps:
	$(PY) -m pip install -r requirements.txt

# The environment the experiments ran in, every package pinned (resolved on
# the platform its header names; pip applies its environment markers).
deps-lock:
	$(PY) -m pip install -r requirements-lock.txt

# The whole resolved environment (requirements.txt and everything it pulls
# in), pinned; every experiment sidecar checks the environment against it.
lock:
	cd $(CODE) && $(PY) environment_lock.py

test:
	cd $(CODE) && $(PY) -m pytest -q -rfE tests/

# The two standalone verification scripts cited in the paper's
# Verification section. They are not pytest tests (one needs the
# datasets, the other sweeps 105 generated floor plans), so they get
# their own target rather than slowing `make test` down.
verify:
	cd $(CODE) && $(PY) architecture_smoke.py
	cd $(CODE) && $(PY) dataset_smoke.py

smoke:
	cd $(CODE) && $(PY) -m experiments.run_synthetic_gt --n-scenarios 3 \
	    --n-seeds 2 --mc-iters 200 --n-gens 8 --pop-size 16 --n-spearman-samples 40 \
	    --workers $(WORKERS)
	cd $(CODE) && $(PY) -m experiments.run_baseline_comparison --n-scenarios 3 \
	    --n-seeds 2 --mc-iters 200 --n-gens 8 --pop-size 16 --workers $(WORKERS)

figures-headless:
	cd $(CODE) && $(PY) -m experiments.make_paper_figures
	cd $(CODE) && $(PY) -m experiments.restyle_figures
	cd $(CODE) && $(PY) -m experiments.make_layout_previews

macros:
	cd $(CODE) && $(PY) make_results_macros.py

# The manuscript source is withheld from the public repository while the
# paper is under submission; there the target says so and succeeds, so
# paper-grade still completes everything the repository can rebuild.
paper:
ifeq ($(wildcard $(PAPER).tex),)
	@echo "paper: $(PAPER).tex is not in this repository (the manuscript source is withheld); nothing to typeset."
else
	pdflatex -interaction=nonstopmode -halt-on-error $(PAPER).tex
	bibtex $(PAPER)
	pdflatex -interaction=nonstopmode -halt-on-error $(PAPER).tex
	pdflatex -interaction=nonstopmode -halt-on-error $(PAPER).tex
endif

# ── Paper-grade experiment regeneration (HOURS of compute) ──────────────
# Reproduces every artefact the macros consume. Figure C and the four
# live diagnostics additionally need the UCI Online Retail II workbook
# (found under DATASETS/, or named by RETAIL / UCI_RETAIL_XLSX).
experiments-paper:
	cd $(CODE) && $(PY) -m experiments.run_synthetic_gt --n-scenarios 30 \
	    --n-seeds 10 --mc-iters 2000 --n-gens 25 --pop-size 30 --workers $(WORKERS)
	cd $(CODE) && $(PY) -m experiments.run_baseline_comparison --n-scenarios 30 \
	    --n-seeds 10 --mc-iters 2000 --n-gens 25 --pop-size 30 --workers $(WORKERS)
	cd $(CODE) && $(PY) -m experiments.run_mc_groundtruth --n-scenarios 6 \
	    --normal-budget 750 --big-budget 7500 --mc-iters 1000 --workers $(WORKERS)
	cd $(CODE) && $(PY) -m experiments.run_ga_sensitivity --n-scenarios 10 \
	    --n-seeds 3 --n-gens 25 --pop-size 30 --mc-iters 800 --workers $(WORKERS)
	cd $(CODE) && $(PY) -m experiments.run_elasticity_lhs --n-scenarios 12 \
	    --n-seeds 3 --n-draws 256 --n-gens 15 --pop-size 24 --mc-iters 500 \
	    --workers $(WORKERS)
	cd $(CODE) && $(PY) -m experiments.run_objective_alignment \
	    --workers $(WORKERS)
	cd $(CODE) && $(PY) -m experiments.make_paper_figures
	cd $(CODE) && $(PY) -m experiments.run_live_protocol_study \
	    --workers $(WORKERS) $(RETAIL_ARG)
	cd $(CODE) && $(PY) -m experiments.run_abm_diagnostics --reps 10 \
	    --workers $(WORKERS) $(RETAIL_ARG)
	cd $(CODE) && $(PY) -m experiments.run_structural_sensitivity \
	    --workers $(WORKERS) $(RETAIL_ARG)
	cd $(CODE) && $(PY) -m experiments.run_structural_exit_check $(RETAIL_ARG)
	cd $(CODE) && $(PY) -m experiments.measure_queueing \
	    --workers $(WORKERS) $(RETAIL_ARG)
	cd $(CODE) && $(PY) -m experiments.run_validation_gof \
	    --workers $(WORKERS) $(RETAIL_ARG)
	cd $(CODE) && $(PY) -m experiments.make_heatmap_figure $(RETAIL_ARG)
	cd $(CODE) && $(PY) -m experiments.profile_uci $(RETAIL_ARG)
	cd $(CODE) && $(PY) -m experiments.run_real_data_example $(RETAIL_ARG)
	cd $(CODE) && $(PY) -m experiments.run_real_data_example \
	    --exclude-anonymous $(RETAIL_ARG)
	cd $(CODE) && $(PY) -m experiments.run_figc_rescore
	cd $(CODE) && $(PY) -m experiments.make_layout_figure $(RETAIL_ARG)
	cd $(CODE) && $(PY) -m experiments.run_real_data_seeds \
	    --workers $(WORKERS) $(RETAIL_ARG)
	cd $(CODE) && $(PY) -m experiments.run_real_data_budget \
	    --workers $(WORKERS) $(RETAIL_ARG)
	cd $(CODE) && $(PY) -m experiments.run_input_uncertainty \
	    --workers $(WORKERS) $(RETAIL_ARG)
	cd $(CODE) && $(PY) -m experiments.run_heldout_transfer \
	    --workers $(WORKERS) $(RETAIL_ARG)

# Ordered recipe lines rather than prerequisites: `make -j` runs
# prerequisites concurrently, but restyle_figures, macros and the paper
# all read what the experiment runs write. The GUI screenshots
# CODE/make_gui_figures.py draws are the only figures not regenerated here:
# they need a display. The emergent heat map comes from a seeded headless
# run (experiments.make_heatmap_figure, in experiments-paper).
paper-grade:
	$(MAKE) experiments-paper
	$(MAKE) figures-headless
	$(MAKE) macros
	$(MAKE) paper

all: deps test smoke

clean:
	rm -f $(PAPER).aux $(PAPER).log $(PAPER).out $(PAPER).bbl $(PAPER).blg
