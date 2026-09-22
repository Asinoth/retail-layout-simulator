# retail-layout-simulator

Agent-based retail-layout simulator and optimization pipeline, released as
the software artifact for an ACM TOMACS submission.

## What this is

A multi-floor, 2D agent-based simulator for retail shop layouts, with
dataset calibration (UCI Online Retail II, the Omnichannel bundle,
OpenTraj/ETH), Monte Carlo / Markov / GA analytical engines, and a
synthetic-ground-truth methodology validation pipeline. Every module
carries a docstring describing its contract; `CODE/retail_literature.py`
is the single source of truth for every literature-derived coefficient.

## Repository layout

- `CODE/` — all Python (flat module layout by design; run everything from
  inside `CODE/`). `CODE/experiments/` holds the headless, reproducible
  experiment runners; `CODE/tests/` the pytest suite.
- `DATASETS/` — the calibration datasets, downloaded separately (see
  *Datasets* below).
- `figs/` — figures and the JSON summaries the manuscript's numbers come
  from.
- `CODE/experiments/results/` — the per-run artifacts behind every
  reported number (`results.csv` / `sidecar.json` / `summary.json`).
- `requirements.txt` — pinned dependencies (Python 3.13.2; numpy/scipy/
  pandas/matplotlib/openpyxl). Every experiment sidecar and dataset
  provenance record also stamps the resolved versions at run time.

The manuscript source and its generated macro file are withheld while the
paper is under submission and are added on publication.

## Reproduce

Cross-platform driver: `make <target>` (Unix/macOS) or
`.\reproduce.ps1 <target>` (Windows PowerShell). Both wrap the same steps.

```bash
make deps        # pip install -r requirements.txt
make test        # 104 pytest cases: MC engine vs closed form, Markov
                 #   properties, elasticity monotonicity, layout
                 #   feasibility and repair parity, GA determinism,
                 #   fixed-step determinism, fixture obstacles, the
                 #   arrival loop, calibrated rates, worker invariance
make verify      # architecture invariants (7 shop types x 5 sizes x 3
                 #   seeds, non-overlapping zones, aisle clearances,
                 #   keepouts, flood-fill reachability through item
                 #   access points) and the dataset pipeline smoke
                 #   (needs DATASETS/)
make smoke       # fast end-to-end experiment smokes (minutes)
```

```powershell
.\reproduce.ps1 deps ; .\reproduce.ps1 test ; .\reproduce.ps1 smoke
```

**Rebuild the paper's numbers from existing artifacts:**

```bash
make macros             # regenerate the macro file from the newest artifacts
```

`make macros` reads the per-run artifact files shipped under
`CODE/experiments/results/` and refuses to run if they are missing or too
small to be paper-grade, so it can never silently replace the shipped
numbers with placeholders.

**Full paper-grade regeneration from scratch:**

```bash
make paper-grade        # experiments -> figures -> macros -> manuscript
```

Its last step typesets the manuscript and therefore needs the withheld
source; everything before it — the experiments, the figures and the macro
file — runs from this repository alone.

The deterministic experiments (Figures A/B/C, the elasticity sweep, the
Monte Carlo ground truth, the GA sensitivity sweep) take a few hours on
one core and accept `--workers N` to run independent scenarios in
parallel; the output is identical for any worker count. The live
diagnostics (`run_abm_diagnostics`, `run_structural_sensitivity`,
`measure_queueing`, `run_validation_gof`) run the agent model in a seeded
fixed-step headless mode, so they are reproducible from their seeds and
also worker-count invariant, and take minutes rather than hours.

## Datasets

Figure C and the live diagnostics need the UCI Online Retail II workbook
under `DATASETS/`; the Validation tab and the trajectory calibration
additionally use the Omnichannel bundle and OpenTraj/ETH. Sources and
licences are listed in the manuscript's data section; none of the three is
redistributed here.

### Notes

- **GUI figures** (`figs/gui_layout.png`, `gui_simulation.png`,
  `emergent_heatmap.png`) require a live Tk display and are regenerated
  separately by `cd CODE && python make_gui_figures.py` on a machine with
  a desktop; `make figures-headless` produces everything else, including
  the floor-plan previews.
- **Determinism.** Every headless path — the experiment runners and the
  fixed-step live simulation — reproduces bit-identically from its seeds
  on a single machine. The interactive GUI runs a simulation worker thread
  whose tick size follows the wall clock, so it carries no such guarantee;
  see the `run_ga_headless` and `run_headless` docstrings and
  `CODE/tests/test_ga_determinism.py` / `test_fixed_step.py`.
- **Terminology:** the code and result files use the historical
  identifier `oracle` (`oracle.py`, the `oracle_R` column, the `oracle`
  method rows) for what the manuscript calls the *analytical reference
  solution*; the name predates the paper's terminology change and is
  kept so the released artifact files stay byte-stable.
- **Running the app:** `cd CODE && python main.py`.
