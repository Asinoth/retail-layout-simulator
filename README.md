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
make test        # 233 pytest cases: MC engine vs closed form, Markov
                 #   properties, elasticity monotonicity, layout
                 #   feasibility and repair parity, GA determinism,
                 #   fixed-step determinism, fixture obstacles, the
                 #   arrival loop, calibrated rates, worker invariance,
                 #   invoice-drawn shopping lists, the basket-level
                 #   category test and its replica yardstick, common
                 #   search starts, the weight sweep, dataset discovery
                 #   and the search-seed replication
make verify      # architecture invariants (7 shop types x 5 sizes x 3
                 #   seeds, non-overlapping zones, aisle clearances,
                 #   keepouts, flood-fill reachability through item
                 #   access points) and the dataset pipeline smoke
                 #   (needs the three datasets; see Datasets below)
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

Its last step typesets the manuscript. The manuscript source is not in
this repository, so here that step prints a one-line notice and succeeds;
everything before it — the experiments, the figures and the macro file —
runs from this repository alone.

The deterministic experiments (Figures A/B/C, the elasticity sweep, the
Monte Carlo ground truth, the GA sensitivity sweep) take about three
hours on four workers and accept `--workers N` to run independent scenarios in
parallel; the output is identical for any worker count. The live
diagnostics (`run_abm_diagnostics`, `run_structural_sensitivity`,
`measure_queueing`, `run_validation_gof`) run the agent model in a seeded
fixed-step headless mode, so they are reproducible from their seeds and
also worker-count invariant, and take minutes rather than hours.

Two variables tune the driver — `WORKERS` and `RETAIL` for `make`,
`-Workers` and `-Retail` for `reproduce.ps1`:

```bash
make paper-grade WORKERS=8          # --workers 8 for every runner that takes it (default 1)
make paper-grade RETAIL=/data/online_retail_II.xlsx
                                    # --retail-path for Figure C and the live
                                    #   diagnostics (default: found automatically)
```

```powershell
.\reproduce.ps1 paper-grade -Workers 8 -Retail D:\data\online_retail_II.xlsx
```

A relative `RETAIL` path is taken from `CODE/`, where the runners run.

### Compute budget

Wall time of each run behind the shipped numbers, all run one after another
at four workers from the same committed tree on the authors' Windows 11
workstation. Times come from each run's `sidecar.json` where it records
one, otherwise from the start and end of the run:

| Experiment | Runner | Workers | Wall time |
|---|---|---|---|
| Figure A — regret and objective agreement | `run_synthetic_gt` | 4 | 34 min |
| Figure B — method comparison | `run_baseline_comparison` | 4 | 2 h 15 min |
| Figure C — UCI worked example | `run_real_data_example` | 1 (no `--workers`) | 2 min |
| MC-objective ground truth | `run_mc_groundtruth` | 4 | 14 min |
| GA operator sensitivity | `run_ga_sensitivity` | 1 (no `--workers`) | 12 min |
| Elasticity and weight sweeps | `run_elasticity_lhs` | 4 | 2 min |
| Markov order and perimeter ratio (live) | `run_abm_diagnostics` | 4 | 3 min |
| Structural sweep (live) | `run_structural_sensitivity` | 4 | 5 min |
| Queueing (live) | `measure_queueing` | 4 | 5 min |
| Goodness of fit, in-sample and held-out (live) | `run_validation_gof` | 4 | 5 min |
| Figure C across ten search seeds | `run_real_data_seeds` | 4 | 6 min |

About three and a half hours in all. Times are machine-dependent; outputs
are not: they follow from the seeds, not from the host's speed or the
worker count.

### Provenance of the shipped artifacts

Every run under `CODE/experiments/results/` records in its `sidecar.json`
the commit it ran from, and that the working tree was clean when it
started and when it finished (`git_dirty` and
`git_state_changed_during_run` both false). Those commits belong to the
authors' development history, which this repository does not carry, so
this is how they map onto the release:

| Recorded commit | Runs | Code relative to this release |
|---|---|---|
| `df2317c` | goodness of fit | identical |
| `cdff891` | ABM diagnostics, structural sweep, queueing | differs only in `run_validation_gof.py` (adds the size-matched replicas), `make_results_macros.py` and the added `tests/test_gof_replicas.py`, none of which those runs use |
| `4ac7843` | Figure C across search seeds | differs in those files and in the live agents' shopping-list law and category test (`customer.py`, `dataset_calibration.py`, `dataset_validation.py`, `retail_literature.py`, `sim_analytics.py`, `viz_edit.py`, `experiments/_live_store.py`), in the four live runners and in three further test files. The run seeds its store's calibration through `dataset_calibration.py`, whose new entries only the live agents read; its Monte Carlo search runs no agents, and Figure C re-run from this release reproduces its `results.csv` byte for byte |
| `f18e38f` | MC ground truth | differs in all of the above and in `run_real_data_example.py` (split into reusable steps; its outputs are byte-identical for the same arguments) and the added `run_real_data_seeds.py` and its test. The run reads `retail_literature.py` only for constants this release leaves unchanged (it removes the two old shopping-list constants) and uses none of the other files |
| `1292e1e` | Figures A, B and C, the elasticity and weight sweeps, the GA sensitivity sweep, the paper figures | differs in all of the above and in `run_mc_groundtruth.py`. Figure C uses its runner and the calibration seeding, and re-running it from this release reproduces its `results.csv` byte for byte; the synthetic-scenario runs read `retail_literature.py` only for unchanged constants and use none of the other files |

Re-running any runner from this release therefore reproduces its shipped
artifact (see *Determinism* below).

## Datasets

None of the three calibration sources is redistributed here. Download
each from its publisher and put it under `DATASETS/` (beside `CODE/`)
under the name listed, or point the listed environment variable at it.
Every entry point — the GUI's Dataset chooser, `make verify`, the figure
scripts, the experiment runners and the tests — finds the data through one
module, `CODE/dataset_paths.py`, so they agree on where it is, and a
missing file is reported with every path that was tried.
`RETAIL_DATASETS_DIR` names a datasets folder to search in place of
`DATASETS/`.

Figure C and the live diagnostics need the UCI workbook; the dataset
smoke in `make verify` needs all three; the dataset panels of the layout
previews use UCI and Omnichannel and are skipped without them; `make
test` runs without any of them, skipping the tests that need the UCI
workbook.

**UCI Online Retail II** (Chen, 2012) — transactional; basket, revenue,
price, co-purchase and arrival calibration.

- Source: <https://archive.ics.uci.edu/dataset/502/online+retail+ii>
- DOI: [10.24432/C5CG6D](https://doi.org/10.24432/C5CG6D)
- Licence: see the source repository
- Looked for as: `DATASETS/online_retail_II.xlsx` (the name UCI
  distributes); `Online Retail II.xlsx`, `online_retail_II.xls` and the
  older local name `UCI Online Retail II .xlsx.xlsx` are also accepted
- Override: `UCI_RETAIL_XLSX=<path to the workbook>`; the runners also
  take `--retail-path`

**Omnichannel Retail Datasets** (Bhowmick and Pazour, 2024) — aggregate in-store
behaviour; purchase probabilities, dwell, impulse rates, hour×weekday
arrivals.

- Source: <https://github.com/JoyjitBhowmick/Omnichannel-Retail-Datasets>
- Licence: see the source repository
- Looked for as: the folder `DATASETS/Omnichannel-Retail-Datasets-main/`
  (GitHub's ZIP download) or `DATASETS/Omnichannel-Retail-Datasets/` (a
  clone)
- Override: `OMNICHANNEL_DIR=<path to the folder>`

**OpenTraj, ETH scene** (Amirian et al., ACCV 2020) — pedestrian
trajectories; walking-speed and dwell distributions.

- Source: <https://github.com/crowdbotp/OpenTraj>
- Licence: see the source repository
- Looked for as: `DATASETS/OpenTraj-master/datasets/ETH/seq_eth/obsmat.txt`
  (GitHub's ZIP download), or the same path under `DATASETS/OpenTraj/` (a
  clone)
- Override: `OPENTRAJ_ETH_OBSMAT=<path to obsmat.txt>`

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
