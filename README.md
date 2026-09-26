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
- `requirements.txt` — the direct dependencies (numpy/scipy/pandas/
  matplotlib/openpyxl); `requirements-lock.txt` pins their whole resolved
  closure as the experiments ran it (CPython 3.13.2, Windows on AMD64).
  Every experiment sidecar and dataset provenance record stamps the
  resolved versions at run time and checks them against the lock.

The manuscript source is withheld while the paper is under submission. Its
generated macro file (`paper_results_macros.tex`) and budget table
(`paper_table_budgets.tex`) accompany the submission as supplementary
material; `make macros` regenerates both from the artifacts shipped here,
byte for byte.

## Reproduce

Cross-platform driver: `make <target>` (Unix/macOS) or
`.\reproduce.ps1 <target>` (Windows PowerShell). Both wrap the same steps.

```bash
make deps        # pip install -r requirements.txt
make deps-lock   # pip install -r requirements-lock.txt (the exact environment)
make test        # 896 pytest cases: the Monte Carlo engine against its
                 #   closed-form mean, the anchored layout objective,
                 #   layout feasibility and the floor-plan invariants,
                 #   GA and fixed-step determinism, worker-count
                 #   invariance, fixture obstacles, the arrival process,
                 #   calibrated rates, invoice-drawn shopping lists, the
                 #   goodness-of-fit tests and their replica yardstick,
                 #   common search starts and equal budgets, the weight
                 #   sweep, the coefficient registry, the macro
                 #   generator's artifact checks and every runner's
                 #   summary
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

The runs behind the shipped numbers took under seven hours of wall time
in all (table below). Runners that take `--workers N` spread independent
scenarios, seeds or replications over N processes. Every runner whose
worker-count invariance is tested (`CODE/tests/test_runner_workers.py`,
`test_live_runner_workers.py` and the per-runner tests) gives identical
output at any N; the live protocol study takes `--workers` but has no such
test. The live runners run the agent model in a seeded fixed-step headless
mode, so they reproduce from their seeds whatever the host's speed.

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

Wall time of each run behind the shipped numbers, from its `sidecar.json`,
run one at a time on the authors' workstation (Intel Core i7-10700KF,
16 logical processors, 16 GB of memory, Windows 11). Runners without a
`--workers` option run in one process:

| Experiment | Runner | Workers | Wall time |
|---|---|---|---|
| Recovery and objective agreement | `run_synthetic_gt` | 4 | 39 min |
| Method comparison | `run_baseline_comparison` | 4 | 2 h 23 min |
| Monte Carlo ground truth | `run_mc_groundtruth` | 4 | 19 min |
| GA operator and annealer sweeps | `run_ga_sensitivity` | 4 | 43 min |
| Elasticity and weight sweeps | `run_elasticity_lhs` | 4 | 3 min |
| Objective alignment | `run_objective_alignment` | 4 | 1 min |
| Paper figures (realized scores) | `make_paper_figures` | 1 | 2 min |
| Live protocol study | `run_live_protocol_study` | 4 | 1 h 1 min |
| Markov order and perimeter ratio (live) | `run_abm_diagnostics` | 4 | 3 min |
| Structural sweep (live) | `run_structural_sensitivity` | 4 | 5 min |
| Structural sweep's exit-route check (live) | `run_structural_exit_check` | 1 | 7 min |
| Queueing (live) | `measure_queueing` | 4 | 4 min |
| Goodness of fit, in sample and held out (live) | `run_validation_gof` | 4 | 2 min |
| Emergent heat map (live) | `make_heatmap_figure` | 1 | 1 min |
| UCI worked example | `run_real_data_example` | 1 | 2 min |
| UCI example without anonymous invoices | `run_real_data_example --exclude-anonymous` | 1 | 2 min |
| UCI example re-scored in closed form | `run_figc_rescore` | 1 | under 1 min |
| UCI example across ten search seeds | `run_real_data_seeds` | 4 | 6 min |
| UCI budget sweep | `run_real_data_budget` | 4 | 43 min |
| UCI input uncertainty | `run_input_uncertainty` | 4 | 9 min |
| UCI held-out transfer | `run_heldout_transfer` | 4 | 6 min |
| UCI calendar profile | `profile_uci` | 1 | 1 min |

Times are machine-dependent; outputs follow from the seeds, not from the
host's speed.

### Provenance of the shipped artifacts

Every run under `CODE/experiments/results/` records in its `sidecar.json`
the commit it ran from, and that the working tree was clean when it
started and when it finished (`git_dirty` and
`git_state_changed_during_run` both false). Those commits belong to the
authors' development history, which this repository does not carry, so
this is how they map onto the release:

| Recorded commit | Runs | Code relative to this release |
|---|---|---|
| `768bd66` | UCI calendar profile | differs only in the added `tests/test_profile_uci.py` |
| `e14bf23` | structural sweep's exit-route check, the UCI layout figure | differs only in comments and labels of `retail_literature.py` (no value changes), comments of `run_elasticity_lhs.py` and `run_real_data_example.py`, the macro generator, the added `profile_uci.py`, the tests and the `Makefile` / `reproduce.ps1` targets; neither run executes the changed code |
| `0f895e6` | UCI example re-scored in closed form | differs in the files listed for `e14bf23`, in `run_structural_sensitivity.py` (writes the exit-route tally to its summary) and in the added `run_structural_exit_check.py`; the re-scoring executes none of them |
| `8129169` | every other run | differs in the files above, in `figstyle.py` (adds the layout figure's print width) and in exit-route bookkeeping: `customer.py` records why an unpaid agent left and `sim_analytics.py` tallies it, which `run_structural_sensitivity.py` now writes to its summary. Nothing reads the record to decide anything and it draws no random number; the exit-route check re-ran two replications of every setting of the structural sweep from `e14bf23` and reproduced the sweep's counts and revenue exactly. The added runners (`run_figc_rescore.py`, `run_structural_exit_check.py`, `make_layout_figure.py`, `profile_uci.py`) are used by no run of this commit |

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

- **GUI figures** (`figs/gui_layout.png`, `gui_simulation.png`) require
  a live Tk display and are regenerated separately by
  `cd CODE && python make_gui_figures.py` on a machine with a desktop; no
  reported number or paper figure comes from them. The emergent heat map
  is drawn headless from a seeded run (`experiments.make_heatmap_figure`),
  and `make figures-headless` produces the rest, including the floor-plan
  previews.
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
