# retail-layout-simulator

An agent-based retail-layout simulator and optimization pipeline, built as
the software artifact for a paper under submission to *ACM Transactions on
Modeling and Computer Simulation*.

## What this is

A multi-floor, 2D agent-based simulator for retail shop layouts, with
dataset calibration (UCI Online Retail II, an omnichannel transactional
bundle, OpenTraj/ETH trajectories), Monte Carlo / Markov / genetic-algorithm
analytical engines, and a synthetic-ground-truth methodology validation
pipeline.

Design goals, in priority order: every behavioural coefficient traceable to
the empirical retail-science literature, and every reported number
reproducible from a hashed data source and a recorded random seed.

### Highlights

- **Literature-calibrated behaviour.** Every coefficient lives in a single
  auditable module with its citation attached, rather than inline as a
  magic number.
- **Agent model.** Customers follow a piecewise-deterministic Markov
  process over a five-state behavioural automaton, with A* pathfinding,
  non-homogeneous Poisson arrivals on an hour-of-day intensity profile,
  and multi-lane checkout with least-loaded lane selection.
- **Floor-plan engine.** Generates the three archetypes from the retail
  literature — grid, racetrack, freeform — guaranteeing non-overlapping
  fixtures, minimum aisle widths, and door-to-item reachability, all
  verified headlessly by a flood-fill check.
- **Synthetic ground truth.** Shops are generated with a known closed-form
  optimum so the optimizer's *regret* can be measured, not asserted.
- **Equal-budget comparators.** The GA is benchmarked against random
  search and simulated annealing at an identical evaluation budget, plus
  four informed heuristic baselines — so any advantage isn't just compute.
- **Common random numbers** and multi-seed final selection throughout the
  paired comparisons, so Monte Carlo noise doesn't pick the winner.

## Repository layout

- `CODE/` — all Python (flat module layout by design; run everything from
  inside `CODE/`). `CODE/experiments/` holds the headless, reproducible
  experiment runners; `CODE/tests/` the pytest numeric-core suite.
- `DATASETS/` — the calibration datasets, **not included**: they are
  third-party and downloaded separately (see *Datasets* below).
- `figs/` — generated figures.
- `requirements.txt` — pinned dependencies (Python 3.13.2; numpy/scipy/
  pandas/matplotlib/openpyxl). Every experiment sidecar and dataset
  provenance record also stamps the resolved versions at run time.

## Reproduce

Cross-platform driver: `make <target>` (Unix/macOS) or
`.\reproduce.ps1 <target>` (Windows PowerShell). Both wrap the same steps.

```bash
make deps        # pip install -r requirements.txt
make test        # pytest numeric-core suite (~5s): MC vs closed-form,
                 #   Markov row-stochasticity/absorption, elasticity
                 #   monotonicity, layout feasibility, GA determinism
make verify      # architecture invariants (7 shop types x 4 sizes x 3
                 #   seeds + flood-fill reachability) and the dataset
                 #   pipeline smoke (needs DATASETS/)
make smoke       # fast end-to-end experiment smokes (minutes)
```

```powershell
.\reproduce.ps1 deps ; .\reproduce.ps1 test ; .\reproduce.ps1 smoke
```

**Full paper-grade regeneration from scratch** (hours of compute — 30
scenarios x 10 seeds x 2000 MC iters for Figures A/B, plus the MC
ground-truth, GA-sensitivity, LHS, and real-data runs):

```bash
make experiments-paper   # the experiment runs
make figures-headless    # then the display-free figures
```

`make macros` regenerates the numeric result macros from the per-run
artifact files (`results.csv` / `sidecar.json` / `summary.json`) shipped
under `CODE/experiments/results/`. It refuses to run if those artifacts
are missing, and rejects undersized runs, so a quick smoke can never
silently replace paper-grade numbers with toy ones.

## Datasets

Not redistributed here — each is third-party with its own terms. The
loader resolves them from a sibling `DATASETS/` directory:

- **UCI Online Retail II** — <https://archive.ics.uci.edu/dataset/502/online+retail+ii>
  (transactional calibration: basket size, per-visit revenue, prices,
  co-purchase pairs, arrival rate, hourly profile)
- **OpenTraj / ETH** — pedestrian trajectories (walking-speed and dwell
  distributions)

Every load records a provenance stamp: SHA-256 of the source file, schema
version, adapter version, currency, seed, and the resolved Python and
package versions.

## Running the app

```bash
cd CODE && python main.py
```

Asks for shop dimensions, then opens the visualizer. `Dataset` calibrates
and lays out automatically; `Optimize` runs the five-phase pipeline; the
`Validation` tab runs the goodness-of-fit tests.

## Notes

- **GUI figures** (`figs/gui_layout.png`, `gui_simulation.png`,
  `emergent_heatmap.png`) require a live Tk display and are regenerated
  separately by `python CODE/make_gui_figures.py` on a machine with a
  desktop; the headless targets above do not produce them.
- **Determinism** is guaranteed on the single-threaded headless
  experiment path (from which every reported number comes). The live GUI
  runs a simulation worker thread and does not carry a bit-identical
  reproducibility guarantee — see the `run_ga_headless` docstring and
  `CODE/tests/test_ga_determinism.py`.
- **Terminology:** the code and result files use the historical identifier
  `oracle` (`oracle.py`, the `oracle_R` column, the `oracle` method rows)
  for what the paper calls the *analytical reference solution*; the name
  predates the terminology change and is kept so the released artifact
  files stay byte-stable.

## Manuscript

The manuscript is **not included in this repository** while the paper is
under submission. It will be added, with its bibliography and generated
result macros, once the paper is published.

## Licence

MIT — see [LICENSE](LICENSE). © 2026 Alexandros Tsagkanos and
Chrysostomos Zeginis.
