# CLAUDE.md — retail-layout simulator for TOMACS

You're working on a 2D retail-shop simulator whose academic goal is a
TOMACS submission demonstrating that better item placement produces a
measurable, dataset-calibrated, projected revenue lift in a confined
retail space. The user is preparing this for ACM Transactions on
Modeling and Computer Simulation (TOMACS) — the bar is calibration +
validation + reproducibility, not just "it runs".

Always prioritize scientific defensibility over feature surface area.
Reviewers will read `retail_literature.py` alongside the paper.

---

## What the simulator does

1. User builds a 2D shop floor plan (items, walls, sections, multi-floor
   support via connectors/stairs) on the **Layout** tab.
2. User loads a real dataset on the **Dataset** button — UCI Online
   Retail II for transactional realism, ATC Shopping Mall trajectories
   for spatial realism. The pipeline calibrates simulator parameters
   against the data and stamps provenance (SHA-256 of source, schema
   version, adapter version).
3. User runs a customer-flow micro-simulation. Pedestrians spawn (now
   with hour-of-day non-homogeneous Poisson when a dataset is loaded),
   walk the shop, visit items, check out or abandon.
4. **Optimize** button runs a 5-phase pipeline:
   PRE-window data collection → analytical engines (Monte Carlo
   baseline, sensitivity / tornado, Markov chain, GA layout
   optimization) → apply best layout → POST-window data collection →
   A/B test PRE vs POST with Welch's t-test + KS-2sample + Cohen's d.
5. **Validation** tab shows empirical-vs-simulated distribution overlays
   (basket size, per-visit revenue, category share, walking speed) with
   KS / chi-square goodness-of-fit tests at α=0.05.

---

## Architecture (flat module layout — no packages)

### Simulation core
- `simulation.py` — `CustomerFlowSimulation`: spawn loop, threading,
  heat-map maintenance, customer despawn. **NHPP arrivals via
  `hourly_profile` + `sim_clock_start_hour`** (defaults uniform/9:00).
- `customer.py` — `Customer` agent state machine
  (`entering / moving / shopping / checking_out / exiting`). The
  `_handle_connector_arrival()` helper is shared between `moving` and
  `exiting` states — do NOT re-inline it.
- `customer_pathfinding.py`, `sim_geometry.py`, `sim_analytics.py` —
  pathfinding, wall caching, analytics rollups.
- `sim_calibration.py` — empirical Markov transition matrix, Monte
  Carlo engine (`mc_engine`), parameter extraction from analytics.
  **MC engine is daily-aggregated; intra-day NHPP is live-simulator
  only** (mean-preserving normalization keeps daily totals correct).

### Dataset pipeline (NEW — designed for TOMACS rigor)
- `dataset_schema.py` — typed contracts (`TRANSACTIONAL_FIELDS`,
  `TRAJECTORY_FIELDS`), `FieldStatus`, `ValidationReport`. Versioned
  via `SCHEMA_VERSION`.
- `dataset_adapters.py` — `OnlineRetailIIAdapter` (UCI), `GenericTransactionalAdapter`,
  `ATCShoppingMallAdapter`, `GenericTrajectoryAdapter`. Auto-detection
  via `detect_schema_kind()` + `best_*_adapter()`. Excel multi-sheet
  via `list_excel_sheets()` + `read_excel_sheets()`.
- `dataset_calibration.py` — `calibrate_transactional()` returns
  `CalibratedParams`, `calibrate_trajectory()` returns `SpatialParams`.
  Each has a `.seed_into(sim)` side effect that writes empirical
  distributions straight into `sim.analytics` so MC/Markov/GA see real
  values from t=0.
- `dataset_layout.py` — `build_layout_from_calibration()`: clears
  **every** floor (not just `self.current_floor` — that bug bit us
  once), groups items by inferred category, builds `Section_<cat>`
  walls, places items grid-style inside their section.
- `dataset_validation.py` — KS-2sample on basket / revenue /
  inter-arrival; chi-square on category visit shares with sparse-bin
  merging. Returns structured `ValidationResult`.
- `dataset_provenance.py` — SHA-256 of source, schema version,
  adapter version, currency, seed, Python+platform. Stored under
  `analytics['provenance']` (transactional) and
  `analytics['spatial_provenance']` (trajectory).

### Scientific constants (THE module reviewers will read)
- `retail_literature.py` — every coefficient that used to be inline
  magic. Citations dict, GA composite weights (equal-weighted per
  Dawes 1979), elasticity midpoints from Hui 2009 / Hui Inman 2013 /
  Larson 2005 / Sorensen 2009 / Botsali Peters 2005, abandonment
  recovery coefficients, default operating hours, hourly profile
  fallback. **GA positive weights sum to 1.0 (asserted at import).**

### UI (Tk + matplotlib)
- `visualizer.py` — `ShopVisualizer` (mega-class via mixins).
  Multi-floor data lives in `self.floors[fid]`; `self.items / .walls
  / .prices` are **property proxies onto floor 1 only** — do not use
  them to mutate other floors. `WM_DELETE_WINDOW` is wired through
  `_on_window_close()` which cancels every tracked after-id
  (`_analytics_after_id`, `_sim_gui_drain_after_id`, `_resize_after_id`,
  `_opt_after_id`, `_heat_after_id`) and calls `sim.hard_stop()` before
  destroying root.
- `viz_layout.py`, `viz_edit.py`, `viz_events.py`, `viz_addops.py`,
  `viz_generate.py`, `viz_simtab.py`, `viz_heatmap.py`,
  `viz_metrics.py`, `drag.py` — layout editor, drag-and-drop, dataset
  generation, simulation tab, heat map, metrics panel.
- `viz_projections.py`, `viz_sensitivity.py`, `viz_markov.py`,
  `viz_ga.py`, `viz_ga_run.py`, `viz_whatif.py` — analytical tabs.
- `viz_optimize.py`, `viz_optimize_helpers.py`,
  `viz_optimize_results.py` — the optimize pipeline + progress dialog
  + report renderer. **Always `customer_simulation.hard_stop()` before
  the optimize pipeline runs**, otherwise the paused worker thread
  floods `_gui_queue` with canvas redraws that starve the main loop
  (was the original 2 %% hang).
- `viz_dataset.py` — dataset orchestrator (file → adapter → validation
  dialog → calibrate → layout → seed → provenance → refresh validation
  tab). Has both transactional and trajectory branches.
- `viz_validation.py` — **Validation tab** with provenance pane,
  KS/chi-square table, four empirical-vs-simulated distribution
  panels (basket / revenue / category / walking speed).

---

## Conventions and gotchas

1. **Multi-floor properties are floor-1 only.** `self.items.clear()`
   clears floor 1; if you need every floor, iterate `self.floors`.
   This bit the dataset loader; fix in `dataset_layout.py`.
2. **`pause_simulation()` doesn't stop the worker thread.** Use
   `hard_stop()` when downstream code is about to do heavy main-thread
   work (optimize pipeline). It joins the thread within one frame.
3. **All tk.after callbacks must be tracked + cancelled in
   `_on_window_close`.** Otherwise Tk emits `bgerror invalid command
   name "...callback"` after the user closes the window.
4. **`Customer._handle_connector_arrival()` is the ONLY place the
   floor-teleport happens.** Both `moving` and `exiting` states route
   through it. If you add a new state that might be on a non-ground
   floor when an `_set_exit_target` is called, route through this
   helper too or the customer will freeze on the stair tile forever.
5. **The 2 %% optimize hang is fixed but the root cause is brittle.**
   `_create_optimization_progress_dialog` uses `dialog.update()` (not
   `update_idletasks()`) so the bar repaints. `pump_events()` must be
   called between MC chunks.
6. **`retail_literature.py` is load-bearing for the paper.** Every
   magic number lives there with a citation. Do not re-inline.
7. **Provenance is two records** (`provenance` for transactional,
   `spatial_provenance` for trajectory) so loading both doesn't
   overwrite either.
8. **`analytics_latest.json` is regenerated on every `_stop_simulation`.**
   It can hit several MB; not a bug, just expected.

---

## Datasets the paper uses

- **UCI Online Retail II** — `online_retail_II.xlsx` from
  https://archive.ics.uci.edu/dataset/502/online+retail+ii
  Two sheets (Year 2009-2010, Year 2010-2011); the load dialog
  defaults to both checked → concatenated. Calibrates basket size,
  per-visit revenue, prices, co-purchase pairs, arrival rate, hourly
  profile, return-customer rate.
  *Limitation:* conversion rate is unobservable from transactional
  data (no non-buyers); it's an explicit assumption (default 0.30)
  flagged in `analytics['calibration']['conversion_rate_source']`.

- **ATC Shopping Mall trajectories** (Brščić et al., IROS 2013) —
  https://dil.atr.jp/sets/ATC/. ~92 days of pedestrian tracking.
  8-column headerless CSV per day. Calibrates walking-speed
  distribution, per-track dwell, traffic-density heat map.
  *Limitation:* ATC's actual mall layout differs from any user shop;
  spatial calibration rescales the bbox to fit the current shop. The
  paper validates *distributions* (speed, dwell) not point-level
  spatial fidelity.

---

## Tier-1 experiments (NEW)

Headless reproducible runners for the two paper-grade figures.

### Modules
- `synthetic_shops.py` — parameterized synthetic shop generator
  (8–12 items, sections sized to fit items, per-item
  literature-cited elasticities). `analytical_revenue` is the
  closed-form revenue model the oracle optimizes. Includes a smooth
  overlap penalty (overlap-area-proportional) so L-BFGS-B can
  navigate it.
- `oracle.py` — multi-start L-BFGS-B over `analytical_revenue` with
  section bounds. Anti-circular: imports neither `simulation` nor
  `viz_ga` nor `viz_optimize`. Verified by
  `test_oracle_anti_circularity()`.
- `baselines.py` — `random_valid`, `perimeter_only`,
  `popularity_rank`, `greedy_swap`. All respect section bounds.
- `experiments/_common.py` — `HeadlessShop` (Tk-free shop with the
  GA / projection mixins), `build_headless_shop`, `run_ga_headless`,
  `paired_mc_revenue`, `bootstrap_ci`, `write_sidecar`. Every
  experiment run writes a JSON sidecar with git SHA + RNG seed +
  elasticity snapshot for reproducibility.
- `experiments/run_synthetic_gt.py` — Figure A (regret + Spearman).
- `experiments/run_baseline_comparison.py` — Figure B (paired-MC
  bar chart of GA vs baselines vs oracle).

### Smoke commands
```powershell
python -m experiments.run_synthetic_gt --n-scenarios 3 --n-seeds 2 `
    --mc-iters 200 --n-gens 8 --pop-size 16 --n-spearman-samples 40

python -m experiments.run_baseline_comparison --n-scenarios 3 `
    --n-seeds 2 --mc-iters 200 --n-gens 8 --pop-size 16
```

### Paper-grade commands
```powershell
python -m experiments.run_synthetic_gt --n-scenarios 30 --n-seeds 10 `
    --mc-iters 2000 --n-gens 25 --pop-size 30

python -m experiments.run_baseline_comparison --n-scenarios 30 `
    --n-seeds 10 --mc-iters 2000 --n-gens 25 --pop-size 30
```

### Gotchas for the experiments
1. **`paired_mc_revenue` requires the SAME `seed` across candidates**
   on the same scenario; otherwise MC noise dominates. The runners
   enforce this via `mc_seed = 5000 + scenario_idx * 100 + seed`.
2. **Spearman test must use overlap-repaired random layouts** —
   raw random samples have overlapping items which the GA's
   overlap_penalty dominates, polluting the correlation. Fixed in
   `run_synthetic_gt.py` via `baselines._repair_within_section`.
3. **Oracle layout must be overlap-validated** before MC scoring.
   The smooth overlap penalty in `analytical_revenue` handles this
   during optimization; if you change the penalty, re-verify oracle
   produces overlap-free layouts with `python oracle.py`.
4. **Substantive finding to preserve in the paper**: Spearman rho
   between GA fitness and analytical revenue is ≈ 0 across
   scenarios. This is NOT a bug. It validates the simulation-based
   approach over closed-form: the simulator captures behavioral
   components (dwell, flow, queueing) the closed-form model omits.

---

## Done in the most recent sessions

- Fixed the optimize-hangs-at-2%% bug (hard_stop + drain gui_queue +
  full `dialog.update()`).
- Fixed customer-stuck-on-stairs (added `_handle_connector_arrival`
  shared between `moving` / `exiting`).
- Fixed `bgerror` on window close (`WM_DELETE_WINDOW` handler).
- Fixed `_clear_current_data` writing to the wrong object.
- Fixed `viz_sensitivity` recomputing `rev_std` from `rev_mean`.
- Built the full dataset pipeline (schema → adapters → calibration →
  layout → validation → provenance).
- Built the cited-constants module `retail_literature.py` and
  refactored GA composite + score-to-conversion / impulse / basket
  elasticities + abandonment recovery to use it (now applied in
  `viz_optimize`, `viz_whatif`, AND `viz_ga_run`).
- Built `ATCShoppingMallAdapter` + `calibrate_trajectory()` +
  Validation tab speed-distribution panel.
- Wired non-homogeneous Poisson arrivals into the live spawn loop.
- **Tier-1 methodology validation**: synthetic ground truth +
  baselines + headless experiment harness. See dedicated section
  above.
- **Tier-1 paper-grade run DONE (2026-05-18).** 30 scenarios × 10
  seeds × 2000 MC iters × 25 gens × pop 30. Wall ~1h 42m each.
  Headline results:
  - **GA regret vs analytical oracle**: median 1.46%, mean 1.58%,
    p5–p95 [0.34%, 3.28%]. GA recovers ~98% of analytical optimum.
  - **GA significantly beats every informed baseline AND the
    analytical-optimum oracle** (n=300 paired-MC, all CIs exclude 0):
    GA − oracle = **+\$1,393 [+\$872, +\$1,975]** (the paper's most
    novel finding). GA − popularity_rank = +\$683 [+\$643, +\$723].
  - **Spearman ρ(GA-fit, analytical-R) = +0.107** (median +0.128,
    max +0.385 sig.). Substantive: simulator and closed-form model
    rank layouts on partially-divergent axes; the simulator catches
    dwell/flow/queue effects the analytical model omits — which is
    why GA outperforms the analytical oracle under MC.
  - Artefacts: `experiments/results/synthetic_gt_20260518-192904/`
    and `experiments/results/baseline_comparison_20260518-212709/`
    (CSV, PNG, sidecar JSON each).
  - Paper text updated: `paper_outline.tex` §Validation +
    §Results headline numbers; `patch_notes.tex` §Tier-1
    paper-grade run results.

---

## Open / deferred work

In rough priority for the paper:

1. **Paper-quality figure styling** — current matplotlib outputs are
   functional but not paper-grade. Editorial pass: consistent
   typography, color-blind palette, descriptive captions, vector
   PDF output.
2. **Sensitivity sweep over the cited-elasticity uncertainty bands**
   (Hui 2009: 50-75 %% conversion lift; Hui Inman 2013: 50-70 %%
   impulse gap). Latin Hypercube over 6-D band, ~200 evaluations.
3. **UCI Online Retail II + ATC worked example (Figure C)** —
   end-to-end calibrate against the real datasets, run the optimize
   pipeline, report projected lift. The "this actually works on
   real data" showpiece for §Application.
4. **Fix oracle non-convergence on synthetic_gt scenarios 21 and 23**
   (negative-regret seeds suggest oracle's L-BFGS-B multi-start
   needs n_restarts > 6 or differential evolution). Doesn't change
   the substantive story but cleans the numbers.
5. **MC engine hourly aggregation** — currently daily, intra-day
   NHPP is live-only. Modest accuracy gain, ~150-line refactor.
6. **Higher-order or non-Markov path model** — Hui 2009 explicitly
   argues against memoryless. Either defend the first-order
   assumption with a sentence + citation, or move to a mixture model.
7. **Fit elasticities from data instead of citing literature** — much
   bigger workstream; treats simulator output as a function of
   coefficients and inverts against held-out UCI. Defer unless
   reviewers push back.
8. **`viz_dataset` still calls `messagebox.showinfo` synchronously
   inside threads in a few code paths.** Audit for Tk-from-worker
   violations.

---

## How to run

```powershell
python main.py
```

Asks for shop dimensions (area / width / depth), opens the visualizer.
For a calibration run, click `Dataset` → pick the UCI .xlsx → both
sheets → adjust assumed conversion → Calibrate. Then `Optimize`. Then
`Validation` tab → `Run validation now`.

`openpyxl` is required for .xlsx (`pip install openpyxl`).

---

## When in doubt

- Read `retail_literature.py` before changing any coefficient.
- Read this file's "Conventions and gotchas" before touching the
  multi-floor model, the sim thread, or the optimize pipeline.
- Ask the user before adding a major analytical feature; ad-hoc
  additions weaken the paper's defensibility.
