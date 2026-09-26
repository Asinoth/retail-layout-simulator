<#
.SYNOPSIS
  Reproduction driver for the retail-layout simulator,
  the Windows/PowerShell counterpart of the Makefile.

.DESCRIPTION
  Quick start (minutes):   .\reproduce.ps1 deps; .\reproduce.ps1 test; .\reproduce.ps1 smoke
  Rebuild the PDF:         .\reproduce.ps1 macros; .\reproduce.ps1 paper   (needs figs/)
  Full paper-grade regen:  .\reproduce.ps1 paper-grade                     (HOURS)

.PARAMETER Target
  deps | deps-lock | test | verify | smoke | figures-headless | macros | paper |
  experiments-paper | paper-grade | help

.PARAMETER Retail
  Path to the UCI Online Retail II workbook, absolute or relative to CODE/
  (the runners run there); the Makefile's RETAIL. Empty (the default,
  unless $env:RETAIL is set): the runners find it themselves, under
  DATASETS/ or wherever UCI_RETAIL_XLSX points; see CODE/dataset_paths.py.

.PARAMETER Workers
  Worker processes for every runner that takes --workers; the Makefile's
  WORKERS (default 1, unless $env:WORKERS is set). Outputs are identical
  for any value; only wall time changes.
#>
param(
  [ValidateSet('help','deps','deps-lock','lock','test','verify','smoke','figures-headless',
               'macros','paper','experiments-paper','paper-grade')]
  [string]$Target = 'help',
  [string]$Retail = $env:RETAIL,
  [int]$Workers = $(if ($env:WORKERS) { [int]$env:WORKERS } else { 1 })
)

$ErrorActionPreference = 'Stop'
$Root   = $PSScriptRoot
$Code   = Join-Path $Root 'CODE'
$Paper  = 'TOMACS_submission'
$PY     = 'python'
# '--retail-path <Retail>' when Retail is set, nothing otherwise.
$RetailArg = @()
if ($Retail) { $RetailArg = @('--retail-path', $Retail) }

function Invoke-In($dir, [string[]]$argv) {
  Push-Location $dir
  try { & $PY @argv; if ($LASTEXITCODE -ne 0) { throw "exit $LASTEXITCODE" } }
  finally { Pop-Location }
}

# Native commands ignore $ErrorActionPreference, so a failed pip, pdflatex
# or bibtex call would otherwise let the script continue and exit 0.
function Invoke-Native($exe, [string[]]$argv) {
  & $exe @argv; if ($LASTEXITCODE -ne 0) { throw "$exe exit $LASTEXITCODE" }
}

function Do-Deps    { Invoke-Native $PY @('-m','pip','install','-r',(Join-Path $Root 'requirements.txt')) }
# The environment the experiments ran in, every package pinned.
function Do-DepsLock { Invoke-Native $PY @('-m','pip','install','-r',(Join-Path $Root 'requirements-lock.txt')) }
# The whole resolved environment, pinned; every sidecar checks against it.
function Do-Lock    { Invoke-In $Code @('environment_lock.py') }
function Do-Test    { Invoke-In $Code @('-m','pytest','-q','-rfE','tests/') }
function Do-Smoke {
  Invoke-In $Code @('-m','experiments.run_synthetic_gt','--n-scenarios','3',
    '--n-seeds','2','--mc-iters','200','--n-gens','8','--pop-size','16',
    '--n-spearman-samples','40','--workers',"$Workers")
  Invoke-In $Code @('-m','experiments.run_baseline_comparison','--n-scenarios','3',
    '--n-seeds','2','--mc-iters','200','--n-gens','8','--pop-size','16',
    '--workers',"$Workers")
}
function Do-Verify {
  # The two standalone verification scripts cited in the paper's
  # Verification section (architecture invariants; dataset pipeline).
  Invoke-In $Code @('architecture_smoke.py')
  Invoke-In $Code @('dataset_smoke.py')
}
function Do-FiguresHeadless {
  Invoke-In $Code @('-m','experiments.make_paper_figures')
  Invoke-In $Code @('-m','experiments.restyle_figures')
  Invoke-In $Code @('-m','experiments.make_layout_previews')
}
function Do-Macros { Invoke-In $Code @('make_results_macros.py') }
# The manuscript source is withheld from the public repository while the
# paper is under submission; there the target says so and succeeds, so
# paper-grade still completes everything the repository can rebuild.
function Do-Paper {
  if (-not (Test-Path (Join-Path $Root "$Paper.tex"))) {
    Write-Host "paper: $Paper.tex is not in this repository (the manuscript source is withheld); nothing to typeset."
    return
  }
  Push-Location $Root
  try {
    Invoke-Native 'pdflatex' @('-interaction=nonstopmode','-halt-on-error',"$Paper.tex")
    Invoke-Native 'bibtex' @($Paper)
    Invoke-Native 'pdflatex' @('-interaction=nonstopmode','-halt-on-error',"$Paper.tex")
    Invoke-Native 'pdflatex' @('-interaction=nonstopmode','-halt-on-error',"$Paper.tex")
  } finally { Pop-Location }
}
function Do-ExperimentsPaper {
  Invoke-In $Code @('-m','experiments.run_synthetic_gt','--n-scenarios','30',
    '--n-seeds','10','--mc-iters','2000','--n-gens','25','--pop-size','30',
    '--workers',"$Workers")
  Invoke-In $Code @('-m','experiments.run_baseline_comparison','--n-scenarios','30',
    '--n-seeds','10','--mc-iters','2000','--n-gens','25','--pop-size','30',
    '--workers',"$Workers")
  Invoke-In $Code @('-m','experiments.run_mc_groundtruth','--n-scenarios','6',
    '--normal-budget','750','--big-budget','7500','--mc-iters','1000',
    '--workers',"$Workers")
  Invoke-In $Code @('-m','experiments.run_ga_sensitivity','--n-scenarios','10',
    '--n-seeds','3','--n-gens','25','--pop-size','30','--mc-iters','800',
    '--workers',"$Workers")
  Invoke-In $Code @('-m','experiments.run_elasticity_lhs','--n-scenarios','12',
    '--n-seeds','3','--n-draws','256','--n-gens','15','--pop-size','24',
    '--mc-iters','500','--workers',"$Workers")
  Invoke-In $Code @('-m','experiments.run_objective_alignment',
    '--workers',"$Workers")
  Invoke-In $Code @('-m','experiments.make_paper_figures')
  Invoke-In $Code (@('-m','experiments.run_live_protocol_study',
    '--workers',"$Workers") + $RetailArg)
  Invoke-In $Code (@('-m','experiments.run_abm_diagnostics','--reps','10',
    '--workers',"$Workers") + $RetailArg)
  Invoke-In $Code (@('-m','experiments.run_structural_sensitivity',
    '--workers',"$Workers") + $RetailArg)
  Invoke-In $Code (@('-m','experiments.run_structural_exit_check') + $RetailArg)
  Invoke-In $Code (@('-m','experiments.measure_queueing',
    '--workers',"$Workers") + $RetailArg)
  Invoke-In $Code (@('-m','experiments.run_validation_gof',
    '--workers',"$Workers") + $RetailArg)
  Invoke-In $Code (@('-m','experiments.make_heatmap_figure') + $RetailArg)
  Invoke-In $Code (@('-m','experiments.profile_uci') + $RetailArg)
  Invoke-In $Code (@('-m','experiments.run_real_data_example') + $RetailArg)
  Invoke-In $Code (@('-m','experiments.run_real_data_example',
    '--exclude-anonymous') + $RetailArg)
  Invoke-In $Code @('-m','experiments.run_figc_rescore')
  Invoke-In $Code (@('-m','experiments.make_layout_figure') + $RetailArg)
  Invoke-In $Code (@('-m','experiments.run_real_data_seeds',
    '--workers',"$Workers") + $RetailArg)
  Invoke-In $Code (@('-m','experiments.run_real_data_budget',
    '--workers',"$Workers") + $RetailArg)
  Invoke-In $Code (@('-m','experiments.run_input_uncertainty',
    '--workers',"$Workers") + $RetailArg)
  Invoke-In $Code (@('-m','experiments.run_heldout_transfer',
    '--workers',"$Workers") + $RetailArg)
}

switch ($Target) {
  'deps'              { Do-Deps }
  'deps-lock'         { Do-DepsLock }
  'lock'              { Do-Lock }
  'test'              { Do-Test }
  'verify'            { Do-Verify }
  'smoke'             { Do-Smoke }
  'figures-headless'  { Do-FiguresHeadless }
  'macros'            { Do-Macros }
  'paper'             { Do-Paper }
  'experiments-paper' { Do-ExperimentsPaper }
  # restyle_figures, macros and the paper read what the experiment runs
  # write, so they run after them. The GUI screenshots
  # CODE/make_gui_figures.py draws are the only figures not regenerated
  # here: they need a display. The emergent heat map comes from a seeded
  # headless run (experiments.make_heatmap_figure, in experiments-paper).
  'paper-grade'       { Do-ExperimentsPaper; Do-FiguresHeadless; Do-Macros; Do-Paper }
  default {
    Write-Host 'Targets: deps deps-lock lock test verify smoke figures-headless macros paper experiments-paper paper-grade'
    Write-Host 'Options: -Retail <workbook path> (default: discovered), -Workers <n> (default: 1)'
  }
}
