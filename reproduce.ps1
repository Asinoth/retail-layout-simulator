<#
.SYNOPSIS
  Reproduction driver for the retail-layout simulator (audit R5.2),
  the Windows/PowerShell counterpart of the Makefile.

.DESCRIPTION
  Quick start (minutes):   .\reproduce.ps1 deps; .\reproduce.ps1 test; .\reproduce.ps1 smoke
  Rebuild the PDF:         .\reproduce.ps1 macros; .\reproduce.ps1 paper   (needs figs/)
  Full paper-grade regen:  .\reproduce.ps1 paper-grade                     (HOURS)

.PARAMETER Target
  deps | test | verify | smoke | figures-headless | macros | paper |
  experiments-paper | paper-grade | help
#>
param(
  [ValidateSet('help','deps','test','verify','smoke','figures-headless','macros',
               'paper','experiments-paper','paper-grade')]
  [string]$Target = 'help'
)

$ErrorActionPreference = 'Stop'
$Root   = $PSScriptRoot
$Code   = Join-Path $Root 'CODE'
$Paper  = 'TOMACS_submission'
$Retail = 'DATASETS/UCI Online Retail II .xlsx.xlsx'
$PY     = 'python'

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
function Do-Test    { Invoke-In $Code @('-m','pytest','-q','tests/') }
function Do-Smoke {
  Invoke-In $Code @('-m','experiments.run_synthetic_gt','--n-scenarios','3',
    '--n-seeds','2','--mc-iters','200','--n-gens','8','--pop-size','16',
    '--n-spearman-samples','40')
  Invoke-In $Code @('-m','experiments.run_baseline_comparison','--n-scenarios','3',
    '--n-seeds','2','--mc-iters','200','--n-gens','8','--pop-size','16')
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
function Do-Paper {
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
    '--n-seeds','10','--mc-iters','2000','--n-gens','25','--pop-size','30')
  Invoke-In $Code @('-m','experiments.run_baseline_comparison','--n-scenarios','30',
    '--n-seeds','10','--mc-iters','2000','--n-gens','25','--pop-size','30')
  Invoke-In $Code @('-m','experiments.run_mc_groundtruth','--n-scenarios','6',
    '--normal-budget','750','--big-budget','7500','--mc-iters','1000')
  Invoke-In $Code @('-m','experiments.run_ga_sensitivity','--n-scenarios','3',
    '--n-seeds','2','--n-gens','25','--mc-iters','800')
  Invoke-In $Code @('-m','experiments.run_elasticity_lhs','--n-scenarios','12',
    '--n-seeds','3','--n-draws','256','--n-gens','15','--pop-size','24',
    '--mc-iters','500')
  Invoke-In $Code @('-m','experiments.make_paper_figures')
  Invoke-In $Code @('-m','experiments.run_abm_diagnostics','--reps','10')
  Invoke-In $Code @('-m','experiments.run_structural_sensitivity')
  Invoke-In $Code @('-m','experiments.measure_queueing')
  Invoke-In $Code @('-m','experiments.run_validation_gof')
  Invoke-In $Code @('-m','experiments.run_real_data_example',
    '--retail-path',"../$Retail")
}

switch ($Target) {
  'deps'              { Do-Deps }
  'test'              { Do-Test }
  'verify'            { Do-Verify }
  'smoke'             { Do-Smoke }
  'figures-headless'  { Do-FiguresHeadless }
  'macros'            { Do-Macros }
  'paper'             { Do-Paper }
  'experiments-paper' { Do-ExperimentsPaper }
  # restyle_figures, macros and the paper read what the experiment runs
  # write, so they run after them. The figures CODE/make_gui_figures.py
  # draws (the GUI screenshots and the emergent heat map) are the only
  # ones not regenerated here: they need a display.
  'paper-grade'       { Do-ExperimentsPaper; Do-FiguresHeadless; Do-Macros; Do-Paper }
  default {
    Write-Host 'Targets: deps test verify smoke figures-headless macros paper experiments-paper paper-grade'
  }
}
