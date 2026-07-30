<#
.SYNOPSIS
  Reproduction driver for the retail-layout simulator (audit R5.2),
  the Windows/PowerShell counterpart of the Makefile.

.DESCRIPTION
  Quick start (minutes):   .\reproduce.ps1 deps; .\reproduce.ps1 test; .\reproduce.ps1 smoke
  Rebuild the PDF:         .\reproduce.ps1 macros; .\reproduce.ps1 paper   (needs figs/)
  Full paper-grade regen:  .\reproduce.ps1 paper-grade                     (HOURS)

.PARAMETER Target
  deps | test | verify | smoke | figures-headless | macros |
  experiments-paper | paper-grade | help
#>
param(
  [ValidateSet('help','deps','test','verify','smoke','figures-headless','macros',
               'experiments-paper','paper-grade')]
  [string]$Target = 'help'
)

$ErrorActionPreference = 'Stop'
$Root   = $PSScriptRoot
$Code   = Join-Path $Root 'CODE'
$Retail = 'DATASETS/UCI Online Retail II .xlsx.xlsx'
$PY     = 'python'

function Invoke-In($dir, [string[]]$argv) {
  Push-Location $dir
  try { & $PY @argv; if ($LASTEXITCODE -ne 0) { throw "exit $LASTEXITCODE" } }
  finally { Pop-Location }
}

function Do-Deps    { & $PY -m pip install -r (Join-Path $Root 'requirements.txt') }
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
}
function Do-Macros { Invoke-In $Code @('make_results_macros.py') }
function Do-ExperimentsPaper {
  Invoke-In $Code @('-m','experiments.run_synthetic_gt','--n-scenarios','30',
    '--n-seeds','10','--mc-iters','2000','--n-gens','25','--pop-size','30')
  Invoke-In $Code @('-m','experiments.run_baseline_comparison','--n-scenarios','30',
    '--n-seeds','10','--mc-iters','2000','--n-gens','25','--pop-size','30')
  Invoke-In $Code @('-m','experiments.run_mc_groundtruth','--n-scenarios','6',
    '--normal-budget','750','--big-budget','7500','--mc-iters','1000')
  Invoke-In $Code @('-m','experiments.run_ga_sensitivity','--n-scenarios','3',
    '--n-seeds','2','--n-gens','25','--mc-iters','800')
  Invoke-In $Code @('-m','experiments.run_elasticity_lhs')
  Invoke-In $Code @('-m','experiments.make_paper_figures')
  Invoke-In $Code @('-m','experiments.run_abm_diagnostics','--reps','3',
    '--seconds','120','--warmup','30')
  Invoke-In $Code @('-m','experiments.run_structural_sensitivity',
    '--seconds','90','--warmup','30')
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
  'experiments-paper' { Do-ExperimentsPaper }
  'paper-grade'       { Do-ExperimentsPaper; Do-Macros }
  default {
    Write-Host 'Targets: deps test verify smoke figures-headless macros experiments-paper paper-grade'
  }
}
