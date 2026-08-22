param([int]$Threads = 8)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$julia = if ($env:JULIA_BIN) { $env:JULIA_BIN } else { 'julia' }
$outputRoot = Join-Path $projectRoot 'outputs\DirectionalJVPvsAlmquist'
$warfarinData = Join-Path $projectRoot 'data\warfarin_dat.csv'

if (-not (Test-Path -LiteralPath $warfarinData)) {
    throw "Warfarin data were not found at $warfarinData. Run scripts/download_warfarin_data.ps1 first."
}

New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null

$env:FLIPFLOP_JULIA_RESIDUAL_MODEL = 'combined'
$env:FLIPFLOP_JULIA_OUTDIR = Join-Path $outputRoot 'flipflop'
$env:FLIPFLOP_JULIA_METHODS = 'ALMQUIST_FORWARD,FULL_IMPLICIT_DIRECTIONAL_JVP'
$env:FLIPFLOP_JULIA_N_SUBJ = '50'
$env:FLIPFLOP_JULIA_N_STARTS = '100'
$env:FLIPFLOP_JULIA_MAXITER_ETA = '40'
$env:FLIPFLOP_JULIA_MAXITER_OUTER = '50'
$env:FLIPFLOP_JULIA_DATA_SEED = '123'
$env:FLIPFLOP_JULIA_LOGDET_MODE = 'raw'

& $julia -t $Threads "--project=$projectRoot" (Join-Path $projectRoot 'flipflop_combined_multistart.jl')
if ($LASTEXITCODE -ne 0) { throw "One-compartment comparison exited with code $LASTEXITCODE" }

$env:WARFARIN_JULIA_RESIDUAL_MODEL = 'combined'
$env:WARFARIN_JULIA_DATA = $warfarinData
$env:WARFARIN_JULIA_OUTDIR = Join-Path $outputRoot 'warfarin'
$env:WARFARIN_JULIA_METHODS = 'ALMQUIST_FORWARD,FULL_IMPLICIT_DIRECTIONAL_JVP'
$env:WARFARIN_JULIA_REPRESENTATIONS = 'ode'
$env:WARFARIN_JULIA_N_SUBJ = '32'
$env:WARFARIN_JULIA_N_STARTS = '10'
$env:WARFARIN_JULIA_MAXITER_ETA = '40'
$env:WARFARIN_JULIA_MAXITER_OUTER = '50'
$env:WARFARIN_JULIA_DT = '0.25'
$env:WARFARIN_JULIA_LOGDET_MODE = 'raw'

& $julia -t $Threads "--project=$projectRoot" (Join-Path $projectRoot 'warfarin_combined_multistart.jl')
if ($LASTEXITCODE -ne 0) { throw "Warfarin comparison exited with code $LASTEXITCODE" }