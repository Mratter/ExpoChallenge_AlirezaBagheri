param([ValidateSet('cpu','gpu')][string]$Profile = 'cpu', [switch]$Full)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$Root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'project_environment.ps1')

if ($Profile -ne 'cpu') { throw 'The City Recovery Model Workbench supports the CPU profile only.' }
foreach ($Path in @(
    'uv.lock',
    'frontend/package-lock.json',
    'frontend/dist/index.html',
    'artifacts/workbench/overview.v1.json',
    'artifacts/workbench/manifest.v1.json',
    'artifacts/workbench/benchmarks/adaptive-cascades-showcase-v2/final/manifest.json',
    'artifacts/workbench/benchmarks/adaptive-cascades-showcase-v2/final/result.json',
    'artifacts/workbench/benchmarks/adaptive-cascades-showcase-v2/final/replay-report.json',
    'artifacts/workbench/benchmarks/adaptive-cascades-showcase-v2/final/anti-gaming-report.json',
    'artifacts/workbench/benchmarks/adaptive-cascades-showcase-v2/final/terminal.json',
    'artifacts/workbench/benchmarks/adaptive-cascades-showcase-v2/candidate/adaptive-cascade-mlp-v2-300k.onnx',
    'artifacts/workbench/benchmarks/adaptive-cascades-showcase-v2/candidate/adaptive-cascade-mlp-v2-300k.pt',
    'artifacts/workbench/benchmarks/adaptive-cascades-showcase-v2/candidate/candidate-manifest.json',
    'artifacts/workbench/benchmarks/adaptive-cascades-showcase-v2/candidate/training-receipt.json',
    'artifacts/workbench/benchmarks/adaptive-cascades-showcase-v1/final/manifest.json',
    'artifacts/workbench/benchmarks/adaptive-cascades-showcase-v1/final/result.json'
)) {
    if (-not (Test-Path -LiteralPath (Join-Path $Root $Path))) { throw "Required dependency is missing: $Path" }
}
$Occupied = Get-NetTCPConnection -State Listen -LocalPort 4117 -ErrorAction SilentlyContinue
if ($Occupied) { throw 'Fixed port 4117 is already occupied.' }

$ProjectEnvironment = $null
$LocationPushed = $false
try {
    $ProjectEnvironment = Enter-Ai17ProjectEnvironment -Root $Root
    Push-Location $Root
    $LocationPushed = $true
    & $ProjectEnvironment.PythonPath scripts/workbench_preflight.py
    if ($LASTEXITCODE -ne 0) { throw 'Workbench evidence, frontend, or ONNX smoke preflight failed' }
    if ($Full) {
        & $ProjectEnvironment.PythonPath -m pytest backend/tests/test_workbench_runtime.py backend/tests/test_workbench_api.py
        if ($LASTEXITCODE -ne 0) { throw 'workbench runtime tests failed' }
        & $ProjectEnvironment.PythonPath -m ruff check backend/app/workbench_main.py backend/app/workbench_service.py backend/app/workbench_toolbox.py backend/tests/test_workbench_runtime.py backend/tests/test_workbench_api.py scripts/workbench_preflight.py
        if ($LASTEXITCODE -ne 0) { throw 'workbench runtime lint failed' }
        & npm test --prefix frontend
        if ($LASTEXITCODE -ne 0) { throw 'frontend tests failed' }
        & npm run build --prefix frontend
        if ($LASTEXITCODE -ne 0) { throw 'frontend build failed' }
    }
    Write-Host '[preflight] PASS'
}
finally {
    if ($LocationPushed) { Pop-Location }
    if ($null -ne $ProjectEnvironment) { Exit-Ai17ProjectEnvironment -Context $ProjectEnvironment }
}
