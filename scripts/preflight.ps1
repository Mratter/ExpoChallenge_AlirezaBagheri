param(
    [ValidateSet('cpu')][string]$Profile = 'cpu',
    [ValidateRange(1, 65535)][int]$Port = 4117,
    [switch]$SkipPortCheck
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'project_environment.ps1')

# This is a V3-only release. simulator_v2.py remains because the sealed V3
# simulator imports its shared logistics mechanics and hashes that exact file.
$RequiredPaths = @(
    'requirements.txt',
    'backend/app/main.py',
    'backend/app/models.py',
    'backend/app/persistence.py',
    'backend/app/scenarios_v3.py',
    'backend/app/simulator_core.py',
    'backend/app/simulator_v2.py',
    'backend/app/simulator_v3.py',
    'model/__init__.py',
    'model/ppo_v3.py',
    'artifacts/city_recovery_ppo.v3.zip',
    'artifacts/city_recovery_ppo.v3.selected.onnx',
    'artifacts/model_manifest.v3.selected.json',
    'training/v3/protocol.json',
    'training/v3/source-seal.json',
    'training/v3/checkpoint-selection-receipt.json',
    'training/v3/selected-onnx-parity.json',
    'training/v3/final-authorization.json',
    'training/v3/final-use-receipt.json',
    'training/v3/final-terminal.json',
    'internal/training_runs/v3/checkpoint-ledger.jsonl',
    'internal/training_runs/v3/checkpoints/stage-01-transition-000092160.zip',
    'internal/training_runs/v3/checkpoints/stage-02-transition-000184320.zip',
    'internal/training_runs/v3/checkpoints/stage-03-transition-000276480.zip',
    'internal/training_runs/v3/checkpoints/stage-04-transition-000368640.zip',
    'internal/training_runs/v3/checkpoints/stage-05-transition-000460800.zip',
    'internal/training_runs/v3/checkpoints/stage-06-transition-000552960.zip',
    'internal/training_runs/v3/checkpoints/stage-07-transition-000645120.zip',
    'internal/training_runs/v3/final-ledger.jsonl',
    'benchmarks/v3/final-40.json',
    'frontend/package-lock.json',
    'frontend/dist/index.html'
)
foreach ($RelativePath in $RequiredPaths) {
    $FullPath = Join-Path $Root $RelativePath
    if (-not (Test-Path -LiteralPath $FullPath -PathType Leaf)) {
        throw "Required PPO-v3 release file is missing: $RelativePath. This package is not release-ready; restore the complete signed release or rerun scripts\setup.ps1 after all release artifacts are present."
    }
}

$Context = Get-CityRecoveryEnvironmentContext -Root $Root
Assert-CityRecoveryPython312 -PythonPath $Context.PythonPath
Assert-CityRecoveryNativePathBudget -Context $Context

if (-not $SkipPortCheck) {
    $Listener = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback, $Port)
    try {
        $Listener.Start()
    }
    catch {
        throw "Port $Port is already in use. Close the program using it or choose another port, for example: .\scripts\run.ps1 -Port 4120"
    }
    finally {
        $Listener.Stop()
    }
}

Push-Location $Root
try {
    & $Context.PythonPath scripts/preflight_check.py
    if ($LASTEXITCODE -ne 0) {
        throw 'The selected PPO-v3 release, frozen evidence, or simulator check failed.'
    }
}
finally {
    Pop-Location
}
Write-Host '[preflight] PPO-v3 RELEASE READY'
