param(
    [ValidateSet('cpu')][string]$Profile = 'cpu',
    [ValidateRange(1, 65535)][int]$Port = 4117,
    [switch]$SkipPortCheck
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'project_environment.ps1')

$RequiredPaths = @(
    'requirements.txt',
    'scripts/preflight_check.py',
    'backend/app/main.py',
    'backend/app/models.py',
    'backend/app/persistence.py',
    'backend/app/shared_evidence.py',
    'backend/app/city/__init__.py',
    'backend/app/city/environment.py',
    'backend/app/city/optimizer.py',
    'backend/app/city/outcome.py',
    'backend/app/city/physics.py',
    'backend/app/city/planners.py',
    'backend/app/city/scenarios.py',
    'model/__init__.py',
    'model/policy.py',
    'frontend/package-lock.json',
    'frontend/dist/index.html'
)
foreach ($RelativePath in $RequiredPaths) {
    $FullPath = Join-Path $Root $RelativePath
    if (-not (Test-Path -LiteralPath $FullPath -PathType Leaf)) {
        throw "Required runtime file is missing: $RelativePath. Rerun scripts\setup.ps1 or restore the working tree."
    }
}

$PolicyPathText = [string]$env:INNOVERSE_POLICY_PATH
if ([string]::IsNullOrWhiteSpace($PolicyPathText)) {
    throw 'INNOVERSE_POLICY_PATH is required. Set it to the ONNX policy that this runtime should serve.'
}
try {
    $ResolvedPolicyPath = (Resolve-Path -LiteralPath $PolicyPathText -ErrorAction Stop).Path
}
catch {
    throw "INNOVERSE_POLICY_PATH does not resolve to a readable policy file: $PolicyPathText"
}
if (-not (Test-Path -LiteralPath $ResolvedPolicyPath -PathType Leaf)) {
    throw "INNOVERSE_POLICY_PATH is not a file: $ResolvedPolicyPath"
}
$env:INNOVERSE_POLICY_PATH = $ResolvedPolicyPath
$env:INNOVERSE_PROFILE = $Profile

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
        throw 'The configured policy or runtime smoke comparison failed.'
    }
}
finally {
    Pop-Location
}
Write-Host '[preflight] RUNTIME READY'
