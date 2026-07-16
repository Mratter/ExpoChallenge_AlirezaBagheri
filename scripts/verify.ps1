param([ValidateSet('cpu','gpu')][string]$Profile = 'cpu')
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$Root = Split-Path -Parent $PSScriptRoot
& (Join-Path $PSScriptRoot 'preflight.ps1') -Profile $Profile -Full
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$RunDirectory = Join-Path $Root '.run'
New-Item -ItemType Directory -Force $RunDirectory | Out-Null
$Stdout = Join-Path $RunDirectory 'verify-server.stdout.log'
$Stderr = Join-Path $RunDirectory 'verify-server.stderr.log'
$Capture = Join-Path $RunDirectory 'verify-persisted-result.json'
$env:INNOVERSE_PROFILE = $Profile
$env:INNOVERSE_RUNTIME = '1'
$env:INNOVERSE_COMMIT = (& git -C $Root rev-parse --short=12 HEAD).Trim()
$Python = Join-Path $Root '.venv\Scripts\python.exe'
$Process = Start-Process -FilePath $Python -ArgumentList @('-m','uvicorn','backend.app.main:app','--host','127.0.0.1','--port','4117','--no-access-log') -WorkingDirectory $Root -WindowStyle Hidden -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -PassThru
try {
    Push-Location $Root
    try {
        & uv run --frozen python scripts/verify_runtime.py --base-url http://127.0.0.1:4117 --capture-result $Capture
        if ($LASTEXITCODE -ne 0) { throw 'runtime verification failed' }
    }
    finally { Pop-Location }
}
finally {
    if (-not $Process.HasExited) { Stop-Process -Id $Process.Id -Force }
    $Process.WaitForExit()
}

$Process = Start-Process -FilePath $Python -ArgumentList @('-m','uvicorn','backend.app.main:app','--host','127.0.0.1','--port','4117','--no-access-log') -WorkingDirectory $Root -WindowStyle Hidden -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -PassThru
try {
    Push-Location $Root
    try {
        & uv run --frozen python scripts/verify_runtime.py --base-url http://127.0.0.1:4117 --restore-result $Capture
        if ($LASTEXITCODE -ne 0) { throw 'restart persistence verification failed' }
    }
    finally { Pop-Location }
}
finally {
    if (-not $Process.HasExited) { Stop-Process -Id $Process.Id -Force }
    $Process.WaitForExit()
}
Write-Host '[verify] PASS'
