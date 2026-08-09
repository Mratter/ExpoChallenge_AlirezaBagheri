param(
    [ValidateSet('cpu')][string]$Profile = 'cpu',
    [ValidateRange(1, 65535)][int]$Port = 4117,
    [switch]$NoBrowser
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'project_environment.ps1')
& (Join-Path $PSScriptRoot 'preflight.ps1') -Profile $Profile -Port $Port

$Context = Get-CityRecoveryEnvironmentContext -Root $Root
$env:INNOVERSE_PROFILE = $Profile
$env:INNOVERSE_RUNTIME = '1'
$SourceSeal = Get-Content -LiteralPath (Join-Path $Root 'training\v3\source-seal.json') -Raw | ConvertFrom-Json
$SourceIdentity = [string]$SourceSeal.scientific_source.semantic_sha256
if ($SourceIdentity -notmatch '^[0-9a-f]{64}$') {
    throw 'The V3 source-seal identity is invalid.'
}
$env:INNOVERSE_COMMIT = "portable-ppo-v3-$($SourceIdentity.Substring(0, 12))"
$ToolboxUrl = "http://127.0.0.1:$Port/#/toolbox"
$ReadyUrl = "http://127.0.0.1:$Port/health/ready"
$BrowserJob = $null

if (-not $NoBrowser) {
    $BrowserJob = Start-Job -ScriptBlock {
        param($HealthUrl, $LaunchUrl)
        for ($Attempt = 0; $Attempt -lt 60; $Attempt++) {
            try {
                $Response = Invoke-WebRequest -Uri $HealthUrl -UseBasicParsing -TimeoutSec 2
                if ($Response.StatusCode -eq 200) {
                    Start-Process $LaunchUrl
                    return
                }
            }
            catch {
                Start-Sleep -Milliseconds 500
            }
        }
    } -ArgumentList $ReadyUrl, $ToolboxUrl
}

Push-Location $Root
try {
    Write-Host "[run] PPO-v3 toolbox: $ToolboxUrl"
    Write-Host '[run] Selected PPO-v3 deployment and sealed 40-case evidence loaded.'
    Write-Host '[run] Press Ctrl+C to stop.'
    & $Context.PythonPath -m uvicorn backend.app.main:app --host 127.0.0.1 --port $Port --no-access-log
    exit $LASTEXITCODE
}
finally {
    Pop-Location
    if ($null -ne $BrowserJob) {
        Stop-Job -Job $BrowserJob -ErrorAction SilentlyContinue
        Remove-Job -Job $BrowserJob -Force -ErrorAction SilentlyContinue
    }
}
