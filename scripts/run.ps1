param(
    [ValidateSet('cpu')][string]$Profile = 'cpu',
    [ValidateRange(1, 65535)][int]$Port = 4117,
    [string]$PolicyPath = $env:INNOVERSE_POLICY_PATH,
    [switch]$NoBrowser
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'project_environment.ps1')
if ([string]::IsNullOrWhiteSpace($PolicyPath)) {
    throw 'A policy is required. Set INNOVERSE_POLICY_PATH or pass -PolicyPath with the ONNX file to serve.'
}
try {
    $ResolvedPolicyPath = (Resolve-Path -LiteralPath $PolicyPath -ErrorAction Stop).Path
}
catch {
    throw "The selected policy does not exist: $PolicyPath"
}
if (-not (Test-Path -LiteralPath $ResolvedPolicyPath -PathType Leaf)) {
    throw "The selected policy is not a file: $ResolvedPolicyPath"
}
$env:INNOVERSE_POLICY_PATH = $ResolvedPolicyPath
& (Join-Path $PSScriptRoot 'preflight.ps1') -Profile $Profile -Port $Port

$Context = Get-CityRecoveryEnvironmentContext -Root $Root
$env:INNOVERSE_PROFILE = $Profile
$env:INNOVERSE_RUNTIME = '1'
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
    Write-Host "[run] City Recovery toolbox: $ToolboxUrl"
    Write-Host "[run] Policy: $ResolvedPolicyPath"
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
