param([ValidateSet('cpu','gpu')][string]$Profile = 'cpu', [switch]$Full)
$ErrorActionPreference = 'Stop'
$marker = Join-Path (Split-Path -Parent $PSScriptRoot) '.gate2-passed'
if (-not (Test-Path -LiteralPath $marker)) {
    Write-Error 'Startup blocked: Gate 2 has not passed and no real vertical slice is available.'
    exit 2
}
exit 0
