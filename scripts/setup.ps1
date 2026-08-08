param([ValidateSet('cpu','gpu')][string]$Profile = 'cpu')
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$Root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'project_environment.ps1')

if ($Profile -ne 'cpu') { throw 'The City Recovery Model Workbench supports the CPU profile only.' }
foreach ($Command in @('uv', 'node', 'npm')) {
    if (-not (Get-Command $Command -ErrorAction SilentlyContinue)) {
        throw "Required setup command is missing: $Command"
    }
}
$UvVersionText = [string](& uv --version)
if ($LASTEXITCODE -ne 0 -or $UvVersionText -notmatch '^uv\s+(\d+\.\d+\.\d+)') {
    throw "Unable to determine the uv version from: $UvVersionText"
}
$UvVersion = [version]$Matches[1]
$MinimumUvVersion = [version]'0.7.21'
if ($UvVersion -lt $MinimumUvVersion) {
    throw "uv $MinimumUvVersion or newer is required; found $UvVersion."
}

# Large pinned wheels can otherwise saturate constrained Windows/CDN paths. Keep
# hash verification and the frozen lock unchanged while using conservative,
# process-scoped defaults. Explicit caller settings still take precedence.
$UvTransportDefaults = [ordered]@{
    UV_CONCURRENT_DOWNLOADS = '4'
    UV_CONCURRENT_INSTALLS = '2'
    UV_HTTP_RETRIES = '6'
    UV_HTTP_TIMEOUT = '120'
}
$UvTransportOriginal = @{}
foreach ($Entry in $UvTransportDefaults.GetEnumerator()) {
    $CurrentValue = [Environment]::GetEnvironmentVariable($Entry.Key, 'Process')
    $UseDefault = [string]::IsNullOrWhiteSpace($CurrentValue)
    $UvTransportOriginal[$Entry.Key] = @{
        WasPresent = $null -ne $CurrentValue
        WasDefaulted = $UseDefault
        Value = $CurrentValue
    }
    if ($UseDefault) {
        [Environment]::SetEnvironmentVariable($Entry.Key, $Entry.Value, 'Process')
    }
}

$ProjectEnvironment = $null
$LocationPushed = $false
try {
    $ProjectEnvironment = Enter-Ai17ProjectEnvironment -Root $Root
    Push-Location $Root
    $LocationPushed = $true
    $UvTransportSummary = ($UvTransportDefaults.Keys | ForEach-Object {
        "$_=$([Environment]::GetEnvironmentVariable($_, 'Process'))"
    }) -join ', '
    Write-Host "[setup] uv transport: $UvTransportSummary"
    Write-Host '[setup] Syncing pinned Python 3.12 environment'
    & uv sync --frozen --python 3.12 --no-group training
    if ($LASTEXITCODE -ne 0) { throw 'uv sync failed' }
    Assert-Ai17NativePathBudget -Context $ProjectEnvironment
    & $ProjectEnvironment.PythonPath -c 'from onnx import onnx_cpp2py_export; import fastapi, numpy, onnxruntime'
    if ($LASTEXITCODE -ne 0) { throw 'native runtime dependency smoke failed' }
    Write-Host '[setup] Installing pinned frontend dependencies'
    & npm ci --prefix frontend
    if ($LASTEXITCODE -ne 0) { throw 'npm ci failed' }
    Write-Host '[setup] Building the production frontend'
    & npm run build --prefix frontend
    if ($LASTEXITCODE -ne 0) { throw 'frontend build failed' }
    Write-Host '[setup] Complete'
}
finally {
    if ($LocationPushed) { Pop-Location }
    if ($null -ne $ProjectEnvironment) { Exit-Ai17ProjectEnvironment -Context $ProjectEnvironment }
    foreach ($Entry in $UvTransportDefaults.GetEnumerator()) {
        $Original = $UvTransportOriginal[$Entry.Key]
        if (-not $Original.WasDefaulted) { continue }
        if ($Original.WasPresent) {
            [Environment]::SetEnvironmentVariable($Entry.Key, $Original.Value, 'Process')
        }
        else {
            Remove-Item -LiteralPath "Env:$($Entry.Key)" -ErrorAction SilentlyContinue
        }
    }
}
