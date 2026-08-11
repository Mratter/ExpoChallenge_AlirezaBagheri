param(
    [ValidateSet('cpu')][string]$Profile = 'cpu',
    [AllowEmptyString()][string]$PolicyPath,
    [switch]$SkipToolBootstrap
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'project_environment.ps1')

function Install-CityRecoveryTool {
    param(
        [Parameter(Mandatory = $true)][string]$Id,
        [Parameter(Mandatory = $true)][string]$DisplayName
    )

    if ($SkipToolBootstrap) {
        throw "$DisplayName is missing. Install it manually, or rerun setup.ps1 without -SkipToolBootstrap."
    }
    $Winget = Get-Command winget.exe -ErrorAction SilentlyContinue
    if ($null -eq $Winget) {
        throw "$DisplayName is missing and Windows Package Manager (winget) is unavailable. Install App Installer from Microsoft Store, then rerun setup.ps1."
    }
    Write-Host "[setup] Installing $DisplayName with Windows Package Manager"
    & $Winget.Source install `
        --id $Id `
        --exact `
        --source winget `
        --accept-package-agreements `
        --accept-source-agreements `
        --disable-interactivity `
        --silent | Out-Host
    $InstallerExitCode = $LASTEXITCODE
    Update-CityRecoveryProcessPath
    return $InstallerExitCode
}

Update-CityRecoveryProcessPath

try {
    $Launcher = Get-CityRecoveryPython312Launcher
}
catch {
    $PythonInstallerExitCode = Install-CityRecoveryTool -Id 'Python.Python.3.12' -DisplayName '64-bit Python 3.12'
    $Launcher = Wait-CityRecoveryPython312Launcher -InstallerExitCode $PythonInstallerExitCode
}

$NodeCommand = Get-CityRecoveryNodeCommand
if ($null -eq $NodeCommand) {
    $NodeInstallerExitCode = Install-CityRecoveryTool -Id 'OpenJS.NodeJS.LTS' -DisplayName 'Node.js LTS and npm'
    $NodeCommand = Wait-CityRecoveryNodeCommand -InstallerExitCode $NodeInstallerExitCode
}
$Npm = Get-Command npm.cmd -ErrorAction SilentlyContinue
if ($null -eq $Npm) { $Npm = Get-Command npm -ErrorAction SilentlyContinue }
if ($null -eq $Npm) {
    throw 'npm is missing from the Node.js installation. Reinstall a current Node.js LTS release.'
}
$NpmCommand = $Npm.Source

$NodeVersionText = [string](& $NodeCommand --version)
if ($LASTEXITCODE -ne 0 -or -not (Test-CityRecoveryNodeVersion -VersionText $NodeVersionText)) {
    throw "A supported Node.js LTS is required (^20.19, ^22.12, or a newer even-numbered LTS); found $NodeVersionText."
}
& $NpmCommand --version | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'npm is installed but could not start.' }

$Context = Get-CityRecoveryEnvironmentContext -Root $Root
$LauncherArguments = @($Launcher.Prefix) + @('-m', 'venv')
if (Test-Path -LiteralPath $Context.PythonPath -PathType Leaf) {
    & $Context.PythonPath -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) and sys.maxsize > 2**32 else 1)'
    if ($LASTEXITCODE -ne 0) {
        Write-Host '[setup] Replacing an incompatible Python environment'
        $LauncherArguments += '--clear'
        $LauncherArguments += $Context.EnvironmentPath
        & $Launcher.Command @LauncherArguments
        if ($LASTEXITCODE -ne 0) { throw 'Python environment creation failed.' }
    }
}
else {
    Write-Host '[setup] Creating the Python 3.12 environment'
    $LauncherArguments += $Context.EnvironmentPath
    & $Launcher.Command @LauncherArguments
    if ($LASTEXITCODE -ne 0) { throw 'Python environment creation failed.' }
}

Assert-CityRecoveryPython312 -PythonPath $Context.PythonPath
Push-Location $Root
try {
    Write-Host '[setup] Updating pip'
    & $Context.PythonPath -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) {
        throw 'pip could not update itself. Check the internet connection and rerun setup.ps1.'
    }

    Write-Host '[setup] Installing the pinned Python 3.12 runtime'
    & $Context.PythonPath -m pip install --requirement requirements.txt
    if ($LASTEXITCODE -ne 0) {
        throw 'Python package installation failed. Check the internet connection and rerun setup.ps1.'
    }
    & $Context.PythonPath -c 'import backend.app.city.environment, backend.app.city.optimizer, fastapi, gymnasium, model.policy, numpy, onnx, onnxruntime, ortools, pydantic, uvicorn'
    if ($LASTEXITCODE -ne 0) { throw 'The installed Python runtime failed its import check.' }
    Assert-CityRecoveryNativePathBudget -Context $Context

    Write-Host '[setup] Installing locked frontend packages with npm ci'
    & $NpmCommand ci --prefix frontend
    if ($LASTEXITCODE -ne 0) {
        throw 'Frontend package installation failed. Check the internet connection and rerun setup.ps1.'
    }

    Write-Host '[setup] Building the browser Toolbox'
    & $NpmCommand run build --prefix frontend
    if ($LASTEXITCODE -ne 0) { throw 'The frontend production build failed.' }

    Write-Host '[setup] Verifying the bundled or selected ONNX policy and runtime'
    $PreflightParameters = @{
        Profile = $Profile
        SkipPortCheck = $true
    }
    if ($PSBoundParameters.ContainsKey('PolicyPath')) {
        $PreflightParameters.PolicyPath = $PolicyPath
    }
    & (Join-Path $PSScriptRoot 'preflight.ps1') @PreflightParameters
    if ($LASTEXITCODE -ne 0) { throw 'Runtime preflight failed.' }

    Write-Host ''
    Write-Host '[setup] COMPLETE'
    Write-Host '[setup] Start the Toolbox with: powershell -ExecutionPolicy Bypass -File .\scripts\run.ps1'
}
finally {
    Pop-Location
}
