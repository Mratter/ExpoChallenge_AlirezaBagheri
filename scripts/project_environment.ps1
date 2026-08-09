function Get-CityRecoveryRootHash {
    param([Parameter(Mandatory = $true)][string]$Value)

    $Hasher = [Security.Cryptography.SHA256]::Create()
    try {
        $Digest = $Hasher.ComputeHash([Text.Encoding]::UTF8.GetBytes($Value))
    }
    finally {
        $Hasher.Dispose()
    }
    return (($Digest | ForEach-Object { $_.ToString('x2') }) -join '')
}

function Get-CityRecoveryEnvironmentContext {
    param([Parameter(Mandatory = $true)][string]$Root)

    $CanonicalRoot = [IO.Path]::GetFullPath($Root).TrimEnd([char[]]'\/')
    $NativeSuffix = 'Lib\site-packages\onnxruntime\capi\onnxruntime_providers_shared.dll'
    $RepositoryEnvironment = Join-Path $CanonicalRoot '.venv'
    $RepositoryNativePath = Join-Path $RepositoryEnvironment $NativeSuffix

    if ($RepositoryNativePath.Length -lt 240) {
        $EnvironmentPath = $RepositoryEnvironment
        $Mode = 'project-local'
    }
    else {
        if ([string]::IsNullOrWhiteSpace($env:LOCALAPPDATA)) {
            throw 'LOCALAPPDATA is unavailable, so a Windows-loader-safe Python environment cannot be created.'
        }
        $RootHash = Get-CityRecoveryRootHash -Value $CanonicalRoot.ToUpperInvariant()
        $EnvironmentPath = Join-Path $env:LOCALAPPDATA "Innoverse\city-recovery-ppo-v3\py312-$($RootHash.Substring(0, 16))"
        $EnvironmentPath = [IO.Path]::GetFullPath($EnvironmentPath)
        $Mode = 'short-path'
    }

    return [pscustomobject]@{
        EnvironmentPath = $EnvironmentPath
        Mode = $Mode
        NativePath = Join-Path $EnvironmentPath $NativeSuffix
        PythonPath = Join-Path $EnvironmentPath 'Scripts\python.exe'
        Root = $CanonicalRoot
    }
}

function Update-CityRecoveryProcessPath {
    $Segments = @(
        [Environment]::GetEnvironmentVariable('Path', 'Machine'),
        [Environment]::GetEnvironmentVariable('Path', 'User'),
        $env:Path
    ) -join ';'
    $env:Path = (($Segments -split ';' | Where-Object {
        -not [string]::IsNullOrWhiteSpace($_)
    } | Select-Object -Unique) -join ';')
}

function Get-CityRecoveryPython312KnownPaths {
    param(
        [AllowNull()][string]$LocalAppDataRoot = $env:LOCALAPPDATA,
        [AllowNull()][string[]]$ProgramFilesRoots = @($env:ProgramFiles, ${env:ProgramFiles(x86)})
    )

    $Paths = [Collections.Generic.List[string]]::new()
    if (-not [string]::IsNullOrWhiteSpace($LocalAppDataRoot)) {
        $PerUserRoot = Join-Path $LocalAppDataRoot 'Programs\Python'
        $Paths.Add((Join-Path $PerUserRoot 'Python312\python.exe'))
        if (Test-Path -LiteralPath $PerUserRoot -PathType Container) {
            foreach ($Directory in @(Get-ChildItem -LiteralPath $PerUserRoot -Directory -Filter 'Python312*' -ErrorAction SilentlyContinue)) {
                $Paths.Add((Join-Path $Directory.FullName 'python.exe'))
            }
        }
    }
    foreach ($ProgramFilesRoot in @($ProgramFilesRoots)) {
        if ([string]::IsNullOrWhiteSpace($ProgramFilesRoot)) { continue }
        $Paths.Add((Join-Path $ProgramFilesRoot 'Python312\python.exe'))
        $Paths.Add((Join-Path $ProgramFilesRoot 'Python 3.12\python.exe'))
    }

    # The python.org installer registers PEP 514 metadata before a new shell
    # necessarily observes its PATH and py-launcher changes.
    foreach ($RegistryPath in @(
        'HKCU:\Software\Python\PythonCore\3.12\InstallPath',
        'HKLM:\Software\Python\PythonCore\3.12\InstallPath',
        'HKLM:\Software\WOW6432Node\Python\PythonCore\3.12\InstallPath'
    )) {
        if (-not (Test-Path -LiteralPath $RegistryPath)) { continue }
        try {
            $InstallPath = [string](Get-Item -LiteralPath $RegistryPath -ErrorAction Stop).GetValue('')
            if (-not [string]::IsNullOrWhiteSpace($InstallPath)) {
                $Paths.Add((Join-Path $InstallPath 'python.exe'))
            }
            $ExecutablePath = [string](Get-Item -LiteralPath $RegistryPath -ErrorAction Stop).GetValue('ExecutablePath')
            if (-not [string]::IsNullOrWhiteSpace($ExecutablePath)) {
                $Paths.Add($ExecutablePath)
            }
        }
        catch {
            # Registry discovery is supplementary; other candidates remain.
        }
    }

    return @($Paths | Where-Object {
        -not [string]::IsNullOrWhiteSpace($_)
    } | Select-Object -Unique)
}

function Test-CityRecoveryPython312Launcher {
    param([Parameter(Mandatory = $true)]$Candidate)

    $Arguments = @($Candidate.Prefix) + @(
        '-c',
        'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) and sys.maxsize > 2**32 else 1)'
    )
    try {
        & $Candidate.Command @Arguments 2>$null
        return ($LASTEXITCODE -eq 0)
    }
    catch {
        return $false
    }
}

function Test-CityRecoveryPythonAppExecutionAlias {
    param([Parameter(Mandatory = $true)][string]$Path)

    return ($Path -match '(?i)[\\/]Microsoft[\\/]WindowsApps[\\/]python(?:3)?(?:\.exe)?$')
}

function Test-CityRecoveryNodeVersion {
    param([Parameter(Mandatory = $true)][string]$VersionText)

    if ($VersionText.Trim() -notmatch '^v?(\d+)\.(\d+)\.(\d+)(?:[-+].*)?$') {
        return $false
    }
    $Version = [version]"$($Matches[1]).$($Matches[2]).$($Matches[3])"
    $Major = $Version.Major

    # Vite requires ^20.19.0 or >=22.12.0. For fresh-device stability we
    # additionally stay on even-numbered (LTS) Node release lines, rejecting
    # short-lived odd majors such as 21 and 23.
    if ($Major -eq 20) { return ($Version -ge [version]'20.19.0') }
    if ($Major -lt 22 -or ($Major % 2) -ne 0) { return $false }
    if ($Major -eq 22) { return ($Version -ge [version]'22.12.0') }
    return $true
}

function Test-CityRecoveryNodeCommand {
    param([Parameter(Mandatory = $true)][string]$Command)

    try {
        $VersionText = [string](& $Command --version 2>$null)
        return ($LASTEXITCODE -eq 0 -and (Test-CityRecoveryNodeVersion -VersionText $VersionText))
    }
    catch {
        return $false
    }
}

function Get-CityRecoveryNodeCommand {
    $Candidates = [Collections.Generic.List[string]]::new()
    $PathCommand = Get-Command node.exe -ErrorAction SilentlyContinue
    if ($null -ne $PathCommand) { $Candidates.Add($PathCommand.Source) }

    foreach ($Root in @($env:ProgramFiles, $env:LOCALAPPDATA)) {
        if ([string]::IsNullOrWhiteSpace($Root)) { continue }
        $RelativePath = if ($Root -eq $env:LOCALAPPDATA) {
            'Programs\nodejs\node.exe'
        }
        else {
            'nodejs\node.exe'
        }
        $Candidates.Add((Join-Path $Root $RelativePath))
    }

    foreach ($Candidate in @($Candidates | Select-Object -Unique)) {
        if ((Test-Path -LiteralPath $Candidate -PathType Leaf) -and
            (Test-CityRecoveryNodeCommand -Command $Candidate)) {
            $env:Path = "$(Split-Path -Parent $Candidate);$env:Path"
            return $Candidate
        }
    }
    return $null
}

function Wait-CityRecoveryNodeCommand {
    param(
        [ValidateRange(1, 120)][int]$Attempts = 30,
        [ValidateRange(0, 5000)][int]$DelayMilliseconds = 500,
        [AllowNull()][Nullable[int]]$InstallerExitCode = $null
    )

    for ($Attempt = 1; $Attempt -le $Attempts; $Attempt += 1) {
        Update-CityRecoveryProcessPath
        $Command = Get-CityRecoveryNodeCommand
        if ($null -ne $Command) { return $Command }
        if ($Attempt -lt $Attempts -and $DelayMilliseconds -gt 0) {
            Start-Sleep -Milliseconds $DelayMilliseconds
        }
    }

    if ($null -ne $InstallerExitCode -and $InstallerExitCode -ne 0) {
        throw "winget returned exit code $InstallerExitCode and no compatible Node.js LTS installation became discoverable. Install Node.js 20.19+, 22.12+, or a newer even-numbered LTS and rerun setup.ps1."
    }
    throw 'Node.js installation completed, but no compatible LTS runtime became discoverable. Open a new PowerShell window and rerun setup.ps1.'
}

function Get-CityRecoveryPython312Launcher {
    $Candidates = @()
    $PyLaunchers = @()
    $PathLauncher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($null -ne $PathLauncher) { $PyLaunchers += $PathLauncher.Source }
    foreach ($LauncherPath in @(
        $(if (-not [string]::IsNullOrWhiteSpace($env:LOCALAPPDATA)) {
            Join-Path $env:LOCALAPPDATA 'Programs\Python\Launcher\py.exe'
        }),
        $(if (-not [string]::IsNullOrWhiteSpace($env:WINDIR)) {
            Join-Path $env:WINDIR 'py.exe'
        })
    )) {
        if (-not [string]::IsNullOrWhiteSpace($LauncherPath) -and (Test-Path -LiteralPath $LauncherPath -PathType Leaf)) {
            $PyLaunchers += $LauncherPath
        }
    }
    foreach ($PyLauncher in @($PyLaunchers | Select-Object -Unique)) {
        $Candidates += [pscustomobject]@{ Command = $PyLauncher; Prefix = @('-3.12') }
        try {
            foreach ($Line in @(& $PyLauncher -0p 2>$null)) {
                if ([string]$Line -match '^\s*-V:\S+\s+(.+python(?:\.exe)?)\s*$') {
                    $Candidates += [pscustomobject]@{
                        Command = $Matches[1].Trim()
                        Prefix = @()
                    }
                }
            }
        }
        catch {
            # The standard -3.12 launcher entry and PATH candidates remain available.
        }
    }
    # Probe deterministic install and registry paths before generic PATH
    # commands. A fresh Windows profile often exposes a WindowsApps
    # AppExecutionAlias named python.exe that opens the Store instead of an
    # interpreter, so it must never be executed by unattended setup.
    foreach ($CandidatePath in @(Get-CityRecoveryPython312KnownPaths)) {
        if (Test-Path -LiteralPath $CandidatePath -PathType Leaf) {
            $Candidates += [pscustomobject]@{ Command = $CandidatePath; Prefix = @() }
        }
    }
    foreach ($CommandName in @('python.exe', 'python3.exe', 'python', 'python3')) {
        foreach ($PathCommand in @(Get-Command $CommandName -All -CommandType Application -ErrorAction SilentlyContinue)) {
            $CommandPath = [string]$PathCommand.Source
            if ([string]::IsNullOrWhiteSpace($CommandPath) -or
                (Test-CityRecoveryPythonAppExecutionAlias -Path $CommandPath)) {
                continue
            }
            $Candidates += [pscustomobject]@{ Command = $CommandPath; Prefix = @() }
        }
    }

    foreach ($Candidate in $Candidates) {
        if (Test-CityRecoveryPython312Launcher -Candidate $Candidate) {
            return $Candidate
        }
    }

    throw 'Python 3.12 was not found. Rerun setup.ps1 with Windows Package Manager available, or install 64-bit Python 3.12 from https://www.python.org/downloads/ and rerun setup.'
}

function Wait-CityRecoveryPython312Launcher {
    param(
        [ValidateRange(1, 120)][int]$Attempts = 30,
        [ValidateRange(0, 5000)][int]$DelayMilliseconds = 500,
        [AllowNull()][Nullable[int]]$InstallerExitCode = $null
    )

    $LastDiscoveryError = $null
    for ($Attempt = 1; $Attempt -le $Attempts; $Attempt += 1) {
        Update-CityRecoveryProcessPath
        try {
            return Get-CityRecoveryPython312Launcher
        }
        catch {
            $LastDiscoveryError = $_
        }
        if ($Attempt -lt $Attempts -and $DelayMilliseconds -gt 0) {
            Start-Sleep -Milliseconds $DelayMilliseconds
        }
    }

    if ($null -ne $InstallerExitCode -and $InstallerExitCode -ne 0) {
        throw "winget returned exit code $InstallerExitCode and no verified 64-bit Python 3.12 installation became discoverable. Install Python 3.12 manually and rerun setup.ps1."
    }
    if ($null -ne $LastDiscoveryError) { throw $LastDiscoveryError }
    throw 'Python 3.12 was not found after installation.'
}

function Assert-CityRecoveryPython312 {
    param([Parameter(Mandatory = $true)][string]$PythonPath)

    if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
        throw "The project Python environment is missing: $PythonPath. Run scripts\setup.ps1 first."
    }
    & $PythonPath -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) and sys.maxsize > 2**32 else 1)'
    if ($LASTEXITCODE -ne 0) {
        throw "The project environment is not 64-bit Python 3.12: $PythonPath. Rerun scripts\setup.ps1."
    }
}

function Assert-CityRecoveryNativePathBudget {
    param([Parameter(Mandatory = $true)]$Context)

    if ($Context.NativePath.Length -ge 240) {
        throw "The ONNX Runtime native-library path is too long for reliable loading on Windows: $($Context.NativePath)"
    }
    if (-not (Test-Path -LiteralPath $Context.NativePath -PathType Leaf)) {
        throw "ONNX Runtime was not installed correctly: $($Context.NativePath) is missing."
    }
    Write-Host "[environment] Python: $($Context.PythonPath)"
    Write-Host "[environment] Native path budget: PASS ($($Context.NativePath.Length) characters, $($Context.Mode))"
}
