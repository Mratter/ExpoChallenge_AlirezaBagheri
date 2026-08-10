$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = Split-Path -Parent $PSScriptRoot
. (Join-Path $Root 'scripts\project_environment.ps1')

function Assert-True {
    param(
        [Parameter(Mandatory = $true)][bool]$Condition,
        [Parameter(Mandatory = $true)][string]$Message
    )
    if (-not $Condition) { throw $Message }
}

$TemporaryRoot = Join-Path ([IO.Path]::GetTempPath()) "city-recovery-python-discovery-$([Guid]::NewGuid().ToString('N'))"
try {
    $ExpectedPath = Join-Path $TemporaryRoot 'Programs\Python\Python312\python.exe'
    New-Item -ItemType Directory -Path (Split-Path -Parent $ExpectedPath) -Force | Out-Null
    New-Item -ItemType File -Path $ExpectedPath -Force | Out-Null

    $KnownPaths = @(Get-CityRecoveryPython312KnownPaths `
        -LocalAppDataRoot $TemporaryRoot `
        -ProgramFilesRoots @())
    Assert-True `
        -Condition ($KnownPaths -contains $ExpectedPath) `
        -Message 'The standard per-user Winget/python.org Python 3.12 location was not enumerated.'
    Assert-True `
        -Condition (Test-CityRecoveryPythonAppExecutionAlias -Path 'C:\Users\Demo\AppData\Local\Microsoft\WindowsApps\python.exe') `
        -Message 'The Windows Store Python AppExecutionAlias was not recognized.'
    Assert-True `
        -Condition (-not (Test-CityRecoveryPythonAppExecutionAlias -Path 'C:\Users\Demo\AppData\Local\Programs\Python\Python312\python.exe')) `
        -Message 'A real per-user Python installation was misclassified as a Store alias.'

    $LongRoot = Join-Path $TemporaryRoot (('nested-' * 28).TrimEnd('-'))
    $ShortPathContext = Get-CityRecoveryEnvironmentContext -Root $LongRoot
    Assert-True `
        -Condition ($ShortPathContext.Mode -eq 'short-path') `
        -Message 'A long repository path did not select the loader-safe environment.'
    Assert-True `
        -Condition ($ShortPathContext.EnvironmentPath -like '*\Innoverse\city-recovery\py312-*') `
        -Message 'The loader-safe environment path still exposes a retired release name.'

    $OriginalPath = $env:Path
    try {
        # Reproduce a stale parent shell: neither the newly installed Python
        # directory nor its launcher has been added to this process PATH.
        $env:Path = "$env:SystemRoot\System32"
        $InstalledPython = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'
        Assert-True `
            -Condition (Test-Path -LiteralPath $InstalledPython -PathType Leaf) `
            -Message "The regression host is missing the Winget Python fixture: $InstalledPython"
        $DirectCandidate = [pscustomobject]@{ Command = $InstalledPython; Prefix = @() }
        Assert-True `
            -Condition (Test-CityRecoveryPython312Launcher -Candidate $DirectCandidate) `
            -Message 'The direct Winget-installed Python fixture did not pass the 3.12/64-bit probe.'
        $Launcher = Get-CityRecoveryPython312Launcher
        Assert-True `
            -Condition (Test-CityRecoveryPython312Launcher -Candidate $Launcher) `
            -Message 'A verified 64-bit Python 3.12 was not found outside stale PATH state.'

        # Winget uses a non-zero result for "already installed/no upgrade" on
        # some versions. A verified interpreter must win over that advisory.
        $Resolved = Wait-CityRecoveryPython312Launcher `
            -Attempts 1 `
            -DelayMilliseconds 0 `
            -InstallerExitCode 1
        Assert-True `
            -Condition (Test-CityRecoveryPython312Launcher -Candidate $Resolved) `
            -Message 'A verified existing Python was rejected because Winget returned non-zero.'
    }
    finally {
        $env:Path = $OriginalPath
    }

    foreach ($AcceptedNodeVersion in @('v20.19.0', '20.20.1', 'v22.12.0', 'v24.0.0')) {
        Assert-True `
            -Condition (Test-CityRecoveryNodeVersion -VersionText $AcceptedNodeVersion) `
            -Message "Supported Node LTS version was rejected: $AcceptedNodeVersion"
    }
    foreach ($RejectedNodeVersion in @('v20.18.9', 'v21.7.3', 'v22.11.0', 'v23.11.1', 'not-a-version')) {
        Assert-True `
            -Condition (-not (Test-CityRecoveryNodeVersion -VersionText $RejectedNodeVersion)) `
            -Message "Unsupported Node version was accepted: $RejectedNodeVersion"
    }
    $CurrentNode = Get-CityRecoveryNodeCommand
    if ($null -ne $CurrentNode) {
        Assert-True `
            -Condition (Test-CityRecoveryNodeCommand -Command $CurrentNode) `
            -Message 'The discovered Node executable failed its LTS compatibility probe.'
        $ResolvedNode = Wait-CityRecoveryNodeCommand `
            -Attempts 1 `
            -DelayMilliseconds 0 `
            -InstallerExitCode 1
        Assert-True `
            -Condition (Test-CityRecoveryNodeCommand -Command $ResolvedNode) `
            -Message 'A compatible installed Node was rejected because Winget returned non-zero.'
    }

    $Context = Get-CityRecoveryEnvironmentContext -Root $Root
    Assert-CityRecoveryPython312 -PythonPath $Context.PythonPath

    Write-Host 'setup-python-discovery-regression-passed'
}
finally {
    if (Test-Path -LiteralPath $TemporaryRoot) {
        Remove-Item -LiteralPath $TemporaryRoot -Recurse -Force
    }
}
