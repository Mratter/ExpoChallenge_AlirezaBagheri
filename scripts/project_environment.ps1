function Get-Ai17RootHash {
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

function Enter-Ai17ProjectEnvironment {
    param([Parameter(Mandatory = $true)][string]$Root)

    $CanonicalRoot = [IO.Path]::GetFullPath($Root).TrimEnd([char[]]'\/')
    $OriginalValue = [Environment]::GetEnvironmentVariable('UV_PROJECT_ENVIRONMENT', 'Process')
    $WasPresent = $null -ne $OriginalValue
    $UseDefault = [string]::IsNullOrWhiteSpace($OriginalValue)
    $NativeSuffix = 'Lib\site-packages\onnxruntime\capi\onnxruntime_providers_shared.dll'

    if ($UseDefault) {
        $RepositoryEnvironment = Join-Path $CanonicalRoot '.venv'
        $RepositoryNativePath = Join-Path $RepositoryEnvironment $NativeSuffix
        if ($RepositoryNativePath.Length -lt 240) {
            $EnvironmentPath = $RepositoryEnvironment
            $Mode = 'repo-local'
        }
        else {
            if ([string]::IsNullOrWhiteSpace($env:LOCALAPPDATA)) {
                throw 'LOCALAPPDATA is required to create a loader-safe Python environment for this long project path.'
            }
            $RootHash = Get-Ai17RootHash -Value $CanonicalRoot.ToUpperInvariant()
            $EnvironmentPath = Join-Path $env:LOCALAPPDATA "Innoverse\ai17-city-recovery\environments\py312-$($RootHash.Substring(0, 16))"
            $EnvironmentPath = [IO.Path]::GetFullPath($EnvironmentPath)
            $Mode = 'loader-safe'
        }
    }
    else {
        if ([IO.Path]::IsPathRooted($OriginalValue)) {
            $EnvironmentPath = [IO.Path]::GetFullPath($OriginalValue)
        }
        else {
            $EnvironmentPath = [IO.Path]::GetFullPath((Join-Path $CanonicalRoot $OriginalValue))
        }
        $Mode = 'caller'
    }

    $ProjectedNativePath = Join-Path $EnvironmentPath $NativeSuffix
    if ($ProjectedNativePath.Length -ge 240) {
        throw "UV_PROJECT_ENVIRONMENT must resolve to a shorter path outside this long clone; projected ONNX native path is $($ProjectedNativePath.Length) characters: $ProjectedNativePath"
    }

    if ($UseDefault) {
        [Environment]::SetEnvironmentVariable('UV_PROJECT_ENVIRONMENT', $EnvironmentPath, 'Process')
    }
    Write-Host "[environment] Project root ($($CanonicalRoot.Length) chars): $CanonicalRoot"
    Write-Host "[environment] Python environment ($($EnvironmentPath.Length) chars, $Mode): $EnvironmentPath"

    return [pscustomobject]@{
        EnvironmentPath = $EnvironmentPath
        Mode = $Mode
        OriginalValue = $OriginalValue
        PythonPath = Join-Path $EnvironmentPath 'Scripts\python.exe'
        Root = $CanonicalRoot
        WasDefaulted = $UseDefault
        WasPresent = $WasPresent
    }
}

function Exit-Ai17ProjectEnvironment {
    param([Parameter(Mandatory = $true)]$Context)

    if (-not $Context.WasDefaulted) { return }
    if ($Context.WasPresent) {
        [Environment]::SetEnvironmentVariable('UV_PROJECT_ENVIRONMENT', $Context.OriginalValue, 'Process')
    }
    else {
        Remove-Item -LiteralPath 'Env:UV_PROJECT_ENVIRONMENT' -ErrorAction SilentlyContinue
    }
}

function Assert-Ai17NativePathBudget {
    param([Parameter(Mandatory = $true)]$Context)

    $SitePackages = Join-Path $Context.EnvironmentPath 'Lib\site-packages'
    $NativeFiles = @()
    foreach ($Requirement in @(
        @{ Directory = 'onnx'; Filter = 'onnx_cpp2py_export*.pyd' },
        @{ Directory = 'onnxruntime\capi'; Filter = 'onnxruntime_pybind11_state*.pyd' },
        @{ Directory = 'onnxruntime\capi'; Filter = 'onnxruntime_providers_shared.dll' },
        @{ Directory = 'onnxruntime\capi'; Filter = 'onnxruntime.dll' }
    )) {
        $NativeDirectory = Join-Path $SitePackages $Requirement.Directory
        $Matches = @(Get-ChildItem -LiteralPath $NativeDirectory -Filter $Requirement.Filter -File -ErrorAction SilentlyContinue)
        if ($Matches.Count -eq 0) {
            $ExpectedPath = Join-Path $Requirement.Directory $Requirement.Filter
            throw "Required native dependency is missing after setup: $ExpectedPath"
        }
        $NativeFiles += $Matches
    }

    $TooLong = @($NativeFiles | Where-Object { $_.FullName.Length -ge 240 })
    if ($TooLong.Count -gt 0) {
        $Paths = ($TooLong | ForEach-Object { "$($_.FullName.Length):$($_.FullName)" }) -join '; '
        throw "Native dependency path exceeds the safe Windows loader budget: $Paths"
    }
    $Longest = $NativeFiles | Sort-Object { $_.FullName.Length } -Descending | Select-Object -First 1
    Write-Host "[environment] Native dependency path budget PASS (max $($Longest.FullName.Length) chars)"
}
