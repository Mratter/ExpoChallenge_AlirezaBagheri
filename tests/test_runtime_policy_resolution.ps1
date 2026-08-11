$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = Split-Path -Parent $PSScriptRoot
. (Join-Path $Root 'scripts\runtime_policy.ps1')

function Assert-Equal {
    param(
        [Parameter(Mandatory = $true)]$Actual,
        [Parameter(Mandatory = $true)]$Expected,
        [Parameter(Mandatory = $true)][string]$Message
    )
    if ($Actual -ne $Expected) {
        throw "$Message Expected '$Expected', got '$Actual'."
    }
}

$TemporaryRoot = Join-Path ([IO.Path]::GetTempPath()) "city-recovery-policy-$([Guid]::NewGuid().ToString('N'))"
try {
    $Artifacts = Join-Path $TemporaryRoot 'artifacts'
    New-Item -ItemType Directory -Path $Artifacts -Force | Out-Null
    $Bundled = Join-Path $Artifacts 'city_recovery_ppo.v4.onnx'
    $Environment = Join-Path $TemporaryRoot 'environment.onnx'
    $Explicit = Join-Path $TemporaryRoot 'explicit.onnx'
    New-Item -ItemType File -Path $Bundled, $Environment, $Explicit -Force | Out-Null

    $DefaultResult = Resolve-CityRecoveryPolicyPath `
        -Root $TemporaryRoot `
        -ExplicitPolicyProvided $false
    Assert-Equal `
        -Actual $DefaultResult `
        -Expected (Resolve-Path -LiteralPath $Bundled).Path `
        -Message 'The zero-configuration path did not select the bundled policy.'

    $EnvironmentResult = Resolve-CityRecoveryPolicyPath `
        -Root $TemporaryRoot `
        -ExplicitPolicyProvided $false `
        -EnvironmentPolicyPath $Environment
    Assert-Equal `
        -Actual $EnvironmentResult `
        -Expected (Resolve-Path -LiteralPath $Environment).Path `
        -Message 'The environment path did not override the bundled policy.'

    $ExplicitResult = Resolve-CityRecoveryPolicyPath `
        -Root $TemporaryRoot `
        -ExplicitPolicyPath $Explicit `
        -ExplicitPolicyProvided $true `
        -EnvironmentPolicyPath $Environment
    Assert-Equal `
        -Actual $ExplicitResult `
        -Expected (Resolve-Path -LiteralPath $Explicit).Path `
        -Message 'The explicit path did not override the environment path.'

    $ExplicitFailure = $false
    try {
        Resolve-CityRecoveryPolicyPath `
            -Root $TemporaryRoot `
            -ExplicitPolicyPath (Join-Path $TemporaryRoot 'missing-explicit.onnx') `
            -ExplicitPolicyProvided $true `
            -EnvironmentPolicyPath $Environment | Out-Null
    }
    catch {
        $ExplicitFailure = $true
    }
    Assert-Equal `
        -Actual $ExplicitFailure `
        -Expected $true `
        -Message 'An invalid explicit path silently fell back to a lower-priority policy.'

    $MissingBundleRoot = Join-Path $TemporaryRoot 'missing-bundle-root'
    New-Item -ItemType Directory -Path $MissingBundleRoot -Force | Out-Null
    $DefaultFailure = $false
    try {
        Resolve-CityRecoveryPolicyPath `
            -Root $MissingBundleRoot `
            -ExplicitPolicyProvided $false | Out-Null
    }
    catch {
        $DefaultFailure = $true
    }
    Assert-Equal `
        -Actual $DefaultFailure `
        -Expected $true `
        -Message 'A missing bundled policy did not fail closed.'

    Write-Host 'runtime-policy-resolution-regression-passed'
}
finally {
    if (Test-Path -LiteralPath $TemporaryRoot) {
        Remove-Item -LiteralPath $TemporaryRoot -Recurse -Force
    }
}
