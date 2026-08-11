function Resolve-CityRecoveryPolicyPath {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [AllowNull()][AllowEmptyString()][string]$ExplicitPolicyPath,
        [bool]$ExplicitPolicyProvided = $false,
        [AllowNull()][AllowEmptyString()][string]$EnvironmentPolicyPath
    )

    if ($ExplicitPolicyProvided) {
        if ([string]::IsNullOrWhiteSpace($ExplicitPolicyPath)) {
            throw 'The explicit -PolicyPath value must not be empty.'
        }
        $PolicyPathText = $ExplicitPolicyPath
    }
    elseif (-not [string]::IsNullOrWhiteSpace($EnvironmentPolicyPath)) {
        $PolicyPathText = $EnvironmentPolicyPath
    }
    else {
        $PolicyPathText = Join-Path $Root 'artifacts\city_recovery_ppo.v4.onnx'
    }

    try {
        $ResolvedPolicyPath = (Resolve-Path -LiteralPath $PolicyPathText -ErrorAction Stop).Path
    }
    catch {
        throw "The selected policy does not exist: $PolicyPathText"
    }
    if (-not (Test-Path -LiteralPath $ResolvedPolicyPath -PathType Leaf)) {
        throw "The selected policy is not a file: $ResolvedPolicyPath"
    }
    return $ResolvedPolicyPath
}
