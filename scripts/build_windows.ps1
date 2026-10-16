<#
.SYNOPSIS
Build the unsigned Windows one-folder ZIP and per-user Inno Setup installer.
.DESCRIPTION
Use a dedicated build environment with the two generated Windows locks.
Internal mode is never publishable. Release mode requires an explicit owner
attestation, an owner-supplied EULA and its approved SHA256 digest.
This script does not install tools, accept vendor licenses, or run installers.
#>
[CmdletBinding()]
param(
    [ValidateSet('internal', 'release')]
    [Parameter(Mandatory = $true)]
    [string]$Mode,
    [string]$Python = '',
    [string]$NativeProbe = '',
    [string]$OutputDirectory = '',
    [string]$InnoCompiler = '',
    [switch]$OwnerApprovedVendorRedistribution,
    [string]$VendorEula = '',
    [string]$VendorEulaSha256 = '',
    [string]$ExpectedTag = ''
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
if (-not $Python) {
    $Python = Join-Path $root 'build\windows-env\Scripts\python.exe'
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Build Python is missing: $Python. Create the dedicated build venv first."
}
$arguments = @(
    (Join-Path $PSScriptRoot 'build_windows.py'),
    '--mode', $Mode
)
$options = @{
    '--native-probe' = $NativeProbe
    '--output-dir' = $OutputDirectory
    '--iscc' = $InnoCompiler
    '--vendor-eula' = $VendorEula
    '--vendor-eula-sha256' = $VendorEulaSha256
    '--expected-tag' = $ExpectedTag
}
foreach ($option in $options.GetEnumerator()) {
    if ($option.Value) {
        $arguments += @($option.Key, $option.Value)
    }
}
if ($OwnerApprovedVendorRedistribution) {
    $arguments += '--owner-approved-vendor-redistribution'
}
& $Python @arguments
exit $LASTEXITCODE
