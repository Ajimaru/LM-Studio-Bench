#requires -Version 5.1
<#
.SYNOPSIS
Install/check the Windows x64 source environment without machine-wide changes.
.DESCRIPTION
Requires Windows 10/11 x64 and an already installed Python >=3.11 x64.
Creates or reuses .venv and installs requirements.txt using that interpreter.
Never activates a venv, changes PATH/ExecutionPolicy, installs system tools,
starts models, builds native code, or accepts vendor SDK licenses.

Exit codes: 0 = ready (LM Studio may be offline), 1 = core setup failure,
2 = native telemetry incomplete, 4 = lms CLI missing/unusable,
6 = both native telemetry and lms CLI incomplete.
.PARAMETER Python
Optional path to a real x64 Python executable, not a Windows Store alias.
.PARAMETER CheckOnly
Check the existing .venv, CLI and native helper without installing anything.
.EXAMPLE
.\setup.ps1
.EXAMPLE
.\setup.ps1 -Python 'C:\Program Files\Python313\python.exe'
.EXAMPLE
.\setup.ps1 -CheckOnly
#>
[CmdletBinding()]
param(
    [string]$Python,
    [switch]$CheckOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function ConvertTo-ProcessArgument {
    <# Quote an argv element using Windows CRT rules, including trailing \. #>
    param([AllowEmptyString()][string]$Value)

    $quoted = New-Object System.Text.StringBuilder
    [void]$quoted.Append('"')
    $slashes = 0
    foreach ($character in $Value.ToCharArray()) {
        if ($character -eq '\') {
            $slashes++
        }
        elseif ($character -eq '"') {
            [void]$quoted.Append(('\' * (2 * $slashes + 1)))
            [void]$quoted.Append('"')
            $slashes = 0
        }
        else {
            [void]$quoted.Append(('\' * $slashes))
            [void]$quoted.Append($character)
            $slashes = 0
        }
    }
    [void]$quoted.Append(('\' * (2 * $slashes)))
    [void]$quoted.Append('"')
    return $quoted.ToString()
}

function Invoke-SetupProcess {
    <# Run a bounded, read-only external check with lossless argv forwarding. #>
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [string[]]$ArgumentList = @(),
        [int]$TimeoutSeconds = 10,
        [string]$WorkingDirectory = $PSScriptRoot
    )

    $startInfo = New-Object System.Diagnostics.ProcessStartInfo
    $startInfo.FileName = $FilePath
    $startInfo.Arguments = (
        $ArgumentList | ForEach-Object { ConvertTo-ProcessArgument $_ }
    ) -join ' '
    $startInfo.WorkingDirectory = $WorkingDirectory
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    $utf8 = New-Object System.Text.UTF8Encoding
    $startInfo.StandardOutputEncoding = $utf8
    $startInfo.StandardErrorEncoding = $utf8
    $startInfo.EnvironmentVariables['PYTHONUTF8'] = '1'
    $startInfo.EnvironmentVariables['PYTHONIOENCODING'] = 'utf-8'
    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $startInfo
    try {
        if (-not $process.Start()) {
            throw "Could not start $FilePath"
        }
        $stdout = $process.StandardOutput.ReadToEndAsync()
        $stderr = $process.StandardError.ReadToEndAsync()
        if (-not $process.WaitForExit($TimeoutSeconds * 1000)) {
            Stop-Process -Id $process.Id -Force -ErrorAction Stop
            [void]$process.WaitForExit(2000)
            throw "Timed out after ${TimeoutSeconds}s: $FilePath"
        }
        # A child inheriting a pipe must not turn a bounded check into a hang.
        if (-not $stdout.Wait(2000) -or -not $stderr.Wait(2000)) {
            throw "Output pipes did not close: $FilePath"
        }
        return [pscustomobject]@{
            ExitCode = $process.ExitCode
            StdOut = $stdout.Result.Trim()
            StdErr = $stderr.Result.Trim()
        }
    }
    finally {
        $process.Dispose()
    }
}

function Test-SetupPython {
    <# Return verified interpreter metadata, or null for an unusable candidate. #>
    param([string]$FilePath, [string[]]$Prefix = @())

    if ($FilePath -match '(?i)\\Microsoft\\WindowsApps\\') {
        Write-Verbose "Skipping Windows Store execution alias: $FilePath"
        return $null
    }
    $probe = @'
import json, platform, struct, sys
print(json.dumps({
    "executable": sys.executable,
    "version": list(sys.version_info[:3]),
    "bits": struct.calcsize("P") * 8,
    "machine": platform.machine(),
    "platform": sys.platform,
}))
'@
    try {
        $result = Invoke-SetupProcess $FilePath ($Prefix + @('-I', '-c', $probe))
        if ($result.ExitCode -ne 0) {
            Write-Verbose "Python candidate failed: $FilePath; $($result.StdErr)"
            return $null
        }
        $info = $result.StdOut | ConvertFrom-Json
        $version = $info.version
        if (
            $info.platform -ne 'win32' -or $info.bits -ne 64 -or
            $info.machine -notin @('AMD64', 'x86_64') -or
            $version[0] -ne 3 -or $version[1] -lt 11 -or
            -not (Test-Path -LiteralPath $info.executable -PathType Leaf)
        ) {
            Write-Verbose "Not a supported Python >=3.11 x64: $FilePath"
            return $null
        }
        return $info
    }
    catch {
        Write-Verbose "Python candidate rejected: $FilePath; $_"
        return $null
    }
}

function Find-SetupPython {
    <# Try explicit Python, py launcher, real PATH candidates, then existing uv. #>
    param([string]$PythonPath)

    if ($PythonPath) {
        $info = Test-SetupPython $PythonPath
        if ($null -eq $info) {
            throw "The requested Python is not usable Python >=3.11 x64: $PythonPath"
        }
        return $info
    }
    $launchers = @(Get-Command py.exe -All -CommandType Application `
        -ErrorAction SilentlyContinue)
    foreach ($launcher in $launchers) {
        foreach ($selector in @('-3-64', '-3')) {
            $info = Test-SetupPython $launcher.Source @($selector)
            if ($null -ne $info) { return $info }
        }
    }
    $candidates = @(Get-Command python.exe, python3.exe -All `
        -CommandType Application -ErrorAction SilentlyContinue)
    foreach ($candidate in $candidates) {
        $info = Test-SetupPython $candidate.Source
        if ($null -ne $info) { return $info }
    }
    $uv = Get-Command uv.exe -CommandType Application -ErrorAction SilentlyContinue
    if ($null -ne $uv) {
        try {
            $found = Invoke-SetupProcess $uv.Source @(
                'python', 'find', '--no-python-downloads', '--no-config', '>=3.11'
            )
            if ($found.ExitCode -eq 0) {
                $info = Test-SetupPython $found.StdOut
                if ($null -ne $info) { return $info }
            }
            else {
                Write-Verbose "uv did not find Python: $($found.StdErr)"
            }
        }
        catch {
            Write-Verbose "Existing uv Python discovery failed: $_"
        }
    }
    throw (
        'Install Python >=3.11 x64 yourself, then rerun with ' +
        "-Python 'C:\path\to\python.exe'. Windows Store aliases are not Python."
    )
}

function Install-SourceRequirements {
    <# Use pip, avoiding uv hardlinks in OneDrive checkouts. #>
    param([string]$Interpreter, [string]$Root)

    $pip = Invoke-SetupProcess $Interpreter @('-m', 'pip', '--version')
    if ($pip.ExitCode -ne 0) {
        & $Interpreter -m ensurepip --upgrade
        if ($LASTEXITCODE -ne 0) { throw 'Could not bootstrap venv pip.' }
    }
    $requirements = Join-Path $Root 'requirements.txt'
    Write-Host "Installing source requirements into $Interpreter"
    & $Interpreter -m pip install --disable-pip-version-check `
        --timeout 30 --retries 2 -r $requirements
    if ($LASTEXITCODE -ne 0) {
        throw "pip install failed (exit $LASTEXITCODE); core setup is incomplete."
    }
}

function Test-SourceRequirements {
    <# Check imports and dependency consistency without running benchmarks. #>
    param([string]$Interpreter)

    $imports = @'
import importlib, json, sys
modules = (
    "lmstudio", "tqdm", "psutil", "reportlab", "plotly", "fastapi", "uvicorn",
    "jinja2", "httpx", "cpuinfo", "scipy", "pydantic", "yaml", "pystray", "PIL"
)
errors = []
for name in modules:
    try:
        importlib.import_module(name)
    except Exception as exc:
        errors.append(f"{name}: {exc}")
print(json.dumps(errors))
sys.exit(bool(errors))
'@
    $result = Invoke-SetupProcess $Interpreter @('-I', '-c', $imports) `
        -TimeoutSeconds 60
    if ($result.ExitCode -ne 0) {
        throw "Runtime imports failed: $($result.StdOut) $($result.StdErr)"
    }
    $result = Invoke-SetupProcess $Interpreter @('-m', 'pip', 'check') `
        -TimeoutSeconds 30
    if ($result.ExitCode -ne 0) {
        throw "Runtime dependency check failed: $($result.StdOut) $($result.StdErr)"
    }
}

function Test-LmStudioReadiness {
    <# Distinguish a missing CLI from an installed but offline LM Studio. #>
    $lms = Get-Command lms.exe -CommandType Application -ErrorAction SilentlyContinue
    $lmsPath = if ($null -ne $lms) { $lms.Source } else {
        Join-Path ([Environment]::GetFolderPath('UserProfile')) '.lmstudio\bin\lms.exe'
    }
    if (-not (Test-Path -LiteralPath $lmsPath -PathType Leaf)) {
        Write-Warning (
            'LM Studio CLI: MISSING. Install/configure the lms CLI from LM Studio, ' +
            'then reopen PowerShell and check: lms --version; lms status'
        )
        return $false
    }
    try {
        $version = Invoke-SetupProcess $lmsPath @('--version') -TimeoutSeconds 5
        if ($version.ExitCode -ne 0) {
            throw "lms --version failed: $($version.StdErr)"
        }
    }
    catch {
        Write-Warning "LM Studio CLI: UNUSABLE. $_"
        return $false
    }
    Write-Host "LM Studio CLI: installed ($($version.StdOut))"
    try {
        $status = Invoke-SetupProcess $lmsPath @('status') -TimeoutSeconds 5
        if ($status.ExitCode -ne 0 -or $status.StdOut -match
            '(?i)not running|offline|not ready|not started|stopped') {
            Write-Host 'LM Studio: OFFLINE/not ready (not a setup failure).'
        }
        else {
            Write-Host 'LM Studio status (no models were started):'
        }
        Write-Host "$($status.StdOut) $($status.StdErr)".Trim()
    }
    catch {
        Write-Warning "LM Studio status unavailable/offline (not a setup failure): $_"
    }
    return $true
}

function Show-NativeBuildInstructions {
    <# Print opt-in build instructions; never execute license acceptance. #>
    Write-Host @'
Native AMD/Intel telemetry requires the local CMake helper and vendor licenses.
Review native\gpu_probe\CMakeLists.txt and its pinned ADLX/IGCL license links.
Install CMake >=3.24 and MSVC x64 C++ build tools yourself if needed.
Only if you accept BOTH vendor SDK licenses, run from this checkout:
  cmake -S ".\native\gpu_probe" -B ".\build\windows_gpu" -A x64 `
    -DGPU_PROBE_ACCEPT_VENDOR_LICENSES=ON
  cmake --build ".\build\windows_gpu" --config Release
  & ".\build\windows_gpu\bin\gpu_probe.exe" --once
  .\setup.ps1 -CheckOnly
Do not redistribute the helper/SDKs without reviewing their separate terms.
Missing sensors on unsupported hardware are not fabricated or reported as zero.
'@
}

function Test-NativeTelemetry {
    <# Validate the local component, not the presence of any particular GPU. #>
    param([string]$Root)

    $directory = Join-Path $Root 'build\windows_gpu\bin'
    $helper = Join-Path $directory 'gpu_probe.exe'
    $requiredFiles = @(
        $helper,
        (Join-Path $directory 'licenses\ADLX SDK License Agreement.pdf'),
        (Join-Path $directory 'licenses\IGCL-LICENSE.txt'),
        (Join-Path $directory 'licenses\NOTICE.txt')
    )
    foreach ($file in $requiredFiles) {
        if (-not (Test-Path -LiteralPath $file -PathType Leaf)) {
            Write-Warning "Native telemetry: INCOMPLETE; missing $file"
            Show-NativeBuildInstructions
            return $false
        }
    }
    try {
        $probe = Invoke-SetupProcess $helper @('--once') -TimeoutSeconds 20
        if ($probe.ExitCode -ne 0) {
            throw "gpu_probe failed (exit $($probe.ExitCode)): $($probe.StdErr)"
        }
        $snapshot = $probe.StdOut | ConvertFrom-Json
        if ($snapshot.protocol_version -ne 1 -or $null -eq $snapshot.devices) {
            throw 'The helper returned an incompatible snapshot.'
        }
        foreach ($device in $snapshot.devices) {
            Write-Host (
                "  $($device.vendor) $($device.name): " +
                "temperature=$($device.status.temp_celsius), " +
                "power=$($device.status.power_watts), " +
                "VRAM=$($device.status.vram_gb), GTT=$($device.status.gtt_gb)"
            )
        }
        if (@($snapshot.errors).Count -gt 0) {
            throw ($snapshot.errors -join '; ')
        }
        Write-Host (
            'Native telemetry component: READY. Actual sensor availability ' +
            'is hardware/driver-dependent; absent AMD hardware is not an error.'
        )
        return $true
    }
    catch {
        Write-Warning "Native telemetry: INCOMPLETE. $_"
        Write-Host "Diagnostic: & '$($helper.Replace("'", "''"))' --once"
        Show-NativeBuildInstructions
        return $false
    }
}

function Invoke-SourceSetup {
    <# Install the core, then report CLI and native readiness independently. #>
    param(
        [string]$Root = $PSScriptRoot,
        [string]$PythonPath,
        [switch]$OnlyCheck
    )

    $architecture = $env:PROCESSOR_ARCHITEW6432
    if (-not $architecture) { $architecture = $env:PROCESSOR_ARCHITECTURE }
    if (
        [Environment]::OSVersion.Platform -ne 'Win32NT' -or
        [Environment]::OSVersion.Version.Major -lt 10 -or
        $architecture -ne 'AMD64'
    ) {
        throw 'This source setup supports Windows 10/11 x64 only, not ARM64.'
    }
    $venv = Join-Path $Root '.venv'
    $interpreter = Join-Path $venv 'Scripts\python.exe'
    if (Test-Path -LiteralPath $venv) {
        $info = Test-SetupPython $interpreter
        if ($null -eq $info) {
            throw (
                "Existing .venv is not usable Python >=3.11 x64: $venv. " +
                'Rename it after backing up any needed data, then rerun setup.'
            )
        }
        Write-Host "Reusing Python $($info.version -join '.') x64: $interpreter"
    }
    elseif ($OnlyCheck) {
        throw 'Core source environment: MISSING. Run .\setup.ps1 first.'
    }
    else {
        $info = Find-SetupPython $PythonPath
        Write-Host "Creating .venv with Python $($info.version -join '.') x64"
        & $info.executable -m venv $venv
        if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
        if ($null -eq (Test-SetupPython $interpreter)) {
            throw 'The created virtual environment failed its interpreter check.'
        }
    }
    if (-not $OnlyCheck) {
        Install-SourceRequirements $interpreter $Root
    }
    Test-SourceRequirements $interpreter
    Write-Host 'Core source installation: READY (no venv activation required).'

    $exitCode = 0
    if (-not (Test-LmStudioReadiness)) { $exitCode = $exitCode -bor 4 }
    if (-not (Test-NativeTelemetry $Root)) { $exitCode = $exitCode -bor 2 }
    $escapedRoot = $Root.Replace("'", "''")
    Write-Host "Run from this checkout: & '$escapedRoot\.venv\Scripts\python.exe' run.py"
    Write-Host "Source setup readiness exit code: $exitCode"
    return $exitCode
}

if ($MyInvocation.InvocationName -ne '.') {
    try {
        exit (Invoke-SourceSetup -PythonPath $Python -OnlyCheck:$CheckOnly)
    }
    catch {
        Write-Error "Core source installation: FAILED. $_" -ErrorAction Continue
        exit 1
    }
}
