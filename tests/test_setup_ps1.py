"""Source setup checks in Windows PowerShell 5.1 and PowerShell 7."""

from __future__ import annotations

import base64
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "setup.ps1"
SHELLS = [
    executable
    for name in ("powershell.exe", "pwsh.exe")
    if (executable := shutil.which(name)) is not None
]
pytestmark = pytest.mark.skipif(
    sys.platform != "win32" or not SHELLS,
    reason="Windows source setup requires a Windows PowerShell host",
)


def ps_string(value: str | Path) -> str:
    """Quote a literal inside PowerShell test source, without interpolation."""
    return "'" + str(value).replace("'", "''") + "'"


def run_ps(shell: str, body: str, *, source: bool = True) -> subprocess.CompletedProcess:
    """Run a bounded shell with profile-free, UTF-8 output."""
    code = (
        "[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding\n"
        "$ErrorActionPreference = 'Stop'\n"
        + (f". {ps_string(SCRIPT)}\n" if source else "")
        + body
    )
    encoded = base64.b64encode(code.encode("utf-16-le")).decode("ascii")
    return subprocess.run(
        [shell, "-NoLogo", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
        timeout=30, check=False,
    )


@pytest.mark.parametrize("shell", SHELLS)
def test_script_parses_and_dot_sourcing_is_side_effect_free(shell):
    """Both shells parse the complete script; loading helpers never installs."""
    result = run_ps(
        shell,
        f"""
$tokens = $null
$errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile(
    {ps_string(SCRIPT)}, [ref]$tokens, [ref]$errors
)
if ($errors.Count -gt 0) {{ throw ($errors | Out-String) }}
$commands = $ast.FindAll({{
    param($node)
    $node -is [System.Management.Automation.Language.CommandAst]
}}, $true) | ForEach-Object {{ $_.GetCommandName() }}
foreach ($forbidden in @(
    'Set-ExecutionPolicy', 'Set-ItemProperty', 'Invoke-Expression',
    'winget', 'msiexec', 'cmake'
)) {{
    if ($commands -contains $forbidden) {{ throw "Unexpected call: $forbidden" }}
}}
Write-Output 'parsed'
""",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "parsed"


@pytest.mark.parametrize("shell", SHELLS)
def test_native_arguments_preserve_spaces_quotes_unicode_and_apostrophes(shell):
    """The PS5-compatible ProcessStartInfo quoting round-trips arbitrary argv."""
    values = [
        "", "two words", "O'Brien", 'a"quoted"value',
        "C:\\folder with spaces\\", "\\\\server\\share\\path",
        "Grüße 日本語", "C:\\O'Brien\\模型\\file name.json",
        'backslashes\\\\"and\\',
    ]
    arguments = ", ".join(ps_string(value) for value in values)
    probe = "import json, sys; print(json.dumps(sys.argv[1:], ensure_ascii=False))"
    result = run_ps(
        shell,
        f"""
$values = @({arguments})
$result = Invoke-SetupProcess {ps_string(sys.executable)} (
    @('-I', '-c', {ps_string(probe)}) + $values
)
if ($result.ExitCode -ne 0) {{ throw $result.StdErr }}
Write-Output $result.StdOut
""",
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == values


@pytest.mark.parametrize("shell", SHELLS)
def test_existing_python_is_actually_executed_and_validated(shell):
    """Architecture/version come from the interpreter rather than its name."""
    result = run_ps(
        shell,
        f"""
$info = Test-SetupPython {ps_string(sys.executable)}
if ($null -eq $info) {{ throw 'Python validation failed' }}
$info | ConvertTo-Json -Compress
""",
    )
    assert result.returncode == 0, result.stderr
    info = json.loads(result.stdout)
    assert info["bits"] == 64
    assert tuple(info["version"]) >= (3, 11)
    assert Path(info["executable"]) == Path(sys.executable)


@pytest.mark.parametrize("shell", SHELLS)
@pytest.mark.parametrize(
    ("version", "bits", "machine", "valid"),
    [
        ([3, 10, 9], 64, "AMD64", False),
        ([3, 11, 0], 32, "AMD64", False),
        ([3, 13, 0], 64, "ARM64", False),
        ([3, 11, 0], 64, "AMD64", True),
    ],
)
def test_python_candidates_reject_unsupported_architectures_and_versions(
    shell, version, bits, machine, valid
):
    """A real executable is still rejected if it is old, x86, or ARM64."""
    metadata = json.dumps({
        "executable": sys.executable,
        "version": version, "bits": bits, "machine": machine, "platform": "win32",
    })
    result = run_ps(
        shell,
        f"""
function Invoke-SetupProcess {{
    return [pscustomobject]@{{
        ExitCode = 0; StdOut = {ps_string(metadata)}; StdErr = ''
    }}
}}
$info = Test-SetupPython 'candidate.exe'
Write-Output ($null -ne $info).ToString()
""",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().lower() == str(valid).lower()


@pytest.mark.parametrize("shell", SHELLS)
def test_store_alias_is_never_launched(shell):
    """WindowsApps aliases must not open the Store or hang Python discovery."""
    result = run_ps(
        shell,
        r"""
function Invoke-SetupProcess { throw 'Store alias was executed' }
$info = Test-SetupPython 'C:\Users\Example\AppData\Local\Microsoft\WindowsApps\python.exe'
Write-Output ($null -eq $info).ToString()
""",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "True"


@pytest.mark.parametrize("shell", SHELLS)
def test_nonzero_external_exit_codes_are_preserved(shell):
    """A started process is not automatically considered a successful check."""
    result = run_ps(
        shell,
        f"""
$result = Invoke-SetupProcess {ps_string(sys.executable)} @(
    '-I', '-c', 'import sys; sys.exit(23)'
)
Write-Output $result.ExitCode
""",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "23"


@pytest.mark.parametrize("shell", SHELLS)
def test_external_checks_are_bounded_and_stop_only_their_process(shell):
    """Timeouts produce visible failure rather than hanging setup indefinitely."""
    result = run_ps(
        shell,
        f"""
try {{
    Invoke-SetupProcess {ps_string(sys.executable)} @(
        '-I', '-c', 'import time; time.sleep(20)'
    ) -TimeoutSeconds 1
    throw 'Timeout was not enforced'
}}
catch {{
    if ("$_" -notmatch 'Timed out after 1s') {{ throw }}
    Write-Output 'bounded'
}}
""",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "bounded"


@pytest.mark.parametrize("shell", SHELLS)
@pytest.mark.parametrize(("cli", "native", "exit_code"), [
    (True, True, 0), (True, False, 2), (False, True, 4), (False, False, 6),
])
def test_check_only_reuses_venv_and_reports_separate_readiness(
    shell, cli, native, exit_code
):
    """A reusable venv is untouched; optional running state is not readiness."""
    result = run_ps(
        shell,
        f"""
function Find-SetupPython {{ throw 'Existing venv was ignored' }}
function Install-SourceRequirements {{ throw 'CheckOnly attempted installation' }}
function Test-SourceRequirements {{ param($Interpreter) }}
function Test-LmStudioReadiness {{ return ${str(cli).lower()} }}
function Test-NativeTelemetry {{ param($Root); return ${str(native).lower()} }}
$code = Invoke-SourceSetup -Root {ps_string(ROOT)} -OnlyCheck
Write-Output "RESULT=$code"
""",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-1] == f"RESULT={exit_code}"
    assert "Reusing Python" in result.stdout
    assert "Core source installation: READY" in result.stdout


@pytest.mark.parametrize("shell", SHELLS)
def test_missing_native_helper_is_explicit_and_never_built(shell, tmp_path):
    """Missing native telemetry is not hidden behind successful pip imports."""
    result = run_ps(
        shell,
        f"""
$ready = Test-NativeTelemetry {ps_string(tmp_path)}
Write-Output "READY=$ready"
""",
    )
    assert result.returncode == 0, result.stderr
    assert "Native telemetry: INCOMPLETE" in result.stdout
    assert "gpu_probe.exe" in result.stdout
    assert "Only if you accept BOTH" in result.stdout
    assert result.stdout.splitlines()[-1] == "READY=False"
    assert not (tmp_path / "build").exists()


@pytest.mark.parametrize("shell", SHELLS)
def test_offline_lmstudio_is_not_a_setup_failure(shell, tmp_path):
    """A verified lms CLI with an offline status is not treated as missing."""
    lms = tmp_path / "lms.exe"
    lms.write_bytes(b"test-only; never executed")
    result = run_ps(
        shell,
        f"""
function Get-Command {{
    return [pscustomobject]@{{ Source = {ps_string(lms)} }}
}}
function Invoke-SetupProcess {{
    param($FilePath, $ArgumentList, $TimeoutSeconds)
    if ($ArgumentList[0] -eq '--version') {{
        return [pscustomobject]@{{ ExitCode = 0; StdOut = '1.0.0'; StdErr = '' }}
    }}
    return [pscustomobject]@{{
        ExitCode = 1; StdOut = 'LM Studio is not running'; StdErr = ''
    }}
}}
$ready = Test-LmStudioReadiness
Write-Output "READY=$ready"
""",
    )
    assert result.returncode == 0, result.stderr
    assert "OFFLINE/not ready" in result.stdout
    assert result.stdout.splitlines()[-1] == "READY=True"


@pytest.mark.parametrize("shell", SHELLS)
def test_native_helper_without_amd_hardware_is_ready(shell, tmp_path):
    """No AMD adapter and unsupported sensors are valid, explicit outcomes."""
    native = tmp_path / "build" / "windows_gpu" / "bin"
    for relative in (
        "gpu_probe.exe", "licenses\\ADLX SDK License Agreement.pdf",
        "licenses\\IGCL-LICENSE.txt", "licenses\\NOTICE.txt",
    ):
        path = native / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test-only")
    snapshot = json.dumps({
        "protocol_version": 1,
        "devices": [{
            "vendor": "Intel", "name": "Test GPU",
            "status": {
                "temp_celsius": "unsupported", "power_watts": "unavailable",
                "vram_gb": "ok", "gtt_gb": "ok",
            },
        }],
        "errors": [],
    })
    result = run_ps(
        shell,
        f"""
function Invoke-SetupProcess {{
    return [pscustomobject]@{{
        ExitCode = 0; StdOut = {ps_string(snapshot)}; StdErr = ''
    }}
}}
$ready = Test-NativeTelemetry {ps_string(tmp_path)}
Write-Output "READY=$ready"
""",
    )
    assert result.returncode == 0, result.stderr
    assert "temperature=unsupported" in result.stdout
    assert "Native telemetry component: READY" in result.stdout
    assert result.stdout.splitlines()[-1] == "READY=True"
