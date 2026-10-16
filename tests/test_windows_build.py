"""Frozen launch, portable-package integrity and release-license gate tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock
import zipfile

import pytest

from core import runtime
import run
from scripts import build_windows as builder
from scripts import windows_gui


def _pe_bytes() -> bytes:
    """Return a minimal x64 PE header for non-executable validation fixtures."""
    value = bytearray(128)
    value[:2] = b"MZ"
    struct.pack_into("<I", value, 60, 64)
    value[64:70] = b"PE\x00\x00\x64\x86"
    return bytes(value)


def _vendor_eula(directory: Path) -> Path:
    """Write an explicitly non-legal test fixture, never an actual release EULA."""
    path = directory / "owner-approved-eula.txt"
    path.write_text(
        "UNIT TEST FIXTURE ONLY. This is NOT a redistribution agreement.\n",
        encoding="utf-8",
    )
    return path


def _fake_artifacts(
    output: Path, mode: str = "release", *, manifest_mode: str | None = None,
) -> tuple[dict[str, str], str]:
    """Create inert fake artifacts for checksum/policy verification only."""
    version = "v0.3.1"
    names = builder.artifact_names(version, mode)
    eula = b"UNIT TEST FIXTURE; not a legal agreement or a distributable release."
    eula_digest = hashlib.sha256(eula).hexdigest()
    native = _pe_bytes()
    manifest = {
        "mode": manifest_mode or mode,
        "version": version,
        "owner_approved_vendor_redistribution": mode == "release",
        "vendor_eula_sha256": eula_digest,
        "native_probe_sha256": hashlib.sha256(native).hexdigest(),
    }
    data = {
        "BUILD-METADATA.json": json.dumps(manifest).encode(),
        "LM-Studio-Bench.exe": _pe_bytes(),
        "lmstudio-bench.exe": _pe_bytes(),
        "LICENSE": b"test MIT notice",
        "_internal/native/gpu_probe.exe": native,
        "_internal/code_runtime/python.exe": _pe_bytes(),
        "_internal/code_runtime/LICENSE.txt": b"Python license fixture",
        "_internal/web/templates/dashboard.html.jinja": b"<html></html>",
        "_internal/cli/report_template.html.template": b"<html></html>",
        "VENDOR-EULA.txt": eula,
    }
    data.update({
        f"_internal/native/licenses/{name}": b"Vendor notice fixture"
        for name in builder.VENDOR_LICENSES
    })
    with zipfile.ZipFile(output / names["zip"], "w") as archive:
        for name, content in data.items():
            archive.writestr(f"{builder.PRODUCT}/{name}", content)
    (output / names["installer"]).write_bytes(b"MZinert test fixture")
    _write_checksums(output, names)
    return names, eula_digest


def _write_checksums(output: Path, names: dict[str, str]) -> None:
    """Write checksums for inert synthetic test artifacts."""
    (output / builder.CHECKSUMS).write_text(
        "".join(
            f"{builder.sha256(output / names[kind])}  {names[kind]}\n"
            for kind in ("zip", "installer")
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("role", "module_name"),
    [
        ("webapp", "web.app"), ("classic", "cli.benchmark"),
        ("capability", "cli.main"), ("tray", "core.tray"),
        ("metadata", "tools.scrape_metadata"),
        ("selftest", "scripts.windows_selftest"),
    ],
)
def test_internal_dispatch_preserves_argv_cwd_and_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, role: str, module_name: str,
) -> None:
    """Every fixed role gets untouched arguments and the packaged resource cwd."""
    original_argv = sys.argv
    original_cwd = Path.cwd()
    args = ["--sample", "C:\\Grüße 日本語's\\One Drive\\file.txt", "--help"]
    called = []

    def entrypoint() -> int:
        called.append((list(sys.argv), Path.cwd()))
        return 7

    module = ModuleType(module_name)
    module.main = entrypoint
    monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.setattr(run, "project_root", tmp_path)
    assert run.main(["--internal-role", role, *args]) == 7
    assert called == [([sys.executable, *args], tmp_path)]
    assert sys.argv is original_argv
    assert Path.cwd() == original_cwd


@pytest.mark.parametrize(
    "args", [["--internal-role"], ["--internal-role", "os.system"],
             ["--internal-role", "..\\arbitrary.py"]],
)
def test_internal_dispatch_rejects_unknown_roles(args: list[str]) -> None:
    """Internal flags cannot evaluate arbitrary modules or files."""
    assert run.main(args) == 2


def test_dispatch_restores_state_after_argparse_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Argparse's help/error exit semantics survive the dispatch boundary."""
    original_argv = sys.argv
    original_cwd = Path.cwd()
    module = ModuleType("cli.main")
    module.main = Mock(side_effect=SystemExit(2))
    monkeypatch.setitem(sys.modules, "cli.main", module)
    monkeypatch.setattr(run, "project_root", tmp_path)
    with pytest.raises(SystemExit) as error:
        run.main(["--internal-role", "capability", "--invalid"])
    assert error.value.code == 2
    assert sys.argv is original_argv
    assert Path.cwd() == original_cwd


def test_no_tray_is_launcher_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Headless startup strips --no-tray without changing the classic default."""
    command = Mock(return_value=["worker", "classic"])
    child = Mock(return_value=42)
    tray = Mock(side_effect=AssertionError("Tray must not start"))
    monkeypatch.setattr(run, "entrypoint_command", command)
    monkeypatch.setattr(run, "_run_child_process", child)
    monkeypatch.setattr(run, "_start_tray_process", tray)
    assert run.main(["--no-tray", "--limit", "1"]) == 42
    assert command.call_args.args == ("classic", ["--limit", "1"])
    tray.assert_not_called()


@pytest.mark.parametrize("reason", [None, "pystray missing"])
def test_windows_tray_gate_is_lazy_and_does_not_require_gtk(
    monkeypatch: pytest.MonkeyPatch, reason: str | None,
) -> None:
    """Windows checks its native presenter rather than attempting any gi import."""
    module = ModuleType("core.windows_tray")
    module.windows_tray_unavailable_reason = Mock(return_value=reason)
    monkeypatch.setitem(sys.modules, "core.windows_tray", module)
    monkeypatch.setattr(run, "IS_LINUX", False)
    monkeypatch.setattr(run, "IS_MACOS", False)
    monkeypatch.setattr(run, "IS_WINDOWS", True)
    monkeypatch.setattr(
        run.importlib.util, "find_spec",
        Mock(side_effect=AssertionError("No GTK lookup")),
    )
    result = getattr(run, "_tray_skip_reason")()
    assert result is None if reason is None else reason in result


def test_gui_always_requests_dashboard_not_benchmark(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even a GUI launch without arguments must explicitly select the dashboard."""
    child = Mock(return_value=SimpleNamespace(returncode=0))
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(windows_gui, "python_executable", lambda: "console-worker.exe")
    monkeypatch.setattr(windows_gui, "USER_LOGS_DIR", tmp_path)
    monkeypatch.setattr(windows_gui.subprocess, "run", child)
    assert windows_gui.main([]) == 0
    assert child.call_args.args[0] == ["console-worker.exe", "--webapp"]
    assert child.call_args.kwargs["stdin"] == subprocess.DEVNULL
    assert child.call_args.kwargs["env"]["_LM_STUDIO_BENCH_WINDOWLESS"] == "1"


def test_gui_workers_do_not_allocate_a_console(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The GUI's nested dashboard worker also inherits windowless creation flags."""
    child = Mock(return_value=SimpleNamespace(returncode=0))
    monkeypatch.setattr(run, "IS_WINDOWS", True)
    monkeypatch.setenv("_LM_STUDIO_BENCH_WINDOWLESS", "1")
    monkeypatch.setattr(run, "child_creation_flags", lambda: 0x08000200)
    monkeypatch.setattr(run.subprocess, "run", child)
    assert getattr(run, "_run_child_process")(["worker"]) == 0
    assert child.call_args.kwargs["creationflags"] == 0x08000200


def test_console_workers_retain_normal_console_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Interactive CLI processes keep inherited stdio and Ctrl-C behavior."""
    child = Mock(return_value=SimpleNamespace(returncode=0))
    monkeypatch.delenv("_LM_STUDIO_BENCH_WINDOWLESS", raising=False)
    monkeypatch.setattr(run.subprocess, "run", child)
    assert getattr(run, "_run_child_process")(["worker"]) == 0
    assert "creationflags" not in child.call_args.kwargs
    assert "stdout" not in child.call_args.kwargs


def test_gui_uses_console_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A GUI executable must not recursively relaunch itself for background roles."""
    worker = tmp_path / runtime.WINDOWS_CONSOLE_EXECUTABLE
    worker.touch()
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(
        sys, "executable", str(tmp_path / runtime.WINDOWS_GUI_EXECUTABLE)
    )
    assert runtime.python_executable() == str(worker)
    assert runtime.entrypoint_command("webapp", []) == [
        str(worker), "--internal-role", "webapp"
    ]
    worker.unlink()
    with pytest.raises(FileNotFoundError, match="console worker"):
        runtime.python_executable()


def test_frozen_environment_drops_external_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A portable bundle cannot inherit a checkout or system Python import root."""
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setenv("PYTHONHOME", "external-python")
    monkeypatch.setenv("PYTHONPATH", "external-checkout")
    environment = runtime.child_environment()
    assert "PYTHONHOME" not in environment
    assert "PYTHONPATH" not in environment
    assert environment["PYTHONUTF8"] == "1"
    assert os.environ["PYTHONPATH"] == "external-checkout"


def test_release_requires_explicit_approval(tmp_path: Path) -> None:
    """Copying a valid EULA/notice is not by itself redistribution approval."""
    eula = _vendor_eula(tmp_path)
    with pytest.raises(ValueError, match="explicit owner"):
        builder.validate_release_policy("release", False, eula, builder.sha256(eula))


@pytest.mark.parametrize("missing", ["file", "digest", "mismatch", "format"])
def test_release_requires_the_particular_owner_eula(
    tmp_path: Path, missing: str,
) -> None:
    """Release approval must be bound to a present, usable agreement's hash."""
    eula = _vendor_eula(tmp_path)
    digest = builder.sha256(eula)
    if missing == "file":
        eula.unlink()
    elif missing == "digest":
        digest = ""
    elif missing == "mismatch":
        digest = "0" * 64
    else:
        eula = eula.rename(eula.with_suffix(".pdf"))
    with pytest.raises(ValueError, match="EULA"):
        builder.validate_release_policy("release", True, eula, digest)


def test_explicit_policy_records_owner_attestation(tmp_path: Path) -> None:
    """Policy validates supplied approval inputs, not the document's legal content."""
    eula = _vendor_eula(tmp_path)
    result = builder.validate_release_policy(
        "release", True, eula, builder.sha256(eula).upper()
    )
    assert result["owner_approved_vendor_redistribution"] is True
    assert result["vendor_eula_sha256"] == builder.sha256(eula)


def test_internal_policy_never_claims_clearance(tmp_path: Path) -> None:
    """Internal builds are distinct from and cannot masquerade as approved releases."""
    policy = builder.validate_release_policy("internal", False, None, None)
    assert policy["owner_approved_vendor_redistribution"] is False
    assert policy["vendor_eula_sha256"] is None
    with pytest.raises(ValueError, match="Internal mode"):
        builder.validate_release_policy("internal", True, _vendor_eula(tmp_path), None)


@pytest.mark.parametrize("value", ["", "latest", "v1.2", "1.2.3/evil", "v1.2.3;cmd"])
def test_invalid_versions_are_rejected(value: str) -> None:
    """Version-controlled names cannot contain paths, shell syntax or ambiguous tags."""
    with pytest.raises(ValueError, match="release version"):
        builder.normalize_version(value)


def test_artifact_names_follow_reference_conventions() -> None:
    """Version, architecture, setup.exe and platform checksum names are stable."""
    assert builder.artifact_names("v0.3.1", "release") == {
        "zip": "lm-studio-bench-0.3.1-windows-x86_64.zip",
        "installer": "lm-studio-bench-0.3.1-windows-x86_64-setup.exe",
        "checksums": "SHA256SUMS-windows.txt",
    }
    internal = builder.artifact_names("0.3.1", "internal")
    assert builder.INTERNAL_MARKER in internal["zip"]
    assert builder.INTERNAL_MARKER in internal["installer"]


def test_version_must_match_metadata_and_tag(tmp_path: Path) -> None:
    """Changing a tag or only one version source fails before any compilation."""
    (tmp_path / "VERSION").write_text("v0.3.1\n", encoding="utf-8")
    project = tmp_path / "pyproject.toml"
    project.write_text('[project]\nversion="0.3.1"\n', encoding="utf-8")
    assert builder.read_version(tmp_path, "v0.3.1") == "v0.3.1"
    with pytest.raises(ValueError, match="tag"):
        builder.read_version(tmp_path, "v0.3.2")
    project.write_text('[project]\nversion="0.2.0"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="versions disagree"):
        builder.read_version(tmp_path)


def test_native_probe_and_notices_are_mandatory(tmp_path: Path) -> None:
    """No fallback or notice-only package can omit its real native executable."""
    probe = tmp_path / "gpu_probe.exe"
    with pytest.raises(FileNotFoundError, match="executable"):
        builder.validate_native_probe(probe)
    probe.write_bytes(_pe_bytes())
    with pytest.raises(FileNotFoundError, match="notice"):
        builder.validate_native_probe(probe)
    licenses = tmp_path / "licenses"
    licenses.mkdir()
    for name in builder.VENDOR_LICENSES:
        (licenses / name).write_bytes(b"fixture notice")
    builder.validate_native_probe(probe)
    probe.write_bytes(b"not an executable")
    with pytest.raises(ValueError, match="Windows executable"):
        builder.validate_native_probe(probe)


def test_explicit_missing_inno_compiler_fails(tmp_path: Path) -> None:
    """An explicitly configured absent compiler is never silently substituted."""
    with pytest.raises(FileNotFoundError, match="ISCC.exe"):
        builder.find_inno_compiler(tmp_path / "ISCC.exe")


def test_complete_artifacts_verify(tmp_path: Path) -> None:
    """Both checksummed artifacts and embedded approval data must agree."""
    _, digest = _fake_artifacts(tmp_path)
    builder.verify_artifacts(tmp_path, "v0.3.1", "release", digest)


@pytest.mark.parametrize("kind", ["zip", "installer", "checksums"])
def test_missing_release_artifacts_fail_closed(tmp_path: Path, kind: str) -> None:
    """Release verification cannot silently omit the installer or portable ZIP."""
    names, digest = _fake_artifacts(tmp_path)
    (tmp_path / names[kind]).unlink()
    with pytest.raises(FileNotFoundError, match="artifact"):
        builder.verify_artifacts(tmp_path, "v0.3.1", "release", digest)


def test_changed_artifacts_fail_checksum_verification(tmp_path: Path) -> None:
    """Tampering or a stale installer invalidates the bundle's checksums."""
    names, digest = _fake_artifacts(tmp_path)
    with (tmp_path / names["installer"]).open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        builder.verify_artifacts(tmp_path, "v0.3.1", "release", digest)


def test_renamed_internal_bundle_is_not_publishable(tmp_path: Path) -> None:
    """Renaming and new checksums cannot turn internal output into a release."""
    _, digest = _fake_artifacts(tmp_path, manifest_mode="internal")
    with pytest.raises(ValueError, match="mode/version"):
        builder.verify_artifacts(tmp_path, "v0.3.1", "release", digest)


def test_release_verification_requires_external_approved_digest(tmp_path: Path) -> None:
    """Self-asserted approval inside an archive is insufficient for publication."""
    _fake_artifacts(tmp_path)
    with pytest.raises(ValueError, match="approval/digest"):
        builder.verify_artifacts(tmp_path, "v0.3.1", "release")


@pytest.mark.parametrize(
    "library", ["amdadlx64.dll", "ControlLib.dll", "igcl64.dll", "nvml.dll"],
)
def test_vendor_driver_dlls_cannot_be_distributed(tmp_path: Path, library: str) -> None:
    """Only helper object code/notices, never installed driver DLLs, may ship."""
    names, digest = _fake_artifacts(tmp_path)
    with zipfile.ZipFile(tmp_path / names["zip"], "a") as archive:
        archive.writestr(
            f"{builder.PRODUCT}/_internal/native/{library}", b"fake driver"
        )
    _write_checksums(tmp_path, names)
    with pytest.raises(ValueError, match="must not be bundled"):
        builder.verify_artifacts(tmp_path, "v0.3.1", "release", digest)


@pytest.mark.parametrize(
    "missing",
    [
        "VENDOR-EULA.txt", "LM-Studio-Bench.exe",
        "_internal/native/gpu_probe.exe", "_internal/code_runtime/python.exe",
        "_internal/native/licenses/IGCL-LICENSE.txt",
    ],
)
def test_incomplete_portable_content_is_rejected(
    tmp_path: Path, missing: str,
) -> None:
    """Correct outer checksums cannot disguise missing helper, GUI, Python or EULA."""
    names, digest = _fake_artifacts(tmp_path)
    path = tmp_path / names["zip"]
    with zipfile.ZipFile(path) as archive:
        content = {
            item: archive.read(item) for item in archive.namelist()
            if item != f"{builder.PRODUCT}/{missing}"
        }
    with zipfile.ZipFile(path, "w") as archive:
        for item, data in content.items():
            archive.writestr(item, data)
    _write_checksums(tmp_path, names)
    with pytest.raises(ValueError, match="missing|EULA"):
        builder.verify_artifacts(tmp_path, "v0.3.1", "release", digest)


def test_build_checks_policy_before_native_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A blocked release must fail before invoking any compiler or packaging tool."""
    native = Mock(side_effect=AssertionError("Must not reach native tools"))
    monkeypatch.setattr(builder, "read_version", Mock(return_value="v0.3.1"))
    monkeypatch.setattr(builder, "validate_native_probe", native)
    args = argparse.Namespace(
        mode="release", expected_tag=None, vendor_eula=None,
        owner_approved_vendor_redistribution=False, vendor_eula_sha256=None,
    )
    with pytest.raises(ValueError, match="Release blocked"):
        builder.build(args)
    native.assert_not_called()


def test_required_resource_inputs_are_present() -> None:
    """The spec's data inputs include capability samples and all role templates."""
    for relative in (*builder.RESOURCE_DIRS, *builder.RESOURCE_FILES):
        assert builder.ROOT.joinpath(*relative.split("/")).exists(), relative
    assert "tests/data" in builder.RESOURCE_DIRS
    assert "tests/prompts" in builder.RESOURCE_DIRS
    assert "web/templates" in builder.RESOURCE_DIRS
    assert "cli/report_template.html.template" in builder.RESOURCE_FILES


def test_windows_locks_do_not_include_gtk() -> None:
    """The Windows dependency closure is generated, not copied from the Linux lock."""
    lock = (builder.ROOT / "requirements-windows-lock.txt").read_text(encoding="utf-8")
    assert "uv pip compile" in lock
    assert "pystray==" in lock
    assert "pillow==" in lock
    assert "pygobject==" not in lock
    assert "pycairo==" not in lock


def test_reporting_selftest_uses_real_libraries_without_writing_files() -> None:
    """Default PDF/font, HTML/JS and scientific operations work without LM Studio."""
    from scripts.windows_selftest import reporting_checks

    checks = reporting_checks()
    assert all(checks.values())
    assert json.loads(json.dumps(checks)) == checks
