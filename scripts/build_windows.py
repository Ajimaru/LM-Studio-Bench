"""Build unsigned Windows x64 bundles; release redistribution is opt-in.

Internal builds reuse an already-built native probe for local validation only.
They are visibly marked and rejected by the release verifier. Release builds
require an explicit owner attestation and an owner-supplied vendor EULA whose
SHA256 matches the approved digest. This checks approval, not legal adequacy.
It neither accepts SDK terms nor compiles/downloads any vendor SDK.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import platform
import re
import shutil
import struct
import subprocess
import sys
import tomllib
from typing import Any
import uuid
import zipfile

ROOT = Path(__file__).resolve().parent.parent
PRODUCT = "LM-Studio-Bench"
INTERNAL_MARKER = "INTERNAL-NOT-FOR-DISTRIBUTION"
MANIFEST = "BUILD-METADATA.json"
CHECKSUMS = "SHA256SUMS-windows.txt"
VERSION_PATTERN = re.compile(
    r"v?(\d+\.\d+\.\d+(?:-(?:alpha|beta|rc|dev|test))?)"
)
VENDOR_LICENSES = (
    "ADLX SDK License Agreement.pdf", "IGCL-LICENSE.txt", "NOTICE.txt"
)
RESOURCE_DIRS = (
    "assets", "config", "prompts", "samples", "tests/data", "tests/prompts",
    "web/templates",
)
RESOURCE_FILES = (
    "VERSION", "AUTHORS", "LICENSE", "THIRD_PARTY_LICENSES.md",
    "requirements-windows-lock.txt", "requirements-windows-build-lock.txt",
    "cli/report_template.html.template",
)
LOGGER = logging.getLogger(__name__)


def sha256(path: Path) -> str:
    """Return a file's SHA256 digest without loading it all into memory."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def normalize_version(value: str) -> str:
    """Return a safe v-prefixed release version, rejecting other syntax."""
    match = VERSION_PATTERN.fullmatch(value.strip())
    if not match:
        raise ValueError(f"Unsupported release version: {value!r}")
    return f"v{match.group(1)}"


def read_version(root: Path = ROOT, expected_tag: str | None = None) -> str:
    """Check VERSION, project metadata and an optional release tag agree."""
    version = normalize_version((root / "VERSION").read_text(encoding="utf-8"))
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    if normalize_version(project["project"]["version"]) != version:
        raise ValueError("VERSION and pyproject.toml versions disagree")
    if expected_tag is not None and normalize_version(expected_tag) != version:
        raise ValueError("Release tag does not match VERSION")
    return version


def artifact_names(version: str, mode: str) -> dict[str, str]:
    """Return stable names for every required Windows release artifact."""
    if mode not in {"internal", "release"}:
        raise ValueError("Build mode must be internal or release")
    base = f"lm-studio-bench-{normalize_version(version)[1:]}-windows-x86_64"
    if mode == "internal":
        base += f"-{INTERNAL_MARKER}"
    return {"zip": f"{base}.zip", "installer": f"{base}-setup.exe",
            "checksums": CHECKSUMS}


def validate_release_policy(
    mode: str,
    approved: bool,
    eula: Path | None,
    approved_digest: str | None,
) -> dict[str, Any]:
    """Validate explicit owner approval, never infer it from local SDK consent.

    Args:
        mode: Internal local validation, or release distribution.
        approved: Owner's explicit redistribution attestation.
        eula: Owner-supplied plain text or RTF agreement for both artifacts.
        approved_digest: SHA256 of the particular agreement the owner approved.

    Returns:
        Auditable policy fields to embed in the bundle.

    Raises:
        ValueError: If release approval or its matching agreement is absent.
    """
    if mode not in {"internal", "release"}:
        raise ValueError("Unknown distribution mode")
    policy = {
        "mode": mode,
        "owner_approved_vendor_redistribution": False,
        "vendor_eula_sha256": None,
    }
    if mode == "internal":
        if approved or eula is not None or approved_digest:
            raise ValueError("Internal mode cannot claim redistribution approval")
        return policy
    if not approved:
        raise ValueError(
            "Release blocked: explicit owner redistribution approval is required; "
            "local SDK-license acceptance is not redistribution authorization"
        )
    if eula is None or not eula.is_file() or eula.stat().st_size == 0:
        raise ValueError("Release blocked: an owner-supplied vendor EULA is required")
    if eula.suffix.lower() not in {".txt", ".rtf"}:
        raise ValueError("The owner-supplied EULA must be an Inno-compatible TXT/RTF")
    if not approved_digest or not re.fullmatch(r"[a-fA-F0-9]{64}", approved_digest):
        raise ValueError("Release blocked: the approved EULA SHA256 is required")
    if sha256(eula) != approved_digest.lower():
        raise ValueError("Release blocked: vendor EULA differs from approved SHA256")
    policy.update(
        owner_approved_vendor_redistribution=True,
        vendor_eula_sha256=approved_digest.lower(),
    )
    return policy


def require_x64_pe(path: Path) -> None:
    """Reject missing, truncated or non-x64 executables before packaging."""
    if not path.is_file():
        raise FileNotFoundError(f"Required executable is missing: {path}")
    with path.open("rb") as handle:
        header = handle.read(64)
        if len(header) != 64 or header[:2] != b"MZ":
            raise ValueError(f"Not a Windows executable: {path}")
        handle.seek(struct.unpack_from("<I", header, 60)[0])
        signature = handle.read(6)
    if signature != b"PE\x00\x00\x64\x86":
        raise ValueError(f"Not a Windows x64 executable: {path}")


def validate_native_probe(probe: Path) -> None:
    """Require the real helper and all vendor notices; no fallback is allowed."""
    require_x64_pe(probe)
    for name in VENDOR_LICENSES:
        license_file = probe.parent / "licenses" / name
        if not license_file.is_file() or license_file.stat().st_size == 0:
            raise FileNotFoundError(
                f"Required vendor notice is missing: {license_file}"
            )


def find_inno_compiler(explicit: Path | None = None) -> Path:
    """Find an existing Inno compiler or fail, without installing anything."""
    if explicit is not None:
        if not explicit.is_file():
            raise FileNotFoundError(f"The specified ISCC.exe is missing: {explicit}")
        return explicit.resolve()
    candidates: list[Path] = []
    for variable in ("LOCALAPPDATA", "ProgramFiles(x86)", "ProgramFiles"):
        value = os.environ.get(variable)
        if value:
            base = (
                Path(value) / "Programs"
                if variable == "LOCALAPPDATA" else Path(value)
            )
            candidates.extend(sorted(base.glob("Inno Setup */ISCC.exe"), reverse=True))
    if shutil.which("ISCC.exe"):
        candidates.append(Path(str(shutil.which("ISCC.exe"))))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError("ISCC.exe is required; the installer may not be skipped")


def copy_code_runtime(destination: Path) -> None:
    """Copy this interpreter and its stdlib, not a venv or an app executable."""
    base = Path(sys.base_prefix)
    executable = base / "python.exe"
    require_x64_pe(executable)
    license_file = base / "LICENSE.txt"
    if not license_file.is_file():
        raise FileNotFoundError("The installed Python's LICENSE.txt is missing")
    destination.mkdir()
    shutil.copy2(executable, destination / "python.exe")
    shutil.copy2(license_file, destination / "LICENSE.txt")
    for library in base.glob("*.dll"):
        shutil.copy2(library, destination / library.name)
    shutil.copytree(base / "DLLs", destination / "DLLs")
    tag = f"python{sys.version_info.major}{sys.version_info.minor}"
    ignored = {
        "__pycache__", "site-packages", "test", "tests",
        "idlelib", "turtledemo", "ensurepip",
    }
    # zipimport never creates __pycache__ in the installation directory, even
    # with -I (which intentionally ignores PYTHONDONTWRITEBYTECODE).
    with zipfile.ZipFile(destination / f"{tag}.zip", "w") as archive:
        for source in sorted((base / "Lib").rglob("*")):
            relative = source.relative_to(base / "Lib")
            if (
                source.is_file() and not ignored.intersection(relative.parts)
                and source.suffix != ".pyc"
            ):
                archive.write(source, relative)
    # tkinter is part of the stdlib; preserve its Tcl runtime too.
    if (base / "tcl").is_dir():
        shutil.copytree(base / "tcl", destination / "tcl")
    (destination / f"{tag}._pth").write_text(
        f"{tag}.zip\nDLLs\n.\n", encoding="utf-8"
    )
    result = subprocess.run(  # nosec B603 - copied real Python, fixed probe
        [
            str(destination / "python.exe"), "-I", "-S", "-c",
            "import json,sys,ssl,sqlite3; "
            "print(json.dumps(list(sys.version_info[:3])))",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=20,
        check=True,
    )
    if json.loads(result.stdout) != list(sys.version_info[:3]):
        raise ValueError("The bundled code runtime does not match the build Python")


def create_icon(destination: Path) -> None:
    """Rasterize the project's simple SVG geometry with the existing Pillow."""
    from PIL import Image, ImageDraw

    image = Image.new("RGBA", (256, 256))
    draw = ImageDraw.Draw(image)
    scale = 256 / 48

    def box(values: tuple[int, int, int, int]) -> tuple[float, ...]:
        return tuple(value * scale for value in values)

    draw.rounded_rectangle(box((0, 0, 48, 48)), radius=8 * scale, fill="#5294e2")
    for rectangle in ((10, 12, 14, 36), (10, 32, 20, 36)):
        draw.rectangle(box(rectangle), fill="white")
    points = (
        (22, 12), (22, 36), (26, 36), (26, 16), (30, 24), (34, 16),
        (34, 36), (38, 36), (38, 12), (34, 12), (30, 22), (26, 12),
    )
    draw.polygon([(x * scale, y * scale) for x, y in points], fill="white")
    for rectangle in ((10, 40, 16, 43), (18, 38, 24, 43),
                      (26, 36, 32, 43), (34, 40, 40, 43)):
        draw.rectangle(box(rectangle), fill=(255, 255, 255, 204))
    image.save(destination, sizes=[(n, n) for n in (16, 24, 32, 48, 64, 256)])
    image.close()


def write_version_info(destination: Path, version: str) -> None:
    """Write Windows executable version metadata for PyInstaller."""
    numeric = tuple(int(part) for part in version[1:].split("-")[0].split("."))
    numeric += (0,)
    destination.write_text(
        "VSVersionInfo(\n"
        f" ffi=FixedFileInfo(filevers={numeric!r}, prodvers={numeric!r},\n"
        "  mask=0x3f, flags=0x0, OS=0x40004, fileType=0x1, subtype=0,\n"
        "  date=(0, 0)),\n"
        " kids=[StringFileInfo([StringTable('040904B0', [\n"
        "  StringStruct('ProductName', 'LM Studio Bench'),\n"
        f"  StringStruct('ProductVersion', '{version}'),\n"
        f"  StringStruct('FileVersion', '{version}'),\n"
        "  StringStruct('FileDescription', 'LM Studio Bench'),\n"
        "  StringStruct('LegalCopyright', 'See AUTHORS and LICENSE')\n"
        " ])]), VarFileInfo([VarStruct('Translation', [1033, 1200])])])\n",
        encoding="utf-8",
    )


def collect_distribution_data(stage: Path) -> list[tuple[str, str]]:
    """Preserve wheel metadata, licenses and a machine-readable inventory."""
    data: list[tuple[str, str]] = []
    inventory = []
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata["Name"]
        files = distribution.files or []
        metadata_file = next(
            (file for file in files if str(file).endswith(".dist-info/METADATA")),
            None,
        )
        if metadata_file is None:
            raise ValueError(f"Wheel license/metadata directory missing: {name}")
        directory = Path(distribution.locate_file(metadata_file)).parent
        data.append((str(directory), directory.name))
        inventory.append({
            "name": name,
            "version": distribution.version,
            "license": distribution.metadata.get(
                "License-Expression", distribution.metadata.get("License", "")
            ),
            "metadata_directory": directory.name,
        })
    inventory_path = stage / "THIRD-PARTY-PYTHON.json"
    inventory_path.write_text(
        json.dumps(sorted(inventory, key=lambda item: item["name"].lower()), indent=2),
        encoding="utf-8",
    )
    data.append((str(inventory_path), "licenses"))
    return data


def prepare_inputs(
    stage: Path, probe: Path, version: str, policy: dict[str, Any], eula: Path | None,
) -> dict[str, Any]:
    """Stage only required resources, the native helper and isolated Python."""
    data = []
    for relative in (*RESOURCE_DIRS, *RESOURCE_FILES):
        source = ROOT.joinpath(*relative.split("/"))
        if not source.exists():
            raise FileNotFoundError(f"Required bundle resource is missing: {source}")
        target = relative if source.is_dir() else str(Path(relative).parent)
        data.append((str(source), target))
    data.append((str(probe.parent / "licenses"), "native/licenses"))
    copy_code_runtime(stage / "code_runtime")
    data.append((str(stage / "code_runtime"), "code_runtime"))
    create_icon(stage / "lmstudio-bench.ico")
    data.append((str(stage / "lmstudio-bench.ico"), "assets/icons"))
    data += collect_distribution_data(stage)
    write_version_info(stage / "version-info.txt", version)
    manifest = {
        "schema_version": 1,
        "product": PRODUCT,
        "version": version,
        "architecture": "windows-x64",
        "python": platform.python_version(),
        "signed": False,
        "native_probe_sha256": sha256(probe),
        "runtime_lock_sha256": sha256(ROOT / "requirements-windows-lock.txt"),
        "build_lock_sha256": sha256(ROOT / "requirements-windows-build-lock.txt"),
        **policy,
    }
    manifest_path = stage / MANIFEST
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if eula is not None:
        license_path = stage / f"VENDOR-EULA{eula.suffix.lower()}"
        shutil.copy2(eula, license_path)
    else:
        license_path = stage / f"{INTERNAL_MARKER}.txt"
        license_path.write_text(
            "INTERNAL LOCAL VALIDATION ONLY - NOT FOR DISTRIBUTION.\n\n"
            "The native helper contains AMD ADLX and Intel IGCL SDK code.\n"
            "Local build consent and copied notices do not authorize distribution.\n"
            "This is NOT a vendor EULA or a claim of legal clearance.\n",
            encoding="utf-8",
        )
    data.extend([(str(manifest_path), "."), (str(license_path), "licenses")])
    inputs = {"datas": data, "native_probe": str(probe)}
    (stage / "freeze-inputs.json").write_text(
        json.dumps(inputs, indent=2), encoding="utf-8"
    )
    return {"manifest": manifest, "license_path": license_path}


def write_bundle_notice(bundle: Path, mode: str) -> None:
    """Explain the GUI/CLI distinction and unsigned status inside the ZIP."""
    (bundle / "README-WINDOWS.txt").write_text(
        f"LM Studio Bench - Windows 10/11 x64 - {mode.upper()} BUILD\n\n"
        "UNSIGNED: Windows/SmartScreen can warn about this application.\n"
        "LM-Studio-Bench.exe: opens the dashboard and optional tray; never "
        "automatically benchmarks.\n"
        "lmstudio-bench.exe: console CLI; no arguments retains the classic "
        "benchmark default. Use --help for options.\n"
        "Headless dashboard: lmstudio-bench.exe --webapp --no-browser --no-tray\n"
        "LM Studio and its lms CLI are separate prerequisites for model work.\n"
        "Python, GTK and a compiler are NOT required on the target machine.\n\n"
        "Configuration: %APPDATA%\\LM-Studio-Bench\n"
        "Results/logs/cache: %LOCALAPPDATA%\\LM-Studio-Bench\n"
        "No autostart, global PATH or ExecutionPolicy changes are made.\n"
        "Uninstall leaves user configuration, results and logs intact.\n\n"
        "Repository MIT LICENSE does not relicense vendor SDKs or dependencies.\n"
        "Read the supplied vendor agreement/notices and third-party licenses.\n"
        "Owner approval is an attestation, not automated legal clearance.\n",
        encoding="utf-8",
    )


def verify_artifacts(
    output: Path,
    version: str,
    mode: str,
    approved_eula_digest: str | None = None,
) -> None:
    """Fail closed on missing artifacts, changed checksums or internal releases."""
    names = artifact_names(version, mode)
    for name in names.values():
        path = output / name
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Required release artifact is missing: {path}")
    expected = {names["zip"], names["installer"]}
    checksums = {}
    for line in (output / CHECKSUMS).read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([a-f0-9]{64})  ([^/\\]+)", line)
        if not match or match[2] in checksums:
            raise ValueError("Malformed or duplicate Windows checksum entry")
        checksums[match[2]] = match[1]
    if set(checksums) != expected:
        raise ValueError("Windows checksums must include exactly ZIP and installer")
    for name, digest in checksums.items():
        if sha256(output / name) != digest:
            raise ValueError(f"Artifact checksum mismatch: {name}")
    with (output / names["installer"]).open("rb") as handle:
        if handle.read(2) != b"MZ":
            raise ValueError("The installer is not a Windows executable")
    prefix = f"{PRODUCT}/"
    with zipfile.ZipFile(output / names["zip"]) as archive:
        entries = set(archive.namelist())
        for entry in entries:
            filename = entry.rsplit("/", 1)[-1].lower()
            if filename.endswith(".dll") and filename.startswith(
                ("amdadlx", "controllib", "igcl", "nvapi", "nvml", "nvcuda")
            ):
                raise ValueError(f"Driver/vendor SDK DLL must not be bundled: {entry}")
        manifest = json.loads(archive.read(prefix + MANIFEST))
        if manifest.get("mode") != mode or manifest.get("version") != version:
            raise ValueError("Bundle mode/version does not match the requested release")
        required = [
            "LM-Studio-Bench.exe", "lmstudio-bench.exe", "LICENSE",
            "_internal/native/gpu_probe.exe",
            "_internal/code_runtime/python.exe",
            "_internal/code_runtime/LICENSE.txt",
            "_internal/web/templates/dashboard.html.jinja",
            "_internal/cli/report_template.html.template",
        ]
        required += [f"_internal/native/licenses/{name}" for name in VENDOR_LICENSES]
        for name in required:
            if (
                prefix + name not in entries
                or archive.getinfo(prefix + name).file_size == 0
            ):
                raise ValueError(f"Required packaged component missing: {name}")
        native = archive.read(prefix + "_internal/native/gpu_probe.exe")
        if hashlib.sha256(native).hexdigest() != manifest["native_probe_sha256"]:
            raise ValueError("Packaged native helper differs from the validated input")
        if mode == "release":
            if any(INTERNAL_MARKER in item for item in entries):
                raise ValueError("Internal packages must never be published")
            if manifest.get("owner_approved_vendor_redistribution") is not True:
                raise ValueError("Release has no explicit owner approval")
            digest = manifest.get("vendor_eula_sha256")
            if not approved_eula_digest or digest != approved_eula_digest.lower():
                raise ValueError("Release EULA approval/digest is missing or different")
            eulas = [
                entry for entry in entries
                if entry in {prefix + "VENDOR-EULA.txt", prefix + "VENDOR-EULA.rtf"}
            ]
            if len(eulas) != 1:
                raise ValueError("Release ZIP must contain the owner's vendor EULA")
            if hashlib.sha256(archive.read(eulas[0])).hexdigest() != digest:
                raise ValueError("Packaged EULA differs from the approved agreement")


def build(args: argparse.Namespace) -> Path:
    """Build both EXEs, ZIP, installer and checksums without running the installer."""
    version = read_version(expected_tag=args.expected_tag)
    eula = args.vendor_eula.resolve() if args.vendor_eula is not None else None
    policy = validate_release_policy(
        args.mode, args.owner_approved_vendor_redistribution,
        eula, args.vendor_eula_sha256,
    )
    if sys.platform != "win32" or platform.machine().lower() not in {"amd64", "x86_64"}:
        raise ValueError("The Windows bundle must be built on Windows x64")
    if struct.calcsize("P") != 8 or sys.version_info[:2] != (3, 13):
        raise ValueError(
            "The Windows lock/build uses 64-bit CPython 3.13; "
            "source execution still supports Python >=3.11"
        )
    default_probe = ROOT / "build" / "windows_gpu" / "bin" / "gpu_probe.exe"
    probe = (args.native_probe or default_probe).resolve()
    validate_native_probe(probe)
    compiler = find_inno_compiler(args.iscc)
    output = (args.output_dir or ROOT / "dist" / "windows" / args.mode).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output must be empty; refusing to overwrite files: {output}")
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    stage = ROOT / "build" / "windows" / f"{stamp}-{uuid.uuid4().hex[:8]}"
    stage.mkdir(parents=True)
    scratch = stage / "scratch"
    scratch.mkdir()
    prepared = prepare_inputs(stage, probe, version, policy, eula)
    spec = stage / "windows.spec"
    shutil.copy2(ROOT / "scripts" / "windows_spec.py", spec)
    environment = os.environ.copy()
    environment.update(
        LMSB_BUILD_ROOT=str(ROOT), LMSB_BUILD_STAGE=str(stage),
        TEMP=str(scratch), TMP=str(scratch),
        PYINSTALLER_CONFIG_DIR=str(stage / "pyinstaller-cache"),
    )
    subprocess.run(  # nosec B603 - fixed build tool, controlled paths
        [
            sys.executable, "-m", "PyInstaller", "--noconfirm",
            "--distpath", str(output), "--workpath", str(stage / "pyinstaller"),
            str(spec),
        ],
        cwd=ROOT,
        env=environment,
        timeout=1800,
        check=True,
    )
    bundle = output / PRODUCT
    for name in ("LM-Studio-Bench.exe", "lmstudio-bench.exe"):
        require_x64_pe(bundle / name)
    for name in ("LICENSE", "AUTHORS", "THIRD_PARTY_LICENSES.md", "VERSION"):
        shutil.copy2(ROOT / name, bundle / name)
    shutil.copy2(stage / MANIFEST, bundle / MANIFEST)
    license_path = prepared["license_path"]
    shutil.copy2(license_path, bundle / license_path.name)
    write_bundle_notice(bundle, args.mode)
    names = artifact_names(version, args.mode)
    numeric = version[1:].split("-")[0] + ".0"
    subprocess.run(  # nosec B603 - compile only; never execute the installer
        [
            str(compiler), "/Qp",
            f"/DBundleDir={bundle}", f"/DVendorEula={license_path}",
            f"/DAppVersion={version}", f"/DNumericVersion={numeric}",
            f"/DSetupName={Path(names['installer']).stem}",
            f"/DOutputPath={output}",
            str(ROOT / "scripts" / "windows_installer.iss"),
        ],
        cwd=ROOT,
        env=environment,
        timeout=1200,
        check=True,
    )
    archive_path = output / names["zip"]
    with zipfile.ZipFile(
        archive_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for file in sorted(bundle.rglob("*")):
            if file.is_file():
                archive.write(file, file.relative_to(output))
    (output / CHECKSUMS).write_text(
        "".join(
            f"{sha256(output / names[kind])}  {names[kind]}\n"
            for kind in ("zip", "installer")
        ),
        encoding="utf-8",
    )
    verify_artifacts(output, version, args.mode, args.vendor_eula_sha256)
    LOGGER.info("Validated %s Windows artifacts in %s", args.mode, output)
    return output


def main(argv: list[str] | None = None) -> int:
    """Parse build/verification arguments and return a failing exit code on error."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("internal", "release"))
    parser.add_argument("--native-probe", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--iscc", type=Path)
    parser.add_argument("--expected-tag")
    parser.add_argument("--owner-approved-vendor-redistribution", action="store_true")
    parser.add_argument("--vendor-eula", type=Path)
    parser.add_argument("--vendor-eula-sha256")
    parser.add_argument("--check-policy", action="store_true")
    parser.add_argument("--verify-artifacts", type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        version = read_version(expected_tag=args.expected_tag)
        if args.verify_artifacts is not None:
            verify_artifacts(
                args.verify_artifacts, version, args.mode, args.vendor_eula_sha256
            )
        elif args.check_policy:
            validate_release_policy(
                args.mode, args.owner_approved_vendor_redistribution,
                args.vendor_eula, args.vendor_eula_sha256,
            )
            LOGGER.info(
                "Explicit policy inputs validated; legal adequacy is not checked"
            )
        else:
            build(args)
        return 0
    except (
        OSError, ValueError, subprocess.SubprocessError, zipfile.BadZipFile
    ) as error:
        LOGGER.error("%s", error)
        return 1


if __name__ == "__main__":
    sys.exit(main())
