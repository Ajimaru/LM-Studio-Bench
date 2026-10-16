# Native Windows support

The Windows implementation targets **Windows 10/11 x64**. Source installations
require **Python 3.11 or newer**. ARM64 is not a verified native target.

Local verification covers Windows 11, NVIDIA Quadro T1000 and Intel UHD
Graphics. The AMD ADLX implementation is compiled and automatically tested;
real AMD validation is performed on a separate system. Windows 10 and other
GPU/driver combinations need their own hardware checks.

## Installation from source

Install LM Studio separately and make its `lms` CLI available. Use a working
x64 Python installation, not the Microsoft Store placeholder executable.

From the repository directory:

```powershell
.\setup.ps1
```

The setup script prepares the local virtual environment and reports Python,
LM Studio and native telemetry readiness separately. It does not download
models, start a benchmark, install drivers, change the global PATH or change
machine-wide PowerShell execution policy.

Manual fallback, without activating a virtual environment:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe .\run.py --webapp
```

An already configured `uv` environment is also suitable. Always invoke the
interpreter inside `.venv` when a normal `python` command opens the Store.
If a downloaded setup script is blocked, verify its source before unblocking
that particular file; do not disable execution restrictions system-wide.

The dashboard prints its selected URL. An offline LM Studio server is shown
as offline rather than preventing the dashboard from opening.

## Native GPU component

NVIDIA can use `nvidia-smi.exe` from PATH, Windows System32 or
`Program Files\NVIDIA Corporation\NVSMI`. The native component adds
adapter-wide Windows memory counters and the AMD/Intel sensor APIs.

For source development, install Visual Studio 2022 Build Tools with:

- MSVC x64/x86 C++ build tools;
- a Windows 10/11 SDK;
- C++ CMake tools for Windows.

Initialize each ordinary PowerShell terminal before building:

```powershell
$vs = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\BuildTools"
& "$vs\Common7\Tools\Launch-VsDevShell.ps1" `
  -Arch amd64 -HostArch amd64 -SkipAutomaticLocation
Get-Command cl, cmake
```

Installing Build Tools alone does not add this environment to an already
open VS Code terminal. A Developer PowerShell shortcut is an alternative.

### Vendor SDK consent

The native build downloads hash-pinned ADLX and IGCL SDK archives.
Review their terms before explicitly enabling the local build:

```powershell
cmake -S native\gpu_probe -B build\windows_gpu `
  -G "Visual Studio 17 2022" -A x64 `
  -DGPU_PROBE_ACCEPT_VENDOR_LICENSES=ON
cmake --build build\windows_gpu --config Release
ctest --test-dir build\windows_gpu -C Release --output-on-failure
```

Without that explicit option, the vendor SDK build stops. This option covers
local build consent only: it is **not redistribution authorization**.
The native component is not relicensed under the application's MIT license.
See [third-party licensing](../THIRD_PARTY_LICENSES.md).

The development executable is:

```powershell
.\build\windows_gpu\bin\gpu_probe.exe --once
```

It reports JSON containing adapter LUID/PCI identity, memory capacities,
readings, availability, sources and diagnostics. Continuous mode is available
with `--watch --interval-ms 1000`; stop it with Ctrl+C or its stdin stop command.
The Python application drains both output streams and validates the protocol.

## Telemetry and memory semantics

| Provider | Measurements |
| --- | --- |
| NVIDIA SMI | GPU temperature, power, dedicated memory usage and capacity |
| AMD ADLX | Supported AMD temperature and power sensors |
| Intel IGCL | Supported Intel temperature and energy-delta power sensors |
| Windows WDDM/PDH | Adapter-wide dedicated/shared memory usage |
| DXGI/D3D12 | Adapter identity, capacity and verified UMA/integrated status |

Adapters are associated using LUID and PCI identity, not matching names or
assuming that every vendor tool uses the same index. Windows can expose
multiple logical adapter LUIDs for the same friendly name; those records are
not silently merged.

The existing `gtt_gb_*` result fields carry **shared GPU memory usage** on
Windows. A shared-memory budget is borrowed from system RAM, not extra
physical memory. Do not add that budget to the system RAM total.
NVIDIA offload prediction still uses dedicated memory. AMD/Intel shared-memory
headroom is bounded by available system RAM and respects the shared-memory
option.

Missing measurements remain missing. A real zero is preserved as zero.
Availability distinguishes `ok`, `unsupported`, `unavailable` and `error`;
loader/protocol errors are not disguised as unsupported hardware.
Per-adapter result metadata includes identities, sensor status and sources.
All measurements remain compatible with the existing SQLite and export paths.

`--max-temp` and `--max-power` are warning thresholds, not an automatic
hardware shutdown mechanism. If the corresponding sensor is unavailable,
the application reports that the threshold cannot be checked.

### Intel runtime diagnostics

On the locally tested Intel UHD driver, the documented System32
`ControlLib.dll` load returns Windows error 126. Shared/dedicated memory
readings work, but Intel temperature and power remain explicitly unavailable.
This result does not establish whether other DriverStore copies exist.

Use the GPU vendor's supported driver package and its support matrix.
Do not download standalone DLLs from arbitrary sites or use undocumented
runtime-path overrides. No additional kernel driver is installed by this tool.

## Benchmark controls and code tests

Windows pause/resume controls the owned process tree, including Python's
virtual-environment redirector and its real interpreter child. Stop first
requests model cleanup, then uses a bounded termination fallback when needed.
Other Python and LM Studio processes are not killed by executable name.

Generated-code tests use a separate real Python interpreter with an isolated
working directory, timeout and Windows Job Object CPU/memory limits. This
contains resource accidents; it is **not a security sandbox** for hostile code.
POSIX file-size limits have no identical Windows implementation.

## User data

| Data | Location |
| --- | --- |
| Configuration and presets | `%APPDATA%\LM-Studio-Bench\` |
| Results and databases | `%LOCALAPPDATA%\LM-Studio-Bench\results\` |
| Logs | `%LOCALAPPDATA%\LM-Studio-Bench\logs\` |
| Cache | `%LOCALAPPDATA%\LM-Studio-Bench\cache\` |

Latest-log references are regular `*_latest.txt` files containing a
timestamped log filename. Symlink privileges and Developer Mode are not
required. See [User Data & Paths](USER_DATA.md) for manual migration from
older XDG-style Windows directories.

## Portable packages and installer

The Windows distribution uses a self-contained PyInstaller one-folder bundle,
a portable ZIP and an Inno Setup installer. Keep the entire extracted bundle,
not just its launcher executable. End users do not need Python or CMake.
LM Studio and supported GPU drivers remain external prerequisites.

The initial artifacts are unsigned. Verify SHA256 checksums and the source of
the download. Checksums do not replace code signing, and disabling Defender
or SmartScreen is not an installation step.

Vendor SDK binary redistribution needs separate owner approval and an
appropriate vendor EULA. Internal test artifacts must not be published as
releases. Release packaging is required to fail closed without those approvals
and the complete native component/license files. Installer execution,
upgrade/uninstall and clean-machine checks are separate from compilation.

## External AMD verification

On the AMD test system:

1. Install a supported official AMD driver and the approved application/native
   component. No extra monitoring kernel driver is needed.
2. Run `gpu_probe.exe --once` and retain its JSON and stderr diagnostics.
3. Check that the AMD LUID, PCI address and name identify the intended adapter.
4. Check temperature/power support flags and dedicated/shared memory units.
   Unsupported sensors must remain missing, not zero.
5. Run a representative benchmark with profiling:

   ```powershell
   .\.venv\Scripts\python.exe .\run.py `
     --enable-profiling --limit 1 --runs 3 --context 2048
   ```

6. Compare the readings with the corresponding vendor/Windows source and
   inspect SQLite, JSON/CSV/HTML/PDF output.
7. Exercise pause, resume, stop and quit; check that the model is unloaded and
   owned helper processes exit. Record unavailable sensors and driver versions.

On a mixed-vendor machine, verify LM Studio's actual GPU selection rather
than assuming its GPU index matches another tool's index.
