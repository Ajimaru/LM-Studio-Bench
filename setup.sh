#!/usr/bin/env bash

set -u
set -o pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="${LOG_DIR}/setup_${TIMESTAMP}.log"

mkdir -p "${LOG_DIR}"
touch "${LOG_FILE}"
ln -sf "$(basename "${LOG_FILE}")" "${LOG_DIR}/setup_latest.log"

LOG_FD=3
exec {LOG_FD}>> "${LOG_FILE}"

if [[ -t 1 ]]; then
    C_RESET="\033[0m"
    C_INFO="\033[1;34m"
    C_OK="\033[1;32m"
    C_WARN="\033[1;33m"
    C_ERR="\033[1;31m"
    C_HEAD="\033[1;36m"
else
    C_RESET=""
    C_INFO=""
    C_OK=""
    C_WARN=""
    C_ERR=""
    C_HEAD=""
fi

PKG_MANAGER=""
PKG_INSTALL_CMD=""
PKG_UPDATE_CMD=""
PKG_UPDATED="0"
INTERACTIVE="1"
DRY_RUN="0"

log() {
    local level="$1"
    local message="$2"
    local color="${C_INFO}"
    local timestamp
    timestamp="$(date +"%Y-%m-%d %H:%M:%S")"

    case "${level}" in
        INFO)
            color="${C_INFO}"
            ;;
        OK)
            color="${C_OK}"
            ;;
        WARN)
            color="${C_WARN}"
            ;;
        ERROR)
            color="${C_ERR}"
            ;;
    esac

    printf "%b[%s] [%s] %s%b\n" \
        "${color}" "${timestamp}" "${level}" \
        "${message}" "${C_RESET}"
    printf "[%s] [%s] %s\n" \
        "${timestamp}" "${level}" \
        "${message}" >&${LOG_FD}
}

section() {
    local title="$1"
    printf "\n%b========== %s ==========%b\n" "${C_HEAD}" "${title}" "${C_RESET}"
    printf "\n========== %s ==========\n" "${title}" >&${LOG_FD}
}

sanitize_path() {
    local path="$1"
    if [[ -n "${HOME:-}" && "${path}" == "${HOME}"* ]]; then
        echo "${path/#${HOME}/\~}"
    else
        echo "${path}"
    fi
}

ask_yes_no() {
    local prompt="$1"
    local default_ans="${2:-no}"
    local answer=""

    if [[ "${INTERACTIVE}" == "0" ]]; then
        if [[ "${default_ans}" == "yes" ]]; then
            log "INFO" "${prompt} [y/N]: ja (auto, --yes Mode)"
            return 0
        else
            log "INFO" "${prompt} [y/N]: nein (auto, --yes Mode)"
            return 1
        fi
    fi

    while true; do
        read -r -p "${prompt} [y/N]: " answer
        case "${answer}" in
            [yY]|[yY][eE][sS])
                return 0
                ;;
            ""|[nN]|[nN][oO])
                return 1
                ;;
            *)
                log "WARN" "Please answer with y(es) or n(o)."
                ;;
        esac
    done
}

print_help() {
    cat <<'HELP'
Usage: setup.sh [OPTIONS]

Setup script for LM-Studio-Bench project dependencies.
Checks and optionally installs system dependencies, Python packages,
GPU drivers/tools, and creates a virtual environment.

OPTIONS:
  --help              Show this help message and exit
  --dry-run           Preview mode: show what would be done, no changes
  --yes               Non-interactive mode: automatically answer 'no' to
                      optional prompts (safe for CI/automation)
  --interactive       Force interactive mode (default)

EXAMPLES:
  ./setup.sh                    # Interactive mode (asks for each decision)
  ./setup.sh --dry-run          # Preview: show what would be done
  ./setup.sh --yes              # Automation mode (safe defaults)
  ./setup.sh --dry-run --yes    # Preview with automation defaults
  ./setup.sh --help             # Show this message

OUTPUT:
  Log file: logs/setup_YYYYMMDD_HHMMSS.log

REQUIREMENTS:
  - Linux system (primary support)
  - Root access or sudo for package installation
  - Supported package managers: apt, dnf, pacman, zypper, apk

NOTE:
  LM Studio / llmster must be installed separately if not found.
  Check https://lmstudio.ai/ for download and documentation.
HELP
}

ensure_linux() {
    if [[ "$(uname -s)" != "Linux" ]]; then
        log "ERROR" "This setup script currently only supports Linux."
        exit 1
    fi
    log "OK" "Linux detected: $(uname -sr)"
}

detect_pkg_manager() {
    if command -v apt-get >/dev/null 2>&1; then
        PKG_MANAGER="apt"
        PKG_INSTALL_CMD="apt-get install -y"
        PKG_UPDATE_CMD="apt-get update"
    elif command -v dnf >/dev/null 2>&1; then
        PKG_MANAGER="dnf"
        PKG_INSTALL_CMD="dnf install -y"
        PKG_UPDATE_CMD="dnf makecache"
    elif command -v pacman >/dev/null 2>&1; then
        PKG_MANAGER="pacman"
        PKG_INSTALL_CMD="pacman -S --noconfirm --needed"
        PKG_UPDATE_CMD="pacman -Sy"
    elif command -v zypper >/dev/null 2>&1; then
        PKG_MANAGER="zypper"
        PKG_INSTALL_CMD="zypper --non-interactive install"
        PKG_UPDATE_CMD="zypper --non-interactive refresh"
    elif command -v apk >/dev/null 2>&1; then
        PKG_MANAGER="apk"
        PKG_INSTALL_CMD="apk add"
        PKG_UPDATE_CMD="apk update"
    fi

    if [[ -z "${PKG_MANAGER}" ]]; then
        log "WARN" "No supported package manager detected (apt/dnf/pacman/zypper/apk)."
        return 1
    fi

    log "OK" "Package manager detected: ${PKG_MANAGER}"
    return 0
}

run_pkg_update_once() {
    if [[ "${PKG_UPDATED}" == "1" ]]; then
        return 0
    fi

    if [[ "${PKG_MANAGER}" == "apt" || "${PKG_MANAGER}" == "pacman" || \
          "${PKG_MANAGER}" == "zypper" || "${PKG_MANAGER}" == "apk" ]]; then
        if ! run_with_privileges "${PKG_UPDATE_CMD}"; then
            log "WARN" "Could not update package index. Proceeding anyway."
        fi
    fi

    PKG_UPDATED="1"
    return 0
}

run_with_privileges() {
    local cmd="$1"

    if [[ "${DRY_RUN}" == "1" ]]; then
        log "INFO" "[DRY-RUN] Would execute: ${cmd}"
        return 0
    fi

    if [[ "${EUID}" -eq 0 ]]; then
        bash -lc "${cmd}"
        return $?
    fi

    if command -v sudo >/dev/null 2>&1; then
        sudo bash -lc "${cmd}"
        return $?
    fi

    log "ERROR" "Root privileges required, but sudo is not available."
    return 1
}

pkg_name_for_key() {
    local key="$1"

    case "${PKG_MANAGER}" in
        apt)
            case "${key}" in
                python3) echo "python3" ;;
                pip3) echo "python3-pip" ;;
                venv) echo "python3-venv" ;;
                git) echo "git" ;;
                curl) echo "curl" ;;
                pkgconf) echo "pkg-config" ;;
                python_dev) echo "python3-dev" ;;
                gobj_dev) echo "libgirepository1.0-dev" ;;
                cairo_dev) echo "libcairo2-dev" ;;
                pciutils) echo "pciutils" ;;
                lm_sensors) echo "lm-sensors" ;;
                intel_gpu_tools) echo "intel-gpu-tools" ;;
                rocm-smi) echo "rocm-smi" ;;
                rocminfo) echo "rocminfo" ;;
                *) echo "" ;;
            esac
            ;;
        dnf)
            case "${key}" in
                python3) echo "python3" ;;
                pip3) echo "python3-pip" ;;
                venv) echo "python3" ;;
                git) echo "git" ;;
                curl) echo "curl" ;;
                pkgconf) echo "pkgconf-pkg-config" ;;
                python_dev) echo "python3-devel" ;;
                gobj_dev) echo "gobject-introspection-devel" ;;
                cairo_dev) echo "cairo-devel" ;;
                pciutils) echo "pciutils" ;;
                lm_sensors) echo "lm_sensors" ;;
                intel_gpu_tools) echo "intel-gpu-tools" ;;
                rocm-smi) echo "rocm-smi" ;;
                rocminfo) echo "rocminfo" ;;
                *) echo "" ;;
            esac
            ;;
        pacman)
            case "${key}" in
                python3) echo "python" ;;
                pip3) echo "python-pip" ;;
                venv) echo "python" ;;
                git) echo "git" ;;
                curl) echo "curl" ;;
                pkgconf) echo "pkgconf" ;;
                python_dev) echo "python" ;;
                gobj_dev) echo "gobject-introspection" ;;
                cairo_dev) echo "cairo" ;;
                pciutils) echo "pciutils" ;;
                lm_sensors) echo "lm_sensors" ;;
                intel_gpu_tools) echo "intel-gpu-tools" ;;
                rocm-smi) echo "rocm-smi" ;;
                rocminfo) echo "rocminfo" ;;
                *) echo "" ;;
            esac
            ;;
        zypper)
            case "${key}" in
                python3) echo "python3" ;;
                pip3) echo "python3-pip" ;;
                venv) echo "python3-virtualenv" ;;
                git) echo "git" ;;
                curl) echo "curl" ;;
                pkgconf) echo "pkg-config" ;;
                python_dev) echo "python3-devel" ;;
                gobj_dev) echo "gobject-introspection-devel" ;;
                cairo_dev) echo "cairo-devel" ;;
                pciutils) echo "pciutils" ;;
                lm_sensors) echo "lm_sensors" ;;
                intel_gpu_tools) echo "intel-gpu-tools" ;;
                rocminfo) echo "" ;;
                *) echo "" ;;
            esac
            ;;
        apk)
            case "${key}" in
                python3) echo "python3" ;;
                pip3) echo "py3-pip" ;;
                venv) echo "python3" ;;
                git) echo "git" ;;
                curl) echo "curl" ;;
                pkgconf) echo "pkgconf" ;;
                python_dev) echo "python3-dev" ;;
                gobj_dev) echo "gobject-introspection-dev" ;;
                cairo_dev) echo "cairo-dev" ;;
                pciutils) echo "pciutils" ;;
                lm_sensors) echo "lm-sensors" ;;
                intel_gpu_tools) echo "" ;;
                rocm-smi) echo "" ;;
                rocminfo) echo "" ;;
                *) echo "" ;;
            esac
            ;;
        *)
            echo ""
            ;;
    esac
}

install_package_key() {
    local key="$1"
    local package_name=""

    package_name="$(pkg_name_for_key "${key}")"
    if [[ -z "${package_name}" ]]; then
        log "WARN" "No package mapping for '${key}' on ${PKG_MANAGER} available."
        return 1
    fi

    if [[ "${DRY_RUN}" == "0" ]]; then
        run_pkg_update_once
    fi

    local install_cmd="${PKG_INSTALL_CMD} ${package_name}"
    if [[ "${DRY_RUN}" == "1" ]]; then
        log "INFO" "[DRY-RUN] Would install: '${package_name}' via ${PKG_MANAGER}"
        return 0
    fi

    log "INFO" "Installiere Paket '${package_name}' via ${PKG_MANAGER} ..."
    if run_with_privileges "${install_cmd}"; then
        log "OK" "Paket '${package_name}' installiert."
        return 0
    fi

    log "ERROR" "Installation von '${package_name}' fehlgeschlagen."
    return 1
}

check_binary_dependency() {
    local cmd_name="$1"
    local label="$2"
    local package_key="$3"

    if command -v "${cmd_name}" >/dev/null 2>&1; then
        log "OK" "${label} found (${cmd_name})."
        return 0
    fi

    log "WARN" "${label} missing (${cmd_name})."
    if [[ -n "${PKG_MANAGER}" ]] && ask_yes_no "Install ${label} now?"; then
        install_package_key "${package_key}" || true
    fi

    if command -v "${cmd_name}" >/dev/null 2>&1; then
        log "OK" "${label} is now available."
    else
        log "WARN" "${label} still not available."
    fi
}

check_system_libs() {
    local missing_dev="0"

    if command -v pkg-config >/dev/null 2>&1; then
        if ! pkg-config --exists gobject-introspection-1.0; then
            missing_dev="1"
            log "WARN" "gobject-introspection dev files missing."
            if [[ -n "${PKG_MANAGER}" ]] && \
               ask_yes_no "Install gobject-introspection dev package?"; then
                install_package_key "gobj_dev" || true
            fi
        fi

        if ! pkg-config --exists cairo; then
            missing_dev="1"
            log "WARN" "cairo dev files missing."
            if [[ -n "${PKG_MANAGER}" ]] && \
               ask_yes_no "Install cairo dev package?"; then
                install_package_key "cairo_dev" || true
            fi
        fi
    else
        missing_dev="1"
        log "WARN" "pkg-config missing, therefore no exact lib check possible."
    fi

    if ! python3 - <<'PY' >/dev/null 2>&1
import gi
print("ok")
PY
    then
        missing_dev="1"
        log "WARN" "Python module 'gi' missing (PyGObject)."
        if [[ -n "${PKG_MANAGER}" ]] && \
           ask_yes_no "Install Python header + GObject dev packages?"; then
            install_package_key "python_dev" || true
            install_package_key "gobj_dev" || true
            install_package_key "cairo_dev" || true
        fi
    else
        log "OK" "Python module 'gi' is available."
    fi

    if [[ "${missing_dev}" == "0" ]]; then
        log "OK" "System libraries for PyGObject/Cairo seem complete."
    fi
}

open_download_link() {
    local label="$1"
    local url="$2"

    if ! ask_yes_no "Should I open the download link for ${label} in the browser?"; then
        log "INFO" "Link for ${label} skipped: ${url}"
        return 0
    fi

    if command -v xdg-open >/dev/null 2>&1; then
        if xdg-open "${url}" >/dev/null 2>&1; then
            log "OK" "Browser opened for ${label}: ${url}"
        else
            log "WARN" "Could not open browser. Open URL manually: ${url}"
        fi
    else
        log "WARN" "xdg-open not found. Open URL manually: ${url}"
    fi
}

check_lmstudio_stack() {
    local has_lms="0"
    local has_llmster="0"
    local llmster_path=""

    if command -v lms >/dev/null 2>&1; then
        has_lms="1"
        log "OK" "LM Studio CLI found: $(sanitize_path "$(command -v lms)")"
    else
        log "WARN" "LM Studio CLI (lms) not found."
    fi

    if command -v llmster >/dev/null 2>&1; then
        has_llmster="1"
        llmster_path="$(command -v llmster)"
        log "OK" "Headless CLI found: $(sanitize_path "${llmster_path}")"
    else
        local search_paths=(
            "${HOME}/.lmstudio/llmster"
            "${HOME}/.local/share/lmstudio/llmster"
            "/opt/lmstudio/llmster"
            "/usr/local/lmstudio/llmster"
        )

        for base_path in "${search_paths[@]}"; do
            if [[ -d "${base_path}" ]]; then
                llmster_path=$(find "${base_path}" -name "llmster" -type f -executable 2>/dev/null | head -n1)
                if [[ -n "${llmster_path}" && -x "${llmster_path}" ]]; then
                    has_llmster="1"
                    log "OK" "Headless CLI found: $(sanitize_path "${llmster_path}")"
                    log "INFO" "Tip: Add '$(sanitize_path "${llmster_path}")' to your PATH or create a symlink."
                    break
                fi
            fi
        done

        if [[ "${has_llmster}" == "0" ]]; then
            log "WARN" "Headless CLI (llmster) not found."
        fi
    fi

    if [[ "${has_lms}" == "0" && "${has_llmster}" == "0" ]]; then
        log "WARN" "Neither LM Studio nor llmster found."
        log "INFO" "Note: At least one of the tools is required for benchmarks."
        open_download_link "LM Studio" "https://lmstudio.ai/download"
        open_download_link \
            "LM Studio Headless (llmster)" \
            "https://lmstudio.ai/docs/developer/core/headless_llmster/"
    elif [[ "${has_lms}" == "0" || "${has_llmster}" == "0" ]]; then
        log "INFO" "Note: Only one of the tools was found. Both are optional."
    fi
}

check_optional_tool() {
    local cmd_name="$1"
    local label="$2"
    local package_key="$3"
    local help_url="$4"

    if command -v "${cmd_name}" >/dev/null 2>&1; then
        log "OK" "${label} found (${cmd_name})."
        return 0
    fi

    log "WARN" "${label} missing (${cmd_name})."
    if [[ -n "${PKG_MANAGER}" ]] && ask_yes_no "Install ${label} now?"; then
        install_package_key "${package_key}" || true
    fi

    if command -v "${cmd_name}" >/dev/null 2>&1; then
        log "OK" "${label} is now available."
        return 0
    fi

    if [[ -n "${help_url}" ]]; then
        open_download_link "${label}" "${help_url}"
    fi
    return 1
}

check_gpu_and_monitoring() {
    local has_nvidia="0"
    local has_amd="0"
    local has_intel="0"
    local gpu_lines=""

    section "GPU Detection & Monitoring"

    if ! command -v lspci >/dev/null 2>&1; then
        log "WARN" "lspci missing. GPU detection will be limited."
        if [[ -n "${PKG_MANAGER}" ]] && ask_yes_no "install lspci (pciutils)?"; then
            install_package_key "pciutils" || true
        fi
    fi

    if command -v lspci >/dev/null 2>&1; then
        gpu_lines="$(lspci | grep -Ei 'vga|3d|display' || true)"
        if [[ -n "${gpu_lines}" ]]; then
            log "INFO" "Detected graphics adapters:"
            while IFS= read -r line; do
                [[ -n "${line}" ]] && log "INFO" "  - ${line}"
            done <<< "${gpu_lines}"
        else
            log "WARN" "No GPU entries found via lspci."
        fi

        if grep -Eiq 'nvidia' <<< "${gpu_lines}"; then
            has_nvidia="1"
        fi
        if grep -Eiq 'amd|advanced micro devices|radeon' <<< "${gpu_lines}"; then
            has_amd="1"
        fi
        if grep -Eiq 'intel' <<< "${gpu_lines}"; then
            has_intel="1"
        fi
    else
        log "WARN" "GPU details not available (lspci missing)."
    fi

    check_optional_tool "sensors" "lm-sensors (alle GPUs)" "lm_sensors" "" || true

    if [[ "${has_nvidia}" == "1" ]]; then
        log "INFO" "NVIDIA GPU detected."
        check_optional_tool \
            "nvidia-smi" \
            "NVIDIA Treiber-Tool (nvidia-smi)" \
            "" \
            "https://www.nvidia.com/Download/index.aspx" || true
    fi

    if [[ "${has_amd}" == "1" ]]; then
        log "INFO" "AMD GPU detected."
        check_optional_tool \
            "rocm-smi" \
            "ROCm SMI for AMD (rocm-smi)" \
            "" \
            "https://rocm.docs.amd.com/projects/install-on-linux/en/latest/" || true
    fi

    if [[ "${has_intel}" == "1" ]]; then
        log "INFO" "Intel GPU detected."
        check_optional_tool \
            "intel_gpu_top" \
            "Intel GPU Tools (intel_gpu_top)" \
            "intel_gpu_tools" \
            "https://gitlab.freedesktop.org/drm/igt-gpu-tools" || true
    fi

    if [[ "${has_nvidia}" == "0" && "${has_amd}" == "0" && \
          "${has_intel}" == "0" ]]; then
        log "WARN" "GPU vendor could not be clearly detected."
    fi
}

check_amd_drivers() {
    section "AMD GPU Treiber & ROCm"
    
    local gpu_device_id=""
    local gpu_sku=""
    
    if lspci -nn 2>/dev/null | grep -i "AMD/ATI" | grep -qE "\[03[0-9a-f]{2}\]"; then
        gpu_device_id=$(lspci -nn 2>/dev/null | grep -i "AMD/ATI" | grep -oP '\[1002:[0-9a-f]{4}\]' | head -1 | tr -d '[]')
        log "OK" "AMD GPU found: ${gpu_device_id}"
    else
        log "INFO" "No AMD GPU detected. Skipping AMD driver check."
        return 0
    fi
    
    if lsmod 2>/dev/null | grep -q "^amdgpu "; then
        log "OK" "amdgpu kernel driver loaded"
    elif lspci -k 2>/dev/null | grep -A 2 "AMD/ATI" | grep -q "Kernel driver in use: amdgpu"; then
        log "OK" "amdgpu kernel driver active (lspci)"
    else
        log "WARN" "amdgpu kernel driver not loaded or not active"
        log "INFO" "Note: A proprietary or old driver may be running."
        log "INFO" "Recommendation: Install 'linux-firmware' and 'xserver-xorg-video-amdgpu'"
    fi
    
    if dpkg -l xserver-xorg-video-amdgpu 2>/dev/null | grep -q "^ii"; then
        local xorg_version
        xorg_version=$(dpkg -l xserver-xorg-video-amdgpu 2>/dev/null | grep "^ii" | awk '{print $3}')
        log "OK" "X.Org display driver (amdgpu): v${xorg_version}"
    elif rpm -qa xserver-xorg-video-amdgpu 2>/dev/null | grep -q "xserver"; then
        log "OK" "X.Org display driver (amdgpu) installed"
    else
        log "INFO" "xserver-xorg-video-amdgpu not installed (optional for display)"
        log "INFO" "If display problems occur, install: sudo apt install xserver-xorg-video-amdgpu"
    fi
    
    if command -v rocm-smi >/dev/null 2>&1; then
        log "OK" "rocm-smi found: $(command -v rocm-smi)"
        
        if rocm-smi --showproductname >/dev/null 2>&1; then
            gpu_sku=$(rocm-smi --showproductname 2>/dev/null | grep "Card SKU:" | awk '{print $NF}')
            if [[ -n "${gpu_sku}" ]]; then
                log "OK" "ROCm recognizes GPU: ${gpu_sku}"
            else
                log "OK" "ROCm recognizes GPU (SKU not determinable)"
            fi
        else
            log "WARN" "rocm-smi installed, but GPU query failed"
            log "INFO" "ROCm kernel modules or permissions may be missing."
        fi
    else
        log "WARN" "rocm-smi not found"
        log "INFO" "rocm-smi is important for GPU monitoring (temperature, VRAM, etc.)"
        
        if [[ "${INTERACTIVE}" == "1" ]] && [[ "${DRY_RUN}" == "0" ]]; then
            if ask_yes_no "Install rocm-smi now?"; then
                install_package_key "rocm-smi" || log "ERROR" "Installation failed"
            fi
        else
            log "INFO" "Install with: sudo apt install rocm-smi  # Ubuntu/Debian"
        fi
    fi
    
    if command -v rocminfo >/dev/null 2>&1; then
        log "OK" "rocminfo found: $(command -v rocminfo)"
    else
        log "INFO" "rocminfo not installed (optional)"
        log "INFO" "rocminfo shows detailed ROCm platform information"
        
        if [[ "${INTERACTIVE}" == "1" ]] && [[ "${DRY_RUN}" == "0" ]]; then
            if ask_yes_no "Install rocminfo now? (optional)"; then
                install_package_key "rocminfo" || log "WARN" "Installation failed"
            fi
        fi
    fi
    
    local rocm_found="0"
    local rocm_version="unknown"
    
    if [[ -d "/opt/rocm" ]]; then
        rocm_found="1"
    else
        for dir in /opt/rocm-*; do
            if [[ -d "${dir}" ]]; then
                rocm_found="1"
                break
            fi
        done
    fi
    
    if [[ "${rocm_found}" == "1" ]]; then
        local rocm_path
        rocm_path=$(find /opt -maxdepth 1 -type d -name "rocm*" 2>/dev/null | head -1)
        if [[ -n "${rocm_path}" ]]; then
            rocm_version=$(basename "${rocm_path}" | grep -oP 'rocm-\K[\d.]+' || echo "unknown")
        fi
        log "OK" "ROCm SDK installed: ${rocm_version}"
    else
        log "INFO" "ROCm SDK not found in /opt/rocm"
        log "INFO" "For machine learning with AMD GPUs, the complete ROCm SDK can be useful."
        log "INFO" "Download: https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html"
        log "INFO" "Note: LM Studio primarily uses GGML/llama.cpp integration,"
        log "INFO" "         the complete ROCm SDK is only needed for special frameworks."
    fi
    
    if dpkg -l libdrm-amdgpu1 2>/dev/null | grep -q "^ii"; then
        log "OK" "libdrm-amdgpu1 installed"
    elif rpm -qa 2>/dev/null | grep -q "^libdrm-amdgpu"; then
        log "OK" "libdrm-amdgpu installed"
    else
        log "WARN" "libdrm-amdgpu not found"
        log "INFO" "This package is important for userspace communication with AMD GPUs"
    fi
    
    if [[ "${gpu_device_id}" == "1002:150e" ]]; then
        log "WARN" "Radeon 890M (STRIX Point) is a very new iGPU"
        log "INFO" "For optimal support you need:"
        log "INFO" "  - Kernel 6.12+ (current: $(uname -r))"
        log "INFO" "  - Mesa 24.2+ or AMDGPU-PRO driver"
        log "INFO" "  - ROCm 6.2+ for computing workloads"
        log "INFO" "If problems occur: Check LM Studio logs for GPU errors"
    fi
}

check_python_requirements() {
    if ! command -v python3 >/dev/null 2>&1; then
        log "ERROR" "Python3 missing. Python dependencies cannot be checked."
        return 1
    fi

    if [[ ! -f "${PROJECT_ROOT}/requirements.txt" ]]; then
        log "WARN" "requirements.txt not found."
        return 0
    fi

    section "Python Dependencies"

    if [[ -n "${VIRTUAL_ENV:-}" ]]; then
        log "INFO" "Virtual environment active: $(sanitize_path "${VIRTUAL_ENV}")"
    else
        log "INFO" "No venv active. Note: 'source .venv/bin/activate' recommended before installation."
    fi

    if ! python3 -m pip --version >/dev/null 2>&1; then
        log "ERROR" "pip not available in current Python environment."
        if [[ -z "${VIRTUAL_ENV:-}" ]] && [[ -d "${PROJECT_ROOT}/.venv" ]]; then
            log "INFO" "Run 'source .venv/bin/activate' and restart the setup."
        fi
        return 1
    fi

    log "OK" "pip found: $(python3 -m pip --version | head -n1)"

    if ! ask_yes_no "Should I run 'python3 -m pip install -r requirements.txt'?"; then
        log "INFO" "Installation of requirements.txt skipped."
        return 0
    fi

    if [[ -z "${VIRTUAL_ENV:-}" ]] && [[ -d "${PROJECT_ROOT}/.venv" ]]; then
        log "INFO" "Activating .venv internally for installation..."
        # shellcheck source=/dev/null
        if source "${PROJECT_ROOT}/.venv/bin/activate" 2>/dev/null; then
            log "OK" "venv internally activated: $(sanitize_path "${VIRTUAL_ENV}")"
        else
            log "ERROR" "Could not activate .venv."
            log "INFO" "Please activate manually: source .venv/bin/activate"
            return 1
        fi
    fi

    if [[ "${DRY_RUN}" == "0" ]]; then
        if python3 -m pip check >/dev/null 2>&1; then
            log "OK" "Current Python environment has no pip conflicts."
        else
            log "WARN" "pip reports possible package conflicts in current environment."
        fi
    else
        log "INFO" "[DRY-RUN] Pip conflict check skipped."
    fi

    if [[ "${DRY_RUN}" == "1" ]]; then
        log "INFO" "[DRY-RUN] Would execute: python3 -m pip install -r requirements.txt"
    else
        if python3 -m pip install -r "${PROJECT_ROOT}/requirements.txt"; then
            log "OK" "Python dependencies installed/updated."
        else
            log "ERROR" "Installation from requirements.txt failed."
            return 1
        fi
    fi

    return 0
}

create_project_venv() {
    section "Python Virtual Environment"

    if ! command -v python3 >/dev/null 2>&1; then
        log "ERROR" "python3 missing. .venv cannot be created."
        return 1
    fi

    if ! python3 -m venv --help >/dev/null 2>&1; then
        log "WARN" "Python venv module missing."
        if [[ -n "${PKG_MANAGER}" ]] && ask_yes_no "Install venv module now?"; then
            install_package_key "venv" || true
        fi
    fi

    if ! python3 -m venv --help >/dev/null 2>&1; then
        log "ERROR" "venv module still not available."
        return 1
    fi

    if [[ -d "${PROJECT_ROOT}/.venv" ]]; then
        log "INFO" ".venv already exists."
        if ask_yes_no "Delete existing .venv and recreate?"; then
            if [[ "${DRY_RUN}" == "1" ]]; then
                log "INFO" "[DRY-RUN] Would delete: $(sanitize_path "${PROJECT_ROOT}")/.venv"
            else
                rm -rf "${PROJECT_ROOT}/.venv"
            fi
        else
            log "INFO" "Existing .venv will continue to be used."
            log "INFO" "Activate with: source .venv/bin/activate"
            return 0
        fi
    fi

    if [[ "${DRY_RUN}" == "1" ]]; then
        log "INFO" "[DRY-RUN] Would create: $(sanitize_path "${PROJECT_ROOT}")/.venv"
        log "INFO" "[DRY-RUN] Activation would work with: source .venv/bin/activate"
        return 0
    fi

    if python3 -m venv "${PROJECT_ROOT}/.venv"; then
        log "OK" "Virtual environment created: $(sanitize_path "${PROJECT_ROOT}")/.venv"
        log "INFO" "Activate with: source .venv/bin/activate"
        log "INFO" "Python requirements will be checked in next step."
        return 0
    fi

    log "ERROR" "Creation of virtual environment failed."
    return 1
}

summary() {
    section "Summary"
    log "INFO" "Setup check completed."
    log "INFO" "Log file: $(sanitize_path "${LOG_FILE}")"
    echo ""
    log "INFO" "Next steps:"
    log "INFO" ""
    log "INFO" "1. Activate virtual environment:"
    log "INFO" "   source .venv/bin/activate"
    log "INFO" ""
    log "INFO" "2. Start webapp (Web Dashboard):"
    log "INFO" "   python run.py --webapp"
    log "INFO" "   or: python run.py -w"
    log "INFO" ""
    log "INFO" "3. Run benchmark directly:"
    log "INFO" "   python run.py"
    log "INFO" "   python run.py --limit 5        # Test only 5 models"
    log "INFO" "   python run.py --export-only    # Generate reports from cache"
    log "INFO" ""
    log "INFO" "4. Activate debug mode:"
    log "INFO" "   python run.py --debug"
    log "INFO" "   python run.py -d"
    log "INFO" ""
    log "INFO" "For all options: python run.py --help"
}

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --help)
                print_help
                exit 0
                ;;
            --dry-run)
                DRY_RUN="1"
                INTERACTIVE="0"
                log "INFO" "Dry-run mode activated (--dry-run)"
                shift
                ;;
            --yes)
                INTERACTIVE="0"
                log "INFO" "Non-interactive mode activated (--yes)"
                shift
                ;;
            --interactive)
                INTERACTIVE="1"
                log "INFO" "Interactive mode enforced"
                shift
                ;;
            *)
                log "ERROR" "Unknown option: $1"
                print_help
                exit 1
                ;;
        esac
    done
}

main() {
    parse_arguments "$@"

    section "LM-Studio-Bench Setup Check"
    log "INFO" "Project path: $(sanitize_path "${PROJECT_ROOT}")"
    log "INFO" "Log file: $(sanitize_path "${LOG_FILE}")"
    if [[ "${DRY_RUN}" == "1" ]]; then
        log "INFO" "Mode: Dry-Run (Preview Only)"
    elif [[ "${INTERACTIVE}" == "0" ]]; then
        log "INFO" "Mode: Non-Interactive (Automation)"
    else
        log "INFO" "Mode: Interactive"
    fi

    ensure_linux
    detect_pkg_manager || true

    section "Core Dependencies"
    check_binary_dependency "python3" "Python 3" "python3"
    check_binary_dependency "git" "Git" "git"
    check_binary_dependency "curl" "curl" "curl"
    check_binary_dependency "pkg-config" "pkg-config" "pkgconf"

    section "System Libraries"
    check_system_libs

    section "LM Studio / llmster"
    check_lmstudio_stack

    check_gpu_and_monitoring
    check_amd_drivers

    create_project_venv || true
    check_python_requirements || true
    summary
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi