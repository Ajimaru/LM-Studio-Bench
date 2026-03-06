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
                log "WARN" "Bitte mit y(es) oder n(o) antworten."
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
        log "ERROR" "Dieses Setup-Skript unterstützt aktuell nur Linux."
        exit 1
    fi
    log "OK" "Linux erkannt: $(uname -sr)"
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
        log "WARN" "Kein unterstützter Paketmanager erkannt (apt/dnf/pacman/zypper/apk)."
        return 1
    fi

    log "OK" "Paketmanager erkannt: ${PKG_MANAGER}"
    return 0
}

run_pkg_update_once() {
    if [[ "${PKG_UPDATED}" == "1" ]]; then
        return 0
    fi

    if [[ "${PKG_MANAGER}" == "apt" || "${PKG_MANAGER}" == "pacman" || \
          "${PKG_MANAGER}" == "zypper" || "${PKG_MANAGER}" == "apk" ]]; then
        if ! run_with_privileges "${PKG_UPDATE_CMD}"; then
            log "WARN" "Paketindex konnte nicht aktualisiert werden. Fahre fort."
        fi
    fi

    PKG_UPDATED="1"
    return 0
}

run_with_privileges() {
    local cmd="$1"

    if [[ "${DRY_RUN}" == "1" ]]; then
        log "INFO" "[DRY-RUN] Würde ausführen: ${cmd}"
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

    log "ERROR" "Root-Rechte nötig, aber sudo ist nicht verfügbar."
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
        log "WARN" "Kein Paket-Mapping für '${key}' auf ${PKG_MANAGER} vorhanden."
        return 1
    fi

    if [[ "${DRY_RUN}" == "0" ]]; then
        run_pkg_update_once
    fi

    local install_cmd="${PKG_INSTALL_CMD} ${package_name}"
    if [[ "${DRY_RUN}" == "1" ]]; then
        log "INFO" "[DRY-RUN] Würde installieren: '${package_name}' via ${PKG_MANAGER}"
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
        log "OK" "${label} gefunden (${cmd_name})."
        return 0
    fi

    log "WARN" "${label} fehlt (${cmd_name})."
    if [[ -n "${PKG_MANAGER}" ]] && ask_yes_no "${label} jetzt installieren?"; then
        install_package_key "${package_key}" || true
    fi

    if command -v "${cmd_name}" >/dev/null 2>&1; then
        log "OK" "${label} ist jetzt verfügbar."
    else
        log "WARN" "${label} weiterhin nicht verfügbar."
    fi
}

check_system_libs() {
    local missing_dev="0"

    if command -v pkg-config >/dev/null 2>&1; then
        if ! pkg-config --exists gobject-introspection-1.0; then
            missing_dev="1"
            log "WARN" "gobject-introspection dev-Dateien fehlen."
            if [[ -n "${PKG_MANAGER}" ]] && \
               ask_yes_no "gobject-introspection dev-Paket installieren?"; then
                install_package_key "gobj_dev" || true
            fi
        fi

        if ! pkg-config --exists cairo; then
            missing_dev="1"
            log "WARN" "cairo dev-Dateien fehlen."
            if [[ -n "${PKG_MANAGER}" ]] && \
               ask_yes_no "cairo dev-Paket installieren?"; then
                install_package_key "cairo_dev" || true
            fi
        fi
    else
        missing_dev="1"
        log "WARN" "pkg-config fehlt, daher keine genaue Lib-Prüfung möglich."
    fi

    if ! python3 - <<'PY' >/dev/null 2>&1
import gi
print("ok")
PY
    then
        missing_dev="1"
        log "WARN" "Python-Modul 'gi' fehlt (PyGObject)."
        if [[ -n "${PKG_MANAGER}" ]] && \
           ask_yes_no "Python-Header + GObject-Dev-Pakete installieren?"; then
            install_package_key "python_dev" || true
            install_package_key "gobj_dev" || true
            install_package_key "cairo_dev" || true
        fi
    else
        log "OK" "Python-Modul 'gi' ist verfügbar."
    fi

    if [[ "${missing_dev}" == "0" ]]; then
        log "OK" "System-Libraries für PyGObject/Cairo wirken vollständig."
    fi
}

open_download_link() {
    local label="$1"
    local url="$2"

    if ! ask_yes_no "Soll ich den Download-Link für ${label} im Browser öffnen?"; then
        log "INFO" "Link für ${label} übersprungen: ${url}"
        return 0
    fi

    if command -v xdg-open >/dev/null 2>&1; then
        if xdg-open "${url}" >/dev/null 2>&1; then
            log "OK" "Browser geöffnet für ${label}: ${url}"
        else
            log "WARN" "Konnte Browser nicht öffnen. URL manuell öffnen: ${url}"
        fi
    else
        log "WARN" "xdg-open nicht gefunden. URL manuell öffnen: ${url}"
    fi
}

check_lmstudio_stack() {
    local has_lms="0"
    local has_llmster="0"
    local llmster_path=""

    if command -v lms >/dev/null 2>&1; then
        has_lms="1"
        log "OK" "LM Studio CLI gefunden: $(sanitize_path "$(command -v lms)")"
    else
        log "WARN" "LM Studio CLI (lms) nicht gefunden."
    fi

    if command -v llmster >/dev/null 2>&1; then
        has_llmster="1"
        llmster_path="$(command -v llmster)"
        log "OK" "Headless CLI gefunden: $(sanitize_path "${llmster_path}")"
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
                    log "OK" "Headless CLI gefunden: $(sanitize_path "${llmster_path}")"
                    log "INFO" "Tipp: Füge '$(sanitize_path "${llmster_path}")' zu deinem PATH hinzu oder erstelle einen Symlink."
                    break
                fi
            fi
        done

        if [[ "${has_llmster}" == "0" ]]; then
            log "WARN" "Headless CLI (llmster) nicht gefunden."
        fi
    fi

    if [[ "${has_lms}" == "0" && "${has_llmster}" == "0" ]]; then
        log "WARN" "Weder LM Studio noch llmster gefunden."
        log "INFO" "Hinweis: Mindestens eines der Tools wird für Benchmarks benötigt."
        open_download_link "LM Studio" "https://lmstudio.ai/download"
        open_download_link \
            "LM Studio Headless (llmster)" \
            "https://lmstudio.ai/docs/developer/core/headless_llmster/"
    elif [[ "${has_lms}" == "0" || "${has_llmster}" == "0" ]]; then
        log "INFO" "Hinweis: Es wurde nur eins der Tools gefunden. Beide sind optional."
    fi
}

check_optional_tool() {
    local cmd_name="$1"
    local label="$2"
    local package_key="$3"
    local help_url="$4"

    if command -v "${cmd_name}" >/dev/null 2>&1; then
        log "OK" "${label} gefunden (${cmd_name})."
        return 0
    fi

    log "WARN" "${label} fehlt (${cmd_name})."
    if [[ -n "${PKG_MANAGER}" ]] && ask_yes_no "${label} jetzt installieren?"; then
        install_package_key "${package_key}" || true
    fi

    if command -v "${cmd_name}" >/dev/null 2>&1; then
        log "OK" "${label} ist jetzt verfügbar."
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
        log "WARN" "lspci fehlt. GPU-Erkennung wird eingeschränkt sein."
        if [[ -n "${PKG_MANAGER}" ]] && ask_yes_no "lspci (pciutils) installieren?"; then
            install_package_key "pciutils" || true
        fi
    fi

    if command -v lspci >/dev/null 2>&1; then
        gpu_lines="$(lspci | grep -Ei 'vga|3d|display' || true)"
        if [[ -n "${gpu_lines}" ]]; then
            log "INFO" "Erkannte Grafikadapter:"
            while IFS= read -r line; do
                [[ -n "${line}" ]] && log "INFO" "  - ${line}"
            done <<< "${gpu_lines}"
        else
            log "WARN" "Keine GPU-Einträge über lspci gefunden."
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
        log "WARN" "GPU-Details nicht verfügbar (lspci fehlt)."
    fi

    check_optional_tool "sensors" "lm-sensors (alle GPUs)" "lm_sensors" "" || true

    if [[ "${has_nvidia}" == "1" ]]; then
        log "INFO" "NVIDIA GPU erkannt."
        check_optional_tool \
            "nvidia-smi" \
            "NVIDIA Treiber-Tool (nvidia-smi)" \
            "" \
            "https://www.nvidia.com/Download/index.aspx" || true
    fi

    if [[ "${has_amd}" == "1" ]]; then
        log "INFO" "AMD GPU erkannt."
        check_optional_tool \
            "rocm-smi" \
            "ROCm SMI für AMD (rocm-smi)" \
            "" \
            "https://rocm.docs.amd.com/projects/install-on-linux/en/latest/" || true
    fi

    if [[ "${has_intel}" == "1" ]]; then
        log "INFO" "Intel GPU erkannt."
        check_optional_tool \
            "intel_gpu_top" \
            "Intel GPU Tools (intel_gpu_top)" \
            "intel_gpu_tools" \
            "https://gitlab.freedesktop.org/drm/igt-gpu-tools" || true
    fi

    if [[ "${has_nvidia}" == "0" && "${has_amd}" == "0" && \
          "${has_intel}" == "0" ]]; then
        log "WARN" "GPU-Hersteller konnte nicht eindeutig erkannt werden."
    fi
}

check_amd_drivers() {
    section "AMD GPU Treiber & ROCm"
    
    local gpu_device_id=""
    local gpu_sku=""
    
    if lspci -nn 2>/dev/null | grep -i "AMD/ATI" | grep -qE "\[03[0-9a-f]{2}\]"; then
        gpu_device_id=$(lspci -nn 2>/dev/null | grep -i "AMD/ATI" | grep -oP '\[1002:[0-9a-f]{4}\]' | head -1 | tr -d '[]')
        log "OK" "AMD GPU gefunden: ${gpu_device_id}"
    else
        log "INFO" "Keine AMD GPU erkannt. Überspringe AMD-Treiber-Check."
        return 0
    fi
    
    if lsmod 2>/dev/null | grep -q "^amdgpu "; then
        log "OK" "amdgpu Kernel-Treiber geladen"
    elif lspci -k 2>/dev/null | grep -A 2 "AMD/ATI" | grep -q "Kernel driver in use: amdgpu"; then
        log "OK" "amdgpu Kernel-Treiber aktiv (lspci)"
    else
        log "WARN" "amdgpu Kernel-Treiber nicht geladen oder nicht aktiv"
        log "INFO" "Hinweis: Möglicherweise läuft ein proprietärer oder alter Treiber."
        log "INFO" "Empfehlung: Installiere 'linux-firmware' und 'xserver-xorg-video-amdgpu'"
    fi
    
    if dpkg -l xserver-xorg-video-amdgpu 2>/dev/null | grep -q "^ii"; then
        local xorg_version
        xorg_version=$(dpkg -l xserver-xorg-video-amdgpu 2>/dev/null | grep "^ii" | awk '{print $3}')
        log "OK" "X.Org Display-Treiber (amdgpu): v${xorg_version}"
    elif rpm -qa xserver-xorg-video-amdgpu 2>/dev/null | grep -q "xserver"; then
        log "OK" "X.Org Display-Treiber (amdgpu) installiert"
    else
        log "INFO" "xserver-xorg-video-amdgpu nicht installiert (optional für Display)"
        log "INFO" "Falls Displayprobleme auftreten, installiere: sudo apt install xserver-xorg-video-amdgpu"
    fi
    
    if command -v rocm-smi >/dev/null 2>&1; then
        log "OK" "rocm-smi gefunden: $(command -v rocm-smi)"
        
        if rocm-smi --showproductname >/dev/null 2>&1; then
            gpu_sku=$(rocm-smi --showproductname 2>/dev/null | grep "Card SKU:" | awk '{print $NF}')
            if [[ -n "${gpu_sku}" ]]; then
                log "OK" "ROCm erkennt GPU: ${gpu_sku}"
            else
                log "OK" "ROCm erkennt GPU (SKU nicht ermittelbar)"
            fi
        else
            log "WARN" "rocm-smi installiert, aber GPU-Abfrage fehlgeschlagen"
            log "INFO" "Möglicherweise fehlen ROCm-Kernel-Module oder Permissions."
        fi
    else
        log "WARN" "rocm-smi nicht gefunden"
        log "INFO" "rocm-smi ist wichtig für GPU-Monitoring (Temperatur, VRAM, etc.)"
        
        if [[ "${INTERACTIVE}" == "1" ]] && [[ "${DRY_RUN}" == "0" ]]; then
            if ask_yes_no "rocm-smi jetzt installieren?"; then
                install_package_key "rocm-smi" || log "ERROR" "Installation fehlgeschlagen"
            fi
        else
            log "INFO" "Installation mit: sudo apt install rocm-smi  # Ubuntu/Debian"
        fi
    fi
    
    if command -v rocminfo >/dev/null 2>&1; then
        log "OK" "rocminfo gefunden: $(command -v rocminfo)"
    else
        log "INFO" "rocminfo nicht installiert (optional)"
        log "INFO" "rocminfo zeigt detaillierte ROCm-Plattform-Informationen"
        
        if [[ "${INTERACTIVE}" == "1" ]] && [[ "${DRY_RUN}" == "0" ]]; then
            if ask_yes_no "rocminfo jetzt installieren? (optional)"; then
                install_package_key "rocminfo" || log "WARN" "Installation fehlgeschlagen"
            fi
        fi
    fi
    
    local rocm_found="0"
    local rocm_version="unbekannt"
    
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
            rocm_version=$(basename "${rocm_path}" | grep -oP 'rocm-\K[\d.]+' || echo "unbekannt")
        fi
        log "OK" "ROCm SDK installiert: ${rocm_version}"
    else
        log "INFO" "ROCm SDK nicht in /opt/rocm gefunden"
        log "INFO" "Für Machine Learning mit AMD GPUs kann das vollständige ROCm SDK nützlich sein."
        log "INFO" "Download: https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html"
        log "INFO" "Hinweis: LM Studio nutzt primär die GGML/llama.cpp Integration,"
        log "INFO" "         das vollständige ROCm SDK ist nur für spezielle Frameworks nötig."
    fi
    
    if dpkg -l libdrm-amdgpu1 2>/dev/null | grep -q "^ii"; then
        log "OK" "libdrm-amdgpu1 installiert"
    elif rpm -qa 2>/dev/null | grep -q "^libdrm-amdgpu"; then
        log "OK" "libdrm-amdgpu installiert"
    else
        log "WARN" "libdrm-amdgpu nicht gefunden"
        log "INFO" "Dieses Paket ist wichtig für die Userspace-Kommunikation mit AMD GPUs"
    fi
    
    if [[ "${gpu_device_id}" == "1002:150e" ]]; then
        log "WARN" "Radeon 890M (STRIX Point) ist eine sehr neue iGPU"
        log "INFO" "Für optimale Unterstützung benötigst du:"
        log "INFO" "  - Kernel 6.12+ (aktuell: $(uname -r))"
        log "INFO" "  - Mesa 24.2+ oder AMDGPU-PRO Treiber"
        log "INFO" "  - ROCm 6.2+ für Computing-Workloads"
        log "INFO" "Bei Problemen: Prüfe LM Studio logs für GPU-Fehler"
    fi
}

check_python_requirements() {
    if ! command -v python3 >/dev/null 2>&1; then
        log "ERROR" "Python3 fehlt. Python-Abhängigkeiten können nicht geprüft werden."
        return 1
    fi

    if [[ ! -f "${PROJECT_ROOT}/requirements.txt" ]]; then
        log "WARN" "requirements.txt nicht gefunden."
        return 0
    fi

    section "Python Dependencies"

    if [[ -n "${VIRTUAL_ENV:-}" ]]; then
        log "INFO" "Virtuelle Umgebung aktiv: $(sanitize_path "${VIRTUAL_ENV}")"
    else
        log "INFO" "Keine venv aktiv. Hinweis: 'source .venv/bin/activate' vor Installation empfohlen."
    fi

    if ! python3 -m pip --version >/dev/null 2>&1; then
        log "ERROR" "pip nicht verfügbar im aktuellen Python-Environment."
        if [[ -z "${VIRTUAL_ENV:-}" ]] && [[ -d "${PROJECT_ROOT}/.venv" ]]; then
            log "INFO" "Führe 'source .venv/bin/activate' aus und starte das Setup erneut."
        fi
        return 1
    fi

    log "OK" "pip gefunden: $(python3 -m pip --version | head -n1)"

    if ! ask_yes_no "Soll ich 'python3 -m pip install -r requirements.txt' ausführen?"; then
        log "INFO" "Installation von requirements.txt übersprungen."
        return 0
    fi

    if [[ -z "${VIRTUAL_ENV:-}" ]] && [[ -d "${PROJECT_ROOT}/.venv" ]]; then
        log "INFO" "Aktiviere .venv intern für Installation..."
        if source "${PROJECT_ROOT}/.venv/bin/activate" 2>/dev/null; then
            log "OK" "venv intern aktiviert: $(sanitize_path "${VIRTUAL_ENV}")"
        else
            log "ERROR" "Konnte .venv nicht aktivieren."
            log "INFO" "Bitte manuell aktivieren: source .venv/bin/activate"
            return 1
        fi
    fi

    if [[ "${DRY_RUN}" == "0" ]]; then
        if python3 -m pip check >/dev/null 2>&1; then
            log "OK" "Aktuelles Python-Environment hat keine pip-Konflikte."
        else
            log "WARN" "pip meldet mögliche Paketkonflikte im aktuellen Environment."
        fi
    else
        log "INFO" "[DRY-RUN] Pip-Konflikt-Check übersprungen."
    fi

    if [[ "${DRY_RUN}" == "1" ]]; then
        log "INFO" "[DRY-RUN] Würde ausführen: python3 -m pip install -r requirements.txt"
    else
        if python3 -m pip install -r "${PROJECT_ROOT}/requirements.txt"; then
            log "OK" "Python-Abhängigkeiten installiert/aktualisiert."
        else
            log "ERROR" "Installation aus requirements.txt fehlgeschlagen."
            return 1
        fi
    fi

    return 0
}

create_project_venv() {
    section "Python Virtual Environment"

    if ! command -v python3 >/dev/null 2>&1; then
        log "ERROR" "python3 fehlt. .venv kann nicht erstellt werden."
        return 1
    fi

    if ! python3 -m venv --help >/dev/null 2>&1; then
        log "WARN" "Python venv-Modul fehlt."
        if [[ -n "${PKG_MANAGER}" ]] && ask_yes_no "venv-Modul jetzt installieren?"; then
            install_package_key "venv" || true
        fi
    fi

    if ! python3 -m venv --help >/dev/null 2>&1; then
        log "ERROR" "venv-Modul weiterhin nicht verfügbar."
        return 1
    fi

    if [[ -d "${PROJECT_ROOT}/.venv" ]]; then
        log "INFO" ".venv existiert bereits."
        if ask_yes_no "Bestehende .venv löschen und neu erstellen?"; then
            if [[ "${DRY_RUN}" == "1" ]]; then
                log "INFO" "[DRY-RUN] Würde löschen: $(sanitize_path "${PROJECT_ROOT}")/.venv"
            else
                rm -rf "${PROJECT_ROOT}/.venv"
            fi
        else
            log "INFO" "Bestehende .venv wird weiterverwendet."
            log "INFO" "Aktivieren mit: source .venv/bin/activate"
            return 0
        fi
    fi

    if [[ "${DRY_RUN}" == "1" ]]; then
        log "INFO" "[DRY-RUN] Würde erstellen: $(sanitize_path "${PROJECT_ROOT}")/.venv"
        log "INFO" "[DRY-RUN] Aktivieren würde damit funktionieren: source .venv/bin/activate"
        return 0
    fi

    if python3 -m venv "${PROJECT_ROOT}/.venv"; then
        log "OK" "Virtuelle Umgebung erstellt: $(sanitize_path "${PROJECT_ROOT}")/.venv"
        log "INFO" "Aktivieren mit: source .venv/bin/activate"
        log "INFO" "Python-Requirements werden im nächsten Schritt geprüft."
        return 0
    fi

    log "ERROR" "Erstellung der virtuellen Umgebung fehlgeschlagen."
    return 1
}

summary() {
    section "Zusammenfassung"
    log "INFO" "Setup-Check abgeschlossen."
    log "INFO" "Logdatei: $(sanitize_path "${LOG_FILE}")"
    echo ""
    log "INFO" "Nächste Schritte:"
    log "INFO" ""
    log "INFO" "1. Virtuelle Umgebung aktivieren:"
    log "INFO" "   source .venv/bin/activate"
    log "INFO" ""
    log "INFO" "2. Webapp starten (Web Dashboard):"
    log "INFO" "   python run.py --webapp"
    log "INFO" "   oder: python run.py -w"
    log "INFO" ""
    log "INFO" "3. Benchmark direkt ausführen:"
    log "INFO" "   python run.py"
    log "INFO" "   python run.py --limit 5        # Nur 5 Modelle testen"
    log "INFO" "   python run.py --export-only    # Reports aus Cache generieren"
    log "INFO" ""
    log "INFO" "4. Debug-Modus aktivieren:"
    log "INFO" "   python run.py --debug"
    log "INFO" "   python run.py -d"
    log "INFO" ""
    log "INFO" "Für alle Optionen: python run.py --help"
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
                log "INFO" "Dry-Run Mode aktiviert (--dry-run)"
                shift
                ;;
            --yes)
                INTERACTIVE="0"
                log "INFO" "Non-interactive Mode aktiviert (--yes)"
                shift
                ;;
            --interactive)
                INTERACTIVE="1"
                log "INFO" "Interactive Mode erzwungen"
                shift
                ;;
            *)
                log "ERROR" "Unbekannte Option: $1"
                print_help
                exit 1
                ;;
        esac
    done
}

main() {
    parse_arguments "$@"

    section "LM-Studio-Bench Setup Check"
    log "INFO" "Projektpfad: $(sanitize_path "${PROJECT_ROOT}")"
    log "INFO" "Logdatei: $(sanitize_path "${LOG_FILE}")"
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