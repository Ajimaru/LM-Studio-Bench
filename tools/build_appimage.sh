#!/usr/bin/env bash
# Build an AppImage for LM-Studio-Bench.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DIST_DIR="$ROOT_DIR/dist"
APPDIR="$DIST_DIR/AppDir"
VERSION="$(tr -d '[:space:]' < "$ROOT_DIR/VERSION" 2>/dev/null || echo 'unknown')"
APPIMAGE_NAME="${APPIMAGE_NAME:-LM-Studio-Bench-${VERSION}-x86_64.AppImage}"
APPIMAGE_OUT="$DIST_DIR/$APPIMAGE_NAME"
PROJECT_DIR="$APPDIR/usr/share/lm-studio-bench"
VENV_DIR="$APPDIR/usr/venv"

if ! command -v python3 >/dev/null 2>&1; then
    echo "Error: python3 is required." >&2
    exit 1
fi

if ! command -v appimagetool >/dev/null 2>&1; then
    echo "Error: appimagetool is required in PATH." >&2
    echo "Download: https://github.com/AppImage/appimagetool/releases/tag/continuous" >&2
    exit 1
fi

rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/bin" "$APPDIR/usr/share"

cp "$ROOT_DIR/assets/icons/lmstudio-bench.svg" "$APPDIR/lmstudio-bench.svg"

cp "$ROOT_DIR/tools/io.github.Ajimaru.LMStudioBench.desktop" "$APPDIR/"

mkdir -p "$APPDIR/usr/share/applications"
cp "$ROOT_DIR/tools/io.github.Ajimaru.LMStudioBench.desktop" \
    "$APPDIR/usr/share/applications/io.github.Ajimaru.LMStudioBench.desktop"

mkdir -p "$APPDIR/usr/share/metainfo"
cp "$ROOT_DIR/tools/io.github.Ajimaru.LMStudioBench.appdata.xml" \
    "$APPDIR/usr/share/metainfo/io.github.Ajimaru.LMStudioBench.appdata.xml"

cat >"$APPDIR/AppRun" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
APPDIR="$(cd "$(dirname "$0")" && pwd)"
exec "$APPDIR/usr/bin/lmstudio-bench" "$@"
EOF
chmod +x "$APPDIR/AppRun"

mkdir -p "$PROJECT_DIR"
cp -a "$ROOT_DIR/run.py" "$PROJECT_DIR/"
cp -a "$ROOT_DIR/src" "$PROJECT_DIR/"
cp -a "$ROOT_DIR/web" "$PROJECT_DIR/"
cp -a "$ROOT_DIR/config" "$PROJECT_DIR/"
cp -a "$ROOT_DIR/tools" "$PROJECT_DIR/"
cp -a "$ROOT_DIR/assets" "$PROJECT_DIR/"
cp -a "$ROOT_DIR/requirements.txt" "$PROJECT_DIR/"
cp -a "$ROOT_DIR/VERSION" "$PROJECT_DIR/"

python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install --upgrade pip setuptools wheel
"$VENV_DIR/bin/pip" install -r "$ROOT_DIR/requirements.txt"

cat >"$APPDIR/usr/bin/lmstudio-bench" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
APPDIR="$(cd "$(dirname "$0")/../.." && pwd)"
PROJECT_DIR="$APPDIR/usr/share/lm-studio-bench"

if ! command -v lms >/dev/null 2>&1; then
    echo "LM Studio CLI not found. Install LM Studio and ensure 'lms' is in PATH."
    exit 1
fi

APPIMAGE_GI_PATH="$APPDIR/usr/lib/x86_64-linux-gnu/girepository-1.0"
APPIMAGE_GI_PATH="$APPIMAGE_GI_PATH:$APPDIR/usr/lib/girepository-1.0"
if [ -n "${GI_TYPELIB_PATH:-}" ]; then
    export GI_TYPELIB_PATH="$APPIMAGE_GI_PATH:${GI_TYPELIB_PATH}"
else
    export GI_TYPELIB_PATH="$APPIMAGE_GI_PATH"
fi

APPIMAGE_LD_PATH="$APPDIR/usr/lib:$APPDIR/usr/lib/x86_64-linux-gnu"
if [ -n "${LD_LIBRARY_PATH:-}" ]; then
    export LD_LIBRARY_PATH="$APPIMAGE_LD_PATH:${LD_LIBRARY_PATH}"
else
    export LD_LIBRARY_PATH="$APPIMAGE_LD_PATH"
fi

export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR"

# When started with no real arguments (--debug/-d are exempt), launch only
# the tray app so it stays in the system tray without auto-running a
# benchmark.  With any other argument, delegate to run.py as usual.
HAS_REAL_ARGS=0
for _arg in "$@"; do
    case "$_arg" in
        --debug|-d) ;;
        *) HAS_REAL_ARGS=1; break ;;
    esac
done

if [ "$HAS_REAL_ARGS" -eq 0 ]; then
    exec "$APPDIR/usr/venv/bin/python" \
    "$PROJECT_DIR/core/tray.py" "$@"
else
    exec "$APPDIR/usr/venv/bin/python" \
        "$PROJECT_DIR/run.py" "$@"
fi
EOF
chmod +x "$APPDIR/usr/bin/lmstudio-bench"

mkdir -p "$APPDIR/usr/lib/x86_64-linux-gnu/girepository-1.0"

copy_if_exists() {
    local source_file="$1"
    local target_dir="$2"
    if [ -f "$source_file" ]; then
        cp -a "$source_file" "$target_dir/"
    fi
}

copy_matches() {
    local pattern="$1"
    local target_dir="$2"
    local matched=1
    local -a matches

    mapfile -t matches < <(compgen -G "$pattern")

    for source_file in "${matches[@]}"; do
        if [ -f "$source_file" ]; then
            cp -a "$source_file" "$target_dir/"
            matched=0
        fi
    done
    return "$matched"
}

TYPELIB_DIR="/usr/lib/x86_64-linux-gnu/girepository-1.0"
copy_if_exists "$TYPELIB_DIR/AyatanaAppIndicator3-0.1.typelib" \
    "$APPDIR/usr/lib/x86_64-linux-gnu/girepository-1.0"
copy_if_exists "$TYPELIB_DIR/AppIndicator3-0.1.typelib" \
    "$APPDIR/usr/lib/x86_64-linux-gnu/girepository-1.0"
copy_if_exists "$TYPELIB_DIR/Gtk-3.0.typelib" \
    "$APPDIR/usr/lib/x86_64-linux-gnu/girepository-1.0"

mkdir -p "$APPDIR/usr/lib/x86_64-linux-gnu"

copy_matches "/usr/lib/x86_64-linux-gnu/libayatana-appindicator3.so.1*" \
    "$APPDIR/usr/lib/x86_64-linux-gnu" || true
copy_matches "/usr/lib/x86_64-linux-gnu/libdbusmenu-glib.so.4*" \
    "$APPDIR/usr/lib/x86_64-linux-gnu" || true
copy_matches "/usr/lib/x86_64-linux-gnu/libdbusmenu-gtk3.so.4*" \
    "$APPDIR/usr/lib/x86_64-linux-gnu" || true

mkdir -p "$DIST_DIR"
ARCH=x86_64 appimagetool "$APPDIR" "$APPIMAGE_OUT"

echo "AppImage created: $APPIMAGE_OUT"
