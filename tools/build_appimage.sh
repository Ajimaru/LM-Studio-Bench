#!/usr/bin/env bash
# Build an AppImage for LM-Studio-Bench.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DIST_DIR="$ROOT_DIR/dist"
APPDIR="$DIST_DIR/AppDir"
APPIMAGE_NAME="${APPIMAGE_NAME:-LM-Studio-Bench-x86_64.AppImage}"
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

export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR"
exec "$APPDIR/usr/venv/bin/python" "$PROJECT_DIR/run.py" "$@"
EOF
chmod +x "$APPDIR/usr/bin/lmstudio-bench"

mkdir -p "$DIST_DIR"
ARCH=x86_64 appimagetool "$APPDIR" "$APPIMAGE_OUT"

echo "AppImage created: $APPIMAGE_OUT"
