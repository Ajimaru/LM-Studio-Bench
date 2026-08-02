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

cp "$ROOT_DIR/scripts/io.github.Ajimaru.LMStudioBench.desktop" "$APPDIR/"

mkdir -p "$APPDIR/usr/share/applications"
cp "$ROOT_DIR/scripts/io.github.Ajimaru.LMStudioBench.desktop" \
    "$APPDIR/usr/share/applications/io.github.Ajimaru.LMStudioBench.desktop"

mkdir -p "$APPDIR/usr/share/metainfo"
cp "$ROOT_DIR/scripts/io.github.Ajimaru.LMStudioBench.appdata.xml" \
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
cp -a "$ROOT_DIR/core" "$PROJECT_DIR/"
cp -a "$ROOT_DIR/cli" "$PROJECT_DIR/"
cp -a "$ROOT_DIR/agents" "$PROJECT_DIR/"
cp -a "$ROOT_DIR/web" "$PROJECT_DIR/"
cp -a "$ROOT_DIR/config" "$PROJECT_DIR/"
cp -a "$ROOT_DIR/tools" "$PROJECT_DIR/"
mkdir -p "$PROJECT_DIR/tests"
cp -a "$ROOT_DIR/tests/data" "$PROJECT_DIR/tests/"
cp -a "$ROOT_DIR/tests/prompts" "$PROJECT_DIR/tests/"
cp -a "$ROOT_DIR/scripts" "$PROJECT_DIR/"
cp -a "$ROOT_DIR/assets" "$PROJECT_DIR/"
cp -a "$ROOT_DIR/requirements.txt" "$PROJECT_DIR/"
cp -a "$ROOT_DIR/VERSION" "$PROJECT_DIR/"
# The tray reads AUTHORS at runtime for its Contributors tab; without it the
# About screen silently drops everyone but the maintainer.
cp -a "$ROOT_DIR/AUTHORS" "$PROJECT_DIR/"
cp -a "$ROOT_DIR/LICENSE" "$PROJECT_DIR/"

# Bundle the interpreter itself.
#
# A plain "python3 -m venv" leaves bin/python as a symlink to the build
# machine's /usr/bin/python3 and puts the packages under
# lib/pythonX.Y/site-packages. At runtime the host interpreter is used, so
# a host with a different minor version looks for site-packages of *its*
# version and finds nothing - every dependency appears missing. Copying the
# interpreter and its standard library into the AppDir makes the AppImage
# independent of what the user has installed.
PY_TAG="$(python3 -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")')"
# sys.executable is /usr/bin/python3, itself a symlink. Resolve it, otherwise
# the copy is a symlink named python3.12 pointing at "python3.12" - itself.
PY_REAL="$(readlink -f "$(python3 -c 'import sys; print(sys.executable)')")"
PY_STDLIB="$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["stdlib"])')"
PY_HOME="$APPDIR/usr/python"

echo "Bundling $PY_TAG from $PY_REAL"

mkdir -p "$PY_HOME/bin" "$PY_HOME/lib"
cp -L "$PY_REAL" "$PY_HOME/bin/$PY_TAG"
chmod +x "$PY_HOME/bin/$PY_TAG"
ln -sf "$PY_TAG" "$PY_HOME/bin/python3"
ln -sf "$PY_TAG" "$PY_HOME/bin/python"

if [ ! -s "$PY_HOME/bin/$PY_TAG" ]; then
    echo "Error: bundled interpreter is empty or missing." >&2
    exit 1
fi

cp -a "$PY_STDLIB" "$PY_HOME/lib/$PY_TAG"

# Trim what a benchmark run never touches. Keeps the image ~40 MB smaller.
rm -rf "$PY_HOME/lib/$PY_TAG/test" \
    "$PY_HOME/lib/$PY_TAG/idlelib" \
    "$PY_HOME/lib/$PY_TAG/tkinter" \
    "$PY_HOME/lib/$PY_TAG/turtledemo" \
    "$PY_HOME/lib/$PY_TAG/lib2to3"
find "$PY_HOME/lib/$PY_TAG" -type d -name "__pycache__" -prune -exec rm -rf {} + \
    2>/dev/null || true

# Ubuntu links python3.12 statically against libpython, so there is usually
# nothing to copy here - but a distribution that builds it shared would fail
# to start without this.
mkdir -p "$APPDIR/usr/lib/x86_64-linux-gnu"
for lib in $(ldd "$PY_REAL" | awk '/libpython/ {print $3}'); do
    cp -a "$lib"* "$APPDIR/usr/lib/x86_64-linux-gnu/" 2>/dev/null || true
done

# Install dependencies into the bundled interpreter's own site-packages
# rather than a venv. A venv records absolute paths in pyvenv.cfg, and the
# AppImage mounts under a different /tmp/.mount_* path on every start, so it
# could never resolve. Python derives its prefix from the location of its
# executable, so bin/ and lib/ next to each other are found wherever the
# AppDir happens to be mounted.
SITE_PACKAGES="$PY_HOME/lib/$PY_TAG/site-packages"
mkdir -p "$SITE_PACKAGES"

python3 -m pip install --upgrade --target "$SITE_PACKAGES" pip setuptools wheel
python3 -m pip install --target "$SITE_PACKAGES" -r "$ROOT_DIR/requirements.txt"

find "$SITE_PACKAGES" -type d -name "__pycache__" -prune -exec rm -rf {} + \
    2>/dev/null || true

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

# Debian and Ubuntu patch site.py to prefer dist-packages, so a bundled
# prefix does not always pick up its own site-packages. Naming it here makes
# the dependencies findable regardless of that patch.
APPIMAGE_SITE_PACKAGES="$(echo "$APPDIR"/usr/python/lib/python*/site-packages)"
APPIMAGE_PYTHONPATH="$PROJECT_DIR:$APPIMAGE_SITE_PACKAGES"
if [ -n "${PYTHONPATH:-}" ]; then
    export PYTHONPATH="$APPIMAGE_PYTHONPATH:${PYTHONPATH}"
else
    export PYTHONPATH="$APPIMAGE_PYTHONPATH"
fi

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
    exec "$APPDIR/usr/python/bin/python3" \
    "$PROJECT_DIR/core/tray.py" "$@"
else
    exec "$APPDIR/usr/python/bin/python3" \
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

# Prove the bundle stands on its own before packing it: run the interpreter
# from inside the AppDir with the host's PATH and PYTHONHOME cleared, and
# import the dependency that broke first when the interpreter was not
# bundled. A build that ships a broken bundle is worse than a failed build.
echo "Verifying runtime data files..."
for required in VERSION AUTHORS LICENSE; do
    if [ ! -f "$PROJECT_DIR/$required" ]; then
        echo "Error: $required missing from the bundle." >&2
        exit 1
    fi
done

echo "Verifying bundled interpreter..."
env -i "$PY_HOME/bin/python3" -c "
import sys
sys.path.insert(0, '$PY_HOME/lib/$PY_TAG/site-packages')
import httpx, fastapi, psutil  # noqa: F401
print(f'  interpreter {sys.version.split()[0]} from {sys.prefix}')
print(f'  httpx {httpx.__version__}')
" || {
    echo "Error: bundled interpreter cannot import its dependencies." >&2
    exit 1
}

mkdir -p "$DIST_DIR"
ARCH=x86_64 appimagetool "$APPDIR" "$APPIMAGE_OUT"

echo "AppImage created: $APPIMAGE_OUT"
