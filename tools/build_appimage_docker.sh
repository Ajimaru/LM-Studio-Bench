#!/usr/bin/env bash
# Build LM-Studio-Bench AppImage using Docker (Ubuntu 24.04 environment).

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE_NAME="lmstudio-bench-appimage-builder"
IMAGE_TAG="ubuntu24.04"
CONTAINER_NAME="${CONTAINER_NAME:-lmstudio-bench-appimage-build-cache}"
REUSE_CONTAINER="${REUSE_CONTAINER:-1}"
FAST_MODE="${FAST_MODE:-0}"
BUILD_CONTEXT_DIR=""

# Check if Docker is available
if ! command -v docker >/dev/null 2>&1; then
    echo "Error: docker is required but not found in PATH." >&2
    exit 1
fi

# Check if Docker daemon is running
if ! docker info >/dev/null 2>&1; then
    echo "Error: Docker daemon is not running." >&2
    exit 1
fi

cleanup_container() {
    if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
        docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
    fi
}

cleanup_build_context() {
    if [ -n "$BUILD_CONTEXT_DIR" ] && [ -d "$BUILD_CONTEXT_DIR" ]; then
        rm -rf "$BUILD_CONTEXT_DIR"
    fi
}

prepare_build_context() {
    BUILD_CONTEXT_DIR="$(mktemp -d)"

    cp -a "$ROOT_DIR/run.py" "$BUILD_CONTEXT_DIR/"
    cp -a "$ROOT_DIR/src" "$BUILD_CONTEXT_DIR/"
    cp -a "$ROOT_DIR/web" "$BUILD_CONTEXT_DIR/"
    cp -a "$ROOT_DIR/config" "$BUILD_CONTEXT_DIR/"
    cp -a "$ROOT_DIR/assets" "$BUILD_CONTEXT_DIR/"
    cp -a "$ROOT_DIR/tools" "$BUILD_CONTEXT_DIR/"
    cp -a "$ROOT_DIR/requirements.txt" "$BUILD_CONTEXT_DIR/"

    find "$BUILD_CONTEXT_DIR" -type d -name "__pycache__" -prune -exec rm -rf {} +
}

container_exists() {
    docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1
}

container_is_running() {
    if ! container_exists; then
        return 1
    fi
    docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" | grep -q true
}

image_exists() {
    docker image inspect "${IMAGE_NAME}:${IMAGE_TAG}" >/dev/null 2>&1
}

trap cleanup_build_context EXIT

if [ "$FAST_MODE" = "1" ] && image_exists; then
    echo "FAST_MODE=1: Skipping docker build (image exists: ${IMAGE_NAME}:${IMAGE_TAG})"
else
    echo "Building Docker image: ${IMAGE_NAME}:${IMAGE_TAG}..."
    docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" -f \
    "$ROOT_DIR/tools/Dockerfile.AppImage" "$ROOT_DIR/tools/"
fi

echo ""
if [ "$REUSE_CONTAINER" = "1" ] && container_exists; then
    echo "Reusing existing build container: $CONTAINER_NAME"
else
    if container_exists; then
        echo "Removing existing build container: $CONTAINER_NAME"
        cleanup_container
    fi
    echo "Creating build container: $CONTAINER_NAME"
    docker create \
        --name "$CONTAINER_NAME" \
        -w /build \
        "${IMAGE_NAME}:${IMAGE_TAG}" \
        /bin/bash -lc "./tools/build_appimage.sh" >/dev/null
fi

if container_is_running; then
    echo "Stopping running container: $CONTAINER_NAME"
    docker stop "$CONTAINER_NAME" >/dev/null
fi

echo "Preparing minimal build context..."
prepare_build_context

echo "Copying minimal build context into container..."
docker cp "$BUILD_CONTEXT_DIR/." "$CONTAINER_NAME:/build"

echo "Running AppImage build in Docker container..."
docker start -a "$CONTAINER_NAME"

echo ""
echo "Copying AppImage artifact back to host..."
mkdir -p "$ROOT_DIR/dist"
if ! docker cp \
    "$CONTAINER_NAME:/build/dist/LM-Studio-Bench-x86_64.AppImage" \
    "$ROOT_DIR/dist/LM-Studio-Bench-x86_64.AppImage"; then
    echo "Error: AppImage artifact not found in container output." >&2
    exit 1
fi

if [ "$REUSE_CONTAINER" != "1" ]; then
    echo "Cleaning up build container..."
    cleanup_container
fi

echo ""
echo "Build completed!"

if [ -f "$ROOT_DIR/dist/LM-Studio-Bench-x86_64.AppImage" ]; then
    APPIMAGE_SIZE=$(du -h "$ROOT_DIR/dist/LM-Studio-Bench-x86_64.AppImage" | cut -f1)
    echo "AppImage: dist/LM-Studio-Bench-x86_64.AppImage (${APPIMAGE_SIZE})"
    echo ""
    echo "Test run:"
    echo "  ./dist/LM-Studio-Bench-x86_64.AppImage --help"
fi
