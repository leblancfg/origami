#!/bin/bash
set -euo pipefail

readonly REVISION="bea3b12daee45876b0129a3602dc8f534ce30bf0"
readonly REMOTE="https://github.com/ggml-org/llama.cpp.git"
DEPS_ROOT="${ORIGAMI_DEPS_ROOT:-/private/tmp/origami-deps}"
SOURCE_DIR="${DEPS_ROOT}/llama.cpp-${REVISION}"
BUILD_DIR="${SOURCE_DIR}/build-origami"

command -v git >/dev/null || { echo "error: git is required" >&2; exit 1; }
command -v cmake >/dev/null || { echo "error: cmake is required" >&2; exit 1; }

mkdir -p "${DEPS_ROOT}"
if [[ ! -d "${SOURCE_DIR}/.git" ]]; then
    git init "${SOURCE_DIR}"
    git -C "${SOURCE_DIR}" remote add origin "${REMOTE}"
fi

if [[ "$(git -C "${SOURCE_DIR}" remote get-url origin)" != "${REMOTE}" ]]; then
    echo "error: ${SOURCE_DIR} has an unexpected origin" >&2
    exit 1
fi

git -C "${SOURCE_DIR}" fetch --depth=1 origin "${REVISION}"
git -C "${SOURCE_DIR}" checkout --detach "${REVISION}"
git -C "${SOURCE_DIR}" reset --hard "${REVISION}"
git -C "${SOURCE_DIR}" clean -ffd -e build-origami/
[[ "$(git -C "${SOURCE_DIR}" rev-parse HEAD)" == "${REVISION}" ]] || {
    echo "error: llama.cpp checkout is not at ${REVISION}" >&2
    exit 1
}

cmake -S "${SOURCE_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF \
    -DLLAMA_CURL=OFF \
    -DLLAMA_BUILD_APP=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_BUILD_TESTS=ON \
    -DLLAMA_BUILD_UI=OFF \
    -DLLAMA_USE_PREBUILT_UI=OFF

jobs="${ORIGAMI_BUILD_JOBS:-$(sysctl -n hw.logicalcpu 2>/dev/null || getconf _NPROCESSORS_ONLN)}"
cmake --build "${BUILD_DIR}" --config Release -j "${jobs}" --target llama-cli test-llama-archs
printf '%s\n' "${REVISION}" > "${BUILD_DIR}/origami-revision.txt"

CLI="${BUILD_DIR}/bin/llama-cli"
"${CLI}" --version
"${CLI}" --list-devices
help="$("${CLI}" --help)"
for flag in '--ctx-size' '--batch-size' '--ubatch-size' '--load-mode' '--gpu-layers' \
            '--override-tensor' '--cache-ram' '--no-warmup' '--seed' '--temp' '--simple-io'; do
    grep -Fq -- "${flag}" <<< "${help}" || {
        echo "error: pinned llama-cli help does not contain ${flag}" >&2
        exit 1
    }
done
"${CLI}" --override-tensor '^per_layer_token_embd$=CPU' --version >/dev/null
printf 'llama.cpp source: %s\nllama.cpp build:  %s\n' "${SOURCE_DIR}" "${BUILD_DIR}"
