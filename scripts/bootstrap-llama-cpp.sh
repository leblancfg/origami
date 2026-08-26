#!/bin/bash
set -euo pipefail

readonly REVISION="bea3b12daee45876b0129a3602dc8f534ce30bf0"
readonly REMOTE="https://github.com/ggml-org/llama.cpp.git"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PATCH="${ROOT}/patches/llama.cpp-bea3b12-mmap-prefetch-optout.patch"
DEPS_ROOT="${ORIGAMI_DEPS_ROOT:-/private/tmp/origami-deps}"
SOURCE_DIR="${DEPS_ROOT}/llama.cpp-${REVISION}"
BUILD_DIR="${SOURCE_DIR}/build-origami"

command -v git >/dev/null || { echo "error: git is required" >&2; exit 1; }
command -v cmake >/dev/null || { echo "error: cmake is required" >&2; exit 1; }
command -v shasum >/dev/null || { echo "error: shasum is required" >&2; exit 1; }
[[ -f "${PATCH}" ]] || { echo "error: patch not found: ${PATCH}" >&2; exit 1; }

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

before_patch="$(git -C "${SOURCE_DIR}" rev-parse HEAD)"
[[ "${before_patch}" == "${REVISION}" ]] || {
    echo "error: source revision before patch is ${before_patch}, expected ${REVISION}" >&2
    exit 1
}

if git -C "${SOURCE_DIR}" apply --check "${PATCH}"; then
    git -C "${SOURCE_DIR}" apply "${PATCH}"
elif git -C "${SOURCE_DIR}" apply --reverse --check "${PATCH}"; then
    printf 'mmap prefetch patch already applied\n'
else
    echo "error: mmap prefetch patch does not apply cleanly to ${REVISION}" >&2
    exit 1
fi

after_patch="$(git -C "${SOURCE_DIR}" rev-parse HEAD)"
[[ "${after_patch}" == "${REVISION}" ]] || {
    echo "error: source revision changed while applying the patch" >&2
    exit 1
}
git -C "${SOURCE_DIR}" apply --reverse --check "${PATCH}" || {
    echo "error: mmap prefetch patch is not present after application" >&2
    exit 1
}
git -C "${SOURCE_DIR}" diff --check
[[ "$(git -C "${SOURCE_DIR}" diff --name-only)" == "src/llama-model.cpp" ]] || {
    echo "error: patched source has changes outside src/llama-model.cpp" >&2
    git -C "${SOURCE_DIR}" status --short >&2
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
cmake --build "${BUILD_DIR}" --config Release -j "${jobs}" \
    --target llama-cli test-llama-archs test-backend-ops
printf '%s\n' "${REVISION}" > "${BUILD_DIR}/origami-revision.txt"
shasum -a 256 "${PATCH}" | awk '{print $1}' > "${BUILD_DIR}/origami-patch.sha256"

CLI="${BUILD_DIR}/bin/llama-cli"
"${CLI}" --version
"${CLI}" --list-devices
help="$("${CLI}" --help 2>&1)"
for flag in '--offline' '--model' '--prompt' '--ctx-size' '--batch-size' '--ubatch-size' \
            '--n-predict' '--load-mode' '--gpu-layers' '--override-tensor' '--cache-ram' '--no-warmup' \
            '--seed' '--temp' '--color' '--simple-io' '--single-turn' \
            '--no-display-prompt'; do
    grep -Fq -- "${flag}" <<< "${help}" || {
        echo "error: pinned llama-cli help does not contain ${flag}" >&2
        exit 1
    }
done
"${CLI}" \
    --offline --load-mode mmap --gpu-layers all \
    --ctx-size 512 --batch-size 32 --ubatch-size 32 \
    --cache-ram 0 --no-warmup --color off --simple-io --single-turn \
    --override-tensor '^output=CPU' \
    --model /tmp/origami-flag-check-does-not-exist.gguf \
    --prompt Hello --n-predict 1 --seed 424242 --temp 0 --no-display-prompt \
    --version >/dev/null 2>&1
printf 'llama.cpp source revision before patch: %s\n' "${before_patch}"
printf 'llama.cpp source revision after patch:  %s\n' "${after_patch}"
printf 'llama.cpp source: %s\nllama.cpp build:  %s\n' "${SOURCE_DIR}" "${BUILD_DIR}"
