#!/bin/bash
set -euo pipefail

readonly REVISION="213df585b9aed6a09be30d8401f267bf603c104c"
readonly REMOTE="https://github.com/ggml-org/llama.cpp.git"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LONG_CONTEXT_PATCH="${ROOT}/patches/llama.cpp-${REVISION}-long-context.patch"
EXPERT_PATCH="${ROOT}/patches/llama.cpp-${REVISION}-explicit-expert-streaming.patch"
CAPABILITIES="${ROOT}/validation/expert-streaming-capabilities.json"
if [[ -n "${ORIGAMI_DEPS_ROOT:-}" ]]; then
    DEPS_ROOT="${ORIGAMI_DEPS_ROOT}"
else
    DEPS_ROOT="$(mktemp -d /private/tmp/origami-expert-graph-bootstrap.XXXXXX)"
fi
SOURCE_DIR="${DEPS_ROOT}/llama.cpp-${REVISION}-expert-graph"
BUILD_DIR="${DEPS_ROOT}/build-${REVISION}-expert-graph"

for command in git cmake ctest shasum; do
    command -v "${command}" >/dev/null || { echo "error: ${command} is required" >&2; exit 1; }
done
for patch in "${LONG_CONTEXT_PATCH}" "${EXPERT_PATCH}"; do
    [[ -f "${patch}" ]] || { echo "error: patch not found: ${patch}" >&2; exit 1; }
done
[[ -f "${CAPABILITIES}" ]] || { echo "error: capabilities not found: ${CAPABILITIES}" >&2; exit 1; }

mkdir -p "${DEPS_ROOT}"
if [[ ! -d "${SOURCE_DIR}/.git" ]]; then
    git init "${SOURCE_DIR}"
    git -C "${SOURCE_DIR}" remote add origin "${REMOTE}"
fi
[[ "$(git -C "${SOURCE_DIR}" remote get-url origin)" == "${REMOTE}" ]] || {
    echo "error: ${SOURCE_DIR} has an unexpected origin" >&2
    exit 1
}

git -C "${SOURCE_DIR}" fetch --depth=1 origin "${REVISION}"
git -C "${SOURCE_DIR}" checkout --detach "${REVISION}"
git -C "${SOURCE_DIR}" reset --hard "${REVISION}"
git -C "${SOURCE_DIR}" clean -ffd
[[ "$(git -C "${SOURCE_DIR}" rev-parse HEAD)" == "${REVISION}" ]] || {
    echo "error: source revision does not match ${REVISION}" >&2
    exit 1
}

for patch in "${LONG_CONTEXT_PATCH}" "${EXPERT_PATCH}"; do
    git -C "${SOURCE_DIR}" apply --check "${patch}"
    git -C "${SOURCE_DIR}" apply "${patch}"
    git -C "${SOURCE_DIR}" apply --reverse --check "${patch}"
done
git -C "${SOURCE_DIR}" diff --check
expected_changes=$'ggml/src/ggml-backend.cpp\nggml/src/ggml-metal/ggml-metal-device.m\nsrc/CMakeLists.txt\nsrc/llama-context.cpp\nsrc/llama-context.h\nsrc/llama-expert-stream.cpp\nsrc/llama-expert-stream.h\nsrc/llama-graph.cpp\nsrc/llama-graph.h\nsrc/llama-kv-cache.cpp\nsrc/llama-kv-cache.h\nsrc/llama-memory-hybrid-idx.cpp\nsrc/llama-memory-hybrid-idx.h\nsrc/llama-model-loader.cpp\nsrc/llama-model-loader.h\nsrc/llama-model.cpp\nsrc/llama-model.h\nsrc/models/models.h\nsrc/models/qwen4exp.cpp\ntests/CMakeLists.txt\ntests/test-expert-stream.cpp'
actual_changes="$({
    git -C "${SOURCE_DIR}" diff --name-only
    git -C "${SOURCE_DIR}" ls-files --others --exclude-standard
} | LC_ALL=C sort)"
[[ "${actual_changes}" == "${expected_changes}" ]] || {
    echo "error: combined patches changed unexpected files" >&2
    git -C "${SOURCE_DIR}" status --short >&2
    exit 1
}

cmake -S "${SOURCE_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF \
    -DLLAMA_CURL=OFF \
    -DLLAMA_BUILD_APP=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_SERVER=OFF \
    -DLLAMA_BUILD_TESTS=ON \
    -DLLAMA_BUILD_TOOLS=OFF \
    -DLLAMA_BUILD_UI=OFF \
    -DLLAMA_USE_PREBUILT_UI=OFF
jobs="${ORIGAMI_BUILD_JOBS:-$(sysctl -n hw.logicalcpu 2>/dev/null || getconf _NPROCESSORS_ONLN)}"
cmake --build "${BUILD_DIR}" --config Release -j "${jobs}" --target test-expert-stream
ctest --test-dir "${BUILD_DIR}" -R '^test-expert-stream$' --output-on-failure

printf '%s\n' "${REVISION}" > "${BUILD_DIR}/origami-revision.txt"
shasum -a 256 "${LONG_CONTEXT_PATCH}" | awk '{print $1}' > "${BUILD_DIR}/origami-long-context-patch.sha256"
shasum -a 256 "${EXPERT_PATCH}" | awk '{print $1}' > "${BUILD_DIR}/origami-expert-streaming-patch.sha256"
cp "${CAPABILITIES}" "${BUILD_DIR}/origami-expert-streaming-capabilities.json"

printf 'expert-streaming llama.cpp source: %s\nexpert-streaming build: %s\n' "${SOURCE_DIR}" "${BUILD_DIR}"
