#!/bin/bash
set -euo pipefail

readonly REVISION="213df585b9aed6a09be30d8401f267bf603c104c"
readonly REMOTE="https://github.com/ggml-org/llama.cpp.git"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PATCH="${ROOT}/patches/llama.cpp-${REVISION}-long-context.patch"
DEPS_ROOT="${ORIGAMI_DEPS_ROOT:-/private/tmp/origami-deps}"
SOURCE_DIR="${DEPS_ROOT}/llama.cpp-${REVISION}"
BUILD_DIR="${SOURCE_DIR}/build-origami"

for command in git cmake shasum; do
    command -v "${command}" >/dev/null || { echo "error: ${command} is required" >&2; exit 1; }
done
[[ -f "${PATCH}" ]] || { echo "error: patch not found: ${PATCH}" >&2; exit 1; }

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
git -C "${SOURCE_DIR}" clean -ffd -e build-origami/
[[ "$(git -C "${SOURCE_DIR}" rev-parse HEAD)" == "${REVISION}" ]] || {
    echo "error: source revision does not match ${REVISION}" >&2
    exit 1
}

git -C "${SOURCE_DIR}" apply --check "${PATCH}"
git -C "${SOURCE_DIR}" apply "${PATCH}"
git -C "${SOURCE_DIR}" apply --reverse --check "${PATCH}"
git -C "${SOURCE_DIR}" diff --check
expected_changes=$'ggml/src/ggml-metal/ggml-metal-device.m\nsrc/llama-kv-cache.cpp\nsrc/llama-kv-cache.h\nsrc/llama-memory-hybrid-idx.cpp\nsrc/llama-memory-hybrid-idx.h\nsrc/llama-model.cpp\nsrc/models/models.h\nsrc/models/qwen4exp.cpp'
[[ "$(git -C "${SOURCE_DIR}" diff --name-only)" == "${expected_changes}" ]] || {
    echo "error: long-context patch changed unexpected files" >&2
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
    --target llama-cli llama-server test-llama-archs test-backend-ops

printf '%s\n' "${REVISION}" > "${BUILD_DIR}/origami-revision.txt"
shasum -a 256 "${PATCH}" | awk '{print $1}' > "${BUILD_DIR}/origami-patch.sha256"
cat > "${BUILD_DIR}/origami-context-capabilities.json" <<EOF
{
  "schema_version": "origami.backend-capabilities.v1",
  "runtime_revision": "${REVISION}",
  "capabilities": {
    "lazy_mmap_safeguards": "checked-in long-context patch",
    "qwen4exp_indexer_cache_coupled_and_reported": "035e22731a7fd70b9854b3a2d64ec68e9b1a45d3",
    "qwen4exp_qsa_quantized_kv_rotation": "0ac4b18025c2e255dd76252cd3b465683d08b257",
    "qwen4exp_large_graph_node_budget": "c52ed2a0b0b865e82eb1b393106c48df1c39cb32",
    "qwen4exp_f16_indexer_cache_split": "${REVISION} plus $(shasum -a 256 "${PATCH}" | awk '{print $1}')",
    "qwen4exp_shared_qsa_graph_inputs": "${REVISION} plus $(shasum -a 256 "${PATCH}" | awk '{print $1}')"
  }
}
EOF

CLI="${BUILD_DIR}/bin/llama-cli"
SERVER="${BUILD_DIR}/bin/llama-server"
"${CLI}" --version
"${CLI}" --list-devices
server_help="$("${SERVER}" --help 2>&1)"
for flag in '--ctx-size' '--batch-size' '--ubatch-size' '--parallel' '--load-mode' \
            '--gpu-layers' '--override-tensor' '--fit' '--cache-type-k' '--cache-type-v' \
            '--flash-attn' '--kv-offload' '--no-kv-unified' '--no-context-shift' \
            '--cache-ram' '--no-cache-prompt' '--cache-reuse' '--ctx-checkpoints' \
            '--no-cache-idle-slots' '--no-warmup' '--metrics' '--slots' '--perf' \
            '--log-verbosity' '--rope-scaling' '--rope-scale' '--yarn-orig-ctx' '--override-kv'; do
    grep -Fq -- "${flag}" <<< "${server_help}" || {
        echo "error: long-context llama-server help does not contain ${flag}" >&2
        exit 1
    }
done
printf 'long-context llama.cpp source: %s\nlong-context build: %s\n' "${SOURCE_DIR}" "${BUILD_DIR}"
