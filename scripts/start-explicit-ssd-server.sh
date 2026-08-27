#!/bin/bash
set -euo pipefail

readonly REVISION="213df585b9aed6a09be30d8401f267bf603c104c"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUNTIME_ROOT="${ORIGAMI_EXPERT_RUNTIME_ROOT:-/private/tmp/origami-expert-runtime}"
BUILD_DIR="${RUNTIME_ROOT}/build-${REVISION}-expert-graph"
SERVER="${BUILD_DIR}/bin/llama-server"
PORT="${ORIGAMI_PORT:-18080}"

[[ $# -eq 1 ]] || { echo "usage: $0 MODEL_DIRECTORY" >&2; exit 2; }
MODEL_DIR="$1"
MODEL="${MODEL_DIR}/Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf"

[[ -x "${SERVER}" ]] || {
    echo "error: explicit SSD runtime is absent; run ORIGAMI_BUILD_RUNTIME=1 scripts/bootstrap-expert-streaming-llama-cpp.sh" >&2
    exit 1
}
"${ROOT}/scripts/verify-model.sh" "${MODEL_DIR}" >/dev/null
[[ -f "${BUILD_DIR}/origami-expert-streaming-capabilities.json" ]] || {
    echo "error: expert-streaming capability record is absent" >&2
    exit 1
}

export LLAMA_EXPLICIT_EXPERT_STREAMING=1
export LLAMA_EXPERT_CACHE_SLOTS=10
export LLAMA_QSA_SHARED_INPUTS=1
export GGML_METAL_NO_RESIDENCY=1
export LLAMA_MMAP_PREFETCH=0

exec "${SERVER}" \
    --offline --host 127.0.0.1 --port "${PORT}" \
    --alias origami-qwen38-ssd-262144 --model "${MODEL}" \
    --load-mode mmap --gpu-layers all --override-tensor '^output=CPU' \
    --fit off --no-repack --ctx-size 262144 \
    --batch-size 32 --ubatch-size 1 --parallel 1 \
    --no-kv-unified --kv-offload --cache-type-k q8_0 --cache-type-v q8_0 \
    --flash-attn on --no-context-shift \
    --cache-ram 0 --no-cache-prompt --cache-reuse 0 --ctx-checkpoints 0 \
    --no-cache-idle-slots --no-warmup --jinja --metrics --slots --perf --log-verbosity 4
