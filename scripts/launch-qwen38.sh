#!/bin/bash
set -euo pipefail

readonly REVISION="bea3b12daee45876b0129a3602dc8f534ce30bf0"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEPS_ROOT="${ORIGAMI_DEPS_ROOT:-/private/tmp/origami-deps}"
SOURCE_DIR="${DEPS_ROOT}/llama.cpp-${REVISION}"
BUILD_DIR="${SOURCE_DIR}/build-origami"
BIN="${ORIGAMI_LLAMA_CLI:-${BUILD_DIR}/bin/llama-cli}"
REVISION_FILE="${ORIGAMI_LLAMA_REVISION_FILE:-${BUILD_DIR}/origami-revision.txt}"
PRINT_ONLY=0

usage() {
    cat >&2 <<EOF
usage: $0 [--print-command] MODEL_DIRECTORY [llama-cli arguments ...]

Environment overrides:
  ORIGAMI_CONTEXT       context tokens (default: 512)
  ORIGAMI_PREDICT       generated tokens (default: 32)
  ORIGAMI_BATCH         logical batch size (default: 64)
  ORIGAMI_UBATCH        physical batch size (default: 32)
  ORIGAMI_DEPS_ROOT     external dependency root (default: /private/tmp/origami-deps)
EOF
}

if [[ "${1:-}" == "--print-command" ]]; then
    PRINT_ONLY=1
    shift
fi
[[ $# -ge 1 ]] || { usage; exit 2; }
MODEL_DIR="$1"
shift

[[ -x "${BIN}" ]] || {
    echo "error: pinned llama-cli not found at ${BIN}; run scripts/bootstrap-llama-cpp.sh" >&2
    exit 1
}
[[ -f "${REVISION_FILE}" && "$(<"${REVISION_FILE}")" == "${REVISION}" ]] || {
    echo "error: backend revision marker is absent or does not match ${REVISION}" >&2
    exit 1
}
"${ROOT}/scripts/verify-model.sh" "${MODEL_DIR}" >&2

MODEL="${MODEL_DIR}/Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf"
args=(
    "${BIN}"
    --offline
    --model "${MODEL}"
    --load-mode mmap
    --gpu-layers all
    --override-tensor '^per_layer_token_embd$=CPU'
    --ctx-size "${ORIGAMI_CONTEXT:-512}"
    --batch-size "${ORIGAMI_BATCH:-64}"
    --ubatch-size "${ORIGAMI_UBATCH:-32}"
    --predict "${ORIGAMI_PREDICT:-32}"
    --cache-ram 0
    --no-warmup
    --seed 1
    --temp 0
    --color off
    --simple-io
    --no-display-prompt
    --no-show-timings
)
args+=("$@")

if [[ ${PRINT_ONLY} -eq 1 ]]; then
    printf '%q ' "${args[@]}"
    printf '\n'
    exit 0
fi
exec "${args[@]}"
