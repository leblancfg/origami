#!/bin/bash
set -euo pipefail

readonly REVISION="bea3b12daee45876b0129a3602dc8f534ce30bf0"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEPS_ROOT="${ORIGAMI_DEPS_ROOT:-/private/tmp/origami-deps}"
BUILD_DIR="${DEPS_ROOT}/llama.cpp-${REVISION}/build-origami"
BIN="${ORIGAMI_LLAMA_CLI:-${BUILD_DIR}/bin/llama-cli}"
MANIFEST="${ROOT}/config/qwen38-flash-next-ud-iq1_s.json"
PROFILE="first-token"
OUTPUT=""
EXPECTED_OUTPUT_SHA256=""
PREFLIGHT_ONLY=0
VERIFY_HASHES=0
FULL_METAL_OUTPUT=0

usage() {
    echo "usage: $0 [--preflight-only] [--validation] [--verify-shards-sha256] [--full-metal-output] [--expected-output-sha256 HASH] [--output FILE] MODEL_DIRECTORY" >&2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --preflight-only) PREFLIGHT_ONLY=1; shift ;;
        --validation) PROFILE="validation"; shift ;;
        --verify-shards-sha256) VERIFY_HASHES=1; shift ;;
        --full-metal-output) FULL_METAL_OUTPUT=1; shift ;;
        --expected-output-sha256)
            [[ $# -ge 2 ]] || { usage; exit 2; }
            EXPECTED_OUTPUT_SHA256="$2"
            shift 2
            ;;
        --output)
            [[ $# -ge 2 ]] || { usage; exit 2; }
            OUTPUT="$2"
            shift 2
            ;;
        --*) usage; exit 2 ;;
        *) break ;;
    esac
done
[[ $# -eq 1 ]] || { usage; exit 2; }
MODEL_DIR="$1"

[[ -x "${BIN}" ]] || {
    echo "error: pinned llama-cli not found at ${BIN}; run scripts/bootstrap-llama-cpp.sh" >&2
    exit 1
}
[[ -f "${BUILD_DIR}/origami-revision.txt" && "$(<"${BUILD_DIR}/origami-revision.txt")" == "${REVISION}" ]] || {
    echo "error: backend revision marker does not match ${REVISION}" >&2
    exit 1
}
expected_patch="$(shasum -a 256 "${ROOT}/patches/llama.cpp-bea3b12-mmap-prefetch-optout.patch" | awk '{print $1}')"
[[ -f "${BUILD_DIR}/origami-patch.sha256" && "$(<"${BUILD_DIR}/origami-patch.sha256")" == "${expected_patch}" ]] || {
    echo "error: backend patch marker is missing or stale; rebuild the backend" >&2
    exit 1
}

if [[ -z "${OUTPUT}" ]]; then
    if [[ ${PREFLIGHT_ONLY} -eq 1 ]]; then
        OUTPUT="${ROOT}/artifacts/qwen38-preflight.json"
    elif [[ ${FULL_METAL_OUTPUT} -eq 1 ]]; then
        OUTPUT="${ROOT}/artifacts/qwen38-${PROFILE}-full-metal-output.json"
    else
        OUTPUT="${ROOT}/artifacts/qwen38-${PROFILE}.json"
    fi
fi

args=(
    python3 "${ROOT}/tools/origami_validate.py"
    --executable "${BIN}"
    --runtime-revision "${REVISION}"
    --model-manifest "${MANIFEST}"
    --model-root "${MODEL_DIR}"
    --output "${OUTPUT}"
    --profile "${PROFILE}"
)
[[ ${VERIFY_HASHES} -eq 0 ]] || args+=(--verify-shards-sha256)
[[ -z "${EXPECTED_OUTPUT_SHA256}" ]] || args+=(--expected-output-sha256 "${EXPECTED_OUTPUT_SHA256}")
if [[ ${PREFLIGHT_ONLY} -eq 1 ]]; then
    args+=(--preflight-only)
else
    args+=(
        --lazy-mmap-safeguards --
        --offline
        --load-mode mmap
        --gpu-layers all
        --ctx-size 512
        --batch-size 32
        --ubatch-size 32
        --cache-ram 0
        --no-warmup
        --color off
        --simple-io
        --single-turn
        --log-verbosity 4
        --perf
    )
    if [[ ${FULL_METAL_OUTPUT} -eq 0 ]]; then
        args+=(--override-tensor '^output=CPU')
    fi
fi
exec "${args[@]}"
