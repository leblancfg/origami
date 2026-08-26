#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
args=(--preflight-only)
if [[ "${1:-}" == "--sha256" ]]; then
    args+=(--verify-shards-sha256)
    shift
fi
[[ $# -eq 1 ]] || { echo "usage: $0 [--sha256] MODEL_DIRECTORY" >&2; exit 2; }
exec "${ROOT}/scripts/smoke-test.sh" "${args[@]}" "$1"
