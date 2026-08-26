#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
echo "note: launch is recorded by the validation harness; use scripts/smoke-test.sh directly for new automation" >&2
exec "${ROOT}/scripts/smoke-test.sh" "$@"
