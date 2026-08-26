#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR=""
EXPECT_SHA256=""
PREFLIGHT_ONLY=0

usage() {
    cat >&2 <<EOF
usage: $0 [--output DIR] [--expect-sha256 HEX] [--preflight-only] MODEL_DIRECTORY
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output)
            [[ $# -ge 2 ]] || { usage; exit 2; }
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --expect-sha256)
            [[ $# -ge 2 ]] || { usage; exit 2; }
            EXPECT_SHA256="$2"
            shift 2
            ;;
        --preflight-only)
            PREFLIGHT_ONLY=1
            shift
            ;;
        --*)
            usage
            exit 2
            ;;
        *)
            break
            ;;
    esac
done
[[ $# -eq 1 ]] || { usage; exit 2; }
MODEL_DIR="$1"

if [[ ${PREFLIGHT_ONLY} -eq 1 ]]; then
    "${ROOT}/scripts/launch-qwen38.sh" --print-command "${MODEL_DIR}" \
        --single-turn --prompt 'Reply with exactly: ORIGAMI_OK'
    exit 0
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
    OUTPUT_DIR="${ROOT}/artifacts/smoke-$(date -u +%Y%m%dT%H%M%SZ)"
fi
mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" && pwd)"

"${ROOT}/scripts/launch-qwen38.sh" --print-command "${MODEL_DIR}" \
    --single-turn --prompt 'Reply with exactly: ORIGAMI_OK' > "${OUTPUT_DIR}/command.txt"
sysctl -n vm.swapusage > "${OUTPUT_DIR}/swap-before.txt"
vm_stat > "${OUTPUT_DIR}/vm-stat-before.txt"
memory_pressure -Q > "${OUTPUT_DIR}/memory-pressure-before.txt" 2>&1 || true

printf 'sample_s,pid,rss_bytes,vsz_bytes,swap_used_mib,compressor_pages,pageouts\n' > "${OUTPUT_DIR}/telemetry.csv"
SECONDS=0
set +e
"${ROOT}/scripts/launch-qwen38.sh" "${MODEL_DIR}" \
    --single-turn --prompt 'Reply with exactly: ORIGAMI_OK' \
    > "${OUTPUT_DIR}/response.raw.txt" 2> "${OUTPUT_DIR}/backend.log" &
model_pid=$!
set -e
trap 'kill "${model_pid}" 2>/dev/null || true' INT TERM EXIT

while kill -0 "${model_pid}" 2>/dev/null; do
    read -r rss_kib vsz_kib < <(ps -o rss=,vsz= -p "${model_pid}" 2>/dev/null || printf '0 0\n')
    rss_kib="${rss_kib:-0}"
    vsz_kib="${vsz_kib:-0}"
    swap_used="$(sysctl -n vm.swapusage | awk '{for (i=1;i<=NF;i++) if ($i=="used") {gsub(/M/,"",$(i+2)); print $(i+2)}}')"
    compressor="$(vm_stat | awk '/Pages occupied by compressor/ {gsub(/\./,"",$NF); print $NF}')"
    pageouts="$(vm_stat | awk '/Pageouts/ {gsub(/\./,"",$NF); print $NF}')"
    printf '%d,%d,%d,%d,%s,%s,%s\n' \
        "${SECONDS}" "${model_pid}" "$((rss_kib * 1024))" "$((vsz_kib * 1024))" \
        "${swap_used:-0}" "${compressor:-0}" "${pageouts:-0}" >> "${OUTPUT_DIR}/telemetry.csv"
    sleep 1
done

set +e
wait "${model_pid}"
status=$?
set -e
trap - INT TERM EXIT

sysctl -n vm.swapusage > "${OUTPUT_DIR}/swap-after.txt"
vm_stat > "${OUTPUT_DIR}/vm-stat-after.txt"
memory_pressure -Q > "${OUTPUT_DIR}/memory-pressure-after.txt" 2>&1 || true

python3 - "${OUTPUT_DIR}/response.raw.txt" "${OUTPUT_DIR}/response.txt" <<'PY'
import re
import sys
raw, clean = sys.argv[1:]
with open(raw, "rb") as f:
    text = f.read().decode("utf-8", "replace")
text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text).replace("\r", "")
with open(clean, "w", encoding="utf-8") as f:
    f.write(text)
PY
response_sha256="$(shasum -a 256 "${OUTPUT_DIR}/response.txt" | awk '{print $1}')"
printf '%s  response.txt\n' "${response_sha256}" > "${OUTPUT_DIR}/response.sha256"

python3 - "${OUTPUT_DIR}" "${status}" "${response_sha256}" <<'PY'
import json
import os
import platform
import subprocess
import sys

out, status, digest = sys.argv[1:]
def command(*args):
    return subprocess.check_output(args, text=True).strip()
report = {
    "exit_status": int(status),
    "response_sha256": digest,
    "hardware_model": command("sysctl", "-n", "hw.model"),
    "physical_memory_bytes": int(command("sysctl", "-n", "hw.memsize")),
    "machine": platform.machine(),
    "macos": command("sw_vers", "-productVersion"),
    "telemetry_scope": "process RSS/VSZ sampled once per second; system-wide swap, compressor pages, and pageouts",
}
with open(os.path.join(out, "report.json"), "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, sort_keys=True)
    f.write("\n")
PY

if [[ -n "${EXPECT_SHA256}" && "${response_sha256}" != "${EXPECT_SHA256}" ]]; then
    echo "error: response SHA-256 ${response_sha256}; expected ${EXPECT_SHA256}" >&2
    exit 1
fi
if [[ ${status} -ne 0 ]]; then
    echo "error: llama-cli exited ${status}; artifacts: ${OUTPUT_DIR}" >&2
    exit "${status}"
fi
printf 'smoke artifacts: %s\nresponse SHA-256: %s\n' "${OUTPUT_DIR}" "${response_sha256}"
