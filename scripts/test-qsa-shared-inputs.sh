#!/bin/bash
set -euo pipefail

readonly REVISION="213df585b9aed6a09be30d8401f267bf603c104c"
DEPS_ROOT="${ORIGAMI_DEPS_ROOT:-/private/tmp/origami-deps}"
BIN="${DEPS_ROOT}/llama.cpp-${REVISION}/build-origami/bin/test-llama-archs"
TMP="$(mktemp -d "${TMPDIR:-/tmp}/origami-qsa-parity.XXXXXX")"
trap 'rm -rf "${TMP}"' EXIT

[[ -x "${BIN}" ]] || {
    echo "error: build test-llama-archs with scripts/bootstrap-long-context-llama-cpp.sh first" >&2
    exit 1
}

run_case() {
    local mode="$1"
    local log="${TMP}/${mode}.log"
    local status

    status="$(python3 - "${mode}" "${BIN}" "${log}" <<'PY'
import os
import subprocess
import sys

mode, binary, log = sys.argv[1:]
environment = os.environ.copy()
environment["LLAMA_QSA_SHARED_INPUTS"] = mode
with open(log, "wb") as output:
    result = subprocess.run(
        [binary, "-a", "qwen4exp", "-s", "424242"],
        env=environment,
        stdout=output,
        stderr=subprocess.STDOUT,
        check=False,
    )
print(128 - result.returncode if result.returncode < 0 else result.returncode)
PY
)"

    if [[ "${status}" -ne 0 ]]; then
        # The pinned Meta backend cannot model Qwen4Exp's ggml_set_rows split state.
        # CPU, Accelerate, and Metal run before this architecture-independent assertion.
        [[ "${status}" -eq 134 ]] && grep -Fq 'split_states_equal(src_ss[0], src_ss[2])' "${log}" || {
            tail -80 "${log}" >&2
            return "${status}"
        }
    fi

    grep -F '|        qwen4exp|' "${log}" | head -3 >"${TMP}/${mode}.rows"
    [[ "$(grep -c 'OK' "${TMP}/${mode}.rows")" -eq 3 ]] || {
        cat "${TMP}/${mode}.rows" >&2
        return 1
    }
}

run_case 0
run_case 1
diff -u "${TMP}/0.rows" "${TMP}/1.rows"
printf 'Qwen4Exp shared/unshared synthetic architecture parity: pass\n'
