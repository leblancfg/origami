#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MANIFEST="${ROOT}/config/qwen38-flash-next-ud-iq1_s.json"
CHECK_HASHES=0

usage() {
    echo "usage: $0 [--sha256] MODEL_DIRECTORY" >&2
}

if [[ "${1:-}" == "--sha256" ]]; then
    CHECK_HASHES=1
    shift
fi
[[ $# -eq 1 ]] || { usage; exit 2; }
MODEL_DIR="$1"

python3 - "${MANIFEST}" "${MODEL_DIR}" "${CHECK_HASHES}" <<'PY'
import hashlib
import json
import os
import sys

manifest_path, model_dir, check_hashes = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
with open(manifest_path, encoding="utf-8") as f:
    manifest = json.load(f)

errors = []
for item in manifest["files"]:
    path = os.path.join(model_dir, item["name"])
    if not os.path.isfile(path):
        errors.append(f"missing: {path}")
        continue
    actual_size = os.path.getsize(path)
    if actual_size != item["size"]:
        errors.append(f"incomplete: {path} is {actual_size} bytes; expected {item['size']}")
        continue
    if check_hashes:
        digest = hashlib.sha256()
        with open(path, "rb", buffering=8 * 1024 * 1024) as model_file:
            for block in iter(lambda: model_file.read(8 * 1024 * 1024), b""):
                digest.update(block)
        actual_hash = digest.hexdigest()
        if actual_hash != item["sha256"]:
            errors.append(f"sha256 mismatch: {path}: {actual_hash}")

if errors:
    print("model verification failed:", file=sys.stderr)
    for error in errors:
        print(f"  {error}", file=sys.stderr)
    sys.exit(1)

mode = "sizes and SHA-256" if check_hashes else "sizes"
print(f"verified {mode}: {manifest['repository']}@{manifest['revision']} {manifest['quantization']}")
PY
