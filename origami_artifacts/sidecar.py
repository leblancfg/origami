"""Byte-exact, resumable expert sidecars for split GGUF artifacts.

The sidecar stores quantized bytes exactly as they appear in the source GGUF.
It never decodes tensor data and uses bounded positional reads and writes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import struct
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

from .gguf import GGUFError, QUANT_TYPES, inspect_artifact, parse_size

SCHEMA_VERSION = "origami.expert-sidecar.v1"
PACK_MAGIC = b"ORIGXPK\0"
PACK_VERSION = 1
PACK_HEADER_BYTES = 4096
PACK_HEADER = struct.Struct("<8sIIIIQQQ32s")
PROJECTION_ORDER = ("gate", "up", "down")
DEFAULT_ALIGNMENT = 4096
DEFAULT_CHUNK_BYTES = 8 * 1024 * 1024
_HEX_SHA256 = set("0123456789abcdef")


def _checked_alignment(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise GGUFError("sidecar alignment must be an integer")
    if value < PACK_HEADER_BYTES or value > 1024 * 1024 or value & (value - 1):
        raise GGUFError("sidecar alignment must be a power of two from 4096 through 1048576")
    return value


def _checked_chunk_bytes(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 4096 or value > 1024 ** 3:
        raise GGUFError("chunk bytes must be from 4096 through 1073741824")
    return value


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _sha256_file_range(path: Path, length: int, *, chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> str:
    digest = hashlib.sha256()
    fd = os.open(str(path), os.O_RDONLY)
    try:
        offset = 0
        while offset < length:
            wanted = min(chunk_bytes, length - offset)
            data = os.pread(fd, wanted, offset)
            if len(data) != wanted:
                raise GGUFError("%s: short read while hashing [%d, %d)" % (path, offset, offset + wanted))
            digest.update(data)
            offset += wanted
    finally:
        os.close(fd)
    return digest.hexdigest()


def _sha256_file(path: Path, *, chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> str:
    return _sha256_file_range(path, path.stat().st_size, chunk_bytes=chunk_bytes)


def _load_manifest(path: os.PathLike, revision: str) -> Dict[str, Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, ValueError) as exc:
        raise GGUFError("cannot read source manifest %s: %s" % (path, exc))
    model = manifest.get("model")
    if not isinstance(model, dict) or model.get("revision") != revision:
        raise GGUFError("source manifest revision does not match %s" % revision)
    entries = manifest.get("shards")
    if not isinstance(entries, list) or not entries:
        raise GGUFError("source manifest has no shards")
    result: Dict[str, Dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise GGUFError("source manifest shard entry is not an object")
        name = entry.get("path")
        size = entry.get("size_bytes")
        digest = entry.get("sha256")
        if not isinstance(name, str) or not name or Path(name).name != name:
            raise GGUFError("source manifest shard path must be a basename")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise GGUFError("source manifest shard %s has invalid size" % name)
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in _HEX_SHA256 for c in digest):
            raise GGUFError("source manifest shard %s has invalid sha256" % name)
        if name in result:
            raise GGUFError("source manifest repeats shard %s" % name)
        result[name] = {"size_bytes": size, "sha256": digest}
    return result


def _stat_identity(path: Path) -> Dict[str, int]:
    stat = path.stat()
    return {
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _source_digest_material(source: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "revision": source["revision"],
        "identity_mode": source["identity_mode"],
        "shards": [
            {
                "number": shard["number"],
                "name": shard["name"],
                "size_bytes": shard["size_bytes"],
                "header_bytes": shard["header_bytes"],
                "header_sha256": shard["header_sha256"],
                **({"sha256": shard["sha256"]} if "sha256" in shard else {}),
                **({"local_identity": shard["local_identity"]} if source["identity_mode"] == "local-stat+header-sha256" else {}),
            }
            for shard in source["shards"]
        ],
    }


def _plan_digest_material(index: Mapping[str, Any]) -> Dict[str, Any]:
    records = []
    for record in index["records"]:
        projections = []
        for projection in record["projections"]:
            projections.append({key: projection[key] for key in (
                "projection", "tensor", "type", "type_code", "tensor_dimensions",
                "slice_dimensions", "quant_block_elements", "quant_block_bytes",
                "source_shard", "source_tensor_offset", "source_tensor_relative_offset",
                "source_slice_relative_offset", "source_offset", "sidecar_offset", "length",
            )})
        records.append({
            "layer": record["layer"],
            "expert": record["expert"],
            "offset": record["offset"],
            "read_length": record["read_length"],
            "logical_bytes": record["logical_bytes"],
            "projections": projections,
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "alignment": index["alignment"],
        "source": _source_digest_material(index["source"]),
        "record_count": index["record_count"],
        "projection_count": index["projection_count"],
        "logical_bytes": index["logical_bytes"],
        "packed_bytes": index["packed_bytes"],
        "records": records,
    }


def _plan_digest(index: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(_plan_digest_material(index))).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.tmp.%d" % (path.name, os.getpid()))
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(str(temporary), flags, 0o644)
    try:
        payload = json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("short write")
            view = view[written:]
        os.fsync(fd)
    except BaseException:
        os.close(fd)
        try:
            temporary.unlink()
        except OSError:
            pass
        raise
    else:
        os.close(fd)
    os.replace(str(temporary), str(path))
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def plan_sidecar(
    paths: Sequence[os.PathLike],
    *,
    source_revision: str,
    source_manifest: Optional[os.PathLike] = None,
    allow_stat_identity: bool = False,
    verify_source_sha256: bool = False,
    alignment: int = DEFAULT_ALIGNMENT,
    max_metadata_bytes: int = 64 * 1024 * 1024,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
) -> Dict[str, Any]:
    """Build an index without reading tensor bodies unless full hashes are requested."""

    alignment = _checked_alignment(alignment)
    chunk_bytes = _checked_chunk_bytes(chunk_bytes)
    if not isinstance(source_revision, str) or not source_revision.strip():
        raise GGUFError("source revision is required")
    if source_manifest is None and not allow_stat_identity:
        raise GGUFError("a source manifest is required unless local stat identity is explicitly allowed")
    if verify_source_sha256 and source_manifest is None:
        raise GGUFError("full source SHA-256 verification requires a source manifest")

    report = inspect_artifact(paths, include_slices=True, max_metadata_bytes=max_metadata_bytes)
    if not report["complete"]:
        raise GGUFError("sidecar planning requires every declared GGUF shard")
    routed_tensors = {
        tensor["name"]: tensor for tensor in report["tensors"] if tensor["classification"] == "routed"
    }
    if not routed_tensors or not report["expert_slices"]:
        raise GGUFError("artifact contains no supported routed expert slices")
    if any(not item["available"] for item in report["expert_slices"]):
        raise GGUFError("one or more routed expert slices exceed their source shard")

    manifest = _load_manifest(source_manifest, source_revision) if source_manifest is not None else None
    source_shards = []
    manifest_names = set(manifest or {})
    for shard in sorted(report["shards"], key=lambda item: item["number"]):
        path = Path(shard["path"]).resolve()
        name = path.name
        entry = manifest.get(name) if manifest is not None else None
        if manifest is not None and entry is None:
            raise GGUFError("source manifest has no entry for %s" % name)
        if entry is not None and entry["size_bytes"] != shard["file_size"]:
            raise GGUFError(
                "%s size %d does not match source manifest size %d"
                % (name, shard["file_size"], entry["size_bytes"])
            )
        header_bytes = shard["data_offset"]
        if header_bytes > shard["file_size"]:
            raise GGUFError("%s metadata/data offset exceeds file size" % name)
        source_entry: Dict[str, Any] = {
            "number": shard["number"],
            "name": name,
            "path": str(path),
            "size_bytes": shard["file_size"],
            "header_bytes": header_bytes,
            "header_sha256": _sha256_file_range(path, header_bytes, chunk_bytes=chunk_bytes),
            "local_identity": _stat_identity(path),
        }
        if entry is not None:
            source_entry["sha256"] = entry["sha256"]
            source_entry["sha256_verified"] = False
        source_shards.append(source_entry)
        manifest_names.discard(name)
    if manifest_names:
        raise GGUFError("source manifest contains shards not present in the artifact: %s" % ", ".join(sorted(manifest_names)))

    if verify_source_sha256:
        for shard in source_shards:
            actual = _sha256_file(Path(shard["path"]), chunk_bytes=chunk_bytes)
            if actual != shard["sha256"]:
                raise GGUFError("%s SHA-256 does not match source manifest" % shard["name"])
            shard["sha256_verified"] = True

    source = {
        "revision": source_revision,
        "identity_mode": "manifest-sha256" if manifest is not None else "local-stat+header-sha256",
        "manifest": str(Path(source_manifest).resolve()) if source_manifest is not None else None,
        "shards": source_shards,
    }

    by_key: Dict[Tuple[int, int], Dict[str, Dict[str, Any]]] = {}
    for item in report["expert_slices"]:
        key = (item["layer"], item["expert"])
        projections = by_key.setdefault(key, {})
        if item["projection"] in projections:
            raise GGUFError("duplicate routed slice for layer %d expert %d projection %s" % (key[0], key[1], item["projection"]))
        projections[item["projection"]] = item

    records: List[Dict[str, Any]] = []
    cursor = PACK_HEADER_BYTES
    logical_bytes = 0
    for layer, expert in sorted(by_key):
        found = by_key[(layer, expert)]
        if set(found) != set(PROJECTION_ORDER):
            raise GGUFError("layer %d expert %d does not have gate/up/down slices" % (layer, expert))
        cursor = _align(cursor, alignment)
        record_offset = cursor
        projections_json = []
        record_logical = 0
        for projection_name in PROJECTION_ORDER:
            item = found[projection_name]
            tensor = routed_tensors[item["tensor"]]
            quant = QUANT_TYPES[tensor["type_code"]]
            cursor = _align(cursor, alignment)
            projection_json = {
                "projection": projection_name,
                "tensor": item["tensor"],
                "type": tensor["type"],
                "type_code": tensor["type_code"],
                "tensor_dimensions": tensor["dimensions"],
                "slice_dimensions": tensor["dimensions"][:-1],
                "quant_block_elements": quant.block_elements,
                "quant_block_bytes": quant.block_bytes,
                "source_shard": item["shard"],
                "source_tensor_offset": tensor["offset"],
                "source_tensor_relative_offset": tensor["relative_offset"],
                "source_slice_relative_offset": item["offset"] - tensor["offset"],
                "source_offset": item["offset"],
                "sidecar_offset": cursor,
                "length": item["length"],
            }
            projections_json.append(projection_json)
            cursor += item["length"]
            record_logical += item["length"]
        cursor = _align(cursor, alignment)
        records.append({
            "layer": layer,
            "expert": expert,
            "offset": record_offset,
            "read_length": cursor - record_offset,
            "logical_bytes": record_logical,
            "projections": projections_json,
        })
        logical_bytes += record_logical

    index: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "mode": "plan",
        "pack_format": {
            "magic_ascii": "ORIGXPK\\0",
            "version": PACK_VERSION,
            "header_bytes": PACK_HEADER_BYTES,
            "projection_order": list(PROJECTION_ORDER),
            "padding_byte": 0,
        },
        "alignment": alignment,
        "source": source,
        "record_count": len(records),
        "projection_count": len(records) * len(PROJECTION_ORDER),
        "logical_bytes": logical_bytes,
        "packed_bytes": cursor,
        "records": records,
        "data": None,
    }
    index["plan_sha256"] = _plan_digest(index)
    return index


def write_index_only(index: Mapping[str, Any], path: os.PathLike) -> Dict[str, Any]:
    """Atomically publish a dry-run index. No pack data is created."""

    output = json.loads(json.dumps(index))
    if output.get("schema_version") != SCHEMA_VERSION or _plan_digest(output) != output.get("plan_sha256"):
        raise GGUFError("sidecar plan digest is invalid")
    output["mode"] = "index-only"
    output["data"] = None
    _atomic_json(Path(path), output)
    return output


def _pack_header(index: Mapping[str, Any]) -> bytes:
    digest = bytes.fromhex(index["plan_sha256"])
    raw = PACK_HEADER.pack(
        PACK_MAGIC,
        PACK_VERSION,
        PACK_HEADER_BYTES,
        index["alignment"],
        0,
        index["record_count"],
        index["logical_bytes"],
        index["packed_bytes"],
        digest,
    )
    return raw + b"\0" * (PACK_HEADER_BYTES - len(raw))


def _parse_pack_header(raw: bytes) -> Dict[str, Any]:
    if len(raw) != PACK_HEADER_BYTES:
        raise GGUFError("sidecar header is truncated")
    values = PACK_HEADER.unpack(raw[:PACK_HEADER.size])
    magic, version, header_bytes, alignment, flags, records, logical, packed, digest = values
    if magic != PACK_MAGIC or version != PACK_VERSION or header_bytes != PACK_HEADER_BYTES or flags != 0:
        raise GGUFError("sidecar binary header is unsupported or malformed")
    if any(raw[PACK_HEADER.size:]):
        raise GGUFError("sidecar binary header reserved bytes are nonzero")
    return {
        "alignment": alignment,
        "record_count": records,
        "logical_bytes": logical,
        "packed_bytes": packed,
        "plan_sha256": digest.hex(),
    }


def _validate_pack_header(raw: bytes, index: Mapping[str, Any]) -> None:
    header = _parse_pack_header(raw)
    expected = {
        "alignment": index["alignment"],
        "record_count": index["record_count"],
        "logical_bytes": index["logical_bytes"],
        "packed_bytes": index["packed_bytes"],
        "plan_sha256": index["plan_sha256"],
    }
    if header != expected:
        raise GGUFError("sidecar binary header does not match its index")


def _revalidate_sources(index: Mapping[str, Any], *, require_local_identity: bool = True, chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> None:
    for shard in index["source"]["shards"]:
        path = Path(shard["path"])
        try:
            identity = _stat_identity(path)
        except OSError as exc:
            raise GGUFError("cannot stat source shard %s: %s" % (path, exc))
        if identity["size_bytes"] != shard["size_bytes"]:
            raise GGUFError("source shard %s changed size" % shard["name"])
        if require_local_identity and identity != shard["local_identity"]:
            raise GGUFError("source shard %s local identity changed" % shard["name"])
        digest = _sha256_file_range(path, shard["header_bytes"], chunk_bytes=chunk_bytes)
        if digest != shard["header_sha256"]:
            raise GGUFError("source shard %s header changed" % shard["name"])


def _pwrite_all(fd: int, data: Union[bytes, bytearray, memoryview], offset: int) -> None:
    view = memoryview(data)
    while view:
        written = os.pwrite(fd, view, offset)
        if written <= 0:
            raise OSError("short positional write")
        offset += written
        view = view[written:]


def _write_zeros(fd: int, start: int, end: int, digest: "hashlib._Hash", chunk_bytes: int) -> None:
    zero = b"\0" * min(chunk_bytes, 1024 * 1024)
    offset = start
    while offset < end:
        data = zero[:min(len(zero), end - offset)]
        _pwrite_all(fd, data, offset)
        digest.update(data)
        offset += len(data)


def _copy_projection(
    source_fd: int,
    destination_fd: int,
    projection: MutableMapping[str, Any],
    packed_digest: "hashlib._Hash",
    chunk_bytes: int,
) -> str:
    digest = hashlib.sha256()
    source_offset = projection["source_offset"]
    destination_offset = projection["sidecar_offset"]
    left = projection["length"]
    while left:
        wanted = min(left, chunk_bytes)
        data = os.pread(source_fd, wanted, source_offset)
        if len(data) != wanted:
            raise GGUFError("%s: short source read at byte %d" % (projection["tensor"], source_offset))
        _pwrite_all(destination_fd, data, destination_offset)
        digest.update(data)
        packed_digest.update(data)
        source_offset += wanted
        destination_offset += wanted
        left -= wanted
    return digest.hexdigest()


def _load_journal(path: Path, record_count: int) -> Tuple[List[Dict[str, Any]], int]:
    entries: List[Dict[str, Any]] = []
    valid_bytes = 0
    if not path.exists():
        return entries, valid_bytes
    with path.open("rb") as handle:
        while True:
            line = handle.readline()
            if not line:
                break
            try:
                entry = json.loads(line)
            except (UnicodeDecodeError, ValueError):
                break
            expected = len(entries)
            if not isinstance(entry, dict):
                break
            hashes = entry.get("projection_sha256")
            if entry.get("record") != expected or not isinstance(hashes, list) or len(hashes) != 3:
                break
            if any(not isinstance(value, str) or len(value) != 64 or any(c not in _HEX_SHA256 for c in value) for value in hashes):
                break
            entries.append(entry)
            valid_bytes += len(line)
            if len(entries) > record_count:
                raise GGUFError("sidecar journal contains too many records")
    return entries, valid_bytes


def _hash_prefix(fd: int, length: int, chunk_bytes: int) -> "hashlib._Hash":
    digest = hashlib.sha256()
    offset = 0
    while offset < length:
        wanted = min(chunk_bytes, length - offset)
        data = os.pread(fd, wanted, offset)
        if len(data) != wanted:
            raise GGUFError("sidecar partial file is shorter than its journal")
        digest.update(data)
        offset += wanted
    return digest


def _relative_data_path(data_path: Path, index_path: Path) -> str:
    return os.path.relpath(str(data_path.resolve()), str(index_path.resolve().parent))


def pack_sidecar(
    index: Mapping[str, Any],
    data_path: os.PathLike,
    index_path: os.PathLike,
    *,
    resume: bool = True,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    checkpoint_records: int = 8,
    _interrupt_after_records: Optional[int] = None,
) -> Dict[str, Any]:
    """Pack routed slices, resuming from a durable record journal when possible."""

    chunk_bytes = _checked_chunk_bytes(chunk_bytes)
    if not isinstance(checkpoint_records, int) or isinstance(checkpoint_records, bool) or checkpoint_records < 1 or checkpoint_records > 4096:
        raise GGUFError("checkpoint_records must be from 1 through 4096")
    plan = json.loads(json.dumps(index))
    if plan.get("schema_version") != SCHEMA_VERSION or _plan_digest(plan) != plan.get("plan_sha256"):
        raise GGUFError("sidecar plan digest is invalid")
    if plan.get("mode") not in ("plan", "index-only"):
        raise GGUFError("pack_sidecar requires a plan or index-only index")

    data_path = Path(data_path).resolve()
    index_path = Path(index_path).resolve()
    if data_path == index_path:
        raise GGUFError("sidecar data and index paths must be different")
    data_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = data_path.with_name(data_path.name + ".partial")
    journal_path = data_path.with_name(data_path.name + ".journal")
    state_path = data_path.with_name(data_path.name + ".state.json")
    lock_path = data_path.with_name(data_path.name + ".lock")
    if index_path.exists():
        raise GGUFError("refusing to replace existing sidecar index %s" % index_path)

    try:
        lock_fd = os.open(str(lock_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        raise GGUFError("sidecar pack is locked by %s" % lock_path)
    os.close(lock_fd)
    source_fds: Dict[int, int] = {}
    data_fd: Optional[int] = None
    journal_fd: Optional[int] = None
    try:
        _revalidate_sources(plan, chunk_bytes=chunk_bytes)
        state = {
            "schema_version": SCHEMA_VERSION,
            "plan_sha256": plan["plan_sha256"],
            "data_path": str(data_path),
        }
        existing_state = None
        if state_path.exists():
            try:
                existing_state = json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise GGUFError("cannot read sidecar resume state: %s" % exc)
            if existing_state != state:
                raise GGUFError("sidecar resume state does not match this plan/output")
        elif any(path.exists() for path in (partial_path, journal_path, data_path)):
            raise GGUFError("sidecar partial artifacts exist without matching resume state")
        else:
            _atomic_json(state_path, state)

        if not resume and any(path.exists() for path in (partial_path, journal_path, data_path)):
            raise GGUFError("partial sidecar exists and resume is disabled")

        entries, journal_valid_bytes = _load_journal(journal_path, plan["record_count"])
        completed = len(entries)
        resume_offset = PACK_HEADER_BYTES if completed == 0 else plan["records"][completed - 1]["offset"] + plan["records"][completed - 1]["read_length"]

        packed_file = data_path if data_path.exists() else partial_path
        if data_path.exists() and completed != plan["record_count"]:
            raise GGUFError("final sidecar data exists but journal is incomplete")
        if not packed_file.exists():
            data_fd = os.open(str(partial_path), os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o644)
            header = _pack_header(plan)
            _pwrite_all(data_fd, header, 0)
            os.fsync(data_fd)
        else:
            data_fd = os.open(str(packed_file), os.O_RDWR)
            header = os.pread(data_fd, PACK_HEADER_BYTES, 0)
            _validate_pack_header(header, plan)
            if os.fstat(data_fd).st_size < resume_offset:
                raise GGUFError("sidecar partial file is shorter than its journal")
            os.ftruncate(data_fd, resume_offset)

        journal_fd = os.open(str(journal_path), os.O_RDWR | os.O_CREAT, 0o644)
        os.ftruncate(journal_fd, journal_valid_bytes)
        os.lseek(journal_fd, journal_valid_bytes, os.SEEK_SET)
        packed_digest = _hash_prefix(data_fd, resume_offset, chunk_bytes)
        source_fds = {
            shard["number"]: os.open(shard["path"], os.O_RDONLY)
            for shard in plan["source"]["shards"]
        }

        pending: List[bytes] = []
        for record_index in range(completed, plan["record_count"]):
            record = plan["records"][record_index]
            current = os.lseek(data_fd, 0, os.SEEK_END)
            if current > record["offset"]:
                raise GGUFError("sidecar record offsets overlap during packing")
            _write_zeros(data_fd, current, record["offset"], packed_digest, chunk_bytes)
            projection_hashes = []
            cursor = record["offset"]
            for projection in record["projections"]:
                _write_zeros(data_fd, cursor, projection["sidecar_offset"], packed_digest, chunk_bytes)
                projection_hashes.append(
                    _copy_projection(
                        source_fds[projection["source_shard"]], data_fd, projection,
                        packed_digest, chunk_bytes,
                    )
                )
                cursor = projection["sidecar_offset"] + projection["length"]
            record_end = record["offset"] + record["read_length"]
            _write_zeros(data_fd, cursor, record_end, packed_digest, chunk_bytes)
            os.lseek(data_fd, record_end, os.SEEK_SET)
            pending.append(_canonical_bytes({"record": record_index, "projection_sha256": projection_hashes}) + b"\n")

            should_commit = len(pending) >= checkpoint_records or record_index + 1 == plan["record_count"]
            if should_commit:
                os.fsync(data_fd)
                for line in pending:
                    view = memoryview(line)
                    while view:
                        written = os.write(journal_fd, view)
                        if written <= 0:
                            raise OSError("short journal write")
                        view = view[written:]
                os.fsync(journal_fd)
                pending = []
            if _interrupt_after_records is not None and record_index + 1 >= _interrupt_after_records:
                raise RuntimeError("test interruption after %d records" % (record_index + 1))

        if os.fstat(data_fd).st_size != plan["packed_bytes"]:
            raise GGUFError("packed sidecar size does not match its plan")
        packed_sha256 = packed_digest.hexdigest()
        _revalidate_sources(plan, chunk_bytes=chunk_bytes)

        os.close(journal_fd)
        journal_fd = None
        all_entries, _ = _load_journal(journal_path, plan["record_count"])
        if len(all_entries) != plan["record_count"]:
            raise GGUFError("sidecar journal did not commit every record")
        for record, entry in zip(plan["records"], all_entries):
            for projection, digest in zip(record["projections"], entry["projection_sha256"]):
                projection["sha256"] = digest

        os.fsync(data_fd)
        os.close(data_fd)
        data_fd = None
        if packed_file == partial_path:
            os.replace(str(partial_path), str(data_path))
            _fsync_directory(data_path.parent)

        plan["mode"] = "packed"
        plan["data"] = {
            "path": _relative_data_path(data_path, index_path),
            "size_bytes": plan["packed_bytes"],
            "sha256": packed_sha256,
        }
        _atomic_json(index_path, plan)
        for cleanup in (journal_path, state_path):
            try:
                cleanup.unlink()
            except FileNotFoundError:
                pass
        _fsync_directory(data_path.parent)
        return plan
    finally:
        if journal_fd is not None:
            os.close(journal_fd)
        if data_fd is not None:
            os.close(data_fd)
        for fd in source_fds.values():
            os.close(fd)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def load_sidecar_index(path: os.PathLike) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            index = json.load(handle)
    except (OSError, ValueError) as exc:
        raise GGUFError("cannot read sidecar index %s: %s" % (path, exc))
    if not isinstance(index, dict) or index.get("schema_version") != SCHEMA_VERSION:
        raise GGUFError("unsupported sidecar index schema")
    if _plan_digest(index) != index.get("plan_sha256"):
        raise GGUFError("sidecar index plan digest is invalid")
    return index


class SidecarReader:
    """Aligned positional-read API for a future bounded expert cache."""

    def __init__(self, index: Union[os.PathLike, Mapping[str, Any]], data_path: Optional[os.PathLike] = None):
        if isinstance(index, Mapping):
            self.index_path: Optional[Path] = None
            self.index = dict(index)
            if self.index.get("schema_version") != SCHEMA_VERSION or _plan_digest(self.index) != self.index.get("plan_sha256"):
                raise GGUFError("sidecar index plan digest is invalid")
        else:
            self.index_path = Path(index).resolve()
            self.index = load_sidecar_index(self.index_path)
        if self.index.get("mode") != "packed" or not isinstance(self.index.get("data"), dict):
            raise GGUFError("sidecar index does not describe packed data")
        if data_path is None:
            if self.index_path is None:
                raise GGUFError("data_path is required when opening an in-memory index")
            data_path = self.index_path.parent / self.index["data"]["path"]
        self.data_path = Path(data_path).resolve()
        try:
            size = self.data_path.stat().st_size
        except OSError as exc:
            raise GGUFError("cannot stat sidecar data %s: %s" % (self.data_path, exc))
        if size != self.index["data"]["size_bytes"]:
            raise GGUFError("sidecar data size does not match index")
        self.fd = os.open(str(self.data_path), os.O_RDONLY)
        try:
            _validate_pack_header(os.pread(self.fd, PACK_HEADER_BYTES, 0), self.index)
        except BaseException:
            os.close(self.fd)
            raise
        self.records = {(record["layer"], record["expert"]): record for record in self.index["records"]}
        if len(self.records) != self.index["record_count"]:
            self.close()
            raise GGUFError("sidecar index repeats a layer/expert key")
        self.max_read_length = max(record["read_length"] for record in self.index["records"])

    def close(self) -> None:
        if getattr(self, "fd", None) is not None:
            os.close(self.fd)
            self.fd = None

    def __enter__(self) -> "SidecarReader":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def record(self, layer: int, expert: int) -> Dict[str, Any]:
        try:
            return self.records[(layer, expert)]
        except KeyError:
            raise GGUFError("sidecar has no layer %d expert %d" % (layer, expert))

    def read_record_into(self, layer: int, expert: int, buffer: bytearray) -> int:
        """Read one aligned gate/up/down record into caller-owned storage."""

        record = self.record(layer, expert)
        length = record["read_length"]
        if len(buffer) < length:
            raise GGUFError("read buffer needs at least %d bytes" % length)
        view = memoryview(buffer)[:length]
        offset = record["offset"]
        done = 0
        while done < length:
            if hasattr(os, "preadv"):
                got = os.preadv(self.fd, [view[done:]], offset + done)
            else:
                data = os.pread(self.fd, length - done, offset + done)
                got = len(data)
                view[done:done + got] = data
            if got <= 0:
                raise GGUFError("short sidecar read for layer %d expert %d" % (layer, expert))
            done += got
        return length

    def read_expert(self, layer: int, expert: int) -> Dict[str, bytes]:
        """Return exact quantized projection bytes; no dequantization is performed."""

        record = self.record(layer, expert)
        buffer = bytearray(record["read_length"])
        self.read_record_into(layer, expert, buffer)
        result = {}
        for projection in record["projections"]:
            relative = projection["sidecar_offset"] - record["offset"]
            result[projection["projection"]] = bytes(buffer[relative:relative + projection["length"]])
        return result


def _source_paths(index: Mapping[str, Any], paths: Optional[Sequence[os.PathLike]]) -> Dict[int, Path]:
    if paths is None:
        mapped = {shard["number"]: Path(shard["path"]) for shard in index["source"]["shards"]}
    else:
        report = inspect_artifact(paths, include_slices=False)
        if not report["complete"]:
            raise GGUFError("verification requires every source shard")
        mapped = {shard["number"]: Path(shard["path"]) for shard in report["shards"]}
    for expected in index["source"]["shards"]:
        path = mapped.get(expected["number"])
        try:
            matches_stat = (
                path is not None
                and path.name == expected["name"]
                and path.stat().st_size == expected["size_bytes"]
            )
        except OSError:
            matches_stat = False
        if not matches_stat:
            raise GGUFError("verification source shard %d does not match index identity" % expected["number"])
        digest = _sha256_file_range(path, expected["header_bytes"])
        if digest != expected["header_sha256"]:
            raise GGUFError("verification source shard %s header does not match index" % expected["name"])
    return mapped


def _verification_indices(count: int, sample_count: int, full: bool) -> List[int]:
    if full:
        return list(range(count))
    if not isinstance(sample_count, int) or isinstance(sample_count, bool) or sample_count < 1:
        raise GGUFError("verification sample count must be positive")
    sample_count = min(sample_count, count)
    if sample_count == 1:
        return [0]
    return sorted({index * (count - 1) // (sample_count - 1) for index in range(sample_count)})


def _compare_projection(source_fd: int, data_fd: int, projection: Mapping[str, Any], chunk_bytes: int) -> int:
    source_digest = hashlib.sha256()
    packed_digest = hashlib.sha256()
    done = 0
    while done < projection["length"]:
        wanted = min(chunk_bytes, projection["length"] - done)
        source = os.pread(source_fd, wanted, projection["source_offset"] + done)
        packed = os.pread(data_fd, wanted, projection["sidecar_offset"] + done)
        if len(source) != wanted or len(packed) != wanted:
            raise GGUFError("short read while verifying %s" % projection["tensor"])
        if source != packed:
            raise GGUFError(
                "byte mismatch for layer projection %s at relative byte %d"
                % (projection["projection"], done)
            )
        source_digest.update(source)
        packed_digest.update(packed)
        done += wanted
    expected = projection.get("sha256")
    if expected is None or source_digest.hexdigest() != expected or packed_digest.hexdigest() != expected:
        raise GGUFError("projection SHA-256 mismatch for %s" % projection["tensor"])
    return done


def verify_sidecar(
    index_path: os.PathLike,
    *,
    source_paths: Optional[Sequence[os.PathLike]] = None,
    full: bool = False,
    sample_count: int = 16,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    verify_source_sha256: bool = False,
) -> Dict[str, Any]:
    """Compare deterministic record samples, or every packed byte projection, to GGUF."""

    chunk_bytes = _checked_chunk_bytes(chunk_bytes)
    index_file = Path(index_path).resolve()
    index = load_sidecar_index(index_file)
    sources = _source_paths(index, source_paths)
    if verify_source_sha256:
        for expected in index["source"]["shards"]:
            if "sha256" not in expected:
                raise GGUFError("index has no declared full SHA-256 for %s" % expected["name"])
            if _sha256_file(sources[expected["number"]], chunk_bytes=chunk_bytes) != expected["sha256"]:
                raise GGUFError("source shard %s SHA-256 mismatch" % expected["name"])
    indices = _verification_indices(index["record_count"], sample_count, full)
    source_fds = {number: os.open(str(path), os.O_RDONLY) for number, path in sources.items()}
    data_path = (index_file.parent / index["data"]["path"]).resolve()
    data_fd = os.open(str(data_path), os.O_RDONLY)
    compared = 0
    projections = 0
    try:
        if os.fstat(data_fd).st_size != index["data"]["size_bytes"]:
            raise GGUFError("sidecar data size does not match index")
        _validate_pack_header(os.pread(data_fd, PACK_HEADER_BYTES, 0), index)
        for record_index in indices:
            record = index["records"][record_index]
            for projection in record["projections"]:
                compared += _compare_projection(source_fds[projection["source_shard"]], data_fd, projection, chunk_bytes)
                projections += 1
    finally:
        os.close(data_fd)
        for fd in source_fds.values():
            os.close(fd)
    data_sha256_verified = False
    if full:
        actual = _sha256_file(data_path, chunk_bytes=chunk_bytes)
        if actual != index["data"]["sha256"]:
            raise GGUFError("full packed-data SHA-256 mismatch")
        data_sha256_verified = True
    return {
        "ok": True,
        "mode": "full" if full else "sample",
        "records_verified": len(indices),
        "projections_verified": projections,
        "projection_bytes_compared": compared,
        "data_sha256_verified": data_sha256_verified,
        "source_sha256_verified": bool(verify_source_sha256),
        "record_indices": indices,
    }


def benchmark_sidecar(
    index_path: os.PathLike,
    *,
    request_count: int = 100,
    warmup: int = 0,
    pattern: str = "sequential",
    seed: int = 0,
) -> Dict[str, Any]:
    """Benchmark aligned record preads with one reusable caller-owned buffer."""

    if not isinstance(request_count, int) or request_count < 1 or request_count > 1_000_000:
        raise GGUFError("request_count must be from 1 through 1000000")
    if not isinstance(warmup, int) or warmup < 0 or warmup > 1_000_000:
        raise GGUFError("warmup must be from 0 through 1000000")
    if pattern not in ("sequential", "random"):
        raise GGUFError("benchmark pattern must be sequential or random")
    with SidecarReader(index_path) as reader:
        records = reader.index["records"]
        rng = random.Random(seed)
        if pattern == "sequential":
            choices = [index % len(records) for index in range(warmup + request_count)]
        else:
            choices = [rng.randrange(len(records)) for _ in range(warmup + request_count)]
        buffer = bytearray(reader.max_read_length)
        for choice in choices[:warmup]:
            record = records[choice]
            reader.read_record_into(record["layer"], record["expert"], buffer)
        latencies = []
        byte_count = 0
        started = time.perf_counter_ns()
        for choice in choices[warmup:]:
            record = records[choice]
            before = time.perf_counter_ns()
            byte_count += reader.read_record_into(record["layer"], record["expert"], buffer)
            latencies.append(time.perf_counter_ns() - before)
        elapsed_ns = time.perf_counter_ns() - started
    ordered = sorted(latencies)
    percentile = lambda fraction: ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]
    return {
        "requests": request_count,
        "warmup_requests": warmup,
        "pattern": pattern,
        "seed": seed,
        "bytes_read": byte_count,
        "elapsed_ns": elapsed_ns,
        "bytes_per_second": byte_count * 1_000_000_000 / elapsed_ns if elapsed_ns else None,
        "requests_per_second": request_count * 1_000_000_000 / elapsed_ns if elapsed_ns else None,
        "latency_ns": {"min": ordered[0], "p50": percentile(0.50), "p95": percentile(0.95), "max": ordered[-1]},
        "read_api": "pread-aligned-record-into-reused-buffer",
    }


def _size_argument(value: str) -> int:
    try:
        return parse_size(value)
    except GGUFError as exc:
        raise argparse.ArgumentTypeError(str(exc))


def _add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("paths", nargs="+", help="GGUF shard(s) or a directory")
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-manifest", help="manifest with exact revision, shard sizes, and SHA-256 values")
    parser.add_argument("--allow-stat-identity", action="store_true", help="allow local inode/mtime/header identity without a hash manifest")
    parser.add_argument("--verify-source-sha256", action="store_true", help="read and hash every source shard")
    parser.add_argument("--alignment", type=_size_argument, default=DEFAULT_ALIGNMENT)
    parser.add_argument("--chunk-size", type=_size_argument, default=DEFAULT_CHUNK_BYTES)
    parser.add_argument("--max-metadata", type=_size_argument, default=64 * 1024 * 1024)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python3 -m origami_artifacts.sidecar")
    commands = parser.add_subparsers(dest="command", required=True)
    index_parser = commands.add_parser("index", help="write an atomic index-only dry run")
    _add_source_arguments(index_parser)
    index_parser.add_argument("--index", required=True)

    pack_parser = commands.add_parser("pack", help="pack or resume a sidecar")
    _add_source_arguments(pack_parser)
    pack_parser.add_argument("--data", required=True)
    pack_parser.add_argument("--index", required=True)
    pack_parser.add_argument("--no-resume", action="store_true")
    pack_parser.add_argument("--checkpoint-records", type=int, default=8)
    verification = pack_parser.add_mutually_exclusive_group()
    verification.add_argument("--verify-samples", type=int, default=0, metavar="N")
    verification.add_argument("--verify-full", action="store_true")

    verify_parser = commands.add_parser("verify", help="verify packed bytes against GGUF")
    verify_parser.add_argument("index")
    verify_parser.add_argument("--source", action="append", default=None, help="relocated source shard or directory")
    verify_parser.add_argument("--samples", type=int, default=16)
    verify_parser.add_argument("--full", action="store_true")
    verify_parser.add_argument("--verify-source-sha256", action="store_true")
    verify_parser.add_argument("--chunk-size", type=_size_argument, default=DEFAULT_CHUNK_BYTES)

    benchmark_parser = commands.add_parser("benchmark", help="benchmark aligned expert-record preads")
    benchmark_parser.add_argument("index")
    benchmark_parser.add_argument("--requests", type=int, default=100)
    benchmark_parser.add_argument("--warmup", type=int, default=0)
    benchmark_parser.add_argument("--pattern", choices=("sequential", "random"), default="sequential")
    benchmark_parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command in ("index", "pack"):
            plan = plan_sidecar(
                args.paths,
                source_revision=args.source_revision,
                source_manifest=args.source_manifest,
                allow_stat_identity=args.allow_stat_identity,
                verify_source_sha256=args.verify_source_sha256,
                alignment=args.alignment,
                max_metadata_bytes=args.max_metadata,
                chunk_bytes=args.chunk_size,
            )
            if args.command == "index":
                result = write_index_only(plan, args.index)
                summary = {
                    "mode": result["mode"], "index": str(Path(args.index).resolve()),
                    "records": result["record_count"], "logical_bytes": result["logical_bytes"],
                    "planned_packed_bytes": result["packed_bytes"], "plan_sha256": result["plan_sha256"],
                }
            else:
                result = pack_sidecar(
                    plan, args.data, args.index, resume=not args.no_resume,
                    chunk_bytes=args.chunk_size, checkpoint_records=args.checkpoint_records,
                )
                summary = {
                    "mode": result["mode"], "data": str(Path(args.data).resolve()),
                    "index": str(Path(args.index).resolve()), "records": result["record_count"],
                    "logical_bytes": result["logical_bytes"], "packed_bytes": result["packed_bytes"],
                    "plan_sha256": result["plan_sha256"], "data_sha256": result["data"]["sha256"],
                }
                if args.verify_full or args.verify_samples:
                    summary["verification"] = verify_sidecar(
                        args.index, full=args.verify_full,
                        sample_count=args.verify_samples or 16,
                        chunk_bytes=args.chunk_size,
                    )
        elif args.command == "verify":
            summary = verify_sidecar(
                args.index, source_paths=args.source, full=args.full,
                sample_count=args.samples, chunk_bytes=args.chunk_size,
                verify_source_sha256=args.verify_source_sha256,
            )
        else:
            summary = benchmark_sidecar(
                args.index, request_count=args.requests, warmup=args.warmup,
                pattern=args.pattern, seed=args.seed,
            )
        sys.stdout.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        return 0
    except (GGUFError, OSError, RuntimeError) as exc:
        parser.exit(2, "error: %s\n" % exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
