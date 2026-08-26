"""Bounded GGUF metadata inspection and allocation accounting.

This module intentionally uses positional reads instead of mmap.  Tensor bytes are
only touched by ``probe_tensor_spans``, which reads a caller-bounded number of
bytes at the two ends of selected spans.
"""

from __future__ import annotations

import os
import re
import struct
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


class GGUFError(ValueError):
    """The artifact cannot be inspected safely."""


@dataclass(frozen=True)
class QuantType:
    code: int
    name: str
    block_elements: int
    block_bytes: int


# The smallest set needed by the current UD-IQ1_S artifact. Unknown types are
# rejected rather than estimated. Values match ggml's GGMLType enum and structs.
QUANT_TYPES: Dict[int, QuantType] = {
    0: QuantType(0, "F32", 1, 4),
    8: QuantType(8, "Q8_0", 32, 34),
    12: QuantType(12, "Q4_K", 256, 144),
    13: QuantType(13, "Q5_K", 256, 176),
    14: QuantType(14, "Q6_K", 256, 210),
    16: QuantType(16, "IQ2_XXS", 256, 66),
    19: QuantType(19, "IQ1_S", 256, 50),
    20: QuantType(20, "IQ4_NL", 32, 18),
    30: QuantType(30, "BF16", 1, 2),
}

_VALUE_FORMATS: Dict[int, str] = {
    0: "B",   # UINT8
    1: "b",   # INT8
    2: "H",   # UINT16
    3: "h",   # INT16
    4: "I",   # UINT32
    5: "i",   # INT32
    6: "f",   # FLOAT32
    7: "B",   # BOOL, checked separately
    10: "Q",  # UINT64
    11: "q",  # INT64
    12: "d",  # FLOAT64
}

_KEPT_METADATA = {
    "general.alignment",
    "general.architecture",
    "general.file_type",
    "general.name",
    "general.quantization_version",
    "general.type",
    "split.count",
    "split.no",
    "split.tensors.count",
    "qwen4exp.block_count",
    "qwen4exp.embedding_length",
    "qwen4exp.embedding_length_per_layer_input",
    "qwen4exp.expert_count",
    "qwen4exp.expert_feed_forward_length",
    "qwen4exp.expert_shared_feed_forward_length",
    "qwen4exp.expert_used_count",
    "qwen4exp.ple.head_vocab_sizes",
    "qwen4exp.ple.layers",
}

_ROUTED_RE = re.compile(r"^blk\.(\d+)\.ffn_(down|gate|up)_exps\.weight$")
_SIZE_RE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*(B|KB|MB|GB|TB|KiB|MiB|GiB|TiB)?\s*$", re.ASCII)


class _BoundedReader:
    """Sequential exact reads with a hard byte ceiling and no read-ahead."""

    def __init__(self, path: Path, limit: int):
        self.path = path
        self.limit = limit
        self.offset = 0
        self.fd = os.open(str(path), os.O_RDONLY)

    def close(self) -> None:
        os.close(self.fd)

    def read(self, size: int) -> bytes:
        if size < 0 or self.offset + size > self.limit:
            raise GGUFError(
                "%s: metadata exceeds the %d-byte read limit" % (self.path, self.limit)
            )
        chunks: List[bytes] = []
        left = size
        while left:
            chunk = os.pread(self.fd, left, self.offset)
            if not chunk:
                raise GGUFError(
                    "%s: truncated metadata at byte %d (wanted %d more bytes)"
                    % (self.path, self.offset, left)
                )
            chunks.append(chunk)
            got = len(chunk)
            self.offset += got
            left -= got
        return b"".join(chunks)

    def unpack(self, fmt: str) -> Tuple[Any, ...]:
        return struct.unpack("<" + fmt, self.read(struct.calcsize("<" + fmt)))

    def string(self, max_string_bytes: int) -> str:
        (length,) = self.unpack("Q")
        if length > max_string_bytes:
            raise GGUFError(
                "%s: string length %d exceeds limit %d at byte %d"
                % (self.path, length, max_string_bytes, self.offset - 8)
            )
        raw = self.read(length)
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise GGUFError("%s: invalid UTF-8 metadata: %s" % (self.path, exc))


@dataclass
class ParsedTensor:
    name: str
    dimensions: Tuple[int, ...]
    type_code: int
    relative_offset: int


@dataclass
class ParsedShard:
    path: Path
    file_size: int
    version: int
    metadata: Dict[str, Any]
    tensors: List[ParsedTensor]
    metadata_end: int
    data_offset: int
    metadata_bytes_read: int


def _read_value(
    reader: _BoundedReader,
    value_type: int,
    keep: bool,
    max_string_bytes: int,
    max_array_elements: int,
) -> Any:
    if value_type in _VALUE_FORMATS:
        (value,) = reader.unpack(_VALUE_FORMATS[value_type])
        if value_type == 7:
            if value not in (0, 1):
                raise GGUFError("%s: invalid boolean value %d" % (reader.path, value))
            value = bool(value)
        return value if keep else None
    if value_type == 8:  # STRING
        value = reader.string(max_string_bytes)
        return value if keep else None
    if value_type == 9:  # ARRAY
        element_type, count = reader.unpack("IQ")
        if element_type == 9:
            raise GGUFError("%s: nested GGUF arrays are unsupported" % reader.path)
        if count > max_array_elements:
            raise GGUFError(
                "%s: array element count %d exceeds limit %d"
                % (reader.path, count, max_array_elements)
            )
        if element_type not in _VALUE_FORMATS and element_type != 8:
            raise GGUFError(
                "%s: unsupported GGUF array element type %d" % (reader.path, element_type)
            )
        values = [] if keep else None
        if element_type in _VALUE_FORMATS:
            fmt = _VALUE_FORMATS[element_type]
            item_size = struct.calcsize("<" + fmt)
            raw = reader.read(count * item_size)
            if keep:
                unpacked = struct.unpack("<" + (fmt * count), raw) if count else ()
                if element_type == 7:
                    if any(value not in (0, 1) for value in unpacked):
                        raise GGUFError("%s: invalid boolean in array" % reader.path)
                    values = [bool(value) for value in unpacked]
                else:
                    values = list(unpacked)
        else:
            for _ in range(count):
                value = reader.string(max_string_bytes)
                if keep:
                    values.append(value)
        return values
    raise GGUFError("%s: unsupported GGUF metadata type %d" % (reader.path, value_type))


def parse_shard(
    path: os.PathLike,
    *,
    max_metadata_bytes: int = 64 * 1024 * 1024,
    max_tensors: int = 1_000_000,
    max_kv: int = 100_000,
    max_dimensions: int = 4,
    max_string_bytes: int = 16 * 1024 * 1024,
    max_array_elements: int = 2_000_000,
) -> ParsedShard:
    """Parse one GGUF v3 shard without reading or mapping its tensor body."""

    shard_path = Path(path).resolve()
    try:
        file_size = shard_path.stat().st_size
    except OSError as exc:
        raise GGUFError("cannot stat %s: %s" % (shard_path, exc))
    reader = _BoundedReader(shard_path, max_metadata_bytes)
    try:
        if reader.read(4) != b"GGUF":
            raise GGUFError("%s: expected little-endian GGUF magic" % shard_path)
        version, tensor_count, kv_count = reader.unpack("IQQ")
        if version != 3:
            raise GGUFError("%s: unsupported GGUF version %d" % (shard_path, version))
        if tensor_count > max_tensors:
            raise GGUFError("%s: tensor count %d exceeds limit" % (shard_path, tensor_count))
        if kv_count > max_kv:
            raise GGUFError("%s: metadata count %d exceeds limit" % (shard_path, kv_count))

        metadata: Dict[str, Any] = {}
        seen_keys = set()
        for _ in range(kv_count):
            key = reader.string(max_string_bytes)
            if key in seen_keys:
                raise GGUFError("%s: duplicate metadata key %r" % (shard_path, key))
            seen_keys.add(key)
            (value_type,) = reader.unpack("I")
            keep = key in _KEPT_METADATA
            value = _read_value(
                reader, value_type, keep, max_string_bytes, max_array_elements
            )
            if keep:
                metadata[key] = value

        tensors: List[ParsedTensor] = []
        seen_names = set()
        for _ in range(tensor_count):
            name = reader.string(max_string_bytes)
            if not name or name in seen_names:
                reason = "empty" if not name else "duplicate"
                raise GGUFError("%s: %s tensor name %r" % (shard_path, reason, name))
            seen_names.add(name)
            (dimension_count,) = reader.unpack("I")
            if dimension_count < 1 or dimension_count > max_dimensions:
                raise GGUFError(
                    "%s: tensor %s has unsupported dimension count %d"
                    % (shard_path, name, dimension_count)
                )
            dimensions = reader.unpack("Q" * dimension_count)
            if any(value == 0 for value in dimensions):
                raise GGUFError("%s: tensor %s has a zero dimension" % (shard_path, name))
            type_code, relative_offset = reader.unpack("IQ")
            tensors.append(
                ParsedTensor(name, tuple(dimensions), type_code, relative_offset)
            )

        metadata_end = reader.offset
        alignment = metadata.get("general.alignment", 32)
        if not isinstance(alignment, int) or alignment < 1 or alignment > 4096:
            raise GGUFError("%s: invalid general.alignment %r" % (shard_path, alignment))
        if alignment & (alignment - 1):
            raise GGUFError("%s: alignment must be a power of two" % shard_path)
        data_offset = (metadata_end + alignment - 1) & ~(alignment - 1)
        return ParsedShard(
            shard_path,
            file_size,
            version,
            metadata,
            tensors,
            metadata_end,
            data_offset,
            metadata_end,
        )
    finally:
        reader.close()


def tensor_nbytes(dimensions: Sequence[int], type_code: int, name: str = "tensor") -> int:
    """Return ggml's contiguous byte size, rejecting partial quantization blocks."""

    quant = QUANT_TYPES.get(type_code)
    if quant is None:
        raise GGUFError("%s uses unsupported GGML type %d" % (name, type_code))
    if not dimensions or any(not isinstance(value, int) or value <= 0 for value in dimensions):
        raise GGUFError("%s has invalid dimensions %r" % (name, dimensions))
    if dimensions[0] % quant.block_elements:
        raise GGUFError(
            "%s: first dimension %d is not divisible by %s block size %d"
            % (name, dimensions[0], quant.name, quant.block_elements)
        )
    row_bytes = (dimensions[0] // quant.block_elements) * quant.block_bytes
    size = row_bytes
    for dimension in dimensions[1:]:
        size *= dimension
        if size > (1 << 63) - 1:
            raise GGUFError("%s byte size overflows signed 64-bit range" % name)
    return size


def _resolve_paths(paths: Sequence[os.PathLike]) -> List[Path]:
    resolved: List[Path] = []
    for item in paths:
        path = Path(item)
        if path.is_dir():
            resolved.extend(sorted(path.glob("*.gguf")))
        else:
            resolved.append(path)
    unique: List[Path] = []
    seen = set()
    for path in resolved:
        canonical = path.resolve()
        if canonical not in seen:
            seen.add(canonical)
            unique.append(canonical)
    if not unique:
        raise GGUFError("no GGUF files found")
    return unique


def parse_size(value: str) -> int:
    """Normalize a byte quantity. GB is decimal; GiB is binary."""

    match = _SIZE_RE.match(value)
    if not match:
        raise GGUFError("invalid byte quantity %r (for example: 4GB or 4GiB)" % value)
    number_text, unit = match.groups()
    unit = unit or "B"
    powers = {
        "B": 1,
        "KB": 1000,
        "MB": 1000 ** 2,
        "GB": 1000 ** 3,
        "TB": 1000 ** 4,
        "KiB": 1024,
        "MiB": 1024 ** 2,
        "GiB": 1024 ** 3,
        "TiB": 1024 ** 4,
    }
    try:
        byte_value = Decimal(number_text) * powers[unit]
    except (InvalidOperation, KeyError):
        raise GGUFError("invalid byte quantity %r" % value)
    if byte_value != byte_value.to_integral_value():
        raise GGUFError("byte quantity %r is not an integral number of bytes" % value)
    result = int(byte_value)
    if result > (1 << 63) - 1:
        raise GGUFError("byte quantity %r exceeds signed 64-bit range" % value)
    return result


def format_size(size: int) -> str:
    return "%.3f GB (%.3f GiB)" % (size / 1_000_000_000, size / (1024 ** 3))


def _classify(name: str) -> Tuple[str, Optional[Tuple[int, str]]]:
    routed = _ROUTED_RE.match(name)
    if routed:
        return "routed", (int(routed.group(1)), routed.group(2))
    if "_exps" in name:
        raise GGUFError("unsupported routed tensor family %s" % name)
    if name == "per_layer_token_embd.weight":
        return "ple", None
    if "shexp" in name:
        return "shared", None
    return "dense", None


def inspect_artifact(
    paths: Sequence[os.PathLike],
    *,
    expert_cache_bytes: int = 0,
    ple_cache_bytes: int = 0,
    temporary_bytes: int = 0,
    include_slices: bool = True,
    max_metadata_bytes: int = 64 * 1024 * 1024,
) -> Dict[str, Any]:
    """Inspect GGUF shards and return a JSON-serializable allocation ledger."""

    parsed = [
        parse_shard(path, max_metadata_bytes=max_metadata_bytes)
        for path in _resolve_paths(paths)
    ]

    split_flags = ["split.count" in shard.metadata for shard in parsed]
    if any(split_flags) and not all(split_flags):
        raise GGUFError("cannot mix split and non-split GGUF files")

    shard_numbers: Dict[int, ParsedShard] = {}
    if all(split_flags):
        split_counts = {shard.metadata.get("split.count") for shard in parsed}
        declared_counts = {shard.metadata.get("split.tensors.count") for shard in parsed}
        if len(split_counts) != 1 or len(declared_counts) != 1:
            raise GGUFError("split shards disagree on split.count or split.tensors.count")
        split_count = split_counts.pop()
        declared_tensor_count = declared_counts.pop()
        if not isinstance(split_count, int) or split_count < 1:
            raise GGUFError("invalid split.count %r" % split_count)
        if not isinstance(declared_tensor_count, int) or declared_tensor_count < 0:
            raise GGUFError("invalid split.tensors.count %r" % declared_tensor_count)
        for shard in parsed:
            number = shard.metadata.get("split.no")
            if not isinstance(number, int) or number < 0 or number >= split_count:
                raise GGUFError("%s: invalid split.no %r" % (shard.path, number))
            if number in shard_numbers:
                raise GGUFError("duplicate split.no %d" % number)
            shard_numbers[number] = shard
    else:
        if len(parsed) != 1:
            raise GGUFError("multiple non-split GGUF files are unsupported")
        split_count = 1
        declared_tensor_count = len(parsed[0].tensors)
        shard_numbers[0] = parsed[0]

    missing_shards = sorted(set(range(split_count)) - set(shard_numbers))
    complete = not missing_shards
    parsed_tensor_count = sum(len(shard.tensors) for shard in parsed)
    if parsed_tensor_count > declared_tensor_count:
        raise GGUFError("parsed more tensors than split.tensors.count declares")
    if complete and parsed_tensor_count != declared_tensor_count:
        raise GGUFError(
            "complete split declares %d tensors but contains %d"
            % (declared_tensor_count, parsed_tensor_count)
        )

    global_metadata: Dict[str, Any] = {}
    for shard in parsed:
        for key, value in shard.metadata.items():
            if key.startswith("split."):
                continue
            if key in global_metadata and global_metadata[key] != value:
                raise GGUFError("shards disagree on metadata key %s" % key)
            global_metadata[key] = value

    alignment = global_metadata.get("general.alignment", 32)
    if not isinstance(alignment, int) or alignment < 1 or alignment > 4096 or alignment & (alignment - 1):
        raise GGUFError("invalid aggregate GGUF alignment %r" % alignment)

    expert_count = global_metadata.get("qwen4exp.expert_count")
    embedding_length = global_metadata.get("qwen4exp.embedding_length")
    expert_ff_length = global_metadata.get("qwen4exp.expert_feed_forward_length")
    block_count = global_metadata.get("qwen4exp.block_count")

    tensors_json: List[Dict[str, Any]] = []
    slices: List[Dict[str, Any]] = []
    totals = {"dense": 0, "shared": 0, "routed": 0, "ple": 0}
    all_names = set()
    routed_inventory = set()

    shard_reports: List[Dict[str, Any]] = []
    for shard_number, shard in sorted(shard_numbers.items()):
        # Split tensor shards commonly omit general.alignment. Apply the
        # aggregate value from the metadata shard before resolving offsets.
        data_offset = (shard.metadata_end + alignment - 1) & ~(alignment - 1)
        spans: List[Tuple[int, int, str]] = []
        expected_min_size = data_offset
        available_count = 0
        for tensor in shard.tensors:
            if tensor.name in all_names:
                raise GGUFError("duplicate tensor across shards: %s" % tensor.name)
            all_names.add(tensor.name)
            if tensor.relative_offset % alignment:
                raise GGUFError(
                    "%s: tensor %s offset %d is not %d-byte aligned"
                    % (shard.path, tensor.name, tensor.relative_offset, alignment)
                )
            byte_length = tensor_nbytes(tensor.dimensions, tensor.type_code, tensor.name)
            start = data_offset + tensor.relative_offset
            end = start + byte_length
            if end > (1 << 63) - 1:
                raise GGUFError("tensor %s span overflows signed 64-bit range" % tensor.name)
            spans.append((start, end, tensor.name))
            expected_min_size = max(expected_min_size, end)
            available = end <= shard.file_size
            available_count += int(available)
            category, routed_key = _classify(tensor.name)
            totals[category] += byte_length
            quant = QUANT_TYPES[tensor.type_code]
            record = {
                "name": tensor.name,
                "classification": category,
                "dimensions": list(tensor.dimensions),
                "type": quant.name,
                "type_code": tensor.type_code,
                "shard": shard_number,
                "relative_offset": tensor.relative_offset,
                "offset": start,
                "length": byte_length,
                "end": end,
                "available": available,
            }
            tensors_json.append(record)

            if routed_key is not None:
                layer, projection = routed_key
                if not isinstance(expert_count, int) or expert_count <= 0:
                    raise GGUFError("%s: routed layout requires qwen4exp.expert_count" % tensor.name)
                if len(tensor.dimensions) != 3 or tensor.dimensions[2] != expert_count:
                    raise GGUFError(
                        "%s: only [input, output, expert] routed layout is supported"
                        % tensor.name
                    )
                if isinstance(block_count, int) and (layer < 0 or layer >= block_count):
                    raise GGUFError("%s: layer is outside qwen4exp.block_count" % tensor.name)
                if isinstance(embedding_length, int) and isinstance(expert_ff_length, int):
                    expected_dims = (
                        (expert_ff_length, embedding_length, expert_count)
                        if projection == "down"
                        else (embedding_length, expert_ff_length, expert_count)
                    )
                    if tensor.dimensions != expected_dims:
                        raise GGUFError(
                            "%s: dimensions %r do not match expected current layout %r"
                            % (tensor.name, tensor.dimensions, expected_dims)
                        )
                if byte_length % expert_count:
                    raise GGUFError("%s: bytes do not divide evenly by expert" % tensor.name)
                slice_length = byte_length // expert_count
                routed_inventory.add((layer, projection))
                if include_slices:
                    for expert in range(expert_count):
                        slice_start = start + expert * slice_length
                        slices.append(
                            {
                                "layer": layer,
                                "expert": expert,
                                "projection": projection,
                                "tensor": tensor.name,
                                "shard": shard_number,
                                "offset": slice_start,
                                "length": slice_length,
                                "end": slice_start + slice_length,
                                "available": slice_start + slice_length <= shard.file_size,
                            }
                        )

        for previous, current in zip(sorted(spans), sorted(spans)[1:]):
            if current[0] < previous[1]:
                raise GGUFError(
                    "%s: tensor spans overlap (%s and %s)"
                    % (shard.path, previous[2], current[2])
                )
        shard_reports.append(
            {
                "number": shard_number,
                "path": str(shard.path),
                "file_size": shard.file_size,
                "metadata_end": shard.metadata_end,
                "data_offset": data_offset,
                "metadata_bytes_read": shard.metadata_bytes_read,
                "tensor_count": len(shard.tensors),
                "available_tensor_count": available_count,
                "expected_min_size": expected_min_size,
                "body_complete": expected_min_size <= shard.file_size,
            }
        )

    if complete and global_metadata.get("general.architecture") == "qwen4exp":
        if isinstance(block_count, int) and block_count > 0:
            expected_routed = {
                (layer, projection)
                for layer in range(block_count)
                for projection in ("down", "gate", "up")
            }
            if routed_inventory != expected_routed:
                missing = sorted(expected_routed - routed_inventory)[:5]
                extra = sorted(routed_inventory - expected_routed)[:5]
                raise GGUFError(
                    "unsupported qwen4exp routed inventory; missing=%r extra=%r"
                    % (missing, extra)
                )
        if "qwen4exp.ple.layers" in global_metadata and not any(
            item["classification"] == "ple" for item in tensors_json
        ):
            raise GGUFError("qwen4exp declares PLE layers but has no supported PLE tensor")

    logical_tensor_bytes = sum(totals.values())
    resident = totals["dense"] + totals["shared"]
    streamed = totals["routed"] + totals["ple"]
    cache = expert_cache_bytes + ple_cache_bytes
    runtime_accounted = resident + cache + temporary_bytes
    allocations = [
        {"name": "dense_weights", "kind": "resident", "bytes": totals["dense"]},
        {"name": "shared_experts", "kind": "resident", "bytes": totals["shared"]},
        {"name": "routed_experts", "kind": "streamed", "bytes": totals["routed"]},
        {"name": "ple", "kind": "streamed", "bytes": totals["ple"]},
        {"name": "expert_cache", "kind": "cache", "bytes": expert_cache_bytes},
        {"name": "ple_cache", "kind": "cache", "bytes": ple_cache_bytes},
        {"name": "temporary", "kind": "temporary", "bytes": temporary_bytes},
    ]

    return {
        "schema_version": 1,
        "format": "GGUF",
        "gguf_version": 3,
        "complete": complete,
        "split": {
            "count": split_count,
            "present": sorted(shard_numbers),
            "missing": missing_shards,
            "declared_tensor_count": declared_tensor_count,
            "parsed_tensor_count": parsed_tensor_count,
        },
        "metadata": global_metadata,
        "shards": shard_reports,
        "inventory": {
            "tensor_count": parsed_tensor_count,
            "expert_slice_count": len(slices) if include_slices else None,
            "bytes": dict(totals, total=logical_tensor_bytes),
        },
        "ledger": {
            "allocations": allocations,
            "resident_weight_bytes": resident,
            "streamed_weight_bytes": streamed,
            "cache_budget_bytes": cache,
            "temporary_budget_bytes": temporary_bytes,
            "runtime_accounted_bytes": runtime_accounted,
            "logical_tensor_bytes": logical_tensor_bytes,
            "scope": "complete artifact" if complete else "present metadata shards only",
        },
        "tensors": tensors_json,
        "expert_slices": slices,
    }


def probe_tensor_spans(
    report: Dict[str, Any], names: Iterable[str], *, edge_bytes: int = 1
) -> List[Dict[str, Any]]:
    """Check selected tensor bounds and pread exactly each span's edge bytes."""

    if edge_bytes < 1 or edge_bytes > 4096:
        raise GGUFError("probe edge_bytes must be between 1 and 4096")
    tensors = {tensor["name"]: tensor for tensor in report["tensors"]}
    shards = {shard["number"]: shard for shard in report["shards"]}
    results = []
    for name in names:
        if name not in tensors:
            raise GGUFError("cannot probe unknown tensor %s" % name)
        tensor = tensors[name]
        shard = shards[tensor["shard"]]
        length = min(edge_bytes, tensor["length"])
        first = tensor["offset"]
        last = tensor["end"] - length
        if tensor["end"] > shard["file_size"]:
            raise GGUFError(
                "%s: span [%d, %d) exceeds shard size %d"
                % (name, tensor["offset"], tensor["end"], shard["file_size"])
            )
        fd = os.open(shard["path"], os.O_RDONLY)
        try:
            first_data = os.pread(fd, length, first)
            last_data = os.pread(fd, length, last)
        finally:
            os.close(fd)
        if len(first_data) != length or len(last_data) != length:
            raise GGUFError("%s: exact bounded probe returned a short read" % name)
        results.append(
            {
                "tensor": name,
                "shard": tensor["shard"],
                "span": [tensor["offset"], tensor["end"]],
                "shard_size": shard["file_size"],
                "reads": [[first, length], [last, length]],
                "ok": True,
            }
        )
    return results
