#!/usr/bin/env python3
"""Read GGUF v3 tensor metadata without mapping or reading tensor bodies.

The placement report models llama.cpp Qwen4Exp with at least one Metal layer:
input tensors stay on CPU; output and block tensors use the Metal buffer type.
llama.cpp creates one mmap-backed Metal buffer envelope per shard and buffer
type, so bytes between the first and last Metal tensor are Metal-visible too.
"""

from __future__ import annotations

import argparse
import math
import struct
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Any

VALUE_FORMAT = {
    0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i",
    6: "f", 7: "?", 10: "Q", 11: "q", 12: "d",
}

# GGML type id: (name, elements per block, bytes per block).
# Types outside this table are rejected so totals cannot be silently wrong.
GGML_TYPES = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210), 16: ("IQ2_XXS", 256, 66),
    17: ("IQ2_XS", 256, 74), 18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50), 20: ("IQ4_NL", 32, 18),
    21: ("IQ3_S", 256, 110), 22: ("IQ2_S", 256, 82),
    23: ("IQ4_XS", 256, 136), 30: ("BF16", 1, 2),
}


@dataclass
class Tensor:
    name: str
    shape: tuple[int, ...]
    type_id: int
    offset: int
    size: int

    @property
    def type_name(self) -> str:
        return GGML_TYPES.get(self.type_id, (f"TYPE_{self.type_id}", 0, 0))[0]


def read_num(f: BinaryIO, fmt: str) -> Any:
    size = struct.calcsize("<" + fmt)
    raw = f.read(size)
    if len(raw) != size:
        raise EOFError("truncated GGUF metadata")
    values = struct.unpack("<" + fmt, raw)
    return values[0] if len(values) == 1 else values


def read_string(f: BinaryIO) -> str:
    length = read_num(f, "Q")
    raw = f.read(length)
    if len(raw) != length:
        raise EOFError("truncated GGUF string")
    return raw.decode("utf-8")


def read_value(f: BinaryIO, type_id: int) -> Any:
    if type_id in VALUE_FORMAT:
        return read_num(f, VALUE_FORMAT[type_id])
    if type_id == 8:
        return read_string(f)
    if type_id == 9:
        element_type = read_num(f, "I")
        count = read_num(f, "Q")
        return [read_value(f, element_type) for _ in range(count)]
    raise ValueError(f"unsupported GGUF value type {type_id}")


def tensor_nbytes(shape: tuple[int, ...], type_id: int) -> int:
    if type_id not in GGML_TYPES:
        raise ValueError(f"unknown GGML tensor type {type_id}")
    _, block, block_bytes = GGML_TYPES[type_id]
    if not shape or shape[0] % block:
        raise ValueError(f"invalid row width {shape[0] if shape else 0} for {GGML_TYPES[type_id][0]}")
    return shape[0] // block * block_bytes * math.prod(shape[1:])


def read_gguf(path: Path) -> tuple[dict[str, Any], list[Tensor], int]:
    with path.open("rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        version = read_num(f, "I")
        if version != 3:
            raise ValueError(f"unsupported GGUF version {version}")
        tensor_count, kv_count = read_num(f, "QQ")
        fields: dict[str, Any] = {}
        for _ in range(kv_count):
            key = read_string(f)
            fields[key] = read_value(f, read_num(f, "I"))
        raw_tensors = []
        for _ in range(tensor_count):
            name = read_string(f)
            n_dims = read_num(f, "I")
            shape = tuple(read_num(f, "Q") for _ in range(n_dims))
            type_id = read_num(f, "I")
            offset = read_num(f, "Q")
            raw_tensors.append((name, shape, type_id, offset))
        alignment = int(fields.get("general.alignment", 32))
        data_start = (f.tell() + alignment - 1) // alignment * alignment

    tensors = [Tensor(n, s, t, o, tensor_nbytes(s, t)) for n, s, t, o in raw_tensors]
    return fields, tensors, data_start


def placement(name: str) -> str:
    if name in {"per_layer_token_embd.weight", "token_embd.weight"}:
        return "CPU-input"
    if name.startswith("output"):
        return "Metal"
    if name.startswith("blk."):
        return "Metal"
    return "other"


def gib(n: int) -> str:
    return f"{n / (1024 ** 3):.3f} GiB"


def audit(path: Path, list_tensors: bool) -> Counter[str]:
    fields, tensors, data_start = read_gguf(path)
    split_no = fields.get("split.no", 0)
    split_count = fields.get("split.count", 1)
    print(f"{path}: split {split_no + 1}/{split_count}, tensors={len(tensors)}, data_start={data_start}")

    totals: Counter[str] = Counter()
    ranges: dict[str, list[int]] = {}
    for tensor in tensors:
        place = placement(tensor.name)
        totals[place] += tensor.size
        if place != "other":
            bounds = ranges.setdefault(place, [tensor.offset, tensor.offset + tensor.size])
            bounds[0] = min(bounds[0], tensor.offset)
            bounds[1] = max(bounds[1], tensor.offset + tensor.size)
        if tensor.name == "per_layer_token_embd.weight":
            print(f"  PLE: {tensor.type_name} {tensor.shape}, {gib(tensor.size)}, file bytes "
                  f"[{data_start + tensor.offset}, {data_start + tensor.offset + tensor.size})")
        if list_tensors:
            print(f"  {tensor.name}\t{tensor.type_name}\t{tensor.shape}\t{tensor.size}\t{place}")

    for place in sorted(ranges):
        first, last = ranges[place]
        print(f"  {place}: tensor bytes={gib(totals[place])}, mmap/MTL envelope={gib(last - first)}, "
              f"file bytes=[{data_start + first}, {data_start + last})")
    if "Metal" in ranges and "CPU-input" in ranges:
        mf, ml = ranges["Metal"]
        cf, cl = ranges["CPU-input"]
        overlap = max(0, min(ml, cl) - max(mf, cf))
        print(f"  Metal envelope overlap with CPU-input envelope: {gib(overlap)}")
    return totals


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gguf", nargs="+", type=Path, help="GGUF shard(s); tensor bodies may still be incomplete")
    parser.add_argument("--list-tensors", action="store_true")
    args = parser.parse_args()
    grand: Counter[str] = Counter()
    try:
        for path in args.gguf:
            grand.update(audit(path, args.list_tensors))
    except (OSError, EOFError, ValueError, struct.error) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    if len(args.gguf) > 1:
        print("totals: " + ", ".join(f"{key}={gib(value)}" for key, value in sorted(grand.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
