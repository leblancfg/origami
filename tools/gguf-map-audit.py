#!/usr/bin/env python3
"""Report llama.cpp mmap buffer envelopes from bounded GGUF metadata reads.

The parser comes from :mod:`origami_artifacts`; this command adds only the
pinned Qwen4Exp placement model. Tensor bodies are never mapped or read.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from origami_artifacts.gguf import GGUFError, QUANT_TYPES, parse_shard, tensor_nbytes


def placement(name: str) -> str:
    """Match the pinned llama.cpp Qwen4Exp input/output/layer placement."""
    if name in {"per_layer_token_embd.weight", "token_embd.weight"}:
        return "CPU-input"
    if name.startswith("output") or name.startswith("blk."):
        return "Metal"
    return "other"


def gib(value: int) -> str:
    return f"{value / (1024 ** 3):.3f} GiB"


def audit(path: Path, list_tensors: bool = False) -> Counter[str]:
    shard = parse_shard(path)
    split_no = int(shard.metadata.get("split.no", 0))
    split_count = int(shard.metadata.get("split.count", 1))
    print(
        f"{path}: split {split_no + 1}/{split_count}, tensors={len(shard.tensors)}, "
        f"data_start={shard.data_offset}"
    )

    totals: Counter[str] = Counter()
    ranges: Dict[str, List[int]] = {}
    for tensor in shard.tensors:
        size = tensor_nbytes(tensor.dimensions, tensor.type_code, tensor.name)
        place = placement(tensor.name)
        totals[place] += size
        if place != "other":
            bounds = ranges.setdefault(
                place,
                [tensor.relative_offset, tensor.relative_offset + size],
            )
            bounds[0] = min(bounds[0], tensor.relative_offset)
            bounds[1] = max(bounds[1], tensor.relative_offset + size)
        type_name = QUANT_TYPES[tensor.type_code].name
        if tensor.name == "per_layer_token_embd.weight":
            start = shard.data_offset + tensor.relative_offset
            print(
                f"  PLE: {type_name} {tensor.dimensions}, {gib(size)}, "
                f"file bytes [{start}, {start + size})"
            )
        if list_tensors:
            print(
                f"  {tensor.name}\t{type_name}\t{tensor.dimensions}\t"
                f"{size}\t{place}"
            )

    for place in sorted(ranges):
        first, last = ranges[place]
        print(
            f"  {place}: tensor bytes={gib(totals[place])}, "
            f"mmap/MTL envelope={gib(last - first)}, "
            f"file bytes=[{shard.data_offset + first}, {shard.data_offset + last})"
        )
    if "Metal" in ranges and "CPU-input" in ranges:
        metal_first, metal_last = ranges["Metal"]
        cpu_first, cpu_last = ranges["CPU-input"]
        overlap = max(0, min(metal_last, cpu_last) - max(metal_first, cpu_first))
        print(f"  Metal envelope overlap with CPU-input envelope: {gib(overlap)}")
    return totals


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "gguf",
        nargs="+",
        type=Path,
        help="GGUF shard(s); tensor bodies may be incomplete",
    )
    parser.add_argument("--list-tensors", action="store_true")
    args = parser.parse_args()
    grand: Counter[str] = Counter()
    try:
        for path in args.gguf:
            grand.update(audit(path, args.list_tensors))
    except (GGUFError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    if len(args.gguf) > 1:
        print(
            "totals: "
            + ", ".join(f"{key}={gib(value)}" for key, value in sorted(grand.items()))
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
