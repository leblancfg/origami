"""Command-line interface for the GGUF allocation ledger."""

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

from .gguf import GGUFError, format_size, inspect_artifact, parse_size, probe_tensor_spans


def _size_argument(value: str) -> int:
    try:
        return parse_size(value)
    except GGUFError as exc:
        raise argparse.ArgumentTypeError(str(exc))


def render_summary(report: Dict[str, Any], probes: Optional[List[Dict[str, Any]]] = None) -> str:
    split = report["split"]
    inventory = report["inventory"]
    ledger = report["ledger"]
    state = "complete" if report["complete"] else "INCOMPLETE"
    lines = [
        "GGUF artifact: %s" % state,
        "  shards: %d/%d present%s"
        % (
            len(split["present"]),
            split["count"],
            " (missing %s)" % ", ".join(map(str, split["missing"]))
            if split["missing"]
            else "",
        ),
        "  tensors: %d parsed / %d declared"
        % (split["parsed_tensor_count"], split["declared_tensor_count"]),
        "  ledger scope: %s" % ledger["scope"],
        "",
        "Tensor inventory (logical bytes, excluding GGUF alignment gaps):",
    ]
    for category in ("dense", "shared", "routed", "ple", "total"):
        lines.append("  %-8s %15d  %s" % (category, inventory["bytes"][category], format_size(inventory["bytes"][category])))
    lines.extend(
        [
            "",
            "Allocation ledger:",
            "  resident weights %15d  %s"
            % (ledger["resident_weight_bytes"], format_size(ledger["resident_weight_bytes"])),
            "  streamed weights %15d  %s"
            % (ledger["streamed_weight_bytes"], format_size(ledger["streamed_weight_bytes"])),
            "  cache budget     %15d  %s"
            % (ledger["cache_budget_bytes"], format_size(ledger["cache_budget_bytes"])),
            "  temporary budget %15d  %s"
            % (ledger["temporary_budget_bytes"], format_size(ledger["temporary_budget_bytes"])),
            "  runtime accounted%15d  %s"
            % (ledger["runtime_accounted_bytes"], format_size(ledger["runtime_accounted_bytes"])),
            "",
            "Shard bounds:",
        ]
    )
    for shard in report["shards"]:
        lines.append(
            "  %d: %d tensors, metadata read %d bytes, file %d / expected minimum %d bytes, body %s"
            % (
                shard["number"],
                shard["tensor_count"],
                shard["metadata_bytes_read"],
                shard["file_size"],
                shard["expected_min_size"],
                "complete" if shard["body_complete"] else "truncated",
            )
        )
    if probes:
        lines.extend(["", "Bounded span probes:"])
        for probe in probes:
            lines.append(
                "  %s: [%d, %d) within %d bytes; read %d bytes at each edge"
                % (
                    probe["tensor"],
                    probe["span"][0],
                    probe["span"][1],
                    probe["shard_size"],
                    probe["reads"][0][1],
                )
            )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m origami_artifacts",
        description="Inspect split GGUF metadata without mapping tensor bodies.",
    )
    parser.add_argument("paths", nargs="+", help="GGUF shard or directory containing shards")
    parser.add_argument("--json", action="store_true", help="emit the ledger and inventory as JSON")
    parser.add_argument("--output", help="write output to this file instead of stdout")
    parser.add_argument("--no-slices", action="store_true", help="omit per-expert slices from JSON")
    parser.add_argument("--expert-cache", type=_size_argument, default=0, metavar="SIZE")
    parser.add_argument("--ple-cache", type=_size_argument, default=0, metavar="SIZE")
    parser.add_argument("--temporary", type=_size_argument, default=0, metavar="SIZE")
    parser.add_argument(
        "--max-metadata", type=_size_argument, default=64 * 1024 * 1024, metavar="SIZE"
    )
    parser.add_argument(
        "--probe", action="append", default=[], metavar="TENSOR", help="pread bounded bytes at both ends of a selected tensor span"
    )
    parser.add_argument(
        "--probe-bytes", type=int, default=1, metavar="N", help="bytes per probed edge (1..4096)"
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = inspect_artifact(
            args.paths,
            expert_cache_bytes=args.expert_cache,
            ple_cache_bytes=args.ple_cache,
            temporary_bytes=args.temporary,
            include_slices=not args.no_slices,
            max_metadata_bytes=args.max_metadata,
        )
        probes = probe_tensor_spans(report, args.probe, edge_bytes=args.probe_bytes) if args.probe else []
        if probes:
            report["probes"] = probes
        output = json.dumps(report, indent=2, sort_keys=True) if args.json else render_summary(report, probes)
        output += "\n"
        if args.output:
            with open(args.output, "w", encoding="utf-8") as handle:
                handle.write(output)
        else:
            sys.stdout.write(output)
        return 0
    except (GGUFError, OSError) as exc:
        parser.exit(2, "error: %s\n" % exc)
        return 2
