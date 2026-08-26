#!/usr/bin/env python3
"""Run Origami's fixed llama.cpp smoke test with macOS telemetry."""

import argparse
import datetime as dt
import hashlib
import json
import os
import plistlib
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

SCHEMA_VERSION = "origami.validation-result.v1"
MANIFEST_VERSION = "origami.model-manifest.v1"
PROMPT = "Reply with only the next four integers: 2 4 6 8"
PROMPT_SHA256 = hashlib.sha256(PROMPT.encode("utf-8")).hexdigest()
N_PREDICT = 16
SEED = 424242
TEMPERATURE = 0
MAX_CAPTURE_BYTES = 16 * 1024 * 1024

COMMANDS = {
    "df": "/bin/df",
    "diskutil": "/usr/sbin/diskutil",
    "memory_pressure": "/usr/bin/memory_pressure",
    "ps": "/bin/ps",
    "sw_vers": "/usr/bin/sw_vers",
    "sysctl": "/usr/sbin/sysctl",
    "vm_stat": "/usr/bin/vm_stat",
}

TIMING_RE = re.compile(
    r"(?P<label>load time|prompt eval time|eval time|total time)\s*=\s*"
    r"(?P<ms>[0-9]+(?:\.[0-9]+)?)\s*ms"
    r"(?:\s*/\s*(?P<tokens>[0-9]+)\s*tokens?\s*"
    r"\(\s*(?P<ms_token>[0-9]+(?:\.[0-9]+)?)\s*ms per token,\s*"
    r"(?P<tps>[0-9]+(?:\.[0-9]+)?)\s*tokens per second\s*\))?",
    re.IGNORECASE,
)
SPLIT_RE = re.compile(r"-(?P<index>[0-9]{5})-of-(?P<count>[0-9]{5})\.gguf$")


class ValidationError(Exception):
    """A clear, user-actionable validation failure."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.details = details or {}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def run_command(command: Sequence[str], timeout: float = 10.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        list(command), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, timeout=timeout, check=False,
    )


def require_dependencies() -> None:
    if sys.platform != "darwin":
        raise ValidationError("this PoC telemetry harness requires macOS (Darwin)")
    missing = [path for path in COMMANDS.values() if not os.path.isfile(path)]
    if missing:
        raise ValidationError("missing required macOS commands: " + ", ".join(missing))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValidationError("cannot read model manifest {}: {}".format(path, error))
    if not isinstance(value, dict):
        raise ValidationError("model manifest root must be a JSON object")
    return value


def validate_manifest(path: Path, verify_hashes: bool) -> Tuple[Dict[str, Any], Path]:
    manifest = read_json(path)
    if manifest.get("schema_version") != MANIFEST_VERSION:
        raise ValidationError("model manifest schema_version must be " + MANIFEST_VERSION)

    model = manifest.get("model")
    if not isinstance(model, dict):
        raise ValidationError("model manifest needs a model object")
    required_identity = ("id", "revision", "format", "quantization")
    missing_identity = [key for key in required_identity if not model.get(key)]
    if missing_identity:
        raise ValidationError("model identity is missing: " + ", ".join(missing_identity))
    if str(model["format"]).upper() != "GGUF":
        raise ValidationError("first PoC manifest format must be GGUF")

    raw_shards = manifest.get("shards")
    if not isinstance(raw_shards, list) or not raw_shards:
        raise ValidationError("model manifest needs a non-empty shards array")

    base = path.resolve().parent
    statuses: List[Dict[str, Any]] = []
    errors: List[str] = []
    resolved_by_name: Dict[str, Path] = {}
    split_parts: List[Tuple[int, int, str]] = []

    for position, shard in enumerate(raw_shards):
        if not isinstance(shard, dict):
            errors.append("shards[{}] is not an object".format(position))
            continue
        relative = shard.get("path")
        expected_size = shard.get("size_bytes")
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            errors.append("shards[{}].path must be a non-empty relative path".format(position))
            continue
        if not isinstance(expected_size, int) or isinstance(expected_size, bool) or expected_size <= 0:
            errors.append("{}: size_bytes must be a positive integer".format(relative))
            continue
        resolved = (base / relative).resolve()
        try:
            resolved.relative_to(base)
        except ValueError:
            errors.append("{}: path escapes the manifest directory".format(relative))
            continue
        if relative in resolved_by_name:
            errors.append("{}: duplicate shard path".format(relative))
            continue
        resolved_by_name[relative] = resolved

        status: Dict[str, Any] = {
            "path": relative,
            "expected_size_bytes": expected_size,
            "exists": resolved.is_file(),
            "complete": False,
        }
        partial_marker = Path(str(resolved) + ".aria2")
        status["partial_marker_exists"] = partial_marker.exists()
        if not resolved.is_file():
            errors.append("{}: shard is missing".format(relative))
        else:
            actual_size = resolved.stat().st_size
            status["actual_size_bytes"] = actual_size
            if actual_size != expected_size:
                errors.append(
                    "{}: shard size is {} bytes; expected {} (download is incomplete or wrong)".format(
                        relative, actual_size, expected_size
                    )
                )
            if partial_marker.exists():
                errors.append("{}: partial-download marker exists: {}".format(relative, partial_marker.name))
            expected_hash = shard.get("sha256")
            if expected_hash is not None:
                if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_hash):
                    errors.append("{}: sha256 must be 64 hexadecimal characters".format(relative))
                else:
                    status["expected_sha256"] = expected_hash.lower()
                    status["sha256_verified"] = False
                    if verify_hashes and actual_size == expected_size and not partial_marker.exists():
                        actual_hash = sha256_file(resolved)
                        status["actual_sha256"] = actual_hash
                        status["sha256_verified"] = True
                        if actual_hash != expected_hash.lower():
                            errors.append("{}: sha256 mismatch".format(relative))
            status["complete"] = actual_size == expected_size and not partial_marker.exists()
        statuses.append(status)

        match = SPLIT_RE.search(relative)
        if match:
            split_parts.append((int(match.group("index")), int(match.group("count")), relative))

    if split_parts:
        counts = {count for _, count, _ in split_parts}
        if len(split_parts) != len(raw_shards) or len(counts) != 1:
            errors.append("split GGUF shard names are inconsistent")
        else:
            count = next(iter(counts))
            indices = {index for index, _, _ in split_parts}
            expected_indices = set(range(1, count + 1))
            if indices != expected_indices or len(raw_shards) != count:
                errors.append(
                    "split GGUF manifest has indices {}; expected 1 through {}".format(
                        sorted(indices), count
                    )
                )

    entrypoint = manifest.get("entrypoint")
    if not isinstance(entrypoint, str) or entrypoint not in resolved_by_name:
        errors.append("entrypoint must exactly match one shards[].path")
        entrypoint_path = base
    else:
        entrypoint_path = resolved_by_name[entrypoint]

    manifest["shard_status"] = statuses
    manifest["total_expected_size_bytes"] = sum(
        shard.get("size_bytes", 0) for shard in raw_shards if isinstance(shard, dict)
        and isinstance(shard.get("size_bytes"), int) and not isinstance(shard.get("size_bytes"), bool)
    )
    manifest["total_actual_size_bytes"] = sum(
        status.get("actual_size_bytes", 0) for status in statuses
    )
    if errors:
        model_result = dict(model)
        model_result.update({
            "manifest_path": str(path.resolve()),
            "entrypoint": manifest.get("entrypoint"),
            "total_expected_size_bytes": manifest["total_expected_size_bytes"],
            "total_actual_size_bytes": manifest["total_actual_size_bytes"],
            "shards": statuses,
        })
        raise ValidationError(
            "model shard preflight failed:\n- " + "\n- ".join(errors),
            {"model": model_result},
        )
    return manifest, entrypoint_path


def sysctl_value(name: str) -> str:
    result = run_command([COMMANDS["sysctl"], "-n", name])
    if result.returncode != 0:
        raise ValidationError("sysctl {} failed: {}".format(name, result.stderr.strip()))
    return result.stdout.strip()


def collect_machine() -> Dict[str, Any]:
    product = run_command([COMMANDS["sw_vers"]])
    sw_values: Dict[str, str] = {}
    if product.returncode == 0:
        for line in product.stdout.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                sw_values[key.strip()] = value.strip()
    return {
        "hardware_model": sysctl_value("hw.model"),
        "chip": sysctl_value("machdep.cpu.brand_string"),
        "memory_bytes": int(sysctl_value("hw.memsize")),
        "physical_cpu_count": int(sysctl_value("hw.physicalcpu")),
        "logical_cpu_count": int(sysctl_value("hw.logicalcpu")),
        "page_size_bytes": int(sysctl_value("hw.pagesize")),
        "os": {
            "name": sw_values.get("ProductName", "macOS"),
            "version": sw_values.get("ProductVersion"),
            "build": sw_values.get("BuildVersion", sysctl_value("kern.osversion")),
            "kernel_release": sysctl_value("kern.osrelease"),
        },
    }


def collect_storage(path: Path) -> Dict[str, Any]:
    result = run_command([COMMANDS["df"], "-P", str(path)])
    if result.returncode != 0 or len(result.stdout.splitlines()) < 2:
        raise ValidationError("df failed for model storage: " + result.stderr.strip())
    device = result.stdout.splitlines()[-1].split()[0]
    raw = subprocess.run(
        [COMMANDS["diskutil"], "info", "-plist", device], stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, timeout=10, check=False,
    )
    if raw.returncode != 0:
        raise ValidationError(
            "diskutil failed for model storage: " + raw.stderr.decode("utf-8", "replace").strip()
        )
    try:
        info = plistlib.loads(raw.stdout)
    except Exception as error:
        raise ValidationError("cannot parse diskutil model-storage data: {}".format(error))
    # Deliberately omit volume UUIDs and device-tree identifiers.
    return {
        "device": device,
        "parent_whole_disk": info.get("ParentWholeDisk"),
        "mount_point": info.get("MountPoint"),
        "filesystem": info.get("FilesystemName"),
        "internal": info.get("Internal"),
        "solid_state": info.get("SolidState"),
        "bus_protocol": info.get("BusProtocol"),
        "smart_status": info.get("SMARTStatus"),
        "volume_size_bytes": info.get("TotalSize"),
        "container_size_bytes": info.get("APFSContainerSize"),
        "container_free_bytes": info.get("APFSContainerFree"),
    }


def parse_vm_stat(text: str) -> Dict[str, Any]:
    page_match = re.search(r"page size of ([0-9]+) bytes", text)
    if not page_match:
        raise ValidationError("cannot parse vm_stat page size")
    page_size = int(page_match.group(1))
    pages: Dict[str, int] = {}
    for line in text.splitlines()[1:]:
        match = re.match(r'\s*"?([^":]+)"?:\s*([0-9]+)\.\s*$', line)
        if match:
            key = re.sub(r"[^a-z0-9]+", "_", match.group(1).lower()).strip("_")
            pages[key] = int(match.group(2))
    return {
        "page_size_bytes": page_size,
        "pages": pages,
        "compressor_occupied_bytes": pages.get("pages_occupied_by_compressor", 0) * page_size,
        "pages_stored_in_compressor_bytes": pages.get("pages_stored_in_compressor", 0) * page_size,
    }


def bytes_from_unit(value: str, unit: str) -> int:
    factors = {"K": 1024, "M": 1024 ** 2, "G": 1024 ** 3, "T": 1024 ** 4}
    return int(round(float(value) * factors[unit.upper()]))


def parse_swapusage(text: str) -> Dict[str, int]:
    values: Dict[str, int] = {}
    for key in ("total", "used", "free"):
        match = re.search(r"\b{}\s*=\s*([0-9]+(?:\.[0-9]+)?)([KMGT])\b".format(key), text, re.I)
        if not match:
            raise ValidationError("cannot parse vm.swapusage field " + key)
        values[key + "_bytes"] = bytes_from_unit(match.group(1), match.group(2))
    return values


def collect_system_snapshot(elapsed_seconds: Optional[float] = None) -> Dict[str, Any]:
    vm = run_command([COMMANDS["vm_stat"]])
    pressure = run_command([COMMANDS["memory_pressure"], "-Q"])
    swap = run_command([COMMANDS["sysctl"], "-n", "vm.swapusage"])
    failures = [
        name for name, result in (("vm_stat", vm), ("memory_pressure -Q", pressure), ("vm.swapusage", swap))
        if result.returncode != 0
    ]
    if failures:
        raise ValidationError("telemetry command failed: " + ", ".join(failures))
    pressure_match = re.search(r"free percentage:\s*([0-9]+)%", pressure.stdout)
    if not pressure_match:
        raise ValidationError("cannot parse memory_pressure -Q output")
    snapshot: Dict[str, Any] = {
        "captured_at": utc_now(),
        "memory_pressure_free_percent": int(pressure_match.group(1)),
        "vm": parse_vm_stat(vm.stdout),
        "swap": parse_swapusage(swap.stdout),
    }
    if elapsed_seconds is not None:
        snapshot["elapsed_seconds"] = round(elapsed_seconds, 6)
    return snapshot


def process_tree_rss(root_pid: int) -> Dict[str, Any]:
    result = run_command([COMMANDS["ps"], "-axo", "pid=,ppid=,rss="])
    if result.returncode != 0:
        raise ValidationError("ps failed while sampling RSS: " + result.stderr.strip())
    rows: Dict[int, Tuple[int, int]] = {}
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) == 3:
            try:
                rows[int(fields[0])] = (int(fields[1]), int(fields[2]) * 1024)
            except ValueError:
                pass
    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, (ppid, _) in rows.items():
            if ppid in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    present = sorted(pid for pid in descendants if pid in rows)
    return {
        "root_rss_bytes": rows.get(root_pid, (0, 0))[1],
        "process_tree_rss_bytes": sum(rows[pid][1] for pid in present),
        "process_count": len(present),
    }


def parse_timings(stderr: str) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}
    names = {
        "load time": "load",
        "prompt eval time": "prefill",
        "eval time": "decode",
        "total time": "total",
    }
    for match in TIMING_RE.finditer(stderr):
        item: Dict[str, Any] = {"elapsed_ms": float(match.group("ms"))}
        if match.group("tokens") is not None:
            item.update({
                "tokens": int(match.group("tokens")),
                "ms_per_token": float(match.group("ms_token")),
                "tokens_per_second": float(match.group("tps")),
            })
        parsed[names[match.group("label").lower()]] = item
    return parsed


def git_identity(root: Path) -> Dict[str, Any]:
    try:
        commit = run_command(["/usr/bin/git", "-C", str(root), "rev-parse", "HEAD"])
        dirty = run_command(["/usr/bin/git", "-C", str(root), "status", "--porcelain"])
    except OSError:
        return {"commit": None, "worktree_dirty": None}
    return {
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "worktree_dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
    }


def executable_identity(path: Path, supplied_revision: str) -> Dict[str, Any]:
    version = run_command([str(path), "--version"], timeout=10)
    version_text = (version.stdout + version.stderr).strip()
    if version.returncode != 0:
        raise ValidationError("executable --version failed (exit {}): {}".format(
            version.returncode, version_text[:1000]
        ))
    stat = path.stat()
    return {
        "path": str(path),
        "revision": supplied_revision,
        "version_output": version_text[:4000],
        "size_bytes": stat.st_size,
        "sha256": sha256_file(path),
    }


def terminate_process_group(process: subprocess.Popen, grace_seconds: float = 2.0) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def read_capture(path: Path) -> Tuple[bytes, bool]:
    with path.open("rb") as stream:
        data = stream.read(MAX_CAPTURE_BYTES + 1)
    return data[:MAX_CAPTURE_BYTES], len(data) > MAX_CAPTURE_BYTES


def telemetry_summary(before: Dict[str, Any], during: List[Dict[str, Any]], after: Dict[str, Any], rss: List[Dict[str, Any]]) -> Dict[str, Any]:
    system_samples = [before] + during + [after]
    vm_before = before["vm"]["pages"]
    vm_after = after["vm"]["pages"]
    summary: Dict[str, Any] = {
        "peak_root_rss_bytes": max((item["root_rss_bytes"] for item in rss), default=0),
        "peak_process_tree_rss_bytes": max((item["process_tree_rss_bytes"] for item in rss), default=0),
        "minimum_memory_pressure_free_percent": min(
            item["memory_pressure_free_percent"] for item in system_samples
        ),
        "peak_compressor_occupied_bytes": max(
            item["vm"]["compressor_occupied_bytes"] for item in system_samples
        ),
        "compressor_occupied_delta_bytes": (
            after["vm"]["compressor_occupied_bytes"] - before["vm"]["compressor_occupied_bytes"]
        ),
        "swap_used_delta_bytes": after["swap"]["used_bytes"] - before["swap"]["used_bytes"],
    }
    for field in ("swapins", "swapouts", "pageins", "pageouts", "compressions", "decompressions"):
        if field in vm_before and field in vm_after:
            summary[field + "_delta_pages"] = vm_after[field] - vm_before[field]
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the fixed Origami llama.cpp smoke test and emit one JSON result."
    )
    parser.add_argument("--executable", required=True, type=Path, help="llama.cpp-compatible CLI")
    parser.add_argument("--runtime-revision", required=True, help="exact llama.cpp commit or build revision")
    parser.add_argument("--model-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="result JSON path")
    parser.add_argument("--expected-output-sha256", help="fail unless raw stdout has this SHA-256")
    parser.add_argument("--verify-shards-sha256", action="store_true", help="read and hash shards that declare sha256")
    parser.add_argument("--sample-interval", type=float, default=1.0, help="telemetry interval in seconds")
    parser.add_argument("--timeout", type=float, default=1800.0, help="hard run timeout in seconds")
    parser.add_argument(
        "extra_args", nargs=argparse.REMAINDER,
        help="additional runtime arguments after -- (fixed deterministic flags are appended last)",
    )
    return parser


def write_result(path: Path, result: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    started_at = utc_now()
    script_root = Path(__file__).resolve().parents[1]
    result: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "error",
        "started_at": started_at,
        "smoke_test": {
            "name": "fixed-greedy-sequence-v1",
            "prompt": PROMPT,
            "prompt_sha256": PROMPT_SHA256,
            "n_predict": N_PREDICT,
            "seed": SEED,
            "temperature": TEMPERATURE,
        },
        "project": git_identity(script_root),
    }
    result["project"].update({
        "harness_path": str(Path(__file__).resolve().relative_to(script_root)),
        "harness_sha256": sha256_file(Path(__file__).resolve()),
    })
    exit_code = 2
    try:
        require_dependencies()
        if args.sample_interval <= 0:
            raise ValidationError("--sample-interval must be greater than zero")
        if args.timeout <= 0:
            raise ValidationError("--timeout must be greater than zero")
        executable = args.executable.expanduser().resolve()
        if not executable.is_file() or not os.access(str(executable), os.X_OK):
            raise ValidationError("executable does not exist or is not executable: " + str(executable))
        if not args.runtime_revision.strip():
            raise ValidationError("--runtime-revision cannot be empty")
        if args.expected_output_sha256 and not re.fullmatch(r"[0-9a-fA-F]{64}", args.expected_output_sha256):
            raise ValidationError("--expected-output-sha256 must be 64 hexadecimal characters")

        manifest_path = args.model_manifest.expanduser().resolve()
        manifest, model_entrypoint = validate_manifest(manifest_path, args.verify_shards_sha256)
        result["model"] = manifest["model"]
        result["model"].update({
            "manifest_path": str(manifest_path),
            "entrypoint": manifest["entrypoint"],
            "total_expected_size_bytes": manifest["total_expected_size_bytes"],
            "total_actual_size_bytes": manifest["total_actual_size_bytes"],
            "shards": manifest["shard_status"],
        })
        result["machine"] = collect_machine()
        result["model_storage"] = collect_storage(model_entrypoint)
        if result["model_storage"].get("internal") is not True or result["model_storage"].get("solid_state") is not True:
            raise ValidationError("the first PoC requires model shards on an internal solid-state volume")
        result["runtime"] = executable_identity(executable, args.runtime_revision.strip())

        extra_args = list(args.extra_args)
        if extra_args and extra_args[0] == "--":
            extra_args = extra_args[1:]
        command = [str(executable)] + extra_args + [
            "--model", str(model_entrypoint),
            "--prompt", PROMPT,
            "--n-predict", str(N_PREDICT),
            "--seed", str(SEED),
            "--temp", str(TEMPERATURE),
            "--no-display-prompt",
        ]
        result["command"] = command

        before = collect_system_snapshot()
        during: List[Dict[str, Any]] = []
        rss_samples: List[Dict[str, Any]] = []
        timed_out = False
        wall_start = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="origami-validation-") as temporary_dir:
            stdout_path = Path(temporary_dir) / "stdout"
            stderr_path = Path(temporary_dir) / "stderr"
            with stdout_path.open("wb") as stdout_file, stderr_path.open("wb") as stderr_file:
                process = subprocess.Popen(
                    command, stdout=stdout_file, stderr=stderr_file,
                    start_new_session=True, stdin=subprocess.DEVNULL,
                )
                try:
                    while process.poll() is None:
                        elapsed = time.monotonic() - wall_start
                        if elapsed >= args.timeout:
                            timed_out = True
                            terminate_process_group(process)
                            break
                        rss_item = process_tree_rss(process.pid)
                        rss_item["elapsed_seconds"] = round(elapsed, 6)
                        rss_samples.append(rss_item)
                        during.append(collect_system_snapshot(elapsed))
                        remaining = args.sample_interval - (time.monotonic() - wall_start - elapsed)
                        if remaining > 0:
                            try:
                                process.wait(timeout=min(remaining, max(0.0, args.timeout - (time.monotonic() - wall_start))))
                            except subprocess.TimeoutExpired:
                                pass
                except BaseException:
                    terminate_process_group(process)
                    raise
                returncode = process.wait()
            stdout_bytes, stdout_truncated = read_capture(stdout_path)
            stderr_bytes, stderr_truncated = read_capture(stderr_path)
        wall_elapsed = time.monotonic() - wall_start
        after = collect_system_snapshot(wall_elapsed)

        stdout_text = stdout_bytes.decode("utf-8", "replace")
        stderr_text = stderr_bytes.decode("utf-8", "replace")
        output_hash = hashlib.sha256(stdout_bytes).hexdigest()
        timings = parse_timings(stderr_text)
        result["run"] = {
            "exit_code": returncode,
            "timed_out": timed_out,
            "wall_elapsed_seconds": round(wall_elapsed, 6),
            "stdout": stdout_text,
            "stdout_sha256": output_hash,
            "stdout_truncated": stdout_truncated,
            "stderr": stderr_text,
            "stderr_truncated": stderr_truncated,
            "timings": timings,
        }
        result["telemetry"] = {
            "before": before,
            "during": during,
            "after": after,
            "process_rss_samples": rss_samples,
            "summary": telemetry_summary(before, during, after, rss_samples),
        }

        failures: List[str] = []
        if timed_out:
            failures.append("runtime exceeded the {} second timeout".format(args.timeout))
        if returncode != 0:
            failures.append("runtime exited with status {}".format(returncode))
        if stdout_truncated or stderr_truncated:
            failures.append("runtime output exceeded the {} byte capture limit".format(MAX_CAPTURE_BYTES))
        for required_timing in ("prefill", "decode"):
            if required_timing not in timings:
                failures.append("llama.cpp {} timing was not found on stderr".format(required_timing))
        if args.expected_output_sha256 and output_hash != args.expected_output_sha256.lower():
            failures.append("stdout SHA-256 does not match --expected-output-sha256")
        if not stdout_bytes:
            failures.append("runtime produced empty stdout")
        if failures:
            raise ValidationError("; ".join(failures))

        result["status"] = "pass"
        exit_code = 0
    except ValidationError as error:
        result.update(error.details)
        result["error"] = str(error)
        exit_code = 3 if "run" in result else 2
    except KeyboardInterrupt:
        result["error"] = "interrupted"
        exit_code = 130
    except Exception as error:
        result["error"] = "unexpected harness failure: {}: {}".format(type(error).__name__, error)
        exit_code = 2
    finally:
        result["finished_at"] = utc_now()
        try:
            write_result(args.output.expanduser().resolve(), result)
        except Exception as error:
            print("cannot write result {}: {}".format(args.output, error), file=sys.stderr)
            return 2

    if exit_code != 0:
        print("validation failed: " + result.get("error", "unknown error"), file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
