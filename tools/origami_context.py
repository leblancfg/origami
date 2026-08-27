#!/usr/bin/env python3
"""Render, allocate, and probe the pinned Qwen3.8 long-context server profile."""

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = ROOT / "config" / "qwen38-context-262144.json"
DEFAULT_DEPS = Path("/private/tmp/origami-deps")
STATE_VERSION = "origami.context-state.v1"
RESULT_VERSION = "origami.context-probe.v1"


class ContextError(Exception):
    pass


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ContextError(f"cannot read JSON {path}: {error}")
    if not isinstance(value, dict):
        raise ContextError(f"JSON root must be an object: {path}")
    return value


def load_profile(path: Path) -> Dict[str, Any]:
    profile = read_json(path)
    if profile.get("schema_version") != "origami.context-profile.v1":
        raise ContextError("unsupported context profile schema")
    server = profile.get("server")
    memory = profile.get("memory")
    probe = profile.get("probe")
    if not all(isinstance(item, dict) for item in (server, memory, probe)):
        raise ContextError("context profile needs server, memory, and probe objects")
    if memory.get("context_tokens") != 262144:
        raise ContextError("native profile must declare exactly 262144 context tokens")
    runtime_patch = profile.get("runtime_patch")
    if not isinstance(runtime_patch, str) or not runtime_patch.startswith("patches/"):
        raise ContextError("context profile needs a repository-relative runtime_patch")
    if memory.get("total_kv_bytes") != memory.get("attention_kv_bytes", 0) + memory.get("qsa_indexer_kv_bytes", 0):
        raise ContextError("context profile KV byte ledger does not add up")
    gate = profile.get("execution_gate")
    if not isinstance(gate, dict) or not isinstance(gate.get("capability_manifest"), str):
        raise ContextError("context profile needs an execution capability gate")
    manifest_name = gate["capability_manifest"]
    if Path(manifest_name).name != manifest_name:
        raise ContextError("capability manifest must be a build-directory file name")
    requirements = gate.get("required_capabilities")
    if not isinstance(requirements, list) or not requirements:
        raise ContextError("execution gate needs required capabilities")
    names = [item.get("name") for item in requirements if isinstance(item, dict)]
    if len(names) != len(requirements) or not all(isinstance(name, str) and name for name in names):
        raise ContextError("each required capability needs a name")
    if len(names) != len(set(names)):
        raise ContextError("required capability names must be unique")
    args = server.get("arguments")
    if not isinstance(args, list) or not all(isinstance(item, str) for item in args):
        raise ContextError("server.arguments must be an array of strings")
    required = server.get("required_log_markers")
    forbidden = server.get("forbidden_log_markers")
    if not isinstance(required, list) or not isinstance(forbidden, list):
        raise ContextError("server log marker lists are required")
    return profile


def model_entrypoint(profile: Dict[str, Any], model_root: Path) -> Path:
    manifest_path = ROOT / str(profile["model_manifest"])
    manifest = read_json(manifest_path)
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ContextError("model manifest has no shards")
    errors = []
    for shard in shards:
        path = model_root / str(shard["path"])
        expected = shard.get("size_bytes")
        if not path.is_file():
            errors.append(f"missing {path.name}")
        elif path.stat().st_size != expected:
            errors.append(f"{path.name} is {path.stat().st_size} bytes; expected {expected}")
        if Path(str(path) + ".aria2").exists():
            errors.append(f"partial-download marker exists for {path.name}")
    if errors:
        raise ContextError("model preflight failed: " + "; ".join(errors))
    return (model_root / str(manifest["entrypoint"])).resolve()


def capability_gate_failures(profile: Dict[str, Any], manifest: Dict[str, Any]) -> List[str]:
    failures = []
    if manifest.get("schema_version") != "origami.backend-capabilities.v1":
        failures.append("unsupported backend capability manifest schema")
    if manifest.get("runtime_revision") != profile.get("runtime_revision"):
        failures.append("capability manifest runtime revision does not match the profile")
    provided = manifest.get("capabilities")
    if not isinstance(provided, dict):
        return failures + ["capability manifest needs a capabilities object"]
    for requirement in profile["execution_gate"]["required_capabilities"]:
        name = requirement["name"]
        evidence = provided.get(name)
        if not isinstance(evidence, str) or not evidence.strip():
            failures.append(f"missing backend capability: {name}")
            continue
        expected_commit = requirement.get("upstream_commit")
        if expected_commit is not None and evidence != expected_commit:
            failures.append(f"backend capability {name} needs exact evidence {expected_commit}")
    return failures


def enforce_execution_gate(profile: Dict[str, Any], build: Path) -> None:
    relative = profile["execution_gate"]["capability_manifest"]
    manifest_path = build / relative
    if not manifest_path.is_file():
        names = ", ".join(item["name"] for item in profile["execution_gate"]["required_capabilities"])
        raise ContextError(f"execution blocked: backend capability manifest is missing ({names})")
    manifest = read_json(manifest_path)
    failures = capability_gate_failures(profile, manifest)
    if failures:
        raise ContextError("execution blocked: " + "; ".join(failures))


def server_executable(profile: Dict[str, Any], deps_root: Path) -> Path:
    revision = str(profile["runtime_revision"])
    build = deps_root / f"llama.cpp-{revision}" / "build-origami"
    executable = build / "bin" / "llama-server"
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise ContextError(f"pinned llama-server is missing: {executable}")
    marker = build / "origami-revision.txt"
    if not marker.is_file() or marker.read_text(encoding="utf-8").strip() != revision:
        raise ContextError("pinned backend revision marker is missing or stale")
    patch_value = profile.get("runtime_patch")
    if not isinstance(patch_value, str) or not patch_value:
        raise ContextError("context profile needs a runtime_patch")
    patch = (ROOT / patch_value).resolve()
    try:
        patch.relative_to(ROOT)
    except ValueError:
        raise ContextError("runtime patch must remain inside the repository")
    patch_marker = build / "origami-patch.sha256"
    patch_hash = hashlib.sha256(patch.read_bytes()).hexdigest()
    if not patch_marker.is_file() or patch_marker.read_text(encoding="utf-8").strip() != patch_hash:
        raise ContextError("pinned backend patch marker is missing or stale")
    enforce_execution_gate(profile, build)
    return executable.resolve()


def render_command(
    profile: Dict[str, Any], executable: Path, model: Path,
    host: Optional[str] = None, port: Optional[int] = None,
) -> Tuple[Dict[str, str], List[str]]:
    server = profile["server"]
    values = {
        "host": host or str(server["host"]),
        "port": str(port if port is not None else int(server["port"])),
        "alias": str(server["alias"]),
        "model": str(model),
    }
    arguments = [item.format(**values) for item in server["arguments"]]
    environment = {str(k): str(v) for k, v in server["environment"].items()}
    return environment, [str(executable)] + arguments


def run_text(command: Sequence[str], timeout: float = 10.0) -> str:
    result = subprocess.run(
        list(command), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace", timeout=timeout, check=False,
        env={**os.environ, "LANG": "C", "LC_ALL": "C"},
    )
    if result.returncode != 0:
        raise ContextError(f"command failed ({' '.join(command)}): {result.stderr.strip()}")
    return result.stdout


def parse_pressure(text: str) -> int:
    match = re.search(r"free percentage:\s*([0-9]+)%", text)
    if not match:
        raise ContextError("cannot parse memory_pressure -Q")
    return int(match.group(1))


def bytes_from_unit(value: str, unit: str) -> int:
    return int(round(float(value) * {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}[unit.upper()]))


def parse_swapusage(text: str) -> Dict[str, int]:
    parsed: Dict[str, int] = {}
    for name in ("total", "used", "free"):
        match = re.search(rf"\b{name}\s*=\s*([0-9]+(?:\.[0-9]+)?)([KMGT])\b", text, re.I)
        if not match:
            raise ContextError(f"cannot parse vm.swapusage {name}")
        parsed[name + "_bytes"] = bytes_from_unit(match.group(1), match.group(2))
    return parsed


def parse_vm_stat(text: str) -> Dict[str, Any]:
    page = re.search(r"page size of ([0-9]+) bytes", text)
    if not page:
        raise ContextError("cannot parse vm_stat page size")
    page_size = int(page.group(1))
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
    }


def collect_snapshot(elapsed: Optional[float] = None) -> Dict[str, Any]:
    pressure = run_text(["/usr/bin/memory_pressure", "-Q"])
    swap = run_text(["/usr/sbin/sysctl", "-n", "vm.swapusage"])
    vm = run_text(["/usr/bin/vm_stat"])
    result: Dict[str, Any] = {
        "captured_at": utc_now(),
        "memory_pressure_free_percent": parse_pressure(pressure),
        "swap": parse_swapusage(swap),
        "vm": parse_vm_stat(vm),
    }
    if elapsed is not None:
        result["elapsed_seconds"] = round(elapsed, 6)
    return result


def gate_failures(profile: Dict[str, Any], baseline: Dict[str, Any], sample: Dict[str, Any]) -> List[str]:
    limits = profile["memory"]
    failures = []
    pressure = sample["memory_pressure_free_percent"]
    if pressure < int(limits["minimum_memory_pressure_free_percent"]):
        failures.append(f"memory-pressure free percentage {pressure}% is below {limits['minimum_memory_pressure_free_percent']}%")
    swap_growth = sample["swap"]["used_bytes"] - baseline["swap"]["used_bytes"]
    if swap_growth > int(limits["maximum_swap_growth_bytes"]):
        failures.append(f"swap grew by {swap_growth} bytes")
    page_size = sample["vm"]["page_size_bytes"]
    swapouts = sample["vm"]["pages"].get("swapouts", 0) - baseline["vm"]["pages"].get("swapouts", 0)
    if swapouts > int(limits["maximum_swapout_growth_pages"]):
        failures.append(f"swapouts grew by {swapouts} pages ({swapouts * page_size} bytes)")
    compressor = sample["vm"]["compressor_occupied_bytes"] - baseline["vm"]["compressor_occupied_bytes"]
    if compressor > int(limits["maximum_compressor_growth_bytes"]):
        failures.append(f"compressor occupancy grew by {compressor} bytes")
    return failures


def http_json(url: str, data: Optional[Dict[str, Any]] = None, timeout: float = 10.0) -> Dict[str, Any]:
    body = None if data is None else json.dumps(data, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(url, data=body)
    if body is not None:
        request.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise ContextError(f"expected JSON object from {url}")
    return value


def http_text(url: str, timeout: float = 10.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8", "replace")


def parse_metrics(text: str) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) == 2:
            try:
                values[fields[0]] = float(fields[1])
            except ValueError:
                pass
    return values


def process_command(pid: int) -> Optional[str]:
    result = subprocess.run(
        ["/bin/ps", "-p", str(pid), "-o", "command="], stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, text=True, check=False,
    )
    command = result.stdout.strip()
    return command or None


def list_llama_servers() -> List[Tuple[int, str]]:
    output = run_text(["/bin/ps", "-axo", "pid=,command="])
    found = []
    for line in output.splitlines():
        fields = line.strip().split(None, 1)
        if len(fields) != 2:
            continue
        try:
            pid = int(fields[0])
        except ValueError:
            continue
        executable = fields[1].split(None, 1)[0]
        if Path(executable).name == "llama-server":
            found.append((pid, fields[1]))
    return found


def port_is_open(host: str, port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, port)) == 0


def verify_owned_process(state: Dict[str, Any]) -> int:
    pid = int(state["pid"])
    command = process_command(pid)
    if command is None:
        raise ContextError(f"recorded server PID {pid} is not running")
    for expected in (state["executable"], state["model_entrypoint"], "--port " + str(state["port"])):
        if expected not in command:
            raise ContextError(f"PID {pid} no longer matches the recorded server: missing {expected}")
    return pid


def terminate_owned_process(state: Dict[str, Any], grace: float = 5.0) -> None:
    pid = verify_owned_process(state)
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        return
    if pgid != pid:
        raise ContextError(f"refusing to signal unexpected process group {pgid} for PID {pid}")
    os.killpg(pgid, signal.SIGTERM)
    deadline = time.monotonic() + grace
    while process_command(pid) is not None and time.monotonic() < deadline:
        time.sleep(0.05)
    if process_command(pid) is not None:
        os.killpg(pgid, signal.SIGKILL)


def check_log(profile: Dict[str, Any], text: str) -> List[str]:
    failures = [f"missing startup marker: {item}" for item in profile["server"]["required_log_markers"] if item not in text]
    failures.extend(f"forbidden backend marker: {item}" for item in profile["server"]["forbidden_log_markers"] if item in text)
    return failures


def write_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def base_url(state: Dict[str, Any]) -> str:
    return f"http://{state['host']}:{state['port']}"


def status_action(profile: Dict[str, Any]) -> int:
    gate = profile["execution_gate"]
    print(json.dumps({
        "name": profile["name"],
        "status": profile.get("status"),
        "runtime_revision": profile["runtime_revision"],
        "required_capabilities": gate["required_capabilities"],
        "configured_context_tokens": profile["memory"]["context_tokens"],
        "validated_prompt_tokens": 0,
    }, indent=2))
    return 0


def command_action(args: argparse.Namespace, profile: Dict[str, Any]) -> int:
    executable = server_executable(profile, args.deps_root.expanduser().resolve())
    model = model_entrypoint(profile, args.model_root.expanduser().resolve())
    environment, command = render_command(profile, executable, model, args.host, args.port)
    for key, value in environment.items():
        print(f"export {key}={shlex.quote(value)}")
    print("exec " + shlex.join(command))
    return 0


def allocate_action(args: argparse.Namespace, profile: Dict[str, Any]) -> int:
    if sys.platform != "darwin":
        raise ContextError("allocation telemetry requires macOS")
    competing = list_llama_servers()
    if competing:
        detail = "; ".join(f"PID {pid}: {command}" for pid, command in competing)
        raise ContextError("refusing to start a competing llama-server: " + detail)

    executable = server_executable(profile, args.deps_root.expanduser().resolve())
    model = model_entrypoint(profile, args.model_root.expanduser().resolve())
    environment, command = render_command(profile, executable, model, args.host, args.port)
    host = args.host or str(profile["server"]["host"])
    port = args.port if args.port is not None else int(profile["server"]["port"])
    if port_is_open(host, port):
        raise ContextError(f"refusing to use occupied address {host}:{port}")

    state_dir = args.state_dir.expanduser().resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / "state.json"
    if state_path.exists():
        old = read_json(state_path)
        if old.get("status") == "pass" and process_command(int(old.get("pid", -1))) is not None:
            raise ContextError(f"state directory already owns live PID {old['pid']}")
    log_path = state_dir / "server.log"
    baseline = collect_snapshot()
    initial_failures = gate_failures(profile, baseline, baseline)
    if initial_failures:
        raise ContextError("initial memory gate failed: " + "; ".join(initial_failures))

    env = {**os.environ, **environment, "LANG": "C", "LC_ALL": "C"}
    log_file = log_path.open("wb")
    process = subprocess.Popen(
        command, stdout=log_file, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        start_new_session=True, env=env,
    )
    state: Dict[str, Any] = {
        "schema_version": STATE_VERSION,
        "status": "starting",
        "started_at": utc_now(),
        "pid": process.pid,
        "host": host,
        "port": port,
        "executable": str(executable),
        "model_entrypoint": str(model),
        "profile_path": str(args.profile.expanduser().resolve()),
        "log_path": str(log_path),
        "command": command,
        "environment": environment,
        "allocation": {"baseline": baseline, "samples": []},
    }
    write_json(state_path, state)
    started = time.monotonic()
    try:
        while time.monotonic() - started < args.startup_timeout:
            if process.poll() is not None:
                raise ContextError(f"llama-server exited during allocation with status {process.returncode}")
            sample = collect_snapshot(time.monotonic() - started)
            state["allocation"]["samples"].append(sample)
            failures = gate_failures(profile, baseline, sample)
            if failures:
                raise ContextError("allocation memory gate failed: " + "; ".join(failures))
            try:
                health = http_json(base_url(state) + "/health", timeout=1)
                props = http_json(base_url(state) + "/props", timeout=1)
                models = http_json(base_url(state) + "/v1/models", timeout=1)
            except (OSError, urllib.error.URLError, json.JSONDecodeError, ContextError):
                time.sleep(args.sample_interval)
                continue
            log_file.flush()
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
            log_failures = check_log(profile, log_text)
            forbidden = [item for item in log_failures if item.startswith("forbidden")]
            if forbidden:
                raise ContextError("backend startup failed: " + "; ".join(forbidden))
            declared = props.get("default_generation_settings", {}).get("n_ctx")
            if declared != int(profile["memory"]["context_tokens"]):
                raise ContextError(f"/props reports n_ctx={declared}, expected 262144")
            if health.get("status") != "ok":
                raise ContextError(f"/health is not ok: {health}")
            if log_failures:
                time.sleep(args.sample_interval)
                continue
            state["status"] = "pass"
            state["ready_at"] = utc_now()
            state["allocation"]["after"] = sample
            state["health"] = health
            state["props"] = props
            state["models"] = models
            write_json(state_path, state)
            print(state_path)
            return 0
        raise ContextError(f"llama-server did not pass allocation checks within {args.startup_timeout:g} seconds")
    except BaseException as error:
        try:
            terminate_owned_process(state)
        except Exception:
            pass
        state["status"] = "error"
        state["error"] = f"{type(error).__name__}: {error}"
        state["finished_at"] = utc_now()
        write_json(state_path, state)
        raise
    finally:
        log_file.close()


def request_with_gates(
    profile: Dict[str, Any], state: Dict[str, Any], baseline: Dict[str, Any],
    length: int, samples: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Optional[str]]:
    outcome: Dict[str, Any] = {}
    failure: List[str] = []
    finished = threading.Event()

    def send() -> None:
        try:
            outcome["response"] = http_json(
                base_url(state) + "/completion",
                {
                    "prompt": [int(profile["probe"]["token_id"])] * length,
                    "n_predict": int(profile["probe"]["n_predict"]),
                    "temperature": 0,
                    "seed": 424242,
                    "cache_prompt": False,
                    "stream": False,
                },
                timeout=float(profile["probe"]["request_timeout_seconds"]),
            )
        except Exception as error:
            outcome["exception"] = f"{type(error).__name__}: {error}"
        finally:
            finished.set()

    worker = threading.Thread(target=send, daemon=True)
    worker.start()
    started = time.monotonic()
    interval = float(profile["probe"]["sample_interval_seconds"])
    timeout = float(profile["probe"]["request_timeout_seconds"])
    while not finished.wait(interval):
        sample = collect_snapshot(time.monotonic() - started)
        samples.append(sample)
        failures = gate_failures(profile, baseline, sample)
        if failures:
            failure.extend(failures)
            terminate_owned_process(state)
            break
        if time.monotonic() - started > timeout:
            failure.append(f"prompt stage exceeded hard timeout of {timeout:g} seconds")
            terminate_owned_process(state)
            break
    worker.join(timeout=5)
    if failure:
        return outcome, "; ".join(failure)
    final_sample = collect_snapshot(time.monotonic() - started)
    samples.append(final_sample)
    final_failures = gate_failures(profile, baseline, final_sample)
    if final_failures:
        terminate_owned_process(state)
        return outcome, "; ".join(final_failures)
    if "exception" in outcome:
        return outcome, outcome["exception"]
    return outcome, None


def probe_action(args: argparse.Namespace, profile: Dict[str, Any]) -> int:
    if sys.platform != "darwin":
        raise ContextError("probe telemetry requires macOS")
    state_path = args.state_dir.expanduser().resolve() / "state.json"
    state = read_json(state_path)
    if state.get("schema_version") != STATE_VERSION or state.get("status") != "pass":
        raise ContextError("probe requires a passing allocation state")
    verify_owned_process(state)
    log_text = Path(state["log_path"]).read_text(encoding="utf-8", errors="replace")
    failures = check_log(profile, log_text)
    if failures:
        raise ContextError("server startup proof failed: " + "; ".join(failures))
    props = http_json(base_url(state) + "/props")
    if props.get("default_generation_settings", {}).get("n_ctx") != 262144:
        raise ContextError("server no longer reports the native 262144 context")

    lengths = args.lengths or [int(item) for item in profile["probe"]["lengths"]]
    if lengths != sorted(set(lengths)) or not lengths or lengths[-1] >= 262144:
        raise ContextError("probe lengths must be unique, increasing, and below 262144")
    output_path = args.output.expanduser().resolve()
    result: Dict[str, Any] = {
        "schema_version": RESULT_VERSION,
        "status": "error",
        "started_at": utc_now(),
        "profile_path": state["profile_path"],
        "state_path": str(state_path),
        "server_pid": state["pid"],
        "configured_context_tokens": 262144,
        "stages": [],
    }
    baseline = collect_snapshot()
    result["baseline"] = baseline
    try:
        for length in lengths:
            metrics_before = parse_metrics(http_text(base_url(state) + "/metrics"))
            stage: Dict[str, Any] = {
                "prompt_tokens_requested": length,
                "started_at": utc_now(),
                "metrics_before": metrics_before,
                "telemetry": [],
            }
            result["stages"].append(stage)
            outcome, error = request_with_gates(profile, state, baseline, length, stage["telemetry"])
            stage["finished_at"] = utc_now()
            stage.update(outcome)
            if error:
                stage["status"] = "error"
                stage["error"] = error
                raise ContextError(f"prompt stage {length} failed: {error}")
            metrics_after = parse_metrics(http_text(base_url(state) + "/metrics"))
            stage["metrics_after"] = metrics_after
            observed = metrics_after.get("llamacpp:n_tokens_max", 0)
            response = outcome.get("response", {})
            evaluated = response.get("tokens_evaluated", response.get("timings", {}).get("prompt_n", 0))
            stage["prompt_tokens_evaluated"] = evaluated
            if observed < length or evaluated < length:
                stage["status"] = "error"
                raise ContextError(
                    f"prompt stage {length} lacks proof of evaluation "
                    f"(metrics n_tokens_max={observed}, response evaluated={evaluated})"
                )
            stage["status"] = "pass"
        result["status"] = "pass"
        result["maximum_prompt_tokens_validated"] = lengths[-1]
        return 0
    except ContextError as error:
        result["error"] = str(error)
        try:
            terminate_owned_process(state)
        except ContextError:
            pass
        return 3
    finally:
        result["finished_at"] = utc_now()
        write_json(output_path, result)
        if result["status"] != "pass":
            print("context probe failed: " + result.get("error", "unknown error"), file=sys.stderr)


def health_action(args: argparse.Namespace, profile: Dict[str, Any]) -> int:
    state = read_json(args.state_dir.expanduser().resolve() / "state.json")
    verify_owned_process(state)
    health = http_json(base_url(state) + "/health")
    props = http_json(base_url(state) + "/props")
    log_text = Path(state["log_path"]).read_text(encoding="utf-8", errors="replace")
    failures = check_log(profile, log_text)
    if health.get("status") != "ok" or props.get("default_generation_settings", {}).get("n_ctx") != 262144:
        failures.append("health or configured context does not match the profile")
    print(json.dumps({"health": health, "n_ctx": props.get("default_generation_settings", {}).get("n_ctx"), "failures": failures}, indent=2))
    return 0 if not failures else 3


def stop_action(args: argparse.Namespace) -> int:
    state_path = args.state_dir.expanduser().resolve() / "state.json"
    state = read_json(state_path)
    terminate_owned_process(state)
    state["status"] = "stopped"
    state["stopped_at"] = utc_now()
    write_json(state_path, state)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--deps-root", type=Path, default=DEFAULT_DEPS)
    sub = parser.add_subparsers(dest="action", required=True)

    sub.add_parser("status", help="print the research status and required backend capabilities")

    command = sub.add_parser("command", help="print the command only after all execution gates pass")
    command.add_argument("model_root", type=Path)
    command.add_argument("--host")
    command.add_argument("--port", type=int)

    allocate = sub.add_parser("allocate", help="start one owned server and stop at allocation/health proof")
    allocate.add_argument("model_root", type=Path)
    allocate.add_argument("--state-dir", type=Path, required=True)
    allocate.add_argument("--host")
    allocate.add_argument("--port", type=int)
    allocate.add_argument("--startup-timeout", type=float, default=600)
    allocate.add_argument("--sample-interval", type=float, default=0.25)

    health = sub.add_parser("health", help="repeat the non-generating allocation checks")
    health.add_argument("--state-dir", type=Path, required=True)

    probe = sub.add_parser("probe", help="run increasing token-ID prompts against an owned server")
    probe.add_argument("--state-dir", type=Path, required=True)
    probe.add_argument("--output", type=Path, required=True)
    probe.add_argument("--lengths", type=lambda value: [int(x) for x in value.split(",")])

    stop = sub.add_parser("stop", help="stop only the server recorded in a state directory")
    stop.add_argument("--state-dir", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        profile = load_profile(args.profile.expanduser().resolve())
        if args.action == "status":
            return status_action(profile)
        if args.action == "command":
            return command_action(args, profile)
        if args.action == "allocate":
            if args.startup_timeout <= 0 or args.sample_interval <= 0:
                raise ContextError("allocation timeout and sample interval must be positive")
            return allocate_action(args, profile)
        if args.action == "health":
            return health_action(args, profile)
        if args.action == "probe":
            return probe_action(args, profile)
        if args.action == "stop":
            return stop_action(args)
        raise ContextError("unknown action")
    except ContextError as error:
        print("error: " + str(error), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("error: interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
