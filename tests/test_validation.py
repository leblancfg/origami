import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("origami_validate", ROOT / "tools" / "origami_validate.py")
VALIDATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATE)


class ParserTests(unittest.TestCase):
    def test_parse_llama_cpp_timings(self):
        stderr = """
llama_print_timings: prompt eval time =  80.00 ms / 10 tokens ( 8.00 ms per token, 125.00 tokens per second)
llama_perf_context_print: eval time = 120.00 ms / 4 tokens (30.00 ms per token, 33.33 tokens per second)
"""
        parsed = VALIDATE.parse_timings(stderr)
        self.assertEqual(parsed["prefill"]["tokens"], 10)
        self.assertEqual(parsed["decode"]["elapsed_ms"], 120.0)

    def test_parse_vm_and_swap(self):
        vm = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free: 10.
Pages occupied by compressor: 3.
Swapins: 7.
"""
        parsed = VALIDATE.parse_vm_stat(vm)
        self.assertEqual(parsed["compressor_occupied_bytes"], 3 * 16384)
        swap = VALIDATE.parse_swapusage("total = 4.00G used = 1.50G free = 2.50G (encrypted)")
        self.assertEqual(swap["used_bytes"], int(1.5 * 1024 ** 3))

    def test_manifest_rejects_invalid_identity_unknown_fields_and_aliases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard = root / "model.gguf"
            shard.write_bytes(b"1234")
            base = {
                "schema_version": VALIDATE.MANIFEST_VERSION,
                "model": {"id": "test/model", "revision": "abc", "format": "GGUF", "quantization": "test"},
                "entrypoint": shard.name,
                "shards": [{"path": shard.name, "size_bytes": 4}],
            }
            path = root / "manifest.json"

            invalid = dict(base, model=dict(base["model"], revision=123))
            path.write_text(json.dumps(invalid))
            with self.assertRaisesRegex(VALIDATE.ValidationError, "non-empty strings"):
                VALIDATE.validate_manifest(path, False)

            invalid = dict(base, unexpected=True)
            path.write_text(json.dumps(invalid))
            with self.assertRaisesRegex(VALIDATE.ValidationError, "unknown model manifest fields"):
                VALIDATE.validate_manifest(path, False)

            invalid = dict(base, shards=[
                {"path": shard.name, "size_bytes": 4},
                {"path": "./" + shard.name, "size_bytes": 4},
            ])
            path.write_text(json.dumps(invalid))
            with self.assertRaisesRegex(VALIDATE.ValidationError, "duplicate shard path"):
                VALIDATE.validate_manifest(path, False)

    def test_manifest_requires_one_split_family_and_first_entrypoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = ["a-00001-of-00002.gguf", "b-00002-of-00002.gguf"]
            for name in names:
                (root / name).write_bytes(b"x")
            manifest = {
                "schema_version": VALIDATE.MANIFEST_VERSION,
                "model": {"id": "test/model", "revision": "abc", "format": "GGUF", "quantization": "test"},
                "entrypoint": names[1],
                "shards": [{"path": name, "size_bytes": 1} for name in names],
            }
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(VALIDATE.ValidationError, "names are inconsistent"):
                VALIDATE.validate_manifest(path, False)

            names[1] = "a-00002-of-00002.gguf"
            (root / names[1]).write_bytes(b"x")
            manifest["entrypoint"] = names[1]
            manifest["shards"] = [{"path": name, "size_bytes": 1} for name in names]
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(VALIDATE.ValidationError, "first split shard"):
                VALIDATE.validate_manifest(path, False)

    def test_process_tree_rss_includes_reparented_process_group_member(self):
        output = "10 1 10 100\n11 1 10 200\n12 10 99 300\n"
        completed = subprocess.CompletedProcess([], 0, output, "")
        with mock.patch.object(VALIDATE, "run_command", return_value=completed):
            sample = VALIDATE.process_tree_rss(10)
        self.assertEqual(sample["root_rss_bytes"], 100 * 1024)
        self.assertEqual(sample["process_tree_rss_bytes"], 600 * 1024)
        self.assertEqual(sample["process_count"], 3)

    def test_manifest_can_resolve_checked_in_manifest_from_model_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.gguf").write_bytes(b"1234")
            manifest = {
                "schema_version": VALIDATE.MANIFEST_VERSION,
                "model": {"id": "test/model", "revision": "abc", "format": "GGUF", "quantization": "test"},
                "entrypoint": "model.gguf",
                "shards": [{"path": "model.gguf", "size_bytes": 4}],
            }
            manifest_path = root.parent / (root.name + "-manifest.json")
            manifest_path.write_text(json.dumps(manifest))
            try:
                parsed, entrypoint = VALIDATE.validate_manifest(manifest_path, False, root)
            finally:
                manifest_path.unlink()
            self.assertEqual(entrypoint, (root / "model.gguf").resolve())
            self.assertEqual(parsed["total_actual_size_bytes"], 4)

    def test_checked_in_model_manifest_matches_schema_contract_and_pins(self):
        manifest = json.loads((ROOT / "config" / "qwen38-flash-next-ud-iq1_s.json").read_text())
        self.assertEqual(set(manifest), {"schema_version", "model", "entrypoint", "shards"})
        self.assertEqual(manifest["schema_version"], VALIDATE.MANIFEST_VERSION)
        self.assertEqual(manifest["model"]["id"], "unsloth/Qwen3.8-Flash-Next-GGUF")
        self.assertEqual(manifest["model"]["revision"], "d3bc75ee6ccef3efc1e228ec00a6cc2cdb1e2249")
        self.assertEqual(
            [item["size_bytes"] for item in manifest["shards"]],
            [10946624, 49990818368, 22544696352],
        )
        self.assertEqual(
            [item["sha256"] for item in manifest["shards"]],
            [
                "88a1420825a9304063e882ada29d438263617f51ac8923d438d927496693bafd",
                "3a62e35bbf9add4733bd1438ebd3a67649d5edd6cb0e72bb78e33c913992b2b6",
                "0e25ceaeb89b8a80aa973c6c0c7448943682f7408c2855b2ebd016b7643a861a",
            ],
        )

    def test_manifest_rejects_partial_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard = root / "model-00001-of-00001.gguf"
            shard.write_bytes(b"1234")
            Path(str(shard) + ".aria2").write_text("partial")
            manifest = {
                "schema_version": VALIDATE.MANIFEST_VERSION,
                "model": {"id": "test/model", "revision": "abc", "format": "GGUF", "quantization": "test"},
                "entrypoint": shard.name,
                "shards": [{"path": shard.name, "size_bytes": 4}],
            }
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(VALIDATE.ValidationError, "partial-download marker"):
                VALIDATE.validate_manifest(path, False)


@unittest.skipUnless(sys.platform == "darwin", "macOS telemetry integration")
class HarnessIntegrationTests(unittest.TestCase):
    def command(self, output, *extra):
        return [
            sys.executable,
            str(ROOT / "tools" / "origami_validate.py"),
            "--executable", str(ROOT / "tools" / "mock_llama_cli.py"),
            "--runtime-revision", "mock-v1",
            "--model-manifest", str(ROOT / "tests" / "fixtures" / "mock-model-manifest.json"),
            "--output", str(output),
            "--sample-interval", "0.1",
            *extra,
        ]

    def test_preflight_only_does_not_start_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "preflight.json"
            completed = subprocess.run(
                self.command(output, "--preflight-only", "--", "--mock-exit-code", "9"),
                text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads(output.read_text())
            self.assertEqual(result["status"], "pass")
            self.assertNotIn("run", result)

    def test_mock_run_parsing_and_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "result.json"
            environment = os.environ.copy()
            environment["TMPDIR"] = str(root)
            completed = subprocess.run(
                self.command(
                    output,
                    "--expected-output-sha256", "3b0c8ba590d96fdafce61f18ec139bcc6195dbf4bf69f22c3659448d43361c33",
                    "--verify-shards-sha256",
                    "--profile", "first-token",
                    "--lazy-mmap-safeguards",
                ),
                text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                env=environment, timeout=20,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads(output.read_text())
            self.assertEqual(result["status"], "pass")
            self.assertEqual(result["run"]["timings"]["prefill"]["tokens"], 10)
            self.assertEqual(result["smoke_test"]["n_predict"], 1)
            self.assertEqual(result["environment"], VALIDATE.SAFEGUARD_ENVIRONMENT)
            self.assertEqual(result["run"]["timings"]["decode"]["tokens"], 4)
            self.assertGreater(result["telemetry"]["summary"]["peak_process_tree_rss_bytes"], 0)
            self.assertTrue(result["model"]["shards"][0]["sha256_verified"])
            self.assertEqual(list(root.glob("origami-validation-*")), [])

    def test_orphaned_process_group_member_is_killed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "orphan.json"
            pid_file = root / "child.pid"
            completed = subprocess.run(
                self.command(
                    output, "--", "--mock-sleep", "0.1",
                    "--mock-orphan-pid-file", str(pid_file),
                ),
                text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            pid = int(pid_file.read_text())
            deadline = time.monotonic() + 2
            while True:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    break
                if time.monotonic() >= deadline:
                    self.fail("mock descendant survived harness cleanup")
                time.sleep(0.02)

    def test_timeout_kills_mock_and_cleans_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "timeout.json"
            environment = os.environ.copy()
            environment["TMPDIR"] = str(root)
            completed = subprocess.run(
                self.command(output, "--timeout", "0.2", "--", "--mock-sleep", "5"),
                text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                env=environment, timeout=20,
            )
            self.assertEqual(completed.returncode, 3)
            result = json.loads(output.read_text())
            self.assertTrue(result["run"]["timed_out"])
            self.assertEqual(list(root.glob("origami-validation-*")), [])


if __name__ == "__main__":
    unittest.main()
