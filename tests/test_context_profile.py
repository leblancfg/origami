import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("origami_context", ROOT / "tools" / "origami_context.py")
CONTEXT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONTEXT)
PROFILE_PATH = ROOT / "config" / "qwen38-context-262144.json"


class ContextProfileTests(unittest.TestCase):
    def setUp(self):
        self.profile = CONTEXT.load_profile(PROFILE_PATH)

    def test_native_profile_has_fail_closed_runtime_flags(self):
        environment, command = CONTEXT.render_command(
            self.profile,
            Path("/tmp/llama-server"),
            Path("/tmp/model.gguf"),
        )
        rendered = "\0".join(command)
        for option, value in (
            ("--ctx-size", "262144"),
            ("--gpu-layers", "43"),
            ("--ubatch-size", "8"),
            ("--cache-type-k", "q8_0"),
            ("--cache-type-v", "q8_0"),
            ("--flash-attn", "on"),
            ("--parallel", "1"),
            ("--cache-ram", "0"),
            ("--ctx-checkpoints", "0"),
            ("--fit", "off"),
        ):
            self.assertIn(option + "\0" + value, rendered)
        for flag in (
            "--no-repack",
            "--no-context-shift",
            "--no-cache-prompt",
            "--no-cache-idle-slots",
            "--no-kv-unified",
            "--kv-offload",
        ):
            self.assertIn(flag, command)
        self.assertIn("^output=CPU", command)
        self.assertNotIn("--rope-scaling", command)
        self.assertNotIn("--rope-scale", command)
        self.assertEqual(environment, {
            "GGML_METAL_NO_RESIDENCY": "1",
            "LLAMA_MMAP_PREFETCH": "0",
            "LLAMA_QSA_SHARED_INPUTS": "1",
        })

    def test_q8_main_and_f16_key_only_indexer_ledger_is_byte_exact(self):
        memory = self.profile["memory"]
        ctx = memory["allocated_cells"]
        layers = memory["attention_layers"]

        def q8_row(elements):
            self.assertEqual(elements % 32, 0)
            return elements // 32 * 34

        attention = ctx * layers * (q8_row(512) + q8_row(512))
        indexer = ctx * layers * (2 * 128)
        self.assertEqual(attention, 3422552064)
        self.assertEqual(indexer, 805306368)
        self.assertEqual(memory["total_kv_bytes"], attention + indexer)
        self.assertEqual(memory["persistent_payload_bytes"], 4383965184)
        self.assertEqual(memory["unshared_qsa_graph_input_floor_bytes_at_ubatch_8"], 138412032)
        self.assertEqual(memory["shared_qsa_graph_input_bound_bytes_at_ubatch_8"], 11534336)
        self.assertEqual(memory["qsa_graph_input_reduction_bytes"], 126877696)
        self.assertEqual(memory["context_and_graph_input_lower_bound_bytes"], 4395499520)
        self.assertEqual(memory["pinned_default_32_checkpoint_copy_bytes"], 4190109696)

    def test_log_proof_requires_capacity_qsa_flash_and_safeguards(self):
        text = "\n".join(self.profile["server"]["required_log_markers"])
        self.assertEqual(CONTEXT.check_log(self.profile, text), [])
        missing = text.replace("n_ctx_seq             = 262144", "n_ctx_seq             = 16384")
        self.assertTrue(any("n_ctx_seq" in item for item in CONTEXT.check_log(self.profile, missing)))
        failed = text + "\nError: Compute error."
        self.assertTrue(any("forbidden" in item for item in CONTEXT.check_log(self.profile, failed)))

    def test_memory_and_swap_gates_are_growth_based(self):
        vm = {
            "page_size_bytes": 16384,
            "pages": {"swapouts": 100},
            "compressor_occupied_bytes": 1024,
        }
        baseline = {
            "memory_pressure_free_percent": 80,
            "swap": {"used_bytes": 8 * 1024**3},
            "vm": vm,
        }
        self.assertEqual(CONTEXT.gate_failures(self.profile, baseline, baseline), [])
        sample = json.loads(json.dumps(baseline))
        sample["swap"]["used_bytes"] += 1
        sample["vm"]["pages"]["swapouts"] += 1
        failures = CONTEXT.gate_failures(self.profile, baseline, sample)
        self.assertTrue(any("swap grew" in item for item in failures))
        self.assertTrue(any("swapouts grew" in item for item in failures))

    def test_parsers_match_macos_telemetry_and_prometheus(self):
        self.assertEqual(
            CONTEXT.parse_pressure("System-wide memory free percentage: 37%\n"),
            37,
        )
        swap = CONTEXT.parse_swapusage("total = 8.00G used = 7.25G free = 768.00M (encrypted)")
        self.assertEqual(swap["used_bytes"], int(7.25 * 1024**3))
        metrics = CONTEXT.parse_metrics(
            "# TYPE llamacpp:n_tokens_max counter\nllamacpp:n_tokens_max 980\n"
        )
        self.assertEqual(metrics["llamacpp:n_tokens_max"], 980)

    def test_pi_sample_is_deliberately_not_loadable(self):
        config = json.loads((ROOT / "config" / "pi-model-origami-262144.json").read_text())
        self.assertEqual(config["status"], "blocked-not-a-pi-configuration")
        self.assertEqual(config["proposed_context_window"], 262144)
        self.assertNotIn("providers", config)

    def test_yarn_targets_are_separate_and_unvalidated(self):
        index = json.loads((ROOT / "config" / "qwen38-context-yarn-research.json").read_text())
        self.assertEqual(index["status"], "research-only-not-launchable")
        self.assertEqual(len(index["profiles"]), 2)
        profiles = [json.loads((ROOT / item).read_text()) for item in index["profiles"]]
        target_500k, target_1m = profiles
        self.assertEqual(target_500k["official_recipe"]["context_tokens"], 524288)
        self.assertEqual(target_500k["official_recipe"]["yarn_factor"], 2.0)
        self.assertEqual(target_500k["memory_at_ubatch_32"]["all_q8_0_kv_bytes_unapproved"], 9412018176)
        self.assertEqual(target_500k["memory_at_ubatch_32"]["q8_0_main_f16_index_kv_bytes_candidate"], 11676942336)
        self.assertEqual(target_1m["official_recipe"]["context_tokens"], 1000000)
        self.assertEqual(target_1m["official_recipe"]["yarn_factor"], 4.0)
        self.assertEqual(target_1m["memory_at_ubatch_32"]["allocated_cells"], 1000192)
        self.assertEqual(target_1m["memory_at_ubatch_32"]["all_q8_0_kv_bytes_unapproved"], 17955446784)
        self.assertEqual(target_1m["memory_at_ubatch_32"]["q8_0_main_f16_index_kv_bytes_candidate"], 22276276224)
        for target in profiles:
            self.assertEqual(target["status"], "research-only-not-launchable")
            self.assertEqual(target["validated_prompt_tokens"], 0)
            self.assertIsNone(target["pi_context_window"])
            recipe = target["pinned_llama_cpp_argument_recipe"]
            tokens = target["official_recipe"]["context_tokens"]
            self.assertIn("qwen4exp.context_length=int:" + str(tokens), recipe)
            self.assertIn("--yarn-orig-ctx", recipe)

    def test_execution_gate_rejects_the_pinned_capability_set(self):
        pinned = {
            "schema_version": "origami.backend-capabilities.v1",
            "runtime_revision": self.profile["runtime_revision"],
            "capabilities": {"lazy_mmap_safeguards": "checked-in patch"},
        }
        failures = CONTEXT.capability_gate_failures(self.profile, pinned)
        names = {item["name"] for item in self.profile["execution_gate"]["required_capabilities"]}
        self.assertEqual(len(failures), len(names))
        for name in names:
            self.assertTrue(any(name in failure for failure in failures))

    def test_execution_gate_accepts_named_evidence_only_at_matching_revision(self):
        manifest = {
            "schema_version": "origami.backend-capabilities.v1",
            "runtime_revision": self.profile["runtime_revision"],
            "capabilities": {
                item["name"]: item.get("upstream_commit", "local source and test evidence")
                for item in self.profile["execution_gate"]["required_capabilities"]
            },
        }
        self.assertEqual(CONTEXT.capability_gate_failures(self.profile, manifest), [])
        first = self.profile["execution_gate"]["required_capabilities"][0]
        manifest["capabilities"][first["name"]] = "wrong evidence"
        self.assertTrue(any("exact evidence" in item for item in CONTEXT.capability_gate_failures(self.profile, manifest)))
        manifest["capabilities"][first["name"]] = first["upstream_commit"]
        manifest["runtime_revision"] = "wrong"
        self.assertTrue(any("revision" in item for item in CONTEXT.capability_gate_failures(self.profile, manifest)))

    def test_pinned_server_help_contains_every_profile_flag_when_available(self):
        revision = self.profile["runtime_revision"]
        binary = Path("/private/tmp/origami-deps") / f"llama.cpp-{revision}" / "build-origami" / "bin" / "llama-server"
        if not binary.is_file():
            self.skipTest("pinned llama-server is not built")
        completed = subprocess.run(
            [str(binary), "--help"], text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, timeout=20,
        )
        self.assertEqual(completed.returncode, 0)
        flags = {item for item in self.profile["server"]["arguments"] if item.startswith("-")}
        self.assertEqual(sorted(flag for flag in flags if flag not in completed.stdout), [])

    def test_help_and_command_render_are_non_invasive(self):
        completed = subprocess.run(
            [sys.executable, str(ROOT / "tools" / "origami_context.py"), "--help"],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("status", completed.stdout)
        self.assertIn("allocate", completed.stdout)
        self.assertIn("probe", completed.stdout)

        status = subprocess.run(
            [sys.executable, str(ROOT / "tools" / "origami_context.py"), "status"],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10,
        )
        self.assertEqual(status.returncode, 0, status.stderr)
        value = json.loads(status.stdout)
        self.assertEqual(value["status"], "candidate-ready-for-allocation")
        self.assertEqual(value["runtime_revision"], "213df585b9aed6a09be30d8401f267bf603c104c")
        self.assertEqual(value["validated_prompt_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
