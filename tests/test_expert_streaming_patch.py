import hashlib
import json
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REVISION = "213df585b9aed6a09be30d8401f267bf603c104c"
PATCH = ROOT / "patches" / f"llama.cpp-{REVISION}-explicit-expert-streaming.patch"
BOOTSTRAP = ROOT / "scripts" / "bootstrap-expert-streaming-llama-cpp.sh"
CAPABILITIES = ROOT / "validation" / "expert-streaming-capabilities.json"


class ExpertStreamingPatchTests(unittest.TestCase):
    def test_patch_is_pinned_and_limited_to_the_vertical_slice(self):
        data = PATCH.read_bytes()
        self.assertEqual(
            hashlib.sha256(data).hexdigest(),
            "d6a49170fdadd147896c4ab1e40402aaa3f7b5afaa647887a0a0075df4079848",
        )
        changed = {
            line.removeprefix("+++ b/")
            for line in data.decode().splitlines()
            if line.startswith("+++ b/")
        }
        self.assertEqual(changed, {
            "ggml/src/ggml-backend.cpp",
            "src/CMakeLists.txt",
            "src/llama-context.cpp",
            "src/llama-context.h",
            "src/llama-expert-stream.cpp",
            "src/llama-expert-stream.h",
            "src/llama-graph.cpp",
            "src/llama-graph.h",
            "src/llama-model-loader.cpp",
            "src/llama-model-loader.h",
            "src/llama-model.cpp",
            "src/llama-model.h",
            "src/models/models.h",
            "src/models/qwen4exp.cpp",
            "tests/CMakeLists.txt",
            "tests/test-expert-stream.cpp",
        })

    def test_patch_contains_the_safety_and_byte_contracts(self):
        text = PATCH.read_text()
        for marker in (
            "LLAMA_EXPLICIT_EXPERT_STREAMING",
            "std::strncmp(name, \"MTL\", 3)",
            "n_tokens != 1",
            "GGML_TYPE_IQ1_S",
            "GGML_TYPE_IQ2_XXS",
            "GGML_TYPE_IQ4_NL",
            "pread(fd",
            "ggml_backend_event_record",
            "all expert cache slots are owned by unfinished device work",
            "std::memcmp(view.gate.data",
            "std::memcmp(view.up.data",
            "std::memcmp(view.down.data",
            "routed tensors will not be mapped or loaded",
            "TENSOR_SKIP",
            "expert_stream_layer->boundary = selected_experts",
            "expert_stream_layer->routed_selected",
            "ggml_backend_sched_set_tensor_backend",
            "selected expert IDs were assigned away",
            "GGML_STATUS_ABORTED",
            "test_scheduler_separate_routed_ids",
        ):
            self.assertIn(marker, text)

    def test_bootstrap_is_valid_shell_and_does_not_launch_a_model(self):
        completed = subprocess.run(
            ["bash", "-n", str(BOOTSTRAP)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        text = BOOTSTRAP.read_text()
        self.assertIn("targets=(test-expert-stream)", text)
        self.assertIn("targets+=(llama-cli llama-server)", text)
        self.assertIn("ORIGAMI_BUILD_RUNTIME", text)
        self.assertIn("mktemp -d /private/tmp/origami-expert-graph-bootstrap", text)
        self.assertIn("expert-streaming-capabilities.json", text)
        self.assertNotIn("llama-cli --model", text)
        self.assertNotIn("llama-server --model", text)

    def test_capability_record_reports_strict_real_model_parity(self):
        value = json.loads(CAPABILITIES.read_text())
        self.assertEqual(value["schema_version"], "origami.expert-streaming-capabilities.v3")
        self.assertEqual(value["strict_feature_flag"], "LLAMA_EXPLICIT_EXPERT_STREAMING=1")
        caps = value["capabilities"]
        self.assertTrue(caps["fixed_cache_buffer_assigned_before_scheduler_split"])
        self.assertTrue(caps["gguf_derived_exact_expert_slices"])
        self.assertFalse(caps["dynamic_tensor_buffer_substitution"])
        self.assertFalse(caps["mmap_routed_fallback"])
        self.assertFalse(caps["prefill"])
        self.assertTrue(caps["original_ids_never_mutated"])
        self.assertTrue(caps["dedicated_remapped_id_inputs"])
        self.assertTrue(value["validation"]["real_model_launch"])
        self.assertTrue(value["validation"]["callback_tensor_byte_parity"])


if __name__ == "__main__":
    unittest.main()
