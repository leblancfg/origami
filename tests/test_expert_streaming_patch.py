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
            "4f6814129139c2070bcf564bf4f109311802af122be61325c9765b219035f5fd",
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
            "expert_stream_layer->boundary = weights",
            "selected expert IDs were assigned away",
            "GGML_STATUS_ABORTED",
            "test_scheduler_id_remap_boundary",
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
        self.assertIn("--target test-expert-stream", text)
        self.assertIn("mktemp -d /private/tmp/origami-expert-graph-bootstrap", text)
        self.assertIn("expert-streaming-capabilities.json", text)
        self.assertNotIn("llama-cli --model", text)
        self.assertNotIn("llama-server --model", text)

    def test_capability_record_is_strict_and_does_not_claim_a_model_run(self):
        value = json.loads(CAPABILITIES.read_text())
        self.assertEqual(value["schema_version"], "origami.expert-streaming-capabilities.v2")
        self.assertEqual(value["strict_feature_flag"], "LLAMA_EXPLICIT_EXPERT_STREAMING=1")
        caps = value["capabilities"]
        self.assertTrue(caps["fixed_cache_buffer_assigned_before_scheduler_split"])
        self.assertTrue(caps["gguf_derived_exact_expert_slices"])
        self.assertFalse(caps["dynamic_tensor_buffer_substitution"])
        self.assertFalse(caps["mmap_routed_fallback"])
        self.assertFalse(caps["prefill"])
        self.assertFalse(value["validation"]["real_model_launch"])


if __name__ == "__main__":
    unittest.main()
