import hashlib
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REVISION = "213df585b9aed6a09be30d8401f267bf603c104c"
PATCH = ROOT / "patches" / f"llama.cpp-{REVISION}-explicit-expert-streaming.patch"
BOOTSTRAP = ROOT / "scripts" / "bootstrap-expert-streaming-llama-cpp.sh"


class ExpertStreamingPatchTests(unittest.TestCase):
    def test_patch_is_pinned_and_limited_to_the_vertical_slice(self):
        data = PATCH.read_bytes()
        self.assertEqual(
            hashlib.sha256(data).hexdigest(),
            "049c779efc9e1a8e5269995bcc4de4efb00bc5fd81ed7e8c6b3414014f57be54",
        )
        changed = {
            line.removeprefix("+++ b/")
            for line in data.decode().splitlines()
            if line.startswith("+++ b/")
        }
        self.assertEqual(changed, {
            "src/CMakeLists.txt",
            "src/llama-expert-stream.cpp",
            "src/llama-expert-stream.h",
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
        self.assertIn("vertical-slice-only-not-launchable", text)
        self.assertNotIn("llama-cli --model", text)
        self.assertNotIn("llama-server --model", text)


if __name__ == "__main__":
    unittest.main()
