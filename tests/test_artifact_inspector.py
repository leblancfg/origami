import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from origami_artifacts.gguf import (
    GGUFError,
    inspect_artifact,
    parse_shard,
    parse_size,
    probe_tensor_spans,
    tensor_nbytes,
)


ALIGNMENT = 32


def _string(value):
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _value(type_code, value):
    formats = {0: "B", 2: "H", 4: "I", 5: "i", 6: "f", 7: "B", 10: "Q", 11: "q", 12: "d"}
    if type_code == 8:
        return _string(value)
    if type_code == 9:
        element_type, items = value
        return struct.pack("<IQ", element_type, len(items)) + b"".join(
            _value(element_type, item) for item in items
        )
    return struct.pack("<" + formats[type_code], value)


def write_gguf(path, metadata, tensors, include_body=True):
    content = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata)))
    for key, (type_code, value) in metadata:
        content += _string(key) + struct.pack("<I", type_code) + _value(type_code, value)
    for tensor in tensors:
        content += _string(tensor["name"])
        content += struct.pack("<I", len(tensor["dimensions"]))
        content += struct.pack("<" + "Q" * len(tensor["dimensions"]), *tensor["dimensions"])
        content += struct.pack("<IQ", tensor["type"], tensor["offset"])
    content += b"\0" * ((-len(content)) % ALIGNMENT)
    if include_body and tensors:
        body_size = max(
            tensor["offset"]
            + (
                tensor_nbytes(tensor["dimensions"], tensor["type"], tensor["name"])
                + ALIGNMENT - 1
            ) // ALIGNMENT * ALIGNMENT
            for tensor in tensors
        )
        content += bytes((index % 251) + 1 for index in range(body_size))
    Path(path).write_bytes(content)


def fixture_metadata(shard_number, tensor_count=6):
    values = [
        ("split.no", (2, shard_number)),
        ("split.tensors.count", (5, tensor_count)),
        ("split.count", (2, 2)),
    ]
    if shard_number == 0:
        values = [
            ("general.architecture", (8, "qwen4exp")),
            ("general.name", (8, "synthetic")),
            ("general.alignment", (4, ALIGNMENT)),
            ("qwen4exp.block_count", (4, 1)),
            ("qwen4exp.embedding_length", (4, 256)),
            ("qwen4exp.expert_count", (4, 2)),
            ("qwen4exp.expert_feed_forward_length", (4, 32)),
            ("qwen4exp.ple.layers", (9, (4, [0]))),
        ] + values
    return values


def fixture_tensors():
    definitions = [
        ("blk.0.attn_norm.weight", (256,), 0),
        ("blk.0.ffn_gate_shexp.weight", (256,), 0),
        ("blk.0.ffn_gate_exps.weight", (256, 32, 2), 19),
        ("blk.0.ffn_up_exps.weight", (256, 32, 2), 16),
        ("blk.0.ffn_down_exps.weight", (32, 256, 2), 20),
        ("per_layer_token_embd.weight", (160, 5), 20),
    ]
    tensors = []
    offset = 0
    for name, dimensions, type_code in definitions:
        offset = (offset + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
        tensors.append(
            {"name": name, "dimensions": dimensions, "type": type_code, "offset": offset}
        )
        offset += tensor_nbytes(dimensions, type_code, name)
    return tensors


class ArtifactInspectorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.metadata_path = self.directory / "model-00001-of-00002.gguf"
        self.tensor_path = self.directory / "model-00002-of-00002.gguf"
        write_gguf(self.metadata_path, fixture_metadata(0), [])
        write_gguf(self.tensor_path, fixture_metadata(1), fixture_tensors())

    def tearDown(self):
        self.temp.cleanup()

    def test_split_inventory_ledger_and_expert_slices(self):
        report = inspect_artifact(
            [self.directory],
            expert_cache_bytes=parse_size("1GB"),
            ple_cache_bytes=parse_size("1GiB"),
            temporary_bytes=parse_size("0.5GB"),
        )
        self.assertTrue(report["complete"])
        self.assertEqual(report["split"]["declared_tensor_count"], 6)
        self.assertEqual(report["inventory"]["tensor_count"], 6)
        self.assertEqual(
            report["inventory"]["bytes"],
            {"dense": 1024, "shared": 1024, "routed": 16640, "ple": 450, "total": 19138},
        )
        self.assertEqual(report["inventory"]["expert_slice_count"], 6)
        slices = {
            (item["expert"], item["projection"]): item
            for item in report["expert_slices"]
        }
        self.assertEqual(slices[(0, "gate")]["length"], 1600)
        self.assertEqual(slices[(1, "gate")]["offset"], slices[(0, "gate")]["end"])
        self.assertEqual(slices[(0, "up")]["length"], 2112)
        self.assertEqual(slices[(0, "down")]["length"], 4608)
        self.assertEqual(report["ledger"]["resident_weight_bytes"], 2048)
        self.assertEqual(report["ledger"]["streamed_weight_bytes"], 17090)
        self.assertEqual(report["ledger"]["cache_budget_bytes"], 1_000_000_000 + 1024 ** 3)
        self.assertEqual(report["ledger"]["temporary_budget_bytes"], 500_000_000)
        self.assertTrue(all(shard["body_complete"] for shard in report["shards"]))

    def test_probe_reads_only_bounded_span_edges(self):
        report = inspect_artifact([self.directory])
        result = probe_tensor_spans(report, ["blk.0.ffn_gate_exps.weight"], edge_bytes=7)[0]
        self.assertTrue(result["ok"])
        self.assertEqual(result["reads"][0][1], 7)
        self.assertEqual(result["reads"][1][1], 7)
        self.assertEqual(result["span"][1] - result["span"][0], 3200)

    def test_truncated_body_is_reported_and_probe_fails_closed(self):
        last = fixture_tensors()[-1]
        last_length = tensor_nbytes(last["dimensions"], last["type"], last["name"])
        trailing_padding = (-last_length) % ALIGNMENT
        with self.tensor_path.open("r+b") as handle:
            handle.truncate(self.tensor_path.stat().st_size - trailing_padding - 1)
        report = inspect_artifact([self.directory])
        self.assertFalse(report["shards"][1]["body_complete"])
        with self.assertRaisesRegex(GGUFError, "exceeds shard size"):
            probe_tensor_spans(report, ["per_layer_token_embd.weight"])

    def test_incomplete_metadata_shard_reports_declared_inventory(self):
        report = inspect_artifact([self.metadata_path])
        self.assertFalse(report["complete"])
        self.assertEqual(report["split"]["missing"], [1])
        self.assertEqual(report["split"]["declared_tensor_count"], 6)
        self.assertEqual(report["split"]["parsed_tensor_count"], 0)
        self.assertEqual(report["ledger"]["scope"], "present metadata shards only")

    def test_rejects_unsupported_quantization_and_layouts(self):
        bad_type = fixture_tensors()
        bad_type[0] = dict(bad_type[0], type=1)
        write_gguf(self.tensor_path, fixture_metadata(1), bad_type, include_body=False)
        with self.assertRaisesRegex(GGUFError, "unsupported GGML type 1"):
            inspect_artifact([self.directory])

        bad_layout = fixture_tensors()
        bad_layout[2] = dict(bad_layout[2], dimensions=(256, 64, 1))
        write_gguf(self.tensor_path, fixture_metadata(1), bad_layout, include_body=False)
        with self.assertRaisesRegex(GGUFError, "layout"):
            inspect_artifact([self.directory])

    def test_rejects_overlapping_spans_and_short_metadata(self):
        overlapping = fixture_tensors()
        overlapping[1] = dict(overlapping[1], offset=overlapping[0]["offset"])
        write_gguf(self.tensor_path, fixture_metadata(1), overlapping)
        with self.assertRaisesRegex(GGUFError, "expected contiguous GGUF offset"):
            inspect_artifact([self.directory])
        with self.assertRaisesRegex(GGUFError, "read limit"):
            parse_shard(self.metadata_path, max_metadata_bytes=16)

    def test_quantized_block_sizes_match_current_ggml_traits(self):
        expected = {
            0: (1, 4),
            8: (32, 34),
            12: (256, 144),
            13: (256, 176),
            14: (256, 210),
            16: (256, 66),
            19: (256, 50),
            20: (32, 18),
            30: (1, 2),
        }
        for type_code, (elements, byte_count) in expected.items():
            with self.subTest(type_code=type_code):
                self.assertEqual(tensor_nbytes((elements,), type_code), byte_count)
                self.assertEqual(tensor_nbytes((elements, 3), type_code), 3 * byte_count)

    def test_rejects_noncanonical_split_types_offsets_and_overflow(self):
        wrong_type = fixture_metadata(1)
        wrong_type[1] = ("split.tensors.count", (10, 6))
        write_gguf(self.tensor_path, wrong_type, fixture_tensors())
        with self.assertRaisesRegex(GGUFError, "split.tensors.count has type 10; expected 5"):
            inspect_artifact([self.directory])

        write_gguf(self.tensor_path, fixture_metadata(1), fixture_tensors())
        malformed = fixture_tensors()
        malformed[1] = dict(malformed[1], offset=malformed[1]["offset"] + ALIGNMENT)
        write_gguf(self.tensor_path, fixture_metadata(1), malformed, include_body=False)
        with self.assertRaisesRegex(GGUFError, "expected contiguous GGUF offset"):
            inspect_artifact([self.directory])

        with self.assertRaisesRegex(GGUFError, "element count overflows"):
            tensor_nbytes(((1 << 63) - 1,), 0, "huge")
        with self.assertRaisesRegex(GGUFError, "cache budget overflows"):
            inspect_artifact(
                [self.metadata_path],
                expert_cache_bytes=(1 << 63) - 1,
                ple_cache_bytes=1,
            )

    def test_partial_tensor_shard_infers_expert_axis(self):
        report = inspect_artifact([self.tensor_path])
        self.assertFalse(report["complete"])
        self.assertEqual(report["split"]["missing"], [0])
        self.assertEqual(report["inventory"]["expert_slice_count"], 6)
        self.assertEqual({item["expert"] for item in report["expert_slices"]}, {0, 1})

    def test_decimal_gb_and_binary_gib_are_distinct(self):
        self.assertEqual(parse_size("1GB"), 1_000_000_000)
        self.assertEqual(parse_size("1GiB"), 1_073_741_824)
        self.assertEqual(parse_size("0.5 GB"), 500_000_000)
        self.assertEqual(parse_size("0.5GiB"), 536_870_912)
        with self.assertRaises(GGUFError):
            parse_size("1G")

    def test_cli_emits_json(self):
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "origami_artifacts",
                str(self.directory),
                "--json",
                "--no-slices",
                "--expert-cache",
                "1GB",
            ],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
        report = json.loads(completed.stdout)
        self.assertEqual(report["inventory"]["bytes"]["total"], 19138)
        self.assertEqual(report["ledger"]["cache_budget_bytes"], 1_000_000_000)
        self.assertEqual(report["expert_slices"], [])


class RealMetadataRegressionTest(unittest.TestCase):
    PATH = Path(
        "/Users/leblancfg/src/github.com/leblancfg/origami/models/"
        "unsloth-Qwen3.8-Flash-Next-GGUF/"
        "d3bc75ee6ccef3efc1e228ec00a6cc2cdb1e2249/UD-IQ1_S/"
        "Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf"
    )

    @unittest.skipUnless(PATH.is_file(), "real first metadata shard is not available")
    def test_real_first_shard_declares_expected_inventory_without_body_reads(self):
        self.assertEqual(self.PATH.stat().st_size, 10_946_624)
        shard = parse_shard(self.PATH)
        self.assertEqual(shard.metadata_bytes_read, 10_946_618)
        self.assertEqual(shard.data_offset, self.PATH.stat().st_size)
        self.assertEqual(len(shard.tensors), 0)
        self.assertEqual(shard.metadata["split.no"], 0)
        self.assertEqual(shard.metadata["split.count"], 3)
        self.assertEqual(shard.metadata["split.tensors.count"], 1224)
        self.assertEqual(shard.metadata["qwen4exp.block_count"], 48)
        self.assertEqual(shard.metadata["qwen4exp.expert_count"], 512)
        ple_vocab_rows = sum(shard.metadata["qwen4exp.ple.head_vocab_sizes"])
        self.assertEqual(ple_vocab_rows, 320_001_446)
        # The owning shard's known tensor extent includes 90 padding rows.
        known_ple_tensor_rows = 320_001_536
        self.assertEqual(
            tensor_nbytes(
                (
                    shard.metadata["qwen4exp.embedding_length_per_layer_input"],
                    known_ple_tensor_rows,
                ),
                20,
                "per_layer_token_embd.weight",
            ),
            28_800_138_240,
        )

        report = inspect_artifact([self.PATH], include_slices=False)
        self.assertFalse(report["complete"])
        self.assertEqual(report["split"]["missing"], [1, 2])
        self.assertEqual(report["split"]["declared_tensor_count"], 1224)
        self.assertEqual(report["split"]["parsed_tensor_count"], 0)


if __name__ == "__main__":
    unittest.main()
