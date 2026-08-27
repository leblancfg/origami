import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from origami_artifacts.gguf import GGUFError
from origami_artifacts.sidecar import (
    PACK_HEADER_BYTES,
    PACK_MAGIC,
    SCHEMA_VERSION,
    SidecarReader,
    benchmark_sidecar,
    load_sidecar_index,
    pack_sidecar,
    plan_sidecar,
    verify_sidecar,
    write_index_only,
)
from tests.test_artifact_inspector import fixture_metadata, fixture_tensors, write_gguf


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            data = handle.read(4096)
            if not data:
                return digest.hexdigest()
            digest.update(data)


class ExpertSidecarTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.metadata_path = self.directory / "model-00001-of-00002.gguf"
        self.tensor_path = self.directory / "model-00002-of-00002.gguf"
        write_gguf(self.metadata_path, fixture_metadata(0), [])
        write_gguf(self.tensor_path, fixture_metadata(1), fixture_tensors())
        self.manifest_path = self.directory / "manifest.json"
        self.manifest_path.write_text(
            json.dumps(
                {
                    "model": {"revision": "synthetic-revision"},
                    "shards": [
                        {
                            "path": path.name,
                            "size_bytes": path.stat().st_size,
                            "sha256": sha256(path),
                        }
                        for path in (self.metadata_path, self.tensor_path)
                    ],
                }
            ),
            encoding="utf-8",
        )
        self.data_path = self.directory / "experts.oxp"
        self.index_path = self.directory / "experts.index.json"

    def tearDown(self):
        self.temp.cleanup()

    def plan(self, **kwargs):
        return plan_sidecar(
            [self.directory],
            source_revision="synthetic-revision",
            source_manifest=self.manifest_path,
            chunk_bytes=4096,
            **kwargs,
        )

    def test_index_only_preserves_layout_and_plans_aligned_records(self):
        plan = self.plan()
        self.assertEqual(plan["schema_version"], SCHEMA_VERSION)
        self.assertEqual(plan["record_count"], 2)
        self.assertEqual(plan["projection_count"], 6)
        self.assertEqual(plan["logical_bytes"], 16640)
        self.assertEqual(plan["packed_bytes"], 36864)
        self.assertEqual(plan["source"]["identity_mode"], "manifest-sha256")
        self.assertTrue(all(not shard["sha256_verified"] for shard in plan["source"]["shards"]))

        record = plan["records"][0]
        self.assertEqual((record["layer"], record["expert"]), (0, 0))
        self.assertEqual(record["offset"], PACK_HEADER_BYTES)
        self.assertEqual(record["read_length"], 16384)
        self.assertEqual([item["projection"] for item in record["projections"]], ["gate", "up", "down"])
        self.assertTrue(all(item["sidecar_offset"] % 4096 == 0 for item in record["projections"]))
        gate = record["projections"][0]
        self.assertEqual(gate["type"], "IQ1_S")
        self.assertEqual(gate["type_code"], 19)
        self.assertEqual(gate["tensor_dimensions"], [256, 32, 2])
        self.assertEqual(gate["slice_dimensions"], [256, 32])
        self.assertEqual(gate["quant_block_elements"], 256)
        self.assertEqual(gate["quant_block_bytes"], 50)
        self.assertEqual(gate["length"], 1600)
        self.assertEqual(gate["source_slice_relative_offset"], 0)
        second_gate = plan["records"][1]["projections"][0]
        self.assertEqual(second_gate["source_tensor_offset"], gate["source_tensor_offset"])
        self.assertEqual(second_gate["source_slice_relative_offset"], gate["length"])

        output = write_index_only(plan, self.index_path)
        self.assertEqual(output["mode"], "index-only")
        self.assertTrue(self.index_path.is_file())
        self.assertFalse(self.data_path.exists())
        self.assertEqual(load_sidecar_index(self.index_path)["plan_sha256"], plan["plan_sha256"])

    def test_pack_is_byte_exact_and_reader_uses_one_grouped_record(self):
        packed = pack_sidecar(
            self.plan(), self.data_path, self.index_path,
            chunk_bytes=4096, checkpoint_records=1,
        )
        self.assertEqual(packed["mode"], "packed")
        self.assertEqual(self.data_path.stat().st_size, 36864)
        self.assertEqual(sha256(self.data_path), packed["data"]["sha256"])
        packed_bytes = self.data_path.read_bytes()
        self.assertEqual(packed_bytes[:8], PACK_MAGIC)
        payload_ranges = [
            (projection["sidecar_offset"], projection["sidecar_offset"] + projection["length"])
            for record in packed["records"]
            for projection in record["projections"]
        ]
        for start, end in zip(
            [PACK_HEADER_BYTES] + [item[1] for item in payload_ranges],
            [item[0] for item in payload_ranges] + [len(packed_bytes)],
        ):
            self.assertFalse(any(packed_bytes[start:end]))

        source_fd = os.open(str(self.tensor_path), os.O_RDONLY)
        data_fd = os.open(str(self.data_path), os.O_RDONLY)
        try:
            for record in packed["records"]:
                for projection in record["projections"]:
                    expected = os.pread(source_fd, projection["length"], projection["source_offset"])
                    actual = os.pread(data_fd, projection["length"], projection["sidecar_offset"])
                    self.assertEqual(actual, expected)
                    self.assertEqual(hashlib.sha256(actual).hexdigest(), projection["sha256"])
        finally:
            os.close(source_fd)
            os.close(data_fd)

        with SidecarReader(self.index_path) as reader:
            record = reader.record(0, 1)
            buffer = bytearray(reader.max_read_length)
            self.assertEqual(reader.read_record_into(0, 1, buffer), record["read_length"])
            expert = reader.read_expert(0, 1)
            self.assertEqual(set(expert), {"gate", "up", "down"})
            for projection in record["projections"]:
                expected = self.tensor_path.read_bytes()[
                    projection["source_offset"]:projection["source_offset"] + projection["length"]
                ]
                self.assertEqual(expert[projection["projection"]], expected)

    def test_interrupted_pack_resumes_to_identical_output(self):
        plan = self.plan()
        with self.assertRaisesRegex(RuntimeError, "test interruption"):
            pack_sidecar(
                plan, self.data_path, self.index_path,
                chunk_bytes=4096, checkpoint_records=1,
                _interrupt_after_records=1,
            )
        self.assertFalse(self.data_path.exists())
        self.assertFalse(self.index_path.exists())
        self.assertTrue(Path(str(self.data_path) + ".partial").is_file())
        self.assertTrue(Path(str(self.data_path) + ".journal").is_file())

        resumed = pack_sidecar(
            plan, self.data_path, self.index_path,
            chunk_bytes=4096, checkpoint_records=1,
        )
        resumed_bytes = self.data_path.read_bytes()

        second_data = self.directory / "fresh.oxp"
        second_index = self.directory / "fresh.index.json"
        fresh = pack_sidecar(
            plan, second_data, second_index,
            chunk_bytes=4096, checkpoint_records=2,
        )
        self.assertEqual(resumed_bytes, second_data.read_bytes())
        self.assertEqual(resumed["data"]["sha256"], fresh["data"]["sha256"])
        self.assertFalse(Path(str(self.data_path) + ".state.json").exists())
        self.assertFalse(Path(str(self.data_path) + ".journal").exists())

    def test_sample_full_verification_corruption_and_benchmark(self):
        pack_sidecar(self.plan(), self.data_path, self.index_path, chunk_bytes=4096)
        sample = verify_sidecar(self.index_path, sample_count=1, chunk_bytes=4096)
        self.assertEqual(sample["mode"], "sample")
        self.assertEqual(sample["records_verified"], 1)
        self.assertEqual(sample["projections_verified"], 3)
        full = verify_sidecar(self.index_path, full=True, chunk_bytes=4096)
        self.assertEqual(full["records_verified"], 2)
        self.assertTrue(full["data_sha256_verified"])

        benchmark = benchmark_sidecar(
            self.index_path, request_count=3, warmup=1, pattern="random", seed=7
        )
        self.assertEqual(benchmark["requests"], 3)
        self.assertEqual(benchmark["bytes_read"], 3 * 16384)
        self.assertEqual(benchmark["read_api"], "pread-aligned-record-into-reused-buffer")

        packed = load_sidecar_index(self.index_path)
        offset = packed["records"][0]["projections"][0]["sidecar_offset"]
        with self.data_path.open("r+b") as handle:
            handle.seek(offset)
            original = handle.read(1)
            handle.seek(offset)
            handle.write(bytes([original[0] ^ 0xFF]))
        with self.assertRaisesRegex(GGUFError, "byte mismatch"):
            verify_sidecar(self.index_path, sample_count=1, chunk_bytes=4096)

    def test_source_identity_and_manifest_fail_closed(self):
        with self.assertRaisesRegex(GGUFError, "manifest is required"):
            plan_sidecar([self.directory], source_revision="synthetic-revision")
        local = plan_sidecar(
            [self.directory], source_revision="local", allow_stat_identity=True,
            chunk_bytes=4096,
        )
        self.assertEqual(local["source"]["identity_mode"], "local-stat+header-sha256")

        bad_manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        bad_manifest["model"]["revision"] = "wrong"
        self.manifest_path.write_text(json.dumps(bad_manifest), encoding="utf-8")
        with self.assertRaisesRegex(GGUFError, "revision"):
            self.plan()

        self.manifest_path.write_text(
            json.dumps(
                {
                    "model": {"revision": "synthetic-revision"},
                    "shards": [
                        {
                            "path": path.name,
                            "size_bytes": path.stat().st_size,
                            "sha256": sha256(path),
                        }
                        for path in (self.metadata_path, self.tensor_path)
                    ],
                }
            ),
            encoding="utf-8",
        )
        plan = self.plan()
        with self.tensor_path.open("r+b") as handle:
            handle.seek(-1, os.SEEK_END)
            value = handle.read(1)
            handle.seek(-1, os.SEEK_END)
            handle.write(bytes([value[0] ^ 1]))
        with self.assertRaisesRegex(GGUFError, "local identity changed"):
            pack_sidecar(plan, self.data_path, self.index_path, chunk_bytes=4096)

    def test_cli_index_only_reads_headers_and_writes_no_pack(self):
        completed = subprocess.run(
            [
                sys.executable, "-m", "origami_artifacts.sidecar", "index",
                str(self.directory), "--source-revision", "synthetic-revision",
                "--source-manifest", str(self.manifest_path), "--chunk-size", "4KiB",
                "--index", str(self.index_path),
            ],
            cwd=ROOT, check=True, capture_output=True, text=True,
        )
        summary = json.loads(completed.stdout)
        self.assertEqual(summary["mode"], "index-only")
        self.assertEqual(summary["records"], 2)
        self.assertTrue(self.index_path.exists())
        self.assertFalse(self.data_path.exists())


if __name__ == "__main__":
    unittest.main()
