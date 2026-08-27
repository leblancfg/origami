# Expert sidecar pack and index

`origami_artifacts.sidecar` repacks routed GGUF expert slices without decoding them. Each record contains one `(layer, expert)` gate/up/down triple. The record and every projection begin at an explicit-read alignment boundary. A cache can fetch a complete expert with one `pread`, then address the three quantized projections from offsets in the index.

## Commands

Create an index-only dry run. This parses and hashes GGUF headers, validates the pinned manifest, and does not read tensor bodies or create a pack file:

```sh
python3 -m origami_artifacts.sidecar index /path/to/UD-IQ1_S \
  --source-revision d3bc75ee6ccef3efc1e228ec00a6cc2cdb1e2249 \
  --source-manifest config/qwen38-flash-next-ud-iq1_s.json \
  --index experts.index-only.json
```

Pack or resume with an 8 MiB transfer ceiling and eight-record durability batches:

```sh
python3 -m origami_artifacts.sidecar pack /path/to/UD-IQ1_S \
  --source-revision d3bc75ee6ccef3efc1e228ec00a6cc2cdb1e2249 \
  --source-manifest config/qwen38-flash-next-ud-iq1_s.json \
  --data experts.oxp --index experts.index.json \
  --chunk-size 8MiB --checkpoint-records 8 --verify-samples 16
```

`--verify-full` compares all 73,728 projections with their GGUF slices and verifies the complete pack SHA-256. It does not dequantize. `--verify-source-sha256` separately reads all source shards and checks the manifest hashes; this is optional because it reads the dense and PLE data as well as the routed data.

Verification and read benchmarks can run later:

```sh
python3 -m origami_artifacts.sidecar verify experts.index.json --samples 32
python3 -m origami_artifacts.sidecar verify experts.index.json --full
python3 -m origami_artifacts.sidecar benchmark experts.index.json \
  --requests 1000 --warmup 16 --pattern random --seed 1
```

Use repeated `--source` options if the GGUF moved after packing. Verification checks shard number, basename, exact size, and header SHA-256 before comparing bytes.

## Binary pack format, version 1

All integers are unsigned little-endian. The first 4,096 bytes are:

| Offset | Size | Field |
|---:|---:|---|
| 0 | 8 | magic `ORIGXPK\0` |
| 8 | 4 | format version, `1` |
| 12 | 4 | header size, `4096` |
| 16 | 4 | alignment in bytes |
| 20 | 4 | flags, `0` |
| 24 | 8 | record count |
| 32 | 8 | logical projection bytes |
| 40 | 8 | total pack file bytes |
| 48 | 32 | binary SHA-256 of the canonical plan |
| 80 | 4016 | reserved zero bytes |

There are no host-sized fields and no binary record headers. Records are ordered by ascending `(layer, expert)`. Within each record, projections are ordered `gate`, `up`, `down`. Every record offset and projection offset is a multiple of the declared alignment, 4,096 by default. Gaps and trailing record padding are zero. Projection payloads are byte-for-byte copies of their GGUF slices. The index is the authority for offsets and lengths.

A record's `read_length` includes internal and trailing alignment padding. Reading `[record.offset, record.offset + record.read_length)` therefore fetches the full triple in one aligned operation. `logical_bytes` excludes all headers and padding; `packed_bytes` includes them.

## JSON index contract

The schema identifier is `origami.expert-sidecar.v1`. Top-level fields are:

- `mode`: `index-only` or `packed` in a published index.
- `alignment`, `record_count`, `projection_count`, `logical_bytes`, and `packed_bytes`.
- `plan_sha256`: SHA-256 of canonical compact JSON with sorted keys over the source identity, layout counts, and every record/projection metadata field. Publication mode, local paths, verification status, projection hashes, and packed-data metadata are excluded, so a resumed pack keeps the same plan identity.
- `source`: revision, identity mode, and numbered shard identities. Each shard records basename, exact size, header extent and SHA-256, local path/stat identity, and the manifest SHA-256 when supplied.
- `pack_format`: magic, version, header size, projection order, and padding byte.
- `records`: the complete read map.
- `data`: `null` for index-only output. A packed index records the pack path relative to the index, exact size, and SHA-256.

Each record has `layer`, `expert`, `offset`, `read_length`, `logical_bytes`, and three `projections`. Each projection preserves:

- projection and source tensor names;
- GGML type name/code and quantization block elements/bytes;
- full tensor dimensions and the two-dimensional expert-slice shape;
- source shard number, absolute tensor and slice offsets, GGUF-relative tensor offset, and slice-relative tensor offset;
- sidecar offset and exact length;
- exact projection SHA-256 after packing.

Offsets and counts are JSON integers. A future C++ reader does not need to interpret byte units, infer quantization sizes, or parse GGUF.

## Source identity and failure rules

The default path requires a source manifest whose revision equals `--source-revision` and whose shard basenames and exact sizes match the files. Manifest SHA-256 values are retained in the index. Planning hashes each GGUF header through its aligned data offset. Packing also captures device, inode, size, and nanosecond mtime, then checks that local identity and the header hash before and after copying. Every copied projection receives its own SHA-256.

`--verify-source-sha256` checks the complete manifest hashes. For unmanifested local fixtures, `--allow-stat-identity` explicitly selects `local-stat+header-sha256`; it is not the default. A changed revision, size, stat identity, header, slice, pack header, plan digest, or verified hash fails closed.

## Resume and atomic publication

Packing writes `DATA.partial`, `DATA.journal`, and `DATA.state.json`. The state binds the output path to `plan_sha256`. A journal line commits one completed record and its three hashes. At each checkpoint batch the pack is synced before journal entries are synced. Resume accepts only sequential valid journal lines, truncates uncommitted pack bytes, re-hashes the committed prefix, and continues at the next record. Source identity is checked again.

On completion the data file is synced and renamed to its final name. The packed JSON index is then written, synced, and atomically renamed. The index is the publication point. A crash between the two renames leaves recoverable data without a published index; rerunning completes publication. Existing published indexes are never overwritten, and an exclusive lock rejects concurrent packers.

Body transfer memory is bounded by `--chunk-size`; the reader and benchmark use one caller-owned buffer sized to the largest record. Planning memory is proportional to the 24,576-record index, not model bytes. No code path uses `mmap`.

## Python read API

```python
from origami_artifacts.sidecar import SidecarReader, benchmark_sidecar

with SidecarReader("experts.index.json") as reader:
    buffer = bytearray(reader.max_read_length)
    count = reader.read_record_into(layer=7, expert=123, buffer=buffer)
    metadata = reader.record(7, 123)

    # Convenience API; values remain quantized bytes.
    projections = reader.read_expert(7, 123)

result = benchmark_sidecar(
    "experts.index.json", request_count=1000,
    warmup=16, pattern="random", seed=1,
)
```

`read_record_into` performs an exact positional read into reusable storage and returns the record byte count. The index exposes the relative projection offsets needed to point a future C++ cache entry at gate, up, and down bytes.

## Real-header dry-run evidence

The index-only command above was run against the three pinned `UD-IQ1_S` shard headers. It did not pack or verify bodies. Header hashing read 11,024,544 bytes in total. The resulting plan reported:

| Field | Value |
|---|---:|
| records | 24,576 |
| projections | 73,728 |
| logical routed bytes | 39,845,888,000 |
| planned pack bytes | 40,022,052,864 |
| alignment overhead including header | 176,164,864 bytes (0.4421%) |
| smallest/largest aligned record | 1,568,768 / 1,773,568 bytes |
| plan SHA-256 | `ab387aac249d245c21e14b3a7c92a506eb6ec00337008f00c60a0edd5a6bee07` |

The generated index was 55,743,298 bytes. Synthetic tests, rather than the 39.8 GB payload, cover exact payload equality, projection hashes, zero padding, preserved format metadata, interruption/resume equivalence, atomic non-publication after interruption, deterministic sample and full verification, corruption detection, explicit reads, benchmark accounting, and manifest failure cases.
