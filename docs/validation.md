# PoC validation harness

`tools/origami_validate.py` runs one fixed greedy prompt through a llama.cpp-compatible CLI and writes a JSON record. It targets macOS because it reads `vm_stat`, `memory_pressure`, `vm.swapusage`, `ps`, and `diskutil`. Python's standard library is its only language dependency.

The harness does not inspect GGUF metadata or tensor data. The GGUF inventory tool owns that work. This harness checks artifact names, exact file sizes, split numbering, and partial-download markers before it starts inference.

## Model manifest

Create a manifest beside the model shards. Record an immutable provider revision and copy exact byte sizes from the provider's artifact metadata. Relative paths are resolved from the manifest directory.

```json
{
  "schema_version": "origami.model-manifest.v1",
  "model": {
    "id": "unsloth/Qwen3.8-Flash-Next-GGUF",
    "revision": "d3bc75ee6ccef3efc1e228ec00a6cc2cdb1e2249",
    "format": "GGUF",
    "quantization": "UD-IQ1_S"
  },
  "entrypoint": "Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf",
  "shards": [
    {
      "path": "Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf",
      "size_bytes": 10946624
    },
    {
      "path": "Qwen3.8-Flash-Next-UD-IQ1_S-00002-of-00003.gguf",
      "size_bytes": "<exact byte size from provider metadata>"
    },
    {
      "path": "Qwen3.8-Flash-Next-UD-IQ1_S-00003-of-00003.gguf",
      "size_bytes": "<exact byte size from provider metadata>"
    }
  ]
}
```

Replace the last two placeholders with integers. Do not use this example as an artifact manifest. `validation/model-manifest.schema.json` defines the format.

A shard fails preflight when it is missing, has the wrong size, or has a neighboring `.aria2` marker. Split names must cover every index from 1 through the declared shard count. An optional `sha256` field pins content. Pass `--verify-shards-sha256` when a cold full-file read is acceptable; hashing a large model can change the filesystem cache state.

## Fixed smoke test

The script owns these settings so benchmark commands cannot change them through extra arguments:

- prompt: `Reply with only the next four integers: 2 4 6 8`
- prediction limit: 16 tokens
- seed: 424242
- temperature: 0 (greedy)
- prompt display: disabled

The command places fixed flags after user-supplied flags. Raw stdout and its SHA-256 are stored in the result. Establish a golden hash with a trusted reference executable, then require it on later runs with `--expected-output-sha256`. A passing first run without that option records output but does not prove parity.

Run the real CLI as follows:

```sh
python3 tools/origami_validate.py \
  --executable /path/to/llama-cli \
  --runtime-revision LLAMA_CPP_COMMIT \
  --model-manifest /path/to/model/model-manifest.json \
  --output bench-results/smoke.json \
  --expected-output-sha256 GOLDEN_SHA256 \
  -- --ctx-size 512 --threads 12
```

The executable must accept the usual llama.cpp CLI flags and print llama.cpp performance lines on stderr. The first PoC also rejects a model volume that `diskutil` does not identify as internal solid-state storage. The parser recognizes both `llama_print_timings` and `llama_perf_context_print` prefixes. A run fails if prompt-evaluation or decode timing is absent.

## Captured data

The JSON result includes:

- Mac model, chip, CPU count, unified-memory size, macOS version, build, and kernel release
- project commit and dirty state; runtime revision, version output, binary size, and binary SHA-256
- provider model ID, immutable revision, quantization, shard status, and byte totals
- model volume device, internal/solid-state flags, bus protocol, capacity, and free space
- exact command, wall time, exit status, stdout, stderr, prefill timing, and decode timing
- root-process and process-tree RSS samples
- system snapshots before, during, and after the run
- memory-pressure free percentage, compressor occupancy, compression counters, swap usage, and swap-in/swap-out counters

The summary reports peak RSS, minimum free percentage, compressor growth, and swap growth. VM and swap counters are system-wide. Stop unrelated memory-heavy jobs before comparing runs, and retain the samples when a counter delta needs investigation.

`validation/result.schema.json` defines the result envelope. The script writes an error result on preflight and runtime failures. Exit status 2 means preflight or harness failure; status 3 means the child ran but validation failed; 130 means interruption. The output file is replaced atomically.

The harness starts the child in a new process group. Timeout and interrupt handling terminate that group, wait briefly, then kill remaining processes. Captures live in a temporary directory that Python removes on success and failure.

## Harmless self-test

The repository includes a mock CLI and a 16-byte fixture. It loads no model.

```sh
python3 tools/origami_validate.py \
  --executable tools/mock_llama_cli.py \
  --runtime-revision mock-v1 \
  --model-manifest tests/fixtures/mock-model-manifest.json \
  --output /tmp/origami-mock-result.json \
  --expected-output-sha256 3b0c8ba590d96fdafce61f18ec139bcc6195dbf4bf69f22c3659448d43361c33 \
  --verify-shards-sha256 \
  --sample-interval 0.1

python3 -m unittest discover -s tests -v
```

The integration tests cover timing parsing, RSS sampling, shard rejection, output hashing, timeout termination, and capture cleanup.
