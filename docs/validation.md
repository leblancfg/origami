# PoC validation harness

`tools/origami_validate.py` runs deterministic llama.cpp profiles and writes one JSON record. It is the repository's only telemetry implementation. Python's standard library is its only language dependency.

The harness targets macOS. It records `vm_stat`, `memory_pressure`, `vm.swapusage`, process-tree RSS, storage identity, runtime identity, stdout, stderr, and parsed llama.cpp timings. VM and swap counters are system-wide.

## Integrated command

The Qwen3.8 wrapper supplies every pinned argument:

```sh
scripts/smoke-test.sh --preflight-only /path/to/UD-IQ1_S
scripts/smoke-test.sh /path/to/UD-IQ1_S
scripts/smoke-test.sh --validation /path/to/UD-IQ1_S
```

The first-token profile generates one token. The validation profile uses the prompt `Reply with only the next four integers: 2 4 6 8` and generates at most 16 tokens. Both use seed 424242 and temperature 0. The wrapper also selects the measured CPU-output placement. Pass `--output FILE` to select the result path, or `--expected-output-sha256 HASH` to compare the extracted generated text with a prior result.

The wrapper enables the lazy mmap safeguards, performance counters, and trace-level backend logging. The backend's early Metal diagnostic bypasses the CLI logger because Metal initializes before argument parsing sets the requested verbosity. The harness records the environment and fails unless stderr proves that mmap prefetch and Metal residency sets are disabled.

## Manifest checks

`validation/model-manifest.schema.json` defines the model manifest. `config/qwen38-flash-next-ud-iq1_s.json` is the canonical manifest for this PoC. `--model-root` lets the harness resolve its relative shard paths from an external model directory.

A shard fails preflight if it is missing, has the wrong size, or has a neighboring `.aria2` marker. Split names must share one prefix, include every declared index, and use shard 1 as the entrypoint. Canonical paths cannot alias one file. `--verify-shards-sha256` hashes shards that declare a digest; omit it before a cold-cache run if exact byte sizes are sufficient.

Preflight writes a normal result without starting inference:

```sh
python3 tools/origami_validate.py \
  --executable /path/to/llama-cli \
  --runtime-revision LLAMA_CPP_COMMIT \
  --model-manifest config/qwen38-flash-next-ud-iq1_s.json \
  --model-root /path/to/UD-IQ1_S \
  --output /tmp/origami-preflight.json \
  --preflight-only
```

## Direct harness use

Use the wrapper for the pinned PoC. Direct use is available for another llama.cpp-compatible executable:

```sh
python3 tools/origami_validate.py \
  --executable /path/to/llama-cli \
  --runtime-revision LLAMA_CPP_COMMIT \
  --model-manifest /path/to/model-manifest.json \
  --output bench-results/smoke.json \
  --profile validation \
  --expected-output-sha256 GOLDEN_SHA256 \
  -- --ctx-size 512 --threads 12
```

Extra arguments cannot replace the fixed model, prompt, prediction count, seed, temperature, or prompt-display options. The executable must print llama.cpp prefill and decode timings on stderr. The harness rejects known compute, out-of-memory, scheduler, and decode failure markers even when the CLI exits zero. The first PoC rejects storage that `diskutil` does not identify as internal solid state.

The harness extracts generated text from the pinned CLI's presentation wrapper and hashes that text separately from raw stdout. A trusted reference run can establish `--expected-output-sha256`. A passing run without that option records deterministic output but does not prove parity with an oracle.

## Result and failure handling

`validation/result.schema.json` defines the result envelope. The harness records:

- project commit and dirty state
- executable path, declared revision, version output, size, and SHA-256
- immutable model identity, shard status, and byte totals
- Mac, OS, memory, and storage properties
- exact argv and safeguard environment
- generated text and SHA-256, raw stdout, stderr, timings, and exit status
- process-tree RSS and system memory snapshots

Exit status 2 indicates preflight or harness failure. Status 3 means the child ran but validation failed; status 130 means interruption. The harness writes error results atomically.

The child runs in a new process group. Timeout, interruption, and ordinary leader exit terminate remaining group members. Captures are bounded and held in a temporary directory.

## Harmless self-test

The mock CLI loads no GGUF tensor data:

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
