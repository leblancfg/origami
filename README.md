# Origami

Origami is an experimental inference project for sparse models whose full weights do not fit in memory. The first target is Qwen3.8-Flash-Next on 64 GB Apple Silicon. [INTENT.md](INTENT.md) defines the hardware budget and engineering rules.

## Pinned llama.cpp PoC

The reference path uses llama.cpp commit `bea3b12daee45876b0129a3602dc8f534ce30bf0` with the checked-in mmap prefetch patch. It never downloads model shards.

```sh
# Build and verify the patched backend outside the worktree.
scripts/bootstrap-llama-cpp.sh

# Reject missing, partial, or incorrectly sized shards.
scripts/smoke-test.sh --preflight-only /path/to/UD-IQ1_S

# Run the default short-context, one-token profile with telemetry.
scripts/smoke-test.sh /path/to/UD-IQ1_S

# Run the fixed 16-token validation profile.
scripts/smoke-test.sh --validation /path/to/UD-IQ1_S

# Reproduce the unsafe full-Metal placement failure when diagnosing it.
scripts/smoke-test.sh --full-metal-output /path/to/UD-IQ1_S
```

`tools/origami_validate.py` is the only telemetry implementation. `scripts/smoke-test.sh` supplies the pinned executable, manifest, runtime profile, and both required environment safeguards. The JSON result captures the command, backend log, host identity, process RSS, memory pressure, compression, and swap counters.

The default 64 GB run sets `GGML_METAL_NO_RESIDENCY=1` and `LLAMA_MMAP_PREFETCH=0`, enables performance logs, and places the output tensor on CPU. The CPU placement avoids the 46.6 GiB Metal mmap envelope that caused an out-of-memory command-buffer failure and 6.72 GB of swap growth in the first full-Metal run. A pass requires backend log lines proving that residency sets and mmap prefetch are disabled. This mmap path has no cache ceiling, so its memory use is not bounded. See [docs/poc.md](docs/poc.md) and [the measured result](docs/results/qwen38-poc-m2-max.md).

## Working hypothesis

Qwen3.8-Flash-Next has 125B parameters but activates about 6B per token. Its 48 MoE layers select 10 of 512 routed experts plus one shared expert. The Unsloth `UD-IQ1_S` GGUF is 72.5 GB: about 39.8 GB of routed experts, a 28.8 GB n-gram table, and 3.9 GB of always-used tensors.

The native design will keep dense weights, shared experts, recurrent state, and a modest context resident. It will fetch selected expert slices through a bounded cache and serve n-gram rows through a separate page-aligned cache. Greedy output must match the pinned llama.cpp Qwen4Exp implementation.

## Artifact inspection

The dependency-free inspector reads split GGUF metadata without mapping tensor bodies. It emits a tensor inventory, allocation ledger, and contiguous expert slices:

```sh
python3 -m origami_artifacts /path/to/gguf/shards
python3 -m origami_artifacts /path/to/gguf/shards --json --output ledger.json
```

`tools/gguf-map-audit.py` imports the same bounded parser and adds the pinned llama.cpp placement model. See [docs/artifact-inspector.md](docs/artifact-inspector.md), [docs/plan.md](docs/plan.md), and [docs/validation.md](docs/validation.md).

## Routed-expert sidecar

The sidecar packer groups each `(layer, expert)` gate/up/down triple into an aligned record while preserving the original quantized bytes and GGUF metadata. An index-only run checks real headers without copying the 39.8 GB routed payload:

```sh
python3 -m origami_artifacts.sidecar index /path/to/UD-IQ1_S \
  --source-revision d3bc75ee6ccef3efc1e228ec00a6cc2cdb1e2249 \
  --source-manifest config/qwen38-flash-next-ud-iq1_s.json \
  --index experts.index-only.json
```

Packing is resumable and atomically publishes a fixed-format binary pack plus JSON index. Sample/full byte verification, an aligned `pread` API, and a benchmark command are included. See [docs/expert-sidecar.md](docs/expert-sidecar.md) for the exact format and recovery contract.

## Exact expert graph boundary

The pinned `213df585` patch now connects fixed-size Metal expert slots to Qwen4Exp single-token evaluation. It derives exact GGUF slices, omits routed bodies from model buffers, stops after final normalized routing weights, remaps selected IDs to fixed cache slots for routed `MUL_MAT_ID`, restores the IDs after synchronized `ffn_moe_out`, and retires slot leases behind Metal events. Enabled mode has no mmap routed fallback and rejects prefill.

```sh
scripts/bootstrap-expert-streaming-llama-cpp.sh
```

The full static library and synthetic scheduler/cache test build without launching a model. Real-model one-token decode and parity remain unrun. See [docs/expert-streaming.md](docs/expert-streaming.md) and [the capability record](validation/expert-streaming-capabilities.json).

## Long-context research profile

The GGUF declares a native 262,144-token window. The guarded candidate uses the 213df585 runtime plus the checked-in patch: Q8_0 main attention KV, an F16 key-only indexer cache, quantized-cache rotation support, and opt-in shared QSA graph inputs. It requests 250,000 tokens and has not completed an allocation or prompt probe.

```sh
# Prints status and the missing backend capabilities. It does not inspect shards.
scripts/context-profile.sh status

# Requires a matching bootstrap build and a complete model manifest.
scripts/context-profile.sh command /path/to/UD-IQ1_S
```

No native-context allocation has run on the test host. The shared-input prototype cuts the guarded profile's logical QSA host inputs from 132,059,136 to 11,004,928 bytes, but leaves dense masks and dense attention in the graph. Factor-2 524,288 and factor-4 1,000,000 static-YaRN work use separate nonlaunchable configs. See [docs/context-profile.md](docs/context-profile.md) and [the QSA build record](docs/results/qsa-shared-inputs-213df585.md).
