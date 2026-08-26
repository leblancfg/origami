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
```

`tools/origami_validate.py` is the only telemetry implementation. `scripts/smoke-test.sh` supplies the pinned executable, manifest, runtime profile, and both required environment safeguards. The JSON result captures the command, backend log, host identity, process RSS, memory pressure, compression, and swap counters.

The default run sets `GGML_METAL_NO_RESIDENCY=1` and `LLAMA_MMAP_PREFETCH=0`. A pass requires backend log lines proving that residency sets and mmap prefetch are disabled. This mmap path has no cache ceiling, so its memory use is not bounded. See [docs/poc.md](docs/poc.md) for the exact identities, expected log evidence, and Metal-envelope fallback.

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
