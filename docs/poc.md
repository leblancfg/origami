# Qwen3.8-Flash-Next llama.cpp PoC

## Fixed identities

| Component | Identity |
|---|---|
| llama.cpp | `bea3b12daee45876b0129a3602dc8f534ce30bf0` |
| mmap patch | `patches/llama.cpp-bea3b12-mmap-prefetch-optout.patch` |
| model | `unsloth/Qwen3.8-Flash-Next-GGUF` |
| model revision | `d3bc75ee6ccef3efc1e228ec00a6cc2cdb1e2249` |
| quantization | `UD-IQ1_S` |

`config/qwen38-flash-next-ud-iq1_s.json` conforms to `validation/model-manifest.schema.json` and pins each shard's byte size and SHA-256. None of the repository commands download model files.

## Build

The build needs CMake, Git, and the macOS command-line developer tools.

```sh
scripts/bootstrap-llama-cpp.sh
```

The script creates `/private/tmp/origami-deps/llama.cpp-bea3b12daee45876b0129a3602dc8f534ce30bf0` by default. `ORIGAMI_DEPS_ROOT` selects another external directory. The script performs these checks before producing revision and patch markers:

1. Fetch and detach at the full pinned revision.
2. Verify `HEAD`, apply the mmap patch idempotently, and verify that `HEAD` is unchanged.
3. Reject patched changes outside `src/llama-model.cpp`.
4. Build `llama-cli`, `test-llama-archs`, and `test-backend-ops`.
5. check every PoC CLI option against the built binary's `--help` output.

## Preflight and launch

Use the checked-in manifest against the directory containing the three existing shards:

```sh
scripts/smoke-test.sh --preflight-only /path/to/UD-IQ1_S
```

Preflight rejects missing shards, wrong byte sizes, inconsistent split names, path aliases, and neighboring `.aria2` markers. Add `--verify-shards-sha256` for a full digest pass. Hashing reads every shard and changes filesystem cache state.

The default launch is the first-token profile:

```sh
scripts/smoke-test.sh /path/to/UD-IQ1_S
```

It uses a 512-token context, batch and microbatch sizes of 32, greedy sampling, mmap, full layer offload, no warm-up, and one generated token. The fixed sequence profile uses the same launch envelope and generates 16 tokens:

```sh
scripts/smoke-test.sh --validation /path/to/UD-IQ1_S
```

Both commands run through `tools/origami_validate.py`; no shell telemetry loop remains. Results are written under `artifacts/` unless `--output FILE` is supplied.

## Lazy mmap safeguards

The harness launches the backend with:

```text
GGML_METAL_NO_RESIDENCY=1
LLAMA_MMAP_PREFETCH=0
```

A successful result must contain both backend markers:

```text
mmap prefetch disabled by LLAMA_MMAP_PREFETCH=0
use residency sets    = false
```

The patch changes only the initial mmap advice. The existing Metal variable prevents residency-set requests. Neither safeguard bounds the Darwin file cache, compressor use, or physical pages retained by the OS. Treat this as a memory-fit reference, not Origami's bounded expert-streaming design.

Qwen4Exp already classifies `per_layer_token_embd.weight` as a CPU input tensor. The launch profile therefore has no tensor override for PLE. If Metal fails while creating the shard-2 virtual envelope, run the diagnostic fallback:

```sh
scripts/smoke-test.sh --shrink-metal-envelope /path/to/UD-IQ1_S
```

The wrapper adds `--override-tensor '^output=CPU'`. This moves the output projection to CPU so the Metal envelope starts after PLE; it is not part of the default profile.

## Targeted Metal quant test

The bootstrap builds the required test binary. Run the pinned backend's filtered comparison with residency sets disabled:

```sh
REV=bea3b12daee45876b0129a3602dc8f534ce30bf0
GGML_METAL_NO_RESIDENCY=1 \
  /private/tmp/origami-deps/llama.cpp-${REV}/build-origami/bin/test-backend-ops \
  test -b MTL0 -o MUL_MAT_ID -p 'type_a=(iq1_s|iq2_xxs|iq4_nl)' -j 1
```

This covers the quantized expert dispatch types used by the pinned model. A full model result still depends on complete local shards.
