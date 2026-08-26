# Qwen3.8-Flash-Next llama.cpp proof of concept

This path uses llama.cpp as an external backend. The bootstrap script fetches commit `bea3b12daee45876b0129a3602dc8f534ce30bf0`, the head of Qwen4Exp PR #27742 used for this bring-up, and builds it outside the Origami worktree.

## Build

The build needs CMake, Git, and the macOS command-line developer tools.

```sh
scripts/bootstrap-llama-cpp.sh
```

The default dependency directory is `/private/tmp/origami-deps`. Set `ORIGAMI_DEPS_ROOT` to put it elsewhere. The script checks the remote URL and full commit before building `llama-cli` and `test-llama-archs`. It does not fetch model files.

This exact build completed on the M2 Max host. `llama-cli --list-devices` reported the M2 Max Metal device with 53,084 MiB. The synthetic Qwen4Exp architecture test passed its Metal and CPU comparisons, then aborted in the test's Meta backend. That Meta failure remains unresolved.

## Model

Use the existing Unsloth download at revision `d3bc75ee6ccef3efc1e228ec00a6cc2cdb1e2249`. The repository manifest records all three shard sizes and SHA-256 digests. Check sizes before launch:

```sh
scripts/verify-model.sh /path/to/UD-IQ1_S
```

A full digest pass is optional because it reads 72.5 GB:

```sh
scripts/verify-model.sh --sha256 /path/to/UD-IQ1_S
```

Neither command downloads a model.

## Launch profile

```sh
scripts/launch-qwen38.sh /path/to/UD-IQ1_S \
  --single-turn --prompt 'Reply with exactly: ORIGAMI_OK'
```

The checked-in profile uses the following flags from the pinned binary's help:

- 512 context tokens, a 64-token logical batch, and a 32-token physical batch
- greedy sampling with seed 1 and temperature 0
- `--load-mode mmap`, `--gpu-layers all`, and `--override-tensor '^per_layer_token_embd$=CPU'`
- no warm-up and no CLI RAM prompt cache

The tensor name comes from the pinned Qwen4Exp loader. Its graph calculates PLE row IDs on the host and calls `ggml_get_rows` on `per_layer_token_embd`. The profile requests a CPU buffer for that 28.8 GB table and Metal placement for all repeating layers. The full model run must confirm the resulting placement from the backend log.

This is demand-paged mmap access, not Origami's planned SSD cache. The pinned backend has no bounded PLE row cache or explicit expert-streaming scheduler. Its direct-I/O load mode allocates model buffers and reads weights into them, so that mode does not solve the 72.5 GB capacity problem. The mmap profile is the shortest path that can plausibly run on this machine, but macOS controls eviction and its memory ceiling is not deterministic.

Set `ORIGAMI_CONTEXT`, `ORIGAMI_PREDICT`, `ORIGAMI_BATCH`, or `ORIGAMI_UBATCH` to adjust the profile. Print the resolved invocation without starting inference:

```sh
scripts/launch-qwen38.sh --print-command /path/to/UD-IQ1_S
```

## Recorded smoke test

```sh
scripts/smoke-test.sh --output artifacts/qwen38-smoke /path/to/UD-IQ1_S
```

The harness saves the resolved command, raw and ANSI-stripped output, an output SHA-256, backend logs, and a JSON host report. It samples process RSS and virtual size once per second. System-wide swap use, compressor pages, and pageouts are recorded alongside before-and-after `vm_stat` and `memory_pressure` snapshots.

Pass a known output hash on later runs to enforce exact output:

```sh
scripts/smoke-test.sh --expect-sha256 HASH /path/to/UD-IQ1_S
```

The full smoke run is blocked until all three model shards finish downloading. No full-model memory or output result has been recorded yet.
