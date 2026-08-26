# TDR 0001: lazy mmap PoC for Qwen3.8-Flash-Next on M2 Max

Status: accepted with CPU output placement for the 64 GB profile

## Decision

Use llama.cpp PR #27742 at `bea3b12daee45876b0129a3602dc8f534ce30bf0` with mmap, all model layers on Metal, the output tensor on CPU, and these safeguards:

```text
GGML_METAL_NO_RESIDENCY=1
LLAMA_MMAP_PREFETCH=0
```

`GGML_METAL_NO_RESIDENCY` exists in the pinned ggml source. [`patches/llama.cpp-bea3b12-mmap-prefetch-optout.patch`](../../patches/llama.cpp-bea3b12-mmap-prefetch-optout.patch) adds the second switch and an early stderr diagnostic for the existing Metal switch. The diagnostic is needed because the pinned CLI initializes Metal before it applies command-line log verbosity. The patch does not change tensor data or inference math.

The default llama.cpp path advises every shard with `POSIX_MADV_WILLNEED` and requests Metal residency for large mmap envelopes. Disabling those requests permits a test of Darwin and Metal page faults on selected expert data. mmap still has no cache ceiling or useful model-bytes-read telemetry. A generated token demonstrates memory fit only; it does not establish bounded serving behavior.

Use the build and validation commands in [`docs/poc.md`](../poc.md). The full-Metal output placement remains a failure-reproduction option, not an alternate supported path.

## Fixed revisions

| Source | Revision inspected |
|---|---|
| llama.cpp PR #27742 | `bea3b12daee45876b0129a3602dc8f534ce30bf0` |
| llama.cpp `master` comparison | `bf942164697d2d62c2237a17b677dc2c017ea8e7` |
| DS4 | `c1d4597a80e300b803dc642519718f2c999589da` |
| Transformers PR #48337 | `b61b98bea4cd99ff97da2ca0aa4fa34e8800d10e` |
| Unsloth GGUF | `d3bc75ee6ccef3efc1e228ec00a6cc2cdb1e2249`, `UD-IQ1_S` |

The relevant Metal files are byte-identical between the two llama.cpp revisions. The test machine is an M2 Max with 64 GiB unified memory. Metal reports `recommendedMaxWorkingSetSize = 55,662.79 MB` (53,084 MiB).

The pinned shard sizes are 10,946,624, 49,990,818,368, and 22,544,696,352 bytes. The first shard contains split metadata and no tensors.

## Tensor placement and mapping

`per_layer_token_embd.weight` is a CPU input tensor. `src/llama-arch.cpp` classifies `LLM_TENSOR_PER_LAYER_TOKEN_EMBD` as `LLM_TENSOR_LAYER_INPUT` with `GGML_OP_GET_ROWS`; `src/llama-model.cpp` fixes `dev_input` to CPU. `src/models/qwen4exp.cpp` gathers PLE rows with `ggml_get_rows`. The launch profile does not override PLE placement.

The pinned PLE tensor has this layout:

```text
per_layer_token_embd.weight  IQ4_NL  [160, 320001536]
size                         28,800,138,240 bytes (26.822 GiB)
shard-2 file interval        [364622656, 29164760896)
row size                     90 bytes
```

On a 16 KiB VM page system, 16 unrelated row lookups can fault at most 256 KiB before filesystem read-ahead.

Tensor placement differs from Metal visibility. llama.cpp groups tensors by buffer type and computes one minimum-to-maximum mmap envelope per shard and context. Metal output tensors precede PLE in shard 2, while Metal block tensors follow it. Full offload therefore produces:

```text
CPU-input tensor bytes/envelope   27.155 GiB
Metal tensor bytes                19.402 GiB
Metal mmap envelope               46.558 GiB
CPU/Metal envelope overlap        27.155 GiB
```

The PLE operation remains on CPU, but its pages lie inside a no-copy Metal virtual range. `tools/gguf-map-audit.py` reproduces this report using `origami_artifacts` for bounded metadata parsing.

`ggml_metal_buffer_map()` wraps mmap addresses in shared no-copy MTLBuffers. The MTLBuffer length is virtual accounting, not a second byte-for-byte weight allocation. Physical pages arrive when CPU or GPU operations access them. Darwin decides when clean file-backed pages are evicted, so this mapping cannot support a bounded-memory claim.

The first full-Metal inference command created the large shard-2 envelope but failed its command buffer with `kIOGPUCommandBufferCallbackErrorOutOfMemory`. It emitted no token, drove measured memory-pressure free percentage down to 7%, and grew swap by 6,721,172,930 bytes. Moving `output` to CPU makes the first Metal tensor occur after PLE and shrinks the virtual envelope. The resulting CPU output projection emitted a token with zero swap growth, so this placement is the 64 GB default.

## Quantized MoE kernels

Qwen4Exp calls `build_moe_ffn()` for routed gate, up, and down tensors. The pinned Metal backend provides typed `MUL_MAT_ID` MV and MM kernels for IQ1_S, IQ2_XXS, and IQ4_NL. The filtered `test-backend-ops` command in [`docs/poc.md`](../poc.md) exercises both dispatch branches. No quant kernel patch is part of this PoC.

## Boundary

Ten selected experts require about 0.778 GB of weight reads per generated token without cache hits. Darwin may evict clean mapped pages, but latency and memory pressure depend on OS policy. The native Origami path must replace routed-expert faults with explicit reads and a bounded cache. PLE can use a separate row cache; its cold demand is much smaller than routed-expert demand.
