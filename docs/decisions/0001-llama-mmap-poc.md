# TDR 0001: lazy mmap PoC for Qwen3.8-Flash-Next on M2 Max

Status: accepted for the first memory-fit run, 2026-08-26

## Decision

Use llama.cpp PR #27742 at `bea3b12daee45876b0129a3602dc8f534ce30bf0` with full Metal layer placement, mmap, and two safeguards:

```sh
GGML_METAL_NO_RESIDENCY=1 LLAMA_MMAP_PREFETCH=0 \
  ./build/bin/llama-cli --load-mode mmap -ngl all ...
```

`GGML_METAL_NO_RESIDENCY` already exists in ggml. Apply [`patches/llama.cpp-bea3b12-mmap-prefetch-optout.patch`](../../patches/llama.cpp-bea3b12-mmap-prefetch-optout.patch) to add the second switch. It changes only the initial mmap advice. It does not change tensor data or inference math.

This is enough to test whether Darwin and Metal can fault selected expert pages from the ordinary file mapping. Do not build an expert cache or a PLE reader before this test. The default llama.cpp load path is not a safe 64 GB test because it advises every shard with `POSIX_MADV_WILLNEED` and requests Metal residency for mmap envelopes larger than the recommended working set.

A successful token is a PoC, not a stable serving design. mmap has no cache ceiling and no useful bytes-read telemetry.

## Revisions and machine

| Source | Revision inspected |
|---|---|
| llama.cpp PR #27742 | `bea3b12daee45876b0129a3602dc8f534ce30bf0` |
| llama.cpp `master`, including current ggml/Metal | `bf942164697d2d62c2237a17b677dc2c017ea8e7` |
| DS4 | `c1d4597a80e300b803dc642519718f2c999589da` |
| Transformers PR #48337 | `b61b98bea4cd99ff97da2ca0aa4fa34e8800d10e` |
| Unsloth GGUF | `d3bc75ee6ccef3efc1e228ec00a6cc2cdb1e2249`, `UD-IQ1_S` |

The relevant Metal files are byte-identical between the two llama.cpp revisions above. Tests ran on an M2 Max with 64 GiB unified memory, macOS 26.5.2 (25F84). Metal reports `recommendedMaxWorkingSetSize = 55,662.79 MB` (53,084 MiB).

The GGUF shard sizes from the pinned Hugging Face revision are 10,946,624, 49,990,818,368, and 22,544,696,352 bytes. The first shard contains split metadata and no tensors. Research commands read the shard-2 header already present in the primary worktree; they did not start another model download.

## Tensor placement and mapping

`per_layer_token_embd.weight` is not a Metal weight in the graph. llama.cpp classifies `LLM_TENSOR_PER_LAYER_TOKEN_EMBD` as `LLM_TENSOR_LAYER_INPUT` with `GGML_OP_GET_ROWS` in `src/llama-arch.cpp`. `llama_model_base::load_tensors()` fixes `dev_input` to CPU in `src/llama-model.cpp`. `llama_model_qwen4exp::graph::build_ple()` gathers the rows with `ggml_get_rows` in `src/models/qwen4exp.cpp`.

The pinned artifact confirms the concrete layout:

```
per_layer_token_embd.weight  IQ4_NL  [160, 320001536]
size                         28,800,138,240 bytes (26.822 GiB)
shard-2 file interval        [364622656, 29164760896)
row size                     90 bytes
```

CPU `ggml_compute_forward_get_rows_q()` calls the IQ4_NL `to_float` function for the selected rows (`ggml/src/ggml-cpu/ops.cpp`, `dequantize_row_iq4_nl` in `ggml/src/ggml-quants.c`). The pinned metadata names one PLE layer, layer 1. Its graph selects 16 rows per token, producing 16 x 160 F32 values. On this Mac's 16 KiB VM pages, a cold token can fault at most 16 unrelated PLE pages, about 256 KiB, before filesystem read-ahead.

Transformers independently makes the same placement choice. `Qwen4ExpTextNGramEmbedding.forward()` moves only the IDs to the embedding weight's device and moves the gathered result back. `Qwen4ExpPreTrainedModel._no_placement_params` excludes `ple.ple_embedding.ngram_embedding.weight` from automatic accelerator placement.

There is an important distinction between tensor placement and Metal visibility. llama.cpp groups all tensors with the same buffer type in one `ctx_map` entry. `llama_model_loader::get_mapping_range()` takes one minimum-to-maximum envelope per shard and context. In shard 2, Metal output tensors precede PLE and Metal block tensors follow it. Full offload therefore creates this layout:

```
CPU-input tensor bytes/envelope   27.155 GiB
Metal tensor bytes                19.402 GiB
Metal mmap envelope               46.558 GiB
CPU/Metal envelope overlap        27.155 GiB
```

The overlap contains PLE and `token_embd.weight`. The PLE tensor still executes on CPU, but its file pages lie inside the no-copy MTLBuffer's virtual range. [`tools/gguf-map-audit.py`](../../tools/gguf-map-audit.py) reproduces these numbers without reading tensor bodies.

`ggml_metal_buffer_map()` wraps mmap addresses with `newBufferWithBytesNoCopy:... MTLResourceStorageModeShared` in `ggml/src/ggml-metal/ggml-metal-device.m`. `ggml_backend_tensor_alloc()` points tensors into that buffer. There is no second 40.73 GiB weight allocation on unified memory. The MTLBuffer length is virtual/accounting size; physical pages arrive when CPU or GPU accesses them. If a mapping exceeds `maxBufferLength`, ggml creates overlapping views so no tensor crosses a view boundary. Those views also alias the file mapping.

Thus all non-PLE weights can stay mapped and Metal-visible without a byte-for-byte copy. They should not all be forced resident. The 40.73 GiB non-PLE payload leaves roughly 11.1 GiB under Metal's recommended working set for state, graph buffers, macOS, and other processes. The ordinary selected-expert accesses need much less resident expert data, but the OS may retain old pages until pressure forces eviction.

## Quantized MoE kernels

`llama_model_qwen4exp::graph::build_layer_ffn()` calls `build_moe_ffn()` for gate, up, and down expert tensors. Both Metal `MUL_MAT_ID` branches cover the artifact's quant types:

- Decode and small batches use `kernel_mul_mv_id_iq1_s_f32`, `kernel_mul_mv_id_iq2_xxs_f32`, and `kernel_mul_mv_id_iq4_nl_f32`, instantiated in `ggml/src/ggml-metal/kernels/mul_mv.metal`.
- Batches of at least 32 use the corresponding `kernel_mul_mm_id_*` templates in `ggml/src/ggml-metal/kernels/mul_mm.metal`.
- `ggml_metal_op_mul_mat_id()` selects MV or MM at `ne21_mm_id_min = 32`; `ggml_metal_library_get_pipeline_mul_mv_id()` and `_mul_mm_id()` resolve the typed pipelines.

A filtered `test-backend-ops` run on MTL0 passed 79/79 comparisons for `type_a=(iq1_s|iq2_xxs|iq4_nl)`. It exercised `n=1` and `n=32`, and the log showed all six typed MV/MM pipelines compiling. No quantized `MUL_MAT_ID` kernel patch is needed.

## Why the two safeguards are required

`llama_model_base::load_tensors()` calls `ml.init_mappings(true, ...)`. On Darwin, `llama_mmap::impl` maps each full shard and then calls `posix_madvise(..., POSIX_MADV_WILLNEED)` for the full file. This asks the OS to read all 67.56 GiB of files, including PLE, before sparse access has helped.

ggml/Metal enables residency sets by default on this OS. `ggml_metal_buffer_rset_init()` adds every no-copy view and calls `requestResidency`. Full offload's shard-2 Metal envelope alone is 46.558 GiB; the remaining Metal envelopes take the declared set beyond the 53,084 MiB recommended working set. `GGML_METAL_NO_RESIDENCY=1` disables this path.

DS4's source supports this choice. `model_open()` creates a shared file mmap for Metal without prefetching it. `ds4_gpu_model_residency_request_views()` returns early in SSD-streaming mode, and its comments state that pages then fault through the same no-copy view buffers. DS4 adds explicit span maps, page-in jobs, and bounded expert caches for production behavior. The first llama.cpp test does not need those mechanisms.

The remaining hazard is unbounded cache growth. Ten experts need about 0.778 GB of weight reads per generated token with no page hits. Darwin can evict clean file-backed pages, so a first run is possible, but latency and memory pressure depend on OS policy. This path cannot satisfy Origami's sustained-memory contract by itself.

## Ranked experiments

### 1. One cold decode token with lazy mappings

Apply and build:

```sh
git -C /path/to/llama.cpp checkout bea3b12daee45876b0129a3602dc8f534ce30bf0
git -C /path/to/llama.cpp apply /path/to/origami/patches/llama.cpp-bea3b12-mmap-prefetch-optout.patch
cmake -S /path/to/llama.cpp -B /path/to/llama.cpp/build-origami \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF
cmake --build /path/to/llama.cpp/build-origami -j 12 --target llama-cli
```

After all three existing downloads finish:

```sh
MODEL=/path/to/UD-IQ1_S/Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf
sysctl vm.swapusage
vm_stat > /tmp/origami-vm.before
GGML_METAL_NO_RESIDENCY=1 LLAMA_MMAP_PREFETCH=0 \
  /usr/bin/time -l /path/to/llama.cpp/build-origami/bin/llama-cli \
  --load-mode mmap -ngl all -m "$MODEL" -c 512 -b 32 -ub 32 \
  -p 'Hello' -n 1 2>&1 | tee /tmp/origami-first-token.log
vm_stat > /tmp/origami-vm.after
sysctl vm.swapusage
```

Go: the log contains `mmap prefetch disabled`, model load completes, and one token is emitted without swap growth or a Metal allocation error. Record compressor-page deltas from the two `vm_stat` files. No-go: process kill, allocation failure, any swap growth, or sustained yellow/red `memory_pressure` during the single-token run.

### 2. Remove PLE from the Metal envelope if buffer creation fails

Keep experiment 1's environment and add:

```sh
-ot '^output=CPU'
```

This puts the output tensors before PLE in the CPU context, so the first Metal tensor in shard 2 is `blk.0.*`, after PLE and `token_embd.weight`. Go: the model reaches one token and the reported Metal model-buffer total drops by about 27.155 GiB of virtual envelope. No-go: the envelope shrinks but allocation still fails. This fallback costs a CPU output projection and is only for isolating Metal VM accounting.

### 3. Exercise both expert dispatch branches

This source-level check is complete:

```sh
GGML_METAL_NO_RESIDENCY=1 build/bin/test-backend-ops test -b MTL0 \
  -o MUL_MAT_ID -p 'type_a=(iq1_s|iq2_xxs|iq4_nl)' -j 1
```

Observed: 79/79 passed, including `n=1` MV and `n=32` MM cases. The model run must still confirm its exact shapes.

### 4. Measure page behavior over 16 decode tokens

Repeat experiment 1 with `-n 16`, first after a reboot/cold cache and then warm. Go: all tokens complete, swap remains unchanged, and compressed pages return near baseline after exit. Record elapsed time, major faults, maximum RSS, and before/after VM counters. No-go: monotonic swap or compression growth, a deadlock on GPU faults, or later tokens slowing without recovery.

### 5. Establish correctness before optimization

Run the same greedy prompt against a resident oracle at the pinned llama.cpp and GGUF revisions. Compare token IDs for short prompts, then compare PLE output and MoE output callbacks if tokens diverge. A memory-fit run with different greedy tokens is a no-go for further performance work.

## Follow-up boundary

If experiments 1 and 4 pass, retain this path only as the reference PoC. The next runtime change should replace routed-expert page faults with explicit reads and a bounded cache. A separate bounded PLE row cache is lower priority: PLE's cold demand is at most a few hundred KiB per token, while routed experts demand about 0.778 GB per token.
