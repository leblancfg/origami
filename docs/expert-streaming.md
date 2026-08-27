# Explicit Qwen4Exp expert reads

## Artifact

`patches/llama.cpp-213df585b9aed6a09be30d8401f267bf603c104c-explicit-expert-streaming.patch` adds a compilable private llama.cpp subsystem and its C++ test. The patch applies to the pinned revision by itself and after the checked-in long-context patch.

The subsystem does four jobs:

- It opens the GGUF shards and reads each gate, up, and down slice by exact offset and byte count. macOS uses `pread`; a short read fails the request.
- It allocates a fixed number of fixed-size backend slots. Capacity is `cache_slots * slot_bytes`. Acquisition fails when Metal still owns every eviction candidate.
- It accepts IQ1_S or IQ2_XXS independently for gate and up, and requires IQ4_NL for down. It copies the quantized payload without conversion and returns views in router order.
- It records a Metal shared event after submitted graph work. A worker releases slot references only after that event signals.

`LLAMA_EXPLICIT_EXPERT_STREAMING` must equal `1` before the cache can be created. Missing, `0`, and malformed values leave the path unavailable. Acquisition also rejects batches with any token count other than one. There is no fallback to mmap expert access inside this subsystem.

## Build and test

The bootstrap uses a separate source and build directory. It applies the long-context patch first, applies the expert patch, builds only the synthetic test target and its dependencies, and does not start llama-cli or llama-server.

```sh
scripts/bootstrap-expert-streaming-llama-cpp.sh
```

`test-expert-stream` checks mixed quant types against patterned spans in two synthetic shards. It also checks router order, cache hits and eviction, EOF rejection, feature and token gates, blocked-slot behavior with a controlled completion, and the real Metal event path without loading a model.

## Remaining graph boundary

The patch stops at the first boundary that cannot fit into the current monolithic Qwen4Exp graph:

```text
llama_model_qwen4exp::graph::build_layer_ffn
  -> llm_graph_context::build_moe_ffn
       ffn_moe_logits
       ffn_moe_probs
       ffn_moe_topk / normalized weights
       ggml_build_forward_expand(gf, weights)
       [missing host-visible routing boundary]
       first MUL_MAT_ID for up/gate
       second MUL_MAT_ID for down
```

The runtime must submit the graph through `ffn_moe_topk` and normalized weights, read the ten selected IDs without changing their order or values, acquire their slices, then resume the layer with compact expert tensors. Existing `MUL_MAT_ID` assumes one tensor containing all 512 experts and indexes it with the original IDs. The resume graph therefore needs either:

1. compact gate/up/down tensors plus selected IDs remapped to compact positions while retaining the original weights and order, or
2. a Metal `MUL_MAT_ID` variant that accepts per-expert buffer addresses.

After submitting the resumed Metal graph, the caller must create `llama_expert_stream_metal_completion()` and pass it with the lease to `retire()`. Full graph integration must also stop routed tensor bodies from entering the mmap Metal envelopes. Until both changes exist, the capability manifest reports `vertical-slice-only-not-launchable`, and the ordinary llama.cpp path remains unchanged.
