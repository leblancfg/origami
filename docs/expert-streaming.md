# Exact Qwen4Exp expert reads and graph boundary

## Artifact

`patches/llama.cpp-213df585b9aed6a09be30d8401f267bf603c104c-explicit-expert-streaming.patch` applies after the checked-in long-context patch. It integrates explicit routed-expert reads into the Qwen4Exp single-token graph at the pinned revision.

The path is enabled only when `LLAMA_EXPLICIT_EXPERT_STREAMING=1`. Missing or `0` leaves upstream loading and graph construction unchanged. Any other value fails model loading. Once enabled, there is no mmap routed-tensor fallback:

- Qwen4Exp records each routed tensor's GGUF shard, absolute body offset, type, shape, and expert stride from `llama_model_loader`.
- It requires separate, contiguous gate/up/down tensors and complete metadata for all layers and experts.
- The routed tensors are consumed with `TENSOR_SKIP`, so no routed `ggml_tensor`, model-buffer allocation, or load destination is created.
- Routed tensors are absent from model contexts and mmap envelopes. Dense, shared-expert, PLE, and router tensors may retain ordinary mmap loading; this keeps the 28.8 GB PLE table sparse without providing any routed-expert fallback.
- The cache reopens the exact GGUF shard paths and uses bounded positional reads. A short read, unsupported quant type, malformed shape, missing path, or out-of-range span aborts.

Gate and up accept IQ1_S or IQ2_XXS independently. Down requires IQ4_NL. Bytes are copied without conversion.

## Single-token launch boundary

The Qwen4Exp graph replaces each omitted routed tensor with a strided view over one fixed Metal cache buffer. The views are allocated before scheduler split and backend assignment. Their buffer, type, dimensions, and stride never change during evaluation.

The scheduler callback creates two synchronized boundaries per layer:

1. When `ffn_moe_topk` completes, the callback reads the original selected IDs. It never changes that tensor. The normalized weight lookup continues to use the router output.
2. The cache acquires the corresponding gate/up/down records. The callback writes physical cache-slot numbers to a dedicated I32 graph input assigned to Metal before scheduler allocation.
3. Routed gate, up, and down `MUL_MAT_ID` nodes use the dedicated IDs and fixed cache views. The graph expands the weight lookup first, which keeps the callback and routed work in the required order.
4. After synchronized `ffn_moe_out`, the callback retires the lease behind a Metal event.

`LLAMA_EXPERT_CACHE_SLOTS` sets the fixed slot count. It must be an unsigned integer at least as large as `expert_used_count`; the default is twice that count. Slot size is derived from the largest real GGUF gate/up/down record and padded to 4 KiB. Capacity is exactly `slots * slot_bytes`.

The scheduler callback now returns `GGML_STATUS_ABORTED` when a callback rejects a boundary. It no longer continues into later backend splits after a fail-closed routing error.

## Why tensor-buffer substitution was rejected

Changing a tensor's buffer after scheduler allocation is not safe at this revision. Backend assignment and cross-backend copies are decided in `ggml_backend_sched_split_graph()`. The scheduler may replace a node source with a copy tensor, and its MoE copy optimization runs before eval callbacks. Swapping the original tensor's buffer at the callback would therefore be invisible to some scheduled nodes or would disagree with their precomputed backend.

The integrated path does not substitute buffers. Cache tensors are pre-allocated in the fixed Metal buffer before graph splitting. Each layer also has a dedicated remapped-ID input pinned to Metal. At the routing boundary the runtime verifies that all three `MUL_MAT_ID` nodes still reference the exact cache tensors and that dedicated input. A source or backend mismatch aborts before routed work is submitted.

## Parity failure and correction

The first real run completed but emitted `,` instead of the resident reference token `The`. A callback comparison found the first divergence at layer 0: the resident `ffn_moe_out` sum was `-0.197129`, while the streamed result was zero. The original IDs were correct (`434,308,37,309,367,2,226,386,118,201`), the remapped slots were `0..9`, and all 30 selected gate/up/down cache records matched positional reads from the GGUF byte for byte.

The fault was the ID handoff. The old graph changed the computed `ffn_moe_topk` payload at the normalized-weight boundary and expected later scheduled work to consume the change. A computed intermediate is not a stable publication input across scheduler splits. The routed `MUL_MAT_ID` work observed the original IDs, which are outside the 64-slot cache tensor, and returned zeros.

The correction leaves `ffn_moe_topk` untouched and publishes slots through a separate Metal-assigned I32 input. A reference callback then captured all 48 `ffn_moe_topk` and all 48 `ffn_moe_out` tensors for one token. Every captured byte matched between resident and streamed runs. The final dirty-tree generation emitted the expected `The`; swap stayed at 14131.44 MiB and free memory stayed at 87%. Layer 0 and layer 47 output hashes are recorded in the capability file.

## Scope and failure gates

Only ubatches with exactly one token may execute while the feature is enabled. Prefill, multi-sequence token-generation batches, malformed cache settings, non-Metal execution, merged gate/up tensors, LoRA/scaled routed tensors, and scheduler source rewrites fail closed. The ordinary path is unchanged when the flag is disabled.

This patch does not implement PLE row streaming or an address-table kernel. Its correctness claim is limited to the verified one-token path.

## Build and test

The bootstrap creates a unique `/private/tmp/origami-expert-graph-bootstrap.*` source/build root unless `ORIGAMI_DEPS_ROOT` is explicitly supplied. It applies the long-context and expert patches, builds the full static llama library plus `test-expert-stream`, and never starts llama-cli or llama-server:

```sh
scripts/bootstrap-expert-streaming-llama-cpp.sh
```

The C++ test covers exact mixed-quant reads, cache hits and eviction, completion ownership, EOF rejection, strict feature/token gates, Metal cache-tensor binding, and the separate original/remapped ID contract across a scheduler callback.

`validation/expert-streaming-capabilities.json` records the compiled capabilities and real-model evidence. Validation used batch size 32, ubatch size 1, and 64 cache slots in a unique private tree. The one-token callback tensors matched the resident reference byte for byte, with no mmap routed fallback.
