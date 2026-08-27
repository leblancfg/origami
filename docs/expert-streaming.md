# Exact Qwen4Exp expert reads and graph boundary

## Artifact

`patches/llama.cpp-213df585b9aed6a09be30d8401f267bf603c104c-explicit-expert-streaming.patch` applies after the checked-in long-context patch. It integrates explicit routed-expert reads into the Qwen4Exp single-token graph at the pinned revision.

The path is enabled only when `LLAMA_EXPLICIT_EXPERT_STREAMING=1`. Missing or `0` leaves upstream loading and graph construction unchanged. Any other value fails model loading. Once enabled, there is no mmap routed-tensor fallback:

- Qwen4Exp records each routed tensor's GGUF shard, absolute body offset, type, shape, and expert stride from `llama_model_loader`.
- It requires separate, contiguous gate/up/down tensors and complete metadata for all layers and experts.
- The routed tensors are consumed with `TENSOR_SKIP`, so no routed `ggml_tensor`, model-buffer allocation, or load destination is created.
- mmap is disabled for the enabled model load. Dense, shared-expert, PLE, and router tensors use ordinary allocated backend buffers. This is deliberately stricter than retaining an unused routed mmap envelope.
- The cache reopens the exact GGUF shard paths and uses bounded positional reads. A short read, unsupported quant type, malformed shape, missing path, or out-of-range span aborts.

Gate and up accept IQ1_S or IQ2_XXS independently. Down requires IQ4_NL. Bytes are copied without conversion.

## Single-token launch boundary

The Qwen4Exp graph replaces each omitted routed tensor with a strided view over one fixed Metal cache buffer. The views are allocated before scheduler split and backend assignment. Their buffer, type, dimensions, and stride never change during evaluation.

The scheduler callback creates two synchronized boundaries per layer:

1. `build_moe_ffn()` expands the final normalized and scaled `ffn_moe_weights`. The callback then reads the original selected IDs. Weight lookup has already consumed those IDs, preserving router order and values.
2. The cache acquires the corresponding gate/up/down records. The callback rewrites only the allocated I32 selected-ID payload from original expert numbers to physical cache-slot numbers.
3. Routed gate, up, and down `MUL_MAT_ID` nodes execute against the fixed cache views.
4. After synchronized `ffn_moe_out`, the callback restores the original selected IDs and retires the lease behind a Metal event recorded after the submitted work.

`LLAMA_EXPERT_CACHE_SLOTS` sets the fixed slot count. It must be an unsigned integer at least as large as `expert_used_count`; the default is twice that count. Slot size is derived from the largest real GGUF gate/up/down record and padded to 4 KiB. Capacity is exactly `slots * slot_bytes`.

The scheduler callback now returns `GGML_STATUS_ABORTED` when a callback rejects a boundary. It no longer continues into later backend splits after a fail-closed routing error.

## Why tensor-buffer substitution was rejected

Changing a tensor's buffer after scheduler allocation is not safe at this revision. Backend assignment and cross-backend copies are decided in `ggml_backend_sched_split_graph()`. The scheduler may replace a node source with a copy tensor, and its MoE copy optimization runs before eval callbacks. Swapping the original tensor's buffer at the callback would therefore be invisible to some scheduled nodes or would disagree with their precomputed backend.

The integrated path does not substitute buffers. Cache tensors are pre-allocated in the fixed Metal buffer before graph splitting. At the routing boundary it verifies that all three `MUL_MAT_ID` nodes still reference those exact tensors, the selected IDs are on the same Metal backend, and no scheduler copy rewrote either source. A mismatch aborts before routed work is submitted.

## Scope and failure gates

Only ubatches with exactly one token may execute while the feature is enabled. Prefill, multi-sequence token-generation batches, malformed cache settings, non-Metal execution, merged gate/up tensors, LoRA/scaled routed tensors, and scheduler source rewrites fail closed. The ordinary path is unchanged when the flag is disabled.

This patch does not implement PLE row streaming or an address-table kernel. It also does not claim real-model correctness parity.

## Build and test

The bootstrap creates a unique `/private/tmp/origami-expert-graph-bootstrap.*` source/build root unless `ORIGAMI_DEPS_ROOT` is explicitly supplied. It applies the long-context and expert patches, builds the full static llama library plus `test-expert-stream`, and never starts llama-cli or llama-server:

```sh
scripts/bootstrap-expert-streaming-llama-cpp.sh
```

The C++ test covers exact mixed-quant reads, cache hits and eviction, completion ownership, EOF rejection, strict feature/token gates, Metal cache-tensor binding, and a scheduler callback graph proving that an ID rewritten after a synchronized boundary is consumed by the following `MUL_MAT_ID` while buffer identity and backend assignment stay fixed.

`validation/expert-streaming-capabilities.json` records the compiled capabilities and validation limit. The combined source built and the synthetic tests passed in a unique private tree. A real model was not launched because the primary 65K probe was active. The remaining boundary is a one-token-at-a-time real Qwen4Exp load/decode and logits/token parity run; no compile-time blocker is currently known.
