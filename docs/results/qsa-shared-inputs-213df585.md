# Qwen4Exp QSA shared-input prototype at 213df585

## Scope

The 213df585 graph creates one `llm_graph_input_qsa` for each of the twelve QSA layers. Every instance has the same cache-layout tensors and the same `F32[n_kv, tokens_per_stream, streams]` visibility/tail bias. `llama_memory_hybrid_idx_context::set_input_qsa()` then rebuilds all twelve copies on the host. The layer index does not enter that function.

The patch adds an opt-in `LLAMA_QSA_SHARED_INPUTS=1` path. The first QSA layer owns one input set through `llm_graph_result`; later layers reuse its tensors. Reuse checks the memory context, compression ratio, and every tensor dimension with `GGML_ASSERT`. A ratio or shape change aborts graph construction rather than using mismatched routing metadata. Values, score expansion, `ggml_top_k`, selected indices, and the dense attention mask are unchanged.

The normal path remains available with the variable unset or set to `0`. Any other value asserts. The guarded 250,000-token profile sets the variable and requires the matching build capability.

## Bound

For allocated cells `C` and microbatch tokens `U`, one logical input set is:

```text
cell_blk + blk_cells + blk_pos + bias
= 4C + 4C + 4C + 4CU
= 12C + 4CU bytes
```

At the native allocation boundary (`C=262144`, `U=32`), the host-input payload falls from 440,401,920 bytes to 36,700,160 bytes. The reduction is 403,701,760 bytes (11/12, or 91.67%).

The guarded profile uses `C=250112` and `U=8`. Its payload falls from 132,059,136 bytes to 11,004,928 bytes, reducing the context-plus-input lower bound from 4,320,817,152 to 4,199,762,944 bytes.

These figures cover graph inputs only. The graph still expands block scores to `C x U`, constructs a dense top-k mask, and calls dense masked attention. This patch does not claim a scheduler-buffer or attention-work bound. A block-index gather or sparse Metal kernel remains the production fix.

## Build and checks

The patch was applied to a clean checkout of `213df585b9aed6a09be30d8401f267bf603c104c` and built on arm64 macOS with Metal enabled. Its SHA-256 is `d46082bfc399f4039048df5c2751b7bb4226dece91712420decca853fffb958a`.

```text
cmake --build build-origami --config Release -j 8 \
  --target llama-cli llama-server test-llama-archs test-backend-ops
result: pass
```

`test-backend-ops -b CPU -o ADD` passed 99/99 cases. `scripts/test-qsa-shared-inputs.sh` runs the synthetic Qwen4Exp architecture test twice with seed 424242, once with sharing disabled and once enabled. The CPU, Accelerate, and Metal result rows were byte-for-byte identical:

```text
Apple M2 Max  MoE  OK (8.61e-08)  OK
Accelerate    MoE  OK (0.00e+00)  OK
Apple M2 Max  MoE  OK (8.72e-14)  OK
```

Both architecture-test invocations later hit the same pre-existing Meta backend assertion in `handle_set_rows()` (`split_states_equal(src_ss[0], src_ss[2])`). The failure is independent of the environment switch. No model weights were loaded and no active model was run.

Repository tests cover the one-set tensor shape formula, the 12:1 reduction, boundary values, invalid dimensions, profile accounting, and the execution gate.
