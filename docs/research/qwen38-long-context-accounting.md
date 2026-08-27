# Qwen3.8-Flash-Next context accounting on the 64 GB M2 Max

## Decision

The pinned llama.cpp path cannot construct the configured Q8_0 QSA graph at 262,144 tokens. It is also uncertified for F16 native context and for either static-YaRN target. The checked-in execution gate therefore refuses all long-context launches.

Allocation arithmetic alone does not establish a no-go. At native context, F16 leaves a small provisioning margin and Q8_0 leaves a larger one, but the calculation mixes Metal virtual mmap envelopes with committed context buffers. It is neither an RSS forecast nor a substitute for scheduler allocation and filled-prompt telemetry. At 500K and 1M, the same screen exceeds Metal's recommendation for F16. That result rejects the configuration under the current provisioning rule; it does not prove that every bounded runtime must fail.

Quantized main attention K/V is the smallest credible capacity reduction. The pin asserts during QSA graph construction, and its shared cache-type API also quantizes the routing indexer key. Later PR #27742 code fixes the rotation assertion. Origami still needs a separate F16 indexer type before attempting the guarded native profile. The official static-YaRN math is present in the pinned RoPE kernels, but no long-context parity test exists.

These findings rule out certifying the current mmap reference as the long-context runtime. They do not rule out a native Origami runtime.

`250K`, `500K`, and `1M` in the accounting tables mean 250,000, 500,000, and 1,000,000 tokens. llama.cpp pads those requests to 250,112, 500,224, and 1,000,192 cells. The official factor-2 research profile uses 524,288 rather than the 500K comparison row; factor 4 uses a 1,000,000-token request.

## Sources inspected

| Source | Fixed revision or digest | Relevant symbols |
|---|---|---|
| Official model repository | `Qwen/Qwen3.8-Flash-Next` `f5d08274bafd880402bd16f5e3e6c514136ec06c`; `config.json` SHA-256 `889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b` | `text_config.max_position_embeddings`, `rope_parameters`, README “Processing Ultra-Long Texts” |
| Official technical report | Qwen repository `513aa6e18a335296fc13e538232a8735b230877d`; PDF SHA-256 `04f263446d74a35cb7cea368574e0c561f3b05c133be2c777ac884404063655d` | §§2.1.1–2.1.2, equations 12–20, Tables 3–4, Figure 6 |
| Unsloth GGUF | `unsloth/Qwen3.8-Flash-Next-GGUF` `d3bc75ee6ccef3efc1e228ec00a6cc2cdb1e2249`, `UD-IQ1_S` | GGUF metadata and 1,224-tensor inventory |
| Transformers Qwen4Exp | PR #48337 commit `b61b98bea4cd99ff97da2ca0aa4fa34e8800d10e` | `Qwen4ExpTextGatedDeltaNet`, `Qwen4ExpTextQSAIndexer`, `Qwen4ExpTextNGramEmbedding`, `Qwen4ExpTextPLELayer`, `Qwen4ExpTextRotaryEmbedding`; generic `_compute_yarn_parameters` |
| Pinned llama.cpp | PR #27742 commit `bea3b12daee45876b0129a3602dc8f534ce30bf0` plus Origami's mmap diagnostic patch | `llama_model_qwen4exp::graph`, `build_qsa_top_k`, `build_layer_attn`, `build_ple`, `llama_memory_hybrid`, `llama_kv_cache`, `llama_memory_recurrent`, `llama_context::sched_reserve`, Metal `kernel_rope_multi` |
| Later PR #27742 head, comparison only | `0b19188e9` | Post-pin fixes `035e22731`, `cfbdc0a50`, `d22d2be2b`, `0ac4b1802`, `c52ed2a0b`, `24ea62df4`, `0b19188e9` |

The model metadata was read positionally. No tensor body was executed, and the 72.5 GB model was not run for this work.

## Native context semantics

The official config declares `max_position_embeddings = 262144`, default RoPE with `rope_theta = 10000000`, partial rotary factor `0.25`, and no scaling block. The GGUF agrees:

```text
qwen4exp.context_length                    262144
qwen4exp.rope.freq_base                    10000000
qwen4exp.rope.dimension_count              64
qwen4exp.rope.dimension_sections           [11, 11, 10, 0]
qwen4exp.attention.compress_ratios         [0,0,0,4] repeated 12 times
```

The report says QSA continued pretraining used 256K-token sequences. It reports RULER and MRCR results through 1M, but does not describe a different trained RoPE range. The official README supplies static YaRN for lengths above 262,144. Therefore:

| Requested length | Position semantics |
|---:|---|
| 250,000 | Native; llama.cpp allocates 250,112 cells |
| 262,144 | Native limit; no padding change |
| 500,000 | Extended; official guidance is static YaRN factor 2, which covers 524,288 |
| 1,000,000 | Extended; official guidance is static YaRN factor 4, which covers 1,048,576 |

QSA changes the selected attention set, not cache length. Twelve layers retain a full K/V record for every token. Each query selects 512 complete four-token blocks, or 2,048 tokens, plus at most three tokens from the incomplete tail. The pinned Metal path turns those indices back into a dense full-context mask and calls ordinary masked attention; it does not provide Qwen's fused sparse-core kernel or its long-context speedup. The 36 GDN layers use fixed recurrent state.

## Exact persistent payload equations

Let

```text
N = requested tokens
C = 256 * ceil(N / 256)           allocated cache cells
R_t(d) = ggml row bytes for d elements of type t
```

For F16, `R_f16(d) = 2d`. For Q8_0, `R_q8_0(d) = 34d/32`; for Q4_0 and IQ4_NL, `R(d) = 18d/32`.

### QSA K/V

The 12 QSA layers have two 256-dimensional K heads and two 256-dimensional V heads:

```text
main_KV(C, tk, tv) = 12 C [R_tk(512) + R_tv(512)]
```

The pinned indexer cache is another generic `llama_kv_cache`. It needs one raw 128-dimensional key per token, but it also allocates an unused 256-dimensional V tensor:

```text
index_KV(C, tk, tv) = 12 C [R_tk(128) + R_tv(256)]
```

With F16 this is 24,576 bytes per cell for attention K/V and 9,216 bytes per cell for the indexer cache, 33,792 bytes per cell in total. The implementation stores no persistent pooled-block or top-k cache; it rebuilds those values in the graph. `llama_memory_hybrid::memory_breakdown()` omits `mem_idx`, so the pinned runtime under-reports context allocation by exactly `index_KV`. The 512-cell PoC log confirms the omission: llama.cpp printed 12.00 MiB main K/V, 124.88 MiB recurrent state, and a separate 4.50 MiB indexer K/V buffer, while its memory breakdown reported only 136 MiB of context. Both K/V constructors clear their complete backend buffers at startup, regardless of the number of occupied cells.

### GDN and PLE state

For one sequence and no recurrent rollback snapshots:

```text
GDN matrix = 36 * 128 * 6144 * 4                    = 113,246,208 bytes
GDN conv   = 36 * 3 * (6144 + 2*16*128) * 4         =   4,423,680 bytes
GDN total                                                117,669,888 bytes

PLE conv   = 36 * (4-1) * 3 * (4*2560) * 4          =  13,271,040 bytes
RS buffer                                                130,940,928 bytes
```

Only layer 2 (one-indexed) runs PLE. The pinned recurrent allocator uses uniform rows and reserves the 92,160-element PLE convolution slice in all 36 GDN layers. A model-specific allocator would need only 368,640 bytes for the one logical PLE layer, so 12,902,400 of the pinned PLE-state bytes are avoidable.

PLE also keeps two predecessor token IDs per live sequence. That host map is negligible in capacity terms, but it belongs to the model object in the pinned commit rather than to a context. Later PR commit `d22d2be2b` moved it into context state and serialized it.

### Host cell metadata

On the measured arm64 libc++ ABI, each cache eagerly sizes four arrays per cell:

```text
llama_pos                         4 bytes
llama_kv_cell_ext                 8 bytes
shift                             4 bytes
bitset<LLAMA_MAX_SEQ=256>        32 bytes
```

The main and index caches therefore reserve `2 * 48C = 96C` bytes of array payload. This excludes allocator size classes and the `std::set`/`std::map` nodes created as cells become occupied. Those nodes depend on libc++ and malloc internals and must be measured as an envelope, not inferred as portable tensor bytes.

### Exact payload table

The following table uses F16 K/V. “Persistent” is main K/V + index K/V + GDN + PLE + the 96C eager cell-array payload. Backend alignment adds no tensor padding for these dimensions and padded values of C; C++ allocator overhead is excluded.

| Request | C | Main K/V bytes | Index K/unused-V bytes | GDN bytes | PLE bytes | Cell arrays | Persistent bytes |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 250,000 | 250,112 | 6,146,752,512 | 2,305,032,192 | 117,669,888 | 13,271,040 | 24,010,752 | 8,606,736,384 |
| 262,144 | 262,144 | 6,442,450,944 | 2,415,919,104 | 117,669,888 | 13,271,040 | 25,165,824 | 9,014,476,800 |
| 500,000 | 500,224 | 12,293,505,024 | 4,610,064,384 | 117,669,888 | 13,271,040 | 48,021,504 | 17,082,531,840 |
| 1,000,000 | 1,000,192 | 24,580,718,592 | 9,217,769,472 | 117,669,888 | 13,271,040 | 96,018,432 | 34,025,447,424 |

For capacity comparison only, replacing both K and V with Q8_0 or Q4_0 gives:

| Request | Q8_0 K/V payload | Q8_0 persistent | Q4_0 K/V payload | Q4_0 persistent |
|---:|---:|---:|---:|---:|
| 250,000 | 4,490,010,624 | 4,644,962,304 | 2,377,064,448 | 2,532,016,128 |
| 262,144 | 4,706,009,088 | 4,862,115,840 | 2,491,416,576 | 2,647,523,328 |
| 500,000 | 8,980,021,248 | 9,158,983,680 | 4,754,128,896 | 4,933,091,328 |
| 1,000,000 | 17,955,446,784 | 18,182,406,144 | 9,505,824,768 | 9,732,784,128 |

These quantized rows are not an approved pinned profile; see “Quantized K/V.”

## Graph and scratch

The pinned QSA implementation materializes full-context inputs for every QSA layer. With `U` tokens in the microbatch, each layer creates `cell_blk`, `blk_cells`, `blk_pos`, and a dense F32 `bias[C,U]`. Since C is divisible by four:

```text
QSA graph-input bytes(C,U)
  = 12 * [4C + 4C + 4C + 4CU]
  = 144C + 48CU
```

For the repository's `--ubatch-size 32` profile this exact logical input payload is:

| Request | QSA graph-input bytes |
|---:|---:|
| 250,000 | 420,188,160 |
| 262,144 | 440,401,920 |
| 500,000 | 840,376,320 |
| 1,000,000 | 1,680,322,560 |

This is a lower bound, not the scheduler buffer size. The graph also holds or schedules pooled indexer keys, gathered block members, expanded scores, top-k indices, attention masks, PLE gathers, MoE activations, and backend copies. Lifetime reuse, Metal alignment, and fused-op selection determine the final buffer. At `C=512,U=32`, the measured reserve was already 32.45 MiB on Metal plus 2.89 MiB on CPU, while the QSA input term above is only 0.82 MiB. `llama_context::sched_reserve()` builds against the full cache at startup, so the allocation is eager.

A production QSA graph must avoid twelve dense `C x U` bias inputs and the dense masks rebuilt from top-k. A compact block-validity representation plus a true sparse gather or Metal kernel should make scratch and core-attention work proportional to compressed blocks and the 2,051-token attention budget. Until the revised graph reports an actual scheduler allocation for each target C and U, no byte-accurate upper envelope exists.

The server's recurrent context checkpoints add another dynamic allocation. Each partial checkpoint is 130,940,928 bytes; the observed default allows 32, or 4,190,109,696 bytes. The pinned state writer omits the indexer cache and PLE history, so these checkpoints are also incomplete correctness state. Later PR commits `cfbdc0a50` and `d22d2be2b` address those omissions.

## Metal envelope on this Mac

The CPU-output placement wraps two no-copy mmap ranges as Metal buffers:

```text
shard 2 Metal envelope   20,468,476,672 bytes
shard 3 Metal envelope   22,544,656,128 bytes
total                    43,013,132,800 bytes (40.059 GiB)
```

The PoC log reports the same ranges as 19,520.26 and 21,500.26 MiB. They are virtual mappings, not second copies of all weights, but they count as Metal buffer envelopes and accessed pages consume unified memory. The full-Metal-output placement expands the first range to 49,990,780,672 bytes (46.558 GiB) and already failed at a 512-token context.

A direct Metal API query on this host reports `recommendedMaxWorkingSetSize = 55,662,788,608` bytes and `maxBufferLength = 41,747,087,360` bytes. Adding the current CPU-output envelopes, F16 persistent context, and only the QSA graph-input floor gives:

| Request | Provisioning lower bound | Margin to Metal recommendation |
|---:|---:|---:|
| 250,000 | 52,040,057,344 | 3,622,731,264 |
| 262,144 | 52,468,011,520 | 3,194,777,088 |
| 500,000 | 60,936,040,960 | -5,273,252,352 |
| 1,000,000 | 78,718,902,784 | -23,056,114,176 |

The lower bound omits the rest of the compute graph, CPU repack and output buffers, resident file pages, active expert and PLE pages, C++ tree nodes, checkpoint copies, macOS, and ordinary development tools. It also mixes virtual mmap envelopes with committed buffers, so it is a conservative provisioning test rather than an RSS prediction. A negative margin fails this provisioning rule but does not prove physical out-of-memory from virtual-envelope arithmetic. The roughly 3 GB native F16 margin cannot certify the uncounted allocations; Q8_0 or a split-cache design must still pass measured allocation and filled-prompt gates.

PLE has no explicit cache in this path. Its 28,800,138,240-byte IQ4_NL table remains mmap-backed, with Darwin deciding page retention and read-ahead. A bounded Origami row cache must account separately for page-aligned PLE reads. The existing worst-case cold demand remains 16 rows and at most sixteen 16 KiB pages, 262,144 bytes, per token before cache hits.

## Quantized K/V

The CLI accepts F32, F16, BF16, Q8_0, Q4_0, Q4_1, IQ4_NL, Q5_0, and Q5_1 for K and V. Quantized V forces Flash Attention. The dimensions 256, 128, and 512 satisfy all listed 32-element block sizes.

That option surface does not establish Qwen4Exp QSA support at the pin. Quantized main K/V enables Hadamard cache rotations. The pinned top-k overload of `llm_graph_context::build_attn()` asserts that `self_k_rot` and `self_v_rot` are null, so a normal quantized QSA cache cannot construct the graph. Disabling rotation with `LLAMA_ATTN_ROT_DISABLE=1` avoids that assertion but creates an untested lossy path, including quantization of the raw indexer key that decides discrete top-k routing.

Later PR commit `0ac4b1802` explicitly adds quantized-K/V rotation handling to the QSA attention path. It postdates `bea3b12`, so it cannot be credited to the pinned runtime. Commit `c52ed2a0b` separately adds Qwen4Exp to the large graph-node budget. Both belong in any quantized native candidate.

Even on that later code, indexer-key quantization needs token, logit, and selected-index parity tests. The first candidate keeps main attention K/V at Q8_0 and the 128-dimensional indexer K at F16. The current shared `type_k` and `type_v` settings cannot express that split. The smallest implementation adds separate indexer types to `llama_memory_hybrid_idx`; using its generic cache initially retains an unused 256-dimensional F16 V side. A key-only QSA allocator would remove that side, but requires a model-specific cache API.

At 262,144 cells, Q8_0 main attention plus the generic F16 indexer cache uses 5,838,471,168 KV bytes. Persistent context is 5,994,577,920 bytes, and adding the exact 440,401,920-byte dense QSA input floor gives 6,434,979,840 bytes. A key-only F16 indexer would reduce the indexer from 2,415,919,104 to 805,306,368 bytes.

## YaRN and 524,288/1M correctness

The official Hugging Face keys are:

```json
{
  "rope_type": "yarn",
  "factor": 2.0,
  "original_max_position_embeddings": 262144
}
```

Use factor 4.0 for 1M. The object must retain `mrope_interleaved`, `mrope_section`, `rope_theta`, and `partial_rotary_factor` from the released config. GGUF uses different names:

```text
qwen4exp.rope.scaling.type
qwen4exp.rope.scaling.factor
qwen4exp.rope.scaling.original_context_length
```

The Unsloth artifact contains none of those scaling keys. At the pin, llama.cpp therefore reports `rope scaling = linear` and `freq_scale = 1`. Increasing only `--ctx-size` gives unscaled positions and a training-overflow warning.

The equivalent pinned CLI settings are:

```text
524,288: --ctx-size 524288  --override-kv qwen4exp.context_length=int:524288  --rope-scaling yarn --rope-scale 2 --yarn-orig-ctx 262144
1M:      --ctx-size 1000000 --override-kv qwen4exp.context_length=int:1000000 --rope-scaling yarn --rope-scale 4 --yarn-orig-ctx 262144
```

The metadata override is needed because the pinned server caps slot context to the declared training context. It changes the exposed ceiling; `--yarn-orig-ctx 262144` retains the real scaling origin.

The implementation is more than an upstream claim: `llama-context.cpp` derives the static YaRN parameters, Qwen4Exp passes them to `ggml_rope_multi()` for both core Q/K and indexer Q/pooled-K, and Metal `kernel_rope_multi` handles interleaved MRoPE with YaRN. Its correction dimensions use `n_dims=64`, base 10,000,000, original context 262,144, and default beta values 32/1, matching the Transformers YaRN inputs. llama.cpp cancels its generic kernel's internal magnitude multiplier before the kernel reapplies it, yielding the standard static-YaRN attention scaling.

The pinned branch still lacks evidence for an end-to-end correctness claim. No test at the pin compares RoPE values with Transformers at factor 2 or 4, exercises QSA block selection near 524,288/1M, or runs long-context retrieval. The only measured Origami run used 512 cells. The official report's 1M benchmark validates Qwen's serving stack, not this llama.cpp commit. Later PR #27742 needed fixes for indexer state serialization, PLE context ownership, quantized QSA, and Qwen4Exp graph sizing. Treat the 524,288 factor-2 and 1,000,000 factor-4 paths as implemented static-YaRN code paths that remain uncertified and fail the current provisioning gate.

## Go/no-go gates

A context profile may be marked go only when all of these checks pass:

1. The allocation ledger uses padded C and includes main K/V, indexer K without an unused V side, recurrent state, PLE state/cache, graph buffers, expert cache, PLE row cache, repack/output buffers, and all checkpoint or prefix-cache copies. The sum must remain at or below 55,662,788,608 bytes, no individual Metal buffer may exceed 41,747,087,360 bytes, and telemetry must show no swap growth or swap-outs.
2. The runtime replaces the twelve `F32[C,U]` QSA bias tensors and dense top-k masks with bounded sparse metadata and a sparse Metal attention path. Record actual Metal and CPU scheduler buffer sizes at each target C and production U; source-derived lower bounds are insufficient.
3. The indexer cache appears in memory reporting and state save/restore. PLE history must be per context. Rewrites, multi-turn reuse, and prefix restore must preserve selected QSA indices and greedy tokens.
4. F16/BF16 K/V must match the Transformers reference on logits, selected blocks, and greedy tokens before any cache quantization. A quantized profile needs the same checks at short, native-boundary, and extended positions. Quantizing indexer K is a separate lossy experiment.
5. Factor-2 and factor-4 RoPE values must match Transformers for text positions around 262,143/262,144, 524,287, and 999,999 in both core attention and the pooled indexer. Retrieval and generation tests must then pass at 524,288 and 1M.
6. The bounded native runtime must keep application allocations at or below 55,662,788,608 bytes and reserve at least 12 GiB of the Mac's 68,719,476,736 bytes for macOS and development tools. The current artifact's non-streamed resident-weight basis is 3,889,410,560 bytes; the 39,845,888,000 routed-expert bytes and 28,800,138,240 PLE bytes must stay behind explicit caches.

Until those gates pass, 250K and 262,144 remain algorithmically native but unavailable in the pinned mmap runtime. The configured Q8_0 profile fails graph construction, while F16 has no allocation or filled-context proof. The 524,288 factor-2 and 1,000,000 factor-4 static-YaRN profiles remain separate, nonlaunchable experiments with failed provisioning screens and no correctness evidence.

## Reproduction artifact

`tools/qwen38_context_accounting.py` implements the equations without opening a GGUF:

```sh
python3 tools/qwen38_context_accounting.py
python3 tools/qwen38_context_accounting.py --type-k q8_0 --type-v q8_0
python3 tools/qwen38_context_accounting.py 262144 \
  --type-k q8_0 --type-v q8_0 \
  --index-type-k f16 --index-type-v f16 --checkpoints 0
```

Pass `--index-key-only` only to model the required allocator change. It does not describe the pin or the later inspected PR head. `--checkpoints 32` adds the pinned maximum of recurrent-state copies. `tests/test_qwen38_context_accounting.py` fixes the padding, state, F16, Q8_0, Q4_0, split-indexer, checkpoint, graph-floor, and YaRN-band values used above.
