# Qwen3.8 long-context integration record

## Operational status

The earlier blocked analysis below has been superseded by the validated `213df585` explicit-SSD profile. Origami now allocates 262,144 context cells, streams routed experts through ten bounded `pread` slots, keeps the QSA routing key F16, and exposes the full window to temporary Pi. Allocation, one-token intermediate parity, sustained decode, and Pi end-to-end checks pass without swap growth or swap-outs.

Use [`config/qwen38-explicit-ssd-262144.json`](../config/qwen38-explicit-ssd-262144.json), [`scripts/start-explicit-ssd-server.sh`](../scripts/start-explicit-ssd-server.sh), and [the measured result](results/qwen38-explicit-ssd-262k.md). The remainder of this document records the constraints and evidence that led to that profile; statements describing the path as blocked are historical.

The two original reports measured different parts of the mmap reference path. At pinned llama.cpp commit `bea3b12daee45876b0129a3602dc8f534ce30bf0`, quantized QSA cache construction asserted and long-context operation was unavailable. The validated profile moved to `213df585`, integrated the post-pin QSA/state fixes, split cache types, shared QSA inputs, and explicit expert streaming.

The evidence supports five separate findings:

| Question | Finding |
|---|---|
| Configured allocation feasibility | Plausible, not proved. The proposed Q8_0-main/F16-indexer context and dense-input lower bound is 6,434,979,840 bytes. Adding the measured 43,013,132,800-byte CPU-output Metal virtual envelope leaves 6,214,675,968 bytes under Metal's recommendation. Virtual mappings are not resident bytes, so this is a provisioning screen rather than an RSS prediction or a no-go proof. |
| Graph-construction feasibility | Failed at the pin for Q8_0. `build_attn_qsa()` asserts on quantized-cache rotation. The pin also omits Qwen4Exp from the large graph-node budget. Later PR commits fix both defects, but Origami has not built or run them. |
| Filled-context memory safety | Unknown. No 262,144-cell allocation or filled prompt has run. The lower bound omits the full scheduler allocation, active mmap pages, tree nodes, output/repack buffers, and OS demand. |
| Long-prompt operation | Unknown. The longest observed prompt is 977 tokens and the largest observed sequence length is 980. The pinned graph builds twelve dense `F32[C,U]` QSA bias inputs and fills them on the host for every microbatch. |
| Correctness certification | Failed by lack of evidence. The pin omits indexer state from save/restore, keeps PLE history on the model, and has no quantized-cache, native-boundary, or YaRN parity result. |

A healthy `/health` response and `n_ctx = 262144` would prove allocation and declared capacity only. They would not change the final three findings.

## Fail-closed profile

[`config/qwen38-context-262144.json`](../config/qwen38-context-262144.json) retains the exact one-slot server arguments, telemetry thresholds, cache ledger, and staged prompt plan. It requires a build-local `origami-context-capabilities.json`. The long-context bootstrap records the upstream fixes and each checked-in local delta. A missing, stale, or incomplete manifest blocks execution.

```sh
# Safe and non-invasive.
scripts/context-profile.sh status

# Refuses before it prints a shell command or starts a server.
scripts/context-profile.sh command /path/to/UD-IQ1_S
scripts/context-profile.sh allocate /path/to/UD-IQ1_S \
  --state-dir /private/tmp/origami-context-262144
```

The gate requires these named capabilities:

1. `qwen4exp_indexer_cache_coupled_and_reported`, corresponding to upstream `035e22731a7fd70b9854b3a2d64ec68e9b1a45d3`.
2. `qwen4exp_qsa_quantized_kv_rotation`, corresponding to `0ac4b18025c2e255dd76252cd3b465683d08b257`.
3. `qwen4exp_large_graph_node_budget`, corresponding to `c52ed2a0b0b865e82eb1b393106c48df1c39cb32`.
4. `qwen4exp_f16_indexer_cache_split`, an Origami delta that is not present at the inspected PR head.
5. `qwen4exp_shared_qsa_graph_inputs`, the opt-in, shape-checked one-set QSA input prototype.

An upstream capability entry must contain the exact fixing commit recorded by the profile. The local split needs nonempty build evidence, and the manifest's runtime revision must match the profile. Revision and patch markers remain mandatory. The gate is an execution interlock, not a correctness certificate.

The Pi file at [`config/pi-model-origami-262144.json`](../config/pi-model-origami-262144.json) is now a loadable provider example for the validated explicit-SSD server. Earlier revisions kept it deliberately blocked until the server passed allocation and generation gates.

## Upstream trace

Origami pins `bea3b12`. The later PR #27742 head inspected for this integration is `0b19188e9`; it includes the following post-pin changes:

| Commit | Effect on this review |
|---|---|
| `7fee670a9` | Keeps the Qwen4Exp top-k mask implementation architecture-local. |
| `035e22731` | Introduces `llama_memory_hybrid_idx`, couples indexer and attention cell layouts, and includes indexer memory in the breakdown. |
| `cfbdc0a50` | Saves and restores indexer KV state. |
| `d22d2be2b` | Moves PLE n-gram history into context state and serializes it. |
| `d4a943f9a` | Cleans up Qwen4Exp comments and image-token handling without closing a long-context gate. |
| `0ac4b1802` | Applies and reverses quantized-cache Hadamard rotations in the QSA attention path, removing the pinned assertion. |
| `c52ed2a0b` | Gives Qwen4Exp the large graph-node budget used by other large hybrid graphs. |
| `6a69a0c12` | Removes an unused Qwen4Exp variable that breaks warning-as-error builds. |
| `ef9fa1ba1` | Quantizes large tensors in row bands; it does not change context memory. |
| `5674c73aa` | Segments fused QKV for tensor split; it does not split main and indexer cache types. |
| `24ea62df4` | Fixes PLE history erasure during `seq_rm(-1)` and a fatal-warning build failure. |
| `0b19188e9` | Adds the explicit include needed by the Hadamard helper. |

Later commits repair real state and graph defects, but they add neither a split indexer cache type nor a bounded sparse QSA scheduler allocation.

## Smallest credible native runtime delta

Use the later Qwen4Exp fixes as the base. Keep main attention K and V at Q8_0, while keeping the routing indexer's K at F16. `llama_memory_hybrid_idx` currently passes the main `type_k` and `type_v` to its generic indexer `llama_kv_cache`, so the public settings cannot express that split.

The smallest first implementation is model-local:

1. Add separate indexer K/V types to `llama_memory_hybrid_idx`.
2. Select F16 indexer K and F16 indexer V for Qwen4Exp while retaining Q8_0 main attention K/V.
3. Emit an explicit cache-type log marker and account for the indexer in `memory_breakdown()`.
4. Keep checkpoints, prompt reuse, context shift, and the RAM prompt cache disabled during allocation and long-prompt probes.

This version still pays for a 256-element F16 indexer V tensor that QSA does not use. Removing it needs a key-only indexer allocator or a QSA-specific cache API; the generic KV allocator always creates both sides. A key-only cache would reduce native indexer payload from 2,415,919,104 to 805,306,368 bytes. That allocator improvement is useful but not required for the first guarded allocation attempt.

The delta must retain the later cache-rotation and graph-budget fixes. Applying only the F16 split to `bea3b12` would leave the main Q8_0 attention assertion in place.

## Native memory ledger

At 262,144 cells and microbatch 32:

| Configuration | Main attention KV | Indexer K/unused V | Persistent context | Dense QSA input floor | Context plus input floor |
|---|---:|---:|---:|---:|---:|
| Pinned F16 | 6,442,450,944 | 2,415,919,104 | 9,014,476,800 | 440,401,920 | 9,454,878,720 |
| Integrated Q8_0/Q8_0 profile, graph-invalid at pin | 3,422,552,064 | 1,283,457,024 | 4,862,115,840 | 440,401,920 | 5,302,517,760 |
| Proposed Q8_0 main, F16 indexer | 3,422,552,064 | 2,415,919,104 | 5,994,577,920 | 440,401,920 | 6,434,979,840 |
| Proposed Q8_0 main, F16 key-only indexer | 3,422,552,064 | 805,306,368 | 4,383,965,184 | 440,401,920 | 4,824,367,104 |

Persistent context includes 117,669,888 bytes of GDN state, 13,271,040 bytes of PLE state, and 25,165,824 bytes of eager host cell arrays. It excludes allocator overhead and occupied-cell tree nodes.

The upstream dense QSA figure is an exact logical input payload, not the complete scheduler buffer:

```text
12 * (12C + 4CU) = 440,401,920 bytes, C=262144, U=32
```

The checked-in opt-in prototype shares that layer-independent input set. With `LLAMA_QSA_SHARED_INPUTS=1`, the same boundary uses `12C + 4CU = 36,700,160` input bytes. It preserves the existing score expansion, top-k operation, and dense attention mask, so this is a host-input bound rather than a sparse-attention implementation or scheduler-buffer bound. See [the build and parity record](results/qsa-shared-inputs-213df585.md).

At the 512-cell PoC, the actual scheduler reserve was much larger than the input term. The guarded allocation must record Metal and CPU scheduler buffers at the target context.

Each recurrent checkpoint copies 130,940,928 logical bytes. The pinned default maximum of 32 adds 4,190,109,696 bytes. The research profile sets `--ctx-checkpoints 0`. It also sets `--cache-ram 0`, `--no-cache-prompt`, and `--cache-reuse 0`.

The measured CPU-output Metal mapping envelope is 43,013,132,800 bytes. It is virtual address coverage for no-copy mappings. Adding it to committed context buffers is conservative for provisioning, while treating the total as physical occupancy is incorrect. Allocation telemetry and a filled prompt are required to settle memory safety.

## Native proof sequence after the capability gate

A capability-complete build may run the existing guarded sequence:

1. `allocate` checks shards, revision and patch markers, the capability manifest, startup logs, `/health`, `/props`, and system memory counters.
2. `health` repeats non-generating checks against the owned PID.
3. `probe` sends increasing raw-token prompts through 258,048 tokens. Any pressure, swap, timeout, HTTP, token-count, or backend-log failure terminates the owned process group.
4. A separate boundary probe must evaluate positions 262,143 and 262,144 before the profile can claim the full native window.

Passing these steps establishes allocation, filled-context safety on the measured host, and basic long-prompt operation. Correctness still needs logits, selected QSA blocks, and greedy-token parity against a trusted Qwen4Exp implementation. Tests must cover fresh prompts, rewrites, state restore, and multi-turn reuse.

## Static-YaRN research profiles

The official model guidance uses static YaRN only above the native 262,144-token range. Origami records two independent, nonlaunchable experiments:

- [`qwen38-context-yarn-524288-research.json`](../config/qwen38-context-yarn-524288-research.json): 524,288 tokens, factor 2.
- [`qwen38-context-yarn-1000000-research.json`](../config/qwen38-context-yarn-1000000-research.json): 1,000,000 requested tokens, factor 4, with 1,000,192 allocated cells after llama.cpp padding.

Each profile preserves `original_max_position_embeddings = 262144`. Neither changes the native profile or supplies a Pi model. They need independent factor-specific RoPE parity for core attention and the pooled indexer, allocation evidence, prompts crossing 262,144, and retrieval/generation comparison. Static YaRN can affect short-context quality, so one factor cannot stand in for the other.

## Recommendation

Continue using the 512-token pinned PoC for reference work. Treat the native profile as an allocation candidate, not a certified server profile, and do not start the 524,288 or 1,000,000 profiles. Run the guarded native allocation and staged prompt sequence on an otherwise quiet 64 GB host. Keep checkpoints disabled. Do not publish the Pi profile until filled-context safety, long-prompt operation, and correctness have separate passing records.
