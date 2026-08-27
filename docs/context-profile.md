# Qwen3.8 native long-context profile

## Status

The pinned model declares a native training context of 262,144 tokens. The profile in [`config/qwen38-context-262144.json`](../config/qwen38-context-262144.json) describes a single-slot server at that capacity with Q8_0 K and V caches. It has not been allocated on the 64 GiB test host. The user's temporary 16,384-token server was active during this work, so no second server or 262,144-token allocation was started.

The current evidence reaches a 977-token prompt and a maximum observed sequence length of 980 tokens. It does not validate a 16,384-token prompt, much less the model's native limit. Server allocation, `/health`, and a declared `n_ctx` prove configured capacity. Only a completed prompt stage proves operation at that length.

## Exact native server profile

Stop the temporary server before using this command. `scripts/context-profile.sh allocate` refuses to run while any `llama-server` process is present.

```sh
STATE=/private/tmp/origami-context-262144
MODEL=/path/to/UD-IQ1_S

# Print the command without starting a server.
scripts/context-profile.sh command "$MODEL"

# Start one owned server, allocate its caches, enforce the memory gates, and
# stop after health and startup-log checks. The server remains running.
scripts/context-profile.sh allocate "$MODEL" --state-dir "$STATE"

# Repeat the allocation checks without generation.
scripts/context-profile.sh health --state-dir "$STATE"

# Run increasing token-ID prompts. Any gate failure terminates the owned server.
scripts/context-profile.sh probe --state-dir "$STATE" \
  --output "$STATE/probe.json"

scripts/context-profile.sh stop --state-dir "$STATE"
```

The rendered server command contains these material settings:

```text
GGML_METAL_NO_RESIDENCY=1
LLAMA_MMAP_PREFETCH=0

llama-server --offline --host 127.0.0.1 --port 18080 \
  --alias origami-qwen38-262144 --model MODEL.gguf \
  --load-mode mmap --gpu-layers all --override-tensor '^output=CPU' \
  --fit off --ctx-size 262144 --batch-size 32 --ubatch-size 32 \
  --parallel 1 --no-kv-unified --kv-offload \
  --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on \
  --no-context-shift --cache-ram 0 --no-cache-prompt --cache-reuse 0 \
  --ctx-checkpoints 0 --no-cache-idle-slots --no-warmup \
  --jinja --metrics --slots --perf --log-verbosity 4
```

`--fit off` makes an allocation failure visible instead of changing placement. The pinned fit code does not reduce an explicitly set context, but disabling fit also prevents an automatic layer-placement change. `--parallel 1` gives the one slot the whole context. `--no-context-shift` makes the server stop at the boundary instead of discarding old tokens. Native 262,144 operation uses the model's RoPE metadata unchanged, so this profile has no YaRN or other RoPE override.

The two mmap safeguards and CPU output placement are the same settings that passed the short PoC. CPU output keeps the 28.8 GB PLE table out of the unsafe Metal mmap envelope. The safeguards suppress eager prefetch and Metal residency requests. They do not impose a physical cache ceiling on routed weights or the PLE table.

## KV and QSA accounting

Qwen4Exp has 12 attention layers among its 48 layers. Each attention token stores 512 K elements and 512 V elements. QSA also stores an indexer cache with 128 K elements and 256 V elements on those 12 layers. The pinned hybrid-memory constructor passes `--cache-type-k` and `--cache-type-v` to both caches. There is no separate QSA cache-type flag.

Q8_0 stores each 32-element row block in 34 bytes. At 262,144 cells:

| Allocation | Bytes | GiB |
|---|---:|---:|
| Attention K and V | 3,422,552,064 | 3.1875 |
| QSA indexer K and V | 1,283,457,024 | 1.1953125 |
| Total KV | 4,706,009,088 | 4.3828125 |

The active 16,384-token server uses F16 and reports 384 MiB for attention plus 144 MiB for the QSA indexer. F16 would scale to 8.25 GiB at native context. Q4_0 would use 2.3203125 GiB. Q8_0 is the initial profile because it halves the F16 allocation without starting with a 4-bit cache. The cache quantization still needs output-quality testing.

Quantized V requires Flash Attention in the pinned context constructor. The profile sets `--flash-attn on` rather than relying on `auto`. The pinned Metal source includes Q8_0 Flash Attention kernels for the model's 256 by 256 attention heads, and 128 and 256 are divisible by Q8_0's 32-element block. Startup must report both `flash_attn = on` and `Flash Attention enabled`.

QSA limits the selected attention width to `min(n_kv, 2048 + ratio - 1)`, with ratio 4 for these layers. It does not make the caches bounded: attention and indexer KV still allocate one cell per context token. The current graph also creates an F32 `[n_kv, n_ubatch]` QSA bias for every QSA layer and fills it on the host. With a microbatch of 32, those 12 bias tensors alone represent 384 MiB at 262,144 cells and about 1.43 GiB at one million cells. The allocation probe must measure actual compute buffers rather than treating the KV table as the whole context cost.

## Prompt cache and checkpoint overhead

The temporary server already uses `--cache-ram 0`, so the bounded server prompt cache is disabled. Its request default still permits slot-prefix reuse. More importantly, the default 32 context checkpoints remain enabled. Qwen4Exp's recurrent memory cannot support partial sequence removal, and the log shows three checkpoints per chat request at 124.876 MiB each. The configured maximum could retain about 3.90 GiB per slot.

The native profile sets all of the following:

- `--cache-ram 0` to disable the extra RAM prompt cache
- `--no-cache-prompt` and `--cache-reuse 0` to keep staged probes independent
- `--ctx-checkpoints 0` to remove recurrent checkpoint copies
- `--no-cache-idle-slots` to avoid an option that requires the RAM cache
- no `--slot-save-path`, so no serialized KV state is written

These choices trade repeat-request speed and checkpoint rollback for a smaller, easier-to-account allocation. They do not remove the live KV and recurrent state needed by the active slot.

## Staged proof and abort behavior

`allocate` performs model shard size checks, verifies the pinned revision marker, rejects every existing `llama-server`, and starts the server in a new process group. It samples macOS system telemetry while waiting. A pass requires:

- `/health` returns `{"status":"ok"}`
- `/props` reports `n_ctx = 262144`
- `/v1/models` is readable
- logs prove the two mmap safeguards, CPU output, Q8_0 attention and indexer caches, forced Flash Attention, disabled prompt cache, and disabled checkpoints
- logs contain no compute, scheduler, decode, Metal out-of-memory, or context-reduction marker

The default hard gates are a memory-pressure free percentage of at least 15%, zero growth in swap used, zero swap-out pages, and no more than 2 GiB growth in compressor occupancy. The gates compare counters with the allocation or probe baseline, so pre-existing swap does not masquerade as inference growth. A failing allocation terminates the new process group.

`probe` accepts only a server PID recorded by a passing `allocate` state. It sends raw token-ID prompts with one generated token at these lengths:

```text
1024, 4096, 16384, 32768, 65536, 131072, 196608, 245760, 258048
```

Each request disables prompt reuse. A stage passes only when the response reports at least the requested prompt length and `llamacpp:n_tokens_max` reaches it. The monitor samples every 0.25 seconds. A memory gate, swap gate, request timeout, HTTP failure, or missing token-count proof terminates the recorded server. The tool validates the PID's executable, model path, port, and process-group ownership before signaling it.

System counters are global. Another memory-intensive process can correctly trip a gate even if llama.cpp did not cause the pressure. Run the probe on an otherwise quiet host and retain the JSON record and server log.

## Current temporary server

The active temporary process was inspected without submitting another prompt. Its command configures 16,384 tokens, one slot, F16 caches, `--cache-ram 0`, automatic Flash Attention, CPU output, and both lazy-mmap environment variables. Its startup log proves:

- model metadata and `n_ctx_train`: 262,144
- allocated `n_ctx` and slot context: 16,384
- Flash Attention resolved on
- attention KV: 384 MiB F16
- QSA indexer KV: 144 MiB F16
- recurrent state: 124.88 MiB
- context checkpoints: 32 maximum, 8,192 minimum spacing

The `/props` snapshot reported `n_ctx = 16384`. Metrics reported 1,943 prompt tokens processed, 169 predicted tokens, and `llamacpp:n_tokens_max = 980`. The longest log record was a 977-token prompt followed by generation to sequence length 980. Later requests were shorter. `/health` was healthy, and the sampled server was idle.

At the snapshot, system memory pressure reported 80% free and system swap already had about 7.04 GiB in use. `vmmap` reported a 1.5 GiB physical footprint for the server, 67.3 GiB of mapped files with about 129 MiB resident, and 1.5 GiB swapped writable regions. These idle values do not account for all file-backed pages that Metal may make resident during a long prompt.

The temporary Pi profile declares `contextWindow: 16384` and `maxTokens: 2048`, matching that server. It does not declare 262,144. [`config/pi-model-origami-262144.json`](../config/pi-model-origami-262144.json) is a replacement example for use only with the native server. Its `contextWindow` is 262,144. Pi's declaration controls client-side token budgeting and compaction; it cannot enlarge the server.

## Honest path to 524,288 and 1,000,000

The official model configuration uses default mRoPE, a 10,000,000 base, and `max_position_embeddings = 262144`. The official model card calls 262,144 native and recommends static YaRN beyond that point. It gives factor 2 for a typical 524,288-token workload and factor 4 for the advertised one-million-token mode, with `original_max_position_embeddings = 262144`. Static YaRN can reduce short-context quality, so these must be separate profiles rather than changes to the native default.

[`config/qwen38-context-yarn-research.json`](../config/qwen38-context-yarn-research.json) records both unvalidated targets without making either one launchable through the native profile tool. The matching pinned llama.cpp flags would be:

```text
# 524,288 configured cells
--ctx-size 524288 \
--override-kv qwen4exp.context_length=int:524288 \
--rope-scaling yarn --rope-scale 2 --yarn-orig-ctx 262144

# Official one-million-token ceiling, not 1,048,576
--ctx-size 1000000 \
--override-kv qwen4exp.context_length=int:1000000 \
--rope-scaling yarn --rope-scale 4 --yarn-orig-ctx 262144
```

The metadata override is necessary in this pinned server. `server-context.cpp` caps each slot to `n_ctx_train`; setting only `--ctx-size` and YaRN would allocate a larger context and then expose a 262,144-token slot. The override changes the server's declared training ceiling, while `--yarn-orig-ctx 262144` preserves the actual YaRN origin. The pinned parser accepts the `int:` override for the GGUF's U32 context key. For one million, llama.cpp pads the internal allocation to a 256-cell boundary and the server cap exposes the requested 1,000,000 cells.

These flags are a configuration recipe, not a validated backend. Before either profile can be called supported, it needs:

1. a dedicated config and Pi declaration that name it as YaRN rather than native;
2. startup allocation proof with exact cache and compute-buffer logs;
3. Q8_0 quality comparison, followed by an explicitly lossy Q4_0 cache profile only if memory requires it;
4. staged prompts that cross 262,144 and reach the target range;
5. long-context retrieval or passkey tests against a trusted implementation, not only repeated synthetic token IDs.

At 524,288 cells, Q8_0 attention plus indexer KV is 8.765625 GiB; Q4_0 is 4.640625 GiB. At 1,000,000 cells, those values are about 16.72 GiB and 8.85 GiB. The latter also carries the QSA graph costs described above. The mmap weight path remains unbounded, so fitting the nominal KV allocation does not establish safe operation on 64 GiB.

The pinned Qwen4Exp support is experimental and absent from the comparison `master` revision recorded in the PoC decision. Its architecture tests cover a small synthetic QSA graph, not YaRN at 524,288 or one million, and no independent resident oracle has validated the current GGUF. Generic `ggml_rope_multi` receives YaRN parameters, but Qwen4Exp has no model-specific long-context regression. The host-side QSA map and bias construction, unbounded routed-weight residency, and lack of end-to-end long-prompt telemetry are the smallest concrete backend gaps to close before advertising either extended window.

## Sources inspected

- Pinned `llama-server --help` at `bea3b12daee45876b0129a3602dc8f534ce30bf0`
- `common/arg.cpp` for context, KV, Flash Attention, checkpoint, prompt-cache, and YaRN parsing
- `src/llama-context.cpp` for context padding, Flash Attention requirements, and YaRN defaults
- `src/llama-model.cpp` for the Qwen4Exp hybrid attention, recurrent, and indexer caches
- `src/models/qwen4exp.cpp` and `src/llama-kv-cache.cpp` for QSA width, tensors, and host setup
- `tools/server/server-context.cpp` for slot capping, context shift, checkpoints, and oversized-prompt rejection
- Metal `kernels/fa.metal` for Q8_0 Flash Attention head-size specializations
- [Official Qwen3.8-Flash-Next model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), including its native and static-YaRN recipes
- Pi `docs/models.md` for `contextWindow` and `maxTokens` semantics
