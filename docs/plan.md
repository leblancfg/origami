# Bring-up plan

## Constraints observed in the first GGUF

Snapshot: 2026-08-26, Unsloth `UD-IQ1_S`.

| Group | On-disk size |
|---|---:|
| Routed experts | 39.846 GB |
| PLE n-gram table | 28.800 GB |
| Dense backbone | 3.697 GB |
| Shared experts | 0.193 GB |
| Total tensor data | 72.535 GB |

The routed expert tensors use dynamic quantization:

- Down projections: IQ4_NL
- Gate and up projections: IQ1_S or IQ2_XXS
- Dense tensors: Q4_K through Q8_0, with selected BF16/F32 tensors

At 10 selected experts out of 512, the uncached routed-expert demand is approximately 0.778 GB per token. The GGUF stores each expert contiguously within the layer's gate, up, and down tensors, which should permit medium-sized parallel reads rather than page-sized random I/O.

The current GGUF appears text-only. Its tensor inventory contains neither vision nor MTP tensors.

## Milestone 0: reference oracle

- Build the Qwen4Exp llama.cpp branch from upstream PR #27742.
- Run the smallest available Unsloth quant on a machine with enough memory.
- Capture greedy tokens and selected intermediate tensors for short prompts.
- Record the exact GGUF revision and llama.cpp commit.

Exit condition: repeatable reference output and a known-good command line.

## Milestone 1: metadata and allocation ledger

- Read split GGUF metadata without mapping tensor bodies.
- Resolve every required tensor and reject unknown required families.
- Build a byte-accurate inventory for resident, streamed, cached, and temporary allocations.
- Decode expert offsets into `(layer, expert, projection)` records.

Exit condition: the loader can explain every byte before allocating memory.

## Milestone 2: resident scalar path

- Implement the Qwen4Exp text graph for one sequence.
- Start with ordinary resident weights on a large-memory machine.
- Add golden checks around PLE, hyper-connections, GDN, QSA, routing, and logits.

Exit condition: greedy token parity with the llama.cpp reference.

## Milestone 3: PLE on SSD

- Keep the 28.8 GB n-gram tensor unmapped or explicitly evictable.
- Compute the 16 row IDs from prior token IDs.
- Fetch page-aligned row groups into a bounded cache.
- Measure cold and warm latency independently.

Exit condition: parity with the resident path and bounded PLE memory.

## Milestone 4: streamed experts

- Keep dense and shared tensors resident.
- Submit the selected gate, up, and down reads in parallel within each layer.
- Bypass the operating system page cache where the platform permits it.
- Add a fixed-size expert cache with deterministic eviction and telemetry.
- Report bytes read, cache hit rate, I/O wait, compute time, and wall time per token.

Exit condition: the model runs below the 64 GB machine's safe resident-memory limit with no swap growth and no output divergence.

## Milestone 5: performance work

Test one change at a time:

- Cache sizing and replacement policy
- Packed expert sidecar versus direct GGUF slices
- Read coalescing and lane count
- Overlap between storage and compute
- Metal versus CPU placement for dense operations
- Context-cache quantization

Predictive prefetch should wait until the baseline is measured. A router cannot usually predict the next layer before the current layer finishes, so speculative reads must repay their wasted bandwidth.

## Deferred work

- MTP speculative decoding
- Vision encoder
- Multiple concurrent sequences
- Million-token YaRN mode
- Expert dropping or reduced top-k
- Network or multi-device expert storage
