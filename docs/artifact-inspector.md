# GGUF artifact inspector

`origami_artifacts` reads GGUF v3 metadata with positional, bounded reads. It does not use `mmap`, and ordinary inspection never requests tensor-body bytes. The current type table covers the GGML types found in the Qwen3.8 Flash Next `UD-IQ1_S` artifact. An unknown type, malformed quantization row, overlapping span, or unfamiliar routed tensor layout stops inspection.

## Usage

Inspect one file or every `*.gguf` file in a directory:

```sh
python3 -m origami_artifacts /path/to/UD-IQ1_S
python3 -m origami_artifacts /path/to/UD-IQ1_S --json --output ledger.json
```

The JSON includes tensor spans and `(layer, expert, projection)` slices. Omit the slices when only aggregate accounting is needed:

```sh
python3 -m origami_artifacts /path/to/UD-IQ1_S --json --no-slices
```

Cache and scratch budgets become entries in the allocation ledger:

```sh
python3 -m origami_artifacts /path/to/UD-IQ1_S \
  --expert-cache 8GiB --ple-cache 2GB --temporary 512MiB
```

`GB` uses powers of 1000. `GiB` uses powers of 1024. JSON records normalized integer byte counts so later C++ code does not need to reinterpret the input units.

Inspection permits missing shards and truncated tensor bodies because metadata can arrive before weights. The report marks its scope as `present metadata shards only` and identifies unavailable spans. It does not present partial totals as full-artifact totals. If every declared shard is present, the inspector requires the parsed tensor count to match `split.tensors.count`.

## Bounded span probe

A probe checks a selected computed span against the current shard size, then calls `pread` at both ends. It reads one byte per edge unless `--probe-bytes` changes the bound:

```sh
python3 -m origami_artifacts /path/to/UD-IQ1_S \
  --probe blk.0.ffn_gate_exps.weight --probe-bytes 8
```

The command fails if the tensor is unknown, its computed end exceeds the file, or either exact read is short. The maximum edge read is 4096 bytes.

## Current layout contract

The Qwen4Exp adapter accepts routed tensors named:

```text
blk.<layer>.ffn_{gate,up,down}_exps.weight
```

Each tensor must have three GGUF dimensions. Dimension 2 is the expert axis and must equal `qwen4exp.expert_count`. Gate and up dimensions must be `(embedding_length, expert_feed_forward_length, expert_count)`; down dimensions reverse the first two axes. One expert therefore occupies a contiguous `tensor_bytes / expert_count` range. A complete Qwen4Exp split must contain all three routed projections for every declared block.

The classifier assigns `per_layer_token_embd.weight` to PLE, names containing `shexp` to shared experts, the routed family above to routed experts, and all remaining recognized tensors to the dense group.

## First-shard regression

The checked Unsloth first shard is 10,946,624 bytes. It contains the global metadata and no tensor descriptors. It declares three shards and 1,224 tensors; its parsed metadata ends at byte 10,946,618 and aligns exactly to the file size. Tests assert those values when the local shard is available. They also check the 320,001,446 vocabulary-row sum and the known padded PLE tensor baseline: 320,001,536 IQ4_NL rows occupying 28,800,138,240 bytes.

The remaining byte totals cannot be reconstructed from that file alone because GGUF stores each tensor's dimensions, quantization type, and relative offset in the shard that owns the tensor. The inspector reports zero parsed tensor bytes and an incomplete scope for the first shard rather than substituting the rounded totals in `docs/plan.md`. Synthetic split fixtures cover byte-exact totals, quantized expert slicing, body truncation, and bounded probes. A full regression will run unchanged once all three metadata regions are available; model weights do not need to be resident or mapped.
