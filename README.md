# Origami

Origami is an experimental inference project for running sparse models whose full weights do not fit in memory. The first target is Qwen3.8-Flash-Next on 64 GB Apple Silicon, with routed experts streamed from SSD.

The repository is private and exploratory. There is no working runtime yet.

## Working hypothesis

Qwen3.8-Flash-Next is a good fit for explicit weight streaming:

- The language model has 125B parameters but activates about 6B per token.
- Its 48 MoE layers select 10 of 512 routed experts, plus one shared expert.
- The 51B n-gram embedding table needs a handful of row lookups per token and can remain on SSD behind a small row cache.
- The always-used tensors in Unsloth's first GGUF occupy roughly 3.9 GB. They can remain resident while expert tensors and the n-gram table stay on disk.

The current Unsloth `UD-IQ1_S` GGUF is 72.5 GB. About 39.8 GB belongs to routed experts and 28.8 GB to the n-gram table. Reading every selected expert without cache hits would move about 778 MB per generated token.

## Initial design

1. Keep dense weights, shared experts, recurrent state, and a modest context resident.
2. Build an index of contiguous expert slices in the GGUF shards.
3. Fetch only the selected gate, up, and down slices with asynchronous direct I/O.
4. Use a bounded expert cache rather than the operating system page cache.
5. Serve n-gram rows from SSD through a separate page-aligned row cache.
6. Compare greedy output against the upstream llama.cpp Qwen4Exp implementation at every milestone.

The first implementation should optimize correctness and memory accounting. Expert prediction, speculative reads, MTP, vision, and lossy expert dropping come later.

## First target

- Hardware: a 64 GB Apple Silicon Mac with fast internal NVMe
- Model: `unsloth/Qwen3.8-Flash-Next-GGUF`
- Mode: text-only, single sequence, short context during bring-up
- Memory: enough headroom to avoid macOS compression and swap storms
- Correctness: greedy token parity with a resident reference run
- Performance: stable interactive decoding without relying on accidental `mmap` cache behavior

See [docs/plan.md](docs/plan.md) for the bring-up sequence and [docs/sources.md](docs/sources.md) for upstream work.
