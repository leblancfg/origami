# Intent

Origami exists to make capable open-weight models practical on Macs that ordinary developers can afford. Its primary target is a 64 GB Apple Silicon machine. A 32 GB machine is a valid target when the model and quantization permit it. Designs that require 128, 256, or 512 GB may be useful elsewhere, but they do not satisfy this project's goal.

DS4 demonstrated one version of this idea for DeepSeek-V4-Flash. Origami takes the underlying approach and makes it reusable: keep the small, frequently used part of a sparse model in memory, fetch the large conditional part only when routing selects it, and control the storage path instead of leaving virtual memory to improvise under pressure.

Lucebox's [PR #625](https://github.com/Luce-Org/lucebox/pull/625) demonstrates another part of the intended approach on consumer GPUs: pair the target with a model-specific drafter, fuse the expensive recurrent operations, adapt speculation to the workload and context length, and validate end-to-end speed against output quality. Origami will apply that same degree of model and runtime co-design to Metal rather than treating Apple Silicon as a generic CPU with shared memory.

## What we are building

Origami will be a local inference runtime and a collection of model integrations for memory-constrained Apple Silicon. It should make models usable, fast enough for interactive work, and predictable under sustained generation.

"Usable" includes the whole process:

- The model loads without consuming every byte of memory or forcing macOS into a swap storm.
- Generation remains stable after caches warm up and across longer responses.
- Context, KV state, compute buffers, and the operating system fit alongside the weights.
- Outputs remain faithful to a trusted resident implementation unless the user explicitly enables a lossy mode.
- The runtime exposes a conventional local API so editors, agents, and command-line tools can use it without custom integration.

A model that emits one demonstration response while the machine is under severe memory pressure is not considered supported.

## Hardware budget

The hardware budget is a design constraint, not a benchmark category added at the end.

The default development target is:

- Apple Silicon with 64 GB unified memory
- Metal as the primary accelerated compute backend
- The machine's internal NVMe SSD
- No second computer, external accelerator, or distributed storage requirement
- Enough free memory for macOS and normal development tools

CPU execution remains useful for reference operations and placements that Metal cannot run efficiently. Unsupported graph variants must fail closed rather than silently execute different math on CPU. CUDA, HIP, and Vulkan implementations are valuable references, but their performance assumptions do not define Origami's design.

Selected models should also receive a 32 GB profile. That profile may use a smaller quant, shorter context, lower cache budget, or a slower storage path. Any quality reduction must be named and measured.

Machines with 128 GB or more are useful for reference runs and correctness comparisons. We will not use their capacity to hide allocations that break the 64 GB target.

## More than one model

Qwen3.8-Flash-Next is the first integration, not the architecture of the repository. Model-specific code should sit behind narrow adapters for configuration, tensor naming, routing, state, and graph construction. Storage, caching, scheduling, telemetry, and API serving should remain shared.

The same rule applies to providers. Official checkpoints, Unsloth dynamic GGUFs, and other reputable weight publishers may package or quantize the same model differently. Origami should describe those differences in provider manifests and tensor maps rather than accumulating provider-specific forks. Every supported artifact must record its source, revision, format, quantization, license, and known quality results.

A provider may supply more than the target weights. Draft models, MTP heads, multimodal projectors, and cache metadata are versioned artifacts with their own tensor layouts and compatibility rules. The manifest should describe the complete serving combination and reject mismatched pieces before inference begins.

Support is earned per model and artifact. Recognizing a model name is not enough; a supported combination must load, pass correctness checks, stay within its memory profile, and have reproducible performance measurements.

## Techniques in scope

Origami can combine several methods instead of expecting one trick to solve every model:

- Dynamic, tensor-aware quantization that preserves sensitive weights at higher precision
- Explicit SSD streaming of routed experts or other conditionally accessed tensors
- Bounded expert caches with measured hit rates and deterministic memory ceilings
- Sparse row access for large embedding, memory, or lookup tables
- Packed on-disk layouts, aligned reads, direct I/O, and asynchronous read lanes
- Deliberate placement across CPU, Metal, unified memory, and storage
- Quantized KV caches and context policies sized for the target machine
- Prefix caching and reusable disk-backed state with versioned cache identities
- Model-specific draft heads, MTP, block diffusion, and other target-verified speculative decoding
- Metal kernels and graph fusions for recurrent state, normalization, quantized matrix operations, and verification batches
- Workload-aware policies that fall back to plain autoregressive decoding when speculation costs more than it saves
- Context-aware draft and verification widths that account for growing attention costs
- Overlap of I/O and compute where the model's dependency graph allows it
- Optional reduced top-k, expert dropping, or pruning as clearly marked lossy modes

Ordinary `mmap` remains useful for resident tensors and reference paths. It is not the main strategy for a model larger than available memory. The operating system page cache has no knowledge of routing, expert reuse, or the runtime's memory budget; Origami should make those decisions explicitly.

## Engineering rules

Memory must be accounted for before optimization. Each integration needs an allocation ledger covering resident weights, streamed weights, caches, model state, KV state, temporary buffers, and safety headroom.

Correctness comes before speed. New execution paths should be compared with a trusted upstream implementation using greedy token parity, logits, or intermediate-state checks. Speculative decoding must remain target-verified. Silent tensor omissions and plausible but incorrect approximations are release blockers.

Optimization decisions must use end-to-end measurements. A smaller quant can have slower kernels. A drafter that wins on code can lose on prose. A wider verification block can become slower as context grows. Origami should measure the serving combination instead of selecting components from isolated bandwidth or compression figures.

Measurements must include enough context to reproduce them: hardware, operating system, model revision, quant, context length, cache state, prompt, and command line. We will report sustained decode, prefill, time to first token, bytes read per token, cache hit rate, I/O wait, compute time, peak resident memory, compression, and swap activity where available.

Lossless and lossy results belong in separate tables. A faster result obtained by routing fewer experts cannot be presented as an optimization of the original model.

A tuned path should become the documented default once it is validated. Requiring a private collection of environment variables makes performance accidental and results difficult to compare. Escape hatches can remain, but the ordinary launch command should select the safest fast path for the detected model, artifact set, and Mac.

## Scope boundaries

Origami focuses on local inference. Training, fine-tuning, datacenter throughput, and broad hardware portability are outside the initial scope. Consumer GPU projects provide useful algorithms and baselines, but Origami's production backend is Metal on Apple Silicon.

Sparse and hybrid models receive priority because selective activation creates an opportunity to trade storage bandwidth for memory capacity. Dense models are in scope only when quantization and placement can fit them within the same hardware budget.

Text-only, single-user inference comes first. Vision, audio, high concurrency, very long context, and distributed execution can follow once the basic memory and correctness contracts are solid.

## Direction

The repository should become a practical toolbox for adapting large open models to small Macs. Each new model will need some custom graph and tensor work, but it should reuse the same storage engine, cache, telemetry, tests, and serving layer. Over time, adding a model should mean describing its conditional weights and implementing its novel operations, not rebuilding the runtime around another checkpoint.

The measure of success is simple: a developer with a 64 GB Mac, and sometimes a 32 GB one, can run a model that would otherwise be dismissed as too large, at a speed and quality level useful for real work.
