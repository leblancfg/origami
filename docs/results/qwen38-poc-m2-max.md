# Qwen3.8-Flash-Next mmap PoC result

## Outcome

The 64 GB M2 Max emitted tokens with the output projection on CPU. The full-Metal output placement failed during inference with a Metal out-of-memory command buffer and emitted no token. The measured CPU-output placement is therefore the repository default.

This result proves a short mmap reference run, not bounded serving. Darwin still controls file-backed residency, and the process RSS samples do not account for all mapped pages resident in unified memory.

## Identities

| Component | Measured identity |
|---|---|
| Host | Mac14,6, Apple M2 Max, 68,719,476,736 bytes unified memory, 12 CPUs, 16 KiB pages |
| OS | macOS 26.5.2, build 25F84, kernel 25.5.0 |
| Storage | internal APFS SSD on Apple Fabric, `/dev/disk3s5` |
| llama.cpp source | `bea3b12daee45876b0129a3602dc8f534ce30bf0` |
| Origami patch SHA-256 | `3c0bcb3ea48d6313c9ec1cff0ab748b79e3891e1c45fd197e76b211ba9787e55` |
| Passing `llama-cli` SHA-256 | `c2ac5ab75b09982833263f922acbc7fd74268ac77a6b53cd60d5c5a206ccea8e` |
| Model | `unsloth/Qwen3.8-Flash-Next-GGUF` at `d3bc75ee6ccef3efc1e228ec00a6cc2cdb1e2249`, `UD-IQ1_S` |
| Shard bytes | 10,946,624; 49,990,818,368; 22,544,696,352 (72,546,461,344 total) |
| Shard SHA-256 | `88a1420825a9304063e882ada29d438263617f51ac8923d438d927496693bafd`; `3a62e35bbf9add4733bd1438ebd3a67649d5edd6cb0e72bb78e33c913992b2b6`; `0e25ceaeb89b8a80aa973c6c0c7448943682f7408c2855b2ebd016b7643a861a` |
| Repository base during measurement | `1857ee93f127e1c38a9cc4fec56c82d9d9038f24` plus the changes described here |

The build source remained detached at the pinned llama.cpp revision. Its only changes were the checked-in patch to `src/llama-model.cpp` and `ggml/src/ggml-metal/ggml-metal-device.m`.

## Commands

The passing first-token command was:

```text
llama-cli --offline --load-mode mmap --gpu-layers all --ctx-size 512 \
  --batch-size 32 --ubatch-size 32 --cache-ram 0 --no-warmup \
  --color off --simple-io --single-turn --log-verbosity 4 --perf \
  --override-tensor '^output=CPU' \
  --model Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf \
  --prompt Hello --n-predict 1 --seed 424242 --temp 0 --no-display-prompt
```

The harness supplied:

```text
GGML_METAL_NO_RESIDENCY=1
LLAMA_MMAP_PREFETCH=0
```

The 16-token run changed only the prompt and prediction count. It used `Reply with only the next four integers: 2 4 6 8` and `--n-predict 16`.

## Full-Metal failure

The orchestrated run without the output override reached the compute graph, then stderr reported `kIOGPUCommandBufferCallbackErrorOutOfMemory`, scheduler failure, and `llama_decode: failed to decode, ret = -3`. The CLI returned zero even though stdout ended with `Error: Compute error.`. The harness now rejects these markers independently of process status.

| Measurement | Full-Metal output |
|---|---:|
| Token emitted | no |
| Wall time | 34.502999 s |
| Minimum memory-pressure free percentage | 7% |
| Peak compressor occupied | 13,572,554,752 bytes |
| Compressor occupied delta | +2,808,954,880 bytes |
| Swap used | 2,748,191,867 to 9,469,364,797 bytes |
| Swap delta | +6,721,172,930 bytes |
| Swap-out delta | 423,540 pages (6,939,279,360 bytes) |
| Page-in delta | 4,853,642 pages (79,521,251,328 bytes) |

This failure is concrete evidence that the 46.6 GiB Metal mmap envelope is unsafe on this host. The CLI's zero exit status was not evidence of success.

## Certified CPU-output runs

The rebuilt backend emitted both safeguard markers:

```text
load_tensors: mmap prefetch disabled by LLAMA_MMAP_PREFETCH=0
ggml_metal_device_init: use residency sets    = false (GGML_METAL_NO_RESIDENCY=1)
```

The fixed first-token profile passed with generated text `The` and generated-output SHA-256 `b344d80e24a3679999fa964450b34bc24d1578a35509f934c1418b0a20d21a67`.

| Measurement | First token | 16-token validation |
|---|---:|---:|
| Generated output | `The` | `The user wants me to reply with only the next four integers after "2 ` |
| Generated-output SHA-256 | `b344d80e24a3679999fa964450b34bc24d1578a35509f934c1418b0a20d21a67` | `3b358acdfe359f0da7c59474860cd1acd20f83cc0f53fb83ddba43b0b748cf95` |
| Wall time | 5.778105 s | 4.319915 s |
| llama.cpp prefill | 53 tokens, 915.33 ms, 57.90 t/s | 68 tokens, 963.12 ms, 70.60 t/s |
| llama.cpp decode | 1 token reported | 16 tokens, 699.98 ms, 21.43 t/s |
| Minimum memory-pressure free percentage | 20% | 18% |
| Peak process-tree RSS | 839,057,408 bytes | 885,817,344 bytes |
| Peak compressor occupied | 4,908,072,960 bytes | 5,963,710,464 bytes |
| Compressor occupied delta | +170,934,272 bytes | +43,663,360 bytes |
| Swap used delta | 0 bytes | -8,388,608 bytes |
| Swap-out delta | 0 pages | 0 pages |
| Page-in delta | 392,489 pages (6,430,539,776 bytes) | 17,343 pages (284,147,712 bytes) |

The one-token llama.cpp report assigns the sampled token to prompt evaluation and reports zero decode milliseconds. It still reports one decoded token, and the extracted output contains `The`.

The 16-token generated-output hash matched the immediately preceding run. This establishes repeatability for the same patched backend and warm filesystem state. It does not establish parity with an independent resident implementation. The 16-token limit also ends during the model's reasoning, before it produces the requested four-number answer.

The successful measurements began with about 9.1 GB of system swap already allocated from the earlier failed run. Neither successful run increased swap or caused swap-outs. The negative validation delta means macOS reclaimed 8 MiB during the run; it is not counted as an inference improvement.

## Test results

The repository's 26 Python tests passed, including all macOS harness integrations. The filtered Metal `MUL_MAT_ID` suite passed 79 of 79 cases for IQ1_S, IQ2_XXS, and IQ4_NL. Shell syntax checks, Python bytecode compilation, patch reverse-application, and pinned-source revision checks also passed.

The pinned backend's broad `test-llama-archs` executable aborts in the experimental Meta backend at `ggml-backend-meta.cpp:756` while testing Qwen4Exp. With seed 424242, Qwen4Exp passes on Metal, Accelerate, and CPU before `split_states_equal(src_ss[0], src_ss[2])` asserts on Meta. `tests/test-llama-archs.cpp` in the pinned source says these tests are disabled in macOS arm64 CI because they segfault. No test was skipped or weakened in Origami, and the failing Meta path is not used by the PoC.

## Remaining boundary

The PoC is working for one token and a short deterministic sequence. It does not satisfy Origami's sustained-generation support standard: mmap has no deterministic cache ceiling, the OS controls eviction, and no independent correctness oracle has been recorded. The native runtime still needs explicit expert reads, bounded caches, long-run memory telemetry, and parity checks. The upstream Meta-backend assertion remains a non-PoC test failure on this host.
