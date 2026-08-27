# Upstream sources

## Model and weights

- [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
- [Qwen technical report](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf)
- [Unsloth dynamic GGUFs](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF), pinned for the proof of concept at revision [`d3bc75ee6ccef3efc1e228ec00a6cc2cdb1e2249`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/d3bc75ee6ccef3efc1e228ec00a6cc2cdb1e2249/UD-IQ1_S)
- [Official FP8 checkpoint](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8)
- [Official Qwen3.8-Flash-Next model card and long-context YaRN recipe](https://huggingface.co/Qwen/Qwen3.8-Flash-Next#processing-ultra-long-texts)

## Reference implementations

- [llama.cpp Qwen4Exp support, PR #27742](https://github.com/ggml-org/llama.cpp/pull/27742), pinned for the proof of concept at commit [`bea3b12daee45876b0129a3602dc8f534ce30bf0`](https://github.com/ggml-org/llama.cpp/commit/bea3b12daee45876b0129a3602dc8f534ce30bf0)
- [Unsloth llama.cpp Qwen4Exp branch, PR #111](https://github.com/unslothai/llama.cpp/pull/111)
- [Transformers Qwen4Exp implementation](https://github.com/huggingface/transformers/pull/48337)
- [vLLM deployment recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next)

## Storage-oriented work

- [BigMoeOnEdge](https://github.com/Helldez/BigMoeOnEdge)
- [BigMoeOnEdge Qwen3.8-Flash-Next draft, PR #172](https://github.com/Helldez/BigMoeOnEdge/pull/172)
- [Atlas Qwen4Exp single-Spark draft, PR #754](https://github.com/Avarok-Cybersecurity/atlas/pull/754)
- [Atlas LongCat NVMe n-gram cache, PR #746](https://github.com/Avarok-Cybersecurity/atlas/pull/746)
- [Earlier llama.cpp n-gram embedding work, PR #19167](https://github.com/ggml-org/llama.cpp/pull/19167)
- [Lucebox Qwen3.8 speculative decoding and GPU optimization, PR #625](https://github.com/Luce-Org/lucebox/pull/625)

## Local baseline

- [DS4](https://github.com/antirez/ds4), for explicit expert streaming, cache accounting, and SSD-oriented scheduling ideas

Claims and measurements copied into this repository should record a source URL, commit or model revision, hardware, and date.
