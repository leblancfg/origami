# Qwen3.8 explicit SSD streaming at 262K

## Outcome

Origami runs the pinned `UD-IQ1_S` artifact with a 262,144-cell context and bounded routed-expert reads on the 64 GB M2 Max. Routed tensors are omitted from model buffers. Each Qwen4Exp routing boundary reads the ten selected gate/up/down records into ten fixed Metal cache slots, remaps dedicated routed IDs to those slots, executes the routed matmuls, and retires the lease after Metal completion.

PLE and resident tensors remain mmap-backed. This keeps the 28.8 GB PLE table sparse. Routed tensors have no mmap fallback.

## Identities

| Component | Identity |
|---|---|
| llama.cpp | `213df585b9aed6a09be30d8401f267bf603c104c` |
| model | `unsloth/Qwen3.8-Flash-Next-GGUF` |
| model revision | `d3bc75ee6ccef3efc1e228ec00a6cc2cdb1e2249` |
| quant | `UD-IQ1_S` |
| expert patch | `75584a1370fc16bf97f32d6993e250b5ae674e2af6e7935a058e2f9779868617` before the scheduler-ID parity correction; see the checked-in test for the current digest |
| host | Mac14,6, Apple M2 Max, 68,719,476,736 bytes unified memory |

## Correctness evidence

The first streamed implementation completed but emitted `,` instead of the resident token `The`. Intermediate callbacks traced the first divergence to layer 0. The callback had overwritten computed router IDs, which later scheduler splits did not treat as a stable input.

The corrected graph uses separate Metal-assigned routed-ID inputs. Validation then recorded:

- 30 of 30 selected layer-0 gate/up/down slices equal independent GGUF positional reads;
- 48 of 48 `ffn_moe_topk` tensors equal the resident reference;
- 48 of 48 `ffn_moe_out` tensors equal the resident reference byte for byte;
- generated token `The` equals the resident result;
- swap unchanged during the parity run.

The capability record is [`validation/expert-streaming-capabilities.json`](../../validation/expert-streaming-capabilities.json).

## Native-context allocation

The validated server uses:

```text
context cells                 262,144
main KV                       Q8_0
QSA routing key               F16, key-only
QSA graph inputs              shared across 12 layers
expert cache                  10 slots
physical ubatch               1 token
output projection             CPU
routed weights                explicit pread
PLE                           sparse mmap
context checkpoints           disabled
```

Allocation telemetry:

| Measurement | Result |
|---|---:|
| Declared context | 262,144 |
| Minimum memory-pressure free percentage | 14% |
| Compressor growth | 940,244,992 bytes |
| Swap growth | 0 bytes |
| Swap-out growth | 0 pages |

## Decode evidence

A monitored 32-token request completed at 4.99 tokens/s with no swap growth or swap-outs. A later request stopped naturally after 35 tokens at 5.07 tokens/s. Its minimum memory-pressure free percentage was 16% and compressor occupancy did not grow.

The temporary Pi provider declared `contextWindow: 262144` and returned the exact requested `SSD_262K_OK` response.

## Launch

Build the explicit runtime:

```sh
ORIGAMI_DEPS_ROOT=/private/tmp/origami-expert-runtime \
ORIGAMI_BUILD_RUNTIME=1 \
scripts/bootstrap-expert-streaming-llama-cpp.sh
```

Start the server:

```sh
scripts/start-explicit-ssd-server.sh /path/to/UD-IQ1_S
```

`config/pi-model-origami-262144.json` is a temporary Pi `models.json` example. It does not require editing user dotfiles when copied into a temporary `PI_CODING_AGENT_DIR`.

## Boundary

This is a correctness-first single-sequence path. Prefill is processed as one-token ubatches, PLE still relies on sparse mmap, and the ten-slot cache has no inter-token expert retention. The full 262,144-token prompt has not been filled end to end. Those are optimization and extended-validation tasks, not hidden capabilities.
