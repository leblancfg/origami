#!/usr/bin/env python3
"""Source-derived Qwen3.8-Flash-Next context payload accounting.

This does not load or map model weights. It models the single-sequence Qwen4Exp
path in llama.cpp bea3b12daee45876b0129a3602dc8f534ce30bf0.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


N_QSA_LAYERS = 12
N_GDN_LAYERS = 36
ATTN_K_DIM = 2 * 256
ATTN_V_DIM = 2 * 256
INDEX_K_DIM = 128
# The pinned index cache is a generic KV cache. It allocates this unused V side.
INDEX_V_DIM = 256
CONTEXT_PAD = 256

# ggml block elements and block bytes at the pinned revision.
KV_TYPES = {
    "f32": (1, 4),
    "f16": (1, 2),
    "bf16": (1, 2),
    "q8_0": (32, 34),
    "q4_0": (32, 18),
    "q4_1": (32, 20),
    "iq4_nl": (32, 18),
    "q5_0": (32, 22),
    "q5_1": (32, 24),
}

# llama-hparams.cpp: n_embd_s(), n_embd_r(), and ple_conv_state().
GDN_MATRIX_BYTES = N_GDN_LAYERS * 128 * 6144 * 4
GDN_CONV_BYTES = N_GDN_LAYERS * (4 - 1) * (6144 + 2 * 16 * 128) * 4
# ple_conv_state is included in every uniform recurrent row, although PLE is on one layer.
PLE_CONV_BYTES = N_GDN_LAYERS * (4 - 1) * 3 * 4 * 2560 * 4
GDN_STATE_BYTES = GDN_MATRIX_BYTES + GDN_CONV_BYTES
RECURRENT_BUFFER_BYTES = GDN_STATE_BYTES + PLE_CONV_BYTES

# On the measured arm64 libc++ ABI: llama_pos=4, llama_kv_cell_ext=8,
# shift=4, bitset<LLAMA_MAX_SEQ=256>=32. Main and index caches each allocate
# all four vectors to C elements eagerly. Tree nodes and allocator rounding are excluded.
CELL_ARRAY_BYTES_PER_CACHE_TOKEN = 4 + 8 + 4 + 32
N_CELL_ARRAYS = 2


def pad_context(requested: int) -> int:
    if requested <= 0:
        raise ValueError("context must be positive")
    return ((requested + CONTEXT_PAD - 1) // CONTEXT_PAD) * CONTEXT_PAD


def row_bytes(kind: str, elements: int) -> int:
    block, size = KV_TYPES[kind]
    if elements % block:
        raise ValueError(f"{elements} is not divisible by the {kind} block size {block}")
    return elements // block * size


def main_kv_bytes(cells: int, type_k: str = "f16", type_v: str = "f16") -> int:
    return N_QSA_LAYERS * cells * (row_bytes(type_k, ATTN_K_DIM) + row_bytes(type_v, ATTN_V_DIM))


def index_kv_bytes(cells: int, type_k: str = "f16", type_v: str = "f16") -> int:
    return N_QSA_LAYERS * cells * (row_bytes(type_k, INDEX_K_DIM) + row_bytes(type_v, INDEX_V_DIM))


def eager_cell_array_bytes(cells: int) -> int:
    return N_CELL_ARRAYS * CELL_ARRAY_BYTES_PER_CACHE_TOKEN * cells


def qsa_graph_input_floor(cells: int, ubatch: int) -> int:
    """Logical bytes for QSA's per-layer graph inputs, not the full scheduler buffer.

    For each QSA layer: cell_blk=4C, blk_cells=4C, blk_pos=4C, and
    bias=4*C*U because C is padded to a multiple of the compression ratio 4.
    """
    if ubatch <= 0:
        raise ValueError("ubatch must be positive")
    return N_QSA_LAYERS * (12 * cells + 4 * cells * ubatch)


@dataclass(frozen=True)
class Ledger:
    requested: int
    cells: int
    main_kv: int
    index_kv: int
    gdn_state: int
    ple_state: int
    cell_arrays: int
    qsa_input_floor: int

    @property
    def persistent_payload(self) -> int:
        return self.main_kv + self.index_kv + self.gdn_state + self.ple_state + self.cell_arrays

    @property
    def startup_lower_bound(self) -> int:
        return self.persistent_payload + self.qsa_input_floor


def ledger(requested: int, type_k: str = "f16", type_v: str = "f16", ubatch: int = 32) -> Ledger:
    cells = pad_context(requested)
    return Ledger(
        requested=requested,
        cells=cells,
        main_kv=main_kv_bytes(cells, type_k, type_v),
        index_kv=index_kv_bytes(cells, type_k, type_v),
        gdn_state=GDN_STATE_BYTES,
        ple_state=PLE_CONV_BYTES,
        cell_arrays=eager_cell_array_bytes(cells),
        qsa_input_floor=qsa_graph_input_floor(cells, ubatch),
    )


def yarn_factor(requested: int) -> int | None:
    if requested <= 262_144:
        return None
    if requested <= 524_288:
        return 2
    if requested <= 1_048_576:
        return 4
    raise ValueError("the official static-YaRN guidance covers at most factor 4")


def fmt_bytes(value: int) -> str:
    return f"{value:,} ({value / 2**30:.3f} GiB)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contexts", metavar="N", type=int, nargs="*", default=[250_000, 262_144, 500_000, 1_000_000])
    parser.add_argument("--type-k", choices=KV_TYPES, default="f16")
    parser.add_argument("--type-v", choices=KV_TYPES, default="f16")
    parser.add_argument("--ubatch", type=int, default=32)
    args = parser.parse_args()

    print(f"K={args.type_k}, V={args.type_v}, ubatch={args.ubatch}")
    for requested in args.contexts:
        item = ledger(requested, args.type_k, args.type_v, args.ubatch)
        factor = yarn_factor(requested)
        rope = "native" if factor is None else f"static YaRN factor {factor}"
        print(f"\ncontext {requested:,} -> {item.cells:,} allocated cells; {rope}")
        for label, value in (
            ("QSA main K/V", item.main_kv),
            ("QSA index K/unused-V", item.index_kv),
            ("GDN recurrent", item.gdn_state),
            ("PLE recurrent", item.ple_state),
            ("eager cell-array payload", item.cell_arrays),
            ("persistent payload", item.persistent_payload),
            ("QSA graph-input floor", item.qsa_input_floor),
            ("startup lower bound", item.startup_lower_bound),
        ):
            print(f"  {label:27s} {fmt_bytes(value)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
