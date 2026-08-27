import unittest

from tools.qwen38_context_accounting import (
    GDN_STATE_BYTES,
    PLE_CONV_BYTES,
    RECURRENT_BUFFER_BYTES,
    checkpoint_bytes,
    index_kv_bytes,
    ledger,
    main_kv_bytes,
    pad_context,
    qsa_graph_input_floor,
    yarn_factor,
)


class Qwen38ContextAccountingTests(unittest.TestCase):
    def test_context_is_padded_to_256_cells(self):
        self.assertEqual(pad_context(250_000), 250_112)
        self.assertEqual(pad_context(262_144), 262_144)
        self.assertEqual(pad_context(500_000), 500_224)
        self.assertEqual(pad_context(1_000_000), 1_000_192)

    def test_f16_cache_payload_at_native_context(self):
        cells = 262_144
        self.assertEqual(main_kv_bytes(cells), 6_442_450_944)
        self.assertEqual(index_kv_bytes(cells), 2_415_919_104)

    def test_recurrent_payload_is_context_independent(self):
        self.assertEqual(GDN_STATE_BYTES, 117_669_888)
        self.assertEqual(PLE_CONV_BYTES, 13_271_040)
        self.assertEqual(ledger(250_000).gdn_state, ledger(1_000_000).gdn_state)

    def test_qsa_graph_input_floor_matches_pinned_profile(self):
        self.assertEqual(qsa_graph_input_floor(262_144, 32), 440_401_920)

    def test_quantized_payload_coefficients(self):
        cells = 1_000_192
        q8 = ledger(1_000_000, "q8_0", "q8_0")
        q4 = ledger(1_000_000, "q4_0", "q4_0")
        self.assertEqual(q8.main_kv + q8.index_kv, 17_955_446_784)
        self.assertEqual(q4.main_kv + q4.index_kv, 9_505_824_768)
        self.assertEqual(q8.cells, cells)

    def test_split_f16_indexer_and_key_only_allocator_are_explicit(self):
        split = ledger(
            262_144, "q8_0", "q8_0",
            index_type_k="f16", index_type_v="f16",
        )
        key_only = ledger(
            262_144, "q8_0", "q8_0",
            index_type_k="f16", index_key_only=True,
        )
        self.assertEqual(split.main_kv, 3_422_552_064)
        self.assertEqual(split.index_kv, 2_415_919_104)
        self.assertEqual(split.persistent_payload, 5_994_577_920)
        self.assertEqual(split.startup_lower_bound, 6_434_979_840)
        self.assertEqual(key_only.index_kv, 805_306_368)
        self.assertEqual(key_only.startup_lower_bound, 4_824_367_104)

    def test_checkpoint_copies_are_separate_from_startup_payload(self):
        self.assertEqual(checkpoint_bytes(32), 4_190_109_696)
        item = ledger(262_144, "q8_0", "q8_0", checkpoints=32)
        self.assertEqual(item.checkpoint_copies, 32 * RECURRENT_BUFFER_BYTES)
        self.assertEqual(item.filled_context_lower_bound, item.startup_lower_bound + item.checkpoint_copies)
        with self.assertRaises(ValueError):
            checkpoint_bytes(-1)

    def test_official_static_yarn_bands(self):
        self.assertIsNone(yarn_factor(262_144))
        self.assertEqual(yarn_factor(500_000), 2)
        self.assertEqual(yarn_factor(1_000_000), 4)
        with self.assertRaises(ValueError):
            yarn_factor(1_048_577)


if __name__ == "__main__":
    unittest.main()
