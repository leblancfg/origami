import unittest

from tools.qwen38_context_accounting import (
    GDN_STATE_BYTES,
    PLE_CONV_BYTES,
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

    def test_official_static_yarn_bands(self):
        self.assertIsNone(yarn_factor(262_144))
        self.assertEqual(yarn_factor(500_000), 2)
        self.assertEqual(yarn_factor(1_000_000), 4)
        with self.assertRaises(ValueError):
            yarn_factor(1_048_577)


if __name__ == "__main__":
    unittest.main()
