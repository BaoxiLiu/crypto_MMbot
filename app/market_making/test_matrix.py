import json
import tempfile
import unittest
from pathlib import Path

from .core import Account, D, ZERO
from .core import account_configs
from .test_core import NOW, book, config, rules


def matrix_config():
    return json.loads((Path(__file__).parent/'fixtures/account_matrix.json').read_text())


NAMES = ["fixed_a", "fixed_b", "as_a", "as_b", "inv_a", "inv_b",
         "fv_d100", "fv_t100", "fv_t10", "fvi_d100", "fvi_t100", "fvi_t10",
         "fv_t100_sc", "fv_t10_sc"]


class MatrixConfigTests(unittest.TestCase):
    def test_fourteen_accounts_built_with_expected_overrides(self):
        rows = account_configs(matrix_config())
        self.assertEqual([name for name, _ in rows], NAMES)
        cfgs = dict(rows)
        self.assertEqual(cfgs["fixed_a"]["type"], "fixed")
        self.assertEqual(cfgs["fv_d100"]["fv_source"], "depth")
        self.assertEqual(cfgs["fv_t100"]["fv_source"], "ticker")
        self.assertEqual(cfgs["fv_t100"]["order_latency_ms"], 100)
        self.assertEqual(cfgs["fv_t10"]["order_latency_ms"], 10)
        self.assertEqual(cfgs["fv_t10"]["cancel_latency_ms"], 10)
        self.assertTrue(cfgs["fv_t10"]["fast_trigger"])
        self.assertTrue(cfgs["fvi_t10"]["inventory_control_enabled"])
        self.assertTrue(cfgs["inv_a"]["inventory_control_enabled"])
        self.assertNotIn("fair_value", cfgs["as_a"])
        self.assertTrue(cfgs["fv_t100_sc"]["spread_control_enabled"])
        self.assertEqual(cfgs["fv_t100_sc"]["sc_window_fills"], 50)
        self.assertTrue(cfgs["fv_t10_sc"]["spread_control_enabled"])
        self.assertNotIn("spread_control_enabled", cfgs["fv_t100"])

    def test_matrix_sc_accounts_materialize_new_thresholds(self):
        # Traceability (spec v2.1): the v2.1 sc thresholds must be explicit
        # in every matrix batch config, not just code defaults, so the real
        # values a batch ran with are reconstructible from batch material.
        for filename in ("account_matrix.json",):
            batch = json.loads((Path(__file__).parent/'fixtures'/filename).read_text())
            resolved = dict(account_configs(batch))
            sc_names = [a["name"] for a in batch["accounts"]
                        if a.get("spread_control_enabled")]
            self.assertEqual(len(sc_names), 2, filename)
            for name in sc_names:
                effective = resolved[name]
                self.assertEqual(effective["sc_min_new_fills"], 10, (filename, name))
                self.assertEqual(effective["sc_window_max_age_min"], 480, (filename, name))
                self.assertEqual(effective["sc_restore_stale_min"], 720, (filename, name))

    def test_capital_and_risk_identical_across_accounts(self):
        rows = account_configs(matrix_config())
        for key in ("initial_collateral", "max_position_qty",
                    "max_position_notional", "daily_loss_limit", "maker_fee"):
            self.assertEqual(len({cfg[key] for _, cfg in rows}), 1, key)

    def test_explicit_accounts_reject_bad_input(self):
        base = matrix_config()
        with self.assertRaises(ValueError):
            account_configs({**base, "accounts": {"name": "x"}})
        with self.assertRaises(ValueError):
            account_configs({**base, "accounts": [{"type": "as"}]})
        with self.assertRaises(ValueError):
            account_configs({**base, "accounts": [{"name": "a"}, {"name": "a"}]})
        with self.assertRaises(ValueError):
            account_configs({**base, "accounts": [{"name": "a", "max_position_qty": "100"}]})

    def test_legacy_candidate_path_unchanged(self):
        rows = account_configs({**config(), "as_candidate": {"fair_value": "micro"}})
        self.assertEqual([name for name, _ in rows], ["fixed", "as", "as_candidate"])
