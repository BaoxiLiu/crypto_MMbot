"""Smoke tests for the frozen-15 baseline recomputation (P3).

Anchors run against the certified v3 hour BTCUSDC 2026-09-19/00 (real data,
skipped if absent): the baseline machine must agree with the independently
implemented v3 factor grid to float precision on shared semantics (spread,
1s aggressive flow, signed notional, mid returns, and the 5-second OFI
cumulative relation), plus a synthetic chaining/shape test.
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.factors.compute_v3_baseline15 import (  # noqa: E402
    BaselineMachine, BtcBaselineMachine, run_hour, _prepare)
from app.prediction.builder import infer_tick_size, load_hour  # noqa: E402

CERT = ROOT / "$QUANT_DATA_ROOT/crypto/raw/hft/binance_futures/2026-09-19/00"
V3HOUR = ROOT / "data/prediction-v3/features/BTCUSDC/2026-09-19/00.parquet"


def _prep(symbol, date, hour):
    ob, tr, _mk = load_hour(symbol, date, hour)
    ob["price"] = pd.to_numeric(ob["price"], errors="coerce")
    ob["quantity"] = pd.to_numeric(ob["quantity"], errors="coerce")
    tr["price"] = pd.to_numeric(tr["price"], errors="coerce")
    tr["quantity"] = pd.to_numeric(tr["quantity"], errors="coerce")
    tick = infer_tick_size(ob.price.head(200000), tr.price)
    return ob, tr, tick


@unittest.skipUnless(CERT.exists() and V3HOUR.exists(),
                      "certified hour or v3 features not present")
class TestBaseline15Anchors(unittest.TestCase):
    def test_anchors_vs_v3_grid(self):
        ob, tr, tick = _prep("BTCUSDC", "2026-09-19", "00")
        start = 1_789_776_000_000
        machine = BaselineMachine(tick, start)
        rows = run_hour(machine, _prepare(ob, tr, tick))
        df = pd.DataFrame(rows)
        self.assertEqual(len(df), 3600)
        self.assertEqual(len(df.columns), 11)   # sec + 10 own columns
        v3 = pd.read_parquet(V3HOUR)
        m = df.merge(v3[["sec", "spread_bp", "buy_qty_1s", "sell_qty_1s",
                         "buy_n_1s", "sell_n_1s", "mid", "D4"]],
                     on="sec", suffixes=("_b", ""))
        self.assertLess((m["spread_bp_b"] - m["spread_bp"]).abs().max(), 1e-9)
        self.assertLess((m["buy_qty"] - m["buy_qty_1s"]).abs().max(), 1e-9)
        self.assertLess((m["sell_qty"] - m["sell_qty_1s"]).abs().max(), 1e-9)
        sn = m["buy_n_1s"] - m["sell_n_1s"]
        self.assertLess((m["signed_notional"] - sn).abs().max(), 1e-6)
        pm = m["mid"].shift(1)
        ok = m["mid_return"].notna() & pm.notna()
        self.assertLess((m.loc[ok, "mid_return"]
                         - (m["mid"][ok] / pm[ok] - 1.0)).abs().max(), 1e-12)
        self.assertEqual(int(m["mid_return"].isna().sum()), 1)  # unchained sec 0
        ofi5 = m["ofi"].rolling(5).sum()
        ok5 = m["D4"].notna() & ofi5.notna()
        self.assertLess((m.loc[ok5, "D4"] - ofi5[ok5]).abs().max(), 1e-9)
        self.assertTrue(m["flow_imbalance_1s"].between(-1, 1).all())
        self.assertTrue((m["book_age_ms"] >= 0).all())

    def test_chaining_gives_sec0_windows(self):
        ob, tr, tick = _prep("BTCUSDC", "2026-09-19", "00")
        start = 1_789_776_000_000
        m1 = BaselineMachine(tick, start)
        run_hour(m1, _prepare(ob, tr, tick))
        state = m1.export_state()
        m2 = BaselineMachine(tick, start + 3_600_000, state)
        ob2, tr2, _ = _prep("BTCUSDC", "2026-09-19", "01")
        rows = run_hour(m2, _prepare(ob2, tr2, tick))
        df = pd.DataFrame(rows)
        self.assertFalse(np.isnan(df["mid_return"].iloc[0]))
        self.assertFalse(np.isnan(df["ofi"].iloc[0]))
        self.assertFalse(np.isnan(df["book_age_ms"].iloc[0]))


class TestBaseline15Synthetic(unittest.TestCase):
    def test_shape_and_columns(self):
        m = BaselineMachine(0.1, 0)
        m.apply_snapshot(["bid", "ask"], [1000, 1002], [10.0, 8.0], 500)
        m.apply_trade(1500, 100.1, 2.0, False)
        m.apply_trade(1600, 100.1, 1.0, True)
        rows = m.finalize_upto(3)
        df = pd.DataFrame(rows)
        self.assertEqual(len(df), 4)
        self.assertIn("flow_imbalance_1s", df.columns)
        # t=1500/1600 -> target sec 2 ((t-1)//1000 + 1)
        r1 = df[df["sec"] == 1].iloc[0]
        self.assertEqual(r1["buy_qty"], 0.0)
        self.assertEqual(r1["flow_imbalance_1s"], 0.0)   # no trades -> 0 (P014)
        r2 = df[df["sec"] == 2].iloc[0]
        self.assertEqual(r2["buy_qty"], 2.0)
        self.assertEqual(r2["sell_qty"], 1.0)
        self.assertAlmostEqual(r2["flow_imbalance_1s"], 1.0 / 3.0)

    def test_btc_machine_columns(self):
        m = BtcBaselineMachine(0.1, 0)
        m.apply_snapshot(["bid", "ask"], [1000, 1002], [10.0, 8.0], 500)
        rows = m.finalize_upto(6)
        df = pd.DataFrame(rows)
        for c in ("btc_mid_ret_1s", "btc_mid_ret_5s", "btc_ofi_1s",
                  "btc_ofi_5s", "btc_flow_imb_1s"):
            self.assertIn(c, df.columns)
        self.assertTrue(np.isnan(df["btc_mid_ret_5s"].iloc[4]))  # no 5s history
        self.assertFalse(np.isnan(df["btc_mid_ret_5s"].iloc[5]))


if __name__ == "__main__":
    unittest.main()
