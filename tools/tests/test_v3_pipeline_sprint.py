"""Smoke tests for the sprint pipeline: dataset assembly (labels, NaN
handling, exclusions) and the LightGBM trainer. Run from repo root:
    python3 -m pytest tools/tests/test_v3_pipeline_sprint.py -q
"""
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.models.assemble_v3_dataset import _labels, _labels_vwap, assemble  # noqa: E402

FACTORS = ["A1", "C1", "D6", "E5", "E8"]  # minimal stand-in feature set
BASE_SEC = 1_800_000_000  # 2027-01-15 ~


def make_hour(features_root, symbol, date, hour, mids, capture=0.75,
              status="ok"):
    d = Path(features_root) / symbol / date
    d.mkdir(parents=True, exist_ok=True)
    n = len(mids)
    day_off = int((pd.Timestamp(date) - pd.Timestamp("2027-01-15")).total_seconds())
    sec = np.arange(n, dtype=np.int64) + BASE_SEC + day_off + int(hour) * 3600
    df = pd.DataFrame({
        "sec": sec,
        "mid": np.asarray(mids, dtype=float),
        "complete": [True] * n,
        "capture_rate": np.nan,
        "A1": np.linspace(-1, 1, n),
        "C1": np.where(np.isnan(mids), np.nan, 0.5),
        "D6": np.where(np.isnan(mids), np.nan, 1.0),
        "E5": [np.nan] * n,           # single-coin: all NaN
        "E8": [0.0001] * n,
        # aux per-second columns feeding the class-3/4 targets
        "buy_n_1s": np.where(np.isnan(mids), 0.0, 10.0),
        "sell_n_1s": np.where(np.isnan(mids), 0.0, 5.0),
        "buy_qty_1s": np.where(np.isnan(mids), 0.0, 1.0),
        "sell_qty_1s": np.where(np.isnan(mids), 0.0, 0.5),
        "buy_vwap_1s": np.where(np.isnan(mids), np.nan, 100.1),
        "sell_vwap_1s": np.where(np.isnan(mids), np.nan, 99.9),
    })
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False),
                   d / f"{hour}.parquet")
    meta = {"status": status, "symbol": symbol, "date": date, "hour": hour,
            "gates": {"capture_rate": capture}}
    (d / f"{hour}.meta.json").write_text(json.dumps(meta))


class TestLabels(unittest.TestCase):
    def test_label_literals(self):
        mids = np.array([100.0] * 10)
        mids[1] = 100.1
        mids[5] = 101.0
        t1, t3, t2 = _labels(mids)
        self.assertAlmostEqual(t1[0], (100.1 / 100.0 - 1) * 1e4, places=9)
        self.assertAlmostEqual(t3[0], (101.0 / 100.0 - 1) * 1e4, places=9)
        # tail rows have no forward window
        self.assertTrue(np.isnan(t1[-1]))
        self.assertTrue(np.isnan(t3[-5:]).all())
        self.assertTrue(np.isnan(t2[-30:]).all())

    def test_t2_rv_math(self):
        # constant log-return -> zero forward rv; alternating +/-r -> r
        mids = 100.0 * np.exp(0.001 * np.arange(40))
        _, _, t2 = _labels(mids)
        self.assertAlmostEqual(t2[0], 0.0, delta=1e-6)
        steps = np.tile([0.001, -0.001], 20)
        mids = 100.0 * np.exp(np.concatenate(([0.0], np.cumsum(steps))))
        _, _, t2 = _labels(mids)
        expect = 0.001 * math.sqrt(30.0 / 29.0) * 1e4
        self.assertAlmostEqual(t2[0], expect, places=6)
        self.assertAlmostEqual(t2[5], expect, places=6)

    def test_gap_makes_labels_nan(self):
        mids = np.array([100.0] * 20)
        mids[10] = np.nan
        t1, t3, t2 = _labels(mids)
        self.assertTrue(np.isnan(t1[9]))   # window (9,10] hits the gap
        self.assertTrue(np.isnan(t1[10]))  # window (10,11] starts at the gap
        self.assertFalse(np.isnan(t1[11]))  # (11,12] fine
        self.assertTrue(np.isnan(t3[6]))   # (6,11] hits the gap
        self.assertTrue(np.isnan(t2[0]))   # 30s window hits the gap


class TestAssembler(unittest.TestCase):
    def test_assemble_filters_and_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            sym = "TESTPEPEUSDC"
            # day 1: two hours, second has a mid gap at its start
            mids1 = [100.0] * 3600
            make_hour(tmp, sym, "2027-01-15", "00", mids1, capture=0.7)
            mids2 = [np.nan] + [100.5] * 3599
            make_hour(tmp, sym, "2027-01-15", "01", mids2, capture=0.8)
            # a rejected hour must be ignored
            make_hour(tmp, sym, "2027-01-15", "02",
                      [101.0] * 3600, status="rejected")
            # day 2
            make_hour(tmp, sym, "2027-01-16", "00",
                      100.0 * np.exp(0.0001 * np.arange(3600)), capture=0.9)

            out = Path(tmp) / "assembled" / "test.parquet"
            df, report = assemble(sym, "2027-01-15", "2027-01-16", tmp, out,
                                  features=["A1", "C1", "D6", "E5", "E8"])

            self.assertTrue(out.exists())
            self.assertIn("E5", report["excluded_features"])
            self.assertEqual(report["auto_excluded_features"], [])
            self.assertIn("A1", report["kept_features"])
            # day1 = 7200 rows: gap-crossing union {3570..3600} (31) plus the
            # day tail {7170..7199} (30); day2 = 3600 rows: tail 30
            self.assertEqual(report["rows_after_label_drop"],
                             3600 * 3 - (31 + 30 + 30))
            # rows with NaN feature: none after label filter (C1/D6 NaN only
            # where mid NaN, already label-dropped) -> complete-case loss 0
            self.assertEqual(report["complete_case_drop_rows"], 0)
            # capture_rate carried from the hour meta
            self.assertAlmostEqual(df["capture_rate"].max(), 0.9)
            self.assertAlmostEqual(df["capture_rate"].min(), 0.7)
            # label sanity: flat day -> T1 == 0; exp day (1bp/s steps) -> ~1bp
            d1 = df[df["date"] == "2027-01-15"]
            d2 = df[df["date"] == "2027-01-16"]
            self.assertAlmostEqual(d1["T1"].abs().max(), 0.0, places=6)
            self.assertAlmostEqual(d2["T1"].iloc[0], 1.0, delta=1e-3)
            rep = json.loads(out.with_suffix(".report.json").read_text())
            self.assertEqual(rep["days"]["2027-01-15"]["rejected_hours"], 1)


class TestTrainerSmoke(unittest.TestCase):
    def test_train_runs_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            rng = np.random.default_rng(3)
            n = 400
            dates = ["2026-09-2%d" % d for d in range(10)]
            rows = []
            for i, d in enumerate(dates):
                x = rng.normal(size=n)
                rows.append(pd.DataFrame({
                    "sec": np.arange(n) + i * 100000,
                    "date": d, "mid": 100.0, "complete": True,
                    "capture_rate": 0.8,
                    "A1": x, "C1": rng.normal(size=n), "D6": rng.normal(size=n),
                    "E8": [0.0001] * n,
                    "T1": x * 0.1 + rng.normal(size=n) * 0.01,
                    "T2": np.abs(x) + rng.normal(size=n) * 0.01,
                    "T3": -x * 0.1 + rng.normal(size=n) * 0.01,
                }))
            # test-window days need >30 rows/hour blocks for bootstrap variety
            for i, d in enumerate(["2026-10-01"] * 100 + ["2026-10-02"] * 100):
                x = rng.normal(size=10)
                rows.append(pd.DataFrame({
                    "sec": np.arange(10) + 9000000 + i * 20,
                    "date": d, "mid": 100.0, "complete": True,
                    "capture_rate": 0.8,
                    "A1": np.tile(x, 1), "C1": rng.normal(size=10),
                    "D6": rng.normal(size=10), "E8": [0.0001] * 10,
                    "T1": x * 0.1, "T2": np.abs(x), "T3": -x * 0.1,
                }))
            df = pd.concat(rows, ignore_index=True)
            ds = Path(tmp) / "ds.parquet"
            pq.write_table(pa.Table.from_pandas(df, preserve_index=False), ds)
            out = Path(tmp) / "report.json"
            r = subprocess.run(
                [sys.executable, str(ROOT / "tools/models/train_v3_lgbm.py"),
                 "--dataset", str(ds), "--targets", "T2", "--max-combos", "2",
                 "--out", str(out)],
                capture_output=True, text=True, cwd=ROOT, timeout=600)
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            report = json.loads(out.read_text())
            ev = report["targets"]["T2"]["test_eval"]["all"]
            self.assertGreater(ev["blocks"], 0)
            self.assertIn("importance_top15", report["targets"]["T2"])


if __name__ == "__main__":
    unittest.main()


class TestVwapTargets(unittest.TestCase):
    """Class-3/class-4 volume-weighted targets (V3_SPEC v1.3)."""

    def _day(self, n=40):
        rng = np.random.default_rng(5)
        mids = 100.0 * np.exp(0.0002 * np.arange(n))
        buy_n = np.zeros(n); sell_n = np.zeros(n)
        buy_q = np.zeros(n); sell_q = np.zeros(n)
        buy_p = np.full(n, np.nan); sell_p = np.full(n, np.nan)
        # second k trades: alternate buy/sell, 2 units each
        for k in range(1, n):
            if k % 2 == 1:
                buy_n[k] = 200.0; buy_q[k] = 2.0; buy_p[k] = mids[k] + 0.1
            else:
                sell_n[k] = 200.0; sell_q[k] = 2.0; sell_p[k] = mids[k] - 0.1
        return (mids, buy_n, sell_n, buy_q, sell_q, buy_p, sell_p)

    def test_hand_computed(self):
        mids, bn, sn, bq, sq, bp, sp = self._day()
        out = _labels_vwap(mids, bn, sn, bq, sq, bp, sp)
        t, T = 0, 5
        # window seconds 1..5; vol: buy_n+sell_n = 200 each -> total 1000
        w = [bn[t + k] + sn[t + k] for k in range(1, T + 1)]
        mw = sum(mids[t + k] * w[k - 1] for k in range(1, T + 1)) / sum(w)
        self.assertAlmostEqual(out["R_vwap_5"][t], (mw / mids[t] - 1) * 1e4,
                               places=9)
        qa = sum(bq[t + k] for k in range(1, T + 1))
        pa = sum(bp[t + k] * bq[t + k] for k in range(1, T + 1)
                 if bq[t + k] > 0) / qa   # NaN vwap pairs with zero qty
        self.assertAlmostEqual(out["R_askw_5"][t], (pa / mids[t] - 1) * 1e4,
                               places=9)
        qb = sum(sq[t + k] for k in range(1, T + 1))
        pb = sum(sp[t + k] * sq[t + k] for k in range(1, T + 1)
                 if sq[t + k] > 0) / qb
        self.assertAlmostEqual(out["R_bidw_5"][t], (pb / mids[t] - 1) * 1e4,
                               places=9)
        tw = sum(mids[t + k] for k in range(1, T + 1)) / T
        self.assertAlmostEqual(out["R_twap_5"][t], (tw / mids[t] - 1) * 1e4,
                               places=9)
        # ask side trades at mid+0.1, bid at mid-0.1 -> askw > vwap > bidw
        self.assertGreater(out["R_askw_5"][t], out["R_vwap_5"][t])
        self.assertGreater(out["R_vwap_5"][t], out["R_bidw_5"][t])

    def test_tail_and_zero_volume_nan(self):
        mids, bn, sn, bq, sq, bp, sp = self._day()
        out = _labels_vwap(mids, bn, sn, bq, sq, bp, sp)
        n = len(mids)
        # tail: last T rows undefined for every target
        for c in ("R_vwap_5", "R_vwap_30", "R_askw_5", "R_askw_30",
                  "R_bidw_5", "R_bidw_30", "R_twap_5", "R_twap_30"):
            self.assertTrue(np.isnan(out[c][-5:]).all(), c)
        self.assertTrue(np.isnan(out["R_vwap_30"][-30:]).all())
        self.assertFalse(np.isnan(out["R_vwap_5"][: n - 5]).any())
        # zero-volume window -> NaN
        bn2, sn2, bq2, sq2 = bn.copy(), sn.copy(), bq.copy(), sq.copy()
        bn2[1:6] = 0; sn2[1:6] = 0; bq2[1:6] = 0; sq2[1:6] = 0
        out2 = _labels_vwap(mids, bn2, sn2, bq2, sq2, bp, sp)
        self.assertTrue(np.isnan(out2["R_vwap_5"][0]))
        self.assertFalse(np.isnan(out2["R_twap_5"][0]))  # twap needs no volume
        # zero BUY volume only -> askw NaN, bidw fine
        bq3, bn3 = bq.copy(), bn.copy()
        bq3[1:6] = 0; bn3[1:6] = 0
        out3 = _labels_vwap(mids, bn3, sn3 := sn.copy(), bq3, sq.copy(),
                            bp, sp)
        self.assertTrue(np.isnan(out3["R_askw_5"][0]))
        self.assertFalse(np.isnan(out3["R_bidw_5"][0]))

    def test_gap_window_nan(self):
        mids, bn, sn, bq, sq, bp, sp = self._day()
        mids2 = mids.copy(); mids2[3] = np.nan
        out = _labels_vwap(mids2, bn, sn, bq, sq, bp, sp)
        self.assertTrue(np.isnan(out["R_vwap_5"][0]))   # window (0,5] hits gap
        self.assertFalse(np.isnan(out["R_vwap_5"][5]))  # window (5,10] clean


def test_explicit_baseline_comparison_preserves_paired_statistic(tmp_path):
    from tools.factors.compare_v3_baseline import paired_increment
    for name,values in [('v3.json',{'a':.1,'b':.4,'c':.2}),('base.json',{'a':.0,'b':.2,'c':.3})]:
        (tmp_path/name).write_text(json.dumps({'targets':{'T1':{'test_eval':{'paired':{'rankic_by_hour':values}}}}}))
    expected={'hours':3,'mean_v3':0.2333333333333333,'mean_base':0.16666666666666666,
              'increment':0.0666666666666667,'ci_lo':-0.09999999999999998,'ci_hi':0.16666666666666666}
    actual=paired_increment(tmp_path/'v3.json',tmp_path/'base.json','T1',n_boot=100,seed=11)
    assert actual==expected
