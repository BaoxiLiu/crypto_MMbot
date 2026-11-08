"""Kimi v3 P1 factor-library tests: hand-computed scenarios, causal guard,
NaN semantics, pipeline gates, and the P0.5 probe reconciliation anchors.

Run from the repo root:
    python3 -m pytest tools/tests/test_prediction_v3_factors.py -q
"""
import json
import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.prediction import replay  # noqa: E402
from app.prediction.builder import (  # noqa: E402
    ValidationError, build_hour, build_hour_multi, run_gates, trade_capture_rate)
from app.prediction.thresholds import load_thresholds  # noqa: E402
from app.prediction.factors import FactorMachine  # noqa: E402
from app.prediction.schema import FACTOR_NAMES, FactorParams  # noqa: E402

TICK = 0.1
NAN = float("nan")


def make_machine(params=None, hour_start=0, symbol="TESTUSDC"):
    return FactorMachine(symbol, TICK, hour_start, params)


SNAP_BIDS = {100.0: 10.0, 99.9: 5.0, 99.8: 4.0, 99.7: 3.0, 99.6: 2.0,
             99.5: 1.0, 99.4: 1.0, 99.3: 1.0, 99.2: 1.0, 99.1: 1.0}
SNAP_ASKS = {100.2: 8.0, 100.3: 6.0, 100.4: 4.0, 100.5: 3.0, 100.6: 2.0,
             100.7: 1.0, 100.8: 1.0, 100.9: 1.0, 101.0: 1.0, 101.1: 1.0}


def snap(m):
    sides = ["bid"] * len(SNAP_BIDS) + ["ask"] * len(SNAP_ASKS)
    idxs = [int(round(p / TICK)) for p in list(SNAP_BIDS) + list(SNAP_ASKS)]
    qtys = list(SNAP_BIDS.values()) + list(SNAP_ASKS.values())
    m.apply_snapshot(sides, idxs, qtys)


def upd(m, sec, side, price, qty):
    m.apply_update(sec * 1000, side, int(round(price / TICK)), qty, price)


def trade(m, sec, price, qty, buyer_maker):
    m.apply_trade(sec * 1000, price, qty, buyer_maker)


def build_s1():
    """Main hand-computed scenario (secs 0..400).

    Book: see SNAP_BIDS/ASKS; mid 100.1, spread 0.2.
    sec 1: cancel 0.5 @99.5 (49.75, far)   sec 2: delete ask 100.7 (100.7, far)
    sec 3: add 2 @100.0 (200)              sec 4: cancel 2 @100.2 (200.4, near)
    sec 5: add 3 @99.9 (299.7)
    sec 10: delete bid 100.0 (cancel 1200, near), delete ask 100.2 (cancel
            601.2, near), add bid 100.2:8 (801.6), add ask 100.4:10 (+6 notional
            on an existing level). New mid 100.25, spread 0.1.
    Trades at every mark k=1..40: odd k buyer-initiated 100*k, even k
    seller-initiated 100*k (price 100.0).
    """
    p = FactorParams(cancel_p90_notional=100.0, median_cancel_qty=10.0,
                     trade_p90_notional=3500.0)
    m = make_machine(params=p)
    snap(m)
    m.apply_mark(1000, 50000.0, 0.0001, 7200000)
    for j in range(0, 401):
        m.feed_btc_second(j, 200.0 if j < 50 else 202.0,
                          100.0 if j % 2 == 0 else 0.0,
                          0.0 if j % 2 == 0 else 50.0)
        m.feed_market_mid(j, 50.0 if j < 50 else 51.0)
    for k in range(1, 41):
        if k == 1:
            upd(m, 1, "bid", 99.5, 0.5)
        elif k == 2:
            upd(m, 2, "ask", 100.7, 0.0)
        elif k == 3:
            upd(m, 3, "bid", 100.0, 12.0)
        elif k == 4:
            upd(m, 4, "ask", 100.2, 6.0)
        elif k == 5:
            upd(m, 5, "bid", 99.9, 8.0)
        elif k == 10:
            upd(m, 10, "bid", 100.0, 0.0)
            upd(m, 10, "ask", 100.2, 0.0)
            upd(m, 10, "bid", 100.2, 8.0)
            upd(m, 10, "ask", 100.4, 10.0)
        if k % 2 == 1:
            trade(m, k, 100.0, float(k), False)
        else:
            trade(m, k, 100.0, float(k), True)
    m.finalize_upto(400)
    return m


class TestShapeFactors(unittest.TestCase):
    def setUp(self):
        self.m = build_s1()

    def b(self, sec):
        return self.m.blocks[sec]

    def test_snapshot_shape(self):
        b = self.b(0)
        self.assertAlmostEqual(b["A1"], 1.0 / 43.0, places=12)   # 22 vs 21
        self.assertAlmostEqual(b["A2"], 1.0 / 57.0, places=12)   # 29 vs 28
        self.assertTrue(math.isnan(b["A3"]))                     # spread 0.2 > 10bp
        self.assertAlmostEqual(b["A4"], 1.0 / 9.0, places=12)    # 10 vs 8 within 10bp
        self.assertEqual(b["A5"], 0.0)                           # symmetric slopes
        self.assertAlmostEqual(b["mid"], 100.1, places=9)
        self.assertAlmostEqual(b["micro_off"], 1.0 / 18.0, places=12)

    def test_shape_after_step(self):
        b = self.b(10)
        # bids <=3t of 100.2: 8+8; asks <=3t of 100.3: 6+10+3+2
        self.assertAlmostEqual(b["A1"], -5.0 / 37.0, places=12)
        # A1(5): bids 12+8+4+3=27, asks 6+6+4+3=19
        self.assertAlmostEqual(self.b(5)["A1"], 4.0 / 23.0, places=12)
        self.assertAlmostEqual(b["A6"], -5.0 / 37.0 - 4.0 / 23.0, places=12)
        self.assertEqual(self.b(15)["A6"], 0.0)
        self.assertAlmostEqual(b["A3"], 1.0 / 7.0, places=12)     # 8 vs 6 within 5bp
        self.assertAlmostEqual(b["A4"], 1.0 / 7.0, places=12)
        # slopes: bid (28.5-16)/7, ask (25-21)/7 -> (12.5-4)/(12.5+4) = 17/33
        self.assertAlmostEqual(b["A5"], 17.0 / 33.0, places=12)
        self.assertAlmostEqual(b["spread_bp"], 0.1 / 100.25 * 1e4, places=9)

    def test_book_update_rate(self):
        self.assertAlmostEqual(self.b(10)["A7"], 0.9)   # 9 rows in (0,10]
        self.assertAlmostEqual(self.b(15)["A7"], 0.4)   # 4 rows in (5,15]
        self.assertEqual(self.b(20)["A7"], 0.0)

    def test_A8_C5_rolling_hour(self):
        m = make_machine()
        m.apply_snapshot(["bid", "ask"], [1000, 1002], [10.0, 8.0])
        for j in range(1, 3701):
            if j % 2 == 1:
                m.apply_update(j * 1000, "ask", 1002, 0.0, 100.2)
                m.apply_update(j * 1000, "ask", 1006, 8.0, 100.6)
                m.apply_trade(j * 1000, 100.0, 3.0, False)
            else:
                m.apply_update(j * 1000, "ask", 1006, 0.0, 100.6)
                m.apply_update(j * 1000, "ask", 1002, 8.0, 100.2)
                m.apply_trade(j * 1000, 100.0, 1.0, False)
        m.finalize_upto(3700)
        sp_e = 0.2 / 100.1 * 1e4
        sp_o = 0.6 / 100.3 * 1e4
        self.assertAlmostEqual(m.blocks[3700]["A8"],
                               sp_e - (sp_e + sp_o) / 2.0, places=9)
        self.assertAlmostEqual(m.blocks[3700]["C5"], 0.5, places=12)
        self.assertTrue(math.isnan(m.blocks[3599]["A8"]))
        self.assertTrue(math.isnan(m.blocks[3599]["C5"]))


class TestCancelFactors(unittest.TestCase):
    def setUp(self):
        self.m = build_s1()

    def b(self, sec):
        return self.m.blocks[sec]

    def test_cancel_intensity(self):
        # window (0,30]: bid cancels 49.75 + 1200; ask cancels 100.7+200.4+601.2
        self.assertAlmostEqual(self.b(30)["B1"], 1249.75 / 30.0, places=9)
        self.assertAlmostEqual(self.b(30)["B2"], 902.3 / 30.0, places=9)
        self.assertAlmostEqual(self.b(31)["B1"], 40.0, places=9)   # secs 2..31

    def test_cancel_add_ratio(self):
        # cancels 2152.05; adds 200+299.7+801.6+602.4 = 1903.7
        self.assertAlmostEqual(self.b(30)["B3"], 2152.05 / 1903.7, places=9)

    def test_near_touch_share(self):
        # window (1,61] secs 2..61, distance at each second's grid state:
        # only the sec-4 cancel @100.2 is <=1 tick of the mark-4 state
        self.assertAlmostEqual(self.b(61)["B4"], 200.4 / 2102.3, places=9)
        self.assertTrue(math.isnan(self.b(30)["B4"]))

    def test_cancel_imbalance(self):
        self.assertAlmostEqual(self.b(30)["B5"],
                               (1249.75 - 902.3) / 2152.05, places=9)
        self.assertTrue(math.isnan(self.b(30 + 31)["B5"]))  # no cancels -> NaN

    def test_large_cancel(self):
        # threshold notional>100 and qty>1.0: only 200.4 (sec4), 1200, 601.2
        self.assertAlmostEqual(self.b(61)["B6"], 2001.6 / 60.0, places=9)
        m = build_s1_with_params(None)
        self.assertTrue(math.isnan(m.blocks[61]["B6"]))


def build_s1_with_params(params):
    m = make_machine(params=params)
    snap(m)
    m.apply_mark(1000, 50000.0, 0.0001, 7200000)
    for k in range(1, 41):
        if k == 1:
            upd(m, 1, "bid", 99.5, 0.5)
        elif k == 2:
            upd(m, 2, "ask", 100.7, 0.0)
        elif k == 4:
            upd(m, 4, "ask", 100.2, 6.0)
        elif k == 10:
            upd(m, 10, "bid", 100.0, 0.0)
            upd(m, 10, "ask", 100.2, 0.0)
            upd(m, 10, "bid", 100.2, 8.0)
            upd(m, 10, "ask", 100.4, 10.0)
        if k % 2 == 1:
            trade(m, k, 100.0, float(k), False)
        else:
            trade(m, k, 100.0, float(k), True)
    m.finalize_upto(61)
    return m


class TestTakerFactors(unittest.TestCase):
    def setUp(self):
        self.m = build_s1()

    def b(self, sec):
        return self.m.blocks[sec]

    def test_taker_imb(self):
        self.assertEqual(self.b(2)["C1"], -1.0)
        self.assertEqual(self.b(3)["C1"], 1.0)
        self.assertTrue(math.isnan(self.b(0)["C1"]))
        self.assertAlmostEqual(self.b(6)["C2"], -0.2, places=12)   # secs 2..6
        self.assertAlmostEqual(self.b(15)["C2"], 0.2, places=12)
        self.assertAlmostEqual(self.b(40)["C3"], -1.0 / 51.0, places=12)

    def test_trade_intensity_and_size(self):
        self.assertEqual(self.b(19)["C4"], 1.0)
        self.assertEqual(self.b(61)["C4"], 0.0)
        self.assertEqual(self.b(30)["C6"], 0.0)  # no trade > 3500 in secs 1..30
        self.assertAlmostEqual(self.b(40)["C6"], 19000.0 / 76500.0, places=9)

    def test_persistence_accel_vpin(self):
        self.assertEqual(self.b(40)["C7"], 0.5)
        self.assertAlmostEqual(self.b(40)["C8"], -0.4, places=12)
        self.assertAlmostEqual(self.b(31)["C8"], 0.4, places=12)
        self.assertEqual(self.b(40)["C9"], 1.0)
        # VPIN(30): every second has exactly one trade -> sum|b-s| == sum(b+s)
        self.assertEqual(self.b(30)["C9"], 1.0)
        self.assertTrue(math.isnan(self.b(29)["C9"]))
        self.assertTrue(math.isnan(self.b(10)["C9"]))


class TestKyleLambda(unittest.TestCase):
    def test_kyle_slope(self):
        m = make_machine()
        m.apply_snapshot(["bid", "ask"], [1000, 1002], [10.0, 8.0])
        for j in range(1, 41):
            if j >= 11:
                m.apply_update(j * 1000, "bid", 1000 + (j - 11), 0.0,
                               100.0 + 0.1 * (j - 11))
                m.apply_update(j * 1000, "ask", 1002 + (j - 11), 0.0,
                               100.2 + 0.1 * (j - 11))
                m.apply_update(j * 1000, "bid", 1000 + (j - 10), 10.0,
                               100.0 + 0.1 * (j - 10))
                m.apply_update(j * 1000, "ask", 1002 + (j - 10), 8.0,
                               100.2 + 0.1 * (j - 10))
            m.apply_trade(j * 1000, 100.0, float(j), False)
        m.finalize_upto(70)
        xs = [100.0 * j if 11 <= j <= 40 else 0.0 for j in range(11, 71)]
        ys = [0.1 if 11 <= j <= 40 else 0.0 for j in range(11, 71)]
        n = 60
        xbar = sum(xs) / n
        ybar = sum(ys) / n
        sxy = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
        sxx = sum((x - xbar) ** 2 for x in xs)
        self.assertAlmostEqual(m.blocks[70]["C10"], sxy / sxx, places=15)
        self.assertAlmostEqual(m.blocks[40]["mid"], 103.1, places=9)

    def test_kyle_degenerate_nan(self):
        m = make_machine()
        m.apply_snapshot(["bid", "ask"], [1000, 1002], [10.0, 8.0])
        for j in range(1, 41):
            m.apply_trade(j * 1000, 100.0, 1.0, False)
        m.finalize_upto(40)
        self.assertTrue(math.isnan(m.blocks[40]["C10"]))


def build_s3():
    """Mid staircase for D-group factors. Mids:
    secs 0-9: 100.1 | 10-99: 100.3 | 100-199: 100.5 | 200-384: 100.7 |
    385-399: 101.9 | 400-419: 102.1 | 420-429: 102.3 | 430+: 103.3.
    """
    m = make_machine()
    snap(m)

    def step(sec, del_bid, del_ask, add_bid, add_ask, extra_del=()):
        for p in extra_del:
            m.apply_update(sec * 1000, "ask", int(round(p / TICK)), 0.0, p)
        m.apply_update(sec * 1000, "bid", int(round(del_bid / TICK)), 0.0, del_bid)
        m.apply_update(sec * 1000, "ask", int(round(del_ask / TICK)), 0.0, del_ask)
        m.apply_update(sec * 1000, "bid", int(round(add_bid / TICK)), 8.0, add_bid)
        m.apply_update(sec * 1000, "ask", int(round(add_ask / TICK)), 10.0, add_ask)

    step(10, 100.0, 100.2, 100.2, 100.4, extra_del=(100.3,))
    step(100, 100.2, 100.4, 100.4, 100.6, extra_del=(100.5,))
    step(200, 100.4, 100.6, 100.6, 100.8, extra_del=(100.7,))
    for p in (100.8, 100.9, 101.0, 101.1):
        m.apply_update(385 * 1000, "ask", int(round(p / TICK)), 0.0, p)
    m.apply_update(385 * 1000, "bid", 1006, 0.0, 100.6)
    m.apply_update(385 * 1000, "bid", int(round(101.8 / TICK)), 8.0, 101.8)
    m.apply_update(385 * 1000, "ask", int(round(102.0 / TICK)), 10.0, 102.0)
    step(400, 101.8, 102.0, 102.0, 102.2)
    step(420, 102.0, 102.2, 102.2, 102.4)
    step(430, 102.2, 102.4, 103.2, 103.4)
    m.finalize_upto(450)
    return m


S3_MIDS = {0: 100.1, 10: 100.3, 100: 100.5, 200: 100.7, 385: 101.9,
           400: 102.1, 420: 102.3, 430: 103.3}


def s3_mid(j):
    keys = sorted(k for k in S3_MIDS if k <= j)
    return S3_MIDS[keys[-1]]


def s3_rets():
    return {j: math.log(s3_mid(j) / s3_mid(j - 1))
            for j in range(1, 451) if s3_mid(j) != s3_mid(j - 1)}


def win_rv(rets, s, lag):
    r = [rets.get(j, 0.0) for j in range(s - lag + 1, s + 1)]
    n = len(r)
    mean = sum(r) / n
    return math.sqrt(sum((x - mean) ** 2 for x in r) / (n - 1)) * 1e4


class TestTimeVolFactors(unittest.TestCase):
    def setUp(self):
        self.m = build_s3()
        self.rets = s3_rets()

    def b(self, sec):
        return self.m.blocks[sec]

    def test_mid_returns(self):
        self.assertAlmostEqual(self.b(10)["D1"],
                               (100.3 / 100.1 - 1.0) * 1e4, places=9)
        self.assertAlmostEqual(self.b(110)["D2"],
                               (100.5 / 100.3 - 1.0) * 1e4, places=9)
        self.assertAlmostEqual(self.b(120)["D3"],
                               (100.5 / 100.3 - 1.0) * 1e4, places=9)
        self.assertEqual(self.b(5)["D1"], 0.0)
        self.assertTrue(math.isnan(self.b(4)["D1"]))

    def test_rv_windows(self):
        r = self.rets
        c = r[200]
        self.assertAlmostEqual(self.b(220)["D6"], c / math.sqrt(30) * 1e4, places=9)
        self.assertAlmostEqual(self.b(220)["D7"], c / math.sqrt(60) * 1e4, places=9)
        self.assertAlmostEqual(self.b(310)["D8"], win_rv(self.rets, 310, 300), places=9)
        self.assertTrue(math.isnan(self.b(220)["D8"]))
        self.assertEqual(self.b(310)["D6"], 0.0)
        d6, d8 = self.b(400)["D6"], self.b(400)["D8"]
        self.assertAlmostEqual(self.b(400)["D9"], d6 / (d8 + 1e-12), places=12)

    def test_semivariance(self):
        e, f, g = (self.rets[400], self.rets[420], self.rets[430])
        up = math.sqrt(e * e + f * f + g * g) * 1e4
        self.assertAlmostEqual(self.b(450)["D10"], up, places=9)
        self.assertEqual(self.b(450)["D11"], 0.0)
        self.assertAlmostEqual(self.b(450)["D12"], 1.0, places=9)
        self.assertAlmostEqual(self.b(450)["D7"], win_rv(self.rets, 450, 60), places=9)
        self.assertAlmostEqual(self.b(450)["D8"], win_rv(self.rets, 450, 300), places=7)

    def test_microprice_mom(self):
        m1 = build_s1()
        # micro_off is 1/18 before the sec-10 step and 1/14 after (constant)
        self.assertAlmostEqual(m1.blocks[0]["micro_off"], 1.0 / 18.0, places=9)
        self.assertAlmostEqual(m1.blocks[20]["micro_off"], 1.0 / 14.0, places=9)
        self.assertEqual(m1.blocks[20]["D13"], 0.0)
        self.assertAlmostEqual(m1.blocks[15]["D13"], 1.0 / 14.0 - 1.0 / 6.0,
                               places=9)

    def test_range_and_jumps(self):
        self.assertAlmostEqual(self.b(450)["D14"], 1.0, places=9)
        self.assertEqual(self.b(450)["D15"], 1.0)
        self.assertEqual(self.b(61)["D15"], 0.0)

    def test_ofi(self):
        m = build_s1()
        # touch OFI: sec3 +2 (bid qty up), sec4 +2 (ask qty down), sec10:
        # -12 (bid delete) +6 (ask touch up) +8 (bid touch up); ask 100.4 was
        # never the touch -> no event
        self.assertEqual(m.blocks[5]["D4"], 4.0)
        self.assertEqual(m.blocks[10]["D4"], 2.0)
        self.assertEqual(m.blocks[35]["D5"], 2.0)
        self.assertEqual(m.blocks[40]["D5"], 0.0)


class TestEnvFactors(unittest.TestCase):
    def setUp(self):
        self.m = build_s1()

    def b(self, sec):
        return self.m.blocks[sec]

    def test_btc_factors_self_feed(self):
        # S1 is a non-BTC symbol: E1-E7 come from the explicit external feeds
        self.assertAlmostEqual(self.b(54)["E1"], 100.0, places=9)   # 202/200-1
        self.assertEqual(self.b(15)["E1"], 0.0)
        self.assertAlmostEqual(self.b(65)["E2"], 100.0, places=9)   # 202/200-1
        self.assertAlmostEqual(self.b(80)["E3"],
                               math.log(1.01) / math.sqrt(60) * 1e4, places=9)
        self.assertAlmostEqual(self.b(40)["E4"], 1.0 / 3.0, places=12)
        self.assertAlmostEqual(self.b(54)["E5"], 200.0, places=9)   # 51/50-1
        self.assertAlmostEqual(self.b(65)["E6"], 200.0, places=9)
        self.assertAlmostEqual(self.b(80)["E7"],
                               math.log(1.02) / math.sqrt(60) * 1e4, places=9)
        self.assertTrue(math.isnan(self.b(40)["E3"]))   # window ( -20,40] incomplete

    def test_btc_symbol_self_feed(self):
        p = FactorParams()
        m = make_machine(params=p, symbol="BTCUSDC")
        m.apply_snapshot(["bid", "ask"], [1000, 1002], [10.0, 8.0])
        for j in range(1, 41):
            m.apply_trade(j * 1000, 100.0, 2.0, False)
        m.finalize_upto(40)
        b = m.blocks[40]
        self.assertAlmostEqual(b["mid"], 100.1, places=9)
        self.assertAlmostEqual(b["E4"], 1.0, places=12)  # own taker imb
        self.assertTrue(math.isnan(b["E5"]))             # no market feed

    def test_funding_and_clock(self):
        self.assertAlmostEqual(self.b(40)["E8"], 0.0001, places=12)
        self.assertTrue(math.isnan(self.b(0)["E8"]))
        self.assertAlmostEqual(self.b(40)["E9"],
                               (7200000 - 40000) / 60000.0, places=9)
        m = make_machine(hour_start=6 * 3600 * 1000)
        m.apply_snapshot(["bid", "ask"], [1000, 1002], [10.0, 8.0])
        m.finalize_upto(0)
        self.assertAlmostEqual(m.blocks[0]["E10"], 1.0, places=12)
        self.assertAlmostEqual(m.blocks[0]["E11"], 0.0, places=12)


class TestNanSemantics(unittest.TestCase):
    def test_fresh_machine_nan_semantics(self):
        m = make_machine()
        m.apply_snapshot(["bid", "ask"], [1000, 1002], [10.0, 8.0])
        m.finalize_upto(0)
        b = m.blocks[0]
        instant = ("A1", "A2", "A4", "A5", "E10", "E11")
        for name in FACTOR_NAMES:
            if name in instant:
                self.assertFalse(math.isnan(b[name]), name)
            else:
                self.assertTrue(math.isnan(b[name]), name)
        # spread 0.2 = ~20bp: no depth inside +/-5bp of mid
        self.assertTrue(math.isnan(b["A3"]))

    def test_window_coverage(self):
        m = make_machine()
        m.apply_snapshot(["bid", "ask"], [1000, 1002], [10.0, 8.0])
        for j in range(1, 101):
            m.apply_trade(j * 1000, 100.0, 1.0, False)
        m.finalize_upto(300)
        self.assertFalse(math.isnan(m.blocks[100]["C3"]))  # 30s covered at 100
        self.assertFalse(math.isnan(m.blocks[5]["C2"]))    # 5s covered at 5
        self.assertTrue(math.isnan(m.blocks[4]["C2"]))     # 5s not covered at 4
        self.assertTrue(math.isnan(m.blocks[100]["D8"]))   # 300s not covered
        self.assertTrue(math.isnan(m.blocks[200]["D8"]))
        self.assertTrue(math.isnan(m.blocks[29]["B1"]))


class TestCausalGuard(unittest.TestCase):
    def test_past_values_unchanged_by_future_events(self):
        m = build_s1()
        marks = list(range(5000, 40001, 1000))
        before = [dict(m.query(t)) for t in marks]

        # inject a burst of future events: cancels, adds, trades, marks
        m.apply_mark(40500, 51000.0, 0.0002, 9000000)
        for sec in range(41, 81):
            if sec % 3 == 0:
                upd(m, sec, "bid", 99.9, 1.0)
            if sec % 4 == 0:
                upd(m, sec, "ask", 100.5, 9.0)
            trade(m, sec, 100.2, 5.0, sec % 2 == 0)
        m.finalize_upto(80)

        for t, snap_before in zip(marks, before):
            after = m.query(t)
            for name in FACTOR_NAMES:
                v0, v1 = snap_before[name], after[name]
                if math.isnan(v0) and math.isnan(v1):
                    continue
                self.assertEqual(v0, v1, f"{name} at t={t} changed: {v0} -> {v1}")

    def test_out_of_order_feed_rejected(self):
        m = make_machine()
        m.apply_snapshot(["bid", "ask"], [1000, 1002], [10.0, 8.0])
        m.apply_update(2000, "bid", 1000, 9.0, 100.0)
        with self.assertRaises(ValueError):
            m.apply_update(1000, "bid", 999, 1.0, 99.9)
        with self.assertRaises(ValueError):
            m.apply_trade(1500, 100.0, 1.0, False)


# ---------------------------------------------------------------- gates

def _write_hour(tmp, ob, tr, mk, symbol="TESTUSDC", date="2026-01-01", hour="00"):
    base = Path(tmp) / date / hour
    base.mkdir(parents=True, exist_ok=True)
    ob.to_parquet(base / f"{symbol}_orderbook.parquet")
    tr.to_parquet(base / f"{symbol}_trades.parquet")
    mk.to_parquet(base / f"{symbol}_mark_price.parquet")


H0 = 1_767_225_600_000  # 2026-01-01 00:00:00 UTC, ms


def _tiny_frames(trade_ids=(100,), trade_price="100.0", trade_qty="1.0",
                 cancel_qty=4.0, base_ms=H0, symbol="TESTUSDC",
                 bid="100.0", ask="100.2", step_to=None):
    # step_to: (new_bid, new_ask) -> mid step via delete+add at base+10000
    et = [base_ms + 1, base_ms + 2, base_ms + 3000, base_ms + 3001]
    sides = ["bid", "ask", "bid", "ask"]
    prices = [bid, ask, bid, ask]
    qtys = ["5.0", "5.0", str(cancel_qty), "4.0"]
    if step_to is not None:
        et += [base_ms + 10000] * 4
        sides += ["bid", "ask", "bid", "ask"]
        prices += [bid, ask, step_to[0], step_to[1]]
        qtys += ["0.0", "0.0", "5.0", "5.0"]
    n_rows = len(et)
    ob = pd.DataFrame({
        "received_time": [(e) * 1_000_000 for e in et],
        "event_time": et,
        "transaction_time": [np.nan] * n_rows,
        "symbol": [symbol] * n_rows,
        "event_type": ["snapshot"] * 2 + ["update"] * (n_rows - 2),
        "first_update_id": list(range(n_rows)),
        "final_update_id": list(range(n_rows)),
        "prev_final_update_id": [0] + list(range(n_rows - 1)),
        "last_update_id": list(range(n_rows)),
        "side": sides,
        "price": prices,
        "quantity": qtys,
        "order_count": [np.nan] * n_rows,
    })
    n = len(trade_ids)
    tr = pd.DataFrame({
        "received_time": (base_ms + np.arange(n) + 4000) * 1_000_000,
        "event_time": base_ms + np.arange(n) + 4000,
        "trade_id": list(trade_ids),
        "price": [trade_price] * n,
        "quantity": [trade_qty] * n,
        "trade_time": base_ms + np.arange(n) + 3016,
        "is_buyer_maker": [True] * n,
        "order_type": ["MARKET"] * n,
    })
    mk = pd.DataFrame({
        "received_time": [(base_ms + 1) * 1_000_000], "event_time": [base_ms + 1],
        "symbol": [symbol],
        "mark_price": [str((float(bid) + float(ask)) / 2)],
        "index_price": [str((float(bid) + float(ask)) / 2)],
        "estimated_settle_price": [str((float(bid) + float(ask)) / 2)],
        "funding_rate": ["0.0001"],
        "next_funding_time": [base_ms + 7_200_000],
    })
    return ob, tr, mk


class TestPipelineGates(unittest.TestCase):
    def test_capture_rate_math(self):
        decreases = [(5000, "bid", 100.0, 1.0)]
        tr = pd.DataFrame({"trade_time": [5016], "price": ["100.0"],
                           "quantity": ["1.0"], "is_buyer_maker": [True]})
        rate, attr, total = trade_capture_rate(decreases, tr)
        self.assertEqual(rate, 1.0)
        tr_far = pd.DataFrame({"trade_time": [5016], "price": ["200.0"],
                               "quantity": ["1.0"], "is_buyer_maker": [True]})
        rate, _, _ = trade_capture_rate(decreases, tr_far)
        self.assertEqual(rate, 0.0)

    def test_gate_rejections_and_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            ob, tr, mk = _tiny_frames()
            _write_hour(tmp, ob, tr, mk)
            detail = run_gates(ob, tr)
            self.assertTrue(detail["ok"], detail)
            meta = build_hour("TESTUSDC", "2026-01-01", "00", base_dir=tmp,
                              out_root=str(Path(tmp) / "out"))
            self.assertEqual(meta["status"], "ok")
            self.assertEqual(meta["rows"], 3600)
            out = Path(tmp) / "out" / "TESTUSDC" / "2026-01-01" / "00.parquet"
            self.assertTrue(out.exists())
            # resume: second run skips
            meta2 = build_hour("TESTUSDC", "2026-01-01", "00", base_dir=tmp,
                               out_root=str(Path(tmp) / "out"))
            self.assertTrue(meta2.get("skipped_existing"))

            # trade_id gap -> reject
            ob2, tr2, mk2 = _tiny_frames(trade_ids=(100, 102))
            _write_hour(tmp, ob2, tr2, mk2)
            with self.assertRaises(ValidationError):
                build_hour("TESTUSDC", "2026-01-01", "00", base_dir=tmp,
                           out_root=str(Path(tmp) / "out2"), force=True)

            # capture rate 0 -> reject
            ob3, tr3, mk3 = _tiny_frames(trade_price="200.0")
            _write_hour(tmp, ob3, tr3, mk3)
            with self.assertRaises(ValidationError):
                build_hour("TESTUSDC", "2026-01-01", "00", base_dir=tmp,
                           out_root=str(Path(tmp) / "out3"), force=True)


# ------------------------------------------------------- P1.5: chaining

def build_chain_hour1():
    """Relative-hour stream: constant book (mid 100.1), one buy trade every
    second, and one cancel 0.5 @99.5 (49.75) in each of secs 3580..3599."""
    m = make_machine()
    m.apply_snapshot(["bid", "ask"], [1000, 1002], [10.0, 8.0])
    m.apply_mark(1000, 100.1, 0.0001, 7200000)
    for j in range(1, 3600):
        m.apply_trade(j * 1000, 100.0, 1.0, False)
        if 3580 <= j < 3600:
            m.apply_update(j * 1000, "bid", 995, 0.5, 99.5)
            m.apply_update(j * 1000, "bid", 995, 1.0, 99.5)
    m.finalize_upto(3599)
    return m


class TestCrossHourChaining(unittest.TestCase):
    def test_export_import_roundtrip(self):
        m1 = build_chain_hour1()
        m2 = make_machine(hour_start=3600 * 1000)
        # different book on purpose: proves the book rebuilds from the new
        # snapshot while only window state crosses the boundary
        m2.apply_snapshot(["bid", "ask"], [1000, 1002], [20.0, 16.0])
        m2.import_state(m1.export_state())
        m2.apply_trade(3600 * 1000 + 1000, 100.0, 1.0, False)
        m2.apply_trade(3600 * 1000 + 10000, 100.0, 1.0, False)
        m2.finalize_upto(30)

        b = lambda j: m2._blk(j)   # noqa: E731
        # book rebuilt from the hour-2 snapshot: A1 uses 20/16 only
        self.assertAlmostEqual(b(0)["A1"], (20.0 - 16.0) / 36.0, places=12)
        self.assertAlmostEqual(b(0)["mid"], 100.1, places=9)
        # C1 windows never cross the boundary (each 1s window lies inside
        # one hour); the 30s/60s flow windows and OFI prefix do
        self.assertTrue(math.isnan(b(0)["C1"]))                # no trade in hour-2 sec 0
        self.assertEqual(b(1)["C1"], 1.0)
        self.assertEqual(b(1)["C3"], 1.0)                      # 30 buys across boundary
        self.assertEqual(b(5)["D4"], 0.0)                      # OFI prefix chained
        self.assertEqual(b(10)["D6"], 0.0)                     # 31 mids, in-hour by snapshot
        # B1 at rel sec 10 covers (-20, 10] = hour1 secs 3581..3599:
        # 19 cancels x 49.75 (the sec-30 window is already fully in-hour)
        self.assertAlmostEqual(b(10)["B1"], 19 * 49.75 / 30.0, places=9)
        self.assertEqual(b(30)["B1"], 0.0)
        # 1h rolling medians covered by the imported tail
        self.assertEqual(b(10)["A8"], 0.0)
        self.assertEqual(b(10)["C5"], 1.0)
        # mark stream carries: funding known before hour-2's first mark row
        self.assertAlmostEqual(b(0)["E8"], 0.0001, places=12)
        # E10/E11 follow absolute time (hour 2 -> sin(pi/12))
        self.assertAlmostEqual(b(0)["E10"], math.sin(2 * math.pi / 24), places=9)

    def test_chained_equals_continuous(self):
        """Chained two-hour build matches one continuous machine (window
        factors must not see the seam)."""
        ref = make_machine()
        ref.apply_snapshot(["bid", "ask"], [1000, 1002], [10.0, 8.0])
        for j in range(1, 3600):
            ref.apply_trade(j * 1000, 100.0, 1.0, False)
            if 3580 <= j < 3600:
                ref.apply_update(j * 1000, "bid", 995, 0.5, 99.5)
                ref.apply_update(j * 1000, "bid", 995, 1.0, 99.5)
        # hour 2 has its own snapshot with IDENTICAL levels (so the book
        # state matches the continuous reference), then one trade at sec 3601
        ref.apply_snapshot(["bid", "ask"], [1000, 1002], [10.0, 8.0])
        ref.apply_trade(3601 * 1000, 100.0, 1.0, False)
        ref.apply_trade(3610 * 1000, 100.0, 1.0, False)
        ref.finalize_upto(3630)

        m1 = build_chain_hour1()
        m2 = make_machine(hour_start=3600 * 1000)
        m2.apply_snapshot(["bid", "ask"], [1000, 1002], [10.0, 8.0])
        m2.import_state(m1.export_state())
        m2.apply_trade(3600 * 1000 + 1000, 100.0, 1.0, False)
        m2.apply_trade(3600 * 1000 + 10000, 100.0, 1.0, False)
        m2.finalize_upto(30)

        for j in (list(range(0, 11)) + [30]):
            rb, cb = ref._blk(3600 + j), m2._blk(j)
            for name in ("C1", "C3", "B1", "D1", "D4", "D6", "A7"):
                rv, cv = rb[name], cb[name]
                if math.isnan(rv) and math.isnan(cv):
                    continue
                self.assertEqual(rv, cv, f"{name} at rel {j}")

    def test_segment_head_still_nan(self):
        m2 = make_machine(hour_start=3600 * 1000)
        m2.apply_snapshot(["bid", "ask"], [1000, 1002], [20.0, 16.0])
        m2.apply_trade(3600 * 1000 + 1000, 100.0, 1.0, False)
        m2.finalize_upto(30)
        self.assertTrue(math.isnan(m2._blk(0)["C1"]))
        # mid-lag factors never cross (snapshot gives mid(0)); flow windows do
        self.assertEqual(m2._blk(5)["D1"], 0.0)
        self.assertTrue(math.isnan(m2._blk(10)["B1"]))   # window crosses boundary
        self.assertEqual(m2._blk(30)["B1"], 0.0)         # window fully in-hour
        self.assertTrue(math.isnan(m2._blk(1)["C3"]))


class TestChainStateFiles(unittest.TestCase):
    def test_builder_chains_through_state_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            date = "2026-01-01"
            ob0, tr0, mk0 = _tiny_frames()
            ob1, tr1, mk1 = _tiny_frames(base_ms=H0 + 3_600_000)
            _write_hour(tmp, ob0, tr0, mk0, hour="00")
            _write_hour(tmp, ob1, tr1, mk1, hour="01")
            out = str(Path(tmp) / "out")

            meta0 = build_hour("TESTUSDC", date, "00", base_dir=tmp, out_root=out)
            self.assertTrue(meta0["chain"]["state"] == "exported")
            self.assertTrue((Path(out) / "TESTUSDC" / date / "00.state.pkl").exists())

            build_hour("TESTUSDC", date, "01", base_dir=tmp, out_root=out)
            df1 = pd.read_parquet(Path(out) / "TESTUSDC" / date / "01.parquet")
            # mid is constant across the boundary -> D1 at sec 5 defined (0)
            row5 = df1[df1["sec"] == H0 // 1000 + 3600 + 5].iloc[0]
            self.assertEqual(row5["D1"], 0.0)
            row3 = df1[df1["sec"] == H0 // 1000 + 3600 + 3].iloc[0]
            self.assertFalse(math.isnan(row3["D4"]))   # OFI window crosses
            row0 = df1[df1["sec"] == H0 // 1000 + 3600].iloc[0]
            self.assertFalse(math.isnan(row0["A8"]))   # 1h spread window crosses

            # control: same hour without chaining -> segment-head NaN
            build_hour("TESTUSDC", date, "01", base_dir=tmp,
                       out_root=str(Path(tmp) / "out_nc"), chain=False)
            df1nc = pd.read_parquet(Path(tmp) / "out_nc" / "TESTUSDC" / date / "01.parquet")
            row3nc = df1nc[df1nc["sec"] == H0 // 1000 + 3600 + 3].iloc[0]
            self.assertTrue(math.isnan(row3nc["D4"]))
            row0nc = df1nc[df1nc["sec"] == H0 // 1000 + 3600].iloc[0]
            self.assertTrue(math.isnan(row0nc["A8"]))


class TestMultiSymbolLockstep(unittest.TestCase):
    def test_market_and_btc_feeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            date, hour = "2026-01-01", "00"
            obA, trA, mkA = _tiny_frames(symbol="COINA", bid="100.0", ask="100.2")
            obB, trB, mkB = _tiny_frames(symbol="COINB", bid="200.0", ask="200.2",
                                         trade_price="200.0",
                                         step_to=("200.2", "200.4"))
            _write_hour(tmp, obA, trA, mkA, symbol="COINA", date=date, hour=hour)
            _write_hour(tmp, obB, trB, mkB, symbol="COINB", date=date, hour=hour)
            out = str(Path(tmp) / "out")
            metas = build_hour_multi(["COINA", "COINB"], date, hour,
                                     base_dir=tmp, out_root=out)
            self.assertEqual(set(metas), {"COINA", "COINB"})
            self.assertTrue(metas["COINA"]["external_feeds"]["market"])

            dfA = pd.read_parquet(Path(out) / "COINA" / date / "00.parquet")
            sec0 = H0 // 1000
            # mkt mid: secs 0..9 = (100.1+200.1)/2 = 150.1; secs 10+ = 150.2
            mkt_early, mkt_late = (100.1 + 200.1) / 2, (100.1 + 200.3) / 2
            e5 = dfA[dfA["sec"] == sec0 + 14].iloc[0]["E5"]
            self.assertAlmostEqual(e5, (mkt_late / mkt_early - 1) * 1e4, places=6)
            e7 = dfA[dfA["sec"] == sec0 + 69].iloc[0]["E7"]
            self.assertAlmostEqual(e7,
                                   math.log(mkt_late / mkt_early) / math.sqrt(60) * 1e4,
                                   places=6)
            # no BTC symbol in the set -> E1-E4 stay NaN
            self.assertTrue(math.isnan(dfA.iloc[40]["E1"]))
            # COINB stepped its own mid at sec 10
            dfB = pd.read_parquet(Path(out) / "COINB" / date / "00.parquet")
            self.assertAlmostEqual(dfB[dfB["sec"] == sec0 + 15].iloc[0]["mid"],
                                   200.3, places=9)


class TestFrozenThresholds(unittest.TestCase):
    def test_frozen_table_preferred(self):
        with tempfile.TemporaryDirectory() as tmp:
            tpath = Path(tmp) / "thresholds.json"
            tpath.write_text(json.dumps({
                "version": 1,
                "thresholds": {"TESTUSDC": {"cancel_p90_notional": 123.0,
                                            "median_cancel_qty": 4.0,
                                            "trade_p90_notional": 567.0}}}))
            ob, tr, mk = _tiny_frames()
            _write_hour(tmp, ob, tr, mk)
            out = str(Path(tmp) / "out")
            meta = build_hour("TESTUSDC", "2026-01-01", "00", base_dir=tmp,
                              out_root=out, thresholds=load_thresholds(tpath))
            self.assertEqual(meta["params"]["source"], "frozen")
            self.assertEqual(meta["params"]["value"]["trade_p90_notional"], 567.0)

            meta2 = build_hour("TESTUSDC", "2026-01-01", "00", base_dir=tmp,
                               out_root=str(Path(tmp) / "out2"))
            self.assertEqual(meta2["params"]["source"], "provisional")


# ---------------------------------------------------------------- anchors

PROBE_DIR = ROOT / "data" / "prediction-v3" / "probe-p0.5"
RAW_HOUR = ROOT / "data" / "cryptohftdata" / "raw" / "binance_futures" / \
    "2026-09-19" / "00"
ANCHOR_SYMBOL, ANCHOR_DATE, ANCHOR_HOUR = "BTCUSDC", "2026-09-19", "00"


@unittest.skipUnless((PROBE_DIR / "probe_results.json").exists()
                      and (RAW_HOUR / "BTCUSDC_orderbook.parquet").exists(),
                      "probe anchors or raw hour data not present")
class TestProbeAnchors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.probe = json.loads((PROBE_DIR / "probe_results.json").read_text())
        cls.ob = pd.read_parquet(RAW_HOUR / "BTCUSDC_orderbook.parquet")
        cls.tr = pd.read_parquet(RAW_HOUR / "BTCUSDC_trades.parquet")
        cls.tmp = tempfile.mkdtemp()
        cls.meta = build_hour(ANCHOR_SYMBOL, ANCHOR_DATE, ANCHOR_HOUR,
                              base_dir=str(RAW_HOUR.parent.parent),
                              out_root=cls.tmp,
                              extra_marks=[f["T_ms"] for f in cls.probe["factors"]])
        cls.anchors = pd.read_parquet(
            Path(cls.tmp) / ANCHOR_SYMBOL / ANCHOR_DATE / "00.anchors.parquet")
        cls.grid = pd.read_parquet(
            Path(cls.tmp) / ANCHOR_SYMBOL / ANCHOR_DATE / "00.parquet")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_reference_replay_matches_probe_exact_ms(self):
        """The replay reference reproduces the probe anchors at exact ms."""
        cache = {}
        for f in self.probe["factors"]:
            t = f["T_ms"]
            book = replay.book_at(self.ob, t, cache)
            a1, bq3, aq3, bb, ba = replay.depth_imb(book, 0.1, 3)
            self.assertAlmostEqual(a1, f["A1"], delta=1e-9)
            self.assertAlmostEqual(bq3, f["bid_qty_3t"], delta=1e-9)
            self.assertAlmostEqual(aq3, f["ask_qty_3t"], delta=1e-9)
            self.assertAlmostEqual((bb + ba) / 2, f["mid"], delta=1e-9)
            c1, buy, sell, n = replay.taker_imb_1s(self.tr, t)
            self.assertAlmostEqual(c1, f["C1"], delta=1e-9)
            self.assertAlmostEqual(buy, f["taker_buy_notional_1s"], delta=1e-3)
            self.assertAlmostEqual(sell, f["taker_sell_notional_1s"], delta=1e-3)
            self.assertEqual(n, f["n_trades_1s"])
            start = self.meta["hour_start_ms"]
            mids = replay.grid_mids(self.ob, start)
            d6 = replay.rv_30s(mids, (t - start) // 1000)
            self.assertAlmostEqual(d6, f["D6_rv_30s"], delta=1e-9)

    def test_builder_anchor_rows(self):
        """Builder anchor rows: C1/D6 exact at the probe timestamps; A1 is the
        1s-grid sample (state at the second mark), cross-checked against the
        reference replay at the same marks."""
        cache = {}
        start = self.meta["hour_start_ms"]
        for f in self.probe["factors"]:
            t = f["T_ms"]
            row = self.anchors[self.anchors["mark_ms"] == t].iloc[0]
            self.assertAlmostEqual(row["C1"], f["C1"], delta=1e-6)
            self.assertAlmostEqual(row["D6"], f["D6_rv_30s"], delta=1e-6)
            mark = start + ((t - start) // 1000) * 1000
            book = replay.book_at(self.ob, mark, cache)
            a1_mark, _, _, _, _ = replay.depth_imb(book, 0.1, 3)
            self.assertAlmostEqual(row["A1"], a1_mark, delta=1e-6)

    def test_builder_grid_rows(self):
        start = self.meta["hour_start_ms"]
        for f in self.probe["factors"]:
            sec = (f["T_ms"] - start) // 1000
            mark = start + sec * 1000
            row = self.grid[self.grid["sec"] == start // 1000 + sec].iloc[0]
            self.assertAlmostEqual(row["D6"], f["D6_rv_30s"], delta=1e-6)
            # grid C1 uses the grid-second window (mark-1000, mark]
            c1_grid, _, _, _ = replay.taker_imb_1s(self.tr, mark)
            if math.isnan(c1_grid):
                self.assertTrue(math.isnan(row["C1"]))
            else:
                self.assertAlmostEqual(row["C1"], c1_grid, delta=1e-6)
            _, _, _, bb, ba = replay.depth_imb(
                replay.book_at(self.ob, mark, {}), 0.1, 3)
            self.assertAlmostEqual(row["mid"], (bb + ba) / 2, delta=1e-6)

    def test_gates_pass_on_certified_hour(self):
        self.assertTrue(self.meta["gates"]["ok"], self.meta["gates"]["problems"])
        self.assertGreaterEqual(self.meta["gates"]["capture_rate"], 0.90)
        self.assertTrue(self.meta["gates"]["trade_id_contiguous"])
        self.assertEqual(self.meta["rows"], 3600)


if __name__ == "__main__":
    unittest.main()
