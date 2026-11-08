import json
import math
import unittest
from copy import deepcopy

from .core import Account, D, encode
from .policy import as_quotes
from .spread_control import SpreadController
from .core import account_configs
from .s1_config import make_s1_config
from .test_core import NOW, book, config, rules
from .test_spread_control import sc_config
from .vol_floor import RollingVolatility


def ticker(mid):
    return {"b": str(D(mid) - D("0.001")), "a": str(D(mid) + D("0.001"))}


class VolatilityTests(unittest.TestCase):
    def estimator(self, **kwargs):
        return RollingVolatility({"s1_window_seconds": 4, **kwargs}, D("0.001"))

    def test_grid_does_not_use_ticker_arriving_after_boundary(self):
        v = self.estimator()
        v.on_ticker(ticker("100"), NOW + 1)
        v.on_ticker(ticker("110"), NOW + 1500)
        self.assertEqual(v.points[-1], ((NOW + 1000) // 1000, D("100")))
        v.advance(NOW + 2000)
        self.assertEqual(v.points[-1][1], D("110"))

    def test_population_std_and_full_window(self):
        v = self.estimator()
        mids = [D("100"), D("101"), D("100"), D("102"), D("101")]
        for i, mid in enumerate(mids):
            v.on_ticker(ticker(mid), NOW + 1000 * i + 1)
            self.assertFalse(v.snapshot(NOW + 1000 * i + 1)["valid"])
        s = v.snapshot(NOW + 5000)
        self.assertTrue(s["valid"])
        r = [float(b / a - 1) for a, b in zip(mids, mids[1:])]
        mean = sum(r) / 4
        expected = math.sqrt(sum((x - mean) ** 2 for x in r) / 4) * 101 / 0.001
        self.assertAlmostEqual(float(s["sigma_ticks"]), expected, places=7)

    def test_gaps_are_missing_points_not_zero_or_compressed(self):
        v = self.estimator(s1_ticker_stale_ms=1000)
        v.on_ticker(ticker("100"), NOW + 1)
        v.on_ticker(ticker("120"), NOW + 3001)
        s = v.snapshot(NOW + 5000)
        self.assertFalse(s["valid"])
        # Spec v0.2 section 2: the missing share is measured over the window's
        # 1s GRID POINTS (3 of 4 missing here), not over returns — one missing
        # point breaks two returns, a return-based ratio would double-count.
        self.assertEqual(s["missing_ratio"], 0.75)
        self.assertIsNone(v.points[1][1])

    def test_stale_ticker_invalidates_even_with_full_history(self):
        v = self.estimator()
        for i in range(6):
            v.on_ticker(ticker(str(100 + i % 2)), NOW + i * 1000 + 1)
        self.assertTrue(v.snapshot(NOW + 6000)["valid"])
        self.assertFalse(v.snapshot(NOW + 10002)["valid"])

    def test_long_restart_gap_is_bounded_and_never_uses_old_price(self):
        v = self.estimator()
        v.on_ticker(ticker("100"), NOW + 1)
        restored = self.estimator().load(json.loads(encode(v.dump())))
        s = restored.snapshot(NOW + 86400000)
        self.assertEqual(len(restored.points), 5)
        self.assertTrue(all(m is None for _, m in restored.points))
        self.assertEqual(s["missing_ratio"], 1.0)
        self.assertFalse(s["valid"])

    def test_invalid_or_time_reversed_ticker_rejected(self):
        v = self.estimator()
        self.assertFalse(v.on_ticker({"b": "NaN", "a": "1"}, NOW))
        self.assertFalse(v.on_ticker({"b": "2", "a": "1"}, NOW))
        self.assertTrue(v.on_ticker(ticker("100"), NOW))
        self.assertFalse(v.on_ticker(ticker("110"), NOW - 1))


class S1StateTests(unittest.TestCase):
    def account(self):
        return Account("D_s1_sc", sc_config(s1_enabled=True, s1_k="2",
                       sc_signal="median_bp", sc_epsilon_bp="0.25"), rules(), lambda *args: None)

    def test_volatility_and_sc_do_not_overwrite_static_or_each_other(self):
        a = self.account()
        original = deepcopy(a.config)
        a.update_volatility({"valid": True, "sigma_ticks": D("5"), "missing_ratio": 0}, NOW)
        c = SpreadController(a)
        for i in range(50):
            c.feed(D("-0.01"), NOW + i, notional=D("100"), fill_key=str(i))
        self.assertEqual(a.vol_floor_ticks, 10)
        self.assertEqual(a.sc_target_ticks, 9)
        self.assertEqual(a.floor_state()["as_floor_ticks"], 10)
        self.assertEqual(a.config, original)
        a.update_volatility({"valid": True, "sigma_ticks": D("8"), "missing_ratio": 0}, NOW + 1000)
        self.assertEqual(a.sc_target_ticks, 9)
        self.assertEqual(a.floor_state()["as_floor_ticks"], 16)

    def test_stale_freeze_degraded_cancel_then_static_and_recover(self):
        a = self.account()
        good = {"valid": True, "sigma_ticks": D("5"), "missing_ratio": 0}
        bad = {"valid": False, "sigma_ticks": None, "missing_ratio": 1}
        a.update_volatility(good, NOW)
        a.quote(D("7.58"), D("7.62"), D("7.6"), NOW, book())
        self.assertTrue(a.orders)
        a.update_volatility(bad, NOW + 1000)
        self.assertEqual(a.vol_floor_ticks, 10)
        a.update_volatility(bad, NOW + 601000)
        self.assertTrue(a.vol_degraded)
        self.assertTrue(all(o.cancel_ms for o in a.orders.values()))
        self.assertEqual(a.vol_floor_ticks, 10)
        a.advance(NOW + 602000, book())
        a.update_volatility(bad, NOW + 602000)
        self.assertEqual(a.vol_floor_ticks, a.static_floor_ticks)
        a.quote(D("7.58"), D("7.62"), D("7.6"), NOW + 603000, book())
        self.assertFalse(a.orders)
        a.update_volatility(good, NOW + 604000)
        self.assertFalse(a.vol_degraded)
        self.assertEqual(a.vol_status, "ready")

    def test_sigma_state_and_floor_survive_checkpoint(self):
        a = self.account()
        a.update_volatility({"valid": True, "sigma_ticks": D("6"), "missing_ratio": 0}, NOW)
        a.update_volatility({"valid": False, "sigma_ticks": None, "missing_ratio": 1}, NOW + 1)
        b = self.account()
        b.restore(json.loads(encode(a.dump())), NOW + 1000)
        self.assertEqual(b.vol_floor_ticks, 12)
        self.assertEqual(b.vol_invalid_since_ms, NOW + 1)

    def test_order_logs_components_and_caps_and_fv_bound_is_static(self):
        a = self.account()
        a.config["fair_value"] = "micro"
        a.sc_target_ticks = 30
        a.vol_floor_ticks = 12
        class FakeAS:
            def quote(self, b, position, now):
                return D("7.599"), D("7.601")
        bid, ask, _, _, shift = as_quotes(FakeAS(), book(), D("0"), NOW,
            a.config, rules(), micro=D("7.8"), floor_ticks=a.floor_state()["as_floor_ticks"])
        self.assertEqual(shift, D("0.006"))
        a.quote(bid, ask, D("7.6"), NOW, book())
        order = a.orders[-1]
        self.assertEqual(order.as_floor_ticks, 24)
        self.assertEqual(order.vol_floor_ticks, 12)
        self.assertEqual(order.sc_target_ticks, 30)
        self.assertTrue(order.floor_cap_hit)
        self.assertEqual(order.floor_reason, "cap")


class MedianSignalTests(unittest.TestCase):
    def controller(self):
        return SpreadController(Account("sc", sc_config(sc_signal="median_bp", sc_epsilon_bp="0.25"),
                                        rules(), lambda *args: None))

    def test_median_widens_even_when_currency_mean_is_positive(self):
        c = self.controller()
        for i in range(49):
            self.assertIsNone(c.feed(D("-0.01"), NOW + i, notional=D("100"), fill_key=str(i)))
        change = c.feed(D("100"), NOW + 49, notional=D("100"), fill_key="outlier")
        self.assertGreater(D(change["window_mean"]), 0)
        self.assertEqual(D(change["signal"]), D("-1"))
        self.assertEqual(change["new_ticks"], 9)

    def test_bp_deadzone_endpoints_and_invalid_samples(self):
        for pnl in ("0.0025", "-0.0025"):
            c = self.controller()
            self.assertIsNone(c.feed(D("1"), NOW, notional=None))
            self.assertIsNone(c.feed(D("NaN"), NOW, notional=D("100")))
            self.assertEqual(c.feedback_seq, 0)
            for i in range(50):
                self.assertIsNone(c.feed(D(pnl), NOW + i, notional=D("100"), fill_key=str(i)))
            self.assertEqual(c.last_eval_reason, "dead_zone")
            c.feed(D("-100"), NOW + 50, notional=D("100"), fill_key="49")
            self.assertEqual(c.feedback_seq, 50)

    def test_profile_isolated_arms_preserve_risk_and_input(self):
        base = config()
        original = deepcopy(base)
        cfg = make_s1_config(base)
        rows = dict(account_configs(cfg))
        self.assertEqual(base, original)
        self.assertEqual(list(rows), ["A_static", "B_sc", "C_s1", "D_s1_sc"])
        for c in rows.values():
            for k in ("initial_collateral", "max_position_notional", "daily_loss_limit", "maker_fee"):
                self.assertEqual(c[k], base[k])
            self.assertEqual(c["sc_signal"], "median_bp")
        self.assertFalse(rows["A_static"]["s1_enabled"])
        self.assertTrue(rows["D_s1_sc"]["s1_enabled"])
        self.assertEqual(cfg["warmup_seconds"], 1800)



class KGridProfileTests(unittest.TestCase):
    def test_kgrid_six_arms_with_distinct_k(self):
        from .s1_config import make_kgrid_config
        cfg = make_kgrid_config({"as_candidate": {}, "_matrix_note": "x",
                                 "capital": "600"})
        names = [a["name"] for a in cfg["accounts"]]
        self.assertEqual(names, ["A_static", "K_100", "K_150", "K_200",
                                 "K_250", "K_300"])
        ks = {a["name"]: a.get("s1_k") for a in cfg["accounts"]}
        self.assertIsNone(ks["A_static"])
        self.assertEqual([ks["K_100"], ks["K_300"]], ["1.0", "3.0"])
        # SC must be off on every grid arm: isolates the S1 effect
        for a in cfg["accounts"]:
            self.assertFalse(a["spread_control_enabled"])
        self.assertEqual(cfg["primary_account"], "K_200")
        self.assertIn("kgrid", cfg["experiment_status"])
