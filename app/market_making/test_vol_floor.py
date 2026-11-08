import math
import unittest

from .core import Account, D
from .test_core import NOW, rules
from .test_spread_control import sc_config
from .vol_floor import RollingVolatility


def ticker(mid):
    return {"b": str(D(mid) - D("0.001")), "a": str(D(mid) + D("0.001"))}


class VolFloorConfigTests(unittest.TestCase):
    def test_vol_window_min_parses_minutes(self):
        v = RollingVolatility({"vol_window_min": "30"}, D("0.001"))
        self.assertEqual(v.window_seconds, 1800)
        # Fractional minutes for small test windows.
        v = RollingVolatility({"vol_window_min": "0.084"}, D("0.001"))
        self.assertEqual(v.window_seconds, 5)

    def test_legacy_s1_keys_still_accepted(self):
        v = RollingVolatility({"s1_window_seconds": 4, "s1_ticker_stale_ms": 1000,
                               "s1_max_missing_ratio": 0.2}, D("0.001"))
        self.assertEqual(v.window_seconds, 4)
        self.assertEqual(v.stale_ms, 1000)
        # New keys win over legacy aliases.
        mixed = RollingVolatility({"vol_window_min": "0.084", "s1_window_seconds": 4},
                                  D("0.001"))
        self.assertEqual(mixed.window_seconds, 5)

    def test_invalid_config_rejected(self):
        with self.assertRaises(ValueError):
            RollingVolatility({"vol_missing_ratio": "1.5"}, D("0.001"))
        with self.assertRaises(ValueError):
            RollingVolatility({"vol_stale_ms": 0}, D("0.001"))


class VolFloorSigmaTests(unittest.TestCase):
    def estimator(self, **kwargs):
        return RollingVolatility({"vol_window_min": "0.084", "vol_stale_ms": "5000",
                                  "vol_missing_ratio": "0.2", **kwargs}, D("0.001"))

    def test_hand_computed_sigma_population_std(self):
        v = self.estimator()
        mids = [D("100"), D("101"), D("100"), D("102"), D("101"), D("100")]
        for i, mid in enumerate(mids):
            v.on_ticker(ticker(mid), NOW + 1000 * i + 1)
            self.assertFalse(v.snapshot(NOW + 1000 * i + 1)["valid"])
        s = v.snapshot(NOW + 6001)
        self.assertTrue(s["valid"])
        returns = [float(b / a - 1) for a, b in zip(mids, mids[1:])]
        mean = sum(returns) / len(returns)
        expected = math.sqrt(sum((r - mean) ** 2 for r in returns) / len(returns))
        expected *= float(mids[-1]) / 0.001
        self.assertAlmostEqual(float(s["sigma_ticks"]), expected, places=7)

    def test_causality_future_ticker_does_not_fill_past_grid_point(self):
        v = self.estimator()
        v.on_ticker(ticker("100"), NOW + 1)
        v.on_ticker(ticker("110"), NOW + 2500)  # arrives after the s0+2 boundary
        # The s0+2 grid point was stamped with the OLD mid (100): a ticker
        # only covers grid points strictly after its own arrival, so an
        # estimate at time t never observes data received at or after t.
        self.assertEqual(v.points[-1], ((NOW + 2000) // 1000, D("100")))

    def test_stale_point_marked_missing_not_zero_return(self):
        v = self.estimator(vol_stale_ms="1500")
        v.on_ticker(ticker("100"), NOW + 1)
        v.on_ticker(ticker("101"), NOW + 5001)
        s = v.snapshot(NOW + 6001)
        # s0+2 had age 1999ms > 1500ms at its grid second: missing, and the
        # returns across missing points are excluded (never zero-filled).
        self.assertIsNone(v.points[1][1])
        self.assertEqual(s["missing_ratio"], 0.8)
        self.assertFalse(s["valid"])

    def test_missing_ratio_boundary_20_percent(self):
        # Exactly 20% missing points -> sigma still valid (spec: invalid only
        # when the share is STRICTLY greater than 20%).
        v = self.estimator(vol_stale_ms="3500")
        v.on_ticker(ticker("100"), NOW + 1)
        v.on_ticker(ticker("101"), NOW + 4001)
        v.on_ticker(ticker("100"), NOW + 5001)
        s = v.snapshot(NOW + 6001)
        self.assertEqual(s["missing_ratio"], 0.2)
        self.assertTrue(s["valid"])
        # Push past the boundary: two missing points of five (40%).
        w = self.estimator(vol_stale_ms="3500")
        w.on_ticker(ticker("100"), NOW + 1)
        w.on_ticker(ticker("101"), NOW + 5001)
        w.on_ticker(ticker("100"), NOW + 6001)
        s = w.snapshot(NOW + 7001)
        self.assertEqual(s["missing_ratio"], 0.4)
        self.assertFalse(s["valid"])


class VolFloorMappingTests(unittest.TestCase):
    def account(self, **overrides):
        cfg = sc_config(s1_enabled=True, s1_k="2",
                        vol_floor_min_ticks="5", vol_floor_max_ticks="10",
                        vol_degraded_min="1")
        cfg.update(overrides)
        return Account("s1", cfg, rules(), lambda *args: None)

    def valid(self, sigma):
        return {"valid": True, "sigma_ticks": D(sigma), "missing_ratio": 0}

    def invalid(self, ratio="1"):
        return {"valid": False, "sigma_ticks": None, "missing_ratio": D(ratio)}

    def test_k_mapping_clamps_to_configured_bounds(self):
        a = self.account()
        a.update_volatility(self.valid("20"), NOW)   # ceil(2x20)=40 -> 10
        self.assertEqual(a.vol_floor_ticks, 10)
        a.update_volatility(self.valid("2"), NOW + 1)   # ceil(4)=4 -> 5
        self.assertEqual(a.vol_floor_ticks, 5)
        a.update_volatility(self.valid("3"), NOW + 2)   # ceil(6)=6 inside
        self.assertEqual(a.vol_floor_ticks, 6)

    def test_stale_freeze_never_below_static(self):
        a = self.account()
        a.update_volatility(self.valid("2"), NOW)    # ceil(4)=4, clamped to min 5
        self.assertEqual(a.vol_floor_ticks, 5)
        events = []
        a.emit = lambda name, kind, ms, fields: events.append(kind)
        a.update_volatility(self.invalid(), NOW + 1000)
        self.assertEqual(a.vol_floor_ticks, 6)       # frozen AND >= static
        self.assertEqual(a.vol_status, "stale_frozen")
        self.assertIn("vol_floor_stale", events)

    def test_degraded_after_configured_minutes_then_recovers(self):
        a = self.account()
        a.update_volatility(self.valid("5"), NOW)    # floor 10
        events = []
        a.emit = lambda name, kind, ms, fields: events.append(kind)
        a.update_volatility(self.invalid(), NOW + 1000)
        self.assertFalse(a.vol_degraded)
        a.update_volatility(self.invalid(), NOW + 61000)  # 1 minute without sigma
        self.assertTrue(a.vol_degraded)
        self.assertEqual(a.vol_status, "degraded_paused")
        self.assertEqual(a.vol_floor_ticks, a.static_floor_ticks)
        self.assertIn("vol_floor_degraded", events)
        a.update_volatility(self.valid("5"), NOW + 62000)
        self.assertFalse(a.vol_degraded)
        self.assertEqual(a.vol_status, "ready")
        self.assertEqual(a.vol_floor_ticks, 10)

    def test_invalid_bounds_rejected(self):
        with self.assertRaises(ValueError):
            self.account(vol_floor_min_ticks="10", vol_floor_max_ticks="5")
        with self.assertRaises(ValueError):
            self.account(vol_degraded_min="0")


if __name__ == "__main__":
    unittest.main()
