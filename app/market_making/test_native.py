import unittest

from .as_quote import NativeAS
from .core import D
from .test_core import NOW, book, config, rules


class NativeTests(unittest.TestCase):
    def test_intensity_window_does_not_change_native_volatility(self):
        a=NativeAS(config(),rules())
        cfg=config();cfg['as_intensity_window_seconds']=3600
        b=NativeAS(cfg,rules())
        for i in range(650):
            mid=D('7.6')+D(i%7)*D('.001')
            a.sample(mid,NOW+i*1000);b.sample(mid,NOW+i*1000)
        self.assertEqual(len(a.samples),config()['calibration_window_seconds'])
        self.assertEqual(len(b.samples),649)
        self.assertEqual(a.vol.current_value,b.vol.current_value)
        cfg['as_intensity_window_seconds']=True
        with self.assertRaises(ValueError):NativeAS(cfg,rules())

    def test_arrival_failure_detail_and_bp_mapping(self):
        model=NativeAS(config(),rules());model.reference=D('10')
        values=[-1.]*120+[.002]*4
        model.samples.extend(values);model.sample_references.extend([10.]*len(values))
        model.fit(NOW);snap=model.fit_snapshot(NOW)
        self.assertEqual(snap['arrival_detail'],'insufficient_distance_points')
        self.assertAlmostEqual(snap['distance_bp_at_fit'][0],.5)
        self.assertEqual(snap['bp_exceed_seconds'][0],4)
        model.samples.clear();model.samples.extend([-1.]*120+[.01]*4)
        model.fit(NOW+30000)
        self.assertEqual(model.fit_snapshot(NOW+30000)['arrival_detail'],'flat_arrival_curve')

    def test_prediction_observer_uses_parameters_available_at_bucket_start(self):
        import math
        model=NativeAS(config(),rules());rows=[]
        model.sample_observer=lambda *row:rows.append(row)
        model.sample(D('7.6'),NOW)
        model.alpha=D('1');model.kappa=D('300');model.last_fit_ms=NOW
        model.sample(D('7.601'),NOW+1000)
        self.assertEqual(rows[-1][3],1.)
        model.last_fit_ms=NOW+1500
        model.sample(D('7.602'),NOW+2000)
        self.assertTrue(math.isnan(rows[-1][3]))
        n=len(rows);model.sample(D('7.603'),NOW+5000)
        self.assertEqual(len(rows),n)

    def test_native_formula_and_inventory_sign(self):
        model = NativeAS(config(), rules())
        for price in [7.600, 7.601, 7.599, 7.602, 7.601]:
            model.vol.add_sample(price)
        model.alpha = D("1")
        model.kappa = D("300")
        model.gamma = D("1")
        model.last_fit_ms = NOW
        b = book()
        bid, ask = model.quote(b, D("0"), NOW)
        volatility = D(str(model.vol.current_value))
        expected_width = volatility + 2 * (1 + D("1") / D("300")).ln()
        self.assertAlmostEqual(ask - bid, expected_width, places=10)
        self.assertAlmostEqual((bid + ask) / 2, b.mid, places=10)
        long_bid, long_ask = model.quote(b, D("5"), NOW)
        short_bid, short_ask = model.quote(b, D("-5"), NOW)
        self.assertLess(long_bid, bid)
        self.assertLess(long_ask, ask)
        self.assertGreater(short_bid, bid)
        self.assertGreater(short_ask, ask)

    def test_stale_estimator_refuses_quotes(self):
        model = NativeAS(config(), rules())
        with self.assertRaises(ValueError):
            model.quote(book(), D("0"), NOW)

    def test_mask_keeps_valid_history_and_long_gap_fails_coverage(self):
        import numpy as np
        cfg=config(); cfg['as_missing_policy']='mask'
        model=NativeAS(cfg,rules())
        values=[-1.0]*60+[.001]*30+[.002]*20+[.003]*10+[.005]*8+[.009]*4
        model.samples.extend(values); model.fit(NOW)
        self.assertGreater(model.last_fit_ms,0)
        model.last_second=NOW//1000
        model.sample(D('7.6'),NOW+3000)
        self.assertEqual(list(model.samples)[:len(values)],values)
        self.assertEqual(sum(not np.isfinite(v) for v in model.samples),3)
        self.assertGreater(model.last_fit_ms,0)
        later=NOW+(cfg['calibration_window_seconds']+10)*1000
        model.sample(D('7.6'),later); model.fit(later)
        self.assertEqual(model.fit_error,'insufficient_observation_coverage')
        self.assertFalse(model.ready(later))

    def test_book_sampling_ignores_mark_block_but_rejects_stale_depth(self):
        cfg = config(); cfg['as_sampling_policy'] = 'book'
        model = NativeAS(cfg, rules()); b = book(); b.received_ms = NOW
        self.assertTrue(model.sampling_allowed(b, NOW, 'stale_mark_price'))
        self.assertFalse(model.sampling_allowed(b, NOW + cfg['stale_book_ms'] + 1, ''))
        b.ready = False
        self.assertFalse(model.sampling_allowed(b, NOW, ''))
        cfg['as_sampling_policy'] = 'quote_health'
        self.assertFalse(model.sampling_allowed(b, NOW, 'stale_mark_price'))

    def test_quiet_seconds_are_samples_but_missing_seconds_reset(self):
        model = NativeAS(config(), rules())
        model.sample(D('7.6'), NOW)
        model.sample(D('7.6'), NOW + 1000)
        self.assertEqual(list(model.samples), [-1.0])
        model.last_fit_ms = NOW
        model.sample(D('7.6'), NOW + 3000)
        self.assertEqual(list(model.samples), [])
        self.assertEqual(model.last_fit_ms, 0)
        self.assertEqual(model.diagnostics['missing_seconds'], 1)

    def test_readiness_reason_matches_ready_and_reports_expired_fit(self):
        model = NativeAS(config(), rules())
        for value in [7.6, 7.601, 7.599]:
            model.vol.add_sample(value)
        model.alpha, model.kappa, model.last_fit_ms = D('1'), D('300'), NOW
        self.assertEqual(model.readiness_reason(NOW), 'ready')
        self.assertTrue(model.ready(NOW))
        later = NOW + model.config['calibration_max_age_seconds'] * 1000 + 1
        self.assertTrue(model.readiness_reason(later).startswith('fit_expired:'))
        self.assertFalse(model.ready(later))

    def test_fit_uses_quiet_seconds(self):
        model = NativeAS(config(), rules())
        model.samples.extend([-1.0] * 60 + [.001] * 30 + [.002] * 20 + [.003] * 10 + [.005] * 8 + [.009] * 4)
        model.fit(NOW)
        self.assertGreater(model.alpha, 0)
        self.assertGreater(model.kappa, 0)
        self.assertGreater(model.fit_r2, .3)


if __name__ == "__main__":
    unittest.main()
