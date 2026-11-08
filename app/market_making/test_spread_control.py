import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from .core import Account, D, ZERO, encode
from .spread_control import SpreadController
from .core import account_configs
from .test_core import NOW, book, config, rules


def sc_config(**overrides):
    base = {**config(), 'as_min_half_spread_ticks': '6',
            'spread_control_enabled': True, 'sc_window_fills': 50,
            'sc_adjust_interval_min': 5, 'sc_min_ticks': 3, 'sc_max_ticks': 24,
            'sc_up_factor': '1.5', 'sc_down_factor': '1.25',
            'sc_max_jump_ticks': 4, 'sc_epsilon': '0.001'}
    base.update(overrides)
    return base


def sc_account(**overrides):
    return Account('sc', sc_config(**overrides), rules(), lambda *args: None)


def feed(controller, n, pnl, start_ms=NOW, step_ms=1000):
    # Blocks may contain an adjust followed by interval-blocked feeds; report
    # the last actual change, not the final feed's (possibly gated) result.
    change = None
    for i in range(n):
        result = controller.feed(D(pnl), start_ms + i * step_ms)
        if result is not None:
            change = result
    return change


class SpreadControllerTests(unittest.TestCase):
    def test_cold_start_waits_for_full_window(self):
        a = sc_account()
        c = SpreadController(a)
        self.assertIsNone(feed(c, 49, '-0.01'))
        self.assertEqual(a.config['as_min_half_spread_ticks'], '6')
        self.assertEqual(c.ticks, 6)

    def test_negative_mean_widens_multiplicatively(self):
        a = sc_account()
        c = SpreadController(a)
        change = feed(c, 50, '-0.01')
        self.assertEqual(change['old_ticks'], 6)
        self.assertEqual(change['new_ticks'], 9)  # 6 x 1.5
        self.assertEqual(change['n'], 50)
        self.assertEqual(a.sc_target_ticks, 9)
        self.assertEqual(a.config['as_min_half_spread_ticks'], '6')
        self.assertAlmostEqual(float(change['window_mean']), -0.01, places=9)

    def test_positive_mean_tightens_multiplicatively(self):
        a = sc_account()
        c = SpreadController(a)
        change = feed(c, 50, '0.01')
        self.assertEqual((change['old_ticks'], change['new_ticks']), (6, 5))  # 6 / 1.25 = 4.8 -> 5
        self.assertEqual(a.sc_target_ticks, 5)
        self.assertEqual(a.config['as_min_half_spread_ticks'], '6')

    def test_tick_rounding_is_nearest_half_away_from_zero(self):
        # Widening: 5 x 1.5 = 7.5 -> 8; 3 x 1.5 = 4.5 -> 5 (never stalls at the low bound).
        a = sc_account(as_min_half_spread_ticks='5')
        c = SpreadController(a)
        self.assertEqual(feed(c, 50, '-0.01')['new_ticks'], 8)
        a = sc_account(as_min_half_spread_ticks='3')
        c = SpreadController(a)
        self.assertEqual(feed(c, 50, '-0.01')['new_ticks'], 5)
        # Tightening: 7 / 1.25 = 5.6 -> 6; 12 / 1.25 = 9.6 -> 10.
        a = sc_account(as_min_half_spread_ticks='7')
        c = SpreadController(a)
        self.assertEqual(feed(c, 50, '0.01')['new_ticks'], 6)
        a = sc_account(as_min_half_spread_ticks='12')
        c = SpreadController(a)
        self.assertEqual(feed(c, 50, '0.01')['new_ticks'], 10)

    def test_single_jump_clamped_both_directions(self):
        high = sc_account(as_min_half_spread_ticks='20')
        c = SpreadController(high)
        # 20 x 1.5 = 30 -> jump-clamped to 20 + 4 = 24, then range-clamped to 24.
        self.assertEqual(feed(c, 50, '-0.01')['new_ticks'], 24)
        self.assertEqual(c.ticks, 24)
        # 24 / 1.25 = 19.2 -> 19 is a 5-tick jump, clamped to 24 - 4 = 20.
        # Feed beyond the 5-minute adjust interval of the first change.
        change = feed(c, 50, '0.01', start_ms=NOW + 350000)
        self.assertEqual(change['new_ticks'], 20)
        self.assertEqual(c.ticks, 20)

    def test_dead_zone_leaves_spread_alone(self):
        a = sc_account()
        c = SpreadController(a)
        self.assertIsNone(feed(c, 50, '0.0005'))
        self.assertEqual(c.ticks, 6)
        # Exactly at epsilon is still the dead zone (comparison is strict).
        self.assertIsNone(feed(c, 50, '0.001', start_ms=NOW + 60000))
        self.assertEqual(c.ticks, 6)

    def test_clamped_at_hard_bounds(self):
        high = sc_account(as_min_half_spread_ticks='24')
        c = SpreadController(high)
        self.assertIsNone(feed(c, 50, '-0.01'))
        self.assertEqual(c.ticks, 24)
        self.assertEqual(high.config['as_min_half_spread_ticks'], '24')

        low = sc_account(as_min_half_spread_ticks='3')
        c = SpreadController(low)
        self.assertIsNone(feed(c, 50, '0.01'))
        self.assertEqual(c.ticks, 3)
        self.assertEqual(low.config['as_min_half_spread_ticks'], '3')

    def test_bounds_default_to_3_24_when_not_configured(self):
        cfg = sc_config()
        del cfg['sc_min_ticks']
        del cfg['sc_max_ticks']
        a = Account('sc', cfg, rules(), lambda *args: None)
        c = SpreadController(a)
        self.assertEqual(feed(c, 50, '-0.01')['new_ticks'], 9)  # 6 x 1.5, no low-bound effect
        a = Account('sc', {**cfg, 'as_min_half_spread_ticks': '20'}, rules(), lambda *args: None)
        c = SpreadController(a)
        self.assertEqual(feed(c, 50, '-0.01')['new_ticks'], 24)  # default high bound 24

    def test_adjust_interval_limits_changes(self):
        a = sc_account()
        c = SpreadController(a)
        first = feed(c, 50, '-0.01')
        self.assertIsNotNone(first)
        self.assertEqual(first['new_ticks'], 9)
        adjust_ms = c.last_adjust_ms
        # Rolling window keeps the mean adverse, but no second change within 5 min.
        for i in range(299):
            self.assertIsNone(c.feed(D('-0.01'), adjust_ms + 1000 * (i + 1)))
        self.assertEqual(c.ticks, 9)
        change = c.feed(D('-0.01'), adjust_ms + 300000)
        self.assertIsNotNone(change)
        # 9 x 1.5 = 13.5 -> 14, jump-clamped to 9 + 4 = 13.
        self.assertEqual(change['new_ticks'], 13)

    def test_window_rolls_dropping_oldest(self):
        a = sc_account(sc_adjust_interval_min=0)
        c = SpreadController(a)
        self.assertIsNotNone(feed(c, 50, '-0.01'))
        self.assertEqual(c.ticks, 9)
        # Ten mildly positive fills must drop the ten oldest negatives: the
        # mean stays adverse only because the window slid, proving the
        # eviction. Ten is also the fresh-evidence minimum (sc_min_new_fills),
        # so the no-adjust gate must not block this change either.
        change = feed(c, 10, '0.02', start_ms=NOW + 60000)
        self.assertIsNotNone(change)
        self.assertEqual(change['new_ticks'], 13)  # jump-clamped from 9 x 1.5 = 13.5 -> 14
        self.assertAlmostEqual(float(change['window_mean']), -0.004, places=9)
        self.assertEqual(len(c.window), 50)


class SpreadControllerIsolationTests(unittest.TestCase):
    def test_config_mutation_stays_on_own_account(self):
        rows = account_configs({**config(), 'accounts': [
            {'name': 'a_sc', 'type': 'as', 'spread_control_enabled': True,
             'sc_window_fills': 50, 'sc_adjust_interval_min': 5,
             'sc_min_ticks': 3, 'sc_max_ticks': 24,
             'sc_up_factor': '1.5', 'sc_down_factor': '1.25',
             'sc_max_jump_ticks': 4, 'sc_epsilon': '0.001'},
            {'name': 'b_plain', 'type': 'as'}]})
        a = Account('a_sc', rows[0][1], rules(), lambda *args: None)
        b = Account('b_plain', rows[1][1], rules(), lambda *args: None)
        c = SpreadController(a)
        feed(c, 50, '-0.01')
        self.assertEqual(a.sc_target_ticks, 9)
        self.assertNotIn('as_min_half_spread_ticks', b.config)
        self.assertNotIn('spread_control_enabled', b.config)
        self.assertEqual(c.ticks, 9)


class SpreadControllerWhitelistTests(unittest.TestCase):
    def test_sc_keys_whitelisted_and_unknown_rejected(self):
        rows = account_configs({**config(), 'accounts': [
            {'name': 's1', 'type': 'as', 'spread_control_enabled': True,
             'sc_window_fills': 50, 'sc_adjust_interval_min': 5,
             'sc_min_ticks': 3, 'sc_max_ticks': 24,
             'sc_up_factor': '1.5', 'sc_down_factor': '1.25',
             'sc_max_jump_ticks': 4, 'sc_epsilon': '0.001'}]})
        self.assertTrue(rows[0][1]['spread_control_enabled'])
        self.assertEqual(rows[0][1]['sc_window_fills'], 50)
        self.assertEqual(rows[0][1]['sc_max_ticks'], 24)
        self.assertEqual(rows[0][1]['sc_up_factor'], '1.5')
        with self.assertRaises(ValueError):
            account_configs({**config(), 'accounts': [{'name': 's1', 'sc_hack': '1'}]})




def collector():
    events = []
    return events, (lambda kind, ms, **fields: events.append((kind, fields)))


def feed_keyed(controller, n, pnl, start_ms, prefix, emit=None):
    change = None
    for i in range(n):
        result = controller.feed(D(pnl), start_ms + i * 1000, oid=f'{prefix}{i}',
                                 fill_key=f'{prefix}{i}', event=emit)
        if result is not None:
            change = result
    return change




class SpreadControllerEvidenceGateTests(unittest.TestCase):
    """F5 (i)/(ii)/(iv)/(v): the fresh-evidence gate and the fixed filter order."""

    def test_insufficient_new_evidence_blocks_adjust_regardless_of_sign(self):
        a = sc_account(sc_adjust_interval_min=0)
        c = SpreadController(a)
        events, emit = collector()
        self.assertEqual(feed_keyed(c, 50, '-0.01', NOW, 'a', emit)['new_ticks'], 9)
        # Nine fresh favorable samples: below sc_min_new_fills=10, so the
        # controller must not act even though the window mean turned positive.
        for i in range(9):
            self.assertIsNone(c.feed(D('0.05'), NOW + 60000 + i * 1000,
                                     fill_key=f'p{i}', event=emit))
        self.assertEqual(c.ticks, 9)
        self.assertEqual(c.last_eval_reason, 'insufficient_evidence')
        # The tenth fresh sample unblocks the gate; the now-positive mean
        # tightens 9 -> 7 (9 / 1.25 = 7.2).
        change = c.feed(D('0.05'), NOW + 60000 + 9000, fill_key='p9', event=emit)
        self.assertEqual(change['new_ticks'], 7)
        # Sign-independence: nine strongly ADVERSE fresh samples are equally
        # blocked — the gate counts evidence, not direction.
        for i in range(9):
            self.assertIsNone(c.feed(D('-0.5'), NOW + 120000 + i * 1000,
                                     fill_key=f'n{i}', event=emit))
        self.assertEqual(c.ticks, 7)
        self.assertEqual(c.last_eval_reason, 'insufficient_evidence')

    def test_full_evidence_with_still_negative_mean_widens(self):
        # Reaching the fresh-evidence threshold does not by itself stop
        # widening: when the eligible window mean is still below -epsilon,
        # widening IS the standing rule (F5 ii).
        a = sc_account(sc_adjust_interval_min=0)
        c = SpreadController(a)
        self.assertEqual(feed_keyed(c, 50, '-0.01', NOW, 'a')['new_ticks'], 9)
        change = feed_keyed(c, 10, '-0.01', NOW + 60000, 'b')
        self.assertIsNotNone(change)
        self.assertEqual(change['new_ticks'], 13)  # 9 x 1.5 = 13.5 -> 14, jump-clamped

    def test_age_eviction_below_full_threshold_never_acts(self):
        # Anti-V1: after age eviction the remaining count is checked against
        # the FULL window threshold, not treated as actionable evidence.
        a = sc_account(sc_adjust_interval_min=0, sc_window_max_age_min=1)
        c = SpreadController(a)
        events, emit = collector()
        self.assertEqual(feed_keyed(c, 50, '-0.01', NOW, 'a', emit)['new_ticks'], 9)
        # Jump two minutes: all 50 old samples exceed the 1-minute fill_ms age.
        # Even a strongly adverse new sample must not act on a 1-sample window.
        self.assertIsNone(c.feed(D('-1'), NOW + 120000, fill_ms=NOW + 120000,
                                 oid='x', fill_key='x', event=emit))
        self.assertEqual(c.ticks, 9)
        self.assertEqual(c.last_eval_reason, 'warming_up')
        evals = [f for k, f in events if k == 'sc_eval']
        # The deque itself evicted the 50th old sample on append; 49 of the
        # remaining ones were then aged out. Only the fresh sample is left.
        self.assertEqual(evals[-1]['eligible_count'], 1)
        self.assertEqual(evals[-1]['aged_out'], 49)
        self.assertEqual(evals[-1]['age_basis'], 'fill_ms')

    def test_dedup_and_restart_do_not_bypass_min_new(self):
        a = sc_account(sc_adjust_interval_min=0)
        c = SpreadController(a)
        events, emit = collector()
        self.assertEqual(feed_keyed(c, 50, '-0.01', NOW, 'a', emit)['new_ticks'], 9)
        seq_before = c.feedback_seq
        # Re-feeding the same fill_keys is deduplicated before any counting
        # and is silent: it does not even leave an evaluation trace.
        for i in range(50):
            self.assertIsNone(c.feed(D('0.05'), NOW + 60000 + i * 1000,
                                     fill_key=f'a{i}', event=emit))
        self.assertEqual(c.feedback_seq, seq_before)
        self.assertEqual(c.last_eval_reason, 'adjust')  # untouched by dup feeds
        self.assertEqual(c.ticks, 9)
        # A restart replaying already-persisted samples hits the same dedup:
        # the restored controller sees them in seen_fill_keys.
        for i in range(9):
            c.feed(D('0.05'), NOW + 120000 + i * 1000, fill_key=f'r{i}', event=emit)
        checkpoint = json.loads(encode(c.dump(NOW + 130000)))
        clone = SpreadController(a).load(checkpoint)
        self.assertEqual(clone.ticks, 9)
        for i in range(9):
            self.assertIsNone(clone.feed(D('0.5'), NOW + 140000 + i * 1000,
                                         fill_key=f'r{i}', event=emit))
        self.assertEqual(clone.ticks, 9)
        self.assertEqual(clone.last_eval_reason, 'insufficient_evidence')


class SpreadControllerHeartbeatTests(unittest.TestCase):
    """F5 (iii): zero-feedback runs stay observable with an unchanged regime."""

    def test_zero_feedback_two_hours_reports_state_and_rate_limited_heartbeats(self):
        a = sc_account()
        c = SpreadController(a)
        events, emit = collector()
        states = [c.observe(NOW + minute * 60000, event=emit) for minute in range(0, 121, 30)]
        self.assertEqual([s['status'] for s in states], ['warming_up'] * 5)
        self.assertTrue(all(s['feedback_rate_per_hour'] == 0 for s in states))
        self.assertTrue(all(s['oldest_sample_age_ms'] is None for s in states))
        self.assertTrue(all(s['last_feedback_ms'] is None for s in states))
        self.assertEqual(c.ticks, 6)
        self.assertEqual(a.config['as_min_half_spread_ticks'], '6')
        beats = [f for k, f in events if k == 'sc_heartbeat']
        self.assertEqual(len(beats), 3)  # t=0, 60min, 120min; nothing in between

    def test_full_window_then_feed_silence_turns_stale(self):
        a = sc_account()
        c = SpreadController(a)
        events, emit = collector()
        feed_keyed(c, 50, '-0.01', NOW, 'a', emit)
        state = c.observe(NOW + 481 * 60000, event=emit)
        self.assertEqual(state['status'], 'stale_feedback')
        self.assertIsNotNone(state['since_last_feedback_ms'])
        self.assertGreater(state['since_last_feedback_ms'], 480 * 60000)
        beats = [f for k, f in events if k == 'sc_heartbeat']
        self.assertEqual(len(beats), 1)  # only the stale observation fires
        self.assertEqual(c.ticks, 9)     # observation never adjusts

    def test_observe_never_feeds_or_adjusts(self):
        a = sc_account()
        c = SpreadController(a)
        events, emit = collector()
        c.observe(NOW + 3600000, event=emit)
        self.assertEqual(c.feedback_seq, 0)
        self.assertEqual(len(c.window), 0)
        self.assertEqual([k for k, _ in events], ['sc_heartbeat'])


class SpreadControllerRestoreTests(unittest.TestCase):
    """F5 (vi)/(vii)/(viii): restore semantics, regime writeback, commit boundary."""











    def test_feedback_rate_survives_restore_untruncated(self):
        # F1: the 60-minute feedback rate needs an independent restorable
        # source. The deque is already bounded to 60 minutes, so persist it
        # whole — reconstructing from the 50-sample window would cap the
        # reported rate at 50 after every restart.
        a = sc_account()
        c = SpreadController(a)
        events, emit = collector()
        for i in range(100):
            c.feed(D('-0.01'), NOW + i * 1000, fill_key=f'f{i}', event=emit)
        self.assertEqual(len(c.window), 50)  # decision window stays capped
        checkpoint = json.loads(encode(c.dump(NOW + 100000)))
        clone = SpreadController(a).load(checkpoint)
        state = clone.state(NOW + 100000)
        self.assertEqual(state['feedback_rate_per_hour'], 100)
        self.assertEqual(state['window_samples'], 50)
        # Legacy checkpoints without the field fall back to the window
        # reconstruction (rate capped at 50), but still load.
        legacy = {k: v for k, v in checkpoint.items() if k != 'feedback_times'}
        legacy_clone = SpreadController(a).load(legacy)
        self.assertEqual(legacy_clone.state(NOW + 100000)['feedback_rate_per_hour'], 50)


class SpreadControllerEvalEventTests(unittest.TestCase):
    """F5 (ix): every evaluation leaves a multi-basis sc_eval trace."""

    def test_sc_eval_payload_carries_all_bases(self):
        a = sc_account(sc_window_fills=4, sc_min_new_fills=2, sc_adjust_interval_min=0)
        c = SpreadController(a)
        events, emit = collector()
        # Two partial fills of one order plus two other orders, mixed notionals.
        rows = [('o1', '0.01', '10'), ('o1', '0.01', '10'), ('o2', '0.01', '100'),
                ('o3', '-0.02', '50')]
        for i, (oid, pnl, notional) in enumerate(rows):
            c.feed(D(pnl), NOW + i * 1000, notional=notional, oid=oid, fill_key=f'k{i}',
                   event=emit)
        evals = [f for k, f in events if k == 'sc_eval']
        self.assertEqual(len(evals), 4)
        fields = evals[-1]
        for key in ('decision', 'reason', 'eligible_count', 'eligible_new_count',
                    'aged_out', 'age_basis', 'mean', 'per_trade_bp_mean',
                    'notional_weighted_bp', 'median_bp', 'notional_total',
                    'distinct_orders', 'partial_fill_ratio', 'seq', 'last_adjust_seq'):
            self.assertIn(key, fields)
        self.assertEqual(fields['decision'], 'adjust')
        self.assertEqual(fields['reason'], 'adjust')
        self.assertEqual(fields['eligible_count'], 4)
        self.assertEqual(fields['eligible_new_count'], 4)
        self.assertEqual(fields['aged_out'], 0)
        self.assertEqual(fields['age_basis'], 'fill_ms')
        self.assertEqual(fields['distinct_orders'], 3)
        self.assertEqual(D(fields['notional_total']), D('170'))
        self.assertEqual(D(fields['per_trade_bp_mean']), D('4.25'))    # mean(10, 10, 1, -4)
        self.assertEqual(D(fields['median_bp']), D('5.5'))             # median(10, 10, 1, -4)
        self.assertEqual(D(fields['notional_weighted_bp']), D('0.01') / D('170') * 10000)
        self.assertEqual(D(fields['partial_fill_ratio']), D(1) / D(3))


class SpreadControllerSignalVariantTests(unittest.TestCase):
    """Spec v0.2 section 4: sc_signal_variant with a bp dead zone."""

    def v2_account(self, **overrides):
        return sc_account(sc_signal_variant='v2_median_bp', sc_deadband_bp='0.25',
                          **overrides)

    def test_default_variant_is_v1_and_explicit_v2_resolves(self):
        self.assertEqual(SpreadController(sc_account()).signal_mode, 'v1_mean_usdt')
        self.assertEqual(SpreadController(self.v2_account()).signal_mode, 'v2_median_bp')
        # Legacy alias keeps older S1 batch configs on the v2 signal.
        legacy = sc_account(sc_signal='median_bp')
        self.assertEqual(SpreadController(legacy).signal_mode, 'v2_median_bp')
        with self.assertRaises(ValueError):
            SpreadController(sc_account(sc_signal_variant='v3'))

    def test_median_signal_tightens_when_currency_mean_is_adverse(self):
        # SUI fv_t10_sc case: a few large-losing fills make the currency MEAN
        # adverse while most fills are mildly favorable in bp; the v2 median
        # sees the majority and tightens where v1 would widen.
        c = SpreadController(self.v2_account())
        events, emit = collector()
        rows = ([('-0.5', '1000')] * 24) + ([('0.025', '100')] * 26)
        change = None
        for i, (pnl, notional) in enumerate(rows):
            result = c.feed(D(pnl), NOW + i, notional=D(notional),
                            fill_key=f'k{i}', event=emit)
            if result is not None:
                change = result
        self.assertIsNotNone(change)
        self.assertLess(D(change['window_mean']), ZERO)      # USDT mean adverse
        self.assertEqual(D(change['signal']), D('2.5'))      # bp median favorable
        self.assertEqual(change['new_ticks'], 5)             # 6 / 1.25
        # The same window under v1 widens on the currency mean.
        c1 = SpreadController(sc_account())
        for i, (pnl, notional) in enumerate(rows):
            c1.feed(D(pnl), NOW + i, notional=D(notional), fill_key=f'k{i}')
        self.assertEqual(c1.ticks, 9)

    def test_bp_deadband_endpoints_are_inside(self):
        # abs(signal) <= 0.25bp never adjusts, endpoints included (spec: the
        # comparison is -deadband <= signal <= deadband).
        for pnl, notional in (('0.0025', '100'), ('-0.0025', '100')):
            c = SpreadController(self.v2_account())
            events, emit = collector()
            for i in range(50):
                c.feed(D(pnl), NOW + i, notional=D(notional), fill_key=f'k{i}',
                       event=emit)
            self.assertEqual(c.ticks, 6)
            self.assertEqual(c.last_eval_reason, 'dead_zone')
        # One hundredth of a bp outside the dead zone DOES adjust.
        c = SpreadController(self.v2_account())
        for i in range(50):
            c.feed(D('0.0026'), NOW + i, notional=D('100'), fill_key=f'k{i}')
        self.assertEqual(c.ticks, 5)

    def test_sc_eval_marks_signal_bp_and_median_flag(self):
        c = SpreadController(self.v2_account(sc_window_fills=4, sc_min_new_fills=2,
                                             sc_adjust_interval_min=0))
        events, emit = collector()
        for i in range(4):
            c.feed(D('-0.01'), NOW + i, notional=D('100'), fill_key=f'k{i}', event=emit)
        fields = [f for k, f in events if k == 'sc_eval'][-1]
        self.assertEqual(fields['signal_mode'], 'v2_median_bp')
        self.assertTrue(fields['median_bp_is_signal'])
        self.assertEqual(D(fields['signal_bp']), D('-1'))  # -0.01/100*1e4
        self.assertEqual(D(fields['signal']), D('-1'))
        # V1: median stays audit-only.
        c1 = SpreadController(sc_account(sc_window_fills=4, sc_min_new_fills=2,
                                         sc_adjust_interval_min=0))
        events1, emit1 = collector()
        for i in range(4):
            c1.feed(D('-0.01'), NOW + i, notional=D('100'), fill_key=f'k{i}', event=emit1)
        fields1 = [f for k, f in events1 if k == 'sc_eval'][-1]
        self.assertFalse(fields1['median_bp_is_signal'])
        self.assertIsNone(fields1['signal_bp'])
        self.assertEqual(fields1['signal_unit'], 'quote_currency')
        self.assertIsNotNone(fields1['median_bp'])  # audit value still recorded


class FloorCompositionTests(unittest.TestCase):
    """Spec v0.2 section 3: three independent floors + cap truncation audit."""

    def test_composite_and_truncation_reason(self):
        a = sc_account(s1_enabled=True, s1_k='2')
        a.update_volatility({'valid': True, 'sigma_ticks': D('5'), 'missing_ratio': 0}, NOW)
        self.assertEqual(a.floor_state()['as_floor_ticks'], 10)      # max(6, 10, 6)
        self.assertEqual(a.floor_state()['floor_trunc_reason'], 'none')
        a.sc_target_ticks = 30                                        # above cap 24
        state = a.floor_state()
        self.assertEqual(state['as_floor_ticks'], 24)
        self.assertEqual(state['floor_trunc_reason'], 'cap')
        self.assertEqual(state['floor_reason'], 'cap')
        self.assertTrue(state['floor_cap_hit'])
        # Exactly AT the cap is not a truncation.
        a.sc_target_ticks = 24
        state = a.floor_state()
        self.assertEqual(state['as_floor_ticks'], 24)
        self.assertEqual(state['floor_trunc_reason'], 'none')
        self.assertTrue(state['floor_cap_hit'])

    def test_sc_adjust_leaves_vol_and_static_untouched(self):
        a = sc_account(s1_enabled=True, s1_k='2')
        a.update_volatility({'valid': True, 'sigma_ticks': D('5'), 'missing_ratio': 0}, NOW)
        c = SpreadController(a)
        feed(c, 50, '-0.01')
        self.assertEqual(a.sc_target_ticks, 9)
        self.assertEqual(a.static_floor_ticks, 6)
        self.assertEqual(a.vol_floor_ticks, 10)
        self.assertEqual(a.config['as_min_half_spread_ticks'], '6')

    def test_vol_update_leaves_sc_target_untouched(self):
        a = sc_account(s1_enabled=True, s1_k='2')
        c = SpreadController(a)
        feed(c, 50, '-0.01')
        self.assertEqual(a.sc_target_ticks, 9)
        a.update_volatility({'valid': True, 'sigma_ticks': D('9'), 'missing_ratio': 0}, NOW)
        self.assertEqual(a.vol_floor_ticks, 18)
        self.assertEqual(a.sc_target_ticks, 9)  # SC state not polluted by S1


class SpreadControlNewKeysWhitelistTests(unittest.TestCase):
    def test_v02_keys_whitelisted(self):
        rows = account_configs({**config(), 'accounts': [
            {'name': 's1', 'type': 'as', 'spread_control_enabled': True,
             'sc_window_fills': 50, 'sc_adjust_interval_min': 5,
             'sc_signal_variant': 'v2_median_bp', 'sc_deadband_bp': '0.25',
             's1_enabled': True, 's1_k': '2',
             'vol_window_min': '30', 'vol_stale_ms': '5000',
             'vol_missing_ratio': '0.2', 'vol_degraded_min': '10',
             'vol_floor_min_ticks': 3, 'vol_floor_max_ticks': 24}]})
        effective = rows[0][1]
        self.assertEqual(effective['sc_signal_variant'], 'v2_median_bp')
        self.assertEqual(effective['vol_floor_max_ticks'], 24)
        self.assertEqual(effective['vol_degraded_min'], '10')
