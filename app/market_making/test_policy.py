import unittest

from .core import Account, D
from .quote_policy import apply_as_policy
from .test_core import NOW, book, config, rules


class PolicyTests(unittest.TestCase):
    @staticmethod
    def inventory_config():
        return {**config(), 'as_min_half_spread_ticks': '6',
                'inventory_control_enabled': True,
                'inventory_resume_qty': '2', 'inventory_soft_qty': '3',
                'inventory_hard_qty': '6', 'inventory_soft_order_notional': '21',
                'inventory_unwind_after_seconds': 90,
                'inventory_unwind_improve_ticks': '1'}

    def test_defaults_preserve_native_quote(self):
        self.assertEqual(apply_as_policy(D('7.598'), D('7.602'), D('7.6'), D('.001'), {}),
                         (D('7.598'), D('7.602')))

    def test_floor_retains_inventory_center(self):
        bid, ask = apply_as_policy(D('7.597'), D('7.601'), D('7.6'), D('.001'),
                                  {'as_min_half_spread_ticks': '4', 'as_inventory_skew_multiplier': '2'})
        self.assertEqual((bid, ask), (D('7.594'), D('7.602')))

    def test_negative_policy_denied(self):
        with self.assertRaises(ValueError):
            apply_as_policy(D('7.59'), D('7.61'), D('7.6'), D('.001'), {'as_min_half_spread_ticks': '-1'})

    def test_adverse_quote_cancel_bypasses_refresh_not_cancel_latency(self):
        c = {**config(), 'adverse_requote_ticks': '2'}
        a = Account('test', c, rules(), lambda *args: None)
        b = book()
        a.quote(D('7.58'), D('7.62'), D('7.6'), NOW, b)
        a.advance(NOW + 400, b)
        a.quote(D('7.578'), D('7.621'), D('7.6'), NOW + 500, b)
        self.assertEqual(a.orders[1].cancel_ms, NOW + 800)
        self.assertEqual(a.orders[-1].cancel_ms, 0)
        a.trade(-1, D('7.57'), D('1'), NOW+600, NOW+600, b)
        self.assertEqual(a.ledger.position, D('1'))

    def test_improving_old_order_keeps_queue_until_regular_refresh(self):
        a = Account('test', {**config(), 'adverse_requote_ticks': '2'}, rules(), lambda *args: None)
        b = book()
        a.quote(D('7.58'), D('7.62'), D('7.6'), NOW, b)
        a.advance(NOW + 400, b)
        a.quote(D('7.583'), D('7.617'), D('7.6'), NOW + 500, b)
        self.assertTrue(all(o.cancel_ms == 0 for o in a.orders.values()))

    def test_event_policy_requotes_asymmetrically_without_time_refresh(self):
        c = {**config(), 'quote_refresh_seconds': 0, 'adverse_requote_ticks': '1',
             'favorable_requote_ticks': '3'}
        a = Account('test', c, rules(), lambda *args: None)
        b = book()
        a.quote(D('7.580'), D('7.620'), D('7.6'), NOW, b)
        a.advance(NOW + 400, b)
        a.quote(D('7.579'), D('7.618'), D('7.6'), NOW + 500, b)
        self.assertGreater(a.orders[1].cancel_ms, 0)
        self.assertEqual(a.orders[-1].cancel_ms, 0)
        a.quote(D('7.579'), D('7.617'), D('7.6'), NOW + 600, b)
        self.assertGreater(a.orders[-1].cancel_ms, 0)

    def test_new_order_rate_limit_suppresses_excess_actions(self):
        events = []
        c = {**config(), 'quote_refresh_seconds': 0, 'max_new_orders_10s': 2,
             'max_new_orders_60s': 2}
        a = Account('test', c, rules(), lambda *args: events.append(args))
        b = book()
        a.quote(D('7.580'), D('7.620'), D('7.6'), NOW, b)
        self.assertEqual(len(a.orders), 2)
        a.orders.clear()
        a.quote(D('7.579'), D('7.621'), D('7.6'), NOW + 100, b)
        self.assertEqual(len(a.orders), 0)
        self.assertTrue(any(row[1] == 'action_rate_limited' for row in events))

    def test_unlimited_control_still_reports_action_usage(self):
        a = Account('test', config(), rules(), lambda *args: None)
        b = book()
        a.quote(D('7.580'), D('7.620'), D('7.6'), NOW, b)
        self.assertEqual(a.action_usage(NOW)['new_order_60s'], 2)
        a.cancel_all(NOW + 100)
        self.assertEqual(a.action_usage(NOW + 100)['cancel_60s'], 2)

    def test_soft_inventory_reduces_increasing_order_and_labels_exit(self):
        a = Account('test', self.inventory_config(), rules(), lambda *args: None)
        a.ledger.fill(1, D('3'), D('7.6'))
        a.quote(D('7.580'), D('7.620'), D('7.6'), NOW, book())
        self.assertEqual(a.inventory_mode, 'reduce')
        self.assertEqual(a.orders[1].remaining, D('3'))
        self.assertEqual(a.orders[1].purpose, 'inventory_limited')
        self.assertTrue(a.orders[-1].reduce_only)
        self.assertEqual(a.orders[-1].purpose, 'inventory_reduce')

    def test_pending_inventory_cannot_cross_hard_threshold(self):
        a = Account('test', self.inventory_config(), rules(), lambda *args: None)
        a.ledger.fill(1, D('2.9'), D('7.6'))
        a.quote(D('7.580'), D('7.620'), D('7.6'), NOW, book())
        self.assertLessEqual(a.ledger.position + a.orders[1].remaining, D('6'))

        b = Account('test', self.inventory_config(), rules(), lambda *args: None)
        b.ledger.fill(1, D('3.47'), D('7.6'))
        b.quote(D('7.580'), D('7.620'), D('7.6'), NOW, book())
        self.assertNotIn(1, b.orders)

    def test_hard_inventory_only_quotes_reducing_side(self):
        events = []
        a = Account('test', self.inventory_config(), rules(), lambda *args: events.append(args))
        a.ledger.fill(1, D('6'), D('7.6'))
        a.quote(D('7.580'), D('7.620'), D('7.6'), NOW, book())
        self.assertEqual(a.inventory_mode, 'reduce_only')
        self.assertNotIn(1, a.orders)
        self.assertTrue(a.orders[-1].reduce_only)
        self.assertTrue(any(row[1] == 'inventory_mode_changed' for row in events))

    def test_timed_unwind_improves_one_tick_without_crossing(self):
        a = Account('test', self.inventory_config(), rules(), lambda *args: None)
        a.ledger.fill(1, D('6'), D('7.6'))
        b = book()
        a.quote(D('7.580'), D('7.620'), D('7.6'), NOW, b)
        a.orders.clear()
        a.quote(D('7.580'), D('7.620'), D('7.6'), NOW + 90000, b)
        self.assertEqual(a.orders[-1].price, D('7.619'))
        self.assertTrue(a.orders[-1].reduce_only)
        self.assertEqual(a.orders[-1].purpose, 'inventory_unwind')
        self.assertGreater(a.orders[-1].price, b.bid)

    def test_inventory_hysteresis_requires_resume_threshold(self):
        a = Account('test', self.inventory_config(), rules(), lambda *args: None)
        a.ledger.position = D('6')
        self.assertEqual(a.update_inventory_mode(NOW), 'reduce_only')
        a.ledger.position = D('2.5')
        self.assertEqual(a.update_inventory_mode(NOW + 1), 'reduce')
        a.ledger.position = D('1.99')
        self.assertEqual(a.update_inventory_mode(NOW + 2), 'normal')


if __name__ == '__main__':
    unittest.main()
