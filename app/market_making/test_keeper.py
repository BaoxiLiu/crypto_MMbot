import unittest

from .core import Account, Book, D, ZERO
from .core import account_configs
from .test_core import NOW, book, config, rules
from .test_matrix import matrix_config


def keeper_config(**overrides):
    base = {**config(), 'keeper_enabled': True, 'fv_buffer_ticks': '1'}
    base.update(overrides)
    return base


def book_at(best_bid, best_ask):
    b = Book()
    b.snapshot({"bids": [[str(best_bid), "5"], ["7.570", "2"]],
                "asks": [[str(best_ask), "7"]], "lastUpdateId": 10})
    b.diff({"U": 9, "u": 11, "pu": 8, "b": [], "a": []}, NOW)
    return b


class KeeperCancelTests(unittest.TestCase):
    def test_fv_cross_cancels_and_respects_buffer(self):
        events = []
        a = Account('test', keeper_config(), rules(), lambda *args: events.append(args))
        b = book()
        a.quote(D('7.580'), D('7.620'), D('7.6'), NOW, b, fv=D('7.6'))
        a.advance(NOW + 400, b)
        self.assertEqual(a.orders[1].cancel_ms, 0)
        a.quote(D('7.578'), D('7.618'), D('7.6'), NOW + 500, b, fv=D('7.578'))
        self.assertGreater(a.orders[1].cancel_ms, 0)
        cancel = [row for row in events if row[1] == 'cancel_requested']
        self.assertEqual(cancel[-1][3]['reason'], 'keeper_fv_cross')

        events.clear()
        a.quote(D('7.578'), D('7.618'), D('7.6'), NOW + 600, b, fv=D('7.5795'))
        self.assertFalse(any(row[1] == 'cancel_requested' for row in events))

    def test_mid_cross_cancels_beyond_fv_buffer_gap(self):
        events = []
        a = Account('test', keeper_config(), rules(), lambda *args: events.append(args))
        a.quote(D('7.580'), D('7.620'), D('7.6'), NOW, book())
        b = book_at('7.578', '7.580')
        a.quote(D('7.578'), D('7.620'), D('7.6'), NOW + 500, b, fv=None)
        self.assertGreater(a.orders[1].cancel_ms, 0)
        cancel = [row for row in events if row[1] == 'cancel_requested']
        self.assertEqual(cancel[-1][3]['reason'], 'keeper_mid_cross')

        events.clear()
        b = book_at('7.580', '7.582')
        a.quote(D('7.578'), D('7.620'), D('7.6'), NOW + 600, b, fv=None)
        self.assertFalse(any(row[1] == 'cancel_requested' for row in events))

    def test_fv_none_degrades_fv_cross_to_mid(self):
        events = []
        a = Account('test', keeper_config(), rules(), lambda *args: events.append(args))
        a.quote(D('7.580'), D('7.620'), D('7.6'), NOW, book())
        b = book_at('7.576', '7.578')
        a.quote(D('7.578'), D('7.620'), D('7.6'), NOW + 500, b, fv=None)
        cancel = [row for row in events if row[1] == 'cancel_requested']
        self.assertEqual(cancel[-1][3]['reason'], 'keeper_fv_cross')

    def test_keeper_suppresses_periodic_requote(self):
        events = []
        a = Account('test', keeper_config(quote_refresh_seconds=1),
                    rules(), lambda *args: events.append(args))
        b = book()
        a.quote(D('7.580'), D('7.620'), D('7.6'), NOW, b, fv=D('7.6'))
        a.quote(D('7.581'), D('7.619'), D('7.6'), NOW + 1000, b, fv=D('7.6'))
        self.assertTrue(all(o.cancel_ms == 0 for o in a.orders.values()))
        self.assertFalse(any(row[1] == 'cancel_requested' for row in events))

    def test_periodic_requote_survives_with_keeper_off(self):
        events = []
        c = {**config(), 'quote_refresh_seconds': 1}
        a = Account('test', c, rules(), lambda *args: events.append(args))
        b = book()
        a.quote(D('7.580'), D('7.620'), D('7.6'), NOW, b)
        a.quote(D('7.581'), D('7.619'), D('7.6'), NOW + 1000, b)
        self.assertTrue(all(o.cancel_ms > 0 for o in a.orders.values()))
        cancel = [row for row in events if row[1] == 'cancel_requested']
        self.assertTrue(all(row[3]['reason'] == 'periodic_requote' for row in cancel))

    def test_favorable_requote_survives_with_keeper_on(self):
        events = []
        a = Account('test', keeper_config(favorable_requote_ticks='3'),
                    rules(), lambda *args: events.append(args))
        b = book()
        a.quote(D('7.580'), D('7.620'), D('7.6'), NOW, b, fv=D('7.6'))
        a.quote(D('7.584'), D('7.616'), D('7.6'), NOW + 500, b, fv=D('7.6'))
        cancel = [row for row in events if row[1] == 'cancel_requested']
        self.assertTrue(any(row[3]['reason'] == 'favorable_requote' for row in cancel))


class KeeperNewOrderTests(unittest.TestCase):
    def test_dangerous_quote_never_placed(self):
        a = Account('test', keeper_config(), rules(), lambda *args: None)
        b = book()
        a.quote(D('7.600'), D('7.620'), D('7.6'), NOW, b, fv=D('7.599'))
        self.assertNotIn(1, a.orders)
        self.assertIn(-1, a.orders)
        self.assertEqual(a.reason, 'keeper_dangerous_quote')

    def test_dangerous_quote_symmetric_on_ask(self):
        a = Account('test', keeper_config(), rules(), lambda *args: None)
        b = book()
        a.quote(D('7.580'), D('7.620'), D('7.6'), NOW, b, fv=D('7.621'))
        self.assertIn(1, a.orders)
        self.assertNotIn(-1, a.orders)
        self.assertEqual(a.reason, 'keeper_dangerous_quote')


class KeeperOverrideKeyTests(unittest.TestCase):
    def test_keeper_keys_whitelisted_and_unknown_rejected(self):
        base = matrix_config()
        rows = account_configs({**base, "accounts": [
            {"name": "k1", "keeper_enabled": True, "fv_buffer_ticks": "1"}]})
        self.assertTrue(rows[0][1]["keeper_enabled"])
        self.assertEqual(rows[0][1]["fv_buffer_ticks"], "1")
        with self.assertRaises(ValueError):
            account_configs({**base, "accounts": [{"name": "k1", "keeper_secret": "1"}]})


class PolicyChangeSizeToleranceTests(unittest.TestCase):
    def sized_account(self, remaining):
        events = []
        a = Account('test', config(), rules(), lambda *args: events.append(args))
        b = book()
        a.quote(D('7.580'), D('7.620'), D('7.6'), NOW, b)
        a.orders[1].remaining = remaining
        return a, events

    def test_one_step_size_jitter_does_not_requote(self):
        # floor(25/7.58)=3.29 in place; a 2-tick rise makes floor(25/7.60)=3.28:
        # remaining == qty + 1 step must be tolerated (no churn requote).
        a, events = self.sized_account(D('3.29'))
        a.quote(D('7.600'), D('7.620'), D('7.6'), NOW + 1000, book())
        self.assertEqual(a.orders[1].cancel_ms, 0)
        self.assertFalse(any(row[1] == 'cancel_requested' for row in events))

    def test_real_size_reduction_still_requotes(self):
        a, events = self.sized_account(D('3.33'))
        a.quote(D('7.600'), D('7.620'), D('7.6'), NOW + 1000, book())
        self.assertGreater(a.orders[1].cancel_ms, 0)
        cancel = [row for row in events if row[1] == 'cancel_requested']
        self.assertEqual(cancel[-1][3]['reason'], 'inventory_policy_change')
