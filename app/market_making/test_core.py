import json
import unittest
from dataclasses import asdict
from pathlib import Path

from .core import Account, Book, BookGap, D, Ledger, Order, Rules, ZERO, day, encode


NOW = 1789365000000


def config():
    return json.loads((Path(__file__).parent/'fixtures/account.json').read_text())


def rules():
    return Rules(D("0.001"), D("0.01"), D("20"), D("0.01"), D("400000"), D("0.9"), D("1.1"))


def book():
    b = Book()
    b.snapshot({"bids": [["7.599", "5"], ["7.580", "2"]], "asks": [["7.601", "7"]], "lastUpdateId": 10})
    b.diff({"U": 9, "u": 11, "pu": 8, "b": [], "a": []}, NOW)
    return b


class LedgerTests(unittest.TestCase):
    def test_open_long_keeps_collateral(self):
        l = Ledger()
        l.fill(1, D("3"), D("8"))
        self.assertEqual(l.wallet, D("600"))
        self.assertEqual(l.equity(D("7")), D("597"))

    def test_close_long(self):
        l = Ledger()
        l.fill(1, D("3"), D("8"))
        l.fill(-1, D("2"), D("9"))
        self.assertEqual((l.position, l.average_entry, l.realized), (D("1"), D("8"), D("2")))
        self.assertEqual(l.equity(D("7")), D("601"))

    def test_close_short(self):
        l = Ledger()
        l.fill(-1, D("3"), D("8"))
        l.fill(1, D("3"), D("7"))
        self.assertEqual((l.position, l.average_entry, l.realized), (ZERO, ZERO, D("3")))

    def test_flip_short_to_long(self):
        l = Ledger()
        l.fill(-1, D("2"), D("8"))
        l.fill(1, D("5"), D("9"))
        self.assertEqual((l.position, l.average_entry, l.realized), (D("3"), D("9"), D("-2")))

    def test_flip_long_to_short(self):
        l = Ledger()
        l.fill(1, D("2"), D("8"))
        l.fill(-1, D("5"), D("9"))
        self.assertEqual((l.position, l.average_entry, l.realized), (D("-3"), D("9"), D("2")))

    def test_weighted_average(self):
        l = Ledger()
        l.fill(1, D("2"), D("7"))
        l.fill(1, D("1"), D("10"))
        self.assertEqual(l.average_entry, D("8"))

    def test_zero_fees_both_directions(self):
        l = Ledger()
        l.fill(1, D("3"), D("7"), ZERO)
        l.fill(-1, D("3"), D("8"), ZERO)
        self.assertEqual(l.fees, ZERO)

    def test_funding_long_pays_short_receives(self):
        a, b = Ledger(), Ledger()
        self.assertEqual(a.settle_funding(D("3"), D("8"), D(".001"), NOW), D("-.024"))
        self.assertEqual(b.settle_funding(D("-3"), D("8"), D(".001"), NOW), D(".024"))

    def test_negative_funding_reverses_payment(self):
        l = Ledger()
        self.assertEqual(l.settle_funding(D("3"), D("8"), D("-.001"), NOW), D(".024"))

    def test_equity_reconciles(self):
        l = Ledger()
        for side, qty, price in [(1, "3.17", "7.514"), (-1, "1.15", "7.513"), (-1, "4.12", "7.611"), (1, "1.19", "7.432")]:
            l.fill(side, D(qty), D(price))
        l.settle_funding(D("-2"), D("7.61"), D(".0001"), NOW)
        r = l.report(D("7.5"), config())
        self.assertAlmostEqual(r["total_pnl"], r["realized_trading_pnl"] + r["unrealized_pnl"] + r["funding_pnl"], places=18)

    def test_daily_stop_holds_position_and_persists(self):
        l = Ledger()
        l.risk(D("8"), NOW, config())
        l.fill(1, D("5"), D("8"))
        self.assertEqual(l.risk(D("6"), NOW + 1000, config()), "daily_loss_stop_hold_position")
        restored = Ledger.restore(json.loads(encode(asdict(l))))
        self.assertTrue(restored.day_stopped)
        self.assertEqual(restored.position, D("5"))
        self.assertEqual(restored.risk(D("8"), NOW + 2000, config()), "daily_loss_stop_hold_position")

    def test_next_day_resumes_without_erasing_pnl(self):
        l = Ledger()
        l.risk(D("8"), NOW, config())
        l.fill(1, D("5"), D("8"))
        l.risk(D("6"), NOW + 1000, config())
        l.risk(D("6"), NOW + 86400000, config())
        self.assertFalse(l.day_stopped)
        self.assertEqual(l.equity(D("6")), D("590"))

    def test_margin_stress_is_sticky(self):
        l = Ledger()
        l.fill(-1, D("10"), D("8"))
        self.assertEqual(l.risk(D("80"), NOW, config()), "margin_stress_breach_evaluation_halted")
        l.risk(D("8"), NOW + 86400000, config())
        self.assertTrue(l.margin_breach)


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.a = Account("fixed", config(), rules(), lambda *args: self.events.append(args))
        self.b = book()

    def test_price_and_quantity_rules(self):
        r = rules()
        self.assertEqual(r.price(D("7.5997"), 1), D("7.599"))
        self.assertEqual(r.price(D("7.5997"), -1), D("7.600"))
        self.assertFalse(r.valid(D("7.6"), D("1"), D("7.6")))
        self.assertFalse(r.valid(D("7.6001"), D("3"), D("7.6")))
        self.assertTrue(r.valid(D("7.6"), D("1"), D("7.6"), reduce_only=True))

    def test_pending_sells_do_not_release_buy_capacity(self):
        self.a.ledger.position = D("9")
        self.a.orders[-1] = Order("sell", -1, D("7.61"), D("3"), NOW, NOW)
        self.assertFalse(self.a.allowed(1, D("2"), D("7.6"), D("7.6")))

    def test_pending_buys_do_not_release_short_capacity(self):
        self.a.ledger.position = D("-9")
        self.a.orders[1] = Order("buy", 1, D("7.59"), D("3"), NOW, NOW)
        self.assertFalse(self.a.allowed(-1, D("2"), D("7.6"), D("7.6")))

    def test_pending_cancel_still_consumes_capacity(self):
        self.a.ledger.position = D("7")
        self.a.orders[1] = Order("buy", 1, D("7.59"), D("2"), NOW, NOW, cancel_ms=NOW + 300)
        self.assertFalse(self.a.allowed(1, D("2"), D("7.6"), D("7.6")))

    def test_notional_limit_at_higher_price(self):
        self.assertFalse(self.a.allowed(1, D("9"), D("10"), D("10")))

    def test_no_fill_before_activation(self):
        self.a.orders[1] = Order("buy", 1, D("7.580"), D("3"), NOW, NOW + 300)
        self.a.trade(-1, D("7.57"), D("10"), NOW + 100, NOW + 100, self.b)
        self.assertEqual(self.a.ledger.position, ZERO)

    def test_delayed_old_trade_cannot_fill_new_order(self):
        self.a.orders[1] = Order("buy", 1, D("7.580"), D("3"), NOW, NOW + 300)
        self.a.trade(-1, D("7.57"), D("10"), NOW + 100, NOW + 500, self.b)
        self.assertEqual(self.a.ledger.position, ZERO)

    def test_queue_volume_then_partial_fill(self):
        self.a.orders[1] = Order("buy", 1, D("7.580"), D("3"), NOW, NOW + 300)
        self.a.trade(-1, D("7.580"), D("1"), NOW + 400, NOW + 400, self.b)
        self.assertEqual(self.a.ledger.position, ZERO)
        self.a.trade(-1, D("7.580"), D("1.50"), NOW + 500, NOW + 500, self.b)
        self.assertEqual(self.a.ledger.position, D(".50"))
        self.assertEqual(self.a.orders[1].remaining, D("2.50"))

    def test_fill_during_cancel_latency(self):
        self.a.orders[1] = Order("buy", 1, D("7.580"), D("3"), NOW - 1000, NOW - 700, active=True)
        self.a.cancel(1, NOW)
        self.a.trade(-1, D("7.57"), D("1"), NOW + 100, NOW + 100, self.b)
        self.assertEqual(self.a.ledger.position, D("1"))
        self.a.trade(-1, D("7.57"), D("5"), NOW + 400, NOW + 400, self.b)
        self.assertEqual(self.a.ledger.position, D("1"))
        self.assertNotIn(1, self.a.orders)

    def test_reduce_only_never_flips_position(self):
        self.a.ledger.fill(1, D("1"), D("7.5"))
        self.a.orders[-1] = Order("sell", -1, D("7.61"), D("3"), NOW - 1000, NOW - 700, reduce_only=True, active=True)
        self.a.trade(1, D("7.62"), D("5"), NOW, NOW, self.b)
        self.assertEqual(self.a.ledger.position, ZERO)

    def test_crossing_post_only_rejected(self):
        self.a.orders[1] = Order("buy", 1, D("7.61"), D("3"), NOW - 1000, NOW - 700)
        self.a.advance(NOW, self.b)
        self.assertNotIn(1, self.a.orders)
        self.assertEqual(self.events[-1][1], "post_only_rejected")

    def test_unchanged_quote_keeps_queue(self):
        self.a.quote(D("7.58"), D("7.62"), D("7.6"), NOW, self.b)
        self.a.advance(NOW + 400, self.b)
        oid = self.a.orders[1].oid
        self.a.quote(D("7.58"), D("7.62"), D("7.6"), NOW + 10000, self.b)
        self.assertEqual(self.a.orders[1].oid, oid)
        self.assertEqual(self.a.orders[1].cancel_ms, 0)

    def test_replacement_does_not_wait_twice_after_cancel(self):
        self.a.quote(D("7.58"), D("7.62"), D("7.6"), NOW, self.b)
        self.a.advance(NOW + 400, self.b)
        old_id = self.a.orders[1].oid
        self.a.quote(D("7.579"), D("7.62"), D("7.6"), NOW + 5000, self.b)
        self.assertGreater(self.a.orders[1].cancel_ms, 0)
        self.a.advance(NOW + 5300, self.b)
        self.a.quote(D("7.579"), D("7.62"), D("7.6"), NOW + 5300, self.b)
        self.assertIn(1, self.a.orders)
        self.assertNotEqual(self.a.orders[1].oid, old_id)
        self.assertEqual(self.a.orders[1].price, D("7.579"))

    def test_drop_pauses_new_long(self):
        self.a.quote(D("7.58"), D("7.62"), D("7.6"), NOW, self.b, guard_return=D("-.02"))
        self.assertNotIn(1, self.a.orders)
        self.assertIn(-1, self.a.orders)

    def test_rally_pauses_new_short(self):
        self.a.quote(D("7.58"), D("7.62"), D("7.6"), NOW, self.b, guard_return=D(".02"))
        self.assertIn(1, self.a.orders)
        self.assertNotIn(-1, self.a.orders)

    def test_pause_allows_closing_short(self):
        self.a.ledger.fill(-1, D("1"), D("7.6"))
        self.a.quote(D("7.58"), D("7.62"), D("7.6"), NOW, self.b, guard_return=D("-.02"))
        self.assertTrue(self.a.orders[1].reduce_only)
        self.assertEqual(self.a.orders[1].remaining, D("1"))

    def test_restart_keeps_funds_and_stop(self):
        self.a.ledger.fill(-1, D("2"), D("7.6"))
        self.a.ledger.day_stopped = True
        state = json.loads(encode(self.a.dump()))
        other = Account("fixed", config(), rules(), lambda *args: None)
        other.restore(state, NOW)
        self.assertEqual(other.ledger.position, D("-2"))
        self.assertTrue(other.ledger.day_stopped)


class DataTests(unittest.TestCase):
    def test_sequence_gap(self):
        b = book()
        with self.assertRaises(BookGap):
            b.diff({"U": 14, "u": 15, "pu": 13, "b": [], "a": []}, NOW)
        self.assertFalse(b.ready)

    def test_missing_snapshot_bridge(self):
        b = Book()
        b.snapshot({"bids": [["7", "1"]], "asks": [["8", "1"]], "lastUpdateId": 10})
        with self.assertRaises(BookGap):
            b.diff({"U": 12, "u": 15, "pu": 11, "b": [], "a": []}, NOW)

    def test_duplicate_diff_ignored(self):
        b = book()
        self.assertFalse(b.diff({"U": 9, "u": 11, "pu": 8, "b": [], "a": []}, NOW))



if __name__ == "__main__":
    unittest.main()
