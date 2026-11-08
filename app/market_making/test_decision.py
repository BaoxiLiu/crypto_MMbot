import unittest
from decimal import Decimal as D

from .core import D, Rules, ZERO
from .policy import as_quotes, fixed_quotes, fv_shift, microprice, microprice_from_book


def rules():
    return Rules(D("0.001"), D("0.01"), D("20"), D("0.01"), D("400000"), D("0.9"), D("1.1"))


def cfg(**over):
    base = {"fixed_half_spread": "0.002", "fixed_inventory_shift_ticks": "1",
            "max_position_qty": "100"}
    base.update(over)
    return base


class FakeModel:
    def __init__(self, bid, ask):
        self._q = (D(bid), D(ask))

    def quote(self, book, position, now):
        return self._q


class TestMicroprice(unittest.TestCase):
    def test_weighted_mid(self):
        # heavy bid size pulls microprice toward the ask
        m = microprice(D("100"), D("90"), D("101"), D("10"))
        self.assertEqual(m, (D("101") * D("90") + D("100") * D("10")) / D("100"))
        self.assertGreater(m, D("100.5"))

    def test_invalid_inputs(self):
        self.assertIsNone(microprice(ZERO, D("1"), D("1"), D("1")))
        self.assertIsNone(microprice(D("1"), ZERO, D("1"), D("1")))

    def test_from_book(self):
        from .core import Book
        book = Book()
        self.assertIsNone(microprice_from_book(book))
        book.bids = {D("100"): D("10")}
        book.asks = {D("101"): D("10")}
        book.ready = True
        self.assertEqual(microprice_from_book(book), D("100.5"))


class TestFvShift(unittest.TestCase):
    def test_clamped(self):
        r = rules()
        self.assertEqual(fv_shift(D("100.004"), D("100"), r.tick, "6"), D("0.004"))
        self.assertEqual(fv_shift(D("100.100"), D("100"), r.tick, "6"), D("0.006"))
        self.assertEqual(fv_shift(None, D("100"), r.tick, "6"), ZERO)


class TestFixedQuotes(unittest.TestCase):
    def test_symmetric_around_mid(self):
        bid, ask = fixed_quotes(D("100"), ZERO, cfg(), rules())
        self.assertEqual(bid, D("99.8"))
        self.assertEqual(ask, D("100.2"))

    def test_inventory_skews_toward_reduction(self):
        # long position pushes both quotes down (sell easier, buy harder)
        bid0, ask0 = fixed_quotes(D("100"), ZERO, cfg(), rules())
        bid1, ask1 = fixed_quotes(D("100"), D("50"), cfg(), rules())
        self.assertLess(bid1, bid0)
        self.assertLess(ask1, ask0)

    def test_inventory_clamped_at_cap(self):
        bid, ask = fixed_quotes(D("100"), D("500"), cfg(), rules())
        bid_cap, ask_cap = fixed_quotes(D("100"), D("100"), cfg(), rules())
        self.assertEqual(bid, bid_cap)
        self.assertEqual(ask, ask_cap)


class TestAsQuotes(unittest.TestCase):
    def test_no_fair_value_zero_shift(self):
        from .core import Book
        book = Book()
        book.bids = {D("100"): D("10")}
        book.asks = {D("100.1"): D("10")}
        book.ready = True
        model = FakeModel("99.9", "100.2")
        bid, ask, nb, na, shift = as_quotes(model, book, ZERO, 1000, cfg(), rules())
        self.assertEqual(shift, ZERO)
        self.assertEqual((nb, na), (D("99.9"), D("100.2")))

    def test_fair_value_micro_applies_clamped_shift(self):
        from .core import Book
        book = Book()
        book.bids = {D("100"): D("90")}
        book.asks = {D("100.1"): D("10")}
        book.ready = True
        model = FakeModel("99.9", "100.2")
        c = cfg(fair_value="micro", as_min_half_spread_ticks="4")
        bid, ask, nb, na, shift = as_quotes(model, book, ZERO, 1000, c, rules(), micro=D("100.09"))
        self.assertEqual(shift, D("0.004"))  # 0.09 clamped to 4 ticks
        self.assertEqual(bid, nb + shift)

    def test_model_error_propagates(self):
        from .core import Book
        book = Book()
        book.bids = {D("100"): D("10")}
        book.asks = {D("100.1"): D("10")}
        book.ready = True

        class Broken:
            def quote(self, book, position, now):
                raise ValueError("AS estimates unavailable")
        with self.assertRaises(ValueError):
            as_quotes(Broken(), book, ZERO, 0, cfg(), rules())


if __name__ == "__main__":
    unittest.main()
