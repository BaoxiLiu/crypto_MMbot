"""A source-neutral Decimal L2 book. Sequence failures invalidate, never patch gaps."""
from dataclasses import dataclass
from decimal import Decimal as D


class BookGap(ValueError):
    pass


@dataclass(frozen=True)
class BookView:
    ns: int
    update_id: int
    bids: tuple
    asks: tuple

    @property
    def mid(self):
        return (self.bids[0][0] + self.asks[0][0]) / 2


class LocalBook:
    def __init__(self, rules):
        self.rules = rules
        self.bids, self.asks = {}, {}
        self.update_id = -1
        self.received_ns = self.source_ns = -1
        self.ready = self.bridged = False

    def invalidate(self):
        self.ready = self.bridged = False
        self.update_id = -1
        self.bids.clear()
        self.asks.clear()

    def _levels(self, rows):
        result = {}
        for p, q in rows:
            p, q = D(str(p)), D(str(q))
            if (not p.is_finite() or not q.is_finite() or p <= 0 or q < 0
                    or p % self.rules.tick or q % self.rules.step or p in result):
                raise BookGap('Invalid/off-grid/duplicate price level')
            result[p] = q
        return result

    @staticmethod
    def _check(bids, asks):
        if not bids or not asks or max(bids) >= min(asks):
            raise BookGap('Empty or crossed book')

    def _clock(self, ns):
        if type(ns) is not int or ns < 0 or ns < self.received_ns:
            raise BookGap('Reversed or invalid receipt clock')

    def snapshot(self, data, ns):
        try:
            self._clock(ns)
            uid = data['lastUpdateId']
            if type(uid) is not int or uid < 0:
                raise BookGap('Invalid snapshot ID')
            bids = {p: q for p, q in self._levels(data['bids']).items() if q}
            asks = {p: q for p, q in self._levels(data['asks']).items() if q}
            self._check(bids, asks)
            self.bids, self.asks = bids, asks
            self.update_id, self.received_ns = uid, ns
            self.source_ns = ns
            self.ready = self.bridged = False
        except (KeyError, ValueError, ArithmeticError, TypeError):
            self.invalidate()
            raise

    def update(self, event, ns):
        try:
            self._clock(ns)
            first, last, previous = event['U'], event['u'], event['pu']
            if any(type(x) is not int for x in (first, last, previous)) or first < 0 or last < first:
                raise BookGap('Invalid update IDs')
            if self.update_id < 0:
                raise BookGap('Snapshot required')
            if last < self.update_id or (self.bridged and last == self.update_id):
                return False
            if ((not self.bridged and not first <= self.update_id <= last)
                    or (self.bridged and previous != self.update_id)):
                raise BookGap('Snapshot bridge or pu sequence gap')
            # Apply the complete message atomically; consumers never see half a packet.
            bids, asks = self.bids.copy(), self.asks.copy()
            for target, key in ((bids, 'b'), (asks, 'a')):
                for price, qty in self._levels(event[key]).items():
                    if qty:
                        target[price] = qty
                    else:
                        target.pop(price, None)
            self._check(bids, asks)
            self.bids, self.asks = bids, asks
            self.update_id, self.received_ns = last, ns
            self.source_ns = ns
            self.ready = self.bridged = True
            return True
        except (KeyError, ValueError, ArithmeticError, TypeError):
            self.invalidate()
            raise

    def view(self, levels=5):
        if not self.ready:
            raise BookGap('Book not bridged')
        if type(levels) is not int or levels < 1:
            raise ValueError('Positive level count required')
        bids = tuple(sorted(self.bids.items(), reverse=True)[:levels])
        asks = tuple(sorted(self.asks.items())[:levels])
        return BookView(self.source_ns, self.update_id, bids, asks)

    def quantity(self, side, price):
        if side not in (-1, 1):
            raise ValueError('Invalid side')
        return (self.bids if side == 1 else self.asks).get(price, D(0))
