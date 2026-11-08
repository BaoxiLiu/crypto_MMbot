"""BookState: causal rebuild of the price-level book from event-style diffs.

Data source is the CryptoHFTData orderbook parquet: one full-depth snapshot at
the head of each hour file + absolute-quantity diff rows (0 = delete level),
see data/prediction-v3/probe-p0.5/probe-report.md §1.

Levels are keyed by integer tick index round(price / tick_size). All prices in
this feed sit on a single tick grid, so integer keys make near-touch lookups
exact and deletion idempotent. The structure is feed-agnostic: any source that
can emit (side, price, absolute qty) rows can drive it.
"""
import heapq


class BookState:
    __slots__ = ("tick", "bids", "asks", "_hb", "_ha")

    def __init__(self, tick_size):
        self.tick = float(tick_size)
        self.bids = {}  # tick idx -> absolute qty (base asset)
        self.asks = {}
        self._hb = []  # max-heap of candidate bid idx (negative), lazy deletion
        self._ha = []  # min-heap of candidate ask idx, lazy deletion

    def clear(self):
        self.bids.clear()
        self.asks.clear()
        self._hb.clear()
        self._ha.clear()

    @staticmethod
    def price_to_idx(tick_size, price):
        return int(round(float(price) / tick_size))

    def apply(self, side, idx, qty):
        """Apply one diff row. qty is the absolute new quantity (0 = delete).

        Returns (old_qty, new_qty). O(log n) worst case, O(1) for levels that
        do not touch the current best.
        """
        if side == "bid":
            d, h = self.bids, self._hb
            key = -idx
        else:
            d, h = self.asks, self._ha
            key = idx
        old = d.get(idx, 0.0)
        if qty == 0.0:
            d.pop(idx, None)
        else:
            heapq.heappush(h, key)  # duplicates are fine: validated lazily
            d[idx] = qty
        return old, qty

    def load_snapshot(self, sides, idxs, qtys):
        """Full replace from the hour-head snapshot (not a cancel/add flow)."""
        self.clear()
        for side, idx, qty in zip(sides, idxs, qtys):
            if qty == 0.0:
                continue
            self.apply(side, idx, float(qty))

    def best_bid(self):
        """(tick_idx, qty) or (None, 0.0)."""
        h, d = self._hb, self.bids
        while h and -h[0] not in d:
            heapq.heappop(h)
        return (-h[0], d[-h[0]]) if h else (None, 0.0)

    def best_ask(self):
        h, d = self._ha, self.asks
        while h and h[0] not in d:
            heapq.heappop(h)
        return (h[0], d[h[0]]) if h else (None, 0.0)

    def qty_at(self, side, idx):
        return (self.bids if side == "bid" else self.asks).get(idx, 0.0)
