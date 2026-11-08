"""Detached market inputs: aggressor side +1 buy/-1 sell, local receipt ns.

QuoteDepth covers the NH action support (0..6 ticks behind each best quote).
Out-of-support queries fail explicitly; they cannot silently invent empty depth.
"""
import numpy as np
from numba import float64, int64
from numba.experimental import jitclass

TRADE_DTYPE = np.dtype([('local_ns', 'i8'), ('side', 'i8'),
                       ('price', 'f8'), ('qty', 'f8')])

@jitclass([('best_bid', float64), ('best_ask', float64),
           ('best_bid_tick', int64), ('best_ask_tick', int64),
           ('bids', float64[:]), ('asks', float64[:])])
class QuoteDepth:
    def __init__(self, bid, ask, bid_tick, ask_tick, bids, asks):
        self.best_bid, self.best_ask = bid, ask
        self.best_bid_tick, self.best_ask_tick = bid_tick, ask_tick
        self.bids, self.asks = bids, asks

    def bid_qty_at_tick(self, tick):
        offset = self.best_bid_tick - tick
        if offset < 0 or offset >= len(self.bids):
            raise ValueError('Bid depth outside declared action support')
        return self.bids[offset]

    def ask_qty_at_tick(self, tick):
        offset = tick - self.best_ask_tick
        if offset < 0 or offset >= len(self.asks):
            raise ValueError('Ask depth outside declared action support')
        return self.asks[offset]
