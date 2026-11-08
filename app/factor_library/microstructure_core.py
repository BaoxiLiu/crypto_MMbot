"""Causal L2 reconstruction at original order decisions, no label access.

Uses only LOCAL_EVENT rows. Atomic timestamp batches, net depth-flow proxies,
and exact trailing time windows; private fills never modify the public book.
"""
import math
import numpy as np
from numba import int64, float64, njit, types
from numba.experimental import jitclass
from numba.typed import Dict

NS = 1_000_000_000
INVALID_ASK = 2**62
WINDOWS_MS = (100, 1000, 5000, 30000)
STATIC_NAMES = ['side', 'spread_bp', 'spread_ticks', 'quote_distance_ticks', 'quote_distance_bp',
                'microprice_signed_bp', 'log_touch_total_notional',
                *[f'log_{s}_notional_l{k}' for k in (1, 5, 10) for s in ('own', 'opposite')],
                *[f'signed_imbalance_l{k}' for k in (1, 5, 10)],
                'own_touch_share_l10', 'opposite_touch_share_l10',
                'own_l10_width_ticks', 'opposite_l10_width_ticks']
HISTORY_NAMES = ['log_trade_count_rate', 'log_trade_notional_rate', 'signed_trade_notional_imbalance',
                 'log_own_aggressor_rate', 'log_opposite_aggressor_rate', 'log_depth_batch_rate',
                 'log_depth_row_rate', 'signed_ofi_scaled', 'realized_vol_bp', 'signed_return_bp',
                 *[f'log_{s}_{action}_ticks{band}_rate'
                   for band in ('0_1', '2_4', '5_9') for action in ('add', 'decrease') for s in ('own', 'opposite')],
                 'log_own_refill_proxy_rate', 'log_opposite_refill_proxy_rate',
                 'own_refill_proxy_mean_ms', 'opposite_refill_proxy_mean_ms',
                 'log_own_refill_proxy_notional_rate', 'log_opposite_refill_proxy_notional_rate',
                 'signed_touch_qty_change']
FEATURE_NAMES = STATIC_NAMES + [f'w{w}ms_{n}' for w in WINDOWS_MS for n in HISTORY_NAMES]


@jitclass([
    ('tick', float64), ('limit', int64), ('capacity', int64),
    ('bid', float64[:]), ('ask', float64[:]), ('bid_decrease', int64[:]), ('ask_decrease', int64[:]),
    ('far_bid', types.DictType(int64,float64)), ('far_ask', types.DictType(int64,float64)),
    ('far_bid_decrease', types.DictType(int64,int64)), ('far_ask_decrease', types.DictType(int64,int64)),
    ('best_bid', int64), ('best_ask', int64), ('clock', int64), ('book_ns', int64),
    ('epoch_ns', int64), ('written', int64), ('previous_mid', float64),
    ('cum', float64[:]), ('times', int64[:]), ('history', float64[:, :]),
    ('history_mid', float64[:]), ('history_bq', float64[:]), ('history_aq', float64[:]),
])
class FeatureState:
    def __init__(self, tick, limit=2000000, capacity=262144):
        self.tick = tick
        self.limit = limit
        self.capacity = capacity
        self.bid = np.zeros(limit)
        self.ask = np.zeros(limit)
        self.bid_decrease = np.zeros(limit, dtype=np.int64)
        self.ask_decrease = np.zeros(limit, dtype=np.int64)
        self.far_bid = Dict.empty(int64,float64)
        self.far_ask = Dict.empty(int64,float64)
        self.far_bid_decrease = Dict.empty(int64,int64)
        self.far_ask_decrease = Dict.empty(int64,int64)
        self.best_bid, self.best_ask = -1, INVALID_ASK
        self.clock = self.book_ns = self.epoch_ns = -1
        self.written = 0
        self.previous_mid = 0.
        # counts, notional, OFI, squared returns, level rows; 3 bands x4;
        # then refill counts/lags/notionals for bids and asks.
        self.cum = np.zeros(26)
        self.times = np.zeros(capacity, dtype=np.int64)
        self.history = np.zeros((capacity, 26))
        self.history_mid = np.zeros(capacity)
        self.history_bq = np.zeros(capacity)
        self.history_aq = np.zeros(capacity)

    def valid(self):
        return 0 < self.best_bid < self.best_ask < INVALID_ASK

    def bid_qty(self, index):
        return self.bid[index] if index < self.limit else self.far_bid.get(index,0.)

    def ask_qty(self, index):
        return self.ask[index] if index < self.limit else self.far_ask.get(index,0.)

    def below(self, start):
        if start >= self.limit:
            found = -1
            for price in self.far_bid:
                if found < price < start:
                    found = price
            if found > 0:
                return found
        j = min(start-1,self.limit-1)
        while j > 0 and self.bid[j] == 0:
            j -= 1
        return j if j > 0 else -1

    def above(self, start):
        j = max(1,start+1)
        while j < self.limit and self.ask[j] == 0:
            j += 1
        if j < self.limit:
            return j
        found = INVALID_ASK
        for price in self.far_ask:
            if start < price < found:
                found = price
        return found

    def past(self, target):
        lo, hi = max(0, self.written-self.capacity), self.written
        while lo < hi:
            mid = (lo+hi)//2
            if self.times[mid % self.capacity] <= target:
                lo = mid+1
            else:
                hi = mid
        return (lo-1) % self.capacity if lo > max(0, self.written-self.capacity) else -1

    def put(self, side, index, qty):
        # Same best-price crossing repair as native HashMapMarketDepth.
        if side == 1:
            if index < self.limit:
                self.bid[index] = qty
            elif qty > 0:
                self.far_bid[index] = qty
            elif index in self.far_bid:
                del self.far_bid[index]
            if qty == 0 and index == self.best_bid:
                self.best_bid = self.below(index)
            elif qty > 0 and index > self.best_bid:
                self.best_bid = index
                if self.best_bid >= self.best_ask:
                    self.best_ask = self.above(self.best_bid)
        else:
            if index < self.limit:
                self.ask[index] = qty
            elif qty > 0:
                self.far_ask[index] = qty
            elif index in self.far_ask:
                del self.far_ask[index]
            if qty == 0 and index == self.best_ask:
                self.best_ask = self.above(index)
            elif qty > 0 and index < self.best_ask:
                self.best_ask = index
                if self.best_bid >= self.best_ask:
                    self.best_bid = self.below(self.best_ask)

    def ingest(self, data, begin, end):
        t = data[begin]['local_ts']
        if t <= self.clock:
            raise ValueError('Timestamp batch repeated or reordered across shards')
        pb, pa = self.best_bid, self.best_ask
        was_valid = self.valid()
        pbq = self.bid_qty(pb) if pb > 0 else 0.
        paq = self.ask_qty(pa) if pa < INVALID_ASK else 0.
        has_depth = False
        for i in range(begin, end):
            row = data[i]
            ev = int(row['ev'])
            if ev & (1 << 30) == 0:
                raise ValueError('Non-local row supplied to feature engine')
            kind = ev & 255
            side = 1 if ev & (1 << 29) else -1
            px, qty = row['px'], row['qty']
            if kind == 2:
                if qty <= 0 or px <= 0:
                    raise ValueError('Invalid public trade')
                j = 1 if side == 1 else 2
                self.cum[j] += 1
                self.cum[j+2] += px*qty
            elif kind == 1 or kind == 4:
                index = int(round(px/self.tick))
                if index <= 0 or index >= INVALID_ASK or qty < 0:
                    raise ValueError('Invalid level')
                has_depth = True
                if kind == 1 and was_valid:
                    old = self.bid_qty(index) if side == 1 else self.ask_qty(index)
                    delta = qty-old
                    touch = pb if side == 1 else pa
                    distance = abs(index-touch)
                    if distance <= 9:
                        band = 0 if distance <= 1 else (1 if distance <= 4 else 2)
                        # band order: bid add, ask add, bid decrease, ask decrease.
                        slot = 8+band*4+(0 if side == 1 else 1)+(2 if delta < 0 else 0)
                        self.cum[slot] += abs(delta)*px
                    decrease = self.bid_decrease if side == 1 else self.ask_decrease
                    far_decrease = self.far_bid_decrease if side == 1 else self.far_ask_decrease
                    last_decrease = decrease[index] if index < self.limit else far_decrease.get(index,0)
                    if delta < 0:
                        if index < self.limit:
                            decrease[index] = t
                        else:
                            far_decrease[index] = t
                    elif delta > 0 and last_decrease > 0:
                        age = t-last_decrease
                        if distance <= 9 and 0 < age <= 5*NS:
                            j = 20 if side == 1 else 21
                            self.cum[j] += 1
                            self.cum[j+2] += age/1e6
                            self.cum[j+4] += delta*px
                        if index < self.limit:
                            decrease[index] = 0
                        elif index in far_decrease:
                            del far_decrease[index]
                    self.cum[7] += 1
                self.put(side, index, qty)
            else:
                raise ValueError('Unsupported depth clear/reset event; require new segment')
        self.clock = t
        if has_depth:
            self.book_ns = t
            self.cum[0] += 1
        if not self.valid():
            self.written = 0
            self.cum[:] = 0
            self.epoch_ns = -1
            self.previous_mid = 0.
            self.bid_decrease[:] = 0
            self.ask_decrease[:] = 0
            self.far_bid_decrease.clear()
            self.far_ask_decrease.clear()
            return
        mid = (self.best_bid+self.best_ask)*self.tick/2
        if self.epoch_ns < 0:
            self.epoch_ns = t
            self.cum[:] = 0
        if self.previous_mid > 0:
            self.cum[6] += (math.log(mid/self.previous_mid)*1e4)**2
        self.previous_mid = mid
        if has_depth and was_valid:
            b, a = self.best_bid, self.best_ask
            self.cum[5] += ((self.bid_qty(b) if b >= pb else 0.)-(pbq if b <= pb else 0.)
                           -(self.ask_qty(a) if a <= pa else 0.)+(paq if a >= pa else 0.))
        slot = self.written % self.capacity
        if self.written >= self.capacity and self.times[slot] >= t-31*NS:
            raise ValueError('Feature history capacity exceeded; never silently truncate')
        self.times[slot] = t
        self.history[slot] = self.cum
        self.history_mid[slot] = mid
        self.history_bq[slot] = self.bid_qty(self.best_bid)
        self.history_aq[slot] = self.ask_qty(self.best_ask)
        self.written += 1

    def query(self, now, side, price, qty):
        out = np.full(20+4*29, np.nan)
        if not self.valid() or self.book_ns < 0 or now < self.clock or now-self.book_ns > NS:
            return out
        b, a = self.best_bid, self.best_ask
        mid = (a+b)*self.tick/2
        bq, aq = self.bid_qty(b), self.ask_qty(a)
        spread = (a-b)*self.tick
        out[0] = side
        out[1] = spread/mid*1e4
        out[2] = a-b
        own_touch = b if side == 1 else a
        out[3] = side*(own_touch-price/self.tick)
        out[4] = side*(mid-price)/mid*1e4
        micro = (a*self.tick*bq+b*self.tick*aq)/(bq+aq)
        out[5] = side*(micro-mid)/mid*1e4
        # Actual command quantity may be clipped by private responses during
        # the compute budget. It is metadata, never a decision-time feature.
        out[6] = math.log1p((bq+aq)*mid)
        bd, ad = np.zeros(10), np.zeros(10)
        bi, ai, nb, na = b, a, 0, 0
        last_bid, last_ask = b, a
        while bi > 0 and nb < 10:
            bd[nb] = self.bid_qty(bi)*bi*self.tick
            nb += 1
            last_bid=bi
            if nb < 10:
                bi = self.below(bi)
        while ai < INVALID_ASK and na < 10:
            ad[na] = self.ask_qty(ai)*ai*self.tick
            na += 1
            last_ask=ai
            if na < 10:
                ai = self.above(ai)
        if nb < 10 or na < 10:
            return out
        for j in range(3):
            k = 1 if j == 0 else (5 if j == 1 else 10)
            bn, an = np.sum(bd[:k]), np.sum(ad[:k])
            out[7+2*j] = math.log1p(bn if side == 1 else an)
            out[8+2*j] = math.log1p(an if side == 1 else bn)
            out[13+j] = side*(bn-an)/(bn+an)
        out[16] = bd[0]/np.sum(bd) if side == 1 else ad[0]/np.sum(ad)
        out[17] = ad[0]/np.sum(ad) if side == 1 else bd[0]/np.sum(bd)
        out[18] = b-last_bid if side == 1 else last_ask-a
        out[19] = last_ask-a if side == 1 else b-last_bid
        for j in range(4):
            w_ms = 100 if j == 0 else (1000 if j == 1 else (5000 if j == 2 else 30000))
            start = now-w_ms*1000000
            p = self.past(start)
            if self.epoch_ns > start or p < 0:
                continue
            sums = self.cum-self.history[p]
            # Squared returns and nonnegative flow accumulators can have
            # floating subtraction dust after long segments.
            for k in range(26):
                if k != 5 and sums[k] < 0 and sums[k] > -1e-7:
                    sums[k] = 0
            sec = w_ms/1000.
            offset = 20+j*29
            buy, sell = sums[3], sums[4]
            out[offset] = math.log1p((sums[1]+sums[2])/sec)
            out[offset+1] = math.log1p((buy+sell)/sec)
            out[offset+2] = side*(buy-sell)/(buy+sell) if buy+sell > 0 else 0.
            own = 0 if side == 1 else 1
            opposite = 1-own
            out[offset+3] = math.log1p(sums[3+own]/sec)
            out[offset+4] = math.log1p(sums[3+opposite]/sec)
            out[offset+5] = math.log1p(sums[0]/sec)
            out[offset+6] = math.log1p(sums[7]/sec)
            out[offset+7] = side*sums[5]/(bq+aq)
            out[offset+8] = math.sqrt(max(0., sums[6]))
            out[offset+9] = side*math.log(mid/self.history_mid[p])*1e4
            for band in range(3):
                for action in range(2):
                    base = 8+band*4+action*2
                    dest = offset+10+band*4+action*2
                    out[dest] = math.log1p(sums[base+own]/sec)
                    out[dest+1] = math.log1p(sums[base+opposite]/sec)
            out[offset+22] = math.log1p(sums[20+own]/sec)
            out[offset+23] = math.log1p(sums[20+opposite]/sec)
            if sums[20+own] > 0:
                out[offset+24] = sums[22+own]/sums[20+own]
            if sums[20+opposite] > 0:
                out[offset+25] = sums[22+opposite]/sums[20+opposite]
            out[offset+26] = math.log1p(sums[24+own]/sec)
            out[offset+27] = math.log1p(sums[24+opposite]/sec)
            out[offset+28] = side*((bq-self.history_bq[p])-(aq-self.history_aq[p]))/(bq+aq)
        return out


def batches(local_rows):
    """Complete timestamp batches; caller must reject cross-shard splits."""
    times = local_rows['local_ts']
    boundaries = np.r_[0, np.flatnonzero(np.diff(times)) + 1, len(times)]
    return boundaries


@njit
def process_chunk(state, data, query_ns, sides, prices, qtys):
    """Batch compilation avoids millions of Python/Numba transitions."""
    result = np.full((len(query_ns), 136), np.nan)
    quotes = np.full((len(query_ns), 2), np.nan)
    e = 0
    for q in range(len(query_ns)):
        while e < len(data) and data[e]['local_ts'] <= query_ns[q]:
            end = e+1
            while end < len(data) and data[end]['local_ts'] == data[e]['local_ts']:
                end += 1
            state.ingest(data, e, end)
            e = end
        result[q] = state.query(query_ns[q], sides[q], prices[q], qtys[q])
        if state.valid():
            quotes[q, 0] = state.best_bid*state.tick
            quotes[q, 1] = state.best_ask*state.tick
    while e < len(data):
        end = e+1
        while end < len(data) and data[end]['local_ts'] == data[e]['local_ts']:
            end += 1
        state.ingest(data, e, end)
        e = end
    return result, quotes
