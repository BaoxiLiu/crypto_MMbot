"""Incremental causal factor state machine for the Kimi v3 line (V3_SPEC §3).

Implements all 50 new factors (A8/B6/C10/D15/E11) as an O(1)-update streaming
state machine over a BookState. Causality: events must be fed with non-decreasing
timestamps; every factor value returned by query(now_ms) uses only events with
t <= now_ms. Window coverage that is not complete yields NaN, never zero-fill
(V3_SPEC §0.3 and §3 empty-value rule).

Grid semantics (frozen with this implementation, matching the P0.5 probe):
the 1s decision grid is left-continuous - grid second s uses the book state
after the last event with tb <= hour_start + s*1000, and flow events (trades,
cancels, adds) with timestamp t are attributed to grid second (t-1)//1000 + 1.
query(now_ms) floors now_ms to its grid second for book-state factors; only
C1/C2/C3/C4/C9 are recomputed at exact millisecond windows from the retained
raw trade stream (used for the probe anchor reconciliation).

Documented interpretation choices (V3_SPEC v1.1 leaves these open):
- A1-A5/A8 instant factors are sampled on the 1s grid (state at the second
  mark). Sub-second decision marks are not part of the 1s row grid.
- B-group (v1.1): any quantity decrease or level delete IS a cancel; trade
  consumption is not subtracted (probe: impact < 0.5%).
- B4 near-touch distance uses the 1s grid state of the event's own grid second
  (probe recommendation #4).
- B6/C6 thresholds come from FactorParams (training-frozen per spec); the P1
  builder estimates them within the hour and marks them provisional.
- A8/C5 rolling 1h medians require the full 3600 s window covered by the
  stream; an isolated single hour therefore yields NaN on every row (P2 chains
  hours within a certified segment).
- C9 VPIN is approximated on 1 s buckets: sum|buy-sell| / sum(buy+sell).
- C5 "current" mean trade size is over the trailing 1 s window.
- C7 counts grid seconds with taker_imb_1s > 0 (strictly).
- C10 Kyle lambda regresses per-second signed taker notional (x) on the 1s mid
  change (y, absolute price units) over the trailing 60 s; O(1) via a bounded
  bucket queue, slope = Sxy/Sxx form.
- D4/D5 OFI follows the frozen Cont-style touch formula of
  app/backtesting/feature_cache.py (add_ofi), prefix-differenced over the window.
- D10/D11 are scaled by 1e4 like D6-D8; D13 uses (microprice-mid)/spread with
  the frozen microprice definition; D14 uses eps on the range denominator;
  D15 baseline = median of nonzero |1s returns| inside the same trailing 60 s
  window (spec does not fix the baseline window), 0.0 when no nonzero return.
- E10/E11 use the integer UTC hour: sin/cos(2*pi*hour/24). E9 is not clamped.
- A7 book_update_rate counts diff rows (level changes applied), not grouped
  diff events; one diff event can carry several rows.

State bounds: the book (<= ~20k levels), per-second blocks/aggs for the
trailing 3750 s, raw trades for 70 s, per-trade sizes for 3660 s, and the L1
OFI prefix stream for 70 s. Everything else is O(1).
"""
import math
from bisect import bisect_right
from collections import deque

import numpy as np

from .book import BookState
from .schema import EPS, FACTOR_NAMES, FactorParams

# per-second aggregate field indices
CB, CA, ADD_B, ADD_A, NEAR_C, LARGE_C = 0, 1, 2, 3, 4, 5
TB, TS, NTR, LB, LS, NUPD = 6, 7, 8, 9, 10, 11
PB_NOT, PB_QTY, PS_NOT, PS_QTY = 12, 13, 14, 15
N_AGG = 16

_ROLLING_WINDOWS = (1, 5, 10, 30, 60)

_BLOCK_RETENTION_SECS = 3750
_TRADE_RETENTION_MS = 70_000
_SIZE_RETENTION_MS = 3_660_000
_OFI_RETENTION_MS = 70_000

_B6_FLOOR_MULT = 0.1


class _RollingSums:
    """Per-second aggregate deques over the trailing W seconds.

    Sums are computed fresh from the raw per-second aggregates at query time
    (deques hold <= 60 entries); incremental running sums were dropped because
    add-then-subtract of ~1e5-magnitude notionals leaves ~1e-10 dust that
    corrupts exact-zero window tests on real data.
    """

    __slots__ = ("windows", "dqs")

    def __init__(self, windows=_ROLLING_WINDOWS):
        self.windows = windows
        self.dqs = {w: deque() for w in windows}

    def push(self, sec, agg):
        for w in self.windows:
            d = self.dqs[w]
            d.append((sec, agg))
            while d[0][0] <= sec - w:
                d.popleft()

    def sums(self, w):
        acc = [0.0] * N_AGG
        for _, agg in self.dqs[w]:
            for i in range(N_AGG):
                acc[i] += agg[i]
        return acc


class FactorMachine:
    """One symbol's factor state machine. Feed events in time order, then read
    the 1s grid with query(now_ms) or finalize_upto + blocks."""

    def __init__(self, symbol, tick_size, hour_start_ms, params=None):
        self.symbol = symbol
        self.tick = float(tick_size)
        self.hour_start_ms = int(hour_start_ms)
        self._sec0 = self.hour_start_ms // 1000   # absolute epoch sec of rel sec 0
        self.params = params if params is not None else FactorParams()
        self.book = BookState(self.tick)
        self._is_btc = symbol.upper().startswith("BTC")

        # blocks/aggs are indexed by ABSOLUTE epoch second; _base is the
        # absolute sec of blocks[0]. A fresh hour starts with _base == _sec0;
        # importing a previous hour's state shifts _base backwards so window
        # factors chain across the hour boundary (P1.5).
        self.blocks = []   # one dict per finalized grid second
        self.aggs = []     # parallel per-second aggregate lists
        self._base = self._sec0
        self._next_sec = 0
        self._cur = [0.0] * N_AGG
        self._cur_cancels = []   # (tb, side, idx, dq, price)
        self._cur_trades = []    # (tb, notional, qty, is_buyer_maker)
        self._last_mark = None   # (mark_price, funding, next_funding_time)

        self._trades = deque()   # raw (t, buy, sell, large_buy, large_sell)
        self._sizes = deque()    # (t, qty)
        self._ofi_t = []
        self._ofi_p = []
        self._ofi_head = 0
        self._prev_touch = None
        self._ofi_prefix = 0.0

        self._rolling = _RollingSums()
        self._btc = {}      # ABS sec -> (mid, buy_n, sell_n), BTC grid feed
        self._mkt = {}      # ABS sec -> equal-weight market mid
        self._last_event_ms = -1

    # ------------------------------------------------------- history access

    def _win(self, j):
        """Aggregate list for relative sec j, or None outside retained history."""
        k = j + self._sec0 - self._base
        if 0 <= k < len(self.aggs):
            return self.aggs[k]
        return None

    def _blk(self, j):
        k = j + self._sec0 - self._base
        if 0 <= k < len(self.blocks):
            return self.blocks[k]
        return None

    # ----------------------------------------------------------- state I/O

    def export_state(self):
        """Serializable window tail for cross-hour chaining (P1.5).

        Covers everything window factors need from the previous hour: the
        retained blocks/aggs (3750s: > 1h medians + 300s rv), raw trades and
        per-trade sizes, the L1 OFI prefix stream + continuation baseline, the
        last mark_price row, and the BTC/market grid feeds. The book itself is
        NOT exported: each hour rebuilds from its own snapshot.
        """
        return {
            "version": 1,
            "base": self._base,
            "blocks": list(self.blocks),
            "aggs": [list(a) for a in self.aggs],
            "trades": list(self._trades),
            "sizes": list(self._sizes),
            "ofi_t": list(self._ofi_t[self._ofi_head:]),
            "ofi_p": list(self._ofi_p[self._ofi_head:]),
            "ofi_prefix": self._ofi_prefix,
            "prev_touch": self._prev_touch,
            "last_mark": self._last_mark,
            "btc": dict(self._btc),
            "mkt": dict(self._mkt),
        }

    def import_state(self, state):
        """Load a previous hour's export_state(); call AFTER apply_snapshot."""
        if state.get("version") != 1:
            raise ValueError("unsupported window state version")
        self.blocks = list(state["blocks"])
        self.aggs = [list(a) for a in state["aggs"]]
        self._base = int(state["base"])
        if len(self.blocks) != len(self.aggs):
            raise ValueError("window state blocks/aggs length mismatch")
        self._trades = deque(state["trades"])
        self._sizes = deque(state["sizes"])
        self._ofi_t = list(state["ofi_t"])
        self._ofi_p = list(state["ofi_p"])
        self._ofi_head = 0
        self._ofi_prefix = float(state["ofi_prefix"])
        # prev_touch intentionally NOT imported: the hour snapshot is the
        # authoritative book baseline (apply_snapshot sets it).
        self._last_mark = state["last_mark"]
        self._btc = dict(state["btc"])
        self._mkt = dict(state["mkt"])
        self._last_event_ms = -1
        self._rolling = _RollingSums()
        for k, a in enumerate(self.aggs):
            self._rolling.push(self._base + k - self._sec0, a)

    # ------------------------------------------------------------------ feeds

    def _advance(self, t):
        # Events slightly before hour_start (collection stragglers of the
        # previous hour) belong to grid second 0: no finalization triggers
        # below hour_start, so they land in the currently open second.
        if t < self._last_event_ms:
            raise ValueError(
                f"out-of-order event t={t} after t={self._last_event_ms}")
        self._last_event_ms = t
        while self.hour_start_ms + self._next_sec * 1000 < t:
            self._finalize(self._next_sec)
            self._next_sec += 1

    def apply_snapshot(self, sides, idxs, qtys):
        """Full replace from the hour-head snapshot. The snapshot's touch is
        the OFI baseline: the first touch change of the hour accumulates
        against it (Cont-style continuation across the boundary)."""
        self.book.load_snapshot(sides, idxs, qtys)
        self._prev_touch = self._touch_sample()

    def apply_update(self, tb, side, idx, qty, price):
        """One orderbook diff row (not snapshot): absolute qty, 0 = delete."""
        self._advance(tb)
        old, new = self.book.apply(side, idx, qty)
        self._cur[NUPD] += 1.0
        if new < old:
            self._cur_cancels.append((tb, side, idx, old - new, price))
        elif new > old:
            f = ADD_B if side == "bid" else ADD_A
            self._cur[f] += (new - old) * price
        self._update_touch(tb)

    def apply_trade(self, tb, price, qty, is_buyer_maker):
        """One trade. is_buyer_maker=True means the seller is the taker."""
        self._advance(tb)
        self._cur_trades.append((tb, price * qty, qty, bool(is_buyer_maker)))

    def apply_mark(self, tb, mark_price, funding_rate, next_funding_time):
        """One mark_price stream row; last value carries (funding level)."""
        self._advance(tb)
        self._last_mark = (mark_price, funding_rate, next_funding_time)

    def feed_btc_second(self, sec, mid, buy_n, sell_n):
        """External BTC 1s grid feed (cross-coin wiring); BTC self-feeds.

        ``sec`` is relative to this machine's hour; stored by absolute sec.
        Must be called BEFORE the corresponding second is finalized (lockstep
        drivers: feed sec s right before/at the finalize of sec s, then call
        recompute_env(s) if sec s was already finalized)."""
        self._btc[self._sec0 + int(sec)] = (mid, buy_n, sell_n)

    def feed_market_mid(self, sec, mid):
        self._mkt[self._sec0 + int(sec)] = float(mid)

    def finalize_upto(self, last_sec):
        """Flush the grid through last_sec (inclusive)."""
        while self._next_sec <= last_sec:
            self._finalize(self._next_sec)
            self._next_sec += 1

    def recompute_env(self, sec):
        """Recompute E1-E11 of an already-finalized second from the current
        BTC/market feeds and mark stream (P1.5 lockstep wiring: external E
        feeds for sec s arrive after finalize(s), so the driver re-runs the
        environment block once the feeds are in)."""
        block = self._blk(sec)
        if block is None:
            raise ValueError(f"sec {sec} not finalized")
        self._env_block(block, sec)

    # ------------------------------------------------------------- OFI prefix

    def _touch_sample(self):
        bbi, bq = self.book.best_bid()
        bai, aq = self.book.best_ask()
        if bbi is None or bai is None:
            return None
        return (bbi, bq, bai, aq)

    def _update_touch(self, tb):
        cur = self._touch_sample()
        prev = self._prev_touch
        if cur is None or prev is None:
            self._prev_touch = cur
            return
        if cur == prev:
            return
        pbbi, pbq, pbai, paq = prev
        bbi, bq, bai, aq = cur
        ch = 0.0
        if bbi >= pbbi:
            ch += bq
        if bbi <= pbbi:
            ch -= pbq
        if bai <= pbai:
            ch -= aq
        if bai >= pbai:
            ch += paq
        self._ofi_prefix += ch
        self._ofi_t.append(tb)
        self._ofi_p.append(self._ofi_prefix)
        self._prev_touch = cur

    def _ofi_at(self, t):
        i = bisect_right(self._ofi_t, t, lo=self._ofi_head) - 1
        if i < self._ofi_head:
            return 0.0
        return self._ofi_p[i]

    # ---------------------------------------------------------------- finalize

    def _instant_block(self, block):
        """A1-A5, mid/spread/microprice sampled from the current book state."""
        tick = self.tick
        bbi, bq = self.book.best_bid()
        bai, aq = self.book.best_ask()
        if bbi is None or bai is None:
            for name in ("mid", "best_bid", "best_ask", "spread_bp",
                         "micro_off", "A1", "A2", "A3", "A4", "A5"):
                block[name] = float("nan")
            return
        bb = round(bbi * tick, 10)
        ba = round(bai * tick, 10)
        mid = (bb + ba) / 2.0
        spread = ba - bb
        block["best_bid"] = bb
        block["best_ask"] = ba
        block["mid"] = mid
        block["spread_bp"] = spread / mid * 1e4 if mid > 0 else float("nan")
        if spread > 0 and bq + aq > 0:
            micro = (bb * aq + ba * bq) / (bq + aq)
            block["micro_off"] = (micro - mid) / spread
        else:
            block["micro_off"] = float("nan")

        bids, asks = self.book.bids, self.book.asks
        b3 = sum(bids.get(bbi - k, 0.0) for k in range(4))
        a3 = sum(asks.get(bai + k, 0.0) for k in range(4))
        block["A1"] = (b3 - a3) / (b3 + a3) if b3 + a3 > 0 else float("nan")
        b10 = sum(bids.get(bbi - k, 0.0) for k in range(11))
        a10 = sum(asks.get(bai + k, 0.0) for k in range(11))
        block["A2"] = (b10 - a10) / (b10 + a10) if b10 + a10 > 0 else float("nan")

        def band_qty(is_bid, band_bp):
            # qty on levels within band_bp of mid on the given side
            band = band_bp * mid
            if is_bid:
                kmax = math.floor((bb - mid + band) / tick + 1e-6)
                if kmax < 0:
                    return 0.0
                return sum(bids.get(bbi - k, 0.0) for k in range(kmax + 1))
            kmax = math.floor((mid + band - ba) / tick + 1e-6)
            if kmax < 0:
                return 0.0
            return sum(asks.get(bai + k, 0.0) for k in range(kmax + 1))

        bb5 = band_qty(True, 5e-4)
        ba5 = band_qty(False, 5e-4)
        block["A3"] = (bb5 - ba5) / (bb5 + ba5) if bb5 + ba5 > 0 else float("nan")
        bb10 = band_qty(True, 1e-3)
        ba10 = band_qty(False, 1e-3)
        block["A4"] = (bb10 - ba10) / (bb10 + ba10) if bb10 + ba10 > 0 else float("nan")

        slope_b = (b10 - b3) / 7.0
        slope_a = (a10 - a3) / 7.0
        block["A5"] = ((slope_b - slope_a)
                       / (abs(slope_b) + abs(slope_a) + EPS))

    def _process_flows(self, block):
        """Attribute the open second's cancels/trades to aggregates and the
        retained raw streams (B4 near-touch uses this second's grid state)."""
        tick = self.tick
        agg = self._cur
        bbi, _ = self.book.best_bid()
        bai, _ = self.book.best_ask()
        bb = round(bbi * tick, 10) if bbi is not None else None
        ba = round(bai * tick, 10) if bai is not None else None
        params = self.params

        for tb, side, idx, dq, price in self._cur_cancels:
            notional = dq * price
            if side == "bid":
                agg[CB] += notional
            else:
                agg[CA] += notional
            if bb is not None and ba is not None:
                dist = (bb - price) if side == "bid" else (price - ba)
                if -1e-9 <= dist <= tick + 1e-9:
                    agg[NEAR_C] += notional
            if params.cancel_p90_notional is not None and params.median_cancel_qty is not None:
                if (notional > params.cancel_p90_notional
                        and dq > _B6_FLOOR_MULT * params.median_cancel_qty):
                    agg[LARGE_C] += notional

        p90t = params.trade_p90_notional
        for tb, notional, qty, is_bm in self._cur_trades:
            large = p90t is not None and notional > p90t
            buy = 0.0 if is_bm else notional
            sell = notional if is_bm else 0.0
            lb = buy if large and not is_bm else 0.0
            ls = sell if large and is_bm else 0.0
            agg[TB] += buy
            agg[TS] += sell
            agg[NTR] += 1.0
            agg[LB] += lb
            agg[LS] += ls
            if buy:
                agg[PB_NOT] += notional
                agg[PB_QTY] += qty
            else:
                agg[PS_NOT] += notional
                agg[PS_QTY] += qty
            self._trades.append((tb, buy, sell, lb, ls))
            self._sizes.append((tb, qty))

        block["buy_n_1s"] = agg[TB]
        block["sell_n_1s"] = agg[TS]
        block["n_trades_1s"] = agg[NTR]
        block["buy_qty_1s"] = agg[PB_QTY]
        block["sell_qty_1s"] = agg[PS_QTY]
        block["buy_vwap_1s"] = agg[PB_NOT] / agg[PB_QTY] if agg[PB_QTY] > 0 else float("nan")
        block["sell_vwap_1s"] = agg[PS_NOT] / agg[PS_QTY] if agg[PS_QTY] > 0 else float("nan")
        if self._last_mark is not None:
            block["mark_price"], block["funding_rate"], block["next_funding_time"] = self._last_mark
        else:
            block["mark_price"] = block["funding_rate"] = float("nan")
            block["next_funding_time"] = 0

    def _finalize(self, s):
        hour_start = self.hour_start_ms
        m = hour_start + s * 1000
        block = {name: float("nan") for name in FACTOR_NAMES}
        self._instant_block(block)
        self._process_flows(block)

        mid = block["mid"]
        if self._is_btc and mid == mid:
            self._btc[self._sec0 + s] = (mid, self._cur[TB], self._cur[TS])

        # append first so windowed factors can read secs <= s
        self.blocks.append(block)
        self.aggs.append(self._cur)

        self._rolling.push(s, self._cur)
        s1, s5, s10, s30, s60 = (self._rolling.sums(w) for w in (1, 5, 10, 30, 60))

        # window (s-W, s] must lie inside the retained stream: fresh hours
        # start at relative sec 0 (segment head -> NaN), chained hours start
        # negative (previous hour's tail covers the boundary).
        base_rel = self._base - self._sec0

        def covered(w):
            return s - w >= base_rel

        def imb(buy, sell):
            return (buy - sell) / (buy + sell) if buy + sell > 0 else float("nan")

        # A6-A8
        if s >= 5:
            prev = self._blk(s - 5)["A1"]
            cur = block["A1"]
            block["A6"] = cur - prev if cur == cur and prev == prev else float("nan")
        if covered(10):
            block["A7"] = s10[NUPD] / 10.0
        if covered(3600):
            spreads = [self._blk(j)["spread_bp"] for j in range(s - 3599, s + 1)]
            if all(x == x for x in spreads):
                block["A8"] = block["spread_bp"] - float(np.median(spreads))

        # B1-B6
        if covered(30):
            block["B1"] = s30[CB] / 30.0
            block["B2"] = s30[CA] / 30.0
            adds = s30[ADD_B] + s30[ADD_A]
            block["B3"] = (s30[CB] + s30[CA]) / adds if adds > 0 else float("nan")
            block["B5"] = imb(s30[CB], s30[CA])
        if covered(60):
            cancels60 = s60[CB] + s60[CA]
            if cancels60 > 0:
                block["B4"] = s60[NEAR_C] / cancels60
            if self.params.cancel_p90_notional is not None:
                block["B6"] = s60[LARGE_C] / 60.0

        # C1-C6
        block["C1"] = imb(s1[TB], s1[TS]) if covered(1) else float("nan")
        if covered(5):
            block["C2"] = imb(s5[TB], s5[TS])
        if covered(30):
            block["C3"] = imb(s30[TB], s30[TS])
            tot30 = s30[TB] + s30[TS]
            if tot30 > 0 and self.params.trade_p90_notional is not None:
                block["C6"] = (s30[LB] + s30[LS]) / tot30
        if covered(10):
            block["C4"] = s10[NTR] / 10.0
        if covered(3600):
            sizes = [q for (t, q) in self._sizes if t > m - 3_600_000]
            recent = [q for (t, q) in self._sizes if t > m - 1000]
            if sizes and recent:
                block["C5"] = (sum(recent) / len(recent)) / float(np.median(sizes))
        if covered(30):
            pos = 0
            ok = True
            for j in range(s - 29, s + 1):
                a = self._win(j)
                if a is None:
                    ok = False
                    break
                if a[TB] > a[TS]:
                    pos += 1
            if ok:
                block["C7"] = pos / 30.0
        if s >= 10:
            prev = self._blk(s - 5)["C2"]
            cur = block["C2"]
            block["C8"] = cur - prev if cur == cur and prev == prev else float("nan")
        if covered(30):
            num = den = 0.0
            ok = True
            for j in range(s - 29, s + 1):
                a = self._win(j)
                if a is None:
                    ok = False
                    break
                num += abs(a[TB] - a[TS])
                den += a[TB] + a[TS]
            if ok:
                block["C9"] = num / den if den > 0 else float("nan")

        # C10 Kyle lambda (60 x/y buckets, mids s-60..s must all be present)
        if covered(60):
            xs, ys = [], []
            ok = True
            for j in range(s - 59, s + 1):
                a = self._win(j)
                pb = self._blk(j - 1)
                cb = self._blk(j)
                if a is None or pb is None or cb is None:
                    ok = False
                    break
                pmid, cmid = pb["mid"], cb["mid"]
                if pmid != pmid or cmid != cmid:
                    ok = False
                    break
                xs.append(a[TB] - a[TS])
                ys.append(cmid - pmid)
            if ok:
                n = 60
                sx = sum(xs)
                sy = sum(ys)
                sxx = sum(x * x for x in xs)
                sxy = sum(x * y for x, y in zip(xs, ys))
                denom = sxx - sx * sx / n
                if denom > 0:
                    block["C10"] = (sxy - sx * sy / n) / denom

        # D1-D3
        def ret(sec_, lag):
            ba = self._blk(sec_)
            bb = self._blk(sec_ - lag)
            if ba is None or bb is None:
                return float("nan")
            a, b = ba["mid"], bb["mid"]
            if a != a or b != b or b <= 0:
                return float("nan")
            return (a / b - 1.0) * 1e4

        if covered(5):
            block["D1"] = ret(s, 5)
        if covered(30):
            block["D2"] = ret(s, 30)
        if covered(60):
            block["D3"] = ret(s, 60)

        # D4/D5 OFI (prefix stream chains via import_state)
        if covered(5):
            block["D4"] = self._ofi_at(m) - self._ofi_at(m - 5000)
        if covered(30):
            block["D5"] = self._ofi_at(m) - self._ofi_at(m - 30000)

        # D6-D15
        def rets(lag):
            out = []
            for j in range(s - lag + 1, s + 1):
                ba = self._blk(j)
                bb = self._blk(j - 1)
                if ba is None or bb is None:
                    return None
                a, b = ba["mid"], bb["mid"]
                if a != a or b != b or a <= 0 or b <= 0:
                    return None
                out.append(math.log(a / b))
            return out

        def rv(lag):
            r = rets(lag)
            if r is None:
                return float("nan")
            n = len(r)
            if n < 2:
                return float("nan")
            mean = sum(r) / n
            var = sum((x - mean) ** 2 for x in r) / (n - 1)
            return math.sqrt(var) * 1e4

        if covered(30):
            block["D6"] = rv(30)
        if covered(60):
            block["D7"] = rv(60)
        if covered(300):
            block["D8"] = rv(300)
        d8 = block["D8"]
        if block["D6"] == block["D6"] and d8 == d8:
            block["D9"] = block["D6"] / (d8 + EPS)
        if covered(60):
            r = rets(60)
            if r is not None:
                up = sum(x * x for x in r if x > 0)
                down = sum(x * x for x in r if x < 0)
                block["D10"] = math.sqrt(up) * 1e4
                block["D11"] = math.sqrt(down) * 1e4
                block["D12"] = ((math.sqrt(up) - math.sqrt(down))
                                / (math.sqrt(up) + math.sqrt(down) + EPS))
            mids = []
            ok = True
            for j in range(s - 60, s + 1):
                bj = self._blk(j)
                if bj is None:
                    ok = False
                    break
                mids.append(bj["mid"])
            if ok and all(x == x for x in mids):
                lo, hi = min(mids), max(mids)
                block["D14"] = (mid - lo) / (hi - lo + EPS)
            r = rets(60)
            if r is not None:
                nz = [abs(x) for x in r if x != 0.0]
                if nz:
                    med = float(np.median(nz))
                    block["D15"] = float(sum(1 for x in r if abs(x) > 4.0 * med))
                else:
                    block["D15"] = 0.0

        # D13
        if s >= 10:
            prev = self._blk(s - 10)["micro_off"]
            cur = block["micro_off"]
            block["D13"] = cur - prev if cur == cur and prev == prev else float("nan")

        self._env_block(block, s)

        block["sec"] = m // 1000

        # retention trims (stream-time based, monotonic => amortized O(1))
        while self._trades and self._trades[0][0] < m - _TRADE_RETENTION_MS:
            self._trades.popleft()
        while self._sizes and self._sizes[0][0] < m - _SIZE_RETENTION_MS:
            self._sizes.popleft()
        while (self._ofi_head < len(self._ofi_t)
               and self._ofi_t[self._ofi_head] < m - _OFI_RETENTION_MS):
            self._ofi_head += 1
        cutoff = self._sec0 + s - _BLOCK_RETENTION_SECS
        if cutoff > self._base:
            n = cutoff - self._base
            del self.blocks[:n]
            del self.aggs[:n]
            self._base = cutoff
            if self._ofi_head > 4096:
                del self._ofi_t[:self._ofi_head]
                del self._ofi_p[:self._ofi_head]
                self._ofi_head = 0
            for j in list(self._btc):
                if j <= cutoff:
                    del self._btc[j]
            for j in list(self._mkt):
                if j <= cutoff:
                    del self._mkt[j]

        self._cur = [0.0] * N_AGG
        self._cur_cancels = []
        self._cur_trades = []

    # -------------------------------------------------------------- E block

    def _env_block(self, block, s):
        """E1-E11 for grid second s from the BTC/market feeds and mark stream.

        External feeds (feed_btc_second/feed_market_mid) for sec s must be in
        place BEFORE this runs; lockstep drivers that feed sec s only after
        finalize(s) call recompute_env(s) afterwards. BTC self-feeds happen
        inside finalize, before this is called.
        """
        m = self.hour_start_ms + s * 1000
        base_rel = self._base - self._sec0

        def covered(w):
            return s - w >= base_rel

        def imb(buy, sell):
            return (buy - sell) / (buy + sell) if buy + sell > 0 else float("nan")

        def btc_mid(j):
            e = self._btc.get(self._sec0 + j)
            return e[0] if e else None

        if covered(5):
            a, b = btc_mid(s), btc_mid(s - 5)
            if a is not None and b is not None and b > 0:
                block["E1"] = (a / b - 1.0) * 1e4
        if covered(30):
            a, b = btc_mid(s), btc_mid(s - 30)
            if a is not None and b is not None and b > 0:
                block["E2"] = (a / b - 1.0) * 1e4
            bl = sl = 0.0
            ok = True
            for j in range(s - 29, s + 1):
                e = self._btc.get(self._sec0 + j)
                if e is None:
                    ok = False
                    break
                bl += e[1]
                sl += e[2]
            if ok:
                block["E4"] = imb(bl, sl)
        if covered(60):
            mseries = [btc_mid(j) for j in range(s - 60, s + 1)]
            if all(x is not None and x > 0 for x in mseries):
                r = [math.log(mseries[k] / mseries[k - 1]) for k in range(1, 61)]
                mean = sum(r) / 60
                var = sum((x - mean) ** 2 for x in r) / 59
                block["E3"] = math.sqrt(var) * 1e4

        def mkt_mid(j):
            return self._mkt.get(self._sec0 + j)

        if covered(5):
            a, b = mkt_mid(s), mkt_mid(s - 5)
            if a is not None and b is not None and b > 0:
                block["E5"] = (a / b - 1.0) * 1e4
        if covered(30):
            a, b = mkt_mid(s), mkt_mid(s - 30)
            if a is not None and b is not None and b > 0:
                block["E6"] = (a / b - 1.0) * 1e4
        if covered(60):
            mseries = [mkt_mid(j) for j in range(s - 60, s + 1)]
            if all(x is not None and x > 0 for x in mseries):
                r = [math.log(mseries[k] / mseries[k - 1]) for k in range(1, 61)]
                mean = sum(r) / 60
                var = sum((x - mean) ** 2 for x in r) / 59
                block["E7"] = math.sqrt(var) * 1e4

        if self._last_mark is not None:
            funding = self._last_mark[1]
            nft = self._last_mark[2]
            if funding == funding:
                block["E8"] = funding
            if nft:
                block["E9"] = (nft - m) / 60000.0
        hour = ((m % 86_400_000)) / 3_600_000.0
        block["E10"] = math.sin(2.0 * math.pi * hour / 24.0)
        block["E11"] = math.cos(2.0 * math.pi * hour / 24.0)

    # ------------------------------------------------------------------ query

    def query(self, now_ms):
        """The 50 factor values as of now_ms (causal: only events <= now_ms).

        Grid-second marks ((now_ms - hour_start) % 1000 == 0) return the stored
        block. Other timestamps return the floor-second block with C1/C2/C3/
        C4/C9 recomputed at the exact millisecond window from the retained raw
        trade stream (anchor-reconciliation path).
        """
        rel_ms = now_ms - self.hour_start_ms
        if rel_ms < 0:
            return {name: float("nan") for name in FACTOR_NAMES}
        sec = rel_ms // 1000
        on_mark = rel_ms % 1000 == 0
        blk = self._blk(sec)
        if on_mark and blk is not None and sec < self._next_sec:
            return dict(blk)
        if sec >= self._next_sec or blk is None:
            return {name: float("nan") for name in FACTOR_NAMES}
        out = dict(blk)
        sums = {1: [0.0] * 5, 5: [0.0] * 5, 10: [0.0] * 5, 30: [0.0] * 5}
        for t, buy, sell, lb, ls in self._trades:
            if t > now_ms:
                break
            for w, acc in sums.items():
                if t > now_ms - w * 1000:
                    acc[0] += buy
                    acc[1] += sell
                    acc[2] += lb
                    acc[3] += ls
                    acc[4] += 1.0
        span = now_ms - self.hour_start_ms
        for w, key in ((1, "C1"), (5, "C2"), (30, "C3")):
            buy, sell = sums[w][0], sums[w][1]
            if span >= w * 1000:
                out[key] = (buy - sell) / (buy + sell) if buy + sell > 0 else float("nan")
        if span >= 10_000:
            out["C4"] = sums[10][4] / 10.0
        if span >= 30_000:
            num = den = 0.0
            for t, buy, sell, _lb, _ls in self._trades:
                if t > now_ms:
                    break
                if t > now_ms - 30_000:
                    num += abs(buy - sell)
                    den += buy + sell
            out["C9"] = num / den if den > 0 else float("nan")
        return out
