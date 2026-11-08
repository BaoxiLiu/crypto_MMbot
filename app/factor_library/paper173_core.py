"""Auditable NumPy reference for paper173, using LOCAL observable inputs.

Engineering choices where the paper omits details are frozen in CONTRACT.md.
This reference is intentionally simple; optimized extraction must match it.
"""
import re
from dataclasses import dataclass

import numpy as np

from app.factor_library.paper173_schema import FEATURE_NAMES

# Names/order now live with factor definitions, independently of report files.
assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES)) == 173
WINDOW_NS = {'100ms':100_000_000, '1s':1_000_000_000, '5s':5_000_000_000,
             '30s':30_000_000_000, '300s':300_000_000_000}


@dataclass
class Observations:
    trade_ns: np.ndarray
    trade_price: np.ndarray
    trade_qty: np.ndarray
    trade_buy: np.ndarray
    book_ns: np.ndarray
    book_bid: np.ndarray
    book_ask: np.ndarray
    epoch_ns: int

    def __post_init__(self):
        if self.trade_ns.dtype != np.int64 or self.book_ns.dtype != np.int64:
            raise ValueError('Integer nanosecond clocks required')
        for clock in (self.trade_ns, self.book_ns):
            if np.any(clock[1:] < clock[:-1]):
                raise ValueError('Unsorted clock')
        if not (len(self.trade_ns) == len(self.trade_price) == len(self.trade_qty) == len(self.trade_buy)):
            raise ValueError('Trade columns differ')
        if not (len(self.book_ns) == len(self.book_bid) == len(self.book_ask)):
            raise ValueError('Book columns differ')
        if not all(np.all(np.isfinite(a)) and np.all(a > 0) for a in (self.trade_price, self.trade_qty)):
            raise ValueError('Invalid trade')
        if np.any(self.book_bid <= 0) or np.any(self.book_ask <= self.book_bid):
            raise ValueError('Invalid BBO')
        if np.any(self.trade_ns < self.epoch_ns) or np.any(self.book_ns < self.epoch_ns):
            raise ValueError('Observations before certified epoch')


def impact_bp(prices, quantities, target, side):
    """Executable-price VWAP distance for fixed notional, fractional last level."""
    prices, quantities = np.asarray(prices, float), np.asarray(quantities, float)
    if side not in (-1, 1) or not np.isfinite(target) or target <= 0:
        raise ValueError('Invalid impact query')
    if len(prices) != len(quantities):
        raise ValueError('Depth columns differ')
    if not len(prices):
        return np.nan
    if (np.any(~np.isfinite(prices)) or np.any(~np.isfinite(quantities)) or
            np.any(prices <= 0) or np.any(quantities <= 0)):
        raise ValueError('Invalid depth')
    # side=+1 denotes the bid ladder, decreasing in price.
    if np.any(np.diff(prices)*side >= 0):
        raise ValueError('Depth ladder not strictly best to worst')
    remaining, total_qty = float(target), 0.
    for price, qty in zip(prices, quantities):
        used = min(remaining, price*qty)
        total_qty += used/price
        remaining -= used
        if remaining <= target*1e-12:
            return side*(1-(target/total_qty)/prices[0])*10_000
    return np.nan  # No extrapolation beyond observed depth.


def compute(obs, now_ns, side, bid_prices, bid_qty, ask_prices, ask_qty):
    """Compute one order-oriented vector, with the supplied book as of now.

    Book ladder provenance must be checked by the replay adapter. This function
    independently rejects future observations when selecting trade/BBO history.
    """
    if side not in (-1, 1) or not isinstance(now_ns, (int, np.integer)):
        raise ValueError('Explicit integer decision and order direction required')
    now = int(now_ns)
    out = dict.fromkeys(FEATURE_NAMES, np.nan)
    bi = np.searchsorted(obs.book_ns, now, side='right')-1
    if bi < 0 or now-obs.book_ns[bi] > 1_000_000_000:
        return np.array(list(out.values()))
    if not len(bid_prices) or not len(ask_prices):
        return np.array(list(out.values()))
    if not np.isclose(bid_prices[0], obs.book_bid[bi], rtol=1e-12, atol=0) or not np.isclose(ask_prices[0], obs.book_ask[bi], rtol=1e-12, atol=0):
        raise ValueError('Current ladder does not match historical as-of BBO')
    touch = obs.book_bid[bi] if side == 1 else obs.book_ask[bi]
    ib = impact_bp(bid_prices, bid_qty, 500_000., 1)
    ia = impact_bp(ask_prices, ask_qty, 500_000., -1)
    out['ob_bid_liq'] = bid_qty[0] if side == 1 else ask_qty[0]
    out['ob_ask_liq'] = ask_qty[0] if side == 1 else bid_qty[0]
    out['ob_half'], out['ob_other_half'] = (ib, ia) if side == 1 else (ia, ib)
    # Actual changes of prices, not quantity updates or receipt freshness.
    b = obs.book_bid[:bi+1]; a = obs.book_ask[:bi+1]; t = obs.book_ns[:bi+1]
    changes = np.flatnonzero((b[1:] != b[:-1]) | (a[1:] != a[:-1]))+1
    ct = t[changes]
    if len(ct):
        out['age'] = (now-int(ct[-1]))/1e9
    if now-600_000_000_000 >= obs.epoch_ns and len(ct) >= 2:
        ends = ct[1:]
        select = ends > now-600_000_000_000
        if np.any(select):
            out['totb_mean'] = np.mean(np.diff(ct)[select])/1e9
    for scale, length in WINDOW_NS.items():
        for lag in range(3):
            end = now-lag*length; start = end-length
            if start < obs.epoch_ns:
                continue
            lo = np.searchsorted(obs.trade_ns, start, side='right')
            hi = np.searchsorted(obs.trade_ns, end, side='right')
            p = obs.trade_price[lo:hi]; q = obs.trade_qty[lo:hi]
            buy = obs.trade_buy[lo:hi]
            same = buy if side == 1 else ~buy
            values = dict(buy_count=float(np.sum(same)), sell_count=float(np.sum(~same)),
                          total_buy=float(np.sum(q[same])), total_sell=float(np.sum(q[~same])))
            if len(p):
                values.update(amplitude=float(np.max(p)-np.min(p)), max_size=float(np.max(q)),
                              avg_size=float(np.mean(q)), ret_vwap=side*np.log(np.sum(p*q)/np.sum(q)/touch))
            if len(p) >= 2:
                r = side*np.diff(np.log(p))
                values['ret_sum'] = float(np.sum(r))
                values['trade_intensity'] = float(np.mean(np.diff(obs.trade_ns[lo:hi]))/1e9)
                if len(r) >= 2:
                    centered = r-np.mean(r)
                    values['ret_autocov'] = float(np.mean(centered[:-1]*centered[1:]))
            for family, value in values.items():
                out[f'{family}_{scale}_w{lag}'] = value
    last_grid = now//10_000_000_000*10_000_000_000
    for count in (100, 500):
        grid = last_grid-np.arange(count, -1, -1, dtype=np.int64)*10_000_000_000
        if grid[0] < obs.epoch_ns:
            continue
        pos = np.searchsorted(obs.trade_ns, grid, side='right')-1
        if np.all(pos >= 0):
            out[f'stdev_{count}'] = float(np.std(np.diff(np.log(obs.trade_price[pos])), ddof=1))
    return np.array([out[n] for n in FEATURE_NAMES], dtype=float)
