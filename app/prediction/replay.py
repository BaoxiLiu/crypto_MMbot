"""Reference (non-incremental) replay used to reconcile the P0.5 probe anchors.

This mirrors data/prediction-v3/probe-p0.5/probe_main.py line for line: float
price keys, tb = transaction_time if present else received_time//1e6, grid mid
= state after the last update with tb <= second mark, A1 over <= 3 ticks, C1
over (T-1000, T] on trade_time, D6 = std(d log mid, ddof=1) x 1e4 over the last
31 grid mids. It is O(n) per query and exists ONLY for validation/anchors; the
production path is FactorMachine.
"""
import math

import numpy as np


def time_base(ob):
    """P0.5 probe time base for orderbook rows (ms)."""
    tb = np.where(ob["transaction_time"].notna(),
                  ob["transaction_time"],
                  ob["received_time"] // 1_000_000)
    return tb.astype(np.int64)


def book_at(ob, t_ms, cache=None):
    """Book state (float-keyed dicts) after applying all rows with tb <= t_ms.

    Pass a dict as ``cache`` to reuse the parsed snapshot/update arrays across
    calls for the same frame.
    """
    if cache is None:
        cache = {}
    key = id(ob)
    if key not in cache:
        snap = ob[ob.event_type == "snapshot"]
        book0 = {"bid": {}, "ask": {}}
        for side, p, q in zip(snap.side, snap.price.astype(float), snap.quantity.astype(float)):
            if q != 0.0:
                book0[side][p] = q
        upd = ob[ob.event_type == "update"]
        cache[key] = (book0, time_base(upd), upd.side.to_numpy(),
                      upd.price.astype(float).to_numpy(),
                      upd.quantity.astype(float).to_numpy())
    book0, u_tb, u_side, u_p, u_q = cache[key]
    bid = dict(book0["bid"])
    ask = dict(book0["ask"])
    for i in range(len(u_tb)):
        if u_tb[i] > t_ms:
            break
        s, p, q = u_side[i], u_p[i], u_q[i]
        b = bid if s == "bid" else ask
        if q == 0.0:
            b.pop(p, None)
        else:
            b[p] = q
    return bid, ask


def depth_imb(book, tick, ticks):
    bid, ask = book
    bb = max(bid)
    ba = min(ask)
    bq = sum(q for p, q in bid.items() if bb - p <= ticks * tick + 1e-9)
    aq = sum(q for p, q in ask.items() if p - ba <= ticks * tick + 1e-9)
    a1 = (bq - aq) / (bq + aq) if bq + aq > 0 else float("nan")
    return a1, bq, aq, bb, ba


def grid_mids(ob, hour_start_ms, n_secs=3600):
    """1s mid grid: grid second s = state after last update tb <= mark s."""
    snap = ob[ob.event_type == "snapshot"]
    book = {"bid": {}, "ask": {}}
    for side, p, q in zip(snap.side, snap.price.astype(float), snap.quantity.astype(float)):
        if q != 0.0:
            book[side][p] = q
    upd = ob[ob.event_type == "update"]
    u_tb = time_base(upd)
    u_side = upd.side.to_numpy()
    u_p = upd.price.astype(float).to_numpy()
    u_q = upd.quantity.astype(float).to_numpy()
    mids = np.full(n_secs, np.nan)
    bi = 0
    n = len(u_tb)
    for s in range(n_secs):
        mark = hour_start_ms + s * 1000
        while bi < n and u_tb[bi] <= mark:
            b = book["bid"] if u_side[bi] == "bid" else book["ask"]
            p, q = u_p[bi], u_q[bi]
            if q == 0.0:
                b.pop(p, None)
            else:
                b[p] = q
            bi += 1
        if book["bid"] and book["ask"]:
            mids[s] = (max(book["bid"]) + min(book["ask"])) / 2
    return mids


def taker_imb_1s(tr, t_ms):
    """C1 over (T-1000, T] on trade_time; buyer-initiated = is_buyer_maker=False."""
    win = tr[(tr.trade_time > t_ms - 1000) & (tr.trade_time <= t_ms)]
    notional = win.price.astype(float) * win.quantity.astype(float)
    buy = float(notional[~win.is_buyer_maker].sum())
    sell = float(notional[win.is_buyer_maker].sum())
    c1 = (buy - sell) / (buy + sell) if buy + sell > 0 else float("nan")
    return c1, buy, sell, int(len(win))


def rv_30s(mids, sec):
    seg = mids[sec - 30: sec + 1]
    rets = np.diff(np.log(seg))
    return float(np.std(rets, ddof=1) * 1e4)
