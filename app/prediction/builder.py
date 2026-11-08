"""Hour-file feature builder for the Kimi v3 line (P1.5).

Reads $QUANT_DATA_ROOT/crypto/raw/hft/binance_futures/<date>/<HH>/<SYMBOL>_{orderbook,
trades,mark_price}.parquet, enforces the V3_SPEC §3 P1 pipeline gates (trade
capture rate >= 90% with a 200 ms attribution window, contiguous trade_id),
rebuilds the book from the hour-head snapshot, streams the 50 factors through
FactorMachine, and writes one parquet of 1s grid rows plus a sidecar metadata
json (source SHA256, row counts, gate results, B6/C6 threshold source, build
timing and peak RSS). Existing complete outputs are skipped (resume).

P1.5 changes:
- Cross-hour window chaining: each finished hour persists its window tail
  (``<HH>.state.pkl``, machine.export_state) and the next hour imports it, so
  rolling windows (up to the 1h medians) cover the boundary inside a certified
  segment. The book itself is still rebuilt from the hour snapshot; only
  factor window state crosses hours. Chaining needs the previous hour's state
  file; without it the hour starts fresh (segment-head semantics, logged).
- E-group wiring: build_hour_multi runs a per-second lockstep across symbols
  (market mid = equal-weight mean of available symbol mids; BTC 1s grid fed to
  non-BTC machines) and recomputes E1-E11 after the feeds land.
- B6/C6 thresholds load from the training-frozen table
  (app/prediction/thresholds.json) when present; otherwise the within-hour
  provisional estimate is used and flagged in metadata.
- Segment-head rows keep the NaN semantics of V3_SPEC (window coverage);
  row removal happens at dataset assembly, not here.
"""
import argparse
import hashlib
import json
import pickle
import platform
import resource
import sys
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import book as _book_mod
from .factors import FactorMachine
from .schema import (
    AUX_COLUMNS,
    CAPTURE_MIN_RATE,
    CAPTURE_WINDOW_MS,
    FACTOR_NAMES,
    GRID_COLUMNS,
    SPEC_VERSION,
    FactorParams,
)
from .thresholds import load_thresholds

RAW_BASE = Path("$QUANT_DATA_ROOT/crypto/raw/hft/binance_futures")
OUT_ROOT = Path("data/prediction-v3/features")


class ValidationError(RuntimeError):
    """An hour file failed a P1 pipeline gate and must not enter the build."""


def _sha256_file(path, _bufsize=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(_bufsize)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _code_sha256():
    root = Path(_book_mod.__file__).parent
    h = hashlib.sha256()
    for name in ("schema.py", "book.py", "factors.py", "thresholds.py", "builder.py"):
        h.update((root / name).read_bytes())
    return h.hexdigest()


def _peak_rss_mb():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KB, macOS reports bytes.
    return rss / (1024.0 ** 2) if platform.system() == "Darwin" else rss / 1024.0


def hour_start_ms(date, hour):
    return int(pd.Timestamp(f"{date}T{int(hour):02d}:00:00", tz="UTC").timestamp() * 1000)


def source_paths(base_dir, symbol, date, hour):
    d = Path(base_dir) / date / f"{int(hour):02d}"
    return {
        "orderbook": d / f"{symbol}_orderbook.parquet",
        "trades": d / f"{symbol}_trades.parquet",
        "mark_price": d / f"{symbol}_mark_price.parquet",
    }


OB_COLUMNS = ["received_time", "event_time", "transaction_time", "event_type",
              "final_update_id", "prev_final_update_id", "side", "price", "quantity"]


def load_hour(symbol, date, hour, base_dir=RAW_BASE):
    paths = source_paths(base_dir, symbol, date, hour)
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing hour inputs: {missing}")
    ob = pq.ParquetFile(paths["orderbook"]).read(columns=OB_COLUMNS).to_pandas()
    tr = pq.ParquetFile(paths["trades"]).read().to_pandas()
    mk = pq.ParquetFile(paths["mark_price"]).read().to_pandas()
    return ob, tr, mk


def infer_tick_size(*price_series):
    """Smallest positive gap between distinct observed prices (the tick).

    Decimal-string prices carry binary representation noise, so the raw minimum
    gap is snapped to 10 decimal places (e.g. 0.0999999999912 -> 0.1).
    """
    parts = [pd.to_numeric(s, errors="coerce").dropna().to_numpy(dtype=float)
             for s in price_series if s is not None and len(s)]
    if not parts:
        return None
    u = np.unique(np.concatenate(parts))
    if len(u) < 2:
        return None
    d = np.diff(u)
    d = d[d > 1e-12]
    if not len(d):
        return None
    return round(float(d.min()), 10)


# ------------------------------------------------------------------ gates

def _replay_book(ob):
    """Snapshot-first book replay; returns (decreases, adds) event lists.

    Decrease event: (tb, side, price, dq). Used by the capture gate and by the
    provisional B6/C6 threshold estimation.
    """
    snap = ob[ob.event_type == "snapshot"]
    book = {"bid": {}, "ask": {}}
    for side, p, q in zip(snap.side, snap.price.astype(float), snap.quantity.astype(float)):
        if q != 0.0:
            book[side][p] = q
    tb = np.where(ob["transaction_time"].notna(),
                  ob["transaction_time"], ob["received_time"] // 1_000_000).astype(np.int64)
    is_snap = (ob.event_type == "snapshot").to_numpy()
    sides = ob.side.to_numpy()
    prices = ob.price.astype(float).to_numpy()
    qtys = ob.quantity.astype(float).to_numpy()
    decreases = []
    adds = []
    for i in range(len(tb)):
        if is_snap[i]:
            continue
        s, p, q = sides[i], prices[i], qtys[i]
        b = book[s]
        old = b.get(p, 0.0)
        if q == 0.0:
            b.pop(p, None)
        else:
            b[p] = q
        if q < old:
            decreases.append((int(tb[i]), s, p, old - q))
        elif q > old:
            adds.append((int(tb[i]), s, p, q - old))
    return decreases, adds


def trade_capture_rate(decreases, tr):
    """Attributed trade notional / total trade notional, 200 ms window.

    Attribution: a decrease at (t, side, price, dq) consumes trades of the same
    side/price with trade_time in [t-W, t+W]; ask-side decrease <=> buyer taker
    (is_buyer_maker=False), bid-side <=> seller taker (is_buyer_maker=True).
    Same algorithm as the P0.5 probe, but attributed in notional (qty x price).
    """
    queues = defaultdict(deque)
    for tt, p, q, is_bm in zip(tr.trade_time.to_numpy(), tr.price.astype(float).to_numpy(),
                               tr.quantity.astype(float).to_numpy(),
                               tr.is_buyer_maker.to_numpy()):
        side = "ask" if not is_bm else "bid"
        queues[(side, p)].append((int(tt), float(q), float(p)))
    w = CAPTURE_WINDOW_MS
    attributed = 0.0
    for tt, side, p, dq in decreases:
        q = queues.get((side, p))
        if not q:
            continue
        matched = 0.0
        while q and q[0][0] < tt - w:
            q.popleft()
        tmp = []
        while q and q[0][0] <= tt + w and matched < dq:
            t2, q2, p2 = q.popleft()
            take = min(q2, dq - matched)
            matched += take
            attributed += take * p2
            if take < q2:
                tmp.append((t2, q2 - take, p2))
        q.extendleft(reversed(tmp))
    total = float((tr.price.astype(float) * tr.quantity.astype(float)).sum())
    return attributed / total if total > 0 else 0.0, attributed, total


def run_gates(ob, tr, decreases=None, capture_min_rate=None):
    """P1 pipeline gates. Raises ValidationError on any hard-gate failure and
    returns a detail dict either way.``decreases`` may carry a pre-computed
    _replay_book result so the caller replays the book only once."""
    if capture_min_rate is None:
        capture_min_rate = CAPTURE_MIN_RATE
    is_snap = ob.event_type == "snapshot"
    n_snap = int(is_snap.sum())
    head_ok = n_snap > 0 and not is_snap.iloc[n_snap:].any()
    trade_ids = np.sort(tr.trade_id.to_numpy())
    trade_id_ok = bool(len(trade_ids) > 0) and bool(np.all(np.diff(trade_ids) == 1))
    if decreases is None:
        decreases, _ = _replay_book(ob)
    rate, attributed, total = trade_capture_rate(decreases, tr)
    n_na = int((tr.order_type == "NA").sum()) if "order_type" in tr.columns else 0

    detail = {
        "snapshot_rows": n_snap,
        "snapshot_head_contiguous": head_ok,
        "trade_rows": int(len(tr)),
        "trade_id_contiguous": trade_id_ok,
        "trade_na_order_type": n_na,
        "capture_rate": rate,
        "capture_attributed_notional": attributed,
        "capture_total_notional": total,
        "decrease_events": len(decreases),
        "update_id_chain_breaks": _update_id_breaks(ob),
    }
    problems = []
    if not head_ok:
        problems.append("snapshot is not a single contiguous head block")
    if not trade_id_ok:
        problems.append("trade_id not contiguous within the hour file")
    if rate < capture_min_rate:
        problems.append(f"trade capture rate {rate:.4f} < {capture_min_rate}")
    detail["ok"] = not problems
    detail["problems"] = problems
    return detail


def _update_id_breaks(ob):
    upd = ob[ob.event_type == "update"]
    if not len(upd):
        return 0
    fid = upd.final_update_id.to_numpy()
    prev = upd.prev_final_update_id.to_numpy()
    new_event = np.ones(len(upd), dtype=bool)
    new_event[1:] = fid[1:] != fid[:-1]
    idx = np.flatnonzero(new_event)
    breaks = 0
    for k in range(1, len(idx)):
        if prev[idx[k]] != fid[idx[k - 1]]:
            breaks += 1
    return int(breaks)


def estimate_params(ob, tr, decreases=None):
    """Provisional per-hour B6/C6 thresholds; used only when the frozen
    training table has no entry for the symbol."""
    if decreases is None:
        decreases, _ = _replay_book(ob)
    if decreases:
        cancel_notional = np.array([dq * p for _, _, p, dq in decreases])
        cancel_qty = np.array([dq for _, _, _, dq in decreases])
        cancel_p90 = float(np.percentile(cancel_notional, 90))
        median_cancel_qty = float(np.median(cancel_qty))
    else:
        cancel_p90 = None
        median_cancel_qty = None
    tr_notional = (tr.price.astype(float) * tr.quantity.astype(float)).to_numpy()
    trade_p90 = float(np.percentile(tr_notional, 90)) if len(tr_notional) else None
    return FactorParams(cancel_p90_notional=cancel_p90,
                        median_cancel_qty=median_cancel_qty,
                        trade_p90_notional=trade_p90)


# ------------------------------------------------------------------ build

def _check_ordered(name, arr):
    if len(arr) > 1 and np.any(arr[1:] < arr[:-1]):
        raise ValidationError(f"{name} is not time-ordered")


def _prev_hour(date, hour):
    dt = datetime.strptime(f"{date} {int(hour):02d}", "%Y-%m-%d %H") - timedelta(hours=1)
    return dt.strftime("%Y-%m-%d"), dt.strftime("%H")


def _state_path(out_root, symbol, date, hour):
    return Path(out_root) / symbol / date / f"{int(hour):02d}.state.pkl"


def _load_chain_state(out_root, symbol, date, hour):
    """Previous hour's window-tail state, or None (fresh/segment-head start)."""
    pdate, phour = _prev_hour(date, hour)
    p = _state_path(out_root, symbol, pdate, phour)
    if not p.exists():
        return None
    try:
        with open(p, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _prepare_symbol_hour(symbol, date, hour, base_dir, frozen, write_meta_fn,
                         capture_min_rate=None):
    """Load, gate and prepare one symbol's hour. Returns the prep dict, or
    None if the hour was already built (resume skip) — the skip meta is
    returned instead via the second element."""
    paths = source_paths(base_dir, symbol, date, hour)
    sources = {k: {"path": str(v), "sha256": _sha256_file(v),
                   "rows": int(pq.ParquetFile(v).metadata.num_rows)}
               for k, v in paths.items()}
    ob, tr, mk = load_hour(symbol, date, hour, base_dir)
    # convert decimal-string columns once; releases the object-dtype storage
    ob["price"] = pd.to_numeric(ob["price"], errors="coerce")
    ob["quantity"] = pd.to_numeric(ob["quantity"], errors="coerce")
    tr["price"] = pd.to_numeric(tr["price"], errors="coerce")
    tr["quantity"] = pd.to_numeric(tr["quantity"], errors="coerce")

    decreases, _adds = _replay_book(ob)
    gates = run_gates(ob, tr, decreases, capture_min_rate=capture_min_rate)
    if not gates["ok"]:
        meta = {"status": "rejected", "symbol": symbol, "date": date, "hour": hour,
                "spec_version": SPEC_VERSION, "code_sha256": _code_sha256(),
                "sources": sources, "gates": gates}
        write_meta_fn(meta)
        raise ValidationError(f"{symbol} {date}/{hour} rejected: {gates['problems']}")

    if symbol in frozen:
        params = frozen[symbol]
        params_source = "frozen"
    else:
        params = estimate_params(ob, tr, decreases)
        params_source = "provisional"

    tick = infer_tick_size(ob.price.head(200000), tr.price)
    if tick is None or tick <= 0:
        raise ValidationError(f"could not infer tick size for {symbol}")

    snap = ob[ob.event_type == "snapshot"]
    upd = ob[ob.event_type == "update"]
    u_tb = np.where(upd["transaction_time"].notna(), upd["transaction_time"],
                    upd["received_time"] // 1_000_000).astype(np.int64)
    _check_ordered("orderbook updates", u_tb)
    tr_tb = tr.trade_time.to_numpy()
    _check_ordered("trades", tr_tb)

    prep = {
        "symbol": symbol, "date": date, "hour": hour,
        "sources": sources, "gates": gates,
        "params": params, "params_source": params_source, "tick": tick,
        "snap_sides": snap.side.to_numpy(),
        "snap_idxs": np.rint(snap.price.astype(float).to_numpy() / tick).astype(np.int64),
        "snap_qtys": snap.quantity.astype(float).to_numpy(),
        "u_tb": u_tb,
        "u_side": upd.side.to_numpy(),
        "u_idx": np.rint(upd.price.astype(float).to_numpy() / tick).astype(np.int64),
        "u_price": upd.price.astype(float).to_numpy(),
        "u_qty": upd.quantity.astype(float).to_numpy(),
        "tr_tb": tr_tb,
        "tr_price": tr.price.astype(float).to_numpy(),
        "tr_qty": tr.quantity.astype(float).to_numpy(),
        "tr_bm": tr.is_buyer_maker.to_numpy(),
        "mk_tb": mk.event_time.to_numpy(),
        "mk_mark": pd.to_numeric(mk.mark_price, errors="coerce").to_numpy(dtype=float),
        "mk_fund": pd.to_numeric(mk.funding_rate, errors="coerce").to_numpy(dtype=float),
        "mk_nft": mk.next_funding_time.to_numpy(),
    }
    return prep


def _feed_events(machine, prep, end_ms, bi=0, ti=0, mi=0):
    """Feed all events with timestamp <= end_ms (absolute ms)."""
    u_tb, tr_tb, mk_tb = prep["u_tb"], prep["tr_tb"], prep["mk_tb"]
    n_u = int(np.searchsorted(u_tb, end_ms, side="right"))
    n_t = int(np.searchsorted(tr_tb, end_ms, side="right"))
    n_m = int(np.searchsorted(mk_tb, end_ms, side="right"))
    while bi < n_u or ti < n_t or mi < n_m:
        cands = []
        if bi < n_u:
            cands.append((u_tb[bi], 0))
        if ti < n_t:
            cands.append((tr_tb[ti], 1))
        if mi < n_m:
            cands.append((mk_tb[mi], 2))
        _, which = min(cands)
        if which == 0:
            machine.apply_update(int(u_tb[bi]), prep["u_side"][bi], int(prep["u_idx"][bi]),
                                 float(prep["u_qty"][bi]), float(prep["u_price"][bi]))
            bi += 1
        elif which == 1:
            machine.apply_trade(int(tr_tb[ti]), float(prep["tr_price"][ti]),
                                float(prep["tr_qty"][ti]), bool(prep["tr_bm"][ti]))
            ti += 1
        else:
            machine.apply_mark(int(mk_tb[mi]), float(prep["mk_mark"][mi]),
                               float(prep["mk_fund"][mi]), int(prep["mk_nft"][mi]))
            mi += 1
    return bi, ti, mi


def _write_output(machine, prep, start_ms, out_root, t0, extra_marks=(),
                  external_feeds=None, chain=True, skipped=False):
    symbol, date, hour = prep["symbol"], prep["date"], prep["hour"]
    out_dir = Path(out_root) / symbol / date
    out_parquet = out_dir / f"{hour}.parquet"
    out_meta = out_dir / f"{hour}.meta.json"

    columns = list(GRID_COLUMNS) + list(FACTOR_NAMES) + list(AUX_COLUMNS)
    data = {c: np.empty(3600, dtype=np.float64) for c in columns if c != "sec"}
    sec_out = np.empty(3600, dtype=np.int64)
    for s in range(3600):
        blk = machine._blk(s)
        sec_out[s] = blk["sec"]
        for c in data:
            data[c][s] = blk.get(c, np.nan)
    df = pd.DataFrame(data, columns=list(data.keys()))
    df.insert(0, "sec", sec_out)

    internal = ("E1", "E2", "E3", "E4", "E5", "E6", "E7")
    if symbol.startswith("BTC"):
        internal = ("E5", "E6", "E7")
    complete = np.ones(3600, dtype=bool)
    for name in FACTOR_NAMES:
        if name in internal:
            continue
        complete &= np.isfinite(df[name].to_numpy())
    df["complete"] = complete

    anchor_rows = []
    for mark in sorted(extra_marks):
        row = machine.query(int(mark))
        anchor_rows.append({"mark_ms": int(mark),
                            **{c: row[c] for c in FACTOR_NAMES},
                            "mid": row["mid"], "best_bid": row["best_bid"],
                            "best_ask": row["best_ask"]})

    out_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), out_parquet)
    if anchor_rows:
        pq.write_table(pa.Table.from_pandas(pd.DataFrame(anchor_rows), preserve_index=False),
                       out_dir / f"{hour}.anchors.parquet")
    if chain:
        with open(_state_path(out_root, symbol, date, hour), "wb") as f:
            pickle.dump(machine.export_state(), f, protocol=pickle.HIGHEST_PROTOCOL)

    elapsed = time.time() - t0
    nan_counts = {name: int((~np.isfinite(df[name].to_numpy())).sum()) for name in FACTOR_NAMES}
    meta = {
        "status": "ok",
        "spec_version": SPEC_VERSION,
        "code_sha256": _code_sha256(),
        "symbol": symbol,
        "date": date,
        "hour": hour,
        "hour_start_ms": start_ms,
        "tick_size": prep["tick"],
        "sources": prep["sources"],
        "gates": prep["gates"],
        "params": {
            "value": prep["params"].__dict__,
            "source": prep["params_source"],
            "provisional": prep["params_source"] == "provisional",
        },
        "chain": {"enabled": chain,
                  "state": "exported" if chain else "off"},
        "external_feeds": external_feeds or {"btc": symbol.startswith("BTC"),
                                             "market": False},
        "rows": int(len(df)),
        "complete_rows": int(complete.sum()),
        "nan_counts": nan_counts,
        "build_seconds": elapsed,
        "peak_rss_mb": _peak_rss_mb(),
        "python": sys.version.split()[0],
    }
    tmp = out_meta.with_suffix(".tmp")
    tmp.write_text(json.dumps(meta, indent=2, default=float))
    tmp.replace(out_meta)
    return meta


def build_hour(symbol, date, hour, base_dir=RAW_BASE, out_root=OUT_ROOT,
               force=False, extra_marks=(), chain=True, thresholds=None,
               capture_min_rate=None):
    """Build one hour of 1s grid features for one symbol.

    Resume: an existing parquet + meta with status "ok" is skipped unless
    force=True. Gate failures write meta with status "rejected" and raise
    ValidationError. With chain=True (default) the previous hour's window-tail
    state is imported so rolling windows cover the hour boundary; E5-E7 stay
    NaN here (they need the multi-symbol lockstep build).
    """
    t0 = time.time()
    symbol = symbol.upper()
    hour = f"{int(hour):02d}"
    start_ms = hour_start_ms(date, hour)
    out_dir = Path(out_root) / symbol / date
    out_parquet = out_dir / f"{hour}.parquet"
    out_meta = out_dir / f"{hour}.meta.json"

    if out_parquet.exists() and out_meta.exists() and not force:
        meta = json.loads(out_meta.read_text())
        if meta.get("status") == "ok":
            meta["skipped_existing"] = True
            return meta

    frozen = thresholds if thresholds is not None else load_thresholds()

    def _write_meta_rejected(meta):
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp = out_meta.with_suffix(".tmp")
        tmp.write_text(json.dumps(meta, indent=2, default=float))
        tmp.replace(out_meta)

    prep = _prepare_symbol_hour(symbol, date, hour, base_dir, frozen, _write_meta_rejected,
                                capture_min_rate=capture_min_rate)
    machine = FactorMachine(symbol, prep["tick"], start_ms, prep["params"])
    machine.apply_snapshot(prep["snap_sides"], prep["snap_idxs"], prep["snap_qtys"])
    if chain:
        state = _load_chain_state(out_root, symbol, date, hour)
        if state is not None:
            machine.import_state(state)
        else:
            print(f"[{symbol} {date}/{hour}] no chain state (segment-head start)",
                  flush=True)

    _feed_events(machine, prep, start_ms + 3599 * 1000 + 999)
    machine.finalize_upto(3599)
    return _write_output(machine, prep, start_ms, out_root, t0,
                         extra_marks=extra_marks, chain=chain)


def build_hour_multi(symbols, date, hour, base_dir=RAW_BASE, out_root=OUT_ROOT,
                     force=False, chain=True, thresholds=None,
                     on_reject="skip", capture_min_rate=None):
    """Lockstep multi-symbol hour build (E-group wiring).

    All symbols advance grid second by grid second together. After each
    second s is finalized on every machine: the market mid (equal-weight mean
    of available symbol mids) and the BTC 1s grid (mid, buy, sell) are fed to
    every machine, then E1-E11 are recomputed for sec s. Symbols failing the
    pipeline gates are excluded (on_reject="skip") or abort the hour
    (on_reject="raise").
    """
    t0 = time.time()
    hour = f"{int(hour):02d}"
    start_ms = hour_start_ms(date, hour)
    symbols = [s.upper() for s in symbols]
    frozen = thresholds if thresholds is not None else load_thresholds()
    preps = {}
    for sym in symbols:
        out_dir = Path(out_root) / sym / date
        if (out_dir / f"{hour}.parquet").exists() and (out_dir / f"{hour}.meta.json").exists() and not force:
            continue
        def _write_rejected(meta, _sym=sym):
            d = Path(out_root) / _sym / date
            d.mkdir(parents=True, exist_ok=True)
            mp = d / f"{hour}.meta.json"
            mp.with_suffix(".tmp").write_text(json.dumps(meta, indent=2, default=float))
            mp.with_suffix(".tmp").replace(mp)

        try:
            preps[sym] = _prepare_symbol_hour(sym, date, hour, base_dir, frozen,
                                              _write_rejected,
                                              capture_min_rate=capture_min_rate)
        except ValidationError as exc:
            if on_reject == "raise":
                raise
            print(f"[multi {sym} {date}/{hour}] REJECTED: {exc}", flush=True)
    if not preps:
        return {}

    machines = {}
    for sym, prep in preps.items():
        m = FactorMachine(sym, prep["tick"], start_ms, prep["params"])
        m.apply_snapshot(prep["snap_sides"], prep["snap_idxs"], prep["snap_qtys"])
        if chain:
            state = _load_chain_state(out_root, sym, date, hour)
            if state is not None:
                m.import_state(state)
        machines[sym] = m
        preps[sym]["_cursors"] = [0, 0, 0]

    btc_sym = next((s for s in machines if s.startswith("BTC")), None)
    for s in range(3600):
        mark = start_ms + s * 1000
        for sym, m in machines.items():
            cur = preps[sym]["_cursors"]
            cur[0], cur[1], cur[2] = _feed_events(m, preps[sym], mark, *cur)
            m.finalize_upto(s)
        mids = [m._blk(s)["mid"] for m in machines.values()]
        valid = [x for x in mids if x == x]
        mkt_mid = float(np.mean(valid)) if valid else float("nan")
        btc_blk = machines[btc_sym]._blk(s) if btc_sym else None
        for sym, m in machines.items():
            if btc_sym and sym != btc_sym and btc_blk is not None:
                m.feed_btc_second(s, btc_blk["mid"], btc_blk["buy_n_1s"],
                                  btc_blk["sell_n_1s"])
            m.feed_market_mid(s, mkt_mid)
            m.recompute_env(s)

    metas = {}
    for sym, m in machines.items():
        metas[sym] = _write_output(
            m, preps[sym], start_ms, out_root, t0, chain=chain,
            external_feeds={"btc": btc_sym is not None, "market": True})
    return metas


def main(argv=None):
    ap = argparse.ArgumentParser(description="Kimi v3 P1.5 hour feature builder")
    ap.add_argument("--symbol", default=None, help="one symbol, or comma list")
    ap.add_argument("--symbols", default=None, help="comma list; enables lockstep multi build")
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--hour", default=None, help="HH, or 'all' with --day")
    ap.add_argument("--day", action="store_true", help="run hours 00..23")
    ap.add_argument("--base", default=str(RAW_BASE))
    ap.add_argument("--out", default=str(OUT_ROOT))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-chain", action="store_true", help="disable cross-hour window chaining")
    ap.add_argument("--capture-min", type=float, default=None,
                    help="override the trade capture-rate gate (default from schema)")
    args = ap.parse_args(argv)
    if not (args.symbol or args.symbols):
        ap.error("one of --symbol / --symbols is required")

    symbols = [s.strip().upper() for s in (args.symbols or args.symbol).split(",")]
    multi = len(symbols) > 1
    hours = [f"{h:02d}" for h in range(24)] if args.day else [args.hour]
    if args.day and args.hour not in (None, "all"):
        hours = [args.hour]
    failures = 0
    for hour in hours:
        try:
            if multi:
                metas = build_hour_multi(symbols, args.date, hour, base_dir=args.base,
                                         out_root=args.out, force=args.force,
                                         chain=not args.no_chain,
                                         capture_min_rate=args.capture_min)
                for sym, meta in metas.items():
                    print(f"[{sym} {args.date}/{hour}] ok rows={meta['rows']} "
                          f"complete={meta['complete_rows']} "
                          f"capture={meta['gates']['capture_rate']:.4f} "
                          f"{meta['build_seconds']:.1f}s", flush=True)
            else:
                meta = build_hour(args.symbol, args.date, hour, base_dir=args.base,
                                  out_root=args.out, force=args.force,
                                  chain=not args.no_chain,
                                  capture_min_rate=args.capture_min)
                state = "skipped" if meta.get("skipped_existing") else \
                    f"ok rows={meta['rows']} complete={meta['complete_rows']} " \
                    f"capture={meta['gates']['capture_rate']:.4f} " \
                    f"{meta['build_seconds']:.1f}s rss={meta['peak_rss_mb']:.0f}MB"
                print(f"[{args.symbol} {args.date}/{hour}] {state}", flush=True)
        except (ValidationError, FileNotFoundError) as exc:
            failures += 1
            print(f"[{args.symbol} {args.date}/{hour}] REJECTED: {exc}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
