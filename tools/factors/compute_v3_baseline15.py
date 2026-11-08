"""Recompute the frozen 15 baseline factors on the v3 1s grid (P3 baseline).

Frozen set (V3_SPEC §0.5; definitions from doc/factors/BASELINE15.md
and doc/models.md):
- old 9 (forecast_dataset.FEATURES): spread_bp, l1_imbalance,
  microprice_offset_bp, ofi, buy_qty, sell_qty, signed_notional, mid_return,
  book_age_ms — window L=1s, Cont-style touch OFI, aggressive flow by
  is_buyer_maker.
- v2 six (R006): flow_imbalance_1s (P014 rule: zero trades -> 0),
  btc_mid_ret_1s/5s, btc_ofi_1s/5s, btc_flow_imb_1s (author candidates; the
  flow ratio uses the same zero->0 rule, documented here).

Faithful-vs-old differences (recorded, no silent drift):
- Clock/grid: old chain sampled on local = exchange + 5ms inside its own
  certified segments; here everything is recomputed on the v3 1s grid (state
  at second mark = last event tb <= mark, flows over (mark-1000, mark]).
  Same causal standard, different transport clock. Targets come from the v3
  assembler, so both sides of the comparison share one clock.
- book_age_ms: last DEPTH RECEIPT time; we use the max received_time of any
  orderbook row (snapshot or update) <= mark. The old line excluded stale
  rows (>3s) at dataset level; the v3 comparison keeps rows, so the stale-row
  exclusion is NOT reproduced (reported as a coverage difference).
- book_age_ms floor: receipt jitter can make (mark - last receipt) slightly
  negative at second 0 (the snapshot block finishes arriving just after the
  mark); we clip at 0. Sub-second effect, far inside the old 3s stale bound.
- Cross-file state: the old cache carried cumulative OFI/flow prefixes and
  book state across files within a segment; we chain a small JSON state
  (last grid mids, OFI prefix tail, un-finalized trade tail, last depth
  receipt) across hours within a day run and start fresh at day boundaries.

Outputs per hour: data/prediction-v3/baseline15/<SYM>/<date>/<HH>.parquet
(columns sec + the symbol's own columns; BTC columns come from a BTC run and
are joined by sec at assembly).
"""
import argparse
import json
import sys
from bisect import bisect_right
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.prediction.book import BookState  # noqa: E402
from app.prediction.builder import infer_tick_size, load_hour  # noqa: E402

OLD9 = ("spread_bp", "l1_imbalance", "microprice_offset_bp", "ofi",
        "buy_qty", "sell_qty", "signed_notional", "mid_return",
        "book_age_ms")
OWN_COLUMNS = OLD9 + ("flow_imbalance_1s",)
BTC_COLUMNS = ("btc_mid_ret_1s", "btc_mid_ret_5s", "btc_ofi_1s",
               "btc_ofi_5s", "btc_flow_imb_1s")


class BaselineMachine:
    """Streaming frozen-9 (+ own flow ratio) computer on the v3 1s grid."""

    def __init__(self, tick_size, hour_start_ms, state=None):
        self.tick = float(tick_size)
        self.hour_start_ms = int(hour_start_ms)
        self.book = BookState(self.tick)
        self._ofi_t = []
        self._ofi_p = []
        self._ofi_base = 0.0
        self._prev_touch = None
        self._last_recv_ms = None
        self._cur = [0.0, 0.0, 0.0, 0.0]   # buy_q, sell_q, buy_n, sell_n
        self._next_sec = 0
        self._last_event_ms = -1
        self._mid_hist = deque(maxlen=8)   # (abs mark, mid) finalized secs
        self._pending_tail = None          # imported unfinalized trade tail
        self._rows = []
        if state:
            self.import_state(state)

    # -------------------------------------------------------------- state

    def export_state(self):
        return {"mid_hist": [[int(m), v] for m, v in self._mid_hist],
                "ofi_tail_t": [int(x) for x in self._ofi_t[-4:]],
                "ofi_tail_p": [float(x) for x in self._ofi_p[-4:]],
                "tail_flows": self._cur,
                "last_recv_ms": self._last_recv_ms}

    def import_state(self, state):
        hist = state.get("mid_hist") or []
        self._mid_hist = deque(hist[-8:], maxlen=8)
        self._ofi_t = [int(x) for x in (state.get("ofi_tail_t") or [])]
        self._ofi_p = [float(x) for x in (state.get("ofi_tail_p") or [])]
        self._ofi_base = self._ofi_p[-1] if self._ofi_p else 0.0
        tail = state.get("tail_flows")
        self._pending_tail = list(tail) if tail else None
        self._last_recv_ms = state.get("last_recv_ms")

    # -------------------------------------------------------------- feeds

    def _advance(self, t):
        if t < self._last_event_ms:
            raise ValueError(f"out-of-order event t={t}")
        self._last_event_ms = t
        while self.hour_start_ms + self._next_sec * 1000 < t:
            self._finalize_sec(self._next_sec)
            self._next_sec += 1

    def _start_sec(self, s):
        """Open second s: fold the imported previous-hour tail into it."""
        if s == 0 and self._pending_tail is not None:
            for i in range(4):
                self._cur[i] += self._pending_tail[i]
            self._pending_tail = None

    def apply_snapshot(self, sides, idxs, qtys, recv_ms):
        self.book.load_snapshot(sides, idxs, qtys)
        self._prev_touch = self._touch_sample()
        self._last_recv_ms = recv_ms

    def apply_update(self, tb, side, idx, qty, recv_ms):
        self._advance(tb)
        self.book.apply(side, idx, qty)
        self._last_recv_ms = recv_ms
        self._update_touch(tb)

    def apply_trade(self, tb, price, qty, is_buyer_maker):
        self._advance(tb)
        notional = price * qty
        if is_buyer_maker:   # seller is taker
            self._cur[1] += qty
            self._cur[3] += notional
        else:
            self._cur[0] += qty
            self._cur[2] += notional

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
        base = self._ofi_p[-1] if self._ofi_p else self._ofi_base
        self._ofi_t.append(tb)
        self._ofi_p.append(base + ch)
        self._prev_touch = cur

    def _ofi_at(self, t):
        i = bisect_right(self._ofi_t, t) - 1
        if i < 0:
            return self._ofi_base
        return self._ofi_p[i]

    def _mid_at(self, t):
        """Mid at grid mark t (absolute ms), or None if unknown."""
        if (t - self.hour_start_ms) % 1000 != 0:
            return None
        for mark, mid in reversed(self._mid_hist):
            if mark == t:
                return mid
        return None

    # ------------------------------------------------------------ finalize

    def _instant(self, out):
        bbi, bq = self.book.best_bid()
        bai, aq = self.book.best_ask()
        if bbi is None or bai is None:
            for k in ("spread_bp", "l1_imbalance", "microprice_offset_bp"):
                out[k] = np.nan
            return None
        bb = round(bbi * self.tick, 10)
        ba = round(bai * self.tick, 10)
        mid = (bb + ba) / 2.0
        out["spread_bp"] = (ba - bb) / mid * 1e4
        out["l1_imbalance"] = (bq - aq) / (bq + aq) if bq + aq > 0 else np.nan
        micro = (bb * aq + ba * bq) / (bq + aq)
        out["microprice_offset_bp"] = (micro - mid) / mid * 1e4
        return mid

    def _finalize(self, s):
        self._start_sec(s)
        mark = self.hour_start_ms + s * 1000
        out = {k: np.nan for k in OWN_COLUMNS}
        mid = self._instant(out)
        bq_, sq_, bn_, sn_ = self._cur
        out["buy_qty"] = bq_
        out["sell_qty"] = sq_
        out["signed_notional"] = bn_ - sn_
        tot = bn_ + sn_
        out["flow_imbalance_1s"] = (bn_ - sn_) / tot if tot > 0 else 0.0
        out["ofi"] = self._ofi_at(mark) - self._ofi_at(mark - 1000)
        if mid is not None:
            pm = self._mid_at(mark - 1000)
            if pm is not None and pm > 0:
                out["mid_return"] = mid / pm - 1.0
            self._mid_hist.append((mark, mid))
        if self._last_recv_ms is not None:
            # receipt jitter can make mark - recv slightly negative at sec 0
            # (snapshot block receipt tail); clip, documented in module notes
            out["book_age_ms"] = max(0.0, float(mark - self._last_recv_ms))
        self._cur = [0.0, 0.0, 0.0, 0.0]
        return out

    def _finalize_sec(self, s):
        row = self._finalize(s)
        row["sec"] = (self.hour_start_ms + s * 1000) // 1000
        self._rows.append(row)

    def finalize_upto(self, last_sec):
        while self._next_sec <= last_sec:
            self._finalize_sec(self._next_sec)
            self._next_sec += 1
        rows = self._rows
        self._rows = []
        return rows


class BtcBaselineMachine(BaselineMachine):
    """BTC grid feeder: only the five BTC v2 columns."""

    def _finalize(self, s):
        base = super()._finalize(s)
        mark = self.hour_start_ms + s * 1000
        out = {k: np.nan for k in BTC_COLUMNS}
        mid = self._mid_hist[-1][1] if self._mid_hist and \
            self._mid_hist[-1][0] == mark else None
        pm1 = self._mid_at(mark - 1000)
        pm5 = self._mid_at(mark - 5000)
        if mid is not None and pm1 is not None and pm1 > 0:
            out["btc_mid_ret_1s"] = mid / pm1 - 1.0
        if mid is not None and pm5 is not None and pm5 > 0:
            out["btc_mid_ret_5s"] = mid / pm5 - 1.0
        out["btc_ofi_1s"] = base["ofi"]
        out["btc_ofi_5s"] = self._ofi_at(mark) - self._ofi_at(mark - 5000)
        out["btc_flow_imb_1s"] = base["flow_imbalance_1s"]
        return out


def _prepare(ob, tr, tick):
    snap = ob[ob.event_type == "snapshot"]
    upd = ob[ob.event_type == "update"]
    u_tb = np.where(upd["transaction_time"].notna(), upd["transaction_time"],
                    upd["received_time"] // 1_000_000).astype(np.int64)
    return {
        "snap_sides": snap.side.to_numpy(),
        "snap_idxs": np.rint(snap.price.astype(float).to_numpy() / tick).astype(np.int64),
        "snap_qtys": snap.quantity.astype(float).to_numpy(),
        "snap_recv": int(snap.received_time.max() // 1_000_000) if len(snap) else None,
        "u_tb": u_tb,
        "u_side": upd.side.to_numpy(),
        "u_idx": np.rint(upd.price.astype(float).to_numpy() / tick).astype(np.int64),
        "u_qty": upd.quantity.astype(float).to_numpy(),
        "u_recv": (upd.received_time // 1_000_000).to_numpy().astype(np.int64),
        "tr_tb": tr.trade_time.to_numpy(),
        "tr_price": tr.price.astype(float).to_numpy(),
        "tr_qty": tr.quantity.astype(float).to_numpy(),
        "tr_bm": tr.is_buyer_maker.to_numpy(),
    }


def run_hour(machine, prep):
    machine.apply_snapshot(prep["snap_sides"], prep["snap_idxs"],
                           prep["snap_qtys"], prep["snap_recv"])
    bi = ti = 0
    n_u, n_t = len(prep["u_tb"]), len(prep["tr_tb"])
    while bi < n_u or ti < n_t:
        if ti >= n_t or (bi < n_u and prep["u_tb"][bi] <= prep["tr_tb"][ti]):
            machine.apply_update(int(prep["u_tb"][bi]), prep["u_side"][bi],
                                 int(prep["u_idx"][bi]), float(prep["u_qty"][bi]),
                                 int(prep["u_recv"][bi]))
            bi += 1
        else:
            machine.apply_trade(int(prep["tr_tb"][ti]), float(prep["tr_price"][ti]),
                                float(prep["tr_qty"][ti]), bool(prep["tr_bm"][ti]))
            ti += 1
    return machine.finalize_upto(3599)


def compute_symbol(symbol, dates, base_dir, out_root, btc=False):
    out_root = Path(out_root) / symbol
    for date in dates:
        prev_state = None
        for hour in range(24):
            hh = f"{hour:02d}"
            try:
                ob, tr, _mk = load_hour(symbol, date, hh, base_dir)
            except FileNotFoundError:
                continue
            ob["price"] = pd.to_numeric(ob["price"], errors="coerce")
            ob["quantity"] = pd.to_numeric(ob["quantity"], errors="coerce")
            tr["price"] = pd.to_numeric(tr["price"], errors="coerce")
            tr["quantity"] = pd.to_numeric(tr["quantity"], errors="coerce")
            tick = infer_tick_size(ob.price.head(200000), tr.price)
            start_ms = int(pd.Timestamp(f"{date}T{hh}:00:00", tz="UTC").timestamp() * 1000)
            cls = BtcBaselineMachine if btc else BaselineMachine
            machine = cls(tick, start_ms, prev_state)
            rows = run_hour(machine, _prepare(ob, tr, tick))
            d = out_root / date
            d.mkdir(parents=True, exist_ok=True)
            cols = BTC_COLUMNS if btc else OWN_COLUMNS
            df = pd.DataFrame(rows, columns=["sec"] + list(cols))
            pq.write_table(pa.Table.from_pandas(df, preserve_index=False),
                           d / f"{hh}.parquet")
            prev_state = machine.export_state()
    return out_root


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--dates", required=True, help="comma list YYYY-MM-DD")
    ap.add_argument("--base", default="$QUANT_DATA_ROOT/crypto/raw/hft/binance_futures")
    ap.add_argument("--out", default="data/prediction-v3/baseline15")
    ap.add_argument("--btc", action="store_true", help="emit only BTC v2 columns")
    args = ap.parse_args(argv)
    compute_symbol(args.symbol.upper(), args.dates.split(","), args.base,
                   args.out, btc=args.btc)
    print("done", args.symbol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
