"""Compute the training-frozen B6/C6 threshold table (V3_SPEC §3, P1.5).

Pools cancel-event notionals / quantities and per-trade notionals over a
sample of the training segment (default 2026-08-05 ~ 2026-09-08) and writes
app/prediction/thresholds.json via app.prediction.thresholds.save_thresholds.

Sampling: ``--hours-per-day K`` takes K evenly spread hours per coin per day
(hour index rotates by day so intraday sessions are covered). The sample is a
percentile estimate; regenerate with a denser sample (or the full segment)
before final P2 evaluation if the table is used for frozen comparisons.

Usage (repo root):
    python3 tools/factors/compute_v3_thresholds.py --hours-per-day 1
"""
import argparse
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.prediction.builder import RAW_BASE, _replay_book  # noqa: E402
from app.prediction.builder import load_hour  # noqa: E402
from app.prediction.schema import FactorParams  # noqa: E402
from app.prediction.thresholds import save_thresholds  # noqa: E402

ALL_COINS = ["BTCUSDC", "SOLUSDC", "SUIUSDC", "UNIUSDC", "DOGEUSDC",
             "1000SHIBUSDC", "WIFUSDC", "ETHUSDC", "1000PEPEUSDC", "LTCUSDC",
             "CRVUSDC", "XRPUSDC", "1000BONKUSDC"]


def daterange(start, end):
    d = date.fromisoformat(start)
    last = date.fromisoformat(end)
    while d <= last:
        yield d.isoformat()
        d += timedelta(days=1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--coins", default=",".join(ALL_COINS))
    ap.add_argument("--start", default="2026-08-05")
    ap.add_argument("--end", default="2026-09-08")
    ap.add_argument("--hours-per-day", type=int, default=1)
    ap.add_argument("--base", default=str(RAW_BASE))
    ap.add_argument("--out", default=None, help="default app/prediction/thresholds.json")
    args = ap.parse_args(argv)

    coins = [c.strip().upper() for c in args.coins.split(",")]
    dates = list(daterange(args.start, args.end))
    table = {}
    for coin in coins:
        t0 = time.time()
        cancel_notional = []
        cancel_qty = []
        trade_notional = []
        n_hours = 0
        for di, d in enumerate(dates):
            for k in range(args.hours_per_day):
                hour = f"{(di * 7 + k * (24 // max(args.hours_per_day, 1))) % 24:02d}"
                try:
                    ob, tr, _mk = load_hour(coin, d, hour, args.base)
                except FileNotFoundError:
                    continue
                ob["price"] = np.asarray(ob["price"], dtype=float)
                ob["quantity"] = np.asarray(ob["quantity"], dtype=float)
                tr["price"] = np.asarray(tr["price"], dtype=float)
                tr["quantity"] = np.asarray(tr["quantity"], dtype=float)
                decreases, _ = _replay_book(ob)
                if decreases:
                    cancel_notional.extend(dq * p for _, _, p, dq in decreases)
                    cancel_qty.extend(dq for _, _, _, dq in decreases)
                trade_notional.extend(
                    (tr.price * tr.quantity).to_numpy(dtype=float))
                n_hours += 1
        if not cancel_notional:
            print(f"[{coin}] no data sampled", flush=True)
            continue
        cn = np.asarray(cancel_notional)
        cq = np.asarray(cancel_qty)
        tn = np.asarray(trade_notional)
        params = FactorParams(
            cancel_p90_notional=float(np.percentile(cn, 90)),
            median_cancel_qty=float(np.median(cq)),
            trade_p90_notional=float(np.percentile(tn, 90)) if len(tn) else None)
        table[coin] = params
        print(f"[{coin}] hours={n_hours} cancels={len(cn)} trades={len(tn)} "
              f"p90_cancel={params.cancel_p90_notional:.4f} "
              f"med_qty={params.median_cancel_qty:.6f} "
              f"p90_trade={params.trade_p90_notional:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    meta = {
        "segment": f"{args.start}~{args.end}",
        "hours_per_day": args.hours_per_day,
        "coins": coins,
        "note": "sampled percentile estimate; regenerate denser before final P2 eval",
    }
    out = save_thresholds(table, args.out, meta)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
