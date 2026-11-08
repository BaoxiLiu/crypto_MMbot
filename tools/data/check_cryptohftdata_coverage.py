#!/usr/bin/env python3
"""Read-only coverage check for the hourly CryptoHFTData archive.

Checks every expected (date, hour, symbol, kind) slot under
$QUANT_DATA_ROOT/crypto/raw/hft/binance_futures, reports missing slots, per-date
coverage and a small parquet decode sample. Never writes inside raw/ and
never downloads anything.
"""
import argparse
import datetime as dt
import json
import random
from pathlib import Path

SYMBOLS = ["BTCUSDC", "SOLUSDC", "SUIUSDC", "UNIUSDC", "DOGEUSDC", "1000SHIBUSDC", "WIFUSDC",
           "ETHUSDC", "1000PEPEUSDC", "LTCUSDC", "CRVUSDC", "XRPUSDC", "1000BONKUSDC"]
KINDS = ["orderbook", "trades", "mark_price"]


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/cryptohftdata")
    p.add_argument("--start", default="2026-08-05")
    p.add_argument("--end", default="2026-10-05", help="exclusive")
    p.add_argument("--sample", type=int, default=30, help="parquet decode sample size")
    p.add_argument("--seed", type=int, default=20261008)
    p.add_argument("--out", default="data/cryptohftdata/reports/coverage-20261008.json")
    a = p.parse_args(argv)
    a.symbols = SYMBOLS
    return a


def scan(root, start, end, symbols):
    base = Path(root) / "raw" / "binance_futures"
    present, missing, sizes = [], [], {}
    day = start
    while day < end:
        for hour in range(24):
            for symbol in symbols:
                for kind in KINDS:
                    path = base / str(day) / f"{hour:02d}" / f"{symbol}_{kind}.parquet"
                    if path.is_file():
                        present.append(str(path.relative_to(base)))
                        sizes[path.stat().st_size] = sizes.get(path.stat().st_size, 0) + 1
                    else:
                        missing.append(f"{day}/{hour:02d}/{symbol}_{kind}.parquet")
        day += dt.timedelta(days=1)
    return base, present, missing


def decode_sample(base, files, count, seed):
    import pyarrow.parquet as pq
    rng = random.Random(seed)
    picks = rng.sample(files, min(count, len(files)))
    out, bad = [], []
    for rel in sorted(picks):
        path = base / rel
        symbol, kind = Path(rel).name[: -len(".parquet")].split("_", 1)
        try:
            pf = pq.ParquetFile(path)
            rows = pf.metadata.num_rows
            cols = set(pf.schema_arrow.names)
            first = pf.read_row_group(0, columns=["symbol"]).column("symbol") if rows else None
            values = set(first.to_pylist()) if first is not None else set()
            ok = rows > 0 and (not values or values == {symbol})
            out.append({"file": rel, "rows": rows, "columns": len(cols), "symbol_ok": ok})
            if not ok:
                bad.append(rel)
        except Exception as error:  # decode failure is a real defect, keep it
            bad.append(rel)
            out.append({"file": rel, "error": type(error).__name__ + ": " + str(error)[:160]})
    return out, bad


def main(argv=None):
    a = parse_args(argv)
    start = dt.date.fromisoformat(a.start)
    end = dt.date.fromisoformat(a.end)
    base, present, missing = scan(a.root, start, end, a.symbols)
    days = (end - start).days
    expected = days * 24 * len(a.symbols) * len(KINDS)
    per_day = {}
    for rel in present:
        per_day[rel.split("/")[0]] = per_day.get(rel.split("/")[0], 0) + 1
    sample, bad = decode_sample(base, present, a.sample, a.seed)
    payload = {
        "checked_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "root": str(Path(a.root).resolve()),
        "window": {"start": str(start), "end_exclusive": str(end)},
        "symbols": a.symbols, "kinds": KINDS,
        "expected_slots": expected, "present": len(present), "missing": len(missing),
        "coverage": round(len(present) / expected, 6) if expected else 0.0,
        "per_day_expected": 24 * len(a.symbols) * len(KINDS),
        "per_day_present": dict(sorted(per_day.items())),
        "missing_files": missing,
        "decode_sample": sample,
        "decode_sample_bad": bad,
        "total_bytes": sum((base / rel).stat().st_size for rel in present),
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: payload[k] for k in
                      ("expected_slots", "present", "missing", "coverage", "total_bytes",
                       "decode_sample_bad")}, ensure_ascii=False))
    if missing:
        print("missing slots:")
        for rel in missing:
            print("  ", rel)
    return 1 if (missing or bad) else 0


if __name__ == "__main__":
    raise SystemExit(main())
