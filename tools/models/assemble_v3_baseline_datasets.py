"""Assemble the baseline-comparison datasets for the PEPE sprint (P3).

Reads the recomputed frozen-15 baseline hour parquets (this repo,
tools/factors/compute_v3_baseline15.py), joins them with the v3 labels, and writes:

- pepe-14d.base15.parquet      complete-case rows on the frozen 15 features
- pepe-14d.d78.parquet         the D7/D8 rv pair alone (the "common sense"
                               share of v3's T2 performance)
- pepe-14d.evalframe.parquet   the SHARED evaluation frame: test-window rows
                               where BOTH feature sets are complete, carrying
                               all 58 feature columns + labels; every model
                               configuration is scored on these identical
                               rows so per-hour rankICs pair exactly.

Also writes pepe-14d.baseline15.report.json with coverage statistics.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

ASSEMBLED = ROOT / "data/prediction-v3/assembled"
BASELINE = ROOT / "data/prediction-v3/baseline15"
SYM = "1000PEPEUSDC"
BTC = "BTCUSDC"

ID_COLUMNS = ["sec", "date", "mid", "complete", "capture_rate", "T1", "T2", "T3"]


def read_hours(root, symbol):
    frames = []
    for d in sorted((root / symbol).iterdir()):
        for h in sorted(d.glob("*.parquet")):
            frames.append(pq.ParquetFile(h).read().to_pandas())
    return pd.concat(frames, ignore_index=True)


def main():
    lv = pq.ParquetFile(ASSEMBLED / "pepe-14d.labelvalid.parquet") \
        .read().to_pandas()
    v3 = pq.ParquetFile(ASSEMBLED / "pepe-14d.parquet").read().to_pandas()

    own = read_hours(BASELINE, SYM)
    btc = read_hours(BASELINE, BTC)[["sec"] + [c for c in
                                               ("btc_mid_ret_1s", "btc_mid_ret_5s",
                                                "btc_ofi_1s", "btc_ofi_5s",
                                                "btc_flow_imb_1s")]]
    base = own.merge(btc, on="sec", how="left")
    report = {"own_rows": int(len(own)), "btc_rows": int(len(btc)),
              "joined_rows": int(len(base)),
              "btc_nan_rows": int(base["btc_mid_ret_1s"].isna().sum())}

    labels = lv[ID_COLUMNS]
    full = labels.merge(base.drop(columns=["mid"], errors="ignore"),
                        on="sec", how="left")
    base_cols = [c for c in full.columns if c not in ID_COLUMNS]
    base_ok = full.dropna(subset=base_cols)
    report["base15_complete_rows"] = int(len(base_ok))
    pq.write_table(pa.Table.from_pandas(
        base_ok[ID_COLUMNS + base_cols], preserve_index=False),
        ASSEMBLED / "pepe-14d.base15.parquet")

    v3_feats = [c for c in v3.columns if c not in ID_COLUMNS]
    d78 = v3[ID_COLUMNS + ["D7", "D8"]].dropna(subset=["D7", "D8"])
    report["d78_complete_rows"] = int(len(d78))
    pq.write_table(pa.Table.from_pandas(d78, preserve_index=False),
                   ASSEMBLED / "pepe-14d.d78.parquet")

    # shared evaluation frame: v3 complete-case secs where baseline15 is
    # complete too, carrying both feature sets
    inter = v3.merge(base, on="sec", how="inner")
    inter = inter.dropna(subset=v3_feats + base_cols)
    report["evalframe_rows_total"] = int(len(inter))
    report["evalframe_rows_test"] = int((inter["date"] >= "2026-10-01").sum())
    pq.write_table(pa.Table.from_pandas(inter, preserve_index=False),
                   ASSEMBLED / "pepe-14d.evalframe.parquet")

    (ASSEMBLED / "pepe-14d.baseline15.report.json").write_text(
        json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
