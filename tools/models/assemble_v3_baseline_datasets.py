"""Assemble frozen-15/D7-D8 comparisons on an explicit shared evaluation frame.

Writes baseline15.parquet, d78.parquet, evalframe.parquet and coverage.json.
Input symbol, files, test boundary and output directory are always explicit.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

ID_COLUMNS = ["sec", "date", "mid", "complete", "capture_rate", "T1", "T2", "T3"]


def read_hours(root, symbol):
    frames = []
    for d in sorted((root / symbol).iterdir()):
        for h in sorted(d.glob("*.parquet")):
            frames.append(pq.ParquetFile(h).read().to_pandas())
    return pd.concat(frames, ignore_index=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('labels','v3','baseline','out'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--symbol',required=True)
    parser.add_argument('--btc-symbol',default='BTCUSDC')
    parser.add_argument('--test-start',required=True)
    args=parser.parse_args()
    pd.Timestamp(args.test_start)  # validate before allocating output
    args.out.mkdir(parents=True,exist_ok=False)
    lv = pq.ParquetFile(args.labels) \
        .read().to_pandas()
    v3 = pq.ParquetFile(args.v3).read().to_pandas()

    own = read_hours(args.baseline, args.symbol)
    btc = read_hours(args.baseline, args.btc_symbol)[["sec"] + [c for c in
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
        args.out / "baseline15.parquet")

    v3_feats = [c for c in v3.columns if c not in ID_COLUMNS]
    d78 = v3[ID_COLUMNS + ["D7", "D8"]].dropna(subset=["D7", "D8"])
    report["d78_complete_rows"] = int(len(d78))
    pq.write_table(pa.Table.from_pandas(d78, preserve_index=False),
                   args.out / "d78.parquet")

    # shared evaluation frame: v3 complete-case secs where baseline15 is
    # complete too, carrying both feature sets
    inter = v3.merge(base, on="sec", how="inner")
    inter = inter.dropna(subset=v3_feats + base_cols)
    report["evalframe_rows_total"] = int(len(inter))
    report["evalframe_rows_test"] = int((inter["date"] >= args.test_start).sum())
    pq.write_table(pa.Table.from_pandas(inter, preserve_index=False),
                   args.out / "evalframe.parquet")

    (args.out / "coverage.json").write_text(
        json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
