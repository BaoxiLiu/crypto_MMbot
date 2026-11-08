"""Assemble the Kimi v3 modeling dataset from hour feature parquets (P2/P3).

Labels (V3_SPEC §1, mid = the 1s grid mid, bp = 1e4):
- T1 = (mid[t+1s]/mid[t] - 1) x 1e4
- T3 = (mid[t+5s]/mid[t] - 1) x 1e4
- T2 = std(d log mid, (t, t+30s], ddof=1) x 1e4  (30 forward 1s returns)

Assembly rules (V3_SPEC §7):
- Labels are computed per day on the continuous 1s mid grid; any label whose
  forward window is incomplete (day tail, rejected-hour gap, NaN mid, or a
  data gap inside the window) is NaN and the ROW IS DROPPED (label must
  exist; no interpolation across gaps).
- E1-E7 are excluded in this single-coin sprint: E5-E7 have no market feed,
  and E1-E4 turn out all-NaN in single-coin builds too (they need the BTC
  feed that only the multi-symbol lockstep provides). Any other feature that
  is entirely NaN across the sample is auto-excluded and reported.
- Remaining feature-NaN rows are dropped as incomplete cases (never zero
  filled); the loss rate is reported per factor.
- capture_rate of the source hour is kept as an auxiliary column.

Outputs one parquet plus a <out>.report.json statistics report.
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

from app.prediction.schema import FACTOR_NAMES  # noqa: E402

DEFAULT_EXCLUDE = ("E1", "E2", "E3", "E4", "E5", "E6", "E7")

# raw per-second columns needed for the class-3/class-4 volume-weighted
# targets (V3_SPEC v1.3; doc/models.md §4.1)
AUX_COLUMNS = ("buy_n_1s", "sell_n_1s", "buy_qty_1s", "sell_qty_1s",
               "buy_vwap_1s", "sell_vwap_1s")
VWAP_TARGETS = ("R_vwap_5", "R_vwap_30", "R_askw_5", "R_askw_30",
                "R_bidw_5", "R_bidw_30", "R_twap_5", "R_twap_30")


def _labels_vwap(mids, buy_n, sell_n, buy_qty, sell_qty, buy_vwap, sell_vwap):
    """Class-3 (volume-weighted path) and class-4 (TWAP) targets.

    All windows are (t, t+T] on the 1s grid; only mid(t) from the decision
    second itself. NaN rules: window mid gap / tail -> NaN (no interpolation
    across gaps, same policy as T2); zero total weight on the corresponding
    side -> NaN (rows dropped, never zero-filled). Stored in bp (x1e4) like
    T1/T3 for scale comparability.
    """
    n = len(mids)
    m = mids
    valid = np.isfinite(m) & (m > 0)
    bad = np.concatenate(([0], np.cumsum(~valid)))

    vol = np.nan_to_num(buy_n) + np.nan_to_num(sell_n)
    qa = np.nan_to_num(buy_qty)
    qb = np.nan_to_num(sell_qty)
    pa_qty = np.nan_to_num(buy_vwap * buy_qty)   # NaN vwap pairs with 0 qty
    pb_qty = np.nan_to_num(sell_vwap * sell_qty)

    out = {}
    for T in (5, 30):
        end = n - T                      # t valid for 0 <= t < n - T
        tag = np.full(n, np.nan)
        vwap = np.full(n, np.nan)
        askw = np.full(n, np.nan)
        bidw = np.full(n, np.nan)
        if end > 0:
            # window elements are x[t+1..t+T] (decision t uses only mid(t))
            win_ok = (bad[T + 1:T + 1 + end] - bad[1:end + 1]) == 0
            base_ok = valid[0:end]
            denom = m[0:end]

            def wsum(x):
                pref = np.concatenate(([0.0], np.cumsum(x)))
                return pref[T + 1:T + 1 + end] - pref[1:end + 1]

            sv = wsum(np.where(valid, vol, 0.0))
            ok = win_ok & base_ok & (sv > 0)
            vwap[0:end][ok] = (wsum(np.where(valid, m * vol, 0.0))[ok]
                               / sv[ok] / denom[ok] - 1.0) * 1e4
            sa = wsum(qa)
            ok_a = win_ok & base_ok & (sa > 0)
            askw[0:end][ok_a] = (wsum(pa_qty)[ok_a] / sa[ok_a] / denom[ok_a]
                                 - 1.0) * 1e4
            sb = wsum(qb)
            ok_b = win_ok & base_ok & (sb > 0)
            bidw[0:end][ok_b] = (wsum(pb_qty)[ok_b] / sb[ok_b] / denom[ok_b]
                                 - 1.0) * 1e4
            st = wsum(np.where(valid, 1.0, 0.0))
            ok_t = win_ok & base_ok & (st >= T)   # all T mids present
            tag[0:end][ok_t] = (wsum(np.where(valid, m, 0.0))[ok_t]
                                / st[ok_t] / denom[ok_t] - 1.0) * 1e4
        out[f"R_vwap_{T}"] = vwap
        out[f"R_askw_{T}"] = askw
        out[f"R_bidw_{T}"] = bidw
        out[f"R_twap_{T}"] = tag
    return out


def daterange(start, end):
    d = pd.Timestamp(start)
    last = pd.Timestamp(end)
    while d <= last:
        yield d.strftime("%Y-%m-%d")
        d += pd.Timedelta(days=1)


def _labels(mids):
    """Forward labels from a 1s mid grid (NaN where the window is incomplete
    or spans a gap)."""
    n = len(mids)
    t1 = np.full(n, np.nan)
    t3 = np.full(n, np.nan)
    t2 = np.full(n, np.nan)
    m = mids
    valid = ~np.isnan(m) & (m > 0)
    bad = np.concatenate(([0], np.cumsum(~valid)))   # bad[k] = #invalid in m[:k]
    for h, out in ((1, t1), (5, t3)):
        for t in range(0, n - h):
            if bad[t + h + 1] - bad[t] == 0:
                out[t] = (m[t + h] / m[t] - 1.0) * 1e4
    if n > 31:
        r = np.where(valid[:-1] & valid[1:], np.log(m[1:] / m[:-1]), np.nan)
        for t in range(0, n - 30):
            seg = r[t:t + 30]
            if np.isnan(seg).any():
                continue
            mean = seg.mean()
            var = ((seg - mean) ** 2).sum() / (len(seg) - 1)
            t2[t] = np.sqrt(var) * 1e4
    return t1, t3, t2


def assemble(symbol, start, end, features_root, out_path,
             exclude=DEFAULT_EXCLUDE, features=None):
    feat_root = Path(features_root) / symbol
    if features is not None:
        feature_candidates = [f for f in features if f not in exclude]
    else:
        feature_candidates = [f for f in FACTOR_NAMES if f not in exclude]
    # aux per-second columns are always loaded: they feed the class-3/4
    # volume-weighted targets
    cols = ["sec", "mid", "complete"] + feature_candidates + list(AUX_COLUMNS)

    day_frames = []
    day_dates = []
    report = {"symbol": symbol, "range": [start, end], "days": {},
              "excluded_features": list(exclude), "auto_excluded_features": []}

    for date in daterange(start, end):
        ddir = feat_root / date
        hours = sorted(ddir.glob("*.parquet")) if ddir.exists() else []
        hours = [h for h in hours if not h.name.endswith(".anchors.parquet")]
        frames = []
        capture = {}
        n_ok = n_rej = 0
        for h in hours:
            meta_p = h.with_name(h.name.replace(".parquet", ".meta.json"))
            if meta_p.exists():
                meta = json.loads(meta_p.read_text())
                if meta.get("status") != "ok":
                    n_rej += 1
                    continue
                capture[h.stem] = float(meta["gates"]["capture_rate"])
                n_ok += 1
            df = pq.ParquetFile(h).read(columns=cols).to_pandas()
            df["capture_rate"] = capture.get(h.stem, np.nan)
            frames.append(df)
        day_info = {"ok_hours": n_ok, "rejected_hours": n_rej}
        if not frames:
            report["days"][date] = day_info
            continue
        day = pd.concat(frames, ignore_index=True).sort_values("sec")
        day_info["rows_raw"] = int(len(day))
        report["days"][date] = day_info
        day_frames.append(day)
        day_dates.append(date)

    if not day_frames:
        raise RuntimeError(f"no ok hours found for {symbol} {start}~{end}")

    # ---- per-day labels on the continuous grid; drop rows without labels
    all_rows = []
    label_drop = 0
    target_nan = {t: 0 for t in ("T1", "T2", "T3") + VWAP_TARGETS}
    target_tot = 0
    for date, day in zip(day_dates, day_frames):
        day = day.copy()
        day["T1"], day["T3"], day["T2"] = _labels(
            day["mid"].to_numpy(dtype=float))
        vwap = _labels_vwap(day["mid"].to_numpy(dtype=float),
                            day["buy_n_1s"].to_numpy(dtype=float),
                            day["sell_n_1s"].to_numpy(dtype=float),
                            day["buy_qty_1s"].to_numpy(dtype=float),
                            day["sell_qty_1s"].to_numpy(dtype=float),
                            day["buy_vwap_1s"].to_numpy(dtype=float),
                            day["sell_vwap_1s"].to_numpy(dtype=float))
        for k, v in vwap.items():
            day[k] = v
        target_tot += len(day)
        for t in target_nan:
            target_nan[t] += int(day[t].isna().sum())
        # row policy (unchanged): rows must carry T1/T2/T3 labels; the new
        # targets' own NaN rows are dropped per-target at train time
        before = len(day)
        day = day[~day[["T1", "T2", "T3"]].isna().any(axis=1)]
        label_drop += before - len(day)
        report["days"][date]["rows_label_ok"] = int(len(day))
        all_rows.append(day)

    report["target_nan_counts"] = target_nan
    report["target_nan_rate"] = {t: float(c / max(target_tot, 1))
                                 for t, c in target_nan.items()}

    df = pd.concat(all_rows, ignore_index=True)
    df["date"] = pd.to_datetime(df["sec"], unit="s").dt.strftime("%Y-%m-%d")
    report["rows_after_label_drop"] = int(len(df))
    report["label_drop_rows"] = int(label_drop)

    # ---- feature NaN audit, auto-exclusion of all-NaN features, complete cases
    nan_counts = {f: int(df[f].isna().sum()) for f in feature_candidates}
    report["feature_nan_counts"] = nan_counts
    auto = [f for f in feature_candidates if nan_counts[f] == len(df)]
    report["auto_excluded_features"] = auto
    keep = [f for f in feature_candidates if f not in auto]
    report["kept_features"] = keep
    df_labelvalid = df.copy()
    before = len(df)
    df = df.dropna(subset=keep)
    report["rows_final"] = int(len(df))
    report["complete_case_drop_rows"] = int(before - len(df))
    report["complete_case_drop_rate"] = float((before - len(df)) / max(before, 1))
    per_day = df.groupby("date").size().to_dict()
    for d, info in report["days"].items():
        info["rows_final"] = int(per_day.get(d, 0))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keep_cols = ["sec", "date", "mid", "complete", "capture_rate",
                 "T1", "T2", "T3"] + list(VWAP_TARGETS) + keep
    pq.write_table(pa.Table.from_pandas(df[keep_cols], preserve_index=False),
                   out_path)
    # full label-valid frame (float32 to bound memory/disk): the evaluation
    # "all seconds" framing; NaN features are allowed at LightGBM predict time
    lv = df_labelvalid
    for c in keep_cols:
        # sec stays int64: float32 cannot represent epoch seconds (2^24),
        # which would corrupt any per-second join on this frame
        if c not in ("date", "sec"):
            lv[c] = lv[c].astype(np.float32)
    lv_path = out_path.with_suffix(".labelvalid.parquet")
    pq.write_table(pa.Table.from_pandas(lv[keep_cols], preserve_index=False),
                   lv_path)
    report["labelvalid_path"] = str(lv_path)
    out_path.with_suffix(".report.json").write_text(
        json.dumps(report, indent=2, default=float))
    return df, report


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--features-root", default="data/prediction-v3/features")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    _, report = assemble(args.symbol, args.start, args.end,
                         args.features_root, args.out)
    print(json.dumps({k: report[k] for k in
                      ("rows_after_label_drop", "rows_final",
                       "complete_case_drop_rate", "auto_excluded_features")},
                     indent=2))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
