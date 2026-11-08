"""Train per-coin LightGBM models on an assembled Kimi v3 dataset (P3).

Split (14-day sprint, time-ordered): train 09-19~09-28 / valid 09-29~09-30 /
test 10-01~10-02. The validation window does early stopping AND combo
selection on a small frozen grid (num_leaves {15,31} x learning_rate
{0.03,0.1} x max_depth {4,7} = 8 combos) - this is a sprint sanity pass, not
the formal <=27-combo tuning budget of V3_SPEC §4.

Evaluation: per-day Spearman rankIC on the test window, the mean, and a
hour-block bootstrap 95% CI (each hour is one block; guards against
intraday autocorrelation; the 2-day test window is too short for pure
day-block bootstrap, so days are reported separately). Two framings:
"all" = every label-valid second in the test window (LightGBM predicts
fine with NaN features), "complete_only" = the assembled complete-case
rows. Pass bar from V3_SPEC §4: T2 rankIC >= 0.15 with a CI excluding
zero.

Trees need no scaling, so V3_SPEC §0.4 robust normalization is skipped here
(it matters for the frozen-ridge baseline, not for LightGBM).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import lightgbm as lgb  # noqa: E402

ID_COLUMNS = ("sec", "date", "mid", "complete", "capture_rate", "T1", "T2",
              "T3", "R_vwap_5", "R_vwap_30", "R_askw_5", "R_askw_30",
              "R_bidw_5", "R_bidw_30", "R_twap_5", "R_twap_30")

GRID = [(nl, lr, md)
        for nl in (15, 31)
        for lr in (0.03, 0.1)
        for md in (4, 7)]


def rank_ic(g, pred, target):
    if len(g) < 10 or g[pred].nunique() < 2 or g[target].nunique() < 2:
        return np.nan
    return g[pred].corr(g[target], method="spearman")


def hour_block_bootstrap(ic_by_hour, n_boot=2000, seed=7):
    vals = ic_by_hour.dropna().to_numpy()
    if len(vals) == 0:
        return {"mean": float("nan"), "ci_lo": float("nan"),
                "ci_hi": float("nan"), "blocks": 0}
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        means[b] = vals[rng.integers(0, len(vals), len(vals))].mean()
    return {"mean": float(vals.mean()),
            "ci_lo": float(np.percentile(means, 2.5)),
            "ci_hi": float(np.percentile(means, 97.5)),
            "blocks": int(len(vals))}


def _quintiles(sub, pred_col, target):
    """Mean realized target by predicted-quintile bucket (complete rows)."""
    s = sub[np.isfinite(sub[target]) & np.isfinite(sub[pred_col])]
    if len(s) < 500 or s[pred_col].nunique() < 5:
        return {}
    b = pd.qcut(s[pred_col], 5, labels=False, duplicates="drop")
    g = s.groupby(b, observed=True)[target].agg(["mean", "count"])
    out = {f"Q{i + 1}": {"mean_bp": float(r["mean"]), "n": int(r["count"])}
           for i, (_, r) in enumerate(g.iterrows())}
    if len(g) == 5:
        out["Q5_minus_Q1_bp"] = float(g["mean"].iloc[-1] - g["mean"].iloc[0])
    return out


def _eval_framing(sub, pred_col, target):
    sub = sub.copy()
    sub["_hour"] = (sub["sec"] // 3600).astype(int)
    by_day = sub.groupby("date").apply(lambda g: rank_ic(g, pred_col, target))
    by_hour = sub.groupby("_hour").apply(lambda g: rank_ic(g, pred_col, target))
    boot = hour_block_bootstrap(by_hour)
    return {"rankic_by_day": {d: float(v) for d, v in by_day.items()},
            "rankic_by_hour": {str(h): float(v) for h, v in by_hour.items()},
            "rankic_mean": boot["mean"], **boot,
            "rows": int(len(sub))}


RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0)


def fit_ridge(train_x, train_y, alpha):
    """Frozen ridge protocol (app/backtesting/forecast_model.py): population
    standardization from training rows, conventional ridge with unpenalized
    intercept solved analytically from joint moments."""
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    scale = np.where(std > 0, std, 1.0)
    z = (train_x - mean) / scale
    zx = np.column_stack((z, np.ones(len(z))))
    scatter = zx.T @ zx
    k = train_x.shape[1]
    pen = alpha * np.eye(k + 1)
    pen[k, k] = 0.0                      # intercept unpenalized
    weights = np.linalg.solve(scatter + pen, zx.T @ train_y)
    return {"mean": mean, "scale": scale,
            "w": weights[:k], "b": float(weights[k])}


def predict_ridge(model, x):
    return ((x - model["mean"]) / model["scale"]) @ model["w"] + model["b"]


def ridge_valid_mse(train, valid, feats, target, alpha):
    model = fit_ridge(train[feats].to_numpy(), train[target].to_numpy(), alpha)
    pred = predict_ridge(model, valid[feats].to_numpy())
    return float(np.mean((valid[target].to_numpy() - pred) ** 2))


def select_ridge_alpha(train, valid, feats, target="T1"):
    """Frozen selection rule: 1s-target validation MSE, ties -> larger alpha."""
    scores = {a: ridge_valid_mse(train, valid, feats, target, a)
              for a in RIDGE_ALPHAS}
    return min(sorted(scores, reverse=True), key=lambda a: scores[a]), scores


def train_target(df, target, grid=GRID, seed=7, labelvalid=None,
                 model_kind="lgbm", ridge_alpha=None, eval_frame=None):
    feats = [c for c in df.columns if c not in ID_COLUMNS]
    # per-target label availability: class-3/4 windows drop zero-volume rows
    df = df[np.isfinite(df[target])].reset_index(drop=True)
    train = df[df["date"] <= "2026-09-28"].reset_index(drop=True)
    valid = df[(df["date"] >= "2026-09-29") & (df["date"] <= "2026-09-30")] \
        .reset_index(drop=True)
    best = None
    if model_kind == "ridge":
        assert ridge_alpha is not None
        model = fit_ridge(train[feats].to_numpy(), train[target].to_numpy(),
                          ridge_alpha)
        pred = predict_ridge(model, valid[feats].to_numpy())
        vic = valid.assign(_p=pred).groupby("date").apply(
            lambda g: rank_ic(g, "_p", target)).mean()
        best = {"model": ("ridge", model), "params": {"alpha": ridge_alpha},
                "valid_rankic": float(vic), "best_iteration": 0}
    else:
        for nl, lr, md in grid:
            params = {"objective": "regression", "metric": "l2",
                      "num_leaves": nl, "learning_rate": lr, "max_depth": md,
                      "min_child_samples": 50, "feature_fraction": 0.8,
                      "verbose": -1, "seed": seed}
            dtrain = lgb.Dataset(train[feats], train[target])
            model = lgb.train(
                params, dtrain, num_boost_round=2000,
                valid_sets=[lgb.Dataset(valid[feats], valid[target],
                                        reference=dtrain)],
                callbacks=[lgb.early_stopping(100, verbose=False)])
            pred = model.predict(valid[feats])
            vic = valid.assign(_p=pred).groupby("date").apply(
                lambda g: rank_ic(g, "_p", target)).mean()
            if best is None or vic > best["valid_rankic"]:
                best = {"model": model,
                        "params": {"num_leaves": nl, "learning_rate": lr,
                                   "max_depth": md},
                        "valid_rankic": float(vic),
                        "best_iteration": model.best_iteration}

    def predict(x):
        m = best["model"]
        if model_kind == "ridge":
            return predict_ridge(m[1], x[feats].to_numpy())
        return m.predict(x[feats])

    # framing 1: complete-case assembled rows only
    test_c = df[df["date"] >= "2026-10-01"].reset_index(drop=True)
    test_c = test_c.assign(_p=predict(test_c))
    # framing 2: every label-valid second (NaN features predict via the
    # default tree direction)
    if labelvalid is not None:
        test_a = labelvalid[labelvalid["date"] >= "2026-10-01"] \
            .reset_index(drop=True)
        test_a = test_a.assign(_p=predict(test_a))
    else:
        test_a = test_c
    evals = {"all": _eval_framing(test_a, "_p", target),
             "complete_only": _eval_framing(test_c, "_p", target)}
    evals["all"]["quintiles"] = _quintiles(test_a, "_p", target)
    evals["complete_only"]["quintiles"] = _quintiles(test_c, "_p", target)
    # paired-comparison frame: identical evaluation rows for every config
    if eval_frame is not None:
        ef = eval_frame[eval_frame["date"] >= "2026-10-01"].copy()
        ef = ef.assign(_p=predict(ef))
        evals["paired"] = _eval_framing(ef, "_p", target)
    imp = None
    if model_kind != "ridge":
        imp = pd.Series(best["model"].feature_importance(importance_type="gain"),
                        index=feats).sort_values(ascending=False)
    return {"target": target, "model": model_kind, "features": feats,
            "grid_size": len(grid),
            "selected_params": best["params"],
            "best_iteration": int(best["best_iteration"]),
            "valid_rankic": best["valid_rankic"],
            "test_eval": evals,
            "importance_top15": ({k: float(v) for k, v in imp.head(15).items()}
                                 if imp is not None else {})}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--targets", default="T1,T2,T3")
    ap.add_argument("--max-combos", type=int, default=None,
                    help="truncate the sprint grid (smoke tests)")
    ap.add_argument("--labelvalid", default=None,
                    help="path to the .labelvalid.parquet frame (all-seconds "
                         "evaluation framing)")
    ap.add_argument("--model", choices=("lgbm", "ridge"), default="lgbm")
    ap.add_argument("--eval-frame", default=None,
                    help="shared evaluation frame (paired comparison rows)")
    ap.add_argument("--out", default=None, help="optional JSON report path")
    args = ap.parse_args(argv)

    df = pd.read_parquet(args.dataset)
    labelvalid = pd.read_parquet(args.labelvalid) if args.labelvalid else None
    eval_frame = pd.read_parquet(args.eval_frame) if args.eval_frame else None
    grid = GRID if args.max_combos is None else GRID[:args.max_combos]
    report = {"dataset": args.dataset, "model": args.model,
              "rows": int(len(df)),
              "date_range": [df["date"].min(), df["date"].max()],
              "targets": {}}
    ridge_alpha = None
    ridge_scores = None
    if args.model == "ridge":
        # frozen selection rule: alpha chosen on the 1s target's validation
        # MSE, ties -> larger alpha; the same alpha serves all targets
        feats = [c for c in df.columns if c not in ID_COLUMNS]
        tr = df[df["date"] <= "2026-09-28"].reset_index(drop=True)
        va = df[(df["date"] >= "2026-09-29") & (df["date"] <= "2026-09-30")] \
            .reset_index(drop=True)
        ridge_alpha, ridge_scores = select_ridge_alpha(tr, va, feats)
        report["ridge_alpha"] = ridge_alpha
        report["ridge_alpha_valid_mse"] = {str(k): v for k, v in ridge_scores.items()}
    for target in args.targets.split(","):
        report["targets"][target] = train_target(
            df, target, grid=grid, labelvalid=labelvalid,
            model_kind=args.model, ridge_alpha=ridge_alpha,
            eval_frame=eval_frame)
        e = report["targets"][target]["test_eval"]["all"]
        print(f"[{args.model} {target}] test rankIC mean={e['mean']:.4f} "
              f"CI=[{e['ci_lo']:.4f}, {e['ci_hi']:.4f}] "
              f"valid={report['targets'][target]['valid_rankic']:.4f} "
              f"params={report['targets'][target]['selected_params']}",
              flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, default=float))
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
