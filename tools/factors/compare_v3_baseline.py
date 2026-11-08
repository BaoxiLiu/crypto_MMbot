"""Paired increment test: v3-43 vs frozen-15 rankIC on identical rows (P3).

Reads train reports (tools/models/train_v3_lgbm.py --eval-frame pepe-14d.evalframe.parquet,
which exports per-hour rankICs on the shared frame) and computes, per target
and model class, the paired per-hour increment

    d_h = rankIC_h(v3 config) - rankIC_h(baseline config)

with an hour-block bootstrap 95% CI on mean(d). Verdict rule (V3_SPEC §4.1
spirit, single-coin sprint): increment >= 0.01 with CI excluding zero = real
added value; 0 <= increment < 0.01 = marginal; negative = the new factors do
not beat the frozen set on this coin.

Usage (repo root):
    python3 tools/factors/compare_v3_baseline.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ASSEMBLED = ROOT / "data/prediction-v3/assembled"

PAIRS = {
    "lgbm": ("pepe-14d.train-report.json", "pepe-14d.base15.lgbm.report.json"),
    "ridge": ("pepe-14d.ridge.report.json", "pepe-14d.base15.ridge.report.json"),
}
TARGETS = ("T1", "T2", "T3")


def paired_increment(v3_path, base_path, target, n_boot=5000, seed=11):
    v3 = json.loads((ASSEMBLED / v3_path).read_text())["targets"][target]
    base = json.loads((ASSEMBLED / base_path).read_text())["targets"][target]
    a = v3["test_eval"]["paired"]["rankic_by_hour"]
    b = base["test_eval"]["paired"]["rankic_by_hour"]
    hours = sorted(set(a) & set(b))
    d = np.array([a[h] - b[h] for h in hours if np.isfinite(a[h]) and np.isfinite(b[h])])
    if not len(d):
        return None
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = d[rng.integers(0, len(d), len(d))].mean()
    return {"hours": int(len(d)), "mean_v3": float(np.mean(
                [a[h] for h in hours if np.isfinite(a[h]) and np.isfinite(b[h])])),
            "mean_base": float(np.mean(
                [b[h] for h in hours if np.isfinite(a[h]) and np.isfinite(b[h])])),
            "increment": float(d.mean()),
            "ci_lo": float(np.percentile(means, 2.5)),
            "ci_hi": float(np.percentile(means, 97.5))}


def main():
    out = {}
    for model, (v3_rep, base_rep) in PAIRS.items():
        for target in TARGETS:
            r = paired_increment(v3_rep, base_rep, target)
            out[f"{model}:{target}"] = r
            if r:
                verdict = ("REAL" if r["increment"] >= 0.01 and r["ci_lo"] > 0
                           else "MARGINAL" if r["increment"] >= 0 else "NEGATIVE")
                print(f"[{model:5} {target}] paired hours={r['hours']} "
                      f"v3={r['mean_v3']:.4f} base={r['mean_base']:.4f} "
                      f"increment={r['increment']:+.4f} "
                      f"CI=[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] -> {verdict}",
                      flush=True)
    (ASSEMBLED / "pepe-14d.increment.json").write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
