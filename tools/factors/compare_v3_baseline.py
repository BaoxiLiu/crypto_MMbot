"""Paired hourly rankIC comparison of two explicit model reports.

Both inputs must be generated on the same evaluation frame. This exploratory
bootstrap describes predictive differences, never economic strategy adoption.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

def paired_increment(v3_path, base_path, target, n_boot=5000, seed=11):
    v3 = json.loads(Path(v3_path).read_text())["targets"][target]
    base = json.loads(Path(base_path).read_text())["targets"][target]
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
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--v3-report',type=Path,required=True)
    parser.add_argument('--baseline-report',type=Path,required=True)
    parser.add_argument('--targets',nargs='+',default=['T1','T2','T3'])
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    result={target:paired_increment(args.v3_report,args.baseline_report,target)
            for target in args.targets}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
