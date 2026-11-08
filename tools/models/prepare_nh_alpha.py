"""Fit NH's initial one-second alpha on certified pre-window events."""
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import hftbacktest as h
import numpy as np

from app.backtesting.experiment_config import load_spec, resolve

from app.market_making.strategies.nh_quote_value.features import NAMES, S
from app.backtesting.nh_sampling import sample
from app.market_making.strategies.nh_quote_value.training import train_alpha


def fit_symbol(dataset, cutoff, tick, lot):
    manifest_path = dataset / 'manifest.json'
    manifest_bytes = manifest_path.read_bytes()
    samples = []
    for segment in json.loads(manifest_bytes)['segments']:
        parts = [part for part in segment['files'] if part['start_ns'] < cutoff]
        if not parts:
            continue
        end = min(cutoff - 1, max(part['clock_end_ns'] for part in parts))
        paths = []
        depth_batches = []
        for part in parts:
            path = dataset / part['path']
            with path.open('rb') as stream:
                assert hashlib.file_digest(stream, 'sha256').hexdigest() == part['sha256']
            paths.append(str(path))
            with np.load(path, allow_pickle=False) as archive:
                tape = archive['data']
                local_depth = ((tape['ev'] & h.LOCAL_EVENT) != 0) & np.isin(tape['ev'] & 255, [1, 4])
                depth_batches.append(tape['local_ts'][local_depth])
        depth_times = np.unique(np.concatenate(depth_batches))
        if not len(depth_times):
            continue
        marker = np.zeros(1, dtype=h.event_dtype)
        marker[0] = (h.EXCH_EVENT | h.LOCAL_EVENT | h.DEPTH_EVENT | h.BUY_EVENT,
                     end + S, end + S, tick, 0., 0, 0, 0.)
        asset = (h.BacktestAsset().data([*paths, marker]).linear_asset(1.)
                 .tick_size(tick).lot_size(lot).last_trades_capacity(200000))
        backtest = h.HashMapMarketDepthBacktest([asset])
        try:
            start = max(int(depth_times[0]), int(segment['start_ns']))
            if end - start <= 61 * S:
                continue
            times, features, mids, valid = sample(backtest, start, end, depth_times)
        finally:
            backtest.close()
        pairs = valid[:-1] & valid[1:] & (np.diff(times) == S) & (times[1:] < cutoff - 5*S)
        samples.append((features[:-1][pairs], np.log(mids[1:][pairs]/mids[:-1][pairs])*1e4, times[1:][pairs]))
    x, y, label_times = (np.concatenate(parts) for parts in zip(*samples))
    if len(x) < 1000 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Insufficient valid pre-window alpha samples')
    mean, scale, beta, intercept = train_alpha(x, y, .1)
    return dict(features=NAMES, mean=mean.tolist(), scale=scale.tolist(),
                beta=beta.tolist(), intercept=intercept, ridge_lambda=.1,
                samples=len(x), last_label_ns=int(label_times.max()), deploy_ns=cutoff,
                dataset_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('spec', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    spec = load_spec(args.spec)
    cutoff = int(datetime.fromisoformat(spec['window']['start']+'T00:00:00+00:00').timestamp())*S
    rules = {row['symbol']: row for row in json.loads(resolve(spec['rules']).read_text())['symbols']}
    args.out.mkdir(parents=True, exist_ok=True)
    for symbol in spec['symbols']:
        filters = {row['filterType']: row for row in rules[symbol]['filters']}
        model = fit_symbol(resolve(spec['datasets'][symbol]['cache']), cutoff,
                           float(filters['PRICE_FILTER']['tickSize']), float(filters['LOT_SIZE']['stepSize']))
        model['symbol'] = symbol
        output = args.out / f'{symbol}.json'
        pending = output.with_suffix('.tmp')
        pending.write_text(json.dumps(model, allow_nan=False, indent=2)+'\n')
        pending.replace(output)
        print(symbol, model['samples'], model['last_label_ns'], flush=True)


if __name__ == '__main__':
    main()
