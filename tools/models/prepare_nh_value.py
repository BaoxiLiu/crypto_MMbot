"""Train initial NH quote-value models from pre-window native probes."""
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import hftbacktest as h
import numpy as np

from app.backtesting.experiment_config import load_spec, resolve

from app.backtesting.nh_probe import collect
from app.market_making.strategies.nh_quote_value.training import train_value
from app.market_making.strategies.nh_quote_value.value_features import NAMES
from app.market_making.fusion.queue_models import configure_queue

S = 1_000_000_000


def fit(dataset, cutoff, rules, execution, queue, notional):
    manifest = dataset / 'manifest.json'
    batches, label_ends = [], []
    for segment in json.loads(manifest.read_text())['segments']:
        files = [p for p in segment['files'] if p['start_ns'] < cutoff]
        if not files:
            continue
        end = min(cutoff-1,max(p['clock_end_ns'] for p in files))
        depth = []
        paths = []
        for part in files:
            path = dataset/part['path']
            with path.open('rb') as stream:
                if hashlib.file_digest(stream,'sha256').hexdigest()!=part['sha256']:
                    raise ValueError(f'Changed input {path}')
            paths.append(str(path))
            with np.load(path,allow_pickle=False) as archive:
                tape = archive['data']
                is_depth = ((tape['ev'] & h.LOCAL_EVENT)!=0) & np.isin(tape['ev'] & 255,[1,4])
                depth.append(tape['local_ts'][is_depth])
        times = np.unique(np.concatenate(depth))
        if not len(times):
            continue
        start = max(int(times[0]),segment['start_ns'])
        if end-start < 68*S:
            continue
        tick,lot,min_qty,min_notional = rules
        marker = np.zeros(1,dtype=h.event_dtype)
        marker[0] = (h.EXCH_EVENT|h.LOCAL_EVENT|h.DEPTH_EVENT|h.BUY_EVENT,
                     end+S,end+S,tick,0.,0,0,0.)
        asset = (h.BacktestAsset().data([*paths,marker]).linear_asset(1.)
                 .tick_size(tick).lot_size(lot).last_trades_capacity(200000)
                 .constant_order_latency(execution['request_ns'],execution['response_ns']))
        asset = configure_queue(asset,queue).partial_fill_exchange().trading_value_fee_model(0.,float(execution['taker_fee']))
        backtest = h.HashMapMarketDepthBacktest([asset])
        try:
            batch = collect(backtest,start,end,times,tick,lot,min_qty,min_notional,
                            notional,execution['decision_ns'],20261029+segment['id'])
        finally:
            backtest.close()
        if len(batch[0]):
            label_ends.append(start+int(batch[-1].max())*100_000_000)
            batches.append(batch)
    x,orders,qty,edge_sum,_ = (np.concatenate(v) for v in zip(*batches))
    if max(label_ends)>=cutoff or not np.isfinite(x).all() or not np.any(qty>0):
        raise ValueError('Insufficient causal probe labels')
    filled = qty>0
    edge = np.divide(edge_sum,qty,out=np.zeros_like(qty),where=filled)
    model = train_value(x,filled,qty/orders[:,2],edge,qty*orders[:,1],orders[:,3].astype(int))
    return dict(**{k:v.tolist() if isinstance(v,np.ndarray) else float(v) for k,v in model.items()},
                features=NAMES, samples=len(x), filled_orders=int(filled.sum()), last_label_ns=max(label_ends),
                deploy_ns=cutoff, queue=queue, order_notional=notional,
                dataset_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('spec',type=Path)
    parser.add_argument('out',type=Path)
    parser.add_argument('--symbol',action='append')
    args = parser.parse_args()
    spec = load_spec(args.spec)
    cutoff = int(datetime.fromisoformat(spec['window']['start']+'T00:00:00+00:00').timestamp())*S
    rules = {row['symbol']:row for row in json.loads(resolve(spec['rules']).read_text())['symbols']}
    args.out.mkdir(parents=True,exist_ok=True)
    for symbol in args.symbol or spec['symbols']:
        item = spec['datasets'][symbol]
        f = {row['filterType']:row for row in rules[symbol]['filters']}
        tick = float(f['PRICE_FILTER']['tickSize'])
        lot = float(f['LOT_SIZE']['stepSize'])
        min_qty = float(f['LOT_SIZE']['minQty'])
        min_notional = float(f.get('MIN_NOTIONAL',{}).get('notional',0))
        reference = float(item['order_reference']['price'])
        notional = max(50.,1.1*(min_notional+lot*reference),1.1*min_qty*reference)
        model = fit(resolve(item['cache']),cutoff,(tick,lot,min_qty,min_notional),
                    spec['execution'],spec['engine']['queue'],notional)
        pending = args.out/f'{symbol}.tmp'
        pending.write_text(json.dumps(dict(symbol=symbol,**model),allow_nan=False,indent=2)+'\n')
        pending.replace(args.out/f'{symbol}.json')
        print(symbol,model['samples'],model['filled_orders'],model['supported'],flush=True)


if __name__=='__main__':
    main()
