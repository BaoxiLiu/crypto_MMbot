"""Synthetic native clock probes; no historical account/model replay.

Expose request/effect/observation ordering under the registered 10ms outbound,
5ms response latency. Native local_timestamp is the request field, not a fill
receipt timestamp. Observations use the replay clock explicitly.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import hftbacktest as h
import numpy as np


def engine(trades):
    flags=h.EXCH_EVENT|h.LOCAL_EVENT
    points=[(0,h.DEPTH_SNAPSHOT_EVENT|h.BUY_EVENT,100.,2.),
            (0,h.DEPTH_SNAPSHOT_EVENT|h.SELL_EVENT,102.,2.),*trades,
            (1_000_000_000,h.DEPTH_EVENT|h.SELL_EVENT,102.,2.)]
    data=np.zeros(len(points),dtype=h.event_dtype)
    for i,(t,event,p,q) in enumerate(points): data[i]=(flags|event,t,t+5_000_000,p,q,0,0,0.)
    asset=(h.BacktestAsset().data(data).linear_asset(1.).tick_size(1.).lot_size(.1)
           .constant_order_latency(10_000_000,5_000_000).risk_adverse_queue_model()
           .partial_fill_exchange().trading_value_fee_model(0.,.0004))
    bt=h.HashMapMarketDepthBacktest([asset]);bt.elapse(100_000_000)
    bt.submit_buy_order(0,1,100.,2.,h.GTX,h.LIMIT,False)
    return bt


def sample(bt):
    order=bt.orders(0).get(1)
    return {'observation_ns':int(bt.current_timestamp),'position':float(bt.position(0)),
            'order':None if order is None else {'status':int(order.status),'remaining':float(order.leaves_qty),
                'native_exchange_processed_ns':int(order.exch_timestamp),
                'native_local_request_ns':int(order.local_timestamp), 'exec_qty':float(order.exec_qty)}}


def advance_to(bt, ns):
    result=bt.elapse(ns-int(bt.current_timestamp))
    if result!=0: raise RuntimeError('Synthetic probe ended early')
    return sample(bt)


def probe():
    trade=lambda t,q:(t,h.TRADE_EVENT|h.SELL_EVENT,100.,q)
    # Queue 2 units ahead; a 3-unit trade at 111ms executes 1 unit after
    # new-order effectiveness (110ms), before NEW response (115ms).
    bt=engine([trade(111_000_000,3.)]);first=[]
    try:
        first.append(advance_to(bt,113_000_000));assert first[-1]['position']==0.
        first.append(advance_to(bt,115_000_000));assert first[-1]['position']==0.
        first.append(advance_to(bt,116_000_000));assert first[-1]['position']==1.
        assert first[-1]['order']['native_exchange_processed_ns']==111_000_000
    finally:bt.close()
    # Cancel requested at 200ms is effective at 210ms, observed at 215ms.
    # 207ms trade executes one unit; the 212ms trade cannot execute more.
    bt=engine([trade(207_000_000,3.),trade(212_000_000,1.)]);second=[]
    try:
        second.append(advance_to(bt,200_000_000));bt.cancel(0,1,False)
        second.append(advance_to(bt,209_000_000));assert second[-1]['position']==0.
        second.append(advance_to(bt,213_000_000));assert second[-1]['position']==1.
        assert second[-1]['order']['status']!=h.CANCELED
        second.append(advance_to(bt,216_000_000));assert second[-1]['position']==1.
        assert second[-1]['order']['status']==h.CANCELED
        assert second[-1]['order']['native_exchange_processed_ns']==210_000_000
    finally:bt.close()
    return {'status':'synthetic_native_clock_ordering_verified_not_historical_execution_audit',
            'request_latency_ms':10,'response_latency_ms':5,'new_effective_before_ack':first,
            'cancel_effective_before_ack_with_inflight_partial_fill':second,
            'limitations':['Fixture only; historical order logs did not retain native exch_timestamp.',
                'native local_timestamp denotes local request, not feedback arrival.',
                'Native exch_timestamp is the last processed exchange event, not every individual partial execution.']}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError('Do not overwrite an existing probe')
    result=probe();result['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['runtime_versions']={n:importlib.metadata.version(n) for n in ('hftbacktest','numpy','numba')}
    import hftbacktest._hftbacktest as native
    result['native_binary_sha256']=hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(result['status'])


if __name__=='__main__':main()
