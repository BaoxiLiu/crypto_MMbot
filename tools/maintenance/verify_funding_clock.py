"""Test constant-response position alignment without changing the native binary.

A zero-response observer receives identical requests; only its acknowledgements
are earlier. Its matching-time net position must equal the ordinary simulator's
position R nanoseconds later. This is a synthetic timing proof, not funding
integration, real latency calibration, or strategy profitability evidence.
"""
import argparse
from decimal import Decimal
import importlib.metadata
import json
from pathlib import Path

import hftbacktest as h
import numpy as np

from app.artifacts import digest as sha, write

NS=1_000_000_000
T=10*NS
RESPONSE=5_000_000


def advance(backend, stamp):
    if stamp<backend.current_timestamp:raise ValueError('Observer clock went backwards')
    if stamp>backend.current_timestamp and backend.elapse(stamp-backend.current_timestamp)!=0:
        raise RuntimeError('Timing fixture ended prematurely')


def wait_to(backend,stamp):
    while backend.current_timestamp<stamp:
        result=backend.wait_next_feed(True,stamp-backend.current_timestamp)
        if result not in (0,2,3):raise RuntimeError('Wait-next-feed fixture ended prematurely')


def run(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    results=[]
    for side,name,tradeflag,px,first_count in ((1,'buy',h.SELL_EVENT,100.,1),(-1,'sell',h.BUY_EVENT,102.,1),
                                            (1,'buy_two_packets',h.SELL_EVENT,100.,2),(-1,'sell_two_packets',h.BUY_EVENT,102.,2)):
        rows=[(NS,h.DEPTH_SNAPSHOT_EVENT|h.BUY_EVENT,99.,1.),
              (NS,h.DEPTH_SNAPSHOT_EVENT|h.SELL_EVENT,101.,1.),
              (T-2_000_000,h.TRADE_EVENT|tradeflag,px,1.),
              (T+2_000_000,h.TRADE_EVENT|tradeflag,px,1.),
              (20*NS,h.DEPTH_EVENT|h.BUY_EVENT,99.,1.)]
        if first_count==2:rows.insert(3,(T-2_000_000,h.TRADE_EVENT|tradeflag,px,1.))
        tape=np.zeros(len(rows),dtype=h.event_dtype)
        for i,(stamp,ev,price,qty) in enumerate(rows):
            tape[i]=(h.EXCH_EVENT|h.LOCAL_EVENT|ev,stamp,stamp+5_000_000,price,qty,0,0,0.)
        path=output/(name+'.npz');np.savez_compressed(path,data=tape)
        backends=[]
        try:
            for response in (0,RESPONSE):
                asset=(h.BacktestAsset().data([str(path)]).linear_asset(1.).constant_latency(10_000_000,response)
                       .risk_adverse_queue_model().partial_fill_exchange().trading_value_fee_model(0.,.0004)
                       .tick_size(1.).lot_size(1.))
                backend=h.HashMapMarketDepthBacktest([asset]);backends.append(backend)
                if backend.elapse(1)!=0:raise RuntimeError('Unable to initialize timing fixture')
                advance(backend,T-20_000_000)
                submit=backend.submit_buy_order if side==1 else backend.submit_sell_order
                assert submit(0,1,px,float(first_count+2),h.GTX,h.LIMIT,False)==0
                # Cancel is requested before the funding timestamp but arrives
                # after both trades. Responses must not duplicate those fills.
                advance(backend,T-1_000_000);assert backend.cancel(0,1,False)==0
            observer,ordinary=backends
            advance(observer,T);wait_to(ordinary,T)
            exchange_position=observer.position(0);unshifted_position=ordinary.position(0)
            wait_to(ordinary,T+RESPONSE);shifted_position=ordinary.position(0)
            assert exchange_position==first_count*side and unshifted_position==0 and shifted_position==first_count*side
            advance(observer,T+2_000_000);advance(ordinary,T+2_000_000+RESPONSE)
            assert observer.position(0)==ordinary.position(0)==(first_count+1)*side
            advance(observer,T+20_000_000);advance(ordinary,T+20_000_000)
            assert observer.position(0)==ordinary.position(0)==(first_count+1)*side
            # Equal quantity gives equal funding; the ordinary unshifted
            # snapshot would incorrectly charge zero in this fixture.
            payment=-Decimal(str(shifted_position))*Decimal('101')*Decimal('.001')
            results.append(dict(side=side,first_timestamp_packets=first_count,wait_next_feed_snapshots=True,settlement_ns=T,response_ns=RESPONSE,
                exchange_observer_position=exchange_position,unshifted_local_position=unshifted_position,
                shifted_local_position=shifted_position,payment=str(payment),
                partial_fill_and_late_cancel_final_position=ordinary.position(0),market_sha256=sha(path)))
        finally:
            for backend in backends:backend.close()
    report=dict(status='synthetic_constant_response_position_alignment_verified',tool_sha256=sha(__file__),
        hftbacktest_version=importlib.metadata.version('hftbacktest'),
        native_binary_sha256=sha(next(Path(h.__file__).parent.glob('_hftbacktest*.so'))),
        response_ns=RESPONSE,cases=results,
        limitations=['Only constant response and supplied identical requests; not variable latency.',
                     'No in-engine funding integration, wallet/risk feedback, certification gaps or terminal exit.',
                     'Matching events strictly before/after settlement; equal-timestamp ordering is not resolved.',
                     'Zero-response observer is a timing oracle for this fixture, not a new strategy backtest.'])
    write(output/'verification.json',report, default=None);print(json.dumps(report),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
