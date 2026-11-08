"""Audit IOC execution accounting on a fixed synthetic two-level book.

The expected fills are declared independently of simulator output. This does
not implement liquidation, certify real depth/impact, or change the runtime.
Known discrepancies are saved as a failed capability audit, never as success.
"""
import argparse
import importlib.metadata
import math
from pathlib import Path

import hftbacktest as h
from hftbacktest.order import IOC, FOK, GTC, MARKET, LIMIT
import numpy as np

from app.artifacts import digest as sha, write

NS = 1_000_000_000
REQUEST = 10_000_000
RESPONSE = 5_000_000
FEE = .0004


def cases(expanded=False):
    for side in (1, -1):
        prices = [101., 102.] if side == 1 else [99., 98.]
        for name, qty, levels in (('one_level_full', 1., 1),
                                  ('two_levels_full', 2., 2),
                                  ('two_levels_partial_ioc', 3., 2)):
            fills = [(p, 1.) for p in prices[:levels]]
            value = sum(p*q for p, q in fills)
            yield dict(id=('buy_' if side == 1 else 'sell_')+name, side=side,
                       qty=qty, limit=prices[levels-1], time_in_force=int(IOC),
                       order_type=int(LIMIT), expected_fills=fills,
                       expected=dict(position=float(side*levels), balance=-side*value,
                                     fee=value*FEE, leaves_qty=qty-levels,
                                     status=int(h.FILLED if qty == levels else h.EXPIRED)))
        if expanded:
            for name,qty,tif,typ,limit,filled in (
                    ('fok_two_levels_full',2.,FOK,LIMIT,prices[1],prices),
                    ('fok_insufficient_rejected',3.,FOK,LIMIT,prices[1],[]),
                    ('gtc_two_levels_full',2.,GTC,LIMIT,prices[1],prices),
                    ('market_two_levels_full',2.,IOC,MARKET,prices[1],prices),
                    ('market_two_levels_partial',3.,IOC,MARKET,prices[1],prices),
                    ('noncross_ioc',1.,IOC,LIMIT,100.,[]),
                    ('noncross_fok',1.,FOK,LIMIT,100.,[])):
                value=sum(filled)
                yield dict(id=('buy_' if side==1 else 'sell_')+name,side=side,
                           qty=qty,limit=limit,time_in_force=int(tif),order_type=int(typ),
                           expected_fills=[(p,1.) for p in filled],
                           expected=dict(position=float(side*len(filled)),balance=-side*value,
                                         fee=value*FEE,leaves_qty=qty-len(filled),
                                         status=int(h.FILLED if len(filled)==qty else h.EXPIRED)))


def run(output, expanded=False):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    declared = list(cases(expanded))
    identity = dict(tool_sha256=sha(__file__), hftbacktest_version=importlib.metadata.version('hftbacktest'),
                    native_binary_sha256=sha(next(Path(h.__file__).parent.glob('_hftbacktest*.so'))))
    write(output/'inputs.json', dict(**identity, cases=declared, tick=1., lot=1.,
                                    request_ns=REQUEST, response_ns=RESPONSE, taker_fee=FEE), default=None)
    rows = [(h.DEPTH_SNAPSHOT_EVENT|h.BUY_EVENT, 99., 1.),
            (h.DEPTH_SNAPSHOT_EVENT|h.BUY_EVENT, 98., 1.),
            (h.DEPTH_SNAPSHOT_EVENT|h.SELL_EVENT, 101., 1.),
            (h.DEPTH_SNAPSHOT_EVENT|h.SELL_EVENT, 102., 1.)]
    tape = np.zeros(5, dtype=h.event_dtype)
    for i, (ev, price, qty) in enumerate(rows):
        tape[i] = (h.EXCH_EVENT|h.LOCAL_EVENT|ev, NS, NS+5_000_000, price, qty, 0, 0, 0.)
    tape[-1] = (h.EXCH_EVENT|h.LOCAL_EVENT|h.DEPTH_EVENT|h.BUY_EVENT,
                4*NS, 4*NS+5_000_000, 99., 1., 0, 0, 0.)
    path=output/'market.npz';np.savez_compressed(path,data=tape)
    results=[]
    for case in declared:
        asset=(h.BacktestAsset().data([str(path)]).linear_asset(1.)
               .constant_order_latency(REQUEST, RESPONSE).risk_adverse_queue_model()
               .partial_fill_exchange().trading_value_fee_model(0., FEE).tick_size(1.).lot_size(1.))
        backend=h.HashMapMarketDepthBacktest([asset])
        try:
            if backend.elapse(1)!=0:raise RuntimeError('Unable to initialize IOC audit')
            if backend.elapse(2*NS-backend.current_timestamp)!=0:raise RuntimeError('IOC audit book unavailable')
            submit=backend.submit_buy_order if case['side']==1 else backend.submit_sell_order
            if submit(0,1,case['limit'],case['qty'],case['time_in_force'],case['order_type'],False)!=0:
                raise RuntimeError('IOC audit request rejected by API')
            if backend.elapse(REQUEST+RESPONSE+1)!=0:raise RuntimeError('IOC audit ended before response')
            state=backend.state_values(0);order=backend.orders(0).get(1)
            if order is None:raise RuntimeError('IOC audit response missing')
            observed=dict(position=float(state.position),balance=float(state.balance),fee=float(state.fee),
                          leaves_qty=float(order.leaves_qty),status=int(order.status))
            mismatch={key:dict(expected=value,observed=observed[key]) for key,value in case['expected'].items()
                      if not math.isclose(value,observed[key],rel_tol=0,abs_tol=1e-10)}
            results.append(dict(**case,observed=observed,mismatches=mismatch,
                                native_last_exec_qty=float(order.exec_qty),
                                native_last_exec_price_tick=int(order.exec_price_tick),
                                effective_ns=int(order.exch_timestamp),observed_ns=int(backend.current_timestamp)))
        finally:backend.close()
    errors=[row['id'] for row in results if row['mismatches']]
    report=dict(**identity,status='failed_ioc_accounting_capability' if errors else 'synthetic_ioc_accounting_verified',
                expanded_order_types=expanded,
                cases=results,failed_cases=errors,market_sha256=sha(path),
                executable_terminal_exit_ready=False,
                limitations=['Synthetic fixed book; optional FOK/GTC/market checks, not a terminal liquidation implementation.',
                             'GTC insufficient-depth synthetic remainder fill and order modification are outside this audit.',
                             'No portfolio carry, cancellation drain, funding/ledger integration or real tape.',
                             'Market impact and available depth during repeated orders are unverified.'])
    write(output/'verification.json',report, default=None)
    print(report['status'], 'failed_cases=',errors, flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--expanded',action='store_true')
    args=parser.parse_args()
    result=run(args.output,args.expanded)
    raise SystemExit(1 if result['failed_cases'] else 0)
