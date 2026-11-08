"""Synthetic exchange tape for pre/post refactor equality; never financial evidence."""
from dataclasses import replace
from decimal import Decimal as D
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

CASES=('as_original','as_improved4','as_improved2','as_improved0',
       'hjb_fixed','hjb_adaptive','hjb_budget50','nh')
ROOT=Path(__file__).resolve().parents[4]
GOLDEN=Path(__file__).parent/'fixtures/strategy_refactor_trace.json'


def nh_params(start,end,symbol="TESTUSDC"):
    alpha=dict(symbol=symbol,last_label_ns=start-1,deploy_ns=start,
               mean=[0.]*23,scale=[1.]*23,beta=[0.]*23,intercept=.2)
    value=dict(symbol=symbol,last_label_ns=start-1,deploy_ns=start,
               mean=[0.]*28,scale=[1.]*28,fill_beta=[0.]*28,edge_beta=[0.]*28,
               fill_intercept=0.,edge_intercept=2.,supported=[True]*6,fraction=[1.]*6,
               queue='cancel_50_50',order_notional=25.)
    return dict(alpha=alpha,value=value,trade_start_ns=start,trade_end_ns=end)


def replay_case(folder,name,*,symbol="TESTUSDC",fee=None,metrics_spec=None,partial=False,return_artifacts=False,plugin_builder=None):
    import hftbacktest as h
    from app.market_making.as_math import ASConfig
    from app.market_making.core import Rules
    from app.market_making.fusion.config import FusionConfig
    from app.market_making.fusion.event_backtest import run_event_arm
    from app.market_making.fusion.execution import ExecutionProfile
    from app.market_making.strategies.hjb_strategy import DEFAULTS
    from app.market_making.strategies.registry import normalize_case,factory_for,bind_symbol
    start=1_785_542_400_000_000_000
    rows=[];marks=[]
    def row(kind,t,price,qty):
        return (h.EXCH_EVENT|h.LOCAL_EVENT|kind,t,t+1_500_000,price,qty,0,0,0.)
    for i in range(1980):
        t=start+i*10**9;bid=100+(i%2)*.01;ask=bid+.02
        if not i:
            for n in range(16):
                rows.append(row(h.DEPTH_SNAPSHOT_EVENT|h.BUY_EVENT,t,bid-n*.01,1.))
                rows.append(row(h.DEPTH_SNAPSHOT_EVENT|h.SELL_EVENT,t,ask+n*.01,1.))
        else:
            prior=100+((i-1)%2)*.01
            rows.extend([row(h.DEPTH_EVENT|h.BUY_EVENT,t,prior,0),row(h.DEPTH_EVENT|h.SELL_EVENT,t,prior+.02,0),
                         row(h.DEPTH_EVENT|h.BUY_EVENT,t,bid,1),row(h.DEPTH_EVENT|h.SELL_EVENT,t,ask,1)])
        mid=(bid+ask)/2;side=1 if i%2 else -1
        px=round(mid+side*(1+i%12)*.01,2)
        rows.append(row(h.TRADE_EVENT|(h.BUY_EVENT if side==1 else h.SELL_EVENT),t+500_000_000,px,20.))
        marks.append((t+1_500_000,mid))
    tape=np.asarray(rows,dtype=h.event_dtype)
    cache=folder/'cache';cache.mkdir(parents=True)
    path=cache/'segment.npz';np.savez_compressed(path,data=tape)
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    end=int(tape['local_ts'][-1]);part=dict(path=path.name,rows=len(tape),sha256=digest(path),
        start_ns=start,end_ns=end,clock_start_ns=start,clock_end_ns=end)
    (cache/'manifest.json').write_text(json.dumps(dict(version=6,symbol=symbol,feed_latency_ms=1.5,
      unresolved_trade_gaps=[],feed_clock='publication_E_plus_transport_exchange_T',
      segments=[dict(id=0,start_ns=start,end_ns=end,files=[part])])))
    mark_path=folder/'marks.npy';np.save(mark_path,np.asarray(marks,dtype=[('local_ts','i8'),('price','f8')]))
    rules=Rules(D('.01'),D('.01'),D('5'),D('.01'),D(10000),D('.5'),D('1.5'))
    if fee is None:fee='-0.00003' if name in ('as_improved4','hjb_adaptive') else '0'
    config=replace(FusionConfig(),as_config=replace(ASConfig(),maker_fee=D(fee),position_cap=D(10),half_spread_floor=0),max_notional=D(1000))
    trade_start=start+1860*10**9
    if name.startswith('as_'):
        params=dict(mode='as_base' if name=='as_original' else 'paper_s1_sc0',
                    paper_config=json.loads((ROOT/'app/market_making/fixtures/account_matrix.json').read_text()),
                    trade_start_ns=trade_start,trade_end_ns=end+1)
        if name in ('as_improved2','as_improved0'):params['min_half_spread_ticks']=int(name[-1])
        plugin=dict(type='paper_native_as',version=1,params=params)
    elif name.startswith('hjb_'):
        plugin=dict(type='fb_as_hjb',version=2,params=dict(DEFAULTS,
                    trade_start_ns=trade_start,trade_end_ns=end+1,
                    adaptive=name!='hjb_fixed',capital_utilization='0.5' if name=='hjb_budget50' else None))
    else:
        plugin=dict(type='nh_quote_value',version=1,params=dict(models={symbol:nh_params(trade_start,end+1,symbol)}))
    if plugin_builder is not None:plugin=plugin_builder(trade_start,end+1,symbol)
    case=normalize_case(dict(id=name,symbol=symbol,plugin=plugin))
    case=bind_symbol(case,symbol)
    output=folder/'result'
    from app.backtesting.experiment_metrics import ReplayMetrics
    observer=(lambda output:ReplayMetrics(output,metrics_spec,'2026-08-01','2026-08-02')) if metrics_spec else None
    result=run_event_arm(cache,output,0,config,mark_path,[],rules,
         ExecutionProfile(name='test',feed_ns=1_500_000,decision_ns=100_000,request_ns=500_000,response_ns=500_000,
                          refresh_ns=None,trigger='event',taker_fee=D('.0004')),
         strategy_factory=factory_for(case),queue_model='cancel_50_50',
         observer_factory=observer,max_files=1 if partial else None)
    fields=('position','wallet','realized','unrealized','fees','funding','equity','pnl','cash_flow',
            'max_position','max_drawdown','halt_reason','orders','policy','markouts',
            'native_fills','native_fees','trace_sha256','invariant_failures','counters')
    summary={key:result[key] for key in fields}
    with np.load(output/'equity.npz',allow_pickle=False) as z:
        summary['equity_sha256']=hashlib.sha256(z['data'].tobytes()).hexdigest()
    assert result['invariant_failures']==0
    assert result['orders'].get('fill_events',0)>0,name
    if return_artifacts:return result,output,cache
    return summary


if __name__=='__main__':
    results={}
    for name in CASES:
        with tempfile.TemporaryDirectory(prefix='strategy-refactor-') as tmp:
            results[name]=replay_case(Path(tmp),name)
        print(name,results[name]['pnl'],results[name]['orders'].get('fill_events',0),flush=True)
    expected=json.loads(GOLDEN.read_text())['cases']
    assert results==expected,'Consolidation changed the frozen native execution/account trace'
    print('All eight pre-refactor traces match exactly',flush=True)
