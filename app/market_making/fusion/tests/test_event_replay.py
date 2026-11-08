"""Causality contracts for event execution; all inputs synthetic/offline."""
import asyncio,hashlib,json
from dataclasses import replace
from decimal import Decimal as D
from types import SimpleNamespace
import numpy as np
import polars as pl
import hftbacktest as h
import pytest
from app.backtesting.datasets import events
from app.market_making.as_math import ASConfig
from app.market_making.core import Rules
from app.market_making.fusion.config import FusionConfig,Factors
from app.market_making.fusion.execution import ExecutionProfile
from app.market_making.fusion.event_backtest import EventCursor,run_event_arm,wake
from app.market_making.live.engine import StrategyEngine
from app.market_making.live.runtime import Runtime
from app.market_making.fusion.timing import next_timer_ns
from app.market_making.fusion.tests.quote_fixture import QuoteFixture


def profile():
    return ExecutionProfile(name='event-fixture',trigger='event',feed_ns=1_500_000,
                            decision_ns=100_000,request_ns=500_000,response_ns=500_000)


def kernel():
    rules=Rules(D('.01'),D('.01'),D('5'),D('.01'),D('10000'),D('.9'),D('1.1'))
    config=replace(FusionConfig(),as_config=replace(ASConfig(),maker_fee=D(0),warmup_samples=2))
    return StrategyEngine(rules,config,Factors(), strategy_factory=QuoteFixture)


def test_publication_wait_is_not_erased_or_added_twice():
    df=pl.DataFrame(dict(is_buyer_maker=[True,False],trade_time=[1000,1001],
                         event_time=[1003,None],trade_id=[1,2],price=[100.,101.],quantity=[1.,1.]))
    data=events(df,trades=True,publication_ns=1_500_000)
    assert data['exch_ts'].tolist()==[1_000_000_000,1_001_000_000]
    assert data['local_ts'][0]==1_004_500_000
    assert int(data['ev'][0]) & h.LOCAL_EVENT
    assert not int(data['ev'][1]) & h.LOCAL_EVENT
    assert int(data['ev'][1]) & h.EXCH_EVENT


def test_runtime_decides_on_three_events_one_ms_apart_without_100ms_gate():
    k=kernel()
    for t in (0,100_000_000,200_000_000):k.book(t,D(100),D('100.02'))
    k.mark_price(200_000_000,D(100))
    clock=SimpleNamespace(value=200_000_000);clock.ns=lambda:clock.value
    journal=SimpleNamespace(record=lambda *a,**kw:None,load=lambda:None,save=lambda *a:None)
    r=Runtime(k,None,journal,'DOGEUSDC','USDC','shadow',clock)
    r.connected.update(public=True,market=True)
    calls=[];original=k.plan
    k.plan=lambda now:(calls.append(now),original(now))[1]
    handle=r.handle
    def on_event(*item):
        clock.value=item[2];handle(*item)
        if item[1][1]['u']==3:r.stopping=True
    r.handle=on_event
    async def reader(*args):pass
    r.reader=reader
    async def run():
        for i in range(3):
            await r.queue.put(('event',('public',dict(e='bookTicker',s='DOGEUSDC',u=i+1,b='100',a='100.02')),200_000_000+i*1_000_000))
        await r.run()
    asyncio.run(run())
    assert calls==[200_000_000,201_000_000,202_000_000]


def test_silence_rearms_staleness_timer_independent_of_new_market_events():
    k=kernel();k.book(0,D(100),D('100.02'));k.mark_price(0,D(100))
    assert next_timer_ns(k,900_000_000)==1_000_000_001
    assert not k.plan(1_000_000_001)


def synthetic_tape(root,extras=()):
    rows=[]
    for t in (1_000_000_000,1_100_000_000,1_200_000_000,1_300_000_000,1_400_000_000):
        typ=h.DEPTH_SNAPSHOT_EVENT if t==1_000_000_000 else h.DEPTH_EVENT
        for side,price in ((h.BUY_EVENT,100.),(h.SELL_EVENT,100.02)):
            rows.extend([(h.EXCH_EVENT|typ|side,t,t+1_500_000,price,10.,0,0,0.),
                         (h.LOCAL_EVENT|typ|side,t,t+1_500_000,price,10.,0,0,0.)])
    for t,typ,side,price,qty in extras:
        rows.extend([(h.EXCH_EVENT|typ|side,t,t+1_500_000,price,qty,0,0,0.),
                     (h.LOCAL_EVENT|typ|side,t,t+1_500_000,price,qty,0,0,0.)])
    data=np.array(rows,dtype=h.event_dtype)
    clock=np.where((data['ev'] & h.EXCH_EVENT)!=0,data['exch_ts'],data['local_ts'])
    data=data[np.argsort(clock,kind='stable')]
    p=root/'data.npz';np.savez(p,data=data)
    part=dict(path=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),rows=len(data),clock_end_ns=1_401_500_000)
    manifest=dict(version=6,symbol='DOGEUSDC',unresolved_trade_gaps=[],feed_clock='publication_E_plus_transport_exchange_T',
                  feed_latency_ms=1.5,segments=[dict(id=0,files=[part])])
    (root/'manifest.json').write_text(json.dumps(manifest))
    marks=np.array([(1_001_500_000,100.),(1_201_500_000,100.)],dtype=[('local_ts','i8'),('price','f8')])
    np.save(root/'marks.npy',marks)
    return part,data


def test_native_wake_applies_all_levels_at_same_receipt_timestamp(tmp_path):
    _,data=synthetic_tape(tmp_path)
    b=h.HashMapMarketDepthBacktest([h.BacktestAsset().data(data).linear_asset(1).tick_size(.01).lot_size(.01)
          .constant_order_latency(500_000,500_000).risk_adverse_queue_model().partial_fill_exchange()])
    try:
        b.elapse(1)
        assert wake(b,1_000_000_000)==2
        assert int(b.current_timestamp)==1_001_500_000
        assert b.depth(0).best_bid==100. and b.depth(0).best_ask==100.02
    finally:b.close()


def test_native_event_replay_observes_ack_without_next_market_tick(tmp_path):
    synthetic_tape(tmp_path);k=kernel()
    result=run_event_arm(tmp_path,tmp_path/'out',0,k.config,tmp_path/'marks.npy',[],k.rules,profile(), strategy_factory=QuoteFixture)
    assert result['invariant_failures']==0
    assert result['orders']['new_requests']==2
    assert result['counters']['response_wakes']>=1
    assert result['counters']['decisions']>=5
    assert result['native_fills']==0


def test_busy_feed_is_not_lost_and_fill_precedes_public_trade_and_funding(tmp_path):
    synthetic_tape(tmp_path,[(1_200_050_000,h.DEPTH_EVENT,h.BUY_EVENT,100.,10.),
                             (1_202_200_000,h.TRADE_EVENT,h.SELL_EVENT,99.95,20.)])
    k=kernel()
    funding=[dict(symbol='DOGEUSDC',fundingTime=1203,fundingRate='.01',markPrice='100')]
    result=run_event_arm(tmp_path,tmp_path/'out',0,k.config,tmp_path/'marks.npy',funding,k.rules,profile(), strategy_factory=QuoteFixture)
    assert result['counters']['busy_rechecks']>=2
    assert result['native_fills']>=1
    receipt=result['funding_receipts'][0]
    assert receipt['receipt_ns']==1_203_500_000
    assert D(receipt['position'])==D('.25')
    assert D(receipt['cashflow'])==D('-.25')


def test_last_exchange_print_and_response_are_drained_before_end_of_tape(tmp_path):
    synthetic_tape(tmp_path,[(1_400_000_000,h.TRADE_EVENT,h.SELL_EVENT,99.95,20.)])
    k=kernel()
    result=run_event_arm(tmp_path,tmp_path/'out',0,k.config,tmp_path/'marks.npy',[],k.rules,profile(), strategy_factory=QuoteFixture)
    assert result['native_fills']==1
    assert D(result['position'])==D('.25')
    assert result['segments'][0]['end_ns']==1_401_500_000


def test_input_grid_is_checked_before_native_rounding_can_hide_bad_data(tmp_path):
    part,data=synthetic_tape(tmp_path)
    data['px'][0]+=.005
    p=tmp_path/'data.npz';np.savez(p,data=data)
    part['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='off-grid'):
        EventCursor(tmp_path,[part],kernel().rules)


def test_compute_ending_exactly_on_cutoff_still_observes_final_mark(tmp_path):
    synthetic_tape(tmp_path,[(1_398_800_000,h.DEPTH_EVENT,h.BUY_EVENT,100.01,10.)])
    marks=np.array([(1_001_500_000,100.),(1_201_500_000,100.),(1_401_500_000,101.)],
                   dtype=[('local_ts','i8'),('price','f8')])
    np.save(tmp_path/'marks.npy',marks)
    k=kernel()
    result=run_event_arm(tmp_path,tmp_path/'out',0,k.config,tmp_path/'marks.npy',[],k.rules,profile(), strategy_factory=QuoteFixture)
    assert D(result['valuation_mark'])==101
    assert result['segments'][0]['end_ns']==1_401_500_000


def test_settlement_price_cannot_leak_into_risk_before_publication(tmp_path):
    synthetic_tape(tmp_path,[(1_202_200_000,h.TRADE_EVENT,h.SELL_EVENT,99.95,20.)])
    k=kernel();config=replace(k.config,drawdown_limit=D('5'))
    # Exchange settlement knows 150, while public marks visible locally are
    # still 100. Charging actual cash must not invent a 150-valued local peak.
    funding=[dict(symbol='DOGEUSDC',fundingTime=1203,fundingRate='.0001',markPrice='150')]
    result=run_event_arm(tmp_path,tmp_path/'out',0,config,tmp_path/'marks.npy',funding,k.rules,profile(), strategy_factory=QuoteFixture)
    assert D(result['funding'])==D('-.00375')
    assert D(result['max_drawdown'])<1
    assert not result['halt_reason']
