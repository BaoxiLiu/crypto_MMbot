from copy import deepcopy
from dataclasses import replace
from decimal import Decimal as D
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.market_making.as_math import ASConfig
import importlib.util
HFT_AVAILABLE = importlib.util.find_spec("hftbacktest") is not None
from app.market_making.core import Account,Rules
from app.market_making.fusion.config import FusionConfig,Factors
from app.market_making.contracts import Phase
from app.market_making.fusion.orders import Working
from app.market_making.strategies.as_strategy import IntentAccount,ASStrategy
from app.market_making.policy import as_quotes
from app.market_making.s1_config import make_two_strategy_config
from app.market_making.spread_control import SpreadController

ROOT=Path(__file__).resolve().parents[4]
BASE=json.loads((ROOT/'app/market_making/fixtures/account_matrix.json').read_text())


def fixture():
    rules=Rules(D('.01'),D('.01'),D('5'),D('.01'),D('100000'),D('.5'),D('1.5'))
    cfg=make_two_strategy_config(BASE)
    row=next(c for c in cfg.pop('accounts') if c['name']=='paper_s1_sc0')
    cfg.update(row)
    book=SimpleNamespace(mid=D('100'),bid=D('99.99'),ask=D('100.01'))
    return rules,cfg,book


def test_original_quote_policy_and_intention_capture_have_identical_new_prices_and_sizes():
    rules,cfg,book=fixture()
    old=Account('test',deepcopy(cfg),rules,lambda *args:None)
    new=IntentAccount('test',deepcopy(cfg),rules)
    old.quote(D('99.9'),D('100.1'),D('100'),1_000_000,book,fv=D('100'))
    targets=new.decide(D('99.9'),D('100.1'),D(0),1_000_000,book,D('100'),[],D(600))
    assert set(targets)==set(old.orders)=={1,-1}
    for side,o in old.orders.items():
        assert (targets[side].price,targets[side].qty,targets[side].reduce_only)==(o.price,o.remaining,o.reduce_only)


def test_original_retain_and_adverse_cancel_are_preserved_without_paper_execution():
    rules,cfg,book=fixture();a=IntentAccount('test',cfg,rules)
    working=[Working('bid',1,1,D('99.9'),D('.25'),1_000_000_000,phase=Phase.LIVE)]
    keep=a.decide(D('99.9'),D('100.1'),D(0),2000,book,D('100'),working,D(600))
    assert keep[1].price==D('99.9') and not a.urgent
    retreat=a.decide(D('99.8'),D('100.1'),D(0),3000,book,D('100'),working,D(600))
    assert 1 not in retreat and 1 in a.urgent


@pytest.mark.skipif(not HFT_AVAILABLE,reason='Run in the verified local HFT runtime')
@pytest.mark.parametrize('mode',['as_base','paper_s1_sc0'])
def test_adapter_uses_actual_native_prices_and_original_floor_policy(mode):
    rules,_,_=fixture()
    config=replace(FusionConfig(),as_config=replace(ASConfig(),maker_fee=D(0),half_spread_floor=0))
    settings=dict(mode=mode,paper_config=BASE,trade_start_ns=1,trade_end_ns=10**15)
    a=ASStrategy(rules,config,Factors(),settings,'TESTUSDC')
    for i in range(30):
        a.book(10**12+i*10**9,D('99.99')+D(i%2)/100,D('100.01')+D(i%2)/100)
    now=10**12+30*10**9;ms=now//10**6
    a.paper_config['warmup_seconds']=0
    a.model.alpha=D('1');a.model.kappa=D('50');a.model.last_fit_ms=ms
    a.model.calibrate_gamma()
    if a.volatility:
        a.volatility.snapshot=lambda now:dict(valid=True,sigma_ticks=D(8),missing_ratio=0)
        a.account.update_volatility(a.volatility.snapshot(ms),ms)
    a.on_top_of_book(now,a.obs.bid,D(10),a.obs.ask,D(10))
    a.before_plan([],D(600))
    book=SimpleNamespace(mid=a.obs.mid,bid=a.obs.bid,ask=a.obs.ask)
    bid,ask,*_=as_quotes(a.model,book,D(0),ms,a.paper_config,rules,micro=a.micro,
                       floor_ticks=a.account.floor_state()['as_floor_ticks'])
    targets=a.desired(D(0),now)
    assert targets
    assert targets[1].price==rules.price(bid,1)
    assert targets[-1].price==rules.price(ask,-1)
    assert a.account.floor_state()['as_floor_ticks']==(16 if mode=='paper_s1_sc0' else 0)
    assert a.identity['implementation_dependencies']
    # Historical context may build estimates but must never open positions.
    a.settings['trade_start_ns']=now+1
    assert a.desired(D(0),now)=={}


@pytest.mark.skipif(not HFT_AVAILABLE,reason='Native runtime required')
def test_sc0_causality_lag_dedup_and_mature_history_across_gap():
    rules,_,_=fixture();config=replace(FusionConfig(),as_config=replace(ASConfig(),maker_fee=D(0)))
    a=ASStrategy(rules,config,Factors(),dict(mode='paper_s1_sc0',paper_config=BASE,trade_start_ns=1,trade_end_ns=10**15),'TESTUSDC')
    initial=a.account.sc_target_ticks
    for i in range(50):
        f=dict(fill=dict(ns=10**12,side=1,qty=D(1),price=D(100),cid=str(i)),fill_id=i,
               horizon_ms=1000,target_ns=10**12+10**9,observed_ns=10**12+10**9,mid=D(99))
        a.on_markout(f)
    assert a.account.sc_target_ticks>initial
    n=a.controller.feedback_seq;a.on_markout(f)
    assert a.controller.feedback_seq==n
    f['fill_id']=100;f['observed_ns']+=250_000_001;a.on_markout(f)
    assert a.controller.feedback_seq==n
    a.reset_market()
    assert a.account.sc_target_ticks>initial and a.controller.feedback_seq==n


def test_rebate_config_preserves_signed_fees():
    assert replace(ASConfig(),maker_fee=D('-0.00003')).maker_fee==D('-0.00003')
