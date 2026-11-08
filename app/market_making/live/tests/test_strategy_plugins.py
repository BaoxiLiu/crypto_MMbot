"""Offline integration of the public plugins with the same live entry/config path."""
from dataclasses import asdict, replace
from decimal import Decimal as D
import json
from pathlib import Path

import numpy as np
import pytest

from app.market_making.core import Rules
from app.market_making.fusion.config import FusionConfig, Factors
from app.market_making.strategies.hjb_strategy import DEFAULTS
from app.market_making.fusion.tests.refactor_trace import nh_params
from app.market_making.live.replay import load_engine
from app.market_making.live.selection import load_selection, validate_source
from app.market_making.live.engine import StrategyEngine
from app.market_making.strategies.base import QuoteStrategy

ROOT=Path(__file__).resolve().parents[4]


def inputs(tmp_path, kind):
    config=tmp_path/'config.json'
    base=FusionConfig()
    base=replace(base,as_config=replace(base.as_config,position_cap=D(1)))
    config.write_text(json.dumps(base.json()))
    rules=tmp_path/'rules.json'
    rules.write_text(json.dumps(asdict(Rules(D('.01'),D('.01'),D('5'),D('.01'),D(100000),D('.5'),D('1.5'))),default=str))
    profile=tmp_path/'strategy.json'
    start,end=1,10**15
    if kind=='nh_quote_value':
        params={'models':{'TESTUSDC':nh_params(start,end)}}
        version=1
    elif kind=='fb_as_hjb':
        params=dict(DEFAULTS,trade_start_ns=start,trade_end_ns=end,capital_utilization=None)
        version=2
    else:
        params=dict(mode='as_base',paper_config=json.loads((ROOT/'app/market_making/fixtures/account_matrix.json').read_text()),
                    trade_start_ns=start,trade_end_ns=end)
        version=1
    profile.write_text(json.dumps(dict(type=kind,version=version,params=params)))
    return config,rules,profile


@pytest.mark.parametrize('kind',['paper_native_as','nh_quote_value','fb_as_hjb'])
def test_public_profile_runs_on_live_depth_and_trade_inputs(tmp_path,kind):
    config,rules,profile=inputs(tmp_path,kind)
    k=load_engine(config,rules,'as',strategy=profile,symbol='TESTUSDC')
    assert k.strategy.identity['type']==kind
    k.snapshot(dict(lastUpdateId=0,bids=[['100','10']],asks=[['100.02','12']]),0)
    targets={}
    for i in range(71):
        ns=i*10**9+100
        k.depth(dict(U=i,u=i+1,pu=i,b=[['100','10']],a=[['100.02','12']]),ns)
        k.mark_price(ns,D('100.01'))
        k.trade(ns,i,1 if i%2 else -1,D('100.01'),D('.2'))
        targets=k.plan(ns)
    if kind=='nh_quote_value':
        assert k.strategy.sample_ns==ns
        assert np.isfinite(k.strategy.x).all()
        assert targets and k.strategy.depth.bid_qty_at_tick(10000)==10.
    elif kind=='fb_as_hjb':
        assert len(k.strategy.reaches)==71 and targets
    else:
        assert k.strategy.micro is not None
        assert k.strategy.account_equity==k.config.initial_cash
    assert not k.pending_trades
    assert k.selected_case['symbol']=='TESTUSDC'
    k.reset_market()
    assert not k.pending_trades and not k.market_ready


def test_nh_profile_requires_depth_and_exact_symbol(tmp_path):
    _,_,profile=inputs(tmp_path,'nh_quote_value')
    case=load_selection(profile,'TESTUSDC')
    with pytest.raises(ValueError,match='full depth'):
        validate_source(case,'bbo')
    validate_source(case,'depth')
    with pytest.raises(ValueError,match='missing'):
        load_selection(profile,'OTHERUSDC')


def test_live_market_input_preserves_receipt_time_deduplicates_and_resets():
    class Observer(QuoteStrategy):
        requires_quote_depth=False
        def observation_due(self,ns):return True
        def observe_market(self,depth,trades,ns):self.received=trades.copy()
        def desired(self,position,ns):return {}
    r=Rules(D('.01'),D('.01'),D('5'),D('.01'),D(100000),D('.5'),D('1.5'))
    k=StrategyEngine(r,FusionConfig(),Factors(),strategy_factory=Observer)
    k.book(100,D('100'),D('100.02'))
    assert k.trade(110,1,1,D('100'),D(1),received_ns=105)
    assert not k.trade(111,1,1,D('100'),D(1),received_ns=105)
    k.observe_market(112)
    assert k.strategy.received['local_ns'].tolist()==[105]
    assert k.strategy.received['side'].tolist()==[1]
    assert k.pending_trades==[]
    k.trade(120,2,-1,D('100'),D(1))
    k.reset_market()
    assert k.pending_trades==[]
    assert not k.observe_market(130)

def test_replay_archives_selected_profile_and_quote_asset(tmp_path):
    from app.market_making.live.replay import run_file
    config,rules,profile=inputs(tmp_path,'fb_as_hjb')
    events=tmp_path/'input.jsonl'
    events.write_text(json.dumps(dict(kind='book',ns=100,bid='100',ask='100.02'))+'\n')
    output=tmp_path/'out'
    summary=run_file(events,output,config,rules,'as',strategy=profile,
                     symbol='TESTUSDC',quote='USDC')
    assert summary['strategy']['plugin']['type']=='fb_as_hjb'
    assert summary['quote']=='USDC'
    assert str(profile) in summary['inputs']
    assert json.loads((output/'summary.json').read_text())==summary
    second=tmp_path/'repeat'
    repeated=run_file(events,second,config,rules,'as',strategy=profile,
                      symbol='TESTUSDC',quote='USDC')
    assert repeated == summary
    assert (second/'events.jsonl').read_bytes() == (output/'events.jsonl').read_bytes()
    with pytest.raises(FileExistsError):
        run_file(events,output,config,rules,'as',strategy=profile,
                 symbol='TESTUSDC',quote='USDC')


def test_replay_api_cannot_select_implicit_legacy_strategy(tmp_path):
    from app.market_making.live.replay import run_file
    config,rules,profile=inputs(tmp_path,'fb_as_hjb')
    with pytest.raises(TypeError, match='strategy'):
        load_engine(config,rules)
    with pytest.raises(TypeError, match='strategy'):
        run_file(tmp_path/'missing.jsonl',tmp_path/'out',config,rules)
    assert not (tmp_path/'out').exists()
    with pytest.raises(ValueError, match='protection policy'):
        load_engine(config,rules,'as-rfi',strategy=profile,symbol='TESTUSDC')
