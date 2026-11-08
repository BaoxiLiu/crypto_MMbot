from dataclasses import replace
from decimal import Decimal as D

import pytest
from app.market_making.as_math import ASConfig
from app.market_making.core import Rules
from app.market_making.fusion.config import Factors,FusionConfig
from app.market_making.strategies.hjb_strategy import DEFAULTS
from app.market_making.fusion.kernel import Kernel
from app.market_making.contracts import Desired

from app.market_making.strategies.hjb_strategy import HJBStrategy


def make():
    rules=Rules(D('.01'),D('.01'),D('5'),D('.01'),D('100000'),D('.5'),D('1.5'))
    config=replace(FusionConfig(),as_config=replace(ASConfig(),position_cap=D(10),
                   maker_fee=D(0),half_spread_floor=0,warmup_samples=2))
    settings=dict(DEFAULTS,trade_start_ns=1,trade_end_ns=100_000_000_000,capital_utilization='0.5')
    strategy=HJBStrategy(rules,config,Factors(),settings)
    for i in range(5):
        strategy.book(1+i*100_000_000,D(100),D('100.02'))
    return strategy,config


@pytest.mark.parametrize('position',[D(4),D(-4)])
def test_budget_breach_only_reduces_without_crossing_flat(position):
    s,config=make()
    s.before_plan([],D(600),D('100.01'))
    quotes=s.desired(position,1_000_000_001)
    assert set(quotes)=={-1 if position>0 else 1}
    for side,q in quotes.items():
        assert q.reduce_only and abs(position+side*q.qty)<abs(position)
        assert position*(position+side*q.qty)>=0
        assert q.price>=s.obs.ask if side<0 else q.price<=s.obs.bid
    assert config.as_config.position_cap==10 and s.original_config is config


def test_cached_targets_are_clipped_each_decision_to_equity_and_quote_price():
    s,_=make()
    s.next_solve=10**12
    s.targets={1:Desired(1,D('105'),D(10)),-1:Desired(-1,D(100),D(10))}
    for equity in (D(600),D(500),D(450)):
        s.before_plan([],equity,D('100.01'))
        position=D('1.7')
        for side,q in s.desired(position,1_000_000_001).items():
            assert abs(position+side*q.qty)*max(q.price,D('100.01'))<=equity/2


def test_pre_window_and_nonpositive_equity_never_open_inventory():
    s,_=make()
    s.before_plan([],D(600),D('100.01'))
    assert s.desired(D(0),0)=={}
    s.before_plan([],D(-1),D('100.01'))
    assert s.desired(D(0),1_000_000_001)=={}


def test_common_kernel_risk_guard_still_latches_before_policy():
    s,config=make()
    kernel=Kernel(s.rules,config,Factors(),strategy_factory=lambda *args:s)
    kernel.oms.ledger.position=D(6)
    kernel.oms.ledger.average_entry=D(100)
    kernel.oms.ledger.wallet=D(600)
    kernel.mark_price(1_000_000_001,D('100.01'))
    assert kernel.plan(1_000_000_001)=={}
    assert kernel.oms.halt_reason=='margin_stress_breach_evaluation_halted'
    assert kernel.oms.config is config and kernel.config is config


def test_budget_shrink_forces_cancel_of_existing_risk_increasing_quote():
    s,config=make()
    kernel=Kernel(s.rules,config,Factors(),strategy_factory=lambda *args:s)
    initial={1:Desired(1,D(100),D(2))}
    commands=kernel.oms.reconcile_targets(initial,D(100),1_000_000_001)
    assert len(commands)==1 and commands[0].kind=='new'
    kernel.oms.update(commands[0].cid,"NEW",D(0),D(0),1_000_000_001)
    # Same price but smaller desired size must force cancellation even before
    # minimum quote lifetime, leaving pending risk reserved until its ACK.
    s.next_solve=10**12
    s.targets=initial
    s.before_plan(kernel.oms.working(),D(100),D(100))
    reduced=s.desired(D(0),1_000_000_002)
    commands=kernel.oms.reconcile_targets(reduced,D(100),1_000_000_002)
    assert len(commands)==1 and commands[0].kind=='cancel'
    assert kernel.oms.pending(1)==D(2)


@pytest.mark.parametrize('value',['0','1','-0.1','NaN','Infinity'])
def test_budget_plugin_rejects_unsafe_or_missing_fraction(value):
    from app.market_making.strategies.registry import normalize_case
    settings=dict(DEFAULTS,trade_start_ns=1,trade_end_ns=100_000_000_000,capital_utilization=value)
    with pytest.raises(ValueError):
        normalize_case(dict(id='invalid',plugin=dict(type='fb_as_hjb',version=2,params=settings)))


def test_registered_budget_has_complete_identity_and_sources():
    from app.market_making.strategies.registry import normalize_case,factory_for,plugin_sources
    s,c=make()
    params=dict(DEFAULTS,trade_start_ns=1,trade_end_ns=100_000_000_000,capital_utilization='0.5')
    case=normalize_case(dict(id='budget',plugin=dict(type='fb_as_hjb',version=2,params=params)))
    plugin=factory_for(case)(s.rules,c,Factors())
    assert plugin.identity['params']==params
    assert plugin.identity['type']=='fb_as_hjb'
    assert plugin.identity['version']==2
    assert 'app/market_making/strategies/hjb_strategy.py' in plugin_sources(case)
