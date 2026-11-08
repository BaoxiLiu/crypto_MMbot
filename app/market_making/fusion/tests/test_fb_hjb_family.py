"""Canonical family variants must preserve the existing trading policies."""
from copy import deepcopy
from dataclasses import replace
from decimal import Decimal as D

import numpy as np
import pytest

from app.market_making.as_math import ASConfig
from app.market_making.core import Rules
from app.market_making.fusion.config import Factors,FusionConfig
from app.market_making.strategies.hjb_strategy import DEFAULTS
from app.market_making.strategies.registry import normalize_case,factory_for,plugin_sources,available_strategies


def params(**values):
    return dict(deepcopy(DEFAULTS),trade_start_ns=1,trade_end_ns=100_000_000_000,**values)


def case(settings,version=2):
    return normalize_case(dict(id='family',plugin=dict(type='fb_as_hjb',version=version,params=settings)))


@pytest.mark.parametrize('adaptive,utilization',[(False,None),(True,None),(True,'0.5'),(False,'0.5')])
def test_family_learns_and_enforces_budget_without_mutating_common_config(adaptive,utilization):
    rules=Rules(D('.01'),D('.01'),D('5'),D('.01'),D('100000'),D('.5'),D('1.5'))
    config=replace(FusionConfig(),max_notional=D(1000),as_config=replace(
        ASConfig(),position_cap=D(10),maker_fee=D(0),half_spread_floor=0,warmup_samples=2))
    settings=params();settings['adaptive']=adaptive
    p=case(dict(settings,capital_utilization=utilization))
    new=factory_for(p)(rules,config,Factors())
    pending=[]
    for i in range(60):
        ns=1+i*100_000_000;bid=D(100)+D(i%3)/100
        position=D(i%4)/10
        new.book(ns,bid,bid+D('.02'))
        new.before_plan([],D(600),new.obs.mid)
        while pending and pending[0]['ns']+1_000_000_000<=ns:
            fill=pending.pop(0)
            completed=dict(horizon_ms=1000,fill=fill,target_ns=fill['ns']+1_000_000_000,
                           observed_ns=ns,mid=bid+D('.01'))
            new.on_markout(completed)
        new.desired(position,ns)
        fill=dict(cid=str(i),ns=ns,side=1 if i%2 else -1,price=bid,qty=D('.25'))
        new.on_fill(fill,position)
        pending.append(fill)
    assert np.isfinite(new.objective).all()
    assert new.counts['mature_labels']>=30
    if adaptive:assert new.counts['objective_updates']>0
    else:assert not new.counts['objective_updates']
    if utilization is not None:
        for position in (D(4),D(-4)):
            quotes=new.desired(position,ns)
            assert quotes and all(x.reduce_only for x in quotes.values())
    assert new.identity['type']=='fb_as_hjb' and new.identity['version']==2
    assert new.identity['params']==p['plugin']['params']
    assert config.as_config.position_cap==10 and config.max_notional==1000
    assert 'app/market_making/strategies/hjb_strategy.py' in plugin_sources(p)


@pytest.mark.parametrize('value',[True,False,'bad','NaN','Infinity',0,1,-.1])
def test_invalid_budget_fraction_rejected(value):
    with pytest.raises(ValueError):case(params(capital_utilization=value))


def test_old_hjb_versions_and_budget_alias_are_rejected():
    with pytest.raises(ValueError):case(params(),version=1)
    with pytest.raises(ValueError):case(params())
    with pytest.raises(ValueError):case(params(capital_utilization=None),version=3)
    with pytest.raises(ValueError):
        normalize_case(dict(id='old',plugin=dict(type='fb_hjb_capital_budget',version=1,
                                               params=params(capital_utilization='0.5'))))
    current={p['type']:p for p in available_strategies()}
    assert current['fb_as_hjb']['versions']==[2]
    assert 'fb_hjb_capital_budget' not in {p['type'] for p in available_strategies()}


@pytest.mark.parametrize('utilization', [None, '0.5'])
def test_hjb_config_normalization_is_pure_and_idempotent(utilization):
    original=dict(id='family',plugin=dict(type='fb_as_hjb',version=2,
                                        params=params(capital_utilization=utilization)))
    before=deepcopy(original)
    result=normalize_case(original)
    assert original==before
    assert result==normalize_case(result)
    assert result['plugin']==original['plugin']
