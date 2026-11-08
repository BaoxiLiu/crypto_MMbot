"""Independent frozen pre-refactor exchange traces and the public family surface."""
import importlib.util
import json
from dataclasses import replace
from decimal import Decimal as D

import pytest

from app.market_making.fusion.tests.refactor_trace import CASES,GOLDEN,replay_case

NATIVE=importlib.util.find_spec('hftbacktest') is not None


@pytest.mark.skipif(not NATIVE,reason='Verified local HFT runtime required')
@pytest.mark.parametrize('name',CASES)
def test_frozen_native_queue_receipts_ledger_and_equity_are_identical(tmp_path,name):
    expected=json.loads(GOLDEN.read_text())['cases'][name]
    assert replay_case(tmp_path,name)==expected


def test_public_registry_only_exposes_three_main_families():
    from app.market_making.strategies.registry import available_strategies
    assert {r['type'] for r in available_strategies()}=={'paper_native_as','nh_quote_value','fb_as_hjb'}
    assert not {'fb_hjb_capital_budget','as_markout','adaptive_as'} & {r['type'] for r in available_strategies()}


def test_all_hjb_variants_use_one_class():
    from app.market_making.strategies.hjb_strategy import DEFAULTS,HJBStrategy,factory_for
    from app.market_making.as_math import ASConfig
    from app.market_making.core import Rules
    from app.market_making.fusion.config import Factors,FusionConfig
    rules=Rules(D('.01'),D('.01'),D('5'),D('.01'),D(10000),D('.5'),D('1.5'))
    config=replace(FusionConfig(),as_config=replace(ASConfig(),position_cap=D(10)))
    for adaptive,utilization in ((False,None),(True,None),(True,'0.5')):
        settings=dict(DEFAULTS,trade_start_ns=1,trade_end_ns=10**12,
                      adaptive=adaptive,capital_utilization=utilization)
        assert type(factory_for(settings)(rules,config,Factors())) is HJBStrategy


@pytest.mark.skipif(not NATIVE,reason='Verified local HFT runtime required')
@pytest.mark.parametrize('volatility,markout',[(False,False),(False,True),(True,False),(True,True)])
def test_as_improvement_modules_are_independently_parameterized(volatility,markout):
    from app.market_making.strategies.as_strategy import ASStrategy
    from app.market_making.fusion.tests.test_paper_native_as import BASE,fixture
    from app.market_making.fusion.config import Factors,FusionConfig
    settings=dict(mode='paper_s1_sc0',paper_config=BASE,trade_start_ns=1,trade_end_ns=10**15,
                  min_half_spread_ticks=2,volatility_adjustment=volatility,markout_adjustment=markout)
    rules,_,_=fixture()
    strategy=ASStrategy(rules,FusionConfig(),Factors(),settings,'TESTUSDC')
    assert (strategy.volatility is not None)==volatility
    assert (strategy.controller is not None)==markout
    assert strategy.paper_config['as_min_half_spread_ticks']=='2'
    # Exercise actual quoting and mature feedback, not just constructor flags.
    for i in range(30):
        strategy.book(10**12+i*10**9,D('99.99')+D(i%2)/100,D('100.01')+D(i%2)/100)
    now=10**12+30*10**9
    strategy.paper_config['warmup_seconds']=0
    strategy.model.alpha=D(1);strategy.model.kappa=D(50)
    strategy.model.last_fit_ms=now//10**6;strategy.model.calibrate_gamma()
    if strategy.volatility:
        strategy.volatility.snapshot=lambda ns:dict(valid=True,sigma_ticks=D(1),missing_ratio=0)
    strategy.on_top_of_book(now,strategy.obs.bid,D(10),strategy.obs.ask,D(10))
    strategy.before_plan([],D(600),D(100))
    assert strategy.desired(D(0),now)
    for i in range(60):
        strategy.on_markout(dict(fill=dict(ns=now-10**9,side=1,qty=D('.25'),price=D(100),cid=str(i)),
            fill_id=i,horizon_ms=1000,target_ns=now,observed_ns=now,mid=D(99)))
    assert bool(strategy.counts['sc_adjustments'])==markout


@pytest.mark.skipif(not NATIVE,reason='Verified local HFT runtime required')
def test_original_as_cannot_silently_enable_improved_modules():
    from app.market_making.strategies.as_strategy import validate_settings
    from app.market_making.fusion.tests.test_paper_native_as import BASE
    for key in ('volatility_adjustment','markout_adjustment'):
        settings=dict(mode='as_base',paper_config=BASE,trade_start_ns=1,trade_end_ns=10**12)
        with pytest.raises(ValueError):validate_settings(dict(settings,**{key:True}))
