"""Registry isolation, baseline parity and causal feedback delivery."""
import copy
from dataclasses import replace
from decimal import Decimal as D
import json
from pathlib import Path

import pytest

from app.market_making.as_math import ASConfig
from app.market_making.core import Rules
from app.market_making.fusion.config import Factors,FusionConfig
from app.market_making.fusion.kernel import Kernel
from app.market_making.strategies.registry import factory_for,normalize_case,plugin_sources

ROOT=Path(__file__).resolve().parents[4]


def fixture():
    r=Rules(D('.01'),D('.01'),D('5'),D('.01'),D('100000'),D('.5'),D('1.5'))
    c=replace(FusionConfig(),as_config=replace(ASConfig(),maker_fee=D(0),half_spread_floor=0,warmup_samples=2))
    return r,c,Factors()


def case(kind,params=None):
    return normalize_case(dict(id='test',plugin=dict(type=kind,version=1,params=params or {})))


def test_kernel_rejects_missing_strategy_instead_of_silent_as():
    r,c,f=fixture()
    with pytest.raises(ValueError,match='Explicit strategy_factory'):
        Kernel(r,c,f)


@pytest.mark.parametrize('row', [{}, {'mask': 0}, {'mask': 7}, {'plugin': None}])
def test_missing_plugin_rejected_at_every_registry_boundary(row):
    from app.market_making.strategies.registry import bind_symbol
    for operation in (normalize_case, factory_for, plugin_sources):
        with pytest.raises(ValueError, match='Plugin requires|masks are retired'):
            operation(row)
    with pytest.raises(ValueError, match='Plugin requires|masks are retired'):
        bind_symbol(row, 'BTCUSDC')


def test_replay_rejects_missing_strategy_before_reading_data(tmp_path):
    from app.market_making.fusion.event_backtest import run_event_arm
    with pytest.raises(ValueError,match='Explicit strategy_factory'):
        run_event_arm(tmp_path,tmp_path/'output',0,None,None,[],None,None)
    assert not (tmp_path/'output').exists()


@pytest.mark.parametrize('value', [None, {}, {'gamma': 0.1}])
def test_retired_adaptive_settings_never_silently_fall_back(value):
    for operation in (normalize_case, factory_for, plugin_sources):
        with pytest.raises(ValueError, match='retired'):
            operation({'mask': 0, 'adaptive_as': value})
    with pytest.raises(ValueError, match='Unknown strategy'):
        case('adaptive_as', {})


@pytest.mark.parametrize('kind', ['as_base','as_fusion','as_factor_model','as_markout',
                                  'adaptive_as','fb_hjb_capital_budget','unknown'])
def test_retired_plugins_rejected_at_each_registry_boundary(kind):
    item=dict(id='old',plugin=dict(type=kind,version=1,params={}))
    for operation in (normalize_case,factory_for,plugin_sources):
        with pytest.raises(ValueError,match='Unknown strategy'):
            operation(item)


def test_supported_plugin_rejects_competing_overlays_and_wrong_versions():
    plugin=json.loads((ROOT/'config/strategies/paper_native_as_v1.json').read_text())
    original={'id':'as','plugin':plugin}
    assert normalize_case(original)['plugin']==plugin
    for changes in ({'mask':7},{'prediction':{}},{'single_long':{}},{'queue_management':{}}):
        with pytest.raises(ValueError):normalize_case(dict(original,**changes))
    for version in (0,2,True,'1'):
        invalid=copy.deepcopy(original);invalid['plugin']['version']=version
        for operation in (normalize_case,factory_for,plugin_sources):
            with pytest.raises(ValueError,match='Unknown strategy'):operation(invalid)
    assert original=={'id':'as','plugin':plugin}


def test_empty_plugin_cannot_use_the_unversioned_fallback():
    for operation in (normalize_case, factory_for, plugin_sources):
        with pytest.raises(ValueError, match='Plugin requires'):
            operation({'id': 'broken', 'plugin': {}})


@pytest.mark.parametrize('key', ['prediction', 'single_long', 'paper_reversal',
                               'quote_placement', 'queue_management'])
def test_retired_overlays_rejected_even_when_explicitly_null(key):
    from app.market_making.strategies.registry import bind_symbol
    plugin = json.loads((ROOT/'config/strategies/paper_native_as_v1.json').read_text())
    row = {'id': 'as', 'plugin': plugin, key: None}
    for operation in (normalize_case, factory_for, plugin_sources):
        with pytest.raises(ValueError, match='retired'):
            operation(row)
    with pytest.raises(ValueError, match='retired'):
        bind_symbol(row, 'BTCUSDC')


@pytest.mark.parametrize('kind', ['paper_native_as','nh_quote_value','fb_as_hjb'])
def test_reused_factory_isolates_every_mutable_strategy_state(kind):
    from app.market_making.fusion.tests.refactor_trace import nh_params
    from app.market_making.strategies.hjb_strategy import DEFAULTS
    from app.market_making.strategies.registry import bind_symbol
    if kind=='paper_native_as':
        plugin=json.loads((ROOT/'config/strategies/paper_native_as_v1.json').read_text())
    elif kind=='nh_quote_value':
        plugin=dict(type=kind,version=1,params={'models':{'BTCUSDC':nh_params(10,10**12,'BTCUSDC')}})
    else:
        plugin=dict(type=kind,version=2,params=dict(DEFAULTS,trade_start_ns=10,
            trade_end_ns=10**12,capital_utilization=None))
    original=copy.deepcopy(plugin)
    bound=bind_symbol(normalize_case(dict(id='isolation',plugin=plugin)),'BTCUSDC')
    factory=factory_for(bound)
    first,second=factory(*fixture()),factory(*fixture())
    first.counts['probe']+=1
    assert not second.counts['probe'] and first.obs is not second.obs
    if kind=='nh_quote_value':
        first.params['alpha']['beta'][0]=99
        assert second.params['alpha']['beta'][0]==0
        assert first.params is not bound['plugin']['params']
    elif kind=='paper_native_as':
        first.settings['paper_config']['symbol']='MUTATED'
        assert second.settings['paper_config'].get('symbol')!='MUTATED'
    else:
        first.settings['prior'][1]=99
        assert second.settings['prior'][1]==0
    assert plugin==original


def test_capabilities_are_central_and_profile_cannot_choose_python_module():
    from app.market_making.strategies.registry import definition_for
    plugin=json.loads((ROOT/'config/strategies/paper_native_as_v1.json').read_text())
    assert definition_for({'plugin':plugin}).supports_prewindow
    plugin['module']='unreviewed.module'
    with pytest.raises(ValueError,match='type/version/params'):
        normalize_case({'plugin':plugin})
