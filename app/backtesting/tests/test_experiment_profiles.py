"""Profile composition and comparison-wide data/execution/account ownership."""
import copy
import json
from pathlib import Path

import pytest

from app.backtesting.experiment_profiles import expand_profiles
from app.backtesting.fusion_experiment import validate_inputs
from app.backtesting.experiment_config import ROOT,load_spec,effective_config

EXAMPLE=ROOT/'config/experiments/default.json'


def test_composed_example_decouples_data_from_strategy_and_shares_risk():
    s=load_spec(EXAMPLE)
    assert s['engine']['queue']=='cancel_50_50'
    assert s['costs']['maker_fee']=='0' and s['execution']['taker_fee']=='0.0004'
    assert 'strategy_config' not in s['datasets']['DOGEUSDC']
    assert len(s['_profile_inputs'])==1
    configs=[effective_config(s,s['datasets']['DOGEUSDC'],c) for c in s['strategies']]
    assert all(c['max_notional']=='1000' and c['initial_cash']=='600' for c in configs)
    assert s['strategies'][0]['plugin']['type']=='paper_native_as'
    assert s['strategies'][0]['plugin']['params']['mode']=='as_base'
    assert s['strategies'][2]['plugin']['type']=='paper_native_as'
    assert s['strategies'][2]['plugin']['params']['mode']=='paper_s1_sc0'


def test_profile_crossing_module_boundary_or_overriding_inline_data_fails(tmp_path):
    p=tmp_path/'profile.json';p.write_text(json.dumps({'symbols':['DOGEUSDC']}))
    with pytest.raises(ValueError,match='boundary'):
        expand_profiles({'profiles':{'risk':str(p)}})
    with pytest.raises(ValueError,match='Conflicting'):
        expand_profiles({'profiles':{'data':str(p)},'symbols':['BTCUSDC']})


def test_retired_model_profile_is_rejected_before_reading_external_file(tmp_path):
    original={'strategies':[{'id':'old','plugin':{'type':'as_factor_model','version':1,
                  'params':{'model_profile':str(tmp_path/'missing.json')}}}]}
    before=copy.deepcopy(original)
    with pytest.raises(ValueError, match='model_profile is retired'):
        expand_profiles(original)
    assert original==before


def test_profile_drift_is_rejected_before_reading_market_data():
    s=load_spec(EXAMPLE)
    path=next(iter(s['_profile_inputs']));s['_profile_inputs'][path]='wrong'
    with pytest.raises(ValueError,match='Profile changed'):validate_inputs(s)


def test_case_cannot_override_comparison_account_budget():
    s=load_spec(EXAMPLE);c=copy.deepcopy(s['strategies'][0]);c['overrides']['max_notional']='9000'
    with pytest.raises(ValueError,match='account risk'):effective_config(s,s['datasets']['DOGEUSDC'],c)


def test_missing_or_ambiguous_parameter_baselines_rejected(tmp_path):
    s=json.loads(EXAMPLE.read_text());s['strategy_parameters']={}
    p=tmp_path/'experiment.json';p.write_text(json.dumps(s))
    with pytest.raises(ValueError,match='baselines'):load_spec(p)


def test_prepare_records_identities_without_copying_source_or_profiles(tmp_path,monkeypatch):
    from app.backtesting import fusion_experiment as module
    raw=json.loads(EXAMPLE.read_text())
    raw['outputs']['runs_root']=str(tmp_path/'runs')
    raw['outputs']['reports_root']=str(tmp_path/'reports')
    config=tmp_path/'external-config.json';config.write_text(json.dumps(raw))
    recorded=[];monkeypatch.setattr(module.run_registry,'append',recorded.append)
    registration=module.prepare(config,'profile-contract-test')
    r=json.loads(registration.read_text())
    assert [c['plugin']['type'] for c in r['cases']]==['paper_native_as']*52
    assert all(c['config']['max_notional']=='1000' for c in r['cases'])
    assert not (registration.parent/'config_snapshot').exists()
    assert not (registration.parent/'source_snapshot').exists()
    assert json.loads((registration.parent/'effective_config.json').read_text())==r['spec']
    assert set(r['spec']['_profile_inputs'])<=set(r['input_sha256'])
    assert 'app/market_making/spread_control.py' in r['source_sha256']
    assert 'app/backtesting/experiment_profiles.py' in r['source_sha256']
    assert 'app/backtesting/experiment_config.py' in r['source_sha256']
    assert json.loads((registration.parent/'status.json').read_text())['status']=='prepared'
    assert recorded[0]['config']==str(config)


def test_parameter_overrides_are_explicit_recursive_pure_and_hashed(tmp_path):
    path=tmp_path/'strategy.json'
    profile={'type':'test','version':1,'params':{'adaptive':True,'nested':{'a':1,'b':2},'actions':[1,2]}}
    path.write_text(json.dumps(profile))
    raw={'strategies':[{'id':'x','profile':str(path),'parameter_overrides':{
        'adaptive':False,'nested':{'a':3},'actions':[4]}}]}
    before=copy.deepcopy(raw)
    result=expand_profiles(raw)
    assert raw==before and json.loads(path.read_text())==profile
    assert result['strategies'][0]['plugin']==dict(profile,params={
        'adaptive':False,'nested':{'a':3,'b':2},'actions':[4]})
    assert 'parameter_overrides' not in result['strategies'][0]
    assert str(path) in result['_profile_inputs']


@pytest.mark.parametrize('value',[None,[],3])
def test_invalid_parameter_overrides_cannot_be_ignored(tmp_path,value):
    path=tmp_path/'strategy.json';path.write_text(json.dumps({'type':'as_base','version':1,'params':{}}))
    with pytest.raises(ValueError):expand_profiles({'strategies':[{'id':'x','profile':str(path),'parameter_overrides':value}]})
    with pytest.raises(ValueError):expand_profiles({'strategies':[{'id':'x','plugin':{'type':'as_base','version':1,'params':{}},'parameter_overrides':{}}]})


def test_canonical_profiles_reproduce_existing_as_presets_and_parameterize_hjb():
    from app.market_making.strategies.registry import normalize_case
    paper='config/strategies/paper_native_as_v1.json'
    hjb='config/strategies/fb_as_hjb_v2.json'
    variants=[dict(id='as_native',profile=paper,parameter_overrides={'mode':'as_base'}),
              dict(id='as_two',profile=paper,parameter_overrides={'min_half_spread_ticks':2,
                     'trade_end_ns':1785801600000000000}),
              dict(id='as_zero',profile=paper,parameter_overrides={'min_half_spread_ticks':0}),
              dict(id='hjb_fixed',profile=hjb,parameter_overrides={'adaptive':False}),
              dict(id='hjb_adaptive',profile=hjb),
              dict(id='hjb_budget',profile=hjb,parameter_overrides={'capital_utilization':'0.5'})]
    cases=expand_profiles({'strategies':variants})['strategies']
    for item in cases:
        normalize_case(item)
    assert [c['plugin']['params'].get('min_half_spread_ticks') for c in cases[1:3]] == [2, 0]
    assert cases[1]['plugin']['params']['trade_end_ns'] == 1785801600000000000
    assert cases[0]['plugin']['params']['mode'] == 'as_base'
    assert {c['plugin']['type'] for c in cases[3:]}=={'fb_as_hjb'}
    assert [c['plugin']['params']['adaptive'] for c in cases[3:]]==[False,True,True]
    assert [c['plugin']['params']['capital_utilization'] for c in cases[3:]]==[None,None,'0.5']
