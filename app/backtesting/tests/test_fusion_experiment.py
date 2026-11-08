import copy
import json
from pathlib import Path

import pytest
from app.backtesting.fusion_experiment import default_run_id,validate_inputs
from app.backtesting.experiment_config import ROOT,effective_config,load_spec


def config():
    return json.loads((ROOT/'config/experiments/default.json').read_text())


def test_effective_config_explicitly_disables_floor_and_only_specified_loss_limits():
    s=config();c=effective_config(s,s['datasets']['DOGEUSDC'],s['strategies'][0])
    original=json.loads((ROOT/s['strategy_parameters']['DOGEUSDC']).read_text())
    assert {k for k in c if c[k]!=original[k]}=={'daily_loss','drawdown_limit','max_notional'}
    expected=dict(original['as_config']);expected['half_spread_floor']=0
    assert c['as_config']==expected
    assert c['daily_loss']==c['drawdown_limit']=='1E+30'
    assert c['max_notional']=='1000' and c['as_config']['position_cap']==original['as_config']['position_cap']


@pytest.mark.parametrize('section,key,value',[
    ('engine','queue','prob_queue'),('engine','exchange','no_partial_fill'),
    ('observations','markout_anchor','exchange_fill'),('observations','markout_horizons_ms',[100,100]),
    ('observations','max_markout_lag_ms',-1),('resources','workers',0),('window','start','2026-10-03'),
])
def test_unsupported_or_inconsistent_configuration_rejected(tmp_path,section,key,value):
    s=config();s[section][key]=value;p=tmp_path/'spec.json';p.write_text(json.dumps(s))
    with pytest.raises(ValueError):load_spec(p)


def test_wrong_cached_date_range_is_not_silently_reused(tmp_path):
    s=config();s['window']['start']='2026-09-20'
    with pytest.raises(ValueError,match='range differs'):validate_inputs(s)


def test_case_specific_fee_confounds_are_rejected():
    s=config();case=copy.deepcopy(s['strategies'][0]);case['overrides']={'as_config':{'maker_fee':'.0002'}}
    with pytest.raises(ValueError,match='fees'):effective_config(s,s['datasets']['DOGEUSDC'],case)


def test_unknown_top_level_setting_is_not_ignored(tmp_path):
    s=config();s['queue_model']='unknown';p=tmp_path/'spec.json';p.write_text(json.dumps(s))
    with pytest.raises(ValueError,match='Unknown'):load_spec(p)


def test_retired_paper_reversal_config_is_rejected(tmp_path):
    s=config();s['strategies'][0]['paper_reversal']={'stop_bp':25}
    p=tmp_path/'spec.json';p.write_text(json.dumps(s))
    with pytest.raises(ValueError,match='retired'):load_spec(p)


def test_isolated_new_engine_does_not_silently_migrate_other_strategies(tmp_path):
    s=config();s['engine']['version']='2.4.4+makerfix3'
    p=tmp_path/'spec.json';p.write_text(json.dumps(s))
    with pytest.raises(ValueError,match='Unsupported engine'):load_spec(p)


def test_automatic_identity_contains_coin_dates_config_and_hash():
    from datetime import datetime,timezone
    s=config();s['name']='doge-floor-5bp-vs-0'
    name=default_run_id(s,'abc12345ffff',datetime(2026,10,20,12,30,tzinfo=timezone.utc))
    assert name.startswith('20261020T123000000000Z_BTCUSDC-n13_20260801-20260804_doge-floor-5bp-vs-0_')
    assert name.endswith('abc12345') and len(name)<=100


def test_multi_coin_long_identity_remains_bounded_and_unique():
    s=config();s['symbols']=['DOGEUSDC']+[f'COIN{i}USDC' for i in range(12)];s['name']='a'*100
    assert len(default_run_id(s,'12345678abcd'))<=100


def test_revised_report_refuses_to_overwrite_an_existing_directory(tmp_path):
    from app.backtesting.experiment_report import report
    out=tmp_path/'frozen';out.mkdir();sentinel=out/'README.md';sentinel.write_text('frozen report')
    registration=tmp_path/'registration.json'
    registration.write_text(json.dumps(dict(kind="new_run",spec={},input_sha256={},source_sha256={},report_output=str(out))))
    with pytest.raises(FileExistsError):report(registration,out)
    assert sentinel.read_text()=='frozen report'


def test_report_output_option_cannot_silently_apply_to_run(monkeypatch,tmp_path):
    import sys
    from app.backtesting.fusion_experiment import main
    monkeypatch.setattr(sys,'argv',['fusion_experiment','run','--report-output',str(tmp_path/'new')])
    with pytest.raises(SystemExit) as error:main()
    assert error.value.code==2 and not (tmp_path/'new').exists()


def test_removing_spread_floor_does_not_disable_as_liquidity_spread():
    from app.market_making.as_math import as_prices
    s=config();case=copy.deepcopy(s['strategies'][0]);case['overrides']={'as_config':{'half_spread_floor':0}}
    c=effective_config(s,s['datasets']['DOGEUSDC'],case)
    original=effective_config(s,s['datasets']['DOGEUSDC'],s['strategies'][0])
    original['as_config']['half_spread_floor']=.0005  # Compare against the explicitly floored historical scenario.
    assert {k for k in c['as_config'] if c['as_config'][k]!=original['as_config'][k]}=={'half_spread_floor'}
    assert c['as_config']['half_spread_floor']==0
    a=c['as_config'];_,half=as_prices(10000,0,0,a['gamma'],a['kappa'],a['horizon_s'])
    assert half>4  # Removing an explicit floor must not turn AS into zero spread.


def test_run_registration_executes_prepared_identity_without_preparing_again(tmp_path, monkeypatch):
    import sys
    from app.backtesting import fusion_experiment as module
    registration=tmp_path/'registration.json'
    calls=[]
    def unexpected_prepare(*args,**kwargs):
        pytest.fail('An explicitly registered run must not prepare another experiment')
    monkeypatch.setattr(module,'prepare',unexpected_prepare)
    monkeypatch.setattr(module,'run',lambda path:calls.append(path))
    monkeypatch.setattr(sys,'argv',['backtest','run','--registration',str(registration)])
    module.main()
    assert calls==[registration]


def test_report_rejects_changed_source_without_requiring_snapshot(tmp_path):
    from app.backtesting.experiment_report import report
    registration=tmp_path/'registration.json'
    registration.write_text(json.dumps(dict(kind='new_run', spec={},
        source_sha256={'app/backtesting/experiment_report.py':'wrong'})))
    with pytest.raises(ValueError,match='Code changed after registration'):
        report(registration)
    assert not (tmp_path/'source_snapshot').exists()


def test_invalid_nh_identity_does_not_allocate_work(tmp_path, monkeypatch):
    import json
    from app.backtesting import fusion_experiment as module
    from app.backtesting.experiment_config import load_spec
    spec = load_spec(module.ROOT / 'config/experiments/default.json')
    models = {symbol: {name: {'symbol': symbol, 'dataset_manifest_sha256': 'wrong'}
                       for name in ('alpha', 'value')} for symbol in spec['symbols']}
    spec['strategies'] = [dict(id='nh', mask=0, overrides={},
                              plugin=dict(type='nh_quote_value', version=1, params={'models': models}))]
    monkeypatch.setattr(module, 'load_spec', lambda _: spec)
    (tmp_path / 'manifest.json').write_text('{}')
    monkeypatch.setattr(module, 'validate_inputs', lambda _: (
        {s: {'dataset': str(tmp_path)} for s in spec['symbols']}, {}))
    native = dict(version=spec['engine']['version'], sha256=spec['engine']['native_sha256'],
                  binary=str(tmp_path / 'native'))
    monkeypatch.setattr(module.subprocess, 'check_output', lambda *a, **k: json.dumps(native))
    def no_outputs(*args):
        raise AssertionError('Invalid model allocated a work directory')
    monkeypatch.setattr(module, 'output_directories', no_outputs)
    with pytest.raises(ValueError, match='NH model training input'):
        module.prepare('unused.json', run_id='bad-model')
