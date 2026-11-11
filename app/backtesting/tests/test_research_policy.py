"""New-run policy must fail before dataset/native preparation has side effects."""
from copy import deepcopy
import pytest
from app.backtesting.campaign_standard import validate_preparation


def spec():
    return dict(stage='custom', symbols=['BTCUSDC','ETHUSDC','DOGEUSDC'],
                fee_scenarios=[{'maker_fee':'0'},{'maker_fee':'-0.00003'}],
                engine={'queue':'cancel_50_50'}, execution={'taker_fee':'0.0004'},
                account={'max_notional':'1000'}, outputs={'format':'html','retention':'final_only'})


def test_new_custom_contract_is_pure():
    s=spec(); before=deepcopy(s)
    assert validate_preparation(s)['mode']=='new_research'
    assert s==before


@pytest.mark.parametrize('section,key,value',[
    (None,'symbols',['DOGEUSDC']), (None,'symbols',['BTCUSDC']),
    ('engine','queue','risk_adverse'), ('execution','taker_fee','0'),
    ('account','max_notional','9000'), ('outputs','retention','all'),
    ('outputs','format','json'), (None,'fee_scenarios',[{'maker_fee':'-0.00003'}]),
    (None,'fee_scenarios',[]), (None,'fee_scenarios',[{'maker_fee':'0.0001'}]),
    (None,'fee_scenarios',[{'maker_fee':'0'},{'maker_fee':'0'}])])
def test_custom_cannot_skip_shared_contract(section,key,value):
    s=spec();(s if section is None else s[section])[key]=value
    with pytest.raises(ValueError):validate_preparation(s)




def test_prepare_checks_policy_before_inputs_or_outputs(monkeypatch):
    from app.backtesting import fusion_experiment as module
    s=spec();s['symbols']=['DOGEUSDC']
    monkeypatch.setattr(module,'load_spec',lambda path:s)
    def must_not_run(*args):
        pytest.fail('Data/native/output preparation occurred before policy check')
    monkeypatch.setattr(module,'validate_inputs',must_not_run)
    with pytest.raises(ValueError,match='BTCUSDC'):
        module.prepare('nonexistent-config.json')


def test_policy_identity_survives_final_failure_html(tmp_path):
    import json
    from app.backtesting.html_report import build_failure_report,read_report
    policy=validate_preparation(spec())
    registration=tmp_path/'registration.json'
    registration.write_text(json.dumps(dict(run_id='history',spec=spec(),full_range=True,
                                            preparation_policy=policy)))
    report=build_failure_report(registration,tmp_path,'native unavailable')
    assert read_report(report)['preparation_policy']==policy


def test_repository_default_obeys_new_research_contract():
    import json
    from pathlib import Path
    root=Path(__file__).resolve().parents[3]
    s=json.loads((root/'config/experiments/default.json').read_text())
    assert validate_preparation(s)['mode']=='new_research'
    assert s['stage']=='quick' and len(s['symbols'])==13
    assert s['window']=={'start':'2026-08-01','end_exclusive':'2026-08-04','timezone':'UTC'}
    assert {c['parameter_overrides']['mode'] for c in s['strategies']}=={'as_base','paper_s1_sc0'}
    assert 'runtime' not in s  # Machine executables belong in paths.local.json.
    for coin in s['symbols']:
        assert (root/s['datasets'][coin]['input_spec']).is_file()
        assert (root/s['strategy_parameters'][coin]).is_file()


@pytest.mark.parametrize('entry', ['run', 'worker'])
def test_registered_execution_cannot_bypass_policy(tmp_path, entry):
    import json
    from app.backtesting import fusion_experiment
    invalid=spec();invalid['symbols']=['DOGEUSDC']
    path=tmp_path/'registration.json'
    path.write_text(json.dumps({'spec':invalid}))
    with pytest.raises(ValueError,match='BTCUSDC'):
        if entry=='worker':fusion_experiment.worker(path,'case')
        else:fusion_experiment.run(path)
    assert list(tmp_path.iterdir())==[path]


def test_maker_zero_only_does_not_inject_rebate_or_mutate_configuration():
    s=spec();s['fee_scenarios']=[{'maker_fee':'0'}];before=deepcopy(s)
    assert validate_preparation(s)['policy_version']==2
    assert s==before
