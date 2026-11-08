"""Actual synthetic AS/NH/HJB replays through metrics, plots, HTML publication and cleanup.

This is software integration evidence, never certified historical market research.
"""
import json
from decimal import Decimal as D
from pathlib import Path

from app.artifacts import digest
from app.backtesting.experiment_report import report
from app.backtesting.html_report import read_report, CHARTS
from app.backtesting.result_retention import create_workspace, finalize
from app.backtesting import run_registry
from app.market_making.fusion.tests.refactor_trace import replay_case


def test_native_replays_publish_self_contained_report_and_release_workspace(tmp_path):
    run_id='synthetic-integration'
    work=tmp_path/'.work'/run_id
    final=tmp_path/'reports'/run_id
    create_workspace(work,final,run_id)
    staging=work/'report';staging.mkdir()
    observations=dict(timezone='UTC',markout_horizons_ms=[100,500,1000,5000,10000,30000,60000],
        max_markout_lag_ms=250,markout_anchor='local_fill_receipt',
        markout_reference='first_observed_mid_at_or_after',retain_midpoints=True)
    fees=[dict(id='maker0',maker_fee='0'),dict(id='rebate03',maker_fee='-0.00003')]
    spec=dict(name='合成行情软件验证（非收益研究）',stage='custom',symbols=['BTCUSDC','ETHUSDC'],
        window=dict(start='2026-08-01',end_exclusive='2026-08-02',timezone='UTC'),
        observations=observations,fee_scenarios=fees,costs={'maker_fee':'0'},loss_stops={},
        outputs=dict(format='html',retention='final_only'),datasets={})
    cases=[];sources={};inputs={};actual={}
    for symbol in spec['symbols']:
        for mode in ('as_original','as_improved4','hjb_fixed','hjb_adaptive','hjb_budget50','nh'):
            for fee in fees:
                strategy=mode+'__'+fee['id'];case_id=symbol+'--'+strategy
                root=work/case_id;root.mkdir()
                summary,out,cache=replay_case(root,mode,fee=fee['maker_fee'],symbol=symbol,metrics_spec=observations,partial=True,return_artifacts=True)
                identity=summary['identity']
                spec['execution']=identity['execution_profile']
                spec['engine']=dict(version=identity['hftbacktest'],queue='cancel_50_50',exchange='partial_fill')
                # The bytes of each symbol's synthetic tape are identical across strategy arms.
                spec['datasets'][symbol]={'cache':str(cache)}
                sources.update(identity['source_sha256'])
                inputs[str(cache/'manifest.json')]=digest(cache/'manifest.json')
                cases.append(dict(id=case_id,symbol=symbol,strategy=strategy,mask=0,
                    config=identity['config'],output=str(out),fee_scenario=fee))
                metrics=json.loads((out/'metrics.json').read_text())
                assert D(metrics['volume']['total_notional'])>0
                assert D(summary['fees'])==0 if fee['id']=='maker0' else D(summary['fees'])<0
                actual[case_id]=(summary['pnl'],metrics['volume']['total_notional'])
    registration=work/'registration.json'
    reg=dict(kind='new_run',run_id=run_id,spec=spec,cases=cases,full_range=False,
        run_output=str(work),report_output=str(staging),final_report_output=str(final),
        source_sha256=sources,input_sha256=inputs,
        preparation_policy={'mode':'synthetic_software_test'})
    registration.write_text(json.dumps(reg))
    (staging/'PREREGISTRATION.json').write_text(json.dumps(reg))
    registry=tmp_path/'registry.json'
    run_registry.append(dict(id=run_id,kind='run',status='prepared'),registry)
    report(registration)
    published=finalize(registration,registry)
    payload=read_report(published)
    assert payload['status']=='completed' and payload['full_range'] is False
    assert payload['spec']['symbols']==['BTCUSDC','ETHUSDC']
    assert len(payload['results']['totals'])==24
    assert all(r['average_daily_turnover'] is None for r in payload['supplemental']['cross_coin_metrics.json'])
    assert all(r['average_daily_turnover_multiple'] is None for r in payload['results']['totals'])
    for row in payload['results']['totals']:
        assert (row['pnl'],row['total_notional'])==actual[row['case']]
    expected={fee['id']+'-'+kind+suffix for fee in fees for kind,_ in CHARTS
              for suffix in ('','-coin-0','-coin-1')}
    assert set(payload['chart_sha256'])==expected and len(expected)==66
    assert payload['fee_split_charts'] is True
    assert not work.exists()
    assert [p.name for p in final.iterdir() if not p.name.startswith('._')]==['report.html']
    assert not (final/'source_snapshot').exists()
    assert 'src="data:image/png;base64,' in published.read_text()
