"""A test-only fourth policy proves extension without changing execution code.

The production registry remains AS/NH/HJB. This fixed-spread probe is software
integration evidence and is never a public trading strategy or return claim.
"""
import json
from decimal import Decimal as D
from types import ModuleType

from app.artifacts import digest
from app.backtesting import run_registry
from app.backtesting.experiment_report import report
from app.backtesting.html_report import read_report
from app.backtesting.result_retention import create_workspace, finalize
from app.market_making.contracts import Desired
from app.market_making.strategies.base import QuoteStrategy
from app.market_making.strategies.registry import DEFINITIONS, StrategyDefinition
from app.market_making.fusion.tests.refactor_trace import replay_case


class FixedSpreadProbe(QuoteStrategy):
    def __init__(self, rules, config, factors, params, symbol):
        super().__init__(rules, config, factors)
        self.params=params
        self.trading_window_ns=params['trade_start_ns'],params['trade_end_ns']

    @property
    def identity(self):
        return dict(type='interface_probe',version=1,params=self.params,
                    scope='synthetic software extension test only')

    def desired(self, position, ns):
        if not self.trading_window_ns[0]<=ns<self.trading_window_ns[1]:return {}
        return {side:Desired(side,self.rules.price(
            self.obs.mid-side*self.params['half_ticks']*self.rules.tick,side),D('.25'),reason='interface_probe')
            for side in (1,-1)}


def validate_settings(params):
    assert set(params)=={'trade_start_ns','trade_end_ns','half_ticks'}
    assert 0<params['trade_start_ns']<params['trade_end_ns'] and params['half_ticks']>0


def test_new_policy_uses_registry_native_engine_metrics_html_and_cleanup(tmp_path,monkeypatch):
    module=ModuleType('app.market_making.strategies.interface_probe')
    module.FixedSpreadProbe=FixedSpreadProbe
    module.validate_settings=validate_settings
    module.SOURCES=()
    monkeypatch.setitem(__import__('sys').modules,module.__name__,module)
    monkeypatch.setitem(DEFINITIONS,'interface_probe',StrategyDefinition(
        'interface_probe','FixedSpreadProbe',(1,),'Test-only fixed-spread policy'))
    def plugin(start,end,symbol):
        return dict(type='interface_probe',version=1,params=dict(
            trade_start_ns=start,trade_end_ns=end,half_ticks=2))
    run_id='extension-contract';work=tmp_path/'.work'/run_id;final=tmp_path/'reports'/run_id
    create_workspace(work,final,run_id)
    staging=work/'report';staging.mkdir()
    observations=dict(timezone='UTC',markout_horizons_ms=[100,500,1000,5000,10000,30000,60000],
        max_markout_lag_ms=250,markout_anchor='local_fill_receipt',
        markout_reference='first_observed_mid_at_or_after',retain_midpoints=False)
    fees=[dict(id='maker0',maker_fee='0'),dict(id='rebate03',maker_fee='-0.00003')]
    spec=dict(name='策略接口扩展验证（合成行情）',stage='custom',symbols=['BTCUSDC','ETHUSDC'],
        window=dict(start='2026-08-01',end_exclusive='2026-08-02',timezone='UTC'),
        observations=observations,fee_scenarios=fees,costs={'maker_fee':'0'},loss_stops={},
        outputs=dict(format='html',retention='final_only'),datasets={})
    cases=[];inputs={};source_hashes={};expected={}
    for symbol in spec['symbols']:
        for fee in fees:
            strategy='interface_probe__'+fee['id'];case_id=symbol+'--'+strategy
            folder=work/case_id;folder.mkdir()
            summary,output,cache=replay_case(folder,'interface_probe',symbol=symbol,fee=fee['maker_fee'],
                metrics_spec=observations,partial=True,return_artifacts=True,plugin_builder=plugin)
            identity=summary['identity'];source_hashes.update(identity['source_sha256'])
            spec['execution']=identity['execution_profile']
            spec['engine']=dict(version=identity['hftbacktest'],queue='cancel_50_50',exchange='partial_fill')
            spec['datasets'][symbol]={'cache':str(cache)}
            inputs[str(cache/'manifest.json')]=digest(cache/'manifest.json')
            cases.append(dict(id=case_id,symbol=symbol,strategy=strategy,mask=0,
                config=identity['config'],output=str(output),fee_scenario=fee))
            expected[case_id]=summary['pnl']
    reg=dict(kind='new_run',run_id=run_id,spec=spec,cases=cases,full_range=False,
        run_output=str(work),report_output=str(staging),final_report_output=str(final),
        source_sha256=source_hashes,input_sha256=inputs)
    registration=work/'registration.json';registration.write_text(json.dumps(reg))
    (staging/'PREREGISTRATION.json').write_text(json.dumps(reg))
    registry=tmp_path/'registry.json';run_registry.append(dict(id=run_id,kind='run',status='prepared'),registry)
    report(registration)
    result=read_report(finalize(registration,registry))
    assert result['status']=='completed' and not work.exists()
    assert len(result['results']['totals'])==4
    assert all(row['pnl']==expected[row['case']] and row['fill_receipts']>0 for row in result['results']['totals'])
    assert result['fee_split_charts'] is True
