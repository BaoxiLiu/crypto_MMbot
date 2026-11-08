import copy
from decimal import Decimal as D
import json
from pathlib import Path

import pytest

from app.backtesting.experiment_config import ROOT,load_spec,effective_config
from datetime import datetime, timezone
from src.metrics.core import position_holding_periods,aggregate_independent_accounts
from app.backtesting.campaign_standard import validate_standard


def ns(date):
    return int(datetime.fromisoformat(date).replace(tzinfo=timezone.utc).timestamp()) * 10**9


def test_two_fee_scenarios_expand_both_algorithms_without_strategy_fee_override(tmp_path):
    raw=json.loads((ROOT/'config/experiments/default.json').read_text())
    raw['fee_scenarios']=[dict(id='maker0',maker_fee='0'),dict(id='rebate03',maker_fee='-0.00003')]
    path=tmp_path/'spec.json';path.write_text(json.dumps(raw))
    s=load_spec(path)
    assert len(s['strategies'])==4
    fees=[effective_config(s,s['datasets']['DOGEUSDC'],c)['as_config']['maker_fee'] for c in s['strategies']]
    assert list(map(D,fees))==[D(0),D('-0.00003'),D(0),D('-0.00003')]
    c=copy.deepcopy(s['strategies'][0]);c['fee_scenario']['maker_fee']='-.1'
    with pytest.raises(ValueError,match='Unregistered'):effective_config(s,s['datasets']['DOGEUSDC'],c)


def test_explicit_runtime_is_not_silently_overridden_by_machine_defaults(tmp_path):
    import sys
    raw=json.loads((ROOT/'config/experiments/default.json').read_text())
    raw['runtime']=dict(backtest_python=sys.executable,report_python=sys.executable)
    p=tmp_path/'spec.json';p.write_text(json.dumps(raw))
    assert load_spec(p)['runtime']==raw['runtime']


def test_plan_has_exact_requested_matrix_and_causal_context():
    p=json.loads((ROOT/'config/experiments/default.json').read_text())
    assert len(set(p['symbols']))==13 and len(p['fee_scenarios'])==2
    assert ns(p['window']['end_exclusive'])-ns(p['window']['start'])==3*86400*10**9
    assert {D(f['maker_fee']) for f in p['fee_scenarios']}=={D(0),D('-0.00003')}


def test_holding_periods_keep_intrasecond_roundtrips_reversals_and_censoring():
    fills=[dict(ns=0,side=1,qty='2'),dict(ns=200_000_000,side=-1,qty='2'),
           dict(ns=300_000_000,side=1,qty='1'),dict(ns=700_000_000,side=-1,qty='2')]
    m=position_holding_periods(fills,10**9)
    assert m['closed_episodes']==2 and D(m['mean_closed_seconds'])==D('.3')
    assert D(m['open_episode_seconds'])==D('.3') and m['terminal_position']=='-1'
    assert position_holding_periods([],10**9)['mean_closed_seconds'] is None
    with pytest.raises(ValueError):position_holding_periods(fills[::-1],10**9)




def test_standard_campaign_cannot_silently_shrink_to_one_coin_or_one_fee():
    p=json.loads((ROOT/'config/experiments/default.json').read_text())
    spec=dict(stage='quick',symbols=p['symbols'],fee_scenarios=p['fee_scenarios'],window=p['window'],
              account={'max_notional':'1000'},engine={'queue':'cancel_50_50'},execution={'taker_fee':'0.0004'})
    validate_standard(spec)
    short=copy.deepcopy(spec);short['symbols']=['DOGEUSDC']
    with pytest.raises(ValueError,match='13 coins'):validate_standard(short)
    short=copy.deepcopy(spec);short['fee_scenarios']=short['fee_scenarios'][:1]
    with pytest.raises(ValueError,match='maker'):validate_standard(short)




def test_cross_coin_summary_uses_total_cash_and_declares_missing_volume():
    rows=[dict(strategy='test',symbol=s,pnl='1',total_notional='100',initial_cash='600') for s in ('A','B')]
    r=aggregate_independent_accounts(rows,3)[0]
    assert D(r['pnl'])==2 and D(r['net_bp'])==100 and D(r['initial_cash'])==1200
    rows[0]['total_notional']=None
    r=aggregate_independent_accounts(rows,3)[0]
    assert r['total_notional'] is None and r['net_bp'] is None


def test_partial_cross_coin_summary_does_not_extrapolate_daily_turnover():
    rows=[dict(strategy='s',symbol=s,pnl='1',total_notional='100',initial_cash='600') for s in ('BTCUSDC','ETHUSDC')]
    result=aggregate_independent_accounts(rows,3,full_range=False)[0]
    assert result['average_daily_turnover'] is None
    assert D(result['total_notional'])==200 and D(result['net_bp'])==100
