"""Comparison reads standard results and fails closed on unmatched conditions."""
from copy import deepcopy
import pytest
from app.backtesting import comparison


def payload():
    return dict(status='completed',run_id='a',full_range=True,
        spec=dict(window={'start':'2026-08-01'},symbols=['BTCUSDC','ETHUSDC'],
                  datasets={'BTCUSDC':{'cache':'frozen'}},account={'max_notional':'1000'}),
        input_sha256={'/raw/manifest.json':'input','/raw/funding/page.json':'funding'},
        source_sha256={'src/metrics/core.py':'metrics','app/market_making/fusion/kernel.py':'kernel'},
        results={'totals':[{'pnl':'1.20','total_notional':'100','strategy':'example'}]})


def test_comparison_preserves_recorded_metrics_and_accepts_different_strategy_parameters(monkeypatch):
    a=payload();b=deepcopy(a);b['run_id']='b'
    b['results']['totals'][0]['pnl']='2.34'
    b['spec']['_profile_inputs']={'/profile/strategy.json':'different'}
    b['input_sha256']['/profile/strategy.json']='different'
    monkeypatch.setattr(comparison,'read_report',lambda p:deepcopy({'a':a,'b':b}[p.name]))
    result=comparison.compare_reports(['a','b'])
    assert result['matched'] and result['runs'][1]['totals'][0]['pnl']=='2.34'
    assert a['results']['totals'][0]['pnl']=='1.20'


@pytest.mark.parametrize('change',['funding','metrics','account','dataset','failed','partial'])
def test_unmatched_or_incomplete_reports_cannot_be_called_matched(monkeypatch,change):
    a=payload();b=deepcopy(a)
    if change=='funding':b['input_sha256']['/raw/funding/page.json']='changed'
    elif change=='metrics':b['source_sha256']['src/metrics/core.py']='changed'
    elif change=='account':b['spec']['account']['max_notional']='2000'
    elif change=='dataset':b['spec']['datasets']['BTCUSDC']['cache']='other'
    elif change=='failed':b['status']='failed'
    else:a['full_range']=b['full_range']=False
    monkeypatch.setattr(comparison,'read_report',lambda p:{'a':a,'b':b}[p.name])
    with pytest.raises(ValueError):comparison.compare_reports(['a','b'])
