"""GLFT reference coefficients, inventory units and native integration."""
from copy import deepcopy
from decimal import Decimal as D
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from app.market_making.strategies.glft_strategy import coefficients,GLFTModel,validate_settings
from app.market_making.fusion.tests.refactor_trace import ROOT,replay_case


def test_published_hftbacktest_example():
    c1,c2=coefficients(.05,.05,2.986162360812285,.04235741115084049)
    assert c1+10.69*c2/2==pytest.approx(20.47208533844371)
    assert 10.69*c2==pytest.approx(9.76326865029227)


@pytest.mark.parametrize('bad',[0,-1,float('inf'),float('nan')])
def test_invalid_coefficients(bad):
    with pytest.raises(ValueError):coefficients(bad,.05,1,.1)


def test_inventory_direction_symmetry_and_lot_units():
    rules=SimpleNamespace(tick=D('.1'),step=D('.001'))
    cfg=dict(symbol='BTCUSDC',calibration_window_seconds=600,min_calibration_seconds=120,
             calibration_max_age_seconds=90,max_position_qty='.02',as_max_half_spread='.015')
    model=GLFTModel(cfg,rules,'.05','.05','.001')
    model.last_fit_ms=1;model.alpha=D(1);model.kappa=D(1)
    model.vol.samples.extend([100,100.1,99.9,100.2])
    book=SimpleNamespace(mid=D(100),bid=D('99.9'),ask=D('100.1'))
    zero=model.quote(book,D(0),2);long=model.quote(book,D('.001'),2);short=model.quote(book,D('-.001'),2)
    assert sum(long)<sum(zero)<sum(short)
    assert long[0]==200-short[1] and long[1]==200-short[0]
    assert model.last_glft['k_per_tick']==.1
    with pytest.raises(ValueError,match='lot'):GLFTModel(cfg,rules,'.05','.05','.0015')


def plugin(start,end,symbol):
    profile=json.loads((ROOT/'config/strategies/glft_v1.json').read_text())
    profile['params'].update(trade_start_ns=start,trade_end_ns=end,inventory_units={symbol:'.25'})
    return profile


def test_native_glft_fills_and_factory_state_isolation(tmp_path):
    s=replay_case(tmp_path,'glft',plugin_builder=plugin)
    assert s['native_fills']>0 and s['invariant_failures']==0
    from app.market_making.strategies.registry import normalize_case,bind_symbol,factory_for
    from app.market_making.fusion.tests.test_strategy_registry import fixture
    c=bind_symbol(normalize_case(dict(id='glft',plugin=plugin(10,10**12,'BTCUSDC'))),'BTCUSDC')
    original=deepcopy(c);first=factory_for(c)(*fixture());second=factory_for(c)(*fixture())
    first.glft_settings['inventory_units']['BTCUSDC']='99'
    assert second.glft_settings['inventory_units']['BTCUSDC']=='.25' and c==original


def test_native_overlapping_publication_segments_keep_monotonic_time(tmp_path,monkeypatch):
    import app.market_making.fusion.event_backtest as module
    from app.artifacts import digest
    original=module.run_event_arm
    def overlapping(dataset,*args,**kwargs):
        manifest=json.loads((dataset/'manifest.json').read_text())
        with np.load(dataset/'segment.npz') as z:data=z['data']
        cut=int(data['exch_ts'][0])+1000*10**9
        snapshot=data[(data['ev']&255)==4].copy()
        snapshot['exch_ts']=cut-10**9;snapshot['local_ts']=cut-10**9+1_500_000
        arrays=[data[data['exch_ts']<cut],np.concatenate([snapshot,data[data['exch_ts']>=cut]])]
        manifest['segments']=[]
        for i,a in enumerate(arrays):
            path=dataset/f'overlap{i}.npz';np.savez_compressed(path,data=a)
            part=dict(path=path.name,rows=len(a),sha256=digest(path),clock_end_ns=int(a['local_ts'].max()))
            manifest['segments'].append(dict(id=i,start_ns=int(a['exch_ts'].min()),files=[part]))
        (dataset/'manifest.json').write_text(json.dumps(manifest))
        result=original(dataset,*args,**kwargs)
        assert any(s['monotonic_bootstrap_delay_ns']>0 for s in result['segments'])
        assert result['invariant_failures']==0
        return result
    monkeypatch.setattr(module,'run_event_arm',overlapping)
    replay_case(tmp_path,'glft',plugin_builder=plugin)
