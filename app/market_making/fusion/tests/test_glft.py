"""GLFT reference coefficients, inventory units and native integration."""
from copy import deepcopy
from decimal import Decimal as D
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from app.market_making.strategies.glft_strategy import (
    coefficients, GLFTModel, MarkoutFeedback, spectral_log_values, validate_settings,
)
from app.market_making.fusion.tests.refactor_trace import ROOT,replay_case


def test_published_hftbacktest_example():
    c1,c2=coefficients(.05,.05,2.986162360812285,.04235741115084049)
    assert c1+10.69*c2/2==pytest.approx(20.47208533844371)
    assert 10.69*c2==pytest.approx(9.76326865029227)


@pytest.mark.parametrize('bad',[0,-1,float('inf'),float('nan')])
def test_invalid_coefficients(bad):
    with pytest.raises(ValueError):coefficients(bad,.05,1,.1)


def calibrated_model(**options):
    rules=SimpleNamespace(tick=D('.1'),step=D('.001'))
    cfg=dict(symbol='BTCUSDC',calibration_window_seconds=600,min_calibration_seconds=120,
             calibration_max_age_seconds=90,max_position_qty='.02',as_max_half_spread='.015')
    model=GLFTModel(cfg,rules,'.05','.05','.001',**options)
    model.last_fit_ms=1;model.alpha=D(1);model.kappa=D(1)
    model.vol.samples.extend([100,100.1,99.9,100.2])
    book=SimpleNamespace(mid=D(100),bid=D('99.9'),ask=D('100.1'))
    return model,book


def test_inventory_direction_symmetry_and_lot_units():
    model,book=calibrated_model()
    zero=model.quote(book,D(0),2);long=model.quote(book,D('.001'),2);short=model.quote(book,D('-.001'),2)
    assert sum(long)<sum(zero)<sum(short)
    assert long[0]==200-short[1] and long[1]==200-short[0]
    assert model.last_glft['k_per_tick']==.1
    with pytest.raises(ValueError,match='lot'):GLFTModel(model.config,model.rules,'.05','.05','.0015')


def test_cached_estimates_refresh_and_fit_expires():
    model,book=calibrated_model()
    model.last_second=1;model.last_attempt_ms=1000;model.reference=book.mid
    model.config['calibration_interval_seconds']=60
    initial=model.quote(book,D(0),1001)
    assert model.quote(book,D(0),1999)==initial
    model.sample(D('100.31'),2000)
    assert model.quote(book,D(0),2000)!=initial
    assert not model.ready(90002)


@pytest.mark.parametrize('risk_ratio', [0., .2, 4.])
def test_spectral_solution_matches_positive_ground_state(risk_ratio):
    q = np.arange(-5,6)
    matrix = np.diag(risk_ratio*q**2) - np.eye(11,k=1) - np.eye(11,k=-1)
    _, vectors = np.linalg.eigh(matrix)
    expected = np.log(vectors[5:,0]/vectors[5,0])
    np.testing.assert_allclose(spectral_log_values(5,risk_ratio),expected,atol=1e-11)
    assert np.isfinite(spectral_log_values(42,1e5)).all()


def test_spectral_partial_lot_and_cap_quotes():
    model,book=calibrated_model(inventory_solution='spectral')
    model.config['as_max_half_spread']='.2'
    for qty in (D('.0005'), D('.02')):
        buy,sell=model.quote(book,qty,2)
        opposite_buy,opposite_sell=model.quote(book,-qty,2)
        assert buy < opposite_buy and sell <= opposite_sell
        assert buy+opposite_sell == sell+opposite_buy == 2*book.mid
    with pytest.raises(ValueError,match='outside cap'):
        model.quote(book,D('.021'),2)
    with pytest.raises(ValueError,match='inventory risk'):
        spectral_log_values(42,float('inf'))


FEEDBACK = dict(orders=5,minimum_orders=3,half_life_ms=20000,max_age_ms=90000,
                deadband_bp=.25,cap_bp=1.)


def feedback_event(cid, side, observed_ms, bp, *, lag_ms=0, horizon_ms=1000):
    ns = observed_ms*1_000_000
    return dict(fill=dict(cid=cid,side=side,price=D(100)),mid=D(100)+D(side)*D(str(bp))/100,
                observed_ns=ns,target_ns=ns-lag_ms*1_000_000,horizon_ms=horizon_ms)


def test_markout_causality_distinct_orders_decay_and_reducing_side():
    model,book=calibrated_model(feedback=FEEDBACK)
    feedback=model.feedback
    assert isinstance(feedback,MarkoutFeedback)
    now=21000
    initial=model.quote(book,D(0),now)
    for cid,stamp,bp in [('a',1000,0.),('a',2000,0.),('b',11000,-1.)]:
        feedback.observe(feedback_event(cid,1,stamp,bp))
    feedback.observe(feedback_event('late',1,now,-10.,lag_ms=251))
    feedback.observe(feedback_event('wrong-horizon',1,now,-10.,horizon_ms=5000))
    assert feedback.widen_bp(1,now)==0
    feedback.observe(feedback_event('c',1,now,-3.))
    assert feedback.widen_bp(1,now-1)==0
    assert feedback.widen_bp(1,now)==1
    assert feedback.widen_bp(1,now+20000)==D('.5')
    assert feedback.widen_bp(1,now+90000)==0
    assert model.quote(book,D(0),now)==(initial[0]-D('.01'),initial[1])
    short=model.quote(book,D('-.001'),now)
    model.feedback=None
    assert model.quote(book,D('-.001'),now)==short


def test_weighted_mid_moves_both_quotes_and_invalid_reference_falls_back():
    model,book=calibrated_model()
    initial=model.quote(book,D(0),2)
    model.fair_value=D('100.02')
    assert model.quote(book,D(0),2)==tuple(p+D('.02') for p in initial)
    model.fair_value=D('100.11')
    assert model.quote(book,D(0),2)==initial


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


@pytest.mark.parametrize('overrides',[
    {'inventory_solution':'spectral'},
    {'reference_price':'weighted_mid'},
    {'markout_feedback':FEEDBACK},
])
def test_variants_complete_native_strategy_lifecycle(tmp_path,overrides):
    def build(start,end,symbol):
        config=plugin(start,end,symbol)
        config['params'].update(overrides)
        return config
    result=replay_case(tmp_path,'glft',plugin_builder=build)
    assert result['native_fills']>0 and result['invariant_failures']==0


@pytest.mark.parametrize('overrides',[
    {'inventory_solution':'spectral','xi':'.02'},
    {'reference_price':'future_mid'},
    {'markout_feedback':dict(FEEDBACK,minimum_orders=6)},
])
def test_rejects_invalid_variant_settings(overrides):
    config=plugin(1,2,'BTCUSDC')['params']
    config.update(overrides)
    with pytest.raises(ValueError):
        validate_settings(config)


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
