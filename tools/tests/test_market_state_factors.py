import numpy as np
from app.factor_library.market_states import (FEATURE_NAMES, NEW_NAMES, MarketStateFactors,
                                            sample_chunk, N_BASE)
from tools.tests.test_microstructure_factors import tape, DT, NS


def ingest(s,a):
    i=0
    while i<len(a):
        j=i+1
        while j<len(a) and a[j]['local_ts']==a[i]['local_ts']:j+=1
        s.ingest(a,i,j);i=j


def test_schema_and_known_spread_depth_changes():
    assert len(FEATURE_NAMES)==len(set(FEATURE_NAMES))
    assert len(NEW_NAMES)==53
    s=MarketStateFactors(1.);a=tape([(100,1,100,0),(200,-1,101,0)])
    ingest(s,a);x=s.query(NS+200_000_000)
    assert x[FEATURE_NAMES.index('w100ms_spread_change_ticks')]==1
    assert x[FEATURE_NAMES.index('w100ms_bbo_change_rate')]==10
    assert np.isnan(x[FEATURE_NAMES.index('w3000ms_spread_change_ticks')])
    assert np.isnan(s.query(NS+2_000_000_000)).all()


def test_future_perturbation_is_causal():
    a=tape([(100,1,100,5),(300,1,100,9),(500,-1,101,1)])
    b=tape([(100,1,100,5),(300,1,100,9),(500,-1,101,99999)])
    x=sample_chunk(MarketStateFactors(1.),a,NS,100_000_000)
    y=sample_chunk(MarketStateFactors(1.),b,NS,100_000_000)
    np.testing.assert_equal(x[2][:5],y[2][:5]);assert np.all(x[3]<=x[1])


def test_synchronized_depletion_is_timestamp_batch_proxy():
    a=tape([(100,1,100,5),(100,-1,101,5),(200,1,100,6)])
    s=MarketStateFactors(1.);ingest(s,a);x=s.query(NS+100_000_000)
    # query may not move backwards, even if the book itself is unchanged.
    assert np.isnan(x[N_BASE:]).all()
    x=s.query(NS+200_000_000)
    assert x[FEATURE_NAMES.index('w500ms_both_decrease_batch_rate')]!=x[FEATURE_NAMES.index('w500ms_both_decrease_batch_rate')]
    assert x[FEATURE_NAMES.index('w100ms_both_decrease_batch_rate')]==0


def test_trade_depletion_uses_opposite_aggressor_and_never_fills():
    rows=list(tape())
    t=NS+1_000_000_000
    rows.append(((1<<30)|(1<<29)|2,t,t,102.,2.,0,0,0.))
    rows.append(((1<<30)|(1<<28)|1,t,t,101.,10.,0,0,0.))
    s=MarketStateFactors(1.);ingest(s,np.array(rows,dtype=DT));x=s.query(t)
    assert x[FEATURE_NAMES.index('w1000ms_buy_multilevel_print_rate')]==1
    assert x[FEATURE_NAMES.index('ask_touch_depletion_seconds')]==1010/204
    assert np.isnan(x[FEATURE_NAMES.index('bid_touch_depletion_seconds')])
    assert not hasattr(s,'position')
