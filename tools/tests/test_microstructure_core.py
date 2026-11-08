
import numpy as np

from app.factor_library.microstructure_core import FEATURE_NAMES, FeatureState, process_chunk

DT = np.dtype([('ev','u8'),('exch_ts','i8'),('local_ts','i8'),('px','f8'),('qty','f8'),
               ('order_id','u8'),('ival','i8'),('fval','f8')])
NS = 1_000_000_000


def tape(extra):
    rows=[]
    for side, levels in ((1,range(100,90,-1)),(-1,range(101,111))):
        for price in levels:
            rows.append(((1<<30)|(1<<29 if side==1 else 1<<28)|4,NS,NS,price,10,0,0,0))
    for ms,kind,side,price,qty in extra:
        t=NS+ms*1_000_000
        rows.append(((1<<30)|(1<<29 if side==1 else 1<<28)|kind,t,t,price,qty,0,0,0))
    return np.array(rows,dtype=DT)


def query(data, at_ms):
    s=FeatureState(1.,1000,1000)
    return process_chunk(s,data,np.array([NS+at_ms*1_000_000]),np.array([1]),np.array([100.]),np.array([1.]))[0][0]


def test_future_rows_cannot_change_decision_features():
    original=tape([(100,1,1,100,5),(300,1,1,100,9),(500,2,1,101,2)])
    perturbed=tape([(100,1,1,100,5),(300,1,1,100,9),(500,2,-1,101,20000)])
    np.testing.assert_equal(query(original,400),query(perturbed,400))


def test_post_decision_clipped_order_quantity_is_not_a_feature():
    data=tape([(100,1,1,100,5)])
    s1,s2=FeatureState(1.,1000,1000),FeatureState(1.,1000,1000)
    args=(data,np.array([NS+100_000_000]),np.array([1]),np.array([100.]))
    a,_=process_chunk(s1,*args,np.array([1.]))
    b,_=process_chunk(s2,*args,np.array([1000000.]))
    np.testing.assert_equal(a,b)


def test_atomic_snapshot_and_exact_left_open_window():
    data=tape([(100,1,1,100,5),(300,1,1,100,9),(500,2,1,101,2),(1499,1,-1,101,10)])
    before=query(data,1499)
    after=query(data,1500)
    j=FEATURE_NAMES.index('w1000ms_log_trade_count_rate')
    assert before[j]==np.log1p(1.)
    assert after[j]==0 # trade exactly at lower boundary is excluded
    assert np.isfinite(before[:20]).all()
    assert np.isnan(before[FEATURE_NAMES.index('w30000ms_log_trade_count_rate')])


def test_refill_is_first_add_after_net_decrease_at_same_price():
    data=tape([(100,1,1,100,5),(300,1,1,100,9),(350,1,1,100,10)])
    x=query(data,400)
    # 100ms excludes both the first refill at 300ms and the decrease at 100ms;
    # another increase at 350ms must not count as a second refill.
    assert x[FEATURE_NAMES.index('w100ms_log_own_refill_proxy_rate')]==0
    data=tape([(1100,1,1,100,5),(1300,1,1,100,9),(1350,1,1,100,10)])
    x=query(data,1400)
    assert x[FEATURE_NAMES.index('w1000ms_own_refill_proxy_mean_ms')]==200


def test_same_timestamp_batch_never_exposes_crossed_intermediate_book():
    data=tape([(100,1,1,102,3),(100,1,-1,101,0),(100,1,-1,103,5)])
    s=FeatureState(1.,1000,1000)
    x,quotes=process_chunk(s,data,np.array([NS+100_000_000]),np.array([1]),np.array([102.]),np.array([1.]))
    assert quotes[0].tolist()==[102.,103.]
    assert np.isfinite(x[0,:7]).all()
    assert np.isnan(x[0,7:]).all() # fewer than ten surviving ask levels


def test_sparse_far_levels_preserve_depth_and_deletions():
    data=tape([(100,1,-1,200,5),(200,1,-1,110,0)])
    near,sparse=FeatureState(1.,1000,1000),FeatureState(1.,105,1000)
    args=(data,np.array([NS+250_000_000]),np.array([1]),np.array([100.]),np.array([1.]))
    a,qa=process_chunk(near,*args)
    b,qb=process_chunk(sparse,*args)
    np.testing.assert_equal(qa,qb)
    np.testing.assert_allclose(a,b,rtol=0,atol=1e-12,equal_nan=True)
