import numpy as np
from app.factor_library.microstructure import FEATURE_NAMES, MicrostructureState, shape, sample_chunk, ingest_until

DT=np.dtype([('ev','u8'),('exch_ts','i8'),('local_ts','i8'),('px','f8'),('qty','f8'),('order_id','u8'),('ival','i8'),('fval','f8')])
NS=1_000_000_000


def tape(changes=()):
    rows=[]
    for side,prices in [(1,range(100,90,-1)),(-1,range(101,111))]:
        for p in prices:rows.append(((1<<30)|(1<<29 if side==1 else 1<<28)|4,NS,NS,p,10.,0,0,0.))
    for ms,side,p,q in changes:
        t=NS+ms*1000000;rows.append(((1<<30)|(1<<29 if side==1 else 1<<28)|1,t,t,p,q,0,0,0.))
    return np.array(rows,dtype=DT)


def test_schema_unique_and_balanced_depth_geometry():
    assert len(FEATURE_NAMES)==len(set(FEATURE_NAMES))==202
    s=MicrostructureState(1.);a=tape();ingest_until(s,a,0,len(a));x=shape(s.base)
    assert np.allclose(x[[2,3,10,11,18,19]],1)
    assert np.allclose(x[27:],1)
    assert np.isclose(x[12],.2) and np.isclose(x[14],.6)
    assert np.isnan(s.query(NS)[164:194]).all()


def test_future_perturbation_does_not_change_earlier_samples():
    a=tape([(100,1,100,5),(300,1,100,9),(500,-1,101,1)])
    b=tape([(100,1,100,5),(300,1,100,9),(500,-1,101,100000)])
    x=sample_chunk(MicrostructureState(1.),a,NS,100000000)
    y=sample_chunk(MicrostructureState(1.),b,NS,100000000)
    np.testing.assert_equal(x[2][:5],y[2][:5])
    assert np.all(x[5]<=x[1])


def test_shape_dynamics_use_past_snapshot_and_stale_queries_are_missing():
    s=MicrostructureState(1.);a=tape([(500,1,100,1),(600,-1,101,10)])
    ingest_until(s,a,0,len(a));x=s.query(NS+600000000)
    assert x[164]!=0  # 500ms change of bid uniformity L5.
    assert np.isnan(s.query(NS+2000000000)).all()




