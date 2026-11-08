import numpy as np
import pytest
from tools.experiments.candidate_trade_reach import query


def tape(times, prices, sides, qty=None):
    dtype=[('local_ts','i8'),('ev','u8'),('px','f8'),('qty','f8')]
    data=np.zeros(len(times),dtype=dtype)
    data['local_ts']=times;data['px']=prices;data['qty']=1 if qty is None else qty
    data['ev']=[2 | (1<<29 if side==1 else 1<<28) for side in sides]
    return data


@pytest.mark.parametrize('side',[1,-1])
def test_opposite_direction_price_and_half_open_boundary(side):
    # Left is included, right excluded; same aggressor cannot hit this quote.
    prices=[10,10-side,10-side,10-side,10+side]
    data=tape([99,100,101,102,103],prices,[-side,-side,side,-side,-side])
    result=query(data,100,102,side,100,0.1)
    assert result['public_prints']==2 and result['reaching_prints']==1
    assert result['first_reaching_local_ns']==100 and result['reaching_qty']==1


def test_touch_equality_and_partial_public_volume():
    result=query(tape([100,100,101],[10,10,11],[-1,-1,-1],[0.2,0.3,4]),100,102,1,100,0.1)
    assert result['reaching_prints']==2 and result['reaching_qty']==pytest.approx(0.5)


def test_clock_conversion_is_request_plus_request_and_feed_not_response():
    # Request at 100, exchange entry at 110, cached market receipt at 115.
    data=tape([114,115,120],[10,10,10],[-1,-1,-1])
    assert query(data,100+10+5,120,1,100,0.1)['reaching_prints']==1


def test_empty_trade_window_is_known_no_public_reach():
    assert query(tape([],[],[]),10,11,1,100,0.1)['reaching_prints']==0


@pytest.mark.parametrize('damage',['off_tick','zero_qty','unordered','empty_window','depth','both_sides','no_side','nan'])
def test_bad_inputs_rejected(damage):
    data=tape([100,101],[10,10],[-1,-1])
    if damage=='off_tick':data['px'][0]=10.03
    if damage=='zero_qty':data['qty'][0]=0
    if damage=='unordered':data['local_ts']=[101,100]
    if damage=='depth':data['ev'][0]=1 | (1<<28)
    if damage=='both_sides':data['ev'][0]=2 | (1<<28) | (1<<29)
    if damage=='no_side':data['ev'][0]=2
    if damage=='nan':data['px'][0]=np.nan
    with pytest.raises(ValueError):query(data,102 if damage=='empty_window' else 100,102,1,100,0.1)
