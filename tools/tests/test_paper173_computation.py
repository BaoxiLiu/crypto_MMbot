import numpy as np
import pytest

from app.factor_library.paper173_core import FEATURE_NAMES, Observations, compute, impact_bp

NS = 1_000_000_000


def obs(times, prices, qty=None, buys=None, epoch=0, end=10*NS):
    n = len(times)
    return Observations(np.array(times, dtype=np.int64), np.array(prices, float),
                        np.ones(n) if qty is None else np.array(qty, float),
                        np.ones(n, bool) if buys is None else np.array(buys, bool),
                        np.array([epoch, end], dtype=np.int64),
                        np.array([100.,100.]), np.array([101.,101.]), epoch)


def query(o, now=10*NS, side=1):
    x = compute(o, now, side, [100.,99.], [10000.,10000.], [101.,102.], [10000.,10000.])
    assert x.shape == (173,)
    return dict(zip(FEATURE_NAMES, x))


def test_lagged_windows_are_disjoint_and_exact_right_closed():
    # Boundary at 9s belongs to w1, not w0. 8s belongs to w2.
    o = obs([7*NS,8*NS,9*NS,10*NS],[100,100,100,100],[1,2,4,8])
    x=query(o)
    assert [x[f'total_buy_1s_w{k}'] for k in range(3)] == [8,4,2]


def test_append_future_trade_and_book_does_not_change_current_features():
    a = obs([8*NS,9*NS,10*NS],[100,99,100])
    b = obs([8*NS,9*NS,10*NS,10*NS+1],[100,99,100,10000])
    b.book_ns=np.r_[b.book_ns,10*NS+1];b.book_bid=np.r_[b.book_bid,200.];b.book_ask=np.r_[b.book_ask,201.]
    np.testing.assert_allclose(list(query(a).values()),list(query(b).values()),equal_nan=True)


def test_direction_swap_preserves_sizes_and_flips_return():
    o=obs([9*NS+1,9*NS+2,10*NS],[99,100,101],[2,3,4],[True,False,True])
    buy,sell=query(o),query(o,side=-1)
    assert buy['total_buy_1s_w0']==sell['total_sell_1s_w0']==6
    assert buy['buy_count_1s_w0']==sell['sell_count_1s_w0']==2
    assert buy['ret_sum_1s_w0']==-sell['ret_sum_1s_w0']
    assert buy['amplitude_1s_w0']==sell['amplitude_1s_w0']==2


def test_empty_vs_uncovered_are_distinct():
    o=obs([],[])
    x=query(o)
    assert x['buy_count_1s_w0']==0
    assert np.isnan(x['avg_size_1s_w0'])
    assert np.isnan(x['buy_count_300s_w0'])


def test_quantity_weighted_vwap_and_max_size():
    o=obs([9*NS+1,10*NS],[99,103],[3,1])
    x=query(o)
    assert x['ret_vwap_1s_w0']==pytest.approx(0)
    assert x['max_size_1s_w0']==3
    assert x['avg_size_1s_w0']==2


def test_insufficient_depth_is_missing_and_fractional_last_level_used():
    assert np.isnan(impact_bp([100,99],[1,1],500000,1))
    # Spend 100 at 100, then 99 at 99 => 2 units; VWAP 99.5.
    assert impact_bp([100,99],[1,10],199,1)==pytest.approx(50)
    assert impact_bp([100,101],[1,10],201,-1)==pytest.approx(50)


def test_bbo_quantity_updates_do_not_reset_price_age():
    o=obs([],[],end=1000*NS)
    o.book_ns=np.array([0,100*NS,400*NS,1000*NS],dtype=np.int64)
    o.book_bid=np.array([99,100,99,100],float);o.book_ask=o.book_bid+1
    x=query(o,now=1000*NS)
    assert x['age']==0
    # Complete spell ending at t=1000 began at t=400, equals 600s.
    assert x['totb_mean']==600


def test_long_std_requires_full_history_and_uses_only_past_trade():
    t=np.arange(0,6001,dtype=np.int64)*NS
    p=np.exp(np.arange(len(t))*1e-6)*100
    o=obs(t,p,end=6000*NS)
    x=query(o,now=6000*NS)
    assert x['stdev_500']<1e-14
    assert np.isnan(query(obs([],[]))['stdev_100'])


def test_reject_clock_loss_or_unsorted_inputs():
    with pytest.raises(ValueError):
        obs([2,1],[100,100])
    with pytest.raises(ValueError):
        query(obs([],[]),now=10.*NS)


def test_unknown_initial_price_age_is_not_fabricated():
    x=query(obs([],[]))
    assert np.isnan(x['age'])
    assert np.isnan(x['totb_mean'])


def test_stale_book_does_not_yield_valid_features():
    assert np.isnan(list(query(obs([],[]),now=12*NS).values())).all()


def test_new_epoch_does_not_reuse_pre_gap_windows():
    o=obs([9*NS],[100],epoch=9*NS,end=10*NS)
    x=query(o)
    assert x['total_buy_1s_w0']==0  # trade exactly at excluded left endpoint
    assert np.isnan(x['total_buy_1s_w1'])
