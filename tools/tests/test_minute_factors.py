import numpy as np
import pytest
from app.factor_library.minute import MINUTE as M, bars, rsi, indicators, align, local_trades


def test_bar_boundary_partial_and_zero_trade():
    b = bars(np.array([M//2, M+1, 2*M, 2*M+1]),
             [90., 100., 105., 95.], [1., 2., 3., 4.],
             [True, True, False, True], M//3, 4*M+1)
    assert b['open_ns'].tolist() == [M, 2*M, 3*M]
    assert b['close'].tolist() == [100., 95., 95.]
    assert b['volume'].tolist() == [2., 7., 0.]
    assert b['trade_count'].tolist() == [1., 2., 0.]
    assert b['high'][1] == 105. and b['low'][1] == 95.
    assert b['buy_volume'][1] == 4.


def test_no_future_or_gap_carry():
    b = bars(np.array([2*M+1]), [99.], [1.], [True], 0, 4*M)
    assert np.isnan(b['close'][:2]).all()
    b2 = bars(np.array([], dtype=np.int64), [], [], [], 5*M, 8*M)
    assert np.isnan(b2['close']).all()
    X = indicators(b)
    out, used, raw = align(b, X, np.array([3*M-1, 3*M]), np.array([1, 1]))
    assert np.isnan(out[0]).all()
    assert used.tolist() == [2*M, 3*M]
    assert out[1, 14] == np.log(2.)


def test_rsi_wilder_seed_recursion_and_missing_reset():
    c = np.array([10., 11., 10., 12., 11., 12., np.nan, 3., 4., 5., 6.])
    v = rsi(c, 3)
    assert np.isnan(v[:3]).all()
    assert v[3] == pytest.approx(75.)
    # Next smoothed gain=2/3, loss=5/9 -> RSI=600/11? Correct direct ratio.
    assert v[4] == pytest.approx(100.-100./(1.+(2./3.)/(5./9.)))
    assert np.isnan(v[6:10]).all() and v[10] == 100.
    assert rsi(np.ones(20), 14)[-1] == 50.
    assert rsi(np.arange(20., 0., -1), 14)[-1] == 0.


def test_sell_orientation_and_closed_cutoff():
    t = np.arange(25)*M+1
    b = bars(t, np.arange(100., 125.), np.ones(25), np.ones(25, bool), 0, 25*M)
    X = indicators(b)
    out, used, raw = align(b, X, [24*M, 24*M, 24*M-1], [1, -1, 1])
    assert used.tolist() == [24*M, 24*M, 23*M]
    assert raw.tolist() == [100., 100., 100.]
    assert out[0, 0] == 1. and out[1, 0] == -1.
    assert out[0, 3] == 0. and out[1, 3] == 1.
    for col in [4, 5, 6, 9, 10, 11, 17]:
        assert out[0, col] == pytest.approx(-out[1, col])
    for col in [7, 8, 12, 13, 14, 15, 16]:
        assert out[0, col] == out[1, col]


def test_future_trade_perturbation_cannot_change_prior_features():
    t = np.arange(30)*M+1
    p = np.linspace(100., 120., 30)
    b = bars(t, p, np.ones(30), np.ones(30, bool), 0, 30*M)
    p[25:] *= 10.
    changed = bars(t, p, np.ones(30), np.ones(30, bool), 0, 30*M)
    np.testing.assert_allclose(indicators(b)[:25], indicators(changed)[:25], equal_nan=True)


def test_malformed_clock_rejected():
    with pytest.raises(ValueError):
        bars([M, 0], [1., 2.], [1., 1.], [True, False], 0, 2*M)
    with pytest.raises(ValueError):
        bars([M], [-1.], [1.], [True], 0, 2*M)


def test_local_trade_filter_avoids_exchange_copy_and_future_receipt():
    d = np.zeros(5, dtype=[('ev', 'u8'), ('local_ts', 'i8'), ('px', 'f8'), ('qty', 'f8')])
    d['ev'] = [(1<<31)|(1<<29)|2, (1<<30)|(1<<29)|2,
               (1<<30)|(1<<28)|2, (1<<30)|1, (1<<30)|2]
    d['local_ts'] = [3, 3, 4, 5, 11]; d['px'] = 100.; d['qty'] = 2.
    t, p, q, buy = local_trades(d, 0, 10)
    assert t.tolist() == [3, 4] and buy.tolist() == [True, False]
    assert p.tolist() == [100., 100.] and q.tolist() == [2., 2.]
