"""Market-only strategy contracts, native conversion and training consumers."""
from pathlib import Path
import subprocess
import sys

import hftbacktest as h
import numpy as np
import pytest

from app.market_making.fusion.native_reads import copy_trades, copy_quote_depth
from app.market_making.strategies.nh_quote_value.features import ingest, new_state, window_values


def engine():
    rows = [(h.DEPTH_SNAPSHOT_EVENT | h.BUY_EVENT, 0, 100., 5.),
            (h.DEPTH_SNAPSHOT_EVENT | h.SELL_EVENT, 0, 102., 7.)]
    for second in range(1, 76):
        rows.append((h.TRADE_EVENT | (h.BUY_EVENT if second % 2 else h.SELL_EVENT),
                     second * 10**9, 102. if second % 2 else 100., 1.))
    rows.append((h.DEPTH_EVENT | h.BUY_EVENT, 80*10**9, 100., 3.))
    data = np.zeros(len(rows), dtype=h.event_dtype)
    for i, (kind, ns, price, qty) in enumerate(rows):
        data[i] = (h.EXCH_EVENT | h.LOCAL_EVENT | kind, ns, ns+1, price, qty, 0, 0, 0.)
    asset = (h.BacktestAsset().data(data).linear_asset(1.).tick_size(1.).lot_size(1.)
             .last_trades_capacity(1000).constant_order_latency(1, 1)
             .risk_adverse_queue_model().partial_fill_exchange().trading_value_fee_model(0., .0004))
    return h.HashMapMarketDepthBacktest([asset]), data


def test_detached_inputs_preserve_side_clock_and_depth_after_native_mutation():
    b, tape = engine()
    try:
        b.elapse(2*10**9+2)
        trades = copy_trades(b.last_trades(0))
        depth = copy_quote_depth(b)
        assert trades['side'].tolist() == [1, -1]
        assert trades['local_ns'].tolist() == [10**9+1, 2*10**9+1]
        assert trades['price'].tolist() == [102., 100.]
        b.clear_last_trades(0)
        b.elapse(79*10**9)
        assert len(trades) == 2
        assert depth.bid_qty_at_tick(100) == 5.
        assert depth.ask_qty_at_tick(102) == 7.
        with pytest.raises(ValueError, match='action support'):
            depth.bid_qty_at_tick(93)
        state = new_state()
        ingest(trades, *state)
        assert window_values(2*10**9+2, *state)[0] == -1.
    finally:
        b.close()


def test_nh_native_training_sampling_and_probe_still_execute():
    from app.backtesting.nh_sampling import sample
    from app.backtesting.nh_probe import collect
    depth_times = np.arange(0, 76*10**9, 10**9, dtype=np.int64)
    b, tape = engine()
    try:
        times, x, mids, valid = sample(b, 0, 75*10**9, depth_times)
        assert x.shape[1] == 23 and valid.any()
        assert np.isfinite(x[valid]).all() and (mids[valid] > 0).all()
    finally:
        b.close()
    b, tape = engine()
    try:
        x, orders, filled, edge, labels = collect(
            b, 0, 75*10**9, depth_times, 1., 1., 1., 5., 102., 1, 42)
        assert len(orders) > 0 and x.shape == (len(orders), 28)
        assert np.isfinite(x).all() and np.isfinite(edge).all()
    finally:
        b.close()


def test_strategy_and_nh_feature_imports_do_not_load_hft():
    root = Path(__file__).resolve().parents[4]
    subprocess.run([sys.executable, '-s', '-c', """
import sys
from app.market_making.strategies import nh_strategy, hjb_strategy
from app.market_making.fusion import market_data
from app.market_making.strategies.nh_quote_value import features, quote_policy
assert 'hftbacktest' not in sys.modules
assert 'app.market_making.fusion.native_reads' not in sys.modules
"""], cwd=root, check=True)
