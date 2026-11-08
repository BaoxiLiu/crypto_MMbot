"""Native fill test proving 50/50 cancellation advances queue ahead of actual trades."""
import pytest

from app.market_making.fusion.queue_models import configure_queue


def test_unknown_queue_is_not_a_silent_default():
    with pytest.raises(ValueError):configure_queue(None,'random_unknown')


def test_native_equal_cancellation_and_partial_execution():
    h=pytest.importorskip('hftbacktest')
    import numpy as np
    from hftbacktest.order import GTX,LIMIT
    rows=[(1_000_000_000,h.DEPTH_SNAPSHOT_EVENT|h.BUY_EVENT,100.,10.),
          (1_000_000_000,h.DEPTH_SNAPSHOT_EVENT|h.SELL_EVENT,102.,20.),
          (1_200_000_000,h.DEPTH_EVENT|h.BUY_EVENT,100.,30.),
          (1_300_000_000,h.DEPTH_EVENT|h.BUY_EVENT,100.,20.),
          (1_400_000_000,h.TRADE_EVENT|h.SELL_EVENT,100.,6.),
          (1_600_000_000,h.DEPTH_EVENT|h.SELL_EVENT,102.,20.)]
    data=np.zeros(len(rows),dtype=h.event_dtype)
    for i,(t,e,p,q) in enumerate(rows):data[i]=(h.EXCH_EVENT|h.LOCAL_EVENT|e,t,t+5_000_000,p,q,0,0,0.)
    positions={}
    for model in ('risk_adverse','cancel_50_50'):
        asset=(h.BacktestAsset().data(data).linear_asset(1).tick_size(1).lot_size(1)
               .constant_order_latency(1_000_000,1_000_000))
        b=h.HashMapMarketDepthBacktest([configure_queue(asset,model).partial_fill_exchange().trading_value_fee_model(0,.0004)])
        try:
            b.elapse(1);b.elapse(1_100_000_000-int(b.current_timestamp))
            b.submit_buy_order(0,1,100.,2.,GTX,LIMIT,False)
            b.elapse(1_350_000_000-int(b.current_timestamp))
            assert b.position(0)==0  # Cancellation alone is not a fill.
            b.elapse(1_450_000_000-int(b.current_timestamp))
            positions[model]=b.position(0)
        finally:b.close()
    assert positions=={'risk_adverse':0.,'cancel_50_50':1.}  # Partial 1 of 2.
