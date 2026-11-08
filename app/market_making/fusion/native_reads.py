"""Read native fields together; never advance time or mutate exchange state."""
import math

from numba import njit


# Native ctypes pointers are process-specific: compile once per worker and do
# not persist them to a Numba disk cache.
@njit
def best_quotes(hbt):
    depth = hbt.depth(0)
    bid, ask = depth.best_bid, depth.best_ask
    if not math.isfinite(bid) or not math.isfinite(ask) or bid <= 0 or ask <= bid:
        return bid, ask, 0, 0, 0., 0.
    return bid, ask, depth.best_bid_tick, depth.best_ask_tick, depth.best_bid_qty, depth.best_ask_qty


@njit
def order_state(hbt, native_id):
    order = hbt.orders(0).get(native_id)
    if order is None:
        return False, 0., 0, 0
    return True, order.leaves_qty, order.status, order.exec_price_tick


@njit
def order_clocks(hbt, native_id):
    """Native simulation timestamps; observations only, never a strategy input."""
    order = hbt.orders(0).get(native_id)
    if order is None:
        return 0, 0, 0
    return order.exch_timestamp, order.local_timestamp, order.req


# Conversion belongs to execution: strategies never own a native backtest handle.
import numpy as np
from .market_data import TRADE_DTYPE, QuoteDepth


@njit
def copy_trades(trades):
    result = np.empty(len(trades), dtype=TRADE_DTYPE)
    for i in range(len(trades)):
        trade = trades[i]
        result[i]['local_ns'] = trade['local_ts']
        result[i]['side'] = 1 if trade['ev'] & (1 << 29) else -1
        result[i]['price'] = trade['px']
        result[i]['qty'] = trade['qty']
    return result


@njit
def copy_quote_depth(backtest):
    depth = backtest.depth(0)
    bids, asks = np.zeros(7), np.zeros(7)
    bid_tick, ask_tick = depth.best_bid_tick, depth.best_ask_tick
    valid = math.isfinite(depth.best_bid) and math.isfinite(depth.best_ask) and 0 < depth.best_bid < depth.best_ask
    for offset in range(7) if valid else range(0):
        bids[offset] = depth.bid_qty_at_tick(bid_tick - offset)
        asks[offset] = depth.ask_qty_at_tick(ask_tick + offset)
    return QuoteDepth(depth.best_bid, depth.best_ask, bid_tick, ask_tick, bids, asks)


def observe_strategy(backtest, strategy, ns):
    if not strategy.observation_due(ns):
        return
    trades = copy_trades(backtest.last_trades(0))
    depth = copy_quote_depth(backtest) if strategy.requires_quote_depth else None
    strategy.observe_market(depth, trades, ns)
    backtest.clear_last_trades(0)
