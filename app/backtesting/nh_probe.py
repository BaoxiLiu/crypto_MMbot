"""Randomized quote probes on the common native execution engine."""
import math

import hftbacktest as h
import numpy as np
from numba import njit
from ..market_making.fusion.native_reads import copy_trades

from ..market_making.strategies.nh_quote_value.features import S, STEP, feature, ingest, new_state, window_values
from ..market_making.strategies.nh_quote_value.value_features import action_features


@njit
def collect(backtest, start, end, depth_times, tick, lot, min_qty, min_notional,
            order_notional, decision_ns, seed):
    np.random.seed(seed)
    steps = int((end - start) // STEP)
    cap = steps // 30 + 8
    features = np.zeros((cap, 28))
    order = np.zeros((cap, 5))  # side, price, quantity, action index, terminal grid
    execs = np.zeros((steps + 2, 2, 2))  # sample, side slot, (row ID + 1, quantity)
    mids = np.full(steps + 2, np.nan)
    active = np.full(2, -1, np.int64)
    sent = np.zeros(2, np.int64)
    filled = np.zeros(cap)
    ts, atoms, ptr, totals, prices = new_state()
    old_bid = old_ask = old_bq = old_aq = 0.
    ema = ofi = buy = sell = variance = last_mid = 0.
    decay = math.exp(-.1)
    var_decay = math.exp(-.1/60)
    offsets = (0, 1, 2, 3, 4, 6)
    row_count = depth_index = 0
    next_probe = start + 60*S
    backtest.elapse(1)
    for i in range(steps):
        now = start + (i+1)*STEP
        if backtest.elapse(now-backtest.current_timestamp) != 0:
            break
        native_orders = backtest.orders(0)
        for j in range(2):
            row = active[j]
            if row < 0:
                continue
            native = native_orders.get(row+1)
            if native is None:
                raise RuntimeError('Probe order disappeared')
            cumulative = max(0., native.qty-native.leaves_qty)
            delta = cumulative-filled[row]
            if delta > lot*1e-8:
                execs[i,j,0] = row+1
                execs[i,j,1] = delta
                filled[row] = cumulative
            if native.status in (h.FILLED,h.CANCELED,h.EXPIRED) and native.req == 0:
                order[row,4] = i+1 if native.status != h.EXPIRED else -1
                active[j] = -1
            elif now-sent[j] >= 5*S and native.cancellable:
                if backtest.cancel(0,row+1,False) != 0:
                    raise RuntimeError('Probe cancel failed')
        backtest.clear_inactive_orders(0)
        trades = backtest.last_trades(0)
        ingest(copy_trades(trades),ts,atoms,ptr,totals,prices)
        buy *= decay
        sell *= decay
        for trade in trades:
            if trade.ev & h.BUY_EVENT:
                buy += trade.qty
            else:
                sell += trade.qty
        backtest.clear_last_trades(0)
        windows = window_values(now,ts,atoms,ptr,totals,prices)
        while depth_index<len(depth_times) and depth_times[depth_index]<=now:
            depth_index += 1
        depth = backtest.depth(0)
        bid, ask = depth.best_bid, depth.best_ask
        if not (math.isfinite(bid) and math.isfinite(ask) and 0<bid<ask):
            continue
        mid = (bid+ask)*.5
        bq = depth.bid_qty_at_tick(depth.best_bid_tick)
        aq = depth.ask_qty_at_tick(depth.best_ask_tick)
        if last_mid>0:
            ret = math.log(mid/last_mid)
            variance = var_decay*variance+(1-var_decay)*ret*ret/.1
            e = (bq if bid>=old_bid else 0.)-(old_bq if bid<=old_bid else 0.)
            e -= (aq if ask<=old_ask else 0.)-(old_aq if ask>=old_ask else 0.)
            ofi = decay*ofi+e
        ema = .99*ema+.01*(bq+aq)*.5
        last_mid,old_bid,old_ask,old_bq,old_aq = mid,bid,ask,bq,aq
        if depth_index==0 or now-depth_times[depth_index-1]>2*S:
            continue
        mids[i] = mid
        if now<next_probe or now+7*S>=end or active[0]>=0 or active[1]>=0:
            continue
        next_probe = now+6*S
        x = feature(.5*(ask-bid)/mid*1e4*(bq-aq)/max(bq+aq,1e-12),
                    ofi/max(ema,1e-12),(buy-sell)/max(buy+sell,1e-12),windows)
        for j in range(2):
            side = 1 if j==0 else -1
            action = np.random.randint(0,6)
            level = (depth.best_bid_tick if j==0 else depth.best_ask_tick)-side*offsets[action]
            price = level*tick
            if price<=0:
                continue
            qty = math.floor(order_notional/price/lot)*lot
            if qty<min_qty or qty*price<min_notional:
                continue
            queue = depth.bid_qty_at_tick(level) if j==0 else depth.ask_qty_at_tick(level)
            features[row_count] = action_features(x,side,mid,bid,ask,tick,price,qty,queue,math.sqrt(variance)*1e4)
            order[row_count,0] = side
            order[row_count,1] = price
            order[row_count,2] = qty
            order[row_count,3] = action
            active[j] = row_count
            row_count += 1
        if decision_ns and backtest.elapse(decision_ns)!=0:
            break
        for j in range(2):
            row = active[j]
            if row<0:
                continue
            sent[j] = now+decision_ns
            price,qty = order[row,1],order[row,2]
            if j==0:
                rc = backtest.submit_buy_order(0,row+1,price,qty,h.GTX,h.LIMIT,False)
            else:
                rc = backtest.submit_sell_order(0,row+1,price,qty,h.GTX,h.LIMIT,False)
            if rc!=0:
                raise RuntimeError('Probe submit failed')
    edge_sum = np.zeros(row_count)
    invalid = np.zeros(row_count,np.bool_)
    label_grid = order[:row_count,4].copy()
    for i in range(steps):
        for j in range(2):
            row = int(execs[i,j,0])-1
            if row<0:
                continue
            target = i+10
            if target>=steps or not math.isfinite(mids[target]):
                target += 1
            if target>=steps or not math.isfinite(mids[target]):
                invalid[row] = True
                continue
            side,price = order[row,0],order[row,1]
            edge_sum[row] += execs[i,j,1]*side*(mids[target]/price-1.)*1e4
            label_grid[row] = max(label_grid[row],target+1)
    valid = (order[:row_count,4]>0) & ~invalid
    return features[:row_count][valid], order[:row_count][valid], filled[:row_count][valid], edge_sum[valid], label_grid[valid]
