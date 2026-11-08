"""Native replay sampling for NH training; not part of strategy inference."""
import math
import numpy as np
import hftbacktest as h
from numba import njit
from ..market_making.strategies.nh_quote_value.features import S, STEP, new_state, ingest, window_values, feature
from ..market_making.fusion.native_reads import copy_trades

@njit
def sample(hb,start,end,depth_times):
    first=start//STEP*STEP+STEP;n=int((end-first)//STEP)+1
    cap=n//10+2;x=np.zeros((cap,23),np.float32);mids=np.full(cap,np.nan)
    valid=np.zeros(cap,np.bool_);times=np.zeros(cap,np.int64)
    state=new_state();ts,atoms,ptr,tot,prices=state
    di=0;oldbid=0.;oldask=0.;oldbq=0.;oldaq=0.;last=0.
    ofi=0.;ema=0.;buy=0.;sell=0.;decay=math.exp(-.1);ni=0
    hb.elapse(1)
    for i in range(n):
        now=first+i*STEP
        if hb.elapse(now-hb.current_timestamp)!=0:break
        trades=hb.last_trades(0);buy*=decay;sell*=decay
        ingest(copy_trades(trades),ts,atoms,ptr,tot,prices)
        for t in trades:
            if t.ev & h.BUY_EVENT:buy+=t.qty
            else:sell+=t.qty
        hb.clear_last_trades(0)
        win=window_values(now,ts,atoms,ptr,tot,prices)
        while di<len(depth_times) and depth_times[di]<=now:di+=1
        fresh=di>0 and now-depth_times[di-1]<=2*S
        d=hb.depth(0);bid=d.best_bid;ask=d.best_ask
        ready=math.isfinite(bid) and math.isfinite(ask) and bid>0 and ask>bid
        if ready:
            mid=(bid+ask)*.5;bq=d.bid_qty_at_tick(d.best_bid_tick);aq=d.ask_qty_at_tick(d.best_ask_tick)
            if last>0:
                e=(bq if bid>=oldbid else 0.)-(oldbq if bid<=oldbid else 0.)
                e-=(aq if ask<=oldask else 0.)-(oldaq if ask>=oldask else 0.)
                ofi=ofi*decay+e
            ema=.99*ema+.01*(bq+aq)*.5
            last=mid;oldbid=bid;oldask=ask;oldbq=bq;oldaq=aq
        if i%10==0:
            times[ni]=now
            if ready:
                mids[ni]=mid
                x[ni]=feature(.5*(ask-bid)/mid*1e4*(bq-aq)/max(bq+aq,1e-12),
                              ofi/max(ema,1e-12),(buy-sell)/max(buy+sell,1e-12),win)
                valid[ni]=fresh and now>=start+60*S
            ni+=1
    return times[:ni],x[:ni],mids[:ni],valid[:ni]
