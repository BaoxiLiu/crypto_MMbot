"""Past-only diagnostics for the photographed market-state taxonomy.

Independent extension: frozen 50/S01-S27/202-feature definitions are unchanged.
Aggregated L2 identifies net changes, never individual cancellations or intent.
Sweep observations count prints crossing >=2 previously visible price levels;
they are not identified parent market orders. No private/exchange queue is read.
"""
import math
import numpy as np
from numba import njit, int64, float64
from numba.experimental import jitclass
from .microstructure import MicrostructureState, FEATURE_NAMES as MICRO_NAMES

# Reuse actual names and formulas, rather than duplicate frozen factors.
SELECT = np.array(list(range(17)) +
                  [i for i,n in enumerate(MICRO_NAMES) if
                   (n.startswith(('w100ms_','w1000ms_')) and any(k in n for k in (
                       'trade_count_rate','trade_notional_rate','trade_notional_imbalance',
                       'aggressor_rate','ofi_scaled','realized_vol_bp','return_bp',
                       'ticks0_1_rate','refill_proxy_rate','refill_proxy_mean_ms')))] +
                  list(range(133,164)), dtype=np.int64)
BASE_NAMES = [MICRO_NAMES[i] for i in SELECT]
WINDOWS_MS = (100, 500, 1000, 2000, 3000)
PER_WINDOW = ('spread_change_ticks','bid_l5_change_fraction','ask_l5_change_fraction',
              'bbo_change_rate','both_decrease_batch_rate','both_add_batch_rate',
              'buy_multilevel_print_rate','sell_multilevel_print_rate')
NEW_NAMES = [f'w{w}ms_{n}' for w in WINDOWS_MS for n in PER_WINDOW]
NEW_NAMES += ['last_trade_age_ms','bid_touch_depletion_seconds','ask_touch_depletion_seconds',
              'bid_trade_adjusted_net_decrease_rate','ask_trade_adjusted_net_decrease_rate']
for w in (2000,3000):
    NEW_NAMES += [f'w{w}ms_{n}' for n in ('return_bp','realized_vol_bp','trade_imbalance','trade_notional_rate')]
FEATURE_NAMES = BASE_NAMES + NEW_NAMES
N_BASE, N_FEATURES = len(BASE_NAMES), len(FEATURE_NAMES)


@njit
def l5_notional(base):
    bid=ask=0.
    b,a=base.best_bid,base.best_ask
    for i in range(5):
        if b<=0 or a>=2**62:return np.nan,np.nan
        bid+=b*base.tick*base.bid_qty(b)
        ask+=a*base.tick*base.ask_qty(a)
        if i<4:b=base.below(b);a=base.above(a)
    return bid,ask


@jitclass([('micro',MicrostructureState.class_type.instance_type),('epoch',int64),
           ('written',int64),('last_trade_ns',int64),('cum',float64[:]),
           ('times',int64[:]),('history',float64[:,:])])
class MarketStateFactors:
    def __init__(self,tick):
        self.micro=MicrostructureState(tick)
        self.epoch=-1;self.written=0;self.last_trade_ns=-1
        self.cum=np.zeros(5)
        self.times=np.zeros(65536,dtype=np.int64)
        # cumulative counters, spread, bid/ask L5 notional.
        self.history=np.zeros((65536,8))

    def ingest(self,data,begin,end):
        base=self.micro.base
        t=int(data[begin]['local_ts'])
        valid=base.valid();pb,pa=base.best_bid,base.best_ask
        bd=ad=ba=aa=0.;buy_sweep=sell_sweep=0.
        for i in range(begin,end):
            row=data[i];ev=int(row['ev']);kind=ev&255
            side=1 if ev&(1<<29) else -1
            px,qty=row['px'],row['qty']
            if kind==2:
                self.last_trade_ns=t
                if valid:
                    tick=int(round(px/base.tick))
                    touch=pa if side==1 else pb
                    second=base.above(touch) if side==1 else base.below(touch)
                    crossed=(second<2**62 and tick>=second) if side==1 else (second>0 and tick<=second)
                    if crossed:
                        if side==1:buy_sweep+=1.
                        else:sell_sweep+=1.
            elif kind==1 and valid:
                tick=int(round(px/base.tick));touch=pb if side==1 else pa
                if abs(tick-touch)<=2:
                    old=base.bid_qty(tick) if side==1 else base.ask_qty(tick)
                    change=(qty-old)*px
                    if side==1:
                        if change>0:ba+=change
                        else:bd-=change
                    else:
                        if change>0:aa+=change
                        else:ad-=change
        self.micro.ingest(data,begin,end)
        if self.epoch!=base.epoch_ns:
            self.epoch=base.epoch_ns;self.written=0;self.cum[:]=0
            # The initial snapshot supplies no historical trade coverage.
            self.last_trade_ns=-1
        if not base.valid():return
        if valid:
            if pb!=base.best_bid or pa!=base.best_ask:self.cum[0]+=1.
            if bd>0 and ad>0:self.cum[1]+=1.
            if ba>0 and aa>0:self.cum[2]+=1.
            self.cum[3]+=buy_sweep;self.cum[4]+=sell_sweep
        slot=self.written%65536
        if self.written>=65536 and self.times[slot]>=t-3_300_000_000:
            raise ValueError('Supplemental history capacity exceeded')
        self.times[slot]=t;self.history[slot,:5]=self.cum
        self.history[slot,5]=base.best_ask-base.best_bid
        self.history[slot,6],self.history[slot,7]=l5_notional(base)
        self.written+=1

    def past(self,target):
        lo=max(0,self.written-65536);hi=self.written
        while lo<hi:
            middle=(lo+hi)//2
            if self.times[middle%65536]<=target:lo=middle+1
            else:hi=middle
        return (lo-1)%65536 if lo>max(0,self.written-65536) else -1

    def query(self,now):
        out=np.full(N_FEATURES,np.nan)
        base=self.micro.base
        x=self.micro.query(now)
        for i in range(N_BASE):out[i]=x[SELECT[i]]
        if not base.valid() or now<base.clock or now-base.book_ns>1_000_000_000:return out
        current_bid,current_ask=l5_notional(base)
        spread=base.best_ask-base.best_bid
        for wi in range(5):
            w=WINDOWS_MS[wi]*1_000_000;target=now-w;p=self.past(target)
            if p<0 or self.epoch>target:continue
            # Last *observable* value; no interpolation from the next message.
            # Stale boundaries do not masquerade as accurate 100ms evolution.
            if target-self.times[p]>200_000_000:continue
            j=N_BASE+8*wi;previous=self.history[p]
            out[j]=spread-previous[5]
            if previous[6]>0:out[j+1]=current_bid/previous[6]-1
            if previous[7]>0:out[j+2]=current_ask/previous[7]-1
            for k in range(5):out[j+3+k]=(self.cum[k]-previous[k])/(w/1e9)
        j=N_BASE+40
        if self.last_trade_ns>=0:out[j]=(now-self.last_trade_ns)/1e6
        p=base.past(now-1_000_000_000)
        if p>=0 and self.epoch<=now-1_000_000_000:
            sums=base.cum-base.history[p]
            # Public intensity-based depletion times, not queue waiting times.
            if sums[4]>0:out[j+1]=base.best_bid*base.tick*base.bid_qty(base.best_bid)/sums[4]
            if sums[3]>0:out[j+2]=base.best_ask*base.tick*base.ask_qty(base.best_ask)/sums[3]
            out[j+3]=max(0.,sums[10]-sums[4])
            out[j+4]=max(0.,sums[11]-sums[3])
        for wi in range(2):
            sec=2. if wi==0 else 3.;target=now-int(sec*1e9);p=base.past(target)
            if p<0 or self.epoch>target or target-base.times[p]>200_000_000:continue
            sums=base.cum-base.history[p];amount=sums[3]+sums[4];j=N_BASE+45+wi*4
            mid=(base.best_bid+base.best_ask)*base.tick/2
            out[j]=math.log(mid/base.history_mid[p])*10000
            out[j+1]=math.sqrt(max(0.,sums[6]))
            out[j+2]=(sums[3]-sums[4])/amount if amount>0 else 0.
            out[j+3]=max(0.,amount)/sec
        return out


@njit
def sample_chunk(state,data,next_ns,step_ns):
    count=max(0,(int(data[-1]['local_ts'])-next_ns)//step_ns+1)
    times=np.empty(count,dtype=np.int64);X=np.full((count,N_FEATURES),np.nan,dtype=np.float32)
    source=np.full(count,-1,dtype=np.int64);epoch=source.copy()
    q=i=0
    while i<len(data):
        t=data[i]['local_ts']
        while next_ns<t and q<count:
            times[q]=next_ns;X[q]=state.query(next_ns)
            source[q]=state.micro.base.clock;epoch[q]=state.epoch
            q+=1;next_ns+=step_ns
        j=i+1
        while j<len(data) and data[j]['local_ts']==t:j+=1
        state.ingest(data,i,j)
        if next_ns==t and q<count:
            times[q]=next_ns;X[q]=state.query(next_ns)
            source[q]=state.micro.base.clock;epoch[q]=state.epoch
            q+=1;next_ns+=step_ns
        i=j
    return next_ns,times[:q],X[:q],source[:q],epoch[:q]
