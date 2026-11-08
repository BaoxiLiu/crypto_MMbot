"""Public-L2 factors for market-direction prediction, never own-fill labels.

Reuse the frozen 136-factor extractor with bid as the fixed orientation. Drop
side/quote-distance columns; extend depth shape and past-only shape dynamics.
All decreases/refills are net-L2 proxies, never identified cancellations.
"""
import math
import numpy as np
from numba import njit, int64, float64
from numba.experimental import jitclass
from app.factor_library.microstructure_core import FeatureState, FEATURE_NAMES as LEGACY_NAMES

KEEP = np.array([i for i in range(136) if i not in (0,3,4)], dtype=np.int64)
BASE_NAMES = [LEGACY_NAMES[i].replace('own','bid').replace('opposite','ask').replace('signed_','bid_signed_') for i in KEEP]
SHAPE_NAMES = [f'{name}_l{n}' for n in (3,5,10) for name in (
    'log_bid_touch_back_ratio','log_ask_touch_back_ratio','bid_uniformity','ask_uniformity',
    'bid_hhi','ask_hhi','bid_outer_share','ask_outer_share')]
SHAPE_NAMES += ['log_bid_notional_l3','log_ask_notional_l3','imbalance_l3']
SHAPE_NAMES += [f'depth_symmetry_l{n}' for n in (1,3,5,10)]
DYNAMIC_INDEX = np.array([10,11,12,13,14,15,24,25,26,30], dtype=np.int64)
EXTRA_NAMES = SHAPE_NAMES + [f'w{w}ms_delta_{SHAPE_NAMES[i]}' for w in (500,1000,5000) for i in DYNAMIC_INDEX]
EXTRA_NAMES += ['w500ms_return_bp','w500ms_log_trade_notional_rate','w500ms_trade_imbalance',
                'w500ms_ofi_scaled','w500ms_realized_vol_bp','volatility_rate_ratio_100ms_5s',
                'bid_fragility_interaction','ask_fragility_interaction']
FEATURE_NAMES = BASE_NAMES + EXTRA_NAMES


@njit
def shape(base):
    out = np.full(31,np.nan)
    if not base.valid():return out
    bq=np.zeros(10);aq=np.zeros(10);bp=np.zeros(10);ap=np.zeros(10)
    b,a=base.best_bid,base.best_ask
    for i in range(10):
        if b<=0 or a>=2**62:return out
        bq[i]=base.bid_qty(b);aq[i]=base.ask_qty(a);bp[i]=b*base.tick;ap[i]=a*base.tick
        if i<9:b=base.below(b);a=base.above(a)
    for j in range(3):
        n=3 if j==0 else 5 if j==1 else 10
        btot,atot=np.sum(bq[:n]),np.sum(aq[:n]);offset=j*8
        if btot<=0 or atot<=0:return out
        out[offset]=math.log1p(bq[0]/np.sum(bq[1:n]))
        out[offset+1]=math.log1p(aq[0]/np.sum(aq[1:n]))
        out[offset+2]=1/(1+np.std(bq[:n])/np.mean(bq[:n]))
        out[offset+3]=1/(1+np.std(aq[:n])/np.mean(aq[:n]))
        out[offset+4]=np.sum((bq[:n]/btot)**2);out[offset+5]=np.sum((aq[:n]/atot)**2)
        out[offset+6]=np.sum(bq[2:n])/btot;out[offset+7]=np.sum(aq[2:n])/atot
    out[24]=math.log1p(np.sum(bq[:3]*bp[:3]));out[25]=math.log1p(np.sum(aq[:3]*ap[:3]))
    out[26]=(np.sum(bq[:3])-np.sum(aq[:3]))/(np.sum(bq[:3])+np.sum(aq[:3]))
    for j in range(4):
        n=1 if j==0 else 3 if j==1 else 5 if j==2 else 10
        out[27+j]=1-abs(np.sum(bq[:n])-np.sum(aq[:n]))/(np.sum(bq[:n])+np.sum(aq[:n]))
    return out


@jitclass([('base',FeatureState.class_type.instance_type),('epoch',int64),('written',int64),
           ('times',int64[:]),('history',float64[:,:])])
class MicrostructureState:
    def __init__(self,tick):
        self.base=FeatureState(tick,2000000,262144)
        self.epoch=-1;self.written=0
        self.times=np.zeros(1024,dtype=np.int64);self.history=np.zeros((1024,31))

    def ingest(self,data,begin,end):
        self.base.ingest(data,begin,end)
        if self.epoch!=self.base.epoch_ns:
            self.epoch=self.base.epoch_ns;self.written=0
        t=self.base.clock
        if self.base.valid() and t-self.base.book_ns<=1_000_000_000 and (self.written==0 or t-self.times[(self.written-1)%1024]>=100_000_000):
            slot=self.written%1024;self.times[slot]=t;self.history[slot]=shape(self.base);self.written+=1

    def query(self,now):
        # Fixed bid orientation, no position or future/current own-fill input.
        x=self.base.query(now,1,self.base.best_bid*self.base.tick,0.)
        out=np.full(202,np.nan)
        for j in range(len(KEEP)):out[j]=x[KEEP[j]]
        if not self.base.valid() or now<self.base.clock or now-self.base.book_ns>1_000_000_000:return out
        current=shape(self.base);out[133:164]=current
        for wi in range(3):
            w=500_000_000 if wi==0 else 1_000_000_000 if wi==1 else 5_000_000_000
            target=now-w;lo=max(0,self.written-1024);hi=self.written
            while lo<hi:
                mid=(lo+hi)//2
                if self.times[mid%1024]<=target:lo=mid+1
                else:hi=mid
            if lo>max(0,self.written-1024):
                p=(lo-1)%1024
                if 0<=target-self.times[p]<=200_000_000:
                    for j in range(10):out[164+wi*10+j]=current[DYNAMIC_INDEX[j]]-self.history[p,DYNAMIC_INDEX[j]]
        p=self.base.past(now-500_000_000)
        if p>=0 and self.base.epoch_ns<=now-500_000_000:
            sums=self.base.cum-self.base.history[p];total=sums[3]+sums[4]
            mid=(self.base.best_bid+self.base.best_ask)*self.base.tick/2
            out[194]=math.log(mid/self.base.history_mid[p])*10000
            out[195]=math.log1p(max(0.,total)/.5)
            out[196]=(sums[3]-sums[4])/total if total>1e-12 else 0.
            out[197]=sums[5]/(self.base.bid_qty(self.base.best_bid)+self.base.ask_qty(self.base.best_ask))
            out[198]=math.sqrt(max(0.,sums[6]))
        # RV is sqrt(sum returns²), so normalize by window duration first.
        if np.isfinite(x[28]) and np.isfinite(x[86]) and x[86]>0:
            out[199]=(x[28]/math.sqrt(.1))/(x[86]/math.sqrt(5.))
            out[200]=current[8]*(1-current[14])*(1+out[199])
            out[201]=current[9]*(1-current[15])*(1+out[199])
        return out


@njit
def ingest_until(state,data,begin,end):
    i=begin
    while i<end:
        j=i+1
        while j<end and data[j]['local_ts']==data[i]['local_ts']:j+=1
        state.ingest(data,i,j);i=j


@njit
def sample_chunk(state,data,next_ns,step_ns):
    count=max(0,(int(data[-1]['local_ts'])-next_ns)//step_ns+1)
    times=np.zeros(count,dtype=np.int64);X=np.full((count,202),np.nan,dtype=np.float32)
    mids=np.full(count,np.nan);epochs=np.full(count,-1,dtype=np.int64);source_ns=np.full(count,-1,dtype=np.int64)
    mt=np.zeros(len(data),dtype=np.int64);mv=np.zeros(len(data));me=np.zeros(len(data),dtype=np.int64)
    q=0;m=0;i=0
    while i<len(data):
        t=data[i]['local_ts']
        while next_ns<t and q<count:
            times[q]=next_ns;X[q]=state.query(next_ns);epochs[q]=state.base.epoch_ns;source_ns[q]=state.base.clock
            if state.base.valid():mids[q]=(state.base.best_bid+state.base.best_ask)*state.base.tick/2
            q+=1;next_ns+=step_ns
        j=i+1
        while j<len(data) and data[j]['local_ts']==t:j+=1
        state.ingest(data,i,j)
        if state.base.valid() and state.base.book_ns==t:
            mt[m]=t;mv[m]=(state.base.best_bid+state.base.best_ask)*state.base.tick/2;me[m]=state.base.epoch_ns;m+=1
        if next_ns==t and q<count:
            times[q]=next_ns;X[q]=state.query(next_ns);epochs[q]=state.base.epoch_ns;source_ns[q]=state.base.clock
            if state.base.valid():mids[q]=(state.base.best_bid+state.base.best_ask)*state.base.tick/2
            q+=1;next_ns+=step_ns
        i=j
    return next_ns,times[:q],X[:q],mids[:q],epochs[:q],source_ns[:q],mt[:m],mv[:m],me[:m]
