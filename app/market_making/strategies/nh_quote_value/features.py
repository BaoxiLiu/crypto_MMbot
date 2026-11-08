"""Online equivalents of cbts_factor_lab/extract.py (only C1's 1s/5s columns)."""
import math
import numpy as np
from numba import njit

S=1_000_000_000
STEP=100_000_000
CAPACITY=200_000
NAMES=['book_offset_bp','ofi_original_tanh','ti_1s']+[f'{n}_{w}s' for w in [1,5] for n in
       ['count_imb','amount_imb','volume_imb','mean_last_momentum','vwap_side_imb_bp',
        'log_mean_amount','log_count_rate','price_cv_bp','trade_rv_bp','posret_fraction']]

@njit
def new_state():
    return (np.zeros(CAPACITY,np.int64),np.zeros((CAPACITY,11)),
            np.zeros(3,np.int64),np.zeros((2,10)),np.zeros(2))

@njit
def ingest(trades,ts,atoms,ptr,tot,prices):
    for t in trades:
        tail=ptr[0]
        if tail-ptr[2]>=len(ts):raise RuntimeError('5-second trade ring capacity exceeded')
        j=tail%len(ts);ts[j]=t.local_ns
        atom=atoms[j]
        for k in range(11):atom[k]=0.
        if prices[0]==0.:prices[0]=t.price
        side=0 if t.side > 0 else 1
        atom[side]=1.;atom[2+side]=t.price*t.qty;atom[4+side]=t.qty
        rel=t.price/prices[0]-1.;atom[6]=rel;atom[7]=rel*rel
        if prices[1]>0.:
            atom[8]=math.log(t.price/prices[1])**2
            atom[9]=1. if t.price>prices[1] else 0.
        atom[10]=t.price;prices[1]=t.price
        for w in range(2):
            for k in range(10):tot[w,k]+=atom[k]
        ptr[0]+=1

@njit
def window_values(now,ts,atoms,ptr,tot,prices):
    out=np.zeros(20,np.float32)
    for w in range(2):
        win=S if w==0 else 5*S
        while ptr[w+1]<ptr[0] and ts[ptr[w+1]%len(ts)]<=now-win:
            j=ptr[w+1]%len(ts)
            for k in range(10):tot[w,k]-=atoms[j,k]
            ptr[w+1]+=1
        count=ptr[0]-ptr[w+1]
        if count==0:
            # Remove subtraction residue when a window empties.
            for k in range(10):tot[w,k]=0.
            continue
        z=tot[w];k=10*w
        nb=max(0.,z[0]);ns=max(0.,z[1]);ab=max(0.,z[2]);ass=max(0.,z[3])
        vb=max(0.,z[4]);vs=max(0.,z[5]);amount=ab+ass;volume=vb+vs
        out[k]=(nb-ns)/count
        if amount>0.:out[k+1]=(ab-ass)/amount
        if volume>0.:out[k+2]=(vb-vs)/volume
        mean_rel=z[6]/count;mean=prices[0]*(1+mean_rel)
        last=atoms[(ptr[0]-1)%len(ts),10]
        out[k+3]=-math.asinh((mean/last-1.)/.000169129652112)
        if nb>0 and ns>0 and vb>0 and vs>0:
            bp=ab/vb;sp=ass/vs;out[k+4]=(bp-sp)/(bp+sp)*1e4
        out[k+5]=math.log1p(amount/count);out[k+6]=math.log1p(count/(win/S))
        if count>1:
            variance=max(0.,(z[7]-count*mean_rel**2)/(count-1))
            out[k+7]=math.sqrt(variance)/(1+mean_rel)*1e4
            out[k+8]=math.sqrt(max(0.,z[8]-atoms[ptr[w+1]%len(ts),8]))*1e4
        out[k+9]=max(0.,min(1.,z[9]/count))
    return out

@njit
def feature(book_offset,ofi,ti,window):
    x=np.empty(23,np.float32)
    x[0]=book_offset;x[1]=math.tanh(ofi);x[2]=ti
    for j in range(20):x[3+j]=window[j]
    return x

@njit
def predict(x,mean,scale,beta,intercept):
    ans=intercept
    for j in range(23):ans+=min(10.,max(-10.,(float(x[j])-mean[j])/scale[j]))*beta[j]
    return ans
