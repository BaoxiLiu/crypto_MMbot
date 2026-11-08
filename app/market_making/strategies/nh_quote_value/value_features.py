"""Shared, causal action features for randomized probes and live quote scoring."""
import math
import numpy as np
from numba import njit

NAMES=['signed_'+n for n in ['book_offset','ofi_tanh','ti','count1','amount1','volume1',
       'momentum1','vwap1','count5','amount5','volume5','momentum5','vwap5']]+[
       'log_count1','log_amount1','rv1','log_count5','log_amount5','rv5',
       'vol_1s','spread_bp','tick_bp','distance_bp','log_queue_notional','log_order_notional',
       'distance_tick','distance_tick_sq','signed_ti_x_distance']
K=len(NAMES)


@njit
def action_features(x,side,mid,bid,ask,tick,px,qty,queue,vol):
    z=np.empty(28,np.float64)
    indices=(0,1,2,3,4,5,6,7,13,14,15,16,17)
    for j in range(13):z[j]=side*x[indices[j]]
    z[13]=x[9];z[14]=x[8];z[15]=x[11]
    z[16]=x[19];z[17]=x[18];z[18]=x[21]
    z[19]=vol;z[20]=(ask-bid)/mid*1e4;z[21]=tick/mid*1e4
    z[22]=side*(mid-px)/mid*1e4
    z[23]=math.log1p(max(0.,queue*px));z[24]=math.log1p(qty*px)
    distance=((bid-px) if side>0 else (px-ask))/tick
    z[25]=distance;z[26]=distance*distance;z[27]=side*x[2]*distance
    return z


@njit
def dot_prediction(z,mean,scale,beta,intercept):
    v=intercept
    for j in range(len(z)):
        v+=min(8.,max(-8.,(z[j]-mean[j])/scale[j]))*beta[j]
    return v


@njit
def value_prediction(z,mean,scale,fill_beta,fill_intercept,edge_beta,edge_intercept,fraction):
    logit=dot_prediction(z,mean,scale,fill_beta,fill_intercept)
    probability=1./(1.+math.exp(-min(25.,max(-25.,logit))))
    edge=dot_prediction(z,mean,scale,edge_beta,edge_intercept)
    return probability,min(50.,max(-50.,edge)),fraction
