"""Frozen quote-selection functions from the winning quote_value variant."""
import math
import numpy as np
from numba import njit
from .value_features import action_features, value_prediction

@njit
def support_fraction(distance,supported,fractions):
    d=int(round(distance))
    if abs(distance-d)>1e-5 or d<0 or d>6:return False,0.,-1
    if d==5:
        return supported[4] and supported[5],.5*(fractions[4]+fractions[5]),4
    a=d if d<5 else 5
    return supported[a],fractions[a],a


@njit
def score_quote(dep,x,side,mid,bid,ask,tick,lot,px,vol,room,mean,scale,fb,fi,eb,ei,supported,fractions,order_notional=50.):
    distance=((bid-px)if side>0 else(px-ask))/tick
    support,frac,action=support_fraction(distance,supported,fractions)
    if not support:return False,0.,0.,0.,-1
    qty=min(math.floor(order_notional/px/lot)*lot,math.floor((room+lot*1e-8)/lot)*lot)
    if qty<=0:return False,0.,0.,0.,-1
    queue=dep.bid_qty_at_tick(int(round(px/tick))) if side>0 else dep.ask_qty_at_tick(int(round(px/tick)))
    z=action_features(x,side,mid,bid,ask,tick,px,qty,queue,vol)
    p,edge,frac=value_prediction(z,mean,scale,fb,fi,eb,ei,frac)
    return True,p,edge,p*qty*px*frac*edge/1e4,action


@njit
def select_quote(dep,x,side,position,room,center,half,mid,bid,ask,tick,lot,vol,
                 mean,scale,fb,fi,eb,ei,supported,fractions,mode,order_notional=50.):
    original_px=math.floor((center-half)/tick)*tick if side>0 else math.ceil((center+half)/tick)*tick
    original_px=min(original_px,ask-tick)if side>0 else max(original_px,bid+tick)
    ok,p,e,score,action=score_quote(dep,x,side,mid,bid,ask,tick,lot,original_px,vol,room,
                                   mean,scale,fb,fi,eb,ei,supported,fractions,order_notional)
    reducing=side*position<0
    if mode==1:
        allowed=not ok or e>0 or reducing
        return original_px,half,allowed,p,e,score,float(action),1. if ok else 2.
    best=0.;chosen=original_px;width=half;bp=0.;be=0.;ba=-1.;found=False;any_support=ok
    for k in range(6):
        candidate_half=half
        if k<5:
            widths=(1.,2.,3.,4.,6.)
            candidate_half=widths[k]*tick
        px=math.floor((center-candidate_half)/tick)*tick if side>0 else math.ceil((center+candidate_half)/tick)*tick
        px=min(px,ask-tick)if side>0 else max(px,bid+tick)
        supported_,p_,e_,value_,a_=score_quote(dep,x,side,mid,bid,ask,tick,lot,px,vol,room,
                                             mean,scale,fb,fi,eb,ei,supported,fractions,order_notional)
        any_support=any_support or supported_
        if supported_ and value_>best:
            best=value_;chosen=px;width=candidate_half;bp=p_;be=e_;ba=float(a_);found=True
    if found:return chosen,width,True,bp,be,best,ba,3.
    if reducing or not any_support:
        return original_px,half,True,p,e,score,float(action),4. if reducing else 2.
    return original_px,half,False,p,e,score,float(action),5.
