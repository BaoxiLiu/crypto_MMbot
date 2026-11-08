"""Frozen numerical training kernels; samples must be causally timestamped."""
import numpy as np
from .value_features import K

def train_alpha(x,y,lam):
    mean=x.mean(axis=0);scale=x.std(axis=0);scale=np.maximum(scale,1e-6)
    z=np.clip((x-mean)/scale,-10,10)
    ym=float(y.mean());n=len(y);zm=z.mean(axis=0);zc=z-zm
    beta=np.linalg.solve(zc.T@zc/n+lam*np.eye(x.shape[1]),zc.T@(y-ym)/n)
    return mean,scale,beta,ym-float(zm@beta)


def predict_alpha(model,x):
    mean,scale,beta,ym=model
    return np.clip((x-mean)/scale,-10,10)@beta+ym


def sigmoid(x):return 1/(1+np.exp(-np.clip(x,-25,25)))


def train_value(x,filled,fraction,edge,notional,action):
    mean=x.mean(axis=0,dtype=float);scale=np.maximum(x.std(axis=0,dtype=float),1e-6)
    z=np.clip((x.astype(float)-mean)/scale,-8,8)
    design=np.column_stack([np.ones(len(z)),z]);n=len(z)
    weights=np.zeros(K+1);p0=(filled.sum()+.5)/(n+1)
    weights[0]=np.log(p0/(1-p0))
    penalty=np.eye(K+1)*.01;penalty[0,0]=1e-8
    for _ in range(12):
        p=sigmoid(design@weights);v=np.maximum(p*(1-p),1e-7)
        gradient=design.T@(p-filled)/n+penalty@weights
        hessian=(design.T*v)@design/n+penalty
        delta=np.linalg.solve(hessian,gradient)
        # Newton steps can overshoot with rare events and near-separation.
        delta*=min(1.,2./max(np.max(np.abs(delta)),1e-9))
        weights-=delta
        if np.max(np.abs(delta))<1e-6:break
    good=filled.astype(bool)
    w=notional[good];w=w/w.sum();d=design[good]
    ridge=np.eye(K+1)*.1;ridge[0,0]=1e-10
    e=np.linalg.solve((d.T*w)@d+ridge,d.T@(w*edge[good]))
    counts=np.bincount(action,minlength=6)
    fills=np.bincount(action[good],minlength=6)
    # Amount weighting of edge is paired with an order-weighted expected fill amount.
    global_fraction=float(np.mean(fraction[good]));fractions=[]
    for a in range(6):
        mask=good&(action==a)
        fractions.append(float((fraction[mask].sum()+20*global_fraction)/(mask.sum()+20)))
    return dict(mean=mean,scale=scale,fill_beta=weights[1:],fill_intercept=weights[0],
                edge_beta=e[1:],edge_intercept=e[0],fraction=np.array(fractions),
                supported=(counts>=200)&(fills>=20))
