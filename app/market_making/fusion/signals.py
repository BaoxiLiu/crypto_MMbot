"""Causal observable market conditions, not an inferred latent market regime.

Inspiration: tfrmma's separation of flow diagnostics and quote protection.
All formulas/state here are independently implemented with an injected clock.
"""
from collections import deque
from dataclasses import dataclass
from decimal import Decimal as D
import math
from ..as_math import Variance


@dataclass(frozen=True)
class MarketState:
    ns: int
    mid: D
    bid: D
    ask: D
    variance: float
    momentum_bp: float
    sigma_bp: float
    flow_ratio: float
    flow_count: int
    flow_notional: float
    regime: str
    adverse_side: int
    ready: bool


class Observations:
    def __init__(self,config,tick):
        self.config,self.tick=config,tick
        self.reference_qty=None
        self.last_clock=None
        self.last_trade_id=-1
        self.reset()

    def reset(self):
        self.vol=Variance(self.config.as_config)
        self.mids=deque()
        self.trades=deque()
        self.buy=self.sell=self.notional=0.0
        self.book_ns=None
        self.bid=self.ask=self.mid=D(0)
        self.stress=False
        self.flow_veto={1:0,-1:0}

    def clock(self,ns):
        if type(ns) is not int or ns<0 or (self.last_clock is not None and ns<self.last_clock):
            raise ValueError('market clock reversed')
        self.last_clock=ns

    def trade(self,ns,trade_id,side,price,qty):
        self.clock(ns)
        if side not in (-1,1) or type(trade_id) is not int or trade_id<0:raise ValueError('trade identity')
        if not price.is_finite() or not qty.is_finite() or price<=0 or qty<=0:raise ValueError('trade values')
        if trade_id<=self.last_trade_id:return False
        self.last_trade_id=trade_id
        amount=float(qty);notional=float(qty*price)
        self.trades.append((ns,side,amount,notional))
        if side==1:self.buy+=amount
        else:self.sell+=amount
        self.notional+=notional
        self._expire_trades(ns)
        self._flow(ns)
        return True

    def _expire_trades(self,ns):
        cutoff=ns-self.config.flow_window_ns
        while self.trades and self.trades[0][0]<cutoff:
            _,side,qty,n=self.trades.popleft()
            if side==1:self.buy-=qty
            else:self.sell-=qty
            self.notional-=n
        # Clamp only roundoff, not observations.
        self.buy=max(0.,self.buy);self.sell=max(0.,self.sell);self.notional=max(0.,self.notional)

    def _flow(self,ns):
        total=self.buy+self.sell
        ratio=(self.buy-self.sell)/total if total>1e-12 else 0.
        if (len(self.trades)>=self.config.flow_min_trades and self.notional>=self.config.flow_min_notional
                and abs(ratio)>=self.config.flow_imbalance):
            adverse=-1 if ratio>0 else 1  # active buys threaten our passive ask
            self.flow_veto[adverse]=ns+self.config.flow_cooldown_ns
        return ratio

    def book(self,ns,bid,ask):
        self.clock(ns)
        if not (bid.is_finite() and ask.is_finite() and D(0)<bid<ask and bid%self.tick==0 and ask%self.tick==0):
            self.reset();raise ValueError('invalid BBO')
        if self.book_ns is not None and ns-self.book_ns>self.config.as_config.stale_ns:
            self.reset()
        self.book_ns,self.bid,self.ask,self.mid=ns,bid,ask,(bid+ask)/2
        self.vol.update(ns,float(self.mid/self.tick))
        self.mids.append((ns,float(self.mid)))
        cutoff=ns-self.config.momentum_ns
        while len(self.mids)>1 and self.mids[1][0]<=cutoff:self.mids.popleft()
        return self.state(ns)

    def state(self,ns):
        self.clock(ns)
        self._expire_trades(ns)
        ratio=self._flow(ns)
        fresh=self.book_ns is not None and ns-self.book_ns<=self.config.as_config.stale_ns
        ready=fresh and self.vol.ready
        momentum=0.
        if self.mids and self.mids[0][0]<=ns-self.config.momentum_ns:
            momentum=(float(self.mid)/self.mids[0][1]-1)*10000
        sigma=math.sqrt(self.vol.value)*float(self.tick/self.mid)*10000 if self.mid else 0.
        if ready:
            threshold=self.config.stress_sigma_bp*(self.config.stress_release_ratio if self.stress else 1)
            self.stress=sigma>=threshold
        regime=('UNKNOWN' if not ready else 'STRESS' if self.stress else
                'UP' if momentum>=self.config.trend_bp else 'DOWN' if momentum<=-self.config.trend_bp else 'NORMAL')
        adverse=1 if self.flow_veto[1]>ns else -1 if self.flow_veto[-1]>ns else 0
        # Both may be active after a rapid flow reversal. Decision checks both.
        return MarketState(ns,self.mid,self.bid,self.ask,self.vol.value,momentum,sigma,ratio,
                           len(self.trades),self.notional,regime,adverse,bool(ready))
