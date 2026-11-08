"""Shared state transition core. No sockets, sleeps, private keys or fill simulation."""
from collections import deque,Counter
from decimal import Decimal as D
from .orders import OrderManager


class Markouts:
    def __init__(self):
        self.pending={h:deque() for h in (1,5)}
        self.sums=Counter();self.weights=Counter();self.counts=Counter();self.censored=Counter()
        self.fill_serial=0

    def fill(self,fill):
        if fill:
            self.fill_serial+=1
            for h,q in self.pending.items():q.append((fill['ns']+h*1_000_000_000,dict(fill),self.fill_serial))

    def mark(self,ns,mid,on_completed=None):
        for h,q in self.pending.items():
            while q and q[0][0]<=ns:
                target,f,fill_id=q.popleft()
                bp=float(f['side']*(mid-f['price'])/f['price']*10000)
                weight=float(f['qty']*f['price'])
                self.sums[h]+=bp*weight;self.weights[h]+=weight;self.counts[h]+=1
                if on_completed is not None:
                    on_completed(dict(fill=dict(f),fill_id=fill_id,horizon_ms=h*1000,
                                      target_ns=target,observed_ns=ns,mid=mid))

    def gap(self):
        for h,q in self.pending.items():self.censored[h]+=len(q);q.clear()

    def report(self):
        return {str(h):dict(notional_weighted_bp=self.sums[h]/self.weights[h] if self.weights[h] else None,
                            observations=self.counts[h],censored=self.censored[h]+len(q)) for h,q in self.pending.items()}


class Kernel:
    def __init__(self,rules,config,factors,prefix='fusion',emit=None,strategy_factory=None):
        self.rules,self.config,self.factors=rules,config,factors
        if not callable(strategy_factory):
            raise ValueError("Explicit strategy_factory required; execution has no default strategy")
        self.strategy=strategy_factory(rules,config,factors)
        self.oms=OrderManager(rules,config,prefix,emit)
        self.markouts=Markouts()
        self.mark=D(0);self.mark_ns=None
        self.market_ready=False
        self.external_healthy=True

    def book(self,ns,bid,ask):
        self.strategy.book(ns,bid,ask)
        self.market_ready=True
        self.markouts.mark(ns,(bid+ask)/2,getattr(self.strategy,'on_markout',None))

    def trade(self,ns,trade_id,side,price,qty):
        return self.strategy.obs.trade(ns,trade_id,side,price,qty)

    def mark_price(self,ns,mark):
        if not mark.is_finite() or mark<=0:raise ValueError('invalid mark')
        self.mark,self.mark_ns=mark,ns

    def reset_market(self):
        self.market_ready=False
        self.strategy.reset_market();self.markouts.gap()

    def plan(self,ns):
        """Freeze intentions at the observation time, before computation delay."""
        m=self.mark or self.strategy.obs.mid
        if m<=0:return {}
        risk=self.oms.risk(m,ns)
        state=self.strategy.obs.state(ns)
        fresh_mark=self.mark_ns is not None and 0<=ns-self.mark_ns<=self.config.mark_stale_ns
        healthy=self.external_healthy and self.market_ready and state.ready and fresh_mark
        if healthy and not risk and hasattr(self.strategy,'before_plan'):
            self.strategy.before_plan(self.oms.working(),self.oms.ledger.equity(m),m)
        return self.strategy.desired(self.oms.ledger.position,ns) if healthy and not risk else {}

    def dispatch(self,targets,ns,urgent_cancel_sides=()):
        # Responses during computation may change inventory. Recheck hard risk
        # and freshness, but never read future market observations into a plan.
        m=self.mark or self.strategy.obs.mid
        if m<=0:return []
        risk=self.oms.risk(m,ns)
        obs=self.strategy.obs
        healthy=(self.external_healthy and self.market_ready and obs.book_ns is not None
                 and 0<=ns-obs.book_ns<=self.config.as_config.stale_ns
                 and self.mark_ns is not None and 0<=ns-self.mark_ns<=self.config.mark_stale_ns)
        urgent_cancel_sides=set(urgent_cancel_sides)|set(getattr(self.strategy,'urgent_cancel_sides',()))
        for order in self.oms.active.values():
            if order.side in urgent_cancel_sides:order.cancel_wanted=True
        return self.oms.reconcile_targets(targets if healthy and not risk else {},m,ns)

    def step(self,ns):
        return self.dispatch(self.plan(ns),ns)

    def update(self,*args,**kwargs):
        fill=self.oms.update(*args,**kwargs)
        self.markouts.fill(fill)
        if fill and hasattr(self.strategy, 'on_fill'):
            self.strategy.on_fill(fill, self.oms.ledger.position)
        return fill

    def report(self):
        o=self.oms;l=o.ledger;m=self.mark or self.strategy.obs.mid
        o.assert_invariants(m)
        return dict(factors=self.factors.label,position=str(l.position),wallet=str(l.wallet),
                    realized=str(l.realized),unrealized=str(l.unrealized(m)),fees=str(l.fees),
                    funding=str(l.funding),equity=str(l.equity(m)),pnl=str(l.equity(m)-l.initial),
                    cash_flow=str(o.cash_flow),max_position=str(o.max_position),
                    max_drawdown=str(l.max_drawdown),halt_reason=o.halt_reason,
                    orders=dict(o.stats),policy=dict(self.strategy.counts),markouts=self.markouts.report())
