"""Shared state transition core. No sockets, sleeps, private keys or fill simulation."""
from decimal import Decimal as D
import numpy as np
from ..fusion.market_data import TRADE_DTYPE, QuoteDepth
from .market import LocalBook
from .queue import OwnQueueObserver
from .decision import DecisionPipeline
from .contracts import DecisionContext, OrderView, QueueView


from app.market_making.fusion.kernel import Kernel


class StrategyEngine(Kernel):
    def __init__(self,rules,config,factors,prefix='maker',emit=None,decision=None,*,strategy_factory=None):
        super().__init__(rules,config,factors,prefix,emit,strategy_factory=strategy_factory)
        self.market_observer=getattr(self.strategy,'observe_market',None)
        if self.market_observer is not None and (
                not callable(self.market_observer)
                or not callable(getattr(self.strategy,'observation_due',None))
                or type(getattr(self.strategy,'requires_quote_depth',None)) is not bool):
            raise ValueError('Incomplete strategy market-input contract')
        self.pending_trades=[]
        self.last_trade_receipt_ns=-1
        self.market=LocalBook(rules)
        self.queue_observer=OwnQueueObserver()
        self.decision=decision or DecisionPipeline()
        if self.decision.price_model is not None and not callable(getattr(self.strategy,'desired_with_alpha',None)):
            raise ValueError('Selected strategy does not support external price forecasts')


    def trade(self,ns,trade_id,side,price,qty,received_ns=None):
        received_ns=ns if received_ns is None else received_ns
        if type(received_ns) is not int or not 0<=received_ns<=ns or received_ns<self.last_trade_receipt_ns:
            raise ValueError('Invalid/reversed trade receipt clock')
        accepted=super().trade(ns,trade_id,side,price,qty)
        if accepted:
            self.last_trade_receipt_ns=received_ns
            if self.market_observer is not None:
                if len(self.pending_trades)>=200000:
                    self.oms.halt_reason='market_buffer_overflow'
                    raise ValueError('Strategy trade buffer capacity exceeded')
                self.pending_trades.append((received_ns,side,float(price),float(qty)))
            for cid,q in tuple(self.queue_observer.orders.items()):
                if (q.initialized and q.side == -side and q.ack_ns <= ns
                        and q.price_tick == int(price/self.rules.tick)):
                    self.queue_observer.advance(cid,float(qty),ns,ns)
        return accepted

    def snapshot(self,data,ns):
        self.reset_market()
        self.market.snapshot(data,ns)

    def depth(self,data,ns,received_ns=None):
        try:
            received_ns=ns if received_ns is None else received_ns
            if type(received_ns) is not int or not 0 <= received_ns <= ns:
                raise ValueError('Invalid source receipt clock')
            accepted=self.market.update(data,ns)
            if accepted:
                view=self.market.view(1)
                self.top_of_book(ns,*view.bids[0],*view.asks[0])
                self.market.source_ns=received_ns
                self.strategy.obs.book_ns=received_ns
            return accepted
        except (ValueError,ArithmeticError):
            self.reset_market()
            raise

    def top_of_book(self,ns,bid,bid_qty,ask,ask_qty):
        hook=getattr(self.strategy,'on_top_of_book',None)
        if hook is not None:
            if any(q is None or not q.is_finite() or q<0 for q in (bid_qty,ask_qty)):
                raise ValueError('Selected strategy requires valid BBO quantities')
            hook(ns,bid,bid_qty,ask,ask_qty)
        self.book(ns,bid,ask)

    def observe_market(self,ns):
        if self.market_observer is None:
            return True
        if not self.market_ready or not self.external_healthy:
            return False
        needs_depth=self.strategy.requires_quote_depth
        if needs_depth and (not self.market.ready or not 0<=ns-self.market.source_ns<=self.config.as_config.stale_ns):
            return False
        if not self.strategy.observation_due(ns):
            return True
        depth=None
        if needs_depth:
            view=self.market.view(1)
            bid,ask=view.bids[0][0],view.asks[0][0]
            bids=np.array([float(self.market.quantity(1,bid-i*self.rules.tick)) for i in range(7)])
            asks=np.array([float(self.market.quantity(-1,ask+i*self.rules.tick)) for i in range(7)])
            depth=QuoteDepth(float(bid),float(ask),int(bid/self.rules.tick),int(ask/self.rules.tick),bids,asks)
        trades=np.array(self.pending_trades,dtype=TRADE_DTYPE)
        self.market_observer(depth,trades,ns)
        self.pending_trades.clear()
        return True

    def context(self,ns):
        working=tuple(OrderView(o.cid,o.side,o.price,o.remaining,o.phase.value,o.reduce_only)
                      for o in self.oms.working())
        queue_views=[]
        fresh=(self.market.ready and 0 <= ns-self.market.source_ns <= self.config.as_config.stale_ns)
        for o in self.oms.working():
            public=float(self.market.quantity(o.side,o.price)) if fresh else 0.
            q,valid=self.queue_observer.estimate(o.cid,public,ns,fresh=fresh)
            queue_views.append(QueueView(o.cid,valid,q.front if valid else None,ns,
                                        'public_trade_proxy' if valid else 'unknown_or_stale'))
        return DecisionContext(ns,self.strategy.obs.state(ns),self.oms.ledger.position,
                               working,tuple(queue_views))


    def reset_market(self):
        super().reset_market()
        self.market.invalidate();self.queue_observer.reset()
        self.pending_trades.clear();self.last_trade_receipt_ns=-1

    def plan(self,ns):
        """Freeze intentions at the observation time, before computation delay."""
        if not self.observe_market(ns):return {}
        m=self.mark or self.strategy.obs.mid
        if m<=0:return {}
        risk=self.oms.risk(m,ns)
        state=self.strategy.obs.state(ns)
        fresh_mark=self.mark_ns is not None and 0<=ns-self.mark_ns<=self.config.mark_stale_ns
        healthy=self.external_healthy and self.market_ready and state.ready and fresh_mark
        if not healthy or risk:return {}
        context=self.context(ns)
        try:
            alpha=self.decision.forecast(context)
            if hasattr(self.strategy,'before_plan'):
                self.strategy.before_plan(self.oms.working(),self.oms.ledger.equity(m),m)
            if self.decision.price_model is not None:
                targets=self.strategy.desired_with_alpha(context.position,ns,alpha)
            else:
                targets=self.strategy.desired(context.position,ns)
            return self.decision.choose(context,targets)
        except ValueError:
            self.oms.halt_reason='model_contract_failure'
            self.decision.last['error']='model_contract_failure'
            return {}

    def dispatch(self,targets,ns,urgent_cancel_sides=()):
        commands=super().dispatch(targets,ns,urgent_cancel_sides)
        fresh=self.market.ready and 0 <= ns-self.market.source_ns <= self.config.as_config.stale_ns
        if fresh:
            for command in commands:
                if command.kind=='new':
                    self.queue_observer.sent(command.cid,command.side,int(command.price/self.rules.tick),
                        ns,self.market.source_ns,float(self.market.quantity(command.side,command.price)))
        return commands


    def update(self,cid,status,cumulative,cum_quote,ns,exchange_id=0,exchange_ms=0):
        fill=super().update(cid,status,cumulative,cum_quote,ns,exchange_id,exchange_ms)
        self.queue_observer.received(cid,status,float(self.oms.find(cid).filled),ns)
        q=self.queue_observer.orders.get(cid)
        if (q and not q.initialized and q.ack_ns>=0 and self.market.ready
                and 0 <= ns-self.market.source_ns <= self.config.as_config.stale_ns):
            self.queue_observer.initialize(cid,float(self.market.quantity(q.side,D(q.price_tick)*self.rules.tick)),
                                           ns,self.market.source_ns)
        return fill

