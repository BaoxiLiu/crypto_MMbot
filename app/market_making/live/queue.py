"""Public/received-private trade-only queue proxy; never produces executions."""
from dataclasses import dataclass, asdict
import math


@dataclass
class Ahead:
    cid: str
    side: int
    price_tick: int
    sent_ns: int
    send_book_ns: int
    send_quantity: float
    ack_ns: int=-1
    initial: float=0.
    front: float=0.
    initialized: bool=False
    cumulative_fill: float=0.
    trade_quantity: float=0.
    last_trade_source_ns: int=-1
    last_trade_local_ns: int=-1


class OwnQueueObserver:
    def __init__(self, emit=None):
        self.emit=emit or (lambda **data:None)
        self.orders={}

    def reset(self):
        self.emit(kind='reset')
        self.orders.clear()

    def sent(self,cid,side,price_tick,sent_ns,book_ns,quantity):
        if cid in self.orders or side not in (-1,1) or book_ns>sent_ns or quantity<0 or not math.isfinite(quantity):
            raise ValueError('Invalid public queue initialization')
        q=Ahead(cid,side,price_tick,sent_ns,book_ns,float(quantity))
        self.orders[cid]=q
        self.emit(kind='sent',**asdict(q))

    def received(self,cid,status,cumulative,now):
        q=self.orders.get(cid)
        if q is None:return
        if now<q.sent_ns or cumulative<q.cumulative_fill:
            raise ValueError('Reversed received order state')
        if status in ('NEW','PARTIALLY_FILLED') and q.ack_ns<0:q.ack_ns=now
        if cumulative>q.cumulative_fill:
            q.cumulative_fill=float(cumulative)
            q.front=0.  # A received own fill, unlike public prints, proves priority.
        self.emit(kind='received',status=status,receipt_ns=now,**asdict(q))
        if status in ('FILLED','CANCELED','EXPIRED','REJECTED'):self.orders.pop(cid,None)

    def initialize(self,cid,quantity,now,book_ns):
        q=self.orders[cid]
        if q.initialized:return
        if q.ack_ns<0 or q.ack_ns>now or book_ns>now or quantity<0 or not math.isfinite(quantity):
            raise ValueError('Queue ACK or public clock unavailable')
        q.initial=max(q.send_quantity,float(quantity))
        q.front=0. if q.cumulative_fill>0 else q.initial
        q.initialized=True
        self.emit(kind='initialized',public_quantity=float(quantity),public_clock_ns=book_ns,receipt_ns=now,**asdict(q))

    def advance(self,cid,quantity,source_ns,local_ns):
        q=self.orders[cid]
        if not q.initialized or source_ns<q.ack_ns or local_ns<source_ns or local_ns<q.last_trade_local_ns:
            raise ValueError('Uncertain pre-ACK or reversed public trade')
        if quantity<0 or not math.isfinite(quantity):raise ValueError('Invalid public consumption')
        old=q.front
        q.front=max(0.,q.front-quantity)
        q.trade_quantity+=quantity
        q.last_trade_source_ns=max(q.last_trade_source_ns,source_ns)
        q.last_trade_local_ns=local_ns
        self.emit(kind='trade',quantity=quantity,previous_front=old,**asdict(q))

    def estimate(self,cid,public_quantity,now,fresh=True):
        q=self.orders.get(cid)
        valid=(q is not None and q.initialized and q.ack_ns<=now
               and q.last_trade_local_ns<=now and fresh and public_quantity>=0
               and q.front<=public_quantity+1e-7)
        return q,bool(valid)
