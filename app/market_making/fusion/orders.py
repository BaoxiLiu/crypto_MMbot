"""Single order ledger. Pending/unknown/canceling quantities reserve risk.

No public-market print is ever interpreted as a real fill here. Adapters supply
cumulative exchange executions; REST/user-stream duplicates reconcile by CID.
"""
from collections import OrderedDict, Counter, deque
from dataclasses import dataclass, asdict
from decimal import Decimal as D
import re
from ..core import Ledger, floor_step
from ..contracts import Phase



TERMINAL={Phase.FILLED,Phase.CANCELED,Phase.EXPIRED,Phase.REJECTED}




@dataclass(frozen=True)
class Command:
    kind: str
    cid: str
    side: int
    price: D
    qty: D
    reduce_only: bool


@dataclass
class Working:
    cid: str
    serial: int
    side: int
    price: D
    qty: D
    created_ns: int
    reduce_only: bool=False
    phase: Phase=Phase.PENDING_NEW
    filled: D=D(0)
    quote_filled: D=D(0)
    cancel_wanted: bool=False
    exchange_id: int=0
    last_exchange_ms: int=0
    actual_fee_qty: D=D(0)
    # Dict of real trade ID -> (qty, commission), retained with the order.
    fee_receipts: dict=None

    def __post_init__(self):
        if self.fee_receipts is None:self.fee_receipts={}

    @property
    def remaining(self):return self.qty-self.filled


class OrderManager:
    def __init__(self,rules,config,prefix='fusion',emit=None):
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,18}',prefix):raise ValueError('CID prefix')
        self.rules,self.config,self.prefix=rules,config,prefix
        self.ledger=Ledger(initial=config.initial_cash,wallet=config.initial_cash,
                           day_start_equity=config.initial_cash,peak_equity=config.initial_cash)
        self.active={};self.recent=OrderedDict();self.serial=0
        self.cash_flow=D(0)
        self.halt_reason='';self.reconciled=True
        self.new_times=deque();self.clock_ns=None
        self.stats=Counter()
        self.emit=emit or (lambda kind,ns,**data:None)
        self.max_position=D(0)

    def clock(self,ns):
        if type(ns) is not int or ns<0 or (self.clock_ns is not None and ns<self.clock_ns):
            self.halt_reason='clock_reversed';raise ValueError(self.halt_reason)
        self.clock_ns=ns

    def working(self):return list(self.active.values())
    def find(self,cid):return self.active.get(cid) or self.recent.get(cid)
    def pending(self,side):return sum((o.remaining for o in self.active.values() if o.side==side),D(0))

    def capacity(self,side,price,mark):
        limit=min(self.config.as_config.position_cap,self.config.max_notional/max(price,mark))
        return max(D(0),limit-side*self.ledger.position-self.pending(side))

    def cancel(self,o):
        o.cancel_wanted=True
        if o.phase in (Phase.LIVE,Phase.PARTIAL):
            o.phase=Phase.PENDING_CANCEL
            self.stats['cancel_requests']+=1
            return Command('cancel',o.cid,o.side,o.price,o.remaining,o.reduce_only)
        # New-order ACK may race with this intent; cancel is deferred, not lost.
        return None

    def reconcile_targets(self,targets,mark,ns):
        self.clock(ns)
        commands=[]
        if self.halt_reason or not self.reconciled:targets={}
        for side in (1,-1):
            rows=[o for o in self.active.values() if o.side==side]
            if len(rows)>1:raise AssertionError('Multiple unconfirmed orders on one side')
            o=rows[0] if rows else None
            wanted=targets.get(side)
            if o:
                cap=min(self.config.as_config.position_cap,self.config.max_notional/max(mark,o.price))
                safe=abs(self.ledger.position+side*o.remaining)<=cap
                if o.phase==Phase.UNKNOWN:
                    continue  # query is scheduled by adapter; never resend or free capacity
                same=(wanted is not None and wanted.price==o.price and wanted.reduce_only==o.reduce_only
                      and o.remaining<=wanted.qty and safe)
                forced=(wanted is None or not safe or o.cancel_wanted or
                        (wanted is not None and (wanted.reduce_only!=o.reduce_only or o.remaining>wanted.qty)))
                if not same and (forced or ns-o.created_ns>=self.config.min_quote_lifetime_ns):
                    cmd=self.cancel(o)
                    if cmd:commands.append(cmd)
                elif o.cancel_wanted:
                    cmd=self.cancel(o)
                    if cmd:commands.append(cmd)
                continue  # no replacement until terminal ACK
            if wanted is None:continue
            while self.new_times and self.new_times[0]<=ns-60_000_000_000:self.new_times.popleft()
            if len(self.new_times)>=self.config.max_new_60s:
                self.stats['rate_limited']+=1;continue
            qty=floor_step(min(wanted.qty,self.capacity(side,wanted.price,mark)),self.rules.step)
            if wanted.reduce_only:
                # Direction is insufficient: a reducing order may not flip flat.
                if side*self.ledger.position>=0:continue
                qty=min(qty,abs(self.ledger.position)-self.pending(side))
            if not self.rules.valid(wanted.price,qty,mark,wanted.reduce_only):continue
            margin=abs(self.ledger.position+side*qty)*max(mark,wanted.price)
            if side*self.ledger.position>=0 and margin>self.ledger.equity(mark):continue
            self.serial+=1
            cid=f'{self.prefix}-{self.serial}'
            if len(cid)>36:raise ValueError('CID exhausted')
            order=Working(cid,self.serial,side,wanted.price,qty,ns,wanted.reduce_only)
            self.active[cid]=order
            self.new_times.append(ns)
            self.stats['new_requests']+=1
            self.emit('new_intent',ns,order=asdict(order),reason=wanted.reason)
            commands.append(Command('new',cid,side,wanted.price,qty,wanted.reduce_only))
        self.assert_invariants(mark)
        return commands

    def uncertain(self,cid,reason):
        o=self.find(cid)
        if o and o.phase not in TERMINAL:o.phase=Phase.UNKNOWN
        self.reconciled=False
        self.stats['unknown_requests']+=1
        self.emit('unknown',self.clock_ns or 0,cid=cid,reason=reason)

    def update(self,cid,status,cumulative,cum_quote,ns,exchange_id=0,exchange_ms=0):
        self.clock(ns)
        o=self.find(cid)
        if o is None:
            if cid.startswith(self.prefix+'-'):
                self.halt_reason='unknown_owned_order';raise ValueError(self.halt_reason)
            return None
        cumulative,cum_quote=D(cumulative),D(cum_quote)
        if (not cumulative.is_finite() or not cum_quote.is_finite() or cumulative<0 or cumulative>o.qty
                or cumulative%self.rules.step or cum_quote<0):
            self.halt_reason='invalid_execution';raise ValueError(self.halt_reason)
        mapped={'NEW':Phase.LIVE,'PARTIALLY_FILLED':Phase.PARTIAL,'FILLED':Phase.FILLED,
                'CANCELED':Phase.CANCELED,'EXPIRED':Phase.EXPIRED,'EXPIRED_IN_MATCH':Phase.EXPIRED,'REJECTED':Phase.REJECTED}
        if status not in mapped:raise ValueError('unsupported order status '+status)
        if cumulative<o.filled:
            self.stats['stale_updates']+=1;return None
        delta=cumulative-o.filled
        was_terminal=o.phase in TERMINAL
        fill=None
        if delta>0:
            price=(cum_quote-o.quote_filled)/delta
            # This system only submits passive limit orders. Improved/taker or
            # amended executions require an explicit different order contract.
            if abs(price-o.price)>self.rules.tick/D(100):
                self.halt_reason='unexpected_execution_price';raise ValueError(self.halt_reason)
            realized,fee=self.ledger.fill(o.side,delta,o.price,self.config.as_config.maker_fee)
            self.cash_flow-=o.side*delta*o.price+fee
            o.filled=cumulative;o.quote_filled=cumulative*o.price
            self.stats['fill_events']+=1
            self.stats['partial_fill_events']+=cumulative<o.qty
            self.max_position=max(self.max_position,abs(self.ledger.position))
            fill=dict(cid=cid,side=o.side,price=o.price,qty=delta,fee=fee,realized=realized,ns=ns)
            self.emit('fill',ns,**{k:v for k,v in fill.items() if k!='ns'})
            if was_terminal:self.halt_reason='late_execution_after_terminal'
        new_phase=mapped[status]
        if new_phase==Phase.LIVE and o.filled>0:new_phase=Phase.PARTIAL
        if new_phase==Phase.FILLED and o.filled!=o.qty:
            self.halt_reason='filled_status_quantity_mismatch';raise ValueError(self.halt_reason)
        if not was_terminal:
            # An ACK cannot erase an already requested cancel.
            if o.phase==Phase.PENDING_CANCEL and new_phase in (Phase.LIVE,Phase.PARTIAL):pass
            else:o.phase=new_phase
        if o.filled==o.qty:o.phase=Phase.FILLED
        if exchange_id:o.exchange_id=int(exchange_id)
        o.last_exchange_ms=max(o.last_exchange_ms,int(exchange_ms))
        if o.phase in TERMINAL:
            self.active.pop(cid,None)
            self.recent[cid]=o;self.recent.move_to_end(cid)
            while len(self.recent)>2048:self.recent.popitem(last=False)
            if not was_terminal:self.stats['terminal_'+o.phase.value]+=1
        self.assert_invariants(o.price)
        return fill

    def actual_fee(self,cid,trade_id,qty,fee,fee_asset,quote_asset,ns):
        """Replace the estimate for THIS receipt, even when REST booked the fill first."""
        o=self.find(cid)
        if o is None:raise ValueError('fee for unknown order')
        self.clock(ns)
        key=str(trade_id)
        qty,fee=D(qty),D(fee)
        if key in o.fee_receipts:
            recorded=o.fee_receipts[key]
            if (D(recorded[0]),D(recorded[1]))!=(qty,fee):
                self.halt_reason='conflicting_fee_receipt';raise ValueError(self.halt_reason)
            return
        if not qty.is_finite() or qty<=0 or not fee.is_finite():raise ValueError('fee values')
        if fee!=0 and fee_asset!=quote_asset:
            self.halt_reason='unsupported_commission_asset';raise ValueError(self.halt_reason)
        if o.actual_fee_qty+qty>o.filled:
            self.halt_reason='fee_receipt_ahead_of_fills';raise ValueError(self.halt_reason)
        estimate=qty*o.price*self.config.as_config.maker_fee
        adjustment=fee-estimate
        self.ledger.fees+=adjustment;self.ledger.wallet-=adjustment;self.cash_flow-=adjustment
        o.fee_receipts[key]=(str(qty),str(fee));o.actual_fee_qty+=qty
        self.stats['actual_fee_receipts']+=1
        self.emit('fee',ns,cid=cid,trade_id=key,fee=fee,adjustment=adjustment)

    def funding(self,amount):
        amount=D(amount)
        if not amount.is_finite():raise ValueError('funding amount')
        self.ledger.wallet+=amount;self.ledger.funding+=amount;self.cash_flow+=amount

    def risk(self,mark,ns):
        self.clock(ns)
        if self.ledger.equity(mark)<=0:self.halt_reason='equity_depleted'
        config=dict(daily_loss_limit=str(self.config.daily_loss),margin_leverage='1')
        result=self.ledger.risk(mark,ns//1_000_000,config)
        if result!='ok':self.halt_reason=result
        if self.ledger.max_drawdown>=self.config.drawdown_limit:self.halt_reason='drawdown_limit'
        return self.halt_reason

    def assert_invariants(self,mark):
        if self.ledger.position.copy_abs()>self.config.as_config.position_cap:
            self.halt_reason='position_cap_breach';raise AssertionError(self.halt_reason)
        for side in (1,-1):
            worst=self.ledger.position+side*self.pending(side)
            if abs(worst)>self.config.as_config.position_cap:
                self.halt_reason='outstanding_cap_breach';raise AssertionError(self.halt_reason)
        for o in self.active.values():
            if o.remaining<=0 or o.remaining%self.rules.step or o.price%self.rules.tick:
                raise AssertionError('order grid or remaining')
        residual=self.ledger.equity(mark)-self.ledger.initial-(self.cash_flow+self.ledger.position*mark)
        if abs(residual)>D('1e-16'):raise AssertionError('cash flow mismatch')

    def dump(self):
        return dict(prefix=self.prefix,serial=self.serial,ledger=asdict(self.ledger),cash_flow=self.cash_flow,
                    active=[asdict(o) for o in self.active.values()],recent=[asdict(o) for o in self.recent.values()],
                    halt_reason=self.halt_reason,reconciled=self.reconciled,stats=dict(self.stats),
                    max_position=self.max_position,new_times=list(self.new_times))

    def restore(self,data):
        self.prefix=data['prefix'];self.serial=int(data['serial'])
        self.ledger=Ledger.restore(data['ledger']);self.cash_flow=D(data['cash_flow'])
        def row(values):
            d=dict(values)
            for key in ('price','qty','filled','quote_filled','actual_fee_qty'):d[key]=D(d[key])
            d['phase']=Phase(d['phase']);return Working(**d)
        self.active={o.cid:o for o in map(row,data['active'])}
        self.recent=OrderedDict((o.cid,o) for o in map(row,data.get('recent',[])))
        self.halt_reason=data.get('halt_reason','');self.reconciled=False
        self.stats.update(data.get('stats',{}))
        self.max_position=D(data.get('max_position',str(abs(self.ledger.position))))
        self.new_times=deque(int(t) for t in data.get('new_times',[]))
        for o in self.active.values():o.phase=Phase.UNKNOWN
