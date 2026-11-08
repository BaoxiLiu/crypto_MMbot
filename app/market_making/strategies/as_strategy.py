"""Original NativeAS/paper quote policy on the certified HFT execution engine.

The archive has L2, not a separate bookTicker stream. Its reconstructed best
price/quantity supplies the L1 inputs; this channel difference is explicit.
NativeAS, S1, SC0, microprice, keeper and quote-retention rules are reused.
All fills, inventory, fees, funding and transport belong to the common HFT OMS.
"""
from collections import Counter,deque
from copy import deepcopy
from decimal import Decimal as D
from types import SimpleNamespace

from ..as_quote import NativeAS
from ..core import Account
from ..policy import as_quotes,microprice
from ..s1_config import make_two_strategy_config
from ..spread_control import SpreadController
from ..vol_floor import RollingVolatility
from app.market_making.contracts import Desired, Phase
from .base import QuoteStrategy as Strategy

SOURCES=('app/market_making/contracts.py','app/market_making/strategies/as_strategy.py','app/market_making/strategies/base.py','app/market_making/as_quote.py',
         'app/market_making/policy.py','app/market_making/quote_policy.py',
         'app/market_making/s1_config.py','app/market_making/vol_floor.py',
         'app/market_making/spread_control.py','app/market_making/core.py')


def validate_settings(p):
    optional={'min_half_spread_ticks','volatility_adjustment','markout_adjustment'}
    if set(p)-optional!={'mode','paper_config','trade_start_ns','trade_end_ns'}:
        raise ValueError('Native paper strategy requires explicit mode/config/trading window')
    if p['mode'] not in ('as_base','paper_s1_sc0'):raise ValueError('Unknown native paper mode')
    if any(type(p[k]) is not int for k in ('trade_start_ns','trade_end_ns')) or not 0<p['trade_start_ns']<p['trade_end_ns']:
        raise ValueError('Invalid native trading window')
    if not isinstance(p['paper_config'],dict):raise ValueError('Explicit paper config required')
    config=make_two_strategy_config(p['paper_config'])
    for key in ('volatility_adjustment','markout_adjustment'):
        if key in p and type(p[key]) is not bool:raise ValueError(key+' must be boolean')
        if p['mode']=='as_base' and p.get(key,False):
            raise ValueError('Original AS cannot enable improved-policy modules')
    if 'min_half_spread_ticks' in p:
        floor=p['min_half_spread_ticks']
        if p['mode']!='paper_s1_sc0' or type(floor) is not int or not 0<=floor<=config['as_floor_cap_ticks']:
            raise ValueError('Static half-spread override requires paper_s1_sc0 and a valid integer tick count')


class IntentAccount(Account):
    """Execute original quote policy, capturing intentions instead of paper fills."""
    def __init__(self,name,cfg,rules):
        self.targets={};self.urgent=set();self.counts=Counter()
        super().__init__(name,cfg,rules,self.capture)

    def capture(self,name,kind,now,fields):
        self.counts[kind]+=1
        if kind=='order_sent':
            self.targets[fields['side']]=Desired(fields['side'],fields['price'],fields['remaining'],fields['reduce_only'],'native_paper')

    def cancel(self,side,now_ms,force=False,reason='strategy_requote'):
        current=self.orders.get(side)
        if current and not current.cancel_ms and self._action_allowed('cancel',now_ms,force=force):
            self.targets.pop(side,None);self.urgent.add(side);self.counts['cancel_'+reason]+=1

    def allowed(self,*args):
        # The common OMS checks pending-side capacity, collateral and all rules
        # again at dispatch, after simulated computation and private responses.
        return True

    def decide(self,bid,ask,position,now,book,fv,working,equity,guard_return=None,mark=None):
        self.orders={};self.targets={};self.urgent=set()
        self.ledger=SimpleNamespace(position=position,risk=lambda *args:'ok',equity=lambda mark:equity)
        for o in working:
            if o.side in self.orders:raise ValueError('Multiple live native orders on one side')
            pending_cancel=o.phase==Phase.PENDING_CANCEL or o.cancel_wanted
            self.orders[o.side]=SimpleNamespace(oid=o.cid,side=o.side,price=o.price,qty=o.qty,
                remaining=o.remaining,reduce_only=o.reduce_only,purpose='market_make',
                cancel_ms=1 if pending_cancel else 0,active=o.phase in (Phase.LIVE,Phase.PARTIAL))
            if not pending_cancel:
                self.targets[o.side]=Desired(o.side,o.price,o.remaining,o.reduce_only,'native_retain')
        super().quote(bid,ask,book.mid if mark is None else mark,now,book,fv=fv,guard_return=guard_return)
        return dict(self.targets)


class ASStrategy(Strategy):
    additional_sources=SOURCES

    def __init__(self,rules,config,factors,settings,symbol):
        super().__init__(rules,config,factors);validate_settings(settings)
        self.settings=deepcopy(settings);self.symbol=symbol
        self.trading_window_ns=(settings['trade_start_ns'],settings['trade_end_ns'])
        self.working=[];self.account_equity=config.initial_cash
        self.account_mark=None
        self.policy_totals=Counter()
        self.floor_samples=[];self.last_floor_sample_ns=0
        self._reset_native()

    def _reset_native(self):
        if hasattr(self,'account'):self.policy_totals.update(self.account.counts)
        p=self.settings;cfg=make_two_strategy_config(p['paper_config'])
        cfg.update(symbol=self.symbol,max_position_qty=str(self.config.as_config.position_cap),
                   order_notional=str(self.config.as_config.order_notional),
                   max_position_notional=str(self.config.max_notional),
                   initial_collateral=str(self.config.initial_cash),
                   maker_fee=str(self.config.as_config.maker_fee))
        account=next(c for c in cfg.pop('accounts') if c['name']==p['mode'])
        cfg.update({k:v for k,v in account.items() if k!='name'})
        if 'min_half_spread_ticks' in p:
            cfg['as_min_half_spread_ticks']=str(p['min_half_spread_ticks'])
        if 'volatility_adjustment' in p:cfg['s1_enabled']=p['volatility_adjustment']
        if 'markout_adjustment' in p:cfg['spread_control_enabled']=p['markout_adjustment']
        self.paper_config=cfg;self.model=NativeAS(cfg,self.rules)
        self.account=IntentAccount(p['mode'],cfg,self.rules)
        self.controller=SpreadController(self.account) if cfg['spread_control_enabled'] else None
        self.volatility=RollingVolatility(cfg,self.rules.tick) if cfg['s1_enabled'] else None
        self.boot_ns=None;self.micro=None;self.urgent_cancel_sides=set()
        self.history=deque();self.last_history_second=None
        original_trade=type(self.obs).trade.__get__(self.obs,type(self.obs))
        def trade(ns,trade_id,side,price,qty):
            accepted=original_trade(ns,trade_id,side,price,qty)
            if accepted:self.model.trade(side,price)
            return accepted
        self.obs.trade=trade

    @property
    def identity(self):
        import hashlib,sys
        from pathlib import Path
        dependencies={}
        for obj in (self.model,self.model.vol):
            module=sys.modules[type(obj).__module__]
            path=Path(module.__file__)
            dependencies[type(obj).__module__]=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        return dict(type='paper_native_as',version=1,params=deepcopy(self.settings),symbol=self.symbol,
          effective_paper_quote_config=deepcopy(self.paper_config),
          implementation_dependencies=dependencies,
          implementation='project AS (Hummingbot conventions) + Account.quote/S1/SC0',
          market_channel='L2-derived BBO quantities replace separate bookTicker; not tick-for-tick live equivalence',
          execution='common HFT queue/OMS/latency/risk; original paper simulator is not used',
          feedback='completed 1s markout, <=250ms lag; retain matured evidence within original 8h age limit; pending samples censored at gaps',
          warmup='no trades before registered window; initial 30min warmup; original NativeAS/S1 missing-observation handling after gaps')

    def reset_market(self):
        # A reconnect must not silently restart the original batch or throw
        # away already observed S1/SC0 evidence. NativeAS.sample handles gaps
        # in its own samples; S1 advances its causal grid and stale/degraded
        # state. Kernel censors pending, not-yet-observed markouts separately.
        super().reset_market();self.micro=None

    def on_top_of_book(self,ns,bid,bid_qty,ask,ask_qty):
        self.micro=microprice(bid,bid_qty,ask,ask_qty)

    def before_plan(self,working,equity,mark=None):
        self.working=working;self.account_equity=equity;self.account_mark=mark

    def book(self,ns,bid,ask):
        # NativeAS has its own inventory normalization; canonical AS's
        # reference-order quantity is not an input and must not reject BTC.
        self.obs.book(ns,bid,ask)
        if self.boot_ns is None:self.boot_ns=ns
        if ns//1_000_000_000!=self.last_history_second:
            self.history.append((ns,self.obs.mid));self.last_history_second=ns//1_000_000_000
            cutoff=ns-self.paper_config['drop_window_seconds']*1_000_000_000
            while len(self.history)>2 and self.history[1][0]<=cutoff:self.history.popleft()
        if self.volatility:
            self.volatility.on_ticker({'b':str(bid),'a':str(ask)},ns//1_000_000)
        self.model.sample(self.obs.mid,ns//1_000_000)

    def on_markout(self,f):
        if not self.controller or f['horizon_ms']!=1000:return
        if f['observed_ns']-f['target_ns']>250_000_000:
            self.counts['late_sc_feedback']+=1;return
        fill=f['fill']
        changed=self.controller.feed(D(fill['side'])*fill['qty']*(f['mid']-fill['price']),
            f['observed_ns']//1_000_000,fill_ms=fill['ns']//1_000_000,qty=fill['qty'],
            notional=fill['qty']*fill['price'],oid=fill['cid'],fill_key=f['fill_id'])
        if changed:self.counts['sc_adjustments']+=1

    def desired(self,position,ns):
        self.urgent_cancel_sides=set();p=self.settings;now=ns//1_000_000
        if self.volatility:self.account.update_volatility(self.volatility.snapshot(now),now)
        if (self.boot_ns is None or ns-self.boot_ns<self.paper_config['warmup_seconds']*1_000_000_000
                or not p['trade_start_ns']<=ns<p['trade_end_ns']):
            self.counts['context_or_warmup']+=1;return {}
        if self.volatility and (self.account.vol_degraded or self.account.vol_last_valid_ms is None):
            self.counts['vol_unavailable']+=1;return {}
        if not self.model.ready(now):self.counts['native_not_ready']+=1;return {}
        book=SimpleNamespace(mid=self.obs.mid,bid=self.obs.bid,ask=self.obs.ask)
        try:
            bid,ask,*_=as_quotes(self.model,book,position,now,self.paper_config,self.rules,
                               micro=self.micro,floor_ticks=self.account.floor_state()['as_floor_ticks'])
            cutoff=ns-self.paper_config['drop_window_seconds']*1_000_000_000
            reference=self.history[0][1] if self.history and self.history[0][0]<=cutoff else None
            change=self.obs.mid/reference-1 if reference else None
            targets=self.account.decide(bid,ask,position,now,book,self.micro if self.micro is not None else self.obs.mid,
                                       self.working,self.account_equity,guard_return=change,mark=self.account_mark)
        except ValueError:
            self.counts['native_quote_unavailable']+=1;return {}
        self.urgent_cancel_sides=set(self.account.urgent)
        self.last=dict(reason='native_quoting',**self.account.floor_state())
        if ns-self.last_floor_sample_ns>=60_000_000_000:
            self.floor_samples.append(dict(ns=ns,**self.account.floor_state()))
            self.last_floor_sample_ns=ns
        self.counts['native_decisions']+=1
        return targets

    def report_diagnostics(self,ns):
        return dict(native=self.model.report(ns//1_000_000),policy_counts=dict(self.policy_totals+self.account.counts),
                    floor_samples=self.floor_samples,final_floor=self.account.floor_state(),
                    note='Native estimator is the final snapshot; counts aggregate the run. Policy floor ticks, not actual AS quote width; sampled every minute while quoting')
