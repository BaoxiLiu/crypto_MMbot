"""NH quote selection with the project's common execution and risk ledger."""
import math
from decimal import Decimal as D

import numpy as np

from ..core import floor_step
from app.market_making.contracts import Desired
from .base import QuoteStrategy as Strategy
S = 1_000_000_000
STEP = 100_000_000

SOURCES = ('app/market_making/contracts.py','app/market_making/strategies/nh_strategy.py',
           'app/market_making/strategies/base.py',
           'app/market_making/fusion/market_data.py')+tuple(
    'app/market_making/strategies/nh_quote_value/'+name+'.py'
    for name in ('features','quote_policy','value_features'))


def validate_settings(params):
    if set(params) != {'alpha','value','trade_start_ns','trade_end_ns'}:
        raise ValueError('NH requires both fitted models and an explicit trading window')
    cutoff = params['trade_start_ns']
    if cutoff >= params['trade_end_ns']:
        raise ValueError('Invalid NH trading window')
    for name, size, vectors in (
        ('alpha',23,('mean','scale','beta')),
        ('value',28,('mean','scale','fill_beta','edge_beta')),
    ):
        model = params[name]
        if not model['last_label_ns'] < model['deploy_ns'] == cutoff:
            raise ValueError('NH model labels must precede first deployment')
        for key in vectors:
            x = np.asarray(model[key],dtype=float)
            if x.shape != (size,) or not np.isfinite(x).all() or (key=='scale' and np.any(x<=0)):
                raise ValueError(f'Invalid NH {name}.{key}')
    value = params['value']
    if len(value['supported'])!=6 or len(value['fraction'])!=6:
        raise ValueError('Invalid NH value support arrays')
    if value['queue']!='cancel_50_50':
        raise ValueError('NH requires common 50/50 probe labels')


def validate_model_identity(params, symbol, manifest_sha256=None):
    for name in ('alpha', 'value'):
        if params[name]['symbol'] != symbol:
            raise ValueError('NH model symbol mismatch')
        if manifest_sha256 is not None and params[name]['dataset_manifest_sha256'] != manifest_sha256:
            raise ValueError('NH model training input does not match '+symbol)


class NHStrategy(Strategy):
    additional_sources = SOURCES

    def __init__(self,rules,config,factors,params,symbol):
        super().__init__(rules,config,factors)
        self.params = params
        self.trading_window_ns = (params['trade_start_ns'],params['trade_end_ns'])
        self.alpha = params['alpha']
        self.value = params['value']
        if self.alpha['symbol']!=symbol or self.value['symbol']!=symbol:
            raise ValueError('NH model symbol mismatch')
        self.a = tuple(np.asarray(self.alpha[k]) for k in ('mean','scale','beta'))
        self.v = tuple(np.asarray(self.value[k]) if isinstance(self.value[k],list) else self.value[k]
                       for k in ('mean','scale','fill_beta','fill_intercept','edge_beta','edge_intercept','supported','fraction'))
        self.notional = D(str(self.value['order_notional']))
        self.working = {}
        self.reset_market()

    @property
    def identity(self):
        return dict(type='nh_quote_value',version=1,params=self.params,
                    execution_policy='common OMS, common loss/settlement policy, 100ms causal sampling, initial weights frozen')

    def book(self,ns,bid,ask):
        self.obs.book(ns,bid,ask)

    def reset_market(self):
        super().reset_market()
        from .nh_quote_value.features import new_state
        self.ring = new_state()
        self.next_sample_ns = 0
        self.sample_ns = self.warm_start = None
        self.old = None
        self.buy = self.sell = self.ofi = self.ema = self.variance = 0.
        self.targets = {}
        self.widths = {}
        self.urgent_cancel_sides = set()

    def before_plan(self,working,equity,mark):
        self.working = {order.side:order for order in working}

    requires_quote_depth = True

    def next_deadline_ns(self):
        return self.next_sample_ns

    def observation_due(self, ns):
        return ns >= self.next_sample_ns

    def observe_market(self, depth, trades, ns):
        self.next_sample_ns = (ns//STEP+1)*STEP
        from .nh_quote_value.features import feature, ingest, window_values
        ingest(trades,*self.ring)
        decay = math.exp(-.1)
        self.buy *= decay
        self.sell *= decay
        for trade in trades:
            if trade['side'] > 0:
                self.buy += float(trade['qty'])
            else:
                self.sell += float(trade['qty'])
        bid,ask = depth.best_bid,depth.best_ask
        if not (math.isfinite(bid) and math.isfinite(ask) and 0<bid<ask):
            return
        mid = (bid+ask)*.5
        bq,aq = depth.bid_qty_at_tick(depth.best_bid_tick),depth.ask_qty_at_tick(depth.best_ask_tick)
        if self.old is not None:
            pb,pa,pbq,paq,pm = self.old
            var_decay = math.exp(-.1/60)
            self.variance = var_decay*self.variance+(1-var_decay)*math.log(mid/pm)**2/.1
            e = (bq if bid>=pb else 0)-(pbq if bid<=pb else 0)
            e -= (aq if ask<=pa else 0)-(paq if ask>=pa else 0)
            self.ofi = decay*self.ofi+e
        self.old = bid,ask,bq,aq,mid
        self.ema = .99*self.ema+.01*(bq+aq)*.5
        self.ti = (self.buy-self.sell)/max(self.buy+self.sell,1e-12)
        self.normalized_ofi = self.ofi/max(self.ema,1e-12)
        self.x = feature(.5*(ask-bid)/mid*1e4*(bq-aq)/max(bq+aq,1e-12),
                         self.normalized_ofi,self.ti,window_values(ns,*self.ring))
        self.depth = depth
        self.sample_ns = ns
        if self.warm_start is None:
            self.warm_start = ns

    def desired(self,position,ns):
        start,end = self.trading_window_ns
        if not start<=ns<end or self.warm_start is None or ns-self.warm_start<60*S:
            return {}
        if self.sample_ns != ns:
            return self.targets
        from .nh_quote_value.features import predict
        from .nh_quote_value.quote_policy import select_quote
        self.urgent_cancel_sides.clear()
        bid,ask,_,_,mid = self.old
        tick,lot = float(self.rules.tick),float(self.rules.step)
        cap = min(float(self.config.as_config.position_cap),float(self.config.max_notional)/mid)
        if cap<=0:
            return {}
        alpha = min(5.,max(-5.,predict(self.x,*self.a,self.alpha['intercept'])))
        center = mid*(1+alpha/1e4-.0002*float(position)/cap)
        vol = math.sqrt(self.variance)
        half = max(mid*.0003,2*tick,.5*vol*mid)
        targets = {}
        for side in (1,-1):
            if (side*self.ti<-.6 and side*self.normalized_ofi<-.5):
                self.counts['flow_veto'] += 1
                continue
            existing = self.working.get(side)
            if existing:
                if ns-existing.created_ns>=5*S:
                    self.urgent_cancel_sides.add(side)
                    continue
                width = self.widths.get(side,half)
                price = self.rules.price(D(str(center-side*width)),side)
                price = min(price,D(str(ask))-self.rules.tick) if side==1 else max(price,D(str(bid))+self.rules.tick)
                if abs(price-existing.price)<2*self.rules.tick:
                    price=existing.price
                targets[side] = Desired(side,price,existing.remaining,reason='nh_retained_width')
                continue
            room = cap-side*float(position)
            if room<=0:
                continue
            px,width,allowed,_,_,_,_,reason = select_quote(
                self.depth,self.x,side,float(position),room,center,half,mid,bid,ask,tick,lot,vol*1e4,
                *self.v,2,float(self.notional))
            self.counts[f'quote_choice_{int(reason)}'] += 1
            if not allowed:
                continue
            price = self.rules.price(D(str(px)),side)
            if price<=0:
                continue
            qty = floor_step(min(self.notional/price,D(str(room))),self.rules.step)
            if qty>=self.rules.minimum_qty and qty*price>=self.rules.minimum_notional:
                targets[side] = Desired(side,price,qty,reason='nh_quote_value')
                self.widths[side] = width
        self.targets = targets
        return targets
