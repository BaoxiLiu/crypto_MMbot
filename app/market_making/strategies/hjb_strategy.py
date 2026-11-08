"""Paper Appendix C adaptation on the common OMS.

Explicit local assumptions are frozen in experiment configs. Public trade reach
estimates arrival rates; it does not replace the replay's execution queue model.
"""
from collections import deque
from copy import deepcopy
from decimal import Decimal as D, InvalidOperation
from dataclasses import replace
import math

import numpy as np

from .base import QuoteStrategy as Strategy
from app.market_making.contracts import Desired
from ..core import floor_step
from .fb_hjb_math import solve_hjb, fit_objective

S = 1_000_000_000
SOURCES = ('app/market_making/contracts.py','app/market_making/strategies/hjb_strategy.py',
           'app/market_making/strategies/base.py',
           'app/market_making/strategies/fb_hjb_math.py',
           'app/market_making/fusion/kernel.py')
DEFAULTS = dict(adaptive=True, step_seconds=1., horizon_steps=60,
                market_window_seconds=1800., reward_window_seconds=1800.,
                half_life_seconds=300., ridge=.01, mixing=.25,
                minimum_labels=30, fit_interval_seconds=10.,
                prior=[1., 0., -.001, -1.],
                distances_ticks=[0., 1., 2., 4., 8., 16., 32.],
                static_floor_ticks=0., max_target_units=10.,
                minimum_rate=.0001, maximum_rate=100.,
                minimum_decay=.001, maximum_decay=10.,
                maximum_feedback_lag_ms=250)


def validate_base_settings(params):
    if set(params) != set(DEFAULTS) | {'trade_start_ns', 'trade_end_ns'}:
        raise ValueError('FB HJB requires complete explicit settings')
    if type(params['adaptive']) is not bool:
        raise ValueError('adaptive must be boolean')
    if not 0 < params['trade_start_ns'] < params['trade_end_ns']:
        raise ValueError('Invalid trading window')
    for key in ('horizon_steps', 'minimum_labels'):
        if type(params[key]) is not int or params[key] <= 0:
            raise ValueError('Invalid positive integer: '+key)
    for key in ('step_seconds', 'market_window_seconds', 'reward_window_seconds',
                'half_life_seconds', 'ridge', 'fit_interval_seconds',
                'minimum_rate', 'maximum_rate', 'minimum_decay', 'maximum_decay',
                'max_target_units'):
        if not math.isfinite(params[key]) or params[key] <= 0:
            raise ValueError('Invalid positive setting: '+key)
    if not 0 <= params['mixing'] <= 1 or params['static_floor_ticks'] < 0:
        raise ValueError('Invalid mixing/floor')
    prior = np.asarray(params['prior'], dtype=float)
    if prior.shape != (4,) or not np.isfinite(prior).all() or prior[0] != 1 or prior[2] >= 0 or prior[3] > 0:
        raise ValueError('Unsafe prior')
    distances = params['distances_ticks']
    if not distances or distances != sorted(set(distances)) or any(not math.isfinite(v) or v < 0 for v in distances):
        raise ValueError('Invalid quote action distances')


def validate_settings(params):
    settings=deepcopy(params)
    if 'capital_utilization' not in settings:
        raise ValueError('HJB v2 requires explicit capital_utilization: null or a fraction')
    utilization=settings.pop('capital_utilization')
    validate_base_settings(settings)
    if utilization is not None:
        if type(utilization) is bool:raise ValueError('capital_utilization cannot be boolean')
        try:value=D(str(utilization))
        except (InvalidOperation,TypeError) as error:
            raise ValueError('capital_utilization must be a finite fraction') from error
        if not value.is_finite() or not D(0)<value<D(1):
            raise ValueError('capital_utilization must be strictly between zero and one')


def factory_for(params):
    validate_settings(params)
    settings=deepcopy(params)
    return lambda rules,config,factors:HJBStrategy(rules,config,factors,deepcopy(settings))


class HJBStrategy(Strategy):
    additional_sources = SOURCES

    def __init__(self, rules, config, factors, settings, symbol=None):
        super().__init__(rules, config, factors)
        validate_settings(settings)
        self.configured_settings=deepcopy(settings)
        settings=deepcopy(settings)
        utilization=settings.pop('capital_utilization')
        self.utilization=None if utilization is None else D(str(utilization))
        self.original_config=config
        self.budget_equity=self.budget_mark=None
        self.settings = deepcopy(settings)
        self.trading_window_ns = (settings['trade_start_ns'], settings['trade_end_ns'])
        self.position = D(0)
        self.diagnostics = []
        self.pending_rows = {}
        self._reset_adaptation()

    def _reset_adaptation(self):
        self.reaches = deque()
        self.labels = deque()
        self.cost_labels = deque()
        self.objective = np.array(self.settings['prior'], dtype=float)
        self.next_solve = self.next_fit = self.next_observation_ns = 0
        self.warm_ns = None
        self.targets = {}
        self.current_risk_scale = 1.
        self.rates = np.zeros(4)
        self.costs = np.zeros(2)
        self.last_diagnostic = 0

    @property
    def identity(self):
        return dict(type='fb_as_hjb',version=2,params=deepcopy(self.configured_settings),
                    provenance='Appendix C local implementation; optional policy inventory budget')

    def before_plan(self, working, equity, mark):
        self.budget_equity=equity
        self.budget_mark=mark

    def reset_market(self):
        super().reset_market()
        self.pending_rows.clear()
        self._reset_adaptation()

    def book(self, ns, bid, ask):
        super().book(ns, bid, ask)
        if self.warm_ns is None:
            self.warm_ns = ns

    requires_quote_depth = False

    def observation_due(self, ns):
        return self.obs.mid > 0 and ns >= self.next_observation_ns

    def observe_market(self, depth, trades, ns):
        self.next_observation_ns = ns+100_000_000
        tick = float(self.rules.tick)
        mid = float(self.obs.mid)
        for trade in trades:
            side = -int(trade['side'])
            reach = side*(mid-float(trade['price']))/tick
            self.reaches.append((ns, side, reach))
        cutoff = ns-int(self.settings['market_window_seconds']*S)
        while self.reaches and self.reaches[0][0] < cutoff:
            self.reaches.popleft()

    def on_fill(self, fill, position):
        self.position = position
        if self.reference_qty is None or self.obs.mid <= 0:
            return
        quantity = float(fill['qty']/self.reference_qty)
        spread = float(fill['side']*(self.obs.mid-fill['price'])/self.rules.tick)*quantity
        q = float(position/self.reference_qty)
        adverse = self.costs[0 if fill['side'] == 1 else 1]*quantity
        key = (fill['cid'], fill['ns'])
        self.pending_rows.setdefault(key, deque()).append((np.array([spread, q, q*q*self.current_risk_scale, adverse]),
                                  quantity, float(self.obs.mid), fill['side']))
        self.counts['fill_rows'] += 1

    def on_markout(self, completed):
        if completed['horizon_ms'] != 1000:
            return
        fill = completed['fill']
        key = (fill['cid'], fill['ns'])
        pending = self.pending_rows.get(key)
        if not pending:
            return
        row = pending.popleft()
        if not pending:
            del self.pending_rows[key]
        if completed['observed_ns']-completed['target_ns'] > self.settings['maximum_feedback_lag_ms']*1_000_000:
            self.counts['late_labels'] += 1
            return
        features, quantity, initial_mid, side = row
        movement = side*(float(completed['mid'])-initial_mid)/float(self.rules.tick)
        reward = features[0]+quantity*movement
        ns = completed['observed_ns']
        self.labels.append((ns, features, reward))
        self.cost_labels.append((ns, side, max(0., -movement), quantity))
        self.counts['mature_labels'] += 1

    def _market_parameters(self, ns):
        p = self.settings
        duration = max(1., min(p['market_window_seconds'], (ns-self.warm_ns)/S))
        distances = np.asarray(p['distances_ticks'])
        for j, side in enumerate((1, -1)):
            reach = np.array([r[2] for r in self.reaches if r[1] == side])
            rates = np.array([np.count_nonzero(reach >= distance)/duration for distance in distances])
            valid = rates > 0
            if valid.sum() >= 2:
                slope, intercept = np.polyfit(distances[valid], np.log(rates[valid]), 1)
                amplitude = math.exp(float(np.clip(intercept, -20, 20)))
                decay = -slope
            else:
                amplitude = rates[0] if len(rates) else p['minimum_rate']
                decay = 1.
            self.rates[j] = np.clip(amplitude, p['minimum_rate'], p['maximum_rate'])
            self.rates[j+2] = np.clip(decay, p['minimum_decay'], p['maximum_decay'])
            rows = [r for r in self.cost_labels if r[1] == side]
            weight = sum(r[3] for r in rows)
            self.costs[j] = sum(r[2]*r[3] for r in rows)/weight if weight else 0.

    def desired(self, position, ns):
        if self.utilization is None:
            return self._desired_hjb(position,ns)
        start,end=self.trading_window_ns
        if not start<=ns<end or self.budget_equity is None:
            return {}
        mark=self.budget_mark
        if mark is None or mark<=0:
            return {}
        budget_notional=max(D(0),self.budget_equity*self.utilization)
        cap=min(self.original_config.as_config.position_cap,
                self.original_config.max_notional/mark,budget_notional/mark)
        cap=floor_step(cap,self.rules.step)
        if abs(position)>cap:
            side=-1 if position>0 else 1
            price=self.obs.ask if side<0 else self.obs.bid
            qty=floor_step(min(self.reference_qty,abs(position)),self.rules.step)
            self.counts['budget_reduce_only']+=1
            if qty<self.rules.minimum_qty:
                return {}
            return {side:Desired(side,price,qty,True,'capital_budget_reduce')}
        if cap<=0:
            return {}
        self.config=replace(self.original_config,as_config=replace(
            self.original_config.as_config,position_cap=cap))
        raw=self._desired_hjb(position,ns)
        safe={}
        for side, wanted in raw.items():
            price_cap=min(cap,budget_notional/max(mark,wanted.price),
                          self.original_config.max_notional/max(mark,wanted.price))
            qty=floor_step(min(wanted.qty,max(D(0),price_cap-side*position)),self.rules.step)
            if qty>=self.rules.minimum_qty:
                safe[side]=Desired(side,wanted.price,qty,wanted.reduce_only,wanted.reason)
        return safe

    def _desired_hjb(self, position, ns):
        p = self.settings
        if not p['trade_start_ns'] <= ns < p['trade_end_ns']:
            return {}
        if ns < self.next_solve:
            return self.targets
        self.next_solve = ns+int(p['step_seconds']*S)
        cutoff = ns-int(p['reward_window_seconds']*S)
        while self.labels and self.labels[0][0] < cutoff:
            self.labels.popleft()
        while self.cost_labels and self.cost_labels[0][0] < cutoff:
            self.cost_labels.popleft()
        self.current_risk_scale = max(1e-8, self.obs.state(ns).variance*p['step_seconds'])
        if p['adaptive'] and ns >= self.next_fit and len(self.labels) >= p['minimum_labels']:
            weights = [math.exp(-(ns-r[0])/S*math.log(2)/p['half_life_seconds']) for r in self.labels]
            self.objective = fit_objective(
                [r[1] for r in self.labels], [r[2] for r in self.labels], weights,
                p['prior'], p['ridge'], p['mixing'],
                p['max_target_units']*self.current_risk_scale)
            self.next_fit = ns+int(p['fit_interval_seconds']*S)
            self.counts['objective_updates'] += 1
        # Volatility rescales q² at every solve. Re-project the target even
        # between regression updates and when the prior is held unchanged.
        target_limit = -2*self.objective[2]*p['max_target_units']*self.current_risk_scale
        self.objective[1] = float(np.clip(self.objective[1], -target_limit, target_limit))
        self._market_parameters(ns)
        cap = int(self.config.as_config.position_cap/self.reference_qty)
        grid = np.arange(-cap, cap+1, dtype=float)
        distances = [-1.] + [max(p['static_floor_ticks'], d) for d in p['distances_ticks']]
        actions = np.array([(b, a) for b in distances for a in distances])
        choices, _ = solve_hjb(grid, actions, self.rates, self.costs, self.objective,
                              p['horizon_steps'], p['step_seconds'], 1., self.current_risk_scale)
        q = float(position/self.reference_qty)
        index = int(np.clip(round(q)+cap, 0, len(grid)-1))
        bid_distance, ask_distance = actions[choices[index]]
        targets = {}
        for side, distance in ((1, bid_distance), (-1, ask_distance)):
            if distance < 0:
                continue
            price = self.rules.price(self.obs.mid-D(str(side*distance))*self.rules.tick, side)
            price = min(price, self.obs.ask-self.rules.tick) if side == 1 else max(price, self.obs.bid+self.rules.tick)
            if price <= 0:
                continue
            qty = self.reference_qty
            targets[side] = Desired(side, price, qty, False, 'fb_hjb')
        self.targets = targets
        self.last = dict(reason='fb_hjb', objective=self.objective.tolist(),
                         rates=self.rates.tolist(), costs=self.costs.tolist())
        if ns-self.last_diagnostic >= 60*S:
            target = self.objective[1]/(-2*self.objective[2]*self.current_risk_scale)
            self.diagnostics.append(dict(ns=ns, q=q, target=target, **self.last))
            self.last_diagnostic = ns
        return targets

    def report_diagnostics(self, ns):
        result=dict(method='FB-AS HJB Appendix C adaptation', counts=dict(self.counts),
                    samples=self.diagnostics, assumptions=self.settings)
        if self.utilization is not None:
            result['capital_utilization']=str(self.utilization)
            result['budget_definition']='policy inventory planning and quote intent cap; common OMS guard remains authoritative'
        return result
