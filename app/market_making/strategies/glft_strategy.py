"""GLFT asymptotic single-level quotes on the existing causal AS estimator.

Reference: nkaz001/hftbacktest GLFT notebook, commit
66b2e7537f0005b1bf691373017160566814ee4a; independently implemented formulas.
Distances/volatility are ticks and ticks/sqrt(second); inventory is fixed
base-asset units. No grid orders, forecast overlay or rebate from the example.
"""
from copy import deepcopy
from decimal import Decimal as D
import math
import numpy as np
from scipy.linalg import eigvalsh_tridiagonal

from ..as_quote import NativeAS
from .as_strategy import ASStrategy, SOURCES as AS_SOURCES
from .as_strategy import validate_settings as validate_as

SOURCES=(*AS_SOURCES,'app/market_making/strategies/glft_strategy.py')


def coefficients(gamma, xi, arrival, decay):
    """Delta=1 GLFT coefficients, evaluated in log space for stability."""
    values=(gamma,xi,arrival,decay)
    if any(not math.isfinite(v) or v<=0 for v in values):
        raise ValueError('GLFT parameters must be finite and positive')
    z=xi/decay
    if not math.isfinite(z) or z<=0:raise ValueError('Unrepresentable GLFT risk/decay ratio')
    log_term=math.log1p(z)
    c1=log_term/xi
    log_c2=(math.log(gamma)-math.log(2)-math.log(arrival)-math.log(decay)
            +log_term+log_term/z)/2
    try:c2=math.exp(log_c2)
    except OverflowError as error:raise ValueError('GLFT coefficient overflow') from error
    if not math.isfinite(c1+c2):raise ValueError('Invalid GLFT coefficients')
    return c1,c2


def validate_settings(p):
    optional = {'inventory_solution', 'reference_price', 'markout_feedback'}
    if set(p)-optional!={'paper_config','trade_start_ns','trade_end_ns','gamma','xi','inventory_units'}:
        raise ValueError('GLFT requires explicit estimator, trading window, gamma, xi and inventory units')
    validate_as(dict(mode='as_base',**{k:p[k] for k in ('paper_config','trade_start_ns','trade_end_ns')}))
    for key in ('gamma','xi'):
        if isinstance(p[key],bool) or not D(str(p[key])).is_finite() or D(str(p[key]))<=0:
            raise ValueError('Invalid GLFT '+key)
    if not isinstance(p['inventory_units'],dict) or not p['inventory_units']:
        raise ValueError('Explicit per-symbol fixed inventory units required')
    for value in p['inventory_units'].values():
        if isinstance(value,bool) or not D(str(value)).is_finite() or D(str(value))<=0:
            raise ValueError('Invalid GLFT inventory unit')
    solution = p.get('inventory_solution', 'quadratic')
    if solution not in ('quadratic', 'spectral') or (solution == 'spectral' and D(str(p['gamma'])) != D(str(p['xi']))):
        raise ValueError('Spectral GLFT requires gamma equal to xi')
    if p.get('reference_price', 'mid') not in ('mid', 'weighted_mid'):
        raise ValueError('Unknown GLFT reference price')
    if 'markout_feedback' in p:
        MarkoutFeedback.validate(p['markout_feedback'])


def spectral_log_values(cap, risk_ratio):
    """Log of the positive stationary eigenvector on 0..cap, relative to v(0).

    The matrix is divided by eta, so its off-diagonals are -1. Reconstruct
    adjacent ratios from the boundary to avoid underflow in tiny entries.
    """
    if not math.isfinite(risk_ratio*cap*cap) or risk_ratio < 0:
        raise ValueError('Unrepresentable GLFT spectral inventory risk')
    states = np.arange(-cap, cap + 1, dtype=float)
    diagonal = risk_ratio * states**2
    eigenvalue = eigvalsh_tridiagonal(diagonal, np.full(2*cap, -1.0),
                                     select='i', select_range=(0, 0))[0]
    log_ratios = np.empty(cap)
    ratio = 0.
    for q in range(cap, 0, -1):
        denominator = diagonal[cap+q] - eigenvalue - ratio
        if denominator <= 0 or not math.isfinite(denominator):
            raise ValueError('Invalid GLFT spectral eigenvector ratio')
        ratio = 1/denominator
        log_ratios[q-1] = math.log(ratio)
    return np.concatenate(([0.], np.cumsum(log_ratios)))


class MarkoutFeedback:
    @staticmethod
    def validate(settings):
        integers = ('orders', 'minimum_orders', 'half_life_ms', 'max_age_ms')
        if not isinstance(settings, dict) or set(settings) != {*integers, 'deadband_bp', 'cap_bp'}:
            raise ValueError('Explicit GLFT markout feedback settings required')
        if (any(type(settings[k]) is not int or settings[k] <= 0 for k in integers)
                or settings['minimum_orders'] > settings['orders']
                or settings['half_life_ms'] > settings['max_age_ms']):
            raise ValueError('Invalid GLFT markout feedback window')
        for key in ('deadband_bp', 'cap_bp'):
            value = settings[key]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError('Invalid GLFT markout feedback bound')

    def __init__(self, settings):
        self.settings = settings
        self.sides = {1: {}, -1: {}}
        self.cache = {}
        self.updates = 0

    def observe(self, markout):
        if markout['horizon_ms'] != 1000 or not 0 <= markout['observed_ns']-markout['target_ns'] <= 250_000_000:
            return
        fill = markout['fill']
        side = fill['side']
        orders = self.sides[side]
        # A split execution occupies one slot. Keep its latest matured sample.
        orders.pop(fill['cid'], None)
        orders[fill['cid']] = (markout['observed_ns']//1_000_000,
                               float(side * (markout['mid']-fill['price'])/fill['price']*10000))
        if len(orders) > self.settings['orders']:
            del orders[next(iter(orders))]
        self.cache.pop(side, None)
        self.updates += 1

    def widen_bp(self, side, now_ms):
        settings = self.settings
        orders = self.sides[side]
        if len(orders) < settings['minimum_orders']:
            return D(0)
        cached = self.cache.get(side)
        if cached is not None and cached[0] == now_ms:
            return cached[1]
        samples = [(now_ms-stamp, bp) for stamp,bp in orders.values()
                   if 0 <= now_ms-stamp < settings['max_age_ms']]
        widening = D(0)
        if len(samples) >= settings['minimum_orders']:
            half_life = settings['half_life_ms']
            weights = [2**(-age/half_life) for age,_ in samples]
            average = sum(w*bp for w,(_,bp) in zip(weights,samples))/sum(weights)
            excess = max(0., -average-settings['deadband_bp'])
            widening = D(str(min(settings['cap_bp'], excess)*weights[-1]))
        self.cache[side] = (now_ms, widening)
        return widening


class GLFTModel(NativeAS):
    def __init__(self,config,rules,gamma,xi,unit,*,inventory_solution='quadratic',feedback=None):
        super().__init__(config,rules)
        self.risk_gamma=float(gamma);self.xi=float(xi);self.unit=D(unit)
        if self.unit<rules.step or self.unit%rules.step:
            raise ValueError('GLFT inventory unit must align to the symbol lot size')
        self.position_cap = D(config['max_position_qty'])
        self.inventory_solution = inventory_solution
        if inventory_solution == 'spectral':
            if self.risk_gamma != self.xi:
                raise ValueError('Spectral GLFT requires gamma equal to xi')
            self.inventory_cap = math.ceil(self.position_cap/self.unit)
            if not 1 <= self.inventory_cap <= 1024:
                raise ValueError('Spectral GLFT inventory grid requires 1 to 1024 lots per side')
        self.feedback = None if feedback is None else MarkoutFeedback(feedback)
        self.fair_value = None
        self._vol_second = None
        self._vol_rms = None
        self._calibration_key = None
        self.last_glft={}

    def ready(self, now_ms):
        if (self.last_fit_ms <= 0 or self.alpha <= 0 or self.kappa <= 0
                or now_ms - self.last_fit_ms > self.config['calibration_max_age_seconds']*1000):
            return False
        # All volatility inputs change only when NativeAS accepts a second.
        if self._vol_rms is None or self._vol_second != self.last_second:
            self._vol_second = self.last_second
            self._vol_rms = self.vol.current_value
        return self._vol_rms > 0

    def _calibrate_quotes(self):
        tick = float(self.rules.tick)
        k = float(self.kappa)*tick
        c1,c2 = coefficients(self.risk_gamma,self.xi,float(self.alpha),k)
        changes = np.diff(np.asarray(self.vol.samples,dtype=float))/tick
        sigma = float(np.std(changes,ddof=0)) if len(changes) else 0.
        half = c1+sigma*c2/2
        skew = sigma*c2
        self._width = D(str(half))*self.rules.tick
        self._skew = D(str(skew))*self.rules.tick
        self.last_glft = dict(A_per_second=str(self.alpha), k_per_tick=k, sigma_ticks_sqrt_second=sigma,
                              c1=c1, c2=c2, half_spread_ticks=half, skew_ticks_per_inventory_unit=skew)
        if self.inventory_solution == 'spectral':
            scaled_skew = k*skew
            self._log_values = spectral_log_values(self.inventory_cap, scaled_skew*scaled_skew)
            self.last_glft.update(inventory_grid_cap=self.inventory_cap, quote_diagnostics='quadratic_reference')

    def _log_value(self, q):
        # Partial fills are continuous in lot units. At the grid edge use
        # its last slope for an indicative quote; actual size obeys common risk.
        q = abs(q)
        lower = min(math.floor(q), self.inventory_cap-1)
        y = self._log_values
        return y[lower]+(q-lower)*(y[lower+1]-y[lower])

    def quote(self,book,position,now_ms):
        if not self.ready(now_ms):raise ValueError('GLFT estimates unavailable or stale')
        if abs(position)>self.position_cap:raise ValueError('GLFT inventory outside cap')
        # The shared estimator samples once/second. A is per-second arrival
        # exceedance proxy, not observed per-order fill probability.
        calibration_key = self.last_second, self.last_fit_ms
        if self._calibration_key != calibration_key:
            self._calibrate_quotes()
            self._calibration_key = calibration_key
        reference = book.mid
        if self.fair_value is not None and book.bid <= self.fair_value <= book.ask:
            reference = self.fair_value
            if reference != book.mid:
                self.diagnostics['weighted_mid_quotes'] += 1
        if self.inventory_solution == 'spectral':
            q = float(position/self.unit)
            value = self._log_value(q)
            c1, k = self.last_glft['c1'], self.last_glft['k_per_tick']
            bid = reference-D(str(c1+(value-self._log_value(q+1))/k))*self.rules.tick
            ask = reference+D(str(c1+(value-self._log_value(q-1))/k))*self.rules.tick
        else:
            center = reference-self._skew*position/self.unit
            bid,ask = center-self._width,center+self._width
        bid,ask = min(bid,book.bid),max(ask,book.ask)
        if self.feedback is not None:
            bp = book.mid/10000
            if position >= 0:
                widening = self.feedback.widen_bp(1,now_ms)
                bid -= bp*widening
                if widening:
                    self.diagnostics['bid_feedback_quotes'] += 1
            if position <= 0:
                widening = self.feedback.widen_bp(-1,now_ms)
                ask += bp*widening
                if widening:
                    self.diagnostics['ask_feedback_quotes'] += 1
        if (not bid.is_finite() or not ask.is_finite() or bid<=0 or bid>=ask or
                max(abs(bid/book.mid-1),abs(ask/book.mid-1))>D(self.config['as_max_half_spread'])):
            raise ValueError('GLFT quote outside configured distance bounds')
        return bid,ask

    def report(self,now_ms):
        result = dict(super().report(now_ms), implementation='glft_asymptotic_delta_one',
                      inventory_solution=self.inventory_solution, gamma=self.risk_gamma, xi=self.xi,
                      inventory_unit=str(self.unit), **self.last_glft)
        if self.feedback is not None:
            result['markout_feedback_updates'] = self.feedback.updates
        return result


class GLFTStrategy(ASStrategy):
    additional_sources=SOURCES

    def __init__(self,rules,config,factors,settings,symbol):
        validate_settings(settings);self.glft_settings=deepcopy(settings)
        if symbol not in settings['inventory_units']:raise ValueError('Missing GLFT inventory unit for '+str(symbol))
        base=dict(mode='as_base',**{k:settings[k] for k in ('paper_config','trade_start_ns','trade_end_ns')})
        super().__init__(rules,config,factors,base,symbol)

    def _reset_native(self):
        super()._reset_native();p=self.glft_settings
        self.model=GLFTModel(self.paper_config,self.rules,p['gamma'],p['xi'],p['inventory_units'][self.symbol],
                             inventory_solution=p.get('inventory_solution','quadratic'),feedback=p.get('markout_feedback'))

    def reset_market(self):
        super().reset_market()
        self.model.fair_value = None

    def on_top_of_book(self, ns, bid, bid_qty, ask, ask_qty):
        super().on_top_of_book(ns,bid,bid_qty,ask,ask_qty)
        if self.glft_settings.get('reference_price') == 'weighted_mid':
            self.model.fair_value = self.micro

    def on_markout(self, markout):
        if self.model.feedback is not None:
            self.model.feedback.observe(markout)

    @property
    def identity(self):
        value=super().identity
        value.update(type='glft',params=deepcopy(self.glft_settings),
                     implementation='GLFT stationary quotes; common AS calibration and order policy',
                     feedback='Optional per-side completed 1s markout; one recent sample per order',
                     estimator='1s causal arrival maxima fit on 0.5..11.5 tick distances; 1s increment population std',
                     reference_commit='66b2e7537f0005b1bf691373017160566814ee4a')
        return value
