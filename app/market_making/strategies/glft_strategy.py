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
    if set(p)!={'paper_config','trade_start_ns','trade_end_ns','gamma','xi','inventory_units'}:
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


class GLFTModel(NativeAS):
    def __init__(self,config,rules,gamma,xi,unit):
        super().__init__(config,rules)
        self.risk_gamma=float(gamma);self.xi=float(xi);self.unit=D(unit)
        if self.unit<rules.step or self.unit%rules.step:
            raise ValueError('GLFT inventory unit must align to the symbol lot size')
        self.last_glft={}

    def quote(self,book,position,now_ms):
        if not self.ready(now_ms):raise ValueError('GLFT estimates unavailable or stale')
        if abs(position)>D(self.config['max_position_qty']):raise ValueError('GLFT inventory outside cap')
        # The shared estimator samples once/second. A is per-second arrival
        # exceedance proxy, not observed per-order fill probability.
        tick=float(self.rules.tick);k=float(self.kappa)*tick
        c1,c2=coefficients(self.risk_gamma,self.xi,float(self.alpha),k)
        changes=np.diff(np.asarray(self.vol.samples,dtype=float))/tick
        sigma=float(np.std(changes,ddof=0)) if len(changes) else 0.
        half=c1+sigma*c2/2;skew=sigma*c2
        center=book.mid-D(str(skew))*self.rules.tick*position/self.unit
        width=D(str(half))*self.rules.tick
        bid=min(center-width,book.bid);ask=max(center+width,book.ask)
        if (not bid.is_finite() or not ask.is_finite() or bid<=0 or bid>=ask or
                max(abs(bid/book.mid-1),abs(ask/book.mid-1))>D(self.config['as_max_half_spread'])):
            raise ValueError('GLFT quote outside configured distance bounds')
        self.last_glft=dict(A_per_second=str(self.alpha),k_per_tick=k,sigma_ticks_sqrt_second=sigma,
                            c1=c1,c2=c2,half_spread_ticks=half,skew_ticks_per_inventory_unit=skew)
        return bid,ask

    def report(self,now_ms):
        return dict(super().report(now_ms),implementation='glft_asymptotic_delta_one',
                    gamma=self.risk_gamma,xi=self.xi,inventory_unit=str(self.unit),**self.last_glft)


class GLFTStrategy(ASStrategy):
    additional_sources=SOURCES

    def __init__(self,rules,config,factors,settings,symbol):
        validate_settings(settings);self.glft_settings=deepcopy(settings)
        if symbol not in settings['inventory_units']:raise ValueError('Missing GLFT inventory unit for '+str(symbol))
        base=dict(mode='as_base',**{k:settings[k] for k in ('paper_config','trade_start_ns','trade_end_ns')})
        super().__init__(rules,config,factors,base,symbol)

    def _reset_native(self):
        super()._reset_native();p=self.glft_settings
        self.model=GLFTModel(self.paper_config,self.rules,p['gamma'],p['xi'],p['inventory_units'][self.symbol])

    @property
    def identity(self):
        value=super().identity
        value.update(type='glft',params=deepcopy(self.glft_settings),
                     implementation='GLFT closed-form asymptotic quotes; common AS calibration and order policy',
                     estimator='1s causal arrival maxima fit on 0.5..11.5 tick distances; 1s increment population std',
                     reference_commit='66b2e7537f0005b1bf691373017160566814ee4a')
        return value
