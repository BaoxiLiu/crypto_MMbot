"""Explicit ablation factors. Mandatory execution/risk guards cannot be disabled."""
from dataclasses import dataclass, field, asdict
from decimal import Decimal as D
import math
from ..as_math import ASConfig


@dataclass(frozen=True)
class FusionConfig:
    as_config: ASConfig = field(default_factory=ASConfig)
    initial_cash: D = D('600')
    max_notional: D = D('300')
    daily_loss: D = D('10')
    drawdown_limit: D = D('30')
    momentum_ns: int = 5_000_000_000
    flow_window_ns: int = 1_000_000_000
    trend_bp: float = 10.0
    stress_sigma_bp: float = 8.0
    stress_release_ratio: float = .7
    flow_imbalance: float = .7
    flow_min_trades: int = 10
    flow_min_notional: float = 250.0
    flow_cooldown_ns: int = 1_000_000_000
    inventory_soft_units: float = 2.0
    inventory_hard_units: float = 4.0
    soft_size_multiplier: D = D('.5')
    min_quote_lifetime_ns: int = 100_000_000
    max_new_60s: int = 1200
    mark_stale_ns: int = 5_000_000_000

    def __post_init__(self):
        for name in ('initial_cash','max_notional','daily_loss','drawdown_limit','soft_size_multiplier'):
            v=getattr(self,name)
            if not isinstance(v,D) or not v.is_finite() or v<=0:raise ValueError(name)
        if self.soft_size_multiplier>1:raise ValueError('soft_size_multiplier')
        for name in ('momentum_ns','flow_window_ns','flow_min_trades','flow_cooldown_ns','min_quote_lifetime_ns','max_new_60s','mark_stale_ns'):
            v=getattr(self,name)
            if type(v) is not int or v<=0:raise ValueError(name)
        for name in ('trend_bp','stress_sigma_bp','flow_min_notional','inventory_soft_units','inventory_hard_units'):
            v=getattr(self,name)
            if isinstance(v,bool) or not math.isfinite(v) or v<=0:raise ValueError(name)
        if not 0<self.flow_imbalance<1 or not 0<self.stress_release_ratio<1:raise ValueError('ratio')
        if self.inventory_soft_units>=self.inventory_hard_units:raise ValueError('inventory units')

    @classmethod
    def from_dict(cls,data):
        data=dict(data)
        a=dict(data.pop('as_config',{}))
        for key in ('order_notional','position_cap','maker_fee'):
            if key in a:a[key]=D(str(a[key]))
        for key in ('initial_cash','max_notional','daily_loss','drawdown_limit','soft_size_multiplier'):
            if key in data:data[key]=D(str(data[key]))
        return cls(as_config=ASConfig(**a),**data)

    def json(self):
        import json
        return json.loads(json.dumps(asdict(self),default=str))


@dataclass(frozen=True)
class Factors:
    regime: bool=False
    flow: bool=False
    inventory: bool=False

    @classmethod
    def from_mask(cls,mask):
        if type(mask) is not int or not 0<=mask<=7:raise ValueError('mask must be 0..7')
        return cls(bool(mask&1),bool(mask&2),bool(mask&4))

    @property
    def label(self):
        return 'AS'+('R' if self.regime else '')+('F' if self.flow else '')+('I' if self.inventory else '')
