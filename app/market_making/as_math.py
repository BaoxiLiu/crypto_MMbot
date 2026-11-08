"""Canonical AS math and causal volatility shared by shadow/live/replay.

Extracted without numerical changes from the certified AS paper v2 core.
"""
from dataclasses import dataclass
from decimal import Decimal as D
import math


def positive(value, name, zero=False):
    if not math.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(f"Invalid {name}: {value}")


@dataclass(frozen=True)
class ASConfig:
    gamma: float = 0.1              # per tick, per reference order
    kappa: float = 0.2              # assumed arrival decay per tick, not fitted
    horizon_s: float = 5.0          # rolling risk horizon, not session expiry
    volatility_tau_s: float = 10.0
    sample_ns: int = 100_000_000
    warmup_samples: int = 20
    stale_ns: int = 1_000_000_000
    half_spread_floor: float = 0.0005  # fraction of mid, explicitly a floor
    order_notional: D = D("25")
    position_cap: D = D("3200")     # asset quantity, independent of q units
    maker_fee: D = D("0.0002")      # scenario assumption; override actual tier

    def __post_init__(self):
        for key in ("gamma", "half_spread_floor"):
            positive(getattr(self, key), key, zero=True)
        for key in ("kappa", "horizon_s", "volatility_tau_s"):
            positive(getattr(self, key), key)
        for key in ("sample_ns", "warmup_samples", "stale_ns"):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError(f"Invalid {key}")
        if self.stale_ns < self.sample_ns:
            raise ValueError("stale_ns must cover at least one volatility sample")
        for key in ("order_notional", "position_cap", "maker_fee"):
            v = getattr(self, key)
            if not isinstance(v, D) or not v.is_finite() or (key != "maker_fee" and v <= 0):
                raise ValueError(f"Invalid {key}")


def as_prices(mid_ticks, inventory_units, variance, gamma, kappa, horizon_s):
    """AS approximate reservation price and half spread, all in tick units."""
    positive(mid_ticks, "mid_ticks")
    positive(variance, "variance ticks²/s", zero=True)
    positive(gamma, "gamma", zero=True)
    positive(kappa, "kappa")
    positive(horizon_s, "horizon_s")
    if not math.isfinite(inventory_units):
        raise ValueError("Nonfinite inventory")
    risk = gamma * variance * horizon_s
    # Stable gamma -> 0 limit; avoid gamma/kappa overflow where possible.
    ratio = gamma / kappa
    liquidity = 1.0 / kappa if ratio < 1e-8 else (math.log1p(ratio) / gamma)
    reservation = mid_ticks - inventory_units * risk
    half = 0.5 * risk + liquidity
    if not math.isfinite(reservation) or not math.isfinite(half):
        raise ValueError("AS numerical overflow")
    return reservation, half


class Variance:
    """Fixed-cadence, causal, normalized EWMA of squared tick changes / second.

    At a boundary use only prices already observed. A later observation is never
    interpolated backwards. Long gaps restart warmup rather than invent returns.
    """
    def __init__(self, config):
        self.config = config
        self.reset()

    def reset(self):
        self.last_ns = self.next_ns = None
        self.observed = self.sampled = 0.0
        self.numerator = self.weight = 0.0
        self.samples = 0

    @property
    def ready(self):
        return self.samples >= self.config.warmup_samples

    @property
    def value(self):
        return self.numerator / self.weight if self.weight else 0.0

    def update(self, ns, mid_ticks):
        positive(mid_ticks, "mid_ticks")
        if type(ns) is not int or ns < 0:
            raise ValueError("Invalid observation time")
        if self.last_ns is not None and ns < self.last_ns:
            raise ValueError("Observation time moved backwards")
        if self.last_ns is None or ns - self.last_ns > self.config.stale_ns:
            self.reset()
            self.last_ns = ns
            self.next_ns = ns + self.config.sample_ns
            self.observed = self.sampled = mid_ticks
            return
        dt = self.config.sample_ns / 1e9
        alpha = -math.expm1(-dt / self.config.volatility_tau_s)
        while self.next_ns <= ns:
            value = mid_ticks if self.next_ns == ns else self.observed
            change = value - self.sampled
            self.numerator = (1 - alpha) * self.numerator + alpha * change * change / dt
            self.weight = (1 - alpha) * self.weight + alpha
            self.samples += 1
            self.sampled = value
            self.next_ns += self.config.sample_ns
        self.observed, self.last_ns = mid_ticks, ns


