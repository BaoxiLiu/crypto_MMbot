"""S1 volatility floor sigma module (spec v0.2 section 2).

Causal 1-second-grid volatility estimated from the bookTicker update stream.
The module itself is a pure sigma estimator plus validity flags; the floor
state machine (vol_floor_ticks, stale freeze, degraded fallback) lives in
core.Account.update_volatility, which consumes snapshot() output.

Interface:
  RollingVolatility(config, tick)
    config keys (vol_* preferred; legacy s1_* keys are accepted as aliases so
    older batch configs stay reproducible):
      vol_window_min        trailing window in minutes (default "30" = 1800s)
      vol_stale_ms          a grid point is missing when the last ticker is
                            older than this at the grid second (default 5000)
      vol_missing_ratio     point-missing share above which sigma is invalid
                            (default "0.2")
  on_ticker(data, now_ms)   data is the raw bookTicker payload: best bid "b",
                            best ask "a" (validated: finite, 0 < bid < ask);
                            now_ms is local RECEIVE time. Grid points use the
                            latest ticker received strictly before the grid
                            second, so an estimate at time t never observes
                            data received at or after t (causality).
  snapshot(now_ms)          advance the grid to now_ms//1000 and return
                            {valid, sigma_ticks, grid_points, window_seconds,
                             missing_ratio, ticker_age_ms, grid_second}.
                            valid requires: full window present, missing_ratio
                            within threshold, and a non-stale last ticker.
  dump()/load()             persistence of the grid (same commit as the
                            ledger, written via core.Store.save).

Estimation (spec section 2):
  Grid point t uses mid_t from the freshest ticker strictly before t*1000ms.
  1s return r_t = (mid_t - mid_{t-1s}) / mid_{t-1s}; a return is only counted
  when BOTH endpoints exist (missing points are dropped from the std, never
  zero-filled). sigma_ticks = std(r_t) x mid / tick_size with the POPULATION
  std over the trailing vol_window_min window; mid is the latest observed mid.
  missing_ratio is the share of MISSING GRID POINTS (not returns) in the
  window, per the spec's "missing share of window points" wording.

Account-level state machine (driven by update_volatility in core.py):
  active   valid sigma -> vol_floor_ticks = clamp(ceil(k x sigma_ticks),
           vol_floor_min_ticks, vol_floor_max_ticks); events vol_floor_updated.
  stale    sigma invalid (missing share or ticker stale beyond 5s) ->
           vol_floor_ticks frozen at the last valid value, raised to at
           least static_floor_ticks; event vol_floor_stale; new quotes are
           NOT paused. A freeze is never silently narrowed back.
  degraded more than vol_degraded_min (default 10) minutes without a valid
           sigma -> quotes paused (floor_state falls back to the static
           floor), event vol_floor_degraded; the next valid sigma returns to
           active. Transition: active -> stale -> degraded -> active.
"""
import math
from collections import deque

from .core import D, ZERO


class RollingVolatility:
    def __init__(self, config, tick):
        window_min = config.get("vol_window_min")
        if window_min is not None:
            # Fractional minutes are allowed ("30" or "0.5").
            self.window_seconds = max(2, int(D(str(window_min)) * 60))
        else:
            self.window_seconds = int(config.get("s1_window_seconds", 1800))
        self.stale_ms = int(config.get("vol_stale_ms",
                                       config.get("s1_ticker_stale_ms", 5000)))
        self.max_missing = float(config.get("vol_missing_ratio",
                                            config.get("s1_max_missing_ratio", 0.2)))
        if self.window_seconds < 2 or self.stale_ms < 1 or not 0 <= self.max_missing < 1:
            raise ValueError("Invalid S1 volatility configuration")
        self.tick = D(tick)
        self.points = deque(maxlen=self.window_seconds + 1)
        self.last_grid_second = None
        self.last_ticker_ms = None
        self.last_mid = None
        self.std_return = None
        self.missing_ratio = 1.0

    def on_ticker(self, data, now_ms):
        bid, ask = D(data["b"]), D(data["a"])
        if not bid.is_finite() or not ask.is_finite() or not ZERO < bid < ask:
            return False
        if self.last_ticker_ms is not None and now_ms < self.last_ticker_ms:
            return False
        self.advance(now_ms)
        self.last_mid = (bid + ask) / 2
        self.last_ticker_ms = now_ms
        return True

    def advance(self, now_ms):
        second = now_ms // 1000
        if self.last_grid_second is None:
            self.last_grid_second = second
            return
        if second <= self.last_grid_second:
            return
        # Bounded work even after a long offline interval.
        start = max(self.last_grid_second + 1, second - self.window_seconds)
        if start > self.last_grid_second + 1:
            self.points.clear()
        for s in range(start, second + 1):
            age = s * 1000 - self.last_ticker_ms if self.last_ticker_ms is not None else None
            mid = self.last_mid if age is not None and 0 < age <= self.stale_ms else None
            self.points.append((s, mid))
        self.last_grid_second = second
        pairs = zip(self.points, list(self.points)[1:])
        returns = [float(b / a - 1) for (ta, a), (tb, b) in pairs
                   if tb == ta + 1 and a is not None and b is not None]
        # Missing share is measured over GRID POINTS (spec section 2), not
        # over returns: one missing interior point breaks two returns, so a
        # return-based ratio would overstate the missing share.
        points = list(self.points)
        # The first point is the anchor for the first return, outside the
        # N-point reporting window (t-N, t]. Do not count N+1 points over N.
        missing = sum(1 for _, m in points[1:] if m is None)
        self.missing_ratio = (missing / self.window_seconds
                              if len(points) == self.window_seconds + 1 else 1.0)
        self.std_return = None
        if len(points) == self.window_seconds + 1 and len(returns) >= 2:
            mean = math.fsum(returns) / len(returns)
            self.std_return = math.sqrt(math.fsum((r - mean) ** 2 for r in returns) / len(returns))

    def snapshot(self, now_ms):
        self.advance(now_ms)
        age = now_ms - self.last_ticker_ms if self.last_ticker_ms is not None else None
        valid = (self.std_return is not None and self.missing_ratio <= self.max_missing + 1e-12
                 and age is not None and 0 <= age <= self.stale_ms)
        sigma = (D(str(self.std_return)) * self.last_mid / self.tick
                 if valid and self.last_mid is not None else None)
        return {"valid": valid, "sigma_ticks": sigma,
                "grid_points": len(self.points), "window_seconds": self.window_seconds,
                "missing_ratio": self.missing_ratio, "ticker_age_ms": age,
                "grid_second": self.last_grid_second}

    def dump(self):
        return {"points": [(s, str(m) if m is not None else None) for s, m in self.points],
                "last_grid_second": self.last_grid_second,
                "last_ticker_ms": self.last_ticker_ms,
                "last_mid": str(self.last_mid) if self.last_mid is not None else None}

    def load(self, data):
        self.points = deque(((int(s), D(m) if m is not None else None)
                             for s, m in data["points"]), maxlen=self.window_seconds + 1)
        self.last_grid_second = data["last_grid_second"]
        self.last_ticker_ms = data["last_ticker_ms"]
        self.last_mid = D(data["last_mid"]) if data["last_mid"] is not None else None
        # Recompute on the next second; never mark restored stale data ready.
        self.std_return = None
        return self
