from collections import Counter, deque
from decimal import Decimal

import numpy as np
from scipy.optimize import curve_fit

from .core import D, ZERO


class InstantVolatilityIndicator:
    """Price-increment RMS; denominator is sample count, matching HB's definition."""
    def __init__(self, sampling_length):
        self.samples = deque(maxlen=sampling_length)

    def add_sample(self, value):
        self.samples.append(float(value))

    @property
    def current_value(self):
        values = np.asarray(self.samples, dtype=float)
        return float(np.sqrt(np.sum(np.square(np.diff(values))) / len(values))) if len(values) else float('nan')


class NativeAS:
    """Project-owned AS estimator and quotes, following Hummingbot infinite-horizon conventions."""

    def __init__(self, config, rules):
        self.config = config
        if config.get('as_sampling_policy', 'quote_health') not in ('quote_health', 'book'):
            raise ValueError('Unknown AS sampling policy')
        if config.get('as_missing_policy', 'reset') not in ('reset', 'mask'):
            raise ValueError('Unknown AS missing-observation policy')
        if not 0 <= float(config.get('as_max_missing_ratio', .2)) < 1:
            raise ValueError('Invalid AS missing ratio')
        self.rules = rules
        quote = next((q for q in ("USDT", "USDC") if config["symbol"].endswith(q)), None)
        if quote is None:
            raise ValueError(f"Cannot infer quote asset from symbol {config['symbol']}")
        self.base = config["symbol"][:-len(quote)]
        self.pair = f"{self.base}-{quote}"
        self.quote_asset = quote
        self.intensity_window_seconds = config.get('as_intensity_window_seconds', config['calibration_window_seconds'])
        if (not isinstance(self.intensity_window_seconds, int) or isinstance(self.intensity_window_seconds, bool)
                or self.intensity_window_seconds < config['min_calibration_seconds']):
            raise ValueError('AS intensity window must be integer seconds >= minimum calibration samples')
        self.samples = deque(maxlen=self.intensity_window_seconds)
        self.sample_references = deque(maxlen=self.intensity_window_seconds)
        self.last_sample_ms = None
        self.fit_observer = None
        self.sample_observer = None
        self.vol = InstantVolatilityIndicator(sampling_length=config["calibration_window_seconds"])
        self.reference = None
        self.depth = -1.0
        self.last_second = None
        self.last_fit_ms = 0
        self.last_attempt_ms = 0
        self.alpha = ZERO
        self.kappa = ZERO
        self.gamma = None
        self.fit_r2 = None
        self.fit_error = "warmup"
        self.diagnostics = Counter()
    def sampling_allowed(self, book, now_ms, quote_blocked):
        # Estimation depends on synchronized depth and trades, not on the
        # mark/funding feed used by account risk. No missing depth is imputed.
        if self.config.get('as_sampling_policy', 'quote_health') == 'quote_health':
            return not quote_blocked
        return book.ready and now_ms - book.received_ms <= self.config['stale_book_ms']

    def trade(self, side, price):
        if self.reference is not None:
            distance = side * (price - self.reference)
            self.depth = max(self.depth, float(distance))

    def sample(self, mid, now_ms):
        second = now_ms // 1000
        if second == self.last_second:
            return
        if self.last_second is not None and second == self.last_second + 1:
            if self.sample_observer is not None:
                eligible = (self.last_fit_ms > 0 and self.last_fit_ms <= self.last_sample_ms
                            and self.last_sample_ms-self.last_fit_ms <= self.config['calibration_max_age_seconds']*1000)
                self.sample_observer(self.last_second, self.depth, float(self.reference),
                                     float(self.alpha) if eligible else float('nan'),
                                     float(self.kappa) if eligible else float('nan'))
            self.samples.append(self.depth)
            self.sample_references.append(float(self.reference))
        elif self.last_second is not None:
            # Missing observations are not zero-trade observations.
            self.diagnostics["sampling_gaps"] += 1
            self.diagnostics["missing_seconds"] += second - self.last_second - 1
            if self.config.get('as_missing_policy', 'reset') == 'mask':
                # Include the interrupted last bucket as unknown. Keep a
                # bounded wall-time window, never count gaps as quiet trading.
                count = min(second - self.last_second, self.samples.maxlen)
                self.samples.extend([float('nan')] * count)
                self.sample_references.extend([float('nan')] * count)
            else:
                self.diagnostics['sample_gap_resets'] += 1
                self.samples.clear()
                self.sample_references.clear()
                self.last_fit_ms = 0
            # Do not splice price changes across missing periods into sigma.
            self.vol = InstantVolatilityIndicator(sampling_length=self.config["calibration_window_seconds"])
        self.vol.add_sample(float(mid))
        self.reference = mid
        self.depth = -1.0
        self.last_second = second
        self.last_sample_ms = now_ms
        if now_ms - self.last_attempt_ms >= self.config["calibration_interval_seconds"] * 1000:
            self.fit(now_ms)

    def fit(self, now_ms):
        self._fit(now_ms)
        if self.fit_observer is not None:
            self.fit_observer(now_ms, self.fit_snapshot(now_ms))

    def fit_snapshot(self, now_ms):
        raw = np.asarray(self.samples)
        observed = raw[np.isfinite(raw)]
        distances = (np.arange(12, dtype=float)+.5)*float(self.rules.tick)
        counts = np.array([(observed > d).sum() for d in distances])
        valid = counts >= 3
        detail = ('insufficient_distance_points' if valid.sum()<3 else
                  'flat_arrival_curve' if counts[0]==counts[valid][-1] else 'distinct_arrival_curve')
        reference = float(self.reference) if self.reference else None
        bp_grid = np.array([1,2,5,10,15,20,30,50,100],dtype=float)
        refs = np.asarray(self.sample_references)
        bp_counts = None
        if len(refs)==len(raw):
            covered = np.isfinite(raw) & np.isfinite(refs) & (refs>0)
            bp_depths = np.where(raw[covered]<0, -1., raw[covered]/refs[covered]*10000)
            bp_counts = [int((bp_depths>d).sum()) for d in bp_grid]
        return {'window_seconds':self.intensity_window_seconds,'observations':len(raw),
                'valid_seconds':len(observed),'missing_ratio':1-len(observed)/max(1,len(raw)),
                'distance_ticks':(distances/float(self.rules.tick)).tolist(),
                'distance_bp_at_fit':(distances/reference*10000).tolist() if reference else None,
                'exceed_seconds':counts.tolist(),'eligible_distance_points':int(valid.sum()),
                'distinct_eligible_counts':len(set(counts[valid].tolist())),
                'arrival_detail':detail,'bp_grid':bp_grid.tolist(),'bp_exceed_seconds':bp_counts,
                'fit_error':self.fit_error,'fit_updated':self.last_fit_ms==now_ms,
                'alpha':str(self.alpha),'kappa':str(self.kappa),'fit_r2':self.fit_r2}

    def _fit(self, now_ms):
        self.last_attempt_ms = now_ms
        self.diagnostics["fit_attempts"] += 1
        observed = np.asarray(self.samples)
        if self.config.get('as_missing_policy', 'reset') == 'mask':
            valid_observations = observed[np.isfinite(observed)]
            missing_ratio = 1 - len(valid_observations) / max(1, len(observed))
            if missing_ratio > float(self.config.get('as_max_missing_ratio', .2)):
                self.fit_error = 'insufficient_observation_coverage'
                self.diagnostics[self.fit_error] += 1
                return
            observed = valid_observations
        if len(observed) < self.config["min_calibration_seconds"]:
            self.fit_error = "insufficient_samples"
            self.diagnostics[self.fit_error] += 1
            return
        # Per-second arrival maxima include quiet seconds in the denominator.
        distances = (np.arange(12, dtype=float) + 0.5) * float(self.rules.tick)
        counts = np.array([(observed > level).sum() for level in distances])
        valid = counts >= 3
        if valid.sum() < 3 or counts[0] == counts[valid][-1]:
            self.fit_error = "insufficient_distinct_arrival_depths"
            detail = 'insufficient_distance_points' if valid.sum()<3 else 'flat_arrival_curve'
            self.diagnostics[detail] += 1
            self.diagnostics[self.fit_error] += 1
            return
        x = distances[valid]
        rates = counts[valid] / len(observed)
        try:
            pars, _ = curve_fit(
                lambda d, a, k: a * np.exp(-k * d), x, rates,
                p0=(max(float(rates[0]), 0.01), 1 / max(float(self.rules.tick) * 3, 1e-12)),
                bounds=([1e-10, 1e-10], [np.inf, np.inf]), maxfev=5000,
            )
            predicted = pars[0] * np.exp(-pars[1] * x)
            r2 = 1 - float(np.sum((rates - predicted) ** 2) / np.sum((rates - rates.mean()) ** 2))
            if not np.isfinite(pars).all() or not np.isfinite(r2) or r2 < 0.3:
                raise ValueError("Unreliable intensity fit")
            self.alpha, self.kappa = (D(str(p)) for p in pars)
            self.fit_r2 = r2
            self.last_fit_ms = now_ms
            self.fit_error = ""
            self.diagnostics["fit_successes"] += 1
        except (RuntimeError, ValueError, FloatingPointError) as error:
            self.fit_error = str(error)
            self.diagnostics["fit_failures"] += 1

    def readiness_reason(self, now_ms):
        if not self.last_fit_ms or self.alpha <= ZERO or self.kappa <= ZERO:
            return self.fit_error or "no_valid_fit"
        if now_ms - self.last_fit_ms > self.config["calibration_max_age_seconds"] * 1000:
            return "fit_expired:" + (self.fit_error or "awaiting_fit")
        if float(self.vol.current_value) <= 0:
            return "zero_volatility"
        return "ready"

    def ready(self, now_ms):
        return (self.last_fit_ms > 0 and self.alpha > ZERO and self.kappa > ZERO
                and now_ms - self.last_fit_ms <= self.config["calibration_max_age_seconds"] * 1000
                and float(self.vol.current_value) > 0)

    def calibrate_gamma(self):
        volatility = D(str(self.vol.current_value))
        if volatility <= ZERO:
            raise ValueError("No volatility estimate")
        # At the position cap, normalized inventory is 0.5: target one tick of skew.
        candidate = 2 * self.rules.tick / volatility
        self.gamma = max(D(self.config["as_gamma_min"]), min(D(self.config["as_gamma_max"]), candidate))

    def quote(self, book, position, now_ms):
        if not self.ready(now_ms):
            raise ValueError("AS estimates unavailable or stale")
        if self.gamma is None:
            self.calibrate_gamma()
        cap = D(self.config["max_position_qty"])
        if abs(position) > cap:
            raise ValueError("Position outside AS inventory adapter domain")
        volatility = D(str(self.vol.current_value))
        inventory = position / (2 * cap)
        reservation = book.mid - inventory * self.gamma * volatility
        spread = self.gamma * volatility + 2 * (1 + self.gamma / self.kappa).ln() / self.gamma
        bid = min(reservation - spread / 2, book.mid)
        ask = max(reservation + spread / 2, book.mid)
        if not all(isinstance(p, Decimal) and p.is_finite() and p > ZERO for p in [bid, ask]):
            raise ValueError("Invalid native AS quotes")
        if bid >= ask or max(abs(bid / book.mid - 1), abs(ask / book.mid - 1)) > D(self.config["as_max_half_spread"]):
            raise ValueError("AS quote outside configured distance bounds")
        return bid, ask

    def report(self, now_ms):
        return {"implementation": "project_as_hummingbot_conventions", "samples": len(self.samples), "valid_samples": int(sum(np.isfinite(v) for v in self.samples)),
                "alpha": self.alpha, "kappa": self.kappa, "gamma": self.gamma,
                "volatility": str(self.vol.current_value), "fit_r2": self.fit_r2,
                "last_fit_ms": self.last_fit_ms, "fit_error": self.fit_error,
                "ready": self.ready(now_ms), "readiness_reason": self.readiness_reason(now_ms),
                "diagnostics": dict(self.diagnostics),
                "intensity_window_seconds": self.intensity_window_seconds,
                "native_volatility_window_seconds": self.config['calibration_window_seconds']}
