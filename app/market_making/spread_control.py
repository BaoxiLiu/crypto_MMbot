"""Layer-3 closed-loop spread control (SC0), driven by rolling markout@1s feedback.

唯一实现;live runner 与离线回测(controllers.py)都从这里取,不再经由 runner.py。
"""
import statistics
from collections import Counter, deque
from decimal import ROUND_HALF_UP

from .core import D, ZERO


class SpreadController:
    """Layer-3 closed-loop spread control driven by rolling markout@1s feedback.

    Multiplicative steps: adverse mean multiplies the floor by sc_up_factor,
    favorable mean divides it by sc_down_factor, each rounded to the nearest
    integer tick (half away from zero) and clamped to a bounded single jump
    and the [sc_min_ticks, sc_max_ticks] range. Only sc_target_ticks is
    mutated; the static config and volatility floor remain independent —
    account.config is never written (spec v0.2 section 3).

    Signal variant (spec v0.2 section 4), account.config sc_signal_variant:
      "v1_mean_usdt" (default): legacy mean of per-fill pnl in quote currency.
      "v2_median_bp": median of the last sc_window_fills valid 1s markouts in
      basis points, signed so POSITIVE is favorable to the maker —
      signed_bp = side x (mid_at_1s - fill_price) / fill_price x 1e4, with
      side = +1 for a maker BUY (long opened; mid rising is favorable) and
      -1 for a maker SELL. This matches the markout pnl sign convention in
      Experiment.markouts (pnl = side x qty x (mid - price)), so
      pnl/notional already equals signed_bp and no extra sign work is needed.
      Legacy aliases: sc_signal="median_bp" selects v2, sc_signal="mean_pnl"
      keeps v1. The v2 dead zone is abs(signal) <= sc_deadband_bp (default
      "0.25" bp, endpoints INSIDE the dead zone); v1 keeps sc_epsilon.

    Evaluation pipeline (fixed order, see spec v2.1):
    dedup by fill_key -> age eviction (basis: fill_ms) -> full-window
    threshold (sc_window_fills, never replaceable by sc_min_new_fills)
    -> fresh-evidence (eligible_new_count >= sc_min_new_fills) and adjust
    interval -> signal and decision. Dead zone / boundary / interval /
    insufficient-evidence outcomes never reset the fresh-evidence count;
    only an actual adjust resets last_adjust_seq. Every evaluation emits an
    sc_eval event with the multi-basis payload, whether or not it acts.
    """

    FEEDBACK_RATE_WINDOW_MS = 3600000
    HEARTBEAT_MIN_INTERVAL_MS = 3600000

    def __init__(self, account):
        self.account = account
        self.window = deque(maxlen=int(account.config["sc_window_fills"]))
        self.ticks = int(str(account.config.get("as_min_half_spread_ticks", "6")))
        self.last_adjust_ms = 0
        self.last_eval_ms = 0
        self.last_eval_reason = "warming_up"
        self.last_feedback_ms = 0
        self.feedback_seq = 0
        self.last_adjust_seq = 0
        self.feedback_times = deque()
        self.seen_fill_keys = set()
        self.last_heartbeat_ms = 0
        self.last_alive_ms = 0
        # Batch identity self-check field; set by Experiment at creation.
        self.batch_start_ms = None
        self.signal_mode = self._resolve_signal_variant(account.config)
        if self._full_threshold() < 1 or self._min_new_fills() < 1:
            raise ValueError("SC sample thresholds must be positive")
        self.account.sc_target_ticks = self.ticks

    @staticmethod
    def _resolve_signal_variant(config):
        # New key wins; legacy sc_signal is an alias so older S1 batch
        # configs (which predate sc_signal_variant) keep their intent.
        variant = config.get("sc_signal_variant")
        if variant is None:
            variant = ("v2_median_bp" if config.get("sc_signal") == "median_bp"
                       else "v1_mean_usdt")
        if variant not in ("v1_mean_usdt", "v2_median_bp"):
            raise ValueError("Unknown SC signal variant")
        return variant

    def _is_v2(self):
        return self.signal_mode == "v2_median_bp"

    def _signal(self, eligible):
        if not eligible:
            return ZERO
        if self._is_v2():
            return statistics.median(s["pnl"] / s["notional"] * 10000 for s in eligible)
        return sum((s["pnl"] for s in eligible), ZERO) / len(eligible)

    def _deadband(self):
        if self._is_v2():
            # Endpoints are INSIDE the dead zone: abs(signal) <= deadband
            # never adjusts (spec v0.2 section 4 gate).
            return D(self.account.config.get("sc_deadband_bp",
                                             self.account.config.get("sc_epsilon_bp", "0.25")))
        return D(self.account.config["sc_epsilon"])

    def _full_threshold(self):
        return int(self.account.config["sc_window_fills"])

    def _max_age_ms(self):
        return int(self.account.config.get("sc_window_max_age_min", 480)) * 60000

    def _min_new_fills(self):
        return int(self.account.config.get("sc_min_new_fills", 10))

    def _interval_ms(self):
        return int(self.account.config["sc_adjust_interval_min"]) * 60000

    def _bounds(self):
        return (int(self.account.config.get("sc_min_ticks", 3)),
                int(self.account.config.get("sc_max_ticks", 24)))

    def feed(self, pnl, now_ms, fill_ms=None, qty=None, notional=None,
             oid=None, fill_key=None, event=None):
        """Consume one non-missing markout@1s; returns an adjust dict or None.

        ``event`` is an optional callable(kind, ms, **fields) used to emit
        sc_eval / spread_adjusted audit events. Sample age is always measured
        from fill_ms (the fill's own market context), never from feedback
        arrival; every eviction reason reports this basis.
        """
        pnl = D(pnl)
        notional = D(notional) if notional is not None else None
        if (not pnl.is_finite() or
                (self._is_v2() and
                 (notional is None or not notional.is_finite() or notional <= ZERO))):
            if event is not None:
                event("sc_invalid_sample", now_ms, fill_key=fill_key, reason="nonfinite_or_missing_notional")
            return None
        if fill_key is not None and fill_key in self.seen_fill_keys:
            return None
        self.feedback_seq += 1
        self.window.append({
            "seq": self.feedback_seq,
            "fill_ms": fill_ms if fill_ms is not None else now_ms,
            "feedback_ms": now_ms,
            "pnl": D(pnl),
            "qty": D(qty) if qty is not None else None,
            "notional": D(notional) if notional is not None else None,
            "oid": oid,
            "fill_key": fill_key,
        })
        if fill_key is not None:
            self.seen_fill_keys.add(fill_key)
        self.last_feedback_ms = now_ms
        self.last_alive_ms = now_ms
        self.feedback_times.append(now_ms)
        cutoff = now_ms - self.FEEDBACK_RATE_WINDOW_MS
        while self.feedback_times and self.feedback_times[0] <= cutoff:
            self.feedback_times.popleft()
        # 1) age eviction (after dedup, which happened above).
        max_age = self._max_age_ms()
        aged_out = 0
        kept = deque(maxlen=self.window.maxlen)
        for sample in self.window:
            if now_ms - sample["fill_ms"] > max_age:
                aged_out += 1
            else:
                kept.append(sample)
        if aged_out:
            self.window = kept
        # 2) full-window threshold, 3) fresh evidence, 4) interval, 5) signal.
        eligible = list(self.window)
        eligible_count = len(eligible)
        eligible_new_count = sum(1 for s in eligible if s["seq"] > self.last_adjust_seq)
        mean = sum((s["pnl"] for s in eligible), ZERO) / eligible_count if eligible_count else ZERO
        signal_value = self._signal(eligible)
        epsilon = self._deadband()
        low, high = self._bounds()
        interval_blocked = bool(self.last_adjust_ms) and now_ms - self.last_adjust_ms < self._interval_ms()
        if eligible_count < self._full_threshold():
            decision, reason = "no_adjust", "warming_up"
        elif eligible_new_count < self._min_new_fills():
            decision, reason = "no_adjust", "insufficient_evidence"
        elif interval_blocked:
            decision, reason = "no_adjust", "waiting_interval"
        elif -epsilon <= signal_value <= epsilon:
            decision, reason = "no_adjust", "dead_zone"
        else:
            up = D(self.account.config.get("sc_up_factor", "1.5"))
            down = D(self.account.config.get("sc_down_factor", "1.25"))
            max_jump = int(self.account.config.get("sc_max_jump_ticks", 4))
            if signal_value < -epsilon:
                target = int((D(self.ticks) * up).to_integral_value(rounding=ROUND_HALF_UP))
            else:
                target = int((D(self.ticks) / down).to_integral_value(rounding=ROUND_HALF_UP))
            # Clamp a single jump in both directions: regime flips are symmetric shocks.
            target = max(self.ticks - max_jump, min(self.ticks + max_jump, target))
            target = max(low, min(high, target))
            if target == self.ticks:
                decision, reason = "no_adjust", "at_boundary"
            else:
                decision, reason = "adjust", "adjust"
        self.last_eval_ms = now_ms
        self.last_eval_reason = reason
        if event is not None:
            event("sc_eval", now_ms, **self._eval_payload(
                decision, reason, eligible, eligible_count, eligible_new_count,
                aged_out, mean, now_ms))
        if decision != "adjust":
            return None
        old_ticks, self.ticks = self.ticks, target
        self.last_adjust_ms = now_ms
        self.last_adjust_seq = self.feedback_seq
        self.account.sc_target_ticks = target
        change = {"old_ticks": old_ticks, "new_ticks": target,
                  "window_mean": str(mean), "n": eligible_count,
                  "eligible_new_count": eligible_new_count,
                  "signal": str(signal_value), "signal_mode": self.signal_mode}
        if event is not None:
            event("spread_adjusted", now_ms, **change)
        return change

    def _eval_payload(self, decision, reason, eligible, eligible_count,
                      eligible_new_count, aged_out, mean, now_ms):
        # Keep legacy bases for audit while explicitly identifying the signal.
        with_notional = [s for s in eligible if s["notional"] is not None and s["notional"] > ZERO]
        per_trade_bp = sorted(s["pnl"] / s["notional"] * 10000 for s in with_notional)
        notional_total = sum((s["notional"] for s in with_notional), ZERO)
        weighted_bp = (sum((s["pnl"] for s in with_notional), ZERO) / notional_total * 10000
                       if notional_total > ZERO else None)
        oid_counts = Counter(s["oid"] for s in eligible if s["oid"] is not None)
        distinct_orders = len(oid_counts)
        partial_orders = sum(1 for n in oid_counts.values() if n > 1)
        return {
            "decision": decision, "reason": reason,
            "signal": str(self._signal(eligible)), "signal_mode": self.signal_mode,
            "signal_bp": str(self._signal(eligible)) if self._is_v2() else None,
            "signal_unit": "bp" if self._is_v2() else "quote_currency",
            # Under v1 the per-trade bp median is audit-only; under v2 it IS
            # the control signal (spec v0.2 section 4).
            "median_bp_is_signal": self._is_v2(),
            "eligible_count": eligible_count, "eligible_new_count": eligible_new_count,
            "window_fills": self._full_threshold(), "min_new_fills": self._min_new_fills(),
            "aged_out": aged_out, "age_basis": "fill_ms",
            "seq": self.feedback_seq, "last_adjust_seq": self.last_adjust_seq,
            "mean": str(mean), "ticks": self.ticks, "last_adjust_ms": self.last_adjust_ms,
            "notional_total": str(notional_total),
            "per_trade_bp_mean": str(sum(per_trade_bp) / len(per_trade_bp)) if per_trade_bp else None,
            "notional_weighted_bp": str(weighted_bp) if weighted_bp is not None else None,
            "median_bp": str(D(str(statistics.median(per_trade_bp)))) if per_trade_bp else None,
            "distinct_orders": distinct_orders,
            "partial_fill_ratio": str(D(partial_orders) / D(distinct_orders)) if distinct_orders else None,
            "oldest_sample_age_ms": max((now_ms - s["fill_ms"] for s in eligible), default=None),
        }

    def state(self, now_ms):
        """Read-only status snapshot; never mutates and never adjusts."""
        max_age = self._max_age_ms()
        active = [s for s in self.window if now_ms - s["fill_ms"] <= max_age]
        if self.last_feedback_ms and now_ms - self.last_feedback_ms > max_age:
            # Feed gap beyond the evidence horizon comes first: a nominally
            # full window whose feedback has been silent that long is stale,
            # not merely warming.
            status = "stale_feedback"
        elif len(active) < self._full_threshold():
            status = "warming_up"
        elif self.last_eval_reason == "adjust":
            status = "ready"
        else:
            status = self.last_eval_reason
        rate_cutoff = now_ms - self.FEEDBACK_RATE_WINDOW_MS
        rate = sum(1 for ts in self.feedback_times if ts > rate_cutoff)  # explicit 0 when empty
        return {
            "status": status, "reason": self.last_eval_reason, "ticks": self.ticks,
            "window_samples": len(active), "window_capacity": self._full_threshold(),
            "oldest_sample_age_ms": (now_ms - min(s["fill_ms"] for s in active)) if active else None,
            "window_span_ms": (max(s["fill_ms"] for s in active) - min(s["fill_ms"] for s in active)) if active else None,
            "feedback_rate_per_hour": rate,
            "last_feedback_ms": self.last_feedback_ms or None,
            "since_last_feedback_ms": (now_ms - self.last_feedback_ms) if self.last_feedback_ms else None,
            "last_eval_ms": self.last_eval_ms or None,
            "since_last_eval_ms": (now_ms - self.last_eval_ms) if self.last_eval_ms else None,
        }

    def observe(self, now_ms, event=None):
        """Report-clock driven observation (spec v2-F1): updates the liveness
        checkpoint and emits a rate-limited sc_heartbeat while the controller
        cannot evaluate (warming / insufficient evidence / stale feedback).
        Never calls feed and never adjusts the spread."""
        self.last_alive_ms = now_ms
        state = self.state(now_ms)
        if state["status"] in ("warming_up", "insufficient_evidence", "stale_feedback"):
            if now_ms - self.last_heartbeat_ms >= self.HEARTBEAT_MIN_INTERVAL_MS:
                self.last_heartbeat_ms = now_ms
                if event is not None:
                    event("sc_heartbeat", now_ms, **state)
        return state

    def dump(self, now_ms):
        return {
            "start_ms": self.batch_start_ms,
            "ticks": self.ticks,
            "window": [{**s, "pnl": str(s["pnl"]),
                        "qty": str(s["qty"]) if s["qty"] is not None else None,
                        "notional": str(s["notional"]) if s["notional"] is not None else None}
                       for s in self.window],
            "last_adjust_ms": self.last_adjust_ms,
            "last_feedback_ms": self.last_feedback_ms,
            "last_eval_ms": self.last_eval_ms,
            "last_eval_reason": self.last_eval_reason,
            "feedback_seq": self.feedback_seq,
            "last_adjust_seq": self.last_adjust_seq,
            "last_heartbeat_ms": self.last_heartbeat_ms,
            # Feedback rate has an independent source: the deque is already
            # bounded to the last 60 minutes by feed(), so persist it whole
            # instead of reconstructing it from the truncated 50-sample
            # decision window.
            "feedback_times": list(self.feedback_times),
            # last_alive_ms is the save-cycle liveness checkpoint: it bounds
            # downtime from above (downtime <= now - last_alive_ms).
            "last_alive_ms": now_ms,
        }

    def load(self, data):
        self.ticks = int(data["ticks"])
        self.window = deque(
            ({**s, "pnl": D(s["pnl"]),
              "qty": D(s["qty"]) if s.get("qty") is not None else None,
              "notional": D(s["notional"]) if s.get("notional") is not None else None}
             for s in data["window"]),
            maxlen=self.window.maxlen)
        self.last_adjust_ms = int(data.get("last_adjust_ms") or 0)
        self.last_feedback_ms = int(data.get("last_feedback_ms") or 0)
        self.last_eval_ms = int(data.get("last_eval_ms") or 0)
        self.last_eval_reason = data.get("last_eval_reason", "warming_up")
        self.feedback_seq = int(data.get("feedback_seq") or 0)
        self.last_adjust_seq = int(data.get("last_adjust_seq") or 0)
        self.last_heartbeat_ms = int(data.get("last_heartbeat_ms") or 0)
        self.last_alive_ms = int(data.get("last_alive_ms") or 0)
        self.seen_fill_keys = {s["fill_key"] for s in self.window if s.get("fill_key") is not None}
        if data.get("feedback_times") is not None:
            self.feedback_times = deque(int(ts) for ts in data["feedback_times"])
        else:
            # Legacy checkpoints without the independent feedback-rate
            # record: fall back to the truncated decision window.
            self.feedback_times = deque(s["feedback_ms"] for s in self.window)
        return self


