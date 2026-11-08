import json
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR


D = Decimal
ZERO = D("0")
DAY_ZONE = timezone(timedelta(hours=8))


def day(ms):
    return datetime.fromtimestamp(ms / 1000, DAY_ZONE).date().isoformat()


def encode(value):
    return json.dumps(value, default=str, ensure_ascii=False, allow_nan=False)


def floor_step(value, step):
    return (value / step).to_integral_value(rounding=ROUND_FLOOR) * step


def ceil_step(value, step):
    return (value / step).to_integral_value(rounding=ROUND_CEILING) * step


@dataclass
class Rules:
    tick: Decimal
    step: Decimal
    minimum_notional: Decimal
    minimum_qty: Decimal
    maximum_qty: Decimal
    price_multiplier_down: Decimal
    price_multiplier_up: Decimal

    @classmethod
    def from_exchange(cls, symbol):
        if (symbol["contractType"] != "PERPETUAL" or symbol["marginAsset"] not in ("USDT", "USDC")
                or symbol["status"] != "TRADING"):
            raise ValueError("Expected an active USDT/USDC perpetual contract")
        filters = {f["filterType"]: f for f in symbol["filters"]}
        return cls(
            D(filters["PRICE_FILTER"]["tickSize"]), D(filters["LOT_SIZE"]["stepSize"]),
            D(filters["MIN_NOTIONAL"]["notional"]), D(filters["LOT_SIZE"]["minQty"]),
            D(filters["LOT_SIZE"]["maxQty"]), D(filters["PERCENT_PRICE"]["multiplierDown"]),
            D(filters["PERCENT_PRICE"]["multiplierUp"]),
        )

    def price(self, value, side):
        return floor_step(value, self.tick) if side == 1 else ceil_step(value, self.tick)

    def valid(self, price, qty, mark, reduce_only=False):
        return (
            price > ZERO and price % self.tick == 0 and qty % self.step == 0
            and self.minimum_qty <= qty <= self.maximum_qty
            and (reduce_only or price * qty >= self.minimum_notional)
            and mark * self.price_multiplier_down <= price <= mark * self.price_multiplier_up
        )


class BookGap(ValueError):
    pass


class Book:
    def __init__(self):
        self.bids = {}
        self.asks = {}
        self.update_id = 0
        self.bridged = False
        self.ready = False
        self.received_ms = 0

    def snapshot(self, data):
        self.bids = {D(p): D(q) for p, q in data["bids"] if D(q) > ZERO}
        self.asks = {D(p): D(q) for p, q in data["asks"] if D(q) > ZERO}
        self.update_id = data["lastUpdateId"]
        self.bridged = False
        self.ready = False

    def diff(self, event, received_ms):
        if event["u"] < self.update_id or (self.bridged and event["u"] == self.update_id):
            return False
        if not self.bridged:
            if not event["U"] <= self.update_id <= event["u"]:
                self.ready = False
                raise BookGap("Snapshot does not bridge the first futures diff")
        elif event["pu"] != self.update_id:
            self.ready = False
            raise BookGap("Futures pu sequence gap")
        for target, updates in [(self.bids, event["b"]), (self.asks, event["a"])]:
            for p, q in updates:
                price, amount = D(p), D(q)
                if amount == ZERO:
                    target.pop(price, None)
                else:
                    target[price] = amount
        self.update_id = event["u"]
        self.bridged = True
        if not self.bids or not self.asks or self.bid >= self.ask:
            self.ready = False
            raise BookGap("Empty or crossed order book")
        self.ready = True
        self.received_ms = received_ms
        return True

    @property
    def bid(self):
        return max(self.bids)

    @property
    def ask(self):
        return min(self.asks)

    @property
    def mid(self):
        return (self.bid + self.ask) / 2

    def ahead(self, side, price):
        return (self.bids if side == 1 else self.asks).get(price, ZERO)


@dataclass
class Ledger:
    initial: Decimal = D("600")
    wallet: Decimal = D("600")
    position: Decimal = ZERO
    average_entry: Decimal = ZERO
    realized: Decimal = ZERO
    funding: Decimal = ZERO
    fees: Decimal = ZERO
    turnover: Decimal = ZERO
    fills: int = 0
    buys: int = 0
    sells: int = 0
    day_key: str = ""
    day_start_equity: Decimal = D("600")
    day_stopped: bool = False
    margin_breach: bool = False
    peak_equity: Decimal = D("600")
    max_drawdown: Decimal = ZERO

    def unrealized(self, mark):
        return self.position * (mark - self.average_entry)

    def equity(self, mark):
        return self.wallet + self.unrealized(mark)

    def fill(self, side, qty, price, fee_rate=ZERO):
        if side not in (-1, 1) or qty <= ZERO or price <= ZERO:
            raise ValueError("Invalid fill")
        signed = side * qty
        previous = self.position
        realized = ZERO
        if previous == ZERO or previous * signed > ZERO:
            self.average_entry = (abs(previous) * self.average_entry + qty * price) / (abs(previous) + qty)
        else:
            closed = min(abs(previous), qty)
            realized = closed * (price - self.average_entry) * (1 if previous > ZERO else -1)
            if qty > abs(previous):
                self.average_entry = price
            elif qty == abs(previous):
                self.average_entry = ZERO
        self.position += signed
        fee = qty * price * fee_rate
        self.realized += realized
        self.fees += fee
        self.wallet += realized - fee
        self.turnover += qty * price
        self.fills += 1
        self.buys += side == 1
        self.sells += side == -1
        return realized, fee

    def settle_funding(self, position_at_settlement, mark, rate, settlement_ms):
        payment = -position_at_settlement * mark * rate
        self.wallet += payment
        self.funding += payment
        if self.day_key and day(settlement_ms) < self.day_key:
            self.day_start_equity += payment
        return payment

    def risk(self, mark, now_ms, config):
        equity = self.equity(mark)
        if self.day_key != day(now_ms):
            self.day_key = day(now_ms)
            self.day_start_equity = equity
            self.day_stopped = False
        self.peak_equity = max(self.peak_equity, equity)
        self.max_drawdown = max(self.max_drawdown, self.peak_equity - equity)
        if equity - self.day_start_equity <= -D(config["daily_loss_limit"]):
            self.day_stopped = True
        # A full-notional collateral stress guard, not an exchange liquidation model.
        if equity <= abs(self.position) * mark / D(config["margin_leverage"]):
            self.margin_breach = True
        if self.margin_breach:
            return "margin_stress_breach_evaluation_halted"
        if self.day_stopped:
            return "daily_loss_stop_hold_position"
        return "ok"

    def report(self, mark, config):
        equity = self.equity(mark)
        total = equity - self.initial
        residual = total - (self.realized + self.unrealized(mark) + self.funding - self.fees)
        if abs(residual) > D("1e-18"):
            raise ArithmeticError("Perpetual equity reconciliation failed")
        return {
            **asdict(self), "mark_price": mark, "equity": equity, "total_pnl": total,
            "realized_trading_pnl": self.realized - self.fees,
            "unrealized_pnl": self.unrealized(mark), "funding_pnl": self.funding,
            "initial_margin_model": abs(self.position) * mark / D(config["margin_leverage"]),
            "day_pnl": equity - self.day_start_equity,
        }

    @classmethod
    def restore(cls, data):
        decimal_fields = {k for k, v in asdict(cls()).items() if isinstance(v, Decimal)}
        return cls(**{k: D(v) if k in decimal_fields else v for k, v in data.items()})


@dataclass
class Order:
    oid: str
    side: int
    price: Decimal
    remaining: Decimal
    sent_ms: int
    active_ms: int
    reduce_only: bool = False
    active: bool = False
    cancel_ms: int = 0
    ahead: Decimal = ZERO
    purpose: str = "market_make"
    # Half-spread floor (in tick units) actually read from config at order
    # creation. Recorded on order_sent so post-hoc analysis can attribute
    # each fill's feedback to the spread regime period it was quoted under.
    as_floor_ticks: int = 0
    static_floor_ticks: int = 0
    vol_floor_ticks: int = 0
    sc_target_ticks: int = 0
    floor_cap_ticks: int = 24
    floor_cap_hit: bool = False
    floor_reason: str = "static"
    # Why the composite was reduced below max(static, vol, sc):
    # "none" = not truncated, "cap" = clamped by floor_cap_ticks, "truncated"
    # reserved for any future non-cap truncation path.
    floor_trunc_reason: str = "none"
    vol_status: str = "disabled"
    sigma_ticks: str = ""


class Account:
    def __init__(self, name, config, rules, emit):
        self.name = name
        self.config = config
        self.rules = rules
        self.emit = emit
        initial = D(config["initial_collateral"])
        self.ledger = Ledger(initial=initial, wallet=initial, day_start_equity=initial, peak_equity=initial)
        self.orders = {}
        self.serial = 0
        self.last_action = {-1: 0, 1: 0}
        self.action_times = {"new_order": deque(), "cancel": deque()}
        self.last_rate_limit_event = {"new_order": 0, "cancel": 0}
        self.pause_buy_until = 0
        self.pause_sell_until = 0
        self.reason = "warmup"
        self.last_quote = {}
        self.inventory_mode = "normal"
        self.inventory_mode_since_ms = 0
        self.static_floor_ticks = int(config.get("as_min_half_spread_ticks", 0))
        self.sc_target_ticks = (self.static_floor_ticks if config.get("spread_control_enabled") else 0)
        self.vol_floor_ticks = (self.static_floor_ticks if config.get("s1_enabled") else 0)
        self.vol_status = "warming_up" if config.get("s1_enabled") else "disabled"
        self.vol_invalid_since_ms = None
        self.vol_last_valid_ms = None
        self.sigma_ticks = None
        self.vol_degraded = False
        self.floor_cap_ticks = int(config.get("as_floor_cap_ticks", 24))
        if not 0 <= self.static_floor_ticks <= self.floor_cap_ticks:
            raise ValueError("Static floor must be within floor cap")
        if config.get("s1_enabled"):
            k = D(config.get("s1_k", "2"))
            if not k.is_finite() or k <= ZERO:
                raise ValueError("S1 k must be finite and positive")
            low = int(config.get("vol_floor_min_ticks", 3))
            high = int(config.get("vol_floor_max_ticks", 24))
            if not 1 <= low <= high:
                raise ValueError("Invalid vol floor tick bounds")
            if self._degrade_after_ms() <= 0:
                raise ValueError("Invalid vol degraded window")
        if config.get("inventory_control_enabled", False):
            resume = D(config["inventory_resume_qty"])
            soft = D(config["inventory_soft_qty"])
            hard = D(config["inventory_hard_qty"])
            soft_notional = D(config["inventory_soft_order_notional"])
            order_notional = D(config["order_notional"])
            minimum_buffered = rules.minimum_notional * D("1.05")
            if not ZERO <= resume < soft < hard <= D(config["max_position_qty"]):
                raise ValueError("Invalid inventory thresholds")
            if not minimum_buffered <= soft_notional <= order_notional:
                raise ValueError("Invalid inventory soft order notional")
            if int(config["inventory_unwind_after_seconds"]) <= 0:
                raise ValueError("Invalid inventory unwind timeout")
            if D(config["inventory_unwind_improve_ticks"]) < ZERO:
                raise ValueError("Invalid inventory unwind price improvement")

    def event(self, kind, now_ms, **fields):
        self.emit(self.name, kind, now_ms, fields)

    def pending(self, side):
        return sum((o.remaining for o in self.orders.values() if o.side == side), ZERO)

    def allowed(self, side, qty, price, mark):
        worst = self.ledger.position + side * (self.pending(side) + qty)
        limit = min(D(self.config["max_position_qty"]), D(self.config["max_position_notional"]) / max(mark, price))
        # Opposite-side pending orders never release inventory capacity.
        if (side == 1 and worst > limit) or (side == -1 and worst < -limit):
            return False
        reducing = abs(worst) < abs(self.ledger.position)
        margin = abs(worst) * max(mark, price) / D(self.config["margin_leverage"])
        return reducing or margin <= self.ledger.equity(mark)

    def _action_allowed(self, kind, now_ms, force=False):
        windows = {
            "new_order": [(10000, "max_new_orders_10s"), (60000, "max_new_orders_60s")],
            "cancel": [(60000, "max_cancels_60s")],
        }[kind]
        configured = [(window, int(self.config[key])) for window, key in windows if key in self.config]
        times = self.action_times[kind]
        longest = max([60000] + [window for window, _ in configured])
        while times and now_ms - times[0] >= longest:
            times.popleft()
        if not force:
            for window, limit in configured:
                if sum(ts > now_ms - window for ts in times) >= limit:
                    if now_ms - self.last_rate_limit_event[kind] >= 1000:
                        self.event("action_rate_limited", now_ms, action=kind, window_ms=window, limit=limit)
                        self.last_rate_limit_event[kind] = now_ms
                    return False
        times.append(now_ms)
        return True

    def action_usage(self, now_ms):
        result = {}
        for kind, times in self.action_times.items():
            while times and now_ms - times[0] >= 60000:
                times.popleft()
            result[f"{kind}_60s"] = len(times)
            if kind == "new_order":
                result["new_order_10s"] = sum(ts > now_ms - 10000 for ts in times)
        return result

    def update_inventory_mode(self, now_ms):
        if not self.config.get("inventory_control_enabled", False):
            return "normal"
        previous = self.inventory_mode
        qty = abs(self.ledger.position)
        resume = D(self.config["inventory_resume_qty"])
        soft = D(self.config["inventory_soft_qty"])
        hard = D(self.config["inventory_hard_qty"])
        if previous == "normal":
            if qty >= hard:
                self.inventory_mode = "reduce_only"
            elif qty >= soft:
                self.inventory_mode = "reduce"
        elif previous == "reduce":
            if qty < resume:
                self.inventory_mode = "normal"
            elif qty >= hard:
                self.inventory_mode = "reduce_only"
        elif previous == "reduce_only":
            if qty < resume:
                self.inventory_mode = "normal"
            elif qty < soft:
                self.inventory_mode = "reduce"
        else:
            raise ValueError("Unknown inventory mode")
        if self.inventory_mode != previous:
            duration = max(0, now_ms - self.inventory_mode_since_ms) if self.inventory_mode_since_ms else 0
            self.inventory_mode_since_ms = now_ms
            self.event("inventory_mode_changed", now_ms, previous=previous, current=self.inventory_mode,
                       position=self.ledger.position, previous_duration_ms=duration)
        return self.inventory_mode

    def inventory_status(self, now_ms):
        enabled = bool(self.config.get("inventory_control_enabled", False))
        unwind = enabled and self.inventory_mode == "reduce_only" and self.inventory_mode_since_ms > 0 and (
            now_ms - self.inventory_mode_since_ms >= int(self.config["inventory_unwind_after_seconds"]) * 1000
        )
        return {"enabled": enabled, "mode": self.inventory_mode,
                "mode_since_ms": self.inventory_mode_since_ms, "unwind_active": unwind}

    def cancel(self, side, now_ms, force=False, reason="strategy_requote"):
        order = self.orders.get(side)
        if order and not order.cancel_ms and self._action_allowed("cancel", now_ms, force=force):
            order.cancel_ms = now_ms + self.config["cancel_latency_ms"]
            self.event("cancel_requested", now_ms, oid=order.oid, effective_ms=order.cancel_ms,
                       reason=reason, order_age_ms=max(0, now_ms - order.sent_ms))

    def cancel_all(self, now_ms):
        for side in list(self.orders):
            self.cancel(side, now_ms, force=True, reason="safety_cancel_all")

    def advance(self, now_ms, book):
        for side, order in list(self.orders.items()):
            if order.cancel_ms and now_ms >= order.cancel_ms:
                self.event("cancel_confirmed", now_ms, oid=order.oid)
                del self.orders[side]
            elif not order.active and now_ms >= order.active_ms and book.ready:
                if (side == 1 and order.price >= book.ask) or (side == -1 and order.price <= book.bid):
                    self.event("post_only_rejected", now_ms, oid=order.oid)
                    del self.orders[side]
                else:
                    order.active = True
                    order.ahead = book.ahead(side, order.price)
                    self.event("order_active", now_ms, **asdict(order))

    def floor_state(self):
        components = {"static": self.static_floor_ticks, "vol": self.vol_floor_ticks,
                      "sc": self.sc_target_ticks}
        raw = max(components.values())
        effective = min(self.floor_cap_ticks, raw)
        truncated = effective < raw
        return {"as_floor_ticks": effective, "static_floor_ticks": self.static_floor_ticks,
                "vol_floor_ticks": self.vol_floor_ticks, "sc_target_ticks": self.sc_target_ticks,
                "floor_cap_ticks": self.floor_cap_ticks, "floor_cap_hit": raw >= self.floor_cap_ticks,
                "floor_reason": "cap" if truncated else "+".join(k for k, v in components.items() if v == raw),
                "floor_trunc_reason": "cap" if truncated else "none",
                "vol_status": self.vol_status,
                "sigma_ticks": str(self.sigma_ticks) if self.sigma_ticks is not None else ""}

    def update_volatility(self, snapshot, now_ms):
        if not self.config.get("s1_enabled"):
            return
        low = int(self.config.get("vol_floor_min_ticks", 3))
        high = int(self.config.get("vol_floor_max_ticks", 24))
        previous = (self.vol_floor_ticks, self.vol_status)
        if snapshot["valid"]:
            self.sigma_ticks = D(snapshot["sigma_ticks"])
            target = int((D(self.config.get("s1_k", "2")) * self.sigma_ticks).to_integral_value(rounding=ROUND_CEILING))
            self.vol_floor_ticks = max(low, min(high, target))
            self.vol_last_valid_ms = now_ms
            self.vol_invalid_since_ms = None
            self.vol_degraded = False
            self.vol_status = "ready"
        elif self.vol_last_valid_ms is not None:
            if self.vol_invalid_since_ms is None:
                self.vol_invalid_since_ms = now_ms
            self.vol_status = "stale_frozen"
            # Spec v0.2 section 2: freeze the last valid floor and never let
            # the frozen value sit below the static floor.
            self.vol_floor_ticks = max(self.vol_floor_ticks, self.static_floor_ticks)
            degrade_ms = self._degrade_after_ms()
            if now_ms - self.vol_invalid_since_ms >= degrade_ms:
                self.vol_degraded = True
                self.vol_status = "degraded_paused"
                self.cancel_all(now_ms)
                # Cancel acknowledgement first, then static fallback. Remain
                # paused until valid sigma recovers; no stale-price reopening.
                if not self.orders:
                    self.vol_floor_ticks = self.static_floor_ticks
        if previous != (self.vol_floor_ticks, self.vol_status):
            kind = ("vol_floor_degraded" if self.vol_degraded else
                    "vol_floor_stale" if self.vol_status == "stale_frozen" else "vol_floor_updated")
            self.event(kind, now_ms, **self.floor_state(), missing_ratio=snapshot["missing_ratio"])

    def _degrade_after_ms(self):
        degraded_min = self.config.get("vol_degraded_min")
        if degraded_min is not None:
            return int(D(str(degraded_min)) * 60000)
        return int(self.config.get("s1_degrade_after_ms", 600000))

    def quote(self, bid, ask, mark, now_ms, book, blocked="", guard_return=None, fv=None,
              blocked_sides=None, observer=None, inventory_admission=None,
              candidate_price_transform=None):
        # Offline research hooks. Absent hooks preserve the live/default path.
        def observe(side, stage, **fields):
            if observer is not None:
                observer(side, stage, **fields)
        if self.vol_degraded:
            blocked = "vol_floor_degraded"
        risk = self.ledger.risk(mark, now_ms, self.config)
        self.reason = risk if risk != "ok" else blocked or "quoting"
        if risk != "ok" or blocked:
            if observer is not None:
                for side in (1,-1): observe(side,"global_risk_or_feed_block",reason=self.reason)
            self.cancel_all(now_ms)
            return
        if guard_return is not None:
            threshold = D(self.config["drop_threshold"])
            until = now_ms + self.config["drop_pause_seconds"] * 1000
            if guard_return <= -threshold:
                self.pause_buy_until = max(self.pause_buy_until, until)
            if guard_return >= threshold:
                self.pause_sell_until = max(self.pause_sell_until, until)
        mode = self.update_inventory_mode(now_ms)
        inventory = self.inventory_status(now_ms)
        self.last_quote = {"bid_raw": bid, "ask_raw": ask, "mid": book.mid,
                           "inventory_mode": mode, "unwind_active": inventory["unwind_active"]}
        for side, value in [(1, bid), (-1, ask)]:
            position = self.ledger.position
            increasing = position != ZERO and side * position > ZERO
            reducing = side * position < ZERO
            purpose = "market_make"
            order_notional = D(self.config["order_notional"])
            if inventory["enabled"] and mode == "reduce_only" and not reducing:
                observe(side,"inventory_reduce_only_cancel_intent",raw_price=value)
                self.cancel(side, now_ms, force=True, reason="inventory_reduce_only")
                continue
            if inventory["enabled"] and mode == "reduce" and increasing:
                soft = D(self.config["inventory_soft_qty"])
                hard = D(self.config["inventory_hard_qty"])
                minimum = D(self.config["inventory_soft_order_notional"])
                fraction = min(D("1"), max(ZERO, (abs(position) - soft) / (hard - soft)))
                order_notional = order_notional - fraction * (order_notional - minimum)
                purpose = "inventory_limited"
            if inventory["enabled"] and mode in ("reduce", "reduce_only") and reducing:
                purpose = "inventory_unwind" if inventory["unwind_active"] else "inventory_reduce"
                if inventory["unwind_active"]:
                    value += side * D(self.config["inventory_unwind_improve_ticks"]) * self.rules.tick
            price = self.rules.price(value, side)
            qty = floor_step(order_notional / price, self.rules.step) if price > ZERO else ZERO
            if inventory["enabled"] and increasing:
                hard_capacity = max(ZERO, D(self.config["inventory_hard_qty"]) - abs(position))
                qty = min(qty, floor_step(hard_capacity, self.rules.step))
            paused = now_ms < (self.pause_buy_until if side == 1 else self.pause_sell_until)
            if paused:
                # During an adverse move, closing the opposite exposure remains allowed.
                if side * self.ledger.position >= ZERO:
                    observe(side,"direction_pause_cancel_intent",raw_price=value,price=price,qty=qty)
                    self.cancel(side, now_ms, force=True, reason="direction_paused")
                    continue
                qty = min(qty, abs(self.ledger.position))
            if inventory["enabled"] and mode in ("reduce", "reduce_only") and reducing:
                qty = min(qty, abs(self.ledger.position))
            reduce_only = side * self.ledger.position < ZERO and qty <= abs(self.ledger.position)
            valid = self.rules.valid(price, qty, mark, reduce_only)
            valid = valid and (price < book.ask if side == 1 else price > book.bid)
            current = self.orders.get(side)
            if candidate_price_transform is not None:
                transformed = candidate_price_transform(
                    self, side, qty, price, mark, current, now_ms,
                    book=book, fv=fv, reduce_only=reduce_only, valid=valid)
                if transformed != price:
                    if (not isinstance(transformed, D) or not transformed.is_finite()
                            or not valid or not reduce_only or qty <= ZERO
                            or qty > abs(self.ledger.position) or side * self.ledger.position >= ZERO
                            or side * (transformed - price) <= ZERO
                            or not self.rules.valid(transformed, qty, mark, True)
                            or not (transformed < book.ask if side == 1 else transformed > book.bid)):
                        raise RuntimeError('Candidate price transform violated reducing-only price contract')
                    price = transformed
            if observer is not None:
                observe(side,"candidate",raw_price=value,price=price,qty=qty,valid=valid,
                        purpose=purpose,reduce_only=reduce_only,inventory_mode=mode,paused=paused)
            if blocked_sides and side in blocked_sides:
                # Preserve the source's explicitly enabled inventory reduction;
                # do not invent extra reduce/unwind behavior for legacy profiles.
                if inventory["enabled"] and mode in ("reduce","reduce_only") and reducing:
                    observe(side,"gate_inventory_reduction_exception",price=price,qty=qty,purpose=purpose)
                else:
                    observe(side,"prediction_gate_cancel_or_suppress_intent",price=price,qty=qty,
                            existing_oid=current.oid if current else None)
                    self.cancel(side, now_ms, force=True, reason="prediction_side_gate")
                    continue
            if inventory_admission is not None and inventory_admission(
                    self, side, qty, price, mark, current, now_ms):
                observe(side,"inventory_headroom_cancel_or_suppress_intent",price=price,qty=qty,
                        existing_oid=current.oid if current else None)
                self.cancel(side, now_ms, force=True,
                            reason=getattr(inventory_admission,"cancel_reason","inventory_headroom"))
                continue
            if current:
                # Keeper: defensive cancels are instant, so check them before any
                # requote logic; without fv the FV check degrades to the mid.
                if self.config.get("keeper_enabled") and not current.cancel_ms:
                    buffer = D(self.config.get("fv_buffer_ticks", "1")) * self.rules.tick
                    keeper_fv = fv if fv is not None else book.mid
                    if side * (current.price - keeper_fv) > buffer:
                        observe(side,"keeper_fv_cancel_intent",existing_oid=current.oid)
                        self.cancel(side, now_ms, reason="keeper_fv_cross")
                        continue
                    if side * (current.price - book.mid) > ZERO:
                        observe(side,"keeper_mid_cancel_intent",existing_oid=current.oid)
                        self.cancel(side, now_ms, reason="keeper_mid_cross")
                        continue
                limit = min(D(self.config["max_position_qty"]), D(self.config["max_position_notional"]) / max(price, mark))
                worst = self.ledger.position + side * self.pending(side)
                unsafe = (side == 1 and worst > limit) or (side == -1 and worst < -limit)
                unsafe = unsafe or (paused and not current.reduce_only)
                # One-step tolerance: floor boundary jitter from a small price move
                # must not churn the order; only a real size reduction requotes.
                policy_change = (current.purpose != purpose or current.reduce_only != reduce_only
                                 or current.remaining > qty + self.rules.step)
                retreat = D(self.config.get("adverse_requote_ticks", "0")) * self.rules.tick
                improve = D(self.config.get("favorable_requote_ticks", "0")) * self.rules.tick
                adverse = retreat > ZERO and side * (current.price - price) >= retreat
                favorable = improve > ZERO and side * (price - current.price) >= improve
                if unsafe or not valid or policy_change:
                    reason = "inventory_policy_change" if policy_change else "unsafe_or_invalid"
                    observe(side,"safety_cancel_intent",reason=reason,existing_oid=current.oid)
                    self.cancel(side, now_ms, force=True, reason=reason)
                elif adverse and not current.cancel_ms:
                    observe(side,"adverse_requote_cancel_intent",existing_oid=current.oid)
                    self.event("adverse_quote_cancel", now_ms, oid=current.oid, desired_price=price)
                    self.cancel(side, now_ms, reason="adverse_requote")
                elif favorable and not current.cancel_ms:
                    observe(side,"favorable_requote_cancel_intent",existing_oid=current.oid)
                    self.event("favorable_quote_cancel", now_ms, oid=current.oid, desired_price=price)
                    self.cancel(side, now_ms, reason="favorable_requote")
                elif (not self.config.get("keeper_enabled")
                      and self.config["quote_refresh_seconds"] > 0 and current.price != price
                      and now_ms - self.last_action[side] >= self.config["quote_refresh_seconds"] * 1000):
                    observe(side,"periodic_requote_cancel_intent",existing_oid=current.oid)
                    self.cancel(side, now_ms, reason="periodic_requote")
                else:
                    observe(side,"retain_existing_or_pending",existing_oid=current.oid,cancel_pending=bool(current.cancel_ms))
                continue
            # Keeper: never place a quote that is immediately adverse to fv.
            if self.config.get("keeper_enabled") and fv is not None and side * (price - fv) >= ZERO:
                observe(side,"keeper_dangerous_new_suppressed",price=price,qty=qty)
                self.reason = "keeper_dangerous_quote"
                continue
            if not valid or not self.allowed(side, qty, price, mark):
                observe(side,"invalid_or_inventory_new_suppressed",price=price,qty=qty,valid=valid)
                continue
            if now_ms - self.last_action[side] < self.config["quote_refresh_seconds"] * 1000:
                observe(side,"refresh_interval_new_suppressed",price=price,qty=qty)
                continue
            if not self._action_allowed("new_order", now_ms):
                observe(side,"action_rate_new_suppressed",price=price,qty=qty)
                continue
            self.serial += 1
            floor_fields = self.floor_state()
            order = Order(f"{self.name}-{now_ms}-{self.serial}", side, price, qty, now_ms,
                          now_ms + self.config["order_latency_ms"], reduce_only, purpose=purpose,
                          **floor_fields)
            self.orders[side] = order
            self.last_action[side] = now_ms
            self.event("order_sent", now_ms, **asdict(order))
            observe(side,"new_order_request",price=price,qty=qty,oid=order.oid)

    def trade(self, aggressor_side, trade_price, trade_qty, event_ms, now_ms, book, trade_id=None):
        self.advance(now_ms, book)
        order = self.orders.get(-aggressor_side)
        if not order or not order.active or event_ms < order.active_ms:
            return
        if (order.side == 1 and trade_price > order.price) or (order.side == -1 and trade_price < order.price):
            return
        available = trade_qty
        if trade_price == order.price:
            consumed = min(order.ahead, available)
            order.ahead -= consumed
            available -= consumed
        else:
            order.ahead = ZERO
        qty = min(order.remaining, available)
        if order.reduce_only:
            qty = min(qty, abs(self.ledger.position)) if order.side * self.ledger.position < ZERO else ZERO
        qty = floor_step(qty, self.rules.step)
        if qty <= ZERO:
            if order.reduce_only and order.side * self.ledger.position >= ZERO:
                self.cancel(order.side, now_ms)
            return
        realized, fee = self.ledger.fill(order.side, qty, order.price, D(self.config["maker_fee"]))
        order.remaining -= qty
        self.event("fill", now_ms, oid=order.oid, side=order.side, qty=qty, price=order.price,
                   fee=fee, realized=realized, event_ms=event_ms, position_after=self.ledger.position,
                   purpose=order.purpose, inventory_mode=self.inventory_mode,
                   reference_mid=book.mid, queue_model="risk_averse_trade_volume", simulated=True,
                   trade_id=trade_id, fill_key=f"{order.oid}:{trade_id}:{event_ms}:{order.remaining}",
                   order_age_ms=now_ms-order.active_ms, cancel_pending=bool(order.cancel_ms),
                   aggressor_price=trade_price, aggressor_qty=trade_qty,
                   book_age_ms=now_ms-book.received_ms, event_lag_ms=now_ms-event_ms)
        if order.remaining == ZERO:
            del self.orders[order.side]

    def dump(self):
        return {"ledger": asdict(self.ledger), "orders": [asdict(o) for o in self.orders.values()],
                "serial": self.serial, "pause_buy_until": self.pause_buy_until,
                "pause_sell_until": self.pause_sell_until, "last_action": self.last_action,
                "action_times": {kind: list(times) for kind, times in self.action_times.items()},
                "inventory_mode": self.inventory_mode,
                "inventory_mode_since_ms": self.inventory_mode_since_ms,
                "s1_state": {"vol_floor_ticks": self.vol_floor_ticks, "vol_status": self.vol_status,
                             "vol_invalid_since_ms": self.vol_invalid_since_ms,
                             "vol_last_valid_ms": self.vol_last_valid_ms,
                             "sigma_ticks": str(self.sigma_ticks) if self.sigma_ticks is not None else None,
                             "vol_degraded": self.vol_degraded}}

    def restore(self, data, now_ms):
        self.ledger = Ledger.restore(data["ledger"])
        self.serial = data["serial"]
        self.pause_buy_until = data["pause_buy_until"]
        self.pause_sell_until = data["pause_sell_until"]
        self.last_action = {int(k): v for k, v in data["last_action"].items()}
        self.action_times = {kind: deque(data.get("action_times", {}).get(kind, []))
                             for kind in ["new_order", "cancel"]}
        self.inventory_mode = data.get("inventory_mode", "normal")
        self.inventory_mode_since_ms = data.get("inventory_mode_since_ms", 0)
        if self.config.get("s1_enabled") and data.get("s1_state"):
            state = data["s1_state"]
            self.vol_floor_ticks = int(state["vol_floor_ticks"])
            self.vol_status = state["vol_status"]
            self.vol_invalid_since_ms = state["vol_invalid_since_ms"]
            self.vol_last_valid_ms = state["vol_last_valid_ms"]
            self.sigma_ticks = D(state["sigma_ticks"]) if state["sigma_ticks"] is not None else None
            self.vol_degraded = bool(state["vol_degraded"])
        for order in data["orders"]:
            self.event("restart_cancel_virtual_order", now_ms, oid=order["oid"])
        self.orders = {}


ACCOUNT_OVERRIDE_KEYS = {
    "type", "fv_source", "fast_trigger",
    "as_min_half_spread_ticks", "as_inventory_skew_multiplier",
    "adverse_requote_ticks", "favorable_requote_ticks", "quote_refresh_seconds",
    "event_driven_quotes", "max_new_orders_10s", "max_new_orders_60s",
    "max_cancels_60s", "order_latency_ms", "cancel_latency_ms",
    "inventory_control_enabled", "inventory_resume_qty",
    "inventory_soft_qty", "inventory_hard_qty", "inventory_soft_order_notional",
    "inventory_unwind_after_seconds", "inventory_unwind_improve_ticks",
    "fair_value", "keeper_enabled", "fv_buffer_ticks",
    "spread_control_enabled", "sc_window_fills", "sc_adjust_interval_min",
    "sc_step_ticks", "sc_min_ticks", "sc_max_ticks", "sc_epsilon",
    "sc_up_factor", "sc_down_factor", "sc_max_jump_ticks",
    "sc_min_new_fills", "sc_window_max_age_min", "sc_restore_stale_min",
    "sc_signal", "sc_epsilon_bp", "sc_signal_variant", "sc_deadband_bp",
    "s1_enabled", "s1_k",
    "vol_window_min", "vol_stale_ms", "vol_missing_ratio",
    "vol_degraded_min", "vol_floor_min_ticks", "vol_floor_max_ticks",
    "as_floor_cap_ticks", "s1_window_seconds", "s1_ticker_stale_ms",
    "s1_max_missing_ratio", "s1_degrade_after_ms", "require_s1_ready"}


def account_configs(config):
    # 显式账户列表（矩阵批次）：每个账户 {name, type, ...覆盖项}，覆盖项白名单制，
    # 资本与风控上限永远只能全局配置，不允许单账户放松。
    explicit = config.get("accounts")
    if explicit is not None:
        if not isinstance(explicit, list) or not explicit:
            raise ValueError("accounts must be a non-empty list")
        rows = []
        seen = set()
        for entry in explicit:
            if not isinstance(entry, dict) or not entry.get("name"):
                raise ValueError("each account needs a name")
            name = entry["name"]
            if name in seen:
                raise ValueError("duplicate account name: %s" % name)
            seen.add(name)
            overrides = {k: v for k, v in entry.items() if k != "name"}
            if set(overrides) - ACCOUNT_OVERRIDE_KEYS:
                raise ValueError("Account %s may only change quote policy, "
                                 "never capital or risk limits" % name)
            rows.append((name, {**config, **overrides}))
        return rows
    rows = [("fixed", config), ("as", config)]
    candidate = config.get("as_candidate")
    if candidate is not None:
        if not isinstance(candidate, dict) or set(candidate) - ACCOUNT_OVERRIDE_KEYS:
            raise ValueError("Candidate may only change quote policy, never capital or risk limits")
        rows.append(("as_candidate", {**config, **candidate}))
    return rows


