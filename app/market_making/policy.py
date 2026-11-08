"""Quote decision core shared by the live runner and the replay.

Pure functions only: no IO, no wall clock (time arrives as an argument).
Every target quote — in production, simulation and research replay — must be
derived through these helpers so research results cannot drift from live
behavior.
"""

from .core import D, ZERO
from .quote_policy import apply_as_policy


def microprice(bid, bid_size, ask, ask_size):
    """Size-weighted top-of-book price, or None when inputs are unusable."""
    if bid <= ZERO or ask <= ZERO or bid_size <= ZERO or ask_size <= ZERO:
        return None
    return (ask * bid_size + bid * ask_size) / (bid_size + ask_size)


def microprice_from_book(book):
    if not (book.ready and book.bids and book.asks and book.bid > ZERO and book.ask > ZERO):
        return None
    return microprice(book.bid, book.bids[book.bid], book.ask, book.asks[book.ask])


def fv_shift(micro, mid, tick, min_half_spread_ticks):
    """Fair-value displacement of the quote center, clamped to +-min_half_spread."""
    if micro is None:
        return ZERO
    clamp = tick * D(str(min_half_spread_ticks))
    return max(-clamp, min(clamp, micro - mid))


def fixed_quotes(mid, position, config, rules):
    width = mid * D(config["fixed_half_spread"])
    offset = rules.tick * D(config["fixed_inventory_shift_ticks"]) * max(
        D("-1"), min(D("1"), position / D(config["max_position_qty"])))
    return mid - offset - width, mid - offset + width


def as_quotes(model, book, position, now, config, rules, micro=None, floor_ticks=None):
    """Native AS target quotes with optional fair-value center shift.

    Returns (bid, ask, native_bid, native_ask, shift). Raises ValueError when
    native estimates are unavailable; the caller decides how to handle that.
    """
    native_bid, native_ask = model.quote(book, position, now)
    shift = ZERO
    if config.get("fair_value") == "micro":
        shift = fv_shift(micro, book.mid, rules.tick,
                         config.get("as_min_half_spread_ticks", "6"))
    # FV shift remains bounded by the immutable static floor. A changing
    # spread floor must not also change the center-shift treatment.
    quote_config = config if floor_ticks is None else {**config, "as_min_half_spread_ticks": str(floor_ticks)}
    bid, ask = apply_as_policy(native_bid + shift, native_ask + shift,
                               book.mid, rules.tick, quote_config)
    return bid, ask, native_bid, native_ask, shift
