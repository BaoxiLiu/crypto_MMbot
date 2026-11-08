"""Optional execution-aware constraints around the native AS calculator."""
from .core import D, ZERO


def apply_as_policy(bid, ask, mid, tick, policy):
    width = (ask - bid) / 2
    center = (ask + bid) / 2
    floor = D(policy.get("as_min_half_spread_ticks", "0")) * tick
    gain = D(policy.get("as_inventory_skew_multiplier", "1"))
    if floor < ZERO or gain < ZERO or not floor.is_finite() or not gain.is_finite():
        raise ValueError("Invalid AS policy")
    center = mid + (center - mid) * gain
    width = max(width, floor)
    bid, ask = center - width, center + width
    if bid <= ZERO or bid >= ask:
        raise ValueError("Invalid constrained AS quote")
    bound = D(policy.get("as_max_half_spread", ".015"))
    if max(abs(bid / mid - 1), abs(ask / mid - 1)) > bound:
        raise ValueError("Constrained AS quote exceeds distance limit")
    return bid, ask
