"""Public entry for the existing causal NumPy paper173 reference calculator.

A correctness/batch interface; not certified to meet the live 100us budget.
The caller supplies as-of public observations and a matching full book ladder.
"""
from .paper173_schema import FEATURE_NAMES
from .paper173_core import Observations, compute, impact_bp


def compute_named(obs, now_ns, side, bid_prices, bid_qty, ask_prices, ask_qty):
    """Return all 173 ordered names, keeping NaN for unavailable quantities."""
    values=compute(obs,now_ns,side,bid_prices,bid_qty,ask_prices,ask_qty)
    return dict(zip(FEATURE_NAMES,values,strict=True))


__all__=['FEATURE_NAMES','Observations','compute','compute_named','impact_bp']
