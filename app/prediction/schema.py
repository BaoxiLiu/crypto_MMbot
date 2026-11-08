"""Kimi v3 factor schema constants (V3_SPEC v1.1, 2026-10-09).

This module is the single place that names the 50 new factors (A8/B6/C10/D15/E11)
and freezes the pipeline gate thresholds from V3_SPEC §3. It imports nothing from
the frozen backtesting paths; V3_SPEC §5 (engineering identity) requires the new
prediction line to stay independent.
"""
from dataclasses import dataclass
from typing import Optional

SPEC_VERSION = "v1.1"

EPS = 1e-12

# P1 pipeline gates (V3_SPEC §3, frozen with the implementation).
CAPTURE_WINDOW_MS = 200
CAPTURE_MIN_RATE = 0.90

FACTOR_NAMES = tuple(
    [f"A{i}" for i in range(1, 9)]
    + [f"B{i}" for i in range(1, 7)]
    + [f"C{i}" for i in range(1, 11)]
    + [f"D{i}" for i in range(1, 16)]
    + [f"E{i}" for i in range(1, 12)]
)
assert len(FACTOR_NAMES) == 50

FACTOR_GROUPS = {
    "A": tuple(f"A{i}" for i in range(1, 9)),
    "B": tuple(f"B{i}" for i in range(1, 7)),
    "C": tuple(f"C{i}" for i in range(1, 11)),
    "D": tuple(f"D{i}" for i in range(1, 16)),
    "E": tuple(f"E{i}" for i in range(1, 12)),
}

# Auxiliary per-second quantities needed to build the T1/T2/T3/T5 labels in P2
# (labels themselves are computed from the clean 1s mid grid in P2).
AUX_COLUMNS = (
    "buy_n_1s", "sell_n_1s", "n_trades_1s",
    "buy_qty_1s", "sell_qty_1s", "buy_vwap_1s", "sell_vwap_1s",
    "mark_price", "funding_rate", "next_funding_time",
)

GRID_COLUMNS = ("sec", "mid", "best_bid", "best_ask", "spread_bp")


@dataclass(frozen=True)
class FactorParams:
    """Per-symbol constants that V3_SPEC freezes from the training segment.

    - cancel_p90_notional: B6 threshold, p90 of cancel-event notional (v1.1
      nominal-notional basis, per-coin).
    - median_cancel_qty: B6 floor = 0.1 x median single cancel qty.
    - trade_p90_notional: C6 large-trade threshold (p90 of per-trade notional).

    P1 has no training segment yet; the builder estimates them within the hour
    (provisional, recorded in metadata). The state machine only consumes frozen
    constants, so P2 can swap in training-frozen tables without code changes.
    """

    cancel_p90_notional: Optional[float] = None
    median_cancel_qty: Optional[float] = None
    trade_p90_notional: Optional[float] = None
