"""Training-frozen B6/C6 thresholds (V3_SPEC §3, P1.5).

V3_SPEC freezes the B6 large-cancel threshold (p90 of cancel-event notional,
plus the 0.1x median-single-cancel floor) and the C6 large-trade threshold
(p90 of per-trade notional) from the TRAINING segment (2026-08-05 ~
2026-09-08), per coin. This module loads that frozen table; the builder falls
back to a provisional within-hour estimate only when a symbol is missing.
"""
import json
from pathlib import Path

from .schema import FactorParams

DEFAULT_PATH = Path(__file__).resolve().parent / "thresholds.json"


def load_thresholds(path=None):
    """symbol (upper) -> FactorParams from the frozen table; {} if absent."""
    p = Path(path) if path else DEFAULT_PATH
    if not p.exists():
        return {}
    raw = json.loads(p.read_text())
    table = {}
    for sym, vals in raw.get("thresholds", {}).items():
        table[sym.upper()] = FactorParams(
            cancel_p90_notional=vals.get("cancel_p90_notional"),
            median_cancel_qty=vals.get("median_cancel_qty"),
            trade_p90_notional=vals.get("trade_p90_notional"))
    return table


def save_thresholds(table, path=None, meta=None):
    """table: symbol -> FactorParams. Writes the frozen table JSON."""
    p = Path(path) if path else DEFAULT_PATH
    out = {
        "version": 1,
        "meta": meta or {},
        "thresholds": {sym.upper(): params.__dict__ for sym, params in table.items()},
    }
    p.write_text(json.dumps(out, indent=2, default=float))
    return p
