"""统一指标口径:全仓库唯一指标真相源(Phase 1,2026-10-31 起)。

口径依据 doc/backtesting.md 与根目录 doc/backtesting.md,指标定义见
doc/backtesting.md。新代码一律从这里取指标,不再各自重算。
"""
from .core import (
    MARKOUT_HORIZONS_MS,
    MARKOUT_MAX_LAG_MS,
    SHARPE_ANNUALIZATION_DAYS,
    SHARPE_MIN_DAILY_SAMPLES,
    aggregate_fills,
    aggregate_markouts,
    average_daily_turnover,
    calendar_days,
    daily_returns,
    max_drawdown,
    net_pnl_per_filled_notional_bp,
    reconcile_ledger,
    return_pct,
    sharpe_daily,
    summarize,
    turnover_initial_cash,
)
from .schema import SCHEMA_VERSION, results_document, to_markdown, write_results_json

__all__ = [
    'MARKOUT_HORIZONS_MS', 'MARKOUT_MAX_LAG_MS',
    'SHARPE_ANNUALIZATION_DAYS', 'SHARPE_MIN_DAILY_SAMPLES',
    'SCHEMA_VERSION',
    'aggregate_fills', 'aggregate_markouts', 'average_daily_turnover', 'calendar_days',
    'daily_returns', 'max_drawdown', 'net_pnl_per_filled_notional_bp', 'reconcile_ledger',
    'return_pct', 'sharpe_daily', 'summarize',
    'results_document', 'to_markdown', 'write_results_json',
]
