"""results.json 固定输出格式(schema_version=1)与 Markdown 报告段。

每个指标带单位、分母、样本数和口径定义;缺失值输出 null 并在 *_reason 注明,
不填 0、不省略字段。
"""
import json
from decimal import Decimal as D
from pathlib import Path

from .core import MARKOUT_HORIZONS_MS, MARKOUT_MAX_LAG_MS, SHARPE_MIN_DAILY_SAMPLES

SCHEMA_VERSION = 1

DEFINITIONS = {
    'net_pnl': 'USDC;期末权益−初始资金,含已实现/未实现盈亏、手续费与资金费,期末按可见价估值不强平',
    'return_pct': '%;(期末权益÷初始资金−1)×100,不年化',
    'net_pnl_per_filled_notional_bp': 'bp;净PnL÷买卖成交额之和×10000;分母不除2,含费用与资金费;无成交为null',
    'turnover': '倍;买卖成交额之和÷初始资金;不除2、不年化',
    'average_daily_turnover': '倍/日;turnover÷请求区间日历天数;局部回放为null',
    'sharpe_daily': f'日收益均值÷样本标准差(ddof=1)×√365,无风险利率0;日样本<{SHARPE_MIN_DAILY_SAMPLES}或零方差为null',
    'max_drawdown': 'USDC;权益序列(前置初始资金)相对历史峰值的最大下跌',
    'fill_rate': 'OMS增量成交回报数÷全部新单请求数;不是订单级任一成交率',
    'markouts': f'成交额加权markout(bp),side×(未来mid−成交价)÷成交价×10000;期限{MARKOUT_HORIZONS_MS}ms,'
                f'取价滞后预算{MARKOUT_MAX_LAG_MS}ms,删失不补0;不含费用和资金费',
}


def _json(value):
    if isinstance(value, D):
        return str(value)
    raise TypeError(f'Unexpected metric type {type(value)}')


def results_document(result, *, run_id=None, notes=None):
    """MetricSet → results.json dict。缺失指标输出 null 并附 *_reason。"""
    document = {
        'schema_version': SCHEMA_VERSION,
        'run_id': run_id,
        'window': {'start': result.window_start, 'end_exclusive': result.window_end_exclusive,
                   'timezone': 'UTC', 'calendar_days': result.calendar_days},
        'initial_cash': _json(result.initial_cash),
        'metrics': {
            'net_pnl': _json(result.net_pnl) if result.net_pnl is not None else None,
            'net_pnl_reason': result.net_pnl_reason,
            'return_pct': _json(result.return_pct) if result.return_pct is not None else None,
            'net_pnl_per_filled_notional_bp': (_json(result.net_pnl_per_filled_notional_bp)
                                               if result.net_pnl_per_filled_notional_bp is not None else None),
            'turnover': _json(result.turnover) if result.turnover is not None else None,
            'average_daily_turnover': (_json(result.average_daily_turnover)
                                       if result.average_daily_turnover is not None else None),
            'sharpe_daily': result.sharpe_daily,
            'sharpe_daily_samples': result.sharpe_daily_samples,
            'sharpe_daily_reason': result.sharpe_daily_reason,
            'max_drawdown': result.max_drawdown,
            'max_drawdown_reason': result.max_drawdown_reason,
            'fill_rate': result.fill_rate,
            'fill_rate_reason': result.fill_rate_reason,
        },
        'activity': {
            'fill_receipts': result.fill_receipts,
            'total_notional': _json(result.total_notional),
            'total_qty': _json(result.total_qty),
            'fees': _json(result.fees),
            'funding': _json(result.funding),
            'net_position_qty': _json(result.net_position_qty),
        },
        'markouts': [dict(row, notional_weighted_bp=(_json(row['notional_weighted_bp'])
                                                     if row['notional_weighted_bp'] is not None else None))
                     for row in result.markouts],
        'reconciliation': result.reconciliation,
        'definitions': DEFINITIONS,
        'notes': notes,
    }
    return document


def write_results_json(path, result, *, run_id=None, notes=None):
    document = results_document(result, run_id=run_id, notes=notes)
    Path(path).write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return document


def _fmt(value, digits=6):
    if value is None:
        return '缺失'
    if isinstance(value, D):
        return f'{value:.{digits}f}'
    return f'{value:.{digits}f}' if isinstance(value, float) else str(value)


def to_markdown(result):
    """人类可读报告段:固定字段表 + 七期限 markout 表。"""
    lines = ['| 指标 | 数值 | 口径 |', '|---|---:|---|']
    rows = [
        ('净PnL (USDC)', result.net_pnl, DEFINITIONS['net_pnl']),
        ('区间收益 (%)', result.return_pct, DEFINITIONS['return_pct']),
        ('单位成交净收益 (bp)', result.net_pnl_per_filled_notional_bp,
         DEFINITIONS['net_pnl_per_filled_notional_bp']),
        ('累计换手 (倍)', result.turnover, DEFINITIONS['turnover']),
        ('平均日换手 (倍/日)', result.average_daily_turnover, DEFINITIONS['average_daily_turnover']),
        ('日度夏普', result.sharpe_daily, DEFINITIONS['sharpe_daily']),
        ('最大回撤 (USDC)', result.max_drawdown, DEFINITIONS['max_drawdown']),
        ('成交率', result.fill_rate, DEFINITIONS['fill_rate']),
    ]
    reasons = {'净PnL (USDC)': result.net_pnl_reason, '日度夏普': result.sharpe_daily_reason,
               '最大回撤 (USDC)': result.max_drawdown_reason, '成交率': result.fill_rate_reason}
    for name, value, definition in rows:
        reason = reasons.get(name)
        shown = _fmt(value) if value is not None else f'缺失({reason})' if reason else '缺失'
        lines.append(f'| {name} | {shown} | {definition} |')
    lines += ['', f"成交回报 {result.fill_receipts} 笔;总成交额 {result.total_notional} USDC;"
              f"费用 {result.fees} USDC;资金费 {result.funding} USDC;"
              f"净持仓 {result.net_position_qty};对账:{result.reconciliation}。", '',
              '| 期限 | 有效样本 | 删失样本 | 成交额加权 markout (bp) | 平均取价滞后 |', '|---|---:|---:|---:|---:|']
    for row in result.markouts:
        bp = _fmt(row['notional_weighted_bp'], 4)
        lag = f"{row['mean_lag_ns'] / 1e6:.1f}ms" if row['mean_lag_ns'] is not None else '缺失'
        label = f"{row['horizon_ms']}ms" if row['horizon_ms'] < 1000 else f"{row['horizon_ms'] / 1000:g}s"
        lines.append(f"| {label} | {row['observed']} | {row['censored']} | {bp} | {lag} |")
    return '\n'.join(lines) + '\n'
