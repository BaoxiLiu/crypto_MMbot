"""统一指标计算核心。唯一真相源,禁止调用方各自重算。

口径(与 doc/backtesting.md、根目录 doc/backtesting.md 一致):

- 成交额 total_notional = 本账户全部增量成交 price×qty,买卖相加,不额外乘 2。
- 换手 turnover = total_notional ÷ initial_cash;平均日换手 = turnover ÷ 请求区间
  日历天数;不除 2、不年化、不按缺口外推;局部回放由调用方决定不填全区间日均。
- 单位成交净收益 net_pnl_per_filled_notional_bp = 净PnL ÷ total_notional × 10000;
  净PnL 含费用、资金费和期末未实现盈亏;无成交(None 或 0)时为 None。
- 区间收益 return_pct = (期末权益 ÷ 初始资金 − 1) × 100,单位 %。
- 日度 Sharpe = 连续 UTC 日收益均值 ÷ 样本标准差(ddof=1) × √365;日收益 = 当日
  PnL ÷ 前一日收盘权益(首日为初始资金),无风险利率 0。样本 <20 或零方差输出
  None 并在 sharpe_daily_reason 注明。
- 最大回撤 = 权益序列(前置初始资金)相对历史峰值的最大下跌金额,单位 USDC。
- Markout 固定七期限,取价滞后预算 250ms;删失样本不补 0。聚合按成交额加权:
  Σ(bp×notional) ÷ Σ(notional),不含费用和资金费。

金额类输入接受 Decimal/int/float/str,内部一律转 Decimal,输出 Decimal 保证与
既有报告 str(Decimal) 序列化逐字节一致;统计类(Sharpe、回撤时刻)用 float。
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal as D
import math

MARKOUT_HORIZONS_MS = (100, 500, 1000, 5000, 10000, 30000, 60000)
MARKOUT_MAX_LAG_MS = 250
SHARPE_MIN_DAILY_SAMPLES = 20
SHARPE_ANNUALIZATION_DAYS = 365
NS_PER_DAY = 86_400_000_000_000


def position_holding_periods(fills, end_ns):
    """Flat-to-flat position episodes from incremental fill receipts.

    Reversing through zero closes one episode and opens the next at the same
    receipt. The final open episode is censored, never counted as a closed
    trade. Durations are elapsed model-position time, including certified
    market gaps across which this replay carries inventory.
    """
    position=D(0);opened=None;durations=[];last=-1
    for fill in fills:
        t=int(fill['ns']);side=int(fill['side']);qty=D(str(fill['qty']))
        if t<last or t>end_ns or side not in (-1,1) or not qty.is_finite() or qty<=0:
            raise ValueError('Invalid holding-period fill sequence')
        previous=position;position+=side*qty
        if previous==0 and position!=0:opened=t
        elif previous!=0 and (position==0 or previous*position<0):
            durations.append(t-opened);opened=t if position else None
        last=t
    return dict(closed_episodes=len(durations),
      mean_closed_seconds=str(D(sum(durations))/D(len(durations))/D(10**9)) if durations else None,
      max_closed_seconds=str(D(max(durations))/D(10**9)) if durations else None,
      open_episode_seconds=str(D(end_ns-opened)/D(10**9)) if opened is not None else None,
      terminal_position=str(position),clock='local incremental fill receipt; elapsed including carried-position gaps')


def aggregate_independent_accounts(rows,days,*,full_range=True):
    """Terminal-value sums by counterfactual strategy, never sums of Sharpe/DD."""
    groups={}
    for row in rows:groups.setdefault(row['strategy'],[]).append(row)
    result=[]
    for strategy,group in groups.items():
        if len({r['symbol'] for r in group})!=len(group):raise ValueError('Duplicate coin in strategy aggregate')
        def complete_sum(key):
            return sum((_d(r[key]) for r in group),D(0)) if all(r.get(key) is not None for r in group) else None
        pnl=complete_sum('pnl');volume=complete_sum('total_notional');cash=complete_sum('initial_cash')
        result.append(dict(strategy=strategy,coins=len(group),pnl=str(pnl) if pnl is not None else None,
          total_notional=str(volume) if volume is not None else None,initial_cash=str(cash) if cash is not None else None,
          net_bp=str(net_pnl_per_filled_notional_bp(pnl,volume)) if pnl is not None and volume else None,
          average_daily_turnover=str(average_daily_turnover(volume,cash,days)) if full_range and volume is not None and cash else None,
          valuation='Sum of independent account terminal marks; final observed mark times can differ; no aligned portfolio Sharpe/drawdown inferred'))
    return result


def _d(value):
    if value is None:
        return None
    result = D(str(value)) if isinstance(value, float) else D(value)
    if not result.is_finite():
        raise ValueError('Non-finite metric input')
    return result


def calendar_days(start, end_exclusive):
    """请求区间日历天数,UTC 半开区间 [start, end_exclusive)。"""
    days = (datetime.fromisoformat(end_exclusive) - datetime.fromisoformat(start)).days
    if days <= 0:
        raise ValueError('Window must span at least one calendar day')
    return days


def net_pnl_per_filled_notional_bp(net_pnl, total_notional):
    """净PnL ÷ 买卖成交额之和 × 10000,单位 bp;无成交时 None。"""
    notional = _d(total_notional)
    if notional is None or notional <= 0:
        return None
    return _d(net_pnl) / notional * 10000


def turnover_initial_cash(total_notional, initial_cash):
    """买卖成交额之和 ÷ 初始资金,单位倍;无成交记录(None)时 None。"""
    notional = _d(total_notional)
    if notional is None:
        return None
    cash = _d(initial_cash)
    if cash <= 0:
        raise ValueError('Initial cash must be positive')
    return notional / cash


def average_daily_turnover(total_notional, initial_cash, days):
    """累计换手 ÷ 请求区间日历天数,单位倍/日;无成交记录时 None。"""
    multiple = turnover_initial_cash(total_notional, initial_cash)
    if multiple is None:
        return None
    if days <= 0:
        raise ValueError('Calendar days must be positive')
    return multiple / days


def return_pct(net_pnl, initial_cash):
    """区间收益 = 净PnL ÷ 初始资金 × 100,单位 %。"""
    cash = _d(initial_cash)
    if cash <= 0:
        raise ValueError('Initial cash must be positive')
    return _d(net_pnl) / cash * 100


def daily_returns(daily_pnl, initial_cash):
    """日收益 = 当日PnL ÷ 前一日收盘权益(首日前为初始资金)。pnl 求全区间覆盖。"""
    prior = _d(initial_cash)
    if prior <= 0:
        raise ValueError('Initial cash must be positive')
    returns = []
    for pnl in daily_pnl:
        pnl = _d(pnl)
        if prior <= 0:
            raise ValueError('Prior-day equity must be positive for daily returns')
        returns.append(pnl / prior)
        prior += pnl
    return returns


def sharpe_daily(returns, min_samples=SHARPE_MIN_DAILY_SAMPLES):
    """日度 Sharpe:均值 ÷ 样本标准差(ddof=1) × √365,无风险利率 0。

    返回 dict(value, samples, reason);样本不足 min_samples 或零方差时 value=None
    并给出中文原因。禁止使用逐秒/逐笔 PnL 年化(doc/backtesting.md 禁令)。
    """
    values = [D(str(r)) if isinstance(r, float) else D(r) for r in returns]
    n = len(values)
    if n < min_samples:
        return dict(value=None, samples=n,
                    reason=f'日收益样本 {n} 个,少于最少 {min_samples} 个;短窗口不输出年化夏普')
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    if variance == 0:
        return dict(value=None, samples=n, reason='日收益零方差,夏普无定义')
    sharpe = mean / variance.sqrt() * D(SHARPE_ANNUALIZATION_DAYS).sqrt()
    return dict(value=float(sharpe), samples=n, reason=None)


def max_drawdown(timestamps_ns, equities, initial_cash):
    """权益序列(前置初始资金)相对历史峰值的最大下跌金额。

    返回 dict(max_drawdown, peak_ns, trough_ns, peak_equity, trough_equity);
    金额为 USDC,峰值取首个达到历史最高的样本。
    """
    values = [float(initial_cash)] + [float(e) for e in equities]
    if not equities:
        raise ValueError('No equity observations')
    times = [int(timestamps_ns[0])] + [int(t) for t in timestamps_ns]
    if any(not math.isfinite(v) for v in values):
        raise ValueError('Non-finite equity sample')
    peaks = []
    running = values[0]
    for value in values:
        running = max(running, value)
        peaks.append(running)
    drops = [peak - value for peak, value in zip(peaks, values)]
    trough = drops.index(max(drops))
    peak = values.index(max(values[:trough + 1]))
    return dict(max_drawdown=drops[trough], peak_ns=times[peak], trough_ns=times[trough],
                peak_equity=values[peak], trough_equity=values[trough])


def aggregate_fills(fills):
    """汇总逐笔成交。fill 字段:ns/receipt_ns(int)、side(±1)、price、qty、fee、
    funding(缺省 0,含资金费的账本把资金费现金流记在此)。"""
    total = dict(fill_receipts=0, buy_qty=D(0), sell_qty=D(0), buy_notional=D(0),
                 sell_notional=D(0), fees=D(0), funding=D(0))
    last_ns = -1
    for fill in fills:
        ns = fill.get('ns', fill.get('receipt_ns'))
        price, qty, fee = _d(fill['price']), _d(fill['qty']), _d(fill['fee'])
        funding = _d(fill.get('funding', 0))
        if type(ns) is not int or ns < last_ns or fill['side'] not in (-1, 1) \
                or price <= 0 or qty <= 0:
            raise ValueError('Invalid fill observation')
        last_ns = ns
        side = 'buy' if fill['side'] == 1 else 'sell'
        total['fill_receipts'] += 1
        total[side + '_qty'] += qty
        total[side + '_notional'] += price * qty
        total['fees'] += fee
        total['funding'] += funding
    total['total_qty'] = total['buy_qty'] + total['sell_qty']
    total['total_notional'] = total['buy_notional'] + total['sell_notional']
    total['net_qty'] = total['buy_qty'] - total['sell_qty']
    return total


def reconcile_ledger(volume, ledger):
    """账本三重对账:成交计数、费用、库存,另核现金恒等式。

    ledger 字段:fill_events/fill_receipts、fees、funding、position、cash_flow
    (cash_flow = 卖成交额 − 买成交额 − 费用 + 资金费)。不一致抛 AssertionError。
    """
    expected = ledger.get('fill_events', ledger.get('fill_receipts'))
    if expected is not None:
        assert volume['fill_receipts'] == expected, 'Fill count reconciliation failed'
    assert volume['fees'] == _d(ledger['fees']), 'Fee reconciliation failed'
    assert volume['net_qty'] == _d(ledger['position']), 'Inventory reconciliation failed'
    assert volume['funding'] == _d(ledger['funding']), 'Funding reconciliation failed'
    cash = (volume['sell_notional'] - volume['buy_notional']
            - volume['fees'] + volume['funding'])
    assert abs(cash - _d(ledger['cash_flow'])) <= D('1e-18'), 'Cash reconciliation failed'


def aggregate_markouts(records, horizons=MARKOUT_HORIZONS_MS,
                       lag_budget_ms=MARKOUT_MAX_LAG_MS):
    """聚合逐笔 markout。record 字段:horizon_ms、notional;observed 样本另含
    markout_bp、lag_ns;删失样本 markout_bp=None 并给 reason。按成交额加权。"""
    groups = {h: dict(horizon_ms=h, observed=0, censored=0, notional=D(0),
                      weighted=D(0), lag_sum=0, reasons={}) for h in horizons}
    for record in records:
        h = record['horizon_ms']
        if h not in groups:
            raise ValueError(f'Horizon {h}ms outside unified {tuple(horizons)}')
        group = groups[h]
        notional = _d(record['notional'])
        bp = record.get('markout_bp')
        if bp is None:
            group['censored'] += 1
            reason = record.get('reason', 'unspecified')
            group['reasons'][reason] = group['reasons'].get(reason, 0) + 1
            continue
        lag = record.get('lag_ns')
        if lag is None or lag < 0 or lag > lag_budget_ms * 1_000_000:
            raise ValueError(f'Observed markout lag {lag} exceeds budget {lag_budget_ms}ms')
        group['observed'] += 1
        group['notional'] += notional
        group['weighted'] += _d(bp) * notional
        group['lag_sum'] += lag
    result = []
    for h in horizons:
        group = groups[h]
        result.append(dict(horizon_ms=h, lag_budget_ms=lag_budget_ms,
                           observed=group['observed'], censored=group['censored'],
                           notional_weighted_bp=(group['weighted'] / group['notional']
                                                 if group['notional'] else None),
                           mean_lag_ns=(group['lag_sum'] / group['observed']
                                        if group['observed'] else None),
                           censor_reasons=dict(sorted(group['reasons'].items()))))
    return result


@dataclass
class MetricSet:
    """统一指标结果。Decimal 字段由 schema 序列化为字符串;None 一律带原因。"""

    initial_cash: D
    window_start: str
    window_end_exclusive: str
    calendar_days: int
    fill_receipts: int
    total_notional: D
    total_qty: D
    fees: D
    funding: D
    net_position_qty: D
    net_pnl: D = None
    net_pnl_reason: str = None
    return_pct: D = None
    net_pnl_per_filled_notional_bp: D = None
    turnover: D = None
    average_daily_turnover: D = None
    sharpe_daily: float = None
    sharpe_daily_samples: int = 0
    sharpe_daily_reason: str = None
    max_drawdown: float = None
    max_drawdown_reason: str = None
    fill_rate: float = None
    fill_rate_reason: str = None
    markouts: list = field(default_factory=list)
    reconciliation: str = 'not_performed'


def summarize(fills=(), equity=None, markouts=(), *, initial_cash, window_start,
              window_end_exclusive, ledger=None, order_requests=None, full_range=True,
              funding_cashflows=(), markout_horizons=MARKOUT_HORIZONS_MS,
              markout_lag_budget_ms=MARKOUT_MAX_LAG_MS):
    """从统一输入计算全部固定输出字段。

    fills:逐笔成交(见 aggregate_fills);funding_cashflows:不依附于成交的带符号
    资金费现金流列表(如事后固定路径叠加),计入 funding 与对账;equity:[(ns, 权益)],
    权益必须已含资金费与期末未实现盈亏,否则 net_pnl/return_pct/drawdown/sharpe
    为 None 并注明;markouts:逐笔 markout(见 aggregate_markouts);ledger:账本
    对账字段;order_requests:全部新单请求数(用于 fill_rate);full_range=False 时
    average_daily_turnover=None(局部回放不折算全区间日均)。
    """
    cash = _d(initial_cash)
    days = calendar_days(window_start, window_end_exclusive)
    volume = aggregate_fills(fills)
    for cashflow in funding_cashflows:
        volume['funding'] += _d(cashflow)
    reconciliation = 'not_performed'
    if ledger is not None:
        reconcile_ledger(volume, ledger)
        reconciliation = 'fills/fees/inventory/funding/cash reconciled'
    result = MetricSet(initial_cash=cash, window_start=window_start,
                       window_end_exclusive=window_end_exclusive, calendar_days=days,
                       fill_receipts=volume['fill_receipts'],
                       total_notional=volume['total_notional'], total_qty=volume['total_qty'],
                       fees=volume['fees'], funding=volume['funding'],
                       net_position_qty=volume['net_qty'], reconciliation=reconciliation)
    result.turnover = turnover_initial_cash(volume['total_notional'], cash)
    result.average_daily_turnover = (average_daily_turnover(volume['total_notional'], cash, days)
                                     if full_range else None)
    if equity:
        times = [int(t) for t, _ in equity]
        values = [float(v) for _, v in equity]
        if any(b < a for a, b in zip(times, times[1:])):
            raise ValueError('Equity samples out of order')
        result.net_pnl = _d(values[-1]) - cash
        result.return_pct = return_pct(result.net_pnl, cash)
        drawdown = max_drawdown(times, values, cash)
        result.max_drawdown = drawdown['max_drawdown']
        closes = {}
        for ns, value in equity:
            day = datetime.fromtimestamp(int(ns) // 1_000_000_000, timezone.utc).date()
            closes[day] = float(value)
        start_day = datetime.fromisoformat(window_start).date()
        expected = [start_day + timedelta(days=i) for i in range(days)]
        missing = [d for d in expected if d not in closes]
        if missing:
            result.sharpe_daily_reason = (f'窗口内 {len(missing)} 个 UTC 日缺少权益样本'
                                          '(行情缺口不填零),日度夏普不输出')
            result.sharpe_daily_samples = len(expected) - len(missing)
        else:
            pnl = []
            prior = float(cash)
            for day in expected:
                pnl.append(closes[day] - prior)
                prior = closes[day]
            sharpe = sharpe_daily(daily_returns(pnl, cash))
            result.sharpe_daily = sharpe['value']
            result.sharpe_daily_samples = sharpe['samples']
            result.sharpe_daily_reason = sharpe['reason']
    else:
        result.net_pnl_reason = '缺少含资金费的权益序列,净PnL/区间收益/回撤/夏普不输出'
    result.net_pnl_per_filled_notional_bp = (
        net_pnl_per_filled_notional_bp(result.net_pnl, volume['total_notional'])
        if result.net_pnl is not None else None)
    if order_requests:
        result.fill_rate = volume['fill_receipts'] / order_requests
    else:
        result.fill_rate_reason = '未提供新单请求总数,成交率不输出'
    result.markouts = aggregate_markouts(markouts, markout_horizons, markout_lag_budget_ms)
    return result
