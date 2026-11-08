import json
from datetime import datetime, timezone
from decimal import Decimal as D

import pytest

from src.metrics import core, schema
from src.metrics.core import (
    aggregate_fills, aggregate_markouts, average_daily_turnover, calendar_days,
    daily_returns, max_drawdown, net_pnl_per_filled_notional_bp, reconcile_ledger,
    return_pct, sharpe_daily, summarize,
)


def ns(day, seconds=0):
    return (int(datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp()) + seconds) * 1_000_000_000


def fill(ns_, side=1, price='100', qty='2', fee='0', funding='0'):
    return dict(ns=ns_, side=side, price=price, qty=qty, fee=fee, funding=funding)


class TestScalarDefinitions:
    def test_hand_computed_values(self):
        # 净PnL 1.5、成交额 251、初始资金 600、3 个日历日。
        assert net_pnl_per_filled_notional_bp(D('1.5'), D('251')) == D('1.5') / D('251') * 10000
        assert core.turnover_initial_cash(D('251'), D('600')) == D('251') / D('600')
        assert average_daily_turnover(D('251'), D('600'), 3) == D('251') / D('600') / 3
        assert return_pct(D('1.5'), D('600')) == D('0.25')

    def test_zero_or_missing_notional_is_null_not_error(self):
        assert net_pnl_per_filled_notional_bp(D('1'), D('0')) is None
        assert net_pnl_per_filled_notional_bp(D('1'), None) is None
        assert core.turnover_initial_cash(None, D('600')) is None
        assert average_daily_turnover(None, D('600'), 10) is None

    def test_calendar_days_half_open(self):
        assert calendar_days('2026-08-01', '2026-08-11') == 10
        with pytest.raises(ValueError):
            calendar_days('2026-08-11', '2026-08-11')

    def test_non_finite_inputs_rejected(self):
        with pytest.raises(ValueError):
            net_pnl_per_filled_notional_bp(D('1'), float('nan'))


class TestSharpe:
    def test_under_20_daily_samples_is_null_with_reason(self):
        result = sharpe_daily([D('0.001')] * 7)
        assert result['value'] is None and result['samples'] == 7
        assert '20' in result['reason']

    def test_zero_variance_is_null(self):
        result = sharpe_daily([D('0.001')] * 25)
        assert result['value'] is None and '零方差' in result['reason']

    def test_hand_computed_sharpe(self):
        returns = [D('0.01'), D('-0.005')] * 10 + [D('0.002')] * 5  # 25 个样本
        values = [float(r) for r in returns]
        mean = sum(values) / len(values)
        sd = (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** .5
        result = sharpe_daily(returns)
        assert result['value'] == pytest.approx(mean / sd * 365 ** .5)
        assert result['samples'] == 25 and result['reason'] is None

    def test_daily_returns_use_prior_day_equity(self):
        # 首日相对初始资金,之后相对前一日收盘。
        result = daily_returns([D('6'), D('6.06'), D('-3.0603')], D('600'))
        assert result[0] == D('0.01')
        assert result[1] == pytest.approx(D('0.01'))
        assert result[2] == D('-0.005')
        with pytest.raises(ValueError):
            daily_returns([D('-600'), D('1')], D('600'))


class TestDrawdown:
    def test_initial_capital_and_first_global_peak(self):
        # 与 app.backtesting.research_report 历史口径一致:600→599→605→594。
        result = max_drawdown([1, 2, 3], [599, 605, 594], 600)
        assert result['max_drawdown'] == pytest.approx(11)
        assert (result['peak_ns'], result['trough_ns']) == (2, 3)

    def test_no_equity_rejected(self):
        with pytest.raises(ValueError):
            max_drawdown([], [], 600)


class TestFillsAndReconciliation:
    def test_single_side_with_funding_and_ledger(self):
        fills = [fill(ns('2026-01-01'), side=1, price='100', qty='2', fee='.02', funding='.1'),
                 fill(ns('2026-01-01', 1), side=-1, price='102', qty='.5', fee='.0051')]
        volume = aggregate_fills(fills)
        assert volume['fill_receipts'] == 2
        assert volume['total_notional'] == D('251') and volume['net_qty'] == D('1.5')
        assert volume['fees'] == D('.0251') and volume['funding'] == D('.1')
        ledger = dict(fill_events=2, fees='.0251', funding='.1', position='1.5',
                      cash_flow='-148.9251')
        reconcile_ledger(volume, ledger)

    def test_reconciliation_mismatch_raises(self):
        volume = aggregate_fills([fill(ns('2026-01-01'))])
        with pytest.raises(AssertionError):
            reconcile_ledger(volume, dict(fees='0', funding='0', position='3', cash_flow='-200'))
        with pytest.raises(AssertionError):
            reconcile_ledger(volume, dict(fill_events=2, fees='0', funding='0',
                                          position='2', cash_flow='-200'))

    def test_out_of_order_or_invalid_fill_rejected(self):
        with pytest.raises(ValueError):
            aggregate_fills([fill(ns('2026-01-01')), fill(ns('2026-01-01') - 1)])
        with pytest.raises(ValueError):
            aggregate_fills([fill(ns('2026-01-01'), side=0)])
        with pytest.raises(ValueError):
            aggregate_fills([fill(ns('2026-01-01'), qty='0')])


class TestMarkoutAggregation:
    def records(self):
        return [dict(horizon_ms=100, notional='200', markout_bp='1', lag_ns=10_000_000),
                dict(horizon_ms=100, notional='51', markout_bp='-2', lag_ns=20_000_000),
                dict(horizon_ms=100, notional='9', markout_bp=None, reason='late_mid'),
                dict(horizon_ms=500, notional='251', markout_bp='3', lag_ns=249_000_000)]

    def test_notional_weighted_hand_computed(self):
        rows = aggregate_markouts(self.records())
        by_horizon = {r['horizon_ms']: r for r in rows}
        h100 = by_horizon[100]
        assert h100['observed'] == 2 and h100['censored'] == 1
        assert h100['notional_weighted_bp'] == (D('1') * 200 + D('-2') * 51) / D('251')
        assert h100['mean_lag_ns'] == 15_000_000
        assert h100['censor_reasons'] == {'late_mid': 1}
        assert by_horizon[500]['notional_weighted_bp'] == D('3')
        empty = by_horizon[60000]
        assert empty['observed'] == 0 and empty['notional_weighted_bp'] is None

    def test_lag_budget_enforced(self):
        with pytest.raises(ValueError, match='250ms'):
            aggregate_markouts([dict(horizon_ms=100, notional='1', markout_bp='1',
                                     lag_ns=251_000_000)])
        with pytest.raises(ValueError, match='outside'):
            aggregate_markouts([dict(horizon_ms=2000, notional='1', markout_bp='1',
                                     lag_ns=0)])


def window(days=3):
    return dict(window_start='2026-01-01',
                window_end_exclusive=f'2026-01-0{1 + days}')


class TestSummarize:
    def equity(self):
        return [(ns('2026-01-01', 100), 601.), (ns('2026-01-02', 200), 602.),
                (ns('2026-01-03', 300), 601.5)]

    def test_empty_fills(self):
        result = summarize([], equity=self.equity(), initial_cash='600', **window())
        assert result.fill_receipts == 0 and result.total_notional == D(0)
        assert result.turnover == D(0)
        assert result.net_pnl_per_filled_notional_bp is None
        assert result.fill_rate is None and result.fill_rate_reason
        assert result.net_pnl == D('1.5')

    def test_full_pipeline_hand_computed(self):
        fills = [fill(ns('2026-01-01'), side=1, price='100', qty='2', fee='.02', funding='.1'),
                 fill(ns('2026-01-02', 1), side=-1, price='102', qty='.5', fee='.0051')]
        ledger = dict(fill_events=2, fees='.0251', funding='.1', position='1.5',
                      cash_flow='-148.9251')
        result = summarize(fills, equity=self.equity(), initial_cash='600',
                           ledger=ledger, order_requests=10, **window())
        assert result.net_pnl == D('1.5')
        assert result.return_pct == D('0.25')
        assert result.turnover == D('251') / D('600')
        assert result.average_daily_turnover == D('251') / D('600') / 3
        assert result.net_pnl_per_filled_notional_bp == D('1.5') / D('251') * 10000
        assert result.fill_rate == pytest.approx(0.2)
        assert result.max_drawdown == pytest.approx(0.5)
        assert result.sharpe_daily is None and '20' in result.sharpe_daily_reason
        assert result.sharpe_daily_samples == 3
        assert result.reconciliation == 'fills/fees/inventory/funding/cash reconciled'

    def test_funding_cashflows_enter_totals_and_reconciliation(self):
        fills = [fill(ns('2026-01-01'), side=1, price='100', qty='2', fee='0')]
        ledger = dict(fill_events=1, fees='0', funding='-.5', position='2',
                      cash_flow='-200.5')
        result = summarize(fills, funding_cashflows=['-.5'], equity=self.equity(),
                           initial_cash='600', ledger=ledger, **window())
        assert result.funding == D('-.5')

    def test_missing_equity_day_blocks_sharpe(self):
        equity = [(ns('2026-01-01', 100), 601.), (ns('2026-01-03', 300), 601.5)]
        result = summarize([], equity=equity, initial_cash='600', **window())
        assert result.sharpe_daily is None and '缺少权益样本' in result.sharpe_daily_reason
        assert result.sharpe_daily_samples == 2

    def test_no_equity_marks_dependent_metrics_missing(self):
        result = summarize([fill(ns('2026-01-01'))], initial_cash='600', **window())
        assert result.net_pnl is None and result.net_pnl_reason
        assert result.turnover == D('200') / D('600')

    def test_partial_range_has_no_average_daily_turnover(self):
        result = summarize([], equity=self.equity(), initial_cash='600',
                           full_range=False, **window())
        assert result.average_daily_turnover is None


class TestSchema:
    def document(self, tmp_path):
        fills = [fill(ns('2026-01-01'), side=1, price='100', qty='2', fee='.02')]
        marks = [dict(horizon_ms=100, notional='200', markout_bp='1', lag_ns=10_000_000)]
        result = summarize(fills, equity=[(ns('2026-01-0%d' % d, 100), 600 + d)
                                          for d in (1, 2, 3)],
                           markouts=marks, initial_cash='600', **window())
        path = tmp_path / 'results.json'
        document = schema.write_results_json(path, result, run_id='synthetic')
        return json.loads(path.read_text()), document, result

    def test_results_json_schema_and_null_reasons(self, tmp_path):
        loaded, document, _ = self.document(tmp_path)
        assert loaded['schema_version'] == 1
        metrics = loaded['metrics']
        assert metrics['net_pnl'] == str(D('3.0'))
        assert metrics['turnover'] == str(D('200') / D('600'))
        assert metrics['sharpe_daily'] is None and metrics['sharpe_daily_samples'] == 3
        assert metrics['sharpe_daily_reason']
        assert metrics['fill_rate'] is None and metrics['fill_rate_reason']
        assert loaded['definitions']['net_pnl']
        assert loaded['markouts'][0]['notional_weighted_bp'] == '1'
        json.dumps(document, allow_nan=False)  # 全文档可序列化,无 NaN。

    def test_to_markdown_table(self, tmp_path):
        _, _, result = self.document(tmp_path)
        text = schema.to_markdown(result)
        assert '| 指标 | 数值 | 口径 |' in text
        assert '净PnL (USDC)' in text and '日度夏普' in text
        assert '100ms' in text and '60s' in text
        assert '缺失' in text  # 夏普样本不足与成交率缺失均标注。
