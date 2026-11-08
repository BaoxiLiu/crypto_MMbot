"""Streaming self-financing PnL attribution with explicit event ordering."""
from decimal import Decimal as D

from .order_context_metrics import OrderPlacementMetrics
from .experiment_metrics import write_json


class PnLAttributionMetrics(OrderPlacementMetrics):
    additional_sources = OrderPlacementMetrics.additional_sources + ('app/backtesting/pnl_attribution.py',)

    def __init__(self, output, spec, start, end_exclusive):
        super().__init__(output, spec, start, end_exclusive)
        self.attribution_mid = None
        self.attribution_position = D(0)
        self.execution_edge = D(0)
        self.inventory_drift = D(0)
        self.gap_drift = D(0)
        self.post_halt_edge = D(0)
        self.post_halt_drift = D(0)
        self.post_halt_gap_drift = D(0)
        self.gap_pending = False
        self.first_halt = None
        self.after_halt_receipts = 0

    def fill(self, fill):
        if self.attribution_mid is None:
            raise ValueError('Fill before any visible midpoint')
        signed_qty = D(fill['side'])*D(fill['qty'])
        edge = signed_qty*(self.attribution_mid-D(fill['price']))
        self.execution_edge += edge
        self.attribution_position += signed_qty
        if self.first_halt is not None:
            self.post_halt_edge += edge
            self.after_halt_receipts += 1
        super().fill(fill)

    def mid(self, ns, mid):
        mid = D(mid)
        if self.attribution_mid is not None:
            drift = self.attribution_position*(mid-self.attribution_mid)
            if self.gap_pending:
                self.gap_drift += drift
                if self.first_halt is not None:
                    self.post_halt_gap_drift += drift
            else:
                self.inventory_drift += drift
                if self.first_halt is not None:
                    self.post_halt_drift += drift
        self.attribution_mid = mid
        self.gap_pending = False
        super().mid(ns, mid)

    def gap(self, reason='segment_boundary'):
        self.gap_pending = True
        return super().gap(reason)

    def halt(self, snapshot):
        if self.first_halt is None:
            self.first_halt = dict(snapshot)
            self.first_halt['visible_mid'] = str(self.attribution_mid)
            self.first_halt['execution_edge'] = str(self.execution_edge)
            self.first_halt['inventory_drift'] = str(self.inventory_drift)
            self.first_halt['gap_drift'] = str(self.gap_drift)

    def finish(self, summary):
        if self.attribution_position != D(summary['position']):
            raise AssertionError('Attribution position differs from ledger')
        mark = D(summary['valuation_mark'])
        basis = self.attribution_position*(mark-self.attribution_mid) if self.attribution_mid else D(0)
        funding, fees = D(summary['funding']), D(summary['fees'])
        total = self.execution_edge+self.inventory_drift+self.gap_drift+basis+funding-fees
        residual = D(summary['pnl'])-total
        if abs(residual) > D('1e-12'):
            raise AssertionError('Self-financing attribution failed: '+str(residual))
        result = dict(
            method='Receipt-time visible-mid execution edge + inventory price changes + final mark basis + funding - fees',
            execution_edge=str(self.execution_edge), inventory_drift=str(self.inventory_drift),
            gap_inventory_drift=str(self.gap_drift), final_mark_basis=str(basis),
            funding=str(funding), fees=str(fees), reconstructed_pnl=str(total),
            ledger_pnl=summary['pnl'], residual=str(residual),
            first_halt=self.first_halt, post_halt_execution_edge=str(self.post_halt_edge),
            post_halt_inventory_drift=str(self.post_halt_drift),
            post_halt_gap_inventory_drift=str(self.post_halt_gap_drift),
            post_halt_fill_receipts=self.after_halt_receipts,
            final_position=summary['position'], final_realized=summary['realized'],
            final_unrealized=summary['unrealized'], halt_reason=summary['halt_reason'],
            final_equity=summary['equity'], first_halt_ns=summary['first_halt_ns'])
        if self.first_halt is not None:
            result['equity_change_after_halt'] = str(D(summary['equity'])-D(self.first_halt['equity']))
            result['funding_after_halt'] = str(funding-D(self.first_halt['funding']))
            result['fees_after_halt'] = str(fees-D(self.first_halt['fees']))
        write_json(self.output/'pnl_attribution_metrics.json', result)
        return super().finish(summary)
