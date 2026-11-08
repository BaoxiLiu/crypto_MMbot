from decimal import Decimal as D
import json

from app.backtesting.pnl_attribution import PnLAttributionMetrics
from app.backtesting.order_context_metrics import OrderPlacementMetrics

START = 1785542400000000000


def test_exact_execution_inventory_gap_and_terminal_basis(tmp_path, monkeypatch):
    spec=dict(timezone='UTC',markout_anchor='local_fill_receipt',
              markout_reference='first_observed_mid_at_or_after',
              markout_horizons_ms=[1000],max_markout_lag_ms=250,retain_midpoints=False)
    observer=PnLAttributionMetrics(tmp_path,spec,'2026-08-01','2026-08-11')
    monkeypatch.setattr(OrderPlacementMetrics,'finish',lambda self,summary:None)
    observer.mid(START,D(100))
    observer.fill(dict(ns=START,cid='buy',side=1,qty='1',price='99',fee='.05'))
    observer.mid(START+1,D(102))
    observer.halt(dict(equity='602.95',funding='0',fees='.05'))
    observer.mid(START+2,D(101))
    observer.fill(dict(ns=START+2,cid='sell',side=-1,qty='.5',price='97',fee='.05'))
    observer.gap()
    observer.mid(START+3,D(95))
    summary=dict(position='.5',valuation_mark='96',funding='.2',fees='.1',
                 pnl='-2.4',realized='-1',unrealized='-1.5',
                 halt_reason='test',equity='597.6',first_halt_ns=START+1)
    observer.finish(summary)
    observer.close()
    result=json.loads((tmp_path/'pnl_attribution_metrics.json').read_text())
    assert D(result['execution_edge']) == -1
    assert D(result['inventory_drift']) == 1
    assert D(result['gap_inventory_drift']) == -3
    assert D(result['final_mark_basis']) == D('.5')
    assert D(result['residual']) == 0
    assert D(result['equity_change_after_halt']) == D('-5.35')
    assert result['post_halt_fill_receipts'] == 1
