from contextlib import ExitStack
from decimal import Decimal as D
import json
import numpy as np
import hftbacktest as h

from app.backtesting.experiment_metrics import ReplayMetrics
from app.market_making.fusion.event_backtest import run_event_arm
from app.market_making.fusion.tests.test_event_replay import synthetic_tape,kernel,profile
from app.market_making.fusion.tests.quote_fixture import QuoteFixture


def test_observer_preserves_all_orders_fills_and_equity_and_exports_metrics(tmp_path):
    synthetic_tape(tmp_path,[(1_202_200_000,h.TRADE_EVENT,h.SELL_EVENT,99.95,20.)])
    k=kernel()
    old=run_event_arm(tmp_path,tmp_path/'plain',0,k.config,tmp_path/'marks.npy',[],k.rules,profile(), strategy_factory=QuoteFixture)
    spec=dict(timezone='UTC',markout_horizons_ms=[100,500,1000,5000,10000,30000],
              max_markout_lag_ms=250,markout_anchor='local_fill_receipt',
              markout_reference='first_observed_mid_at_or_after',retain_midpoints=True)
    with ExitStack() as stack:
        factory=lambda out:stack.enter_context(ReplayMetrics(out,spec,'1970-01-01','1970-01-02'))
        new=run_event_arm(tmp_path,tmp_path/'observed',0,k.config,tmp_path/'marks.npy',[],k.rules,profile(),observer_factory=factory, strategy_factory=QuoteFixture)
    assert {k:v for k,v in old.items() if k not in ('identity','wall_s')}=={k:v for k,v in new.items() if k not in ('identity','wall_s')}
    with np.load(tmp_path/'plain/equity.npz') as a,np.load(tmp_path/'observed/equity.npz') as b:
        assert np.array_equal(a['data'],b['data'])
    metrics=json.loads((tmp_path/'observed/metrics.json').read_text())
    assert D(metrics['volume']['total_notional'])==D('.25') * (D('100') - 4 * D('.01'))
    assert metrics['oms_fill_receipts']==1 and len(metrics['markouts'])==6
    assert all(r['observed']+r['censored']==1 for r in metrics['markouts'])
    from app.backtesting.order_context_metrics import OrderContextMetrics
    with ExitStack() as stack:
        factory=lambda out:stack.enter_context(OrderContextMetrics(out,spec,'1970-01-01','1970-01-02',tmp_path/'observed/fills.jsonl'))
        contextual=run_event_arm(tmp_path,tmp_path/'contextual',0,k.config,tmp_path/'marks.npy',[],k.rules,profile(),observer_factory=factory, strategy_factory=QuoteFixture)
    assert {k:v for k,v in old.items() if k not in ('identity','wall_s')}=={k:v for k,v in contextual.items() if k not in ('identity','wall_s')}
    assert (tmp_path/'observed/fills.jsonl').read_bytes()==(tmp_path/'contextual/fills.jsonl').read_bytes()
    contexts={r['cid']:r for r in map(json.loads,(tmp_path/'contextual/order_context.jsonl').read_text().splitlines())}
    fill=json.loads((tmp_path/'contextual/fills.jsonl').read_text().splitlines()[0])
    c=contexts[fill['cid']]
    assert c['decision_ns']<c['submit_ns']<=fill['receipt_ns']
    assert c['submit_ns']-c['decision_ns']==profile().decision_ns
