"""Risk exits must bypass ordinary quote age without bypassing private ACKs."""
from dataclasses import replace
from decimal import Decimal as D
import gzip,json
from app.market_making.core import Rules
from app.market_making.as_math import ASConfig
from app.market_making.fusion.config import FusionConfig,Factors
from app.market_making.fusion.kernel import Kernel
from app.market_making.contracts import Desired, Phase
from app.backtesting.order_context_metrics import OrderLifecycleMetrics
from app.market_making.fusion.tests.quote_fixture import QuoteFixture


def ready():
    rules=Rules(D('.01'),D(1),D(5),D(1),D(100000),D('.5'),D('1.5'))
    k=Kernel(rules,replace(FusionConfig(),as_config=replace(ASConfig(),warmup_samples=2)),Factors(), strategy_factory=QuoteFixture)
    for ns in (0,100_000_000,200_000_000):
        k.book(ns,D(10),D('10.02'));k.mark_price(ns,D(10))
    return k


def test_urgent_reprice_bypasses_age_but_waits_cancel_ack():
    k=ready();t=200_000_000
    first={1:Desired(1,D(10),D(2))};next_price={1:Desired(1,D('9.99'),D(2))}
    c=k.dispatch(first,t)[0];k.update(c.cid,'NEW',D(0),D(0),t+1)
    assert k.dispatch(next_price,t+2)==[]
    commands=k.dispatch(next_price,t+3,{1})
    assert len(commands)==1 and commands[0].kind=='cancel'
    assert k.oms.find(c.cid).phase==Phase.PENDING_CANCEL
    assert k.dispatch(next_price,t+4)==[] and k.oms.pending(1)==2
    k.update(c.cid,'CANCELED',D(0),D(0),t+5)
    assert k.dispatch(next_price,t+6)[0].kind=='new'


def test_urgent_exit_while_new_pending_is_deferred():
    k=ready();t=200_000_000;targets={1:Desired(1,D(10),D(2))}
    c=k.dispatch(targets,t)[0]
    assert k.dispatch(targets,t+1,{1})==[]
    assert k.oms.find(c.cid).cancel_wanted
    k.update(c.cid,'NEW',D(0),D(0),t+2)
    assert k.dispatch(targets,t+3)[0].kind=='cancel'


def test_lifecycle_clocks_are_observations_and_boundary_is_explicit(tmp_path):
    spec=dict(timezone='UTC',markout_horizons_ms=[100],max_markout_lag_ms=250,
              markout_anchor='local_fill_receipt',markout_reference='first_observed_mid_at_or_after',retain_midpoints=False)
    obs=OrderLifecycleMetrics(tmp_path,spec,'2026-10-02','2026-10-03')
    k=ready();c=k.dispatch({1:Desired(1,D(10),D(2))},200_000_000)[0]
    obs.command(c,200_000_000,200_100_000,10.,10.02)
    obs.order_update(c.cid,'NEW',D(0),201_100_000,200_600_000,200_100_000,0)
    obs.order_update(c.cid,'CANCELED',D(0),202_000_000,0,0,0,True)
    obs.close()
    with gzip.open(tmp_path/'order_lifecycle.jsonl.gz','rt') as f:rows=[json.loads(x) for x in f]
    assert rows[1]['native_exchange_ns']==200_600_000 and rows[1]['receipt_ns']==201_100_000
    assert rows[2]['boundary_assumption'] and rows[2]['native_exchange_ns'] is None
    assert k.oms.find(c.cid).phase==Phase.PENDING_NEW
