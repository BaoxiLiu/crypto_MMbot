from decimal import Decimal as D
from dataclasses import replace
import json,random
import pytest
from app.market_making.core import Rules
from app.market_making.as_math import ASConfig
from app.market_making.fusion.config import FusionConfig,Factors
from app.market_making.contracts import Desired, Phase
from app.market_making.fusion.orders import OrderManager
from app.market_making.fusion.signals import Observations
from app.market_making.fusion.kernel import Kernel
from app.market_making.fusion.tests.quote_fixture import QuoteFixture


def rules():return Rules(D('.01'),D('1'),D('5'),D('1'),D('100000'),D('.5'),D('1.5'))
def cfg(**kw):return replace(FusionConfig(),**kw)
def om(**kw):return OrderManager(rules(),cfg(**kw),'test')
def target(side,qty=2,price='10',ro=False):return Desired(side,D(price),D(qty),ro)
def ack(o,c,ns=1,status='NEW',q='0'):
 return o.update(c.cid,status,D(q),D(q)*c.price,ns)


def test_pending_and_canceling_orders_reserve_both_sides_independently():
 o=om(as_config=replace(ASConfig(),position_cap=D(5)))
 cs=o.reconcile_targets({1:target(1,4),-1:target(-1,4,'10.02')},D(10),0)
 assert len(cs)==2
 assert o.pending(1)==o.pending(-1)==4
 for c in cs:ack(o,c)
 cancels=o.reconcile_targets({},D(10),2)
 assert len(cancels)==2 and o.pending(1)==4
 assert not o.reconcile_targets({1:target(1,5)},D(10),3)


def test_partial_full_and_duplicate_updates_never_fill_twice():
 o=om();c=o.reconcile_targets({1:target(1,2)},D(10),0)[0]
 ack(o,c)
 f=ack(o,c,2,'PARTIALLY_FILLED','1');assert f['qty']==1
 assert ack(o,c,3,'PARTIALLY_FILLED','1') is None
 ack(o,c,4,'FILLED','2')
 assert o.ledger.position==2 and not o.active
 assert ack(o,c,5,'NEW','0') is None
 assert o.ledger.position==2


def test_quote_changes_wait_for_cancel_ack():
 o=om();c=o.reconcile_targets({1:target(1)},D(10),0)[0];ack(o,c)
 t=200_000_000
 cancels=o.reconcile_targets({1:target(1,price='9.99')},D(10),t)
 assert [c.kind for c in cancels]==['cancel']
 assert not o.reconcile_targets({1:target(1,price='9.99')},D(10),t+1)
 ack(o,c,t+2,'CANCELED')
 assert o.reconcile_targets({1:target(1,price='9.99')},D(10),t+3)[0].kind=='new'


def test_cancel_before_new_ack_is_deferred_and_dispatched():
 o=om();c=o.reconcile_targets({1:target(1)},D(10),0)[0]
 assert not o.reconcile_targets({},D(10),1)
 assert o.active[c.cid].cancel_wanted
 ack(o,c,2)
 assert o.reconcile_targets({},D(10),3)[0].kind=='cancel'


def test_fill_while_cancel_pending_is_booked():
 o=om();c=o.reconcile_targets({1:target(1)},D(10),0)[0];ack(o,c)
 o.reconcile_targets({},D(10),2)
 ack(o,c,3,'PARTIALLY_FILLED','1')
 assert o.active[c.cid].phase==Phase.PENDING_CANCEL
 ack(o,c,4,'CANCELED','1')
 assert o.ledger.position==1 and o.pending(1)==0


def test_unknown_keeps_capacity_no_automatic_resend():
 o=om();c=o.reconcile_targets({1:target(1)},D(10),0)[0]
 o.uncertain(c.cid,'timeout')
 assert o.pending(1)==2 and not o.reconciled
 for t in range(1,10):assert not o.reconcile_targets({1:target(1)},D(10),t)
 assert o.serial==1


def test_worst_case_risk_does_not_use_opposite_side_fills():
 o=om(as_config=replace(ASConfig(),position_cap=D(5)))
 # Establish short 4 with real events, then consider independent sides.
 c=o.reconcile_targets({-1:target(-1,4)},D(10),0)[0];ack(o,c,1,'FILLED','4')
 cs=o.reconcile_targets({1:target(1,2),-1:target(-1,2)},D(10),2)
 assert next(c for c in cs if c.side==-1).qty==1
 assert o.ledger.position-o.pending(-1)==-5


def test_reduce_only_cannot_flip_through_flat():
 o=om();c=o.reconcile_targets({1:target(1,1)},D(10),0)[0];ack(o,c,1,'FILLED','1')
 cs=o.reconcile_targets({-1:target(-1,10,ro=True)},D(10),2)
 assert cs[0].qty==1 and cs[0].reduce_only


def test_actual_fee_replaces_estimate_even_if_rest_filled_first():
 o=om();c=o.reconcile_targets({1:target(1,2)},D(10),0)[0];ack(o,c,1,'FILLED','2')
 assert o.ledger.fees==D('.004')
 o.actual_fee(c.cid,9,D(2),D('.002'),'USDT','USDT',2)
 assert o.ledger.fees==D('.002')
 before=o.ledger.wallet
 o.actual_fee(c.cid,9,D(2),D('.002'),'USDT','USDT',3)
 assert o.ledger.wallet==before
 o.assert_invariants(D(10))


def test_fee_receipts_can_arrive_after_newer_cumulative_fill():
 o=om();c=o.reconcile_targets({1:target(1,2)},D(10),0)[0];ack(o,c,1,'FILLED','2')
 o.actual_fee(c.cid,2,D(1),D('.001'),'USDT','USDT',2)
 o.actual_fee(c.cid,1,D(1),D('.001'),'USDT','USDT',3)
 assert o.ledger.fees==D('.002')
 assert ack(o,c,4,'PARTIALLY_FILLED','1') is None
 assert o.ledger.position==2


def test_unsupported_fee_asset_halts():
 o=om();c=o.reconcile_targets({1:target(1,1)},D(10),0)[0];ack(o,c,1,'FILLED','1')
 with pytest.raises(ValueError):o.actual_fee(c.cid,1,D(1),D('.001'),'BNB','USDT',2)
 assert o.halt_reason=='unsupported_commission_asset'


def test_restore_reserves_open_orders_and_requires_reconciliation():
 o=om();c=o.reconcile_targets({1:target(1)},D(10),0)[0];ack(o,c)
 restored=om();restored.restore(json.loads(json.dumps(o.dump(),default=str)))
 assert restored.pending(1)==2 and not restored.reconciled
 assert restored.find(c.cid).phase==Phase.UNKNOWN
 assert not restored.reconcile_targets({1:target(1)},D(10),100)


def test_known_flip_and_cash_flow_oracle():
 o=om();c=o.reconcile_targets({1:target(1,1)},D(10),0)[0];ack(o,c,1,'FILLED','1')
 c=o.reconcile_targets({-1:target(-1,2,'11')},D(11),2)[0];ack(o,c,3,'FILLED','2')
 assert o.ledger.position==-1 and o.ledger.average_entry==11 and o.ledger.realized==1
 o.assert_invariants(D(11))


@pytest.mark.parametrize('mask',range(8))
def test_all_factors_share_same_mandatory_position_guard(mask):
 k=Kernel(rules(),cfg(as_config=replace(ASConfig(),position_cap=D(2),warmup_samples=2)),Factors.from_mask(mask), strategy_factory=QuoteFixture)
 for t in range(3):
  k.book(t*100_000_000,D(10),D('10.02'));k.mark_price(t*100_000_000,D('10.01'))
 cmds=k.step(200_000_000)
 assert cmds and all(c.qty<=2 for c in cmds)


def test_prefix_causality_of_features():
 def run(prices):
  obs=Observations(cfg(as_config=replace(ASConfig(),warmup_samples=2)),D('.01'))
  return [obs.book(i*100_000_000,D(p),D(p)+D('.02')) for i,p in enumerate(prices)]
 p=['10','10.01','10.02']
 assert run(p)==run(p+['11','12'])[:3]


def test_flow_sells_veto_buys_not_asks():
 c=cfg(flow_min_trades=2,flow_min_notional=1.)
 obs=Observations(c,D('.01'))
 obs.trade(0,1,-1,D(10),D(10));obs.trade(1,2,-1,D(10),D(10))
 assert obs.flow_veto[1]>1 and obs.flow_veto[-1]==0
 assert not obs.trade(2,2,-1,D(10),D(10))
 assert len(obs.trades)==2




def test_market_gap_cancels_targets_but_keeps_open_reservations():
 k=Kernel(rules(),cfg(as_config=replace(ASConfig(),warmup_samples=2)),Factors(), strategy_factory=QuoteFixture)
 for i in range(3):k.book(i*100_000_000,D(10),D('10.02'));k.mark_price(i*100_000_000,D(10))
 cs=k.step(200_000_000)
 for c in cs:k.update(c.cid,'NEW',D(0),D(0),200_000_001)
 k.reset_market();cancel=k.step(300_000_000)
 assert len(cancel)==2 and all(c.kind=='cancel' for c in cancel)
 assert k.oms.pending(1)>0


def test_halt_does_not_freeze_valuation():
 k=Kernel(rules(),cfg(),Factors(), strategy_factory=QuoteFixture);o=k.oms
 c=o.reconcile_targets({1:target(1,2)},D(10),0)[0];ack(o,c,1,'FILLED','2')
 o.halt_reason='manual_stop'
 k.mark_price(2,D(9));a=k.report()['equity']
 k.mark_price(3,D(8));b=k.report()['equity']
 assert D(b)<D(a)
