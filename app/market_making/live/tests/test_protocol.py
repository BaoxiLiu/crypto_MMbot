from app.market_making.fusion.tests.quote_fixture import QuoteFixture
from dataclasses import replace,asdict
from decimal import Decimal as D
import asyncio,hashlib,hmac,json
from urllib.parse import parse_qs
from types import SimpleNamespace
import pytest
from app.market_making.live.gateway import *
from app.market_making.live.journal import Journal
from app.market_making.live.runtime import Runtime
from app.market_making.fusion.config import FusionConfig as StrategyConfig, Factors as ProtectionSwitches
from app.market_making.live.engine import StrategyEngine
from app.market_making.contracts import Desired, Phase
from app.market_making.as_math import ASConfig
from app.market_making.core import Rules


def kernel():
 r=Rules(D('.01'),D(1),D(5),D(1),D(100000),D('.5'),D('1.5'))
 return StrategyEngine(r,replace(StrategyConfig(),as_config=replace(ASConfig(),warmup_samples=2)),ProtectionSwitches(),prefix='abc',strategy_factory=QuoteFixture)
def new(k,ns=0):return k.oms.reconcile_targets({1:Desired(1,D(10),D(2))},D(10),ns)[0]
def receipt(c,qty='2',status='FILLED'):
 return dict(e='ORDER_TRADE_UPDATE',T=100,E=101,o=dict(s='DOGEUSDT',c=c.cid,S='BUY',ps='BOTH',
                 X=status,x='TRADE',z=qty,ap='10',i=12,l=qty,L='10',t=100,n='.002',N='USDT',m=True))
def account(k):return dict(canTrade=True,positions=[dict(symbol='DOGEUSDT',positionAmt=str(k.oms.ledger.position),positionSide='BOTH',entryPrice=str(k.oms.ledger.average_entry))],
                         assets=[dict(asset='USDT',walletBalance=str(k.oms.ledger.wallet))])


def test_sign_exact_encoded_payload_and_post_only_contract():
 k=kernel();c=new(k);p=order_params(c,'DOGEUSDT')
 assert p['timeInForce']=='GTX' and p['positionSide']=='BOTH' and p['reduceOnly']=='false'
 assert p['quantity']=='2' and p['newClientOrderId']==c.cid
 payload=signed_query(p,'fixture',123)
 query,signature=payload.rsplit('&signature=',1)
 assert signature==hmac.new(b'fixture',query.encode(),hashlib.sha256).hexdigest()
 assert parse_qs(query)['timestamp']==['123']
 assert order_params(replace(c,kind='cancel'),'DOGEUSDT')['origClientOrderId']==c.cid


def test_rest_then_private_trade_reconciles_execution_and_actual_fee():
 k=kernel();c=new(k)
 apply_order(k,dict(clientOrderId=c.cid,orderId=12,status='FILLED',executedQty='2',cumQuote='20'),1)
 e=PrivateEvents(k,'DOGEUSDT','USDT');event=receipt(c)
 e.apply(event,2);e.apply(event,3)
 assert k.oms.ledger.position==2 and k.oms.ledger.fees==D('.002')
 assert k.oms.stats['fill_events']==1
 assert account_matches(k,account(k),[],'DOGEUSDT','USDT')


def test_funding_is_actual_cash_and_deduplicated():
 k=kernel();e=PrivateEvents(k,'DOGEUSDT','USDT')
 event=dict(e='ACCOUNT_UPDATE',E=100,T=99,a=dict(m='FUNDING_FEE',B=[dict(a='USDT',bc='-.1')]))
 e.apply(event,0);e.apply(event,1)
 assert k.oms.ledger.funding==D('-.1') and k.oms.ledger.wallet==D('599.9')
 k.oms.assert_invariants(D(10))
 e.apply(dict(e='ACCOUNT_UPDATE',a=dict(m='DEPOSIT')),2)
 assert k.oms.halt_reason=='external_account_change'


def test_no_account_overwrite_or_foreign_order_adoption():
 k=kernel();a=account(k);a['assets'][0]['walletBalance']='601'
 assert not account_matches(k,a,[],'DOGEUSDT','USDT')
 assert not account_matches(k,account(k),[dict(symbol='DOGEUSDT',clientOrderId='foreign')],'DOGEUSDT','USDT')
 assert k.oms.ledger.wallet==600


def test_journal_locks_fingerprints_and_restart_requires_exchange_reconciliation(tmp_path):
 k=kernel();c=new(k);events=PrivateEvents(k,'DOGEUSDT','USDT');events.funding_ids={'1:2:USDT:.1'}
 path=tmp_path/'state.sqlite';j=Journal(path,dict(symbol='DOGEUSDT'))
 j.save(k,events)
 with pytest.raises(RuntimeError):Journal(path,dict(symbol='DOGEUSDT'))
 saved=j.load();j.close()
 with pytest.raises(ValueError):Journal(path,dict(symbol='DOGEUSDC'))
 j=Journal(path,dict(symbol='DOGEUSDT'))
 now=SimpleNamespace(ns=lambda:100)
 r=Runtime(kernel(),None,j,'DOGEUSDT','USDT','live',now)
 assert r.k.oms.find(c.cid).phase==Phase.UNKNOWN and not r.k.oms.reconciled
 assert r.k.oms.pending(1)==2 and r.events.funding_ids==events.funding_ids
 assert r.k.oms.serial==1 and list(r.k.oms.new_times)==[0]
 j.close()


def test_runtime_timeout_unknown_never_resends(tmp_path):
 k=kernel();j=Journal(tmp_path/'state',{});r=Runtime(k,None,j,'DOGEUSDT','USDT','live',SimpleNamespace(ns=lambda:10))
 k.oms.reconciled=True;c=new(k)
 r.handle('result',(c,RequestFailure(-1007)),1)
 assert k.oms.find(c.cid).phase==Phase.UNKNOWN
 assert k.oms.pending(1)==2 and not k.oms.reconciled
 r.handle('result',(c,dict(clientOrderId=c.cid,status='NEW',executedQty='0',cumQuote='0')),2)
 assert not k.oms.reconciled # status alone does not prove the account matched
 j.close()


def test_definitive_rejection_releases_only_rejected_quantity(tmp_path):
 k=kernel();j=Journal(tmp_path/'state',{});r=Runtime(k,None,j,'DOGEUSDT','USDT','live',SimpleNamespace(ns=lambda:10))
 k.oms.reconciled=True;c=new(k)
 r.handle('result',(c,RequestFailure(-1111,400,ambiguous=False)),1)
 assert k.oms.pending(1)==0 and k.oms.find(c.cid).phase==Phase.REJECTED
 j.close()


def test_plan_is_frozen_before_decision_delay_and_does_not_advance_market_clock():
 k=kernel()
 for t in (0,100_000_000,200_000_000):k.book(t,D(10),D('10.02'));k.mark_price(t,D(10))
 targets=k.plan(200_000_000)
 k.dispatch(targets,205_000_000)
 # Trade received DURING calculation is processed at the next observation.
 k.trade(202_000_000,1,1,D(10),D(1))
 assert k.strategy.obs.last_clock==202_000_000
 assert all(o.created_ns==205_000_000 for o in k.oms.active.values())


def test_intents_clip_to_account_updates_observed_during_computation():
 k=kernel();k.config # use independent OMS targets; no predictive fills
 c=new(k);k.update(c.cid,'NEW',D(0),D(0),1)
 k.update(c.cid,'FILLED',D(2),D(20),2)
 # Stale reducing intent sized at 2 cannot flip a now smaller position.
 c=k.oms.reconcile_targets({-1:Desired(-1,D(10),D(1))},D(10),3)[0]
 k.update(c.cid,'FILLED',D(1),D(10),4)
 c=k.oms.reconcile_targets({-1:Desired(-1,D(10),D(2),True)},D(10),5)[0]
 assert c.qty==1 and c.reduce_only


def test_partial_ack_does_not_revert_partial_state():
 k=kernel();c=new(k)
 k.update(c.cid,'PARTIALLY_FILLED',D(1),D(10),1)
 k.update(c.cid,'NEW',D(1),D(10),2)
 assert k.oms.find(c.cid).phase==Phase.PARTIAL


class FakeResponse:
 def __init__(self,status,data):self.status,self.data=status,data
 async def __aenter__(self):return self
 async def __aexit__(self,*args):pass
 async def json(self,**kwargs):return self.data

class FakeSession:
 def __init__(self,status=200,data=None,error=None):self.status,self.data,self.error=status,data,error;self.calls=[]
 def request(self,*args,**kwargs):
  self.calls.append((args,kwargs))
  if self.error:raise self.error
  return FakeResponse(self.status,self.data)


@pytest.mark.parametrize('status,code,ambiguous',[(503,-1007,True),(400,-1111,False),(400,-2010,True),(429,-1003,False)])
def test_http_classification_has_no_automatic_resend(status,code,ambiguous):
 session=FakeSession(status,dict(code=code,msg='fixture'))
 client=RestClient(session,'key','secret')
 with pytest.raises(RequestFailure) as error:
  asyncio.run(client.request('POST','/fapi/v1/order',dict(symbol='DOGEUSDT'),mutating=True))
 assert error.value.ambiguous==ambiguous and len(session.calls)==1
 assert 'secret' not in str(error.value) and 'signature' not in str(error.value)


def test_timeout_is_unknown_and_cancel_rejection_requires_query():
 session=FakeSession(error=asyncio.TimeoutError());client=RestClient(session,'key','secret')
 with pytest.raises(RequestFailure) as error:asyncio.run(client.request('POST','/fapi/v1/order'))
 assert error.value.ambiguous and len(session.calls)==1
 session=FakeSession(400,dict(code=-2011));client=RestClient(session,'key','secret')
 with pytest.raises(RequestFailure) as error:asyncio.run(client.request('DELETE','/fapi/v1/order'))
 assert error.value.ambiguous


def test_ack_without_execution_status_never_invents_fill():
 k=kernel();c=new(k)
 apply_order(k,dict(clientOrderId=c.cid,orderId=12),1)
 assert k.oms.ledger.position==0 and k.oms.pending(1)==2


def test_actual_fee_conflict_and_foreign_order_fail_closed():
 k=kernel();c=new(k);p=PrivateEvents(k,'DOGEUSDT','USDT');p.apply(receipt(c),1)
 with pytest.raises(ValueError):k.oms.actual_fee(c.cid,100,D(2),D('.003'),'USDT','USDT',2)
 assert k.oms.halt_reason=='conflicting_fee_receipt'
 k=kernel();p=PrivateEvents(k,'DOGEUSDT','USDT')
 event=receipt(c);event['o']['c']='another_strategy'
 p.apply(event,1);assert k.oms.halt_reason=='unowned_private_order'


def test_reconciliation_checks_economic_cash_when_fill_aggregation_reorders_cost_basis():
 k=kernel();c=new(k)
 apply_order(k,dict(clientOrderId=c.cid,status='FILLED',executedQty='2',cumQuote='20'),1)
 a=account(k)
 # Same net cash/position; different realized/entry decomposition from an
 # exchange-ordered aggregation. Never import or overwrite those balances.
 a['positions'][0]['entryPrice']='10.1';a['assets'][0]['walletBalance']=str(k.oms.ledger.wallet+D('.2'))
 old_wallet=k.oms.ledger.wallet
 assert account_matches(k,a,[],'DOGEUSDT','USDT')
 assert k.oms.ledger.wallet==old_wallet
 a['assets'][0]['walletBalance']=str(old_wallet+D('.21'))
 assert not account_matches(k,a,[],'DOGEUSDT','USDT')


def test_manual_amendment_of_owned_order_does_not_pass_account_reconciliation():
 k=kernel();c=new(k);k.update(c.cid,'NEW',D(0),D(0),1)
 row=dict(symbol='DOGEUSDT',clientOrderId=c.cid,origQty='2',executedQty='0',price='10',side='BUY',
          type='LIMIT',timeInForce='GTX',reduceOnly=False)
 assert account_matches(k,account(k),[row],'DOGEUSDT','USDT')
 row['origQty']='3'
 assert not account_matches(k,account(k),[row],'DOGEUSDT','USDT')


def test_private_order_direction_must_match_the_owned_order():
 k=kernel();c=new(k);event=receipt(c);event['o']['S']='SELL'
 with pytest.raises(ValueError):PrivateEvents(k,'DOGEUSDT','USDT').apply(event,1)
 assert k.oms.ledger.position==0 and k.oms.halt_reason=='private_order_contract_mismatch'


def test_conflicting_duplicate_cumulative_quote_is_not_silently_accepted():
 k=kernel();c=new(k)
 row=dict(clientOrderId=c.cid,status='FILLED',executedQty='2',cumQuote='20')
 apply_order(k,row,1)
 row['cumQuote']='19'
 with pytest.raises(ValueError):apply_order(k,row,2)
 assert k.oms.ledger.position==2 and k.oms.halt_reason=='conflicting_execution_receipt'


def test_post_only_rejection_is_routine_but_invalid_parameters_stop_resubmission(tmp_path):
 for code,halt in ((-5022,False),(-1111,True)):
  k=kernel();j=Journal(tmp_path/str(code),{});r=Runtime(k,None,j,'DOGEUSDT','USDT','live',SimpleNamespace(ns=lambda:10))
  k.oms.reconciled=True;c=new(k)
  r.handle('result',(c,RequestFailure(code,400,False)),1)
  assert bool(k.oms.halt_reason)==halt and k.oms.pending(1)==0
  j.close()


def test_repeated_funding_transaction_with_new_event_time_is_not_double_booked():
 k=kernel();e=PrivateEvents(k,'DOGEUSDT','USDT')
 event=dict(e='ACCOUNT_UPDATE',E=100,T=99,a=dict(m='FUNDING_FEE',B=[dict(a='USDT',bc='-.1')]))
 e.apply(event,0);event['E']=200;e.apply(event,1)
 assert k.oms.ledger.funding==D('-.1')
 event['a']['B'][0]['bc']='-.2'
 with pytest.raises(ValueError):e.apply(event,2)
 assert k.oms.ledger.funding==D('-.1') and k.oms.halt_reason=='conflicting_funding_receipt'
