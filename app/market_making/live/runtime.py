"""Single-owner event loop. Default shadow never accesses API credentials.

Production adapter is implemented but NOT exchange-acceptance tested. Shadow
records intentions only: it must never report public touches as actual fills.
"""
import asyncio,hashlib,json,os,time,uuid
from dataclasses import asdict
from decimal import Decimal as D
from app.market_making.core import Rules
from app.market_making.fusion.config import FusionConfig as StrategyConfig, Factors as ProtectionSwitches
from .engine import StrategyEngine
from ..strategies.registry import factory_for
from .selection import load_selection, validate_source
from app.market_making.contracts import Phase
from app.market_making.fusion.orders import TERMINAL
from .gateway import (RestClient,RequestFailure,PRIVATE,PUBLIC,MARKET,PrivateEvents,
                      apply_order,account_matches)
from .journal import Journal
from app.market_making.fusion.timing import next_timer_ns


class Clock:
    def __init__(self):self.epoch=time.time_ns();self.origin=time.monotonic_ns()
    def ns(self):return self.epoch+time.monotonic_ns()-self.origin


class Runtime:
    def __init__(self,kernel,client,journal,symbol,quote,mode,clock=None,market_source='bbo',depth_speed='100ms'):
        self.k,self.client,self.journal=kernel,client,journal
        self.symbol,self.quote,self.mode=symbol,quote,mode
        if market_source not in ('bbo','depth'):raise ValueError('Unknown market source')
        self.market_source,self.depth_speed=market_source,depth_speed
        self.depth_restart=asyncio.Event()
        self.clock=clock or Clock();self.events=PrivateEvents(kernel,symbol,quote)
        kernel.oms.emit=journal.record
        self.queue=asyncio.Queue(maxsize=10000);self.tasks=set()
        self.connected={'public':False,'market':False,'private':False}
        self.book_id=-1;self.stopping=False;self.last_audit=0;self.last_output=0;self.pause_until=0
        saved=journal.load()
        if saved:
            kernel.oms.restore(saved['oms'])
            ref=saved.get('reference_qty')
            kernel.strategy.reference_qty=D(ref) if ref is not None else None
            self.events.funding_ids=set(saved.get('funding_ids',[]))
        if mode=='live':kernel.oms.reconciled=False

    def spawn(self,coro):
        task=asyncio.create_task(coro);self.tasks.add(task);task.add_done_callback(self.tasks.discard)
        return task

    async def reader(self,label,url):
        import aiohttp
        while not self.stopping:
            try:
                async with self.client.session.ws_connect(url,heartbeat=20,timeout=10) as ws:
                    await self.queue.put(('connected',label,self.clock.ns()))
                    async for msg in ws:
                        if msg.type!=aiohttp.WSMsgType.TEXT:break
                        received_ns=self.clock.ns()
                        data=json.loads(msg.data);data=data.get('data',data)
                        await self.queue.put(('event',(label,data),received_ns))
            except Exception:
                # URLs/private listen keys and keys never enter logs.
                pass
            await self.queue.put(('disconnected',label,self.clock.ns()))
            await asyncio.sleep(2)

    async def result(self,cmd):
        try:result=await self.client.command(cmd,self.symbol)
        except Exception as e:result=e
        await self.queue.put(('result',(cmd,result),self.clock.ns()))

    async def audit(self):
        try:
            # Queries only; never blindly resubmit an uncertain command.
            replies=[]
            for o in list(self.k.oms.active.values()):
                replies.append(await self.client.request('GET','/fapi/v1/order',
                               dict(symbol=self.symbol,origClientOrderId=o.cid)))
            # Fees may arrive only on the user stream after a REST fill; retrieve
            # trade receipts for all orders missing actual commission quantities.
            fees=[]
            for o in list(self.k.oms.active.values())+list(self.k.oms.recent.values()):
                if o.exchange_id and o.filled>o.actual_fee_qty:
                    trades=await self.client.request('GET','/fapi/v1/userTrades',
                                                    dict(symbol=self.symbol,orderId=o.exchange_id,limit=1000))
                    fees.extend((o.cid,t) for t in trades)
            opens=await self.client.request('GET','/fapi/v1/openOrders')
            account=await self.client.request('GET','/fapi/v3/account')
            result=(replies,fees,opens,account)
        except Exception as e:result=e
        await self.queue.put(('audit',result,self.clock.ns()))

    def handle(self,kind,data,received_ns):
        k=self.k;ns=self.clock.ns()
        if ns-received_ns>1_000_000_000:
            k.external_healthy=False;k.oms.halt_reason='event_queue_lag'
        if kind in ('connected','disconnected'):
            self.connected[data]=(kind=='connected')
            if kind=='disconnected':
                k.external_healthy=False;k.reset_market()
                if self.mode=='live':k.oms.reconciled=False
                if data=='public':self.book_id=-1
            return
        if kind=='snapshot':
            k.snapshot(data,ns)
            return
        if kind=='result':
            cmd,result=data
            if isinstance(result,Exception):
                if isinstance(result,RequestFailure) and (result.status in (418,429) or result.code in (-1003,-1015)):
                    self.pause_until=ns+120_000_000_000;k.oms.halt_reason='exchange_rate_limit'
                if (isinstance(result,RequestFailure) and not result.ambiguous
                        and cmd.kind=='new'):
                    o=k.oms.find(cmd.cid)
                    # A preceding execution proves the request was accepted;
                    # preserve it even if a contradictory REST error arrives.
                    if o.phase==Phase.PENDING_NEW and o.filled==0:
                        k.update(o.cid,'REJECTED',D(0),D(0),ns)
                        if result.code==-2022:k.oms.reconciled=False
                        elif result.code!=-5022:k.oms.halt_reason=k.oms.halt_reason or 'exchange_rejection:'+str(result.code)
                    else:k.oms.uncertain(cmd.cid,'contradictory_rejection')
                else:k.oms.uncertain(cmd.cid,'request_outcome_unknown')
                return
            if result.get('clientOrderId')!=cmd.cid:
                k.oms.uncertain(cmd.cid,'response_cid_mismatch');return
            apply_order(k,result,ns);return
        if kind=='audit':
            if isinstance(data,Exception):k.oms.reconciled=False;return
            replies,fees,opens,account=data
            for row in replies:apply_order(k,row,ns)
            for cid,t in fees:
                # Results queried concurrently with fresh executions can be old;
                # absent receipts keep reconciliation closed until the next audit.
                o=k.oms.find(cid)
                if str(t['id']) in o.fee_receipts:continue
                if o.actual_fee_qty+D(t['qty'])>o.filled:continue
                k.oms.actual_fee(cid,t['id'],D(t['qty']),D(t['commission']),t['commissionAsset'],self.quote,ns)
            certain=all(o.phase!=Phase.UNKNOWN for o in k.oms.active.values())
            fees_done=all(o.actual_fee_qty==o.filled for o in [*k.oms.active.values(),*k.oms.recent.values()])
            k.oms.reconciled=(certain and fees_done and account_matches(k,account,opens,self.symbol,self.quote)
                              and self.connected['private'])
            return
        if kind=='event':
            label,event=data
            if label=='private':self.events.apply(event,ns);return
            if event.get('s')!=self.symbol:raise ValueError('foreign market symbol')
            kind=event.get('e')
            if kind=='depthUpdate':
                try:k.depth(event,ns,received_ns)
                except (ValueError,ArithmeticError,KeyError):
                    k.reset_market();self.depth_restart.set()
                return
            if kind=='bookTicker':
                uid=int(event['u'])
                if uid<=self.book_id:return
                self.book_id=uid
                k.top_of_book(ns,D(event['b']),D(event['B']) if 'B' in event else None,
                              D(event['a']),D(event['A']) if 'A' in event else None)
                # Clock monotonicity is processing time, age is actual receipt.
                k.strategy.obs.book_ns=received_ns
            elif kind=='aggTrade':k.trade(ns,int(event['a']),-1 if event['m'] else 1,D(event['p']),D(event['q']),received_ns=received_ns)
            elif kind=='markPriceUpdate':k.mark_price(received_ns,D(event['p']))

    async def keepalive(self):
        while True:
            await asyncio.sleep(25*60)
            try:await self.client.request('PUT','/fapi/v1/listenKey',signed=False,mutating=True)
            except Exception:
                await self.queue.put(('disconnected','private',self.clock.ns()));return

    async def run(self,seconds=0):
        s=self.symbol.lower();started=time.monotonic()
        if self.market_source=='depth':
            from .market_feed import DepthFeed
            self.spawn(DepthFeed(self.client,self.queue,self.clock,self.symbol,
                                 self.depth_restart,self.depth_speed).run())
        else:self.spawn(self.reader('public',PUBLIC+'?streams='+s+'@bookTicker'))
        self.spawn(self.reader('market',MARKET+'?streams='+s+'@aggTrade/'+s+'@markPrice@1s'))
        try:
            if self.mode=='live':
                mode=await self.client.request('GET','/fapi/v1/positionSide/dual')
                multi=await self.client.request('GET','/fapi/v1/multiAssetsMargin')
                if mode['dualSidePosition'] or multi['multiAssetsMargin']:raise ValueError('one-way single-asset account required')
                key=await self.client.request('POST','/fapi/v1/listenKey',signed=False,mutating=True)
                self.spawn(self.reader('private',PRIVATE+key['listenKey']));self.spawn(self.keepalive())
            while not self.stopping:
                if seconds and time.monotonic()-started>=seconds:break
                item=None
                try:
                    before=self.clock.ns()
                    timeout=max(1,next_timer_ns(self.k,before)-before)/1e9
                    item=await asyncio.wait_for(self.queue.get(),timeout=timeout)
                    try:self.handle(*item)
                    except Exception as e:
                        self.k.oms.halt_reason=self.k.oms.halt_reason or 'event_contract_failure:'+type(e).__name__
                        self.k.oms.reconciled=False
                except asyncio.TimeoutError:pass
                ns=self.clock.ns();k=self.k
                k.external_healthy=(self.connected['public'] and self.connected['market']
                                    and (self.mode=='shadow' or self.connected['private']))
                if self.mode=='live':
                    # Every intent including cancel is durable BEFORE transport.
                    # Feed/account/transport events are handled before each
                    # decision. Order lifetime and request caps remain in OMS.
                    commands=k.step(ns) if ns>=self.pause_until else []
                    if commands:
                        self.journal.save(k,self.events)
                        for command in commands:self.spawn(self.result(command))
                    if ns-self.last_audit>=5_000_000_000 and ns>=self.pause_until:
                        self.last_audit=ns
                        if not any(t.get_name()=='audit' for t in self.tasks):
                            self.spawn(self.audit()).set_name('audit')
                else:
                    k.plan(ns) # no fake ACKs/fills, no credentials, no orders
                if ns-self.last_output>=1_000_000_000:
                    self.last_output=ns;self.journal.save(k,self.events)
                    print(json.dumps(dict(mode=self.mode,ns=ns,ready=k.market_ready,
                                          reconciled=k.oms.reconciled,strategy=k.strategy.last,
                                          account=k.report()),default=str),flush=True)
                # Persist every received account/order event, including no-command
                # fills/fees, rather than waiting for the next display interval.
                if self.mode=='live' and item is not None and (item[0] in ('result','audit') or
                        (item[0]=='event' and item[1][0]=='private')):self.journal.save(k,self.events)
        finally:
            # Stop NEW intentions, keep user stream/results alive while canceling.
            self.k.external_healthy=False
            if self.mode=='live':
                self.k.oms.halt_reason=self.k.oms.halt_reason or 'shutdown'
                deadline=time.monotonic()+12
                while self.k.oms.active and time.monotonic()<deadline:
                    commands=self.k.step(self.clock.ns());self.journal.save(self.k,self.events)
                    for cmd in commands:self.spawn(self.result(cmd))
                    try:self.handle(*await asyncio.wait_for(self.queue.get(),timeout=.1))
                    except asyncio.TimeoutError:pass
                    except Exception:self.k.oms.reconciled=False
                for o in self.k.oms.active.values():self.k.oms.uncertain(o.cid,'shutdown_unconfirmed')
            self.journal.save(self.k,self.events)
            for task in list(self.tasks):task.cancel()
            await asyncio.gather(*list(self.tasks),return_exceptions=True)


async def main_async(args):
    case=load_selection(args.strategy,args.symbol)
    validate_source(case,getattr(args,'market_source','bbo'))
    if args.mask:raise ValueError('Public strategy cannot use a legacy RFI overlay')
    import aiohttp
    config=StrategyConfig.from_dict(json.loads(args.config.read_text()))
    async with aiohttp.ClientSession() as session:
        # Credentials are accessed ONLY in an explicitly selected live invocation.
        client=RestClient(session,os.environ['BINANCE_API_KEY'] if args.mode=='live' else '',
                          os.environ['BINANCE_API_SECRET'] if args.mode=='live' else '')
        metadata=await client.request('GET','/fapi/v1/exchangeInfo',signed=False)
        symbol=next(s for s in metadata['symbols'] if s['symbol']==args.symbol)
        rules=Rules.from_exchange(symbol);quote=symbol['marginAsset']
        server=await client.request('GET','/fapi/v1/time',signed=False)
        client.offset_ms=server['serverTime']-time.time_ns()//1_000_000
        if abs(client.offset_ms)>500:raise ValueError('synchronize system clock before running')
        k=StrategyEngine(rules,config,ProtectionSwitches.from_mask(args.mask),prefix='mm'+uuid.uuid4().hex[:12],strategy_factory=factory_for(case))
        identity=dict(application='crypto_MMbot',schema_version=1,symbol=args.symbol,quote=quote,mode=args.mode,config=config.json(),strategy=case,mask=args.mask,rules=asdict(rules),
                      account_key_fingerprint=hashlib.sha256(client.key.encode()).hexdigest() if args.mode=='live' else None)
        journal=Journal(args.journal,identity)
        try:await Runtime(k,client,journal,args.symbol,quote,args.mode,
                          market_source=getattr(args,'market_source','bbo'),
                          depth_speed=getattr(args,'depth_speed','100ms')).run(args.seconds)
        finally:journal.close()

