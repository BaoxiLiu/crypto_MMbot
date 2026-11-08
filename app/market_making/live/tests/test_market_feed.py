import asyncio
import json
from types import SimpleNamespace
from decimal import Decimal as D
import pytest
from app.market_making.live.market_feed import DepthFeed
from app.market_making.live.runtime import Runtime
from app.market_making.live.journal import Journal
from .test_modules import engine, snapshot, depth


class Socket:
    def __init__(self, gate, release_on=1):
        self.gate, self.release_on, self.i = gate, release_on, 0
    def __aiter__(self): return self
    async def __anext__(self):
        if self.i >= 4: raise StopAsyncIteration
        if self.i == self.release_on: self.gate.set()
        await asyncio.sleep(0)
        self.i += 1
        data=dict(e='depthUpdate',s='DEMOUSDT',**depth(100+self.i,99+self.i))
        return SimpleNamespace(type=SimpleNamespace(name='TEXT'),data=json.dumps(data))


class Connection:
    def __init__(self, socket, order): self.socket, self.order=socket, order
    async def __aenter__(self): self.order.append('ws'); return self.socket
    async def __aexit__(self,*args): pass


class Client:
    def __init__(self, gate, release_on=1):
        self.order=[];self.gate=gate;self.socket=Socket(gate,release_on)
        self.session=self;self.cancelled=False
    def ws_connect(self,*args,**kwargs): return Connection(self.socket,self.order)
    async def request(self,*args,**kwargs):
        self.order.append('rest')
        assert kwargs == {'signed':False}
        try:
            await self.gate.wait()
            return snapshot()
        except asyncio.CancelledError:
            self.cancelled=True;raise


def test_subscribe_before_snapshot_and_preserve_buffered_message_order():
    async def run():
        queue=asyncio.Queue();client=Client(asyncio.Event());clock=iter(range(10,100))
        feed=DepthFeed(client,queue,SimpleNamespace(ns=lambda:next(clock)),'DEMOUSDT',asyncio.Event())
        await feed.connection()
        rows=[]
        while not queue.empty(): rows.append(queue.get_nowait())
        assert client.order == ['ws','rest']
        assert [r[0] for r in rows[:2]] == ['connected','snapshot']
        events=[r[1][1] for r in rows if r[0]=='event']
        assert [r['u'] for r in events] == [101,102,103,104]
    asyncio.run(run())


def test_buffer_overflow_fails_and_cancels_inflight_snapshot():
    async def run():
        client=Client(asyncio.Event(),release_on=100)
        feed=DepthFeed(client,asyncio.Queue(),SimpleNamespace(ns=lambda:10),'DEMOUSDT',asyncio.Event(),limit=2)
        with pytest.raises(ValueError,match='overflow'): await feed.connection()
        assert client.cancelled
    asyncio.run(run())


def test_depth_owner_gap_requests_resnapshot_and_keeps_account(tmp_path):
    k=engine();journal=Journal(tmp_path/'depth.sqlite',{})
    runtime=Runtime(k,None,journal,'DEMOUSDT','USDT','shadow',
                    SimpleNamespace(ns=lambda:1000),market_source='depth')
    runtime.handle('snapshot',snapshot(),1000)
    runtime.handle('event',('public',dict(e='depthUpdate',s='DEMOUSDT',**depth())),1000)
    assert k.market.ready
    runtime.handle('event',('public',dict(e='depthUpdate',s='DEMOUSDT',**depth(103,999))),1000)
    assert runtime.depth_restart.is_set() and not k.market.ready
    assert not k.market_ready and k.oms.ledger.position == 0
    assert not k.oms.halt_reason
    journal.close()


def test_source_receipt_age_is_distinct_from_processing_clock():
    k=engine();k.snapshot(snapshot(),100)
    k.depth(depth(),200,received_ns=150)
    assert k.market.received_ns == 200
    assert k.market.source_ns == 150 and k.strategy.obs.book_ns == 150
