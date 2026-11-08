"""Optional L2 adapter: subscribe/buffer, request snapshot, then hand off in order.

The feed never mutates the book or account. Runtime owns state application.
A gap detected by the owner requests a reconnect/snapshot, rather than silently
continuing an uninitialized stream. No credentials are needed for this feed.
"""
import asyncio
from collections import deque
import json
from .gateway import PUBLIC


class DepthFeed:
    def __init__(self, client, queue, clock, symbol, restart, speed='100ms', limit=10000):
        if speed not in ('100ms', '0ms') or type(limit) is not int or limit < 1:
            raise ValueError('Invalid depth channel or buffer limit')
        self.client, self.queue, self.clock = client, queue, clock
        self.symbol, self.restart, self.speed, self.limit = symbol, restart, speed, limit

    async def connection(self):
        url = PUBLIC + '?streams=' + self.symbol.lower() + '@depth@' + self.speed
        async with self.client.session.ws_connect(url, heartbeat=20, timeout=10) as ws:
            await self.queue.put(('connected', 'public', self.clock.ns()))
            # WS is established before the snapshot request. Messages received
            # while REST is outstanding are kept, including the bridge packet.
            task = asyncio.create_task(self.client.request(
                'GET', '/fapi/v1/depth', dict(symbol=self.symbol, limit=1000), signed=False))
            buffered = deque()
            seeded = False
            try:
                async for message in ws:
                    if self.restart.is_set():
                        raise ValueError('Book owner requested resnapshot')
                    if message.type.name != 'TEXT':
                        break
                    ns = self.clock.ns()
                    raw = json.loads(message.data)
                    event = raw.get('data', raw)
                    if not seeded:
                        buffered.append((event, ns))
                        if len(buffered) > self.limit:
                            raise ValueError('Depth bootstrap buffer overflow')
                        if not task.done():
                            continue
                        snapshot = task.result()
                        await self.queue.put(('snapshot', snapshot, self.clock.ns()))
                        seeded = True
                        for data, received_ns in buffered:
                            await self.queue.put(('event', ('public', data), received_ns))
                        buffered.clear()
                    else:
                        await self.queue.put(('event', ('public', event), ns))
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    async def run(self):
        while True:
            self.restart.clear()
            try:
                await self.connection()
            except asyncio.CancelledError:
                raise
            except Exception:
                # Do not leak private transport URLs or arbitrary server payloads.
                pass
            await self.queue.put(('disconnected', 'public', self.clock.ns()))
            await asyncio.sleep(2)
