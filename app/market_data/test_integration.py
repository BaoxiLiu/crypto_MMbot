"""Offline end-to-end collection against a local HTTP/WebSocket exchange."""
import asyncio
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .collector import Collector, load_config

try:
    import aiohttp
    from aiohttp import web
except ImportError:
    aiohttp = web = None


@unittest.skipIf(aiohttp is None, 'aiohttp runtime dependency unavailable')
class OfflineIntegration(unittest.IsolatedAsyncioTestCase):
    async def test_snapshot_buffer_gap_reconnect_and_graceful_shutdown(self):
        connections = {'depth': 0}
        sockets = []

        async def exchange(request):
            stream = request.query.get('streams', request.path.rsplit('/', 1)[-1])
            channel = 'depth' if '@depth' in stream else ('agg_trade' if '@aggTrade' in stream else 'book_ticker')
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            sockets.append(ws)
            if channel == 'depth':
                connections['depth'] += 1
                await ws.send_json({'e': 'depthUpdate', 's': 'SUIUSDC', 'U': 99, 'u': 101,
                                    'pu': 98, 'b': [['1.1', '5']], 'a': [['1.2', '6']]})
                await asyncio.sleep(.08)
                await ws.send_json({'e': 'depthUpdate', 's': 'SUIUSDC', 'U': 102, 'u': 103,
                                    'pu': 101, 'b': [], 'a': []})
                if connections['depth'] == 1:
                    await ws.send_json({'e': 'depthUpdate', 's': 'SUIUSDC', 'U': 105, 'u': 106,
                                        'pu': 104, 'b': [], 'a': []})
            elif channel == 'agg_trade':
                for n in (100, 102):
                    await ws.send_json({'e': 'aggTrade', 's': 'SUIUSDC', 'a': n,
                                        'p': '1.2', 'q': '1', 'm': False})
            else:
                await ws.send_json({'e': 'bookTicker', 's': 'SUIUSDC', 'u': 101,
                                    'b': '1.1', 'B': '5', 'a': '1.2', 'A': '6'})
            async for _ in ws:
                pass
            return ws

        app = web.Application()
        app.router.add_get('/{tail:.*}', exchange)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '127.0.0.1', 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        original = aiohttp.ClientSession.ws_connect

        def redirect(session, url, **kwargs):
            from urllib.parse import urlsplit
            parts = urlsplit(url)
            return original(session, f'http://127.0.0.1:{port}{parts.path}?{parts.query}', **kwargs)

        async def fake_get(self, path, params=None):
            if path.endswith('exchangeInfo'):
                return {'symbols': [{'symbol': 'SUIUSDC', 'quoteAsset': 'USDC', 'marginAsset': 'USDC',
                                     'status': 'TRADING', 'contractType': 'PERPETUAL'}]}
            await asyncio.sleep(.04)
            return {'lastUpdateId': 100, 'bids': [['1.1', '5']], 'asks': [['1.2', '6']]}

        try:
            with tempfile.TemporaryDirectory() as td:
                cfg = load_config(Path(__file__).with_name('config.example.json'))
                cfg.update(output=td, symbols=['SUIUSDC'], min_free_bytes=1)
                collector = Collector(cfg)
                with patch.object(aiohttp.ClientSession, 'ws_connect', redirect), patch.object(Collector, 'get', fake_get):
                    task = asyncio.create_task(collector.run())
                    try:
                        # Await a real reconnect and bridged second session, bounded to 4 seconds.
                        async with asyncio.timeout(4):
                            while connections['depth'] < 2 or not collector.streams.get('SUIUSDC/depth', {}).get('depth_ready'):
                                await asyncio.sleep(.02)
                    finally:
                        collector.stop.set()
                        await task
                records = []
                for p in Path(td).rglob('*.gz'):
                    with gzip.open(p, 'rt') as f:
                        records.extend(json.loads(line) for line in f)
                channels = {r['channel'] for r in records}
                self.assertTrue({'depth', 'snapshot', 'book_ticker', 'agg_trade', 'quality'} <= channels)
                quality = [r['data']['event'] for r in records if r['channel'] == 'quality']
                self.assertIn('disconnected_or_gap', quality)
                self.assertIn('trade_id_discontinuity', quality)
                self.assertGreaterEqual(quality.count('depth_ready'), 2)
                self.assertFalse(list(Path(td).rglob('*.partial')))
                health = json.loads((Path(td) / 'health.json').read_text())
                self.assertEqual(health['state'], 'stopped: signal')
        finally:
            for ws in sockets:
                await ws.close()
            await runner.cleanup()


if __name__ == '__main__':
    unittest.main()
