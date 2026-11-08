"""Public Binance USD-M streams; one isolated collector for multiple symbols."""
import argparse
import asyncio
from collections import deque
import contextlib
import json
import logging
from pathlib import Path
import signal
import time
import uuid

from .storage import DiskFull, Store, archive, atomic_json

LOG = logging.getLogger('market_data')
EVENTS = {'book_ticker': 'bookTicker', 'depth': 'depthUpdate', 'agg_trade': 'aggTrade', 'trade': 'trade'}


class Gap(ValueError):
    pass


class DepthSequence:
    """Futures snapshot bridge: U <= lastUpdateId <= u, then pu == previous u."""
    def __init__(self, snapshot):
        self.last = int(snapshot['lastUpdateId'])
        self.bridged = False

    def accept(self, event):
        first, last = int(event['U']), int(event['u'])
        if first > last:
            raise Gap('inverted depth update range')
        if last < self.last or (self.bridged and last == self.last):
            return False
        if not self.bridged:
            if not first <= self.last <= last:
                raise Gap('snapshot does not bridge depth stream')
        elif int(event['pu']) != self.last:
            raise Gap(f'depth discontinuity: pu={event["pu"]}, expected={self.last}')
        self.last = last
        self.bridged = True
        return True


def load_config(path):
    cfg = json.loads(Path(path).read_text())
    defaults = {'min_free_bytes': 5_000_000_000, 'compression_level': 6,
                'queue_size': 20000, 'snapshot_buffer_size': 10000, 'snapshot_limit': 1000,
                'snapshot_timeout_seconds': 20, 'flush_seconds': 5,
                'health_seconds': 10, 'idle_seconds': 60, 'max_chunk_bytes': 64_000_000}
    cfg = defaults | cfg
    symbols = cfg.get('symbols', [])
    import re
    if not symbols or len(set(symbols)) != len(symbols) or any(
        not isinstance(s, str) or not re.fullmatch(r'[A-Z0-9]+USDC', s) for s in symbols
    ):
        raise ValueError('symbols must be unique uppercase USDC contract names')
    if not cfg.get('output'):
        raise ValueError('output directory is required')
    for key in defaults:
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if not 1 <= cfg['compression_level'] <= 9 or cfg['snapshot_limit'] not in {5, 10, 20, 50, 100, 500, 1000}:
        raise ValueError('invalid compression level or snapshot limit')
    if 'duration_seconds' in cfg and (type(cfg['duration_seconds']) is not int or cfg['duration_seconds'] <= 0):
        raise ValueError('duration_seconds must be positive')
    cfg.setdefault('channels', ['book_ticker', 'depth', 'agg_trade'])
    cfg.setdefault('depth_speed', '100ms')
    if not isinstance(cfg['channels'], list) or not cfg['channels'] or len(set(cfg['channels'])) != len(cfg['channels']) or any(c not in EVENTS for c in cfg['channels']):
        raise ValueError('invalid channels')
    if cfg['depth_speed'] not in {'100ms', '0ms'}:
        raise ValueError('invalid depth_speed')
    return cfg


class Collector:
    def __init__(self, cfg):
        self.cfg = cfg
        self.store = Store(cfg['output'], cfg['min_free_bytes'], cfg['compression_level'], cfg['max_chunk_bytes'])
        self.queue = asyncio.Queue(maxsize=cfg['queue_size'])
        self.seq = 0
        self.stop = asyncio.Event()
        self.streams = {}
        self.trade_ids = {}
        self.ticker_ids = {}
        self.session = None
        self.reason = 'running'

    def emit(self, symbol, channel, data, session_id, received_ms=None, received_ns=None):
        self.seq += 1
        row = {'schema_version': 1, 'run_id': self.store.run_id, 'seq': self.seq,
               'symbol': symbol, 'channel': channel, 'session_id': session_id,
               'received_ms': received_ms if received_ms is not None else int(time.time() * 1000),
               'received_ns': received_ns if received_ns is not None else time.time_ns(),
               'data': data}
        try:
            self.queue.put_nowait(row)
        except asyncio.QueueFull as e:
            # Stop globally: never silently discard events and claim continuous recording.
            raise DiskFull('writer queue overflow; recording has a gap and stopped') from e

    async def get(self, path, params=None):
        import aiohttp
        async with self.session.get('https://fapi.binance.com' + path, params=params,
                                    timeout=aiohttp.ClientTimeout(total=self.cfg['snapshot_timeout_seconds'])) as r:
            r.raise_for_status()
            return await r.json()

    async def validate_symbols(self):
        info = await self.get('/fapi/v1/exchangeInfo')
        valid = {s['symbol'] for s in info['symbols'] if s['status'] == 'TRADING'
                 and s['contractType'] == 'PERPETUAL' and s['quoteAsset'] == 'USDC'
                 and s['marginAsset'] == 'USDC'}
        if missing := set(self.cfg['symbols']) - valid:
            raise ValueError(f'not active USDC-margined perpetual contracts: {sorted(missing)}')

    async def stream(self, symbol, channel):
        import aiohttp
        s = symbol.lower()
        url = {'depth': f"wss://fstream.binance.com/public/stream?streams={s}@depth@{self.cfg.get('depth_speed', '100ms')}",
               'trade': f'wss://fstream.binance.com/public/stream?streams={s}@trade',
               'agg_trade': f'wss://fstream.binance.com/market/stream?streams={s}@aggTrade',
               'book_ticker': f'wss://fstream.binance.com/ws/{s}@bookTicker'}[channel]
        status = self.streams[f'{symbol}/{channel}'] = {
            'connected': False, 'messages': 0, 'reconnects': 0, 'last_received_ms': None}
        backoff = 1
        while not self.stop.is_set():
            sid = uuid.uuid4().hex
            snapshot_task = None
            status['depth_ready'] = False if channel == 'depth' else None
            buffer = deque()
            validator = None
            try:
                async with self.session.ws_connect(url, heartbeat=20, max_msg_size=4 * 1024 * 1024) as ws:
                    status['connected'] = True
                    self.emit(symbol, 'quality', {'event': 'connected', 'stream': channel}, sid)
                    while not self.stop.is_set():
                        message = await asyncio.wait_for(ws.receive(), self.cfg['idle_seconds'])
                        if message.type != aiohttp.WSMsgType.TEXT:
                            if message.type in {aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR}:
                                raise ConnectionError('websocket disconnected')
                            continue
                        received_ns = time.time_ns()
                        now = received_ns // 1_000_000
                        raw = json.loads(message.data)
                        event = raw.get('data', raw)
                        if event.get('e') != EVENTS[channel] or event.get('s') != symbol:
                            raise ValueError('unexpected event type or symbol')
                        self.emit(symbol, channel, event, sid, now, received_ns)
                        status['messages'] += 1
                        status['last_received_ms'] = now
                        backoff = 1
                        if channel == 'depth':
                            if validator is None:
                                if snapshot_task is None:
                                    # Buffer the first live update before requesting the snapshot.
                                    snapshot_task = asyncio.create_task(self.get('/fapi/v1/depth', {
                                        'symbol': symbol, 'limit': self.cfg['snapshot_limit']}))
                                buffer.append(event)
                                if len(buffer) > self.cfg['snapshot_buffer_size']:
                                    raise Gap('snapshot bootstrap buffer overflow')
                                if snapshot_task.done():
                                    snapshot = snapshot_task.result()
                                    self.emit(symbol, 'snapshot', snapshot, sid)
                                    validator = DepthSequence(snapshot)
                                    for item in buffer:
                                        validator.accept(item)
                                    buffer.clear()
                            else:
                                validator.accept(event)
                            if validator and validator.bridged and not status['depth_ready']:
                                status['depth_ready'] = True
                                self.emit(symbol, 'quality', {'event': 'depth_ready', 'last_update_id': validator.last,
                                                            'stream': channel}, sid)
                        elif channel in {'agg_trade', 'trade'}:
                            previous = self.trade_ids.get((symbol, channel))
                            current = int(event['a'] if channel == 'agg_trade' else event['t'])
                            if previous is not None and current != previous + 1:
                                self.emit(symbol, 'quality', {'event': 'trade_id_discontinuity', 'stream': channel,
                                                            'previous': previous, 'current': current}, sid)
                            self.trade_ids[(symbol, channel)] = current
                            if channel == 'trade' and (event.get('X') == 'NA' or float(event['p']) <= 0 or float(event['q']) <= 0):
                                self.emit(symbol, 'quality', {'event': 'trade_placeholder', 'stream': channel, 'trade_id': current}, sid)
                        else:
                            previous = self.ticker_ids.get(symbol)
                            current = int(event['u'])
                            if previous is not None and current < previous:
                                self.emit(symbol, 'quality', {'event': 'ticker_id_regression', 'stream': channel,
                                                            'previous': previous, 'current': current}, sid)
                            self.ticker_ids[symbol] = current
            except DiskFull:
                raise
            except Exception as e:
                status['last_error'] = str(e)
                status['reconnects'] += 1
                self.emit(symbol, 'quality', {'event': 'disconnected_or_gap', 'stream': channel,
                                             'error': str(e)}, sid)
                LOG.warning('%s/%s reconnect: %s', symbol, channel, e)
            finally:
                status['connected'] = False
                if channel == 'depth':
                    status['depth_ready'] = False
                if snapshot_task:
                    snapshot_task.cancel()
                    await asyncio.gather(snapshot_task, return_exceptions=True)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)

    async def writer(self):
        try:
            while True:
                row = await self.queue.get()
                try:
                    if row is None:
                        break
                    # Storage I/O and compression never block socket receipt processing.
                    batch = [row]
                    while len(batch) < 500 and not self.queue.empty():
                        item = self.queue.get_nowait()
                        if item is None:
                            batch.append(item)
                            break
                        batch.append(item)
                    try:
                        await asyncio.to_thread(self.write_batch, batch)
                    finally:
                        for _ in batch[1:]:
                            self.queue.task_done()
                    if batch[-1] is None:
                        break
                finally:
                    self.queue.task_done()
        finally:
            await asyncio.to_thread(self.store.close)

    def write_batch(self, batch):
        for row in batch:
            if row is None:
                continue
            if '_maintenance' in row:
                self.store.rotate(row['_maintenance'])
                self.store.flush()
            else:
                self.store.write(row)

    async def maintenance(self):
        last_flush = 0
        while True:
            await asyncio.sleep(1)
            # Use the same writer queue to avoid concurrent access to gzip handles.
            now = int(time.time() * 1000)
            if time.monotonic() - last_flush >= self.cfg['flush_seconds']:
                # A barrier goes through the writer rather than mutating Store here.
                self.queue.put_nowait({'_maintenance': now})
                last_flush = time.monotonic()

    async def health(self):
        while True:
            await asyncio.sleep(self.cfg['health_seconds'])
            await asyncio.to_thread(self.write_health)

    def write_health(self):
        atomic_json(self.store.root / 'health.json', {
            'updated_ms': int(time.time() * 1000), 'run_id': self.store.run_id,
            'state': self.reason, 'queue_size': self.queue.qsize(),
            'incomplete_files': self.store.incomplete, 'streams': self.streams})

    async def expire(self):
        await asyncio.sleep(self.cfg['duration_seconds'])
        self.reason = 'stopped: pilot duration complete'
        self.stop.set()

    async def run(self):
        import aiohttp
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(sig, self.stop.set)
        tasks = []
        writer = None
        error = None
        async with aiohttp.ClientSession() as session:
            self.session = session
            try:
                self.store.check_space()
                await self.validate_symbols()
                atomic_json(self.store.root / f'run_{self.store.run_id}.json', {
                    'started_ms': int(time.time() * 1000), 'config': self.cfg,
                    'schema_version': 1, 'run_id': self.store.run_id})
                writer = asyncio.create_task(self.writer())
                tasks = [asyncio.create_task(self.stream(s, c)) for s in self.cfg['symbols'] for c in self.cfg.get('channels', ['book_ticker', 'depth', 'agg_trade'])]
                tasks += [asyncio.create_task(self.maintenance()), asyncio.create_task(self.health())]
                if 'duration_seconds' in self.cfg:
                    tasks.append(asyncio.create_task(self.expire()))
                stopper = asyncio.create_task(self.stop.wait())
                tasks.append(stopper)
                done, _ = await asyncio.wait([writer, *tasks], return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    if task is not stopper:
                        task.result()
            except Exception as e:
                error = e
                self.reason = f'stopped: {type(e).__name__}: {e}'
                LOG.error(self.reason)
            finally:
                self.stop.set()
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                if writer and not writer.done():
                    # The writer can fail while a full queue blocks the sentinel.
                    sentinel = asyncio.create_task(self.queue.put(None))
                    await asyncio.wait([sentinel, writer], return_when=asyncio.FIRST_COMPLETED)
                    if not sentinel.done():
                        sentinel.cancel()
                    await asyncio.gather(sentinel, return_exceptions=True)
                    try:
                        await writer
                    except Exception as e:
                        error = error or e
                elif writer:
                    try:
                        writer.result()
                    except Exception as e:
                        error = error or e
                else:
                    await asyncio.to_thread(self.store.close)
                if error:
                    self.reason = f'stopped: {type(error).__name__}: {error}'
                elif self.reason == 'running':
                    self.reason = 'stopped: signal'
                await asyncio.to_thread(self.write_health)
        if error:
            raise error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for cmd in ('collect', 'check-config'):
        sub.add_parser(cmd).add_argument('--config', required=True)
    arch = sub.add_parser('archive')
    arch.add_argument('--source', required=True)
    arch.add_argument('--destination', required=True)
    arch.add_argument('--prune', action='store_true')
    arch.add_argument('--retention-days', type=int, default=7)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if args.command == 'archive':
        print(json.dumps(archive(args.source, args.destination, args.prune, args.retention_days)))
    elif args.command == 'check-config':
        print(json.dumps(load_config(args.config), indent=2))
    else:
        cfg = load_config(args.config)
        try:
            asyncio.run(Collector(cfg).run())
        except (DiskFull, OSError) as e:
            LOG.error('collection stopped; restart manually after fixing capacity/I/O: %s', e)
            raise SystemExit(75)


if __name__ == '__main__':
    main()
