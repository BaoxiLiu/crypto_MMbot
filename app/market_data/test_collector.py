import asyncio
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .collector import Collector, DepthSequence, Gap, load_config
from .storage import DiskFull, Store, archive, digest


def row(ms=0, seq=1, channel='depth'):
    return {'symbol': 'SUIUSDC', 'channel': channel, 'received_ms': ms,
            'seq': seq, 'data': {'b': [['1.1', '5']], 'a': [['1.2', '6']]}}


class SequenceTests(unittest.TestCase):
    def test_snapshot_bridge_duplicates_and_gap(self):
        s = DepthSequence({'lastUpdateId': 100})
        self.assertFalse(s.accept({'U': 80, 'u': 99, 'pu': 79}))
        self.assertTrue(s.accept({'U': 99, 'u': 102, 'pu': 98}))
        self.assertFalse(s.accept({'U': 99, 'u': 102, 'pu': 98}))
        self.assertTrue(s.accept({'U': 103, 'u': 110, 'pu': 102}))
        with self.assertRaises(Gap):
            s.accept({'U': 112, 'u': 120, 'pu': 111})

    def test_missing_snapshot_bridge_rejected(self):
        with self.assertRaises(Gap):
            DepthSequence({'lastUpdateId': 100}).accept({'U': 101, 'u': 110, 'pu': 99})


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / 'source'

    def tearDown(self):
        self.temp.cleanup()

    def store(self, **kwargs):
        return Store(self.root, min_free_bytes=1, **kwargs)

    def test_hour_and_channel_partition_roundtrip_manifest(self):
        s = self.store()
        for r in [row(), row(1000, 2, 'agg_trade'), row(3600000, 3)]:
            s.write(r)
        s.close()
        files = list(self.root.rglob('*.gz'))
        self.assertEqual(len(files), 3)
        records = []
        for f in files:
            with gzip.open(f, 'rt') as h:
                records += [json.loads(l) for l in h]
            m = json.loads(Path(str(f) + '.manifest.json').read_text())
            self.assertEqual(m['sha256'], digest(f))
            self.assertEqual(m['count'], 1)
        self.assertEqual({r['seq'] for r in records}, {1, 2, 3})
        self.assertTrue(all('/1970-01-01/' in str(f) for f in files))

    def test_restart_keeps_finalized_files_and_crash_evidence(self):
        s = self.store()
        s.write(row())
        s.close()
        partial = self.root / 'crash.gz.partial'
        partial.write_bytes(b'incomplete')
        s = self.store()
        self.assertIn('crash.gz.partial', s.incomplete)
        s.write(row(1000, 2))
        s.close()
        self.assertEqual(len(list(self.root.rglob('*.gz'))), 2)
        self.assertEqual(partial.read_bytes(), b'incomplete')

    def test_second_writer_refused(self):
        s = self.store()
        with self.assertRaises(RuntimeError):
            self.store()
        s.close()

    def test_low_disk_stops_without_deleting_unarchived_data(self):
        s = self.store()
        s.write(row())
        with patch('app.market_data.storage.shutil.disk_usage') as usage:
            usage.return_value.free = 0
            with self.assertRaises(DiskFull):
                s.write(row(1000, 2))
        s.close()
        self.assertEqual(len(list(self.root.rglob('*.gz'))), 1)

    def test_archive_copies_and_only_prunes_verified_expired_files(self):
        s = self.store()
        s.write(row())
        s.write(row(9 * 86400000, 2))
        s.close()
        destination = Path(self.temp.name) / 'archive'
        result = archive(self.root, destination, prune=True, now_ms=10 * 86400000)
        self.assertEqual(result, {'verified_files': 2, 'removed_files': 1})
        self.assertEqual(len(list(destination.rglob('*.gz'))), 2)
        self.assertEqual(len(list(self.root.rglob('*.gz'))), 1)

    def test_source_corruption_blocks_archive_and_deletion(self):
        s = self.store()
        s.write(row())
        s.close()
        f = next(self.root.rglob('*.gz'))
        f.write_bytes(b'corrupted')
        with self.assertRaises(ValueError):
            archive(self.root, Path(self.temp.name) / 'archive', prune=True, now_ms=10 * 86400000)
        self.assertTrue(f.exists())

    def test_archive_inside_source_rejected(self):
        s = self.store()
        s.close()
        with self.assertRaises(ValueError):
            archive(self.root, self.root / 'archive')


class CollectorTests(unittest.IsolatedAsyncioTestCase):
    async def test_shutdown_sentinel_preserves_queued_records(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = load_config(Path(__file__).with_name('config.example.json'))
            cfg.update(output=td, min_free_bytes=1)
            c = Collector(cfg)
            for i in range(20):
                c.emit('SUIUSDC', 'depth', {'number': i}, 'session', received_ms=0)
            c.queue.put_nowait(None)
            await c.writer()
            await c.queue.join()
            f = next(Path(td).rglob('*.gz'))
            with gzip.open(f, 'rt') as h:
                self.assertEqual([json.loads(line)['data']['number'] for line in h], list(range(20)))

    async def test_writer_drains_and_finalizes_all_three_channels(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = load_config(Path(__file__).with_name('config.example.json'))
            cfg.update(output=td, min_free_bytes=1)
            collector = Collector(cfg)
            for c in ('depth', 'agg_trade', 'book_ticker'):
                collector.emit('SUIUSDC', c, {'example': True}, 'session', received_ms=0)
            collector.queue.put_nowait({'_maintenance': 1000})
            task = asyncio.create_task(collector.writer())
            await collector.queue.join()
            await collector.queue.put(None)
            await task
            files = list(Path(td).rglob('*.gz'))
            self.assertEqual(len(files), 3)
            self.assertFalse(list(Path(td).rglob('*.partial')))
            for f in files:
                with gzip.open(f, 'rt') as h:
                    data = json.loads(h.readline())
                self.assertEqual(data['session_id'], 'session')
                self.assertEqual(data['run_id'], collector.store.run_id)

    async def test_queue_overflow_is_fatal_not_silent(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = load_config(Path(__file__).with_name('config.example.json'))
            cfg.update(output=td, min_free_bytes=1, queue_size=1)
            c = Collector(cfg)
            try:
                c.emit('SUIUSDC', 'depth', {}, 'session')
                with self.assertRaises(DiskFull):
                    c.emit('SUIUSDC', 'depth', {}, 'session')
            finally:
                c.store.close()


class ConfigTests(unittest.TestCase):
    def test_invalid_capacity_and_symbols_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'config.json'
            for cfg in ({'symbols': ['BTCUSDT'], 'output': td},
                        {'symbols': ['BTCUSDC', 'BTCUSDC'], 'output': td},
                        {'symbols': ['BTCUSDC'], 'output': td, 'min_free_bytes': 0}):
                path.write_text(json.dumps(cfg))
                with self.assertRaises(ValueError):
                    load_config(path)


if __name__ == '__main__':
    unittest.main()

class PilotTests(unittest.TestCase):
    def test_profile_and_legacy_defaults(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'config.json'
            p.write_text(json.dumps({'symbols':['WIFUSDC'], 'output':d}))
            old = load_config(p)
            self.assertEqual(old['channels'], ['book_ticker','depth','agg_trade'])
            self.assertEqual(old['depth_speed'], '100ms')
            self.assertNotIn('duration_seconds', old)
            p.write_text(json.dumps({'symbols':['WIFUSDC'], 'output':d,
                'channels':['trade','depth','book_ticker'], 'depth_speed':'0ms', 'duration_seconds':1}))
            self.assertEqual(load_config(p)['depth_speed'], '0ms')
            c = Collector(load_config(p))
            try:
                c.emit('WIFUSDC','trade',{'t':1,'p':'1','q':'2'},'session',1000,1000000001)
                r = c.queue.get_nowait()
                self.assertEqual(r['received_ns'],1000000001)
                c.store.write(r)
                asyncio.run(c.expire())
                self.assertTrue(c.stop.is_set())
                self.assertEqual(c.reason,'stopped: pilot duration complete')
            finally:
                c.store.close()
