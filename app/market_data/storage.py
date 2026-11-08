"""Hourly, restart-safe gzip segments and verified local archive copies."""
import gzip
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
import uuid
from datetime import datetime, timezone


class DiskFull(RuntimeError):
    pass


def atomic_json(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class Store:
    def __init__(self, root, min_free_bytes=5_000_000_000, compression_level=6,
                 max_chunk_bytes=64_000_000):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = (self.root / '.collector.lock').open('a')
        try:
            fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.lock.close()
            raise RuntimeError('another collector already owns the output directory')
        self.min_free_bytes = min_free_bytes
        self.level = compression_level
        self.max_chunk_bytes = max_chunk_bytes
        self.open_files = {}
        self.run_id = uuid.uuid4().hex
        # Preserve crash leftovers as incomplete evidence, never certify or delete them.
        self.incomplete = [str(p.relative_to(self.root)) for p in self.root.rglob('*.partial')]

    def check_space(self):
        if shutil.disk_usage(self.root).free < self.min_free_bytes:
            raise DiskFull('free space below reserved capacity; collection stopped')

    def write(self, row):
        symbol, channel = row['symbol'], row['channel']
        if not re.fullmatch(r'[A-Z0-9]+USDC', symbol) or channel not in {
            'book_ticker', 'depth', 'agg_trade', 'trade', 'snapshot', 'quality'
        }:
            raise ValueError('invalid symbol/channel')
        self.check_space()
        dt = datetime.fromtimestamp(row['received_ms'] / 1000, timezone.utc)
        hour = dt.strftime('%Y%m%d_%H')
        key = symbol, channel
        state = self.open_files.get(key)
        if state and (state['hour'] != hour or state['raw'].tell() >= self.max_chunk_bytes):
            self.finish(key)
            state = None
        if state is None:
            directory = self.root / symbol / dt.strftime('%Y-%m-%d')
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f'{channel}_{hour}_{uuid.uuid4().hex}.jsonl.gz.partial'
            raw = path.open('xb')
            state = {'path': path, 'raw': raw,
                     'gzip': gzip.GzipFile(fileobj=raw, mode='wb', compresslevel=self.level),
                     'hour': hour, 'count': 0, 'first_ms': row['received_ms'],
                     'last_ms': row['received_ms'], 'first_seq': row['seq'], 'last_seq': row['seq']}
            self.open_files[key] = state
        state['gzip'].write((json.dumps(row, separators=(',', ':'), ensure_ascii=False) + '\n').encode())
        state['count'] += 1
        state['last_ms'], state['last_seq'] = row['received_ms'], row['seq']

    def flush(self):
        self.check_space()
        for state in self.open_files.values():
            state['gzip'].flush()
            state['raw'].flush()
            os.fsync(state['raw'].fileno())

    def finish(self, key):
        state = self.open_files.pop(key)
        state['gzip'].close()
        state['raw'].flush()
        os.fsync(state['raw'].fileno())
        state['raw'].close()
        path = state['path'].with_suffix('')
        os.replace(state['path'], path)
        meta = {k: state[k] for k in ('count', 'first_ms', 'last_ms', 'first_seq', 'last_seq')}
        meta.update(schema_version=1, run_id=self.run_id, symbol=key[0], channel=key[1],
                    path=str(path.relative_to(self.root)), bytes=path.stat().st_size,
                    sha256=digest(path), closed_ms=int(time.time() * 1000))
        atomic_json(Path(str(path) + '.manifest.json'), meta)

    def rotate(self, now_ms):
        hour = datetime.fromtimestamp(now_ms / 1000, timezone.utc).strftime('%Y%m%d_%H')
        for key, state in list(self.open_files.items()):
            if state['hour'] != hour:
                self.finish(key)

    def close(self):
        try:
            for key in list(self.open_files):
                self.finish(key)
        finally:
            self.lock.close()


def archive(root, destination, prune=False, retention_days=7, now_ms=None):
    """Copy finalized files, verify both hashes, then optionally prune expired originals.

    Destination must be a local or mounted archive directory. An SSH upload is
    deliberately outside this tool. Re-run after a crash; unfinished copies get replaced.
    """
    root, destination = Path(root).resolve(), Path(destination).resolve()
    if root == destination or root in destination.parents or destination in root.parents:
        raise ValueError('archive destination must be outside the source tree')
    if retention_days < 1:
        raise ValueError('retention_days must be positive')
    destination.mkdir(parents=True, exist_ok=True)
    cutoff = (now_ms if now_ms is not None else int(time.time() * 1000)) - retention_days * 86400000
    copied = removed = 0
    for manifest in sorted(root.rglob('*.manifest.json')):
        meta = json.loads(manifest.read_text())
        source = (root / meta['path']).resolve()
        if root not in source.parents:
            raise ValueError('manifest path escapes data directory')
        if not source.exists():
            continue
        expected = meta['sha256']
        if digest(source) != expected:
            raise ValueError(f'source hash mismatch: {source}')
        target = destination / source.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or digest(target) != expected:
            tmp = target.with_name(target.name + '.partial')
            shutil.copyfile(source, tmp)
            with tmp.open('rb') as f:
                os.fsync(f.fileno())
            if digest(tmp) != expected:
                raise ValueError(f'archive hash mismatch: {tmp}')
            os.replace(tmp, target)
        atomic_json(Path(str(target) + '.manifest.json'), meta)
        receipt = Path(str(source) + '.archived.json')
        atomic_json(receipt, {'sha256': expected, 'destination': str(target),
                              'verified_ms': int(time.time() * 1000)})
        copied += 1
        # Verify again immediately before removal; a receipt alone is insufficient.
        if prune and meta['last_ms'] < cutoff:
            if digest(target) != expected or digest(source) != expected:
                raise ValueError('hash changed before pruning')
            source.unlink()
            manifest.unlink()
            receipt.unlink()
            removed += 1
    return {'verified_files': copied, 'removed_files': removed}
