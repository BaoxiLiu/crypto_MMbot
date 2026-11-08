"""Resumable hourly Parquet downloader. Run from project root; data stays ignored."""
import argparse, datetime as dt, hashlib, json, os, shutil, time, urllib.request, urllib.error
from pathlib import Path
import pyarrow.parquet as pq

BASE = 'https://api.cryptohftdata.com/v1/download?file='

def inspect(path, kind):
    table = pq.read_table(path)
    required = {'symbol', 'received_time'}
    required |= {'side', 'price', 'quantity', 'event_type'} if kind == 'orderbook' else {'price', 'quantity'} if kind == 'trades' else set()
    missing = required - set(table.column_names)
    if missing: raise ValueError('missing columns: ' + str(sorted(missing)))
    symbols = set(table['symbol'].to_pylist())
    if symbols and symbols != {'WIFUSDC'}: raise ValueError('unexpected symbol')
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): digest.update(chunk)
    return {'bytes': path.stat().st_size, 'sha256': digest.hexdigest(), 'rows': table.num_rows,
            'schema': str(table.schema), 'empty': table.num_rows == 0}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', default='2026-09-18')
    parser.add_argument('--end', default='2026-10-02', help='exclusive UTC date')
    parser.add_argument('--root', default='data/cryptohftdata')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    root = Path(args.root); root.mkdir(parents=True, exist_ok=True)
    lock = root / 'download.lock'
    try: fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError: raise SystemExit('Lock exists. Check process before removing stale lock.')
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    manifest = root / 'manifests/download.jsonl'; manifest.parent.mkdir(exist_ok=True)
    statusfile = root / 'reports/status.json'; statusfile.parent.mkdir(exist_ok=True)
    prior = {}
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            try: item = json.loads(line); prior[item['file']] = item
            except (ValueError, KeyError): pass
    counts = {'verified': 0, 'skipped': 0, 'missing': 0, 'failed': 0}
    def status(state, current=''):
        payload = dict(counts, state=state, current=current, pid=os.getpid(), start=args.start,
                       end_exclusive=args.end, updated_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        tmp = statusfile.with_suffix('.tmp'); tmp.write_text(json.dumps(payload, indent=2)); tmp.replace(statusfile)
    day = dt.date.fromisoformat(args.start); end = dt.date.fromisoformat(args.end); processed = 0
    try:
        while day < end:
            for hour in range(24):
                for kind in ('orderbook', 'trades', 'mark_price'):
                    key = f'binance_futures/{day}/{hour:02d}/WIFUSDC_{kind}.parquet'
                    status('running', key)
                    target = root / 'raw' / key; target.parent.mkdir(parents=True, exist_ok=True)
                    entry = {'file': key, 'downloaded_utc': dt.datetime.now(dt.timezone.utc).isoformat()}
                    try:
                        if shutil.disk_usage(root).free < 5_000_000_000: raise RuntimeError('disk space below 5GB')
                        old = prior.get(key, {})
                        if target.exists() and old.get('status') == 'verified':
                            checked = inspect(target, kind)
                            if checked['sha256'] == old['sha256']:
                                counts['skipped'] += 1; continue
                        part = target.with_suffix('.parquet.part')
                        for attempt in range(5):
                            try:
                                with urllib.request.urlopen(BASE + key, timeout=90) as response, part.open('wb') as out:
                                    shutil.copyfileobj(response, out)
                                break
                            except urllib.error.HTTPError as e:
                                if e.code == 404: raise
                                if e.code not in (429, 500, 502, 503, 504) or attempt == 4: raise
                                time.sleep(max(float(e.headers.get('Retry-After', 0)), 2 ** (attempt + 1)))
                            except (OSError, TimeoutError):
                                if attempt == 4: raise
                                time.sleep(2 ** (attempt + 1))
                        entry.update(inspect(part, kind))
                        if target.exists():
                            previous = hashlib.sha256(target.read_bytes()).hexdigest()
                            if previous != entry['sha256']:
                                target.rename(target.with_name(target.name + '.' + previous[:12] + '.previous'))
                        part.replace(target); entry['status'] = 'verified'; counts['verified'] += 1
                    except urllib.error.HTTPError as e:
                        entry.update(status='missing' if e.code == 404 else 'failed', http_status=e.code)
                        counts['missing' if e.code == 404 else 'failed'] += 1
                        if e.code in (401, 403): raise RuntimeError('authorization required; stopped') from None
                    except RuntimeError: raise
                    except Exception as e:
                        entry.update(status='failed', error=type(e).__name__ + ': ' + str(e)[:200]); counts['failed'] += 1
                    with manifest.open('a') as out: out.write(json.dumps(entry, ensure_ascii=False) + '\n')
                    print(json.dumps({'file': key, 'status': entry['status'], 'rows': entry.get('rows'), 'counts': counts}), flush=True)
                    processed += 1; status('running', key)
                    if args.limit and processed >= args.limit: status('sample_complete'); return
                    time.sleep(1.2)
            day += dt.timedelta(days=1)
        status('complete_with_gaps' if counts['missing'] or counts['failed'] else 'complete')
    except BaseException:
        status('stopped'); raise
    finally:
        lock.unlink(missing_ok=True)

if __name__ == '__main__': main()
