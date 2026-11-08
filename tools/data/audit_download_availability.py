"""Inventory an append-only download manifest without reading market values.

The result proves file-level availability, never certified or unseen data.
Only a frozen, newline-terminated prefix is consumed while downloading continues.
"""
import argparse
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re


KEY = re.compile(r"binance_futures/(\d{4}-\d{2}-\d{2})/(\d{2})/([A-Z0-9]+)_(orderbook|trades|mark_price)\.parquet")
KINDS = ('orderbook', 'trades', 'mark_price')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def prefix_snapshot(manifest, destination):
    with manifest.open('rb') as f:
        stat = f.fileno()
        import os
        identity = os.fstat(stat)
        data = f.read(identity.st_size)
    if len(data) != identity.st_size:
        raise ValueError('Manifest shrank during snapshot')
    end = data.rfind(b'\n') + 1
    prefix = data[:end]
    destination.write_bytes(prefix)
    return dict(path=str(manifest), device=identity.st_dev, inode=identity.st_ino,
                observed_bytes=len(data), prefix_bytes=end,
                incomplete_tail_bytes=len(data)-end, prefix_sha256=digest(prefix))


def check_prefix(identity):
    p = Path(identity['path'])
    with p.open('rb') as f:
        import os
        stat = os.fstat(f.fileno())
        data = f.read(identity['prefix_bytes'])
    if ((stat.st_dev, stat.st_ino) != (identity['device'], identity['inode'])
            or digest(data) != identity['prefix_sha256']):
        raise ValueError('Manifest prefix was replaced or rewritten')
    return dict(observed_bytes=stat.st_size,
                appended_bytes=stat.st_size-identity['observed_bytes'])


def parse_key(key):
    if not isinstance(key, str):
        raise ValueError('Manifest file key must be a string')
    match = KEY.fullmatch(key)
    if not match:
        raise ValueError('Unexpected manifest partition key')
    day, hour, symbol, kind = match.groups()
    date.fromisoformat(day)
    if not 0 <= int(hour) < 24:
        raise ValueError('Unexpected manifest hour')
    return day, int(hour), symbol, kind


def inventory(prefix, root, scope):
    latest = {}; attempts = Counter(); changed_hashes = set()
    verified_hashes = {}; parsed = {}
    for line in prefix.splitlines():
        entry = json.loads(line)
        key = entry['file']; parsed[key] = parse_key(key)
        status = entry['status']
        if status not in ('verified', 'missing', 'failed'):
            raise ValueError('Unknown download status')
        attempts[status] += 1
        if status == 'verified':
            new_hash = entry.get('sha256')
            if key in verified_hashes and verified_hashes[key] != new_hash:
                changed_hashes.add(key)
            verified_hashes[key] = new_hash
        latest[key] = entry
    start = date.fromisoformat(scope['start'])
    end = date.fromisoformat(scope['end_exclusive'])
    symbols = scope['symbols']; kinds = scope['types']
    if (end <= start or not symbols or len(set(symbols)) != len(symbols)
            or set(kinds) != set(KINDS) or len(kinds) != 3
            or any(not re.fullmatch('[A-Z0-9]+', s) for s in symbols)):
        raise ValueError('Invalid expected download scope')
    daily = defaultdict(Counter); schema = defaultdict(Counter)
    verified_bytes = 0; verified_empty = 0; in_scope_keys = set()
    for key, entry in latest.items():
        day, hour, symbol, kind = parsed[key]
        if not (scope['start'] <= day < scope['end_exclusive'] and symbol in symbols):
            continue
        in_scope_keys.add(key)
        status = entry['status']
        category = status
        if status == 'verified':
            size, rows = entry.get('bytes'), entry.get('rows')
            sha = entry.get('sha256'); text = entry.get('schema')
            empty = entry.get('empty')
            if (type(size) is not int or size < 0 or type(rows) is not int or rows < 0
                    or not isinstance(sha, str) or not re.fullmatch('[a-f0-9]{64}', sha)
                    or not isinstance(text, str) or not text
                    or type(empty) is not bool or empty != (rows == 0)):
                category = 'invalid_verified_metadata'
            else:
                path = root/'raw'/key
                if not path.is_file():
                    category = 'missing_on_disk'
                elif path.is_symlink() or path.resolve().is_relative_to((root/'raw').resolve()) is False:
                    category = 'unsafe_disk_path'
                elif path.stat().st_size != size:
                    category = 'size_mismatch'
                else:
                    category = 'verified_present'
                    verified_bytes += size
                    verified_empty += empty
                    schema[(symbol, kind)][digest(text.encode())] += 1
                    daily[(symbol, day)]['empty_files'] += empty
                    daily[(symbol, day)]['present_' + kind] += 1
        daily[(symbol, day)][category] += 1
    days = []
    for offset in range((end-start).days):
        day = str(start+timedelta(days=offset))
        for symbol in symbols:
            c = daily[(symbol, day)]
            attempted = sum(c[k] for k in ('verified_present', 'missing', 'failed',
                            'invalid_verified_metadata', 'missing_on_disk', 'unsafe_disk_path', 'size_mismatch'))
            days.append(dict(symbol=symbol, date=day, expected_files=72,
                not_attempted_files=72-attempted, counts=dict(c),
                file_complete=(c['verified_present'] == 72),
                all_files_nonempty=(c['verified_present'] == 72 and c['empty_files'] == 0)))
    totals = Counter()
    for row in days:
        totals.update(row['counts']); totals['not_attempted_files'] += row['not_attempted_files']
    common = [str(start+timedelta(days=i)) for i in range((end-start).days)
              if all(daily[(s,str(start+timedelta(days=i)))]['verified_present'] == 72 for s in symbols)]
    ranges = []
    for day in common:
        if ranges and date.fromisoformat(ranges[-1]['end_exclusive']) == date.fromisoformat(day):
            ranges[-1]['end_exclusive'] = str(date.fromisoformat(day)+timedelta(days=1))
            ranges[-1]['days'] += 1
        else:
            ranges.append(dict(start=day, end_exclusive=str(date.fromisoformat(day)+timedelta(days=1)), days=1))
    by_symbol = []
    for symbol in symbols:
        rows = [r for r in days if r['symbol'] == symbol]
        present = sum(r['counts'].get('verified_present', 0) for r in rows)
        by_symbol.append(dict(symbol=symbol, expected_files=len(rows)*72,
            verified_present=present, file_complete_days=sum(r['file_complete'] for r in rows),
            nonempty_complete_days=sum(r['all_files_nonempty'] for r in rows)))
    return dict(manifest_records=sum(attempts.values()), unique_files=len(latest),
        repeated_attempt_records=sum(attempts.values())-len(latest), attempt_statuses=dict(attempts),
        latest_statuses=dict(Counter(e['status'] for e in latest.values())),
        files_with_changed_verified_hash=len(changed_hashes),
        out_of_scope_files=len(latest)-len(in_scope_keys), expected_files=len(days)*72,
        scope_counts=dict(totals), verified_present_bytes=verified_bytes,
        verified_present_empty_files=verified_empty, by_symbol=by_symbol,
        common_file_complete_days=len(common), common_file_complete_ranges=ranges,
        schema_fingerprints=[dict(symbol=s,kind=k,hash_counts=dict(c)) for (s,k),c in sorted(schema.items())],
        daily=days)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'scope-status', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError('Availability audit output already exists')
    status_bytes = a.scope_status.read_bytes()
    status = json.loads(status_bytes)
    scope = {key:status[key] for key in ('start','end_exclusive','symbols','types')}
    a.output.mkdir(parents=True)
    try:
        identity = prefix_snapshot(a.root/'manifests/download.jsonl', a.output/'manifest-prefix.jsonl')
        write(a.output/'registration.json', dict(status='frozen_metadata_only_download_scope',
            created_utc=datetime.now(timezone.utc).isoformat(), root=str(a.root), scope=scope,
            scope_status_path=str(a.scope_status), scope_status_sha256=digest(status_bytes),
            downloader_observation={k:status[k] for k in ('pid','state','updated_utc') if k in status},
            manifest=identity, tool_sha256=digest(Path(__file__).read_bytes()),
            reads_market_values=False, selects_confirmation_window=False))
        result = inventory((a.output/'manifest-prefix.jsonl').read_bytes(), a.root, scope)
        result.update(status='completed_file_metadata_availability_not_certification', scope=scope,
            registration_sha256=digest((a.output/'registration.json').read_bytes()),
            source_after_audit=check_prefix(identity), economic_adoption=False,
            unseen_window_verified=False, limitations=[
                'Latest attempt per canonical file key is counted; older attempts do not add coverage.',
                'Disk checks are presence and size only; raw payload hashes are not recomputed.',
                'Empty files count as file coverage, not usable trades or book state.',
                'Hourly files do not prove timestamp coverage, book continuity or causal feature availability.',
                'No market values or new-window outcomes are read. Prior research usage is not audited.',
                'This is a frozen prefix snapshot, not live download percentage or confirmed unseen dates.'])
        write(a.output/'summary.json', result)
        print(json.dumps({k:result[k] for k in ('status','manifest_records','expected_files','scope_counts','common_file_complete_ranges')}, ensure_ascii=False))
    except BaseException as e:
        write(a.output/'failure.json', dict(status='failed_no_auto_retry', error_type=type(e).__name__))
        raise


if __name__ == '__main__':
    main()
