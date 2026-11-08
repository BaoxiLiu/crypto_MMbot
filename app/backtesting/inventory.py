"""Read-only source inventory against the downloader's content hashes.

This audits files and their recorded metadata, not L2 sequence continuity or
the supplier's completeness. Certified replay still runs datasets.prepare.
"""
import argparse
from collections import Counter
from datetime import datetime, timedelta
import json
from pathlib import Path

from .datasets import atomic_json, digest

KINDS = ('orderbook', 'trades', 'mark_price')
SYMBOLS = ('BTCUSDC', 'SOLUSDC', 'SUIUSDC', 'UNIUSDC', 'DOGEUSDC', '1000SHIBUSDC', 'WIFUSDC')


def audit(root, symbols=SYMBOLS, verify_hashes=True):
    root = Path(root)
    manifest = root / 'manifests/download.jsonl'
    latest = {}
    for line in manifest.read_text().splitlines():
        row = json.loads(line)
        latest[row['file']] = row
    rows, complete = [], {}
    for symbol in symbols:
        by_hour = {}
        kinds, source_rows = Counter(), Counter()
        problems, size, fingerprints = [], 0, []
        for key, recorded in sorted(latest.items()):
            kind = next((k for k in KINDS if key.endswith(f'/{symbol}_{k}.parquet')), None)
            if kind is None:
                continue
            pieces = Path(key).parts
            if len(pieces) != 4 or pieces[0] != 'binance_futures':
                raise ValueError('Unexpected download manifest path')
            stamp = datetime.fromisoformat(pieces[1] + 'T' + pieces[2] + ':00:00')
            path = root / 'raw' / key
            reason = None
            if recorded.get('status') != 'verified': reason = 'download_not_verified'
            elif not path.is_file(): reason = 'missing_file'
            elif path.stat().st_size != recorded['bytes']: reason = 'size_mismatch'
            elif recorded.get('empty') or recorded.get('rows', 0) <= 0: reason = 'empty_file'
            elif verify_hashes and digest(path) != recorded['sha256']: reason = 'hash_mismatch'
            if reason:
                problems.append({'file': key, 'reason': reason})
                continue
            by_hour.setdefault(stamp, set()).add(kind)
            kinds[kind] += 1
            source_rows[kind] += recorded['rows']
            size += recorded['bytes']
            fingerprints.append({'file': key, 'sha256': recorded['sha256']})
        complete[symbol] = {stamp for stamp, found in by_hour.items() if set(KINDS) <= found}
        stamps = sorted(complete[symbol])
        missing = []
        if stamps:
            stamp = stamps[0]
            while stamp <= stamps[-1]:
                if stamp not in complete[symbol]: missing.append(stamp.isoformat())
                stamp += timedelta(hours=1)
        rows.append({'symbol': symbol, 'start': stamps[0].isoformat() if stamps else None,
                     'end_exclusive': (stamps[-1] + timedelta(hours=1)).isoformat() if stamps else None,
                     'complete_hours': len(stamps), 'missing_hours_inside_span': missing,
                     'files_by_type': dict(kinds), 'reported_rows_by_type': dict(source_rows),
                     'bytes': size, 'problems': problems, 'source_files': fingerprints})
        print('Inventory', symbol, 'complete hours', len(stamps), 'problems', len(problems), flush=True)
    common = sorted(set.intersection(*(complete[s] for s in symbols))) if symbols else []
    ranges = []
    for stamp in common:
        if ranges and stamp == datetime.fromisoformat(ranges[-1]['end_exclusive']):
            ranges[-1]['end_exclusive'] = (stamp + timedelta(hours=1)).isoformat()
            ranges[-1]['hours'] += 1
        else:
            ranges.append({'start': stamp.isoformat(), 'end_exclusive': (stamp + timedelta(hours=1)).isoformat(), 'hours': 1})
    return {'schema_version': 1, 'status': 'source_inventory_not_replay_certification',
            'download_manifest': str(manifest.resolve()), 'manifest_sha256': digest(manifest),
            'hashes_verified': verify_hashes, 'symbols': rows, 'common_complete_hours': len(common),
            'common_contiguous_ranges': ranges,
            'caveats': ['Rows/schema are downloader metadata; L2 gaps and official trade overlap require replay certification.',
                        'A complete hourly file set does not prove uninterrupted market coverage.']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--symbols', nargs='+', default=list(SYMBOLS))
    args = parser.parse_args()
    if args.output.exists(): parser.error('Inventory output already exists')
    result = audit(args.root, args.symbols)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, result)
    print('Inventory saved', args.output, 'common hours', result['common_complete_hours'], flush=True)


if __name__ == '__main__': main()
