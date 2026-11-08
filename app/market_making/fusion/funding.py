"""Public settled funding history; never interpret a predicted rate as settled."""
import argparse
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def utc_ms(value):
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1000)

ENDPOINT = 'https://fapi.binance.com/fapi/v1/fundingRate'
DOCS = 'https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Get-Funding-Rate-History'


def validate_rows(rows, symbol):
    if not isinstance(rows, list):
        raise ValueError('Funding response must be an array')
    by_time = {}
    for row in rows:
        if row.get('symbol') != symbol:
            raise ValueError('Funding symbol mismatch')
        ts = row.get('fundingTime')
        if not isinstance(ts, int) or isinstance(ts, bool) or ts <= 0:
            raise ValueError('Invalid funding settlement timestamp')
        try:
            rate, price = Decimal(str(row['fundingRate'])), Decimal(str(row['markPrice']))
        except (KeyError, InvalidOperation) as error:
            raise ValueError('Invalid settled funding values') from error
        if not rate.is_finite() or not price.is_finite() or price <= 0:
            raise ValueError('Invalid settled funding values')
        normalized = dict(symbol=symbol, fundingTime=ts, fundingRate=str(rate), markPrice=str(price))
        if ts in by_time:
            previous = by_time[ts]
            if Decimal(previous['fundingRate']) != rate or Decimal(previous['markPrice']) != price:
                raise ValueError('Conflicting funding settlement duplicates')
        by_time[ts] = normalized
    return [by_time[t] for t in sorted(by_time)]


def fetch(symbol, start, end_exclusive, output, request=None):
    """Fetch every API page in the requested half-open UTC interval, with hashes."""
    start_ms, end_ms = utc_ms(start), utc_ms(end_exclusive)
    if start_ms >= end_ms:
        raise ValueError('Empty funding request range')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if request is None:
        def request(url):
            with urlopen(url, timeout=30) as response:
                return json.load(response)
    cursor, collected, pages = start_ms, [], []
    while cursor < end_ms:
        url = ENDPOINT + '?' + urlencode(dict(symbol=symbol, startTime=cursor, endTime=end_ms - 1, limit=1000))
        raw = request(url)
        rows = validate_rows(raw, symbol)
        if any(not cursor <= r['fundingTime'] < end_ms for r in rows):
            raise ValueError('Funding API returned records outside its query')
        path = output / f'page_{len(pages):03d}.json'
        path.write_text(json.dumps(raw, indent=2) + '\n')
        pages.append(dict(path=path.name, sha256=sha256(path), url=url, rows=len(raw)))
        collected.extend(rows)
        if len(raw) < 1000:
            break
        if not rows or rows[-1]['fundingTime'] < cursor:
            raise ValueError('Funding pagination made no progress')
        cursor = rows[-1]['fundingTime'] + 1
    collected = validate_rows(collected, symbol)
    if not collected:
        raise ValueError('No settled funding history; absence must not imply zero funding')
    target = output / 'settlements.json'
    target.write_text(json.dumps(collected, indent=2) + '\n')
    manifest = dict(schema_version=1, symbol=symbol, start=start, end_exclusive=end_exclusive,
                    retrieved_utc=datetime.now(timezone.utc).isoformat(), endpoint=ENDPOINT, docs=DOCS,
                    pages=pages, rows=len(collected), path=target.name, sha256=sha256(target),
                    coverage_policy='All pages of official API in requested range; no assumed fixed funding interval.',
                    status='official_api_history_not_independently_audited')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def load(manifest_path, symbol, start, end_exclusive):
    path = Path(manifest_path)
    manifest = json.loads(path.read_text())
    if manifest['schema_version'] != 1 or manifest['symbol'] != symbol:
        raise ValueError('Funding manifest symbol/schema mismatch')
    if utc_ms(manifest['start']) > utc_ms(start) or utc_ms(manifest['end_exclusive']) < utc_ms(end_exclusive):
        raise ValueError('Funding history does not cover the run request')
    target = path.parent / manifest['path']
    if sha256(target) != manifest['sha256']:
        raise ValueError('Settled funding history hash mismatch')
    for page in manifest['pages']:
        if sha256(path.parent / page['path']) != page['sha256']:
            raise ValueError('Raw funding page hash mismatch')
    rows = validate_rows(json.loads(target.read_text()), symbol)
    if len(rows) != manifest['rows'] or not rows:
        raise ValueError('Funding manifest row count mismatch')
    if any(not utc_ms(manifest['start']) <= r['fundingTime'] < utc_ms(manifest['end_exclusive']) for r in rows):
        raise ValueError('Funding records outside manifest range')
    return [r for r in rows if utc_ms(start) <= r['fundingTime'] < utc_ms(end_exclusive)], manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', required=True)
    parser.add_argument('--start', required=True)
    parser.add_argument('--end-exclusive', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = fetch(args.symbol, args.start, args.end_exclusive, args.output)
    print(json.dumps({'rows': result['rows'], 'manifest': str(args.output / 'manifest.json')}))


if __name__ == '__main__':
    main()
