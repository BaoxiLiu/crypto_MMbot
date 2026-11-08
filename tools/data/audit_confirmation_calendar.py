"""Bind file-availability and known research dates without certifying unseen data.

No market values, predictions or strategy outcomes are read. Absence of a usage
record is deliberately unresolved; it never proves an unused confirmation date.
"""
import argparse
from collections import Counter
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def midnight(text):
    if len(text) == 10:
        return date.fromisoformat(text)
    value = datetime.fromisoformat(text.replace('Z', '+00:00'))
    if value.time().replace(tzinfo=None) != datetime.min.time():
        raise ValueError('Calendar contract requires midnight bounds')
    if value.utcoffset() not in (None, timedelta(0)):
        raise ValueError('Calendar bounds must be UTC')
    return value.date()


def ranges(days):
    result = []
    for day in sorted(set(days)):
        if result and result[-1]['end_exclusive'] == str(day):
            result[-1]['end_exclusive'] = str(day + timedelta(days=1))
            result[-1]['days'] += 1
        else:
            result.append(dict(start=str(day), end_exclusive=str(day + timedelta(days=1)), days=1))
    return result


def assess(availability, contract, control_snapshot, historical_run):
    if availability['status'] != 'completed_file_metadata_availability_not_certification':
        raise ValueError('Need recorded file-level availability')
    scope = availability['scope']; symbols = scope['symbols']
    if not symbols or len(set(symbols)) != len(symbols):
        raise ValueError('Duplicate or empty symbol scope')
    first = midnight(scope['start']); last = midnight(scope['end_exclusive'])
    if first >= last:
        raise ValueError('Empty availability calendar')
    days = [first + timedelta(days=i) for i in range((last-first).days)]
    expected = {(s, str(d)) for s in symbols for d in days}
    daily = availability['daily']
    if len(daily) != len(expected) or {(r['symbol'], r['date']) for r in daily} != expected:
        raise ValueError('Availability daily scope incomplete or duplicated')
    files = {}
    for r in daily:
        if r['expected_files'] != 72:
            raise ValueError('Need hourly book/trades/mark grain')
        valid = (r['counts'].get('verified_present', 0) == 72
                 and all(r['counts'].get('present_'+kind, 0) == 24
                         for kind in ('orderbook', 'trades', 'mark_price')))
        if r['file_complete'] != valid:
            raise ValueError('File completeness contradicts daily counts')
        files[(r['symbol'], r['date'])] = valid
    parity = control_snapshot['parity']; checked = parity['checks']
    if (parity['status'] != 'all_13_full_calendar_controls_exactly_reproduce_original'
            or len(checked) != len(symbols) or {r['symbol'] for r in checked} != set(symbols)
            or not all(r['exact_business_and_equity_bytes']
                       and r['effective_config_gamma_start_and_ledger_equal'] for r in checked)):
        raise ValueError('Need all complete control parities')
    tasks = contract['tasks']; controls = [t for t in tasks if t['arm'] == 'control']
    if len(controls) != len(symbols) or {t['symbol'] for t in controls} != set(symbols):
        raise ValueError('Complete control calendar differs')
    claims = []
    for t in controls:
        cfg = t['config']
        if cfg['symbol'] != t['symbol']:
            raise ValueError('Control symbol mismatch')
        claims.append(dict(symbol=t['symbol'], start=str(midnight(cfg['start'])),
            end_exclusive=str(midnight(cfg['end_exclusive'])), source='full_controls',
            registered_segment_replay=True))
    cfg = historical_run['config']
    if cfg['symbol'] not in symbols:
        raise ValueError('Historical usage outside symbol scope')
    claims.append(dict(symbol=cfg['symbol'], start=str(midnight(cfg['start'])),
        end_exclusive=str(midnight(cfg['end_exclusive'])), source='historical_run',
        registered_segment_replay=False))
    if any(midnight(x['start']) >= midnight(x['end_exclusive']) for x in claims):
        raise ValueError('Invalid research usage interval')
    rows = []
    for d in days:
        for s in symbols:
            used = [x for x in claims if x['symbol'] == s and x['start'] <= str(d) < x['end_exclusive']]
            rows.append(dict(date=str(d), symbol=s, file_complete=files[(s, str(d))],
                research_usage='known_used' if used else 'usage_unresolved',
                usage_sources=sorted({x['source'] for x in used}),
                registered_segment_replay=any(x['registered_segment_replay'] for x in used),
                unused_verified=False, confirmation_eligible=False))
    by_day = {d:[r for r in rows if r['date'] == str(d)] for d in days}
    common_complete = [d for d in days if all(r['file_complete'] for r in by_day[d])]
    any_used = [d for d in common_complete if any(r['research_usage'] == 'known_used' for r in by_day[d])]
    unknown = [d for d in common_complete if all(r['research_usage'] == 'usage_unresolved' for r in by_day[d])]
    after = [d for d in common_complete if any_used and d > max(any_used)]
    return dict(status='calendar_readiness_complete_usage_unresolved_not_confirmation',
        scope=scope, grain='symbol_utc_day', claims=claims, daily=rows,
        file_complete_common_ranges=ranges(common_complete),
        common_dates_with_any_known_usage=ranges(any_used),
        common_dates_without_observed_usage_unverified=ranges(unknown),
        common_file_dates_after_last_known_usage=ranges(after),
        usage_coin_days=dict(Counter(r['research_usage'] for r in rows)),
        verified_unused_common_days=0, confirmation_eligible_common_days=0,
        selects_confirmation_window=False, reads_market_values=False,
        economic_adoption=False,
        limitations=[
            'File metadata is a dated snapshot, not a fresh payload or market-quality audit.',
            'Registered replay calendar includes certified segments with explicit gap policies; it does not prove every instant was observed.',
            'Known use is conservative; other partner work, live observations and unindexed experiments remain unaudited.',
            'No observed use is unresolved, not unseen or eligible for confirmation.',
            'Earlier history cannot prove later-trained models were available at that time.',
            'Dates after the development window are not necessarily after candidate freeze.',
            'This ledger does not freeze the candidate, confirmation domain or 13-coin adoption gates.'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('availability', 'contract', 'control-snapshot', 'historical-run', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError('Calendar audit identity already exists')
    sources = {name:dict(path=str(getattr(a, name)), sha256=digest(getattr(a, name)))
               for name in ('availability', 'contract', 'control_snapshot', 'historical_run')}
    values = {name:json.loads(Path(v['path']).read_text()) for name, v in sources.items()}
    if digest(a.contract) != values['control_snapshot']['source_contract_sha256']:
        raise ValueError('Control snapshot does not bind actual contract')
    result = assess(**values)
    for v in sources.values():
        if digest(v['path']) != v['sha256']:
            raise ValueError('Calendar source changed during read')
    result.update(observed_utc=datetime.now(timezone.utc).isoformat(), sources=sources,
                  tool_sha256=digest(__file__))
    a.output.mkdir(parents=True)
    (a.output/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('daily','claims','sources','limitations')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
