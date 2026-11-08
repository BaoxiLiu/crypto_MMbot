"""Compare immutable standard HTML payloads; never recalculate metrics."""
from pathlib import Path
from .html_report import read_report

SHARED = ('window', 'symbols', 'engine', 'execution', 'account', 'loss_stops',
          'fee_scenarios', 'observations')


def compare_reports(paths):
    if len(paths)<2:raise ValueError('compare requires at least two --html reports')
    reports=[read_report(Path(path)) for path in paths]
    reference=reports[0]
    if not reference['full_range']:
        raise ValueError('Partial reports lack complete matched coverage; inspect individually')
    for report in reports:
        if report['status']!='completed':raise ValueError('Cannot compare failed reports')
        if report['full_range']!=reference['full_range']:
            raise ValueError('Complete and partial reports cannot be compared as matched runs')
        for key in SHARED:
            if report['spec'].get(key)!=reference['spec'].get(key):
                raise ValueError('Comparison assumptions differ: '+key)
        if report['spec']['datasets']!=reference['spec']['datasets']:
            raise ValueError('Comparison datasets differ')
        # Profiles may change strategy parameters; every other frozen input
        # (marks, funding pages, exchange rules and account baseline) must match.
        external=lambda r:{p:h for p,h in r.get('input_sha256',{}).items()
                           if p not in r['spec'].get('_profile_inputs',{})}
        if not external(report) or external(report)!=external(reference):
            raise ValueError('Comparison input identities differ')
        shared_code=lambda r:{p:h for p,h in r.get('source_sha256',{}).items()
                             if p.startswith(('src/metrics/', 'app/market_making/fusion/'))
                             or p in ('app/market_making/core.py','app/market_making/contracts.py','app/market_making/as_math.py')}
        if not shared_code(report) or shared_code(report)!=shared_code(reference):
            raise ValueError('Comparison execution or metrics code differs')
    return dict(schema_version=1,full_range=reference['full_range'],matched=True,
        note='Existing standard metrics only; independent accounts, no new replay or portfolio inference',
        runs=[dict(run_id=r['run_id'],html=str(Path(p).resolve()),
                   totals=r['results']['totals']) for p,r in zip(paths,reports)])
