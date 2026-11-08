import pytest
from tools.data.prepare_quarantined_event_dataset import usable_ranges


def test_quarantine_divides_requested_period_without_replaying_bad_hour():
    ranges=usable_ranges('2026-09-01','2026-10-01',['2026-09-03T09:00:00+00:00'])
    assert [(a.isoformat(),b.isoformat()) for a,b in ranges]==[
        ('2026-09-01T00:00:00+00:00','2026-09-03T09:00:00+00:00'),
        ('2026-09-03T10:00:00+00:00','2026-10-01T00:00:00+00:00')]
    assert sum((b-a).total_seconds() for a,b in ranges)==719*3600


@pytest.mark.parametrize('excluded',[
    ['2026-08-31T23:00:00'],['2026-10-01T00:00:00'],
    ['2026-09-03T09:30:00'],['2026-09-03T09:00:00']*2])
def test_bad_quarantine_spec_is_rejected(excluded):
    with pytest.raises(ValueError):usable_ranges('2026-09-01','2026-10-01',excluded)
