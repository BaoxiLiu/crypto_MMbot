import copy
import unittest

from tools.data.audit_confirmation_calendar import assess, midnight, ranges
from datetime import date


def fixtures():
    symbols = ['A', 'B']; dates = ['2026-09-18', '2026-09-19', '2026-09-20']
    a = dict(status='completed_file_metadata_availability_not_certification',
        scope=dict(start=dates[0], end_exclusive='2026-09-21', symbols=symbols),
        daily=[dict(symbol=s, date=d, expected_files=72, file_complete=True,
                    counts=dict(verified_present=72, present_orderbook=24, present_trades=24, present_mark_price=24))
               for s in symbols for d in dates])
    c = dict(tasks=[dict(symbol=s, arm='control', config=dict(symbol=s,
                    start='2026-09-19T00:00:00Z', end_exclusive='2026-09-20T00:00:00Z')) for s in symbols])
    p = dict(parity=dict(status='all_13_full_calendar_controls_exactly_reproduce_original',
                checks=[dict(symbol=s, exact_business_and_equity_bytes=True,
                            effective_config_gamma_start_and_ledger_equal=True) for s in symbols]))
    h = dict(config=dict(symbol='A', start='2026-09-18', end_exclusive='2026-09-20'))
    return a, c, p, h


class CalendarTest(unittest.TestCase):
    def test_one_coin_used_removes_whole_common_day_and_unknown_is_not_unused(self):
        result = assess(*fixtures())
        self.assertEqual(result['usage_coin_days'], {'known_used':3, 'usage_unresolved':3})
        self.assertEqual(result['common_dates_with_any_known_usage'],
                         [dict(start='2026-09-18', end_exclusive='2026-09-20', days=2)])
        self.assertEqual(result['common_dates_without_observed_usage_unverified'],
                         [dict(start='2026-09-20', end_exclusive='2026-09-21', days=1)])
        self.assertEqual(result['confirmation_eligible_common_days'], 0)
        self.assertTrue(all(not r['unused_verified'] and not r['confirmation_eligible'] for r in result['daily']))

    def test_duplicate_and_missing_daily_rows_refused(self):
        for remove in (True, False):
            a,c,p,h = fixtures()
            if remove:a['daily'].pop()
            else:a['daily'][-1] = copy.deepcopy(a['daily'][0])
            with self.assertRaises(ValueError):assess(a,c,p,h)

    def test_incomplete_file_day_not_common_and_contradictory_metadata_refused(self):
        a,c,p,h = fixtures(); row=a['daily'][-1]
        row['counts']['verified_present'] = 71
        row['counts']['present_trades'] = 23
        with self.assertRaises(ValueError):assess(a,c,p,h)
        row['file_complete'] = False
        result = assess(a,c,p,h)
        self.assertEqual(result['common_dates_without_observed_usage_unverified'], [])

    def test_partial_control_proof_or_configuration_refused(self):
        a,c,p,h = fixtures();p['parity']['checks'].pop()
        with self.assertRaises(ValueError):assess(a,c,p,h)
        a,c,p,h = fixtures();c['tasks'][0]['config']['symbol'] = 'B'
        with self.assertRaises(ValueError):assess(a,c,p,h)

    def test_right_exclusive_midnight_and_non_utc_or_intraday_rejected(self):
        self.assertEqual(midnight('2026-09-19T00:00:00Z'), date(2026,9,19))
        for text in ('2026-09-19T00:01:00Z','2026-09-19T00:00:00+08:00'):
            with self.assertRaises(ValueError):midnight(text)
        self.assertEqual(ranges([date(2026,9,19),date(2026,9,19),date(2026,9,21)]),
            [dict(start='2026-09-19',end_exclusive='2026-09-20',days=1),
             dict(start='2026-09-21',end_exclusive='2026-09-22',days=1)])


if __name__ == '__main__':unittest.main()
