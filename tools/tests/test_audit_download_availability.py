import json
from pathlib import Path
import tempfile
import unittest

from tools.data.audit_download_availability import check_prefix, inventory, prefix_snapshot


class AvailabilityTest(unittest.TestCase):
    def test_prefix_append_allowed_rewrite_rejected_and_partial_tail_excluded(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); source = root/'live'; dest = root/'snapshot'
            source.write_bytes(b'{"one":1}\n{"pending":')
            identity = prefix_snapshot(source, dest)
            self.assertEqual(dest.read_bytes(), b'{"one":1}\n')
            source.write_bytes(source.read_bytes()+b'2}\n')
            check_prefix(identity)
            source.write_bytes(b'{"one":9}\n{"pending":2}\n')
            with self.assertRaises(ValueError):check_prefix(identity)

    def test_latest_attempt_and_disk_size_do_not_inflate_coverage(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); lines=[]
            scope=dict(start='2026-08-05',end_exclusive='2026-08-06',symbols=['SUIUSDC'],types=['orderbook','trades','mark_price'])
            for hour in range(24):
                for kind in scope['types']:
                    key=f'binance_futures/2026-08-05/{hour:02d}/SUIUSDC_{kind}.parquet'
                    path=root/'raw'/key; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(b'x')
                    entry=dict(file=key,status='verified',bytes=1,rows=1,empty=False,sha256='a'*64,schema='received_time: int64')
                    lines.append(json.dumps(entry))
            lines += [lines[0]]
            prefix=('\n'.join(lines)+'\n').encode()
            result=inventory(prefix,root,scope)
            self.assertEqual(result['common_file_complete_days'],1)
            self.assertEqual(result['repeated_attempt_records'],1)
            self.assertEqual(result['scope_counts']['verified_present'],72)
            # A later failed attempt invalidates coverage even if an older file remains.
            lines.append(json.dumps(dict(file=json.loads(lines[0])['file'],status='failed')))
            result=inventory(('\n'.join(lines)+'\n').encode(),root,scope)
            self.assertEqual(result['common_file_complete_days'],0)
            self.assertEqual(result['scope_counts']['verified_present'],71)
            path.write_bytes(b'xx')
            result=inventory(('\n'.join(lines)+'\n').encode(),root,scope)
            self.assertEqual(result['scope_counts']['size_mismatch'],1)
            self.assertEqual(result['scope_counts']['verified_present'],70)


if __name__ == '__main__':unittest.main()
