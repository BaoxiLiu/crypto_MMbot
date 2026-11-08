"""Download flow and source-schema tests; no replay or economic certification."""
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import urllib.error
from unittest.mock import patch

from tools.data import download_cryptohftdata_multi as dl


class DownloadTests(unittest.TestCase):
    def args(self,root,*extra):
        return ['--start','2026-10-03','--end','2026-10-04','--root',str(root),
                '--symbols',*dl.SUPPORTED_SYMBOLS[:7],'--attempts','1','--fail-fast',
                '--max-bytes','100000',*extra]

    def response(self,payload=b'PAR1data'):
        response=io.BytesIO(payload);response.headers={};return response

    def inspect(self,path,symbol,kind):
        return {'bytes':path.stat().st_size,'sha256':dl.sha(path),'rows':1,'schema':'fixture','empty':False}

    def test_full_day_seven_coins_retains_504_files_without_trading(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'raw-source';requests=[]
            def opener(request,timeout):
                requests.append(request.full_url);self.assertEqual(timeout,90);return self.response()
            with patch.object(dl.urllib.request,'urlopen',side_effect=opener),patch.object(dl,'inspect',side_effect=self.inspect),\
                    patch.object(dl.Throttle,'wait'),patch.object(dl.shutil,'disk_usage',return_value=SimpleNamespace(free=20_000_000_000)),\
                    patch('builtins.print'):
                dl.main(self.args(root))
            status=json.loads((root/'reports/selected-coins-download-status.json').read_text())
            self.assertEqual(status['state'],'complete');self.assertEqual(status['target'],504)
            self.assertEqual(status['verified'],504);self.assertEqual(status['requests'],504)
            self.assertEqual(status['retained_file_bytes'],504*8);self.assertFalse((root/'download.lock').exists())
            self.assertEqual(len(set(requests)),504)
            records=[json.loads(line) for line in (root/'manifests/download.jsonl').read_text().splitlines()]
            self.assertEqual(len(records),504);self.assertTrue(all(r['status']=='verified' for r in records))
            self.assertEqual({r['file'].split('/')[2] for r in records},{f'{i:02}' for i in range(24)})

    def test_access_and_service_errors_stop_one_attempt_and_leave_failure(self):
        for code in (401,403,404,418,429,503):
            with self.subTest(code=code),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)/'raw-source'
                with patch.object(dl.urllib.request,'urlopen',side_effect=urllib.error.HTTPError('public',code,'fixture',{},None)) as opener,\
                        patch.object(dl.Throttle,'wait'),patch.object(dl.shutil,'disk_usage',return_value=SimpleNamespace(free=20_000_000_000)),\
                        patch('builtins.print'):
                    with self.assertRaises(RuntimeError):dl.main(self.args(root))
                    self.assertEqual(opener.call_count,1)
                records=(root/'manifests/download.jsonl').read_text().splitlines();self.assertEqual(len(records),1)
                self.assertIn(json.loads(records[0])['status'],('failed','missing'))
                self.assertEqual(json.loads((root/'reports/selected-coins-download-status.json').read_text())['state'],'stopped')
                self.assertFalse((root/'download.lock').exists())

    def test_partial_bytes_respect_budget_and_no_second_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'raw-source';args=self.args(root)
            args[args.index('--max-bytes')+1]='4'
            with patch.object(dl.urllib.request,'urlopen',side_effect=lambda *a,**kw:self.response()) as opener,\
                    patch.object(dl.Throttle,'wait'),patch.object(dl.shutil,'disk_usage',return_value=SimpleNamespace(free=20_000_000_000)),\
                    patch('builtins.print'):
                with self.assertRaisesRegex(RuntimeError,'byte budget'):dl.main(args)
                self.assertEqual(opener.call_count,1)
            self.assertLessEqual(sum(p.stat().st_size for p in (root/'raw').rglob('*') if p.is_file()),4)
        class Stream:
            def __init__(self):self.chunks=iter((b'1234',b'5678',b''))
            def read(self,_):return next(self.chunks)
        out=io.BytesIO()
        with self.assertRaisesRegex(RuntimeError,'byte budget'):dl.copy_bounded(Stream(),out,6)
        self.assertEqual(out.getvalue(),b'1234')

    def test_low_disk_and_existing_lock_make_no_network_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'raw-source'
            with patch.object(dl.urllib.request,'urlopen') as opener,\
                    patch.object(dl.shutil,'disk_usage',return_value=SimpleNamespace(free=0)),patch('builtins.print'):
                with self.assertRaisesRegex(RuntimeError,'Free disk'):dl.main(self.args(root))
                opener.assert_not_called()
            (root/'download.lock').write_text('other owner')
            with patch.object(dl.urllib.request,'urlopen') as opener:
                with self.assertRaises(SystemExit):dl.main(self.args(root))
                opener.assert_not_called()
            self.assertEqual((root/'download.lock').read_text(),'other owner')

    def test_invalid_dates_duplicates_and_attempts_fail_before_root_creation(self):
        for extra in (['--start','2026-10-04'],['--attempts','0'],['--attempts','6'],
                      ['--symbols','BTCUSDC','BTCUSDC'],['--max-bytes','0'],['--end','2999-01-01']):
            with self.subTest(extra=extra),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)/'not-created'
                with patch.object(dl.urllib.request,'urlopen') as opener,patch('sys.stderr',new=io.StringIO()):
                    with self.assertRaises(SystemExit):dl.main(self.args(root,*extra))
                    opener.assert_not_called()
                self.assertFalse(root.exists())

    def test_schema_and_symbol_validation_use_actual_parquet(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        columns={'symbol':['WIFUSDC'],'received_time':[1_791_004_000_000_000_000],
                 'event_time':[1_791_004_000_000],'price':[1.0],'quantity':[2.0],
                 'trade_id':[1],'trade_time':[1_791_004_000_000],'is_buyer_maker':[True]}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'sample.parquet';pq.write_table(pa.table(columns),path)
            checked=dl.inspect(path,'WIFUSDC','trades');self.assertEqual(checked['rows'],1)
            self.assertEqual(checked['sha256'],dl.sha(path))
            with self.assertRaisesRegex(ValueError,'Unexpected symbol'):dl.inspect(path,'BTCUSDC','trades')
            with self.assertRaisesRegex(ValueError,'Missing required columns'):dl.inspect(path,'WIFUSDC','orderbook')

    def test_zstd_transport_decodes_to_same_parquet_and_records_identity(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        cols={'symbol':['BTCUSDC'],'received_time':[1],'event_time':[1],
              'price':[100.],'quantity':[1.],'trade_id':[1],'trade_time':[1],'is_buyer_maker':[False]}
        with tempfile.TemporaryDirectory() as tmp:
            raw=Path(tmp)/'raw.parquet';part=Path(tmp)/'trade.parquet.part'
            pq.write_table(pa.table(cols),raw)
            with pa.output_stream(part,compression='zstd') as out:out.write(raw.read_bytes())
            encoded=part.read_bytes();encoded_sha=dl.sha(part)
            with patch.object(dl.shutil,'disk_usage',return_value=SimpleNamespace(free=20_000_000_000)):
                proof=dl.normalize_partial(part,'BTCUSDC','trades')
            self.assertEqual(part.read_bytes(),raw.read_bytes());self.assertEqual(proof['transport_sha256'],encoded_sha)
            self.assertEqual(proof['transport_bytes'],len(encoded));self.assertEqual(proof['rows'],1)
            self.assertFalse(part.with_name(part.name+'.decoded').exists())

    def test_bad_or_oversized_zstd_preserves_partial(self):
        import pyarrow as pa
        with tempfile.TemporaryDirectory() as tmp:
            part=Path(tmp)/'source.part'
            with pa.output_stream(part,compression='zstd') as out:out.write(b'not a parquet file'*1000)
            original=part.read_bytes()
            with patch.object(dl.shutil,'disk_usage',return_value=SimpleNamespace(free=20_000_000_000)):
                with self.assertRaisesRegex(RuntimeError,'byte budget'):dl.normalize_partial(part,'BTCUSDC','trades',4)
                self.assertEqual(part.read_bytes(),original)
                with self.assertRaises(Exception):dl.normalize_partial(part,'BTCUSDC','trades')
            self.assertEqual(part.read_bytes(),original);self.assertFalse(part.with_name(part.name+'.decoded').exists())

    def test_recover_valid_partial_without_network_and_mark_limited_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);part=root/'raw/binance_futures/2026-10-03/00/BTCUSDC_orderbook.parquet.part'
            part.parent.mkdir(parents=True);part.write_bytes(b'PAR1data')
            with patch.object(dl.urllib.request,'urlopen') as opener,patch.object(dl,'inspect',side_effect=self.inspect),\
                    patch.object(dl.shutil,'disk_usage',return_value=SimpleNamespace(free=20_000_000_000)),patch('builtins.print'):
                dl.main(self.args(root,'--max-files','1','--name','recovery-smoke'))
            opener.assert_not_called();state=json.loads((root/'reports/recovery-smoke-status.json').read_text())
            self.assertEqual(state['verified'],1);self.assertEqual(state['recovered'],1)
            self.assertEqual(state['state'],'complete_requested_limit');self.assertEqual(state['target'],504)

    def test_thirteen_coins_and_legacy_comma_syntax(self):
        args=dl.parse_args(self.args('/tmp/download-test','--symbols',','.join(dl.SUPPORTED_SYMBOLS),'--name','thirteen-coins-backfill'))
        self.assertEqual(len(args.symbols),13);self.assertEqual(args.name,'thirteen-coins-backfill')


if __name__=='__main__':unittest.main()
