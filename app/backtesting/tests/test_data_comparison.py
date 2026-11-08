import json
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest
import zipfile

import polars as pl

from app.backtesting.inventory import audit, KINDS
from app.backtesting.datasets import digest
from app.backtesting.datasets import partition_trades
from app.backtesting.providers.binance_official import OfficialTrades

class InventoryTests(unittest.TestCase):
    def test_midnight_source_rows_match_exchange_dates_without_timestamp_tolerance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);day=root/'X';day.mkdir()
            boundary=1767312000000
            for date,oid,timestamp in (('2026-01-01',1,boundary-50),('2026-01-02',2,boundary+40)):
                archive=day/f'X-trades-{date}.zip'
                with zipfile.ZipFile(archive,'w') as z:
                    z.writestr('trades.csv',f'id,price,qty,quote_qty,time,is_buyer_maker\n{oid},100,1,100,{timestamp},false\n')
                archive.with_name(archive.name+'.CHECKSUM').write_text(digest(archive))
            official=OfficialTrades({'symbol':'X','official_trades_root':str(root)})
            raw=pl.DataFrame({'trade_id':[1,2],'price':['100','100'],'quantity':['1','1'],
                              'trade_time':[boundary-50,boundary+40],'is_buyer_maker':[False,False]})
            hour=datetime(2026,1,2,tzinfo=timezone.utc)
            official.verify(raw,hour)
            self.assertEqual(official.verification[-1]['matched_valid_trades'],2)
            self.assertEqual(official.for_hour(hour)['trade_id'].to_list(),[2])
            with self.assertRaisesRegex(ValueError,'overlap mismatch'):
                official.verify(raw.with_columns((pl.col('trade_time')+1).alias('trade_time')),hour)

    def test_zero_insurance_rows_require_official_absence_and_exact_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);day=root/'X';day.mkdir()
            archive=day/'X-trades-2026-01-01.zip'
            with zipfile.ZipFile(archive,'w') as z:
                z.writestr('trades.csv','id,price,qty,quote_qty,time,is_buyer_maker\n1,100,1,100,1767225600000,false\n')
            archive.with_name(archive.name+'.CHECKSUM').write_text(digest(archive))
            official=OfficialTrades({'symbol':'X','official_trades_root':str(root)})
            hour=datetime(2026,1,1,tzinfo=timezone.utc)
            raw=pl.DataFrame({'trade_id':[2],'trade_time':[1767225600000],'price':[0.],
                              'quantity':[0.],'order_type':['INSURANCE_FUND']})
            cfg={'invalid_trade_policy':'exclude_verified_zero_placeholders'}
            clean,rows=partition_trades(raw,cfg,official,hour)
            self.assertEqual(len(clean),0);self.assertFalse(rows[0]['official_present'])
            for value in ('UNKNOWN',None):
                with self.assertRaises(ValueError):partition_trades(raw.with_columns(pl.lit(value).alias('order_type')),cfg,official,hour)
            with self.assertRaises(ValueError):partition_trades(raw.with_columns(pl.lit(1.).alias('quantity')),cfg,official,hour)
            with self.assertRaises(ValueError):partition_trades(raw,cfg,None,hour)
            with self.assertRaisesRegex(ValueError,'official executions'):
                partition_trades(raw.with_columns(pl.lit(1).alias('trade_id')),cfg,official,hour)
            with self.assertRaises(ValueError):partition_trades(raw,{'invalid_trade_policy':'exclude_marked_na'},official,hour)

    def test_hash_mismatch_and_noncontiguous_hours_not_hidden(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'manifests').mkdir();rows=[]
            for symbol in ('A','B'):
                for hour in ('00','02'):
                    for kind in KINDS:
                        key=f'binance_futures/2026-01-01/{hour}/{symbol}_{kind}.parquet'
                        path=root/'raw'/key;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'fixture')
                        rows.append({'file':key,'status':'verified','rows':1,'bytes':7,'sha256':digest(path)})
            (root/'manifests/download.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            result=audit(root,('A','B'))
            self.assertEqual(result['common_complete_hours'],2)
            self.assertEqual(len(result['common_contiguous_ranges']),2)
            self.assertEqual(result['symbols'][0]['missing_hours_inside_span'],['2026-01-01T01:00:00'])
            path=root/'raw'/rows[-1]['file'];path.write_bytes(b'corrupt')
            result=audit(root,('A','B'))
            self.assertEqual(result['common_complete_hours'],1)
            self.assertEqual(result['symbols'][1]['problems'][0]['reason'],'hash_mismatch')




