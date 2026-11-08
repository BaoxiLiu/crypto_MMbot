import json
from pathlib import Path
import tempfile
import unittest

import hftbacktest as h
import numpy as np
import polars as pl

from app.backtesting.datasets import prepare


class DatasetBoundaryTests(unittest.TestCase):
    def test_uninitialized_prefix_is_rejected_by_default_and_explicitly_audited_when_skipped(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);t=1789689600000
            path=root/'source/2026-09-18/00';path.mkdir(parents=True)
            book=[]
            for event,stamp,uid,prev in [('update',t,8,7),('snapshot',t+10000,10,None),('update',t+11000,11,10)]:
                for side,price in [('bid',100),('ask',102)]:
                    book.append(dict(event_type=event,received_time=stamp*1000000,event_time=stamp,
                        transaction_time=stamp,symbol='WIFUSDC',first_update_id=uid,final_update_id=uid,
                        prev_final_update_id=prev,last_update_id=uid if event=='snapshot' else None,
                        side=side,price=str(price),quantity='2'))
            pl.DataFrame(book).write_parquet(path/'WIFUSDC_orderbook.parquet')
            pl.DataFrame(dict(received_time=[(t+1000)*1000000,(t+12000)*1000000],event_time=[t+1000,t+12000],
                symbol=['WIFUSDC']*2,trade_id=[1,2],trade_time=[t+1000,t+12000],is_buyer_maker=[True]*2,
                price=['100']*2,quantity=['1']*2)).write_parquet(path/'WIFUSDC_trades.parquet')
            pl.DataFrame(dict(symbol=['WIFUSDC'],received_time=[t*1000000])).write_parquet(path/'WIFUSDC_mark_price.parquet')
            cfg=json.loads((Path(__file__).parent/'fixtures/gateway_fixed.json').read_text())
            cfg.update(source=str(root/'source'),cache=str(root/'cache'),start='2026-09-18T00:00:00',
                       end_exclusive='2026-09-18T01:00:00',trade_source='provider')
            with self.assertRaisesRegex(ValueError,'before initialization'):prepare(cfg)
            cfg['bootstrap_policy']='wait_first_snapshot'
            directory,meta=prepare(cfg)
            self.assertEqual(meta['excluded_uninitialized_depth'][0]['rows'],2)
            self.assertEqual(meta['segments'][0]['start_ns'],(t+10000)*1000000)
            with np.load(directory/meta['segments'][0]['files'][0]['path']) as z:
                self.assertGreaterEqual(z['data']['exch_ts'].min(),(t+10000)*1000000)
                trade=z['data'][(z['data']['ev']&255)==h.TRADE_EVENT]
                self.assertEqual(trade['exch_ts'].tolist(),[(t+12000)*1000000])

    def test_prior_hour_tail_and_published_gap_trades_stay_outside_new_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);t=1789689600000
            def book_rows(event,stamp,uid,prev,levels,received=None):
                return [{'event_type':event,'received_time':(received or stamp)*1000000,'event_time':stamp,
                         'transaction_time':stamp,'symbol':'WIFUSDC','first_update_id':prev if event=='update' else uid,
                         'final_update_id':uid,'prev_final_update_id':prev,
                         'last_update_id':uid if event=='snapshot' else None,'side':side,
                         'price':str(price),'quantity':str(quantity)} for side,price,quantity in levels]
            first=book_rows('snapshot',t,10,None,[('bid',100,2),('bid',99,2),('ask',102,2)])
            # Valid bridge received after the snapshot, transaction time 50ms earlier.
            first+=book_rows('update',t-50,11,10,[('bid',100,0)],received=t+10)
            first+=book_rows('update',t+3570000,12,11,[('bid',99,2),('ask',102,2)])
            second=book_rows('snapshot',t+3602000,100,None,[('bid',99,2),('ask',102,2)])
            second+=book_rows('update',t+3603000,101,100,[('bid',99,2),('ask',102,2)])
            for hour,book,times,ids in [('00',first,[t+10000,t+3571000,t+3597000,t+3599000],[1,2,3,4]),
                                      ('01',second,[t+3601000,t+3602500],[5,6])]:
                path=root/'source/2026-09-18'/hour;path.mkdir(parents=True)
                pl.DataFrame(book).write_parquet(path/'WIFUSDC_orderbook.parquet')
                pl.DataFrame({'received_time':[stamp*1000000 for stamp in times],'event_time':times,
                              'symbol':['WIFUSDC']*len(times),'trade_id':ids,'trade_time':times,
                              'is_buyer_maker':[True]*len(times),'price':['99']*len(times),
                              'quantity':['1']*len(times)}).write_parquet(path/'WIFUSDC_trades.parquet')
                pl.DataFrame({'symbol':['WIFUSDC'],'received_time':[t*1000000]}).write_parquet(path/'WIFUSDC_mark_price.parquet')
            cfg=json.loads((Path(__file__).parent/'fixtures/gateway_fixed.json').read_text())
            cfg.update(source=str(root/'source'),cache=str(root/'cache'),start='2026-09-18T00:00:00',
                       end_exclusive='2026-09-18T02:00:00',trade_source='provider')
            directory,meta=prepare(cfg)
            self.assertEqual(meta['version'],5)
            self.assertEqual(len(meta['segments']),2)
            datasets=[]
            for segment in meta['segments']:
                parts=[]
                for part in segment['files']:
                    with np.load(directory/part['path']) as z:parts.append(z['data'])
                datasets.append(np.concatenate(parts))
            old,new=datasets
            self.assertEqual(old['exch_ts'].max(),(t+3570000)*1000000)
            self.assertEqual(new['exch_ts'].min(),(t+3602000)*1000000)
            self.assertEqual(old['exch_ts'][(old['ev']&255)==h.TRADE_EVENT].tolist(),[(t+10000)*1000000])
            self.assertEqual(new['exch_ts'][(new['ev']&255)==h.TRADE_EVENT].tolist(),[(t+3602500)*1000000])
            self.assertEqual(meta['bootstrap_depth_updates'][0]['original_min_ns'],(t-50)*1000000)
            self.assertEqual(meta['bootstrap_depth_updates'][0]['final_update_id'],11)
            self.assertGreater(sum(r['trade_rows'] for r in meta['excluded_interval_events']),0)
            # Snapshot must precede buffered bridge: deleted bid is not reinserted.
            asset=h.BacktestAsset().data(old).linear_asset(1).tick_size(1).lot_size(1).risk_adverse_queue_model().no_partial_fill_exchange()
            bt=h.HashMapMarketDepthBacktest([asset])
            try:
                bt.elapse(5000001)
                self.assertEqual(bt.depth(0).best_bid,99)
                self.assertEqual(bt.depth(0).best_ask,102)
            finally:bt.close()
            # A second preparation must load the identical, sealed cache.
            _,cached=prepare(cfg)
            self.assertEqual(cached,meta)


if __name__=='__main__':unittest.main()
