import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import polars as pl
import hftbacktest as h
from app.backtesting.datasets import events, prepare

BASE = Path(__file__).parent / 'fixtures/gateway_fixed.json'


class BacktestTests(unittest.TestCase):



    def test_trade_direction_and_units(self):
        frame=pl.DataFrame({'is_buyer_maker':[True,False],'trade_time':[1000,1001],
                            'price':[1.,1.1],'quantity':[2.,3.]})
        out=events(frame,trades=True)
        self.assertTrue(int(out[0]['ev']) & h.SELL_EVENT)
        self.assertTrue(int(out[1]['ev']) & h.BUY_EVENT)
        self.assertEqual(out[0]['exch_ts'],1_000_000_000)


    def test_live_queue_full_fill(self):
        data=np.zeros(5,dtype=h.event_dtype)
        flags=h.EXCH_EVENT|h.LOCAL_EVENT
        values=[(0,h.DEPTH_SNAPSHOT_EVENT|h.BUY_EVENT,100.,2.),
                (0,h.DEPTH_SNAPSHOT_EVENT|h.SELL_EVENT,102.,2.),
                (500_000_000,h.TRADE_EVENT|h.SELL_EVENT,100.,3.),
                (700_000_000,h.TRADE_EVENT|h.SELL_EVENT,100.,2.),
                (1_000_000_000,h.DEPTH_EVENT|h.SELL_EVENT,102.,2.)]
        for i,(t,e,p,q) in enumerate(values): data[i]=(flags|e,t,t+5_000_000,p,q,0,0,0.)
        asset=(h.BacktestAsset().data(data).linear_asset(1.).tick_size(1.).lot_size(1.)
               .constant_order_latency(10_000_000,5_000_000).risk_adverse_queue_model()
               .no_partial_fill_exchange().trading_value_fee_model(0.,.0004))
        bt=h.HashMapMarketDepthBacktest([asset])
        try:
            bt.elapse(100_000_000)
            bt.submit_buy_order(0,1,100.,2.,h.GTX,h.LIMIT,False)
            bt.elapse(450_000_000)
            self.assertEqual(bt.position(0),2.) # whole order fills once queue exhausted
            bt.elapse(200_000_000)
            self.assertEqual(bt.position(0),2.)
            self.assertEqual(bt.state_values(0).fee,0.)
        finally: bt.close()

    def test_cancel_can_fill_before_effective_time(self):
        flags=h.EXCH_EVENT|h.LOCAL_EVENT
        data=np.zeros(4,dtype=h.event_dtype)
        rows=[(0,h.DEPTH_EVENT|h.BUY_EVENT,100.,0.),
              (0,h.DEPTH_EVENT|h.SELL_EVENT,102.,2.),
              (205_000_000,h.TRADE_EVENT|h.SELL_EVENT,99.,1.),
              (500_000_000,h.DEPTH_EVENT|h.SELL_EVENT,102.,2.)]
        for i,(t,e,p,q) in enumerate(rows): data[i]=(flags|e,t,t+5_000_000,p,q,0,0,0.)
        a=(h.BacktestAsset().data(data).linear_asset(1.).tick_size(1.).lot_size(1.)
           .constant_order_latency(10_000_000,5_000_000).risk_adverse_queue_model()
           .no_partial_fill_exchange().trading_value_fee_model(0.,.0004))
        b=h.HashMapMarketDepthBacktest([a])
        try:
            b.elapse(100_000_000)
            b.submit_buy_order(0,1,100.,1.,h.GTX,h.LIMIT,False)
            b.elapse(100_000_000)
            b.cancel(0,1,False)  # Effective at 210ms; sell at 205ms still hits order.
            b.elapse(50_000_000)
            self.assertEqual(b.position(0),1.)
        finally: b.close()

    def partial_engine(self):
        flags=h.EXCH_EVENT|h.LOCAL_EVENT
        data=np.zeros(5,dtype=h.event_dtype)
        rows=[(0,h.DEPTH_SNAPSHOT_EVENT|h.BUY_EVENT,100.,2.),
              (0,h.DEPTH_SNAPSHOT_EVENT|h.SELL_EVENT,102.,2.),
              (500_000_000,h.TRADE_EVENT|h.SELL_EVENT,100.,3.),
              (700_000_000,h.TRADE_EVENT|h.SELL_EVENT,100.,2.),
              (1_000_000_000,h.DEPTH_EVENT|h.SELL_EVENT,102.,2.)]
        for i,(t,e,p,q) in enumerate(rows): data[i]=(flags|e,t,t+5_000_000,p,q,0,0,0.)
        a=(h.BacktestAsset().data(data).linear_asset(1.).tick_size(1.).lot_size(1.)
           .constant_order_latency(10_000_000,5_000_000).risk_adverse_queue_model()
           .partial_fill_exchange().trading_value_fee_model(.001,.0004))
        b=h.HashMapMarketDepthBacktest([a]);b.elapse(100_000_000)
        b.submit_buy_order(0,1,100.,2.,h.GTX,h.LIMIT,False)
        return b

    def test_partial_fill_accounts_every_execution(self):
        b=self.partial_engine()
        try:
            b.elapse(450_000_000)
            self.assertEqual(b.position(0),1.)
            self.assertAlmostEqual(b.state_values(0).balance,-100.)
            self.assertAlmostEqual(b.state_values(0).fee,.1)
            b.elapse(200_000_000)
            self.assertEqual(b.position(0),2.)
            self.assertAlmostEqual(b.state_values(0).balance,-200.)
            self.assertAlmostEqual(b.state_values(0).fee,.2)
            self.assertEqual(b.state_values(0).num_trades,2)
        finally: b.close()

    def test_exact_maker_completion_removes_queue_order_on_both_sides(self):
        flags=h.EXCH_EVENT|h.LOCAL_EVENT
        for side in (1,-1):
            for partial in (False,True):
                with self.subTest(side=side,partial=partial):
                    price=100. if side==1 else 102.
                    direction=h.SELL_EVENT if side==1 else h.BUY_EVENT
                    data=np.zeros(6 if partial else 5,dtype=h.event_dtype)
                    rows=[(0,h.DEPTH_SNAPSHOT_EVENT|h.BUY_EVENT,100.,2.),
                          (0,h.DEPTH_SNAPSHOT_EVENT|h.SELL_EVENT,102.,2.),
                          (500_000_000,h.TRADE_EVENT|direction,price,3. if partial else 4.)]
                    if partial:rows.append((600_000_000,h.TRADE_EVENT|direction,price,1.))
                    rows.extend([(700_000_000,h.TRADE_EVENT|direction,price,1.),
                                 (1_000_000_000,h.DEPTH_EVENT|h.SELL_EVENT,102.,2.)])
                    for i,(stamp,event,p,q) in enumerate(rows):data[i]=(flags|event,stamp,stamp+5_000_000,p,q,0,0,0.)
                    asset=(h.BacktestAsset().data(data).linear_asset(1).tick_size(1).lot_size(1)
                           .constant_order_latency(10_000_000,5_000_000).risk_adverse_queue_model()
                           .partial_fill_exchange().trading_value_fee_model(.001,.0004))
                    bt=h.HashMapMarketDepthBacktest([asset]);bt.elapse(100_000_000)
                    try:
                        (bt.submit_buy_order if side==1 else bt.submit_sell_order)(0,1,price,2,h.GTX,h.LIMIT,False)
                        self.assertEqual(bt.elapse(700_000_000),0)
                        self.assertEqual(bt.position(0),side*2)
                        self.assertAlmostEqual(bt.state_values(0).balance,-side*price*2)
                        self.assertAlmostEqual(bt.state_values(0).fee,price*2*.001)
                    finally:bt.close()


    def test_cancel_after_partial_does_not_repeat_fill(self):
        b=self.partial_engine()
        try:
            b.elapse(450_000_000)
            b.cancel(0,1,False)
            b.elapse(200_000_000)
            self.assertEqual(b.position(0),1.)
            self.assertAlmostEqual(b.state_values(0).fee,.1)
            self.assertEqual(b.state_values(0).num_trades,1)
        finally: b.close()

    def test_rejected_cancel_echo_does_not_repeat_fill(self):
        b=self.partial_engine()
        try:
            b.elapse(595_000_000)  # now 695ms, cancel effective 705ms
            b.cancel(0,1,False)  # remaining fill at 700ms, then rejection echo
            b.elapse(100_000_000)
            self.assertEqual(b.position(0),2.)
            self.assertAlmostEqual(b.state_values(0).fee,.2)
            self.assertEqual(b.state_values(0).num_trades,2)
        finally: b.close()

    def test_official_overlap_uses_exact_decimals_and_rejects_conflicts(self):
        import hashlib,zipfile
        from datetime import datetime,timezone
        from app.backtesting.providers.binance_official import OfficialTrades
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'WIFUSDC';root.mkdir()
            archive=root/'WIFUSDC-trades-2026-09-18.zip'
            text='id,price,qty,quote_qty,time,is_buyer_maker\n1,0.2001,1.1,0.22011,1789689601000,false\n3,0.2002,2.0,0.4004,1789689602000,true\n'
            with zipfile.ZipFile(archive,'w') as z:z.writestr('trades.csv',text)
            archive.with_name(archive.name+'.CHECKSUM').write_text(hashlib.sha256(archive.read_bytes()).hexdigest())
            ref=OfficialTrades({'official_trades_root':directory,'symbol':'WIFUSDC'})
            hour=datetime(2026,9,18,tzinfo=timezone.utc)
            raw=pl.DataFrame({'trade_id':[1,2],'price':['0.2001000','0'],
                             'quantity':['1.1','0'],'trade_time':[1789689601000,1789689601500],
                             'is_buyer_maker':[False,False]})
            ref.verify(raw,hour)
            self.assertEqual(ref.verification[0]['na_present_in_official'],0)
            self.assertEqual(ref.for_hour(hour)['trade_id'].to_list(),[1,3])
            with self.assertRaises(ValueError):
                ref.verify(raw.with_columns(pl.when(pl.col('trade_id')==1).then(pl.lit('0.2002')).otherwise(pl.col('price')).alias('price')),hour)


    def test_snapshot_reset_and_missing_sequence(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); hour=root/'source/2026-09-18/00'; hour.mkdir(parents=True)
            t=1789689600000
            rows=[]
            for event,ms,uid,prev in [('snapshot',t,10,None),('update',t+100,11,10),
                                       ('snapshot',t+1000,20,None),('snapshot',t+1000,20,None),
                                       ('update',t+1100,21,20)]:
                for side,p in [('bid',1.),('ask',1.1)]:
                    rows.append({'event_type':event,'received_time':ms*1000000,'event_time':ms,
                                 'transaction_time':ms,'symbol':'WIFUSDC','first_update_id':prev if event=='update' else uid,
                                 'final_update_id':uid,'prev_final_update_id':prev,
                                 'last_update_id':uid if event=='snapshot' else None,
                                 'side':side,'price':str(p),'quantity':'10.'})
            book=pl.DataFrame(rows); book.write_parquet(hour/'WIFUSDC_orderbook.parquet')
            pl.DataFrame({'received_time':[(t+200)*1000000],'event_time':[t+200],
                          'symbol':['WIFUSDC'],'trade_id':[1],'trade_time':[t+200],
                          'is_buyer_maker':[True],'price':['1.'],'quantity':['1.']}).write_parquet(hour/'WIFUSDC_trades.parquet')
            pl.DataFrame({'symbol':['WIFUSDC'],'received_time':[t*1000000]}).write_parquet(hour/'WIFUSDC_mark_price.parquet')
            cfg=json.loads(BASE.read_text()); cfg.update(source=str(root/'source'),cache=str(root/'cache'),
                start='2026-09-18T00:00:00',end_exclusive='2026-09-18T01:00:00',trade_source='provider')
            prepared,m=prepare(cfg)
            self.assertEqual(len(m['segments']),2)
            self.assertEqual(len(m['resets']),1)
            # Duplicate reset snapshot must not create a third account.
            self.assertEqual(m['segments'][0]['events'],4) # gap trade excluded
            self.assertEqual(prepare(cfg)[1],m)
            changed=dict(m);changed['conversion_code_sha256']={'datasets.py':'tampered'}
            (prepared/'manifest.json').write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,'Prepared manifest identity mismatch'):
                prepare(cfg)
            (prepared/'manifest.json').write_text(json.dumps(m))
            bad=book.with_columns(pl.when(pl.col('final_update_id')==21).then(19)
                                 .otherwise(pl.col('prev_final_update_id')).alias('prev_final_update_id'))
            bad.write_parquet(hour/'WIFUSDC_orderbook.parquet')
            bad=bad.with_columns(pl.when(pl.col('final_update_id')==21).then(22).otherwise(pl.col('first_update_id')).alias('first_update_id'))
            bad.write_parquet(hour/'WIFUSDC_orderbook.parquet')
            with self.assertRaises(ValueError): prepare(cfg)


if __name__=='__main__': unittest.main()
