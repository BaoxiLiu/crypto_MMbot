"""Actual HFT feed offset and request/response clocks, not a mock transport."""
import unittest,math
import numpy as np
try:import hftbacktest as h
except ImportError:h=None


@unittest.skipIf(h is None,'requires verified quant_backtest environment')
class ExecutionClock(unittest.TestCase):
 def make(self,offset=0,request=10_000_000,response=5_000_000):
  rows=[(1_000_000_000,h.DEPTH_SNAPSHOT_EVENT|h.BUY_EVENT,100.,2.),
        (1_000_000_000,h.DEPTH_SNAPSHOT_EVENT|h.SELL_EVENT,102.,2.),
        (1_200_000_000,h.TRADE_EVENT|h.SELL_EVENT,100.,3.),
        (1_400_000_000,h.TRADE_EVENT|h.SELL_EVENT,100.,3.),
        (2_000_000_000,h.DEPTH_EVENT|h.SELL_EVENT,102.,2.)]
  data=np.zeros(len(rows),dtype=h.event_dtype)
  for i,(t,e,p,q) in enumerate(rows):data[i]=(h.EXCH_EVENT|h.LOCAL_EVENT|e,t,t+5_000_000,p,q,0,0,0.)
  asset=(h.BacktestAsset().data(data).linear_asset(1).latency_offset(offset).tick_size(1).lot_size(1)
         .constant_order_latency(request,response).risk_adverse_queue_model().partial_fill_exchange()
         .trading_value_fee_model(.0002,.0005))
  b=h.HashMapMarketDepthBacktest([asset]);b.elapse(1);return b

 def advance(self,b,ns):
  self.assertGreaterEqual(ns,int(b.current_timestamp));self.assertEqual(b.elapse(ns-int(b.current_timestamp)),0)

 def test_positive_and_negative_feed_offset_shift_local_depth_not_exchange_events(self):
  for offset in (-3_000_000,7_000_000):
   with self.subTest(offset=offset):
    b=self.make(offset)
    try:
     ready=1_005_000_000+offset
     self.advance(b,ready-1);self.assertFalse(math.isfinite(b.depth(0).best_bid))
     self.advance(b,ready);self.assertEqual(b.depth(0).best_bid,100.)
    finally:b.close()

 def test_entry_and_response_are_distinct_clocks(self):
  b=self.make(request=200_000_000,response=70_000_000)
  try:
   self.advance(b,1_100_000_000);b.submit_buy_order(0,1,100.,1.,h.GTX,h.LIMIT,False)
   self.advance(b,1_300_000_000);self.assertEqual(b.position(0),0.) # 1.2s print preceded arrival
   self.advance(b,1_469_999_999);self.assertEqual(b.position(0),0.) # 1.4s execution, response not arrived
   self.advance(b,1_470_000_000);self.assertEqual(b.position(0),1.)
  finally:b.close()

 def test_public_print_does_not_change_visible_depth(self):
  b=self.make()
  try:
   self.advance(b,1_205_000_000)
   self.assertEqual(b.depth(0).best_bid,100.)
   self.assertEqual(b.depth(0).best_bid_qty,2.)
  finally:b.close()

if __name__=='__main__':unittest.main()
