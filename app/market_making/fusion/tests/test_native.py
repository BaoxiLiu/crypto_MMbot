"""Run with quant_backtest python -m unittest ...test_native -v."""
import unittest
from decimal import Decimal as D
import numpy as np
try:
    import hftbacktest as h
    from hftbacktest.order import PARTIALLY_FILLED
except ImportError:
    h=None;PARTIALLY_FILLED=None
from app.market_making.core import Rules
from app.market_making.fusion.config import FusionConfig,Factors
from app.market_making.fusion.kernel import Kernel
from app.market_making.contracts import Desired
from app.market_making.fusion.native_reads import order_state
from app.market_making.fusion.tests.quote_fixture import QuoteFixture


@unittest.skipIf(h is None,'requires the verified quant_backtest environment')
class NativeContract(unittest.TestCase):
    def make(self,trades):
        rows=[(0,h.DEPTH_SNAPSHOT_EVENT|h.BUY_EVENT,100.,2.),
              (0,h.DEPTH_SNAPSHOT_EVENT|h.SELL_EVENT,102.,2.),*trades,
              (1_000_000_000,h.DEPTH_EVENT|h.SELL_EVENT,102.,2.)]
        data=np.zeros(len(rows),dtype=h.event_dtype)
        for i,(t,ev,p,q) in enumerate(rows):data[i]=(h.EXCH_EVENT|h.LOCAL_EVENT|ev,t,t+5_000_000,p,q,0,0,0.)
        a=(h.BacktestAsset().data(data).linear_asset(1.).tick_size(1.).lot_size(1.)
           .constant_order_latency(10_000_000,5_000_000).risk_adverse_queue_model()
           .partial_fill_exchange().trading_value_fee_model(.0002,.0004))
        b=h.HashMapMarketDepthBacktest([a]);b.elapse(100_000_000)
        rules=Rules(D(1),D(1),D(5),D(1),D(100000),D('.5'),D('1.5'))
        return b,Kernel(rules,FusionConfig(),Factors(), strategy_factory=QuoteFixture)

    def sync(self,b,k,c,ns):
        present,leaves,status,_=order_state(b,c.serial)
        cumulative=c.qty-D(str(leaves))
        labels={h.NEW:'NEW',PARTIALLY_FILLED:'PARTIALLY_FILLED',h.FILLED:'FILLED',h.CANCELED:'CANCELED',h.EXPIRED:'EXPIRED'}
        if status in labels:k.update(c.cid,labels[status],cumulative,cumulative*c.price,ns)
        self.assertEqual(float(k.oms.ledger.position),b.position(0))
        self.assertAlmostEqual(float(k.oms.cash_flow),b.state_values(0).balance-b.state_values(0).fee)

    def test_partial_executions_and_duplicates_use_leaves_not_last_exec_qty(self):
        b,k=self.make([(500_000_000,h.TRADE_EVENT|h.SELL_EVENT,100.,3.),
                       (700_000_000,h.TRADE_EVENT|h.SELL_EVENT,100.,2.)])
        try:
            cmd=k.oms.reconcile_targets({1:Desired(1,D(100),D(2))},D(101),100_000_000)[0]
            o=k.oms.find(cmd.cid);b.submit_buy_order(0,o.serial,100.,2.,h.GTX,h.LIMIT,False)
            b.elapse(450_000_000);self.sync(b,k,o,550_000_000)
            self.assertEqual(k.oms.ledger.position,1)
            self.sync(b,k,o,550_000_001)
            b.elapse(200_000_000);self.sync(b,k,o,750_000_000)
            self.assertEqual(k.oms.ledger.position,2)
            self.assertEqual(k.oms.ledger.fees,D('.04'))
        finally:b.close()

    def test_wrong_aggressor_direction_cannot_fill_passive_buy(self):
        b,k=self.make([(500_000_000,h.TRADE_EVENT|h.BUY_EVENT,100.,100.)])
        try:
            b.submit_buy_order(0,1,100.,2.,h.GTX,h.LIMIT,False)
            b.elapse(600_000_000);self.assertEqual(b.position(0),0.)
        finally:b.close()

    def test_cancel_request_does_not_instantly_remove_fillable_order(self):
        b,k=self.make([(205_000_000,h.TRADE_EVENT|h.SELL_EVENT,100.,3.)])
        try:
            c=k.oms.reconcile_targets({1:Desired(1,D(100),D(2))},D(101),100_000_000)[0]
            o=k.oms.find(c.cid);b.submit_buy_order(0,o.serial,100.,2.,h.GTX,h.LIMIT,False)
            b.elapse(100_000_000);self.sync(b,k,o,200_000_000)
            cancel=k.oms.reconcile_targets({},D(101),200_000_000)
            self.assertEqual(cancel[0].kind,'cancel');b.cancel(0,o.serial,False)
            b.elapse(50_000_000);self.sync(b,k,o,250_000_000)
            self.assertEqual(k.oms.ledger.position,1)
            self.assertFalse(k.oms.active)
        finally:b.close()

if __name__=='__main__':unittest.main()
