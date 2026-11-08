import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from tools.factors.evaluate_flow_factors import forward_labels, ic, bh, signflip_p, interval, CONTROLS, GROUPS, TARGETS, prepare_hour_frame
class EvaluationTests(unittest.TestCase):
    def test_future_return_exact_and_hour_tail(self):
        mid=np.arange(100.,140.)
        r,v=forward_labels(mid)
        self.assertAlmostEqual(r[0],500)
        self.assertTrue(np.isnan(r[-5:]).all())
        self.assertTrue(np.isnan(v[-30:]).all())
    def test_gap_not_interpolated(self):
        mid=np.full(60,100.);mid[2]=np.nan
        r,v=forward_labels(mid)
        self.assertTrue(np.isnan(r[0]));self.assertTrue(np.isnan(v[0]));self.assertEqual(r[3],0)
    def test_volatility_30_returns(self):
        returns=np.array([.001,-.002,.003]*10);mid=np.r_[100,100*np.exp(np.cumsum(returns))]
        r,v=forward_labels(mid)
        self.assertAlmostEqual(v[0],np.std(returns,ddof=1)*1e4)
    def test_double_rank_not_pearson(self):
        a=np.tile([1,3,2,4],30);b=np.tile([10,40,20,30],30)
        self.assertAlmostEqual(ic(a,b),.8)
        self.assertTrue(np.isnan(ic(a,np.ones(120))))
    def test_BH_multiple_testing(self):
        np.testing.assert_allclose(bh([.01,.04,.03]),[.03,.04,.04])
    def test_shared_day_block(self):
        summary=interval([1,2,3,4,5,6,7]);self.assertEqual(summary['days'],7);self.assertEqual(summary['mean'],4)
        self.assertAlmostEqual(signflip_p([1]*7),2/128)
    def test_no_label_columns_in_model(self):
        for extra in GROUPS.values():self.assertFalse(set(CONTROLS+extra)&set(TARGETS))
    def test_zero_effect_bootstrap(self):
        self.assertEqual(interval([0]*7)['ci95'],[0,0])
    def test_half_open_hour_no_next_partition_label(self):
        import pandas as pd,datetime as dt
        start=int(dt.datetime(2026,9,19,1,tzinfo=dt.timezone.utc).timestamp()*1e9)
        frame=pd.DataFrame({'received_ns':np.arange(start,start+3601*1_000_000_000,1_000_000_000,dtype=np.int64),'mid':np.full(3601,100.)})
        frame.loc[3600,'mid']=100000 # next-partition value must never enter labels
        selected=prepare_hour_frame(frame,'2026-09-19','WIFUSDC')
        self.assertEqual(len(selected),3510)
        self.assertEqual(int(selected.received_ns.max()),start+3569*1_000_000_000)
        self.assertTrue((selected.future_rv_30s_bp==0).all())

if __name__=='__main__':unittest.main()
