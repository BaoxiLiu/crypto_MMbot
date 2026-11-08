"""Runtime schema integrity and dense, fully warmed reference-computation checks."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from app.factor_library.paper173 import FEATURE_NAMES, Observations, compute_named
from app.factor_library.paper173_schema import SCHEMA_PATH, load_names
from tools.factors.build_factor_catalog import paper_names

NS=1_000_000_000


class Paper173LibraryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.now=6000*NS
        t=np.arange(0,cls.now+1,20_000_000,dtype=np.int64)
        p=100+0.005*np.sin(np.arange(len(t))*0.17)
        q=2+0.2*np.cos(np.arange(len(t))*0.11)
        buy=np.arange(len(t))%2==0
        bt=np.arange(0,cls.now+1,NS,dtype=np.int64)
        bid=100+0.001*np.sin(np.arange(len(bt))*0.13)
        cls.obs=Observations(t,p,q,buy,bt,bid,bid+0.01,0)
        cls.bid=np.array([bid[-1],bid[-1]-0.01])
        cls.ask=np.array([bid[-1]+0.01,bid[-1]+0.02])
        cls.qty=np.array([10000.,10000.])

    def query(self,obs=None,side=1):
        return compute_named(self.obs if obs is None else obs,self.now,side,
                             self.bid,self.qty,self.ask,self.qty)

    def test_schema_preserves_published_appendix_order_not_sorted_names(self):
        self.assertEqual(FEATURE_NAMES,tuple(paper_names()))
        self.assertEqual(len(set(FEATURE_NAMES)),173)
        self.assertNotEqual(FEATURE_NAMES,tuple(sorted(FEATURE_NAMES)))

    def test_fresh_runtime_import_does_not_open_report_files(self):
        script='''
from pathlib import Path
original=Path.open
def guarded(self,*args,**kwargs):
    if '/doc/reports/' in str(self):
        raise RuntimeError('Runtime attempted to read a report')
    return original(self,*args,**kwargs)
Path.open=guarded
from app.factor_library.paper173 import FEATURE_NAMES, compute_named
assert len(FEATURE_NAMES)==173
assert callable(compute_named)
print('paper173-runtime-ok')
'''
        result=subprocess.run([sys.executable,'-c',script],cwd=SCHEMA_PATH.parents[2],
                              capture_output=True,text=True,check=True)
        self.assertIn('paper173-runtime-ok',result.stdout)

    def test_reordered_or_duplicate_schema_is_rejected(self):
        data=json.loads(SCHEMA_PATH.read_text())
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'schema.json'
            names=data['feature_names']
            names[0],names[1]=names[1],names[0]
            p.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'fingerprint mismatch'):load_names(p)
            names[1]=names[0];p.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'duplicates'):load_names(p)

    def test_full_history_and_deep_book_produce_all_173_values(self):
        values=self.query()
        self.assertEqual(tuple(values),FEATURE_NAMES)
        self.assertTrue(np.isfinite(list(values.values())).all())
        lo=np.searchsorted(self.obs.trade_ns,self.now-NS,side='right')
        select=self.obs.trade_buy[lo:]
        self.assertEqual(values['buy_count_1s_w0'],int(select.sum()))
        self.assertAlmostEqual(values['total_buy_1s_w0'],float(self.obs.trade_qty[lo:][select].sum()))
        self.assertAlmostEqual(values['trade_intensity_1s_w0'],0.02)

    def test_all_173_outputs_are_unchanged_by_future_trade_and_bbo(self):
        a=self.obs
        future=Observations(np.r_[a.trade_ns,self.now+1],np.r_[a.trade_price,999.],
                            np.r_[a.trade_qty,999.],np.r_[a.trade_buy,True],
                            np.r_[a.book_ns,self.now+1],np.r_[a.book_bid,999.],
                            np.r_[a.book_ask,1000.],0)
        np.testing.assert_allclose(list(self.query().values()),list(self.query(future).values()),
                                   rtol=0,atol=0,equal_nan=True)

    def test_insufficient_book_preserves_missing_impact_instead_of_fake_zero(self):
        values=compute_named(self.obs,self.now,1,self.bid,[1.,1.],self.ask,[1.,1.])
        self.assertTrue(np.isnan(values['ob_half']))
        self.assertTrue(np.isnan(values['ob_other_half']))
        self.assertTrue(np.isfinite(values['total_buy_1s_w0']))

    def test_order_side_swaps_quantity_fields_and_return_sign(self):
        buy,sell=self.query(side=1),self.query(side=-1)
        for scale in ('100ms','1s','5s','30s','300s'):
            for lag in range(3):
                suffix=f'_{scale}_w{lag}'
                self.assertEqual(buy['buy_count'+suffix],sell['sell_count'+suffix])
                self.assertAlmostEqual(buy['total_buy'+suffix],sell['total_sell'+suffix])
                self.assertAlmostEqual(buy['ret_sum'+suffix],-sell['ret_sum'+suffix])


if __name__=='__main__':unittest.main()
