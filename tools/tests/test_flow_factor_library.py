import math, random, unittest
from app.factor_library.flow import FlowFactors, FlowParams, SECOND, FACTOR_IDS

SNAP=[('bid',100,10),('bid',99,10),('ask',102,10),('ask',103,10)]
def make(**kwargs):
    x=FlowFactors(FlowParams(tick_size=1,quota_usdc=100,**kwargs));x.snapshot(0,SNAP,10);return x
class FlowTests(unittest.TestCase):
    def test_registry_count_and_freeze(self):
        self.assertEqual(len(FACTOR_IDS),27)
    def test_snapshot_not_flow_and_warmup(self):
        x=make();self.assertTrue(all(math.isnan(v) for v in x.query(0).values()))
        q=x.query(SECOND);self.assertEqual(q['S01'],0);self.assertEqual(q['S02'],0);self.assertEqual(q['S06'],0)
    def test_full_message_count_near_add_decrease(self):
        x=make();x.update(SECOND,[('bid',100,12),('ask',102,8),('bid',90,100)],11,12,10)
        q=x.query(SECOND);self.assertEqual(q['S01'],1);self.assertEqual(q['S02'],200);self.assertEqual(q['S05'],204)
        self.assertAlmostEqual(q['S16'],-204/(102*8+103*10))
    def test_atomic_cross_temporarily(self):
        x=make();x.update(SECOND,[('bid',103,1),('ask',102,0),('ask',103,0),('ask',104,3)],11,12,10)
        self.assertEqual(x.query(SECOND)['S01'],1)
    def test_gap_requires_new_snapshot(self):
        x=make()
        with self.assertRaises(ValueError):x.update(1,[('bid',100,9)],11,12,9)
        self.assertTrue(all(math.isnan(v) for v in x.query(2).values()))
        x.snapshot(3,SNAP,20);self.assertTrue(math.isnan(x.query(3)['S01']))
    def test_duplicates_ignored_not_counted(self):
        x=make();self.assertFalse(x.update(1,[('bid',100,9)],1,10,0));self.assertEqual(x.book['bid'][100],10)
    def test_timestamp_guard_no_future_query(self):
        x=make();x.trade(SECOND,100,1,False,1)
        with self.assertRaises(ValueError):x.query(SECOND-1)
        with self.assertRaises(ValueError):x.trade(SECOND-1,100,1,False,2)
    def test_zero_denominator_and_stale_book(self):
        x=make();q=x.query(5*SECOND);self.assertTrue(math.isnan(q['S08']));self.assertTrue(math.isnan(q['S01']))
        self.assertEqual(q['S06'],0)
        for k in ('S17','S18','S19','S20','S21'):self.assertTrue(math.isnan(q[k]))
        self.assertTrue(math.isnan(q['S23']));self.assertTrue(math.isnan(q['S24']))
        self.assertTrue(math.isnan(q['S22']))
    def test_volume_quota_all_and_direction(self):
        x=make();x.trade(SECOND,10,6,False,1);x.trade(2*SECOND,10,6,True,2)
        q=x.query(3*SECOND);self.assertEqual(q['S11'],2000);self.assertTrue(math.isnan(q['S12']));self.assertTrue(math.isnan(q['S13']))
        self.assertEqual(q['S15'],1000)
    def test_near_depth_static_imbalances(self):
        x=make();q=x.query(SECOND)
        # near bid depth = 100*10+99*10 = 1990; near ask = 102*10+103*10 = 2050
        self.assertAlmostEqual(q['S17'],(1990-2050)/4040)
        self.assertAlmostEqual(q['S18'],(1000-1020)/2020)
        self.assertFalse(math.isnan(q['S17']));self.assertFalse(math.isnan(q['S18']))
    def test_replenish_shares_and_net_flow_share(self):
        x=make();x.update(SECOND,[('bid',100,12),('ask',102,8),('bid',90,100)],11,12,10)
        q=x.query(SECOND)  # S02=200, S05=204, S03=S04=0
        self.assertEqual(q['S19'],1);self.assertEqual(q['S20'],0);self.assertEqual(q['S21'],1)
        q=x.query(2*SECOND)  # no near flow in the last second: zero denominators
        for k in ('S19','S20','S21'):self.assertTrue(math.isnan(q[k]))
        self.assertFalse(math.isnan(q['S17']))
    def test_taker_imbalance_1s(self):
        x=make()
        x.trade(SECOND,10,6,False,1);x.trade(2*SECOND,10,6,True,2)
        self.assertEqual(x.query(2*SECOND+500_000_000)['S22'],-1)  # only the sell is inside the 1s window
        self.assertTrue(math.isnan(make().query(2*SECOND)['S22']))
    def test_taker_vwap_basis_and_concentration_5s(self):
        x=make()
        x.trade(SECOND+500_000_000,100,1,False,1)  # buy 100 USDC, qty 1
        x.trade(2*SECOND,102,1,True,2)  # sell 102 USDC, qty 1
        x.update(5*SECOND,[('bid',100,10),('ask',102,10)],11,12,10)  # refresh book freshness
        q=x.query(6*SECOND)  # mid=(100+102)/2=101
        self.assertAlmostEqual(q['S23'],(100-102)/101*1e4)
        self.assertAlmostEqual(q['S24'],102/202)
        y=make();y.trade(SECOND+500_000_000,100,1,False,1)  # buy side only
        y.update(5*SECOND,[('bid',100,10),('ask',102,10)],11,12,10)
        self.assertTrue(math.isnan(y.query(6*SECOND)['S23']))
        self.assertTrue(math.isnan(make().query(6*SECOND)['S24']))
    def test_bucket_dynamics_signed_streak_cv_momentum(self):
        x=make(bucket_count=4)
        x.trade(SECOND,1,60,False,1);x.trade(2*SECOND,1,80,False,2)   # bucket 1: +1.0
        x.trade(3*SECOND,1,110,True,3)                                 # bucket 2: -0.2
        x.trade(4*SECOND,1,100,False,4)                                # bucket 3: 0.0
        q=x.query(4*SECOND+1)
        self.assertTrue(math.isnan(q['S27']));self.assertTrue(math.isnan(q['S14']))  # partial window
        x.trade(5*SECOND,1,60,True,5)                                  # bucket 4: 0.0
        q=x.query(6*SECOND)
        self.assertAlmostEqual(q['S25'],2)      # trailing same-sign (zero) streak
        self.assertAlmostEqual(q['S26'],0)      # all bucket durations exactly 1000 ms
        self.assertAlmostEqual(q['S27'],0-0.4)  # last-half mean 0 minus first-half mean 0.4
        self.assertAlmostEqual(q['S14'],0.3)    # signed storage must not leak into S14
    def test_bucket_splitting_same_time(self):
        x=make(bucket_count=2);x.trade(SECOND,100,2.5,False,1)
        q=x.query(SECOND);self.assertEqual(q['S14'],1);self.assertEqual(q['S15'],0)
        self.assertEqual(x.bucket_buy,50)
    def test_uncompleted_bucket_not_lookahead(self):
        x=make(bucket_count=2);x.trade(1,100,.5,False,1);before=x.query(1)
        self.assertTrue(math.isnan(before['S14']));self.assertTrue(math.isnan(before['S15']))
        x.trade(2,100,1.5,True,2)
        self.assertEqual(x.query(2)['S14'],.5)
        self.assertTrue(math.isnan(before['S14']))
    def test_placeholders_excluded_keep_ID(self):
        x=make();x.trade(1,0,0,True,1,True);x.trade(2,100,1,False,2)
        self.assertEqual(len(x.trades),1);self.assertEqual(x.invalid_trades,1)
    def test_trade_gap_resets(self):
        x=make();x.trade(1,100,1,False,1)
        with self.assertRaises(ValueError):x.trade(2,100,1,False,3)
        self.assertIsNone(x.epoch_start)
    def test_interval_cv_and_acceleration(self):
        x=make();x.trade(2*SECOND,10,1,False,1);x.trade(3*SECOND,10,1,False,2);x.trade(5*SECOND,10,1,True,3)
        q=x.query(5*SECOND);self.assertAlmostEqual(q['S10'],1/3);self.assertAlmostEqual(q['S08'],20/6)
    def test_memory_bound_and_trim(self):
        x=make(max_events=2);x.trade(1,100,.1,False,1);x.trade(2,100,.1,False,2)
        with self.assertRaises(ValueError):x.trade(3,100,.1,False,3)
        self.assertIsNone(x.epoch_start)
        y=make();y.trade(1,100,1,False,1);q=y.query(61*SECOND)
        self.assertEqual(len(y.trades),0);self.assertTrue(math.isnan(q['S11']));self.assertTrue(math.isnan(q['S15']))
    def test_invalid_inputs_no_mutation(self):
        x=make()
        with self.assertRaises(ValueError):x.update(1,[('bid',100,-1)],11,12,10)
        self.assertEqual(x.last_u,10)
        with self.assertRaises(ValueError):x.trade(1,float('inf'),1,False,1)
        with self.assertRaises(ValueError):x.update(1,[('bid',100,9),('bid',100,8)],11,12,10)
    def test_delete_and_price_changes_fixed_keys(self):
        x=make();x.update(SECOND,[('bid',100,0),('bid',99,12)],11,12,10)
        q=x.query(SECOND);self.assertEqual(q['S04'],1000);self.assertEqual(q['S02'],198)
    def test_randomized_independent_window_sums(self):
        rng=random.Random(42);x=make();history=[];qty={'bid':10,'ask':10};tid=0;u=10
        for i in range(1,201):
            t=i*100000000;side=rng.choice(['bid','ask']);p=100 if side=='bid' else 102
            new=rng.randint(1,20);old=qty[side];qty[side]=new;u+=1
            x.update(t,[(side,p,new)],u,u,u-1);tid+=1
            buy=bool(rng.getrandbits(1));value=rng.randint(1,30)
            x.trade(t,1,value,not buy,tid);history.append((t,side,p,new-old,value,buy))
            q=x.query(t)
            if t<SECOND:continue
            rows=[r for r in history if t-SECOND<r[0]<=t]
            self.assertEqual(q['S01'],len(rows))
            for key,s,sign in [('S02','bid',1),('S03','ask',1),('S04','bid',-1),('S05','ask',-1)]:
                self.assertAlmostEqual(q[key],sum(r[2]*max(sign*r[3],0) for r in rows if r[1]==s))
            self.assertAlmostEqual(q['S06'],sum(r[4] for r in rows if r[5]))
            self.assertAlmostEqual(q['S07'],sum(r[4] for r in rows if not r[5]))
            if t>=5*SECOND:
                slow=sum(r[4] for r in history if r[0]>t-5*SECOND)/5
                fast=sum(r[4] for r in history if r[0]>t-SECOND//2)/.5
                self.assertAlmostEqual(q['S08'],fast/slow)
            for key,flag in [('S11',None),('S12',True),('S13',False)]:
                total=0;expected=float('nan')
                for r in reversed(history):
                    if flag is None or r[5]==flag:
                        total+=r[4]
                        if total>=100:expected=(t-r[0])/1e6;break
                if math.isnan(expected):self.assertTrue(math.isnan(q[key]))
                else:self.assertAlmostEqual(q[key],expected)
            s02=sum(r[2]*max(r[3],0) for r in rows if r[1]=='bid');s04=sum(r[2]*max(-r[3],0) for r in rows if r[1]=='bid')
            s03=sum(r[2]*max(r[3],0) for r in rows if r[1]=='ask');s05=sum(r[2]*max(-r[3],0) for r in rows if r[1]=='ask')
            s06=sum(r[4] for r in rows if r[5]);s07=sum(r[4] for r in rows if not r[5])
            db,da=100*qty['bid']+990,102*qty['ask']+1030
            self.assertAlmostEqual(q['S17'],(db-da)/(db+da))
            self.assertAlmostEqual(q['S18'],(100*qty['bid']-102*qty['ask'])/(100*qty['bid']+102*qty['ask']))
            for key,num,den in [('S19',s02,s02+s04),('S20',s03,s03+s05),
                    ('S21',(s02-s04)-(s03-s05),s02+s03+s04+s05),('S22',s06-s07,s06+s07)]:
                expected=num/den if den>0 else float('nan')
                if math.isnan(expected):self.assertTrue(math.isnan(q[key]))
                else:self.assertAlmostEqual(q[key],expected)
    def test_independent_equal_volume_bucket_reference(self):
        x=make(bucket_count=3);rng=random.Random(17);history=[];cumulative=0.0
        for i in range(1,41):
            value=rng.randint(1,180);buy=bool(rng.getrandbits(1));t=i*100000000
            history.append((t,cumulative,cumulative+value,buy));cumulative+=value
            x.trade(t,1,value,not buy,i)
            completed=int(cumulative//100);reference=[];previous_end=history[0][0]
            for bucket in range(completed):
                low,high=bucket*100,(bucket+1)*100
                buys=sum(max(0,min(end,high)-max(start,low)) for ts,start,end,b in history if b)
                end_time=next(ts for ts,start,end,b in history if end>=high)
                reference.append((abs(2*buys-100)/100,(end_time-previous_end)/1e6))
                previous_end=end_time
            q=x.query(t)
            if reference:self.assertAlmostEqual(q['S15'],reference[-1][1])
            if len(reference)>=3:self.assertAlmostEqual(q['S14'],sum(r[0] for r in reference[-3:])/3)
            else:self.assertTrue(math.isnan(q['S14']))
    def test_snapshot_reset_is_causal_and_restarts_warmup(self):
        x=make();x.trade(SECOND,100,1,False,1);x.query(2*SECOND)
        with self.assertRaises(ValueError):x.snapshot(SECOND,SNAP,30)
        x.snapshot(3*SECOND,SNAP,30)
        q=x.query(3*SECOND);self.assertTrue(math.isnan(q['S06']));self.assertTrue(math.isnan(q['S11']))
        self.assertEqual(len(x.closed),0)

if __name__=='__main__':unittest.main()
