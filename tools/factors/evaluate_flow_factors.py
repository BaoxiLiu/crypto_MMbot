"""Exploratory, registered receipt-clock evaluation. No trading adoption claims."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
import datetime as dt, hashlib, itertools, json, math, sys, time
from collections import deque
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.factors import build_flow_factors as builder
from app.factor_library import FlowFactors,FACTOR_IDS

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/factor-library/evaluation-v1'
SYMBOLS=['BTCUSDC','SOLUSDC','SUIUSDC','UNIUSDC','DOGEUSDC','1000SHIBUSDC','WIFUSDC','ETHUSDC','1000PEPEUSDC','LTCUSDC','CRVUSDC','XRPUSDC','1000BONKUSDC']
DATES=[str(dt.date(2026,9,19)+dt.timedelta(days=i)) for i in range(16)]
CONTROLS=['spread_bp','l1_imbalance','micro_offset_bp','past_return_5s_bp','past_rv_30s_bp','taker_ratio_1s','trade_count_rate_10s','book_age_ms']
TARGETS=['return_5s_bp','future_rv_30s_bp']
GROUPS={'base':[],'near_depth':['S01','S02','S03','S04','S05','S16'],'trade_speed':['S06','S07','S08','S09','S10'],'volume_clock':['S11','S12','S13','S14','S15'],'near_depth_static':['S17','S18'],'replenishment':['S19','S20','S21'],'taker_pressure':['S22','S23','S24'],'bucket_dynamics':['S25','S26','S27'],'all27':list(FACTOR_IDS)}

class EvaluationFlow(FlowFactors):
    def reset(self):
        super().reset();self.mid_history=deque()
    def query(self,t):
        q=super().query(t)
        mid=spread=imb=micro=float('nan');age=float('nan')
        if self.epoch_start is not None:
            age=(t-self.book_time)/1e6
            if age<=self.params.book_stale_ms:
                b,a=self._best('bid'),self._best('ask');bp=b*self.params.tick_size;ap=a*self.params.tick_size
                B,A=self.book['bid'][b],self.book['ask'][a];mid=(bp+ap)/2
                spread=(ap-bp)/mid*1e4;imb=(B-A)/(B+A)
                micro=((bp*A+ap*B)/(B+A)/mid-1)*1e4
        self.mid_history.append((t,mid))
        while self.mid_history and self.mid_history[0][0]<t-30_000_000_000:self.mid_history.popleft()
        previous={ts:m for ts,m in self.mid_history};old=previous.get(t-5_000_000_000,float('nan'))
        pastret=(mid/old-1)*1e4 if math.isfinite(mid) and math.isfinite(old) else float('nan')
        mids=np.array([m for ts,m in self.mid_history])
        rv=float(np.std(np.diff(np.log(mids)),ddof=1)*1e4) if len(mids)==31 and np.all(np.isfinite(mids)) else float('nan')
        rate=sum(1 for r in self.trades if r[0]>t-10_000_000_000)/10 if self.epoch_start is not None and t-self.epoch_start>=10_000_000_000 else float('nan')
        den=q['S06']+q['S07'];ratio=(q['S06']-q['S07'])/den if math.isfinite(den) and den>0 else float('nan')
        return dict(q,mid=mid,spread_bp=spread,l1_imbalance=imb,micro_offset_bp=micro,past_return_5s_bp=pastret,past_rv_30s_bp=rv,taker_ratio_1s=ratio,trade_count_rate_10s=rate,book_age_ms=age)

def forward_labels(mid):
    mid=np.asarray(mid,float);n=len(mid);ret=np.full(n,np.nan);rv=np.full(n,np.nan)
    for horizon,key in [(5,'r'),(30,'v')]:
        if n<=horizon:continue
        windows=np.lib.stride_tricks.sliding_window_view(mid,horizon+1)
        ok=np.all(np.isfinite(windows)&(windows>0),axis=1)
        values=(windows[:,-1]/windows[:,0]-1)*1e4 if key=='r' else np.std(np.diff(np.log(windows),axis=1),axis=1,ddof=1)*1e4
        (ret if key=='r' else rv)[:len(values)]=np.where(ok,values,np.nan)
    return ret,rv

def prepare_hour_frame(frame,date,symbol):
    start=int(dt.datetime.fromisoformat(f'{date}T01:00:00+00:00').timestamp()*1e9)
    # The builder also emits the right endpoint. Evaluation hours are half-open:
    # do not use that endpoint as future label data from the next partition.
    frame=frame[(frame.received_ns>=start)&(frame.received_ns<start+3_600_000_000_000)].copy()
    expected=np.arange(start,start+3_600_000_000_000,1_000_000_000,dtype=np.int64)
    if not np.array_equal(frame.received_ns.to_numpy(),expected):raise ValueError('Incomplete decision grid')
    frame['date']=date;frame['symbol']=symbol
    frame['return_5s_bp'],frame['future_rv_30s_bp']=forward_labels(frame['mid'].to_numpy())
    return frame.iloc[60:-30].copy()

def ic(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float);good=np.isfinite(a)&np.isfinite(b)
    if good.sum()<100 or np.unique(a[good]).size<2 or np.unique(b[good]).size<2:return np.nan
    return float(spearmanr(a[good],b[good]).statistic)

def interval(values,seed=42):
    v=np.asarray(values,float);v=v[np.isfinite(v)]
    if not len(v):return {'mean':None,'ci95':[None,None],'days':0}
    rng=np.random.default_rng(seed);means=v[rng.integers(0,len(v),(5000,len(v)))].mean(axis=1)
    return {'mean':float(v.mean()),'ci95':[float(x) for x in np.quantile(means,[.025,.975])],'days':len(v)}

def signflip_p(values):
    v=np.asarray(values,float);v=v[np.isfinite(v)]
    if not len(v):return 1.0
    signs=np.array(list(itertools.product([-1,1],repeat=len(v))))
    return float(np.mean(np.abs((signs*v).mean(axis=1))>=abs(v.mean())-1e-15))

def bh(pvals):
    p=np.asarray(pvals,float);order=np.argsort(p);adjusted=np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]
    result=np.empty(len(p));result[order]=np.minimum(adjusted,1);return result

def build():
    OUT.mkdir(parents=True,exist_ok=True)
    parameters=json.loads((OUT/'parameters.json').read_text())
    assert dt.datetime.fromisoformat(parameters['fit_end_exclusive_utc']) < dt.datetime(2026,9,19,tzinfo=dt.timezone.utc)
    builder.FlowFactors=EvaluationFlow
    reports=[];frames=[]
    for date in DATES:
        for symbol in SYMBOLS:
            cache=OUT/'hour-features'/f'{symbol}_{date}_01_flow.parquet'
            reportpath=cache.with_suffix('.report.json')
            if cache.exists() and reportpath.exists():r=json.loads(reportpath.read_text())
            else:
                r=builder.run(ROOT/'data/cryptohftdata',date,'01',symbol,3600,cache.parent,parameters['quotas_usdc'][symbol])
                reportpath.write_text(json.dumps(r,indent=2))
            if r['quota_usdc']!=parameters['quotas_usdc'][symbol]:raise ValueError('Cached quota differs from registered parameters')
            reports.append(r)
            if r['error_count']:
                print('REJECT',symbol,date,r['error_count'],flush=True);continue
            frame=prepare_hour_frame(pq.read_table(cache).to_pandas(),date,symbol)
            frames.append(frame)
        print('BUILT',date,'accepted',sum(r['error_count']==0 for r in reports if r['date']==date),flush=True)
    data=pd.concat(frames,ignore_index=True)
    data['split']=np.where(data.date<='2026-09-24','train',np.where(data.date<='2026-09-27','valid','test'))
    data.to_parquet(OUT/'dataset.parquet',index=False)
    (OUT/'source-reports.json').write_text(json.dumps(reports,indent=2));return data

def evaluate(data):
    train=data[data.split=='train'];valid=data[data.split=='valid'];test=data[data.split=='test']
    single=[];daily_records=[]
    for target in TARGETS:
        for factor in FACTOR_IDS:
            records=[]
            for (symbol,date),g in test.groupby(['symbol','date'],sort=True):
                paired=np.isfinite(g[factor])&np.isfinite(g[target])
                record={'factor':factor,'target':target,'symbol':symbol,'date':date,'ic':ic(g[factor],g[target]),'paired_rows':int(paired.sum()),'label_rows':int(np.isfinite(g[target]).sum())}
                records.append(record);daily_records.append(record)
            df=pd.DataFrame(records);days=df.groupby('date').ic.mean().to_numpy()
            stats=interval(days)
            good=np.isfinite(test[factor])&np.isfinite(test[target]);base=np.isfinite(test[target])
            item={'factor':factor,'target':target,'train_ic':ic(train[factor],train[target]),'valid_ic':ic(valid[factor],valid[target]),'test_daily_equal_coin_ic':stats,'test_coverage':float(good.sum()/max(1,base.sum())),'day_signflip_p':signflip_p(days),'per_coin':{coin:interval(g.ic.to_numpy()) for coin,g in df.groupby('symbol')}}
            single.append(item)
    adjusted=bh([x['day_signflip_p'] for x in single])
    for row,q in zip(single,adjusted):row['bh_q_descriptive']=float(q)
    arms=[];prediction_frames=[]
    ids=pd.get_dummies(data.symbol,prefix='coin',dtype=float)
    for target in TARGETS:
        finite=np.isfinite(data[target]);d=data.loc[finite].copy();coins=ids.loc[finite].to_numpy()
        ti=d.split.to_numpy()=='train';vi=d.split.to_numpy()=='valid';te=d.split.to_numpy()=='test'
        if not all([ti.any(),vi.any(),te.any()]):raise ValueError('Empty split')
        y=d[target].to_numpy();base_sq=None;dtest=d.loc[te,['symbol','date','received_ns']].copy()
        for name,extra in GROUPS.items():
            cols=CONTROLS+extra;X=np.column_stack([d[cols].to_numpy(float),coins])
            model=HistGradientBoostingRegressor(max_iter=60,learning_rate=.05,max_leaf_nodes=15,max_depth=5,min_samples_leaf=200,l2_regularization=1,early_stopping=False,random_state=42)
            with threadpool_limits(limits=2):model.fit(X[ti],y[ti]);pred=model.predict(X[te]);pv=model.predict(X[vi])
            sq=(pred-y[te])**2
            if name=='base':base_sq=sq.copy()
            result=dtest.copy();result['mse']=sq;result['base_mse']=base_sq;result['prediction']=pred;result['target_value']=y[te]
            perday=result.groupby(['date','symbol'])[['mse','base_mse']].mean()
            # Equal coins within each shared calendar day, preserving the matched samples.
            day=perday.groupby('date').mean();diff=day.base_mse-day.mse
            overall=float(diff.mean()/day.base_mse.mean()*100)
            ratio_boot=[];rng=np.random.default_rng(42)
            for idx in rng.integers(0,len(day),(5000,len(day))):ratio_boot.append(float((day.base_mse.to_numpy()[idx]-day.mse.to_numpy()[idx]).mean()/day.base_mse.to_numpy()[idx].mean()*100))
            arm={'target':target,'arm':name,'features':cols,'test_rows':int(te.sum()),'train_rows':int(ti.sum()),'valid_mse':float(np.mean((pv-y[vi])**2)),'test_equal_coin_day_mse':float(day.mse.mean()),'mse_improvement_percent':overall,'mse_improvement_ci95':[float(x) for x in np.quantile(ratio_boot,[.025,.975])],'days':len(day),'per_coin':{coin:{'base_mse':float(g.base_mse.mean()),'mse':float(g.mse.mean()),'improvement_percent':float((g.base_mse.mean()-g.mse.mean())/g.base_mse.mean()*100)} for coin,g in perday.groupby('symbol')}}
            arms.append(arm);result['arm']=name;result['target_name']=target;prediction_frames.append(result)
            print('MODEL',target,name,'improvement',round(overall,3),'CI',arm['mse_improvement_ci95'],flush=True)
    pd.concat(prediction_frames).to_parquet(OUT/'test-predictions.parquet',index=False)
    pd.DataFrame(daily_records).to_parquet(OUT/'factor-daily-ic.parquet',index=False)
    sources=json.loads((OUT/'source-reports.json').read_text())
    report={'status':'exploratory_not_adoption','scope':'13 coins, one fixed UTC hour/day, past-only quotas, receipt-clock basic control; NOT original50 incremental model; S17-S27 not yet evaluated in this report','split':{'train':['2026-09-19','2026-09-24'],'valid':['2026-09-25','2026-09-27'],'test':['2026-09-28','2026-10-04']},'rows':{s:int((data.split==s).sum()) for s in ['train','valid','test']},'source_hours':len(sources),'rejected_hours':sum(x['error_count']>0 for x in sources),'single_factors':single,'model_arms':arms,'limitations':['One UTC hour/day only','Seven shared date blocks; descriptive inference and limited power','Historical dates already explored','Not conditional fill markout','No original50 same-clock ablation, latency degradation or economic trading validation'],'hashes':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['tools/factors/evaluate_flow_factors.py','app/factor_library/flow.py','app/factor_library/registry.json','doc/models.md']}}
    def clean(x):
        if isinstance(x,float) and not math.isfinite(x):return None
        if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
        if isinstance(x,list):return [clean(v) for v in x]
        return x
    (OUT/'results.json').write_text(json.dumps(clean(report),indent=2,allow_nan=False));print('EVALUATION COMPLETE',flush=True)
if __name__=='__main__':
    data=pq.read_table(OUT/'dataset.parquet').to_pandas() if (OUT/'dataset.parquet').exists() else build()
    evaluate(data)
