"""Post-result sensitivity checks, not preregistered confirmatory tests."""
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parent))
from tools.factors.evaluate_flow_factors import OUT, CONTROLS, FACTOR_IDS, TARGETS, GROUPS

def model(seed):return HistGradientBoostingRegressor(max_iter=60,learning_rate=.05,max_leaf_nodes=15,max_depth=5,min_samples_leaf=200,l2_regularization=1,early_stopping=False,random_state=seed)
def gain(dtest,sq,base):
    f=dtest.copy();f['sq']=sq;f['base']=base
    per=f.groupby(['date','symbol'])[['sq','base']].mean();day=per.groupby('date').mean()
    return {'improvement_percent':float((day.base.mean()-day.sq.mean())/day.base.mean()*100),'daily_percent':{date:float((r.base-r.sq)/r.base*100) for date,r in day.iterrows()}}

data=pd.read_parquet(OUT/'dataset.parquet');primary=json.loads((OUT/'results.json').read_text());ids=pd.get_dummies(data.symbol,prefix='coin',dtype=float)
records=[]
for target in TARGETS:
    good=np.isfinite(data[target]);d=data.loc[good].copy();coins=ids.loc[good].to_numpy();y=d[target].to_numpy()
    ti=d.split.to_numpy()=='train';te=d.split.to_numpy()=='test';test=d.loc[te,['date','symbol','received_ns']].copy()
    for seed in [7,2026]:
        base=None
        for name in ['base','near_depth','all16']:
            X=np.column_stack([d[CONTROLS+GROUPS[name]].to_numpy(float),coins]);m=model(seed)
            with threadpool_limits(limits=2):m.fit(X[ti],y[ti]);pred=m.predict(X[te])
            sq=(pred-y[te])**2
            if name=='base':base=sq
            result=dict(target=target,seed=seed,arm=name,**gain(test,sq,base));records.append(result)
            print(target,seed,name,round(result['improvement_percent'],4),flush=True)
    # Separate a pure missingness mechanism from the numerical factor values.
    X=np.column_stack([d[CONTROLS].to_numpy(float),d[list(FACTOR_IDS)].isna().to_numpy(float),coins]);m=model(42)
    with threadpool_limits(limits=2):m.fit(X[ti],y[ti]);pred=m.predict(X[te])
    stored=pd.read_parquet(OUT/'test-predictions.parquet');stored=stored[(stored.arm=='base')&(stored.target_name==target)]
    linked=test.merge(stored[['received_ns','symbol','prediction','target_value']],on=['received_ns','symbol'],how='left',validate='one_to_one')
    assert linked.prediction.notna().all() and np.allclose(linked.target_value.to_numpy(),y[te],atol=1e-12,rtol=0)
    base=(linked.prediction.to_numpy()-y[te])**2
    result=dict(target=target,seed=42,arm='missing_only',**gain(test,(pred-y[te])**2,base));records.append(result)
    print(target,42,'missing_only',round(result['improvement_percent'],4),flush=True)
for row in primary['model_arms']:
 if row['arm'] in ['base','near_depth','all16']:records.append({'target':row['target'],'seed':42,'arm':row['arm'],'improvement_percent':row['mse_improvement_percent']})
(OUT/'sensitivity.json').write_text(json.dumps({'status':'post_hoc_descriptive_not_confirmatory','no_model_selected_from_probe':True,'rows':records},indent=2))
