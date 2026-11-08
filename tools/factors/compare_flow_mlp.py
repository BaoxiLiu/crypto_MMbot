"""A small two-hidden-layer MLP, exact same tasks/splits and paired test rows."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
import copy,hashlib,json,sys,time,warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parent))
from tools.factors.evaluate_flow_factors import OUT, CONTROLS, FACTOR_IDS, TARGETS
DEST=OUT/'mlp-comparison';DEST.mkdir(exist_ok=True)

def paired(test,sq,mae,base_sq):
    f=test.copy();f['sq']=sq;f['base']=base_sq;f['mae']=mae
    per=f.groupby(['date','symbol'])[['sq','base','mae']].mean();day=per.groupby('date').mean()
    b=day.base.to_numpy();m=day.sq.to_numpy();rng=np.random.default_rng(42)
    idx=rng.integers(0,len(b),(5000,len(b)));gain=(b[idx]-m[idx]).mean(axis=1)/b[idx].mean(axis=1)*100
    return {'mse_gain_vs_original_tree_percent':float((b.mean()-m.mean())/b.mean()*100),'gain_ci95':[float(v) for v in np.quantile(gain,[.025,.975])],'equal_coin_day_mse':float(m.mean()),'equal_coin_day_mae':float(day.mae.mean()),'per_coin':{coin:{'mse':float(g.sq.mean()),'mse_gain_percent':float((g.base.mean()-g.sq.mean())/g.base.mean()*100)} for coin,g in per.groupby('symbol')}}

def main():
    data=pd.read_parquet(OUT/'dataset.parquet');stored=pd.read_parquet(OUT/'test-predictions.parquet');cols=CONTROLS+list(FACTOR_IDS)
    coins=pd.get_dummies(data.symbol,prefix='coin',dtype=float)
    results=[]
    for target in TARGETS:
        good=np.isfinite(data[target]);d=data.loc[good];onehot=coins.loc[good].to_numpy(float)
        ti=d.split.to_numpy()=='train';vi=d.split.to_numpy()=='valid';te=d.split.to_numpy()=='test'
        x=d[cols].to_numpy(float);missing=~np.isfinite(x)
        med=np.nanmedian(x[ti],axis=0);lo=np.nanquantile(x[ti],.001,axis=0);hi=np.nanquantile(x[ti],.999,axis=0)
        if not np.all(np.isfinite(med)):raise ValueError('No training support for feature')
        imputed=np.where(missing,med,x);clipped=np.clip(imputed,lo,hi)
        scaler=StandardScaler().fit(clipped[ti]);X=np.column_stack([scaler.transform(clipped),missing.astype(float),onehot]).astype(np.float32)
        y=d[target].to_numpy(float);mean=float(y[ti].mean());scale=float(y[ti].std())
        if scale<=0:raise ValueError('Constant target')
        ys=(y-mean)/scale
        predrows=d.loc[te,['symbol','date','received_ns']].copy()
        native=stored[(stored.arm=='all16')&(stored.target_name==target)]
        linked=predrows.merge(native[['symbol','received_ns','prediction','target_value']],on=['symbol','received_ns'],how='left',validate='one_to_one')
        assert linked.prediction.notna().all() and np.allclose(linked.target_value.to_numpy(),y[te],rtol=0,atol=1e-12)
        original=linked.prediction.to_numpy();base_sq=(original-y[te])**2
        params={'columns':cols,'median':med.tolist(),'winsor_lower':lo.tolist(),'winsor_upper':hi.tolist(),'scaler_mean':scaler.mean_.tolist(),'scaler_scale':scaler.scale_.tolist(),'target_mean':mean,'target_scale':scale}
        (DEST/f'{target}-preprocessing.json').write_text(json.dumps(params,indent=2))
        results.append(dict(target=target,model='original_native_tree',seed=42,**paired(predrows,base_sq,np.abs(original-y[te]),base_sq)))
        for seed in [7,42,2026]:
            t=time.perf_counter()
            tree=HistGradientBoostingRegressor(max_iter=60,learning_rate=.05,max_leaf_nodes=15,max_depth=5,min_samples_leaf=200,l2_regularization=1,early_stopping=False,random_state=seed)
            with threadpool_limits(limits=2):tree.fit(X[ti],y[ti]);tp=tree.predict(X[te]);tv=tree.predict(X[vi])
            row=dict(target=target,model='matched_preprocessing_tree',seed=seed,valid_mse=float(np.mean((tv-y[vi])**2)),training_seconds=time.perf_counter()-t,**paired(predrows,(tp-y[te])**2,np.abs(tp-y[te]),base_sq));results.append(row)
            print(target,row['model'],seed,round(row['mse_gain_vs_original_tree_percent'],3),flush=True)
            net=MLPRegressor(hidden_layer_sizes=(64,32),activation='relu',solver='adam',batch_size=2048,learning_rate_init=.001,alpha=.0001,shuffle=True,early_stopping=False,random_state=seed)
            best=np.inf;best_weights=None;best_epoch=None;patience=0;history=[];begin=time.perf_counter()
            for epoch in range(1,31):
                with threadpool_limits(limits=2):
                    net.partial_fit(X[ti],ys[ti]);vp=net.predict(X[vi])*scale+mean
                loss=float(np.mean((vp-y[vi])**2));history.append(loss)
                if not np.isfinite(loss):raise ValueError('Nonfinite MLP validation loss')
                if loss<best:
                    best=loss;best_epoch=epoch;best_weights=([a.copy() for a in net.coefs_],[a.copy() for a in net.intercepts_]);patience=0
                else:patience+=1
                if patience>=5:break
            net.coefs_,net.intercepts_=best_weights
            with threadpool_limits(limits=2):prediction=net.predict(X[te])*scale+mean
            row=dict(target=target,model='mlp_64_32',seed=seed,valid_mse=best,best_epoch=best_epoch,epochs_run=len(history),validation_losses=history,training_seconds=time.perf_counter()-begin,**paired(predrows,(prediction-y[te])**2,np.abs(prediction-y[te]),base_sq));results.append(row)
            weights={f'weight{i}':v for i,v in enumerate(net.coefs_)};weights.update({f'bias{i}':v for i,v in enumerate(net.intercepts_)})
            np.savez_compressed(DEST/f'{target}-seed{seed}-weights.npz',**weights)
            saved=predrows.copy();saved['target_value']=y[te];saved['prediction']=prediction;saved.to_parquet(DEST/f'{target}-seed{seed}-predictions.parquet',index=False)
            print(target,row['model'],seed,'epoch',best_epoch,'gain',round(row['mse_gain_vs_original_tree_percent'],3),flush=True)
        (DEST/'progress.json').write_text(json.dumps(results,indent=2))
    report={'status':'quick_exploratory_comparison','inputs':'8 controls + 16 flow factors + coin indicators','targets':TARGETS,'paired_test_rows':True,'results':results,'hashes':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ['tools/factors/compare_flow_mlp.py','doc/models.md']},'dataset_sha256':hashlib.sha256((OUT/'dataset.parquet').read_bytes()).hexdigest()}
    (DEST/'results.json').write_text(json.dumps(report,indent=2));print('COMPLETE',flush=True)
if __name__=='__main__':main()
