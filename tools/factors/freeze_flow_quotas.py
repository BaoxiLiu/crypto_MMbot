"""Fit volume quotas from explicitly earlier data; never use validation events."""
import argparse,datetime as dt,json,math,statistics,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.factors.build_flow_factors import raw_rows,fingerprint
p=argparse.ArgumentParser();p.add_argument('--date',default='2026-09-30');p.add_argument('--hour',default='00');p.add_argument('--out',default='data/factor-library/flow-v1/parameters.json');p.add_argument('--symbols',nargs='+',required=True);a=p.parse_args()
start=int(dt.datetime.fromisoformat(f'{a.date}T{a.hour}:00:00+00:00').timestamp()*1e9)
quotas={};sources={}
for symbol in a.symbols:
 path=Path(f'$QUANT_DATA_ROOT/crypto/raw/hft/binance_futures/{a.date}/{a.hour}/{symbol}_trades.parquet')
 minutes=[0.0]*60
 for r in raw_rows(path):
  j=(r['received_time']-start)//60000000000
  if 0<=j<60 and r.get('order_type')!='NA' and float(r['price'])>0 and float(r['quantity'])>0:
   minutes[j]+=float(r['price'])*float(r['quantity'])
 median=statistics.median(minutes)
 if median<=0:raise ValueError('Insufficient historical activity to freeze quota: '+symbol)
 quotas[symbol]=max(1.0,median/5)
 sources[symbol]={'sha256':fingerprint(path),'median_minute_notional':median}
result={'version':'1.0.0','fit_start_utc':dt.datetime.fromtimestamp(start/1e9,dt.timezone.utc).isoformat(),'fit_end_exclusive_utc':dt.datetime.fromtimestamp(start/1e9+3600,dt.timezone.utc).isoformat(),'formula':'max(1 USDC, median of 60 complete-minute notional totals / 5)','quotas_usdc':quotas,'sources':sources,'status':'engineering calibration only; model training must refit solely inside training folds'}
out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2));print('已冻结',len(quotas),'币门槛')
