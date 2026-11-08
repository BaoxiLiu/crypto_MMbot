"""Bounded receipt-clock replay for the independently versioned S01-S16 library."""
import argparse,datetime as dt,hashlib,itertools,json,math,time,sys
from pathlib import Path
from decimal import Decimal
from functools import reduce
from math import gcd
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from app.factor_library import FlowFactors,FlowParams,FACTOR_IDS
import pyarrow as pa
import pyarrow.parquet as pq

def raw_rows(path):
    for batch in pq.ParquetFile(path).iter_batches(batch_size=16384):yield from batch.to_pylist()
def fingerprint(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def run(root,date,hour,symbol,seconds,out,quota=10000):
    start=int(dt.datetime.fromisoformat(f'{date}T{hour}:00:00+00:00').timestamp()*1e9)
    end=start+seconds*1000000000
    base=root/'raw/binance_futures'/date/hour
    ob=base/f'{symbol}_orderbook.parquet';tr=base/f'{symbol}_trades.parquet'
    events=[];snapshot=None;updates=0
    fields=lambda r:(r['event_type'],r['received_time'],r['event_time'],r['final_update_id'],r['last_update_id'])
    for key,iterator in itertools.groupby(raw_rows(ob),key=fields):
        group=list(iterator);t=key[1]
        if t>end:break
        levels=[(r['side'],r['price'],r['quantity']) for r in group]
        if key[0]=='snapshot':
            if snapshot is None:snapshot=group
            events.append((t,0,'snapshot',(levels,key[4] if key[4] is not None else key[3])))
        else:
            if any((r['first_update_id'],r['prev_final_update_id'])!=(group[0]['first_update_id'],group[0]['prev_final_update_id']) for r in group):raise ValueError('Mixed update group metadata')
            events.append((t,0,'update',(levels,group[0]['first_update_id'],key[3],group[0]['prev_final_update_id'])));updates+=1
    if not snapshot:raise ValueError('No initialization snapshot')
    # Observed grid, not a claim about exchange historical instrument filters.
    prices=sorted(set(Decimal(r['price']) for r in snapshot));scale=10**max(0,-min(p.as_tuple().exponent for p in prices))
    diffs=[int((b-a)*scale) for a,b in zip(prices,prices[1:])]
    tick=reduce(gcd,diffs)/scale
    for r in raw_rows(tr):
        if r['received_time']>end:break
        events.append((r['received_time'],1,'trade',(r['price'],r['quantity'],r['is_buyer_maker'],r['trade_id'],r.get('order_type')=='NA')))
    events.sort(key=lambda e:(e[0],e[1]))
    machine=FlowFactors(FlowParams(tick_size=tick,quota_usdc=quota));iterator=iter(events);pending=next(iterator,None);output=[];errors=[]
    begin=time.perf_counter()
    for t in range(start,end+1,1000000000):
        while pending and pending[0]<=t:
            received,_,kind,args=pending
            try:getattr(machine,kind)(received,*args)
            except ValueError as e:
                errors.append({'received_ns':received,'kind':kind,'error':str(e)})
                # Preserve unavailable epoch until the next true snapshot, never synthesize it.
                machine.reset()
            pending=next(iterator,None)
        q=machine.query(t)
        output.append({'received_ns':t,'epoch_available':machine.epoch_start is not None,**q})
    elapsed=time.perf_counter()-begin
    out.mkdir(parents=True,exist_ok=True)
    pq.write_table(pa.Table.from_pylist(output),out/f'{symbol}_{date}_{hour}_flow.parquet')
    finite={k:sum(math.isfinite(r[k]) for r in output) for k in FACTOR_IDS}
    return {'symbol':symbol,'date':date,'hour_utc':hour,'seconds':seconds,'rows':len(output),'observed_tick_size':tick,'quota_usdc':quota,'updates':updates,'events':len(events),'errors':errors[:20],'error_count':len(errors),'finite_rows':finite,'replay_seconds':elapsed,'placeholder_excluded':machine.invalid_trades,'source_sha256':{ob.name:fingerprint(ob),tr.name:fingerprint(tr)},'all_finite_values_bounded':all(math.isfinite(v) or math.isnan(v) for r in output for k,v in r.items() if k in FACTOR_IDS)}
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='data/cryptohftdata');p.add_argument('--date',default='2026-10-01');p.add_argument('--hour',default='01');p.add_argument('--seconds',type=int,default=300);p.add_argument('--symbols',nargs='+',default=['WIFUSDC','BTCUSDC','DOGEUSDC']);p.add_argument('--out',default='data/factor-library/flow-v1');p.add_argument('--params',help='Past-only frozen volume quota JSON');a=p.parse_args()
    if not 1<=a.seconds<=3600:raise SystemExit('seconds must be within one hour')
    parameters=json.loads(Path(a.params).read_text()) if a.params else None
    if parameters and dt.datetime.fromisoformat(parameters['fit_end_exclusive_utc']) > dt.datetime.fromisoformat(f'{a.date}T{a.hour}:00:00+00:00'): raise SystemExit('Future parameter calibration forbidden')
    reports=[]
    for symbol in a.symbols:
        r=run(Path(a.root),a.date,a.hour,symbol,a.seconds,Path(a.out),parameters['quotas_usdc'][symbol] if parameters else 10000);reports.append(r);print(symbol,'errors',r['error_count'],'events',r['events'],'seconds',round(r['replay_seconds'],3),flush=True)
    report={'version':'1.1.0','parameters_sha256':fingerprint(Path(a.params)) if a.params else None,'validation_scope':'Bounded real-data receipt-clock replay, not predictive or economic validation','reports':reports,'passed':all(r['error_count']==0 and r['all_finite_values_bounded'] for r in reports)}
    (Path(a.out)/'validation.json').write_text(json.dumps(report,indent=2));raise SystemExit(0 if report['passed'] else 1)
if __name__=='__main__':main()
