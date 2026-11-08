"""Classify off-grid events against both causal native/public book streams."""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from numba import njit

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from app.factor_library.microstructure_core import FeatureState
from app.artifacts import digest as sha, write


@njit
def inspect(data,exchange,local,tick,step):
    flagged=[]
    for i in range(len(data)):
        r=data[i];ev=int(r['ev']);kind=ev&255
        state=local if ev&(1<<30) else exchange
        side=1 if ev&(1<<29) else -1
        units=r['px']/tick;index=int(round(units))
        badpx=abs(units-round(units))>1e-6
        badqty=abs(r['qty']/step-round(r['qty']/step))>1e-6
        if badpx or badqty:
            lower=int(math.floor(units));upper=int(math.ceil(units))
            lq=state.bid_qty(lower) if side==1 else state.ask_qty(lower)
            uq=state.bid_qty(upper) if side==1 else state.ask_qty(upper)
            safe=badpx and not badqty and kind==1 and r['qty']==0. and lq==0. and uq==0.
            flagged.append((i,lq,uq,1. if safe else 0.))
            if safe:continue
        if kind==1 or kind==4:state.put(side,index,r['qty'])
        elif kind!=2:raise ValueError('Unexpected event type')
    return flagged


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-spec',type=Path,required=True)
    p.add_argument('--rules',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    prep=json.loads(a.input_spec.read_text());root=Path(prep['dataset']);m=json.loads((root/'manifest.json').read_text())
    rule=next(r for r in json.loads(a.rules.read_text())['symbols'] if r['symbol']==m['symbol'])
    tick=float(next(f['tickSize'] for f in rule['filters'] if f['filterType']=='PRICE_FILTER'))
    step=float(next(f['stepSize'] for f in rule['filters'] if f['filterType']=='LOT_SIZE'))
    records=[];checked=0
    for segment in m['segments']:
        exchange=FeatureState(tick,limit=100000,capacity=2)
        local=FeatureState(tick,limit=100000,capacity=2)
        for part in segment['files']:
            path=root/part['path'];assert sha(path)==part['sha256']
            with np.load(path,allow_pickle=False) as z:data=z['data']
            for i,lq,uq,safe in inspect(data,exchange,local,tick,step):
                i=int(i);r=data[i]
                records.append(dict(file=part['path'],row=i,ev=int(r['ev']),exch_ns=int(r['exch_ts']),
                    local_ns=int(r['local_ts']),price=float(r['px']),qty=float(r['qty']),
                    previous_floor_qty=lq,previous_ceil_qty=uq,proven_noop=bool(safe)))
            checked+=len(data)
        print('segment',segment['id'],'checked',checked,'off_grid',len(records),flush=True)
    evidence=dict(status='passed_noop_classification' if all(r['proven_noop'] for r in records) else 'unresolved_active_or_nonzero_exception',
        dataset=str(root),manifest_sha256=sha(root/'manifest.json'),rules_sha256=sha(a.rules),code_sha256=sha(__file__),
        book_implementation_sha256=sha(ROOT/'app/factor_library/microstructure_core.py'),
        rows_checked=checked,off_grid_events=len(records),safe_noops=sum(r['proven_noop'] for r in records),
        exceptions=records,meaning='Each safe exception is a zero-quantity depth update at an off-grid price; both adjacent valid prices had zero standing quantity in that causal stream. No data transformed by this audit.')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    write(a.output,evidence,default=None)
    print(evidence['status'],len(records),flush=True)


if __name__=='__main__':main()
