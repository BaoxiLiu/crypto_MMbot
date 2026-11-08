"""Diagnose all vendor/official trade disagreements without accepting or editing them."""
import argparse
from datetime import datetime,timedelta,timezone
import hashlib
import json
from pathlib import Path
import sys

import polars as pl

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from app.backtesting.providers.binance_official import OfficialTrades
from app.backtesting.providers.cryptohftdata import hours


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(a.config.read_text());o=OfficialTrades(cfg)
    reports=[];mismatches=[];inputs={};decimal=pl.Decimal(38,16)
    for hour,paths in hours(cfg):
        p=paths['trades'];raw=pl.read_parquet(p);inputs[str(p.resolve())]=sha(p)
        official=o.reference_for_source(raw,hour)
        valid=raw.filter(pl.col('price').cast(pl.Float64)>0)
        ref=official.rename({'id':'trade_id','price':'official_price','qty':'official_qty',
                             'time':'official_time','is_buyer_maker':'official_maker'})
        joined=valid.join(ref,on='trade_id',how='left')
        missing=joined.filter(pl.col('official_price').is_null())
        conflict=joined.filter(pl.col('official_price').is_not_null()&(
            (pl.col('price').cast(decimal)!=pl.col('official_price').cast(decimal))|
            (pl.col('quantity').cast(decimal)!=pl.col('official_qty').cast(decimal))|
            (pl.col('trade_time')!=pl.col('official_time'))|
            (pl.col('is_buyer_maker')!=pl.col('official_maker'))))
        for kind,frame in [('not_in_official_archive',missing),('conflicting_fields',conflict)]:
            for row in frame.to_dicts():
                mismatches.append(dict(hour=hour.isoformat(),kind=kind,source=str(p),**row))
        reports.append(dict(hour=hour.isoformat(),provider_rows=len(raw),valid_provider_rows=len(valid),
                            archive_absent=len(missing),field_conflicts=len(conflict)))
        if hour.hour==23:
            print(hour.date(),'hours',len(reports),'unresolved',len(mismatches),flush=True)
    # Keep canonical archive identities separate; no inferred correction rule.
    sources={s['path']:s for s in o.sources}
    result=dict(status='diagnostic_only',symbol=cfg['symbol'],start=cfg['start'],end_exclusive=cfg['end_exclusive'],
        hours=len(reports),valid_provider_trades=sum(r['valid_provider_rows'] for r in reports),
        missing_in_official=sum(r['archive_absent'] for r in reports),
        field_conflicts=sum(r['field_conflicts'] for r in reports),
        mismatches=mismatches,hourly_counts=reports,input_sha256=inputs,
        official_archives=list(sources.values()),code_sha256=sha(__file__),
        interpretation='Disagreement is unresolved; checksum validates archive bytes, not completeness of historical executions. No source modified, no row accepted, no zero funding assumed.')
    (a.output/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str)+'\n')
    print('finished',len(reports),'hours, missing',result['missing_in_official'],'conflicts',result['field_conflicts'],flush=True)


if __name__=='__main__':main()
