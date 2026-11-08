"""Checksum-verified Binance daily trades and exact overlap verification.

python -m app.backtesting.providers.binance_official --config CONFIG
"""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile
import polars as pl
from .cryptohftdata import hours, read


def ensure_archive(root, symbol, day):
    root=Path(root)/symbol
    root.mkdir(parents=True,exist_ok=True)
    name=f'{symbol}-trades-{day}.zip'
    path=root/name
    url=f'https://data.binance.vision/data/futures/um/daily/trades/{symbol}/{name}'
    checksum_path=root/(name+'.CHECKSUM')
    if not checksum_path.exists():
        checksum_path.write_bytes(urllib.request.urlopen(url+'.CHECKSUM',timeout=30).read())
    expected=checksum_path.read_text().split()[0]
    if not path.exists():
        tmp=path.with_suffix('.zip.tmp')
        tmp.write_bytes(urllib.request.urlopen(url,timeout=30).read())
        if hashlib.sha256(tmp.read_bytes()).hexdigest()!=expected:
            raise ValueError(f'Official checksum mismatch: {url}')
        tmp.replace(path)
    actual=hashlib.sha256(path.read_bytes()).hexdigest()
    if actual!=expected: raise ValueError(f'Corrupted official archive: {path}')
    return path,{'path':str(path.resolve()),'url':url,'sha256':actual}


class OfficialTrades:
    def __init__(self, config):
        self.root=config['official_trades_root']
        self.symbol=config['symbol']
        self.day=None
        self.data=None
        self.verification=[]
        self.sources=[]

    def load_day(self,day):
        if day==self.day: return
        path,meta=ensure_archive(self.root,self.symbol,day)
        with zipfile.ZipFile(path) as archive:
            names=[n for n in archive.namelist() if n.endswith('.csv')]
            if len(names)!=1: raise ValueError('Expected one official daily CSV')
            frame=pl.read_csv(io.BytesIO(archive.read(names[0])),schema_overrides={'price':pl.String,'qty':pl.String})
        if frame['id'].n_unique()!=len(frame) or not frame['id'].is_sorted():
            raise ValueError('Duplicate/regressing official trade IDs')
        if (frame['price'].cast(pl.Float64)<=0).any() or (frame['qty'].cast(pl.Float64)<=0).any():
            raise ValueError('Nonpositive official trade')
        self.day,self.data=day,frame
        self.sources.append(meta)

    def verify(self,raw,hour):
        data=self.reference_for_source(raw,hour)
        valid=raw.filter(pl.col('price').cast(pl.Float64)>0)
        ref=data.rename({'id':'trade_id','price':'official_price','qty':'official_qty',
                              'time':'official_time','is_buyer_maker':'official_maker'})
        joined=valid.join(ref,on='trade_id',how='inner')
        decimal=pl.Decimal(38,16)
        mismatch=joined.filter((pl.col('price').cast(decimal)!=pl.col('official_price').cast(decimal)) |
                              (pl.col('quantity').cast(decimal)!=pl.col('official_qty').cast(decimal)) |
                              (pl.col('trade_time')!=pl.col('official_time')) |
                              (pl.col('is_buyer_maker')!=pl.col('official_maker')))
        if len(joined)!=len(valid) or len(mismatch):
            raise ValueError(f'Provider/official overlap mismatch: {hour}')
        self.verification.append({'hour':hour.isoformat(),'matched_valid_trades':len(joined),
                                   'na_rows':raw.filter(pl.col('price').cast(pl.Float64)<=0).height,
                                   'na_present_in_official':raw.filter(pl.col('price').cast(pl.Float64)<=0).join(ref,on='trade_id',how='inner').height})

    def reference_for_source(self, raw, hour):
        # Supplier paths are assigned by receipt time. At UTC midnight a
        # file can contain executions from the previous exchange date.
        # Match their actual trade dates; retain exact field comparisons.
        times=raw['trade_time']
        if times.null_count() or (len(times) and times.min() <= 0):
            raise ValueError('Missing source trade timestamp')
        dates={hour.date()} if not len(times) else {
            datetime.fromtimestamp(ts/1000,timezone.utc).date() for ts in (times.min(),times.max())}
        allowed={hour.date()+timedelta(days=offset) for offset in (-1,0,1)}
        if not dates <= allowed: raise ValueError('Source trade date too far from receipt hour')
        frames=[]
        for day in sorted(dates):
            self.load_day(day.isoformat());frames.append(self.data)
        return pl.concat(frames)

    def verify_zero_placeholders(self, invalid, hour):
        if invalid.filter((pl.col('price').cast(pl.Float64) != 0) | (pl.col('quantity').cast(pl.Float64) != 0) |
                          ~pl.col('order_type').is_in(['NA','INSURANCE_FUND']).fill_null(False)).height:
            raise ValueError('Unexpected zero-placeholder values/type')
        reference=self.reference_for_source(invalid,hour).select(pl.col('id').alias('trade_id'),pl.lit(True).alias('official_present'))
        audit=invalid.select('trade_id','trade_time','order_type').join(reference,on='trade_id',how='left')
        if audit.filter((pl.col('order_type')=='INSURANCE_FUND') & pl.col('official_present').fill_null(False)).height:
            raise ValueError('Insurance zero placeholder unexpectedly appears in official executions')
        return audit.with_columns(pl.col('official_present').fill_null(False),
                                  pl.lit('zero_price_zero_quantity_official_checked').alias('exclusion_reason')).to_dicts()

    def for_hour(self,hour):
        self.load_day(hour.strftime('%Y-%m-%d'))
        start=int(hour.timestamp()*1000)
        data=self.data.filter(pl.col('time').is_between(start,start+3_600_000,closed='left'))
        return data.select(pl.col('id').alias('trade_id'),pl.col('price').cast(pl.Float64),
                           pl.col('qty').cast(pl.Float64).alias('quantity'),
                           pl.col('time').alias('trade_time'),pl.col('is_buyer_maker'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();config=json.loads(args.config.read_text());official=OfficialTrades(config)
    for hour,paths in hours(config):
        raw=pl.read_parquet(paths['trades'])
        official.verify(raw,hour)
    output=Path(config['official_trades_root'])/config['symbol']/'overlap_verification.json'
    output.write_text(json.dumps({'sources':official.sources,'hours':official.verification},indent=2))
    print('Matched',sum(x['matched_valid_trades'] for x in official.verification),'provider trades; no mismatches')


if __name__=='__main__':main()
