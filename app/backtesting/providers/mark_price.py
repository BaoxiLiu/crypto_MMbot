"""Causal mark-price input for live strategy risk, without funding settlement."""
import argparse
from pathlib import Path
import json
import numpy as np
import polars as pl
from .cryptohftdata import hours
from ..datasets import atomic_json, digest


def prepare_marks(config, root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    frames=[];sources=[]
    for hour,paths in hours(config):
        path=paths['mark_price']
        data=pl.read_parquet(path)
        if len(data) and data['symbol'].unique().to_list()!=[config['symbol']]:
            raise ValueError(f'Unexpected mark-price symbol: {path}')
        frames.append(data.select('event_time',pl.col('mark_price').cast(pl.Float64).alias('price')))
        sources.append({'path':str(path.resolve()),'sha256':digest(path)})
    frame=pl.concat(frames).sort('event_time')
    if (frame['event_time'].null_count() or frame['price'].null_count() or
        not frame['price'].is_finite().all() or frame['price'].min()<=0):
        raise ValueError('Invalid mark-price data')
    if frame.group_by('event_time').agg(pl.col('price').n_unique().alias('prices'))['prices'].max()>1:
        raise ValueError('Conflicting mark prices at one exchange timestamp')
    frame=frame.unique(subset=['event_time'],keep='first',maintain_order=True)
    data=np.zeros(len(frame),dtype=[('local_ts','i8'),('price','f8')])
    data['local_ts']=frame['event_time'].to_numpy()*1_000_000+int(config['feed_latency_ms']*1e6)
    data['price']=frame['price'].to_numpy()
    # Mappable tape: replay does not load all dates' mark prices into RAM.
    target=root/'marks.npy';temp=root/'marks.tmp.npy'
    np.save(temp,data);temp.replace(target)
    atomic_json(root/'marks_manifest.json',{'sources':sources,'rows':len(data),'sha256':digest(target),
                                           'time_policy':'exchange event_time + configured feed latency'})
    return target


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(prepare_marks(json.loads(args.config.read_text()),args.output))


if __name__=='__main__':main()
