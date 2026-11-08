"""Read hourly CryptoHFTData L2 files, preserving message groups and resets."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import polars as pl


def hours(config):
    start = datetime.fromisoformat(config.get('context_start',config['start'])).replace(tzinfo=timezone.utc)
    if start>datetime.fromisoformat(config['start']).replace(tzinfo=timezone.utc):
        raise ValueError('Context must begin at or before the trading window')
    end = datetime.fromisoformat(config['end_exclusive']).replace(tzinfo=timezone.utc)
    current = start
    while current < end:
        base = Path(config['source']) / current.strftime('%Y-%m-%d/%H')
        paths = {kind: base / f"{config['symbol']}_{kind}.parquet"
                 for kind in ('orderbook', 'trades', 'mark_price')}
        for path in paths.values():
            if not path.is_file():
                raise ValueError(f'Missing required source file: {path}')
        yield current, paths
        current += timedelta(hours=1)


def read(path, symbol):
    data = pl.read_parquet(path)
    if len(data) and (data['symbol'].null_count() or data['symbol'].unique().to_list() != [symbol]):
        raise ValueError(f'Unexpected symbol: {path}')
    if data['received_time'].null_count():
        raise ValueError(f'Missing receipt timestamp: {path}')
    for col in ('price', 'quantity'):
        if col in data.columns:
            data = data.with_columns(pl.col(col).cast(pl.Float64))
            if data[col].null_count() or not data[col].is_finite().all():
                raise ValueError(f'Invalid {col}: {path}')
    if 'price' in data.columns and 'trade_id' not in data.columns and (data['price'] <= 0).any():
        raise ValueError(f'Nonpositive price: {path}')
    if 'quantity' in data.columns and (data['quantity'] < 0).any():
        raise ValueError(f'Negative quantity: {path}')
    return data
