"""Versioned, hashed conversion caches; fail closed and split genuine book resets."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import numpy as np
import polars as pl
import hftbacktest as h
from .providers.cryptohftdata import hours, read

VERSION = 5


def digest(path):
    sha = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    tmp.replace(path)


def events(df, snapshot=False, trades=False, publication_ns=None):
    out = np.zeros(len(df), dtype=h.event_dtype)
    if trades:
        side = np.where(df['is_buyer_maker'].to_numpy(), h.SELL_EVENT, h.BUY_EVENT)
        ts = df['trade_time'].to_numpy()
        typ = h.TRADE_EVENT
    else:
        side = np.where(df['side'].to_numpy() == 'bid', h.BUY_EVENT, h.SELL_EVENT)
        ts = df['transaction_time'].fill_null(df['event_time']).to_numpy()
        typ = h.DEPTH_SNAPSHOT_EVENT if snapshot else h.DEPTH_EVENT
    if np.any(ts == 0):
        raise ValueError('Missing exchange timestamp')
    out['ev'] = side.astype(np.uint64) | np.uint64(typ | h.EXCH_EVENT | h.LOCAL_EVENT)
    out['exch_ts'] = ts.astype(np.int64) * 1_000_000
    out['px'], out['qty'] = df['price'].to_numpy(), df['quantity'].to_numpy()
    if publication_ns is not None:
        # Missing publication times of official-only prints must not become
        # invented public observations. They still participate in exchange fills.
        known = df['event_time'].is_not_null().to_numpy()
        pub = df['event_time'].fill_null(df['trade_time'] if trades else df['transaction_time']).to_numpy()
        out['local_ts'] = np.maximum(pub.astype(np.int64)*1_000_000, out['exch_ts']) + publication_ns
        out['ev'][~known] &= np.uint64(~h.LOCAL_EVENT & ((1 << 64)-1))
        out['ival'] = df['trade_id' if trades else 'final_update_id'].fill_null(0).to_numpy()
    return out


def partition_trades(trades, config, official, hour):
    """Exclude only explicitly recognized non-execution source placeholders."""
    invalid=trades.filter(pl.col('price') <= 0)
    if not len(invalid): return trades, []
    policy=config.get('invalid_trade_policy')
    if policy=='exclude_marked_na':
        if invalid.filter((pl.col('quantity') != 0) | (pl.col('order_type') != 'NA')).height:
            raise ValueError('Invalid trades outside the NA placeholder policy')
        audit=invalid.select('trade_id','trade_time','order_type').to_dicts()
    elif policy=='exclude_verified_zero_placeholders':
        if official is None or invalid.filter((pl.col('price') != 0) | (pl.col('quantity') != 0) |
                                              ~pl.col('order_type').is_in(['NA','INSURANCE_FUND']).fill_null(False)).height:
            raise ValueError('Zero-placeholder policy requires official trades and exact recognized zero rows')
        audit=official.verify_zero_placeholders(invalid,hour)
    else:
        raise ValueError('Invalid trades without an explicit placeholder policy')
    return trades.filter(pl.col('price') > 0), audit


def prepare(config):
    if config['provider'] != 'cryptohftdata':
        raise ValueError('Unsupported provider')
    source_hours = list(hours(config))
    official = None
    if config.get('trade_source') == 'binance_official':
        from .providers.binance_official import OfficialTrades, ensure_archive
        official = OfficialTrades(config)
        official_sources = [ensure_archive(config['official_trades_root'], config['symbol'], day)[1]
                            for day in sorted({hour.strftime('%Y-%m-%d') for hour,_ in source_hours})]
    else:
        official_sources = []
    sources = []
    for _, paths in source_hours:
        for path in paths.values():
            sources.append({'path': str(path.resolve()), 'sha256': digest(path), 'bytes': path.stat().st_size})
    source=Path(__file__).parent
    converters=('datasets.py','providers/cryptohftdata.py','providers/binance_official.py')
    publication = config.get('feed_clock') == 'publication'
    publication_ns = int(config['feed_latency_ms']*1_000_000) if publication else None
    identity = {'version': 6 if publication else VERSION,
                'conversion_code_sha256':{name:digest(source/name) for name in converters},
                'conversion_versions':{name:importlib.metadata.version(name) for name in ('numpy','polars','hftbacktest')},
                'symbol': config['symbol'], 'feed_latency_ms': config['feed_latency_ms'],
                'sources': sources, 'official_sources':official_sources, 'trade_source':config.get('trade_source','provider'), 'gap_policy': config['gap_policy'], 'invalid_trade_policy': config.get('invalid_trade_policy','reject')}
    bootstrap_policy = config.get('bootstrap_policy', 'reject_uninitialized')
    if bootstrap_policy not in ('reject_uninitialized', 'wait_first_snapshot'):
        raise ValueError('Unknown bootstrap policy')
    if 'bootstrap_policy' in config:identity['bootstrap_policy'] = bootstrap_policy
    if 'context_start' in config:
        identity['context_start']=config['context_start']
        identity['prewindow_context']=config['context_start']<config['start']
    if publication: identity['feed_clock']='publication_E_plus_transport_exchange_T'
    if publication:
        from datetime import timedelta
        last_hour,last_paths=source_hours[-1]
        extra=last_paths['trades'].parent.parent.parent/(last_hour+timedelta(hours=1)).strftime('%Y-%m-%d/%H')/last_paths['trades'].name
        identity['supplemental_publication_sources']=([{'path':str(extra.resolve()),'sha256':digest(extra)}] if extra.exists() else [])
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    cache_base = config['cache']
    if cache_base.startswith('$QUANT_DATA_ROOT/'):
        env_root = os.environ.get('QUANT_DATA_ROOT')
        if not env_root:
            raise ValueError('QUANT_DATA_ROOT environment variable not set for: ' + cache_base)
        cache_base = str(Path(env_root) / cache_base[len('$QUANT_DATA_ROOT/'):])
    root = Path(cache_base) / config['symbol'] / key
    manifest_path = root / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('key') != key or any(manifest.get(name) != value for name, value in identity.items()):
            raise ValueError('Prepared manifest identity mismatch')
        for segment in manifest['segments']:
            for part in segment['files']:
                if digest(root / part['path']) != part['sha256']:
                    raise ValueError('Converted cache checksum mismatch')
        for part in manifest.get('official_overlap_sources', []):
            if digest(part['path']) != part['sha256']:
                raise ValueError('Official overlap reference checksum mismatch')
        if manifest.get('unresolved_trade_gaps',manifest['trade_id_gaps']) and config.get('trade_gap_policy','reject') != 'report_exploratory':
            raise ValueError('Dataset has reported trade ID gaps; strict run refused')
        return root, manifest
    root.mkdir(parents=True, exist_ok=True)
    segments, resets, reports = [], [], []
    segment = None
    pending = []
    tail = np.zeros(0, dtype=h.event_dtype)
    previous_id = None
    bridged = False
    last_book_ts = None
    previous_trade = None
    previous_trade_time = None
    trade_gaps = []
    excluded_na = []
    bootstrap_updates = []
    excluded_interval_events = []
    excluded_uninitialized_depth = []

    def flush(cutoff=None):
        nonlocal pending, tail
        arrays = [tail] + pending
        pending = []
        data = np.concatenate(arrays)
        if not len(data):
            return
        valid = data['exch_ts'] >= segment['start_ns']
        if 'certified_end_ns' in segment:
            valid &= data['exch_ts'] <= segment['certified_end_ns']
        if not valid.all():
            excluded_interval_events.append({'segment_id':segment['id'],'rows':int((~valid).sum()),
                                             'trade_rows':int(np.sum((data['ev'][~valid]&255)==h.TRADE_EVENT)),
                                             'reason':'outside_certified_book_interval'})
            data=data[valid]
        if not len(data):
            tail=np.zeros(0,dtype=h.event_dtype)
            return
        if publication:
            # Independent exchange/local streams. Split only new dual-clock
            # records; the retained tail may already contain single-clock rows.
            both=(data['ev'] & (h.EXCH_EVENT | h.LOCAL_EVENT)) == (h.EXCH_EVENT | h.LOCAL_EVENT)
            local=data[both].copy()
            local['ev'] &= np.uint64(~h.EXCH_EVENT & ((1 << 64)-1))
            data['ev'][both] &= np.uint64(~h.LOCAL_EVENT & ((1 << 64)-1))
            data=np.concatenate((data,local))
            clock=np.where((data['ev'] & h.EXCH_EVENT)!=0,data['exch_ts'],data['local_ts'])
            order=np.argsort(clock,kind='stable');data=data[order];clock=clock[order]
        else:
            data = data[np.argsort(data['exch_ts'], kind='stable')]
            data['local_ts'] = data['exch_ts'] + int(config['feed_latency_ms'] * 1_000_000)
            clock=data['exch_ts']
        if cutoff is not None:
            mask = clock < cutoff
            tail, data = data[~mask].copy(), data[mask]
            clock=clock[mask]
        else:
            tail = np.zeros(0, dtype=h.event_dtype)
        if not len(data):
            return
        if publication:
            from hftbacktest.data.validation import validate_event_order
            validate_event_order(data)
        if segment['files'] and int(clock[0]) < segment['files'][-1].get('clock_end_ns',segment['end_ns']):
            raise ValueError('Cross-part exchange timestamp inversion exceeds 2-second buffer')
        name = f"segment_{segment['id']:03d}_{len(segment['files']):04d}.npz"
        tmp = root / (name + '.tmp')
        with tmp.open('wb') as stream:
            np.savez_compressed(stream, data=data)
        tmp.replace(root / name)
        segment['files'].append({'path': name, 'rows': len(data), 'sha256': digest(root / name),
                                 'start_ns':int(data['exch_ts'].min()),'end_ns':int(data['exch_ts'].max()),
                                 **({'clock_start_ns':int(clock[0]),'clock_end_ns':int(clock[-1])} if publication else {})})
        segment['end_ns'] = int(data['exch_ts'].max())
        segment['events'] += len(data)

    def seal(end_ns):
        nonlocal pending, tail
        # A reconnect snapshot can reveal that the preceding hour already
        # contained an unobserved book interval. Clip its published tail too.
        segment['certified_end_ns']=int(end_ns)
        # Keep the old tail owned by the old segment, merging it with this
        # hour's late messages before sorting. Never attach it to the reset.
        hour_events.extend((segment,array) for array in pending)
        pending=[]
        if len(tail): hour_events.append((segment,tail))
        tail=np.zeros(0,dtype=h.event_dtype)
        for part in reversed(segment['files'].copy()):
            if part['end_ns'] <= end_ns: break
            path=root/part['path']
            with np.load(path) as archive: data=archive['data']
            keep=data['exch_ts'] <= end_ns
            excluded_interval_events.append({'segment_id':segment['id'],'rows':int((~keep).sum()),
                                             'trade_rows':int(np.sum((data['ev'][~keep]&255)==h.TRADE_EVENT)),
                                             'reason':'gap_revealed_by_later_snapshot'})
            segment['events']-=int((~keep).sum())
            data=data[keep]
            if not len(data):
                segment['files'].remove(part);path.unlink()
            else:
                tmp=root/(part['path']+'.tmp')
                with tmp.open('wb') as stream: np.savez_compressed(stream,data=data)
                tmp.replace(path)
                part.update(rows=len(data),sha256=digest(path),end_ns=int(data['exch_ts'].max()))
                if publication:
                    actual_clock=np.where((data['ev'] & h.EXCH_EVENT)!=0,data['exch_ts'],data['local_ts'])
                    part.update(clock_start_ns=int(actual_clock[0]),clock_end_ns=int(actual_clock[-1]))
        segment['end_ns']=segment['files'][-1]['end_ns'] if segment['files'] else segment['start_ns']

    for hour, paths in source_hours:
        book = read(paths['orderbook'], config['symbol'])
        if not set(book['event_type'].unique().to_list()) <= {'snapshot', 'update'}:
            raise ValueError('Unknown book event type')
        if not set(book['side'].unique().to_list()) <= {'bid', 'ask'}:
            raise ValueError('Unknown book side')
        trades = read(paths['trades'], config['symbol'])
        ids = trades['trade_id'].to_numpy()
        if len(ids):
            ts = trades['trade_time'].to_numpy()
            before = np.r_[previous_trade if previous_trade is not None else ids[0]-1, ids[:-1]]
            for i in np.flatnonzero(ids-before != 1):
                if ids[i] <= before[i]:
                    raise ValueError('Duplicate or regressing trade IDs')
                trade_gaps.append({'path':str(paths['trades']), 'previous_id':int(before[i]),
                                   'next_id':int(ids[i]), 'missing_count':int(ids[i]-before[i]-1),
                                   'previous_ns':int(previous_trade_time if i==0 else ts[i-1])*1_000_000,
                                   'next_ns':int(ts[i])*1_000_000})
            previous_trade = int(ids[-1])
            previous_trade_time = int(ts[-1])
        trades, excluded = partition_trades(trades,config,official,hour)
        excluded_na.extend(excluded)
        if official is not None:
            raw = pl.read_parquet(paths['trades'])
            official.verify(raw,hour)
            trades = official.for_hour(hour)
            if publication:
                # Receipt-hour files can contain preceding exchange-hour prints.
                # Include the following source hour before matching by trade ID.
                from datetime import timedelta
                next_path=paths['trades'].parent.parent.parent/(hour+timedelta(hours=1)).strftime('%Y-%m-%d/%H')/paths['trades'].name
                receipt_frames=[raw.select('trade_id','event_time')]
                if next_path.exists():
                    receipt_frames.append(pl.read_parquet(next_path).select('trade_id','event_time'))
                pubs=pl.concat(receipt_frames).unique(subset=['trade_id'],keep='last')
                trades=trades.join(pubs,on='trade_id',how='left',maintain_order='left')
        # Rows for each exchange message are consecutive in provider files.
        types = book['event_type'].to_numpy()
        recv = book['received_time'].to_numpy()
        finals = book['final_update_id'].to_numpy()
        boundaries = np.r_[0, np.flatnonzero((types[1:] != types[:-1]) | (recv[1:] != recv[:-1]) | (finals[1:] != finals[:-1])) + 1, len(book)]
        book_events = events(book, publication_ns=publication_ns)
        snap_mask = types == 'snapshot'
        book_events['ev'][snap_mask] = (book_events['ev'][snap_mask] & np.uint64(h.BUY_EVENT | h.SELL_EVENT | h.EXCH_EVENT | h.LOCAL_EVENT)) | np.uint64(h.DEPTH_SNAPSHOT_EVENT)
        hour_events = []
        split_times = []
        for a, b in zip(boundaries[:-1], boundaries[1:]):
            row = book.row(int(a), named=True)
            if row['event_type'] == 'snapshot':
                sid = int(row['last_update_id'])
                if segment is None or (previous_id is not None and sid > previous_id):
                    # Provider reset: terminate the old continuous experiment before the snapshot.
                    snap = book_events[a:b]
                    start_ns = int(snap['exch_ts'][0])
                    if np.any(snap['exch_ts'] != start_ns):
                        raise ValueError('Initial snapshot has inconsistent exchange timestamps')
                    if segment is not None:
                        split_times.append((start_ns, segment))
                        resets.append({'last_valid_ns': last_book_ts, 'recovery_ns': start_ns,
                                       'previous_id': previous_id, 'snapshot_id': sid})
                        seal(last_book_ts)
                    segment = {'id': len(segments), 'files': [], 'start_ns': start_ns,
                               'end_ns': start_ns, 'events': 0}
                    segments.append(segment)
                    hour_events.append((segment, snap))
                    previous_id = sid
                    bridged = False
                    last_book_ts = start_ns
                # Regular hourly and duplicate reconnect snapshots are checkpoints, not new depth events.
            else:
                if previous_id is None:
                    if bootstrap_policy != 'wait_first_snapshot':
                        raise ValueError('Updates before initialization')
                    hour_key = hour.isoformat()
                    if not excluded_uninitialized_depth or excluded_uninitialized_depth[-1]['hour'] != hour_key:
                        excluded_uninitialized_depth.append(dict(hour=hour_key, groups=0, rows=0,
                            first_exchange_ns=int(book_events[a]['exch_ts']),
                            last_exchange_ns=int(book_events[a]['exch_ts']),
                            reason='No initial snapshot; prefix excluded, never reconstructed from future snapshot'))
                    skipped = excluded_uninitialized_depth[-1]
                    skipped['groups'] += 1
                    skipped['rows'] += int(b-a)
                    skipped['last_exchange_ns'] = int(book_events[a]['exch_ts'])
                    continue
                if not bridged and int(row['final_update_id']) <= previous_id:
                    continue
                if not bridged:
                    valid = (int(row['prev_final_update_id']) == previous_id or int(row['first_update_id']) <= previous_id <= int(row['final_update_id']))
                else:
                    valid = int(row['prev_final_update_id']) == previous_id
                if not valid:
                    raise ValueError(f'Unrecoverable depth gap at {paths["orderbook"]}: {row}')
                ev = book_events[a:b]
                early=ev['exch_ts'] < segment['start_ns']
                if early.any():
                    bootstrap_updates.append({'segment_id':segment['id'],'rows':int(early.sum()),
                                              'original_min_ns':int(ev['exch_ts'][early].min()),
                                              'available_ns':segment['start_ns'],
                                              'first_update_id':int(row['first_update_id']),
                                              'final_update_id':int(row['final_update_id']),
                                              'reason':'ID-verified bridge buffered until initial snapshot'})
                    ev=ev.copy();ev['exch_ts'][early]=segment['start_ns']
                    if publication:ev['local_ts']=np.maximum(ev['local_ts'],ev['exch_ts']+publication_ns)
                hour_events.append((segment, ev))
                previous_id = int(row['final_update_id'])
                bridged = True
                last_book_ts = int(ev['exch_ts'][0])
        # Group arrays by segment. Trades in unobserved intervals are excluded conservatively.
        te = events(trades, trades=True, publication_ns=publication_ns)
        active = sorted({s['id']: s for s, _ in hour_events}.values(), key=lambda s: s['id'])
        for index, target in enumerate(active):
            if pending or len(tail):
                old = segments[target['id']-1] if index > 0 else target
                if index > 0:
                    segment = old
                    flush()
            segment = target
            arrays = [e for s, e in hour_events if s is target]
            start = target['start_ns']
            end = resets[target['id']]['last_valid_ns'] if target['id'] < len(resets) else None
            mask = te['exch_ts'] >= start
            if end is not None:
                mask &= te['exch_ts'] <= end
            pending.extend(arrays)
            pending.append(te[mask])
            if index < len(active)-1:
                flush()
            else:
                flush(int(hour.timestamp()*1e9) + 3_598_000_000_000)
        reports.append({'hour': hour.isoformat(), 'book_rows': len(book), 'trade_rows': len(trades),
                        **({'exchange_only_trade_rows':trades['event_time'].null_count()} if publication else {}),
                        'snapshot_groups': sum(types[a] == 'snapshot' for a in boundaries[:-1])})
        if len(reports) % 24 == 0:
            print('Converted', hour.strftime('%Y-%m-%d'), 'segments', len(segments), flush=True)
    flush()
    if not segments:raise ValueError('No initial snapshot anywhere in requested range')
    if trade_gaps and official is None and config.get('trade_gap_policy','reject') != 'report_exploratory':
        raise ValueError(f'Trade ID gaps found: {trade_gaps[:10]}')
    manifest = {**identity, 'key': key, 'segments': segments, 'resets': resets,
                'excluded_uninitialized_depth':excluded_uninitialized_depth,
                'bootstrap_depth_updates':bootstrap_updates,
                'excluded_interval_events':excluded_interval_events,
                'execution_interval_policy':'initial snapshot plus ID-verified buffered bridge; no executions in certified L2 gaps',
                'hours': reports, 'trade_id_gaps': trade_gaps, 'unresolved_trade_gaps': trade_gaps if official is None else [],
                'official_overlap_verification': official.verification if official is not None else [],
                'official_overlap_sources': list({row['path']:row for row in official.sources}.values()) if official is not None else [],
                'excluded_na_trades': excluded_na,
                'time_policy': ('exchange T; local publication E + transport; official-only prints exchange-visible only; independent clock streams' if publication else 'exchange transaction/trade time + configured constant feed latency; supplier receipt time retained in original files')}
    atomic_json(manifest_path, manifest)
    return root, manifest
