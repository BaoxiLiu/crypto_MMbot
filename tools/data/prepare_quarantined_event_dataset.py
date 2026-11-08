"""Prepare verified subranges, explicitly quarantine source hours, preserve all raw data.

No trade is corrected or accepted on inference. The native cache is a byte-for-byte
composition of independently certified child caches, with a declared coverage gap.
"""
import argparse
from copy import deepcopy
from datetime import datetime,timedelta,timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def save(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def usable_ranges(start,end,excluded):
    begin=datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    finish=datetime.fromisoformat(end).replace(tzinfo=timezone.utc)
    blocked=sorted(datetime.fromisoformat(t).replace(tzinfo=timezone.utc) for t in excluded)
    if len(set(blocked))!=len(blocked):raise ValueError('Duplicate quarantined hour')
    ranges=[];cursor=begin
    for hour in blocked:
        if hour.minute or hour.second or hour.microsecond or not begin<=hour<finish:
            raise ValueError('Exclusions must be whole requested source hours')
        if cursor<hour:ranges.append((cursor,hour))
        cursor=hour+timedelta(hours=1)
    if cursor<finish:ranges.append((cursor,finish))
    if not ranges:raise ValueError('No usable source hours')
    return ranges


def dedup(rows):
    out={}
    for row in rows:
        key=row['path']
        if key in out and out[key]['sha256']!=row['sha256']:
            raise ValueError('Conflicting source identity')
        out[key]=row
    return list(out.values())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--input-spec',type=Path,required=True)
    a=p.parse_args()
    if a.input_spec.exists():raise FileExistsError('Preserve existing input specification')
    task=json.loads(a.config.read_text());original=json.loads(Path(task['source_preparation']).read_text())
    cfg=deepcopy(original)
    for key in ('source','cache','official_trades_root'):cfg[key]=str((ROOT/cfg[key]).resolve())
    audit_path=Path(task['audit']);audit=json.loads(audit_path.read_text())
    assert sha(audit_path)==task['audit_sha256']
    assert audit['symbol']==cfg['symbol'] and audit['start']==cfg['start'] and audit['end_exclusive']==cfg['end_exclusive']
    assert audit['field_conflicts']==0
    exclusion=task['exclude_source_hours']
    # Audit, not price/PnL, determines the exact excluded source-hour set.
    assert set(exclusion)=={r['hour'] for r in audit['mismatches']}
    assert audit['missing_in_official']==len(audit['mismatches'])
    for path,expected in audit['input_sha256'].items():assert sha(path)==expected,path
    ranges=usable_ranges(cfg['start'],cfg['end_exclusive'],exclusion)
    from app.backtesting.datasets import prepare
    from app.backtesting.providers.cryptohftdata import hours
    from app.backtesting.providers.mark_price import prepare_marks
    children=[]
    for begin,end in ranges:
        sub=deepcopy(cfg)
        sub.update(start=begin.replace(tzinfo=None).isoformat(),end_exclusive=end.replace(tzinfo=None).isoformat())
        folder,manifest=prepare(sub)
        children.append((folder,manifest,sub))
        print('Verified child',sub['start'],sub['end_exclusive'],len(manifest['segments']),'segments',flush=True)
    identity=dict(adapter='explicit_whole_source_hour_quarantine_v1',config=cfg,exclusions=exclusion,
                  audit_sha256=sha(audit_path),adapter_sha256=sha(__file__),
                  children=[dict(path=str(f),manifest_sha256=sha(f/'manifest.json'),config=c) for f,m,c in children])
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:20]
    destination=Path(cfg['cache'])/cfg['symbol']/('quarantine-'+key)
    destination.mkdir(parents=True,exist_ok=False)
    merged=deepcopy(children[0][1]);merged.update(key='quarantine-'+key,segments=[],resets=[],hours=[])
    list_fields=('sources','official_sources','official_overlap_sources','supplemental_publication_sources','official_overlap_verification',
                 'trade_id_gaps','bootstrap_depth_updates','excluded_interval_events','excluded_uninitialized_depth','excluded_na_trades')
    for name in list_fields:merged[name]=[]
    copied=[]
    for child,m,sub in children:
        assert m['version']==6 and not m['unresolved_trade_gaps']
        offset=len(merged['segments'])
        if merged['segments']:
            assert merged['segments'][-1]['end_ns']<m['segments'][0]['start_ns']
        for segment in m['segments']:
            target=deepcopy(segment);target['id']=len(merged['segments']);target['files']=[]
            for j,part in enumerate(segment['files']):
                source=child/part['path'];assert sha(source)==part['sha256']
                new=deepcopy(part);new['path']=f"segment_{target['id']:03d}_{j:04d}.npz"
                shutil.copyfile(source,destination/new['path'])
                assert sha(destination/new['path'])==part['sha256']
                target['files'].append(new)
                copied.append(dict(source=str(source),destination=new['path'],sha256=part['sha256']))
            merged['segments'].append(target)
        merged['hours'].extend(m['hours'])
        merged['resets'].extend(m.get('resets',[]))
        for name in list_fields:
            for row in m.get(name,[]):
                row=deepcopy(row)
                if 'segment_id' in row:row['segment_id']+=offset
                merged[name].append(row)
    # Account for every requested source hour, explicitly recording excluded
    # hours as unprocessed, not verified zero-activity observations.
    for hour in exclusion:
        merged['hours'].append(dict(hour=hour,book_rows=0,trade_rows=0,exchange_only_trade_rows=0,
            snapshot_groups=0,coverage_status='quarantined_not_replayed',
            reason='Provider positive trade absent from checksum-verified official archive; disagreement unresolved'))
    merged['hours'].sort(key=lambda r:r['hour'])
    expected={h.isoformat() for h,_ in hours(cfg)}
    assert len(merged['hours'])==len(expected) and {h['hour'] for h in merged['hours']}==expected
    for field in ('sources','official_sources','official_overlap_sources','supplemental_publication_sources'):
        merged[field]=dedup(merged[field])
    # Preserve quarantined original file identities as evidence of omission.
    excluded_inputs=[]
    for hour,paths in hours(cfg):
        if hour.isoformat() in exclusion:
            excluded_inputs.extend(dict(path=str(path.resolve()),sha256=sha(path),bytes=path.stat().st_size)
                                   for path in paths.values())
    merged['sources']=dedup(merged['sources']+excluded_inputs)
    merged['unresolved_trade_gaps']=[]
    merged['source_hour_quarantine']=dict(**identity,excluded_original_inputs=excluded_inputs,
        meaning='No correction or acceptance of disputed trade. Verified child caches only; original input files preserved. Requested-hour placeholders are explicitly quarantined, not replay coverage.',
        copied_file_count=len(copied),copied_file_sha256_verified=True)
    save(destination/'manifest.json',merged)
    save(destination/'quarantine-build.json',dict(identity=identity,copied_files=copied))
    prepare_marks(cfg,destination/'project_inputs')
    effective=deepcopy(cfg);effective['excluded_source_hours']=exclusion
    save(a.input_spec,dict(dataset=str(destination),config=effective,
        rows=sum(s['events'] for s in merged['segments']),segments=len(merged['segments']),
        exchange_only_trade_rows=sum(r.get('exchange_only_trade_rows',0) for r in merged['hours']),
        preparation_adapter=str(Path(__file__).resolve()),adapter_sha256=sha(__file__),
        quarantine_task=str(a.config.resolve()),quarantine_task_sha256=sha(a.config),
        coverage_warning='SUI comparison excludes unresolved trade-conflict source hour equally in both arms; not complete calendar coverage'))
    print(destination,flush=True)


if __name__=='__main__':main()
