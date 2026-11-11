"""Prepare a publication-clock cache from an explicit, archived JSON config."""
import argparse
import json
from pathlib import Path
import sys
from copy import deepcopy
import hashlib
import os
import shutil

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def certify_grid(dataset, manifest, rules, exclude_orphan_deletions=False):
    """Check the whole tape; derive a new cache only for proven inert deletions.

    Never round prices, alter trades, or mutate a shared parent cache. A zero
    depth deletion is inert only if its rounded target has no positive depth
    anywhere in the complete context and trading window.
    """
    import numpy as np
    from app.backtesting.datasets import atomic_json, digest
    dataset=Path(dataset); changed={};positive=set();candidates=set();audit=[]
    for segment in manifest['segments']:
        for part in segment['files']:
            path=dataset/part['path']
            if digest(path)!=part['sha256']:raise ValueError('Changed event tape: '+str(path))
            with np.load(path,allow_pickle=False) as archive:data=archive['data']
            if len(data)!=part['rows']:raise ValueError('Event row count mismatch')
            price=data['px']/float(rules.tick);qty=data['qty']/float(rules.step)
            if not np.isfinite(price).all() or not np.isfinite(qty).all():
                raise ValueError('Nonfinite event price/quantity')
            if np.any(np.abs(qty-np.rint(qty))>1e-6):raise ValueError('Off-grid event quantity')
            offgrid=np.abs(price-np.rint(price))>1e-6
            kind=data['ev'] & 255
            if offgrid.any():
                if (not exclude_orphan_deletions or
                        np.any(offgrid & ((kind!=1)|(data['qty']!=0)))):
                    raise ValueError('Off-grid event price is not an eligible zero depth deletion')
                targets=set(np.rint(price[offgrid]).astype('int64').tolist())
                candidates.update(targets);changed[part['path']]=np.flatnonzero(offgrid)
                audit.append(dict(path=part['path'],rows=int(offgrid.sum()),
                                  prices=np.unique(data['px'][offgrid]).tolist()))
            if exclude_orphan_deletions:
                live=((kind==1)|(kind==4)) & (data['qty']>0)
                positive.update(np.unique(np.rint(price[live]).astype('int64')).tolist())
    if candidates & positive:
        raise ValueError('Off-grid deletion rounded target has positive depth in the full window')
    proof=dict(policy='exclude_only_offgrid_zero_depth_deletions_with_no_positive_rounded_target_in_full_window'
               if exclude_orphan_deletions else 'reject_offgrid',tick=str(rules.tick),step=str(rules.step),
               tool_sha256=digest(__file__),excluded_rows=sum(x['rows'] for x in audit),audit=audit)
    if not changed:return dataset,manifest,proof
    identity=dict(parent_manifest_sha256=digest(dataset/'manifest.json'),grid_certification=proof)
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:20]
    target=dataset.parent/key;target.mkdir(exist_ok=True)
    owner=target/'grid_derivation.json'
    if owner.exists():
        if json.loads(owner.read_text())!=identity:raise ValueError('Grid derivation identity conflict')
    elif any(target.iterdir()):raise FileExistsError('Unowned grid derivation directory')
    else:atomic_json(owner,identity)
    result=deepcopy(manifest);result.update(key=key,grid_certification=identity)
    for segment in result['segments']:
        removed=0
        for part in segment['files']:
            source=dataset/part['path'];destination=target/part['path']
            temp=destination.with_suffix('.npz.tmp');temp.unlink(missing_ok=True)
            if part['path'] in changed:
                with np.load(source,allow_pickle=False) as archive:data=archive['data']
                data=np.delete(data,changed[part['path']]);removed+=len(changed[part['path']])
                if not len(data):raise ValueError('Grid remediation would empty an event part')
                with temp.open('wb') as stream:np.savez_compressed(stream,data=data)
                clock=np.where((data['ev'] & (1<<31))!=0,data['exch_ts'],data['local_ts'])
                part.update(rows=len(data),sha256=digest(temp),start_ns=int(data['exch_ts'].min()),
                            end_ns=int(data['exch_ts'].max()),clock_start_ns=int(clock[0]),clock_end_ns=int(clock[-1]))
            else:
                try:os.link(source,temp)
                except OSError:shutil.copyfile(source,temp)
            temp.replace(destination)
        segment['events']-=removed
    atomic_json(target/'manifest.json',result)
    return target,result,proof


def main():
    from app.backtesting.datasets import prepare, atomic_json
    from app.backtesting.experiment_config import resolve
    from app.backtesting.providers.mark_price import prepare_marks
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--input-spec', required=True, type=Path)
    p.add_argument('--rules', type=Path, help='Validate all tape prices/quantities against this exchange snapshot')
    p.add_argument('--exclude-orphan-depth-deletions', action='store_true',
                   help='Explicitly permit only zero depth deletions with a globally unused rounded target')
    args = p.parse_args()
    if args.exclude_orphan_depth_deletions and args.rules is None:p.error('Deletion certification requires --rules')
    if args.input_spec.exists():
        raise FileExistsError('Never overwrite a frozen input specification')
    config = json.loads(args.config.read_text())
    # Resolve paths identically on either invocation directory.
    for key in ('source', 'cache', 'official_trades_root'):
        config[key] = str(resolve(config[key]).resolve())
    dataset, manifest = prepare(config)
    proof={}
    if args.rules:
        from app.market_making.core import Rules
        metadata=json.loads(args.rules.read_text())
        rules=Rules.from_exchange(next(s for s in metadata['symbols'] if s['symbol']==config['symbol']))
        dataset,manifest,proof=certify_grid(dataset,manifest,rules,args.exclude_orphan_depth_deletions)
    if not (dataset / 'project_inputs/marks.npy').exists():
        prepare_marks(config, dataset / 'project_inputs')
    args.input_spec.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.input_spec, dict(dataset=str(dataset), config=config,
        rows=sum(s['events'] for s in manifest['segments']), segments=len(manifest['segments']),
        grid_certification=proof,
        exchange_only_trade_rows=sum(r['exchange_only_trade_rows'] for r in manifest['hours'])))
    print(dataset, flush=True)


if __name__ == '__main__':
    main()
