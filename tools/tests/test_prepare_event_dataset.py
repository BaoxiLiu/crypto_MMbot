"""Grid certification must preserve real depth, executions and parent identity."""
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pytest
from app.artifacts import digest
from tools.data.prepare_event_dataset import certify_grid

RULES=SimpleNamespace(tick=Decimal('.01'),step=Decimal('.01'))
DTYPE=[('ev','u8'),('exch_ts','i8'),('local_ts','i8'),('px','f8'),('qty','f8')]


def tape(tmp_path, rows):
    root=tmp_path/'parent';root.mkdir();parts=[]
    for i,events in enumerate(rows):
        path=root/f'part{i}.npz'
        data=np.array([((1<<31)|(1<<30)|kind,100+j,100+j,px,qty)
                       for j,(kind,px,qty) in enumerate(events)],dtype=DTYPE)
        np.savez_compressed(path,data=data)
        parts.append(dict(path=path.name,rows=len(data),sha256=digest(path)))
    manifest=dict(key='parent',segments=[dict(events=sum(p['rows'] for p in parts),files=parts)])
    (root/'manifest.json').write_text(json.dumps(manifest))
    return root,manifest


def test_orphan_deletion_derives_new_identity_without_mutating_parent(tmp_path):
    root,m=tape(tmp_path,[[(1,100.004,0),(1,100.01,2)],[(2,100.02,1)]])
    hashes={p.name:digest(p) for p in root.iterdir()}
    out,derived,proof=certify_grid(root,m,RULES,True)
    assert out!=root and proof['excluded_rows']==1
    assert derived['segments'][0]['events']==2
    assert {p.name:digest(p) for p in root.iterdir()}==hashes
    with np.load(out/'part0.npz') as z:assert z['data']['px'].tolist()==[100.01]
    assert digest(out/'part1.npz')==digest(root/'part1.npz')
    assert certify_grid(root,m,RULES,True)[0]==out


@pytest.mark.parametrize('bad', [[(1,100.004,1)],[(2,100.004,0)],[(1,100.01,.001)]])
def test_active_offgrid_depth_trades_and_quantities_are_rejected(tmp_path,bad):
    root,m=tape(tmp_path,[bad])
    with pytest.raises(ValueError,match='[Oo]ff-grid'):certify_grid(root,m,RULES,True)
    assert list(tmp_path.iterdir())==[root]


def test_rounded_target_in_different_part_blocks_exclusion(tmp_path):
    root,m=tape(tmp_path,[[(1,100.004,0)],[(4,100,1)]])
    with pytest.raises(ValueError,match='positive depth'):certify_grid(root,m,RULES,True)


def test_default_rejects_orphan_and_checks_hash_before_reading(tmp_path):
    root,m=tape(tmp_path,[[(1,100.004,0)]])
    with pytest.raises(ValueError,match='Off-grid'):certify_grid(root,m,RULES)
    (root/'part0.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='Changed event tape'):certify_grid(root,m,RULES,True)
