"""Hashing/report serialization must not import an execution stack."""
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
from app.artifacts import ROOT, digest, write


def test_digest_and_atomic_json_keep_existing_contract(tmp_path):
    target=tmp_path/'result.json'
    write(target,dict(price=Decimal('1.20'),label='成交'))
    assert json.loads(target.read_text())=={'price':'1.20','label':'成交'}
    assert target.read_text().endswith('\n')
    assert digest(target)==hashlib.sha256(target.read_bytes()).hexdigest()
    before=target.read_bytes()
    with pytest.raises(ValueError):
        write(target,dict(value=float('nan')))
    assert target.read_bytes()==before


def test_artifact_import_has_no_strategy_or_execution_side_effects():
    subprocess.run([sys.executable,'-s','-c',"""
import sys
from app.artifacts import ROOT, digest, write
assert 'hftbacktest' not in sys.modules
assert not any(k.startswith(('app.market_making','app.backtesting')) for k in sys.modules)
"""],cwd=ROOT,check=True)


def test_strict_json_rejects_unsupported_values_without_overwriting(tmp_path):
    target = tmp_path / 'strict.json'
    write(target, {'value': 1}, default=None)
    before = target.read_bytes()
    with pytest.raises(TypeError):
        write(target, {'value': Decimal('1.20')}, default=None)
    assert target.read_bytes() == before


def test_source_identity_excludes_sidecars_and_honors_directory_scope(tmp_path):
    from app.artifacts import source_identity
    folder = tmp_path / 'selected'
    folder.mkdir()
    code = folder / 'code.py'
    code.write_text('value = 1')
    (folder / '._code.py').write_bytes(b'invalid sidecar')
    (folder / 'notes.txt').write_text('not code')
    (tmp_path / 'unselected.py').write_text('not selected')
    assert source_identity(tmp_path, folders=('selected',)) == {
        'selected/code.py': digest(code)}
    code.write_text('value = 2')
    assert source_identity(tmp_path, folders=('selected',))['selected/code.py'] == digest(code)


def test_default_runtime_identity_covers_canonical_metrics_and_skips_tests(tmp_path):
    from app.artifacts import source_identity
    for name in ('app/policy.py', 'src/metrics/core.py', 'app/tests/test_code.py'):
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('value=1')
    identity=source_identity(tmp_path)
    assert set(identity)=={'app/policy.py','src/metrics/core.py'}
    before=identity['src/metrics/core.py']
    (tmp_path/'src/metrics/core.py').write_text('value=2')
    assert source_identity(tmp_path)['src/metrics/core.py']!=before
