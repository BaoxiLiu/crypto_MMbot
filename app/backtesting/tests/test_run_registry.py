import json

import pytest

from app.backtesting import run_registry


def fresh_registry(tmp_path):
    path = tmp_path / 'registry.json'
    path.write_text(json.dumps(dict(schema_version=1, entries=[])))
    return path


def entry(run_id, status='prepared'):
    return dict(id=run_id, kind='run', config='config/experiments/default.json', datasets=None,
                code_ref=None, status=status, artifacts=None, conclusion=None, notes=None)


def test_append_and_roundtrip(tmp_path):
    path = fresh_registry(tmp_path)
    run_registry.append(entry('run-a'), path)
    reg = run_registry.load(path)
    assert [e['id'] for e in reg['entries']] == ['run-a']


def test_duplicate_identity_rejected(tmp_path):
    path = fresh_registry(tmp_path)
    run_registry.append(entry('run-a'), path)
    with pytest.raises(ValueError, match='conflict'):
        run_registry.append(entry('run-a'), path)


def test_unknown_status_rejected(tmp_path):
    path = fresh_registry(tmp_path)
    with pytest.raises(ValueError, match='status'):
        run_registry.append(entry('run-a', status='mystery'), path)
    with pytest.raises(ValueError, match='status'):
        run_registry.update_status('run-a', 'mystery', path)


def test_update_status_only_touches_run_entries(tmp_path):
    path = fresh_registry(tmp_path)
    run_registry.append(entry('run-a'), path)
    batch = entry('run-a-batch', status='frozen')
    batch['kind'] = 'batch'
    run_registry.append(batch, path)
    assert run_registry.update_status('run-a', 'completed', path)
    reg = run_registry.load(path)
    assert reg['entries'][0]['status'] == 'completed'
    assert reg['entries'][1]['status'] == 'frozen'


def test_update_missing_run_warns_without_failing(tmp_path, capsys):
    path = fresh_registry(tmp_path)
    assert not run_registry.update_status('ghost', 'completed', path)
    assert 'not registered' in capsys.readouterr().err


def test_first_run_creates_registry_without_history(tmp_path):
    path = tmp_path / 'new-results' / 'registry.json'
    assert run_registry.load(path) == dict(schema_version=1, entries=[])
    assert not path.exists()
    run_registry.append(entry('first-run'), path)
    assert run_registry.load(path)['entries'] == [entry('first-run')]


def test_corrupt_registry_is_not_silently_reinitialized(tmp_path):
    path = tmp_path / 'registry.json'
    path.write_text('{broken')
    with pytest.raises(json.JSONDecodeError):
        run_registry.append(entry('new-run'), path)
    assert path.read_text() == '{broken'
