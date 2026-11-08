"""Config path semantics independent of workers and attached market archives."""
from pathlib import Path
import pytest
from app.backtesting.experiment_config import ROOT, resolve


def test_paths_accept_strings_and_path_objects():
    assert resolve(Path('config/example.json')) == ROOT/'config/example.json'
    assert resolve(ROOT/'config/example.json') == ROOT/'config/example.json'


def test_dataset_root_is_expanded_for_both_path_types(tmp_path, monkeypatch):
    monkeypatch.setenv('QUANT_DATA_ROOT', str(tmp_path))
    value='$QUANT_DATA_ROOT/crypto/certified/example'
    assert resolve(value) == resolve(Path(value)) == tmp_path/'crypto/certified/example'


def test_missing_dataset_root_fails_explicitly(monkeypatch):
    monkeypatch.delenv('QUANT_DATA_ROOT', raising=False)
    with pytest.raises(ValueError, match='environment variable not set'):
        resolve('$QUANT_DATA_ROOT/crypto/certified/example')


def test_configuration_import_does_not_load_execution_or_report_runtime():
    import subprocess
    import sys
    code = """
import sys
from app.backtesting.experiment_config import load_spec
for name in ('hftbacktest', 'app.backtesting.fusion_experiment',
             'app.backtesting.experiment_report', 'app.backtesting.html_report'):
    assert not any(x == name or x.startswith(name + '.') for x in sys.modules), name
"""
    subprocess.run([sys.executable, '-B', '-s', '-c', code], cwd=ROOT, check=True)


def test_retired_dataset_strategy_reference_has_no_fallback():
    from app.backtesting.experiment_config import parameter_file
    with pytest.raises(ValueError,match='strategy_config is retired'):
        parameter_file({'datasets':{}},{'strategy_config':'old.json'},'BTCUSDC')
