"""Training consumes the same resolved experiment configuration as backtesting."""
import importlib
import json
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize('kind', ['alpha', 'value'])
def test_training_uses_common_loader_and_expands_data_root(tmp_path, monkeypatch, kind):
    module = importlib.import_module('tools.models.prepare_nh_' + kind)
    rules = tmp_path / 'rules.json'
    rules.write_text(json.dumps({'symbols': [{'symbol': 'BTCUSDC', 'filters': [
        {'filterType': 'PRICE_FILTER', 'tickSize': '0.1'},
        {'filterType': 'LOT_SIZE', 'stepSize': '0.001', 'minQty': '0.001'},
        {'filterType': 'MIN_NOTIONAL', 'notional': '10'}]}]}))
    config = tmp_path / 'profile-config.json'
    out = tmp_path / 'models'
    loaded = []
    def load(path):
        loaded.append(path)
        return dict(window={'start': '2026-08-01'}, rules=str(rules), symbols=['BTCUSDC'],
                    datasets={'BTCUSDC': {'cache': '$QUANT_DATA_ROOT/certified/btc',
                                         'order_reference': {'price': '60000'}}},
                    execution={'request_ns': 500000}, engine={'queue': 'cancel_50_50'})
    monkeypatch.setattr(module, 'load_spec', load)
    monkeypatch.setenv('QUANT_DATA_ROOT', str(tmp_path / 'raw-root'))
    fitted = []
    def fit(dataset, cutoff, *args):
        fitted.append((dataset, cutoff))
        return dict(samples=1000, last_label_ns=cutoff-1, filled_orders=500,
                    supported=[True]*6)
    monkeypatch.setattr(module, 'fit_symbol' if kind == 'alpha' else 'fit', fit)
    monkeypatch.setattr(sys, 'argv', ['train', str(config), str(out)])
    module.main()
    assert loaded == [config]
    assert fitted == [(tmp_path / 'raw-root/certified/btc', 1785542400000000000)]
    assert json.loads((out / 'BTCUSDC.json').read_text())['symbol'] == 'BTCUSDC'
    assert not list(out.glob('*.tmp'))


@pytest.mark.parametrize('kind', ['alpha', 'value'])
def test_invalid_training_config_creates_no_output(tmp_path, monkeypatch, kind):
    module = importlib.import_module('tools.models.prepare_nh_' + kind)
    def reject(_):
        raise ValueError('invalid experiment')
    monkeypatch.setattr(module, 'load_spec', reject)
    out = tmp_path / 'models'
    monkeypatch.setattr(sys, 'argv', ['train', 'invalid.json', str(out)])
    with pytest.raises(ValueError, match='invalid experiment'):
        module.main()
    assert not out.exists()
