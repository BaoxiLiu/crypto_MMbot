import json
from pathlib import Path

import pytest

from app.backtesting import dataset_registry as registry
from app.backtesting.experiment_config import ROOT, load_spec


def config():
    return json.loads((ROOT / 'config/experiments/default.json').read_text())


def write(tmp_path, spec):
    path = tmp_path / 'spec.json'
    path.write_text(json.dumps(spec))
    return path


class TestRegistryContents:
    def test_appledouble_metadata_is_ignored_but_bad_actual_json_is_not(self,tmp_path):
        entry=next(iter(registry.load_registry().values()))
        (tmp_path/(entry['name']+'.json')).write_text(json.dumps(entry))
        (tmp_path/('._'+entry['name']+'.json')).write_bytes(b'\x00\x05\x16\x07\xb0')
        assert set(registry.load_registry(tmp_path))=={entry['name']}
        (tmp_path/'broken.json').write_bytes(b'\xb0')
        with pytest.raises(UnicodeDecodeError):registry.load_registry(tmp_path)

    def test_entries_are_complete_unique_and_named_by_file(self):
        entries = registry.load_registry()
        current = config()
        for symbol, item in current['datasets'].items():
            assert registry.find_by_cache(entries, item['cache'], symbol) is not None
        keys = [e['cache_key'] for e in entries.values()]
        assert len(set(keys)) == len(keys)
        for name, entry in entries.items():
            assert (registry.REGISTRY_ROOT / (name + '.json')).is_file()
            assert entry['feed_clock'] == 'publication_E_plus_transport_exchange_T'
            assert entry['gap_policy'] == 'independent_segments'

    def test_manifest_hash_is_verified(self, tmp_path):
        import hashlib
        path = tmp_path / 'manifest.json'
        payload = b'{"version":6}'
        path.write_bytes(payload)
        entry = {'name': 'fixture', 'cache_manifest_sha256': hashlib.sha256(payload).hexdigest()}
        registry.check_manifest(entry, path)
        path.write_bytes(b'{"version":5}')
        with pytest.raises(ValueError, match='manifest changed'):
            registry.check_manifest(entry, path)

    def test_manifest_tampering_rejected(self):
        entries = registry.load_registry()
        entry = entries['quick-aug01-03-context-jul30-dogeusdc-v1']
        with pytest.raises(ValueError, match='manifest changed'):
            registry.check_manifest(entry, ROOT / 'config/experiments/default.json')

    def test_quarantined_and_uncertified_caches_are_not_registered(self):
        keys = {e['cache_key'] for e in registry.load_registry().values()}
        assert 'quarantine-b8290dd044d7aac2be50' not in keys
        assert '1165db1b0af2d4223cd1' not in keys  # SUI 月度训练缓存:供应商/官方冲突,未认证


class TestExperimentGating:
    def test_root_config_passes_strict_registry_and_strategy_checks(self):
        spec = load_spec(ROOT / 'config/experiments/default.json')
        assert spec['schema_version'] == 2

    def test_unregistered_cache_rejected_in_strict_schema(self, tmp_path):
        spec = config()
        spec['datasets']['DOGEUSDC']['cache'] = '$QUANT_DATA_ROOT/crypto/certified/DOGEUSDC/021491db96fd65095361'
        with pytest.raises(ValueError, match='注册表'):
            load_spec(write(tmp_path, spec))

    def test_window_mismatch_with_registered_dataset_rejected(self, tmp_path):
        spec = config()
        spec['window']['start'] = '2026-08-02'
        spec['stage'] = 'custom'
        with pytest.raises(ValueError, match='窗口'):
            load_spec(write(tmp_path, spec))

    @pytest.mark.parametrize('schema', [0, 1, 3])
    def test_unsupported_schema_is_rejected(self, tmp_path, schema):
        spec = config()
        spec['schema_version'] = schema
        with pytest.raises(ValueError):
            load_spec(write(tmp_path, spec))

    def test_implicit_strategy_defaults_rejected_in_strict_schema(self, tmp_path, monkeypatch):
        spec = config()
        baseline = json.loads((ROOT / spec['strategy_parameters']['DOGEUSDC']).read_text())
        del baseline['momentum_ns']
        path = tmp_path / 'strategy.json'
        path.write_text(json.dumps(baseline))
        monkeypatch.setitem(spec['strategy_parameters'], 'DOGEUSDC', str(path))
        with pytest.raises(ValueError, match='momentum_ns'):
            load_spec(write(tmp_path, spec))

    def test_implicit_as_config_default_rejected_in_strict_schema(self, tmp_path):
        spec = config()
        baseline = json.loads((ROOT / spec['strategy_parameters']['DOGEUSDC']).read_text())
        del baseline['as_config']['kappa']
        path = tmp_path / 'strategy.json'
        path.write_text(json.dumps(baseline))
        spec['strategy_parameters']['DOGEUSDC'] = str(path)
        with pytest.raises(ValueError, match='kappa'):
            load_spec(write(tmp_path, spec))
