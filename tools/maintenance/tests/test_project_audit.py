"""The architecture ratchet must distinguish source, fixtures and generated data."""
import unittest
from tools.maintenance.project_audit import violations


class ArchitectureChecks(unittest.TestCase):
    def test_artifacts_and_legacy_experiments_are_detected(self):
        issues = violations(['app/backtesting/experiments/run/result.json',
                             'app/backtesting/curve.png', 'src/metrics/cache.npz',
                             'tools/new_date_script.py'])
        self.assertEqual(len(issues), 4)
        self.assertIn('history-in-code:app/backtesting/experiments/run/result.json', issues)

    def test_declared_sources_and_text_fixtures_are_allowed(self):
        self.assertEqual(violations(['app/market_making/strategies/as_strategy.py',
                                    'tools/maintenance/project_audit.py',
                                    'app/backtesting/tests/fixtures/trace.json',
                                    'doc/reports/experiments/run/report.html']), [])

    def test_new_path_not_hidden_by_same_extension_legacy_debt(self):
        old = set(violations(['app/backtesting/old.png']))
        new = set(violations(['app/backtesting/old.png', 'app/backtesting/new.png']))
        self.assertEqual(new - old, {'artifact-in-code:app/backtesting/new.png'})


if __name__ == '__main__':
    unittest.main()


def test_current_code_has_no_flat_tools_history_or_generated_artifacts():
    from tools.maintenance.project_audit import ROOT,git_paths
    assert violations(git_paths(ROOT))==[]


def test_retired_parallel_backtest_entrypoints_cannot_return():
    from tools.maintenance.project_audit import RETIRED_RUNNERS
    assert {x for x in violations(sorted(RETIRED_RUNNERS)) if x.startswith('retired-runner:')}=={'retired-runner:'+p for p in RETIRED_RUNNERS}


def test_physical_inventory_includes_ignored_outputs_but_not_external_data(tmp_path):
    from tools.maintenance.project_audit import physical_inventory
    project = tmp_path / 'project'
    (project / 'data').mkdir(parents=True)
    (project / 'data' / 'result.bin').write_bytes(b'12345')
    (project / '__pycache__').mkdir()
    (project / '__pycache__' / 'code.pyc').write_bytes(b'abc')
    (project / '.git').mkdir()
    (project / '.git' / 'objects').write_bytes(b'not counted')
    external = tmp_path / 'external'
    external.mkdir()
    (external / 'raw.bin').write_bytes(b'do not traverse')
    (project / 'raw').symlink_to(external, target_is_directory=True)
    report = physical_inventory(project)
    assert report['files'] == 2 and report['bytes'] == 8
    assert report['directories']['__pycache__']['cache_files'] == 1
    assert report['directories']['raw']['symlinks'] == 1
    assert not report['errors']


def test_retired_independent_projects_cannot_return():
    paths = ['latency/src/main.rs', 'baselines/as_prediction_v1/cli.py',
             'experiments/old-run/config.json']
    assert violations(paths) == sorted('retired-tree:' + p for p in paths)
    assert violations(['config/experiments/example.json',
                       'data/experiments/registry.json']) == []


def test_ignored_runtime_files_are_checked_without_following_links(tmp_path):
    from tools.maintenance.project_audit import physical_code_paths
    (tmp_path / 'app/data').mkdir(parents=True)
    (tmp_path / 'app/data/state.json').write_text('{}')
    (tmp_path / 'app/data/._state.json').write_bytes(b'ignored metadata')
    external = tmp_path / 'external'
    external.mkdir()
    (external / 'raw.log').write_text('shared data')
    (tmp_path / 'app/shared').symlink_to(external, target_is_directory=True)
    paths = physical_code_paths(tmp_path)
    assert paths == ['app/data/state.json']
    assert violations(paths) == ['runtime-or-deploy-in-code:app/data/state.json']


def test_retired_subsystems_reject_unseen_filenames_without_blocking_active_peers():
    from tools.maintenance.project_audit import RETIRED_TREES
    paths = [prefix + '/new_subdir/future_entry.py' for prefix in RETIRED_TREES]
    assert violations(paths) == sorted('retired-tree:' + p for p in paths)
    assert violations(['tools/models/new_trainer.py',
                       'app/market_making/fusion/new_math.py',
                       'config/experiments/new_comparison.json']) == []


def test_ignored_raw_data_cannot_return_to_project(tmp_path):
    from tools.maintenance.project_audit import physical_code_paths, violations
    for name in ('data/official_binance/trades/X/day.zip',
                 'data/provider_comparison_new/sample.parquet',
                 'data/raw/sample.gz', 'data/models/current.json',
                 'data/experiments/registry.json'):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}')
    assert violations(physical_code_paths(tmp_path)) == [
        'raw-data-in-project:data/official_binance/trades/X/day.zip',
        'raw-data-in-project:data/provider_comparison_new/sample.parquet',
        'raw-data-in-project:data/raw/sample.gz',
    ]


def test_source_tree_without_git_can_be_checked(tmp_path):
    from tools.maintenance.project_audit import git_paths
    (tmp_path / 'README.md').write_text('project')
    (tmp_path / '._README.md').write_text('metadata')
    assert git_paths(tmp_path) == ['README.md']


def test_documentation_and_source_backups_cannot_accumulate_in_code():
    assert violations(['app/new/README.md','tools/new/backup/source.py']) == [
        'backup-in-code:tools/new/backup/source.py',
        'documentation-in-code:app/new/README.md',
    ]
