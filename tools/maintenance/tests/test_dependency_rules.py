from pathlib import Path
from tools.maintenance.dependency_rules import forbidden_imports,check_boundaries


def test_function_local_and_relative_imports_cannot_bypass_boundary():
    code='def compute():\n    from ..backtesting import html_report\n    import research.old as old\n'
    found=forbidden_imports(code,'app.factor_library.example',('research','app.backtesting'))
    assert 'app.backtesting.html_report' in found and 'research.old' in found


def test_stdlib_and_local_math_are_allowed():
    assert forbidden_imports('from .core import calculate\nimport math','src.metrics.schema',('app','research'))==[]


def test_current_core_boundaries():
    assert check_boundaries(Path(__file__).resolve().parents[3])==[]


def test_paper_implementation_preserves_verified_numerical_source():
    import hashlib,json
    root=Path(__file__).resolve().parents[3]
    record=json.loads((root/'doc/architecture/refactor/factor_migration.json').read_text())
    assert hashlib.sha256((root/record['canonical']).read_bytes()).hexdigest()==record['implementation_sha256']


def test_runtime_cannot_import_controlled_test_quotes():
    source = 'from .tests.quote_fixture import QuoteFixture'
    assert forbidden_imports(source, 'app.market_making.strategies.registry',
                             ('app.market_making.strategies.tests',))


def test_new_strategy_filename_cannot_import_oms_native_or_reports(tmp_path):
    package = tmp_path/'app/market_making/strategies'
    package.mkdir(parents=True)
    (package/'future_policy.py').write_text(
        'def choose():\n    from ..fusion.orders import OrderManager\n'
        '    from app.backtesting import html_report\n    import hftbacktest\n')
    issues = check_boundaries(tmp_path)
    assert any('future_policy.py->app.market_making.fusion.orders' in x for x in issues)
    assert any('future_policy.py->app.backtesting.html_report' in x for x in issues)
    assert any('future_policy.py->hftbacktest' in x for x in issues)


def test_cycle_detection_and_module_budget_cover_new_files(tmp_path):
    package = tmp_path/'app/example'
    package.mkdir(parents=True)
    (package/'a.py').write_text('from .b import value\n'+'\n'*450)
    (package/'b.py').write_text('from .a import value\n')
    issues = check_boundaries(tmp_path)
    assert any(x.startswith('dependency-cycle:') for x in issues)
    assert any(x.startswith('module-budget:app/example/a.py') for x in issues)


def test_test_imports_and_dynamic_loading_cannot_hide_in_new_modules(tmp_path):
    package = tmp_path/'app/example'
    package.mkdir(parents=True)
    (package/'x.py').write_text('from app.market_making.fusion.tests import quote_fixture\n'
                              'import importlib\nimportlib.import_module("app.backtesting")\n')
    issues = check_boundaries(tmp_path)
    assert any(x.startswith('test-dependency:') for x in issues)
    assert any(x.startswith('dynamic-import:') for x in issues)


def test_graph_includes_closed_registry_loading():
    from tools.maintenance.dependency_rules import module_graph
    graph = module_graph({
        'app.market_making.strategies.registry': "x=StrategyDefinition('new_policy', 'Policy', (1,), 'description')",
        'app.market_making.strategies.new_policy': 'value=1',
    })
    assert graph['app.market_making.strategies.registry'] == {'app.market_making.strategies.new_policy'}


def test_new_tools_cannot_reintroduce_container_execution(tmp_path):
    package=tmp_path/'tools/datasets';package.mkdir(parents=True)
    (package/'new_tool.py').write_text('import subprocess\nsubprocess.run(["docker", "exec", "old"])\n')
    assert 'container-runtime:tools/datasets/new_tool.py' in check_boundaries(tmp_path)
