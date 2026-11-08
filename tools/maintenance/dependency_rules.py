"""Architecture gate for every production module, including future filenames."""
import ast
from importlib.util import resolve_name

CODE_ROOTS = ('app', 'src', 'tools')
# These numerical/state-machine implementations have independent frozen tests.
# Budgets are explicit upper bounds, not permission to grow them indefinitely.
MODULE_LIMITS = {
    'app/market_making/core.py': 771,
    'app/prediction/factors.py': 820,
    'app/prediction/builder.py': 675,
}
DEFAULT_MODULE_LIMIT = 450


def is_test(path):
    return 'tests' in path.parts or path.name.startswith('test_')


def imports(source, module):
    package = module.rpartition('.')[0]
    if module.endswith('.__init__'):
        package = module[:-9]
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = resolve_name('.'*node.level+(node.module or ''), package) if node.level else (node.module or '')
            yield base
            yield from (base+'.'+alias.name for alias in node.names if alias.name != '*')


def forbidden_imports(source, module, prefixes):
    return sorted({name for name in imports(source, module)
                   if any(name == p or name.startswith(p+'.') for p in prefixes)})


def prohibited(module):
    """Whole-layer rules apply to new files without editing a filename allowlist."""
    blocked = ['research', 'app.backtesting.compare_runs', 'app.backtesting.metrics']
    blocked += ['app.market_making.fusion.'+name for name in (
        'as_strategy','nh_strategy','hjb_strategy','fb_hjb_math','strategy_registry',
        'strategy_base','nh_quote_value','adaptive_quote','own_queue','queue_reactive','quote_policy')]
    if module.startswith(('app.', 'src.')):
        blocked += ['tools']
    if module.startswith('src.metrics.'):
        blocked += ['app', 'hftbacktest']
    if module == 'app.artifacts':
        blocked += ['app', 'src', 'numpy', 'hftbacktest']
    if module == 'app.market_making.contracts':
        blocked += ['app', 'src', 'hftbacktest', 'numpy']
    if module.startswith(('app.factor_library.', 'app.prediction.')):
        blocked += ['app.backtesting', 'app.market_making', 'hftbacktest']
    if module.startswith('app.market_making.strategies.'):
        blocked += ['app.backtesting', 'app.market_making.live', 'hftbacktest',
                    'app.market_making.fusion.orders', 'app.market_making.fusion.kernel',
                    'app.market_making.fusion.native_reads', 'app.market_making.fusion.event_backtest',
                    'app.market_making.core.Ledger', 'app.market_making.core.Store']
    if module.startswith('app.market_making.fusion.'):
        blocked += ['app.backtesting', 'app.market_making.live', 'app.market_making.strategies']
    if module.startswith('app.market_making.live.'):
        blocked += ['app.backtesting']
        if module.rsplit('.', 1)[-1] not in ('selection', 'runtime', 'replay'):
            blocked += ['app.market_making.strategies']
    if module == 'app.backtesting.experiment_config':
        blocked += ['app.backtesting.fusion_experiment', 'app.backtesting.experiment_report',
                    'app.backtesting.html_report']
    # Shared math/ledger/quote utilities may not depend on orchestration or strategies.
    if module.startswith('app.market_making.') and len(module.split('.')) == 3:
        blocked += ['app.backtesting', 'app.market_making.strategies',
                    'app.market_making.live', 'app.market_making.fusion', 'hftbacktest']
    return blocked


def module_graph(sources):
    """Resolve local imports to modules; include function-local edges conservatively."""
    names = {m.removesuffix('.__init__'): m for m in sources}
    graph = {m: set() for m in sources}
    for module, source in sources.items():
        for target in imports(source, module):
            if target in names and names[target] != module:
                graph[module].add(names[target])
        # Registry loading is closed over reviewed literal module declarations.
        if module == 'app.market_making.strategies.registry':
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'StrategyDefinition':
                    target = 'app.market_making.strategies.'+ast.literal_eval(node.args[0])
                    if target in names: graph[module].add(names[target])
    return graph


def cycles(graph):
    """Tarjan SCCs report each cyclic dependency group once, in stable order."""
    serial = 0
    indices, low, stack, active, found = {}, {}, [], set(), []
    def visit(node):
        nonlocal serial
        indices[node] = low[node] = serial; serial += 1
        stack.append(node); active.add(node)
        for target in sorted(graph[node]):
            if target not in indices:
                visit(target); low[node] = min(low[node], low[target])
            elif target in active:
                low[node] = min(low[node], indices[target])
        if low[node] == indices[node]:
            group = []
            while True:
                target = stack.pop(); active.remove(target); group.append(target)
                if target == node: break
            if len(group) > 1: found.append(sorted(group))
    for node in sorted(graph):
        if node not in indices: visit(node)
    return sorted(found)


def check_boundaries(root):
    sources, issues = {}, []
    for folder in CODE_ROOTS:
        for path in sorted((root/folder).rglob('*.py')):
            if path.name.startswith('._') or path.is_symlink() or is_test(path.relative_to(root)):
                continue
            relative = path.relative_to(root).as_posix()
            module = '.'.join(path.relative_to(root).with_suffix('').parts)
            source = path.read_text()
            try:
                ast.parse(source)
                targets = list(imports(source, module))
            except (SyntaxError, ImportError, ValueError) as error:
                issues.append(f'invalid-module:{relative}:{error}'); continue
            sources[module] = source
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.ImportFrom) and any(a.name == '*' for a in node.names):
                    issues.append(f'wildcard-import:{relative}')
                if isinstance(node, ast.Call) and module != 'app.market_making.strategies.registry':
                    name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, 'attr', '')
                    if name in ('__import__', 'import_module'):
                        issues.append(f'dynamic-import:{relative}')
            for target in targets:
                if any(part == 'tests' or part.startswith('test_') for part in target.split('.')):
                    issues.append(f'test-dependency:{relative}->{target}')
            for target in forbidden_imports(source, module, prohibited(module)):
                issues.append(f'dependency:{relative}->{target}')
            limit = MODULE_LIMITS.get(relative, DEFAULT_MODULE_LIMIT)
            if len(source.splitlines()) > limit:
                issues.append(f'module-budget:{relative}:{len(source.splitlines())}>{limit}')
    issues += ['dependency-cycle:'+' -> '.join(group) for group in cycles(module_graph(sources))]
    return sorted(set(issues))
