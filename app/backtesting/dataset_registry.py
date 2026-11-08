"""认证数据集注册表：统一实验入口强制核对币种、窗口和缓存身份。"""
import hashlib
import json
from pathlib import Path
from ..market_making.strategies.registry import definition_for

ROOT = Path(__file__).resolve().parents[2]
REGISTRY_ROOT = ROOT / 'config' / 'datasets'
REQUIRED_FIELDS = {'schema_version', 'name', 'symbol', 'window', 'cache_key', 'cache_path',
                   'cache_manifest_sha256', 'feed_clock', 'feed_latency_ms', 'gap_policy',
                   'prewindow_context', 'certified_date', 'source', 'notes'}


def load_registry(root=REGISTRY_ROOT):
    """读取全部注册条目并做静态校验;条目名必须等于文件名。"""
    root = Path(root)
    if not root.is_dir():
        raise ValueError('Dataset registry missing: ' + str(root))
    entries = {}
    for path in sorted(root.glob('*.json')):
        # macOS stores binary AppleDouble metadata beside files on exFAT.
        # These are not registrations; real malformed JSON must still fail.
        if path.name.startswith('._'):continue
        entry = json.loads(path.read_text())
        if set(entry) != REQUIRED_FIELDS:
            raise ValueError(f'Registry entry {path.name} fields differ from {sorted(REQUIRED_FIELDS)}')
        if entry['schema_version'] != 1 or entry['name'] != path.stem:
            raise ValueError('Registry entry name/schema mismatch: ' + path.name)
        if entry['window']['timezone'] != 'UTC':
            raise ValueError('Registry windows are UTC: ' + path.name)
        if not entry['cache_path'].endswith('/' + entry['cache_key']):
            raise ValueError('Registry cache_path must end with cache_key: ' + path.name)
        if entry['name'] in entries:
            raise ValueError('Duplicate registry entry: ' + entry['name'])
        entries[entry['name']] = entry
    if not entries:
        raise ValueError('Dataset registry is empty')
    keys = [e['cache_key'] for e in entries.values()]
    if len(set(keys)) != len(keys):
        raise ValueError('A cache key may only be certified once')
    return entries


def find_by_cache(entries, cache_path, symbol):
    """按缓存 key(路径末段)+币种定位注册条目;未登记返回 None。"""
    key = str(cache_path).rstrip('/').rsplit('/', 1)[-1]
    for entry in entries.values():
        if entry['cache_key'] == key and entry['symbol'] == symbol:
            return entry
    return None


def check_experiment(spec, resolve):
    """未登记、窗口不符或策略不支持预热时直接拒绝。"""
    entries = load_registry()
    problems = []
    for symbol in spec['symbols']:
        item = spec['datasets'][symbol]
        entry = find_by_cache(entries, resolve(item['cache']), symbol)
        if entry is None:
            problems.append(f'{symbol}: 缓存 {item["cache"]} 未在 config/datasets 注册表登记')
            continue
        window = dict(start=spec['window']['start'], end_exclusive=spec['window']['end_exclusive'])
        registered = {k: entry['window'][k] for k in ('start', 'end_exclusive')}
        if registered != window:
            problems.append(f'{symbol}: 实验窗口 {window} 与注册数据集 {entry["name"]} 窗口 {registered} 不一致')
        if entry['prewindow_context'] and any(not definition_for(s).supports_prewindow for s in spec['strategies']):
            # 与 validate_inputs 的预热上下文规则同源,这里只复述为注册表层提示。
            problems.append(f'{symbol}: 注册数据集 {entry["name"]} 带预热上下文,需要显式 start-aware 策略')
    if problems:
        raise ValueError('Dataset registry check failed: ' + '; '.join(problems))
    return problems


def check_manifest(entry, manifest_path):
    """prepare 阶段:盘上 manifest 必须与注册条目冻结的 SHA256 一致。"""
    if entry['cache_manifest_sha256'] != digest(manifest_path):
        raise ValueError(f"Registered dataset {entry['name']} manifest changed on disk")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()
