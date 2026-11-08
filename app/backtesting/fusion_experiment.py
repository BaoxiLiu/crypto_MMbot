"""Project-level offline experiment entry: prepare, run, report and compare immutable HTML."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from decimal import Decimal as D
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

from .experiment_metrics import write_json
from . import dataset_registry, run_registry
from .experiment_config import ROOT, resolve, identifier, load_spec, parameter_file, effective_config
from ..market_making.strategies.registry import factory_for,bind_symbol,definition_for
from ..artifacts import digest,source_identity
from ..market_making.fusion.config import FusionConfig
from ..market_making.fusion.execution import ExecutionProfile

def registration_path(path):
    path=Path(path)
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def default_run_id(spec, config_sha256, now=None):
    """Readable identity; the full labels and config stay in registration.json."""
    stamp = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    coins = '-'.join(spec['symbols'])
    if len(coins) > 24:
        coins = spec['symbols'][0][:16] + '-n' + str(len(spec['symbols']))
    window = spec['window']['start'].replace('-', '') + '-' + spec['window']['end_exclusive'].replace('-', '')
    name = identifier(spec['name'])[:24].rstrip('-_')
    return identifier(f'{stamp}_{coins}_{window}_{name}_{config_sha256[:8]}')


def validate_inputs(spec):
    inputs = {}
    hashes = dict(spec.get('_profile_inputs',{}))
    for path,expected in hashes.items():
        if digest(path)!=expected:raise ValueError('Profile changed after resolution: '+path)
    metadata_path = resolve(spec['rules'])
    metadata = json.loads(metadata_path.read_text())
    hashes[str(metadata_path)] = digest(metadata_path)
    from ..market_making.core import Rules
    from ..market_making.fusion.funding import load
    for symbol in spec['symbols']:
        item = spec['datasets'][symbol]
        dataset = resolve(item['cache'])
        manifest_path = dataset / 'manifest.json'
        m = json.loads(manifest_path.read_text())
        registered = dataset_registry.find_by_cache(dataset_registry.load_registry(), dataset, symbol)
        registry_name = None
        if registered is not None:
            # 已登记数据集的 manifest 必须与注册表冻结值一致(任何 schema 都强制)。
            dataset_registry.check_manifest(registered, manifest_path)
            registry_name = registered['name']
        preparation = json.loads(resolve(item['input_spec']).read_text())
        source_spec = preparation['config']
        if resolve(preparation['dataset']).resolve() != dataset.resolve():
            raise ValueError('Input specification points at a different cache')
        if (source_spec['symbol'], source_spec['start'], source_spec['end_exclusive']) != (symbol, spec['window']['start'], spec['window']['end_exclusive']):
            raise ValueError('Dataset preparation range differs from experiment; prepare a matching cache')
        if m['symbol'] != symbol or m['version'] != 6 or m['feed_clock'] != 'publication_E_plus_transport_exchange_T' or m['unresolved_trade_gaps']:
            raise ValueError('Certified publication-clock dataset required')
        if m.get('prewindow_context') and any(not definition_for(s).supports_prewindow for s in spec['strategies']):
            raise ValueError('Historical warmup-context cache requires an explicit start-aware strategy')
        if D(str(m['feed_latency_ms'])) * 1_000_000 != spec['execution']['feed_ns']:
            raise ValueError('Cache/profile feed latency mismatch; never add twice')
        start = datetime.fromisoformat(spec['window']['start']).replace(tzinfo=timezone.utc)
        end = datetime.fromisoformat(spec['window']['end_exclusive']).replace(tzinfo=timezone.utc)
        coverage_start=datetime.fromisoformat(source_spec.get('context_start',spec['window']['start'])).replace(tzinfo=timezone.utc)
        expected = {int((coverage_start + timedelta(hours=h)).timestamp()) for h in range(int((end-coverage_start).total_seconds()/3600))}
        actual = {int(datetime.fromisoformat(h['hour']).timestamp()) for h in m['hours']}
        if expected != actual:
            raise ValueError('Cached hour coverage differs from requested range')
        rows, funding_manifest = load(resolve(item['funding_manifest']), symbol, spec['window']['start'], spec['window']['end_exclusive'])
        rule = Rules.from_exchange(next(x for x in metadata['symbols'] if x['symbol'] == symbol))
        for path in [manifest_path, dataset/'project_inputs/marks.npy', resolve(item['input_spec']), parameter_file(spec,item,symbol), resolve(item['funding_manifest'])]:
            hashes[str(path)] = digest(path)
        for p in [funding_manifest['path'], *(x['path'] for x in funding_manifest['pages'])]:
            path = resolve(item['funding_manifest']).parent / p
            hashes[str(path)] = digest(path)
        inputs[symbol] = dict(dataset=str(dataset), dataset_registry=registry_name, source_spec=source_spec, funding_rows=len(rows),
                              segments=len(m['segments']), rows=sum(f['rows'] for seg in m['segments'] for f in seg['files']))
    context = spec['observations'].get('order_context')
    if context is not None:
        if set(context) != {'source_run'}:
            raise ValueError('Unknown order-context settings')
        source = resolve(context['source_run'])
        original = json.loads((source/'registration.json').read_text())
        if any(hashes[p] != value for p,value in original['input_sha256'].items() if p in hashes):
            raise ValueError('Order-context replay changed an original input')
        for key in ('window', 'engine', 'execution', 'costs', 'loss_stops'):
            if original['spec'][key] != spec[key]:
                raise ValueError('Order-context replay changed original ' + key)
        hashes[str(source/'registration.json')] = digest(source/'registration.json')
        for symbol in spec['symbols']:
            for case in spec['strategies']:
                case_id = symbol + '--' + case['id']
                old = next(c for c in original['cases'] if c['id'] == case_id)
                if old['mask'] != case['mask'] or old['config'] != effective_config(spec, spec['datasets'][symbol], case):
                    raise ValueError('Order-context replay changed original strategy')
                for name in ('fills.jsonl', 'summary.json'):
                    p = source/case_id/name
                    hashes[str(p)] = digest(p)
    return inputs, hashes


def output_directories(spec,run_id):
    from .result_retention import final_only,create_workspace
    final=resolve(spec['outputs']['reports_root'])/run_id
    out=resolve(spec['outputs']['runs_root'])/run_id
    if final_only(spec):out=out.parent/'.work'/run_id
    if out.exists() or final.exists():raise FileExistsError('Run/report identity already exists')
    if final_only(spec):
        create_workspace(out,final,run_id)
        report=out/'report';report.mkdir()
        return out,report,final
    out.mkdir(parents=True);final.mkdir(parents=True)
    return out,final,None


def prepare(config, run_id=None, max_segments=None, max_files=None):
    config = Path(config).resolve()
    spec = load_spec(config)
    from .campaign_standard import validate_preparation
    preparation_policy = validate_preparation(spec)
    inputs, hashes = validate_inputs(spec)
    probe = subprocess.check_output([spec['runtime']['backtest_python'], '-c',
        'import importlib.metadata,json,hashlib;import hftbacktest._hftbacktest as n;print(json.dumps(dict(version=importlib.metadata.version("hftbacktest"),binary=n.__file__,sha256=hashlib.sha256(open(n.__file__,"rb").read()).hexdigest())))'], text=True)
    native = json.loads(probe)
    if native['version'] != spec['engine']['version']:
        raise ValueError('Use verified HFT environment')
    if native.get('sha256',digest(native['binary']) if Path(native['binary']).exists() else None) != spec['engine']['native_sha256']:
        raise ValueError('Unverified native binary')
    run_id = identifier(run_id) if run_id else default_run_id(spec, digest(config))
    source_hashes = source_identity(ROOT)
    cases = []
    for symbol in spec['symbols']:
        for s in spec['strategies']:
            identifier(symbol)
            case_id = symbol + '--' + s['id']
            cases.append(dict(id=case_id, symbol=symbol, strategy=s['id'], mask=s['mask'],
                              config=effective_config(spec, spec['datasets'][symbol], s)))
            manifest_hash=digest(Path(inputs[symbol]['dataset'])/'manifest.json')
            cases[-1]['plugin']=bind_symbol(s,symbol,symbols=spec['symbols'],
                                           manifest_sha256=manifest_hash)['plugin']
            if s.get('fee_scenario') is not None:cases[-1]['fee_scenario']=s['fee_scenario']
    for path, expected in spec.get('_profile_inputs',{}).items():
        if digest(path)!=expected:
            raise ValueError('Profile changed while preparing: '+path)
    # Validate models, effective settings and source identities before allocating work.
    out,report,final_report=output_directories(spec,run_id)
    for case in cases:
        case['output']=str(out/case['id'])
    reg = dict(schema_version=1, kind='new_run', registered_utc=datetime.now(timezone.utc).isoformat(),
               run_id=run_id, root=str(ROOT), run_output=str(out), report_output=str(report),
               config_sha256=digest(config), spec=spec, inputs=inputs, input_sha256=hashes,
               source_sha256=source_hashes, native_binary=native['binary'],
               max_segments=max_segments, max_files=max_files, full_range=max_segments is None and max_files is None,
               cases=cases, real_orders_sent=False, preparation_policy=preparation_policy)
    if final_report is not None:reg['final_report_output']=str(final_report)
    write_json(out/'effective_config.json', spec)
    write_json(out/'registration.json', reg)
    write_json(report/'PREREGISTRATION.json', reg)
    write_json(out/'status.json', dict(status='prepared', full_range=reg['full_range']))
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    except Exception:
        commit = None
    run_registry.append(dict(
        id=run_id, kind='run', config=registration_path(config),
        datasets=sorted({v['dataset_registry'] for v in inputs.values() if v.get('dataset_registry')}) or None,
        code_ref=commit, status='prepared',
        artifacts=[registration_path(out), registration_path(report)],
        conclusion=None, notes=None, preparation_policy=preparation_policy))
    return out/'registration.json'


def worker(path, case_id):
    reg = json.loads(Path(path).read_text())
    from .campaign_standard import validate_preparation
    validate_preparation(reg['spec'])
    for p, h in reg['input_sha256'].items():
        if digest(p) != h:
            raise ValueError('Input changed after registration: ' + p)
    for p, h in reg['source_sha256'].items():
        if digest(ROOT/p) != h:
            raise ValueError('Code changed after registration: ' + p)
    spec = reg['spec']
    if importlib.metadata.version('hftbacktest') != spec['engine']['version'] or digest(reg['native_binary']) != spec['engine']['native_sha256']:
        raise ValueError('Native engine changed after registration')
    case = next(c for c in reg['cases'] if c['id'] == case_id)
    item = spec['datasets'][case['symbol']]
    from ..market_making.fusion.funding import load
    from .experiment_metrics import ReplayMetrics
    from ..market_making.core import Rules
    from ..market_making.fusion.event_backtest import run_event_arm
    metadata = json.loads(resolve(spec['rules']).read_text())
    rules = Rules.from_exchange(next(s for s in metadata['symbols'] if s['symbol'] == case['symbol']))
    rows, _ = load(resolve(item['funding_manifest']), case['symbol'], spec['window']['start'], spec['window']['end_exclusive'])
    with ExitStack() as stack:
        if spec['observations'].get('pnl_attribution'):
            from .pnl_attribution import PnLAttributionMetrics
            factory = lambda output: stack.enter_context(PnLAttributionMetrics(output, spec['observations'], spec['window']['start'], spec['window']['end_exclusive']))
        elif spec['observations'].get('order_lifecycle'):
            from .order_context_metrics import OrderLifecycleMetrics
            factory = lambda output: stack.enter_context(OrderLifecycleMetrics(output, spec['observations'], spec['window']['start'], spec['window']['end_exclusive']))
        elif spec['observations'].get('order_placements'):
            from .order_context_metrics import OrderPlacementMetrics
            factory = lambda output: stack.enter_context(OrderPlacementMetrics(output, spec['observations'], spec['window']['start'], spec['window']['end_exclusive']))
        elif spec['observations'].get('order_context'):
            from .order_context_metrics import OrderContextMetrics
            source = resolve(spec['observations']['order_context']['source_run'])/case_id/'fills.jsonl'
            factory = lambda output: stack.enter_context(OrderContextMetrics(output, spec['observations'], spec['window']['start'], spec['window']['end_exclusive'], source))
        else:
            factory = lambda output: stack.enter_context(ReplayMetrics(output, spec['observations'], spec['window']['start'], spec['window']['end_exclusive']))
        strategy_factory = factory_for(case)
        run_event_arm(Path(reg['inputs'][case['symbol']]['dataset']), Path(case['output']), case['mask'],
                      FusionConfig.from_dict(case['config']), resolve(item['cache'])/'project_inputs/marks.npy',
                      rows, rules, ExecutionProfile.from_dict(spec['execution']), reg['max_segments'], reg['max_files'], factory,
                      strategy_factory=strategy_factory,
                      queue_model=spec['engine']['queue'])


def run(path):
    reg = json.loads(Path(path).read_text())
    from .campaign_standard import validate_preparation
    validate_preparation(reg['spec'])
    out = Path(reg['run_output'])
    write_json(out/'status.json', dict(status='running', full_range=reg['full_range']))
    def launch(case):
        with (out/(case['id']+'.log')).open('w') as log:
            result = subprocess.run([reg['spec']['runtime']['backtest_python'], '-m', 'app.backtesting.fusion_experiment', 'worker',
                         '--registration', str(path), '--case-id', case['id']], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                         env=dict(os.environ, NUMBA_NUM_THREADS='1', POLARS_MAX_THREADS='1', OPENBLAS_NUM_THREADS='1'))
        return dict(case=case['id'], exit=result.returncode)
    try:
        with ThreadPoolExecutor(max_workers=reg['spec']['resources']['workers']) as pool:
            statuses = list(pool.map(launch, reg['cases']))
        write_json(out/'process-results.json', statuses)
        if any(s['exit'] != 0 for s in statuses):
            raise RuntimeError('Experiment workers failed: '+str([s for s in statuses if s['exit'] != 0]))
        with (out/'report.log').open('w') as log:
            subprocess.run([reg['spec']['runtime']['report_python'], '-m', 'app.backtesting.fusion_experiment',
                            'report', '--registration', str(path)], cwd=ROOT, check=True,
                           stdout=log,stderr=subprocess.STDOUT)
    except Exception as error:
        write_json(out/'status.json', dict(status='failed', error=str(error)))
        run_registry.update_status(reg['run_id'], 'failed')
        from .result_retention import final_only,finalize,log_tail
        if final_only(reg['spec']):
            from .html_report import build_failure_report
            build_failure_report(path,reg['report_output'],error,log_tail(out))
            final=finalize(path)
            raise RuntimeError('Experiment failed; final HTML saved: '+str(final)) from error
        raise
    write_json(out/'status.json', dict(status='complete', full_range=reg['full_range']))
    from .result_retention import final_only,finalize
    if final_only(reg['spec']):return finalize(path)
    run_registry.update_status(reg['run_id'], 'completed')
    return Path(reg['report_output'])/'report.html'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'run', 'report', 'compare', 'worker','strategies','validate-config','finalize-results','inspect-report'])
    p.add_argument('--config', type=Path, default=ROOT/'config/experiments/default.json')
    p.add_argument('--run-id')
    p.add_argument('--registration', type=Path)
    p.add_argument('--report-output', type=Path, help='New report directory; report action only, never overwrites')
    p.add_argument('--case-id')
    p.add_argument('--html',type=Path,action='append')
    p.add_argument('--max-segments', type=int)
    p.add_argument('--max-files', type=int)
    args = p.parse_args()
    if args.report_output is not None and args.action != 'report':
        p.error('--report-output is only valid with report')
    if any(v is not None and v <= 0 for v in (args.max_segments, args.max_files)):
        p.error('Partial-run limits must be positive')
    if args.action=='inspect-report':
        if not args.html or len(args.html)!=1:p.error('inspect-report requires exactly one --html')
        from .html_report import read_report
        payload=read_report(args.html[0])
        print(json.dumps({k:payload[k] for k in ('run_id','status','stage','full_range','results')},ensure_ascii=False,indent=2))
    elif args.action=='finalize-results':
        if args.registration is None:p.error('finalize-results requires --registration')
        from .result_retention import finalize
        print(finalize(args.registration))
    elif args.action == 'strategies':
        from ..market_making.strategies.registry import available_strategies
        print(json.dumps(available_strategies(),ensure_ascii=False,indent=2))
    elif args.action == 'validate-config':
        spec=load_spec(args.config)
        print(json.dumps(dict(name=spec['name'],window=spec['window'],symbols=spec['symbols'],
             engine=spec['engine'],profiles=spec.get('_profile_inputs',{}),
             cases=[dict(symbol=s,id=c['id'],plugin=c.get('plugin'),
                         config=effective_config(spec,spec['datasets'][s],c))
                    for s in spec['symbols'] for c in spec['strategies']]),ensure_ascii=False,indent=2))
    elif args.action == 'run' and args.registration is not None:
        print(run(args.registration), flush=True)
    elif args.action == 'worker':
        worker(args.registration, args.case_id)
    elif args.action == 'report':
        from .experiment_report import report
        report(args.registration, args.report_output)
    elif args.action == 'compare':
        from .comparison import compare_reports
        print(json.dumps(compare_reports(args.html or []),ensure_ascii=False,indent=2))
    else:
        path = prepare(args.config, args.run_id, args.max_segments, args.max_files)
        print(path, flush=True)
        if args.action == 'run':
            print(run(path),flush=True)


if __name__ == '__main__':
    main()
