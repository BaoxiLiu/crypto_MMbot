"""Experiment configuration: resolution, validation and effective strategy parameters.

This module does not launch workers or generate reports.
"""
from datetime import datetime, timezone
from decimal import Decimal as D
import json
import os
from pathlib import Path
import re

from . import dataset_registry
from .experiment_profiles import expand_profiles
from ..market_making.strategies.registry import normalize_case
from ..market_making.fusion.config import FusionConfig
from ..market_making.fusion.execution import ExecutionProfile

ROOT = Path(__file__).resolve().parents[2]
REQUIRED = {'pnl_curve', 'daily_volume', 'total_volume', 'markouts', 'fills', 'effective_config', 'input_identity', 'validation'}


def resolve(value):
    value = os.fspath(value)
    if value.startswith('$QUANT_DATA_ROOT/'):
        root = os.environ.get('QUANT_DATA_ROOT')
        if not root:
            raise ValueError('QUANT_DATA_ROOT environment variable not set for: ' + value)
        return Path(root) / value[len('$QUANT_DATA_ROOT/'):]
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def identifier(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,99}', value):
        raise ValueError('Invalid run/case identifier')
    return value


def load_spec(path):
    spec = json.loads(Path(path).read_text())
    allowed = {'schema_version','name','window','symbols','datasets','rules','engine','execution','costs','loss_stops','strategies','observations','outputs','resources','runtime','profiles','account','strategy_parameters','stage','fee_scenarios'}
    if not set(spec) <= allowed:
        raise ValueError('Unknown project config keys: ' + str(set(spec)-allowed))
    spec=expand_profiles(spec,ROOT)
    if 'fee_scenarios' in spec:
        fees=spec['fee_scenarios']
        if not isinstance(fees,list) or not fees:raise ValueError('Nonempty fee scenarios required')
        if any(set(f)!={'id','maker_fee'} or not D(str(f['maker_fee'])).is_finite() for f in fees):
            raise ValueError('Explicit finite maker-fee scenarios required')
        if len({identifier(f['id']) for f in fees})!=len(fees):raise ValueError('Duplicate fee scenario')
        from copy import deepcopy
        expanded=[]
        for c in spec['strategies']:
            if 'fee_scenario' in c:raise ValueError('Fee scenarios belong to the experiment')
            for f in fees:
                row=deepcopy(c);row['id']=c['id']+'__'+f['id'];row['fee_scenario']=deepcopy(f);expanded.append(row)
        spec['strategies']=expanded
    spec['strategies']=[normalize_case(c) for c in spec['strategies']]
    if 'account' in spec:
        if not isinstance(spec['account'],dict) or set(spec['account'])-{'initial_cash','max_notional'}:
            raise ValueError('Account profile supports initial_cash/max_notional only')
        for value in spec['account'].values():
            if not D(str(value)).is_finite() or D(str(value))<=0:raise ValueError('Positive account limits required')
    if 'strategy_parameters' in spec and set(spec['strategy_parameters'])!=set(spec['symbols']):
        raise ValueError('Strategy parameter baselines must cover exactly the requested symbols')
    local = Path(__file__).resolve().parents[2] / 'paths.local.json'
    if local.exists():
        overrides = json.loads(local.read_text())
        if set(overrides) - {'runtime'}:
            raise ValueError('paths.local.json may only define runtime: ' + str(set(overrides)))
        if 'runtime' in overrides and 'runtime' not in spec:
            spec['runtime'] = overrides['runtime']
    if 'runtime' not in spec:
        raise ValueError('runtime missing: define it in the project config or in paths.local.json (gitignored, machine-specific)')
    if spec['schema_version'] != 2 or spec['window']['timezone'] != 'UTC':
        raise ValueError('Experiment schema_version=2 / UTC required')
    from datetime import date
    start, end = (datetime.combine(date.fromisoformat(spec['window'][k]), datetime.min.time(), timezone.utc) for k in ('start', 'end_exclusive'))
    if start >= end or any(t.time().isoformat() != '00:00:00' for t in (start, end)):
        raise ValueError('Nonempty whole UTC dates required')
    engine = spec['engine']
    from ..market_making.fusion.queue_models import QUEUE_MODELS
    if engine['name']!='hftbacktest' or engine['queue'] not in QUEUE_MODELS or engine['exchange']!='partial_fill' or engine['version'] != '2.4.4+makerfix2':
        raise ValueError('Unsupported engine/queue/exchange; no silent fallback')
    profile = ExecutionProfile.from_dict(spec['execution'])
    if profile.trigger != 'event' or profile.wire_prepare_ns or profile.trade_feed_ns or profile.mark_feed_ns:
        raise ValueError('Publication-clock event profile required')
    obs = spec['observations']
    if 'pnl_attribution' in obs and type(obs['pnl_attribution']) is not bool:
        raise ValueError('pnl_attribution must be boolean')
    if obs.get('pnl_attribution') and (not obs.get('order_placements') or obs.get('order_lifecycle') or obs.get('order_context')):
        raise ValueError('PnL attribution requires standard all-order observations')
    if 'order_placements' in obs and type(obs['order_placements']) is not bool:
        raise ValueError('order_placements must be an explicit boolean')
    if obs.get('order_placements') and obs.get('order_context'):
        raise ValueError('Select either all placements or filled-order context sampling')
    if 'order_lifecycle' in obs and type(obs['order_lifecycle']) is not bool:
        raise ValueError('order_lifecycle must be a boolean')
    if obs.get('order_lifecycle') and not obs.get('order_placements'):
        raise ValueError('Lifecycle observations require all-order placements')
    if obs['timezone'] != 'UTC' or obs['markout_anchor'] != 'local_fill_receipt' or obs['markout_reference'] != 'first_observed_mid_at_or_after':
        raise ValueError('Unsupported metric definition')
    hs = obs['markout_horizons_ms']
    if not hs or len(set(hs)) != len(hs) or any(type(h) is not int or h <= 0 for h in hs):
        raise ValueError('Invalid horizons')
    if type(obs['max_markout_lag_ms']) is not int or obs['max_markout_lag_ms'] < 0 or type(obs['retain_midpoints']) is not bool:
        raise ValueError('Invalid observation settings')
    if set(spec['outputs']['required']) != REQUIRED:
        raise ValueError('Standard outputs must be requested')
    if spec['outputs'].get('retention','all') not in ('all','final_only'):
        raise ValueError('Unknown output retention policy')
    if spec['outputs'].get('format','html')!='html':raise ValueError('Final output format must be html')
    if spec.get('stage','custom') not in ('quick','full','custom'):raise ValueError('Unknown experiment stage')
    from .campaign_standard import validate_standard
    validate_standard(spec)
    if type(spec['resources']['workers']) is not int or spec['resources']['workers'] < 1:
        raise ValueError('Invalid worker count')
    for value in spec['runtime'].values():
        if not Path(value).is_file() or not os.access(value, os.X_OK):
            raise ValueError('Configured Python executable unavailable: ' + value)
    if not spec['symbols'] or len(set(spec['symbols'])) != len(spec['symbols']):
        raise ValueError('Unique symbols required')
    ids = [identifier(s['id']) for s in spec['strategies']]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('Unique strategy identifiers required')
    for s in spec['strategies']:
        if s['mask'] != 0:
            raise ValueError('Legacy strategy masks are retired')
        if set(s) - {'id', 'mask', 'overrides', 'plugin', 'fee_scenario'}:
            raise ValueError('Unknown strategy key')
    dataset_registry.check_experiment(spec, resolve)
    check_explicit_strategies(spec)
    return spec


def parameter_file(spec,instrument,symbol=None):
    if 'strategy_config' in instrument:
        raise ValueError('strategy_config is retired; use strategy_parameters')
    if symbol is None:
        matches=[s for s,v in spec['datasets'].items() if v==instrument]
        if len(matches)!=1:raise ValueError('Ambiguous strategy parameter symbol')
        symbol=matches[0]
    path=spec.get('strategy_parameters',{}).get(symbol)
    if path is None:raise ValueError('Missing strategy parameter baseline for '+symbol)
    return resolve(path)


def check_explicit_strategies(spec):
    """策略参数基线必须显式列出全部 FusionConfig/ASConfig 字段(合并 overrides 后),
    禁止 dataclass 默认值悄悄生效(AGENTS.md 禁隐含默认)。"""
    from dataclasses import fields
    from ..market_making.as_math import ASConfig
    top = {f.name for f in fields(FusionConfig)} - {'as_config'}
    inner = {f.name for f in fields(ASConfig)}
    problems = []
    for symbol in spec['symbols']:
        baseline = json.loads(parameter_file(spec,spec['datasets'][symbol],symbol).read_text())
        for case in spec['strategies']:
            merged = dict(baseline)
            merged_as = dict(baseline.get('as_config', {}))
            for key, value in case.get('overrides', {}).items():
                if key == 'as_config':
                    merged_as.update(value)
                else:
                    merged[key] = value
            missing = sorted((top - set(merged)) | (inner - set(merged_as)))
            if missing:
                problems.append(f"{symbol}/{case['id']}: 策略参数基线缺显式字段 {missing}")
    if problems:
        raise ValueError('Explicit strategy parameters required: ' + '; '.join(problems))
    return problems


def effective_config(spec, instrument, case):
    data = json.loads(parameter_file(spec,instrument).read_text())
    overrides = case.get('overrides', {})
    for key, value in overrides.items():
        if key == 'as_config':
            data[key].update(value)
        else:
            data[key] = value
    for key,value in spec.get('account',{}).items():
        if key in overrides and D(str(overrides[key]))!=D(str(value)):
            raise ValueError('Case-specific account risk would confound the comparison')
        data[key]=value
    fee=case.get('fee_scenario')
    if fee is not None and fee not in spec.get('fee_scenarios',[]):raise ValueError('Unregistered fee scenario')
    maker_fee=fee['maker_fee'] if fee is not None else spec['costs']['maker_fee']
    # Only experiment-declared scenarios can change fees; strategy overrides cannot.
    if 'maker_fee' in overrides.get('as_config', {}) and D(str(overrides['as_config']['maker_fee'])) != D(maker_fee):
        raise ValueError('Case-specific fees would confound the comparison')
    data['as_config']['maker_fee'] = maker_fee
    if set(spec['loss_stops']) != {'daily_loss', 'drawdown_limit'}:
        raise ValueError('Explicit daily and drawdown settings required')
    for key, value in spec['loss_stops'].items():
        data[key] = '1E+30' if value is None else str(value)
    return FusionConfig.from_dict(data).json()


