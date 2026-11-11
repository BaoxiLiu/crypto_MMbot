"""Closed, declarative composition root. Profiles cannot import arbitrary code."""
from copy import deepcopy
from dataclasses import dataclass
from importlib import import_module


@dataclass(frozen=True)
class StrategyDefinition:
    module: str
    implementation: str
    versions: tuple
    description: str
    per_symbol_models: bool = False
    requires_depth: bool = False
    supports_prewindow: bool = True

    def load(self):
        return import_module('.'+self.module, __package__)


# One reviewed entry per mathematical family; parameter variants add no entry.
DEFINITIONS = {
    'glft': StrategyDefinition('glft_strategy', 'GLFTStrategy', (1,),
        'GLFT asymptotic single-level quotes with causal arrival calibration'),
    'paper_native_as': StrategyDefinition('as_strategy', 'ASStrategy', (1,),
        'Project-owned AS; optional S1/SC0 policy'),
    'nh_quote_value': StrategyDefinition('nh_strategy', 'NHStrategy', (1,),
        'NH causal alpha and quote value', per_symbol_models=True, requires_depth=True),
    'fb_as_hjb': StrategyDefinition('hjb_strategy', 'HJBStrategy', (2,),
        'Fixed/adaptive inventory-grid HJB with optional capital budget'),
}
SOURCES = ('app/market_making/strategies/registry.py',
           'app/market_making/strategies/base.py')


def available_strategies():
    return [dict(type=kind, versions=list(d.versions), description=d.description)
            for kind, d in DEFINITIONS.items()]


def reject_retired_keys(case):
    retired = {'adaptive_as', 'prediction', 'single_long', 'paper_reversal',
               'quote_placement', 'queue_management'} & set(case)
    if retired:
        raise ValueError('Strategy overlays are retired: ' + ', '.join(sorted(retired)))
    if 'mask' in case and (type(case['mask']) is not int or case['mask'] != 0):
        raise ValueError('Legacy strategy masks are retired')


def validate_plugin_header(plugin):
    if not isinstance(plugin, dict) or set(plugin) != {'type', 'version', 'params'} or not isinstance(plugin['params'], dict):
        raise ValueError('Plugin requires type/version/params')
    kind = plugin['type']
    if not isinstance(kind, str) or kind not in DEFINITIONS or type(plugin['version']) is not int or plugin['version'] not in DEFINITIONS[kind].versions:
        raise ValueError('Unknown strategy plugin/version')


def definition_for(case):
    reject_retired_keys(case)
    validate_plugin_header(case.get('plugin'))
    return DEFINITIONS[case['plugin']['type']]


def normalize_case(case):
    definition = definition_for(case)
    result = deepcopy(case)
    result['mask'] = 0
    params = result['plugin']['params']
    validate = definition.load().validate_settings
    if definition.per_symbol_models:
        if set(params) != {'models'} or not isinstance(params['models'], dict) or not params['models']:
            raise ValueError('Per-symbol models required')
        for settings in params['models'].values():
            validate(settings)
    else:
        validate(params)
    return result


def plugin_sources(case):
    definition = definition_for(case)
    return tuple(dict.fromkeys((*SOURCES, *definition.load().SOURCES)))


def bind_symbol(case, symbol, *, symbols=None, manifest_sha256=None):
    """Bind private parameters; fitted-input checks belong to the strategy module."""
    definition = definition_for(case)
    bound = deepcopy(case)
    bound['symbol'] = symbol
    if definition.per_symbol_models:
        models = bound['plugin']['params']['models']
        if symbol not in models:
            raise ValueError('Strategy model missing for '+symbol)
        if symbols is not None and set(models) != set(symbols):
            raise ValueError('Strategy models must cover exactly the campaign symbols')
        params = models[symbol]
        definition.load().validate_model_identity(params, symbol, manifest_sha256)
        bound['plugin']['params'] = params
    return bound


def factory_for(case):
    """Each construction owns a deep copy, even when one factory is reused."""
    definition = definition_for(case)
    params = deepcopy(case['plugin']['params'])
    module = definition.load()
    module.validate_settings(params)
    implementation = getattr(module, definition.implementation)
    symbol = case.get('symbol')
    return lambda rules, config, factors: implementation(
        rules, config, factors, deepcopy(params), symbol)
