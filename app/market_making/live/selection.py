"""Public strategy profile selection shared by offline replay and online entry."""
import json
from pathlib import Path
from ..strategies.registry import available_strategies, normalize_case, bind_symbol, definition_for


def load_selection(path, symbol):
    if not isinstance(symbol,str) or not symbol:
        raise ValueError('Explicit strategy symbol required')
    plugin=json.loads(Path(path).read_text())
    if not isinstance(plugin,dict) or plugin.get('type') not in {item['type'] for item in available_strategies()}:
        raise ValueError('Live entry requires a public AS/NH/HJB strategy profile')
    case=normalize_case(dict(id='selected',plugin=plugin))
    return bind_symbol(case,symbol)


def validate_source(case, source):
    if definition_for(case).requires_depth and source!='depth':
        raise ValueError('NH requires full depth input; BBO cannot substitute queue depth')
