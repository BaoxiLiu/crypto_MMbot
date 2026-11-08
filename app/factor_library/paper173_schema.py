"""Paper173 names/order; runtime does not read HTML or historical report CSVs."""
import hashlib
import json
from pathlib import Path

SCHEMA_PATH=Path(__file__).with_suffix('.json')
SCHEMA_SOURCES=(Path(__file__),SCHEMA_PATH)


def load_names(path=SCHEMA_PATH):
    data=json.loads(Path(path).read_text(encoding='utf-8'))
    names=tuple(data['feature_names'])
    if data.get('schema_version')!=1 or data.get('feature_count')!=173 or len(names)!=173 or len(set(names))!=173:
        raise ValueError('Invalid paper173 schema/count/duplicates')
    families=('ret_autocov','ret_sum','trade_intensity','amplitude','ret_vwap','max_size',
              'avg_size','buy_count','sell_count','total_buy','total_sell')
    expected={f'{name}_{scale}_w{lag}' for name in families
              for scale in ('100ms','1s','5s','30s','300s') for lag in range(3)}
    expected.update(('age','totb_mean','ob_bid_liq','ob_ask_liq','ob_half','ob_other_half','stdev_100','stdev_500'))
    digest=hashlib.sha256('\n'.join(names).encode()).hexdigest()
    if set(names)!=expected or digest!=data.get('ordered_names_sha256'):
        raise ValueError('Paper173 members/order fingerprint mismatch')
    return names


FEATURE_NAMES=load_names()
