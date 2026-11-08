"""Unified metadata queries; no implicit numerical-backend or trading integration."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PATH = Path(__file__).with_name('catalog.json')


def load_catalog(path=PATH):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if data.get('schema_version') != 1:
        raise ValueError('Unsupported factor catalog schema')
    ids = [r['id'] for r in data['factors']]
    if len(ids) != len(set(ids)):
        raise ValueError('Ambiguous factor IDs')
    return data


def lookup(factor_id, *, catalog=None):
    """Require namespaced IDs; resolve only explicitly proved reuse aliases."""
    data = load_catalog() if catalog is None else catalog
    for row in data['factors']:
        if row['id'] == factor_id:
            return row
    for alias in data['known_reuse']:
        if alias['alias'] == factor_id:
            return lookup(alias['target'], catalog=data)
    raise KeyError(f'Unknown namespaced factor ID: {factor_id}')


def select(*, category=None, collection=None, include_candidates=False,
           include_blocked=False, catalog=None):
    """Select metadata only; presence of source code is not runtime certification."""
    data = load_catalog() if catalog is None else catalog
    if category is not None and category not in data['categories']:
        raise ValueError(f'Unknown category: {category}')
    if collection is not None and collection not in data['collections']:
        raise ValueError(f'Unknown collection: {collection}')
    return [r for r in data['factors']
            if (category is None or r['category'] == category)
            and (collection is None or r['collection'] == collection)
            and (include_candidates or r['collection'] != 'cicc')
            and (include_blocked or r['status'] != 'blocked_missing_dependency')]


def verify_sources(*, catalog=None, root=ROOT):
    """Fail closed on missing/changed definitions; never rebuild during a query."""
    data = load_catalog() if catalog is None else catalog
    mismatches = []
    for rel, expected in data['source_sha256'].items():
        path = Path(root)/rel
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            mismatches.append(rel)
    if mismatches:
        raise ValueError('Factor catalog sources changed or missing: '+', '.join(mismatches))
    for rel in data['missing_sources']:
        if (Path(root)/rel).exists():
            raise ValueError('Previously missing dependency appeared; rebuild catalog: '+rel)
    return True
