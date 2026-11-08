"""Small artifact utilities; no strategy, execution or research imports."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()


def write(path, data, *, default=str):
    path = Path(path)
    temporary = Path(str(path) + '.tmp')
    temporary.write_text(json.dumps(data, default=default, ensure_ascii=False,
                                    allow_nan=False, indent=2) + '\n')
    temporary.replace(path)


def source_identity(root, folders=('app', 'src')):
    """Freeze all runtime sources, including features and canonical metrics.

    Only hashes are retained. Tests, sidecars and symlinks are not runtime code.
    """
    root = Path(root)
    return {str(p.relative_to(root)): digest(p) for folder in folders
            for p in sorted((root / folder).rglob('*.py'))
            if not p.is_symlink() and not p.name.startswith('test_')
            and 'tests' not in p.relative_to(root).parts
            and not any(part.startswith('._') for part in p.relative_to(root).parts)}
