"""Read-only storage/health summary; open gzip files are not certified complete."""
import argparse
import json
from pathlib import Path
import shutil


def summarize(root):
    root = Path(root)
    health_path = root / 'health.json'
    health = json.loads(health_path.read_text()) if health_path.exists() else None
    groups = {}
    for p in root.rglob('*.gz*'):
        if not (p.name.endswith('.gz') or p.name.endswith('.gz.partial')):
            continue
        channel = p.name.split('_20')[0]
        group = groups.setdefault(channel, {'bytes':0,'closed_rows':0,'open_files':0})
        group['bytes'] += p.stat().st_size
        if p.name.endswith('.partial'):
            group['open_files'] += 1
        else:
            manifest = Path(str(p)+'.manifest.json')
            if manifest.exists():
                group['closed_rows'] += json.loads(manifest.read_text())['count']
    return {'root':str(root),'free_bytes':shutil.disk_usage(root).free,
            'channels':groups,'health':health}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('roots', nargs='+')
    args = parser.parse_args()
    print(json.dumps([summarize(r) for r in args.roots],ensure_ascii=False,indent=2))
