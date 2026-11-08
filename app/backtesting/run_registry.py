"""本机实验注册摘要，首次运行自动创建于 data/experiments/registry.json。

仅记录当前运行身份、状态与最终报告；不依赖历史实验或纳入版本控制。
"""
import json
import sys
from pathlib import Path

REGISTRY = Path(__file__).resolve().parents[2] / 'data' / 'experiments' / 'registry.json'
STATUS = {'prepared', 'completed', 'frozen', 'superseded', 'failed'}


def load(path=REGISTRY):
    try:
        raw = Path(path).read_text()
    except FileNotFoundError:
        return dict(schema_version=1, entries=[])
    reg = json.loads(raw)
    if reg.get('schema_version') != 1 or not isinstance(reg.get('entries'), list):
        raise ValueError('Unsupported experiments registry: ' + str(path))
    return reg


def write(reg, path=REGISTRY):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(reg, ensure_ascii=False, indent=2) + '\n')


def append(entry, path=REGISTRY):
    if entry.get('status') not in STATUS:
        raise ValueError('Unknown registry status: ' + str(entry.get('status')))
    reg = load(path)
    if any(e['id'] == entry['id'] for e in reg['entries']):
        raise ValueError('Registry identity conflict: ' + entry['id'])
    reg['entries'].append(entry)
    write(reg, path)


def update_status(run_id, status, path=REGISTRY):
    if status not in STATUS:
        raise ValueError('Unknown registry status: ' + status)
    reg = load(path)
    for entry in reg['entries']:
        if entry['id'] == run_id and entry.get('kind') == 'run':
            entry['status'] = status
            write(reg, path)
            return True
    print(f'run_registry: {run_id} not registered; status {status} not recorded', file=sys.stderr)
    return False


def record_final(run_id,html_path,html_sha256,status,summary,cleanup,path=REGISTRY,**details):
    if status not in ('completed','failed'):raise ValueError('Invalid final run status')
    reg=load(path)
    for e in reg['entries']:
        if e['id']==run_id and e.get('kind')=='run':
            e.update(status=status,artifacts=[str(html_path)],html_report=str(html_path),
                     html_sha256=html_sha256,result_summary=summary,
                     retention=dict(mode='final_only',workspace=cleanup,**details))
            write(reg,path);return
    raise ValueError('Final report has no registered run: '+run_id)
