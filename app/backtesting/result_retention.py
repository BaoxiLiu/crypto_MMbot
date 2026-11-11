"""Publish verified final HTML, then remove only this run's owned workspace."""
from html import escape
import json
from pathlib import Path
import shutil
from urllib.parse import quote

from .html_report import read_report,sha
from . import run_registry

OWNER='.managed-run.json'


def cleanup_error(function, path, exc_info):
    # macOS may remove an AppleDouble sidecar automatically with its data file.
    error = exc_info[1]
    if isinstance(error, FileNotFoundError) and Path(path).name.startswith('._'):
        return
    raise error



def final_only(spec):return spec['outputs'].get('retention','all')=='final_only'


def create_workspace(workspace,final_directory,run_id):
    w=Path(workspace);f=Path(final_directory)
    if w.is_symlink() or w.parent.is_symlink() or w.parent.name!='.work' or w.name!=run_id:
        raise ValueError('Final-only workspace must be a dedicated .work/run_id directory')
    w=w.resolve();f=f.resolve()
    if f==w or f.is_relative_to(w) or w.is_relative_to(f):
        raise ValueError('Final report and workspace must be separate')
    w.mkdir(parents=True,exist_ok=False)
    owner=dict(schema_version=1,run_id=run_id,workspace=str(w),final_directory=str(f))
    (w/OWNER).write_text(json.dumps(owner),encoding='utf-8')


def check_owner(reg):
    if reg['kind']!='new_run':raise ValueError('Only current new_run workspaces can be finalized')
    if not final_only(reg['spec']):raise ValueError('Run did not request final-only cleanup')
    w=Path(reg['run_output']);f=Path(reg['final_report_output'])
    if w.is_symlink() or w.parent.is_symlink() or w.parent.name!='.work' or w.name!=reg['run_id']:
        raise ValueError('Refusing cleanup outside a dedicated workspace')
    owner=json.loads((w/OWNER).read_text())
    expected=dict(schema_version=1,run_id=reg['run_id'],workspace=str(w.resolve()),final_directory=str(f.resolve()))
    if owner!=expected:raise ValueError('Workspace ownership mismatch; cleanup refused')
    if f.resolve().is_relative_to(w.resolve()) or w.resolve().is_relative_to(f.resolve()):
        raise ValueError('Final report overlaps temporary workspace')
    if not Path(reg['report_output']).resolve().is_relative_to(w.resolve()):
        raise ValueError('Report staging directory is not owned by this run')
    for c in reg['cases']:
        if not Path(c['output']).resolve().is_relative_to(w.resolve()):
            raise ValueError('Case output is outside owned workspace')
    return w,f


def update_index(root,registry_path=run_registry.REGISTRY):
    root=Path(root).resolve();items=[]
    for e in run_registry.load(registry_path)['entries']:
        p=Path(e.get('html_report',''))
        if not e.get('html_report') or not p.is_file() or not p.resolve().is_relative_to(root):continue
        meta=e.get('result_summary',{})
        items.append((e,p,meta))
    items.reverse();rows=[]
    for e,p,m in items:
        href=quote(str(p.resolve().relative_to(root)))
        values=[m.get('name',e['id']),m.get('stage','custom'),m.get('window',''),
                ', '.join(m.get('symbols',[])),', '.join(m.get('strategies',[])),
                e.get('status',''),m.get('scope','')]
        rows.append('<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in values)+
                    '<td><a href="'+href+'">打开报告</a></td></tr>')
    text='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>回测实验 · 结果索引</title><style>
body{font:15px/1.7 -apple-system,"PingFang SC",sans-serif;color:#182c3f;background:#f5f8fb;margin:32px}h1{color:#123b5c}table{border-collapse:collapse;background:white;width:100%}td,th{padding:12px;border-bottom:1px solid #d8e3ee;text-align:left}a{color:#006d77}input{font:inherit;padding:10px;width:min(100%,560px);margin-bottom:20px}.scroll{overflow:auto}
</style><h1>回测实验记录</h1><p>每一行对应一次独立运行。完整/局部与运行状态分别展示，不把运行完成当作策略达标。</p><input aria-label="筛选实验" placeholder="搜索实验、币种、策略或日期" id="search"><div class="scroll"><table><thead><tr>'''+''.join('<th>'+v+'</th>' for v in ['实验','阶段','UTC区间','币种','策略','状态','覆盖','报告'])+'</tr></thead><tbody>'+''.join(rows)+'''</tbody></table></div><script>document.getElementById('search').addEventListener('input',e=>{const q=e.target.value.toLowerCase();document.querySelectorAll('tbody tr').forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(q));});</script></html>'''
    target=root/'index.html';tmp=target.with_suffix('.html.tmp');tmp.write_text(text,encoding='utf-8');tmp.replace(target)
    return target


def finalize(registration,registry_path=run_registry.REGISTRY):
    reg=json.loads(Path(registration).read_text());w,final=check_owner(reg)
    source=Path(reg['report_output'])/'report.html';payload=read_report(source)
    if payload['run_id']!=reg['run_id'] or payload['spec']!=reg['spec'] or payload['full_range']!=reg['full_range']:
        raise ValueError('Final report does not describe this registered run')
    raw=source.read_bytes();expected=sha(raw)
    if final.exists():
        entries={p.name for p in final.iterdir()
                 if not (p.name=='._report.html' and p.is_file() and not p.is_symlink())}
        if final.is_symlink() or (final/'report.html').is_symlink() or entries!={'report.html'} or sha((final/'report.html').read_bytes())!=expected:
            raise FileExistsError('Final report identity already exists; will not overwrite')
    else:
        final.mkdir(parents=True,exist_ok=False)
        tmp=final/'report.html.tmp';tmp.write_bytes(raw)
        read_report(tmp);tmp.replace(final/'report.html')
    target=final/'report.html'
    if sha(target.read_bytes())!=expected:raise ValueError('Published HTML copy mismatch')
    # A completed final report and registry reference must exist before deletion.
    summary=dict(name=reg['spec']['name'],stage=payload.get('stage','custom'),
                 window=reg['spec']['window']['start']+' → '+reg['spec']['window']['end_exclusive'],
                 symbols=reg['spec']['symbols'],strategies=list(dict.fromkeys(c['strategy'] for c in reg['cases'])),
                 scope='完整请求区间' if reg['full_range'] else '局部验证',
                 totals=payload['results'].get('totals',[]))
    run_registry.record_final(reg['run_id'],target,expected,payload['status'],summary,'pending',registry_path)
    if payload['status']!='completed':
        # Failed HTML is diagnostic only: it does not contain successful arms'
        # fills/curves. Preserve the entire owned workspace for repair/resume.
        run_registry.record_final(reg['run_id'],target,expected,payload['status'],summary,'retained',registry_path,workspace_path=str(w))
        update_index(final.parent,registry_path)
        return target
    # rmtree does not follow directory symlinks; the root and ownership are checked again.
    check_owner(reg)
    released=sum(p.stat().st_size for p in w.rglob('*') if p.is_file() and not p.is_symlink())
    shutil.rmtree(w,onerror=cleanup_error)
    run_registry.record_final(reg['run_id'],target,expected,payload['status'],summary,'removed',registry_path,
                              released_bytes=released)
    update_index(final.parent,registry_path)
    return target


def log_tail(workspace):
    pieces=[]
    root=Path(workspace)
    statuses=root/'process-results.json'
    failed=[]
    if statuses.exists():
        failed=[root/(row['case']+'.log') for row in json.loads(statuses.read_text()) if row['exit']!=0]
    logs=[p for p in sorted(root.glob('*.log')) if not p.name.startswith('._')]
    selected=list(dict.fromkeys(failed+logs[-10:]))[:20]
    for path in selected:
        if not path.is_file():continue
        with path.open('rb') as f:
            f.seek(0,2);f.seek(max(0,f.tell()-4096));tail=f.read().decode('utf-8',errors='replace')
        pieces.append(path.name+'\n'+tail)
    return '\n\n'.join(pieces)[:65536]
