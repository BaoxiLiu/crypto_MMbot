"""Final HTML must remain usable after deleting only its owned working directory."""
import base64
import json
from pathlib import Path

from PIL import Image
import pytest

from app.backtesting.html_report import CHARTS,read_report,sha,write_report,build_failure_report,build_report
from app.backtesting.result_retention import OWNER,create_workspace,finalize


def setup_run(tmp_path):
    work=tmp_path/'runs'/'.work'/'case-a';final=tmp_path/'reports'/'case-a'
    create_workspace(work,final,'case-a')
    spec=dict(name='测试 </script><script>alert(1)</script>',stage='quick',
              window=dict(start='2026-08-01',end_exclusive='2026-08-04'),symbols=['DOGEUSDC'],
              outputs=dict(retention='final_only'))
    reg=dict(kind='new_run',run_id='case-a',run_output=str(work),final_report_output=str(final),
             report_output=str(work/'report'),full_range=False,spec=spec,
             cases=[dict(id='base',strategy='base',output=str(work/'base'))])
    path=work/'registration.json';path.write_text(json.dumps(reg))
    registry=tmp_path/'registry.json'
    registry.write_text(json.dumps(dict(schema_version=1,entries=[dict(id='case-a',kind='run',status='prepared')])))
    return path,reg,registry


def complete_report(reg,tmp_path):
    png=tmp_path/'chart.png';Image.new('RGB',(4,4),'white').save(png)
    raw=png.read_bytes();images={kind:base64.b64encode(raw).decode() for kind,_ in CHARTS}
    payload=dict(schema_version=1,status='completed',run_id=reg['run_id'],stage='quick',
                 full_range=reg['full_range'],spec=reg['spec'],retention='final_only',
                 results=dict(totals=[dict(symbol='DOGEUSDC',strategy='base',pnl='0',total_notional=None)]),
                 chart_sha256={kind:sha(raw) for kind,_ in CHARTS})
    return write_report(Path(reg['report_output'])/'report.html',payload,images,{k:'数据结论' for k in images})


def test_html_embeds_charts_values_and_escapes_script_injection(tmp_path):
    _,reg,_=setup_run(tmp_path);p=complete_report(reg,tmp_path)
    assert read_report(p)['spec']==reg['spec']
    html=p.read_text()
    assert '<script>alert(1)</script>' not in html
    assert '<td class="num pos">' not in html
    assert '0.0000' in html and '—' in html
    assert html.count('data-chart=')==11
    assert '局部验证' in html and 'quick' in html
    assert 'https://' not in html and 'src="data:image/png;base64,' in html


@pytest.mark.parametrize('target',['payload','image','external'])
def test_corrupt_html_is_rejected(tmp_path,target):
    _,reg,_=setup_run(tmp_path);p=complete_report(reg,tmp_path)
    html=p.read_text()
    if target=='payload':html=html.replace('"schema_version":1','"schema_version":9')
    elif target=='image':html=html.replace('base64,iVBOR','base64,aVBOR',1)
    else:html=html.replace('</head>','<script src="https://example.com/a.js"></script></head>')
    p.write_text(html)
    with pytest.raises(ValueError):read_report(p)


def test_publish_then_cleanup_keeps_single_html_and_shared_inputs(tmp_path):
    path,reg,registry=setup_run(tmp_path);complete_report(reg,tmp_path)
    shared=tmp_path/'market.npy';shared.write_bytes(b'original source')
    work=Path(reg['run_output']);(work/'temporary.bin').write_bytes(b'x'*10000)
    (work/'shared-link').symlink_to(shared)
    result=finalize(path,registry)
    assert not work.exists() and shared.read_bytes()==b'original source'
    assert list(result.parent.iterdir())==[result]
    assert read_report(result)['results']['totals'][0]['pnl']=='0'
    entry=json.loads(registry.read_text())['entries'][0]
    assert entry['status']=='completed' and entry['retention']['workspace']=='removed'
    assert entry['artifacts']==[str(result)] and entry['retention']['released_bytes']>=10000
    index=(result.parent.parent/'index.html').read_text()
    assert 'case-a/report.html' in index and '局部验证' in index


@pytest.mark.parametrize('bad',['marker','case','report','identity','missing-registry','final-conflict'])
def test_cleanup_refuses_unsafe_or_unpublished_results(tmp_path,bad):
    path,reg,registry=setup_run(tmp_path);p=complete_report(reg,tmp_path)
    work=Path(reg['run_output']);sentinel=work/'must-remain';sentinel.write_text('keep')
    if bad=='marker':(work/OWNER).write_text('{}')
    elif bad=='case':reg['cases'][0]['output']=str(tmp_path/'unowned');path.write_text(json.dumps(reg))
    elif bad=='report':p.write_text('invalid')
    elif bad=='identity':reg['spec']['name']='different';path.write_text(json.dumps(reg))
    elif bad=='missing-registry':registry.write_text('{"schema_version":1,"entries":[]}')
    else:
        final=Path(reg['final_report_output']);final.mkdir(parents=True);(final/'keep').write_text('keep')
    with pytest.raises((ValueError,FileExistsError)):finalize(path,registry)
    assert sentinel.read_text()=='keep'


def test_failure_retains_bounded_error_html_and_cleans_work(tmp_path):
    path,reg,registry=setup_run(tmp_path)
    build_failure_report(path,reg['report_output'],'worker crashed','x'*100000)
    result=finalize(path,registry);payload=read_report(result)
    assert payload['status']=='failed' and len(payload['log_tail'])==65536
    assert payload['results']['totals']==[] and not Path(reg['run_output']).exists()
    assert json.loads(registry.read_text())['entries'][0]['status']=='failed'


def test_retired_export_is_rejected_without_deleting_original_cases(tmp_path):
    path,reg,registry=setup_run(tmp_path);reg['kind']='existing_export'
    source=tmp_path/'frozen-case';source.mkdir();(source/'fills').write_text('frozen')
    reg['cases'][0]['output']=str(source);path.write_text(json.dumps(reg))
    complete_report(reg,tmp_path)
    with pytest.raises(ValueError,match='new_run'):finalize(path,registry)
    assert (source/'fills').read_text()=='frozen'


def test_symlink_workspace_cannot_be_created(tmp_path):
    target=tmp_path/'shared';target.mkdir();link=tmp_path/'.work';link.symlink_to(target)
    with pytest.raises(ValueError):create_workspace(link/'case-a',tmp_path/'reports'/'case-a','case-a')


def test_pack_verified_standard_report_and_keep_supplemental_results(tmp_path,monkeypatch):
    from app.backtesting import html_report
    path,reg,_=setup_run(tmp_path);reg['input_sha256']={'market':'identity'};path.write_text(json.dumps(reg))
    folder=Path(reg['report_output']);folder.mkdir()
    png=tmp_path/'chart.png';Image.new('RGB',(4,4)).save(png)
    for kind,_ in CHARTS:(folder/(kind+'.png')).write_bytes(png.read_bytes())
    for name,value in [('results.json',{'totals':[]}),('chart_statistics.json',{}),('prediction_comparison.json',{'pnl':'1.2'})]:
        (folder/name).write_text(json.dumps(value))
    (folder/'README.md').write_text('## 方法\n可验证结果')
    (folder/'ablation_effects.csv').write_text('delta_pnl,delta_max_drawdown,delta_total_notional\n')
    (folder/'daily_volume.csv').write_text('case,total_notional\na,0\n')
    hashes={p.name:sha(p.read_bytes()) for p in folder.iterdir()}
    (folder/'validation.json').write_text(json.dumps(dict(status='passed',report_sha256=hashes)))
    (folder/'._prediction_comparison.json').write_bytes(b'\x00\x05\x16\x07\xb0')
    monkeypatch.setattr(html_report,'chart_conclusion',lambda *args:'事实结论')
    payload=read_report(build_report(path,folder))
    assert payload['daily'][0]['total_notional']=='0'
    assert payload['supplemental']['prediction_comparison.json']['pnl']=='1.2'
    assert payload['retention']=='final_only'
    (folder/'README.md').write_text('modified after validation')
    with pytest.raises(ValueError,match='changed'):build_report(path,folder)


@pytest.mark.parametrize('worker_exit',[0,1])
def test_run_orchestrator_publishes_and_cleans_success_and_failure(tmp_path,monkeypatch,worker_exit):
    from types import SimpleNamespace
    from app.backtesting import fusion_experiment,result_retention,run_registry
    path,reg,registry=setup_run(tmp_path)
    from app.backtesting.campaign_standard import SYMBOLS
    reg['spec'].update(symbols=list(SYMBOLS),
        fee_scenarios=[{'id':'maker0','maker_fee':'0'},{'id':'rebate03','maker_fee':'-0.00003'}],
        engine={'queue':'cancel_50_50'},execution={'taker_fee':'.0004'},
        account={'max_notional':'1000'},outputs={'format':'html','retention':'final_only'},
        resources={'workers':1},runtime={'backtest_python':'python','report_python':'python'})
    path.write_text(json.dumps(reg))
    real_finalize=result_retention.finalize;real_status=run_registry.update_status
    monkeypatch.setattr(result_retention,'finalize',lambda p,**kwargs:real_finalize(p,registry,**kwargs))
    monkeypatch.setattr(run_registry,'update_status',lambda identity,status:real_status(identity,status,registry))
    def process(args,**kwargs):
        if 'report' in args:complete_report(reg,tmp_path)
        return SimpleNamespace(returncode=worker_exit if 'worker' in args else 0)
    monkeypatch.setattr(fusion_experiment.subprocess,'run',process)
    if worker_exit:
        with pytest.raises(RuntimeError,match='final HTML saved'):fusion_experiment.run(path)
    else:assert fusion_experiment.run(path)==Path(reg['final_report_output'])/'report.html'
    assert not Path(reg['run_output']).exists()
    assert read_report(Path(reg['final_report_output'])/'report.html')['status']==('failed' if worker_exit else 'completed')


def test_cleanup_ignores_only_missing_appledouble():
    from app.backtesting.result_retention import cleanup_error
    error = FileNotFoundError('sidecar removed automatically')
    cleanup_error(None, '/owned/._.managed-run.json', (FileNotFoundError,error,None))
    with pytest.raises(FileNotFoundError):
        cleanup_error(None, '/owned/summary.json', (FileNotFoundError,error,None))
    error = PermissionError('denied')
    with pytest.raises(PermissionError):
        cleanup_error(None, '/owned/._summary.json', (PermissionError,error,None))


def publish_with_cleanup_failure(path, registry, monkeypatch):
    from app.backtesting import result_retention
    def fail(*args, **kwargs):
        raise PermissionError('simulated cleanup failure')
    with monkeypatch.context() as patch:
        patch.setattr(result_retention.shutil, 'rmtree', fail)
        with pytest.raises(PermissionError, match='simulated cleanup failure'):
            finalize(path, registry)
    reg=json.loads(path.read_text())
    return Path(reg['final_report_output'])/'report.html'


def test_cleanup_failure_keeps_published_diagnostic_and_owned_workspace(tmp_path,monkeypatch):
    path,reg,registry=setup_run(tmp_path)
    build_failure_report(path,reg['report_output'],'worker crashed','bounded diagnostic')
    result=publish_with_cleanup_failure(path,registry,monkeypatch)
    assert read_report(result)['status']=='failed'
    assert (Path(reg['run_output'])/OWNER).is_file()
    entry=json.loads(registry.read_text())['entries'][0]
    assert entry['status']=='failed' and entry['retention']['workspace']=='pending'
    assert finalize(path,registry)==result
    assert not Path(reg['run_output']).exists()


def test_resume_publication_with_appledouble_metadata(tmp_path,monkeypatch):
    path,reg,registry=setup_run(tmp_path);complete_report(reg,tmp_path)
    report=publish_with_cleanup_failure(path,registry,monkeypatch)
    original=report.read_bytes()
    sidecar=report.parent/'._report.html';sidecar.write_bytes(b'macOS metadata')
    assert Path(reg['run_output']).exists()
    result=finalize(path,registry)
    assert result.read_bytes()==original and sidecar.exists()
    assert not Path(reg['run_output']).exists()


@pytest.mark.parametrize('conflict',['extra-file','sidecar-directory','report-symlink'])
def test_resume_still_refuses_unowned_final_contents(tmp_path,conflict,monkeypatch):
    path,reg,registry=setup_run(tmp_path);complete_report(reg,tmp_path)
    report=publish_with_cleanup_failure(path,registry,monkeypatch)
    if conflict=='extra-file':(report.parent/'other.txt').write_text('not metadata')
    elif conflict=='sidecar-directory':(report.parent/'._report.html').mkdir()
    else:
        external=tmp_path/'external.html';external.write_bytes(report.read_bytes())
        report.unlink();report.symlink_to(external)
    with pytest.raises(FileExistsError):finalize(path,registry)
    assert Path(reg['run_output']).exists()
