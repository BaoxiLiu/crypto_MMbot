"""Self-contained offline final reports; embedded data survives workspace cleanup."""
import base64
import csv
import hashlib
from html import escape
from html.parser import HTMLParser
import json
from pathlib import Path
import re

from .report_sections import CHARTS,chart_conclusion


def sha(raw):return hashlib.sha256(raw).hexdigest()
def json_text(value):return json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',',':'))


class EmbeddedReport(HTMLParser):
    def __init__(self):
        super().__init__();self.in_data=False;self.parts=[];self.digest=None;self.images=[]

    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag=='script' and a.get('id')=='experiment-data':
            if self.digest is not None:raise ValueError('Duplicate embedded report data')
            self.in_data=True;self.digest=a.get('data-sha256')
        if tag=='img':self.images.append((a.get('data-chart'),a.get('src','')))
        if tag in ('script','link','iframe') and a.get('src',a.get('href')):
            raise ValueError('Final report cannot require external assets')

    def handle_endtag(self,tag):
        if tag=='script':self.in_data=False

    def handle_data(self,data):
        if self.in_data:self.parts.append(data)


def read_report(path):
    parser=EmbeddedReport();parser.feed(Path(path).read_text(encoding='utf-8'))
    raw=''.join(parser.parts)
    if not parser.digest or sha(raw.encode())!=parser.digest:raise ValueError('Invalid embedded report checksum')
    data=json.loads(raw)
    if data.get('schema_version')!=1 or data.get('status') not in ('completed','failed'):
        raise ValueError('Unsupported final report schema/status')
    expected=data.get('chart_sha256',{})
    if len(parser.images)!=len(expected):raise ValueError('Final report chart count mismatch')
    seen=set()
    for kind,src in parser.images:
        if kind in seen or not src.startswith('data:image/png;base64,'):
            raise ValueError('Duplicate/nonembedded chart')
        raw=base64.b64decode(src.split(',',1)[1],validate=True)
        if sha(raw)!=expected.get(kind) or not raw.startswith(b'\x89PNG\r\n\x1a\n'):
            raise ValueError('Final report chart checksum mismatch')
        seen.add(kind)
    if data['status']=='completed':
        kinds = {kind for kind, _ in CHARTS}
        panel_coins = data.get('panel_coins', [])
        if len(panel_coins) != len(set(panel_coins)) or not set(panel_coins) <= set(data['spec']['symbols']):
            raise ValueError('Invalid coin panel mapping')
        if data.get('fee_split_charts'):
            kinds = {f'{scenario["id"]}-{kind}' for scenario in data['spec']['fee_scenarios'] for kind in kinds}
        required = kinds | {f'{kind}-coin-{index}' for kind in kinds for index in range(len(panel_coins))}
        if set(expected) != required:
            raise ValueError('Completed final report requires all overview charts and declared coin panels')
    return data


def inline(text):
    text=escape(str(text))
    text=re.sub(r'`([^`]+)`',r'<code>\1</code>',text)
    text=re.sub(r'\*\*([^*]+)\*\*',r'<strong>\1</strong>',text)
    # Report links referred to deleted intermediates; retain labels, not dead links.
    return re.sub(r'\[([^\]]+)\]\([^)]+\)',r'\1',text)


def notes_html(markdown):
    lines=[];paragraph=[]
    def flush():
        if paragraph:lines.append('<p>'+inline(' '.join(paragraph))+'</p>');paragraph.clear()
    for line in markdown.splitlines():
        if line.startswith(('|','![')):flush();continue
        if not line.strip():flush();continue
        if line.startswith('#'):
            flush();lines.append('<h3>'+inline(line.lstrip('# '))+'</h3>')
        else:paragraph.append(line)
    flush();return ''.join(lines)


CSS='''
:root{--ink:#182c3f;--navy:#123b5c;--teal:#006d77;--paper:#f5f8fb;--line:#d8e3ee;--muted:#546b80;--red:#b13d50}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.7 -apple-system,BlinkMacSystemFont,"Avenir Next","PingFang SC","Microsoft YaHei",sans-serif}
a{color:var(--teal)}header{background:var(--navy);color:white;padding:42px max(24px,calc((100vw - 1280px)/2)) 32px}header p{margin:0;opacity:.8}.eyebrow{letter-spacing:.15em;font-size:12px}h1{font-size:clamp(25px,4vw,42px);line-height:1.25;margin:12px 0}h2{font-size:23px;margin:0 0 18px}h3{font-size:18px}.run-id{font:12px/1.6 ui-monospace,Menlo,monospace;overflow-wrap:anywhere}.status{display:inline-block;border:1px solid #8db4d1;border-radius:4px;padding:3px 10px;margin-top:16px}
main{max-width:1328px;margin:auto;padding:24px}.passport{display:grid;grid-template-columns:repeat(4,1fr);background:white;border:1px solid var(--line);margin-bottom:24px}.passport div{padding:16px 20px;border-right:1px solid var(--line)}.passport small{display:block;color:var(--muted)}section{background:white;border:1px solid var(--line);border-radius:8px;padding:26px;margin-bottom:24px}.filters{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-bottom:16px}label{color:var(--muted)}select{font:inherit;padding:7px 12px;border:1px solid #b7cbdc;border-radius:4px;background:white;color:var(--ink)}.table-scroll{overflow:auto}table{width:100%;border-collapse:collapse;white-space:nowrap}th{background:#edf3f8;color:var(--navy);text-align:left}th,td{padding:11px 12px;border-bottom:1px solid var(--line)}td.num{font-family:ui-monospace,Menlo,monospace;text-align:right}td.pos{color:var(--teal)}td.neg{color:var(--red)}.muted{color:var(--muted);font-size:13px}.notice{border-left:4px solid #ce8b2a;background:#fff8e9;padding:12px 16px}.plots{display:grid;gap:22px}figure{margin:0;border-top:1px solid var(--line);padding-top:12px}figure img{display:block;width:100%;height:auto;background:#fff}figcaption{padding:14px 18px;background:#edf5f7;color:#244b5a;line-height:1.9}details{border-top:1px solid var(--line);padding:14px 0}summary{cursor:pointer;font-weight:600}pre{overflow:auto;background:#f0f4f8;padding:18px;font:12px/1.7 ui-monospace,Menlo,monospace;max-height:580px}code{font-family:ui-monospace,Menlo,monospace;overflow-wrap:anywhere}footer{max-width:1280px;margin:20px auto;padding:0 24px 32px;color:var(--muted);font-size:12px}:focus-visible{outline:3px solid #11a1b1;outline-offset:3px}[hidden]{display:none!important}@media(max-width:700px){main{padding:12px}section{padding:16px}.passport{grid-template-columns:1fr 1fr}.passport div{padding:12px}th,td{padding:9px;font-size:13px}}@media print{header{background:white;color:var(--ink)}body{background:white}.filters{display:none}section,figure{break-inside:avoid}main{max-width:none}}
'''
JS='''
const coin=document.getElementById('coin'),strategy=document.getElementById('strategy');
function filterRows(){let count=0;document.querySelectorAll('.comparison tbody tr').forEach(row=>{row.hidden=!!((coin.value&&row.dataset.coin!==coin.value)||(strategy.value&&row.dataset.strategy!==strategy.value));if(!row.hidden)count++;});document.querySelectorAll('[data-native-coin]').forEach(row=>{row.hidden=!!((coin.value&&row.dataset.nativeCoin!==coin.value)||(strategy.value&&row.dataset.nativeStrategy!==strategy.value));});document.querySelectorAll('.plots figure').forEach(figure=>{const panels=Array.from(figure.querySelectorAll('[data-panel-coin]'));const matches=panels.filter(panel=>!coin.value||panel.dataset.panelCoin===coin.value);figure.querySelector('img').hidden=matches.length>0;panels.forEach(panel=>{panel.hidden=!matches.includes(panel);});});document.getElementById('visible-count').textContent='显示 '+count+' 条结果';}
if(coin&&strategy){coin.addEventListener('change',filterRows);strategy.addEventListener('change',filterRows);filterRows();}
'''


def native_plots(payload):
    """Policy diagnostics use original integer ticks, with gaps left disconnected."""
    from datetime import datetime,timezone
    stats=payload.get('supplemental',{}).get('native_strategy_metrics.json',{})
    cases={c['id']:c for c in payload.get('effective_cases',[])}
    charts=[]
    for cid,d in stats.items():
        rows=d.get('floor_samples',[]);c=cases.get(cid)
        if not rows or c is None:continue
        start,end=rows[0]['ns'],rows[-1]['ns'];span=max(1,end-start)
        lines=[]
        fields=[('static_floor_ticks','#546b80'),('vol_floor_ticks','#006d77'),('sc_target_ticks','#b13d50')]
        top=max(1,max(float(r[k]) for r in rows for k,_ in fields))
        for field,color in fields:
            pieces=[];current=[];previous=None
            for r in rows:
                if previous is not None and r['ns']-previous>90_000_000_000:
                    pieces.append(current);current=[]
                current.append(f"{50+850*(r['ns']-start)/span:.2f},{190-160*float(r[field])/top:.2f}")
                previous=r['ns']
            pieces.append(current)
            lines.extend('<polyline fill="none" stroke="'+color+'" stroke-width="2" points="'+' '.join(p)+'"/>' for p in pieces if p)
        times=[datetime.fromtimestamp(t/1e9,timezone.utc).strftime('%m/%d %H:%M UTC') for t in (start,end)]
        charts.append('<details data-native-coin="'+escape(c['symbol'],quote=True)+'" data-native-strategy="'+escape(c['strategy'],quote=True)+'"><summary>'+escape(c['symbol']+' · '+c['strategy'])+'</summary><svg style="display:block;width:100%;height:auto" viewBox="0 0 960 220" role="img" aria-label="三种政策下限，单位tick"><path d="M50 25V190H900" fill="none" stroke="#a3b7c8"/><text x="8" y="36">'+str(top)+'t</text><text x="18" y="190">0</text>'+''.join(lines)+'<text x="50" y="215">'+times[0]+'</text><text x="900" y="215" text-anchor="end">'+times[1]+'</text></svg></details>')
    if not charts:return ''
    return '<section><h2>三个调整模块的变化</h2><p>按币种和费率展开查看。原始AS不启用独立S1/SC0模块。</p>'+''.join(charts)+'</section>'


def cell(value,signed=False,integer=False):
    if value in (None,''):return '<td class="num muted">—</td>'
    from decimal import Decimal
    try:
        v=Decimal(str(value));text=f'{v:,.0f}' if integer else f'{v:+,.4f}' if signed else f'{v:,.4f}'
        cls=' pos' if signed and v>0 else ' neg' if signed and v<0 else ''
    except Exception:text=str(value);cls=''
    return '<td class="num'+cls+'">'+escape(text)+'</td>'


def fee_groups(rows, spec):
    scenarios = spec.get('fee_scenarios', [])
    if not scenarios:
        return [('', rows)]
    from decimal import Decimal
    groups = []
    for scenario in scenarios:
        fee = Decimal(str(scenario['maker_fee'])) * 10000
        label = '有返佣' if fee < 0 else '无返佣'
        groups.append((f'{label}（Maker {fee.normalize():f} bp）',
                       [r for r in rows if r['strategy'].endswith('__' + scenario['id'])]))
    if sum(len(group) for _, group in groups) != len(rows):
        return [('', rows)]
    return groups


def fee_tables(rows, spec, columns, render_row, css=''):
    return ''.join(
        ('<h3>'+escape(label)+'</h3>' if label else '')+
        '<div class="table-scroll"><table class="'+css+'"><thead><tr>'+columns+
        '</tr></thead><tbody>'+''.join(map(render_row, group))+'</tbody></table></div>'
        for label, group in fee_groups(rows, spec)
    )


def cross_coin_table(payload):
    rows=payload.get('supplemental',{}).get('cross_coin_metrics.json',[])
    if not rows:return ''
    def row(r):
        return ('<tr><td>'+escape(r['strategy'])+'</td>'+cell(r['coins'],integer=True)+cell(r['initial_cash'])+
                cell(r['pnl'],True)+cell(r['total_notional'])+cell(r['net_bp'],True)+
                cell(r['average_daily_turnover'])+'</tr>')
    tables=fee_tables(rows,payload['spec'],'<th>策略</th><th>币数</th><th>初始资金 / USDC</th><th>净盈亏（USDC）</th><th>成交额 / USDC</th><th>单位成交净收益（bp）</th><th>日均换手 / 倍</th>',row)
    return '<section><h2>跨币汇总 · 独立账户终值求和</h2>'+tables+'<p class="muted">净盈亏为期末权益减期初资金，含已实现与浮动盈亏、资金费和手续费。单位成交净收益＝净盈亏合计 ÷ 买卖成交额合计 × 10,000，零成交显示“—”。各币末次可见mark的时刻可能不同；不据此推算组合回撤或夏普。不同费率是反事实情景，资金不能跨情景累加。币种筛选不改变本表的完整汇总范围。</p></section>'


def render(payload,images,conclusions,markdown=''):
    spec=payload['spec'];window=spec.get('window',{})
    rows=payload.get('results',{}).get('totals',[])
    coins=spec.get('symbols',[]);strategies=list(dict.fromkeys(r['strategy'] for r in rows))
    status=('完整区间' if payload['full_range'] else '局部验证') if payload['status']=='completed' else '运行失败'
    options=lambda values:''.join('<option value="'+escape(v,quote=True)+'">'+escape(v)+'</option>' for v in values)
    holding=payload.get('supplemental',{}).get('holding_metrics.json',{})
    def row(r):
        return ('<tr data-coin="'+escape(r['symbol'],quote=True)+'" data-strategy="'+escape(r['strategy'],quote=True)+'"><td>'+escape(r['symbol'])+'</td><td>'+escape(r['strategy'])+'</td>'+''.join([
            cell(r.get('pnl'),True),cell(r.get('net_pnl_per_filled_notional_bp'),True),
            cell(r.get('total_notional')),cell(r.get('average_daily_turnover_multiple')),
            cell(r.get('max_drawdown')),cell(r.get('fill_receipts'),integer=True)])+
            (cell(holding.get(r.get('case'),{}).get('mean_closed_seconds'))+
             cell(holding.get(r.get('case'),{}).get('open_episode_seconds')) if holding else '')+'</tr>')
    tables=fee_tables(rows,spec,'<th>币种</th><th>策略</th><th>净盈亏（USDC）</th><th>单位成交净收益（bp）</th><th>成交额 / USDC</th><th>日均换手 / 倍</th><th>最大回撤 / USDC</th><th>成交回报数</th>'+('<th>平均已结束持仓 / s</th><th>未结束持仓 / s</th>' if holding else ''),row,'comparison')
    figures=[]
    chart_entries = CHARTS
    if payload.get('fee_split_charts'):
        chart_entries = [
            (scenario['id']+'-'+kind,title+' · '+label)
            for kind,title in CHARTS
            for scenario,(label,_) in zip(spec['fee_scenarios'],fee_groups(rows,spec))
        ]
    for kind,title in chart_entries:
        if kind not in images:continue
        figures.append('<figure id="'+kind+'"><h3>'+escape(title)+'</h3><img data-chart="'+kind+'" src="data:image/png;base64,'+images[kind]+'" alt="'+escape(title,quote=True)+'">')
        for index, coin in enumerate(payload.get('panel_coins', coins)):
            key = f'{kind}-coin-{index}'
            if key in images:
                figures.append('<img hidden data-panel-coin="'+escape(coin,quote=True)+'" data-chart="'+key+'" src="data:image/png;base64,'+images[key]+'" alt="'+escape(title+' · '+coin,quote=True)+'">')
        figures.append('</figure>')
    legacy_chart_note = '<p class="muted">归档图保留原费率混排口径。按费率重绘需要原始回放曲线。</p>' if spec.get('fee_scenarios') and not payload.get('fee_split_charts') else ''
    failure=''
    if payload['status']=='failed':failure='<section class="notice"><h2>本次未完成</h2><p>'+escape(payload.get('error',''))+'</p><pre>'+escape(payload.get('log_tail',''))+'</pre></section>'
    notice=('本文件是合成数据展示，不代表策略回测表现。' if payload.get('demo') else
            '局部验证：结果只覆盖实际执行片段，不折算完整区间表现。' if not payload['full_range'] else
            '结果覆盖本次全部认证片段，实际行情缺口见覆盖图。')
    if payload['status']=='failed':notice='本次运行失败，不能据此判断策略表现。'
    archive=('本次采用仅保留最终结果模式：报告验收并登记后，清理本次专属临时目录。配置、来源哈希、汇总表与图表均已嵌入本文件；逐笔重算需要重新回放行情。'
             if payload.get('retention')=='final_only' else
             '本文件是独立HTML导出，原有实验产物保持不变。配置、来源哈希、汇总表与图表均已嵌入本文件。')
    serialized=json_text(payload).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    stage={'quick':'快速验证','full':'全面评估','custom':'自定义区间'}.get(payload.get('stage','custom'),'自定义区间')
    badge='<span class="status">'+escape(status)+' · '+stage+'</span>'
    def fee(value):
        from decimal import Decimal
        return '—' if value is None else f'{(Decimal(str(value))*10000).normalize():f} bp'
    data_id=escape(payload['run_id']);title=escape(spec.get('name',payload['run_id']))
    passport=''.join('<div><small>'+label+'</small><strong>'+escape(str(value))+'</strong></div>' for label,value in [
        ('UTC 区间',window.get('start','—')+' → '+window.get('end_exclusive','—')+'（右端不含）'),
        ('币种 / 策略',f'{len(coins)} 币 / {len(strategies)//len(spec["fee_scenarios"])} 策略 × {len(spec["fee_scenarios"])} 费率' if spec.get('fee_scenarios') else f'{len(coins)} 币 / {len(strategies)} 策略'),
        ('队列',{'cancel_50_50':'撤单前后各50%','risk_adverse':'保守队列'}.get(spec.get('engine',{}).get('queue'),'未记录')),
        ('费率',f"maker {' / '.join(fee(f['maker_fee']) for f in spec['fee_scenarios']) if spec.get('fee_scenarios') else fee(spec.get('costs',{}).get('maker_fee'))}；taker {fee(spec.get('execution',{}).get('taker_fee'))}")])
    heading = '做市策略回测报告'
    if window.get('start') and window.get('end_exclusive'):
        from datetime import date
        if (date.fromisoformat(window['end_exclusive']) - date.fromisoformat(window['start'])).days == 3:
            heading = '做市策略三日回测报告'
    return '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>'+title+' · 回测报告</title><style>'+CSS+'</style></head><body><header><p class="eyebrow">实验记录</p><h1>'+heading+'</h1><p class="run-id">'+data_id+'</p>'+badge+'</header><main><div class="passport">'+passport+'</div><p class="notice">'+escape(notice)+'</p>'+failure+cross_coin_table(payload)+'''
<section><h2>逐币 · 策略对照</h2><div class="filters"><label>币种 <select id="coin"><option value="">全部币种</option>'''+options(coins)+'''</select></label><label>策略 <select id="strategy"><option value="">全部策略</option>'''+options(strategies)+'''</select></label><span id="visible-count" class="muted"></span></div>'''+tables+'''<p class="muted">各币、各策略使用独立账户。缺失显示“—”；持仓段按净仓从0出发、回到0统计，未结束持仓单列，含模型跨行情缺口延续的时间。单位成交净收益＝该币该组净盈亏 ÷ 买卖成交额 × 10,000。币种筛选联动下方图表，策略筛选作用于表格及报价下限图。</p></section>
<section><h2>回测图表</h2>'''+legacy_chart_note+'''<div class="plots">'''+''.join(figures)+'''</div></section>'''+native_plots(payload)+'''<section><h2>方法与归档</h2><p>'''+escape(archive)+'''</p><details><summary>完整报告说明与统计口径</summary>'''+notes_html(markdown)+'''</details><details><summary>本次完整配置</summary><pre>'''+escape(json.dumps(spec,ensure_ascii=False,indent=2))+'''</pre></details><details><summary>结果摘要、来源与验证</summary><pre>'''+escape(json.dumps({k:v for k,v in payload.items() if k not in ('spec','report_markdown')},ensure_ascii=False,indent=2))+'''</pre></details></section></main><footer>本地离线报告 · 图片和数据内嵌 · 金额单位USDC · 不加载网络资源</footer><script id="experiment-data" type="application/json" data-sha256="'''+sha(serialized.encode())+'">'+serialized+'</script><script>'+JS+'</script></body></html>'


def write_report(path,payload,images,conclusions,markdown=''):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp')
    tmp.write_text(render(payload,images,conclusions,markdown),encoding='utf-8')
    read_report(tmp);tmp.replace(p)
    return p


def build_report(registration,folder):
    folder=Path(folder);reg=json.loads(Path(registration).read_text())
    validation=json.loads((folder/'validation.json').read_text())
    if validation['status']!='passed':raise ValueError('Cannot publish unvalidated results')
    for name,expected in validation['report_sha256'].items():
        if Path(name).name!=name or sha((folder/name).read_bytes())!=expected:
            raise ValueError('Report source changed before HTML packing: '+name)
    stats=json.loads((folder/'chart_statistics.json').read_text())
    effects=[]
    with (folder/'ablation_effects.csv').open() as f:
        for row in csv.DictReader(f):
            for key in ('delta_pnl','delta_max_drawdown','delta_total_notional'):
                row[key]=float(row[key]) if row[key] else None
            effects.append(row)
    images={};hashes={};conclusions={}
    for kind,_ in CHARTS:
        raw=(folder/(kind+'.png')).read_bytes();hashes[kind]=sha(raw)
        images[kind]=base64.b64encode(raw).decode('ascii')
        conclusions[kind]=('局部验证：' if not reg['full_range'] else '')+chart_conclusion(kind,stats,effects)
    has_panels = 'pnl-coin-0.png' in validation['report_sha256']
    panel_coins = list(dict.fromkeys(case['symbol'] for case in reg['cases'])) if has_panels else []
    for kind, _ in CHARTS:
        for index, _ in enumerate(panel_coins if has_panels else []):
            key = f'{kind}-coin-{index}'
            if key+'.png' not in validation['report_sha256']:
                raise ValueError('Missing validated coin panel: '+key)
            raw = (folder/(key+'.png')).read_bytes()
            hashes[key] = sha(raw)
            images[key] = base64.b64encode(raw).decode('ascii')
    prefixes = [scenario['id'] + '-' for scenario in reg['spec'].get('fee_scenarios', [])]
    fee_split_charts = bool(prefixes and prefixes[0] + 'pnl.png' in validation['report_sha256'])
    if fee_split_charts:
        images, hashes = {}, {}
        for prefix in prefixes:
            for kind, _ in CHARTS:
                keys = [kind] + [f'{kind}-coin-{index}' for index in range(len(panel_coins))]
                for key in keys:
                    key = prefix + key
                    raw = (folder/(key+'.png')).read_bytes()
                    hashes[key] = sha(raw)
                    if hashes[key] != validation['report_sha256'][key+'.png']:
                        raise ValueError('Fee chart checksum mismatch: ' + key)
                    images[key] = base64.b64encode(raw).decode('ascii')
    with (folder/'daily_volume.csv').open() as f:daily=list(csv.DictReader(f))
    payload=dict(schema_version=1,status='completed',run_id=reg['run_id'],stage=reg['spec'].get('stage','custom'),
        full_range=reg['full_range'],registered_utc=reg.get('registered_utc'),
        preparation_policy=reg.get('preparation_policy',{'mode':'legacy_unrecorded'}),spec=reg['spec'],
        results=json.loads((folder/'results.json').read_text()),daily=daily,chart_statistics=stats,
        validation=validation,chart_sha256=hashes,fee_split_charts=fee_split_charts,input_sha256=reg['input_sha256'],
        source_sha256=reg.get('source_sha256',{}),retention=reg['spec']['outputs'].get('retention','all'),
        effective_cases=reg['cases'], panel_coins=panel_coins if has_panels else [],
        dataset_inputs=reg.get('inputs',{}),
        original_observations_retained=reg['spec']['outputs'].get('retention','all')!='final_only',
        report_markdown=(folder/'README.md').read_text(),
        supplemental={p.name:json.loads(p.read_text()) for p in folder.glob('*.json')
                      if not p.name.startswith('._') and p.name.endswith(('_comparison.json','_metrics.json'))})
    return write_report(folder/'report.html',payload,images,conclusions,(folder/'README.md').read_text())


def build_failure_report(registration,folder,error,log_tail=''):
    reg=json.loads(Path(registration).read_text());spec=reg['spec']
    payload=dict(schema_version=1,status='failed',run_id=reg['run_id'],stage=spec.get('stage','custom'),
        full_range=reg['full_range'],registered_utc=reg.get('registered_utc'),
        preparation_policy=reg.get('preparation_policy',{'mode':'legacy_unrecorded'}),spec=spec,
        results={'totals':[]},chart_sha256={},error=str(error),log_tail=log_tail[-65536:],
        input_sha256=reg.get('input_sha256',{}),source_sha256=reg.get('source_sha256',{}),
        retention=spec['outputs'].get('retention','all'),original_observations_retained=True)
    return write_report(Path(folder)/'report.html',payload,{},{} )
