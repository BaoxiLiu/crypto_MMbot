"""Standard report for current, registered native observations."""
import csv
from datetime import datetime, timezone
from decimal import Decimal as D
import json
from pathlib import Path
import shutil

import numpy as np
from src.metrics import core as unified_metrics
from .experiment_metrics import write_json, write_csv
from .report_sections import chart_sections, CHARTS, save_coin_panels, configure_plotting
from ..artifacts import digest, ROOT


def _str(value):
    return str(value) if value is not None else None


def report(registration, output=None):
    from .experiment_config import resolve
    reg = json.loads(Path(registration).read_text())
    if reg['kind']!='new_run':raise ValueError('Only current new_run reports are supported')
    spec = reg['spec']
    if reg['kind']=='new_run':
        for path, expected in reg['source_sha256'].items():
            if digest(ROOT/path)!=expected:
                raise ValueError('Code changed after registration: '+path)
    for path, h in reg['input_sha256'].items():
        assert digest(path) == h, ('registered input changed', path)
    out = resolve(output) if output is not None else Path(reg['report_output'])
    if output is not None:
        # A revised presentation receives a new directory; frozen runs stay intact.
        out.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(registration, out/'PREREGISTRATION.json')
        write_json(out/'REPORT_ORIGIN.json', dict(
            run_id=reg['run_id'], registration=str(Path(registration).resolve()),
            registration_sha256=digest(registration), original_report=reg['report_output'],
            report_only=True, generated_utc=datetime.now(timezone.utc).isoformat()))
    summaries, curves, totals, daily, markouts, hashes = {}, {}, [], [], [], {}
    native_diagnostics={}
    attribution_metrics={}
    holding_metrics={}
    for case in reg['cases']:
        p = Path(case['output'])
        s = json.loads((p/'summary.json').read_text())
        identity = s['identity']
        if reg['spec']['observations'].get('pnl_attribution'):
            attribution_path = p/'pnl_attribution_metrics.json'
            attribution = json.loads(attribution_path.read_text())
            if (abs(D(attribution['residual'])) > D('1e-12')
                    or D(attribution['ledger_pnl']) != D(s['pnl'])):
                raise ValueError('Attribution does not reconcile to replay account')
            attribution_metrics[case['id']] = attribution
            hashes[str(attribution_path)] = digest(attribution_path)
        if 'native_strategy_diagnostics' in s:
            native_diagnostics[case['id']]=s['native_strategy_diagnostics']
        assert case['symbol'] in spec['symbols']
        assert identity['symbol'] == case['symbol'] and identity['mask'] == case['mask']
        assert identity['queue'] == spec['engine']['queue'] and identity['exchange'] == spec['engine']['exchange']
        assert identity['hftbacktest'] == spec['engine']['version']
        assert identity['execution_profile'] == spec['execution']
        assert identity['dataset_manifest_sha256'] == digest(resolve(spec['datasets'][case['symbol']]['cache'])/'manifest.json')
        expected_fee=case.get('fee_scenario',{}).get('maker_fee',spec['costs']['maker_fee'])
        assert case.get('fee_scenario') is None or case['fee_scenario'] in spec.get('fee_scenarios',[])
        assert D(identity['config']['as_config']['maker_fee']) == D(expected_fee)
        for key, value in spec['loss_stops'].items():
            assert D(identity['config'][key]) == D('1E+30' if value is None else str(value))
        assert s['invariant_failures'] == 0 and s['full_range'] == reg['full_range']
        assert D(s['pnl']) == D(s['equity']) - D(identity['config']['initial_cash'])
        assert abs(D(s['pnl']) - (D(s['realized'])+D(s['unrealized'])+D(s['funding'])-D(s['fees']))) < D('1e-18')
        if reg['kind'] == 'new_run':
            assert identity['config'] == case['config']
            for path, h in identity['source_sha256'].items():
                assert reg['source_sha256'][path] == h, path
            if not (p/'metrics.json').exists():
                raise ValueError('New run missing mandatory observations')
        with np.load(p/'equity.npz', allow_pickle=False) as z:
            a = z['data']
        assert np.all(np.diff(a['local_ns']) > 0) and np.isfinite(a['equity']).all()
        summaries[case['id']], curves[case['id']] = s, a
        base = dict(case=case['id'], symbol=case['symbol'], strategy=case['strategy'])
        mpath = p/'metrics.json'
        if mpath.exists():
            if (p/'fills.jsonl').exists():
                with (p/'fills.jsonl').open() as stream:
                    fills = (json.loads(line) for line in stream if line.strip())
                    holding_metrics[case['id']]=unified_metrics.position_holding_periods(
                        (dict(fill, ns=fill.get('receipt_ns', fill.get('ns'))) for fill in fills),
                        max([int(r['end_ns']) for r in s.get('segments',[])]+[int(a['local_ns'][-1]) if len(a) else 0]))
            m = json.loads(mpath.read_text())
            assert m['status'] == 'complete' and m['spec'] == spec['observations']
            assert m['full_range'] == reg['full_range']
            volume = m['volume']
            totals.append(dict(**base, status='recorded', total_notional=volume['total_notional'], total_qty=volume['total_qty'],
                               fill_receipts=m['oms_fill_receipts'], native_executions=s['native_fills'], pnl=s['pnl'], max_drawdown=s['max_drawdown']))
            with (p/'daily_volume.csv').open() as f:
                daily.extend(dict(**base, status='recorded', **r) for r in csv.DictReader(f))
            index = {int(r['horizon_ms']): r for r in m['markouts']}
            for h in spec['observations']['markout_horizons_ms']:
                r = index[h]
                markouts.append(dict(**base, horizon_ms=h, status='recorded', notional_weighted_bp=r['notional_weighted_bp'],
                                     observed=r['observed'], censored=r['censored'], mean_lag_ns=r['mean_lag_ns'],
                                     lag_budget_ms=spec['observations']['max_markout_lag_ms']))
            for name in ('metrics.json', 'fills.jsonl', 'markouts.jsonl', 'daily_volume.csv', 'markout_summary.csv'):
                hashes[str(p/name)] = digest(p/name)
            if spec['observations']['retain_midpoints']:
                hashes[str(p/'midpoints.csv.gz')] = digest(p/'midpoints.csv.gz')
    calendar_days = unified_metrics.calendar_days(spec['window']['start'], spec['window']['end_exclusive'])
    for row in totals:
        cash = D(summaries[row['case']]['identity']['config']['initial_cash'])
        row['net_pnl_per_filled_notional_bp'] = _str(unified_metrics.net_pnl_per_filled_notional_bp(
            row['pnl'], row['total_notional']))
        row['initial_cash'] = str(cash)
        row['turnover_initial_cash_multiple'] = _str(unified_metrics.turnover_initial_cash(
            row['total_notional'], cash))
        row['calendar_days'] = calendar_days
        row['average_daily_turnover_multiple'] = _str(unified_metrics.average_daily_turnover(
            row['total_notional'], cash, calendar_days)) if reg['full_range'] else None
    for row in daily:
        cash = D(summaries[row['case']]['identity']['config']['initial_cash'])
        row['initial_cash'] = str(cash)
        row['turnover_initial_cash_multiple'] = _str(unified_metrics.turnover_initial_cash(
            row['total_notional'] if row['total_notional'] != '' else None, cash))
    # Different coins have distinct prices/lot sizes. Never merge their raw quantities.
    for symbol in spec['symbols']:
        group = [summaries[c['id']]['identity'] for c in reg['cases'] if c['symbol'] == symbol]
        for i in group[1:]:
            for key in ('rules', 'dataset_manifest_sha256', 'marks_sha256', 'funding_sha256', 'execution_profile'):
                assert i[key] == group[0][key], (symbol, key)
    write_csv(out/'total_volume.csv', totals, list(totals[0]))
    fields = list(dict.fromkeys(k for r in daily for k in r))
    write_csv(out/'daily_volume.csv', daily, fields)
    write_csv(out/'markouts.csv', markouts, list(markouts[0]))
    write_json(out/'results.json', dict(full_range=reg['full_range'], totals=totals, markouts=markouts,
        metric_status='complete' if all(r['status']=='recorded' for r in totals) else 'legacy_missing_observations',
        notes='Volume is own filled buy+sell notional, no extra factor of two; UTC receipt dates. Markout clock is local execution receipt.',
        legacy_markouts='Existing 1s/5s aggregates have no lag cap; other requested horizons and volume are not reconstructed from equity.'))
    draw(out, spec, reg['cases'], summaries, curves, daily, markouts, reg['full_range'])
    lines = ['# 标准实验结果', '',
        f"实验：`{reg['run_id']}`；范围：UTC `[{spec['window']['start']}, {spec['window']['end_exclusive']})`；" + ('已运行请求区间全部认证片段，实际覆盖见每日表。' if reg['full_range'] else '**局部验证，不能当成完整区间结果。**'), '',
        f"引擎 {spec['engine']['version']}；队列 `{spec['engine']['queue']}`；撮合 `{spec['engine']['exchange']}`。", '',
        f"行情发布后传输 {spec['execution']['feed_ns']/1e6:g}ms，计算预算 {spec['execution']['decision_ns']/1e3:g}µs，订单去程/回程 {spec['execution']['request_ns']/1e6:g}/{spec['execution']['response_ns']/1e6:g}ms。maker {spec['costs']['maker_fee']}，taker {spec['execution']['taker_fee']}，官方资金费。计算和订单延迟仍是模拟假设。", '',
        '| 策略 | 币种 | 净PnL | 最大回撤 | 总成交额 USDC | 单位成交净收益 bp | 平均日换手率（倍/日） | native撮合笔数 |', '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in totals:
        volume = r['total_notional'] if r['total_notional'] is not None else '未记录'
        turnover = f"{D(r['average_daily_turnover_multiple']):.4f}" if r['average_daily_turnover_multiple'] is not None else ('局部不折算' if not reg['full_range'] else '未记录')
        net_bp = f"{D(r['net_pnl_per_filled_notional_bp']):.4f}" if r['net_pnl_per_filled_notional_bp'] is not None else '缺失'
        lines.append(f"| {r['strategy']} | {r['symbol']} | {D(r['pnl']):.6f} | {D(r['max_drawdown']):.6f} | {volume} | {net_bp} | {turnover} | {r['native_executions']} |")
    if spec.get('fee_scenarios'):
        lines+=['','费用情景按完整回放分别计算：'+ '；'.join(f"{f['id']} maker={D(str(f['maker_fee']))*10000}bp" for f in spec['fee_scenarios'])+'。taker固定4bp，非对同一成交结果事后加返佣。']
    if native_diagnostics:
        lines+=['','**策略实现：** AS家族调用项目本地Python NativeAS（公式约定参考Hummingbot），成交强度alpha/kappa在线拟合、gamma按原代码校准；兼容ASConfig中的固定gamma/kappa不是此插件的报价参数。完整早期组复用原S1、SC0、microprice、keeper和报价保留规则，原始AS关闭附加报价层。NH与HJB使用各自注册插件，其模型参数、模式及诊断保存在报告身份记录中。',
          '', '历史归档没有独立bookTicker，以已可见L2重建的BBO价格/数量替代；这是数据通道差异，不是原paper逐事件等价复现。成交、队列、延迟、敞口、账本使用共同HFT执行层；只用窗口前历史预热，窗口前禁止交易。缺口内不撮合；未到期Markout删失，已观察历史按原NativeAS/S1/SC0的缺失与时效规则处理，不擅自重启30分钟预热。具体代码、运行依赖哈希和有效配置均内嵌。',
          '', '日度夏普按有效日数据和统一指标合同计算，数据不足时不输出；不将独立账户的夏普平均当成组合夏普。持仓段以逐次增量成交回报计算，未结束段单列，包含模型跨认证缺口延续的持仓时间。订单名义金额与最小下单约束以本次有效配置为准，同币各策略共用执行约束，不以评估期收益调节单量。']
    lines += chart_sections(out, reg['cases'], summaries, curves, totals, daily, markouts, reg['full_range'])
    fee_charts = []
    if spec.get('fee_scenarios') and spec['outputs'].get('format') == 'html':
        for scenario in spec['fee_scenarios']:
            fee_id = scenario['id']
            cases = [c for c in reg['cases'] if c['fee_scenario']['id'] == fee_id]
            ids = {c['id'] for c in cases}
            folder = out / ('charts-' + fee_id)
            folder.mkdir(exist_ok=True)
            fee_daily = [r for r in daily if r['case'] in ids]
            fee_markouts = [r for r in markouts if r['case'] in ids]
            fee_totals = [r for r in totals if r['case'] in ids]
            draw(folder, spec, cases, summaries, curves, fee_daily, fee_markouts, reg['full_range'])
            chart_sections(folder, cases, summaries, curves, fee_totals, fee_daily, fee_markouts, reg['full_range'])
            for chart in sorted(folder.glob('*.png')):
                if chart.name.startswith('._'):
                    continue
                target = out / (fee_id + '-' + chart.name)
                chart.replace(target)
                fee_charts.append(target.name)

    lines += ['',
        '成交额按本账户每次增量成交的价格×数量累计，买卖相加，不再乘2；数量按合约报价单位保存。UTC按本地成交回报入账时间划日。OMS回报可能合并native撮合，两个计数分开列出。', '',
        f'累计换手倍数＝买卖成交额之和÷初始资金；平均日换手率（倍/日）＝累计换手倍数÷{calendar_days}个日历日，不额外除2、不年化，不按缺口后的认证小时外推。逐日表按当日成交额÷初始资金计算；局部回放不填全区间日均值。', '',
        '标准markout＝side×(未来mid−成交价)/成交价×10000，按成交额加权；从本地收到增量成交回报起，取期限到达后的第一条可见mid，超过配置滞后预算或跨行情缺口的样本删失。保存每条观测的目标时刻、实际mid时刻、滞后和删失原因。不是交易所成交时钟markout，也不含资金费和手续费扣减。', '',
        '旧实验只有1s、5s聚合markout且没有滞后上限；未记录的期限、每日和总成交额均为缺失，不填0、不从逐秒持仓差反推。新实验必须保存fills、markout逐笔明细及每日汇总，否则报告失败；midpoints按retain_midpoints配置保留，默认开启。', '',
        '单位成交净收益BP＝净PnL÷买卖成交额之和×10000；分母不除2，含资金费和期末未实现盈亏，不是完整开平仓配对的平均收益。无成交时缺失。', '',
        'PnL包括未实现盈亏和资金费，期末不默认平仓；曲线按5分钟保留逐秒采样的首末与极值，并附认证片段精确终点，行情缺口不连线。最大回撤取账本原统计。', '',
        '[有效配置和输入登记](PREREGISTRATION.json) · [总成交额](total_volume.csv) · [每日成交额](daily_volume.csv) · [Markout表](markouts.csv) · [结构化结果](results.json) · [图表统计](chart_statistics.json) · [消融配对差值](ablation_effects.csv) · [验证](validation.json)']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    report_sources = {}
    for name in ('experiment_report.py','report_sections.py','experiment_metrics.py','fusion_experiment.py'):
        source = Path(__file__).parent/name
        report_sources[str(source.relative_to(ROOT))] = digest(source)
    write_json(out/'validation.json', dict(status='passed', full_range=reg['full_range'], cases=len(totals),
        raw_sha256=hashes, registration_sha256=digest(registration),
        report_code_sha256=digest(__file__),
        report_sources_sha256=report_sources,
        source='recorded replay observations' if reg['kind']=='new_run' else 'existing immutable artifacts; missing observations explicitly declared',
        report_sha256={name:digest(out/name) for name in ['README.md','results.json','total_volume.csv','daily_volume.csv','markouts.csv','chart_statistics.json','ablation_effects.csv'] + [kind+'.png' for kind, _ in CHARTS] + sorted(p.name for p in out.glob('*-coin-*.png') if not p.name.startswith('._')) + fee_charts + (['REPORT_ORIGIN.json'] if (out/'REPORT_ORIGIN.json').exists() else [])}))
    if native_diagnostics:
        write_json(out/'native_strategy_metrics.json',native_diagnostics)
        validation=json.loads((out/'validation.json').read_text())
        validation['report_sha256']['native_strategy_metrics.json']=digest(out/'native_strategy_metrics.json')
        write_json(out/'validation.json',validation)
    if attribution_metrics:
        write_json(out/'pnl_attribution_metrics.json',attribution_metrics)
        validation=json.loads((out/'validation.json').read_text())
        validation['report_sha256']['pnl_attribution_metrics.json']=digest(out/'pnl_attribution_metrics.json')
        write_json(out/'validation.json',validation)
    if holding_metrics:
        write_json(out/'holding_metrics.json',holding_metrics)
        validation=json.loads((out/'validation.json').read_text())
        validation['report_sha256']['holding_metrics.json']=digest(out/'holding_metrics.json')
        write_json(out/'validation.json',validation)
    if native_diagnostics:
        write_json(out/'cross_coin_metrics.json',unified_metrics.aggregate_independent_accounts(totals,calendar_days,full_range=reg["full_range"]))
        validation=json.loads((out/'validation.json').read_text())
        validation['report_sha256']['cross_coin_metrics.json']=digest(out/'cross_coin_metrics.json')
        write_json(out/'validation.json',validation)
    if spec['outputs'].get('format','html')=='html' or spec['outputs'].get('retention')=='final_only':
        from .html_report import build_report
        build_report(registration,out)
    print(out, flush=True)


def draw(out, spec, cases, summaries, curves, daily, markouts, full_range):
    configure_plotting()
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    symbols = list(dict.fromkeys(c['symbol'] for c in cases))
    palette = list(plt.get_cmap('tab10').colors)
    strategies = list(dict.fromkeys(c['strategy'] for c in cases))
    colors = {c['id']:palette[strategies.index(c['strategy'])%len(palette)] for c in cases}
    for kind in ('pnl', 'daily_volume', 'markouts'):
        fig, axes = plt.subplots(len(symbols), 1, figsize=(12, 4.7*len(symbols)), squeeze=False)
        for ax, symbol in zip(axes[:,0], symbols):
            group = [c for c in cases if c['symbol']==symbol]
            dates = sorted({r['utc_day'] for r in daily if r['symbol']==symbol and r['total_notional'] not in (None,'')})
            any_data = False
            for j,c in enumerate(group):
                cid = c['id']
                if kind=='pnl':
                    s,a=summaries[cid],curves[cid]
                    xs,ys=[],[]
                    # Ordered time-bin envelope, preserving extrema without smoothing.
                    for seg in s['segments']:
                        lo=np.searchsorted(a['local_ns'],seg['start_ns'],'left');hi=np.searchsorted(a['local_ns'],seg['end_ns'],'right')
                        x=a['local_ns'][lo:hi];y=a['equity'][lo:hi]-float(s['identity']['config']['initial_cash'])
                        if len(x):
                            bins=x//300_000_000_000
                            starts=np.r_[0,np.flatnonzero(np.diff(bins))+1];ends=np.r_[starts[1:],len(x)]
                            for left,right in zip(starts,ends):
                                ix=sorted({left,right-1,left+int(np.argmin(y[left:right])),left+int(np.argmax(y[left:right]))})
                                xs.extend(x[ix]/1e9/86400);ys.extend(y[ix])
                        xs.extend([seg['end_ns']/1e9/86400,np.nan]);ys.extend([float(seg['pnl']),np.nan])
                    ax.plot(xs,ys,label=c['strategy'],color=colors[cid],lw=1.4)
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d',tz=timezone.utc))
                    ax.set_ylabel('净PnL（USDC）');ax.set_xlabel('日期（UTC）');any_data=True
                elif kind=='daily_volume':
                    rows=[r for r in daily if r['case']==cid and r['total_notional'] not in (None,'')]
                    if rows:
                        xs=[dates.index(r['utc_day']) for r in rows]
                        width=.8/len(group)
                        ax.bar(np.array(xs)+(j-(len(group)-1)/2)*width,[float(r['total_notional']) for r in rows],width=width,color=colors[cid],label=c['strategy'])
                        ax.set_xticks(range(len(dates)),[day[5:] for day in dates])
                        ax.set_ylabel('成交额（USDC）');ax.set_xlabel('UTC日期 / 仅已回放时段');any_data=True
                else:
                    rows=[r for r in markouts if r['case']==cid]
                    y=[float(r['notional_weighted_bp']) if r['notional_weighted_bp'] is not None else np.nan for r in rows]
                    ax.plot(range(len(rows)),y,marker='o',label=c['strategy'],color=colors[cid])
                    ax.set_xticks(range(len(rows)),[f"{r['horizon_ms']}ms" if r['horizon_ms']<1000 else f"{r['horizon_ms']/1000:g}s" for r in rows]);ax.set_ylabel('成交额加权markout（bp）');ax.set_xlabel('从本地成交回报起算');any_data |= any(np.isfinite(y))
            if not any_data:
                ax.text(.5,.5,'未记录：旧实验缺少逐笔成交，不能从权益曲线反推',transform=ax.transAxes,ha='center',va='center',color='#596573')
                ax.set_xticks([]);ax.set_yticks([])
            else:
                ax.legend(frameon=False);ax.grid(axis='y',alpha=.2)
            ax.set_title(symbol+(' · 局部验证' if not full_range else ''));ax.spines[['top','right']].set_visible(False)
        title={'pnl':'PnL对比','daily_volume':'每日成交额对比','markouts':'多期限Markout对比（缺失不填0）'}[kind]
        fig.suptitle(title,fontsize=17)
        fig.tight_layout(rect=(0,.07,1,.96))
        fig.text(.03,.025,'同一数据、费用和延迟预算；旧1s/5s为无滞后上限的历史聚合，新记录采用配置的滞后预算。',fontsize=9,color='#596573')
        fig.savefig(out/(kind+'.png'),dpi=170,facecolor='white')
        save_coin_panels(out, fig, axes[:, 0], kind)
        plt.close(fig)
