"""Chart-adjacent conclusions and comparable, paired module effects.

This module consumes recorded artifacts only; it never infers fills from positions.
"""
import gzip
from decimal import Decimal
import math
import json
from pathlib import Path

import numpy as np

from .experiment_metrics import write_csv, write_json



def configure_plotting():
    """Apply the same headless plotting and font settings to every report entry."""
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import font_manager
    font = Path('/System/Library/Fonts/Supplemental/Arial Unicode.ttf')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        matplotlib.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
    matplotlib.rcParams.update({'axes.unicode_minus': False, 'font.size': 10})


def submitted_quantity(value, step=None):
    """Recover legal lot counts from JSON numeric native order observations."""
    quantity = Decimal(str(value))
    if not quantity.is_finite() or quantity <= 0:
        raise ValueError('Invalid recorded submitted quantity')
    if step is None:
        return quantity
    step = Decimal(str(step))
    if not step.is_finite() or step <= 0:
        raise ValueError('Invalid recorded lot size')
    canonical = (quantity / step).to_integral_value() * step
    tolerance = Decimal(str(8 * math.ulp(float(value)))) if isinstance(value, float) else Decimal(0)
    if tolerance >= step / 1000 or abs(quantity - canonical) > tolerance or canonical <= 0:
        raise ValueError('Recorded submitted quantity is off the legal lot grid')
    return canonical


def chart_statistics(cases, summaries, curves, totals, daily, markouts):
    result = {}
    for case in cases:
        cid = case['id']
        s = summaries[cid]
        total = next(r for r in totals if r['case'] == cid)
        a = curves[cid]
        orders = None
        placements = Path(case['output'])/'order_placements.jsonl.gz'
        fills = Path(case['output'])/'fills.jsonl'
        if placements.exists() and fills.exists():
            quantities = {}
            with fills.open() as f:
                for line in f:
                    r = json.loads(line)
                    quantities[r['cid']] = quantities.get(r['cid'], Decimal(0)) + Decimal(r['qty'])
            orders = dict(submitted=0, no_observed_fill=0, partially_filled=0, fully_filled=0)
            seen = set()
            with gzip.open(placements, 'rt') as f:
                for line in f:
                    r = json.loads(line)
                    if r['cid'] in seen:
                        raise ValueError('Duplicate submitted CID')
                    seen.add(r['cid'])
                    qty = quantities.pop(r['cid'], Decimal(0))
                    submitted = submitted_quantity(r['qty'], s.get('identity', {}).get('rules', {}).get('step'))
                    if qty < 0 or qty > submitted:
                        raise ValueError('Recorded fill exceeds submitted order quantity')
                    orders['submitted'] += 1
                    key = 'no_observed_fill' if qty == 0 else 'fully_filled' if qty == submitted else 'partially_filled'
                    orders[key] += 1
            if quantities or orders['submitted'] != s['orders'].get('new_requests', 0):
                raise ValueError('Submitted orders/fill identities do not reconcile')
            n = orders['submitted']
            orders['any_fill_submission_rate'] = (n-orders['no_observed_fill'])/n if n else None
        rows = [r for r in markouts if r['case'] == cid]
        result[cid] = dict(
            strategy=case['strategy'], symbol=case['symbol'], pnl=float(s['pnl']),
            max_drawdown=float(s['max_drawdown']),
            realized=float(s['realized']), unrealized=float(s['unrealized']),
            funding=float(s['funding']), fees=float(s['fees']),
            total_notional=float(total['total_notional']) if total['total_notional'] is not None else None,
            daily_turnover=float(total['average_daily_turnover_multiple']) if total['average_daily_turnover_multiple'] is not None else None,
            terminal_position=float(s['position']),
            max_abs_sampled_position=float(np.max(np.abs(a['position']))) if len(a) else None,
            orders=orders, markouts=rows,
            coverage=[dict(utc_day=r['utc_day'], covered_seconds=r.get('covered_seconds'),
                           coverage_status=r.get('coverage_status', 'not_recorded')) for r in daily if r['case'] == cid])
    return result


def module_effects(cases, stats, summaries):
    """Only compare masks with identical effective parameters and optional policies."""
    rows = []
    for symbol in dict.fromkeys(c['symbol'] for c in cases):
        group = [c for c in cases if c['symbol'] == symbol]
        for bit, module in ((1, 'R'), (2, 'F'), (4, 'I')):
            for view, before_mask, after_mask in (('add_one', 0, bit), ('remove_one', 7, 7 ^ bit)):
                before = [c for c in group if c['mask'] == before_mask]
                after = [c for c in group if c['mask'] == after_mask]
                for b in before:
                    for a in after:
                        if summaries[b['id']]['identity']['config'] != summaries[a['id']]['identity']['config']:
                            continue
                        if any(b.get(k) != a.get(k) for k in ('prediction', 'queue_management', 'quote_placement')):
                            continue
                        bs, ns = stats[b['id']], stats[a['id']]
                        rows.append(dict(symbol=symbol, view=view, module=module, before=b['strategy'], after=a['strategy'],
                            delta_pnl=ns['pnl']-bs['pnl'], delta_max_drawdown=ns['max_drawdown']-bs['max_drawdown'],
                            delta_total_notional=(ns['total_notional']-bs['total_notional']) if ns['total_notional'] is not None and bs['total_notional'] is not None else None,
                            interpretation='after minus before; descriptive paired difference, not causal significance'))
    return rows


def chart_conclusion(kind, stats, effects=()):
    parts = []
    for symbol in dict.fromkeys(s['symbol'] for s in stats.values()):
        group = [s for s in stats.values() if s['symbol'] == symbol]
        shortened = len(group) > 4
        if shortened:
            selected = [group[0], group[-1]]
            if kind == 'pnl':
                selected += [min(group, key=lambda r:r['pnl']), max(group, key=lambda r:r['pnl'])]
            group = list({r['strategy']:r for r in selected}.values())
        values = []
        for s in group:
            name = s['strategy']
            if kind == 'pnl':
                text = f"净PnL {s['pnl']:+.4f} USDC"
            elif kind == 'daily_volume':
                text = '成交额未记录' if s['total_notional'] is None else f"总成交额 {s['total_notional']:.2f} USDC"
            elif kind == 'turnover':
                text = '完整区间日均换手缺失或仅局部验证' if s['daily_turnover'] is None else f"日均换手 {s['daily_turnover']:.3f} 倍/日"
            elif kind == 'drawdown':
                text = f"账本最大回撤 {s['max_drawdown']:.4f} USDC"
            elif kind == 'inventory':
                maximum = s['max_abs_sampled_position']
                text = f"期末净仓位 {s['terminal_position']:+g}，采样绝对仓位峰值 " + (f'{maximum:g}' if maximum is not None else '缺失')
            elif kind == 'pnl_components':
                text = f"已实现 {s['realized']:+.4f}、未实现 {s['unrealized']:+.4f}、资金费 {s['funding']:+.4f}、手续费扣项 {s['fees']:+.4f} USDC"
            elif kind == 'order_outcomes':
                o = s['orders']
                text = '全部发单记录缺失，不能计算成交/未成交比例' if o is None else (
                    f"发单 {o['submitted']}，全成 {o['fully_filled']}、部分成 {o['partially_filled']}、未观察到成交 {o['no_observed_fill']}；"
                    + (f"有成交发单占比 {o['any_fill_submission_rate']:.2%}" if o['any_fill_submission_rate'] is not None else '无发单，比例缺失'))
            elif kind == 'coverage':
                recorded = [r for r in s['coverage'] if r['covered_seconds'] not in (None, '')]
                seconds = sum(float(r['covered_seconds']) for r in recorded)
                text = f"实际认证回放 {seconds/3600:.2f} 小时" if recorded else '认证覆盖时长未记录'
            elif kind in ('markouts', 'markout_coverage'):
                records = [r for r in s['markouts'] if r['observed'] is not None and r['censored'] is not None]
                if shortened:
                    records = [r for r in records if r['horizon_ms'] in (100,1000,60000)]
                if kind == 'markouts':
                    valid = [r for r in records if r['status'] == 'recorded' and r['notional_weighted_bp'] is not None]
                    text = '统一口径Markout缺失' if not valid else '、'.join(
                        f"{r['horizon_ms']/1000:g}s {float(r['notional_weighted_bp']):+.3f}bp" for r in valid)
                else:
                    text = '有效/删失样本未记录' if not records else '、'.join(
                        f"{r['horizon_ms']/1000:g}s 有效{r['observed']}/删失{r['censored']}" for r in records)
            else:
                continue
            values.append(f'{name}：{text}')
        if values:
            parts.append(symbol+('摘要选列（全部组及期限见图和统计表）' if shortened else '')+'；'+'；'.join(values)+'。')
            if len(group) > 1 and kind == 'pnl':
                first, last = group[0], next(s for s in reversed(list(stats.values())) if s['symbol'] == symbol)
                parts.append(f"{last['strategy']}较{first['strategy']}净PnL变化 {last['pnl']-first['pnl']:+.4f} USDC；"
                             + ('两者本期均亏损。' if first['pnl'] < 0 and last['pnl'] < 0 else '盈利情况以本期净值为准。'))
            elif len(group) > 1 and kind == 'daily_volume':
                first, last = group[0], group[-1]
                if first['total_notional'] is not None and first['total_notional'] > 0 and last['total_notional'] is not None:
                    parts.append(f"{last['strategy']}成交额较{first['strategy']}变化 {(last['total_notional']/first['total_notional']-1)*100:+.2f}%；收益变化需结合参与量判断。")
    if kind == 'ablation':
        parts = [f"{r['symbol']} {r['view']} {r['module']}（{r['before']}→{r['after']}）净PnL变化 {r['delta_pnl']:+.4f} USDC。" for r in effects]
        if not parts:
            parts = ['没有参数和附加政策一致的配对消融组，本报告不推断模块贡献。']
        parts.append('正负号表示切换后的变化；去除模块为负意味着该模块在此组合中有帮助。加法与减法结果不必一致，不能据此声称统计显著。')
    qualifications = {
        'pnl': '包含期末未实现盈亏；单个区间的排序不代表长期正期望。',
        'turnover': '分母为初始资金，买卖成交额相加、不除2；缺口不外推。',
        'drawdown': '图为约1秒权益采样的回撤，摘要为账本事件统计，两者可能不同。',
        'inventory': '单位为各合约报价数量，币种之间不直接比较；期末未假设清仓。',
        'order_outcomes': '按唯一CID和截至回放末尾累计成交分类；包含拒单、待确认及边界截尾，不等于已接受订单的最终成交概率，更不是交易胜率。',
        'markouts': '正值表示成交方向有利，未扣手续费和资金费；必须结合样本覆盖与PnL判断。',
        'markout_coverage': '按成交回报条数计数，不是独立订单数；缺口、期末和超预算滞后均删失，不能补零。',
        'coverage': '零成交只对已覆盖时段有意义；未认证时段不能当成零收益。',
    }
    return ' '.join(parts+[qualifications[kind]] if kind in qualifications else parts)


CHARTS = (
    ('pnl', '净PnL曲线'), ('drawdown', '回撤曲线'),
    ('daily_volume', '每日成交额'), ('turnover', '每日换手'),
    ('inventory', '净仓位曲线'), ('order_outcomes', '发单与成交情况'),
    ('markouts', '多期限Markout'), ('markout_coverage', 'Markout有效样本与删失'),
    ('pnl_components', 'PnL构成'), ('coverage', '每日实际数据覆盖'),
    ('ablation', '模块加法与减法对照'),
)


def save_coin_panels(out, fig, axes, kind):
    """Export complete subplots from one rendered canvas."""
    from math import ceil, floor
    from PIL import Image

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = Image.fromarray(np.asarray(renderer.buffer_rgba())).convert('RGB')
    width, height = canvas.size
    for index, ax in enumerate(axes):
        bounds = ax.get_tightbbox(renderer).padded(8)
        box = (max(0, floor(bounds.x0)), max(0, floor(height-bounds.y1)),
               min(width, ceil(bounds.x1)), min(height, ceil(height-bounds.y0)))
        canvas.crop(box).save(out / f'{kind}-coin-{index}.png')


def chart_sections(out, cases, summaries, curves, totals, daily, markouts, full_range,
                   frozen_statistics=None, frozen_effects=None):
    """Write audit tables, draw additional charts, return adjacent Markdown blocks."""
    stats = (chart_statistics(cases, summaries, curves, totals, daily, markouts)
             if frozen_statistics is None else {c["id"]:frozen_statistics[c["id"]] for c in cases})
    effects = module_effects(cases, stats, summaries) if frozen_effects is None else frozen_effects
    write_json(out/'chart_statistics.json', stats)
    write_csv(out/'ablation_effects.csv', effects, ['symbol','view','module','before','after','delta_pnl','delta_max_drawdown','delta_total_notional','interpretation'])
    configure_plotting()
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import timezone
    symbols = list(dict.fromkeys(c['symbol'] for c in cases))
    strategies = list(dict.fromkeys(c['strategy'] for c in cases))
    for kind, title in CHARTS:
        if kind in ('pnl', 'daily_volume', 'markouts'):
            continue  # Existing audited drawing code preserves segment envelopes.
        fig, axes = plt.subplots(len(symbols), 1, figsize=(13, 5*len(symbols)), squeeze=False)
        for ax, symbol in zip(axes[:, 0], symbols):
            group = [c for c in cases if c['symbol'] == symbol]
            available = False
            for j, case in enumerate(group):
                cid = case['id']; s = stats[cid]; a = curves[cid]
                color = plt.get_cmap('tab10')(strategies.index(case['strategy']) % 10)
                if kind in ('drawdown', 'inventory') and len(a):
                    y = (a['equity'] - np.maximum.accumulate(np.r_[float(summaries[cid]['identity']['config']['initial_cash']), a['equity']])[1:]) if kind == 'drawdown' else a['position']
                    # Draw segments separately; never interpolate over a gap.
                    for k, seg in enumerate(summaries[cid]['segments']):
                        ix = np.flatnonzero((a['local_ns'] >= seg['start_ns']) & (a['local_ns'] <= seg['end_ns']))
                        # Time-bin extrema envelope keeps peaks and bounds chart size.
                        if len(ix):
                            bins = a['local_ns'][ix]//300_000_000_000
                            starts = np.r_[0, np.flatnonzero(np.diff(bins))+1]; ends = np.r_[starts[1:], len(ix)]
                            keep = sorted({int(ix[v]) for l, r in zip(starts, ends) for v in (l, r-1, l+np.argmin(y[ix[l:r]]), l+np.argmax(y[ix[l:r]]))})
                            ax.plot(a['local_ns'][keep]/1e9/86400, y[keep], color=color, label=case['strategy'] if k == 0 else None, lw=1)
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M', tz=timezone.utc))
                    ax.margins(y=.08)
                    ax.set_ylabel('采样回撤（USDC，负值）' if kind == 'drawdown' else '净仓位（合约报价数量）'); available = True
                elif kind in ('turnover', 'coverage'):
                    key = 'turnover_initial_cash_multiple' if kind == 'turnover' else 'covered_seconds'
                    rows = [r for r in daily if r['case'] == cid and r.get(key) not in (None, '')]
                    if rows:
                        dates = sorted({r['utc_day'] for r in daily if r['symbol'] == symbol})
                        width = .8/len(group)
                        ax.bar(np.array([dates.index(r['utc_day']) for r in rows])+(j-(len(group)-1)/2)*width,
                               [float(r[key])/(3600 if kind == 'coverage' else 1) for r in rows], width=width, color=color, label=case['strategy'])
                        ax.set_xticks(range(len(dates)), [d[5:] for d in dates]); available = True
                    ax.set_ylabel('换手（初始资金倍数/日）' if kind == 'turnover' else '实际覆盖小时')
                elif kind == 'markout_coverage':
                    rows = [r for r in markouts if r['case'] == cid and r['observed'] is not None and r['censored'] is not None]
                    if rows:
                        y = [int(r['observed'])/(int(r['observed'])+int(r['censored'])) if int(r['observed'])+int(r['censored']) else np.nan for r in rows]
                        ax.plot(range(len(rows)), y, marker='o', color=color, label=case['strategy'])
                        ax.set_xticks(range(len(rows)), [f"{r['horizon_ms']/1000:g}s" for r in rows]); available = True
                    ax.set_ylim(0, 1.05); ax.set_ylabel('有效样本占比（0—1）')
            if kind == 'pnl_components':
                x = np.arange(len(group)); width = .18
                for j, (key, label, sign) in enumerate((('realized','已实现',1),('unrealized','未实现',1),('funding','资金费',1),('fees','手续费对PnL贡献',-1))):
                    ax.bar(x+(j-1.5)*width, [stats[c['id']][key]*sign for c in group], width=width, label=label)
                ax.set_xticks(x, [c['strategy'] for c in group], rotation=20); ax.set_ylabel('USDC'); available = True
            elif kind == 'order_outcomes':
                x = np.arange(len(group)); bottom = np.zeros(len(group))
                for key, label in (('fully_filled','全成'),('partially_filled','部分成交'),('no_observed_fill','未观察到成交')):
                    y = np.array([stats[c['id']]['orders'][key] if stats[c['id']]['orders'] is not None else np.nan for c in group])
                    ax.bar(x, y, bottom=bottom, label=label); bottom += np.nan_to_num(y)
                    available |= bool(np.isfinite(y).any())
                ax.set_xticks(x, [c['strategy'] for c in group], rotation=20); ax.set_ylabel('提交订单数（唯一CID）')
            elif kind == 'ablation':
                rows = [r for r in effects if r['symbol'] == symbol]
                if rows:
                    ax.bar(range(len(rows)), [r['delta_pnl'] for r in rows], color=['#2566a6' if r['view'] == 'add_one' else '#c46818' for r in rows])
                    ax.set_xticks(range(len(rows)), [f"{r['view']} {r['module']}" for r in rows], rotation=20); available = True
                ax.set_ylabel('切换后−切换前 净PnL（USDC）')
            if not available:
                ax.text(.5, .5, '缺少可比组或原始观测；不补零、不推算', transform=ax.transAxes, ha='center')
            elif kind != 'ablation':
                ax.legend(frameon=False, fontsize=8, ncol=2, **(
                    dict(loc='lower center', bbox_to_anchor=(.5, 1.02)) if kind == 'inventory' else {}))
            ax.set_title(symbol+(' · 局部验证' if not full_range else ''), pad=38 if kind == 'inventory' else 6); ax.grid(axis='y', alpha=.2)
        fig.suptitle(title, y=1-.12/fig.get_figheight())
        fig.tight_layout(rect=(0, 0, 1, 1-.45/fig.get_figheight()))
        fig.savefig(out/(kind+'.png'), dpi=150)
        save_coin_panels(out, fig, axes[:, 0], kind)
        plt.close(fig)
    lines = []
    for kind, title in CHARTS:
        lines += ['', '## '+title, '', f'![{title}]({kind}.png)', '',
                  ('**局部验证，以下仅描述已回放时段。** ' if not full_range else '')+chart_conclusion(kind, stats, effects)]
    return lines
