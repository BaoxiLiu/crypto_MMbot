"""Build a metadata inventory without importing legacy numerical backends.

This does not compute factors, infer exact aliases, or change model columns.
Run from any directory; --check verifies the committed/generated inventory.
"""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'doc/factors/unified-v1'
CATEGORIES = {
    'book_structure': '盘口结构',
    'book_flow': '盘口动态与订单流',
    'trade_activity': '主动成交与活跃度',
    'trade_distribution': '成交分布与量时钟',
    'price_path': '价格路径与动量反转',
    'volatility': '波动与尾部风险',
    'liquidity': '流动性与交易成本',
    'technical': '技术指标',
    'cross_market': '跨市场与时间环境',
}
COLLECTIONS = {
    'v3': ('原始预测因子', 50),
    'flow': ('流速扩展', 27),
    'micro': ('微观结构', 202),
    'states': ('盘面诊断增补', 53),
    'minute': ('分钟指标', 18),
    'paper': ('反转论文参考实现', 173),
    'cicc': ('中金公式候选', 79),
}


def read_json(rel):
    return json.loads((ROOT / rel).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paper_names():
    """Use the validated extracted schema; paper full text is not a dependency."""
    from app.factor_library.paper173_schema import load_names
    return list(load_names())


def literal_assignment(rel, name):
    tree = ast.parse((ROOT / rel).read_text())
    return next(ast.literal_eval(n.value) for n in tree.body
                if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))


def v3_definitions():
    """Keep grouped formula descriptions as references, expand stable IDs only."""
    rel = 'doc/prediction/V3_SPEC.md'
    definitions = {}
    for line_no, line in enumerate((ROOT / rel).read_text().splitlines(), 1):
        if not re.match(r'^\| [ABCDE]\d', line):
            continue
        # Markdown cells escape literal vertical bars in formula expressions.
        cells = [x.strip() for x in re.split(r'(?<!\\)\|', line)[1:-1]]
        ids, name, formula, window = cells
        match = re.fullmatch(r'([ABCDE])(\d+)[–—-]([ABCDE])?(\d+)', ids)
        if match:
            group, lo, _, hi = match.groups()
            keys = [f'{group}{i}' for i in range(int(lo), int(hi)+1)]
        else:
            keys = ids.split('/')
        for key in keys:
            definitions[key] = (name, formula, window, f'{rel}:{line_no}')
    expected = [f'{group}{i}' for group, count in [('A',8),('B',6),('C',10),('D',15),('E',11)]
                for i in range(1, count+1)]
    if set(definitions) != set(expected):
        raise ValueError('V3 specification IDs do not partition the 50-factor schema')
    return [(key, *definitions[key]) for key in expected]


def category(name, collection, group=None):
    """One primary category per variant; source taxonomy remains separate."""
    n = name.lower()
    if collection == 'v3':
        if group == 'E': return 'cross_market'
        if group == 'A': return 'liquidity' if name == 'A8' else ('book_flow' if name in ('A6','A7') else 'book_structure')
        if group == 'B': return 'book_flow'
        if group == 'C': return 'liquidity' if name == 'C10' else ('trade_distribution' if name in ('C5','C6','C9') else 'trade_activity')
        return 'book_flow' if name in ('D4','D5') else ('volatility' if name in ('D6','D7','D8','D9','D10','D11','D12','D15') else 'price_path')
    if 'rsi' in n: return 'technical'
    if any(s in n for s in ('depletion','spread','amihud')) or n in ('ob_half','ob_other_half','age','totb_mean'): return 'liquidity'
    if any(s in n for s in ('volatility','realized_vol','stdev','rv_','fragility','range_relative')) or n == 'range1_bp': return 'volatility'
    if any(s in n for s in ('bucket','quota','max_size','avg_size','size_ratio','interval_cv','completed_volume')): return 'trade_distribution'
    if any(s in n for s in ('ofi','refill','replenish','net_add','net_decrease','net_flow','both_decrease','both_add','bbo_change','delta_','change_fraction')) or re.search(r'_(add|decrease)_ticks',n): return 'book_flow'
    if any(s in n for s in ('return','ret_sum','ret_autocov','ret_vwap','microprice','close_minus','body','amplitude')): return 'price_path'
    if any(s in n for s in ('notional_l','imbalance_l','touch_share','touch_back','uniformity','hhi','outer_share','symmetry','width_ticks','touch_qty_imbalance','near_depth_imbalance')) or n in ('log_touch_total_notional','ob_bid_liq','ob_ask_liq'): return 'book_structure'
    if 'touch_qty_change' in n or 'depth_message' in n or 'depth_batch' in n or 'depth_row' in n: return 'book_flow'
    if any(s in n for s in ('trade','aggressor','taker','volume','total_buy','total_sell','buy_count','sell_count','multilevel_print')): return 'trade_activity'
    raise ValueError(f'Unclassified feature: {collection}/{name}')


def family(name, collection):
    """Parameter-normalized grouping WITHIN a source, never exact cross-source aliases."""
    n = re.sub(r'^w\d+ms_', '', name)
    n = re.sub(r'_(100ms|\d+s)_w[012]$', '', n)
    n = re.sub(r'_l\d+$', '_l{levels}', n)
    n = re.sub(r'ticks\d+_\d+', 'ticks{band}', n)
    n = re.sub(r'\b(bid|ask|buy|sell)\b', '{side}', n)  # underscores need explicit token matching
    n = re.sub(r'(^|_)(bid|ask|buy|sell)(?=_|$)', r'\1{side}', n)
    return collection + ':' + n


def build():
    rows, sources = [], set()

    def add(collection, name, implementation, definition, *, formula=None, window=None,
            unit=None, status='implemented_source', inputs='public_l2_and_trades',
            clock='LOCAL receipt time; backend-specific alignment', primary=None,
            notes='', source_id=None, group=None):
        for ref in (implementation, definition):
            if ref:
                sources.add(ref.split(':')[0])
        params = {}
        m = re.match(r'^w(\d+)ms_', name)
        if m: params['window_ms'] = int(m[1])
        m = re.search(r'_(100ms|\d+s)_w([012])$', name)
        if m: params.update(scale=m[1], historical_segment=int(m[2]))
        m = re.search(r'_l(\d+)$', name)
        if m: params['levels'] = int(m[1])
        row = dict(id=f'{collection}:{name}', collection=collection, name=name,
                   source_id=source_id, category=primary or category(name, collection, group),
                   family_key=family(name, collection), parameters=params,
                   formula=formula, formula_status='documented' if formula else 'source_reference',
                   formula_reference=definition, implementation=implementation,
                   unit=unit, window=window, clock=clock, inputs=inputs,
                   status=status, role='candidate_market_feature' if collection=='cicc' else 'market_feature',
                   runtime_adapter='not_added_by_this_inventory',
                   predictive_status='not_reassessed; see source-specific evidence',
                   notes=notes)
        rows.append(row)
        return row

    # Existing frozen schema: source-specific clocks are explicitly not normalized.
    for key, name, formula, window, ref in v3_definitions():
        r = add('v3', key, 'app/prediction/factors.py', ref, formula=formula,
                window=window, group=key[0], inputs='book/trades; E1-E7 require cross-symbol feeds; E8-E9 require mark feed',
                clock='caller-supplied millisecond clock; 1s grid plus selected exact trade windows',
                notes='Grouped specification formula retained; see implementation interpretation choices. B* are net-L2 decrease proxies, not identified cancellations.')
        r['display_name'] = name
    sources.add('app/prediction/schema.py')

    rel = 'app/factor_library/registry.json'
    flow_categories = {
        **dict.fromkeys(['S01','S02','S03','S04','S05','S19','S20','S21'], 'book_flow'),
        **dict.fromkeys(['S06','S07','S08','S09','S22'], 'trade_activity'),
        **dict.fromkeys(['S10','S11','S12','S13','S14','S15','S24','S25','S26','S27'], 'trade_distribution'),
        'S16':'liquidity', 'S17':'book_structure', 'S18':'book_structure', 'S23':'price_path',
    }
    for f in read_json(rel)['factors']:
        r = add('flow', f['name'], 'app/factor_library/flow.py', rel,
                formula=f['formula'], unit=f['unit'], window=f['history'], source_id=f['id'],
                primary=flow_categories[f['id']],
                clock='source received_time, integer nanoseconds; source-specific atomic depth messages',
                notes=f['overlap_note'])
        r['predictive_status'] = f['predictive_status']

    rel = 'doc/factors/microstructure-direction-v1/registry.json'
    for f in read_json(rel)['factors']:
        # Numerical formulas remain at the actual implementation, not guessed from names.
        r = add('micro', f['name'], 'app/factor_library/microstructure.py',
                'app/factor_library/microstructure_core.py' if f['index'] < 133 else 'app/factor_library/microstructure.py',
                source_id=str(f['index']), notes='Fixed bid orientation; public net-L2 proxies. Formula units/warm-up remain backend-specific; no new numerical implementation.')
        r['registry_reference'] = rel
    sources.add(rel)

    rel = 'doc/factors/market-states-v1/registry.json'
    s = read_json(rel)
    for f in s['new_columns']:
        r = add('states', f['name'], 'app/factor_library/market_states.py', rel,
                formula=f['formula'], unit=f['unit'], window=f.get('window_ms'), source_id=f['id'],
                notes='Diagnostics only. ' + f['warmup'] + '; ' + f['reset'])
        r['predictive_status'] = 'diagnostic_only; not an adopted trading filter'
    reused = s['reused_columns']

    rel = 'app/factor_library/minute.py'
    minute_formulas = [
        '(RSI14-50)/50 * side', '(RSI7-50)/50 * side', '(RSI14[i]-RSI14[i-1])/50 * side',
        'I(RSI14<=30) for buy; I(RSI14>=70) for sell',
        '(C[i]/C[i-1]-1)*10000*side', '(C[i]/C[i-5]-1)*10000*side',
        '(C[i]/C[i-15]-1)*10000*side', 'abs((C[i]/C[i-1]-1)*10000)',
        '(H[i]-L[i])/C[i]*10000', '(C[i]/O[i]-1)*10000*side',
        '(C[i]/VWAP[i]-1)*10000*side', '(C[i]/mean(C[i-19:i+1])-1)*10000*side',
        'std(diff(log(C[i-19:i+1])),ddof=1)*10000',
        'range_bp[i]/mean(range_bp[i-19:i+1]); zero mean ->0',
        'log1p(base_volume[i])', 'base_volume[i]/mean(base_volume[i-19:i+1]); zero mean ->0',
        'log1p(trade_count[i])', '(2*buy_base_volume[i]/base_volume[i]-1)*side; empty volume ->0',
    ]
    for name, formula in zip(literal_assignment(rel, 'NAMES'), minute_formulas, strict=True):
        add('minute', name, rel, rel, formula=formula,
            clock='LOCAL trade receipts -> closed minute bars; close_ns<=decision_ns',
            inputs='closed LOCAL-receipt OHLCV/VWAP/trade_count/aggressor-volume bars',
            notes='RSI uses Wilder smoothing; side=+1 buy/-1 sell. Insufficient history NaN. Reset across gaps. Per-column empty-volume rules retained.')

    paper_impl = 'app/factor_library/paper173.py'
    paper_backend = 'app/factor_library/paper173_core.py'
    dependency = 'app/factor_library/paper173_schema.json'
    blocked = not (ROOT / dependency).exists()
    if not blocked: sources.add(dependency)
    paper_formulas = {
        'ret_autocov': 'mean((r[:-1]-mean(r))*(r[1:]-mean(r))); r=side*diff(log(trade_price))',
        'ret_sum': 'sum(side*diff(log(trade_price)))',
        'trade_intensity': 'mean(diff(trade_receipt_ns))/1e9; an interval, NOT trades/second',
        'amplitude': 'max(trade_price)-min(trade_price)',
        'ret_vwap': 'side*log(sum(price*qty)/sum(qty)/current_side_touch_price)',
        'max_size': 'max(trade_base_qty)', 'avg_size': 'mean(trade_base_qty)',
        'buy_count': 'count(order-same-side aggressor trades)',
        'sell_count': 'count(order-opposite-side aggressor trades)',
        'total_buy': 'sum(order-same-side aggressor base_qty)',
        'total_sell': 'sum(order-opposite-side aggressor base_qty)',
        'age': '(now-last_actual_BBO_price_change_ns)/1e9',
        'totb_mean': 'mean(BBO price change intervals ending within trailing 600s)',
        'ob_bid_liq': 'same-order-side touch base_qty', 'ob_ask_liq': 'opposite-order-side touch base_qty',
        'ob_half': 'same-side VWAP impact for 500000 quote notional, bp; insufficient book ->NaN',
        'ob_other_half': 'opposite-side VWAP impact for 500000 quote notional, bp; insufficient book ->NaN',
        'stdev_100': 'std(100 log returns of trailing 10s last-trade grid,ddof=1)',
        'stdev_500': 'std(500 log returns of trailing 10s last-trade grid,ddof=1)',
    }
    for name in paper_names():
        base = re.sub(r'_(100ms|\d+s)_w[012]$', '', name)
        row = add('paper', name, paper_impl, paper_backend, formula=paper_formulas[base],
                  status='blocked_missing_dependency' if blocked else 'implemented_source',
                  clock='LOCAL receipt nanoseconds; current and two nonoverlapping past segments',
                  notes='Reference implementation, not exact author source code. Names/order frozen to the validated runtime schema extracted from Appendix B; unavailable history/depth remains NaN.')
        row['name_source'] = dependency
        row['required_file'] = dependency
        row['required_file_present'] = not blocked
    sources.update(['app/factor_library/paper173_schema.py',
                    'doc/factors/PAPER173_CONTRACT.md'])

    rel = 'app/factor_library/cicc_candidates.json'
    mapping = {'动量反转':'price_path','波动性':'volatility','高阶特征':'volatility',
               '流动性':'liquidity','量价相关性':'price_path','筹码分布':'trade_distribution',
               '拥挤度':'trade_distribution','资金成交':'trade_activity'}
    for f in read_json(rel)['factors']:
        r = add('cicc', f['original_code'], None, rel, formula=f['formula'],
                status='formula_unresolved' if f['status']=='U' else 'formula_candidate',
                primary=mapping[f['category']], inputs=f['required_data'],
                clock='original A-share daily construction; crypto causal rolling adaptation NOT implemented',
                notes=f['notes'])
        r['original_status'] = f['status']
        r['original_category'] = f['category']
        r['candidate_formula'] = f['candidate_formula']
        r['formula_status'] = 'unresolved' if f['status']=='U' else 'public_definition_with_documented_conventions'
        r['predictive_status'] = 'not_evaluated_in_this_project'

    sources.update(['tools/factors/build_factor_catalog.py', 'app/factor_library/catalog.py'])
    counts = Counter(r['collection'] for r in rows)
    assert counts == {k:v[1] for k,v in COLLECTIONS.items()}, counts
    assert len(rows) == len({r['id'] for r in rows}) == 602
    assert len(reused) == 80
    micro = {r['name'] for r in rows if r['collection']=='micro'}
    assert all(n in micro for n in reused)
    fingerprints = {p:sha(ROOT/p) for p in sorted(sources)}
    data = dict(schema_version=1, catalog_version='1.1.0', date='2026-11-01',
                scope='metadata inventory, not unified numerical execution or proof of alpha',
                categories=CATEGORIES,
                collections={k:dict(label=v[0], count=counts[k]) for k,v in COLLECTIONS.items()},
                counts=dict(total_entries=602, historical_entries=523, external_candidates=79,
                            implemented_source=sum(r['status']=='implemented_source' for r in rows),
                            blocked=sum(r['status']=='blocked_missing_dependency' for r in rows),
                            candidate_defined=43, candidate_unresolved=36),
                category_counts=dict(Counter(r['category'] for r in rows)),
                historical_category_counts=dict(Counter(r['category'] for r in rows if r['collection']!='cicc')),
                exact_cross_source_deduplication='not certified; no automatic merge',
                family_grouping='source-local parameter grouping only; not independent economic signals',
                known_reuse=[dict(alias=f'states:{n}', target=f'micro:{n}',
                                  evidence='states BASE_NAMES selects the micro202 output unchanged') for n in reused],
                missing_sources=[dependency] if blocked else [],
                source_sha256=fingerprints, factors=rows)
    return data


def render(data):
    lines = ['# 统一因子目录 v1', '',
             '本目录由 `python tools/factors/build_factor_catalog.py` 从现有定义和已提取的因子schema生成；不导入旧数值后端，不运行回测。', '',
             '**523项历史登记 + 79项外部候选 = 602条目录记录，按9类管理。不能解释成602个可直接运行或独立有效的信号。**', '',
             f'状态：{data["counts"]["implemented_source"]}项有现存实现入口，{data["counts"]["blocked"]}项已知依赖阻断；中金43项为有计算口径的候选、36项原式未确定。paper173已恢复独立名称清单与计算入口；其计算测试不等于13币覆盖、预测或盈利验收。', '',
             '## 在项目中的位置', '',
             '`标准行情 → 因子计算 → 特征值 → 模型预测或策略规则 → 报价意图 → OMS/队列撮合 → 成交账本 → src/metrics分析`', '',
             '原始AS只需其价格、波动、库存等必要输入，可以直接使用原有波动估计器。因子库提供可选特征，不要求原始AS运行全部523项。模型负责预测，策略负责决定怎么报价，执行模块负责订单；因子库不负责下单或计算交易PnL。', '',
             '未来Markout/盈亏是标签和结果；库存、自有排位是私有策略状态。两者不混进公共行情因子清单。撤量前后50/50、费用和敞口属于执行/风控设置，不改变公共L2净减少因子的定义。', '',
             '## 来源与计数', '', '| 集合 | 登记数量 |', '|---|---:|']
    for c in data['collections'].values(): lines.append(f'| {c["label"]} | {c["count"]} |')
    lines += ['', '盘面诊断原输出133列中80列直接复用micro202，已作为别名登记，不再计入523；早期9/15/136、202+18=220、202+173=375等历史子集/组合不重复计数。跨来源同名/近义尚未证明同定义，不强行合并。', '',
              '## 主分类', '', '| 分类 | 历史登记 | 外部候选 | 合计 |', '|---|---:|---:|---:|']
    for key, label in CATEGORIES.items():
        n = data['category_counts'].get(key,0); old = data['historical_category_counts'].get(key,0)
        lines.append(f'| {label} | {old} | {n-old} | {n} |')
    lines += ['', '一项只计入一个主类，便于总量对账；来源原分类仍保留。买卖镜像、档位、窗口、历史段通过family_key/parameters关联；family_key只是来源内部归组，不是经过相关性或经济独立性检验的去重结果。', '',
              '## 标准化边界与当前缺口', '',
              '- 已统一：稳定的命名空间ID、9类主分类、来源/公式引用、窗口参数、时钟说明、实现/候选/阻断状态、来源SHA256和查询接口。',
              '- 尚未统一：旧后端的输入结构、数值计算与缓存接口。旧50使用调用方毫秒时钟和1秒网格；flow使用其来源接收时钟；micro/states/minute/paper使用LOCAL可见时钟，不能按相同整数时间直接拼接。',
              '- micro202的公式继续引用现有实现，未由名称猜写全套公式。缺失和有效零值规则保留逐后端差异；不能统一填0。',
              '- paper173已解除报告CSV依赖：名称/顺序在`app/factor_library/paper173_schema.json`，公开计算入口为`app.factor_library.paper173`。原文全文已删除；schema保留名称/顺序、出处和原文哈希，不再读取报告目录。NumPy参考计算不保证实时100us预算。',
              '- 中金79为公式材料，43项仍包含工程约定和适配要求；36项不能唯一还原。不会被默认查询当成已实现特征。',
              '- 同族重叠只说明需要做增量比较；成交数量与成交金额、末笔成交价与mid、真实撤单与净L2减少、时间桶与等额桶都不得互相冒名替代。', '',
              '## 使用目录接口', '', '```python',
              'from app.factor_library.catalog import lookup, select, verify_sources',
              "spec = lookup('minute:side_rsi14_centered')",
              "items = select(category='book_flow')  # 默认排除候选和已知阻断项",
              "paper = select(collection='paper', include_blocked=True)",
              'verify_sources()  # 来源已变而未重建目录时抛错', '```', '',
              '这是元数据查询接口，不是统一的数值计算函数；接入新回测仍需显式选择后端并完成可见时间/预热/缺失适配。', '',
              '## 全部记录', '', '| ID | 主类 | 状态 | 公式/定义来源 |', '|---|---|---|---|']
    for row in data['factors']:
        lines.append(f'| `{row["id"]}` | {CATEGORIES[row["category"]]} | {row["status"]} | `{row["formula_reference"]}` |')
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    data = build()
    files = {ROOT/'app/factor_library/catalog.json':json.dumps(data, ensure_ascii=False, indent=2)+'\n', OUT/'CATALOG.md':render(data)}
    if args.check:
        for name, content in files.items():
            if not name.exists() or name.read_text() != content:
                raise SystemExit(f'Catalog stale or missing: {name}; rebuild explicitly')
    else:
        OUT.mkdir(parents=True, exist_ok=True)
        for name, content in files.items(): name.write_text(content, encoding='utf-8')
    print(json.dumps(dict(mode='check' if args.check else 'build', **data['counts']), ensure_ascii=False))


if __name__ == '__main__':
    main()
