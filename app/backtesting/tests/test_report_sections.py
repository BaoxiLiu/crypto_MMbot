import gzip
import json

import numpy as np
import pytest

from app.backtesting.report_sections import (
    CHARTS, chart_conclusion, chart_sections, chart_statistics, module_effects,
)


def fixture(tmp_path):
    cases, summaries, curves, totals, daily, markouts = [], {}, {}, [], [], []
    for mask in range(8):
        cid = f'AS{mask}'
        p = tmp_path/cid
        p.mkdir()
        cases.append(dict(id=cid, strategy=cid, symbol='DOGEUSDC', mask=mask, output=str(p)))
        summaries[cid] = dict(identity={'config': {'initial_cash':'600'}}, pnl=str(mask-4),
            realized=str(mask-3), unrealized='-2', funding='2', fees='1', max_drawdown='4',
            position='1', orders={'new_requests':3}, segments=[dict(start_ns=0, end_ns=1000000000, pnl=mask-4)])
        curves[cid] = np.array([(0,600,0),(1000000000,596+mask,1)], dtype=[('local_ns','i8'),('equity','f8'),('position','f8')])
        totals.append(dict(case=cid,total_notional='300',average_daily_turnover_multiple='.05'))
        daily.append(dict(case=cid,symbol='DOGEUSDC',utc_day='2026-08-01',covered_seconds='1',coverage_status='partial',total_notional='300',turnover_initial_cash_multiple='.5'))
        for h in (100,500,1000,5000,10000,30000,60000):
            markouts.append(dict(case=cid,symbol='DOGEUSDC',horizon_ms=h,status='recorded',observed=1,censored=2,notional_weighted_bp='-1'))
    return cases, summaries, curves, totals, daily, markouts


def test_add_and_remove_effects_have_opposite_reference_directions(tmp_path):
    cases, summaries, curves, totals, daily, markouts = fixture(tmp_path)
    stats = chart_statistics(cases, summaries, curves, totals, daily, markouts)
    rows = module_effects(cases, stats, summaries)
    assert len(rows) == 6
    assert [r['delta_pnl'] for r in rows] == [1,-1,2,-2,4,-4]
    assert stats['AS0']['orders'] is None
    assert '缺失' in chart_conclusion('order_outcomes', stats)
    summaries['AS1']['identity']['config']['initial_cash'] = '1000'
    assert len(module_effects(cases, stats, summaries)) == 5


def test_order_outcomes_use_cids_and_exact_cumulative_quantities(tmp_path):
    args = fixture(tmp_path)
    p = tmp_path/'AS0'
    with gzip.open(p/'order_placements.jsonl.gz','wt') as f:
        for cid in ('a','b','c'):
            f.write(json.dumps(dict(cid=cid,qty='0.3'))+'\n')
    (p/'fills.jsonl').write_text('\n'.join(json.dumps(dict(cid=cid,qty=qty)) for cid,qty in [('a','0.1'),('a','0.2'),('b','0.1')]))
    o = chart_statistics(*args)['AS0']['orders']
    assert o == dict(submitted=3,no_observed_fill=1,partially_filled=1,fully_filled=1,any_fill_submission_rate=2/3)
    with (p/'fills.jsonl').open('a') as f:
        f.write('\n'+json.dumps(dict(cid='unknown',qty='1')))
    with pytest.raises(ValueError,match='identities'):
        chart_statistics(*args)


def test_report_has_eleven_charts_each_immediately_followed_by_conclusion(tmp_path):
    import matplotlib
    matplotlib.use('Agg')
    args = fixture(tmp_path)
    lines = chart_sections(tmp_path, *args, full_range=False)
    images = [i for i, line in enumerate(lines) if line.startswith('![')]
    assert len(images) == 11
    for index in images:
        assert lines[index+1] == ''
        assert lines[index+2].startswith('**局部验证')
    assert all((tmp_path/(kind+'.png')).exists() for kind,_ in CHARTS if kind not in ('pnl','daily_volume','markouts'))
    text = '\n'.join(lines)
    assert '60s -1.000bp' in text
    assert '不是交易胜率' in text
    assert len(json.loads((tmp_path/'chart_statistics.json').read_text())) == 8


def test_missing_and_zero_markout_samples_are_not_invented(tmp_path):
    args = fixture(tmp_path)
    args[-1][0].update(observed=0,censored=0,notional_weighted_bp=None)
    stats = chart_statistics(*args)
    assert '0.1s 有效0/删失0' in chart_conclusion('markout_coverage',stats)
    assert '0.1s +0.000bp' not in chart_conclusion('markouts',stats)
