import csv
from decimal import Decimal as D
import json

import pytest
from app.backtesting.experiment_metrics import ReplayMetrics
from app.market_making.fusion.tests.quote_fixture import QuoteFixture


def spec(horizons=(100, 500, 1000, 5000, 10000, 30000), lag=250):
    return dict(timezone='UTC', markout_horizons_ms=list(horizons), max_markout_lag_ms=lag,
                markout_anchor='local_fill_receipt', markout_reference='first_observed_mid_at_or_after', retain_midpoints=True)


def fill(ns=0, side=1, price='100', qty='2', fee='0'):
    return dict(ns=ns, side=side, price=D(price), qty=D(qty), fee=D(fee), cid='order-1')


def summary(count=1, position='2', fees='0', cash='-200', segments=None):
    return dict(orders={'fill_events':count}, position=position, fees=fees, funding='0',
                cash_flow=cash, full_range=False, native_fills=count,
                segments=segments or [dict(start_ns=0, end_ns=60_000_000_000)])


@pytest.mark.parametrize('side,expected', [(1, '10'), (-1, '-10')])
def test_markout_sign_and_receipt_clock(tmp_path, side, expected):
    with ReplayMetrics(tmp_path, spec((100,), 20), '1970-01-01', '1970-01-02') as m:
        m.fill(fill(500_000, side=side))
        m.mid(100_499_999, D('105'))  # Earlier than receipt + horizon, not usable.
        m.mid(110_500_000, D('100.1'))
        result=m.finish(summary(position=str(side*2), cash=str(-side*200)))
    row=result['markouts'][0]
    assert D(row['notional_weighted_bp']) == D(expected)
    assert row['observed']==1 and row['censored']==0 and row['mean_lag_ns']==10_000_000
    detail=json.loads((tmp_path/'markouts.jsonl').read_text())
    assert detail['target_ns']==100_500_000


def test_volume_counts_deltas_once_and_reconciles_buy_sell_cash(tmp_path):
    with ReplayMetrics(tmp_path,spec(),'1970-01-01','1970-01-03') as m:
        m.fill(fill(qty='2',fee='.02'))
        m.fill(fill(ns=1,side=-1,price='102',qty='.5',fee='.0051'))
        m.mid(31_000_000_000,D('101'))
        r=m.finish(summary(2,'1.5','.0251','-149.0251'))
    assert D(r['volume']['total_notional'])==251
    assert D(r['volume']['total_qty'])==D('2.5')
    assert len((tmp_path/'fills.jsonl').read_text().splitlines())==2
    assert all(x['observed']==0 and x['censored']==2 for x in r['markouts'])
    with (tmp_path/'daily_volume.csv').open() as f:days=list(csv.DictReader(f))
    assert D(days[0]['total_notional'])==251
    assert days[1]['total_notional']=='' and days[1]['coverage_status']=='not_replayed'


def test_weighted_markout_is_not_average_of_fill_means(tmp_path):
    with ReplayMetrics(tmp_path,spec((100,)),'1970-01-01','1970-01-02') as m:
        m.fill(fill(qty='3'))
        m.fill(fill(side=-1,qty='1'))
        m.mid(100_000_000,D('100.1'))
        r=m.finish(summary(2,'2','0','-200'))
    assert D(r['markouts'][0]['notional_weighted_bp'])==5


def test_sixty_second_horizon_waits_for_deadline_and_censors_boundary(tmp_path):
    with ReplayMetrics(tmp_path,spec((60000,)),'1970-01-01','1970-01-02') as m:
        m.fill(fill())
        m.mid(59_999_999_999,D('999'))  # Must not use a price before the horizon.
        m.mid(60_000_000_000,D('100.1'))
        m.fill(fill(ns=60_000_000_001))
        m.gap()
        result=m.finish(summary(2,'4','0','-400'))
    row=result['markouts'][0]
    assert row['observed']==1 and row['censored']==1
    assert D(row['notional_weighted_bp'])==10


def test_late_and_gap_prices_are_censored_never_replaced_with_zero(tmp_path):
    with ReplayMetrics(tmp_path,spec((100,500),10),'1970-01-01','1970-01-02') as m:
        m.fill(fill())
        m.mid(111_000_000,D('101'))
        m.gap()
        m.mid(600_000_000,D('999'))
        r=m.finish(summary())
    assert all(x['notional_weighted_bp'] is None for x in r['markouts'])
    assert all(x['observed']==0 and x['censored']==1 for x in r['markouts'])
    assert [json.loads(x)['status'] for x in (tmp_path/'markouts.jsonl').read_text().splitlines()]==['late_mid','segment_boundary']


def test_volume_uses_receipt_utc_day_at_midnight(tmp_path):
    with ReplayMetrics(tmp_path,spec((100,)),'1970-01-01','1970-01-03') as m:
        m.fill(fill(86_399_999_999_999,qty='1'))
        m.fill(fill(86_400_000_000_000,qty='1'))
        m.finish(summary(2,segments=[dict(start_ns=0,end_ns=172_800_000_000_000)]))
    with (tmp_path/'daily_volume.csv').open() as f:rows=list(csv.DictReader(f))
    assert [D(r['total_notional']) for r in rows]==[100,100]


def test_wrong_ledger_total_rejects_publication(tmp_path):
    with ReplayMetrics(tmp_path,spec(),'1970-01-01','1970-01-02') as m:
        m.fill(fill())
        with pytest.raises(AssertionError,match='inventory'):
            m.finish(summary(position='3'))
    assert not (tmp_path/'metrics.json').exists()


def test_partial_and_duplicate_oms_reports_count_only_new_quantity(tmp_path):
    from dataclasses import replace
    from app.market_making.as_math import ASConfig
    from app.market_making.core import Rules
    from app.market_making.fusion.config import FusionConfig,Factors
    from app.market_making.fusion.kernel import Kernel
    from app.market_making.contracts import Desired
    rules=Rules(D('.01'),D('.1'),D(5),D('.1'),D(10000),D('.9'),D('1.1'))
    config=replace(FusionConfig(),as_config=replace(ASConfig(),maker_fee=D(0)))
    k=Kernel(rules,config,Factors(), strategy_factory=QuoteFixture);k.mark_price(0,D(100))
    cmd=k.oms.reconcile_targets({1:Desired(1,D(100),D(1))},D(100),0)[0]
    with ReplayMetrics(tmp_path,spec(),'1970-01-01','1970-01-02') as m:
        for ns,status,cumulative in [(1,'PARTIALLY_FILLED',D('.4')),(2,'PARTIALLY_FILLED',D('.4')),(3,'FILLED',D(1))]:
            f=k.update(cmd.cid,status,cumulative,cumulative*100,ns)
            if f:m.fill(f)
        s=k.report();s.update(full_range=False,native_fills=2,segments=[dict(start_ns=0,end_ns=1_000_000_000)])
        r=m.finish(s)
    assert D(r['volume']['total_notional'])==100
    assert r['oms_fill_receipts']==2
    assert [D(json.loads(line)['qty']) for line in (tmp_path/'fills.jsonl').read_text().splitlines()]==[D('.4'),D('.6')]


@pytest.mark.parametrize('bad', [[100,100],[0],[True],[]])
def test_invalid_horizons_fail_before_opening_files(tmp_path,bad):
    with pytest.raises(ValueError):
        ReplayMetrics(tmp_path,spec(bad),'1970-01-01','1970-01-02')
    assert not (tmp_path/'fills.jsonl').exists()
