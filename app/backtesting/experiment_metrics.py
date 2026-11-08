"""Observation-only trade/volume/markout records. Never produces strategy inputs."""
from collections import Counter, deque
import csv
from datetime import datetime, timedelta, timezone
from decimal import Decimal as D
import gzip
import json
from pathlib import Path


def utc_day(ns):
    return datetime.fromtimestamp(ns // 1_000_000_000, timezone.utc).date().isoformat()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows, fields):
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


class ReplayMetrics:
    """Anchor to observed execution receipts, not reconstructed exchange fills.

    An OMS delta may coalesce native executions. Both its quantity and notional
    are exact; its count is a receipt count, not a native matching count.
    """
    additional_sources = ('app/backtesting/experiment_metrics.py',)

    def __init__(self, output, spec, start, end_exclusive):
        self.output = Path(output)
        self.spec = dict(spec)
        if spec.get('timezone') != 'UTC' or spec.get('markout_anchor') != 'local_fill_receipt' or spec.get('markout_reference') != 'first_observed_mid_at_or_after':
            raise ValueError('Unsupported observation clock/reference')
        hs = spec['markout_horizons_ms']
        if not hs or any(type(h) is not int or h <= 0 for h in hs) or len(set(hs)) != len(hs):
            raise ValueError('Unique positive integer markout horizons required')
        lag = spec['max_markout_lag_ms']
        if type(lag) is not int or lag < 0:
            raise ValueError('Invalid markout lag budget')
        self.pending = {h: deque() for h in sorted(hs)}
        self.lag_ns = lag * 1_000_000
        self.daily = {}
        day = datetime.fromisoformat(start)
        end = datetime.fromisoformat(end_exclusive)
        while day < end:
            self.daily[day.date().isoformat()] = self.empty_volume()
            day += timedelta(days=1)
        self.total = self.empty_volume()
        self.aggregates = {}
        self.fills_count = 0
        self.last_fill_ns = -1
        self.last_mid_ns = -1
        self.closed = False
        self.fills_file = (self.output / 'fills.jsonl').open('w')
        self.markouts_file = (self.output / 'markouts.jsonl').open('w')
        self.mid_file = gzip.open(self.output / 'midpoints.csv.gz', 'wt', compresslevel=1) if spec.get('retain_midpoints') else None
        if self.mid_file:
            self.mid_file.write('local_ns,mid\n')

    @staticmethod
    def empty_volume():
        return dict(fill_receipts=0, buy_qty=D(0), sell_qty=D(0), buy_notional=D(0), sell_notional=D(0), fees=D(0))

    def fill(self, fill):
        ns = fill['ns']
        price, qty, fee = D(fill['price']), D(fill['qty']), D(fill['fee'])
        if type(ns) is not int or ns < self.last_fill_ns or fill['side'] not in (-1, 1) or not price.is_finite() or not qty.is_finite() or price <= 0 or qty <= 0 or not fee.is_finite():
            raise ValueError('Invalid incremental fill observation')
        self.last_fill_ns = ns
        self.fills_count += 1
        day = utc_day(ns)
        if day not in self.daily:
            raise ValueError('Fill outside configured UTC window')
        row = dict(fill_id=self.fills_count, cid=fill['cid'], receipt_ns=ns, utc_day=day,
                   side=fill['side'], price=str(price), qty=str(qty), notional=str(price * qty), fee=str(fee))
        self.fills_file.write(json.dumps(row, separators=(',', ':')) + '\n')
        side = 'buy' if fill['side'] == 1 else 'sell'
        for agg in (self.daily[day], self.total):
            agg['fill_receipts'] += 1
            agg[side + '_qty'] += qty
            agg[side + '_notional'] += price * qty
            agg['fees'] += fee
        for h, q in self.pending.items():
            q.append((ns + h * 1_000_000, row))

    def mid(self, ns, mid):
        mid = D(mid)
        if type(ns) is not int or ns < self.last_mid_ns or not mid.is_finite() or mid <= 0:
            raise ValueError('Invalid midpoint observation')
        self.last_mid_ns = ns
        if self.mid_file:
            self.mid_file.write(f'{ns},{mid}\n')
        for h, q in self.pending.items():
            while q and q[0][0] <= ns:
                target, fill = q.popleft()
                lag = ns - target
                if lag > self.lag_ns:
                    self.outcome(h, fill, 'late_mid', target, ns, mid, None)
                else:
                    bp = D(fill['side']) * (mid - D(fill['price'])) / D(fill['price']) * D(10000)
                    self.outcome(h, fill, 'observed', target, ns, mid, bp)

    def outcome(self, h, fill, status, target, ns=None, mid=None, bp=None):
        row = dict(fill_id=fill['fill_id'], utc_day=fill['utc_day'], horizon_ms=h, status=status,
                   target_ns=target, midpoint_ns=ns, mid=str(mid) if mid is not None else None,
                   lag_ns=ns-target if ns is not None else None, notional=fill['notional'],
                   markout_bp=str(bp) if bp is not None else None)
        self.markouts_file.write(json.dumps(row, separators=(',', ':')) + '\n')
        for day in ('ALL', fill['utc_day']):
            a = self.aggregates.setdefault((day, h), dict(observed=0, censored=0, notional=D(0), weighted=D(0), lag_sum=0, reasons=Counter()))
            if bp is not None:
                a['observed'] += 1
                a['notional'] += D(fill['notional'])
                a['weighted'] += bp * D(fill['notional'])
                a['lag_sum'] += ns - target
            else:
                a['censored'] += 1
                a['reasons'][status] += 1

    def gap(self, reason='segment_boundary'):
        # A future segment must never supply a missing pre-gap markout price.
        for h, q in self.pending.items():
            while q:
                target, fill = q.popleft()
                self.outcome(h, fill, reason, target)

    @staticmethod
    def volume_row(day, a):
        return dict(utc_day=day, fill_receipts=a['fill_receipts'], **{k: str(v) for k, v in a.items() if k != 'fill_receipts'},
                    total_qty=str(a['buy_qty'] + a['sell_qty']), total_notional=str(a['buy_notional'] + a['sell_notional']))

    def finish(self, summary):
        self.gap()
        if self.fills_count != summary['orders'].get('fill_events', 0) or self.total['fees'] != D(summary['fees']):
            raise AssertionError('Observation receipts/fees differ from OMS')
        if self.total['buy_qty'] - self.total['sell_qty'] != D(summary['position']):
            raise AssertionError('Volume inventory reconciliation failed')
        cash = self.total['sell_notional'] - self.total['buy_notional'] - self.total['fees'] + D(summary['funding'])
        if abs(cash - D(summary['cash_flow'])) > D('1e-18'):
            raise AssertionError('Volume cash reconciliation failed')
        for key, value in self.total.items():
            assert sum(a[key] for a in self.daily.values()) == value, key
        volume = [self.volume_row(day, a) for day, a in sorted(self.daily.items())]
        for row in volume:
            start = int(datetime.fromisoformat(row['utc_day']).replace(tzinfo=timezone.utc).timestamp()) * 1_000_000_000
            end = start + 86_400_000_000_000
            covered = sum(max(0, min(end,s['end_ns'])-max(start,s['start_ns'])) for s in summary['segments'])
            row['covered_seconds'] = str(D(covered)/D(1_000_000_000))
            row['coverage_status'] = 'not_replayed' if not covered else ('full_day' if covered >= 86_399_000_000_000 else 'gaps_or_partial_day')
            if not covered:
                for k in list(row):
                    if k not in ('utc_day', 'covered_seconds', 'coverage_status'):
                        row[k] = None
        total = self.volume_row('ALL', self.total)
        write_csv(self.output / 'daily_volume.csv', volume, list(volume[0]))
        markouts = []
        for day in ('ALL', *sorted(self.daily)):
            for h in self.pending:
                a = self.aggregates.get((day, h), dict(observed=0, censored=0, notional=D(0), weighted=D(0), lag_sum=0, reasons={}))
                expected = self.total['fill_receipts'] if day == 'ALL' else self.daily[day]['fill_receipts']
                assert a['observed'] + a['censored'] == expected
                markouts.append(dict(utc_day=day, horizon_ms=h, observed=a['observed'], censored=a['censored'],
                    observed_notional=str(a['notional']), notional_weighted_bp=str(a['weighted']/a['notional']) if a['notional'] else None,
                    mean_lag_ns=a['lag_sum']/a['observed'] if a['observed'] else None,
                    censor_reasons=json.dumps(dict(a['reasons']), sort_keys=True)))
        write_csv(self.output / 'markout_summary.csv', markouts, list(markouts[0]))
        result = dict(schema_version=1, status='complete', full_range=summary['full_range'], spec=self.spec,
                      volume=total, markouts=[r for r in markouts if r['utc_day'] == 'ALL'],
                      native_execution_count=summary['native_fills'], oms_fill_receipts=self.fills_count,
                      validation='daily sums, inventory, fees, cash and horizon coverage reconciled')
        write_json(self.output / 'metrics.json', result)
        self.close()
        return result

    def close(self):
        if not self.closed:
            for f in (self.fills_file, self.markouts_file, self.mid_file):
                if f:
                    f.close()
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
