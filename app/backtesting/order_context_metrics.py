"""Offline observation of original decision clocks for frozen filled orders.

Selecting these CIDs is retrospective sampling, never an input to the strategy.
This dataset supports conditional-on-fill research, not all-order prediction.
"""
import json
import math
import gzip
from pathlib import Path

from .experiment_metrics import ReplayMetrics, write_json


class OrderPlacementMetrics(ReplayMetrics):
    """Observe every submitted new command, including orders that never fill.

    BBO is the frozen decision book; actual submit-time L2 is joined offline.
    A successful submit call is not an exchange acceptance acknowledgement.
    """
    additional_sources = ('app/backtesting/experiment_metrics.py', 'app/backtesting/order_context_metrics.py')

    def __init__(self, output, spec, start, end_exclusive):
        super().__init__(output, spec, start, end_exclusive)
        self.placements_file = gzip.open(self.output / 'order_placements.jsonl.gz', 'wt', compresslevel=1)
        self.placements_count = 0

    def command(self, cmd, decision_ns, submit_ns, bid, ask):
        if cmd.kind != 'new':
            return
        if decision_ns > submit_ns or not (math.isfinite(bid) and math.isfinite(ask) and 0 < bid < ask):
            raise AssertionError('Invalid new-order observation clock/book')
        row = dict(cid=cmd.cid, decision_ns=int(decision_ns), submit_ns=int(submit_ns),
                   side=cmd.side, price=str(cmd.price), qty=str(cmd.qty), reduce_only=cmd.reduce_only,
                   decision_bid=float(bid), decision_ask=float(ask))
        self.placements_file.write(json.dumps(row, separators=(',', ':')) + '\n')
        self.placements_count += 1

    def finish(self, summary):
        if self.placements_count != summary['orders'].get('new_requests', 0):
            raise AssertionError('Missing submitted new orders')
        write_json(self.output / 'order_placements_validation.json',
                   dict(submitted_new_orders=self.placements_count, matches_oms_new_requests=True,
                        sampling='all actual new requests; exchange acceptance not implied',
                        bbo_clock='decision_ns; submit-time book requires raw LOCAL tape join'))
        return super().finish(summary)

    def close(self):
        if getattr(self, 'placements_file', None) is not None:
            self.placements_file.close()
        super().close()


class OrderLifecycleMetrics(OrderPlacementMetrics):
    """Receipt state/command observation without synthesizing exchange clocks."""

    def __init__(self, output, spec, start, end_exclusive):
        super().__init__(output, spec, start, end_exclusive)
        self.lifecycle_file = gzip.open(self.output/'order_lifecycle.jsonl.gz', 'wt', compresslevel=1)
        self.lifecycle_records = 0

    def record(self, row):
        self.lifecycle_file.write(json.dumps(row, separators=(',', ':'))+'\n')
        self.lifecycle_records += 1

    def command(self, cmd, decision_ns, submit_ns, bid, ask):
        super().command(cmd, decision_ns, submit_ns, bid, ask)
        self.record(dict(kind=cmd.kind+'_sent', cid=cmd.cid, side=cmd.side,
                         decision_ns=int(decision_ns), receipt_ns=int(submit_ns),
                         price=str(cmd.price), qty=str(cmd.qty), reduce_only=cmd.reduce_only,
                         decision_bid=float(bid), decision_ask=float(ask)))

    def order_update(self, cid, label, cumulative, now, exchange_ns, request_ns, pending_req,
                     boundary_assumption=False):
        self.record(dict(kind='state_received', cid=cid, status=label, cumulative=str(cumulative),
                         receipt_ns=int(now), native_exchange_ns=int(exchange_ns) if exchange_ns else None,
                         native_request_ns=int(request_ns) if request_ns else None,
                         pending_req=int(pending_req), boundary_assumption=boundary_assumption,
                         clocks='native simulation fields; not measured real exchange timestamps'))

    def close(self):
        if getattr(self, 'lifecycle_file', None) is not None:
            self.lifecycle_file.close()
        super().close()


class OrderContextMetrics(ReplayMetrics):
    additional_sources = ('app/backtesting/experiment_metrics.py', 'app/backtesting/order_context_metrics.py')

    def __init__(self, output, spec, start, end_exclusive, source_fills):
        super().__init__(output, spec, start, end_exclusive)
        self.selected = {json.loads(line)['cid'] for line in Path(source_fills).read_text().splitlines()}
        self.context_file = (self.output / 'order_context.jsonl').open('w')
        self.captured = set()
        self.own_filled = set()

    def command(self, cmd, decision_ns, submit_ns, bid, ask):
        if cmd.kind != 'new' or cmd.cid not in self.selected:
            return
        if cmd.cid in self.captured or decision_ns > submit_ns:
            raise AssertionError('Duplicate order context or reversed decision clock')
        if not (math.isfinite(bid) and math.isfinite(ask) and 0 < bid < ask):
            raise AssertionError('New order without valid decision book')
        row = dict(cid=cmd.cid, decision_ns=int(decision_ns), submit_ns=int(submit_ns),
                   side=cmd.side, price=str(cmd.price), qty=str(cmd.qty),
                   decision_bid=float(bid), decision_ask=float(ask),
                   feature_cutoff='complete local timestamp batch at decision; before compute',
                   sampling='retrospective frozen filled CIDs; not all candidates')
        self.context_file.write(json.dumps(row, separators=(',', ':')) + '\n')
        self.captured.add(cmd.cid)

    def fill(self, fill):
        if fill['cid'] not in self.captured:
            raise AssertionError('Fill lacks original decision context')
        self.own_filled.add(fill['cid'])
        super().fill(fill)

    def finish(self, summary):
        write_json(self.output / 'order_context_validation.json',
                   dict(captured=len(self.captured), filled_orders=len(self.own_filled),
                        all_fills_have_context=self.own_filled <= self.captured,
                        all_frozen_orders_captured=self.selected <= self.captured,
                        full_range=summary['full_range']))
        return super().finish(summary)

    def close(self):
        if getattr(self, 'context_file', None) is not None:
            self.context_file.close()
        super().close()
