from app.market_making.fusion.tests.quote_fixture import QuoteFixture
from dataclasses import FrozenInstanceError, asdict, replace
from decimal import Decimal as D
import json
from pathlib import Path
import subprocess
import sys
import pytest
from app.market_making.live import Rules, StrategyConfig, ProtectionSwitches, StrategyEngine
from app.market_making.as_math import ASConfig
from app.market_making.live.market import LocalBook, BookGap
from app.market_making.live.contracts import PriceForecast, SideAssessment
from app.market_making.live.decision import DecisionPipeline
from app.market_making.contracts import Desired
from app.market_making.live.replay import replay, load_engine, run_file


def rules():
    return Rules(D('.01'), D(1), D(5), D(1), D(100000), D('.5'), D('1.5'))


def engine(decision=None):
    config = replace(StrategyConfig(), as_config=replace(ASConfig(), warmup_samples=2))
    return StrategyEngine(rules(), config, ProtectionSwitches(), decision=decision,strategy_factory=QuoteFixture)


def warm(k):
    for i in range(3):
        t = i * 100_000_000
        k.book(t, D(10), D('10.02'))
        k.mark_price(t, D('10.01'))
    return 200_000_000


def snapshot():
    return dict(lastUpdateId=100, bids=[['10.00', '100'], ['9.99', '50']],
                asks=[['10.02', '100'], ['10.03', '50']])


def depth(uid=101, previous=100, bids=None, asks=None):
    return dict(U=100 if uid == 101 else uid, u=uid, pu=previous,
                b=bids or [], a=asks or [])


def test_l2_snapshot_bridge_absolute_updates_and_nonempty_level_ranking():
    b = LocalBook(rules()); b.snapshot(snapshot(), 0)
    with pytest.raises(BookGap): b.view()
    assert b.update(depth(bids=[['10.00', '12'], ['9.99', '0'], ['9.97', '30']]), 1)
    view = b.view(2)
    assert view.bids == ((D('10'), D(12)), (D('9.97'), D(30)))
    assert view.mid == D('10.01')
    assert not b.update(depth(), 2)  # duplicate has no effect
    with pytest.raises(FrozenInstanceError): view.ns = 100


def test_gap_invalidates_market_without_resetting_account_or_pending_orders():
    k = engine(); t = warm(k)
    cmds = k.step(t)
    before = k.oms.dump()
    k.market.snapshot(snapshot(), t)
    k.market.update(depth(), t+1)
    with pytest.raises(BookGap): k.depth(depth(103, 999), t+2)
    assert not k.market.ready and not k.market_ready
    assert k.oms.dump() == before
    assert k.oms.pending(1) > 0 and k.oms.pending(-1) > 0


@pytest.mark.parametrize('changes', [dict(b=[['10.01', '-1']]),
    dict(b=[['10.001', '1']]), dict(b=[['10.00', '1'], ['10.00', '2']]),
    dict(b=[['10.04', '1']])])
def test_bad_depth_packets_never_leave_a_usable_half_updated_book(changes):
    b = LocalBook(rules()); b.snapshot(snapshot(), 0); b.update(depth(), 1)
    with pytest.raises((BookGap, ArithmeticError)):
        b.update(dict(depth(102, 101), **changes), 2)
    assert not b.ready and not b.bids and not b.asks


def test_price_model_before_quotes_and_evaluator_after_quotes():
    calls = []
    class Model:
        def predict(self, ctx):
            calls.append('price')
            with pytest.raises(FrozenInstanceError): ctx.position = D(100)
            return PriceForecast(50., ctx.ns, ctx.ns + 1, 'fixture')
    class Evaluator:
        def evaluate(self, ctx, candidates):
            calls.append('evaluate')
            assert candidates and all(q.price > 0 for q in candidates)
            with pytest.raises(FrozenInstanceError): candidates[0].qty = D(999)
            return tuple(SideAssessment(q.side, True, ctx.ns, ctx.ns+1, 'fixture') for q in candidates)
    k = engine(DecisionPipeline(Model(), Evaluator())); t = warm(k)
    shifted = k.plan(t)
    baseline = engine(); warm(baseline)
    assert shifted[1].price > baseline.plan(t)[1].price
    assert calls == ['price', 'evaluate']
    assert k.decision.last['stages'] == ['state', 'features', 'price_forecast',
        'as_candidates', 'regime_flow_inventory', 'order_evaluation', 'decision']


def test_unavailable_evaluation_blocks_opening_but_preserves_nonflipping_reduction():
    class Evaluator:
        def evaluate(self, ctx, candidates): return None
    k = engine(DecisionPipeline(order_evaluator=Evaluator())); t = warm(k)
    assert k.plan(t) == {}
    c = k.oms.reconcile_targets({1: Desired(1, D(10), D(1))}, D(10), t+1)[0]
    k.update(c.cid, 'FILLED', D(1), D(10), t+2)
    target = k.plan(t+3)
    assert set(target) == {-1} and target[-1].reduce_only and target[-1].qty == 1
    commands = k.dispatch(target, t+4)
    assert commands[0].side == -1 and commands[0].qty == 1


def test_future_model_cannot_send_new_orders():
    class Future:
        def predict(self, ctx): return PriceForecast(1., ctx.ns+1, ctx.ns+2, 'future')
    k = engine(DecisionPipeline(Future())); t = warm(k)
    assert k.step(t) == []
    assert k.oms.halt_reason == 'model_contract_failure'


def test_model_never_bypasses_latest_risk_or_independent_stop():
    k = engine(); t = warm(k); targets = k.plan(t)
    k.oms.halt_reason = 'supervisor_stop'
    assert k.dispatch(targets, t+1) == []
    assert not k.oms.active


def test_public_queue_estimate_is_not_a_fill_and_stale_receipts_do_not_regress():
    k = engine(); k.snapshot(snapshot(), 0)
    for i in range(3):
        t=i*100_000_000; k.depth(depth(101+i, 100+i), t); k.mark_price(t,D('10.01'))
    c=k.dispatch({1: Desired(1, D(10), D(2))}, t+1)[0]
    k.update(c.cid,'NEW',D(0),D(0),t+2)
    k.trade(t+3,1,-1,D(10),D(100))
    assert k.queue_observer.orders[c.cid].front == 0
    assert k.oms.ledger.position == 0  # public consumption is not execution
    k.update(c.cid,'PARTIALLY_FILLED',D(1),D(10),t+4)
    k.update(c.cid,'NEW',D(0),D(0),t+5)
    assert k.oms.ledger.position == 1
    assert k.queue_observer.orders[c.cid].cumulative_fill == 1







def test_replay_rejects_backwards_time_and_unknown_event():
    k=engine()
    with pytest.raises(ValueError,match='clock reversed'):
        replay(k,[dict(ns=10,kind='mark',price='10'),dict(ns=9,kind='timer')])
    with pytest.raises(ValueError,match='Unknown normalized event'):
        replay(engine(),[dict(ns=0,kind='pretend_fill')])


def test_engine_requires_an_explicit_strategy():
    with pytest.raises(ValueError, match='Explicit strategy_factory'):
        StrategyEngine(rules(), StrategyConfig(), ProtectionSwitches())


def test_injected_strategy_receives_account_context_before_quoting():
    from app.market_making.strategies.base import QuoteStrategy

    class Policy(QuoteStrategy):
        def before_plan(self, working, equity, mark):
            self.context = (tuple(working), equity, mark)

        def desired(self, position, ns):
            assert self.context[1] == self.config.initial_cash
            assert self.context[2] == D('10.01')
            self.counts['injected_decisions'] += 1
            return {1: Desired(1, D('9.99'), D(1))}

    c = replace(StrategyConfig(), as_config=replace(ASConfig(), warmup_samples=2))
    k = StrategyEngine(rules(), c, ProtectionSwitches(), strategy_factory=Policy)
    ns = warm(k)
    assert k.plan(ns)[1].price == D('9.99')
    assert k.strategy.counts['injected_decisions'] == 1


def test_unsupported_forecast_and_native_market_interfaces_fail_explicitly():
    from app.market_making.strategies.base import QuoteStrategy

    with pytest.raises(ValueError, match='external price forecasts'):
        StrategyEngine(rules(), StrategyConfig(), ProtectionSwitches(),
                       decision=DecisionPipeline(price_model=object()),
                       strategy_factory=QuoteStrategy)

    class NativeOnly(QuoteStrategy):
        def observe_market(self, depth, trades, ns):
            pass

    with pytest.raises(ValueError, match='market-input contract'):
        StrategyEngine(rules(), StrategyConfig(), ProtectionSwitches(),
                       strategy_factory=NativeOnly)
