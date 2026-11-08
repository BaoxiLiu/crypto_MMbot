"""Shared-policy parity between replay Kernel and the live event adapter."""
from app.market_making.fusion.tests.quote_fixture import QuoteFixture
from dataclasses import asdict, replace
from decimal import Decimal as D
import json
import random
import pytest
from app.market_making.live import Rules, StrategyConfig, ProtectionSwitches, StrategyEngine
from app.market_making.as_math import ASConfig


def normalized(value):
    return json.loads(json.dumps(value, default=str, sort_keys=True))


@pytest.mark.parametrize('mask', range(8))
@pytest.mark.parametrize('seed', [7, 42, 2026])
def test_live_adapter_preserves_quotes_commands_ledger_and_policy(mask, seed):
    from app.market_making.fusion import kernel as legacy
    from app.market_making.fusion.tests.quote_fixture import QuoteFixture
    from app.market_making.core import Rules as OldRules
    from app.market_making.as_math import ASConfig as OldAS
    from app.market_making.fusion.config import FusionConfig, Factors
    values = (D('.01'), D(1), D(5), D(1), D(100000), D('.5'), D('1.5'))
    old_config = replace(FusionConfig(), as_config=replace(OldAS(),warmup_samples=2))
    new_config = StrategyConfig.from_dict(old_config.json())
    a = legacy.Kernel(OldRules(*values), old_config, Factors.from_mask(mask),prefix='parity',strategy_factory=QuoteFixture)
    b = StrategyEngine(Rules(*values), new_config, ProtectionSwitches.from_mask(mask),prefix='parity',strategy_factory=QuoteFixture)
    rng = random.Random(seed)
    for i in range(100):
        t = i * 100_000_000
        bid = D(10) + D(rng.randrange(-2,3))/100
        qty, side = D(rng.randrange(1,8)), rng.choice([-1,1])
        for k in (a,b):
            k.book(t,bid,bid+D('.02')); k.mark_price(t,bid+D('.01'))
            k.trade(t,i,side,bid,qty)
        x, y = a.step(t), b.step(t)
        assert normalized([asdict(c) for c in x]) == normalized([asdict(c) for c in y])
        for c in x:
            for k in (a,b):
                o=k.oms.find(c.cid)
                k.update(c.cid,'NEW' if c.kind=='new' else 'CANCELED',o.filled,o.quote_filled,t+1)
        if i%11 == 0:
            live=next((o for o in a.oms.active.values() if o.remaining>0),None)
            if live:
                for k in (a,b):
                    o=k.oms.find(live.cid);filled=o.filled+min(D(1),o.remaining)
                    k.update(o.cid,'FILLED' if filled==o.qty else 'PARTIALLY_FILLED',
                             filled,filled*o.price,t+2)
        assert normalized(a.report()) == normalized(b.report())
        assert normalized(a.strategy.last) == normalized(b.strategy.last)
        assert normalized(a.oms.dump()) == normalized(b.oms.dump())
