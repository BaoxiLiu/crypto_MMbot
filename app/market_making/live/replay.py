"""Deterministic input-event replay. No network and no invented matching/fills.

Private executions in the input are explicit recorded/fixture receipts. For
performance/economic backtests keep the existing verified HFT backend.
"""
from dataclasses import asdict
from decimal import Decimal as D
import hashlib
import json
from pathlib import Path
from app.market_making.fusion.config import FusionConfig as StrategyConfig, Factors as ProtectionSwitches
from app.market_making.core import Rules
from .engine import StrategyEngine
from ..strategies.registry import factory_for
from .selection import load_selection
from .gateway import PrivateEvents, apply_order


def encode(value):
    return json.dumps(value, default=str, ensure_ascii=False, allow_nan=False)


def load_engine(config_path, rules_path, protections='as', prefix='replay', *, strategy, symbol):
    config = StrategyConfig.from_dict(json.loads(Path(config_path).read_text()))
    values = json.loads(Path(rules_path).read_text())
    rules = Rules(**{key: D(str(value)) for key, value in values.items()})
    if protections != 'as':
        raise ValueError('Public strategies own their protection policy; select protections=as')
    case=load_selection(strategy,symbol)
    factory=factory_for(case)
    engine=StrategyEngine(rules, config, ProtectionSwitches(), prefix=prefix,strategy_factory=factory)
    engine.selected_case=case
    return engine


def apply_event(engine, event, private):
    ns, kind = event['ns'], event['kind']
    if type(ns) is not int or ns < 0:
        raise ValueError('Nonnegative integer ns required')
    if kind == 'snapshot':
        engine.snapshot(event['data'], ns)
    elif kind == 'depth':
        engine.depth(event['data'], ns)
    elif kind == 'book':
        engine.top_of_book(ns, D(event['bid']), D(event['bid_qty']) if 'bid_qty' in event else None,
                           D(event['ask']), D(event['ask_qty']) if 'ask_qty' in event else None)
    elif kind == 'trade':
        engine.trade(ns, event['id'], event['side'], D(event['price']), D(event['qty']))
    elif kind == 'mark':
        engine.mark_price(ns, D(event['price']))
    elif kind == 'order':
        engine.update(event['cid'], event['status'], D(event['cumulative']),
                      D(event['cumulative_quote']), ns)
    elif kind == 'private':
        private.apply(event['data'], ns)
    elif kind == 'rest_order':
        apply_order(engine, event['data'], ns)
    elif kind == 'disconnect':
        engine.external_healthy = False
        engine.reset_market()
    elif kind == 'connected':
        engine.external_healthy = True
    elif kind != 'timer':
        raise ValueError('Unknown normalized event: ' + str(kind))


def replay(engine, events, symbol='DEMOUSDT', quote='USDT'):
    private = PrivateEvents(engine, symbol, quote)
    previous = -1
    records = []
    for event in events:
        ns = event['ns']
        if type(ns) is not int or ns < previous:
            raise ValueError('Input receipt clock reversed')
        previous = ns
        apply_event(engine, event, private)
        commands = engine.step(ns)
        records.append(dict(ns=ns, event=event['kind'],
                            commands=[asdict(c) for c in commands],
                            account=engine.report(), decision=dict(engine.decision.last)))
    return records


def run_file(input_path, output, config, rules, protections='as', *, strategy, symbol, quote):
    if not isinstance(quote,str) or not quote:
        raise ValueError('Explicit quote asset required for strategy replay')
    output = Path(output)
    if output.exists():
        raise FileExistsError('Use a new output directory')
    input_path, config, rules = map(Path, (input_path, config, rules))
    # Parse all inputs before creating output. Streaming replay is a later optimization.
    events = [json.loads(line) for line in input_path.read_text().splitlines() if line.strip()]
    if not events:
        raise ValueError('Empty replay input')
    engine = load_engine(config, rules, protections, strategy=strategy, symbol=symbol)
    if getattr(engine.strategy,'requires_quote_depth',False) and not {'snapshot','depth'} <= {e['kind'] for e in events}:
        raise ValueError('Selected strategy replay requires snapshot and depth events')
    records = replay(engine, events, symbol=symbol, quote=quote)
    output.mkdir(parents=True)
    (output / 'events.jsonl').write_text(''.join(encode(row) + '\n' for row in records))
    summary = dict(mode='input_event_replay_not_matching_backtest', events=len(records),
                   protections=protections, symbol=symbol, quote=quote,
                   strategy=engine.selected_case, account=engine.report(),
                   inputs={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (input_path, config, rules, Path(strategy))})
    (output / 'summary.json').write_text(encode(summary) + '\n')
    (output / 'effective_config.json').write_text(encode(engine.config.json()) + '\n')
    return summary
