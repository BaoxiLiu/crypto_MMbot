"""Versioned replay timing/cost scenarios, separate from AS strategy parameters."""
from dataclasses import dataclass,field,asdict
from decimal import Decimal as D
import json


@dataclass(frozen=True)
class ExecutionProfile:
    name: str='legacy-5-5-10-5'
    feed_ns: int=5_000_000
    trade_feed_ns: int | None=None
    mark_feed_ns: int | None=None
    refresh_ns: int | None=100_000_000
    decision_ns: int=5_000_000
    wire_prepare_ns: int=0
    request_ns: int=10_000_000
    response_ns: int=5_000_000
    taker_fee: D=D('.0004')
    provenance: dict=field(default_factory=dict)
    trigger: str='interval'

    def __post_init__(self):
        if not isinstance(self.name,str) or not self.name:raise ValueError('profile name')
        for key in ('feed_ns','request_ns','response_ns'):
            if type(getattr(self,key)) is not int or getattr(self,key)<=0:raise ValueError(key)
        for key in ('trade_feed_ns','mark_feed_ns'):
            value=getattr(self,key)
            if value is not None and (type(value) is not int or value<=0):raise ValueError(key)
        if self.trade_feed_ns is not None and self.trade_feed_ns<self.feed_ns:
            raise ValueError('trade feed must not precede depth feed in this cursor adapter')
        if type(self.wire_prepare_ns) is not int or self.wire_prepare_ns<0:raise ValueError('wire_prepare_ns')
        if self.trigger not in ('interval','event'):raise ValueError('trigger')
        if self.refresh_ns is None:
            if self.trigger!='event':raise ValueError('refresh_ns')
        elif type(self.refresh_ns) is not int or self.refresh_ns<=0:raise ValueError('refresh_ns')
        if type(self.decision_ns) is not int or self.decision_ns<0 or (self.trigger=='interval' and self.decision_ns>=self.refresh_ns):raise ValueError('decision_ns')
        if not isinstance(self.taker_fee,D) or not self.taker_fee.is_finite() or self.taker_fee<0:raise ValueError('taker_fee')
        if not isinstance(self.provenance,dict):raise ValueError('provenance')
        if self.provenance.get('order_latency_measured') and not self.provenance.get('private_order_samples',0):
            raise ValueError('cannot claim measured orders without private samples')

    @classmethod
    def from_dict(cls,data):
        data=dict(data)
        if 'taker_fee' in data:data['taker_fee']=D(str(data['taker_fee']))
        return cls(**data)

    def json(self):return json.loads(json.dumps(asdict(self),default=str,allow_nan=False))

    def offset_from_cache(self,cache_feed_ms):return self.feed_ns-int(cache_feed_ms*1_000_000)
