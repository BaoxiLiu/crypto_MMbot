from decimal import Decimal as D
import pytest
from app.market_making.fusion.execution import ExecutionProfile


def test_profile_roundtrip_and_offset_does_not_add_feed_twice():
 p=ExecutionProfile.from_dict(dict(name='fixture',feed_ns=2_000_000,decision_ns=250_000,taker_fee='.0005'))
 assert p.offset_from_cache(5)==-3_000_000 and p.taker_fee==D('.0005')
 assert ExecutionProfile.from_dict(p.json())==p


@pytest.mark.parametrize('field,value',[('feed_ns',0),('response_ns',-1),('request_ns',True),('decision_ns',-1),('decision_ns',100_000_000),('taker_fee','NaN')])
def test_invalid_profile_fails_closed(field,value):
 with pytest.raises(ValueError):ExecutionProfile.from_dict({field:value})


def test_public_proxy_cannot_claim_measured_private_order_latency():
 with pytest.raises(ValueError):ExecutionProfile.from_dict(dict(provenance=dict(order_latency_measured=True,private_order_samples=0)))
