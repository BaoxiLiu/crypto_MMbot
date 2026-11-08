"""Synthetic AS contract; never financial research evidence."""
from decimal import Decimal as D
import numpy as np
import pytest
from app.market_making.fusion.tests.refactor_trace import replay_case


@pytest.mark.parametrize('name,fee',[('as_original','0'),('as_improved4','-0.00003')])
def test_causal_context_native_policy_queue_and_signed_fee_accounting(tmp_path,name,fee):
    result,out,_=replay_case(tmp_path,name,fee=fee,return_artifacts=True)
    assert result['invariant_failures']==0
    assert result['orders'].get('fill_events',0)>0
    assert D(result['fees'])==0 if D(fee)==0 else D(result['fees'])<0
    mode='as_base' if name=='as_original' else 'paper_s1_sc0'
    assert result['identity']['strategy']['params']['mode']==mode
    with np.load(out/'equity.npz') as z:
        assert np.all(z['data']['local_ns']>=1_785_542_400_000_000_000+1860*10**9)
