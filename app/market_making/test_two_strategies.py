"""The current pair retains the real paper arm without contaminating native AS."""
from copy import deepcopy
import json
from pathlib import Path

from .core import Account, account_configs
from .s1_config import make_s1_config,make_two_strategy_config
from .test_core import rules


def test_only_two_accounts_with_original_full_paper_policy():
    base=json.loads((Path(__file__).parent/'fixtures/account_matrix.json').read_text())
    before=deepcopy(base)
    cfg=make_two_strategy_config(base)
    assert base==before
    rows=dict(account_configs(cfg))
    assert set(rows)=={'as_base','paper_s1_sc0'}
    original=dict(account_configs(make_s1_config(base)))['D_s1_sc']
    for key in ('s1_k','s1_window_seconds','sc_window_fills','sc_min_new_fills',
                'sc_adjust_interval_min','sc_up_factor','sc_down_factor','fair_value',
                'fv_source','keeper_enabled','as_floor_cap_ticks'):
        assert rows['paper_s1_sc0'][key]==original[key]
    for name,c in rows.items():
        a=Account(name,c,rules(),lambda *args:None)
        if name=='as_base':
            assert a.floor_state()['as_floor_ticks']==0
            assert not c['s1_enabled'] and not c['spread_control_enabled']
            assert not c['keeper_enabled'] and c['fair_value'] is None
        else:
            assert a.floor_state()['as_floor_ticks']==4
            assert c['s1_enabled'] and c['spread_control_enabled']


def test_pair_selection_preserves_execution_fees_and_risk():
    base=json.loads((Path(__file__).parent/'fixtures/account_matrix.json').read_text())
    base.update(maker_fee='-0.00003',taker_fee='0.0004',max_position_notional='1000')
    cfg=make_two_strategy_config(base)
    for _,c in account_configs(cfg):
        for k in ('maker_fee','taker_fee','max_position_notional','max_position_qty','initial_collateral'):
            assert c[k]==base[k]
