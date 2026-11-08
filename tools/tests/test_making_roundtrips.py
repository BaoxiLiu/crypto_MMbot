from decimal import Decimal as D
from tools.ledger.analyze_making_roundtrips import analyze


def fill(i,side,qty,price,fee='0'):
    return dict(fill_id=i,cid=f'o{i}',receipt_ns=i*1000000000,side=side,qty=str(qty),price=str(price),fee=fee)


def test_partial_close_reversal_and_funding_are_not_double_counted():
    fills=[fill(1,1,2,100,'2'),fill(2,-1,1,110,'1'),fill(3,-1,3,90,'3'),fill(4,1,2,80,'2')]
    funding=[dict(applied_ns=1500000000,position='2',cashflow='2')]
    result,episodes,reductions,_=analyze(fills,funding,D(80))
    assert [r['net_pnl'] for r in episodes]==[D(-2),D(16)]
    assert [r['net_pnl'] for r in reductions]==[D(9),D(-11),D(16)]
    assert result['total_pnl']==14 and result['position']==0
    assert result['completed_inventory_episodes']['count']==2


def test_residual_position_is_not_counted_as_completed_roundtrip():
    result,episodes,reductions,_=analyze([fill(1,1,2,100,'2'),fill(2,-1,1,110,'1')],
                                      [dict(applied_ns=2500000000,position='1',cashflow='2')],D(105))
    assert not episodes and len(reductions)==1
    assert reductions[0]['net_pnl']==8
    assert result['residual_average_cost_net']==6
    assert result['open_episode']['marked_net_pnl']==14
    assert result['total_pnl']==14


def test_flat_roundtrip_at_same_price_is_neutral():
    result,episodes,_,_=analyze([fill(1,1,1,100),fill(2,-1,1,100)],[],D(100))
    assert result['completed_inventory_episodes']['breakeven']==1
    assert result['completed_inventory_episodes']['profitable']==0
    assert result['completed_inventory_episodes']['losing']==0


def test_fill_is_applied_before_funding_at_identical_receipt_time():
    result,episodes,_,_=analyze([fill(1,1,1,100),fill(2,-1,1,100)],
                              [dict(applied_ns=1000000000,position='1',cashflow='-1')],D(100))
    assert episodes[0]['net_pnl']==-1
    assert result['total_pnl']==-1
