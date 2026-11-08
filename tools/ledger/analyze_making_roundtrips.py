"""Actual simulated inventory PnL, independent of forward markout labels.

Two explicit units: flat-to-flat inventory episodes and average-cost reducing
fill receipts. Reversal fills are split at zero. Residual inventory stays open.
"""
import argparse
from collections import Counter
from decimal import Decimal as D
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from app.market_making.core import Ledger

ZERO=D(0)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,default=str,ensure_ascii=False,indent=2)+'\n')


def stats(rows,key):
    values=[D(r[key]) for r in rows]
    wins=[v for v in values if v>0];losses=[v for v in values if v<0]
    total=len(values);nonzero=len(wins)+len(losses)
    return dict(count=total,profitable=len(wins),losing=len(losses),breakeven=total-nonzero,
                win_pct_all=100*len(wins)/total if total else None,
                loss_pct_all=100*len(losses)/total if total else None,
                win_pct_nonzero=100*len(wins)/nonzero if nonzero else None,
                loss_pct_nonzero=100*len(losses)/nonzero if nonzero else None,
                total_net_pnl=sum(values,ZERO),sum_profits=sum(wins,ZERO),sum_losses=sum(losses,ZERO),
                mean_profit=sum(wins,ZERO)/len(wins) if wins else None,
                mean_loss=sum(losses,ZERO)/len(losses) if losses else None)


def analyze(fills,funding,mark,initial=D(600)):
    ledger=Ledger(initial=initial,wallet=initial,day_start_equity=initial,peak_equity=initial)
    episodes=[];reductions=[];legs=[];episode=None;serial=0
    open_fee_pool=ZERO;open_funding_pool=ZERO
    events=[(r['receipt_ns'],0,r.get('fill_id',i),r) for i,r in enumerate(fills)]
    events += [(r['applied_ns'],1,i,r) for i,r in enumerate(funding)]
    for ns,kind,_,row in sorted(events,key=lambda x:x[:3]):
        if kind==1:
            payment=D(row['cashflow'])
            assert ledger.position==D(row['position']), ('funding inventory mismatch',ns)
            ledger.wallet+=payment;ledger.funding+=payment
            open_funding_pool+=payment
            if episode is None:
                assert payment==0 and ledger.position==0
            else:
                episode['funding']+=payment
            continue
        side=row['side'];qty,price,fee=D(row['qty']),D(row['price']),D(row['fee'])
        previous,entry=ledger.position,ledger.average_entry
        close_qty=min(abs(previous),qty) if previous*side<0 else ZERO
        new_qty=qty-close_qty
        closing_fee=fee if close_qty==qty else fee*close_qty/qty
        opening_fee=fee-closing_fee
        allocated_entry_fee=ZERO;allocated_funding=ZERO
        if close_qty:
            ratio=close_qty/abs(previous)
            allocated_entry_fee=open_fee_pool if close_qty==abs(previous) else open_fee_pool*ratio
            allocated_funding=open_funding_pool if close_qty==abs(previous) else open_funding_pool*ratio
            open_fee_pool-=allocated_entry_fee;open_funding_pool-=allocated_funding
        gross,_=ledger.fill(side,qty,price,ZERO)
        ledger.fees+=fee;ledger.wallet-=fee
        open_fee_pool+=opening_fee
        if close_qty:
            reductions.append(dict(fill_id=row.get('fill_id'),cid=row.get('cid'),receipt_ns=ns,
                                   closed_qty=close_qty,prior_average_entry=entry,exit_price=price,
                                   gross_realized=gross,entry_fee=allocated_entry_fee,exit_fee=closing_fee,
                                   funding=allocated_funding,net_pnl=gross-allocated_entry_fee-closing_fee+allocated_funding))
        for piece,piece_fee,action in ((close_qty,closing_fee,'reduce'),(new_qty,opening_fee,'open_or_add')):
            if not piece:
                continue
            if episode is None:
                assert action=='open_or_add'
                serial+=1
                episode=dict(episode_id=serial,open_ns=ns,side=side,position=ZERO,
                             buy_qty=ZERO,sell_qty=ZERO,cashflow=ZERO,fees=ZERO,funding=ZERO,
                             opening_qty=ZERO,closing_qty=ZERO,fill_pieces=0)
            assert action!='reduce' or episode['position']*side<0
            episode['position']+=side*piece
            episode['buy_qty' if side==1 else 'sell_qty']+=piece
            episode['cashflow']-=side*piece*price
            episode['fees']+=piece_fee
            episode['opening_qty' if action=='open_or_add' else 'closing_qty']+=piece
            episode['fill_pieces']+=1
            legs.append(dict(episode_id=serial,fill_id=row.get('fill_id'),cid=row.get('cid'),receipt_ns=ns,
                             action=action,side=side,qty=piece,price=price,fee=piece_fee))
            if episode['position']==0:
                assert episode['buy_qty']==episode['sell_qty']
                episode.update(close_ns=ns,holding_s=D(ns-episode['open_ns'])/D(1_000_000_000),
                               gross_pnl=episode['cashflow'],
                               net_pnl=episode['cashflow']-episode['fees']+episode['funding'])
                episodes.append(episode);episode=None
        assert (episode['position'] if episode else ZERO)==ledger.position
        if ledger.position==0:
            assert open_fee_pool==0 and open_funding_pool==0
    if episode is not None:
        episode.update(status='open_not_counted_as_closed',valuation_mark=mark,
                       marked_net_pnl=episode['cashflow']+episode['position']*mark-episode['fees']+episode['funding'])
    closed_episode_pnl=sum((r['net_pnl'] for r in episodes),ZERO)
    closed_receipt_pnl=sum((r['net_pnl'] for r in reductions),ZERO)
    remaining_episode_pnl=episode['marked_net_pnl'] if episode else ZERO
    residual_cost_basis_net=ledger.unrealized(mark)-open_fee_pool+open_funding_pool
    account_pnl=ledger.equity(mark)-initial
    assert abs(closed_episode_pnl+remaining_episode_pnl-account_pnl)<D('1e-18')
    assert abs(closed_receipt_pnl+residual_cost_basis_net-account_pnl)<D('1e-18')
    return dict(completed_inventory_episodes=stats(episodes,'net_pnl'),
                average_cost_reducing_receipts=stats(reductions,'net_pnl'),
                open_episode=episode,residual_average_cost_net=residual_cost_basis_net,
                position=ledger.position,gross_realized=ledger.realized,fees=ledger.fees,
                funding=ledger.funding,unrealized=ledger.unrealized(mark),total_pnl=account_pnl),episodes,reductions,legs


def main(source,output):
    reg=json.loads((source/'registration.json').read_text())
    validation=json.loads((Path(reg['report_output'])/'validation.json').read_text())
    assert validation['status']=='passed'
    output.mkdir(parents=True,exist_ok=False)
    results={};hashes={}
    for case in reg['cases']:
        folder=Path(case['output'])
        for name in ('fills.jsonl','summary.json'):
            path=folder/name
            hashes[str(path)]=sha(path)
            assert hashes[str(path)]==validation['raw_sha256'][str(path)]
        fills=[json.loads(line) for line in (folder/'fills.jsonl').read_text().splitlines()]
        summary=json.loads((folder/'summary.json').read_text())
        result,episodes,receipts,legs=analyze(fills,summary['funding_receipts'],D(summary['valuation_mark']),D(case['config']['initial_cash']))
        for key,reference in (('position','position'),('gross_realized','realized'),('unrealized','unrealized'),
                              ('fees','fees'),('funding','funding'),('total_pnl','pnl')):
            assert abs(result[key]-D(summary[reference]))<D('1e-18'),key
        results[case['id']]=result
        for name,rows in (('episodes',episodes),('reducing_receipts',receipts),('legs',legs)):
            with (output/(case['id']+'--'+name+'.jsonl')).open('w') as f:
                for row in rows:
                    f.write(json.dumps(row,default=str,separators=(',',':'))+'\n')
    write(output/'results.json',dict(source_run=str(source),window=reg['spec']['window'],
                                   definition='completed net-inventory flat-to-flat episodes; add/reduce included; reversals split at zero; fees and observed funding included',
                                   secondary_definition='average-cost reducing fill receipts; entry fees and funding allocated by closed inventory share',
                                   results=results))
    write(output/'validation.json',dict(status='passed',source_sha256=hashes,script_sha256=sha(__file__),
                                      reconciled='ledger realized, inventory, fees, funding, unrealized, total PnL; two independent PnL partitions',
                                      future_markout_used=False,private_orders_sent=False))
    print(json.dumps(results,default=str,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.source.resolve(),a.output.resolve())
