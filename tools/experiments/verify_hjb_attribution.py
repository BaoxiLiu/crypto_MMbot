"""Verify observer-only replay parity and summarize exact PnL attribution."""
import argparse
from datetime import datetime, timezone
from decimal import Decimal as D
import json
from pathlib import Path

from app.backtesting.html_report import read_report, sha

CONTROLS=('window','symbols','datasets','rules','engine','execution','costs',
          'account','fee_scenarios','loss_stops','strategy_parameters')


def summarize_attribution(row,record):
    components=('execution_edge','inventory_drift','gap_inventory_drift','final_mark_basis','funding')
    reconstructed=sum((D(record[k]) for k in components),D(0))-D(record['fees'])
    if abs(reconstructed-D(row['pnl']))>D('1e-12') or abs(D(record['residual']))>D('1e-12'):
        raise ValueError('Attribution identity failed')
    halt=record['first_halt']
    summary=dict(symbol=row['symbol'],strategy=row['strategy'],**record)
    if halt is not None:
        if D(halt['equity'])>0:
            summary['halt_notional_to_equity']=str(D(halt['notional'])/D(halt['equity']))
        summary['halt_utc']=datetime.fromtimestamp(halt['ns']/1e9,timezone.utc).isoformat()
        summary['pnl_at_first_halt']=str(D(halt['equity'])-D(row['initial_cash']))
        summary['equity_change_after_halt']=str(D(record['final_equity'])-D(halt['equity']))
        halt_basis=D(halt['position'])*(D(halt['mark'])-D(halt['visible_mid']))
        post_pnl=(D(record['post_halt_execution_edge'])+
                  D(record['post_halt_inventory_drift'])+
                  D(record['post_halt_gap_inventory_drift'])+
                  D(record['final_mark_basis'])-halt_basis+
                  D(record['funding_after_halt'])-D(record['fees_after_halt']))
        post_residual=D(summary['equity_change_after_halt'])-post_pnl
        if abs(post_residual)>D('1e-12'):
            raise ValueError('Post-halt accounting does not reconcile')
        summary['halt_mark_basis']=str(halt_basis)
        summary['post_halt_reconstructed_pnl']=str(post_pnl)
        summary['post_halt_residual']=str(post_residual)
        if halt['reason']=='margin_stress_breach_evaluation_halted' and D(halt['equity'])>D(halt['notional']):
            raise ValueError('Margin halt snapshot contradicts the 1x stress guard')
    return summary


def verify(old_path, replay_path, output):
    old=read_report(old_path)
    new=read_report(replay_path)
    if new['status']!='completed' or not new['full_range'] or new['validation']['status']!='passed':
        raise ValueError('A complete validated observation replay is required')
    if len(new['results']['totals'])!=4:
        raise ValueError('Expected all four adaptive baseline cases')
    for key in CONTROLS:
        if old['spec'].get(key)!=new['spec'].get(key):
            raise ValueError('Replay control changed: '+key)
    before=dict(old['spec']['observations'])
    after=dict(new['spec']['observations'])
    if not after.pop('pnl_attribution',False) or after!=before:
        raise ValueError('Expected an observation-only configuration change')
    reference={(r['symbol'],r['strategy']):r for r in old['results']['totals']}
    controls={(c['symbol'],c['strategy']):c for c in old['effective_cases']}
    attribution=new['supplemental']['pnl_attribution_metrics.json']
    if len(attribution)!=4:
        raise ValueError('Incomplete attribution')
    rows=[]
    for case in new['effective_cases']:
        key=(case['symbol'],case['strategy'])
        previous=controls[key]
        if case['config']!=previous['config'] or case['plugin']!=previous['plugin']:
            raise ValueError('Execution or quoting configuration changed')
    for row in new['results']['totals']:
        key=(row['symbol'],row['strategy'])
        previous=reference[key]
        for name in ('pnl','total_notional','total_qty','fill_receipts','native_executions','max_drawdown'):
            if D(str(row[name]))!=D(str(previous[name])):
                raise ValueError(f'Observation replay parity failed: {key} {name}')
        summary=summarize_attribution(row,attribution[row['case']])
        rows.append(summary)
    result=dict(status='passed',cases=4,baseline_parity='exact',
                sources=[dict(path=str(p),sha256=sha(p.read_bytes())) for p in (old_path,replay_path)],
                scope='causal receipt-time-mid self-financing identity; no double-counting future markouts',
                rows=rows)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('old_report',type=Path)
    parser.add_argument('observation_report',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    verify(args.old_report,args.observation_report,args.output)
