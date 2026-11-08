"""Common contract for newly marked quick/full research campaigns."""
from datetime import date
from decimal import Decimal as D

SYMBOLS=('BTCUSDC','ETHUSDC','SOLUSDC','DOGEUSDC','SUIUSDC','UNIUSDC','LTCUSDC',
         'CRVUSDC','XRPUSDC','WIFUSDC','1000PEPEUSDC','1000SHIBUSDC','1000BONKUSDC')


def validate_standard(spec):
    if spec.get('stage') not in ('quick','full'):return
    if set(spec['symbols'])!=set(SYMBOLS) or len(spec['symbols'])!=13:
        raise ValueError('Standard quick/full campaigns require all 13 coins')
    scenarios=spec.get('fee_scenarios',[])
    if len(scenarios)!=2 or {D(str(f['maker_fee'])) for f in scenarios}!={D(0),D('-0.00003')}:
        raise ValueError('Standard campaigns require maker 0 and -0.3bp scenarios')
    if spec['engine']['queue']!='cancel_50_50' or D(str(spec['execution']['taker_fee']))!=D('.0004'):
        raise ValueError('Standard campaigns require 50/50 queue and 4bp taker')
    if D(str(spec.get('account',{}).get('max_notional','0')))!=D(1000):
        raise ValueError('Standard campaigns require 1000 USDC per-coin exposure cap')
    if spec['stage']=='quick' and (date.fromisoformat(spec['window']['end_exclusive'])-date.fromisoformat(spec['window']['start'])).days!=3:
        raise ValueError('Quick campaigns require three UTC calendar days')


def validate_new_research(spec):
    """Current policy for every newly prepared run, including custom subsets.

    Validated before preparation and again before running registered work.
    """
    if not {'BTCUSDC', 'ETHUSDC'} <= set(spec['symbols']):
        raise ValueError('New research requires BTCUSDC and ETHUSDC; both are mandatory')
    validate_standard(spec)
    scenarios = spec.get('fee_scenarios', [])
    if len(scenarios) != 2 or {D(str(f['maker_fee'])) for f in scenarios} != {D(0), D('-0.00003')}:
        raise ValueError('New research requires maker 0 and -0.3bp scenarios')
    if spec['engine']['queue'] != 'cancel_50_50' or D(str(spec['execution']['taker_fee'])) != D('.0004'):
        raise ValueError('New research requires 50/50 queue and 4bp taker')
    if D(str(spec.get('account', {}).get('max_notional', '0'))) != D(1000):
        raise ValueError('New research requires 1000 USDC per-coin exposure cap')
    if spec['outputs'].get('format') != 'html' or spec['outputs'].get('retention') != 'final_only':
        raise ValueError('New research requires HTML and final_only retention')


def validate_preparation(spec):
    validate_new_research(spec)
    return {'mode': 'new_research', 'policy_version': 1}
