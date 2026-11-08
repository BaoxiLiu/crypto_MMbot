"""Binance USD-M REST/private event contract; transport never mutates strategy.

HMAC REST is deliberately independent of the legacy websocket request loop.
Timeout/ambiguous execution retains risk until a query/account reconciliation.
"""
from decimal import Decimal as D
import hashlib,hmac,json,time
from urllib.parse import urlencode

REST='https://fapi.binance.com'
PUBLIC='wss://fstream.binance.com/public/stream'
MARKET='wss://fstream.binance.com/market/stream'
PRIVATE='wss://fstream.binance.com/private/ws/'


def signed_query(params,secret,timestamp):
    values=dict(params,timestamp=timestamp,recvWindow=5000)
    payload=urlencode(sorted(values.items()))
    return payload+'&signature='+hmac.new(secret.encode(),payload.encode(),hashlib.sha256).hexdigest()


def order_params(command,symbol):
    if command.kind=='cancel':return dict(symbol=symbol,origClientOrderId=command.cid)
    if command.kind!='new':raise ValueError('unknown command')
    return dict(symbol=symbol,side='BUY' if command.side==1 else 'SELL',positionSide='BOTH',
                type='LIMIT',timeInForce='GTX',quantity=str(command.qty),price=str(command.price),
                newClientOrderId=command.cid,reduceOnly='true' if command.reduce_only else 'false',
                newOrderRespType='ACK')


class RequestFailure(Exception):
    def __init__(self,code,status=0,ambiguous=True):
        self.code,self.status,self.ambiguous=code,status,ambiguous
        super().__init__(f'Binance request failed: HTTP {status}, code {code}, ambiguous={ambiguous}')


class RestClient:
    def __init__(self,session,key='',secret=''):
        self.session,self.key,self.secret=session,key,secret
        self.offset_ms=0;self.last_request=0

    async def request(self,method,path,params=None,signed=True,mutating=False):
        import asyncio
        # A bounded total timeout; no implicit retry of a trading command.
        query=(signed_query(params or {},self.secret,time.time_ns()//1_000_000+self.offset_ms)
               if signed else urlencode(params or {}))
        headers={'X-MBX-APIKEY':self.key} if self.key else {}
        try:
            async with self.session.request(method,REST+path+('?' +query if query else ''),
                                            headers=headers,timeout=8) as r:
                data=await r.json(content_type=None)
                if r.status>=400 or (isinstance(data,dict) and int(data.get('code',0))<0):
                    code=int(data.get('code',0)) if isinstance(data,dict) else 0
                    # 503 may explicitly reject, but conservatively query every
                    # 5xx/timeout/duplicate outcome. DELETE failure also queries.
                    ambiguous=(r.status>=500 or code in (-1000,-1001,-1006,-1007,-2010) or method=='DELETE')
                    raise RequestFailure(code,r.status,ambiguous=ambiguous)
                return data
        except RequestFailure:raise
        except (asyncio.TimeoutError,ConnectionError,OSError,ValueError) as e:
            raise RequestFailure(type(e).__name__,ambiguous=True) from None

    async def command(self,command,symbol):
        return await self.request('POST' if command.kind=='new' else 'DELETE','/fapi/v1/order',
                                  order_params(command,symbol),mutating=True)


def verify_execution(kernel,cid,quantity,quote):
    o=kernel.oms.find(cid)
    if o is None:raise ValueError('unowned execution response')
    if (not quantity.is_finite() or not quote.is_finite() or quantity<0 or quote<0
            or (quantity==0 and quote!=0)
            or abs(quote-quantity*o.price)>quantity*kernel.rules.tick/D(100)):
        kernel.oms.halt_reason='conflicting_execution_receipt'
        raise ValueError(kernel.oms.halt_reason)


def apply_order(kernel,row,ns):
    """REST cumulative execution; an ACK with no status is not a fill."""
    cid=row['clientOrderId']
    if 'status' not in row:return
    quantity=D(row.get('executedQty','0'))
    quote=D(row['cumQuote']) if 'cumQuote' in row else quantity*D(row.get('avgPrice','0'))
    verify_execution(kernel,cid,quantity,quote)
    return kernel.update(cid,row['status'],quantity,quote,ns,row.get('orderId',0),row.get('updateTime',0))


class PrivateEvents:
    def __init__(self,kernel,symbol,quote_asset):
        self.kernel,self.symbol,self.quote_asset=kernel,symbol,quote_asset
        self.funding_ids=set()

    def apply(self,event,ns):
        k=self.kernel
        k.oms.clock(ns)
        kind=event.get('e')
        if kind=='ORDER_TRADE_UPDATE':
            o=event['o']
            if o['s']!=self.symbol:
                k.oms.halt_reason='foreign_symbol_event';return
            own=k.oms.find(o['c'])
            if own is None:
                k.oms.halt_reason='unowned_private_order';return
            if (o.get('ps','BOTH')!='BOTH' or o['S']!=('BUY' if own.side==1 else 'SELL')
                    or ('q' in o and D(o['q'])!=own.qty) or ('p' in o and D(o['p'])!=own.price)):
                k.oms.halt_reason='private_order_contract_mismatch';raise ValueError(k.oms.halt_reason)
            cumulative=D(o['z']);quote=cumulative*D(o['ap'])
            verify_execution(k,o['c'],cumulative,quote)
            fill=k.update(o['c'],o['X'],cumulative,quote,ns,o.get('i',0),event.get('T',0))
            if o.get('x')=='TRADE':
                if not o.get('m',False):k.oms.halt_reason='unexpected_taker_execution'
                k.oms.actual_fee(o['c'],o['t'],D(o['l']),D(o.get('n','0') or '0'),o.get('N'),self.quote_asset,ns)
            return fill
        if kind=='ACCOUNT_UPDATE':
            account=event['a']
            if account.get('m')=='FUNDING_FEE':
                for b in account.get('B',[]):
                    amount=D(b.get('bc','0'))
                    if b['a']!=self.quote_asset:
                        if amount:k.oms.halt_reason='foreign_funding_asset'
                        continue
                    prefix=f"{event['T']}:{b['a']}:"
                    key=prefix+str(amount)
                    if key not in self.funding_ids and any(old.startswith(prefix) for old in self.funding_ids):
                        k.oms.halt_reason='conflicting_funding_receipt';raise ValueError(k.oms.halt_reason)
                    if key not in self.funding_ids:
                        self.funding_ids.add(key);k.oms.funding(amount)
            elif account.get('m') not in ('ORDER',):
                # Deposits/withdrawals/insurance/margin adjustments must not be
                # silently interpreted as strategy PnL or reset the loss limit.
                k.oms.halt_reason='external_account_change'
        elif kind in ('listenKeyExpired','MARGIN_CALL','ACCOUNT_CONFIG_UPDATE'):
            k.oms.halt_reason=kind;k.oms.reconciled=False


def account_matches(kernel,account,open_orders,symbol,quote_asset):
    """Require a dedicated, one-way account. Never overwrite mismatched cash."""
    if not account.get('canTrade',True):return False
    if any(r['symbol']!=symbol for r in open_orders):return False
    owned={o.cid for o in kernel.oms.active.values()}
    if any(r['clientOrderId'] not in owned for r in open_orders):return False
    if {r['clientOrderId'] for r in open_orders}!=owned:return False
    for row in open_orders:
        o=kernel.oms.find(row['clientOrderId'])
        if (D(row['origQty'])!=o.qty or D(row['executedQty'])!=o.filled or D(row['price'])!=o.price
                or row['side']!=('BUY' if o.side==1 else 'SELL')
                or row.get('positionSide','BOTH')!='BOTH' or row['type']!='LIMIT'
                or row['timeInForce']!='GTX' or row.get('reduceOnly',False)!=o.reduce_only):return False
    positions=account.get('positions',[])
    if any(D(p['positionAmt'])!=0 and p['symbol']!=symbol for p in positions):return False
    if any(p.get('positionSide','BOTH')!='BOTH' for p in positions):return False
    rows=[p for p in positions if p['symbol']==symbol]
    position=sum((D(p['positionAmt']) for p in rows),D(0))
    assets=[b for b in account['assets'] if b['asset']==quote_asset]
    if len(assets)!=1:return False
    wallet=D(assets[0]['walletBalance'])
    if position!=kernel.oms.ledger.position:return False
    # Cumulative REST replies can aggregate opposite-side fills in a different
    # order than the matching engine. That changes average-entry/realized split,
    # NEVER net cash or equity. Compare the independent economic cash identity
    # instead of falsely demanding identical reconstructed average-cost paths.
    # No exchange balance overwrites our ledger. Allow only published 8-decimal
    # entry/wallet rounding; actual fee receipts are required by the caller.
    if len(rows)>1:return False
    entry=D(rows[0]['entryPrice']) if rows else D(0)
    if not entry.is_finite() or not wallet.is_finite():return False
    account_cash=wallet-position*entry-kernel.oms.ledger.initial
    tolerance=D('.00000001')*(1+abs(position))
    return abs(account_cash-kernel.oms.cash_flow)<=tolerance
