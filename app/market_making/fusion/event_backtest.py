"""Event-triggered Kernel/OMS replay on independent exchange/publication clocks.

Native HFT owns all queue/fill/transport mechanics. Python wall-clock speed is
not interpreted as Rust execution speed: the explicit compute budget is a
scenario assumption. Strategy behavior is provided only by the registered plugin.
"""
from dataclasses import asdict
from decimal import Decimal as D
import hashlib,json,math,time
from pathlib import Path
import numpy as np
from numba import njit
from ...artifacts import digest,write,ROOT,source_identity
from .config import Factors
from .kernel import Kernel
from .timing import next_timer_ns

LOCAL=1 << 30


def advance_segment_start(engine, minimum_ns):
    """Bootstrap an independent book without reversing application time.

    Prior-segment late publications can overlap the next exchange snapshot.
    No owned orders exist during bootstrap; queued visible inputs are consumed
    at the monotonic application time, never backdated into earlier decisions.
    """
    now=int(engine.current_timestamp)
    delay=max(0,minimum_ns-now)
    if delay and engine.elapse(delay)!=0:
        raise ValueError('Overlapping segment ends before monotonic bootstrap completes')
    return delay


@njit
def wake(hbt,timeout):
    result=hbt.wait_next_feed(True,timeout)
    if result in (0,2,3):
        # A depth packet has one row per level. Never quote an intermediate
        # half-applied packet when multiple rows share its receipt timestamp.
        advanced=hbt.elapse(0)
        if advanced != 0:return advanced
    return result


class EventCursor:
    def __init__(self,root,files,rules=None):
        self.root=root;self.files=iter(files);self.sequence=0;self.data=None
        self.rules=rules
        self.load()

    def load(self):
        while True:
            part=next(self.files,None)
            if part is None:self.data=None;return
            path=self.root/part['path']
            if digest(path)!=part['sha256']:raise ValueError('changed event tape')
            with np.load(path,allow_pickle=False) as z:data=z['data']
            if len(data)!=part['rows']:raise ValueError('event rows')
            if self.rules is not None:
                for name,step in (('px',self.rules.tick),('qty',self.rules.step)):
                    units=data[name]/float(step)
                    if np.any(np.abs(units-np.rint(units))>1e-6):raise ValueError('off-grid tape '+name)
            self.data=data[(data['ev'] & LOCAL)!=0]
            if not len(self.data):continue
            self.times=self.data['local_ts'];self.index=0
            if np.any(np.diff(self.times)<0):raise ValueError('local timestamp inversion')
            self.kinds=self.data['ev'] & 255
            self.depth=np.flatnonzero((self.kinds==1)|(self.kinds==4))
            self.trades=np.flatnonzero(self.kinds==2)
            return

    @property
    def next_ns(self):
        return int(self.times[self.index]) if self.data is not None else 2**63-1

    def consume(self,now,kernel):
        received=None
        while self.data is not None and self.next_ns<=now:
            end=int(np.searchsorted(self.times,now,side='right'))
            a,b=np.searchsorted(self.depth,[self.index,end])
            if b>a:received=int(self.times[self.depth[b-1]])
            a,b=np.searchsorted(self.trades,[self.index,end])
            for index in self.trades[a:b]:
                row=self.data[index];self.sequence+=1
                # Busy application handles queued events at processing time,
                # like Runtime.handle; source receipt is retained for book age.
                kernel.trade(now,self.sequence,1 if int(row['ev'])&(1<<29) else -1,
                             D(str(float(row['px']))),D(str(float(row['qty']))))
            self.index=end
            if self.index==len(self.data):self.load()
        return received


class EventMarks:
    def __init__(self,path):
        self.data=np.load(path,mmap_mode='r',allow_pickle=False);self.index=0
        if np.any(np.diff(self.data['local_ts'])<=0):raise ValueError('mark timestamps')

    @property
    def next_ns(self):
        return int(self.data['local_ts'][self.index]) if self.index<len(self.data) else 2**63-1

    def consume(self,now,kernel):
        end=int(np.searchsorted(self.data['local_ts'],now,side='right'))
        if end>self.index:
            row=self.data[end-1];kernel.mark_price(int(row['local_ts']),D(str(float(row['price']))))
        self.index=end


def run_event_arm(dataset,output,mask,config,marks_path,funding_rows,rules,execution,
                  max_segments=None,max_files=None,observer_factory=None,
                  strategy_factory=None,queue_model='risk_adverse'):
    if not callable(strategy_factory):
        raise ValueError("Explicit strategy_factory required for replay")
    from .queue_models import QUEUE_MODELS,configure_queue
    if queue_model not in QUEUE_MODELS:raise ValueError('Unsupported queue model')
    import importlib.metadata
    import hftbacktest as h
    from hftbacktest.order import PARTIALLY_FILLED
    from .native_reads import best_quotes,order_state,order_clocks
    version=importlib.metadata.version('hftbacktest')
    if version!='2.4.4+makerfix2':raise RuntimeError('verified makerfix2 required')
    manifest=json.loads((dataset/'manifest.json').read_text())
    if (execution.trigger!='event' or manifest['version']!=6 or manifest.get('unresolved_trade_gaps')
            or manifest.get('feed_clock')!='publication_E_plus_transport_exchange_T'
            or int(manifest['feed_latency_ms']*1e6)!=execution.feed_ns):
        raise ValueError('certified publication-clock tape and event profile required')
    if execution.wire_prepare_ns or execution.trade_feed_ns or execution.mark_feed_ns:
        raise ValueError('event tape already carries per-channel publication times; do not add delays twice')
    output.mkdir(parents=True,exist_ok=False)
    observer=observer_factory(output) if observer_factory else None
    command_observer=getattr(observer,'command',None)
    update_observer=getattr(observer,'order_update',None)
    halt_observer=getattr(observer,'halt',None)
    factors=Factors.from_mask(mask)
    k=Kernel(rules,config,factors,prefix=f'e{mask}',strategy_factory=strategy_factory)
    trade_window=getattr(k.strategy,'trading_window_ns',None)
    marks=EventMarks(marks_path);funding_i=0;funding_log=[];curves=[];reports=[]
    trace=hashlib.sha256();started=time.monotonic();cash_offset=0.;position_offset=D(0)
    native_fills=0;native_fees=0.;first_halt=None;counters=dict(decisions=0,market_wakes=0,response_wakes=0,timer_wakes=0,busy_rechecks=0)
    identity=dict(symbol=manifest['symbol'],mask=mask,factors=asdict(factors),config=config.json(),rules=asdict(rules),
        execution_profile=execution.json(),dataset_manifest_sha256=digest(dataset/'manifest.json'),
        marks_sha256=digest(marks_path),hftbacktest=version,queue=queue_model,exchange='partial_fill',
        clock='exchange T; public E+1.5ms; compute then send; native entry and response',
        same_timestamp_policy='apply complete timestamp batch before decision; unknown intra-ms order not reconstructed',
        funding_sha256=hashlib.sha256(json.dumps(funding_rows,sort_keys=True).encode()).hexdigest(),
        gap_policy='carry inventory/cash; old orders assumed canceled without unknown-interval fills; conditional results',
        cpu_budget_is_assumed=True,private_rtt_is_measured=False,
        historical_rules='current exchange metadata; tick/lot checked against tape; historic dynamic limits not independently verified',
        parameter_selection='preregistered; no PnL tuning',
        source_sha256=source_identity(ROOT))
    if observer is not None:
        identity['observations']=observer.spec
        for name in getattr(observer,'additional_sources',()):
            identity['source_sha256'][name]=digest(ROOT/name)
    strategy_identity=getattr(k.strategy,'identity',None)
    if strategy_identity is not None:
        identity['strategy']=strategy_identity
        for name in getattr(k.strategy,'additional_sources',()):
            identity['source_sha256'][name]=digest(ROOT/name)
    write(output/'run.json',identity)

    signatures={}
    def sync(b,now):
        changed=False
        for o in list(k.oms.active.values()):
            present,leaves,status,exectick=order_state(b,o.serial)
            if not present:raise AssertionError('native lost order')
            if status not in (h.NEW,PARTIALLY_FILLED,h.FILLED,h.CANCELED,h.EXPIRED):continue
            signature=(leaves,status,exectick)
            if signatures.get(o.serial)==signature:continue
            signatures[o.serial]=signature;changed=True
            cumulative=o.qty-D(str(leaves)).quantize(rules.step)
            if cumulative>o.filled and D(int(exectick))*rules.tick!=o.price:
                raise AssertionError('passive native execution price differs from owned limit')
            label={h.NEW:'NEW',PARTIALLY_FILLED:'PARTIALLY_FILLED',h.FILLED:'FILLED',h.CANCELED:'CANCELED',h.EXPIRED:'EXPIRED'}[int(status)]
            fill=k.update(o.cid,label,cumulative,cumulative*o.price,now)
            if update_observer is not None:
                exchange_ns,request_ns,pending_req=order_clocks(b,o.serial)
                update_observer(o.cid,label,cumulative,now,exchange_ns,request_ns,pending_req)
            if o.cid not in k.oms.active:signatures.pop(o.serial,None)
            if fill:
                trace.update(json.dumps(fill,default=str,sort_keys=True).encode())
                if observer is not None:observer.fill(fill)
        b.clear_inactive_orders(0)
        if not math.isclose(float(k.oms.ledger.position-position_offset),b.position(0),abs_tol=1e-8):raise AssertionError('native position mismatch')
        state=b.state_values(0)
        if not math.isclose(float(k.oms.cash_flow),cash_offset+state.balance-state.fee+float(k.oms.ledger.funding),abs_tol=1e-6):raise AssertionError('native cash mismatch')
        return changed

    def settle(now,gap=False):
        nonlocal funding_i
        while funding_i<len(funding_rows) and funding_rows[funding_i]['fundingTime']*1_000_000+execution.response_ns<=now:
            row=funding_rows[funding_i];receipt=row['fundingTime']*1_000_000+execution.response_ns
            if not gap and receipt!=now:raise AssertionError('missed funding clock')
            # With uniform response latency, the position visible at T+response
            # is the exchange position at T, rather than a future inventory.
            position=k.oms.ledger.position;payment=-position*D(row['markPrice'])*D(row['fundingRate'])
            # The settlement price computes exchange cash, but its publication
            # may arrive later than the private cash receipt. Risk may use only
            # a mark/mid already visible to the application.
            observed=k.mark or k.strategy.obs.mid
            if observed>0:k.oms.risk(observed,now)
            k.oms.funding(payment)
            if observed>0:k.oms.risk(observed,now)
            funding_log.append(dict(**row,receipt_ns=receipt,applied_ns=now,position=str(position),cashflow=str(payment),conditional_gap=gap))
            funding_i+=1

    for segment in manifest['segments'][:max_segments]:
        files=segment['files'][:max_files] if max_files else segment['files']
        end=max(p['clock_end_ns'] for p in files)
        if trade_window:
            if segment['start_ns']>=trade_window[1]:continue
            end=min(end,trade_window[1]-1)
        cursor=EventCursor(dataset,files,rules);cursor.sequence=k.strategy.obs.last_trade_id+1
        # Native EOF can be raised while advancing the exchange iterator before
        # its final local feed/fill response has been delivered. An unreachable
        # clock marker keeps both iterators alive until our certified cutoff.
        # The replay never advances to this marker, so it cannot mutate a book.
        marker=np.zeros(1,dtype=h.event_dtype)
        marker_time=end+execution.request_ns+execution.response_ns+execution.decision_ns+1_000_000_000
        marker[0]=(h.EXCH_EVENT|h.LOCAL_EVENT|h.DEPTH_EVENT|h.BUY_EVENT,
                   marker_time,marker_time,float(rules.tick),0.,0,0,0.)
        asset=(h.BacktestAsset().data([*[str(dataset/p['path']) for p in files],marker]).linear_asset(1.)
               .tick_size(float(rules.tick)).lot_size(float(rules.step))
               .constant_order_latency(execution.request_ns,execution.response_ns))
        asset=(configure_queue(asset,queue_model).partial_fill_exchange()
               .trading_value_fee_model(float(config.as_config.maker_fee),float(execution.taker_fee)))
        market_observer=getattr(k.strategy,'observe_market',None)
        if market_observer is not None:asset=asset.last_trades_capacity(200000)
        b=h.HashMapMarketDepthBacktest([asset]);b.elapse(1)
        bootstrap_delay=advance_segment_start(b,max(k.strategy.obs.last_clock or 0,k.oms.clock_ns or 0))
        k.reset_market();signatures.clear();position_offset=k.oms.ledger.position
        now=int(b.current_timestamp);before=k.oms.ledger.equity(k.mark) if k.mark else config.initial_cash
        last_curve=0;pending=False;segment_start=now
        try:
            marks.consume(now,k);settle(now,gap=True)
            while now<=end:
                # elapse(compute) may have already delivered events. Process the
                # queued batch immediately, before waiting for another feed.
                if not pending and cursor.next_ns>now and marks.next_ns>now:
                    deadline=min(end,next_timer_ns(k,now),marks.next_ns)
                    if funding_i<len(funding_rows):deadline=min(deadline,funding_rows[funding_i]['fundingTime']*1_000_000+execution.response_ns)
                    result=wake(b,max(0,deadline-now))
                    if result==1:break
                    if result not in (0,2,3):raise RuntimeError('native wait error '+str(result))
                    counters[{0:'timer_wakes',2:'market_wakes',3:'response_wakes'}[result]]+=1
                else:counters['busy_rechecks']+=1
                now=int(b.current_timestamp);pending=False
                if now>end:break
                sync(b,now)
                received=cursor.consume(now,k)
                bid,ask,bt,at,bq,aq=best_quotes(b)
                if received is not None and math.isfinite(bid) and math.isfinite(ask) and 0<bid<ask:
                    if hasattr(k.strategy,'on_top_of_book'):
                        k.strategy.on_top_of_book(now,D(int(bt))*rules.tick,D(str(bq)),D(int(at))*rules.tick,D(str(aq)))
                    k.book(now,D(int(bt))*rules.tick,D(int(at))*rules.tick)
                    if observer is not None:observer.mid(now,D(int(bt+at))*rules.tick/2)
                    k.strategy.obs.book_ns=received
                elif received is not None:
                    k.reset_market()
                    if observer is not None:observer.gap('invalid_book')
                marks.consume(now,k);settle(now)
                if market_observer is not None:
                    from .native_reads import observe_strategy
                    observe_strategy(b,k.strategy,now)
                targets=k.plan(now);counters['decisions']+=1
                decision_ns,decision_bid,decision_ask=now,bid,ask
                finish=min(now+execution.decision_ns,end)
                truncated=finish<now+execution.decision_ns
                timer=next_timer_ns(k,now)
                while funding_i<len(funding_rows):
                    receipt=funding_rows[funding_i]['fundingTime']*1_000_000+execution.response_ns
                    if not now<receipt<=finish:break
                    if b.elapse(receipt-int(b.current_timestamp))!=0:raise RuntimeError('EOF during funding/compute')
                    sync(b,receipt);settle(receipt)
                if b.elapse(finish-int(b.current_timestamp))!=0:break
                now=int(b.current_timestamp)
                pending=sync(b,now) or timer<=now
                if truncated or now>=end:
                    # Drain the last real observations/responses without
                    # inventing a completed decision after the data boundary.
                    received=cursor.consume(now,k)
                    bid,ask,bt,at,bq,aq=best_quotes(b)
                    if received is not None and math.isfinite(bid) and math.isfinite(ask) and 0<bid<ask:
                        if hasattr(k.strategy,'on_top_of_book'):
                            k.strategy.on_top_of_book(now,D(int(bt))*rules.tick,D(str(bq)),D(int(at))*rules.tick,D(str(aq)))
                        k.book(now,D(int(bt))*rules.tick,D(int(at))*rules.tick)
                        if observer is not None:observer.mid(now,D(int(bt+at))*rules.tick/2)
                        k.strategy.obs.book_ns=received
                    marks.consume(now,k)
                    break
                # Frozen market intentions, refreshed private inventory/risk.
                for cmd in k.dispatch(targets,now):
                    o=k.oms.find(cmd.cid)
                    if cmd.kind=='new':
                        submit=b.submit_buy_order if cmd.side==1 else b.submit_sell_order
                        result=submit(0,o.serial,float(cmd.price),float(cmd.qty),h.GTX,h.LIMIT,False)
                    else:
                        native=b.orders(0).get(o.serial)
                        if native is None or not native.cancellable:raise AssertionError('cancel phase/native mismatch')
                        result=b.cancel(0,o.serial,False)
                    if result!=0:raise RuntimeError('native command error '+str(result))
                    if command_observer is not None:
                        command_observer(cmd,decision_ns,now,decision_bid,decision_ask)
                if now-last_curve>=1_000_000_000:
                    mark=k.mark or k.strategy.obs.mid
                    if mark>0:
                        k.oms.risk(mark,now)
                        curves.append((now,float(k.oms.ledger.equity(mark)),float(k.oms.ledger.position)))
                    last_curve=now
                if first_halt is None and k.oms.halt_reason:
                    first_halt=now
                    if halt_observer is not None:
                        ledger=k.oms.ledger
                        valuation=k.mark or k.strategy.obs.mid
                        halt_observer(dict(ns=now, reason=k.oms.halt_reason, mark=str(valuation),
                            equity=str(ledger.equity(valuation)), wallet=str(ledger.wallet),
                            position=str(ledger.position), average_entry=str(ledger.average_entry),
                            notional=str(abs(ledger.position)*valuation),
                            funding=str(ledger.funding), fees=str(ledger.fees),
                            realized=str(ledger.realized), unrealized=str(ledger.unrealized(valuation)),
                            active_orders=[dict(cid=o.cid,side=o.side,price=str(o.price),
                                                remaining=str(o.remaining),phase=str(o.phase))
                                           for o in k.oms.working()]))
                if now>=end:break
            now=int(b.current_timestamp);sync(b,now)
            state=b.state_values(0);cash_offset+=state.balance-state.fee
            native_fills+=int(state.num_trades);native_fees+=state.fee
            for o in list(k.oms.active.values()):
                k.update(o.cid,'CANCELED',o.filled,o.quote_filled,now);k.oms.stats['boundary_assumed_cancels']+=1
                if update_observer is not None:
                    update_observer(o.cid,'CANCELED',o.filled,now,0,0,0,True)
            k.markouts.gap()
            if observer is not None:observer.gap()
            report=k.report();report.update(segment_id=segment['id'],start_ns=segment_start,end_ns=now,
                monotonic_bootstrap_delay_ns=bootstrap_delay,
                rows=sum(p['rows'] for p in files),native_fills=native_fills,
                incremental_pnl=str(k.oms.ledger.equity(k.mark or k.strategy.obs.mid)-before),trace_sha256=trace.hexdigest())
            reports.append(report);write(output/f"segment_{segment['id']:03d}.json",report)
            write(output/'progress.json',dict(segments=len(reports),wall_s=time.monotonic()-started,**counters))
            print(manifest['symbol'],factors.label,'segment',segment['id'],'pnl',report['pnl'],'fills',native_fills,'wall',round(time.monotonic()-started,1),flush=True)
        finally:b.close()
    if trade_window:
        curves=[row for row in curves if trade_window[0]<=row[0]<trade_window[1]]
        reports=[{**r,'start_ns':max(r['start_ns'],trade_window[0]),'end_ns':min(r['end_ns'],trade_window[1]-1)}
                 for r in reports if r['end_ns']>=trade_window[0] and r['start_ns']<trade_window[1]]
    final=k.report();final.update(identity=identity,segments=reports,rows=sum(r['rows'] for r in reports),
        native_fills=native_fills,native_fees=native_fees,trace_sha256=trace.hexdigest(),
        wall_s=time.monotonic()-started,funding_receipts=funding_log,invariant_failures=0,
        valuation_mark=str(k.mark or k.strategy.obs.mid),first_halt_ns=first_halt,counters=counters,
        terminal='mark to latest observed mark; residual inventory NOT assumed liquidated',
        full_range=max_segments is None and max_files is None)
    if hasattr(k.strategy,'report_diagnostics'):
        final['native_strategy_diagnostics']=k.strategy.report_diagnostics(now)
    write(output/'summary.json',final)
    np.savez_compressed(output/'equity.npz',data=np.asarray(curves,dtype=[('local_ns','i8'),('equity','f8'),('position','f8')]))
    if observer is not None:observer.finish(final)
    return final
