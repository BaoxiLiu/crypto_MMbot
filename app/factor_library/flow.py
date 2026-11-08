"""Causal L2 flow extension v1.1.0; all inputs use observed receipt nanoseconds.

Snapshots/gaps reset the epoch. Full depth messages are atomic, never row events.
Returns NaN for warm-up, incomplete volume history, missing denominators and
stale volume buckets. Quantity declines are net L2 proxies, NOT cancellations.

v1.1.0 adds S17-S27: near-depth statics (S17/S18), per-side replenish shares and
net-flow share (S19-S21), taker pressure (S22-S24) and volume-clock bucket
dynamics (S25-S27). S19-S21 share the net-L2-proxy caveat of S02-S05; S23 uses
the book mid and inherits the book-stale gate; S25/S27 consume signed bucket
imbalances. None of the additions are cancellations or identified order flow.
"""
from collections import deque
from dataclasses import dataclass
from bisect import bisect_left
import heapq
import math
import statistics

FACTOR_IDS = tuple(f'S{i:02d}' for i in range(1, 28))
NAN = float('nan')
SECOND = 1_000_000_000

@dataclass(frozen=True)
class FlowParams:
    tick_size: float
    quota_usdc: float = 10000.0
    near_ticks: int = 2
    retention_seconds: int = 60
    bucket_count: int = 20
    max_events: int = 200000
    book_stale_ms: int = 3000
    def __post_init__(self):
        if not all(math.isfinite(x) and x > 0 for x in (self.tick_size, self.quota_usdc)):
            raise ValueError('Positive finite tick and quota required')
        if self.near_ticks < 0 or self.retention_seconds < 5 or self.bucket_count < 2 or self.max_events < 2 or self.book_stale_ms <= 0:
            raise ValueError('Invalid flow parameters')

class FlowFactors:
    version = '1.1.0'
    def __init__(self, params):
        self.params = params
        self.reset()
    def reset(self):
        self.epoch_start = self.clock = self.book_time = None
        self.book = {'bid': {}, 'ask': {}}
        self.heaps = {'bid': [], 'ask': []}
        self.depth = deque()  # receipt, near bid add, ask add, bid decrease, ask decrease
        self.trades = deque()  # receipt, USDC, buyer initiated, ID, quantity
        self.last_trade = self.last_trade_id = None
        self.last_u = None
        self.closed = deque(maxlen=self.params.bucket_count)
        self.bucket_buy = self.bucket_sell = 0.0
        self.bucket_start = None
        self.invalid_trades = 0
    def _time(self, t):
        if type(t) is not int: raise ValueError('Receipt timestamp must be integer nanoseconds')
        if self.clock is not None and t < self.clock: raise ValueError('Non-causal/reordered receipt timestamp')
        self.clock = t
        cutoff = t - self.params.retention_seconds * SECOND
        for queue in (self.depth, self.trades):
            while queue and queue[0][0] < cutoff: queue.popleft()
        if len(self.depth) + len(self.trades) >= self.params.max_events:
            self.reset(); raise ValueError('Event memory bound exceeded; epoch invalidated')
    def _idx(self, p):
        p=float(p)
        if not math.isfinite(p) or p <= 0: raise ValueError('Invalid price')
        idx=round(p/self.params.tick_size)
        if idx <= 0 or not math.isclose(idx*self.params.tick_size,p,rel_tol=1e-10,abs_tol=self.params.tick_size*1e-7):
            raise ValueError('Price not on tick grid')
        return idx
    def _parse(self, levels):
        parsed=[];seen=set()
        for side,p,q in levels:
            if side not in self.book: raise ValueError('Invalid side')
            idx=self._idx(p);q=float(q)
            if not math.isfinite(q) or q < 0: raise ValueError('Invalid quantity')
            if (side,idx) in seen: raise ValueError('Duplicate level in atomic message')
            seen.add((side,idx));parsed.append((side,idx,q))
        return parsed
    def _put(self,side,idx,q):
        book=self.book[side];heap=self.heaps[side]
        if q:
            if idx not in book: heapq.heappush(heap,-idx if side=='bid' else idx)
            book[idx]=q
        else: book.pop(idx,None)
        if len(heap)>2*len(book)+100:
            heap[:]=[-i if side=='bid' else i for i in book];heapq.heapify(heap)
    def _best(self,side):
        book=self.book[side];heap=self.heaps[side]
        while heap and ((-heap[0] if side=='bid' else heap[0]) not in book):heapq.heappop(heap)
        return (-heap[0] if side=='bid' else heap[0]) if heap else None
    def _valid_book(self):
        b,a=self._best('bid'),self._best('ask')
        return b is not None and a is not None and b<a
    def snapshot(self,t,levels,last_update_id):
        parsed=self._parse(levels)
        if self.clock is not None and t < self.clock: raise ValueError("Reordered snapshot")
        self.reset();self._time(t)
        for side,idx,q in parsed:self._put(side,idx,q)
        if not self._valid_book():self.reset();raise ValueError('Invalid snapshot')
        self.epoch_start=self.book_time=t;self.last_u=int(last_update_id)
    def update(self,t,levels,first_id,final_id,prev_id):
        if self.epoch_start is None:raise ValueError('Snapshot required')
        parsed=self._parse(levels)
        if self.clock is not None and t<self.clock:raise ValueError('Reordered update')
        if int(first_id)>int(final_id):raise ValueError('Inverted IDs')
        if int(final_id)<=self.last_u:return False  # duplicate/snapshot overlap; not flow
        if int(prev_id)!=self.last_u:
            self.reset();raise ValueError('Depth sequence gap; new snapshot required')
        self._time(t)
        touches={'bid':self._best('bid'),'ask':self._best('ask')}
        v=[0.0]*4
        for side,idx,q in parsed:
            dq=q-self.book[side].get(idx,0.0)
            if abs(idx-touches[side]) <= self.params.near_ticks:
                slot=(0 if side=='bid' else 1)+(2 if dq<0 else 0)
                v[slot]+=abs(dq)*idx*self.params.tick_size
            self._put(side,idx,q)
        if not self._valid_book():self.reset();raise ValueError('Crossed/empty post-message book')
        self.depth.append((t,*v));self.book_time=t;self.last_u=int(final_id)
        return True
    def trade(self,t,price,qty,buyer_maker,trade_id,placeholder=False):
        if self.epoch_start is None:raise ValueError('Snapshot required')
        price,qty=float(price),float(qty)
        if type(buyer_maker) is not bool:raise ValueError('Invalid maker flag')
        if not all(math.isfinite(x) and x>=0 for x in (price,qty)):raise ValueError('Invalid trade numbers')
        if self.last_trade_id is not None and int(trade_id)!=self.last_trade_id+1:
            self.reset();raise ValueError('Trade ID gap/duplicate; epoch invalidated')
        self._time(t)
        if placeholder or price==0 or qty==0:
            self.last_trade_id=int(trade_id);self.invalid_trades+=1;return
        value=price*qty
        if not math.isfinite(value):raise ValueError('Notional overflow')
        # Fail before state mutation if one trade would generate unbounded buckets.
        if value/self.params.quota_usdc > 4096:raise ValueError('Trade exceeds bounded bucket split limit')
        self.last_trade_id=int(trade_id);self.last_trade=t
        buy=not buyer_maker
        self.trades.append((t,value,buy,int(trade_id),qty))
        if self.bucket_start is None:self.bucket_start=t
        left=value
        while left>0:
            need=self.params.quota_usdc-self.bucket_buy-self.bucket_sell
            chunk=min(left,need)
            if buy:self.bucket_buy+=chunk
            else:self.bucket_sell+=chunk
            left-=chunk
            if self.bucket_buy+self.bucket_sell >= self.params.quota_usdc*(1-1e-12):
                imbalance=(self.bucket_buy-self.bucket_sell)/self.params.quota_usdc  # signed; S14 takes abs
                self.closed.append((t,imbalance,(t-self.bucket_start)/1e6))
                self.bucket_start=t;self.bucket_buy=self.bucket_sell=0.0
            if left <= value*1e-14:break
    def _near_depth(self,side):
        touch=self._best(side)
        indices = range(touch-self.params.near_ticks,touch+1) if side=='bid' else range(touch,touch+self.params.near_ticks+1)
        return sum(idx*self.params.tick_size*self.book[side].get(idx,0.0) for idx in indices)
    def query(self,t):
        self._time(t);out=dict.fromkeys(FACTOR_IDS,NAN)
        if self.epoch_start is None:return out
        age=t-self.epoch_start
        depth=[r for r in self.depth if r[0]>t-SECOND]
        tr1=[r for r in self.trades if r[0]>t-SECOND]
        tr5=[r for r in self.trades if r[0]>t-5*SECOND]
        if age>=SECOND:
            book_fresh=t-self.book_time<=self.params.book_stale_ms*1000000
            if book_fresh:
                out['S01']=float(len(depth))
                for j,k in enumerate(['S02','S03','S04','S05']):out[k]=sum(r[j+1] for r in depth)
                db,da=self._near_depth('bid'),self._near_depth('ask')
                if db>0 and da>0:out['S16']=out['S04']/db-out['S05']/da
                if db+da>0:out['S17']=(db-da)/(db+da)
                b,a=self._best('bid'),self._best('ask')
                qb=b*self.params.tick_size*self.book['bid'].get(b,0.0)
                qa=a*self.params.tick_size*self.book['ask'].get(a,0.0)
                if qb+qa>0:out['S18']=(qb-qa)/(qb+qa)
                den=out['S02']+out['S04']
                if den>0:out['S19']=out['S02']/den
                den=out['S03']+out['S05']
                if den>0:out['S20']=out['S03']/den
                den=out['S02']+out['S03']+out['S04']+out['S05']
                if den>0:out['S21']=((out['S02']-out['S04'])-(out['S03']-out['S05']))/den
            out['S06']=sum(r[1] for r in tr1 if r[2]);out['S07']=sum(r[1] for r in tr1 if not r[2])
            den=out['S06']+out['S07']
            if den>0:out['S22']=(out['S06']-out['S07'])/den
        if age>=5*SECOND:
            slow=sum(r[1] for r in tr5)/5
            if slow>0:out['S08']=(sum(r[1] for r in self.trades if r[0]>t-SECOND//2)/.5)/slow
            times=[r[0] for r in tr5]
            intervals=[(b-a)/1e6 for a,b in zip(times,times[1:])]
            if len(intervals)>=2 and statistics.mean(intervals)>0:
                out['S10']=statistics.pstdev(intervals)/statistics.mean(intervals)
            total5=sum(r[1] for r in tr5)
            if total5>0:out['S24']=max(r[1] for r in tr5)/total5
            bq=sum(r[4] for r in tr5 if r[2]);aq=sum(r[4] for r in tr5 if not r[2])
            b_,a_=self._best('bid'),self._best('ask')
            if bq>0 and aq>0 and book_fresh and b_ is not None and a_ is not None:
                mid=(b_+a_)*self.params.tick_size/2
                if mid>0:
                    bv=sum(r[1] for r in tr5 if r[2]);av=sum(r[1] for r in tr5 if not r[2])
                    out['S23']=(bv/bq-av/aq)/mid*1e4
        if self.last_trade is not None:out['S09']=(t-self.last_trade)/1e6
        for key,direction in [('S11',None),('S12',True),('S13',False)]:
            total=0.0
            for ts,value,buy,tid,_qty in reversed(self.trades):
                if direction is None or buy==direction:
                    total+=value
                    if total>=self.params.quota_usdc:
                        out[key]=(t-ts)/1e6;break
        if self.closed and t-self.closed[-1][0]<=self.params.retention_seconds*SECOND:
            out['S15']=self.closed[-1][2]
            signed=[r[1] for r in self.closed]
            streak_sign=(signed[-1]>0)-(signed[-1]<0);streak=0
            for v in reversed(signed):
                if (v>0)-(v<0)!=streak_sign:break
                streak+=1
            out['S25']=float(streak)
            durations=[r[2] for r in self.closed]
            if len(durations)>=2 and statistics.mean(durations)>0:
                out['S26']=statistics.pstdev(durations)/statistics.mean(durations)
            if len(self.closed)==self.params.bucket_count:
                out['S14']=sum(abs(v) for v in signed)/self.params.bucket_count
                half=math.ceil(self.params.bucket_count/2)
                out['S27']=sum(signed[-half:])/half-sum(signed[:-half])/(self.params.bucket_count-half)
        return out
