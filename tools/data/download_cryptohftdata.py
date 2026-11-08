"""Single-worker, throttled hourly downloader. No credentials or IP rotation."""
import argparse, collections, datetime as dt, fcntl, hashlib, json, os, re
import shutil, time, urllib.error, urllib.request
from email.utils import parsedate_to_datetime
from pathlib import Path

SYMBOLS = ['BTCUSDC', 'SOLUSDC', 'SUIUSDC', 'UNIUSDC', 'DOGEUSDC', '1000SHIBUSDC']
KINDS = ['orderbook', 'trades', 'mark_price']
SUPPORTED_SYMBOLS = (*SYMBOLS, 'WIFUSDC', 'ETHUSDC', '1000PEPEUSDC', 'LTCUSDC',
                     'CRVUSDC', 'XRPUSDC', '1000BONKUSDC')

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def inspect(path, symbol, kind):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    required = {'symbol', 'received_time', 'event_time'} | {
        'orderbook': {'event_type', 'side', 'price', 'quantity', 'final_update_id'},
        'trades': {'trade_id', 'price', 'quantity', 'trade_time', 'is_buyer_maker'},
        'mark_price': {'mark_price', 'index_price', 'funding_rate', 'next_funding_time'}
    }[kind]
    if required - set(pf.schema_arrow.names): raise ValueError('Missing required columns')
    # Read all columns in bounded batches; validates decoding without whole-hour RAM loading.
    for batch in pf.iter_batches(batch_size=32768):
        values = set(batch.column(batch.schema.get_field_index('symbol')).to_pylist())
        if values and values != {symbol}: raise ValueError('Unexpected symbol')
    return {'rows': pf.metadata.num_rows, 'bytes': path.stat().st_size,
            'sha256': sha(path), 'schema': str(pf.schema_arrow), 'empty': pf.metadata.num_rows == 0}

def seconds(value):
    try: return max(0.0, float(value))
    except (ValueError, TypeError):
        try: return max(0.0, parsedate_to_datetime(value).timestamp() - time.time())
        except (ValueError, TypeError): return 60.0

class Throttle:
    def __init__(self): self.last = 0; self.blocked_until = 0; self.hits = collections.deque(); self.interval = 2.1
    def wait(self):
        now = time.monotonic()
        while self.hits and now-self.hits[0] >= 61: self.hits.popleft()
        due = max(self.last+self.interval, self.blocked_until)
        if len(self.hits) >= 30: due = max(due, self.hits[0]+61)
        time.sleep(max(0, due-now)); self.last=time.monotonic(); self.hits.append(self.last)
    def headers(self, headers):
        selected = {key: headers[key] for key in ['RateLimit','RateLimit-Policy','RateLimit-Limit',
                   'RateLimit-Remaining','RateLimit-Reset','Retry-After'] if key in headers}
        # Support both separate header fields and standard structured forms (r=, t=).
        remaining = headers.get('RateLimit-Remaining')
        reset = headers.get('RateLimit-Reset')
        modern = headers.get('RateLimit', '')
        if remaining is None:
            m=re.search(r'(?:^|[;, ])r=(\d+)',modern); remaining=m.group(1) if m else None
        if reset is None:
            m=re.search(r'(?:^|[;, ])t=(\d+)',modern); reset=m.group(1) if m else None
        policy = headers.get('RateLimit-Policy', '')
        m=re.search(r'q=(\d+).*?w=(\d+)',policy)
        try:
            limit=float(headers.get('RateLimit-Limit',m.group(1) if m else 60))
            window=float(m.group(2)) if m else 60
            if limit > 0: self.interval=max(self.interval,2*window/limit + .1)
            if remaining is not None and float(remaining) <= 5:
                delay=float(reset) if reset is not None else 60
                if delay > 1_000_000_000: delay=max(0,delay-time.time())
                self.blocked_until=max(self.blocked_until,time.monotonic()+delay+2)
        except ValueError: self.blocked_until=max(self.blocked_until,time.monotonic()+60)
        return selected

def parse_args(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--start',required=True);p.add_argument('--end',required=True)
    p.add_argument('--root',required=True);p.add_argument('--sample',action='store_true',help='Latest complete day 23 UTC, all six symbols and three types')
    p.add_argument('--symbols',nargs='+',default=list(SYMBOLS),help='Symbols, separated by spaces or commas')
    p.add_argument('--name',help='Name of the status report')
    p.add_argument('--max-files',type=int,help='Bounded smoke run; not completion of the entire requested range')
    p.add_argument('--attempts',type=int,default=5)
    p.add_argument('--fail-fast',action='store_true')
    p.add_argument('--max-bytes',type=int,help='Retained raw payload bytes including partial files; filesystem overhead is separate')
    a=p.parse_args(argv);a.symbols=[s for group in a.symbols for s in group.split(',')]
    start=dt.date.fromisoformat(a.start);end=dt.date.fromisoformat(a.end)
    if end<=start:p.error('end must be after start')
    if end>dt.datetime.now(dt.timezone.utc).date():p.error('Only complete UTC dates may be downloaded')
    if len(a.symbols)!=len(set(a.symbols)):p.error('Duplicate symbols')
    if not a.symbols or set(a.symbols)-set(SUPPORTED_SYMBOLS):p.error('Unsupported symbols')
    if a.name and not re.fullmatch(r'[A-Za-z0-9_-]+',a.name):p.error('Invalid report name')
    if a.max_files is not None and a.max_files<=0:p.error('max-files must be positive')
    if not 1<=a.attempts<=5:p.error('attempts must be 1 through 5')
    if a.max_bytes is not None and a.max_bytes<=0:p.error('max-bytes must be positive')
    return a


def copy_bounded(response,out,budget):
    written=0
    if budget is not None and budget<0:raise RuntimeError('Registered download byte budget exceeded')
    while block:=response.read(1024*1024):
        if budget is not None and written+len(block)>budget:raise RuntimeError('Registered download byte budget exceeded')
        out.write(block);written+=len(block)
    return written


def normalize_partial(part,symbol,kind,budget=None):
    """Recognise transport compression, decode on disk, then validate all batches.

    Older hourly files may have an outer Zstandard frame despite their .parquet
    suffix. Preserve the compressed partial on failed decoding, and record its
    identity before replacing it with the verified, reusable Parquet payload.
    """
    with part.open('rb') as stream: magic=stream.read(4)
    if magic==b'PAR1':
        return dict(inspect(part,symbol,kind),transport_format='parquet')
    if magic!=b'\x28\xb5\x2f\xfd':raise ValueError('Unsupported transport payload; expected Parquet or Zstandard')
    import pyarrow as pa
    encoded_bytes=part.stat().st_size;encoded_sha256=sha(part)
    decoded=part.with_name(part.name+'.decoded')
    try:
        # Encoded + decoded files coexist briefly. Leave at least 10GB free and
        # respect the same retained-byte budget as the network transfer.
        free_budget=max(0,shutil.disk_usage(part.parent).free-10_000_000_000)
        limit=free_budget if budget is None else min(free_budget,budget)
        with pa.input_stream(part,compression='zstd') as source,decoded.open('wb') as out:
            copy_bounded(source,out,limit)
        checked=inspect(decoded,symbol,kind)
        decoded.replace(part)
        return dict(checked,transport_format='zstd_wrapped_parquet',
                    transport_bytes=encoded_bytes,transport_sha256=encoded_sha256)
    finally:decoded.unlink(missing_ok=True)


def main(argv=None):
    a=parse_args(argv);start=dt.date.fromisoformat(a.start);end=dt.date.fromisoformat(a.end)
    root=Path(a.root);root.mkdir(parents=True,exist_ok=True)
    # A single writer owns each download root.
    lock=root/'download.lock'
    try: fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError: raise SystemExit('Another job/lock exists; inspect process before removing stale lock')
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    name=a.name or (('six-coins-sample' if a.sample else 'six-coins-14d') if a.symbols==SYMBOLS else ('selected-coins-sample' if a.sample else 'selected-coins-download'))
    manifest=root/'manifests/download.jsonl';manifest.parent.mkdir(exist_ok=True)
    report=root/f'reports/{name}-status.json';report.parent.mkdir(exist_ok=True)
    prior={}
    if manifest.exists():
        with manifest.open() as stream:
            for line in stream:
                try: entry=json.loads(line);prior[entry['file']]=entry
                except (ValueError,KeyError): pass
    counts={'verified':0,'recovered':0,'skipped':0,'missing':0,'failed':0,'requests':0,'rate_limits':0}
    retained=sum(f.stat().st_size for f in (root/'raw').rglob('*') if f.is_file())
    target=len(a.symbols)*len(KINDS) if a.sample else (end-start).days*24*len(a.symbols)*len(KINDS)
    current='';throttle=Throttle();rate_streak=0;started=dt.datetime.now(dt.timezone.utc).isoformat()
    def state(status,reason=None):
        d=dict(counts,state=status,current=current,pid=os.getpid(),target=target,start=a.start,end_exclusive=a.end,
               symbols=a.symbols,types=KINDS,started_utc=started,updated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
               request_interval_seconds=throttle.interval,attempt_limit=a.attempts,fail_fast=a.fail_fast,
               retained_file_bytes=retained,max_bytes=a.max_bytes,max_files=a.max_files,
               downloader_sha256=sha(Path(__file__)),reason=reason)
        tmp=report.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2));tmp.replace(report)
    jobs=((end-dt.timedelta(days=1),23,s,k) for s in a.symbols for k in KINDS) if a.sample else (
        (start+dt.timedelta(days=i),hour,s,k) for i in range((end-start).days) for hour in range(24) for s in a.symbols for k in KINDS)
    try:
        for job_index,(day,hour,symbol,kind) in enumerate(jobs):
            if a.max_files is not None and job_index>=a.max_files:
                state('complete_requested_limit');return
            current=f'binance_futures/{day}/{hour:02d}/{symbol}_{kind}.parquet';state('running')
            path=root/'raw'/current;path.parent.mkdir(parents=True,exist_ok=True)
            entry={'file':current,'downloaded_utc':dt.datetime.now(dt.timezone.utc).isoformat()}
            fatal=None
            try:
                if shutil.disk_usage(root).free<10_000_000_000: raise RuntimeError('Free disk space below 10GB')
                old=prior.get(current,{})
                if path.exists() and old.get('status')=='verified' and sha(path)==old['sha256']:
                    counts['skipped']+=1;state('running');continue
                part=path.with_suffix('.parquet.part')
                recovered=False
                if part.exists():
                    previous_size=part.stat().st_size
                    try:
                        entry.update(normalize_partial(part,symbol,kind,None if a.max_bytes is None else a.max_bytes-retained))
                        recovered=True
                    except RuntimeError:raise
                    except Exception as exc:
                        entry['partial_recovery_error']=type(exc).__name__
                    finally:retained+=part.stat().st_size-previous_size
                for attempt in range(a.attempts):
                    if recovered:break
                    throttle.wait();counts['requests']+=1;state('running')
                    try:
                        request=urllib.request.Request('https://api.cryptohftdata.com/v1/download?file='+current,
                                                      headers={'User-Agent':'quant-research-hourly-downloader/1.0'})
                        prior_part=part.stat().st_size if part.exists() else 0
                        with urllib.request.urlopen(request,timeout=90) as response,part.open('wb') as out:
                            entry['rate_headers']=throttle.headers(response.headers)
                            # The live registered run has one attempt and a new
                            # root. Account all actual retained bytes, including
                            # failed partial files, rather than just successes.
                            retained-=prior_part
                            try:copy_bounded(response,out,None if a.max_bytes is None else a.max_bytes-retained)
                            finally:
                                out.flush();retained+=part.stat().st_size
                        rate_streak=0;break
                    except urllib.error.HTTPError as e:
                        entry['rate_headers']=throttle.headers(e.headers)
                        if e.code==429:
                            counts['rate_limits']+=1;rate_streak+=1
                            if a.fail_fast:raise RuntimeError('HTTP429; stopped without retry') from None
                            wait=max(seconds(e.headers.get('Retry-After',60)),60*2**(rate_streak-1))
                            throttle.blocked_until=max(throttle.blocked_until,time.monotonic()+wait+2)
                            state('rate_limit_wait',f'HTTP429; wait {wait+2}s')
                            if rate_streak>=3: raise RuntimeError('Repeated HTTP429; paused without bypassing limit')
                        elif e.code in (401,403,418): raise RuntimeError(f'HTTP{e.code}; stopped for access review') from None
                        elif e.code not in (500,502,503,504): raise
                        else: throttle.blocked_until=max(throttle.blocked_until,time.monotonic()+min(120,10*2**attempt))
                        if attempt==a.attempts-1: raise
                    except (OSError,TimeoutError):
                        if attempt==a.attempts-1: raise
                        throttle.blocked_until=max(throttle.blocked_until,time.monotonic()+10*2**attempt)
                if not recovered:
                    previous_size=part.stat().st_size
                    try:entry.update(normalize_partial(part,symbol,kind,None if a.max_bytes is None else a.max_bytes-retained))
                    finally:retained+=part.stat().st_size-previous_size
                if path.exists():
                    old_hash=sha(path)
                    if old_hash!=entry['sha256']:path.rename(path.with_name(path.name+'.'+old_hash[:12]+'.previous'))
                part.replace(path);entry['status']='verified';counts['verified']+=1
                if recovered:counts['recovered']+=1;entry['reused_existing_partial']=True
            except urllib.error.HTTPError as e:
                entry.update(status='missing' if e.code==404 else 'failed',http_status=e.code)
                counts['missing' if e.code==404 else 'failed']+=1
            except Exception as e:
                entry.update(status='failed',error=type(e).__name__+': '+str(e)[:200]);counts['failed']+=1
                if isinstance(e,RuntimeError):fatal=str(e)
            if a.fail_fast and entry['status']!='verified':fatal=fatal or 'Incomplete source file; stopped without further requests'
            with manifest.open('a') as out:out.write(json.dumps(entry,ensure_ascii=False)+'\n')
            print(json.dumps({'file':current,'status':entry['status'],'bytes':entry.get('bytes'),'counts':counts}),flush=True)
            state('running')
            if fatal:raise RuntimeError(fatal)
            if a.sample and entry['status']!='verified':raise RuntimeError('Sample verification failed; full download not started')
        state('complete_with_gaps' if counts['missing'] or counts['failed'] else 'complete')
    except BaseException as e:
        state('stopped',str(e)[:200]);raise
    finally:lock.unlink(missing_ok=True)

if __name__=='__main__':main()
