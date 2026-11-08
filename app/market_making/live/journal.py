"""Durable write-ahead intent/state journal, single process ownership."""
import hashlib,json,sqlite3,fcntl
from pathlib import Path


class Journal:
    def __init__(self,path,identity):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        self.lock=path.with_suffix(path.suffix+'.lock').open('a+')
        try:fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:self.lock.close();raise RuntimeError('journal already in use') from None
        self.db=sqlite3.connect(str(path))
        self.db.execute('PRAGMA journal_mode=WAL');self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY, identity TEXT, payload TEXT)')
        self.db.execute('CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY, kind TEXT, ns INTEGER, payload TEXT)')
        self.identity=json.dumps(identity,sort_keys=True,default=str)
        row=self.db.execute('SELECT identity FROM state WHERE id=1').fetchone()
        if row and row[0]!=self.identity:self.close();raise ValueError('journal config/symbol/mode/rules mismatch')

    def record(self,kind,ns,**data):
        with self.db:self.db.execute('INSERT INTO events (kind,ns,payload) VALUES (?,?,?)',
                                    (kind,ns,json.dumps(data,default=str,allow_nan=False)))

    def load(self):
        row=self.db.execute('SELECT payload FROM state WHERE id=1').fetchone()
        return json.loads(row[0]) if row else None

    def save(self,kernel,events=None):
        payload=dict(oms=kernel.oms.dump(),reference_qty=kernel.strategy.reference_qty,
                     funding_ids=sorted(events.funding_ids) if events else [])
        with self.db:self.db.execute('INSERT OR REPLACE INTO state VALUES (1,?,?)',
                                    (self.identity,json.dumps(payload,default=str,allow_nan=False)))

    def close(self):
        self.db.close();fcntl.flock(self.lock,fcntl.LOCK_UN);self.lock.close()
