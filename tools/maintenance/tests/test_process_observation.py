"""Process liveness and RSS checks, independent of retired experiment workflows."""
import subprocess
import sys
import pytest
from tools.maintenance.process_observation import observe


class Child:
    pid=123
    def __init__(self,codes):self.codes=iter(codes)
    def poll(self):return next(self.codes)
    def wait(self,timeout):return 0


def status(tmp_path,content):
    p=tmp_path/'123';p.mkdir();(p/'status').write_text(content)


def test_exit_between_poll_and_status_read_is_not_missing_memory_failure(tmp_path):
    status(tmp_path,'State:\tZ (zombie)\n')
    assert observe(Child([None,0]),tmp_path)==dict(exit_code=0,rss_bytes=None)
    assert observe(Child([None,None]),tmp_path)==dict(exit_code=0,rss_bytes=None)


def test_live_rss_checked_and_missing_live_rss_remains_fail_closed(tmp_path):
    status(tmp_path,'State:\tR (running)\nVmRSS:\t1234 kB\n')
    assert observe(Child([None]),tmp_path)['rss_bytes']==1234*1024
    (tmp_path/'123/status').write_text('State:\tR (running)\n')
    with pytest.raises(RuntimeError):observe(Child([None,None]),tmp_path)


def test_disappeared_proc_can_only_be_accepted_after_observed_exit(tmp_path):
    assert observe(Child([None,7]),tmp_path)['exit_code']==7
    with pytest.raises(RuntimeError):observe(Child([None,None]),tmp_path)


def test_real_owned_short_child_exit_and_live_observation():
    p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(.05)'])
    try:
        value=observe(p)
        assert value['exit_code'] is None or value['exit_code']==0
        if value['exit_code'] is None:assert value['rss_bytes']>=0
        p.wait(timeout=3);assert observe(p)['exit_code']==0
    finally:
        if p.poll() is None:p.kill();p.wait(timeout=3)


def test_macos_observation_failure_is_not_misreported_as_exit(monkeypatch):
    from tools.maintenance import process_observation as module
    monkeypatch.setattr(module.sys,'platform','darwin')
    monkeypatch.setattr(module.subprocess,'run',lambda *a,**k:subprocess.CompletedProcess(a,1,'',''))
    with pytest.raises(RuntimeError,match='still live'):
        observe(Child([None,None]))
    assert observe(Child([None,7]))==dict(exit_code=7,rss_bytes=None)


def test_macos_rss_uses_bytes_and_preserves_running_state(monkeypatch):
    from tools.maintenance import process_observation as module
    monkeypatch.setattr(module.sys,'platform','darwin')
    monkeypatch.setattr(module.subprocess,'run',lambda *a,**k:subprocess.CompletedProcess(a,0,' 1234\n',''))
    assert observe(Child([None,None]))==dict(exit_code=None,rss_bytes=1234*1024)

