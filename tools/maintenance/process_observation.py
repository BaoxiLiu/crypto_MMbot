"""Observe owned children without mistaking an exited process for missing RSS."""
from pathlib import Path
import subprocess
import sys


def observe(process,proc_root=Path('/proc')):
    code=process.poll()
    if code is not None: return dict(exit_code=code,rss_bytes=None)
    if sys.platform=='darwin' and proc_root==Path('/proc'):
        result=subprocess.run(['ps','-o','rss=','-p',str(process.pid)],
                              capture_output=True,text=True,timeout=5)
        code=process.poll()
        if code is not None:return dict(exit_code=code,rss_bytes=None)
        value=result.stdout.strip()
        if result.returncode==0 and value.isdigit():
            return dict(exit_code=None,rss_bytes=int(value)*1024)
        raise RuntimeError('Owned child still live but macOS RSS observation failed')
    try: lines=(proc_root/str(process.pid)/'status').read_text().splitlines()
    except FileNotFoundError:
        code=process.poll()
        if code is not None: return dict(exit_code=code,rss_bytes=None)
        raise RuntimeError('Owned child still reports live but /proc status disappeared')
    rss=next((int(x.split()[1])*1024 for x in lines if x.startswith('VmRSS:')),None)
    if rss is not None: return dict(exit_code=None,rss_bytes=rss)
    # Exit can occur between poll and status read. Zombies no longer own RSS.
    code=process.poll()
    if code is None and any(x.startswith('State:') and x.split()[1] in ('Z','X','x') for x in lines):
        try: code=process.wait(timeout=.2)
        except subprocess.TimeoutExpired: pass
    if code is not None: return dict(exit_code=code,rss_bytes=None)
    raise RuntimeError('Owned live child has no VmRSS; resource evidence unknown')
