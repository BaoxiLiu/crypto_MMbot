"""One development gate; native replay certification must be requested explicitly."""
import argparse
import importlib.util
import os
import site
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[2]


def run_checks():
    if site.ENABLE_USER_SITE:
        env=dict(os.environ,PYTHONNOUSERSITE='1')
        raise SystemExit(subprocess.call([sys.executable,'-s','-m',
                         'tools.maintenance.check_project',*sys.argv[1:]],cwd=ROOT,env=env))
    os.environ['PYTHONNOUSERSITE']='1'
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native',action='store_true',help='Also require and run local HFT frozen replay tests')
    parser.add_argument('--tools',action='store_true',help='Run all tool tests in the same interpreter')
    args=parser.parse_args()
    if args.native and any(importlib.util.find_spec(name) is None for name in ('scipy','hftbacktest','aiohttp')):
        parser.error('--native requires the verified local HFT interpreter and aiohttp; no skipped native acceptance')
    commands=[['tools/maintenance/project_audit.py','check'],['-m','tools.factors.build_factor_catalog','--check'],
              ['-m','pytest','tools/maintenance/tests','src/metrics/tests','tools/tests/test_factor_catalog.py',
               'tools/tests/test_paper173_library.py','app/backtesting/tests/test_research_policy.py',
               'app/backtesting/tests/test_experiment_config.py',
               'app/backtesting/tests/test_dataset_registry.py',
               'app/backtesting/tests/test_campaign.py',
               'app/backtesting/tests/test_as_math.py',
               'app/backtesting/tests/test_html_retention.py','app/backtesting/tests/test_run_registry.py','-q']]
    if args.native:
        commands.append(['-m','pytest','app/market_making/fusion/tests','app/market_making/live/tests',
                         'app/market_data/test_collector.py','app/market_data/test_integration.py','-q'])
        commands.append(['-m','pytest','app/backtesting/tests',
                         *[str(p.relative_to(ROOT)) for p in sorted((ROOT/'app/market_making').glob('test_*.py'))],
                         '-q'])
    if args.tools:
        commands.append(['-m','pytest','tools/tests','-q','-rs'])
    for command in commands:
        print('CHECK: '+ ' '.join(command),flush=True)
        result=subprocess.run([sys.executable,'-B','-s',*command],cwd=ROOT)
        if result.returncode:raise SystemExit(result.returncode)
    print('Development gates passed. Native replay: '+('passed' if args.native else 'NOT RUN; use --native in verified runtime'))


def main():
    # Verification may compile native helpers, but must not litter source trees.
    sys.dont_write_bytecode = True
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['PYTEST_ADDOPTS'] = os.environ.get('PYTEST_ADDOPTS', '') + ' -p no:cacheprovider'
    with tempfile.TemporaryDirectory(prefix='crypto-mm-check-') as cache:
        os.environ['NUMBA_CACHE_DIR'] = cache
        run_checks()


if __name__=='__main__':main()
