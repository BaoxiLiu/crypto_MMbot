"""Single project CLI. Domain implementations retain their own option parsers."""
import argparse
import runpy
import os
import site
import subprocess
import sys

COMMANDS = {
    'live': ('app.market_making.live', '接入原型的离线replay；联网模式需明确指定'),
    'backtest': ('app.backtesting.fusion_experiment', '配置校验、回测和报告'),
    'check': ('tools.maintenance.check_project', '开发验收；--native 包含本机HFT回归'),
    'factors': ('tools.factors.build_factor_catalog', '生成因子目录；--check 检查一致性'),
}


def main(argv=None):
    sys.dont_write_bytecode = True
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    args=list(sys.argv[1:] if argv is None else argv)
    if site.ENABLE_USER_SITE:
        env=dict(os.environ,PYTHONNOUSERSITE='1')
        return subprocess.call([sys.executable,'-B','-s','-m','app',*args],env=env)
    os.environ['PYTHONNOUSERSITE']='1'
    parser=argparse.ArgumentParser(description='Crypto MMbot · 本机做市研究')
    parser.add_argument('command',choices=COMMANDS,help='；'.join(k+': '+v[1] for k,v in COMMANDS.items()))
    if not args or args[0] in ('-h','--help'):
        parser.print_help()
        return
    command=parser.parse_args(args[:1]).command
    module=COMMANDS[command][0]
    original=sys.argv
    try:
        sys.argv=[module,*args[1:]]
        runpy.run_module(module,run_name='__main__')
    finally:
        sys.argv=original


if __name__=='__main__':raise SystemExit(main())
