"""One entry point: offline replay, public shadow, explicit live adapter."""
import argparse
import asyncio
from pathlib import Path
from .replay import encode, run_file

BASE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('replay', help='Replay explicit market/private events; no matching')
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--rules', type=Path, required=True)
    p.add_argument('--protections', choices=['as'], default='as')
    p.add_argument('--strategy', type=Path, required=True)
    p.add_argument('--symbol', type=str.upper, required=True)
    p.add_argument('--quote', type=str.upper, required=True)
    for mode in ('shadow', 'live'):
        p = commands.add_parser(mode, help='Public intentions only' if mode == 'shadow'
                               else 'Real trading adapter; must be explicitly selected')
        p.add_argument('--symbol', type=str.upper, default='DOGEUSDC')
        p.add_argument('--market-source', choices=['bbo','depth'], default='bbo')
        p.add_argument('--depth-speed', choices=['100ms','0ms'], default='100ms')
        p.add_argument('--config', type=Path, default=BASE / 'default.json')
        p.add_argument('--journal', type=Path, required=True)
        p.add_argument('--seconds', type=int, default=60)
        p.add_argument('--protections', choices=['as'], default='as')
        p.add_argument('--strategy', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'replay':
        print(encode(run_file(args.input, args.output, args.config, args.rules, args.protections,
                              strategy=args.strategy, symbol=args.symbol, quote=args.quote)))
    else:
        if args.seconds < 0:
            parser.error('seconds must be nonnegative; 0 runs until stopped')
        args.mode = args.command
        args.mask = 0
        from .runtime import main_async
        asyncio.run(main_async(args))


if __name__ == '__main__':
    main()
