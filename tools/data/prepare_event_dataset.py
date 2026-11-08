"""Prepare a publication-clock cache from an explicit, archived JSON config."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    from app.backtesting.datasets import prepare, atomic_json
    from app.backtesting.experiment_config import resolve
    from app.backtesting.providers.mark_price import prepare_marks
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--input-spec', required=True, type=Path)
    args = p.parse_args()
    if args.input_spec.exists():
        raise FileExistsError('Never overwrite a frozen input specification')
    config = json.loads(args.config.read_text())
    # Resolve paths identically on either invocation directory.
    for key in ('source', 'cache', 'official_trades_root'):
        config[key] = str(resolve(config[key]).resolve())
    dataset, manifest = prepare(config)
    if not (dataset / 'project_inputs/marks.npy').exists():
        prepare_marks(config, dataset / 'project_inputs')
    args.input_spec.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.input_spec, dict(dataset=str(dataset), config=config,
        rows=sum(s['events'] for s in manifest['segments']), segments=len(manifest['segments']),
        exchange_only_trade_rows=sum(r['exchange_only_trade_rows'] for r in manifest['hours'])))
    print(dataset, flush=True)


if __name__ == '__main__':
    main()
