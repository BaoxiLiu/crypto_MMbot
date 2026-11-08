"""Read-only architecture inventory and strict checks; Python standard library only."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
CODE_ROOTS = ('app', 'src', 'tools')
# Reject whole retired subsystems, including new filenames under them.
RETIRED_TREES = (
    'latency',
    'baselines',
    'experiments',
    'crypto_mmbot',
    'research',
    'tools/inventory',
    'tools/prediction',
    'tools/queue',
    'tools/reports',
    'app/backtesting/research',
    'app/backtesting/configs',
    'app/backtesting/strategies',
    'app/backtesting/engines',
    'app/market_making/fusion/compat',
    'app/market_making/fusion/profiles',
    'app/market_making/research',
    'src/strategy',
)
# Isolated retired entrypoints in otherwise active packages.
RETIRED_RUNNERS = {'app/market_making/fusion/'+name+'.py' for name in (
    'as_strategy', 'nh_strategy', 'hjb_strategy', 'fb_hjb_math', 'strategy_base',
    'strategy_registry', 'adaptive_quote', 'own_queue', 'queue_reactive', 'quote_policy',
)} | {
    'experiment_config.json', 'app/backtesting/compare_runs.py', 'app/backtesting/metrics.py',
    'tools/experiments/compare_backtest_benchmarks.py',
    'app/backtesting/tests/reproduce_partial_accounting.py',
    'app/backtesting/vendor/hftbacktest-2.4.4-makerfix1.patch',
    'app/backtesting/vendor/hftbacktest-2.4.4-makerfix3.patch',
    'src/runner_lib', 'src/audit', 'src/research',
    'app/market_making/fusion/default.json',
    'tools/experiments/analyze_hjb_budget.py',
    'tools/experiments/capture_hjb_quote_edges.py',
    'app/backtesting/adaptive_top.py',
    'app/backtesting/batch.py',
    'app/backtesting/cached_forecast_model.py',
    'app/backtesting/check_features.py',
    'app/backtesting/check_forecast_dataset.py',
    'app/backtesting/cli.py',
    'app/backtesting/confirmation_data.py',
    'app/backtesting/continuity_experiment.py',
    'app/backtesting/controllers.py',
    'app/backtesting/cross_coin_features.py',
    'app/backtesting/direction_prediction.py',
    'app/backtesting/execution_control.py',
    'app/backtesting/export_preparation.py',
    'app/backtesting/feature_cache.py',
    'app/backtesting/feature_reader.py',
    'app/backtesting/first_stage_native.py',
    'app/backtesting/fixed_gamma_experiment.py',
    'app/backtesting/flow_diagnostics.py',
    'app/backtesting/forecast_dataset.py',
    'app/backtesting/forecast_model.py',
    'app/backtesting/funding_clock.py',
    'app/backtesting/gateway_config.py',
    'app/backtesting/inventory_experiment.py',
    'app/backtesting/inventory_headroom.py',
    'app/backtesting/inventory_policy.py',
    'app/backtesting/inventory_repair_experiment.py',
    'app/backtesting/inventory_thresholds.py',
    'app/backtesting/joint_order_risk.py',
    'app/backtesting/order_audit.py',
    'app/backtesting/order_event_state.py',
    'app/backtesting/paired_coverage.py',
    'app/backtesting/paired_loss_blocks.py',
    'app/backtesting/paper_reversal.py',
    'app/backtesting/performance_experiment.py',
    'app/backtesting/placebo_top.py',
    'app/backtesting/post_fill_exit_quote.py',
    'app/backtesting/post_fill_protection.py',
    'app/backtesting/prediction_cache.py',
    'app/backtesting/prediction_config.py',
    'app/backtesting/prediction_diagnostics.py',
    'app/backtesting/prediction_experiment.py',
    'app/backtesting/prediction_filter.py',
    'app/backtesting/prediction_lifetime.py',
    'app/backtesting/prediction_quote.py',
    'app/backtesting/prediction_risk.py',
    'app/backtesting/prediction_side_gate.py',
    'app/backtesting/prediction_spread.py',
    'app/backtesting/prepare_batch.py',
    'app/backtesting/pressure_protection.py',
    'app/backtesting/pressure_quote.py',
    'app/backtesting/project_config.py',
    'app/backtesting/queue_hold.py',
    'app/backtesting/queue_management.py',
    'app/backtesting/quote_experiment.py',
    'app/backtesting/quote_floor.py',
    'app/backtesting/quote_intent.py',
    'app/backtesting/quote_intent_observation.py',
    'app/backtesting/quote_models.py',
    'app/backtesting/quote_observation.py',
    'app/backtesting/quote_placement.py',
    'app/backtesting/quote_risk_dataset.py',
    'app/backtesting/replay_comparison.py',
    'app/backtesting/replay_reads.py',
    'app/backtesting/research_protocol.py',
    'app/backtesting/research_report.py',
    'app/backtesting/single_long.py',
    'app/backtesting/strategy_experiment.py',
    'app/backtesting/terminal_exit.py',
    'app/backtesting/tests/fixtures/gateway_as.json',
    'app/backtesting/tests/market_fixtures.py',
    'app/backtesting/tests/test_adaptive_top.py',
    'app/backtesting/tests/test_cached_forecast_model.py',
    'app/backtesting/tests/test_confirmation_data.py',
    'app/backtesting/tests/test_continuity_experiment.py',
    'app/backtesting/tests/test_cross_coin_features.py',
    'app/backtesting/tests/test_export_preparation.py',
    'app/backtesting/tests/test_feature_cache.py',
    'app/backtesting/tests/test_feature_reader.py',
    'app/backtesting/tests/test_fixed_gamma_experiment.py',
    'app/backtesting/tests/test_flow_diagnostics.py',
    'app/backtesting/tests/test_forecast_dataset.py',
    'app/backtesting/tests/test_forecast_model.py',
    'app/backtesting/tests/test_funding_clock.py',
    'app/backtesting/tests/test_inventory_experiment.py',
    'app/backtesting/tests/test_inventory_headroom.py',
    'app/backtesting/tests/test_inventory_policy.py',
    'app/backtesting/tests/test_inventory_repair_experiment.py',
    'app/backtesting/tests/test_inventory_thresholds.py',
    'app/backtesting/tests/test_joint_order_risk.py',
    'app/backtesting/tests/test_order_event_state.py',
    'app/backtesting/tests/test_paired_coverage.py',
    'app/backtesting/tests/test_paired_loss_blocks.py',
    'app/backtesting/tests/test_placebo_top.py',
    'app/backtesting/tests/test_post_fill_exit_quote.py',
    'app/backtesting/tests/test_post_fill_protection.py',
    'app/backtesting/tests/test_prediction_cache.py',
    'app/backtesting/tests/test_prediction_diagnostics.py',
    'app/backtesting/tests/test_prediction_experiment.py',
    'app/backtesting/tests/test_prediction_filter.py',
    'app/backtesting/tests/test_prediction_lifetime.py',
    'app/backtesting/tests/test_prediction_quote.py',
    'app/backtesting/tests/test_prediction_risk.py',
    'app/backtesting/tests/test_prediction_side_gate.py',
    'app/backtesting/tests/test_prediction_spread.py',
    'app/backtesting/tests/test_pressure_protection.py',
    'app/backtesting/tests/test_pressure_quote.py',
    'app/backtesting/tests/test_project_config.py',
    'app/backtesting/tests/test_queue_hold.py',
    'app/backtesting/tests/test_queue_management.py',
    'app/backtesting/tests/test_quote_experiment.py',
    'app/backtesting/tests/test_quote_floor.py',
    'app/backtesting/tests/test_quote_intent.py',
    'app/backtesting/tests/test_quote_intent_observation.py',
    'app/backtesting/tests/test_quote_models.py',
    'app/backtesting/tests/test_quote_observation.py',
    'app/backtesting/tests/test_quote_placement.py',
    'app/backtesting/tests/test_quote_risk_dataset.py',
    'app/backtesting/tests/test_replay_reads.py',
    'app/backtesting/tests/test_research.py',
    'app/backtesting/tests/test_strategy_experiment.py',
    'app/backtesting/tests/test_terminal_attribution.py',
    'app/backtesting/tests/test_terminal_exit.py',
    'app/backtesting/tests/test_volatility.py',
    'app/backtesting/volatility.py',
    'app/factor_library/source_identity.py',
    'app/factor_library/source_migrations.json',
    'app/market_making/ENGINE_README.md',
    'app/market_making/binance_ws.py',
    'app/market_making/bookkeeper.py',
    'app/market_making/calibrate_replay.py',
    'app/market_making/config.1000shibusdc.fv.json',
    'app/market_making/config.1000shibusdc.json',
    'app/market_making/config.as_event.json',
    'app/market_making/config.as_floor6.json',
    'app/market_making/config.as_inventory.json',
    'app/market_making/config.json',
    'app/market_making/config.matrix.json',
    'app/market_making/config.matrix.sui.json',
    'app/market_making/config.matrix.uni.json',
    'app/market_making/diagnose.py',
    'app/market_making/feed.py',
    'app/market_making/fusion/analyze.py',
    'app/market_making/fusion/backtest.py',
    'app/market_making/fusion/binance.py',
    'app/market_making/fusion/calibrate.py',
    'app/market_making/fusion/journal.py',
    'app/market_making/fusion/profile_calibration.py',
    'app/market_making/fusion/protection_policy.py',
    'app/market_making/fusion/runtime.py',
    'app/market_making/gateway.py',
    'app/market_making/paper_compare.py',
    'app/market_making/replay.py',
    'app/market_making/runner.py',
    'app/market_making/sc1.py',
    'app/market_making/status.py',
    'app/market_making/test_binance_ws.py',
    'app/market_making/test_bookkeeper.py',
    'app/market_making/test_diagnose.py',
    'app/market_making/test_feed.py',
    'app/market_making/test_gateway.py',
    'app/market_making/test_integration_ws.py',
    'app/market_making/test_paper_compare.py',
    'app/market_making/test_replay_bt.py',
    'app/market_making/test_runtime.py',
    'app/market_making/test_sc1.py',
    'config/experiments/strategy_comparison_v1.json',
    'config/market/exchange-info-event-replay.json',
    'config/profiles/data/doge_aug01_10_v1.json',
    'config/profiles/data/inputs/doge_aug01_10.json',
    'config/strategies/account_baseline.json',
    'config/strategies/as_base_v1.json',
    'config/strategies/as_fusion_v1.json',
    'config/strategies/as_markout_v1.json',
    'tools/data/accept_wif_capture.py',
    'tools/data/audit_btc_causal_join.py',
    'tools/data/finish_july_downloads.py',
    'tools/data/prefetch_monthly_funding.py',
    'tools/data/probe_unseen_data_access.py',
    'tools/data/run_cryptohftdata_incremental_20261013.py',
    'tools/data/verify_quarantined_event_dataset.py',
    'tools/experiments/analyze_doge_spread_floor.py',
    'tools/experiments/analyze_fill_path.py',
    'tools/experiments/analyze_making_matrix.py',
    'tools/experiments/analyze_quote_geometry.py',
    'tools/experiments/analyze_quote_placements.py',
    'tools/experiments/analyze_quote_target_time.py',
    'tools/experiments/analyze_vol_markout.py',
    'tools/experiments/audit_quote_observation_artifacts.py',
    'tools/experiments/audit_single_long.py',
    'tools/experiments/audit_spread_replay.py',
    'tools/experiments/benchmark_backtest.py',
    'tools/experiments/calibrate_markout_spread.py',
    'tools/experiments/compare_execution_runtime.py',
    'tools/experiments/diagnose_doge_fill_quality.py',
    'tools/experiments/diagnose_spread_experiment.py',
    'tools/experiments/export_quote_predictors.py',
    'tools/experiments/prepare_defensive_requote.py',
    'tools/experiments/reconcile_metrics_unification.py',
    'tools/experiments/reconcile_quote_risk_lifecycle.py',
    'tools/experiments/register_research_queue_20261004.py',
    'tools/experiments/report_maker_stop.py',
    'tools/experiments/report_order_placements.py',
    'tools/experiments/report_quote_range_study.py',
    'tools/experiments/report_spread_experiment.py',
    'tools/experiments/run_candidate_trade_reach.py',
    'tools/experiments/run_defensive_requote.py',
    'tools/experiments/run_making_baseline.py',
    'tools/experiments/run_making_matrix.py',
    'tools/experiments/run_terminal_pilot.py',
    'tools/experiments/select_lifetime_research_coins.py',
    'tools/experiments/verify_adaptive_run.py',
    'tools/experiments/verify_quote_observation.py',
    'tools/experiments/verify_research_release_fixture.py',
    'tools/factors/build_paper173_schema.py',
    'tools/factors/review_kimi_factor_analysis_20261004.py',
    'tools/ledger/analyze_cost_matrix.py',
    'tools/ledger/analyze_funded_ledger.py',
    'tools/ledger/analyze_maker_rebate.py',
    'tools/ledger/analyze_making_ledger.py',
    'tools/ledger/analyze_profit_cycles.py',
    'tools/ledger/analyze_terminal_ledger.py',
    'tools/ledger/analyze_thirteen_cost.py',
    'tools/ledger/audit_cost_daily_anchors.py',
    'tools/ledger/audit_cost_path_risk_funding.py',
    'tools/ledger/check_profit_cycle_openings.py',
    'tools/ledger/compare_cost_path_predecessor.py',
    'tools/ledger/replay_analysis.py',
    'tools/ledger/run_funding_pilot.py',
    'tools/ledger/run_making_cost_matrix.py',
    'tools/ledger/summarize_making_costs.py',
    'tools/ledger/summarize_profit_cycles.py',
    'tools/ledger/summarize_thirteen_costs.py',
    'tools/ledger/verify_funding_clock.py',
    'tools/ledger/verify_terminal_ledger.py',
    'tools/tests/test_analyze_cost_matrix.py',
    'tools/tests/test_analyze_fill_path.py',
    'tools/tests/test_analyze_full_inventory.py',
    'tools/tests/test_analyze_funded_ledger.py',
    'tools/tests/test_analyze_inventory_state.py',
    'tools/tests/test_analyze_making_ledger.py',
    'tools/tests/test_analyze_making_matrix.py',
    'tools/tests/test_analyze_prediction_window.py',
    'tools/tests/test_analyze_thirteen_cost.py',
    'tools/tests/test_analyze_vol_markout.py',
    'tools/tests/test_audit_btc_causal_join.py',
    'tools/tests/test_audit_cost_path_risk_funding.py',
    'tools/tests/test_compare_cost_path_predecessor.py',
    'tools/tests/test_post_fill_action_methods.py',
    'tools/tests/test_post_fill_inventory_model.py',
    'tools/tests/test_post_fill_inventory_readout_v2.py',
    'tools/tests/test_post_fill_inventory_task.py',
    'tools/tests/test_prediction_common_valuation.py',
    'tools/tests/test_prediction_lifetime_analysis.py',
    'tools/tests/test_prediction_plain_zero.py',
    'tools/tests/test_prediction_side_gate_analysis.py',
    'tools/tests/test_prediction_side_gate_calibration.py',
    'tools/tests/test_prediction_side_gate_independent_review.py',
    'tools/tests/test_prediction_side_gate_readout_scope.py',
    'tools/tests/test_prediction_side_gate_sampling.py',
    'tools/tests/test_prediction_transmission.py',
    'tools/tests/test_prepare_defensive_requote.py',
    'tools/tests/test_probe_unseen_data_access.py',
    'tools/tests/test_profit_cycles.py',
    'tools/tests/test_quote_intent_methods.py',
    'tools/tests/test_reducing_quote_feasibility.py',
    'tools/tests/test_replay_analysis.py',
    'tools/tests/test_signal_value_methods.py',
    'tools/tests/test_signal_value_timing.py',
    'tools/tests/test_summarize_full_inventory.py',
    'tools/tests/test_summarize_making_costs.py',
    'tools/tests/test_summarize_thirteen_costs.py',
    'tools/tests/test_three_class_research.py',
    'tools/tests/test_verify_inventory_headroom_fixture.py',
    'tools/tests/test_verify_inventory_headroom_readout_identity.py',
}
ARTIFACT_SUFFIXES = {'.npy', '.npz', '.parquet', '.png', '.html', '.tgz', '.zip', '.log', '.pyc'}


def git_paths(root):
    root = Path(root)
    if not (root / '.git').exists():
        # A cleaned source tree can be checked before initializing its new repository.
        return sorted(p.relative_to(root).as_posix() for p in root.rglob('*')
                      if p.is_file() and not p.is_symlink()
                      and not any(x.startswith('._') for x in p.relative_to(root).parts))
    raw = subprocess.check_output(['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'], cwd=root)
    return sorted(set(p for p in raw.decode().split('\0') if p and (root / p).exists() and not any(x.startswith('._') for x in Path(p).parts)))


def physical_code_paths(root):
    """Inspect ignored source and local data files, without following links."""
    root = Path(root)
    paths = []
    for code_root in (*CODE_ROOTS, "data"):
        for folder, dirs, files in os.walk(root / code_root, followlinks=False):
            parent = Path(folder)
            dirs[:] = [name for name in dirs if not name.startswith('._')
                       and not (parent / name).is_symlink()]
            for name in files:
                path = parent / name
                if not name.startswith('._') and not path.is_symlink():
                    paths.append(path.relative_to(root).as_posix())
    return sorted(paths)


def violations(paths):
    """Code directories contain source; tools are grouped by responsibility."""
    issues = set()
    for name in paths:
        p = Path(name)
        if any(name == prefix or name.startswith(prefix + '/') for prefix in RETIRED_TREES):
            issues.add('retired-tree:' + name)
        if name.startswith(('data/official_binance/', 'data/provider_comparison_', 'data/raw/')):
            issues.add('raw-data-in-project:' + name)
        if name.startswith(('app/data/', 'app/logs/', 'app/deploy/')):
            issues.add('runtime-or-deploy-in-code:' + name)
        if name == 'app/market_making/RESEARCH_LOG.md':
            issues.add('history-in-code:' + name)
        if name in RETIRED_RUNNERS:
            issues.add('retired-runner:'+name)
        if p.parts[0] in CODE_ROOTS and p.suffix in ARTIFACT_SUFFIXES:
            issues.add('artifact-in-code:' + name)
        if p.parts[0] in CODE_ROOTS and p.suffix.lower() == '.md':
            issues.add('documentation-in-code:' + name)
        if p.parts[0] in CODE_ROOTS and any(part.lower() in ('legacy','old','backup','archive','snapshots','source_snapshot') for part in p.parts):
            issues.add('backup-in-code:' + name)
        if p.parts[0] == 'tools' and len(p.parts) == 2 and p.suffix == '.py':
            issues.add('flat-tool:' + name)
        if p.parts[0] == 'app' and len(p.parts) > 2 and p.parts[1] == 'backtesting' and any(x in p.parts for x in ('experiments', 'baselines')):
            issues.add('history-in-code:' + name)
    return sorted(issues)


def physical_inventory(root):
    """Count what is on disk, including ignored files; never follow symlinks."""
    root = Path(root)
    groups = {}
    errors = []
    def row(path):
        key = path.relative_to(root).parts[0]
        return groups.setdefault(key, dict(files=0, bytes=0, symlinks=0, cache_files=0))
    for folder, dirs, files in os.walk(root, followlinks=False, onerror=lambda e: errors.append(str(e))):
        parent = Path(folder)
        for name in list(dirs):
            path = parent / name
            if name == '.git':
                dirs.remove(name)
            elif path.is_symlink():
                row(path)['symlinks'] += 1
                dirs.remove(name)
        for name in files:
            path = parent / name
            values = row(path)
            if path.is_symlink():
                values['symlinks'] += 1
                continue
            try:
                size = path.stat().st_size
            except OSError as error:
                errors.append(str(error))
                continue
            values['files'] += 1
            values['bytes'] += size
            if '__pycache__' in path.parts or '.pytest_cache' in path.parts:
                values['cache_files'] += 1
    return dict(scope='Physical files including ignored outputs; excludes .git and symlink targets',
                files=sum(r['files'] for r in groups.values()),
                bytes=sum(r['bytes'] for r in groups.values()),
                directories=dict(sorted(groups.items())), errors=errors)


def inventory(root):
    paths = git_paths(root)
    counts = Counter(Path(p).parts[0] for p in paths)
    sizes = Counter()
    for name in paths:
        p = root / name
        if p.is_file() and not p.is_symlink():
            sizes[Path(name).parts[0]] += p.stat().st_size
    status = subprocess.check_output(['git', 'status', '--porcelain=v1', '-z', '--untracked-files=normal'], cwd=root)
    return dict(schema_version=2, physical=physical_inventory(root), scope='Git tracked and nonignored files only; excludes raw data and ignored reports',
                files=len(paths), counts=dict(sorted(counts.items())), bytes=dict(sorted(sizes.items())),
                dirty_status_sha256=hashlib.sha256(status).hexdigest(),
                violations=violations(paths))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['inventory', 'check'])
    args = parser.parse_args()
    if args.action == 'inventory':
        print(json.dumps(inventory(ROOT), ensure_ascii=False, indent=2))
        return
    current = {'violations': violations(set(git_paths(ROOT)) | set(physical_code_paths(ROOT)))}
    if __package__:
        from .dependency_rules import check_boundaries
    else:
        from dependency_rules import check_boundaries
    issues = current['violations'] + check_boundaries(ROOT)
    print(json.dumps(dict(violations=issues, status='failed' if issues else 'passed'), ensure_ascii=False, indent=2))
    raise SystemExit(bool(issues))


if __name__ == '__main__':
    main()
