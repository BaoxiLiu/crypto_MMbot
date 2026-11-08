"""Explicit new-batch paper profile; old matrix files remain reproducible."""
from copy import deepcopy


def make_s1_config(base):
    cfg = deepcopy(base)
    cfg.pop("as_candidate", None)
    cfg.pop("_matrix_note", None)
    cfg.update({
        "strategy_version": "s1-sc-v2-paper-20260928",
        "primary_account": "D_s1_sc",
        "experiment_status": "forward_paper_exploration_unvalidated_k",
        "maker_fee": "0", "taker_fee": "0",
        "type": "as", "fair_value": "micro", "fv_source": "ticker",
        "keeper_enabled": True, "fv_buffer_ticks": "1",
        "fast_trigger": True, "fast_path": True, "event_driven_quotes": True,
        "order_latency_ms": 10, "cancel_latency_ms": 10,
        "adverse_requote_ticks": "4", "as_floor_cap_ticks": 24,
        "require_s1_ready": True, "warmup_seconds": 1800,
        "s1_window_seconds": 1800, "s1_ticker_stale_ms": 5000,
        "s1_max_missing_ratio": 0.2, "s1_degrade_after_ms": 600000,
        "s1_k": "2.0", "sc_signal": "median_bp", "sc_epsilon_bp": "0.25",
        "sc_window_fills": 50, "sc_min_new_fills": 10,
        "sc_adjust_interval_min": 5, "sc_window_max_age_min": 480,
        "sc_restore_stale_min": 720, "sc_min_ticks": 3, "sc_max_ticks": 24,
        "sc_up_factor": "1.5", "sc_down_factor": "1.25", "sc_max_jump_ticks": 4,
        "accounts": [
            {"name": "A_static", "s1_enabled": False, "spread_control_enabled": False},
            {"name": "B_sc", "s1_enabled": False, "spread_control_enabled": True},
            {"name": "C_s1", "s1_enabled": True, "spread_control_enabled": False},
            {"name": "D_s1_sc", "s1_enabled": True, "spread_control_enabled": True},
        ],
    })
    return cfg


def make_kgrid_config(base):
    """k-grid exploration batch: isolate the S1 effect across k values.

    Six virtual arms share one market path: A_static baseline (no S1, no SC)
    plus five pure-S1 arms with k = 1.0/1.5/2.0/2.5/3.0 (SC disabled on all
    grid arms — the SC interaction is studied separately). Same seeds of
    global parameters as the s1 profile; only per-account s1_k overrides.
    Data from this batch is exploration-grade (select-k input), never
    confirmation evidence.
    """
    cfg = make_s1_config(base)
    cfg.update({
        "strategy_version": "s1-kgrid-paper-20261001",
        "primary_account": "K_200",
        "experiment_status": "forward_paper_exploration_kgrid_unvalidated",
        "accounts": [
            {"name": "A_static", "s1_enabled": False, "spread_control_enabled": False},
            {"name": "K_100", "s1_enabled": True, "spread_control_enabled": False,
             "s1_k": "1.0"},
            {"name": "K_150", "s1_enabled": True, "spread_control_enabled": False,
             "s1_k": "1.5"},
            {"name": "K_200", "s1_enabled": True, "spread_control_enabled": False,
             "s1_k": "2.0"},
            {"name": "K_250", "s1_enabled": True, "spread_control_enabled": False,
             "s1_k": "2.5"},
            {"name": "K_300", "s1_enabled": True, "spread_control_enabled": False,
             "s1_k": "3.0"},
        ],
    })
    return cfg


def make_two_strategy_config(base):
    """Current paper comparison: native AS versus the original full S1+SC0 arm.

    Retain the original engine and estimation path, not the fusion approximation.
    Fees, capital, latency and symbol remain explicit inputs from the base config.
    """
    cfg = make_s1_config(base)
    cfg.update(strategy_version="native-as-vs-paper-s1-sc0-v1",
               primary_account="paper_s1_sc0")
    # make_s1_config historically sets zero fees: do not override the supplied
    # execution scenario when selecting a strategy pair.
    for key in ("maker_fee", "taker_fee"):
        cfg[key] = base[key]
    cfg["accounts"] = [
        {"name": "as_base", "type": "as", "fair_value": None,
         "keeper_enabled": False, "inventory_control_enabled": False,
         "as_min_half_spread_ticks": "0", "as_inventory_skew_multiplier": "1",
         "s1_enabled": False, "spread_control_enabled": False},
        {"name": "paper_s1_sc0", "type": "as",
         "as_min_half_spread_ticks": "4", "as_inventory_skew_multiplier": "1",
         "inventory_control_enabled": False,
         "s1_enabled": True, "spread_control_enabled": True},
    ]
    return cfg
