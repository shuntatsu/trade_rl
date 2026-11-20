"""Frozen constants and digest of the low-turnover strategy family screen.

The normative text is ``docs/specs/lowturn-family-screen.md``. Everything the screen
decides with is pinned here and hashed into :func:`protocol_digest`, so an owner
authorization comment can quote exactly the protocol that runs. Nothing in this module
reads market data or computes a return.
"""

from __future__ import annotations

import hashlib
from typing import Final

import numpy as np

from trade_rl.artifacts import canonical_json_bytes
from trade_rl.evaluation.core7_mtf_study import protocol as core7_protocol
from trade_rl.risk.pretrade import PreTradeRiskConfig
from trade_rl.simulation.execution import ExecutionCostConfig

PROTOCOL_SCHEMA: Final = "lowturn_family_screen_protocol_v1"
PROTOCOL_ID: Final = "lowturn_family_screen_v1"
#: ``tools.core7_mtf_tierd`` Tier D dataset of the Phase 1 run; the screen refuses any other.
FROZEN_DATASET_ID: Final = (
    "92336a938ac1229316a1e5ecce97f5d12a2f8f4864812b2b415e83389dbbaf8b"
)
DATA_END_EXCLUSIVE: Final = core7_protocol.DATA_END_EXCLUSIVE
ROSTER: Final[tuple[str, ...]] = core7_protocol.ROSTER
LEG_ROSTER: Final[tuple[str, ...]] = core7_protocol.LEG_ROSTER
N_PAIRS: Final = len(ROSTER)
BTC_ETH_PAIRS: Final[tuple[str, ...]] = ("BTCUSDT", "ETHUSDT")

CONTROL_FAMILIES: Final[tuple[str, ...]] = ("CASH", "EW_HOLD_SPOT", "VT_HOLD_PERP")
CANDIDATE_FAMILIES: Final[tuple[str, ...]] = (
    "TS_TREND_LS",
    "TS_TREND_LONG_SPOT",
    "TS_TREND_BTCETH",
    "XS_MOM",
    "MA_BTCETH_SPOT",
)
FAMILIES: Final[tuple[str, ...]] = CONTROL_FAMILIES + CANDIDATE_FAMILIES
#: Control families the candidates must beat in the DEV BASE Sharpe.
SHARPE_BENCHMARK_CONTROLS: Final[tuple[str, ...]] = ("EW_HOLD_SPOT", "VT_HOLD_PERP")

SEGMENTS: Final[tuple[str, ...]] = ("DEV", "HOLDOUT")
SCENARIOS: Final[tuple[str, ...]] = ("BASE", "HIGH_COST", "DELAY_1BAR")
# 4h bars
LOOKBACK_BARS: Final[dict[str, int]] = {
    "7d": 42,
    "30d": 180,
    "60d": 360,
    "90d": 540,
    "150d": 900,
    "180d": 1080,
}
R30_BARS: Final = LOOKBACK_BARS["30d"]
R90_BARS: Final = LOOKBACK_BARS["90d"]
R180_BARS: Final = LOOKBACK_BARS["180d"]
VOL_WINDOW_BARS: Final = LOOKBACK_BARS["60d"]
SMA_FAST_BARS: Final = LOOKBACK_BARS["30d"]
SMA_SLOW_BARS: Final = LOOKBACK_BARS["150d"]
#: XS score is ``close[t - XS_RECENT_BARS] / close[t - XS_PAST_BARS] - 1``.
XS_RECENT_BARS: Final = LOOKBACK_BARS["7d"]
XS_PAST_BARS: Final = 402
XS_SIDE_COUNT: Final = 3
PERIODS_PER_YEAR: Final = 2190.0
VOL_FLOOR: Final = 0.10
VOL_TARGET: Final = 0.50
MA_LEG_WEIGHT: Final = 0.5

PER_NAME_CAP: Final = 0.40
GROSS_CAP: Final = 2.0
XS_GROSS_CAP: Final = 1.5
REBALANCE_EVERY_BARS: Final = LOOKBACK_BARS["7d"]
DEADBAND: Final = 0.02
WARMUP_BARS: Final = LOOKBACK_BARS["180d"]
HOLDOUT_SPLIT_TIMESTAMP: Final = "2024-07-01T00:00:00"
INITIAL_CAPITAL: Final = 100_000.0

DEV_MIN_SHARPE: Final = 0.5
HOLDOUT_MIN_SHARPE: Final = 0.3
DSR_TRIALS: Final = 5

LABEL_EXPLORATORY_CANDIDATE: Final = "EXPLORATORY_CANDIDATE"
LABEL_NO_CANDIDATE: Final = "NO_CANDIDATE"
LABEL_INVALID: Final = "INVALID"
CLAIM_CEILING: Final = LABEL_EXPLORATORY_CANDIDATE

_SCENARIO_DELAY_BARS: Final[dict[str, int]] = {
    "BASE": 0,
    "HIGH_COST": 0,
    "DELAY_1BAR": 1,
}


def scenario_cost(scenario: str) -> ExecutionCostConfig:
    """The core7 frozen ``ExecutionCostConfig`` of the scenario, unchanged."""
    if scenario not in _SCENARIO_DELAY_BARS:
        raise ValueError(f"unknown scenario: {scenario!r}")
    return core7_protocol.scenario_cost(scenario)


def scenario_signal_lag_bars(scenario: str) -> int:
    """Signal lag in bars; ``DELAY_1BAR`` lags every signal input by one bar."""
    if scenario not in _SCENARIO_DELAY_BARS:
        raise ValueError(f"unknown scenario: {scenario!r}")
    return _SCENARIO_DELAY_BARS[scenario]


def risk_config() -> PreTradeRiskConfig:
    """The risk values of ``core7_mtf_shadow_anchor_ppo_v4``, unchanged."""
    return core7_protocol.risk_config()


def segment_bounds(timestamps: np.ndarray) -> dict[str, tuple[int, int]]:
    """``{segment: (start_index, stop_index)}`` of the DEV and HOLDOUT replays.

    ``timestamps`` are the dataset's bar timestamps (bar close times). The split index is
    the first bar whose stored timestamp is at or after 2024-07-01T00:00Z. DEV runs from
    the end of the warm-up to that bar and HOLDOUT from that bar to the last row.
    """
    ts = np.asarray(timestamps).astype("datetime64[ns]")
    if ts.ndim != 1 or ts.size == 0:
        raise ValueError("timestamps must be a non-empty 1-D array")
    split = int(
        np.searchsorted(ts, np.datetime64(HOLDOUT_SPLIT_TIMESTAMP, "ns"), side="left")
    )
    last = int(ts.size - 1)
    if not WARMUP_BARS + REBALANCE_EVERY_BARS < split < last - REBALANCE_EVERY_BARS:
        raise ValueError(
            "the dataset does not span the warm-up, the DEV segment and the HOLDOUT segment"
        )
    return {"DEV": (WARMUP_BARS, split), "HOLDOUT": (split, last)}


def protocol_payload() -> dict[str, object]:
    """Canonical frozen protocol payload for digesting."""
    rc = risk_config()
    return {
        "schema_version": PROTOCOL_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "claim_ceiling": CLAIM_CEILING,
        "dataset_id": FROZEN_DATASET_ID,
        "data_end_exclusive": DATA_END_EXCLUSIVE,
        "roster": list(ROSTER),
        "leg_layout": "alternating_spot_perp_even_index_spot_odd_index_perp",
        "controls": list(CONTROL_FAMILIES),
        "candidates": list(CANDIDATE_FAMILIES),
        "sharpe_benchmark_controls": list(SHARPE_BENCHMARK_CONTROLS),
        "segments": {
            "warmup_bars": WARMUP_BARS,
            "dev": "first decision at the warm-up index; stops at the first bar with "
            f"stored timestamp >= {HOLDOUT_SPLIT_TIMESTAMP}Z",
            "holdout": "from that bar to the dataset last index",
            "each_segment_starts_flat_with": INITIAL_CAPITAL,
        },
        "scenarios": {
            name: {
                "cost": core7_protocol.scenario_cost(name).execution_policy_payload(),
                "signal_lag_bars": scenario_signal_lag_bars(name),
            }
            for name in SCENARIOS
        },
        "risk": {
            "max_gross": rc.max_gross,
            "max_abs_weight": rc.max_abs_weight,
            "max_turnover": rc.max_turnover,
            "drawdown_start": rc.drawdown_start,
            "drawdown_stop": rc.drawdown_stop,
            "emergency_turnover_override": rc.emergency_turnover_override,
        },
        "replay": {
            "path": "run_shared_portfolio_replay",
            "settle_terminal_position": True,
            "capture_ledger_evidence": True,
        },
        "lookback_bars": dict(LOOKBACK_BARS),
        "signals": {
            "decision_row": "row t - signal_lag_bars of closes at or before that row",
            "r30_r90_r180_bars": [R30_BARS, R90_BARS, R180_BARS],
            "vol": {
                "window_bars": VOL_WINDOW_BARS,
                "kind": "std_of_4h_log_returns",
                "ddof": 0,
                "periods_per_year": PERIODS_PER_YEAR,
                "floor": VOL_FLOOR,
                "target": VOL_TARGET,
            },
            "xs_score_bars": [XS_RECENT_BARS, XS_PAST_BARS],
            "xs_side_count": XS_SIDE_COUNT,
            "xs_tie_break": "stable_by_pair_index",
            "xs_minimum_available_pairs": 2 * XS_SIDE_COUNT,
            "sma_bars": [SMA_FAST_BARS, SMA_SLOW_BARS],
            "ma_leg_weight": MA_LEG_WEIGHT,
            "btc_eth_pairs": list(BTC_ETH_PAIRS),
            "unavailable_input": "weight 0",
        },
        "families": {
            "CASH": "always flat",
            "EW_HOLD_SPOT": "spot legs at 1/11",
            "VT_HOLD_PERP": "perp legs at (vol_target / vol60) / 11",
            "TS_TREND_LS": "perp w = mean(sign r30, r90, r180) * (vol_target/vol60) / 11",
            "TS_TREND_LONG_SPOT": "TS_TREND_LS on spot legs with s clipped to [0, 1]",
            "TS_TREND_BTCETH": "TS_TREND_LS on BTC and ETH perps with / 2",
            "XS_MOM": "top 3 long bottom 3 short perp at +-(vol_target/vol60)/6",
            "MA_BTCETH_SPOT": "BTC and ETH spot at 0.5 when SMA180 > SMA900 else 0",
        },
        "caps": {
            "per_name": PER_NAME_CAP,
            "gross": GROSS_CAP,
            "gross_xs_mom": XS_GROSS_CAP,
            "order": "clip each |w| to per_name then scale proportionally to gross",
        },
        "execution": {
            "rebalance_every_bars": REBALANCE_EVERY_BARS,
            "rebalance_rule": "(index - segment_start) % rebalance_every_bars == 0",
            "between_rebalances": "exact quantity HOLD",
            "deadband": DEADBAND,
            "deadband_rule": "per symbol on |risk_scale * target - current weight|",
            "risk_scale_decrease": "re-target the last rebalance targets; risk applies the "
            "scale once, an unchanged scale below 1 re-proposes current / scale",
            "spot_legs": "long only, a negative spot target is an error",
        },
        "thresholds": {
            "dev_min_base_sharpe": DEV_MIN_SHARPE,
            "holdout_min_base_sharpe": HOLDOUT_MIN_SHARPE,
            "dsr_trials": DSR_TRIALS,
            "dsr_gates": False,
        },
        "labels": [LABEL_EXPLORATORY_CANDIDATE, LABEL_NO_CANDIDATE, LABEL_INVALID],
        "initial_capital": INITIAL_CAPITAL,
    }


def protocol_digest() -> str:
    """SHA-256 hex digest of the canonical frozen protocol."""
    return hashlib.sha256(canonical_json_bytes(protocol_payload())).hexdigest()


__all__ = [
    "BTC_ETH_PAIRS",
    "CANDIDATE_FAMILIES",
    "CLAIM_CEILING",
    "CONTROL_FAMILIES",
    "DEADBAND",
    "DEV_MIN_SHARPE",
    "DSR_TRIALS",
    "FAMILIES",
    "FROZEN_DATASET_ID",
    "GROSS_CAP",
    "HOLDOUT_MIN_SHARPE",
    "HOLDOUT_SPLIT_TIMESTAMP",
    "INITIAL_CAPITAL",
    "LABEL_EXPLORATORY_CANDIDATE",
    "LABEL_INVALID",
    "LABEL_NO_CANDIDATE",
    "LEG_ROSTER",
    "LOOKBACK_BARS",
    "MA_LEG_WEIGHT",
    "N_PAIRS",
    "PER_NAME_CAP",
    "PERIODS_PER_YEAR",
    "PROTOCOL_ID",
    "PROTOCOL_SCHEMA",
    "R30_BARS",
    "R90_BARS",
    "R180_BARS",
    "REBALANCE_EVERY_BARS",
    "ROSTER",
    "SCENARIOS",
    "SEGMENTS",
    "SHARPE_BENCHMARK_CONTROLS",
    "SMA_FAST_BARS",
    "SMA_SLOW_BARS",
    "VOL_FLOOR",
    "VOL_TARGET",
    "VOL_WINDOW_BARS",
    "WARMUP_BARS",
    "XS_GROSS_CAP",
    "XS_PAST_BARS",
    "XS_RECENT_BARS",
    "XS_SIDE_COUNT",
    "protocol_digest",
    "protocol_payload",
    "risk_config",
    "scenario_cost",
    "scenario_signal_lag_bars",
    "segment_bounds",
]
