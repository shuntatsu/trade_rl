"""Frozen study protocol, controls, scenarios, and disclosures for Core7 MTF Shadow-Anchor PPO."""

from __future__ import annotations

import hashlib
import re
from typing import Final

from trade_rl.artifacts import canonical_json_bytes
from trade_rl.risk.pretrade import PreTradeRiskConfig
from trade_rl.simulation.execution import ExecutionCostConfig

PROTOCOL_SCHEMA: Final = "core7_mtf_shadow_anchor_protocol_v1"
PROTOCOL_ID: Final = "core7_mtf_shadow_anchor_ppo_v1"
DATA_END_EXCLUSIVE: Final = "2026-08-01T00:00:00"
G4_AUTHORIZATION_URL_PATTERN: Final = re.compile(
    r"^https://github\.com/shuntatsu/trade_rl/(issues|pull)/[0-9]+#issuecomment-[0-9]+$"
)

ROSTER: Final[tuple[str, ...]] = (
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "LTCUSDT",
    "DOTUSDT",
)
LEG_ROSTER: Final[tuple[str, ...]] = tuple(
    leg for symbol in ROSTER for leg in (f"{symbol}:spot", f"{symbol}:perp")
)
MIN_PAIRS: Final = 2
MAX_PAIRS: Final = 11

ARMS: Final[tuple[str, ...]] = (
    "CASH",
    "ANC_BIDIR_75_25",
    "ANC_REGIME_GATE_NONRL",
    "CORE7_SPARSE_PPO",
    "ORTH6_CUSH_PPO",
)
NONRL_CONTROL_ARMS: Final[tuple[str, ...]] = (
    "CASH",
    "ANC_BIDIR_75_25",
    "ANC_REGIME_GATE_NONRL",
)
RL_ARMS: Final[tuple[str, ...]] = (
    "CORE7_SPARSE_PPO",
    "ORTH6_CUSH_PPO",
)
SCENARIOS: Final[tuple[str, ...]] = (
    "BASE",
    "LOW_COST",
    "HIGH_COST",
    "DELAY_1BAR",
)

DEFAULT_INITIAL_CAPITAL: Final = 100_000.0
DEFAULT_REBALANCE_EVERY_BARS: Final = 6
DEFAULT_DEADBAND: Final = 0.04

_SCENARIO_RATES: Final[dict[str, tuple[float, float, float]]] = {
    "BASE": (0.0004, 0.0002, 0.0002),
    "LOW_COST": (0.0002, 0.0001, 0.0001),
    "HIGH_COST": (0.0008, 0.0004, 0.0004),
    "DELAY_1BAR": (0.0004, 0.0002, 0.0002),
}

DEVIATIONS: Final[tuple[str, ...]] = (
    "universe_known_used: 11 crypto pairs observed across 2021-01-01 to 2026-07-31 Pre-FS1 "
    "exploratory window; results remain Tier D known-used development evidence and do not open FS1/unused data",
    "delay_1bar_semantics: order_latency_bars 0 and 1 both settle at next-bar open in the shared "
    "execution engine while latency=1 trims agent_stop_index, so DELAY_1BAR keeps order_latency_bars=0 "
    "and applies a genuine 1-bar signal_lag_bars=1 delay inside ShadowAnchorMtfPolicy",
    "capacity_reference: processing_bar_volume_capacity=False across all scenarios so the 1% "
    "participation cap reads only completed bar t-1 volume",
    "nonrl_benchmark: ANC_REGIME_GATE_NONRL applies fixed causal sigmoid/clip regime gates "
    "(g_alpha, g_carry) from 6D opportunity state as a mandatory Non-RL control alongside "
    "CASH and static ANC_BIDIR_75_25",
    "phase0_scope: Phase 0 establishes software/oracle/causality/fail-closed validity on synthetic "
    "inputs only; real-data parity (Phase 1+) requires independent G0-G2 review and separate owner G4 URL",
)

PRIOR_EXPOSURE: Final[tuple[str, ...]] = (
    "pre_fs1_rounds_1_to_11: extensive multi-timeframe feature engineering, carry integration, "
    "and GPU PPO screening conducted on 2021-01-01..2026-07-31 11-symbol Binance spot/perp data "
    "identified CORE7_SPARSE_PPO and ORTH6_CUSH_PPO as candidate shadow-anchor architectures",
    "pre_fs1_round_12_null_result: GPU evaluation of regime-throttled RL vs Non-RL causal regime "
    "control (ANC_REGIME_GATE_NONRL) returned H13-A NOT SUPPORTED and H13-B NOT SUPPORTED, "
    "establishing that non-RL regime gating reproduced drawdown throttling without RL complexity",
    "no_economic_authorization: prior exploratory scratch runs do not constitute Phase 1 "
    "shared-ledger parity evidence or live trading eligibility",
)


def scenario_cost(scenario: str) -> ExecutionCostConfig:
    """Return the frozen ``ExecutionCostConfig`` for a named stress scenario."""
    if scenario not in _SCENARIO_RATES:
        raise ValueError(f"unknown scenario: {scenario!r}")
    fee, spread, impact = _SCENARIO_RATES[scenario]
    return ExecutionCostConfig(
        fee_rate=fee,
        spread_rate=spread,
        impact_rate=impact,
        allow_short=True,
        max_leverage=2.5,
        maintenance_margin_rate=0.05,
        max_participation_rate=0.01,
        minimum_notional=10.0,
        order_latency_bars=0,
        processing_bar_volume_capacity=False,
    )


def scenario_signal_delay_bars(scenario: str) -> int:
    """Return the policy signal delay in bars for a named scenario."""
    if scenario not in _SCENARIO_RATES:
        raise ValueError(f"unknown scenario: {scenario!r}")
    return 1 if scenario == "DELAY_1BAR" else 0


def risk_config() -> PreTradeRiskConfig:
    """Return the frozen ``PreTradeRiskConfig`` compatible with 2.5x gross cap."""
    return PreTradeRiskConfig(
        max_gross=2.5,
        max_abs_weight=0.55,
        max_turnover=2.5,
        drawdown_start=0.10,
        drawdown_stop=0.20,
        emergency_turnover_override=True,
    )


def validate_authorization_url(url: str) -> str:
    """Validate GitHub owner G4 authorization comment URL format."""
    if not isinstance(url, str):
        raise TypeError("authorization_url must be a string")
    cleaned = url.strip()
    if G4_AUTHORIZATION_URL_PATTERN.fullmatch(cleaned) is None:
        raise ValueError(
            "authorization_url must match https://github.com/shuntatsu/trade_rl/(issues|pull)/<id>#issuecomment-<id>"
        )
    return cleaned


def protocol_payload() -> dict[str, object]:
    """Return canonical frozen protocol payload for digesting."""
    rc = risk_config()
    return {
        "schema_version": PROTOCOL_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "data_end_exclusive": DATA_END_EXCLUSIVE,
        "roster": list(ROSTER),
        "leg_roster": list(LEG_ROSTER),
        "min_pairs": MIN_PAIRS,
        "max_pairs": MAX_PAIRS,
        "arms": list(ARMS),
        "nonrl_control_arms": list(NONRL_CONTROL_ARMS),
        "rl_arms": list(RL_ARMS),
        "scenarios": {
            name: {
                "cost": scenario_cost(name).execution_policy_payload(),
                "signal_delay_bars": scenario_signal_delay_bars(name),
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
        "initial_capital": DEFAULT_INITIAL_CAPITAL,
        "rebalance_every_bars": DEFAULT_REBALANCE_EVERY_BARS,
        "deadband": DEFAULT_DEADBAND,
        "deviations": list(DEVIATIONS),
        "prior_exposure": list(PRIOR_EXPOSURE),
    }


def protocol_digest() -> str:
    """Compute SHA-256 hex digest of the canonical frozen protocol."""
    return hashlib.sha256(canonical_json_bytes(protocol_payload())).hexdigest()


__all__ = [
    "ARMS",
    "DATA_END_EXCLUSIVE",
    "DEFAULT_DEADBAND",
    "DEFAULT_INITIAL_CAPITAL",
    "DEFAULT_REBALANCE_EVERY_BARS",
    "DEVIATIONS",
    "G4_AUTHORIZATION_URL_PATTERN",
    "LEG_ROSTER",
    "MAX_PAIRS",
    "MIN_PAIRS",
    "NONRL_CONTROL_ARMS",
    "PRIOR_EXPOSURE",
    "PROTOCOL_ID",
    "PROTOCOL_SCHEMA",
    "RL_ARMS",
    "ROSTER",
    "SCENARIOS",
    "protocol_digest",
    "protocol_payload",
    "risk_config",
    "scenario_cost",
    "scenario_signal_delay_bars",
    "validate_authorization_url",
]
