"""Frozen pre-registration for the trend-following-with-stops development study.

This module is committed BEFORE any real-data run of this rule: no return, price path
or PnL of the candidate rule was computed. The universe and window are NOT unexposed:
see :data:`PRIOR_EXPOSURE`. Every parameter below was fixed on 2026-11-10 from
convention (Turtle-style 20/10 and 55/20 Donchian channels, ATR(14), 2x initial and 3x
trailing ATR stops, 0.5% risk per trade); the thresholds (profit factor 1.2, 4 of 6
sub-years, 100 cycles, 20% drawdown, Calmar versus BTC) are round-number conventions,
and the implementer did not read the Phase 2 TSMOM figures when fixing any value (see
the specification's attestation). Revision 2 (2026-11-13) changes only the roster rule
(complete eleven symbols, no exclusions, no reduced roster) and the causality probe
(one per variant and timeframe); see :data:`SUPERSEDES_PROTOCOL_DIGEST`. The runner
reads the parameters from :data:`PROTOCOL` and never from the command line, and the
result records :func:`protocol_digest` so a result can be matched to this file.

Study: a long/short Donchian breakout with ATR stops on a fixed USD-M perpetual
roster at 4h and 1d, run through the shared-portfolio replay (one account, the repo's
fee / spread / impact / participation model, funding from the Dataset, hard risk,
terminal flat). The development window is known-used and exploratory: the result can
at most label the rule ``DEVELOPMENT_PROMISING`` and never selects a winner or
authorizes unused-data, paper or live use. See ``docs/specs/trend-following-stops.md``
for the mechanism, falsifiers, claim boundary and the later RL design note.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Final

from trade_rl.risk.pretrade import PreTradeRisk, PreTradeRiskConfig
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.rules.trend_stop import TrendStopConfig

PROTOCOL_SCHEMA: Final = "trend_stop_study_protocol_v2"
RESULT_SCHEMA: Final = "trend_stop_study_result_v2"
CELL_SCHEMA: Final = "trend_stop_study_cell_v1"
STOP_MODEL: Final = "bar_extreme_touch_next_open_v1"

SCENARIO_BASE: Final = "base"
SCENARIO_COST_X2: Final = "cost_x2"
SCENARIOS: Final = (SCENARIO_BASE, SCENARIO_COST_X2)

ARM_RULE: Final = "rule"
ARM_BTC_BUY_AND_HOLD: Final = "btc_buy_and_hold"
ARM_EQUAL_WEIGHT_BUY_AND_HOLD: Final = "equal_weight_buy_and_hold"

LABEL_PROMISING: Final = "DEVELOPMENT_PROMISING"
LABEL_NO_WINNER: Final = "NO_WINNER"
LABEL_INVALID: Final = "INVALID"

# Prior reads of this universe / window by other repository work (disclosed, not hidden).
PRIOR_EXPOSURE: Final = (
    "legacy signal metadata tests/fixtures/legacy/realdata_20260713_signal_metadata.json: "
    "a GBM cross-sectional signal gate on 15 symbols (10 of the 11 here, all but DOT), "
    "fit cutoff 2025-12-06; an information-coefficient gate, not a PnL of this rule",
    "Phase 2 trial T1 (time-series momentum, 30d/90d blend, no stops, daily decisions) "
    "on BTC, ETH, BNB, XRP, ADA over 2023-01-01..2026-08-01 reported a positive return; "
    "directly related rule family on 5 of the 11 symbols and the 2023-2026 part of the "
    "window",
    "the M34 development Dataset (BTC, ETH, BNB, XRP, ADA, 2021-01..2026-08, 4h/1d "
    "features) and the Phase 2/3 and directional studies that used it",
)

# Revision 1 (PR #1201, review 5870390989 BLOCKED) had this protocol digest; revision 2
# replaces its roster-exclusion path and its single causality probe.
SUPERSEDES_PROTOCOL_DIGEST: Final = (
    "f57f2d6dedd9a46b4d4ffc57c70dc495f60e517d8cd14f873c1cdab47d02bb9d"
)
# The owner's roster decision for revision 2 (issue #810): all eleven coins, no
# exclusions; a missing archive makes the run INVALID, never a reduced roster.
OWNER_ROSTER_DECISION_URL: Final = (
    "https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-6818495567"
)

# Hard data boundary: the exploratory window ends with the last bar CLOSING at or before
# this instant (exclusive of the FS1 warm-up span that starts here).
DATA_END_EXCLUSIVE: Final = "2026-08-01T00:00:00"

# A G4 authorization is a GitHub issue / PR comment by the owner; the runner checks the
# shape only (it cannot authenticate the author) and records the URL in the result.
G4_AUTHORIZATION_URL_PATTERN: Final = re.compile(
    r"^https://github\.com/shuntatsu/trade_rl/(issues|pull)/[0-9]+#issuecomment-[0-9]+$"
)

DEVIATIONS: Final = (
    "universe: eleven survivors of 2026-11 chosen by name and listing date, not from any "
    "return of this rule; survivorship bias and prior exposure are declared limitations",
    "stop model: bar-extreme touch evaluated on the completed bar, exit at the next "
    "open (no intrabar fill at the stop price); fills better than the stop level are "
    "removed by the preregistered stop haircut; a fired stop takes precedence over a "
    "reversal and keeps exiting until the position is flat (no entry in either "
    "direction meanwhile), so a partly filled stop exit is one stop event",
    "order_latency_bars 0 and 1 are the same next-open fill in this engine and >= 2 "
    "needs pending-order-preserving proposals, so no latency-stress cell is run",
    "entry price for the stop is the observed fill-bar open, not the realized average "
    "fill (spread/impact slippage is not in the stop distance)",
    "profit factor is execution-cost net per closed position cycle and excludes "
    "funding, which is reported per cell instead",
    "hard risk: gross 1.0, per-symbol 0.5, no turnover cap; drawdown brake from 20% "
    "(the gate level) to flat at 30% of running-peak historical drawdown",
    "Bonferroni over the four preregistered (variant, timeframe) configurations; the "
    "bootstrap interval is two-sided at coverage 1 - 0.05/4 and criterion 7 reads only "
    "its lower endpoint (a one-sided test at 0.625%)",
    "all gated metrics use the stop-haircut-adjusted equity and closed-cycle PnL; the "
    "haircut is a first-order multiplicative adjustment, not a re-simulation",
    "the roster is exactly the eleven symbols with no exclusion path and no reduced "
    "roster (owner decision, issue #810 comment 6818495567): a Dataset missing any "
    "roster symbol, or a Binance Vision cache archive with recorded HTTP 404 evidence "
    "(URL, status, time, written by the download step), writes an INVALID result with "
    "no replay; an archive missing without such evidence (not yet downloaded, "
    "transient failure, empty, failed content check) halts with no result and does "
    "not consume the run",
    "the future-perturbation probe runs for every (timeframe, variant) pair and "
    "validity criterion 8 of each configuration uses its own probe",
    "prior exposure of the universe and window is disclosed in PRIOR_EXPOSURE",
)


@dataclass(frozen=True)
class VariantSpec:
    name: str
    entry_bars: int
    exit_bars: int


VARIANTS: Final = (
    VariantSpec("donchian_20_10", 20, 10),
    VariantSpec("donchian_55_20", 55, 20),
)


@dataclass(frozen=True)
class TrendStopProtocol:
    """Every fixed parameter of the study; the defaults ARE the preregistration."""

    schema_version: str = PROTOCOL_SCHEMA
    study_id: str = "trend_stop_tsmom_20261113"
    planner_task: str = "opus-plan-trend-stops-v2-20261113"
    supersedes_protocol_digest: str = SUPERSEDES_PROTOCOL_DIGEST
    roster_decision_url: str = OWNER_ROSTER_DECISION_URL
    # Universe: chosen by name and listing date, not from any return of this rule. The
    # roster is complete or the run is INVALID; there is no exclusion path.
    symbols: tuple[str, ...] = (
        "BTCUSDT",
        "ETHUSDT",
        "BNBUSDT",
        "ADAUSDT",
        "LINKUSDT",
        "DOGEUSDT",
        "DOTUSDT",
        "AVAXUSDT",
        "LTCUSDT",
        "SOLUSDT",
        "XRPUSDT",
    )
    roster_policy: str = "complete_roster_no_exclusions"
    benchmark_symbol: str = "BTCUSDT"
    timeframes: tuple[str, ...] = ("4h", "1d")
    variants: tuple[VariantSpec, ...] = VARIANTS
    # Rule.
    atr_period: int = 14
    initial_stop_atr: float = 2.0
    trailing_stop_atr: float = 3.0
    allow_short: bool = True
    stop_model: str = STOP_MODEL
    # Sizing and hard risk.
    risk_per_trade: float = 0.005
    max_symbol_weight: float = 0.25
    gross_cap: float = 0.95
    risk_max_gross: float = 1.0
    risk_max_abs_weight: float = 0.5
    risk_drawdown_start: float = 0.20
    risk_drawdown_stop: float = 0.30
    initial_capital: float = 10_000.0
    # Window (known-used, exploratory) and data boundary.
    data_start: str = "2021-01-01T00:00:00"
    evaluation_start: str = "2021-03-01T00:00:00"
    evaluation_end: str = DATA_END_EXCLUSIVE
    data_end_exclusive: str = DATA_END_EXCLUSIVE
    sub_years: tuple[str, ...] = ("2021", "2022", "2023", "2024", "2025", "2026")
    # Costs: the repo's explicit model; funding comes from the Dataset.
    fee_rate: float = 0.0005
    spread_rate: float = 0.0002
    impact_rate: float = 0.0001
    max_participation_rate: float = 0.05
    cost_stress_multiplier: float = 2.0
    buy_and_hold_gross_budget: float = 0.95
    # Pass / fail (all must hold for a configuration to be DEVELOPMENT_PROMISING).
    min_positive_sub_years: int = 4
    min_profit_factor: float = 1.2
    max_drawdown_limit: float = 0.20
    min_closed_trades: int = 100
    reconciliation_tolerance: float = 1e-4
    # Statistics: moving-block bootstrap of daily net returns, Bonferroni over m.
    bootstrap_block_days: int = 20
    bootstrap_resamples: int = 2000
    bootstrap_seed: int = 0
    bootstrap_alpha: float = 0.05
    n_configurations: int = 4
    trading_days_per_year: int = 365
    cagr_days_per_year: float = 365.25
    # Causality probe: one per (timeframe, variant); criterion 8 uses its own probe.
    causality_cut: str = "2023-07-01T00:00:00"
    causality_probe_scope: str = "every_timeframe_and_variant"
    deviations: tuple[str, ...] = DEVIATIONS

    def canonical_payload(self) -> dict[str, object]:
        return json.loads(json.dumps(asdict(self), sort_keys=True))


PROTOCOL: Final = TrendStopProtocol()


def protocol_digest(protocol: TrendStopProtocol = PROTOCOL) -> str:
    payload = json.dumps(
        protocol.canonical_payload(),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_authorization_url(url: object) -> str:
    """Return a well-formed G4 authorization comment URL or raise ``ValueError``."""

    if not isinstance(url, str) or G4_AUTHORIZATION_URL_PATTERN.fullmatch(url) is None:
        raise ValueError(
            "a G4 authorization is required: pass the owner's issue or pull-request "
            "comment URL (https://github.com/shuntatsu/trade_rl/<issues|pull>/<n>"
            "#issuecomment-<id>); the runner cannot authenticate its author"
        )
    return url


def scenario_cost(protocol: TrendStopProtocol, scenario: str) -> ExecutionCostConfig:
    """Fee, spread and impact are all scaled in the stress; funding is not a cost."""

    if scenario == SCENARIO_BASE:
        multiplier = 1.0
    elif scenario == SCENARIO_COST_X2:
        multiplier = protocol.cost_stress_multiplier
    else:
        raise ValueError(f"unknown scenario: {scenario}")
    return ExecutionCostConfig(
        fee_rate=protocol.fee_rate * multiplier,
        spread_rate=protocol.spread_rate * multiplier,
        impact_rate=protocol.impact_rate * multiplier,
        max_participation_rate=protocol.max_participation_rate,
    )


def trend_config(protocol: TrendStopProtocol, variant: VariantSpec) -> TrendStopConfig:
    return TrendStopConfig(
        entry_bars=variant.entry_bars,
        exit_bars=variant.exit_bars,
        atr_period=protocol.atr_period,
        initial_stop_atr=protocol.initial_stop_atr,
        trailing_stop_atr=protocol.trailing_stop_atr,
        risk_per_trade=protocol.risk_per_trade,
        max_symbol_weight=protocol.max_symbol_weight,
        gross_cap=protocol.gross_cap,
        allow_short=protocol.allow_short,
    )


def hard_risk(protocol: TrendStopProtocol) -> PreTradeRisk:
    return PreTradeRisk(
        PreTradeRiskConfig(
            max_gross=protocol.risk_max_gross,
            max_abs_weight=protocol.risk_max_abs_weight,
            max_turnover=None,
            drawdown_start=protocol.risk_drawdown_start,
            drawdown_stop=protocol.risk_drawdown_stop,
        )
    )


def check_protocol_window(protocol: TrendStopProtocol) -> None:
    """Refuse any protocol whose window leaves the known-used exploratory span."""

    if protocol.evaluation_end > protocol.data_end_exclusive:
        raise ValueError(
            "evaluation window ends after the exploratory data boundary "
            f"{protocol.data_end_exclusive}; the FS1 warm-up, unused and final spans "
            "are never read by this study"
        )
    if protocol.data_end_exclusive > DATA_END_EXCLUSIVE:
        raise ValueError("data_end_exclusive may not move past the frozen boundary")
    if protocol.evaluation_start >= protocol.evaluation_end:
        raise ValueError("evaluation window must be non-empty")


__all__ = [
    "ARM_BTC_BUY_AND_HOLD",
    "ARM_EQUAL_WEIGHT_BUY_AND_HOLD",
    "ARM_RULE",
    "CELL_SCHEMA",
    "DATA_END_EXCLUSIVE",
    "DEVIATIONS",
    "G4_AUTHORIZATION_URL_PATTERN",
    "LABEL_INVALID",
    "LABEL_NO_WINNER",
    "LABEL_PROMISING",
    "PRIOR_EXPOSURE",
    "PROTOCOL",
    "PROTOCOL_SCHEMA",
    "OWNER_ROSTER_DECISION_URL",
    "RESULT_SCHEMA",
    "SCENARIOS",
    "SCENARIO_BASE",
    "SCENARIO_COST_X2",
    "STOP_MODEL",
    "SUPERSEDES_PROTOCOL_DIGEST",
    "TrendStopProtocol",
    "VARIANTS",
    "VariantSpec",
    "check_protocol_window",
    "hard_risk",
    "protocol_digest",
    "scenario_cost",
    "trend_config",
    "validate_authorization_url",
]
