"""Bounded real-data development study runner and contracts for TD5 (#1025).

This module implements the frozen development study protocol on real market data
under the "real_development" source scope. It enforces strict preflight validation,
causal training isolation, write-once artifact persistence, ledger reconciliation,
and fail-closed causality verification.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path

import numpy as np

from trade_rl._validation import require_sha256
from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.decision_value_fit import (
    fit_conservative_linear_decision_value,
    fit_synthetic_linear_decision_value,
    fit_synthetic_shallow_decision_value,
)
from trade_rl.evaluation.decision_value_plan import DecisionValueStudyPlan
from trade_rl.evaluation.paired_portfolio_action import run_paired_portfolio_action
from trade_rl.evaluation.paired_portfolio_label import (
    SyntheticPairedLabel,
    build_synthetic_paired_label,
    build_synthetic_paired_label_batch,
)
from trade_rl.evaluation.replay import (
    SharedCashPortfolioReplayResult,
    run_shared_portfolio_replay,
)
from trade_rl.risk.pretrade import PreTradeRisk, PreTradeRiskConfig
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.simulation.stateful.portfolio_session import SharedPortfolioSession
from trade_rl.strategies.decision_value import (
    ConservativeLinearDecisionValuePolicy,
    LinearDecisionValuePolicy,
    LinearDecisionValueRuntimeContext,
    compute_effective_posture_weights,
    preserve_posture_aware_proposal,
)
from trade_rl.strategies.decision_value_shallow import (
    ShallowDecisionValuePolicy,
)
from trade_rl.strategies.forecasts.ridge import (
    RidgeForecastModel,
    RidgeForecastStrategy,
    fit_ridge_forecast,
)
from trade_rl.strategies.interface import StrategyObservation
from trade_rl.strategies.portfolio import (
    PortfolioDecisionSnapshot,
    PortfolioPolicy,
    PortfolioTargetProposal,
)
from trade_rl.strategies.portfolio_adapter import (
    CashPortfolioPolicy,
)
from trade_rl.strategies.portfolio_observation import (
    build_portfolio_decision_snapshot,
)
from trade_rl.strategies.portfolio_policy_codec import (
    ACTION_FLAT,
    ACTION_HOLD,
    ACTION_LONG,
    ACTION_SHORT,
)
from trade_rl.strategies.position_intent import PositionIntent

DEFAULT_SCENARIOS: tuple[tuple[str, float, int], ...] = (
    ("base", 1.0, 0),
    ("cost_x2", 2.0, 0),
    ("latency_1bar", 1.0, 1),
    ("latency_2bar", 1.0, 2),
)


def _finite(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a number")
    val = float(value)
    if not math.isfinite(val):
        raise ValueError(f"{field} must be finite")
    return val


def _verify_ledger_reconciliation(
    replay_res: SharedCashPortfolioReplayResult,
    *,
    expected_intervals: int,
) -> bool:
    ledger = replay_res.ledger_evidence
    if ledger is None:
        return False
    iv = ledger.intervals
    if len(iv.intervals) != expected_intervals:
        return False
    book = replay_res.book
    return (
        math.isclose(iv.final_portfolio_value, float(book.portfolio_value))
        and math.isclose(iv.final_cash, float(book.cash))
        and math.isclose(iv.final_total_cost, float(book.total_cost))
        and math.isclose(iv.final_funding_pnl, float(book.funding_pnl))
        and math.isclose(iv.final_borrow_cost, float(book.borrow_cost))
        and math.isclose(iv.final_turnover_total, float(book.turnover_total))
        and math.isclose(iv.final_max_drawdown, float(book.max_drawdown))
    )


@dataclass(frozen=True, slots=True)
class DecisionValueBlockSpec:
    """One chronological walk-forward training, fit-cutoff, and evaluation block."""

    block_id: str
    train_start_index: int
    train_stop_index: int
    train_step: int
    fit_cutoff_index: int
    eval_start_index: int
    eval_stop_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.block_id, str) or not self.block_id.strip():
            raise ValueError("block_id must be a non-empty string")
        for name in (
            "train_start_index",
            "train_stop_index",
            "train_step",
            "fit_cutoff_index",
            "eval_start_index",
            "eval_stop_index",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be a native integer")
        if self.train_start_index < 0:
            raise ValueError("train_start_index must be non-negative")
        if self.train_step <= 0:
            raise ValueError("train_step must be positive")
        if self.train_start_index >= self.train_stop_index:
            raise ValueError("train_start_index must precede train_stop_index")
        if not (
            self.train_stop_index
            < self.fit_cutoff_index
            < self.eval_start_index
            < self.eval_stop_index
        ):
            raise ValueError(
                "indices must satisfy train_stop_index < fit_cutoff_index < eval_start_index < eval_stop_index"
            )

    def to_mapping(self) -> dict[str, object]:
        return {
            "block_id": self.block_id,
            "train_start_index": self.train_start_index,
            "train_stop_index": self.train_stop_index,
            "train_step": self.train_step,
            "fit_cutoff_index": self.fit_cutoff_index,
            "eval_start_index": self.eval_start_index,
            "eval_stop_index": self.eval_stop_index,
        }


def compute_real_scope_authorization_digest(
    *,
    dataset_id: str,
    expected_artifact_digest: str,
    symbols: tuple[str, ...],
    initial_capital: float,
    target_abs_weights: tuple[float, ...],
    horizon_bars: int,
    terminal_reserve_bars: int,
    candidate_actions: tuple[tuple[int, ...], ...],
    fee_rate: float,
    spread_rate: float,
    impact_rate: float,
    max_participation_rate: float,
    risk_config: PreTradeRiskConfig,
    blocks: tuple[DecisionValueBlockSpec, ...],
    ridge_feature_indices: tuple[int, ...],
    ridge_horizon_hours: int,
    ridge_alpha: float,
    ridge_entry_threshold: float,
    ridge_exit_threshold: float,
    prior_usage_disclosure: str,
    learner_roster: tuple[str, ...] = ("linear", "conservative_linear", "shallow"),
    value_ridge_penalty: float = 10.0,
    uncertainty_penalty_weight: float = 1.0,
    scenarios: tuple[tuple[str, float, int], ...] = DEFAULT_SCENARIOS,
) -> str:
    """Canonical lowercase SHA-256 digest of the pre-authorization specification."""
    payload = {
        "schema_version": "decision_value_real_scope_authorization_v1",
        "dataset_id": dataset_id,
        "expected_artifact_digest": expected_artifact_digest,
        "symbols": list(symbols),
        "initial_capital": float(initial_capital),
        "target_abs_weights": [float(w) for w in target_abs_weights],
        "horizon_bars": int(horizon_bars),
        "terminal_reserve_bars": int(terminal_reserve_bars),
        "candidate_actions": [list(a) for a in candidate_actions],
        "fee_rate": float(fee_rate),
        "spread_rate": float(spread_rate),
        "impact_rate": float(impact_rate),
        "max_participation_rate": float(max_participation_rate),
        "risk_config": {
            "max_gross": risk_config.max_gross,
            "max_abs_weight": risk_config.max_abs_weight,
            "max_turnover": risk_config.max_turnover,
            "drawdown_start": risk_config.drawdown_start,
            "drawdown_stop": risk_config.drawdown_stop,
        },
        "blocks": [b.to_mapping() for b in blocks],
        "ridge_feature_indices": list(ridge_feature_indices),
        "ridge_horizon_hours": int(ridge_horizon_hours),
        "ridge_alpha": float(ridge_alpha),
        "ridge_entry_threshold": float(ridge_entry_threshold),
        "ridge_exit_threshold": float(ridge_exit_threshold),
        "prior_usage_disclosure": prior_usage_disclosure,
        "learner_roster": list(learner_roster),
        "value_ridge_penalty": float(value_ridge_penalty),
        "uncertainty_penalty_weight": float(uncertainty_penalty_weight),
        "scenarios": [
            {
                "name": name,
                "cost_multiplier": float(cost_mult),
                "order_latency_bars": int(lat),
            }
            for name, cost_mult, lat in scenarios
        ],
    }
    return content_digest(payload)


@dataclass(frozen=True, slots=True)
class DecisionValueDevStudyProtocol:
    """Frozen protocol specification binding all parameters for the TD5 study."""

    study_plan: DecisionValueStudyPlan
    dataset_id: str
    expected_artifact_digest: str
    blocks: tuple[DecisionValueBlockSpec, ...]
    target_abs_weights: tuple[float, ...]
    risk_config: PreTradeRiskConfig
    base_execution_cost: ExecutionCostConfig
    ridge_feature_indices: tuple[int, ...]
    ridge_horizon_hours: int
    ridge_alpha: float
    ridge_entry_threshold: float
    ridge_exit_threshold: float
    prior_usage_disclosure: str
    value_ridge_penalty: float = 10.0
    uncertainty_penalty_weight: float = 1.0
    scenarios: tuple[tuple[str, float, int], ...] = DEFAULT_SCENARIOS

    def __post_init__(self) -> None:
        if not isinstance(self.study_plan, DecisionValueStudyPlan):
            raise TypeError("study_plan must be DecisionValueStudyPlan")
        if self.study_plan.data_scope != "real_development":
            raise ValueError("study_plan data_scope must be real_development")
        require_sha256(self.dataset_id, field="dataset_id")
        require_sha256(self.expected_artifact_digest, field="expected_artifact_digest")
        if type(self.blocks) is not tuple or not (1 <= len(self.blocks) <= 3):
            raise ValueError("protocol must specify between 1 and 3 blocks")
        for block in self.blocks:
            if not isinstance(block, DecisionValueBlockSpec):
                raise TypeError("blocks must contain DecisionValueBlockSpec instances")
            if (
                block.train_stop_index + self.study_plan.horizon_bars
                >= block.fit_cutoff_index
            ):
                raise ValueError(
                    "train_stop_index + horizon_bars must precede fit_cutoff_index"
                )
        for i in range(len(self.blocks) - 1):
            if (
                self.blocks[i].eval_stop_index > self.blocks[i + 1].eval_start_index
                or self.blocks[i].fit_cutoff_index
                >= self.blocks[i + 1].fit_cutoff_index
            ):
                raise ValueError("blocks must be strictly chronologically ordered")
        if type(self.target_abs_weights) is not tuple or len(
            self.target_abs_weights
        ) != len(self.study_plan.symbols):
            raise ValueError("target_abs_weights must match symbols length")
        for weight in self.target_abs_weights:
            if isinstance(weight, bool) or not isinstance(weight, (int, float)):
                raise TypeError("target_abs_weights must contain real numbers")
            if not math.isfinite(weight) or weight <= 0.0:
                raise ValueError("target_abs_weights must be positive and finite")
        if not isinstance(self.risk_config, PreTradeRiskConfig):
            raise TypeError("risk_config must be PreTradeRiskConfig")
        if not isinstance(self.base_execution_cost, ExecutionCostConfig):
            raise TypeError("base_execution_cost must be ExecutionCostConfig")
        if self.base_execution_cost.order_latency_bars != 0:
            raise ValueError("base_execution_cost order_latency_bars must be 0")
        if not math.isclose(
            self.base_execution_cost.fee_rate, self.study_plan.fee_rate
        ) or not math.isclose(
            self.base_execution_cost.spread_rate, self.study_plan.spread_rate
        ):
            raise ValueError("base_execution_cost must match study_plan cost rates")
        if (
            type(self.ridge_feature_indices) is not tuple
            or not self.ridge_feature_indices
        ):
            raise ValueError("ridge_feature_indices must be a non-empty tuple")
        if len(set(self.ridge_feature_indices)) != len(self.ridge_feature_indices):
            raise ValueError("ridge_feature_indices must be unique")
        for idx in self.ridge_feature_indices:
            if isinstance(idx, bool) or not isinstance(idx, int) or idx < 0:
                raise ValueError("ridge_feature_indices must be non-negative integers")
        for name in (
            "ridge_horizon_hours",
            "ridge_alpha",
            "ridge_entry_threshold",
            "ridge_exit_threshold",
        ):
            val = getattr(self, name)
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise TypeError(f"{name} must be a number")
            if not math.isfinite(val) or val <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if (
            not isinstance(self.prior_usage_disclosure, str)
            or not self.prior_usage_disclosure.strip()
        ):
            raise ValueError("prior_usage_disclosure must be a non-empty string")
        pen = _finite(self.value_ridge_penalty, field="value_ridge_penalty")
        if pen <= 0.0:
            raise ValueError("value_ridge_penalty must be positive and finite")
        unc_w = _finite(
            self.uncertainty_penalty_weight, field="uncertainty_penalty_weight"
        )
        if unc_w < 0.0:
            raise ValueError(
                "uncertainty_penalty_weight must be non-negative and finite"
            )

        expected_digest = compute_real_scope_authorization_digest(
            dataset_id=self.dataset_id,
            expected_artifact_digest=self.expected_artifact_digest,
            symbols=self.study_plan.symbols,
            initial_capital=self.study_plan.initial_capital,
            target_abs_weights=self.target_abs_weights,
            horizon_bars=self.study_plan.horizon_bars,
            terminal_reserve_bars=self.study_plan.terminal_reserve_bars,
            candidate_actions=self.study_plan.candidate_actions,
            fee_rate=self.study_plan.fee_rate,
            spread_rate=self.study_plan.spread_rate,
            impact_rate=self.base_execution_cost.impact_rate,
            max_participation_rate=self.base_execution_cost.max_participation_rate,
            risk_config=self.risk_config,
            blocks=self.blocks,
            ridge_feature_indices=self.ridge_feature_indices,
            ridge_horizon_hours=self.ridge_horizon_hours,
            ridge_alpha=self.ridge_alpha,
            ridge_entry_threshold=self.ridge_entry_threshold,
            ridge_exit_threshold=self.ridge_exit_threshold,
            prior_usage_disclosure=self.prior_usage_disclosure,
            learner_roster=self.study_plan.learner_roster,
            value_ridge_penalty=self.value_ridge_penalty,
            uncertainty_penalty_weight=self.uncertainty_penalty_weight,
            scenarios=self.scenarios,
        )
        if self.study_plan.real_scope_authorization_digest != expected_digest:
            raise ValueError(
                "study_plan real_scope_authorization_digest does not match pre-authorization specification"
            )

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": "decision_value_dev_study_protocol_v1",
            "study_plan": {
                "data_scope": self.study_plan.data_scope,
                "account_mode": self.study_plan.account_mode,
                "symbols": list(self.study_plan.symbols),
                "initial_capital": self.study_plan.initial_capital,
                "horizon_bars": self.study_plan.horizon_bars,
                "terminal_reserve_bars": self.study_plan.terminal_reserve_bars,
                "candidate_actions": [
                    list(a) for a in self.study_plan.candidate_actions
                ],
                "fee_rate": self.study_plan.fee_rate,
                "spread_rate": self.study_plan.spread_rate,
                "max_policy_improvements": self.study_plan.max_policy_improvements,
                "learner_roster": list(self.study_plan.learner_roster),
                "real_scope_authorization_digest": (
                    self.study_plan.real_scope_authorization_digest
                ),
            },
            "dataset_id": self.dataset_id,
            "expected_artifact_digest": self.expected_artifact_digest,
            "blocks": [b.to_mapping() for b in self.blocks],
            "target_abs_weights": list(self.target_abs_weights),
            "risk_config": {
                "max_gross": self.risk_config.max_gross,
                "max_abs_weight": self.risk_config.max_abs_weight,
                "max_turnover": self.risk_config.max_turnover,
                "drawdown_start": self.risk_config.drawdown_start,
                "drawdown_stop": self.risk_config.drawdown_stop,
            },
            "base_execution_cost": {
                "fee_rate": self.base_execution_cost.fee_rate,
                "spread_rate": self.base_execution_cost.spread_rate,
                "impact_rate": self.base_execution_cost.impact_rate,
                "max_participation_rate": self.base_execution_cost.max_participation_rate,
                "order_latency_bars": self.base_execution_cost.order_latency_bars,
            },
            "ridge_feature_indices": list(self.ridge_feature_indices),
            "ridge_horizon_hours": self.ridge_horizon_hours,
            "ridge_alpha": self.ridge_alpha,
            "ridge_entry_threshold": self.ridge_entry_threshold,
            "ridge_exit_threshold": self.ridge_exit_threshold,
            "prior_usage_disclosure": self.prior_usage_disclosure,
            "value_ridge_penalty": self.value_ridge_penalty,
            "uncertainty_penalty_weight": self.uncertainty_penalty_weight,
            "scenarios": [
                {
                    "name": name,
                    "cost_multiplier": float(cost_mult),
                    "order_latency_bars": int(lat),
                }
                for name, cost_mult, lat in self.scenarios
            ],
        }

    @property
    def protocol_digest(self) -> str:
        return content_digest(self.to_mapping())


def default_2024q1_decision_value_dev_protocol() -> DecisionValueDevStudyProtocol:
    """Return the frozen 2024Q1 development study protocol for bot-development-2024q1."""
    dataset_id = "7cff150e0f4d18dcc457009232f53cc7ca3f350db4893c433db6494887979616"
    expected_artifact_digest = (
        "96361d309652557382d8b74b96dac661d7ae568cb78db957e9032b71329d550c"
    )
    symbols = ("BTCUSDT", "ETHUSDT")
    initial_capital = 100_000.0
    target_abs_weights = (0.25, 0.25)
    horizon_bars = 24
    terminal_reserve_bars = 1
    candidate_actions = (
        (ACTION_HOLD, ACTION_HOLD),
        (ACTION_FLAT, ACTION_FLAT),
        (ACTION_LONG, ACTION_FLAT),
        (ACTION_SHORT, ACTION_FLAT),
        (ACTION_FLAT, ACTION_LONG),
        (ACTION_FLAT, ACTION_SHORT),
        (ACTION_LONG, ACTION_LONG),
        (ACTION_SHORT, ACTION_SHORT),
        (ACTION_LONG, ACTION_SHORT),
        (ACTION_SHORT, ACTION_LONG),
    )
    fee_rate = 0.0005
    spread_rate = 0.0002
    impact_rate = 0.0001
    max_participation_rate = 0.05
    risk_config = PreTradeRiskConfig(
        max_gross=0.60,
        max_abs_weight=0.30,
        max_turnover=1.0,
        drawdown_start=0.15,
        drawdown_stop=0.20,
    )
    base_execution_cost = ExecutionCostConfig(
        fee_rate=fee_rate,
        spread_rate=spread_rate,
        impact_rate=impact_rate,
        max_participation_rate=max_participation_rate,
        order_latency_bars=0,
    )
    blocks = (
        DecisionValueBlockSpec(
            block_id="block_1",
            train_start_index=24,
            train_stop_index=954,
            train_step=6,
            fit_cutoff_index=984,
            eval_start_index=985,
            eval_stop_index=1344,
        ),
        DecisionValueBlockSpec(
            block_id="block_2",
            train_start_index=24,
            train_stop_index=1338,
            train_step=6,
            fit_cutoff_index=1368,
            eval_start_index=1369,
            eval_stop_index=1728,
        ),
        DecisionValueBlockSpec(
            block_id="block_3",
            train_start_index=24,
            train_stop_index=1722,
            train_step=6,
            fit_cutoff_index=1752,
            eval_start_index=1753,
            eval_stop_index=2183,
        ),
    )
    ridge_feature_indices = tuple(range(9))
    ridge_horizon_hours = 24
    ridge_alpha = 1.0
    ridge_entry_threshold = 0.0020
    ridge_exit_threshold = 0.0005
    prior_usage_disclosure = (
        "data/bot-development-2024q1 is Tier D exploratory development data "
        "previously consumed in bot-development-2024q1 and bot-cash-control-2024q1; "
        "not eligible for G5 unused validation or live trading promotion."
    )
    value_ridge_penalty = 10.0
    uncertainty_penalty_weight = 1.0
    scenarios = DEFAULT_SCENARIOS
    learner_roster = ("linear", "conservative_linear", "shallow")
    auth_digest = compute_real_scope_authorization_digest(
        dataset_id=dataset_id,
        expected_artifact_digest=expected_artifact_digest,
        symbols=symbols,
        initial_capital=initial_capital,
        target_abs_weights=target_abs_weights,
        horizon_bars=horizon_bars,
        terminal_reserve_bars=terminal_reserve_bars,
        candidate_actions=candidate_actions,
        fee_rate=fee_rate,
        spread_rate=spread_rate,
        impact_rate=impact_rate,
        max_participation_rate=max_participation_rate,
        risk_config=risk_config,
        blocks=blocks,
        ridge_feature_indices=ridge_feature_indices,
        ridge_horizon_hours=ridge_horizon_hours,
        ridge_alpha=ridge_alpha,
        ridge_entry_threshold=ridge_entry_threshold,
        ridge_exit_threshold=ridge_exit_threshold,
        prior_usage_disclosure=prior_usage_disclosure,
        learner_roster=learner_roster,
        value_ridge_penalty=value_ridge_penalty,
        uncertainty_penalty_weight=uncertainty_penalty_weight,
        scenarios=scenarios,
    )
    study_plan = DecisionValueStudyPlan(
        data_scope="real_development",
        account_mode="shared_portfolio",
        symbols=symbols,
        initial_capital=initial_capital,
        horizon_bars=horizon_bars,
        terminal_reserve_bars=terminal_reserve_bars,
        candidate_actions=candidate_actions,
        fee_rate=fee_rate,
        spread_rate=spread_rate,
        max_policy_improvements=1,
        learner_roster=learner_roster,
        real_scope_authorization_digest=auth_digest,
    )
    return DecisionValueDevStudyProtocol(
        study_plan=study_plan,
        dataset_id=dataset_id,
        expected_artifact_digest=expected_artifact_digest,
        blocks=blocks,
        target_abs_weights=target_abs_weights,
        risk_config=risk_config,
        base_execution_cost=base_execution_cost,
        ridge_feature_indices=ridge_feature_indices,
        ridge_horizon_hours=ridge_horizon_hours,
        ridge_alpha=ridge_alpha,
        ridge_entry_threshold=ridge_entry_threshold,
        ridge_exit_threshold=ridge_exit_threshold,
        prior_usage_disclosure=prior_usage_disclosure,
        value_ridge_penalty=value_ridge_penalty,
        uncertainty_penalty_weight=uncertainty_penalty_weight,
        scenarios=scenarios,
    )


@dataclass(frozen=True, slots=True)
class DecisionValueArmMetrics:
    """Metrics for one arm under one scenario in one block."""

    arm: str
    scenario: str
    initial_capital: float
    final_portfolio_value: float
    net_return: float
    total_cost: float
    funding_pnl: float
    max_drawdown: float
    n_decisions: int
    terminal_quantities: tuple[str, ...]
    terminal_flat: bool

    def to_mapping(self) -> dict[str, object]:
        return {
            "arm": self.arm,
            "scenario": self.scenario,
            "initial_capital": self.initial_capital,
            "final_portfolio_value": self.final_portfolio_value,
            "net_return": self.net_return,
            "total_cost": self.total_cost,
            "funding_pnl": self.funding_pnl,
            "max_drawdown": self.max_drawdown,
            "n_decisions": self.n_decisions,
            "terminal_quantities": list(self.terminal_quantities),
            "terminal_flat": self.terminal_flat,
        }


@dataclass(frozen=True, slots=True)
class DecisionValueBlockResult:
    """Results, counters, and verification receipts for one chronological block."""

    block_id: str
    paired_replay_runs: int
    paired_transitions: int
    training_rows: int
    overlap_groups: int
    eval_replay_runs: int
    wall_clock_seconds: float
    b0_model_intercept: float
    b1_model_coefficients_sha256: str
    b1c_model_coefficients_sha256: str
    b2_model_coefficients_sha256: str
    causality_probe_passed: bool
    ledger_evidence_reconciled: bool
    arms: tuple[DecisionValueArmMetrics, ...]

    def to_mapping(self) -> dict[str, object]:
        return {
            "block_id": self.block_id,
            "paired_replay_runs": self.paired_replay_runs,
            "paired_transitions": self.paired_transitions,
            "training_rows": self.training_rows,
            "overlap_groups": self.overlap_groups,
            "eval_replay_runs": self.eval_replay_runs,
            "wall_clock_seconds": self.wall_clock_seconds,
            "b0_model_intercept": self.b0_model_intercept,
            "b1_model_coefficients_sha256": self.b1_model_coefficients_sha256,
            "b1c_model_coefficients_sha256": self.b1c_model_coefficients_sha256,
            "b2_model_coefficients_sha256": self.b2_model_coefficients_sha256,
            "causality_probe_passed": self.causality_probe_passed,
            "ledger_evidence_reconciled": self.ledger_evidence_reconciled,
            "arms": [arm.to_mapping() for arm in self.arms],
        }


@dataclass(frozen=True, slots=True)
class DecisionValueDevStudyResult:
    """Full empirical outcome of the TD5 development study."""

    schema_version: str
    protocol_digest: str
    dataset_id: str
    actual_artifact_digest: str
    verdict: str
    verdict_reason: str
    blocks: tuple[DecisionValueBlockResult, ...]
    aggregate_metrics: dict[str, dict[str, float]]
    total_wall_clock_seconds: float

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "protocol_digest": self.protocol_digest,
            "dataset_id": self.dataset_id,
            "actual_artifact_digest": self.actual_artifact_digest,
            "verdict": self.verdict,
            "verdict_reason": self.verdict_reason,
            "blocks": [b.to_mapping() for b in self.blocks],
            "aggregate_metrics": self.aggregate_metrics,
            "total_wall_clock_seconds": self.total_wall_clock_seconds,
        }


class _HoldQuantityPolicy:
    """Posture-aware and in-flight-preserving HOLD reference policy."""

    __slots__ = ("_target_abs_weights",)

    def __init__(
        self,
        target_abs_weights: tuple[float, ...] = (0.25, 0.25),
    ) -> None:
        self._target_abs_weights = target_abs_weights

    def decide_portfolio(
        self, snapshot: PortfolioDecisionSnapshot
    ) -> PortfolioTargetProposal:
        return preserve_posture_aware_proposal(
            (ACTION_HOLD,) * len(snapshot.symbols),
            snapshot,
            self._target_abs_weights,
            all_scores_non_positive=False,
        )


class _FixedEntryPolicy:
    """One fixed entry on the first decision bar, followed by exact-quantity HOLD."""

    __slots__ = ("_action", "_target_abs_weights", "_entered")

    def __init__(
        self,
        *,
        action: tuple[int, ...],
        target_abs_weights: tuple[float, ...],
    ) -> None:
        self._action = action
        self._target_abs_weights = target_abs_weights
        self._entered = False

    def decide_portfolio(
        self, snapshot: PortfolioDecisionSnapshot
    ) -> PortfolioTargetProposal:
        if not self._entered:
            self._entered = True
            return preserve_posture_aware_proposal(
                self._action,
                snapshot,
                self._target_abs_weights,
                all_scores_non_positive=False,
            )
        return preserve_posture_aware_proposal(
            (ACTION_HOLD,) * len(snapshot.symbols),
            snapshot,
            self._target_abs_weights,
            all_scores_non_positive=False,
        )


class _CostAwareRidgePortfolioPolicy:
    """Evaluate frozen RidgeForecastStrategy across symbols with cost-aware intent control."""

    __slots__ = (
        "_symbols",
        "_budgets",
        "_strategies",
        "_last_index",
        "_last_timestamp_ns",
        "_agent_stop_index",
    )

    def __init__(
        self,
        *,
        dataset: MarketDataset,
        model: RidgeForecastModel,
        target_abs_weights: tuple[float, ...],
        entry_threshold: float,
        exit_threshold: float,
        one_way_switch_cost: float,
    ) -> None:
        self._symbols = tuple(dataset.symbols)
        self._budgets = target_abs_weights
        self._strategies = tuple(
            RidgeForecastStrategy(
                model,
                entry_threshold=entry_threshold,
                exit_threshold=exit_threshold,
                one_way_switch_cost=one_way_switch_cost,
            )
            for _ in self._symbols
        )
        self._last_index: int | None = None
        self._last_timestamp_ns: int | None = None
        self._agent_stop_index: int | None = None

    def decide_portfolio(
        self, snapshot: PortfolioDecisionSnapshot
    ) -> PortfolioTargetProposal:
        if (
            snapshot.account.insolvent
            or snapshot.account.termination_reason is not None
        ):
            raise ValueError("cost-aware policy cannot run after termination")
        timestamp = np.datetime64(snapshot.timestamp_ns, "ns")
        if np.isnat(timestamp):
            raise ValueError("decision timestamp must not be NaT")

        effective_weights = compute_effective_posture_weights(snapshot, self._budgets)
        actions: list[int] = []
        for i, (strategy, budget) in enumerate(
            zip(self._strategies, self._budgets, strict=True)
        ):
            w_eff = effective_weights[i]
            current = (
                PositionIntent.LONG
                if w_eff > 1e-6
                else PositionIntent.SHORT
                if w_eff < -1e-6
                else PositionIntent.FLAT
            )
            obs = StrategyObservation(
                index=snapshot.index,
                timestamp=timestamp,
                symbol=self._symbols[i],
                features=np.asarray(snapshot.features[i], dtype=np.float64),
                feature_available=np.asarray(
                    snapshot.feature_available[i], dtype=np.bool_
                ),
                feature_staleness=np.asarray(
                    snapshot.feature_staleness[i], dtype=np.float32
                ),
                global_features=np.asarray(snapshot.global_features, dtype=np.float64),
                global_feature_available=np.asarray(
                    snapshot.global_feature_available, dtype=np.bool_
                ),
                current_intent=current,
                current_weight=w_eff,
                intent_gross_budget=budget,
            )
            intent = strategy.decide(obs)
            if intent == PositionIntent.LONG:
                actions.append(ACTION_LONG)
            elif intent == PositionIntent.SHORT:
                actions.append(ACTION_SHORT)
            else:
                actions.append(ACTION_FLAT)

        self._last_index = snapshot.index
        self._last_timestamp_ns = snapshot.timestamp_ns
        self._agent_stop_index = snapshot.agent_stop_index
        return preserve_posture_aware_proposal(tuple(actions), snapshot, self._budgets)


def _perturb_after(dataset: MarketDataset, last_unchanged_bar: int) -> MarketDataset:
    """Return a dataset copy whose market arrays differ strictly after last_unchanged_bar."""
    rng = np.random.default_rng(20261029)
    first = last_unchanged_bar + 1
    close = dataset.close.copy()
    n_rows = close.shape[0] - first
    if n_rows <= 0:
        return dataset
    close[first:] *= np.exp(rng.normal(0.0, 0.05, size=(n_rows, close.shape[1])))
    open_price = dataset.open.copy()
    open_price[first:] = np.vstack((close[first - 1], close[first:-1]))
    funding = dataset.funding_rate.copy()
    funding[first:] = rng.normal(0.001, 0.002, size=funding[first:].shape)
    features = dataset.features.copy()
    features[first:] += rng.normal(0.0, 0.1, size=features[first:].shape).astype(
        features.dtype
    )
    return replace(
        dataset,
        close=close,
        open=open_price,
        high=np.maximum(open_price, close) * 1.01,
        low=np.minimum(open_price, close) * 0.99,
        volume=np.where(
            np.arange(close.shape[0])[:, None] >= first, 1e11, dataset.volume
        ),
        funding_rate=funding,
        features=features,
        identity_payload_json=None,
    )


def _format_markdown_summary(
    protocol: DecisionValueDevStudyProtocol,
    result: DecisionValueDevStudyResult,
) -> str:
    """Generate human-readable Markdown summary of the study results."""
    lines = [
        "# Decision Value Development Study Report (#1025)",
        "",
        f"- **Verdict**: `{result.verdict}`",
        f"- **Reason**: {result.verdict_reason}",
        f"- **Dataset ID**: `{result.dataset_id}`",
        f"- **Artifact Digest**: `{result.actual_artifact_digest}`",
        f"- **Protocol Digest**: `{result.protocol_digest}`",
        f"- **Total Wall Clock Time**: {result.total_wall_clock_seconds:.2f} seconds",
        f"- **Prior Usage Disclosure**: {protocol.prior_usage_disclosure}",
        "",
        "## Aggregate Net Returns by Arm and Scenario",
        "",
        "| Arm | Base | Cost x2 | Latency 1 bar | Latency 2 bars |",
        "|---|---|---|---|---|",
    ]
    for arm in (
        "cash",
        "fixed_long",
        "cost_aware_ridge",
        "linear_value",
        "conservative_linear_value",
        "shallow_value",
    ):
        arm_metrics = result.aggregate_metrics.get(arm, {})
        base_r = arm_metrics.get("base", 0.0)
        c2_r = arm_metrics.get("cost_x2", 0.0)
        lat_r = arm_metrics.get("latency_1bar", 0.0)
        lat2_r = arm_metrics.get("latency_2bar", 0.0)
        lines.append(
            f"| `{arm}` | {base_r:+.4%} | {c2_r:+.4%} | {lat_r:+.4%} | {lat2_r:+.4%} |"
        )
    lines.extend(
        [
            "",
            "## Per-Block Results",
            "",
        ]
    )
    for b in result.blocks:
        lines.extend(
            [
                f"### {b.block_id}",
                f"- Training rows: {b.training_rows}, Overlap groups: {b.overlap_groups}, Paired replays: {b.paired_replay_runs}, Wall clock: {b.wall_clock_seconds:.2f}s",
                f"- Causality probe: {'PASSED' if b.causality_probe_passed else 'FAILED'}, Ledger reconciliation: {'PASSED' if b.ledger_evidence_reconciled else 'FAILED'}",
                "",
                "| Arm | Scenario | Final Value | Net Return | Total Cost | Max Drawdown |",
                "|---|---|---|---|---|---|",
            ]
        )
        for arm_res in b.arms:
            lines.append(
                f"| `{arm_res.arm}` | `{arm_res.scenario}` | {arm_res.final_portfolio_value:.2f} | {arm_res.net_return:+.4%} | {arm_res.total_cost:.2f} | {arm_res.max_drawdown:.4%} |"
            )
        lines.append("")
    return "\n".join(lines)


def run_decision_value_dev_study(
    dataset: MarketDataset,
    protocol: DecisionValueDevStudyProtocol,
    *,
    actual_artifact_digest: str | None = None,
    output_dir: str | Path | None = None,
) -> DecisionValueDevStudyResult:
    """Execute the bounded real-development study under strict contracts."""
    study_start_time = time.perf_counter()

    # Preflight verification (G0–G2)
    if not isinstance(dataset, MarketDataset):
        raise TypeError("dataset must be a MarketDataset")
    if not isinstance(protocol, DecisionValueDevStudyProtocol):
        raise TypeError("protocol must be a DecisionValueDevStudyProtocol")
    if dataset.dataset_id != protocol.dataset_id:
        raise ValueError(
            f"dataset dataset_id {dataset.dataset_id} differs from protocol {protocol.dataset_id}"
        )
    resolved_artifact_digest = (
        actual_artifact_digest
        if actual_artifact_digest is not None
        else protocol.expected_artifact_digest
    )
    if resolved_artifact_digest != protocol.expected_artifact_digest:
        raise ValueError(
            f"actual_artifact_digest {resolved_artifact_digest} differs from expected {protocol.expected_artifact_digest}"
        )
    if tuple(dataset.symbols) != protocol.study_plan.symbols:
        raise ValueError("dataset symbols do not match study_plan symbols")
    for block in protocol.blocks:
        if block.eval_stop_index >= dataset.n_bars:
            raise ValueError(
                f"block {block.block_id} eval_stop_index {block.eval_stop_index} exceeds dataset size {dataset.n_bars}"
            )
        train_slice = dataset.feature_available[
            block.train_start_index : block.train_stop_index + 1
        ]
        if not np.all(train_slice[:, :, list(protocol.ridge_feature_indices)]):
            raise ValueError(
                f"features not fully available in block {block.block_id} training slice"
            )

    block_results: list[DecisionValueBlockResult] = []

    for block in protocol.blocks:
        block_t0 = time.perf_counter()
        fit_cutoff = dataset.timestamps[block.fit_cutoff_index]
        eval_start = dataset.timestamps[block.eval_start_index]

        # 1. Fit B0 (Ridge Forecast)
        b0_model = fit_ridge_forecast(
            dataset,
            feature_indices=protocol.ridge_feature_indices,
            fit_cutoff=fit_cutoff,
            horizon_hours=protocol.ridge_horizon_hours,
            alpha=protocol.ridge_alpha,
        )

        # 2. Collect paired action interventions for B1/B2
        ref_digest = content_digest(
            {
                "policy": "in_flight_preserving_hold_quantity_v1",
                "target_abs_weights": [float(w) for w in protocol.target_abs_weights],
            }
        )
        samples: list[SyntheticPairedLabel] = []
        intervention_indices = range(
            block.train_start_index, block.train_stop_index + 1, block.train_step
        )
        non_hold_candidates = protocol.study_plan.candidate_actions[1:]
        paired_replay_runs = 0

        for t in intervention_indices:
            for cand_action in non_hold_candidates:
                if all(a == ACTION_FLAT for a in cand_action):
                    continue
                pair = run_paired_portfolio_action(
                    dataset,
                    reference_policy_factory=lambda: _HoldQuantityPolicy(
                        protocol.target_abs_weights
                    ),
                    start_index=t,
                    stop_index=t + protocol.study_plan.horizon_bars,
                    intervention_index=t,
                    baseline_action=protocol.study_plan.candidate_actions[0],
                    candidate_action=cand_action,
                    target_abs_weights=protocol.target_abs_weights,
                    initial_capital=protocol.study_plan.initial_capital,
                    execution_cost=protocol.base_execution_cost,
                    risk=PreTradeRisk(protocol.risk_config),
                )
                paired_replay_runs += 2  # Baseline + Candidate
                label = build_synthetic_paired_label(
                    dataset,
                    pair,
                    baseline_action=protocol.study_plan.candidate_actions[0],
                    candidate_action=cand_action,
                    target_abs_weights=protocol.target_abs_weights,
                    reference_policy_digest=ref_digest,
                    expected_execution_policy_digest=(
                        protocol.base_execution_cost.execution_policy_digest
                    ),
                    observation_record_available_at=dataset.timestamps[t],
                    settlement_record_available_at=(
                        dataset.timestamps[t + protocol.study_plan.horizon_bars]
                    ),
                    fit_cutoff=fit_cutoff,
                    source_scope="real_development",
                )
                samples.append(label)

        paired_transitions = paired_replay_runs * protocol.study_plan.horizon_bars
        training_rows = len(samples)

        # 3. Build paired label batch
        batch = build_synthetic_paired_label_batch(
            tuple(samples),
            fit_cutoff=fit_cutoff,
            evaluation_start=eval_start,
            expected_reference_policy_digest=ref_digest,
        )
        overlap_groups = len(set(batch.group_ids))

        # 4. Fit B1 (Linear), B1c (Conservative Linear), and B2 (Shallow)
        b1_model = fit_synthetic_linear_decision_value(
            batch,
            candidate_actions=protocol.study_plan.candidate_actions,
            ridge_penalty=protocol.value_ridge_penalty,
        )
        train_switch_cost = (
            protocol.study_plan.fee_rate
            + protocol.study_plan.spread_rate
            + protocol.base_execution_cost.impact_rate
        )
        b1c_model = fit_conservative_linear_decision_value(
            batch,
            candidate_actions=protocol.study_plan.candidate_actions,
            ridge_penalty=protocol.value_ridge_penalty,
            uncertainty_penalty_weight=protocol.uncertainty_penalty_weight,
            one_way_switch_cost=train_switch_cost,
        )
        b2_model = fit_synthetic_shallow_decision_value(
            batch,
            candidate_actions=protocol.study_plan.candidate_actions,
            ridge_penalty=protocol.value_ridge_penalty,
        )

        b1_coef_sha256 = hashlib.sha256(
            canonical_json_bytes([list(row) for row in b1_model.coefficients])
        ).hexdigest()
        b1c_coef_sha256 = hashlib.sha256(
            canonical_json_bytes([list(row) for row in b1c_model.coefficients])
        ).hexdigest()
        b2_coef_sha256 = hashlib.sha256(
            canonical_json_bytes([list(row) for row in b2_model.coefficients])
        ).hexdigest()

        # 5. Causality perturbation probe (Bug D1)
        dataset_perturbed = _perturb_after(dataset, block.fit_cutoff_index)
        b0_perturbed = fit_ridge_forecast(
            dataset_perturbed,
            feature_indices=protocol.ridge_feature_indices,
            fit_cutoff=fit_cutoff,
            horizon_hours=protocol.ridge_horizon_hours,
            alpha=protocol.ridge_alpha,
        )
        b0_fit_ok = np.allclose(
            b0_model.coefficients, b0_perturbed.coefficients
        ) and math.isclose(b0_model.intercept, b0_perturbed.intercept)

        samples_pert: list[SyntheticPairedLabel] = []
        for t in intervention_indices:
            for cand_action in non_hold_candidates:
                if all(a == ACTION_FLAT for a in cand_action):
                    continue
                pair_pert = run_paired_portfolio_action(
                    dataset_perturbed,
                    reference_policy_factory=lambda: _HoldQuantityPolicy(
                        protocol.target_abs_weights
                    ),
                    start_index=t,
                    stop_index=t + protocol.study_plan.horizon_bars,
                    intervention_index=t,
                    baseline_action=protocol.study_plan.candidate_actions[0],
                    candidate_action=cand_action,
                    target_abs_weights=protocol.target_abs_weights,
                    initial_capital=protocol.study_plan.initial_capital,
                    execution_cost=protocol.base_execution_cost,
                    risk=PreTradeRisk(protocol.risk_config),
                )
                label_pert = build_synthetic_paired_label(
                    dataset_perturbed,
                    pair_pert,
                    baseline_action=protocol.study_plan.candidate_actions[0],
                    candidate_action=cand_action,
                    target_abs_weights=protocol.target_abs_weights,
                    reference_policy_digest=ref_digest,
                    expected_execution_policy_digest=(
                        protocol.base_execution_cost.execution_policy_digest
                    ),
                    observation_record_available_at=dataset_perturbed.timestamps[t],
                    settlement_record_available_at=(
                        dataset_perturbed.timestamps[
                            t + protocol.study_plan.horizon_bars
                        ]
                    ),
                    fit_cutoff=fit_cutoff,
                    source_scope="real_development",
                )
                samples_pert.append(label_pert)

        batch_pert = build_synthetic_paired_label_batch(
            tuple(samples_pert),
            fit_cutoff=fit_cutoff,
            evaluation_start=eval_start,
            expected_reference_policy_digest=ref_digest,
        )
        b1_pert = fit_synthetic_linear_decision_value(
            batch_pert,
            candidate_actions=protocol.study_plan.candidate_actions,
            ridge_penalty=protocol.value_ridge_penalty,
        )
        b1c_pert = fit_conservative_linear_decision_value(
            batch_pert,
            candidate_actions=protocol.study_plan.candidate_actions,
            ridge_penalty=protocol.value_ridge_penalty,
            uncertainty_penalty_weight=protocol.uncertainty_penalty_weight,
            one_way_switch_cost=train_switch_cost,
        )
        b2_pert = fit_synthetic_shallow_decision_value(
            batch_pert,
            candidate_actions=protocol.study_plan.candidate_actions,
            ridge_penalty=protocol.value_ridge_penalty,
        )

        b1_pert_sha256 = hashlib.sha256(
            canonical_json_bytes([list(row) for row in b1_pert.coefficients])
        ).hexdigest()
        b1c_pert_sha256 = hashlib.sha256(
            canonical_json_bytes([list(row) for row in b1c_pert.coefficients])
        ).hexdigest()
        b2_pert_sha256 = hashlib.sha256(
            canonical_json_bytes([list(row) for row in b2_pert.coefficients])
        ).hexdigest()

        b1_fit_ok = b1_coef_sha256 == b1_pert_sha256
        b1c_fit_ok = (
            b1c_coef_sha256 == b1c_pert_sha256
            and np.allclose(
                b1c_model.covariance_matrices,
                b1c_pert.covariance_matrices,
                rtol=0.0,
                atol=1e-12,
            )
        )
        b2_fit_ok = b2_coef_sha256 == b2_pert_sha256

        b1_runtime_ctx = LinearDecisionValueRuntimeContext(
            source_scope="real_development",
            feature_config_digest=dataset.feature_config_digest,
            normalization_digest=dataset.normalization_digest,
            execution_policy_digest=b1_model.execution_policy_digest,
            reference_policy_digest=b1_model.reference_policy_digest,
        )
        b1c_runtime_ctx = LinearDecisionValueRuntimeContext(
            source_scope="real_development",
            feature_config_digest=dataset.feature_config_digest,
            normalization_digest=dataset.normalization_digest,
            execution_policy_digest=b1c_model.execution_policy_digest,
            reference_policy_digest=b1c_model.reference_policy_digest,
        )
        b2_runtime_ctx = LinearDecisionValueRuntimeContext(
            source_scope="real_development",
            feature_config_digest=dataset.feature_config_digest,
            normalization_digest=dataset.normalization_digest,
            execution_policy_digest=b2_model.execution_policy_digest,
            reference_policy_digest=b2_model.reference_policy_digest,
        )

        session_clean = SharedPortfolioSession(
            dataset,
            start_index=block.eval_start_index,
            stop_index=block.eval_stop_index,
            initial_capital=protocol.study_plan.initial_capital,
            execution_cost=protocol.base_execution_cost,
            risk=PreTradeRisk(protocol.risk_config),
        )
        snap_clean = build_portfolio_decision_snapshot(
            dataset,
            index=block.eval_start_index,
            agent_stop_index=block.eval_stop_index,
            book=session_clean.book,
            active_orders=session_clean.active_orders,
            risk_config=protocol.risk_config,
        )

        pol_b0_probe = _CostAwareRidgePortfolioPolicy(
            dataset=dataset,
            model=b0_model,
            target_abs_weights=protocol.target_abs_weights,
            entry_threshold=protocol.ridge_entry_threshold,
            exit_threshold=protocol.ridge_exit_threshold,
            one_way_switch_cost=train_switch_cost,
        )
        pol_b0_probe_perturbed = _CostAwareRidgePortfolioPolicy(
            dataset=dataset_perturbed,
            model=b0_perturbed,
            target_abs_weights=protocol.target_abs_weights,
            entry_threshold=protocol.ridge_entry_threshold,
            exit_threshold=protocol.ridge_exit_threshold,
            one_way_switch_cost=train_switch_cost,
        )
        prop_b0_clean = pol_b0_probe.decide_portfolio(snap_clean)
        prop_b0_pert = pol_b0_probe_perturbed.decide_portfolio(snap_clean)
        b0_probe_ok = b0_fit_ok and (
            prop_b0_clean.target_weights == prop_b0_pert.target_weights
        )

        pol_b1_clean = LinearDecisionValuePolicy(
            b1_model,
            runtime_context=b1_runtime_ctx,
            allow_active_orders=True,
            eval_one_way_cost=train_switch_cost,
            train_one_way_cost=train_switch_cost,
        )
        pol_b1_pert_policy = LinearDecisionValuePolicy(
            b1_pert,
            runtime_context=b1_runtime_ctx,
            allow_active_orders=True,
            eval_one_way_cost=train_switch_cost,
            train_one_way_cost=train_switch_cost,
        )
        b1_probe_ok = b1_fit_ok and (
            pol_b1_clean.decide_portfolio(snap_clean).target_weights
            == pol_b1_pert_policy.decide_portfolio(snap_clean).target_weights
        )

        pol_b1c_clean = ConservativeLinearDecisionValuePolicy(
            b1c_model,
            runtime_context=b1c_runtime_ctx,
            allow_active_orders=True,
            eval_one_way_cost=train_switch_cost,
            train_one_way_cost=train_switch_cost,
        )
        pol_b1c_pert_policy = ConservativeLinearDecisionValuePolicy(
            b1c_pert,
            runtime_context=b1c_runtime_ctx,
            allow_active_orders=True,
            eval_one_way_cost=train_switch_cost,
            train_one_way_cost=train_switch_cost,
        )
        b1c_probe_ok = b1c_fit_ok and (
            pol_b1c_clean.decide_portfolio(snap_clean).target_weights
            == pol_b1c_pert_policy.decide_portfolio(snap_clean).target_weights
        )

        pol_b2_clean = ShallowDecisionValuePolicy(
            b2_model,
            runtime_context=b2_runtime_ctx,
            allow_active_orders=True,
            eval_one_way_cost=train_switch_cost,
            train_one_way_cost=train_switch_cost,
        )
        pol_b2_pert_policy = ShallowDecisionValuePolicy(
            b2_pert,
            runtime_context=b2_runtime_ctx,
            allow_active_orders=True,
            eval_one_way_cost=train_switch_cost,
            train_one_way_cost=train_switch_cost,
        )
        b2_probe_ok = b2_fit_ok and (
            pol_b2_clean.decide_portfolio(snap_clean).target_weights
            == pol_b2_pert_policy.decide_portfolio(snap_clean).target_weights
        )

        causality_probe_passed = bool(
            b0_probe_ok and b1_probe_ok and b1c_probe_ok and b2_probe_ok
        )

        # 6. Evaluate 6 arms across scenarios
        scenarios: list[tuple[str, ExecutionCostConfig, float]] = []
        for s_name, cost_mult, lat_bars in protocol.scenarios:
            s_cost = replace(
                protocol.base_execution_cost,
                fee_rate=protocol.base_execution_cost.fee_rate * cost_mult,
                spread_rate=protocol.base_execution_cost.spread_rate * cost_mult,
                impact_rate=protocol.base_execution_cost.impact_rate * cost_mult,
                order_latency_bars=lat_bars,
            )
            scen_switch_cost = s_cost.fee_rate + s_cost.spread_rate + s_cost.impact_rate
            scenarios.append((s_name, s_cost, scen_switch_cost))

        arms_metrics: list[DecisionValueArmMetrics] = []
        ledger_all_reconciled = True
        eval_replay_runs = 0

        for scen_name, scen_cost, scen_switch_cost in scenarios:
            pol_b1_eval = LinearDecisionValuePolicy(
                b1_model,
                runtime_context=b1_runtime_ctx,
                allow_active_orders=True,
                eval_one_way_cost=scen_switch_cost,
                train_one_way_cost=train_switch_cost,
            )
            pol_b1c_eval = ConservativeLinearDecisionValuePolicy(
                b1c_model,
                runtime_context=b1c_runtime_ctx,
                allow_active_orders=True,
                eval_one_way_cost=scen_switch_cost,
                train_one_way_cost=train_switch_cost,
            )
            pol_b2_eval = ShallowDecisionValuePolicy(
                b2_model,
                runtime_context=b2_runtime_ctx,
                allow_active_orders=True,
                eval_one_way_cost=scen_switch_cost,
                train_one_way_cost=train_switch_cost,
            )

            arm_policies: tuple[tuple[str, PortfolioPolicy], ...] = (
                ("cash", CashPortfolioPolicy(symbols=dataset.symbols)),
                (
                    "fixed_long",
                    _FixedEntryPolicy(
                        action=(ACTION_LONG,) * len(dataset.symbols),
                        target_abs_weights=protocol.target_abs_weights,
                    ),
                ),
                (
                    "cost_aware_ridge",
                    _CostAwareRidgePortfolioPolicy(
                        dataset=dataset,
                        model=b0_model,
                        target_abs_weights=protocol.target_abs_weights,
                        entry_threshold=protocol.ridge_entry_threshold,
                        exit_threshold=protocol.ridge_exit_threshold,
                        one_way_switch_cost=scen_switch_cost,
                    ),
                ),
                ("linear_value", pol_b1_eval),
                ("conservative_linear_value", pol_b1c_eval),
                ("shallow_value", pol_b2_eval),
            )

            for arm_name, policy in arm_policies:
                eval_replay_runs += 1
                replay_res = run_shared_portfolio_replay(
                    dataset,
                    policy,
                    start_index=block.eval_start_index,
                    stop_index=block.eval_stop_index,
                    initial_capital=protocol.study_plan.initial_capital,
                    execution_cost=scen_cost,
                    risk=PreTradeRisk(protocol.risk_config),
                    settle_terminal_position=True,
                    capture_ledger_evidence=True,
                )

                expected_intervals = block.eval_stop_index - block.eval_start_index
                if not _verify_ledger_reconciliation(
                    replay_res, expected_intervals=expected_intervals
                ):
                    ledger_all_reconciled = False

                terminal_quantities = tuple(
                    str(q) for q in replay_res.book.exact_quantities
                )
                terminal_flat = all(
                    q in ("0", "0/1") or q == Fraction(0, 1)
                    for q in replay_res.book.exact_quantities
                )
                final_val = float(replay_res.book.portfolio_value)
                net_ret = (final_val / protocol.study_plan.initial_capital) - 1.0

                arms_metrics.append(
                    DecisionValueArmMetrics(
                        arm=arm_name,
                        scenario=scen_name,
                        initial_capital=protocol.study_plan.initial_capital,
                        final_portfolio_value=final_val,
                        net_return=net_ret,
                        total_cost=float(replay_res.book.total_cost),
                        funding_pnl=float(replay_res.book.funding_pnl),
                        max_drawdown=float(replay_res.book.max_drawdown),
                        n_decisions=len(replay_res.decisions),
                        terminal_quantities=terminal_quantities,
                        terminal_flat=terminal_flat,
                    )
                )

        block_wall_clock = time.perf_counter() - block_t0
        block_results.append(
            DecisionValueBlockResult(
                block_id=block.block_id,
                paired_replay_runs=paired_replay_runs,
                paired_transitions=paired_transitions,
                training_rows=training_rows,
                overlap_groups=overlap_groups,
                eval_replay_runs=eval_replay_runs,
                wall_clock_seconds=block_wall_clock,
                b0_model_intercept=float(b0_model.intercept),
                b1_model_coefficients_sha256=b1_coef_sha256,
                b1c_model_coefficients_sha256=b1c_coef_sha256,
                b2_model_coefficients_sha256=b2_coef_sha256,
                causality_probe_passed=causality_probe_passed,
                ledger_evidence_reconciled=ledger_all_reconciled,
                arms=tuple(arms_metrics),
            )
        )

    total_wall_clock = time.perf_counter() - study_start_time

    # Compute aggregate metrics across blocks: arm -> scenario -> net_return
    arm_names = (
        "cash",
        "fixed_long",
        "cost_aware_ridge",
        "linear_value",
        "conservative_linear_value",
        "shallow_value",
    )
    scenario_names = ("base", "cost_x2", "latency_1bar", "latency_2bar")
    aggregate_metrics: dict[str, dict[str, float]] = {arm: {} for arm in arm_names}
    aggregate_max_dd: dict[str, dict[str, float]] = {arm: {} for arm in arm_names}

    for arm in arm_names:
        for scen in scenario_names:
            returns = [
                m.net_return
                for b in block_results
                for m in b.arms
                if m.arm == arm and m.scenario == scen
            ]
            drawdowns = [
                m.max_drawdown
                for b in block_results
                for m in b.arms
                if m.arm == arm and m.scenario == scen
            ]
            compounded = math.prod(1.0 + r for r in returns) - 1.0
            aggregate_metrics[arm][scen] = compounded
            aggregate_max_dd[arm][scen] = max(drawdowns) if drawdowns else 0.0

    # Verdict evaluation
    all_causality = all(b.causality_probe_passed for b in block_results)
    all_ledger = all(b.ledger_evidence_reconciled for b in block_results)
    any_dd_exceeded = any(m.max_drawdown >= 0.20 for b in block_results for m in b.arms)

    if not all_causality:
        verdict = "INVALID / BLOCKED"
        verdict_reason = "Causality perturbation probe failed"
    elif not all_ledger:
        verdict = "INVALID / BLOCKED"
        verdict_reason = "Ledger evidence reconciliation failed"
    elif any_dd_exceeded:
        verdict = "INVALID / BLOCKED"
        verdict_reason = "Observed maximum drawdown exceeded 20% limit"
    else:
        # Check if B1, B1c, or B2 outperforms all baselines (cash, fixed_long, B0) across all scenarios
        def outperforms_all_baselines(candidate_arm: str) -> bool:
            for scen in scenario_names:
                c_ret = aggregate_metrics[candidate_arm][scen]
                for base_arm in ("cash", "fixed_long", "cost_aware_ridge"):
                    if c_ret <= aggregate_metrics[base_arm][scen]:
                        return False
            return True

        def improves_over_ridge(candidate_arm: str) -> bool:
            return any(
                aggregate_metrics[candidate_arm][scen]
                > aggregate_metrics["cost_aware_ridge"][scen]
                for scen in scenario_names
            )

        b1_winner = outperforms_all_baselines("linear_value")
        b1c_winner = outperforms_all_baselines("conservative_linear_value")
        b2_winner = outperforms_all_baselines("shallow_value")

        if b1_winner or b1c_winner or b2_winner:
            verdict = "DEVELOPMENT_IMPROVED"
            winners = []
            if b1_winner:
                winners.append("B1 (linear_value)")
            if b1c_winner:
                winners.append("B1c (conservative_linear_value)")
            if b2_winner:
                winners.append("B2 (shallow_value)")
            winner_str = " and ".join(winners)
            verdict_reason = f"{winner_str} outperformed cash, fixed_long, and cost_aware_ridge across all scenarios"
        elif (
            improves_over_ridge("linear_value")
            or improves_over_ridge("conservative_linear_value")
            or improves_over_ridge("shallow_value")
        ):
            verdict = "INCONCLUSIVE"
            verdict_reason = "Decision-value head improved over cost_aware_ridge in some scenarios but did not dominate all controls"
        else:
            verdict = "NO_IMPROVEMENT"
            verdict_reason = "Neither linear_value, conservative_linear_value, nor shallow_value outperformed baselines across evaluation scenarios"

    result = DecisionValueDevStudyResult(
        schema_version="decision_value_dev_study_result_v1",
        protocol_digest=protocol.protocol_digest,
        dataset_id=dataset.dataset_id,
        actual_artifact_digest=resolved_artifact_digest,
        verdict=verdict,
        verdict_reason=verdict_reason,
        blocks=tuple(block_results),
        aggregate_metrics=aggregate_metrics,
        total_wall_clock_seconds=total_wall_clock,
    )

    # Persistence (write-once)
    if output_dir is not None:
        out_path = Path(output_dir)
        if out_path.exists():
            raise FileExistsError(f"output_dir already exists: {out_path}")
        out_path.mkdir(parents=True, exist_ok=False)
        (out_path / "protocol.json").write_bytes(
            canonical_json_bytes(protocol.to_mapping())
        )
        (out_path / "result.json").write_bytes(
            canonical_json_bytes(result.to_mapping())
        )
        (out_path / "summary.md").write_text(
            _format_markdown_summary(protocol, result),
            encoding="utf-8",
        )

    return result


__all__ = [
    "DecisionValueArmMetrics",
    "DecisionValueBlockResult",
    "DecisionValueBlockSpec",
    "DecisionValueDevStudyProtocol",
    "DecisionValueDevStudyResult",
    "compute_real_scope_authorization_digest",
    "default_2024q1_decision_value_dev_protocol",
    "run_decision_value_dev_study",
]
