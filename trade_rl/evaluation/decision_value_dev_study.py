"""Historical TD8 contracts and current synthetic-only software diagnostics.

The original real-development TD8 protocol remains readable but cannot run on
this corrected source. The current v3 runner requires a published synthetic
artifact and preserves the old research outcomes without reclassification.
Software diagnostics enforce distinct causal teacher, temporary calibration,
and final block feature vintages; an incomplete paired label stops model fitting
and is retained as an invalid attempt rather than a zero-valued label.
"""

from __future__ import annotations

import hashlib
import math
import os
import shutil
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path

import numpy as np

from trade_rl._validation import require_sha256
from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.artifacts.publication import (
    load_published_market_dataset_artifact,
)
from trade_rl.data.identity import parse_identity_json
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.decision_value_feature_transform import (
    apply_feature_transform,
    covariate_shift_report,
    fit_train_feature_transform,
)
from trade_rl.evaluation.decision_value_fit import (
    fit_conservative_linear_decision_value,
    fit_synthetic_linear_decision_value,
    fit_synthetic_shallow_decision_value,
)
from trade_rl.evaluation.decision_value_ledger_oracle import (
    verify_paired_economic_ledger,
)
from trade_rl.evaluation.decision_value_margin import (
    MarginSelection,
    select_margin_multiplier,
)
from trade_rl.evaluation.decision_value_plan import DecisionValueStudyPlan
from trade_rl.evaluation.decision_value_reachable_labels import (
    IneligiblePairOutcome,
    ReachableLabelCollection,
    ReferencePolicyFactory,
    collect_reachable_state_labels,
)
from trade_rl.evaluation.paired_portfolio_label import (
    ReferenceVintage,
    build_synthetic_paired_label_batch,
)
from trade_rl.evaluation.replay import (
    SharedCashPortfolioReplayResult,
    run_shared_portfolio_replay,
)
from trade_rl.evaluation.runs.provenance import build_candidate_run_provenance
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


DEFAULT_MARGIN_GRID: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0)
LEARNER_ARMS: tuple[str, ...] = (
    "linear_value",
    "conservative_linear_value",
    "shallow_value",
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
    dataset: MarketDataset,
    expected_intervals: int,
    expected_start_index: int,
    initial_capital: float,
    execution_cost: ExecutionCostConfig | None = None,
) -> bool:
    """Validate source-backed interval fills/cash/marked equity, not just totals.

    Actual fee/funding/borrow/dividend/interest amounts still originate from the
    observer rather than independently priced financial contracts.
    """
    return verify_paired_economic_ledger(
        replay_res,
        dataset,
        expected_intervals=expected_intervals,
        expected_start_index=expected_start_index,
        initial_capital=initial_capital,
        execution_cost=execution_cost,
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
    label_prefix_bars: int = 48,
    reference_min_fit_bars: int = 96,
    margin_multiplier_grid: tuple[float, ...] = DEFAULT_MARGIN_GRID,
    margin_calibration_fraction: float = 0.25,
    feature_clip: float = 3.0,
    unexcited_prior_std: float = 1.0,
) -> str:
    """Canonical lowercase SHA-256 digest of the pre-authorization specification."""
    payload = {
        "schema_version": "decision_value_real_scope_authorization_v2",
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
        "label_prefix_bars": int(label_prefix_bars),
        "reference_min_fit_bars": int(reference_min_fit_bars),
        "margin_multiplier_grid": [float(k) for k in margin_multiplier_grid],
        "margin_calibration_fraction": float(margin_calibration_fraction),
        "feature_clip": float(feature_clip),
        "unexcited_prior_std": float(unexcited_prior_std),
    }
    return content_digest(payload)


@dataclass(frozen=True, slots=True)
class DecisionValueDevStudyProtocol:
    """Frozen TD8 parameters; real v2 is archival and synthetic v3 is runnable."""

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
    label_prefix_bars: int = 48
    reference_min_fit_bars: int = 96
    margin_multiplier_grid: tuple[float, ...] = DEFAULT_MARGIN_GRID
    margin_calibration_fraction: float = 0.25
    feature_clip: float = 3.0
    unexcited_prior_std: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.study_plan, DecisionValueStudyPlan):
            raise TypeError("study_plan must be DecisionValueStudyPlan")
        if self.study_plan.data_scope not in ("real_development", "synthetic_only"):
            raise ValueError(
                "study_plan data_scope must be real_development or synthetic_only"
            )
        if self.study_plan.data_scope == "synthetic_only":
            # The result table uses a fixed arm/scenario roster; do not execute
            # hidden arms under a smaller protocol declaration.
            if self.study_plan.learner_roster != (
                "linear",
                "conservative_linear",
                "shallow",
            ):
                raise ValueError(
                    "TD8 synthetic learner roster must name all three heads"
                )
            if self.scenarios != DEFAULT_SCENARIOS:
                raise ValueError("TD8 synthetic scenario roster must match four cases")
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
            if block.train_step < self.study_plan.horizon_bars:
                raise ValueError(
                    "train_step must be at least horizon_bars so branch labels do not overlap"
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

        if (
            isinstance(self.label_prefix_bars, bool)
            or not isinstance(self.label_prefix_bars, int)
            or self.label_prefix_bars < 0
        ):
            raise ValueError("label_prefix_bars must be a non-negative integer")
        if (
            isinstance(self.reference_min_fit_bars, bool)
            or not isinstance(self.reference_min_fit_bars, int)
            or self.reference_min_fit_bars < 0
        ):
            raise ValueError("reference_min_fit_bars must be a non-negative integer")
        grid = self.margin_multiplier_grid
        if type(grid) is not tuple or not grid:
            raise ValueError("margin_multiplier_grid must be a non-empty tuple")
        for item in grid:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise ValueError("margin_multiplier_grid must contain real numbers")
            if not math.isfinite(item) or item < 0.0:
                raise ValueError(
                    "margin_multiplier_grid must be finite and non-negative"
                )
        if any(b <= a for a, b in zip(grid, grid[1:], strict=False)):
            raise ValueError("margin_multiplier_grid must be strictly increasing")
        fraction = _finite(
            self.margin_calibration_fraction, field="margin_calibration_fraction"
        )
        if not 0.0 < fraction < 1.0:
            raise ValueError("margin_calibration_fraction must lie inside (0, 1)")
        if _finite(self.feature_clip, field="feature_clip") <= 0.0:
            raise ValueError("feature_clip must be positive and finite")
        if _finite(self.unexcited_prior_std, field="unexcited_prior_std") <= 0.0:
            raise ValueError("unexcited_prior_std must be positive and finite")

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
            label_prefix_bars=self.label_prefix_bars,
            reference_min_fit_bars=self.reference_min_fit_bars,
            margin_multiplier_grid=self.margin_multiplier_grid,
            margin_calibration_fraction=self.margin_calibration_fraction,
            feature_clip=self.feature_clip,
            unexcited_prior_std=self.unexcited_prior_std,
        )
        if self.study_plan.data_scope == "real_development":
            if self.study_plan.real_scope_authorization_digest != expected_digest:
                raise ValueError(
                    "study_plan real_scope_authorization_digest does not match pre-authorization specification"
                )
        elif self.study_plan.real_scope_authorization_digest is not None:
            raise ValueError("synthetic_only study cannot claim real authorization")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": (
                "decision_value_dev_study_protocol_v3_software"
                if self.study_plan.data_scope == "synthetic_only"
                else "decision_value_dev_study_protocol_v2"
            ),
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
            "label_prefix_bars": self.label_prefix_bars,
            "reference_min_fit_bars": self.reference_min_fit_bars,
            "margin_multiplier_grid": [float(k) for k in self.margin_multiplier_grid],
            "margin_calibration_fraction": self.margin_calibration_fraction,
            "feature_clip": self.feature_clip,
            "unexcited_prior_std": self.unexcited_prior_std,
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
            train_step=24,
            fit_cutoff_index=984,
            eval_start_index=985,
            eval_stop_index=1344,
        ),
        DecisionValueBlockSpec(
            block_id="block_2",
            train_start_index=24,
            train_stop_index=1338,
            train_step=24,
            fit_cutoff_index=1368,
            eval_start_index=1369,
            eval_stop_index=1728,
        ),
        DecisionValueBlockSpec(
            block_id="block_3",
            train_start_index=24,
            train_stop_index=1722,
            train_step=24,
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
        label_prefix_bars=48,
        reference_min_fit_bars=96,
        margin_multiplier_grid=DEFAULT_MARGIN_GRID,
        margin_calibration_fraction=0.25,
        feature_clip=3.0,
        unexcited_prior_std=1.0,
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
        label_prefix_bars=48,
        reference_min_fit_bars=96,
        margin_multiplier_grid=DEFAULT_MARGIN_GRID,
        margin_calibration_fraction=0.25,
        feature_clip=3.0,
        unexcited_prior_std=1.0,
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
    turnover_total: float = 0.0

    def to_mapping(self) -> dict[str, object]:
        return {
            "arm": self.arm,
            "scenario": self.scenario,
            "initial_capital": self.initial_capital,
            "final_portfolio_value": self.final_portfolio_value,
            "net_return": self.net_return,
            "total_cost": self.total_cost,
            "turnover_total": self.turnover_total,
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
    chosen_margin_multipliers: dict[str, float]
    margin_selections: dict[str, dict[str, object]]
    b1c_residual_dof: tuple[float, ...]
    distinct_decision_times: int
    ineligible_pairs: int
    nonflat_branch_states: int
    skipped_branches_without_reference: int
    feature_transform_digest: str
    covariate_shift: tuple[dict[str, object], ...]
    ineligible_pair_outcomes: tuple[dict[str, object], ...] = ()
    eligible_pair_outcomes: tuple[dict[str, object], ...] = ()

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
            "chosen_margin_multipliers": dict(self.chosen_margin_multipliers),
            "margin_selections": {
                k: dict(v) for k, v in self.margin_selections.items()
            },
            "b1c_residual_dof": list(self.b1c_residual_dof),
            "distinct_decision_times": self.distinct_decision_times,
            "ineligible_pairs": self.ineligible_pairs,
            "ineligible_pair_outcomes": [
                dict(r) for r in self.ineligible_pair_outcomes
            ],
            "eligible_pair_outcomes": [dict(r) for r in self.eligible_pair_outcomes],
            "nonflat_branch_states": self.nonflat_branch_states,
            "skipped_branches_without_reference": (
                self.skipped_branches_without_reference
            ),
            "feature_transform_digest": self.feature_transform_digest,
            "covariate_shift": [dict(r) for r in self.covariate_shift],
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

    __slots__ = ("_action", "_target_abs_weights", "_entered", "_proposal_version")

    def __init__(
        self,
        *,
        action: tuple[int, ...],
        target_abs_weights: tuple[float, ...],
        proposal_version: str = "legacy_v1",
    ) -> None:
        if proposal_version not in ("legacy_v1", "td8_v2"):
            raise ValueError("unknown fixed-entry proposal version")
        self._proposal_version = proposal_version
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
                version=self._proposal_version,
            )
        return preserve_posture_aware_proposal(
            (ACTION_HOLD,) * len(snapshot.symbols),
            snapshot,
            self._target_abs_weights,
            all_scores_non_positive=False,
            version=self._proposal_version,
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
        "_proposal_version",
        "_reference_feature_dataset",
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
        proposal_version: str = "legacy_v1",
        reference_feature_dataset: MarketDataset | None = None,
    ) -> None:
        if proposal_version not in ("legacy_v1", "td8_v2"):
            raise ValueError("unknown cost-aware proposal version")
        self._proposal_version = proposal_version
        if reference_feature_dataset is not None and (
            reference_feature_dataset.dataset_id != dataset.dataset_id
            or reference_feature_dataset.symbols != dataset.symbols
            or reference_feature_dataset.feature_names != dataset.feature_names
            or reference_feature_dataset.global_feature_names
            != dataset.global_feature_names
            or not np.array_equal(
                reference_feature_dataset.timestamps, dataset.timestamps
            )
        ):
            raise ValueError("reference features differ from the native market roster")
        self._reference_feature_dataset = reference_feature_dataset
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

        # TD8 teachers score on a frozen pre-prefix feature vintage while the
        # native account/intervention remains on the shared block dataset.
        scoring_snapshot = snapshot
        source = self._reference_feature_dataset
        if source is not None:
            if (
                not 0 <= snapshot.index < source.n_bars
                or source.timestamps[snapshot.index] != timestamp
                or source.symbols != snapshot.symbols
            ):
                raise ValueError("reference feature vintage does not match the clock")
            scoring_snapshot = replace(
                snapshot,
                features=tuple(
                    tuple(float(v) for v in row)
                    for row in source.features[snapshot.index]
                ),
                feature_available=tuple(
                    tuple(bool(v) for v in row)
                    for row in source.feature_available[snapshot.index]
                ),
                feature_staleness=tuple(
                    tuple(float(v) for v in row)
                    for row in source.resolved_array("feature_staleness")[
                        snapshot.index
                    ]
                ),
                global_features=tuple(
                    float(v) for v in source.global_features[snapshot.index]
                ),
                global_feature_available=tuple(
                    bool(v)
                    for v in source.resolved_array("global_feature_available")[
                        snapshot.index
                    ]
                ),
            )

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
                features=np.asarray(scoring_snapshot.features[i], dtype=np.float64),
                feature_available=np.asarray(
                    scoring_snapshot.feature_available[i], dtype=np.bool_
                ),
                feature_staleness=np.asarray(
                    scoring_snapshot.feature_staleness[i], dtype=np.float32
                ),
                global_features=np.asarray(
                    scoring_snapshot.global_features, dtype=np.float64
                ),
                global_feature_available=np.asarray(
                    scoring_snapshot.global_feature_available, dtype=np.bool_
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
        return preserve_posture_aware_proposal(
            tuple(actions), snapshot, self._budgets, version=self._proposal_version
        )


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


def causal_b0_reference_provider(
    dataset: MarketDataset,
    *,
    feature_indices: tuple[int, ...],
    horizon_hours: int,
    alpha: float,
    entry_threshold: float,
    exit_threshold: float,
    target_abs_weights: tuple[float, ...],
    one_way_switch_cost: float,
    min_fit_bars: int,
    first_fit_index: int,
    raw_dataset: MarketDataset | None = None,
    causal_feature_clip: float | None = None,
) -> Callable[[int, int], Callable[[], PortfolioPolicy] | None]:
    """Per-branch B0 reference: the B0 recipe refitted causally before the prefix.

    ``RidgeForecastStrategy`` rejects decisions earlier than its ``fit_cutoff``, so
    the block-level B0 model (fitted at the block fit cutoff) cannot legally act
    inside the train window. For a branch whose prefix starts at ``start`` this
    provider refits the identical recipe (same features, horizon, alpha, thresholds)
    with ``fit_cutoff = timestamps[start]``. A raw source and clip fit a
    per-prefix normalizer and serve that same vintage to the teacher. It returns ``None`` when fewer than
    ``min_fit_bars`` bars precede ``start`` in the train window, because no
    informed causal reference exists there.
    """
    if (raw_dataset is None) != (causal_feature_clip is None):
        raise ValueError("reference vintage requires raw_dataset and clip together")
    if raw_dataset is not None and (
        raw_dataset.dataset_id != dataset.dataset_id
        or raw_dataset.symbols != dataset.symbols
        or raw_dataset.feature_names != dataset.feature_names
        or raw_dataset.global_feature_names != dataset.global_feature_names
        or not np.array_equal(raw_dataset.timestamps, dataset.timestamps)
    ):
        raise ValueError("reference raw source differs from replay source")
    cache: dict[int, tuple[RidgeForecastModel, MarketDataset]] = {}
    # Bind the actual source and runtime that produced each reference teacher.
    # Compute this once per provider, not for every paired candidate.
    source_runtime_digest: str | None = None
    if raw_dataset is not None:
        provenance = build_candidate_run_provenance()
        source_runtime_digest = content_digest(
            {
                "implementation_digest": provenance["implementation_digest"],
                "runtime_environment_digest": provenance["runtime_environment_digest"],
                "dataset_id": raw_dataset.dataset_id,
                "source_normalization_digest": raw_dataset.normalization_digest,
                "feature_config_digest": raw_dataset.feature_config_digest,
            }
        )

    def provider(_t: int, start: int) -> Callable[[], PortfolioPolicy] | None:
        minimum = max(min_fit_bars, 2) if raw_dataset is not None else min_fit_bars
        if start - first_fit_index < minimum:
            return None
        fitted = cache.get(start)
        if fitted is None:
            vintage_view = dataset
            if raw_dataset is not None:
                assert causal_feature_clip is not None
                transform = fit_train_feature_transform(
                    raw_dataset,
                    train_start_index=first_fit_index,
                    train_stop_index=start - 1,
                    clip=causal_feature_clip,
                    as_of_index=start,
                )
                vintage_view = apply_feature_transform(raw_dataset, transform)
            model = fit_ridge_forecast(
                vintage_view,
                feature_indices=feature_indices,
                fit_cutoff=vintage_view.timestamps[start],
                horizon_hours=horizon_hours,
                alpha=alpha,
            )
            fitted = (model, vintage_view)
            cache[start] = fitted
        model, vintage_view = fitted

        def factory() -> PortfolioPolicy:
            return _CostAwareRidgePortfolioPolicy(
                dataset=dataset,
                model=model,
                target_abs_weights=target_abs_weights,
                entry_threshold=entry_threshold,
                exit_threshold=exit_threshold,
                one_way_switch_cost=one_way_switch_cost,
                proposal_version="td8_v2",
                reference_feature_dataset=(
                    vintage_view if raw_dataset is not None else None
                ),
            )

        if raw_dataset is None:
            return factory  # legacy reference contract is unchanged
        assert source_runtime_digest is not None
        model_digest = content_digest(
            {
                "schema": "decision_value_b0_reference_model_v1",
                "fit_cutoff_ns": int(model.fit_cutoff.astype(np.int64)),
                "feature_indices": list(model.feature_indices),
                "feature_mean": model.feature_mean.tolist(),
                "feature_scale": model.feature_scale.tolist(),
                "coefficients": model.coefficients.tolist(),
                "intercept": model.intercept,
                "horizon_hours": model.horizon_hours,
                "alpha": model.alpha,
                "n_samples": model.n_samples,
                "entry_threshold": entry_threshold,
                "exit_threshold": exit_threshold,
                "one_way_switch_cost": one_way_switch_cost,
                "target_abs_weights": list(target_abs_weights),
            }
        )
        vintage = ReferenceVintage(
            source_dataset_id=dataset.dataset_id,
            prefix_start_index=start,
            fit_cutoff_ns=int(model.fit_cutoff.astype(np.int64)),
            normalization_digest=vintage_view.normalization_digest,
            model_digest=model_digest,
            source_runtime_digest=source_runtime_digest,
        )
        return ReferencePolicyFactory(factory, vintage)

    return provider


def _runtime_context(
    model: object, dataset: MarketDataset
) -> LinearDecisionValueRuntimeContext:
    return LinearDecisionValueRuntimeContext(
        source_scope=model.source_scope,  # type: ignore[attr-defined]
        feature_config_digest=dataset.feature_config_digest,
        normalization_digest=dataset.normalization_digest,
        execution_policy_digest=model.execution_policy_digest,  # type: ignore[attr-defined]
        reference_policy_digest=model.reference_policy_digest,  # type: ignore[attr-defined]
    )


def _build_value_policy(
    arm: str,
    model: object,
    dataset: MarketDataset,
    *,
    margin_multiplier: float,
    eval_one_way_cost: float,
    train_one_way_cost: float,
) -> PortfolioPolicy:
    """HOLD-relative value policy for one learner arm with the given margin."""
    context = _runtime_context(model, dataset)
    common = {
        "runtime_context": context,
        "allow_active_orders": True,
        "eval_one_way_cost": eval_one_way_cost,
        "train_one_way_cost": train_one_way_cost,
        "hold_relative": True,
        "decision_margin_multiplier": margin_multiplier,
    }
    if arm == "linear_value":
        return LinearDecisionValuePolicy(model, **common)  # type: ignore[arg-type]
    if arm == "conservative_linear_value":
        return ConservativeLinearDecisionValuePolicy(model, **common)  # type: ignore[arg-type]
    if arm == "shallow_value":
        return ShallowDecisionValuePolicy(model, **common)  # type: ignore[arg-type]
    raise ValueError(f"unknown learner arm {arm}")


def _coefficient_sha256(model: object) -> str:
    return hashlib.sha256(
        canonical_json_bytes([list(row) for row in model.coefficients])  # type: ignore[attr-defined]
    ).hexdigest()


def _fit_all_learners(
    batch: object,
    protocol: DecisionValueDevStudyProtocol,
    train_switch_cost: float,
    dof_out: list[float] | None = None,
) -> dict[str, object]:
    actions = protocol.study_plan.candidate_actions
    return {
        "linear_value": fit_synthetic_linear_decision_value(
            batch,  # type: ignore[arg-type]
            candidate_actions=actions,
            ridge_penalty=protocol.value_ridge_penalty,
            solver_version="td8_v2",
        ),
        "conservative_linear_value": fit_conservative_linear_decision_value(
            batch,  # type: ignore[arg-type]
            candidate_actions=actions,
            ridge_penalty=protocol.value_ridge_penalty,
            solver_version="td8_v2",
            uncertainty_penalty_weight=protocol.uncertainty_penalty_weight,
            one_way_switch_cost=train_switch_cost,
            unexcited_prior_std=protocol.unexcited_prior_std,
            residual_dof_out=dof_out,
        ),
        "shallow_value": fit_synthetic_shallow_decision_value(
            batch,  # type: ignore[arg-type]
            candidate_actions=actions,
            ridge_penalty=protocol.value_ridge_penalty,
            solver_version="td8_v2",
        ),
    }


class IncompletePairedLabelsError(ValueError):
    """No learned fit is admitted from outcome-selected survivor pairs."""

    def __init__(
        self, block_id: str, evidence: tuple[IneligiblePairOutcome, ...]
    ) -> None:
        self.block_id = block_id
        self.evidence = evidence
        super().__init__(
            f"TD8 block {block_id} contains {len(evidence)} incomplete/terminated "
            "paired labels; outcomes must be recorded and research fit is blocked"
        )


def _require_complete_reachable_labels(
    collection: ReachableLabelCollection, *, block_id: str
) -> None:
    if collection.ineligible_pairs != len(collection.ineligible_outcomes):
        raise ValueError("incomplete paired-label evidence count mismatch")
    if collection.ineligible_outcomes:
        raise IncompletePairedLabelsError(block_id, collection.ineligible_outcomes)
    if collection.samples and len(collection.eligible_pair_outcomes) != len(
        collection.samples
    ):
        raise ValueError("TD8 eligible paired evidence count mismatch")


def _persist_incomplete_pair_attempt(
    output_dir: Path,
    failure: IncompletePairedLabelsError,
    *,
    dataset_id: str,
    protocol_digest: str,
) -> None:
    """Preserve blocked synthetic-label evidence without publishing a success result.

    Prepare a canonical receipt off to the side, then atomically claim the
    destination with mkdir(exist_ok=False) before installing that receipt. A
    concurrent writer must never be silently overwritten, including an empty
    destination created between preliminary checks and publication.
    """
    destination = Path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"output_dir already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".td8-incomplete-", dir=destination.parent))
    try:
        payload = {
            "schema_version": "decision_value_incomplete_pair_attempt_v1",
            "verdict": "INVALID / BLOCKED",
            "dataset_id": dataset_id,
            "protocol_digest": protocol_digest,
            "block_id": failure.block_id,
            "ineligible_pairs": len(failure.evidence),
            "pairs": [e.to_mapping() for e in failure.evidence],
        }
        (stage / "incomplete-pairs.json").write_bytes(canonical_json_bytes(payload))
        # Directory rename would silently replace a concurrently created EMPTY
        # directory on POSIX. Atomic mkdir is the write-once ownership gate.
        destination.mkdir(exist_ok=False)
        try:
            # Hard-link creation is O_EXCL-like for the destination file:
            # even a writer racing *inside* our claimed directory cannot have
            # their pre-existing evidence replaced by a file rename.
            os.link(
                stage / "incomplete-pairs.json", destination / "incomplete-pairs.json"
            )
        except OSError:
            # Best effort cleanup of our empty reservation. A failed receipt
            # may never be interpreted as a successful research result.
            try:
                destination.rmdir()
            except OSError:
                pass
            raise
    finally:
        if stage.exists():
            shutil.rmtree(stage)


@dataclass(frozen=True, slots=True)
class _BlockTraining:
    """Everything fitted from one dataset view of the train window."""

    dataset: MarketDataset
    b0_model: RidgeForecastModel
    collection_summary: tuple[int, int, int, int, int]
    batch: object
    models: dict[str, object]
    b1c_dof: tuple[float, ...]
    margins: dict[str, MarginSelection]
    ineligible_pair_outcomes: tuple[dict[str, object], ...]
    eligible_pair_outcomes: tuple[dict[str, object], ...]


def _train_block(
    raw_dataset: MarketDataset,
    protocol: DecisionValueDevStudyProtocol,
    block: DecisionValueBlockSpec,
    train_switch_cost: float,
) -> _BlockTraining:
    """Transform, fit B0, collect labels, fit all heads, and select margins.

    Used both for the real block and for the future-perturbed causality probe, so
    the probe exercises exactly the same code path.
    """
    plan = protocol.study_plan
    fit_cutoff = raw_dataset.timestamps[block.fit_cutoff_index]
    eval_start = raw_dataset.timestamps[block.eval_start_index]
    transform = fit_train_feature_transform(
        raw_dataset,
        train_start_index=block.train_start_index,
        train_stop_index=block.train_stop_index,
        clip=protocol.feature_clip,
        as_of_index=block.fit_cutoff_index,
    )
    view = apply_feature_transform(raw_dataset, transform)
    b0_model = fit_ridge_forecast(
        view,
        feature_indices=protocol.ridge_feature_indices,
        fit_cutoff=fit_cutoff,
        horizon_hours=protocol.ridge_horizon_hours,
        alpha=protocol.ridge_alpha,
    )
    ref_digest = content_digest(
        {
            # Version both the teacher feature vintage and native TD8 actions;
            # the archived exploratory run keeps its original provenance.
            "policy": "b0_causal_refit_prefix_hold_relative_v3",
            "teacher_normalization": "preprefix_asof_available_v1",
            "raw_normalization_digest": raw_dataset.normalization_digest,
            "intervention_projection": "posture_v2",
            "native_proposal_version": "td8_v2",
            "target_abs_weights": [float(w) for w in protocol.target_abs_weights],
            "label_prefix_bars": protocol.label_prefix_bars,
            "reference_min_fit_bars": protocol.reference_min_fit_bars,
            "horizon_bars": plan.horizon_bars,
            "ridge_feature_indices": list(protocol.ridge_feature_indices),
            "ridge_horizon_hours": protocol.ridge_horizon_hours,
            "ridge_alpha": protocol.ridge_alpha,
            "ridge_entry_threshold": protocol.ridge_entry_threshold,
            "ridge_exit_threshold": protocol.ridge_exit_threshold,
            "feature_transform_digest": transform.digest,
        }
    )
    provider = causal_b0_reference_provider(
        view,
        feature_indices=protocol.ridge_feature_indices,
        horizon_hours=protocol.ridge_horizon_hours,
        alpha=protocol.ridge_alpha,
        entry_threshold=protocol.ridge_entry_threshold,
        exit_threshold=protocol.ridge_exit_threshold,
        target_abs_weights=protocol.target_abs_weights,
        one_way_switch_cost=train_switch_cost,
        min_fit_bars=protocol.reference_min_fit_bars,
        first_fit_index=block.train_start_index,
        raw_dataset=raw_dataset,
        causal_feature_clip=protocol.feature_clip,
    )
    branches = tuple(
        range(block.train_start_index, block.train_stop_index + 1, block.train_step)
    )
    collection = collect_reachable_state_labels(
        view,
        reference_policy_provider=provider,
        reference_policy_digest=ref_digest,
        branch_indices=branches,
        train_start_index=block.train_start_index,
        prefix_bars=protocol.label_prefix_bars,
        horizon_bars=plan.horizon_bars,
        candidate_actions=plan.candidate_actions,
        target_abs_weights=protocol.target_abs_weights,
        initial_capital=plan.initial_capital,
        execution_cost=protocol.base_execution_cost,
        risk_config=protocol.risk_config,
        fit_cutoff=fit_cutoff,
        source_scope=protocol.study_plan.data_scope,
        require_reference_vintage=True,
    )
    _require_complete_reachable_labels(collection, block_id=block.block_id)
    if not collection.samples:
        raise ValueError(f"block {block.block_id} produced no admissible labels")
    batch = build_synthetic_paired_label_batch(
        collection.samples,
        fit_cutoff=fit_cutoff,
        evaluation_start=eval_start,
        expected_reference_policy_digest=ref_digest,
    )
    dof: list[float] = []
    models = _fit_all_learners(batch, protocol, train_switch_cost, dof)

    margins: dict[str, MarginSelection] = {}
    for arm in LEARNER_ARMS:

        def fit_for_arm(sub_batch: object, _arm: str = arm) -> object:
            return _fit_all_learners(sub_batch, protocol, train_switch_cost)[_arm]

        def policy_for_arm(
            model: object, k: float, calibration_view: MarketDataset, _arm: str = arm
        ) -> PortfolioPolicy:
            return _build_value_policy(
                _arm,
                model,
                calibration_view,
                margin_multiplier=k,
                eval_one_way_cost=train_switch_cost,
                train_one_way_cost=train_switch_cost,
            )

        margins[arm] = select_margin_multiplier(
            dataset=view,
            samples=collection.samples,
            fit_model=fit_for_arm,
            make_policy=policy_for_arm,
            grid=protocol.margin_multiplier_grid,
            calibration_fraction=protocol.margin_calibration_fraction,
            horizon_bars=plan.horizon_bars,
            initial_capital=plan.initial_capital,
            execution_cost=protocol.base_execution_cost,
            risk_config=protocol.risk_config,
            reference_policy_digest=ref_digest,
            raw_dataset=raw_dataset,
            train_start_index=block.train_start_index,
            feature_clip=protocol.feature_clip,
        )
    return _BlockTraining(
        dataset=view,
        b0_model=b0_model,
        collection_summary=(
            collection.paired_replay_runs,
            collection.ineligible_pairs,
            collection.nonflat_branch_states,
            collection.skipped_branches_without_reference,
            len({s.decision_time_ns for s in collection.samples}),
        ),
        batch=batch,
        models=models,
        b1c_dof=tuple(dof),
        margins=margins,
        ineligible_pair_outcomes=tuple(
            record.to_mapping() for record in collection.ineligible_outcomes
        ),
        eligible_pair_outcomes=collection.eligible_pair_outcomes,
    )


def _format_markdown_summary(
    protocol: DecisionValueDevStudyProtocol,
    result: DecisionValueDevStudyResult,
) -> str:
    """Generate human-readable Markdown summary of the study results."""
    lines = [
        "# TD8 Synthetic-only Software Diagnostics (#1025)",
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
        margins = ", ".join(
            f"{arm}={k:g}" for arm, k in sorted(b.chosen_margin_multipliers.items())
        )
        dof = ", ".join(f"{d:.1f}" for d in b.b1c_residual_dof)
        lines.extend(
            [
                f"### {b.block_id}",
                f"- Training rows: {b.training_rows}, Distinct decision times: {b.distinct_decision_times}, Overlap groups: {b.overlap_groups}, Paired replays: {b.paired_replay_runs}, Wall clock: {b.wall_clock_seconds:.2f}s",
                f"- Non-flat branch states: {b.nonflat_branch_states}, Ineligible pairs: {b.ineligible_pairs}, Branches without causal reference: {b.skipped_branches_without_reference}",
                f"- Chosen margin multipliers: {margins}",
                f"- B1c residual dof per head: [{dof}]",
                f"- Causality probe: {'PASSED' if b.causality_probe_passed else 'FAILED'}, Ledger reconciliation: {'PASSED' if b.ledger_evidence_reconciled else 'FAILED'}",
                "",
                "| Arm | Scenario | Final Value | Net Return | Total Cost | Turnover | Max Drawdown |",
                "|---|---|---|---|---|---|---|",
            ]
        )
        for arm_res in b.arms:
            lines.append(
                f"| `{arm_res.arm}` | `{arm_res.scenario}` | {arm_res.final_portfolio_value:.2f} | {arm_res.net_return:+.4%} | {arm_res.total_cost:.2f} | {arm_res.turnover_total:.2f} | {arm_res.max_drawdown:.4%} |"
            )
        lines.append("")
        lines.append("Covariate shift (eval vs train z-score, diagnostic only):")
        lines.append("")
        lines.append("| Feature | Scope | Mean z | Max abs z | Fraction clipped |")
        lines.append("|---|---|---|---|---|")
        for row in b.covariate_shift:
            lines.append(
                f"| `{row['name']}` | {row['scope']} | {float(row['mean_z']):+.2f} | {float(row['max_abs_z']):.2f} | {float(row['fraction_clipped']):.2%} |"  # type: ignore[arg-type]
            )
        lines.append("")
    return "\n".join(lines)


def run_decision_value_dev_study(
    dataset: MarketDataset,
    protocol: DecisionValueDevStudyProtocol,
    *,
    actual_artifact_digest: str | None = None,
    output_dir: str | Path | None = None,
    artifact_root: str | Path | None = None,
) -> DecisionValueDevStudyResult:
    """Execute only verified synthetic TD8 diagnostics; historical real TD8 is frozen."""
    study_start_time = time.perf_counter()

    # Software-only preflight. Identity checks are not independent G0–G2 approval.
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
    # The archived TD8 real-development authorization refers to the original
    # source SHA a775b577e. It cannot authorize this post-review software:
    # versioned fitting, paired intervention, and native proposal all changed.
    if protocol.study_plan.data_scope != "synthetic_only":
        raise ValueError(
            "G0-G3 independent research authorization NOT_ESTABLISHED for revised "
            "TD8 source; historical real_development protocol is read-only"
        )
    if artifact_root is None:
        raise ValueError(
            "synthetic_only TD8 diagnostics require a verified published artifact_root"
        )
    if not dataset.identity_verified:
        raise ValueError(
            "synthetic_only TD8 requires canonical verified input identity"
        )
    artifact, verified_dataset = load_published_market_dataset_artifact(artifact_root)
    if artifact.artifact_digest != protocol.expected_artifact_digest:
        raise ValueError("published artifact digest differs from frozen protocol")
    if verified_dataset.dataset_id != protocol.dataset_id:
        raise ValueError("published dataset identity differs from frozen protocol")
    if not verified_dataset.identity_verified:
        raise ValueError("published dataset identity is not verified")
    provenance = parse_identity_json(verified_dataset.identity_payload_json or "{}")
    # This synthetic flag is source-provided provenance, not independent proof
    # of origin. It must not unlock real-data authorization or G4 claims.
    if provenance.get("synthetic_only") is not True:
        raise ValueError(
            "synthetic_only TD8 diagnostics require declared synthetic_only provenance"
        )
    if verified_dataset.dataset_id != dataset.dataset_id:
        raise ValueError("provided and published dataset identities differ")
    dataset = verified_dataset
    # Fail before costly fit; the write-once check remains at persistence for
    # a racing writer who creates the destination during the study.
    if output_dir is not None and Path(output_dir).exists():
        raise FileExistsError(f"output_dir already exists: {output_dir}")
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

    train_switch_cost = (
        protocol.study_plan.fee_rate
        + protocol.study_plan.spread_rate
        + protocol.base_execution_cost.impact_rate
    )
    block_results: list[DecisionValueBlockResult] = []

    for block in protocol.blocks:
        block_t0 = time.perf_counter()

        # 1-4. Uniform feature transform, B0, reachable-state labels, three heads,
        #      and the train-internal margin selection.
        try:
            training = _train_block(dataset, protocol, block, train_switch_cost)
        except IncompletePairedLabelsError as failure:
            if output_dir is not None:
                _persist_incomplete_pair_attempt(
                    Path(output_dir),
                    failure,
                    dataset_id=dataset.dataset_id,
                    protocol_digest=protocol.protocol_digest,
                )
            raise
        view = training.dataset
        b0_model = training.b0_model
        models = training.models
        (
            paired_replay_runs,
            ineligible_pairs,
            nonflat_states,
            skipped_branches,
            distinct_times,
        ) = training.collection_summary
        paired_transitions = paired_replay_runs * protocol.study_plan.horizon_bars
        batch = training.batch
        training_rows = len(batch.samples)  # type: ignore[attr-defined]
        overlap_groups = len(set(batch.group_ids))  # type: ignore[attr-defined]
        transform = fit_train_feature_transform(
            dataset,
            train_start_index=block.train_start_index,
            train_stop_index=block.train_stop_index,
            clip=protocol.feature_clip,
            as_of_index=block.fit_cutoff_index,
        )
        transform_digest = transform.digest
        shift = covariate_shift_report(
            dataset,
            transform,
            start_index=block.eval_start_index,
            stop_index=block.eval_stop_index,
        )
        chosen = {arm: training.margins[arm].multiplier for arm in LEARNER_ARMS}
        coef_sha = {arm: _coefficient_sha256(models[arm]) for arm in LEARNER_ARMS}

        # 5. Causality perturbation probe: the whole pipeline (transform, B0,
        #    labels, heads, margins) is re-run on a dataset whose bars differ
        #    strictly after the fit cutoff; nothing it produces may change.
        dataset_perturbed = _perturb_after(dataset, block.fit_cutoff_index)
        perturbed = _train_block(dataset_perturbed, protocol, block, train_switch_cost)
        b0_pert = perturbed.b0_model
        b0_fit_ok = bool(
            np.allclose(b0_model.coefficients, b0_pert.coefficients)
            and math.isclose(b0_model.intercept, b0_pert.intercept)
        )
        fits_ok = {
            arm: coef_sha[arm] == _coefficient_sha256(perturbed.models[arm])
            for arm in LEARNER_ARMS
        }
        b1c_model = models["conservative_linear_value"]
        b1c_pert = perturbed.models["conservative_linear_value"]
        fits_ok["conservative_linear_value"] = bool(
            fits_ok["conservative_linear_value"]
            and np.allclose(
                b1c_model.covariance_matrices,  # type: ignore[attr-defined]
                b1c_pert.covariance_matrices,  # type: ignore[attr-defined]
                rtol=0.0,
                atol=1e-12,
            )
        )
        margins_ok = all(
            training.margins[arm].multiplier == perturbed.margins[arm].multiplier
            and training.margins[arm].net_returns == perturbed.margins[arm].net_returns
            for arm in LEARNER_ARMS
        )

        session_clean = SharedPortfolioSession(
            view,
            start_index=block.eval_start_index,
            stop_index=block.eval_stop_index,
            initial_capital=protocol.study_plan.initial_capital,
            execution_cost=protocol.base_execution_cost,
            risk=PreTradeRisk(protocol.risk_config),
        )
        snap_clean = build_portfolio_decision_snapshot(
            view,
            index=block.eval_start_index,
            agent_stop_index=block.eval_stop_index,
            book=session_clean.book,
            active_orders=session_clean.active_orders,
            risk_config=protocol.risk_config,
        )

        pol_b0_probe = _CostAwareRidgePortfolioPolicy(
            dataset=view,
            model=b0_model,
            target_abs_weights=protocol.target_abs_weights,
            entry_threshold=protocol.ridge_entry_threshold,
            exit_threshold=protocol.ridge_exit_threshold,
            one_way_switch_cost=train_switch_cost,
            proposal_version="td8_v2",
        )
        pol_b0_probe_perturbed = _CostAwareRidgePortfolioPolicy(
            dataset=view,
            model=b0_pert,
            target_abs_weights=protocol.target_abs_weights,
            entry_threshold=protocol.ridge_entry_threshold,
            exit_threshold=protocol.ridge_exit_threshold,
            one_way_switch_cost=train_switch_cost,
            proposal_version="td8_v2",
        )
        prop_b0_clean = pol_b0_probe.decide_portfolio(snap_clean)
        prop_b0_pert = pol_b0_probe_perturbed.decide_portfolio(snap_clean)
        b0_probe_ok = b0_fit_ok and (
            prop_b0_clean.target_weights == prop_b0_pert.target_weights
        )

        learner_probe_ok = True
        for arm in LEARNER_ARMS:
            clean_policy = _build_value_policy(
                arm,
                models[arm],
                view,
                margin_multiplier=chosen[arm],
                eval_one_way_cost=train_switch_cost,
                train_one_way_cost=train_switch_cost,
            )
            pert_policy = _build_value_policy(
                arm,
                perturbed.models[arm],
                view,
                margin_multiplier=perturbed.margins[arm].multiplier,
                eval_one_way_cost=train_switch_cost,
                train_one_way_cost=train_switch_cost,
            )
            learner_probe_ok = bool(
                learner_probe_ok
                and fits_ok[arm]
                and (
                    clean_policy.decide_portfolio(snap_clean).target_weights
                    == pert_policy.decide_portfolio(snap_clean).target_weights
                )
            )

        causality_probe_passed = bool(b0_probe_ok and learner_probe_ok and margins_ok)

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
            arm_policies: list[tuple[str, PortfolioPolicy]] = [
                ("cash", CashPortfolioPolicy(symbols=dataset.symbols)),
                (
                    "fixed_long",
                    _FixedEntryPolicy(
                        action=(ACTION_LONG,) * len(dataset.symbols),
                        target_abs_weights=protocol.target_abs_weights,
                        proposal_version="td8_v2",
                    ),
                ),
                (
                    "cost_aware_ridge",
                    _CostAwareRidgePortfolioPolicy(
                        dataset=view,
                        model=b0_model,
                        target_abs_weights=protocol.target_abs_weights,
                        entry_threshold=protocol.ridge_entry_threshold,
                        exit_threshold=protocol.ridge_exit_threshold,
                        one_way_switch_cost=scen_switch_cost,
                        proposal_version="td8_v2",
                    ),
                ),
            ]
            for arm in LEARNER_ARMS:
                arm_policies.append(
                    (
                        arm,
                        _build_value_policy(
                            arm,
                            models[arm],
                            view,
                            margin_multiplier=chosen[arm],
                            eval_one_way_cost=scen_switch_cost,
                            train_one_way_cost=train_switch_cost,
                        ),
                    )
                )

            for arm_name, policy in arm_policies:
                eval_replay_runs += 1
                replay_res = run_shared_portfolio_replay(
                    view,
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
                    replay_res,
                    dataset=view,
                    expected_intervals=expected_intervals,
                    expected_start_index=block.eval_start_index,
                    initial_capital=protocol.study_plan.initial_capital,
                    execution_cost=scen_cost,
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
                        turnover_total=float(replay_res.book.turnover_total),
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
                b1_model_coefficients_sha256=coef_sha["linear_value"],
                b1c_model_coefficients_sha256=coef_sha["conservative_linear_value"],
                b2_model_coefficients_sha256=coef_sha["shallow_value"],
                causality_probe_passed=causality_probe_passed,
                ledger_evidence_reconciled=ledger_all_reconciled,
                arms=tuple(arms_metrics),
                chosen_margin_multipliers=chosen,
                margin_selections={
                    arm: training.margins[arm].to_mapping() for arm in LEARNER_ARMS
                },
                b1c_residual_dof=training.b1c_dof,
                distinct_decision_times=distinct_times,
                ineligible_pairs=ineligible_pairs,
                ineligible_pair_outcomes=training.ineligible_pair_outcomes,
                eligible_pair_outcomes=training.eligible_pair_outcomes,
                nonflat_branch_states=nonflat_states,
                skipped_branches_without_reference=skipped_branches,
                feature_transform_digest=transform_digest,
                covariate_shift=shift,
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

    # The synthetic fixture is NOT a G4 market experiment, even if its
    # internal fixed-control comparison happens to be statistically favorable.
    if verdict != "INVALID / BLOCKED":
        verdict = "SOFTWARE_ONLY"
        verdict_reason = (
            "Synthetic TD8 diagnostic only; revised source has no independent "
            "G0-G3 real-development authorization or G4 economic verdict"
        )

    result = DecisionValueDevStudyResult(
        schema_version="decision_value_dev_study_result_v3_software",
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
