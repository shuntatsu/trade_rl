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
import json
import math
import os
import re
import shutil
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
from pathlib import Path

import numpy as np

from trade_rl._validation import require_sha256
from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.artifacts.verified_file import open_regular_binary
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
from trade_rl.evaluation.runs import build_candidate_run_provenance
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
    """Frozen TD8 parameters; real v2 is archival and synthetic v4 is runnable."""

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
            historical_execution = ExecutionCostConfig(
                fee_rate=self.base_execution_cost.fee_rate,
                spread_rate=self.base_execution_cost.spread_rate,
                impact_rate=self.base_execution_cost.impact_rate,
                max_participation_rate=self.base_execution_cost.max_participation_rate,
                order_latency_bars=self.base_execution_cost.order_latency_bars,
            )
            historical_risk = PreTradeRiskConfig(
                max_gross=self.risk_config.max_gross,
                max_abs_weight=self.risk_config.max_abs_weight,
                max_turnover=self.risk_config.max_turnover,
                drawdown_start=self.risk_config.drawdown_start,
                drawdown_stop=self.risk_config.drawdown_stop,
            )
            if (
                self.base_execution_cost != historical_execution
                or self.risk_config != historical_risk
            ):
                raise ValueError(
                    "historical v2 authorization rejects undeclared execution/risk options"
                )
            if self.study_plan.real_scope_authorization_digest != expected_digest:
                raise ValueError(
                    "study_plan real_scope_authorization_digest does not match pre-authorization specification"
                )
        elif self.study_plan.real_scope_authorization_digest is not None:
            raise ValueError("synthetic_only study cannot claim real authorization")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": (
                "decision_value_dev_study_protocol_v4_software"
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
            "risk_config": (
                asdict(self.risk_config)
                if self.study_plan.data_scope == "synthetic_only"
                else {
                    "max_gross": self.risk_config.max_gross,
                    "max_abs_weight": self.risk_config.max_abs_weight,
                    "max_turnover": self.risk_config.max_turnover,
                    "drawdown_start": self.risk_config.drawdown_start,
                    "drawdown_stop": self.risk_config.drawdown_stop,
                }
            ),
            "base_execution_cost": (
                self.base_execution_cost.execution_policy_payload()
                if self.study_plan.data_scope == "synthetic_only"
                else {
                    "fee_rate": self.base_execution_cost.fee_rate,
                    "spread_rate": self.base_execution_cost.spread_rate,
                    "impact_rate": self.base_execution_cost.impact_rate,
                    "max_participation_rate": self.base_execution_cost.max_participation_rate,
                    "order_latency_bars": self.base_execution_cost.order_latency_bars,
                }
            ),
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
    label_evidence_digest: str | None = None
    label_bound_reference_digest: str | None = None

    def to_mapping(self) -> dict[str, object]:
        if self.label_evidence_digest is not None:
            if (
                not self.eligible_pair_outcomes
                or content_digest(self.eligible_pair_outcomes)
                != self.label_evidence_digest
            ):
                raise ValueError(
                    "TD8 eligible label evidence population digest mismatch"
                )
            for receipt in self.eligible_pair_outcomes:
                if receipt.get("evidence_digest") != content_digest(
                    {k: v for k, v in receipt.items() if k != "evidence_digest"}
                ):
                    raise ValueError("TD8 per-pair evidence digest mismatch")
            recipe_roots = {
                r.get("reference_policy_digest") for r in self.eligible_pair_outcomes
            }
            if len(recipe_roots) != 1:
                raise ValueError("TD8 label evidence has inconsistent teacher recipe")
            recipe_root = next(iter(recipe_roots))
            if not isinstance(
                recipe_root, str
            ) or self.label_bound_reference_digest != content_digest(
                {
                    "schema_version": "td8_reference_with_label_population_v1",
                    "reference_recipe_digest": recipe_root,
                    "label_evidence_digest": self.label_evidence_digest,
                }
            ):
                raise ValueError("TD8 label-bound reference digest mismatch")
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
            **(
                {
                    "label_evidence_digest": self.label_evidence_digest,
                    "label_bound_reference_digest": self.label_bound_reference_digest,
                }
                if self.label_evidence_digest is not None
                else {}
            ),
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
        if self.schema_version == "decision_value_dev_study_result_v4_software":
            if not self.blocks or any(
                block.label_evidence_digest is None
                or block.label_bound_reference_digest is None
                for block in self.blocks
            ):
                raise ValueError(
                    "TD8 v4 result requires a label evidence root per block"
                )
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
        # Set once the write-once failure receipt has been durably published;
        # it lets a caller pin the exact receipt when strictly reloading it.
        self.receipt_digest: str | None = None
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


_DATASET_VIEW_DIGEST_SCHEMA = "decision_value_dataset_view_v3"
_DATASET_VIEW_ARRAYS = (
    "timestamps",
    "features",
    "global_features",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "funding_rate",
    "tradable",
    "feature_available",
)


def _dataset_view_digest(dataset: MarketDataset) -> str:
    """Digest the declared arrays and names of one dataset view.

    Scope, exactly: ``dataset_id``, ``symbols``, ``feature_names``,
    ``global_feature_names`` and the arrays in ``_DATASET_VIEW_ARRAYS``
    (timestamps, features, global features, OHLCV, funding rate, tradable mask
    and feature-availability mask), each framed by name, dtype and shape. The
    optional execution/cost/calendar/availability arrays (``fee_rate``,
    ``mark_price``, ``available_at`` and so on) and ``identity_payload_json``
    are **not** covered, so this identifies the observed view of a probe copy
    for those arrays only, not an arbitrary complete source identity.
    """
    digest = hashlib.sha256()
    digest.update(
        canonical_json_bytes(
            {
                "schema": _DATASET_VIEW_DIGEST_SCHEMA,
                "dataset_id": dataset.dataset_id,
                "symbols": list(dataset.symbols),
                "feature_names": list(dataset.feature_names),
                "global_feature_names": list(dataset.global_feature_names),
            }
        )
    )
    for name in _DATASET_VIEW_ARRAYS:
        array = np.ascontiguousarray(getattr(dataset, name))
        digest.update(f"|{name}|{array.dtype.str}|{array.shape}|".encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


_INCOMPLETE_RECEIPT_FILE = "incomplete-pairs.json"
# Default untrusted-reader resource budget, not a universal v2/v3 schema size.
_DEFAULT_INCOMPLETE_RECEIPT_BYTES = 64 * 1024 * 1024
# Exception for a separately pinned legacy/large receipt, still always bounded.
_MAX_INCOMPLETE_RECEIPT_BYTES = 512 * 1024 * 1024
_INCOMPLETE_RECEIPT_SCHEMA = "decision_value_incomplete_pair_attempt_v3"
# v2 receipts may already exist on disk: the reader keeps accepting them (with
# the plain finite-decimal rules only), but the writer only emits v3.
_INCOMPLETE_RECEIPT_SCHEMA_V2 = "decision_value_incomplete_pair_attempt_v2"
_INCOMPLETE_RECEIPT_READABLE_SCHEMAS = (
    _INCOMPLETE_RECEIPT_SCHEMA_V2,
    _INCOMPLETE_RECEIPT_SCHEMA,
)
_INCOMPLETE_RECEIPT_VERDICT = "INVALID / BLOCKED"
_INCOMPLETE_RECEIPT_REASON = "incomplete_paired_labels"
_INCOMPLETE_RECEIPT_RETRY_POLICY = "write_once_no_restart"
_INCOMPLETE_RECEIPT_PHASES = ("training", "causality_probe")
_INCOMPLETE_PAIR_SCHEMA = "decision_value_ineligible_pair_v1"
_INCOMPLETE_RECEIPT_SHA_FIELDS = (
    "dataset_id",
    "protocol_digest",
    "actual_artifact_digest",
    "implementation_digest",
    "runtime_environment_digest",
    "pairs_digest",
)
_INCOMPLETE_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "verdict",
        "reason",
        "retry_policy",
        "fit_admitted",
        "block_id",
        "phase",
        "observed_view_digest",
        "ineligible_pairs",
        "pairs",
        "receipt_digest",
        *_INCOMPLETE_RECEIPT_SHA_FIELDS,
    }
)

# Bounded numeric-string grammar of the receipt (untrusted input to the reader).
# Anything outside it is rejected before ``Fraction`` ever sees it.
_MAX_RECEIPT_NUMERIC_CHARS = 16_384
_MAX_RECEIPT_NUMERIC_DIGITS = 4_000
_MAX_RECEIPT_EXPONENT = 4_096
_MAX_RECEIPT_INT = 2**63 - 1
_MAX_RECEIPT_JSON_DEPTH = 16
_NONFINITE_PREFIX = "nonfinite:"
_NONFINITE_KINDS = ("nan", "inf", "-inf", "oversized", "unparseable")
_NONFINITE_SENTINELS = frozenset(
    f"{_NONFINITE_PREFIX}{kind}" for kind in _NONFINITE_KINDS
)
_DECIMAL_TEXT = re.compile(
    r"[+-]?(?P<int>[0-9]*)(?:\.(?P<frac>[0-9]*))?(?:[eE](?P<exp>[+-]?[0-9]+))?",
    re.ASCII,
)
_RATIONAL_TEXT = re.compile(r"[+-]?(?P<num>[0-9]+)/(?P<den>[0-9]+)", re.ASCII)
_QUARANTINE_REASON = "receipt:unvalidated_source_record"
_QUARANTINE_FIELD = "unvalidated_source_record"
_QUARANTINE_TEXT_LIMIT = 65_536
_QUARANTINE_ERROR_LIMIT = 512
_QUARANTINE_ENCODINGS = ("canonical_json", "repr")


def _decimal_status(text: str) -> str:
    """Classify numeric text; ``"ok"`` means a bounded, finite exact value.

    The grammar is a strict ASCII decimal (optional sign, digits with an
    optional fraction, optional ``e`` exponent) or ``numerator/denominator``,
    at most 16,384 characters, at most 4,000 mantissa/numerator/denominator
    digits and an exponent magnitude of at most 4,096. ``Fraction`` is only
    called after these bounds hold, so hostile exponents or lengths cost
    constant time. Other results name the failure: ``nan``, ``inf``, ``-inf``,
    ``oversized`` (well formed but beyond a bound) or ``unparseable``.
    """
    if len(text) > _MAX_RECEIPT_NUMERIC_CHARS:
        return "oversized"
    lowered = text.strip().lower()
    if lowered in ("nan", "+nan", "-nan"):
        return "nan"
    if lowered in ("inf", "+inf", "infinity", "+infinity"):
        return "inf"
    if lowered in ("-inf", "-infinity"):
        return "-inf"
    if not text or text != text.strip():
        return "unparseable"
    rational = _RATIONAL_TEXT.fullmatch(text)
    if rational is not None:
        longest = max(len(rational.group("num")), len(rational.group("den")))
        if longest > _MAX_RECEIPT_NUMERIC_DIGITS:
            return "oversized"
    else:
        decimal = _DECIMAL_TEXT.fullmatch(text)
        if decimal is None:
            return "unparseable"
        digits = len(decimal.group("int") or "") + len(decimal.group("frac") or "")
        if digits == 0:
            return "unparseable"
        if digits > _MAX_RECEIPT_NUMERIC_DIGITS:
            return "oversized"
        exponent = decimal.group("exp")
        if exponent is not None:
            magnitude = exponent.lstrip("+-")
            if len(magnitude) > 4 or int(magnitude) > _MAX_RECEIPT_EXPONENT:
                return "oversized"
    try:
        Fraction(text)
    except (ValueError, ZeroDivisionError):
        return "unparseable"
    return "ok"


def _require_finite_decimal(
    value: object, *, field: str, allow_sentinel: bool = False
) -> None:
    """Accept only a bounded finite exact decimal/fraction string.

    ``allow_sentinel`` additionally admits the v3 ``nonfinite:<kind>`` encoding
    that the writer uses for values that are non-finite or unrepresentable.
    """
    message = f"failure receipt {field} must be a finite numeric string"
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(message)
    if allow_sentinel and value in _NONFINITE_SENTINELS:
        return
    if _decimal_status(value) != "ok":
        raise ValueError(message)


def _sentinel(kind: str) -> str:
    return f"{_NONFINITE_PREFIX}{kind}"


def _receipt_decimal(value: object) -> object:
    """Encode one numeric endpoint value for a v3 receipt.

    A bounded finite decimal text is kept verbatim. A non-finite or
    unrepresentable number becomes ``nonfinite:<kind>`` so that the receipt is
    always writable. Values of any other type are returned unchanged and are
    left to the shared validator.
    """
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return value
    if isinstance(value, str) and value in _NONFINITE_SENTINELS:
        return value
    try:
        text = value if isinstance(value, str) else str(value)
    except ValueError:  # an int beyond the interpreter's string-conversion limit
        return _sentinel("oversized")
    status = _decimal_status(text)
    return text if status == "ok" else _sentinel(status)


def _receipt_remainder(value: object) -> object:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        if math.isnan(value):
            return _sentinel("nan")
        return _sentinel("inf" if value > 0 else "-inf")
    if isinstance(value, int):
        return value if abs(value) <= _MAX_RECEIPT_INT else _sentinel("oversized")
    return _receipt_decimal(value)


def _nonfinite_free(value: object, depth: int = 0) -> object:
    """Replace non-finite floats anywhere in a record by their sentinel."""
    if depth > _MAX_RECEIPT_JSON_DEPTH:
        raise ValueError("failure receipt record is nested too deeply")
    if isinstance(value, float) and not math.isfinite(value):
        return _receipt_remainder(value)
    if isinstance(value, dict):
        return {key: _nonfinite_free(item, depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_nonfinite_free(item, depth + 1) for item in value]
    return value


def _normalise_pair_mapping(mapping: dict[str, object]) -> object:
    """Apply the v3 endpoint encoding to a copy of one source pair record."""
    pair = dict(mapping)
    for arm in ("baseline", "candidate"):
        endpoint = pair.get(arm)
        if not isinstance(endpoint, dict):
            continue
        record = dict(endpoint)
        for name in _ENDPOINT_DECIMALS:
            if name in record:
                record[name] = _receipt_decimal(record[name])
        quantities = record.get("terminal_exact_quantities")
        if isinstance(quantities, (list, tuple)):
            record["terminal_exact_quantities"] = [
                _receipt_decimal(item) for item in quantities
            ]
        pending = record.get("pending_order_remainders")
        if isinstance(pending, (list, tuple)):
            record["pending_order_remainders"] = [
                [item[0], _receipt_remainder(item[1])]
                if isinstance(item, (list, tuple)) and len(item) == 2
                else item
                for item in pending
            ]
        pair[arm] = record
    return _nonfinite_free(pair)


def _safe_text(text: str) -> str:
    return text.encode("utf-8", "backslashreplace").decode("utf-8")


def _describe_error(error: BaseException) -> str:
    try:
        text = f"{type(error).__name__}: {error}"
    except Exception:
        text = type(error).__name__
    return _safe_text(text)[:_QUARANTINE_ERROR_LIMIT]


def _quarantine_source(evidence: object, error: BaseException) -> dict[str, object]:
    """Preserve an unvalidatable source record as bounded text plus its digest."""
    encoding = "canonical_json"
    try:
        text = canonical_json_bytes(
            _nonfinite_free(evidence.to_mapping())  # type: ignore[attr-defined]
        ).decode("utf-8")
    except Exception:
        encoding = "repr"
        try:
            text = repr(evidence)
        except Exception:
            text = "<unrepresentable source record>"
    text = _safe_text(text)
    return {
        "encoding": encoding,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "text": text[:_QUARANTINE_TEXT_LIMIT],
        "truncated": len(text) > _QUARANTINE_TEXT_LIMIT,
        "validation_error": _describe_error(error),
    }


def _placeholder_endpoint() -> dict[str, object]:
    """An endpoint whose every field is a placeholder, never evidence."""
    unknown = _sentinel("unparseable")
    return {
        **{name: unknown for name in _ENDPOINT_DECIMALS},
        **{name: 0 for name in _ENDPOINT_INTS},
        **{name: False for name in _ENDPOINT_BOOLS},
        "terminal_exact_quantities": [unknown],
        "termination_reason": None,
        "pending_order_remainders": [],
    }


def _salvaged_index(value: object) -> int:
    if isinstance(value, int) and _is_bounded_int(value, signed=False):
        return value
    return 0


def _salvaged_reasons(value: object) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [
        item
        for item in value
        if isinstance(item, str)
        and item
        and item != _QUARANTINE_REASON
        and _safe_text(item) == item
    ]


def _quarantined_pair(evidence: object, error: BaseException) -> dict[str, object]:
    """Minimal valid pair for a record that cannot be encoded as evidence.

    Structure and every endpoint field are placeholders; the original record is
    kept verbatim (bounded) in ``unvalidated_source_record``. The pair is never
    admissible for fit and the reason codes lead with the quarantine marker.
    """
    action = getattr(evidence, "candidate_action", None)
    if not (
        isinstance(action, (list, tuple))
        and action
        and all(_is_bounded_int(item, signed=True) for item in action)
    ):
        action = (0,)
    return {
        "schema": _INCOMPLETE_PAIR_SCHEMA,
        "intervention_index": _salvaged_index(
            getattr(evidence, "intervention_index", None)
        ),
        "prefix_start_index": _salvaged_index(
            getattr(evidence, "prefix_start_index", None)
        ),
        "terminal_index": _salvaged_index(getattr(evidence, "terminal_index", None)),
        "candidate_action": list(action),
        "reason_codes": [
            _QUARANTINE_REASON,
            *_salvaged_reasons(getattr(evidence, "reason_codes", None)),
        ],
        "baseline": _placeholder_endpoint(),
        "candidate": _placeholder_endpoint(),
        "admissible_for_fit": False,
        _QUARANTINE_FIELD: _quarantine_source(evidence, error),
    }


def _receipt_pair(evidence: object) -> dict[str, object]:
    """Encode one source record as the reloaded form of a v3 pair, never raising.

    The record is normalised (non-finite values to ``nonfinite:<kind>``),
    round-tripped through canonical JSON and then run through the *same*
    validator the strict reader uses. A record that still cannot be validated
    is quarantined instead of aborting the write: the root must be claimed.
    """
    try:
        normalised = _normalise_pair_mapping(
            evidence.to_mapping()  # type: ignore[attr-defined]
        )
        pair = json.loads(canonical_json_bytes(normalised).decode("utf-8"))
        _validate_incomplete_pair_record(pair, allow_sentinel=True)
    except Exception as error:  # every failure must still yield a receipt
        return _quarantined_pair(evidence, error)
    return dict(pair)


def _receipt_pairs(evidence: Sequence[object]) -> list[dict[str, object]]:
    return [_receipt_pair(item) for item in evidence]


def _quarantined_receipt_pairs(evidence: Sequence[object]) -> list[dict[str, object]]:
    error = ValueError("failure receipt self-check rejected the encoded pairs")
    return [_quarantined_pair(item, error) for item in evidence]


def _encode_incomplete_receipt(
    failure: IncompletePairedLabelsError,
    *,
    dataset_id: str,
    protocol_digest: str,
    actual_artifact_digest: str,
    implementation_digest: str,
    runtime_environment_digest: str,
    phase: str,
    observed_view_digest: str | None,
) -> tuple[bytes, str]:
    """Return canonical receipt bytes verified by the shared strict decoder.

    The bytes are checked by the reader's structural/canonical validator
    before they are returned. If the encoding is rejected, every pair is
    quarantined and checked again before claiming the destination root.
    File-based reload also enforces a separate finite input-byte budget and
    may reject an otherwise valid oversized receipt.
    """
    last_error: Exception | None = None
    for build_pairs in (_receipt_pairs, _quarantined_receipt_pairs):
        try:
            pairs = build_pairs(failure.evidence)
            body: dict[str, object] = {
                "schema_version": _INCOMPLETE_RECEIPT_SCHEMA,
                "verdict": _INCOMPLETE_RECEIPT_VERDICT,
                "reason": _INCOMPLETE_RECEIPT_REASON,
                "retry_policy": _INCOMPLETE_RECEIPT_RETRY_POLICY,
                "fit_admitted": False,
                "dataset_id": dataset_id,
                "protocol_digest": protocol_digest,
                "actual_artifact_digest": actual_artifact_digest,
                "implementation_digest": implementation_digest,
                "runtime_environment_digest": runtime_environment_digest,
                "block_id": failure.block_id,
                "phase": phase,
                "observed_view_digest": observed_view_digest,
                "ineligible_pairs": len(pairs),
                "pairs": pairs,
                "pairs_digest": content_digest(pairs),
            }
            receipt_digest = content_digest(body)
            encoded = canonical_json_bytes({**body, "receipt_digest": receipt_digest})
            _decode_incomplete_receipt(encoded, expected_receipt_digest=receipt_digest)
        except Exception as error:  # fall back to the all-quarantine encoding
            last_error = error
            continue
        return encoded, receipt_digest
    raise ValueError(
        "failure receipt could not be encoded in a reloadable form"
    ) from last_error


def _persist_incomplete_pair_attempt(
    output_dir: Path,
    failure: IncompletePairedLabelsError,
    *,
    dataset_id: str,
    protocol_digest: str,
    actual_artifact_digest: str,
    implementation_digest: str,
    runtime_environment_digest: str,
    phase: str,
    observed_view_digest: str | None,
) -> str:
    """Preserve blocked synthetic-label evidence without publishing a success result.

    The receipt binds the exact dataset, frozen protocol, published artifact and
    the package-source / runtime-manifest content digests (see
    ``build_candidate_run_provenance``) to every ineligible branch record and
    carries a self digest. It never admits a fit and never permits a restart
    into the same root. Return that digest so a caller can pin the receipt on
    strict reload.

    ``phase`` names where the block failed: ``training`` (the primary fit) or
    ``causality_probe`` (the perturbed re-run, whose ``observed_view_digest``
    identifies the perturbed view for the arrays that digest covers).
    Implementation/runtime identities are resolved once before the study
    starts, so this abort path never depends on hashing the source tree.

    Order of operations, so that no data-dependent failure can leave the root
    retryable: (1) cheap identity preconditions, (2) encode the receipt and
    decode it with the strict reader's validator (non-finite or unrepresentable
    values are encoded, never raised; see ``_encode_incomplete_receipt``), and
    only then (3) claim the destination. Only environmental I/O failures after
    step 2 remain; they happen before any result is published.

    Prepare a canonical receipt off to the side, then atomically claim the
    destination with mkdir(exist_ok=False) before installing that receipt. A
    concurrent writer must never be silently overwritten, including an empty
    destination created between preliminary checks and publication.
    """
    for name, value in (
        ("dataset_id", dataset_id),
        ("protocol_digest", protocol_digest),
        ("actual_artifact_digest", actual_artifact_digest),
        ("implementation_digest", implementation_digest),
        ("runtime_environment_digest", runtime_environment_digest),
    ):
        require_sha256(value, field=name)
    if phase not in _INCOMPLETE_RECEIPT_PHASES:
        raise ValueError("failure receipt phase is not recognised")
    if (phase == "training") != (observed_view_digest is None):
        raise ValueError("only a causality_probe receipt carries a view digest")
    if observed_view_digest is not None:
        require_sha256(observed_view_digest, field="observed_view_digest")
    if not failure.evidence:
        raise ValueError("a failure receipt requires at least one ineligible pair")
    destination = Path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"output_dir already exists: {destination}")
    encoded, receipt_digest = _encode_incomplete_receipt(
        failure,
        dataset_id=dataset_id,
        protocol_digest=protocol_digest,
        actual_artifact_digest=actual_artifact_digest,
        implementation_digest=implementation_digest,
        runtime_environment_digest=runtime_environment_digest,
        phase=phase,
        observed_view_digest=observed_view_digest,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".td8-incomplete-", dir=destination.parent))
    try:
        (stage / _INCOMPLETE_RECEIPT_FILE).write_bytes(encoded)
        # Directory rename would silently replace a concurrently created EMPTY
        # directory on POSIX. Atomic mkdir is the write-once ownership gate.
        destination.mkdir(exist_ok=False)
        try:
            # Hard-link creation is O_EXCL-like for the destination file:
            # even a writer racing *inside* our claimed directory cannot have
            # their pre-existing evidence replaced by a file rename.
            os.link(
                stage / _INCOMPLETE_RECEIPT_FILE,
                destination / _INCOMPLETE_RECEIPT_FILE,
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
    return receipt_digest


def _is_native_int(value: object) -> bool:
    return type(value) is int


def _is_bounded_int(value: object, *, signed: bool) -> bool:
    """A native int that survives JSON reload (inside the 64-bit range)."""
    if type(value) is not int:
        return False
    if signed:
        return abs(value) <= _MAX_RECEIPT_INT
    return 0 <= value <= _MAX_RECEIPT_INT


def _require_json_safe(value: object, *, field: str, depth: int = 0) -> None:
    """Reject anything that would not survive a canonical-JSON round trip."""
    if depth > _MAX_RECEIPT_JSON_DEPTH:
        raise ValueError(f"failure receipt {field} is nested too deeply")
    if value is None or isinstance(value, (bool, str)):
        return
    if isinstance(value, int):
        if not _is_bounded_int(value, signed=True):
            raise ValueError(f"failure receipt {field} holds an out-of-range integer")
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"failure receipt {field} holds a non-finite float")
        return
    if isinstance(value, list):
        for item in value:
            _require_json_safe(item, field=field, depth=depth + 1)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"failure receipt {field} has a non-string key")
            _require_json_safe(item, field=field, depth=depth + 1)
        return
    raise ValueError(f"failure receipt {field} holds an unsupported value")


_ENDPOINT_DECIMALS = (
    "terminal_marked_equity",
    "observed_equity_change",
    "initial_capital",
    "terminal_cash",
    "total_cost",
    "funding_pnl",
)
_ENDPOINT_INTS = (
    "requested_stop_index",
    "observed_stop_index",
    "observed_intervals",
    "expected_intervals",
)
_ENDPOINT_BOOLS = ("insolvent", "settled_label_eligible")
_ENDPOINT_KEYS = frozenset(
    (
        *_ENDPOINT_DECIMALS,
        *_ENDPOINT_INTS,
        *_ENDPOINT_BOOLS,
        "terminal_exact_quantities",
        "termination_reason",
        "pending_order_remainders",
    )
)


def _validate_incomplete_endpoint(
    endpoint: object, *, arm: str, allow_sentinel: bool = False
) -> None:
    if not isinstance(endpoint, dict) or set(endpoint) != _ENDPOINT_KEYS:
        raise ValueError(f"failure receipt {arm} endpoint fields are malformed")
    for name in _ENDPOINT_DECIMALS:
        _require_finite_decimal(
            endpoint[name], field=f"{arm}.{name}", allow_sentinel=allow_sentinel
        )
    for name in _ENDPOINT_INTS:
        if not _is_bounded_int(endpoint[name], signed=False):
            raise ValueError(f"failure receipt {arm}.{name} must be an integer")
    for name in _ENDPOINT_BOOLS:
        if type(endpoint[name]) is not bool:
            raise ValueError(f"failure receipt {arm}.{name} must be a boolean")
    quantities = endpoint["terminal_exact_quantities"]
    if not isinstance(quantities, list) or not quantities:
        raise ValueError(f"failure receipt {arm} quantities are malformed")
    for quantity in quantities:
        _require_finite_decimal(
            quantity,
            field=f"{arm}.terminal_exact_quantities",
            allow_sentinel=allow_sentinel,
        )
    reason = endpoint["termination_reason"]
    if reason is not None and (not isinstance(reason, str) or not reason):
        raise ValueError(f"failure receipt {arm} termination reason is malformed")
    pending = endpoint["pending_order_remainders"]
    if not isinstance(pending, list):
        raise ValueError(f"failure receipt {arm} pending orders are malformed")
    for item in pending:
        if (
            not isinstance(item, list)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not item[0]
            or isinstance(item[1], bool)
            or not isinstance(item[1], (int, float, str))
        ):
            raise ValueError(f"failure receipt {arm} pending order is malformed")
        remainder = item[1]
        if isinstance(remainder, str):
            _require_finite_decimal(
                remainder,
                field=f"{arm}.pending_order_remainders",
                allow_sentinel=allow_sentinel,
            )
        elif isinstance(remainder, float):
            if not math.isfinite(remainder):
                raise ValueError(f"failure receipt {arm} pending order is not finite")
        elif not _is_bounded_int(remainder, signed=True):
            raise ValueError(f"failure receipt {arm} pending order is out of range")


_PAIR_KEYS = frozenset(
    {
        "schema",
        "intervention_index",
        "prefix_start_index",
        "terminal_index",
        "candidate_action",
        "reason_codes",
        "baseline",
        "candidate",
        "admissible_for_fit",
    }
)
_QUARANTINE_KEYS = frozenset(
    {"encoding", "sha256", "text", "truncated", "validation_error"}
)


def _validate_quarantine_record(record: object) -> None:
    if (
        not isinstance(record, dict)
        or set(record) != _QUARANTINE_KEYS
        or record["encoding"] not in _QUARANTINE_ENCODINGS
        or not isinstance(record["text"], str)
        or len(record["text"]) > _QUARANTINE_TEXT_LIMIT
        or type(record["truncated"]) is not bool
        or not isinstance(record["validation_error"], str)
        or len(record["validation_error"]) > _QUARANTINE_ERROR_LIMIT
    ):
        raise ValueError("failure receipt unvalidated source record is malformed")
    digest = record["sha256"]
    if not isinstance(digest, str):
        raise ValueError("failure receipt unvalidated source digest is malformed")
    require_sha256(digest, field="unvalidated_source_record.sha256")
    # A truncated prefix cannot independently establish the digest of the
    # omitted suffix. In that case require exactly the writer's prefix length;
    # a complete source record must instead match its actual UTF-8 checksum.
    text = record["text"]
    if record["truncated"]:
        if len(text) != _QUARANTINE_TEXT_LIMIT:
            raise ValueError(
                "failure receipt unvalidated source truncated prefix is malformed"
            )
    else:
        try:
            source_bytes = text.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ValueError(
                "failure receipt unvalidated source text is not UTF-8"
            ) from error
        if hashlib.sha256(source_bytes).hexdigest() != digest:
            raise ValueError("failure receipt unvalidated source digest mismatch")


def _require_canonical_round_trip(value: object, *, field: str) -> None:
    try:
        raw = canonical_json_bytes(value)
        again = canonical_json_bytes(json.loads(raw.decode("utf-8")))
    except (ValueError, TypeError, RecursionError) as error:
        raise ValueError(f"failure receipt {field} is not canonical JSON") from error
    if again != raw:
        raise ValueError(f"failure receipt {field} is not canonical JSON")


def _validate_incomplete_pair_record(
    pair: object, *, allow_sentinel: bool = False
) -> None:
    """Structurally validate one ``decision_value_ineligible_pair_v1`` record.

    This is the single validator shared by the writer (before it claims the
    root) and the strict reader. ``allow_sentinel`` is true for v3 receipts and
    admits the ``nonfinite:<kind>`` numeric encoding and the quarantine record;
    v2 receipts are validated without either.
    """
    optional = {"reference_vintage"}
    if allow_sentinel:
        optional.add(_QUARANTINE_FIELD)
    if (
        not isinstance(pair, dict)
        or not _PAIR_KEYS <= set(pair)
        or not set(pair) <= _PAIR_KEYS | optional
        or pair["schema"] != _INCOMPLETE_PAIR_SCHEMA
        or pair["admissible_for_fit"] is not False
    ):
        raise ValueError("failure receipt pair record is malformed or admissible")
    for name in ("intervention_index", "prefix_start_index", "terminal_index"):
        if not _is_bounded_int(pair[name], signed=False):
            raise ValueError(f"failure receipt pair {name} must be an integer")
    action = pair["candidate_action"]
    if not isinstance(action, list) or not action:
        raise ValueError("failure receipt pair candidate_action is malformed")
    if not all(_is_bounded_int(item, signed=True) for item in action):
        raise ValueError("failure receipt pair candidate_action is malformed")
    reasons = pair["reason_codes"]
    if (
        not isinstance(reasons, list)
        or not reasons
        or not all(isinstance(item, str) and item for item in reasons)
    ):
        raise ValueError("failure receipt pair reason_codes are malformed")
    quarantined = _QUARANTINE_FIELD in pair
    if quarantined != (reasons[0] == _QUARANTINE_REASON) or (
        _QUARANTINE_REASON in reasons[1:]
    ):
        raise ValueError("failure receipt quarantine marker is inconsistent")
    if quarantined:
        _validate_quarantine_record(pair[_QUARANTINE_FIELD])
    _validate_incomplete_endpoint(
        pair["baseline"], arm="baseline", allow_sentinel=allow_sentinel
    )
    _validate_incomplete_endpoint(
        pair["candidate"], arm="candidate", allow_sentinel=allow_sentinel
    )
    vintage = pair.get("reference_vintage")
    if "reference_vintage" in pair:
        if not isinstance(vintage, dict):
            raise ValueError("failure receipt pair reference_vintage is malformed")
        _require_json_safe(vintage, field="reference_vintage")
    _require_canonical_round_trip(pair, field="pair record")


def _decode_incomplete_receipt(
    raw: bytes, *, expected_receipt_digest: str | None = None
) -> dict[str, object]:
    """Strictly validate receipt bytes (shared by the reader and the writer)."""
    try:
        loaded = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("failure receipt JSON is malformed") from error
    if not isinstance(loaded, dict) or any(not isinstance(k, str) for k in loaded):
        raise ValueError("failure receipt must contain a JSON object")
    try:
        canonical = canonical_json_bytes(loaded)
    except RecursionError as error:
        raise ValueError("failure receipt JSON is malformed") from error
    if raw != canonical:
        raise ValueError("failure receipt must use canonical JSON bytes")
    if set(loaded) != _INCOMPLETE_RECEIPT_KEYS:
        raise ValueError("failure receipt fields are malformed")
    if (
        loaded["schema_version"] not in _INCOMPLETE_RECEIPT_READABLE_SCHEMAS
        or loaded["verdict"] != _INCOMPLETE_RECEIPT_VERDICT
        or loaded["reason"] != _INCOMPLETE_RECEIPT_REASON
        or loaded["retry_policy"] != _INCOMPLETE_RECEIPT_RETRY_POLICY
        or loaded["fit_admitted"] is not False
    ):
        raise ValueError("failure receipt contract fields are malformed")
    allow_sentinel = loaded["schema_version"] == _INCOMPLETE_RECEIPT_SCHEMA
    for name in _INCOMPLETE_RECEIPT_SHA_FIELDS:
        value = loaded[name]
        if not isinstance(value, str):
            raise ValueError(f"failure receipt {name} must be a SHA-256 digest")
        require_sha256(value, field=name)
    block_id = loaded["block_id"]
    if not isinstance(block_id, str) or not block_id:
        raise ValueError("failure receipt block_id is malformed")
    phase = loaded["phase"]
    view_digest = loaded["observed_view_digest"]
    if phase not in _INCOMPLETE_RECEIPT_PHASES:
        raise ValueError("failure receipt phase is malformed")
    if (phase == "training") != (view_digest is None):
        raise ValueError("failure receipt view digest does not match its phase")
    if view_digest is not None:
        if not isinstance(view_digest, str):
            raise ValueError("failure receipt view digest is malformed")
        require_sha256(view_digest, field="observed_view_digest")
    pairs = loaded["pairs"]
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("failure receipt requires ineligible pair records")
    if type(loaded["ineligible_pairs"]) is not int or loaded["ineligible_pairs"] != len(
        pairs
    ):
        raise ValueError("failure receipt pair count mismatch")
    for pair in pairs:
        _validate_incomplete_pair_record(pair, allow_sentinel=allow_sentinel)
    if loaded["pairs_digest"] != content_digest(pairs):
        raise ValueError("failure receipt pairs digest mismatch")
    body = {k: v for k, v in loaded.items() if k != "receipt_digest"}
    receipt_digest = loaded["receipt_digest"]
    if not isinstance(receipt_digest, str) or receipt_digest != content_digest(body):
        raise ValueError("failure receipt digest mismatch")
    if (
        expected_receipt_digest is not None
        and receipt_digest != expected_receipt_digest
    ):
        raise ValueError("failure receipt differs from the pinned digest")
    return dict(loaded)


def load_incomplete_pair_attempt(
    root: str | Path,
    *,
    expected_receipt_digest: str | None = None,
    max_receipt_bytes: int = _DEFAULT_INCOMPLETE_RECEIPT_BYTES,
) -> dict[str, object]:
    """Strictly reload a TD8 failure receipt; reject any tampering.

    The root must contain exactly the canonical receipt file. The self digest,
    pair digest, pair count, fixed ``INVALID / BLOCKED`` verdict, no-fit and
    no-restart policy, and every pair record's inadmissibility are re-verified
    by the same validator the writer applies before it claims the root. Both
    the current v3 and the earlier v2 schema are accepted; v2 receipts are
    validated without the v3-only ``nonfinite:<kind>`` and quarantine
    encodings. Numeric strings are parsed under a bounded grammar, so hostile
    receipts are rejected quickly. Pass ``expected_receipt_digest`` (the value
    returned at write time) to also reject a fully re-sealed forgery.

    The default read budget is 64 MiB (not an invariant of older receipts).
    An explicitly pinned trusted digest is required to request a larger finite
    budget, up to 512 MiB. Reading a larger file never falls back to an
    unbounded allocation, and changed regular-file metadata fails closed.
    """
    if (
        type(max_receipt_bytes) is not int
        or not 0 < max_receipt_bytes <= _MAX_INCOMPLETE_RECEIPT_BYTES
    ):
        raise ValueError("max_receipt_bytes must be a positive bounded integer")
    if (
        max_receipt_bytes > _DEFAULT_INCOMPLETE_RECEIPT_BYTES
        and expected_receipt_digest is None
    ):
        raise ValueError("larger receipt byte budget requires a pinned digest")
    if expected_receipt_digest is not None:
        require_sha256(expected_receipt_digest, field="expected_receipt_digest")
    directory = Path(root)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("failure receipt root must be a real directory")
    # Verify the single-file root without Path.iterdir()'s eager full scan.
    with os.scandir(directory) as entries:
        first = next(entries, None)
        if (
            first is None
            or first.name != _INCOMPLETE_RECEIPT_FILE
            or next(entries, None) is not None
        ):
            raise ValueError("failure receipt root contents are malformed")
    with open_regular_binary(
        directory / _INCOMPLETE_RECEIPT_FILE, field="failure receipt"
    ) as handle:
        before = os.fstat(handle.fileno())
        if before.st_size > max_receipt_bytes:
            raise ValueError("failure receipt exceeds configured byte budget")
        # Bounded even if the file grows between the first fstat and read.
        raw = handle.read(max_receipt_bytes + 1)
        if len(raw) > max_receipt_bytes:
            raise ValueError("failure receipt exceeds configured byte budget")
        if handle.read(1):
            raise ValueError("failure receipt changed during read")
        after = os.fstat(handle.fileno())
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if len(raw) != before.st_size or before_identity != after_identity:
            raise ValueError("failure receipt changed during read")
    return _decode_incomplete_receipt(
        raw, expected_receipt_digest=expected_receipt_digest
    )


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
    label_evidence_digest: str
    label_bound_reference_digest: str


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
        require_v4_evidence=True,
        study_protocol_digest=protocol.protocol_digest,
    )
    _require_complete_reachable_labels(collection, block_id=block.block_id)
    if not collection.samples:
        raise ValueError(f"block {block.block_id} produced no admissible labels")
    if len(collection.eligible_pair_outcomes) != len(collection.samples):
        raise ValueError("TD8 v4 eligible evidence must cover every training row")
    label_evidence_digest = content_digest(collection.eligible_pair_outcomes)
    # Bind the model to the *observed* teacher/outcome population, not only the
    # common recipe. The pre-binding receipts retain the original recipe root
    # to avoid a circular hash dependency.
    bound_ref_digest = content_digest(
        {
            "schema_version": "td8_reference_with_label_population_v1",
            "reference_recipe_digest": ref_digest,
            "label_evidence_digest": label_evidence_digest,
        }
    )
    bound_samples = tuple(
        replace(
            sample,
            reference_policy_digest=bound_ref_digest,
            reference_policy_recipe_digest=ref_digest,
            label_population_digest=label_evidence_digest,
        )
        for sample in collection.samples
    )
    batch = build_synthetic_paired_label_batch(
        bound_samples,
        fit_cutoff=fit_cutoff,
        evaluation_start=eval_start,
        expected_reference_policy_digest=bound_ref_digest,
        require_v4_evidence=True,
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
            samples=bound_samples,
            fit_model=fit_for_arm,
            make_policy=policy_for_arm,
            grid=protocol.margin_multiplier_grid,
            calibration_fraction=protocol.margin_calibration_fraction,
            horizon_bars=plan.horizon_bars,
            initial_capital=plan.initial_capital,
            execution_cost=protocol.base_execution_cost,
            risk_config=protocol.risk_config,
            reference_policy_digest=bound_ref_digest,
            raw_dataset=raw_dataset,
            train_start_index=block.train_start_index,
            feature_clip=protocol.feature_clip,
            require_v4_evidence=True,
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
        label_evidence_digest=label_evidence_digest,
        label_bound_reference_digest=bound_ref_digest,
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
    # Resolve the source/runtime identity once, before any work, so that a
    # failure on the abort path can never replace the incomplete-label error
    # or skip its receipt (a failure here simply aborts a not-yet-started run).
    run_provenance = build_candidate_run_provenance()

    def _abort_with_receipt(
        failure: IncompletePairedLabelsError,
        *,
        phase: str,
        view: MarketDataset | None,
    ) -> None:
        if output_dir is None:
            return
        failure.receipt_digest = _persist_incomplete_pair_attempt(
            Path(output_dir),
            failure,
            dataset_id=dataset.dataset_id,
            protocol_digest=protocol.protocol_digest,
            actual_artifact_digest=resolved_artifact_digest,
            implementation_digest=str(run_provenance["implementation_digest"]),
            runtime_environment_digest=str(
                run_provenance["runtime_environment_digest"]
            ),
            phase=phase,
            observed_view_digest=(None if view is None else _dataset_view_digest(view)),
        )

    for block in protocol.blocks:
        block_t0 = time.perf_counter()

        # 1-4. Uniform feature transform, B0, reachable-state labels, three heads,
        #      and the train-internal margin selection.
        try:
            training = _train_block(dataset, protocol, block, train_switch_cost)
        except IncompletePairedLabelsError as failure:
            _abort_with_receipt(failure, phase="training", view=None)
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
        try:
            perturbed = _train_block(
                dataset_perturbed, protocol, block, train_switch_cost
            )
        except IncompletePairedLabelsError as failure:
            # Same write-once root and v3 receipt; the phase and the perturbed
            # source identity are bound into the digest.
            _abort_with_receipt(
                failure, phase="causality_probe", view=dataset_perturbed
            )
            raise
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
                label_evidence_digest=training.label_evidence_digest,
                label_bound_reference_digest=training.label_bound_reference_digest,
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
        schema_version="decision_value_dev_study_result_v4_software",
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
    "load_incomplete_pair_attempt",
    "run_decision_value_dev_study",
]
