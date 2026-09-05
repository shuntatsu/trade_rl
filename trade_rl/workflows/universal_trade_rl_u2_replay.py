"""Deterministic synthetic Development replay boundary for Universal Trade RL U2."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final
from weakref import WeakValueDictionary

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.market import MarketDataset
from trade_rl.domain.common import require_sha256
from trade_rl.risk.pretrade import RiskConstrainedTarget
from trade_rl.rl.universal_normalization import UniversalTradeSequenceNormalizer
from trade_rl.rl.universal_trade_contract import UniversalTradePolicyContract
from trade_rl.rl.universal_trade_environment import (
    UniversalTradeEnvironment,
    UniversalTradeMarketEnv,
)
from trade_rl.simulation.execution import ExecutionResult
from trade_rl.simulation.stateful_execution import StatefulExecutionResult
from trade_rl.workflows.universal_trade_rl_u1_contract import (
    UniversalTradeRLU1Contract,
    require_universal_trade_rl_u1_environment_contract,
)
from trade_rl.workflows.universal_trade_rl_u2_contract import (
    U2_TRAINING_SEEDS,
    UniversalTradeRLU2Contract,
)
from trade_rl.workflows.universal_trade_rl_u2_evaluation import (
    UniversalTradeRLU2DevelopmentScopeClosure,
    UniversalTradeRLU2EvaluationScope,
    build_universal_trade_rl_u2_development_scope_closure,
)
from trade_rl.workflows.universal_trade_rl_u2_evaluation_dataset import (
    U2EvaluationSourceArtifactLoader,
    U2EvaluationSourceArtifactLocator,
    load_universal_trade_rl_u2_development_evaluation_datasets,
)
from trade_rl.workflows.universal_trade_rl_u2_time_partition import (
    UniversalTradeRLU2TimePartition,
)
from trade_rl.workflows.universal_trade_rl_universe_manifest import (
    UniversalTradeRLUniverseManifest,
)

U2ReplayEnvironmentFactory = Callable[[MarketDataset], UniversalTradeEnvironment]
U2_REPLAY_EVIDENCE_SCHEMA: Final = "universal_trade_rl_u2_replay_evidence_v1"
_DIAGNOSTIC_TOLERANCE: Final = 1e-6
_TRANSITION_CLASSES: Final = frozenset(
    {"flat", "entry", "exit", "flip", "rebalance", "hold"}
)


class UniversalTradeRLU2ReplayVariant(str, Enum):
    """The four preregistered deterministic Development replay variants."""

    CANDIDATE = "candidate"
    CASH = "cash"
    CONSTANT_LONG = "constant_long"
    CONSTANT_SHORT = "constant_short"


_BASELINE_ACTIONS = {
    UniversalTradeRLU2ReplayVariant.CASH: np.asarray([0.0], dtype=np.float32),
    UniversalTradeRLU2ReplayVariant.CONSTANT_LONG: np.asarray([1.0], dtype=np.float32),
    UniversalTradeRLU2ReplayVariant.CONSTANT_SHORT: np.asarray(
        [-1.0], dtype=np.float32
    ),
}


def _single_symbol_value(value: object, *, field_name: str) -> float:
    try:
        vector = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as error:
        raise ValueError(f"U2 replay {field_name} must be numeric") from error
    if vector.shape != (1,) or not np.isfinite(vector).all():
        raise ValueError(f"U2 replay {field_name} must be one finite symbol value")
    return float(vector[0])


def _finite_number(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"U2 replay {field_name} must be numeric")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"U2 replay {field_name} must be finite")
    return resolved


def _non_negative_count(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"U2 replay {field_name} must be a non-negative integer")
    return value


def _change_count(values: tuple[float, ...]) -> int:
    previous = 0.0
    count = 0
    for value in values:
        if abs(value - previous) > _DIAGNOSTIC_TOLERANCE:
            count += 1
        previous = value
    return count


def _transition_class(before: float, after: float) -> str:
    before_nonflat = abs(before) > _DIAGNOSTIC_TOLERANCE
    after_nonflat = abs(after) > _DIAGNOSTIC_TOLERANCE
    if not before_nonflat and not after_nonflat:
        return "flat"
    if not before_nonflat and after_nonflat:
        return "entry"
    if before_nonflat and not after_nonflat:
        return "exit"
    if before * after < 0.0:
        return "flip"
    if abs(after - before) > _DIAGNOSTIC_TOLERANCE:
        return "rebalance"
    return "hold"


def _hard_risk_violation(
    *,
    projected_target: float,
    risk_scale: float,
    max_abs_weight: float,
    max_gross: float,
    fail_closed_tolerance: float,
) -> bool:
    return bool(
        abs(projected_target) > max_abs_weight * risk_scale + fail_closed_tolerance
        or abs(projected_target) > max_gross * risk_scale + fail_closed_tolerance
        or (risk_scale == 0.0 and abs(projected_target) > fail_closed_tolerance)
    )


@dataclass(frozen=True, slots=True)
class UniversalTradeRLU2ReplayRequest:
    """One paired candidate/baseline replay identity before numeric execution."""

    scope_digest: str
    policy_variant: UniversalTradeRLU2ReplayVariant
    evaluation_seed: int
    paired_candidate_checkpoint_digest: str

    def __post_init__(self) -> None:
        require_sha256(self.scope_digest, field="U2 replay scope digest")
        if not isinstance(self.policy_variant, UniversalTradeRLU2ReplayVariant):
            raise TypeError("U2 replay policy variant is invalid")
        if (
            isinstance(self.evaluation_seed, bool)
            or not isinstance(self.evaluation_seed, int)
            or self.evaluation_seed not in U2_TRAINING_SEEDS
        ):
            raise ValueError("U2 replay evaluation seed must be preregistered")
        require_sha256(
            self.paired_candidate_checkpoint_digest,
            field="U2 replay paired candidate checkpoint digest",
        )


@dataclass(frozen=True, slots=True)
class UniversalTradeRLU2ReplayStepEvidence:
    """One decision's maintained U1 action/risk/execution lifecycle evidence."""

    decision_bar_index: int
    normalized_action: float
    submitted_target: float
    executed_target: float
    risk_projected_target: float
    realized_exposure: float
    requested_turnover: float
    filled_turnover: float
    requested_notional: float
    filled_notional: float
    fill_count: int
    rejected_count: int
    rejection_reasons: tuple[str, ...]
    emergency_deleverage: bool
    liquidation_requested_turnover: float
    liquidation_filled_turnover: float
    liquidation_requested_notional: float
    liquidation_filled_notional: float
    liquidation_fill_count: int
    risk_scale: float
    max_abs_weight: float
    max_gross: float
    fail_closed_tolerance: float
    risk_reasons: tuple[str, ...]
    hard_risk_violation: bool
    transition_class: str

    def __post_init__(self) -> None:
        _non_negative_count(self.decision_bar_index, field_name="decision bar index")
        for field_name, value in (
            ("normalized_action", self.normalized_action),
            ("submitted_target", self.submitted_target),
            ("executed_target", self.executed_target),
            ("risk_projected_target", self.risk_projected_target),
            ("realized_exposure", self.realized_exposure),
            ("requested_turnover", self.requested_turnover),
            ("filled_turnover", self.filled_turnover),
            ("requested_notional", self.requested_notional),
            ("filled_notional", self.filled_notional),
            ("liquidation_requested_turnover", self.liquidation_requested_turnover),
            ("liquidation_filled_turnover", self.liquidation_filled_turnover),
            ("liquidation_requested_notional", self.liquidation_requested_notional),
            ("liquidation_filled_notional", self.liquidation_filled_notional),
            ("risk_scale", self.risk_scale),
            ("max_abs_weight", self.max_abs_weight),
            ("max_gross", self.max_gross),
            ("fail_closed_tolerance", self.fail_closed_tolerance),
        ):
            if not math.isfinite(value):
                raise ValueError(f"U2 replay step {field_name} must be finite")
        if abs(self.normalized_action) > 1.0:
            raise ValueError("U2 replay step normalized action is outside [-1, 1]")
        if any(
            value < 0.0
            for value in (
                self.requested_turnover,
                self.filled_turnover,
                self.requested_notional,
                self.filled_notional,
                self.liquidation_requested_turnover,
                self.liquidation_filled_turnover,
                self.liquidation_requested_notional,
                self.liquidation_filled_notional,
                self.fail_closed_tolerance,
            )
        ):
            raise ValueError(
                "U2 replay step turnover/notional/tolerance cannot be negative"
            )
        if not 0.0 <= self.risk_scale <= 1.0:
            raise ValueError("U2 replay step risk scale must be within [0, 1]")
        if self.max_abs_weight <= 0.0 or self.max_gross <= 0.0:
            raise ValueError("U2 replay step hard-risk limits must be positive")
        _non_negative_count(self.fill_count, field_name="step fill count")
        _non_negative_count(
            self.liquidation_fill_count,
            field_name="step liquidation fill count",
        )
        _non_negative_count(self.rejected_count, field_name="step rejected count")
        if not isinstance(self.emergency_deleverage, bool):
            raise TypeError("U2 replay step emergency-deleverage flag must be boolean")
        if not self.emergency_deleverage and (
            self.liquidation_requested_turnover != 0.0
            or self.liquidation_filled_turnover != 0.0
            or self.liquidation_requested_notional != 0.0
            or self.liquidation_filled_notional != 0.0
            or self.liquidation_fill_count != 0
        ):
            raise ValueError(
                "U2 replay step liquidation evidence requires emergency deleveraging"
            )
        if not isinstance(self.rejection_reasons, tuple) or any(
            not isinstance(reason, str) or not reason
            for reason in self.rejection_reasons
        ):
            raise ValueError("U2 replay step rejection reasons are malformed")
        if len(self.rejection_reasons) != self.rejected_count:
            raise ValueError("U2 replay step rejected count does not match reasons")
        if not isinstance(self.risk_reasons, tuple) or any(
            not isinstance(reason, str) or not reason for reason in self.risk_reasons
        ):
            raise ValueError("U2 replay step risk reasons are malformed")
        if not isinstance(self.hard_risk_violation, bool):
            raise TypeError("U2 replay step hard-risk violation must be boolean")
        expected_violation = _hard_risk_violation(
            projected_target=self.risk_projected_target,
            risk_scale=self.risk_scale,
            max_abs_weight=self.max_abs_weight,
            max_gross=self.max_gross,
            fail_closed_tolerance=self.fail_closed_tolerance,
        )
        if self.hard_risk_violation is not expected_violation:
            raise ValueError("U2 replay step hard-risk evidence is inconsistent")
        if self.transition_class not in _TRANSITION_CLASSES:
            raise ValueError("U2 replay step transition class is invalid")

    def to_payload(self) -> dict[str, object]:
        return {
            "decision_bar_index": self.decision_bar_index,
            "normalized_action": self.normalized_action,
            "submitted_target": self.submitted_target,
            "executed_target": self.executed_target,
            "risk_projected_target": self.risk_projected_target,
            "realized_exposure": self.realized_exposure,
            "requested_turnover": self.requested_turnover,
            "filled_turnover": self.filled_turnover,
            "requested_notional": self.requested_notional,
            "filled_notional": self.filled_notional,
            "fill_count": self.fill_count,
            "rejected_count": self.rejected_count,
            "rejection_reasons": self.rejection_reasons,
            "emergency_deleverage": self.emergency_deleverage,
            "liquidation_requested_turnover": self.liquidation_requested_turnover,
            "liquidation_filled_turnover": self.liquidation_filled_turnover,
            "liquidation_requested_notional": self.liquidation_requested_notional,
            "liquidation_filled_notional": self.liquidation_filled_notional,
            "liquidation_fill_count": self.liquidation_fill_count,
            "risk_scale": self.risk_scale,
            "max_abs_weight": self.max_abs_weight,
            "max_gross": self.max_gross,
            "fail_closed_tolerance": self.fail_closed_tolerance,
            "risk_reasons": self.risk_reasons,
            "hard_risk_violation": self.hard_risk_violation,
            "transition_class": self.transition_class,
        }


@dataclass(frozen=True, slots=True)
class UniversalTradeRLU2ReplayEvidence:
    """Content-addressed raw net-economic evidence for one U2 replay scope."""

    scope_closure_digest: str
    scope_digest: str
    universe_manifest_digest: str
    u1_contract_digest: str
    u2_contract_digest: str
    source_dataset_digest: str
    evaluation_dataset_digest: str
    concrete_symbol: str
    symbol_role: str
    cell: str
    source_window: str
    tile_index: int
    policy_variant: str
    evaluation_seed: int
    paired_candidate_checkpoint_digest: str
    outcome_start_bar_index: int
    outcome_stop_bar_index_exclusive: int
    evaluation_start_bar_index: int
    evaluation_stop_bar_index: int
    runtime_start_bar_index: int
    runtime_end_bar_index: int
    final_current_bar_index: int
    observed_decision_count: int
    normal_completion: bool
    terminated: bool
    truncated: bool
    termination_reason: str | None
    terminal_accounting_mode: str
    terminal_liquidation_cost: float
    initial_capital: float
    final_net_portfolio_value: float
    net_wealth_ratio: float
    net_simple_returns: tuple[float, ...]
    maximum_drawdown: float
    turnover_total: float
    total_execution_cost: float
    funding_pnl: float
    borrow_cost: float
    trade_count: int
    rebalance_count: int
    fill_count: int
    target_change_count: int
    submitted_change_count: int
    executed_change_count: int
    sign_flip_count: int
    hard_risk_violation_count: int
    execution_rejection_count: int
    normalized_action_trace: tuple[float, ...]
    realized_exposure_trace: tuple[float, ...]
    step_evidence: tuple[UniversalTradeRLU2ReplayStepEvidence, ...]
    schema_version: str = U2_REPLAY_EVIDENCE_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != U2_REPLAY_EVIDENCE_SCHEMA:
            raise ValueError("U2 replay evidence schema is unsupported")
        for field_name, value in (
            ("scope_closure_digest", self.scope_closure_digest),
            ("scope_digest", self.scope_digest),
            ("universe_manifest_digest", self.universe_manifest_digest),
            ("u1_contract_digest", self.u1_contract_digest),
            ("u2_contract_digest", self.u2_contract_digest),
            ("source_dataset_digest", self.source_dataset_digest),
            ("evaluation_dataset_digest", self.evaluation_dataset_digest),
            (
                "paired_candidate_checkpoint_digest",
                self.paired_candidate_checkpoint_digest,
            ),
        ):
            require_sha256(value, field=f"U2 replay evidence {field_name}")
        for field_name, value in (
            ("concrete_symbol", self.concrete_symbol),
            ("symbol_role", self.symbol_role),
            ("cell", self.cell),
            ("source_window", self.source_window),
            ("policy_variant", self.policy_variant),
            ("terminal_accounting_mode", self.terminal_accounting_mode),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"U2 replay evidence {field_name} must be non-empty")
        if self.policy_variant not in {
            variant.value for variant in UniversalTradeRLU2ReplayVariant
        }:
            raise ValueError("U2 replay evidence policy variant is invalid")
        if self.termination_reason is not None and (
            not isinstance(self.termination_reason, str) or not self.termination_reason
        ):
            raise ValueError("U2 replay evidence termination reason is invalid")
        for field_name, value in (
            ("tile_index", self.tile_index),
            ("outcome_start_bar_index", self.outcome_start_bar_index),
            (
                "outcome_stop_bar_index_exclusive",
                self.outcome_stop_bar_index_exclusive,
            ),
            ("evaluation_start_bar_index", self.evaluation_start_bar_index),
            ("evaluation_stop_bar_index", self.evaluation_stop_bar_index),
            ("runtime_start_bar_index", self.runtime_start_bar_index),
            ("runtime_end_bar_index", self.runtime_end_bar_index),
            ("final_current_bar_index", self.final_current_bar_index),
            ("observed_decision_count", self.observed_decision_count),
            ("trade_count", self.trade_count),
            ("rebalance_count", self.rebalance_count),
            ("fill_count", self.fill_count),
            ("target_change_count", self.target_change_count),
            ("submitted_change_count", self.submitted_change_count),
            ("executed_change_count", self.executed_change_count),
            ("sign_flip_count", self.sign_flip_count),
            ("hard_risk_violation_count", self.hard_risk_violation_count),
            ("execution_rejection_count", self.execution_rejection_count),
        ):
            _non_negative_count(value, field_name=f"evidence {field_name}")
        if (
            isinstance(self.evaluation_seed, bool)
            or not isinstance(self.evaluation_seed, int)
            or self.evaluation_seed not in U2_TRAINING_SEEDS
        ):
            raise ValueError("U2 replay evidence evaluation seed must be preregistered")
        if self.outcome_stop_bar_index_exclusive <= self.outcome_start_bar_index:
            raise ValueError("U2 replay outcome interval is empty or reversed")
        if self.evaluation_start_bar_index != self.outcome_start_bar_index - 1:
            raise ValueError("U2 replay evaluation start boundary is inconsistent")
        if self.evaluation_stop_bar_index != self.outcome_stop_bar_index_exclusive:
            raise ValueError("U2 replay evaluation stop must remain exclusive")
        if self.runtime_start_bar_index != self.evaluation_start_bar_index:
            raise ValueError("U2 replay runtime start boundary is inconsistent")
        if self.runtime_end_bar_index != self.outcome_stop_bar_index_exclusive - 1:
            raise ValueError("U2 replay runtime end must be inclusive O_stop - 1")
        if self.final_current_bar_index > self.runtime_end_bar_index:
            raise ValueError("U2 replay final current index exceeded runtime end")
        expected_decisions = (
            self.outcome_stop_bar_index_exclusive - self.outcome_start_bar_index
        )
        if self.observed_decision_count > expected_decisions:
            raise ValueError("U2 replay observed more decisions than the outcome scope")
        for field_name, value in (
            ("terminal_liquidation_cost", self.terminal_liquidation_cost),
            ("initial_capital", self.initial_capital),
            ("final_net_portfolio_value", self.final_net_portfolio_value),
            ("net_wealth_ratio", self.net_wealth_ratio),
            ("maximum_drawdown", self.maximum_drawdown),
            ("turnover_total", self.turnover_total),
            ("total_execution_cost", self.total_execution_cost),
            ("funding_pnl", self.funding_pnl),
            ("borrow_cost", self.borrow_cost),
        ):
            if not math.isfinite(value):
                raise ValueError(f"U2 replay evidence {field_name} must be finite")
        if self.initial_capital <= 0.0:
            raise ValueError("U2 replay evidence initial capital must be positive")
        if self.final_net_portfolio_value <= 0.0 or self.net_wealth_ratio <= 0.0:
            raise ValueError("U2 replay evidence final net wealth must stay positive")
        if self.terminal_liquidation_cost < 0.0:
            raise ValueError("U2 replay terminal liquidation cost cannot be negative")
        if not 0.0 <= self.maximum_drawdown <= 1.0:
            raise ValueError("U2 replay maximum drawdown must be within [0, 1]")
        if self.turnover_total < 0.0 or self.total_execution_cost < 0.0:
            raise ValueError("U2 replay turnover and execution cost cannot be negative")
        if self.borrow_cost < 0.0:
            raise ValueError("U2 replay borrow cost cannot be negative")
        if not all(
            isinstance(value, bool)
            for value in (self.normal_completion, self.terminated, self.truncated)
        ):
            raise TypeError("U2 replay completion flags must be booleans")
        if self.terminated and self.truncated:
            raise ValueError("U2 replay cannot be both terminated and truncated")
        if self.terminated and self.termination_reason is None:
            raise ValueError("U2 terminated replay requires a termination reason")

        for field_name, values in (
            ("net_simple_returns", self.net_simple_returns),
            ("normalized_action_trace", self.normalized_action_trace),
            ("realized_exposure_trace", self.realized_exposure_trace),
        ):
            if not isinstance(values, tuple) or not all(
                math.isfinite(value) for value in values
            ):
                raise ValueError(
                    f"U2 replay evidence {field_name} must be finite tuple"
                )
            if len(values) != self.observed_decision_count:
                raise ValueError(
                    f"U2 replay evidence {field_name} length must match decisions"
                )
        if any(abs(value) > 1.0 for value in self.normalized_action_trace):
            raise ValueError("U2 replay normalized action trace is outside [-1, 1]")
        if any(value <= -1.0 for value in self.net_simple_returns):
            raise ValueError("U2 replay returns violate positive-wealth U1 semantics")

        if not isinstance(self.step_evidence, tuple) or not all(
            isinstance(step, UniversalTradeRLU2ReplayStepEvidence)
            for step in self.step_evidence
        ):
            raise TypeError("U2 replay step evidence must be an immutable step tuple")
        if len(self.step_evidence) != self.observed_decision_count:
            raise ValueError("U2 replay step evidence length must match decisions")
        for offset, step in enumerate(self.step_evidence):
            if step.decision_bar_index != self.runtime_start_bar_index + offset:
                raise ValueError("U2 replay step evidence bar alignment drifted")

        action_trace = tuple(step.normalized_action for step in self.step_evidence)
        submitted_trace = tuple(step.submitted_target for step in self.step_evidence)
        executed_trace = tuple(step.executed_target for step in self.step_evidence)
        exposure_trace = tuple(step.realized_exposure for step in self.step_evidence)
        if action_trace != self.normalized_action_trace:
            raise ValueError("U2 replay action trace does not match step evidence")
        if exposure_trace != self.realized_exposure_trace:
            raise ValueError("U2 replay exposure trace does not match step evidence")
        if self.target_change_count != _change_count(action_trace):
            raise ValueError("U2 replay target-change count is inconsistent")
        if self.submitted_change_count != _change_count(submitted_trace):
            raise ValueError("U2 replay submitted-change count is inconsistent")
        if self.executed_change_count != _change_count(executed_trace):
            raise ValueError("U2 replay executed-change count is inconsistent")
        expected_sign_flips = sum(
            step.transition_class == "flip" for step in self.step_evidence
        )
        if self.sign_flip_count != expected_sign_flips:
            raise ValueError("U2 replay sign-flip count is inconsistent")
        expected_hard_risk = sum(
            step.hard_risk_violation for step in self.step_evidence
        )
        if self.hard_risk_violation_count != expected_hard_risk:
            raise ValueError("U2 replay hard-risk count is inconsistent")
        expected_rejections = sum(step.rejected_count for step in self.step_evidence)
        if self.execution_rejection_count != expected_rejections:
            raise ValueError("U2 replay execution rejection count is inconsistent")
        expected_fills = sum(
            step.fill_count + step.liquidation_fill_count for step in self.step_evidence
        )
        if self.fill_count != expected_fills:
            raise ValueError("U2 replay fill count is inconsistent")
        if self.trade_count != self.fill_count:
            raise ValueError("U2 replay trade/fill ledger counts are inconsistent")

        expected_ratio = self.final_net_portfolio_value / self.initial_capital
        if not math.isclose(
            expected_ratio,
            self.net_wealth_ratio,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("U2 replay final wealth ratio is inconsistent")
        wealth_from_returns = math.prod(
            1.0 + value for value in self.net_simple_returns
        )
        if not math.isclose(
            wealth_from_returns,
            self.net_wealth_ratio,
            rel_tol=0.0,
            abs_tol=1e-10,
        ):
            raise ValueError("U2 replay simple returns do not reconcile to net wealth")
        if self.normal_completion:
            if self.observed_decision_count != expected_decisions:
                raise ValueError("U2 normal replay decision count is incomplete")
            if self.terminated or not self.truncated:
                raise ValueError("U2 normal replay completion flags are invalid")
            if self.termination_reason is not None:
                raise ValueError("U2 normal replay cannot have termination reason")
            if self.final_current_bar_index != self.runtime_end_bar_index:
                raise ValueError("U2 normal replay did not finish on runtime end")
            if self.terminal_accounting_mode != "mark_to_market":
                raise ValueError(
                    "U2 normal replay must use mark-to-market terminal accounting"
                )
            if self.terminal_liquidation_cost != 0.0:
                raise ValueError("U2 normal replay cannot charge terminal liquidation")

        expected_digest = content_digest(self.to_payload(include_digest=False))
        if self.digest:
            require_sha256(self.digest, field="U2 replay evidence digest")
            if self.digest != expected_digest:
                raise ValueError("U2 replay evidence digest mismatch")
        object.__setattr__(self, "digest", expected_digest)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "scope_closure_digest": self.scope_closure_digest,
            "scope_digest": self.scope_digest,
            "universe_manifest_digest": self.universe_manifest_digest,
            "u1_contract_digest": self.u1_contract_digest,
            "u2_contract_digest": self.u2_contract_digest,
            "source_dataset_digest": self.source_dataset_digest,
            "evaluation_dataset_digest": self.evaluation_dataset_digest,
            "concrete_symbol": self.concrete_symbol,
            "symbol_role": self.symbol_role,
            "cell": self.cell,
            "source_window": self.source_window,
            "tile_index": self.tile_index,
            "policy_variant": self.policy_variant,
            "evaluation_seed": self.evaluation_seed,
            "paired_candidate_checkpoint_digest": (
                self.paired_candidate_checkpoint_digest
            ),
            "outcome_start_bar_index": self.outcome_start_bar_index,
            "outcome_stop_bar_index_exclusive": self.outcome_stop_bar_index_exclusive,
            "evaluation_start_bar_index": self.evaluation_start_bar_index,
            "evaluation_stop_bar_index": self.evaluation_stop_bar_index,
            "runtime_start_bar_index": self.runtime_start_bar_index,
            "runtime_end_bar_index": self.runtime_end_bar_index,
            "final_current_bar_index": self.final_current_bar_index,
            "observed_decision_count": self.observed_decision_count,
            "normal_completion": self.normal_completion,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "termination_reason": self.termination_reason,
            "terminal_accounting_mode": self.terminal_accounting_mode,
            "terminal_liquidation_cost": self.terminal_liquidation_cost,
            "initial_capital": self.initial_capital,
            "final_net_portfolio_value": self.final_net_portfolio_value,
            "net_wealth_ratio": self.net_wealth_ratio,
            "net_simple_returns": self.net_simple_returns,
            "maximum_drawdown": self.maximum_drawdown,
            "turnover_total": self.turnover_total,
            "total_execution_cost": self.total_execution_cost,
            "funding_pnl": self.funding_pnl,
            "borrow_cost": self.borrow_cost,
            "trade_count": self.trade_count,
            "rebalance_count": self.rebalance_count,
            "fill_count": self.fill_count,
            "target_change_count": self.target_change_count,
            "submitted_change_count": self.submitted_change_count,
            "executed_change_count": self.executed_change_count,
            "sign_flip_count": self.sign_flip_count,
            "hard_risk_violation_count": self.hard_risk_violation_count,
            "execution_rejection_count": self.execution_rejection_count,
            "normalized_action_trace": self.normalized_action_trace,
            "realized_exposure_trace": self.realized_exposure_trace,
            "step_evidence": tuple(step.to_payload() for step in self.step_evidence),
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


@dataclass(slots=True)
class UniversalTradeRLU2DevelopmentReplaySession:
    """Own one canonical synthetic Development dataset generation for replay."""

    manifest: UniversalTradeRLUniverseManifest
    time_partition: UniversalTradeRLU2TimePartition
    u2_contract: UniversalTradeRLU2Contract
    u1_contract: UniversalTradeRLU1Contract
    policy_contract: UniversalTradePolicyContract
    normalizer: UniversalTradeSequenceNormalizer
    scope_closure: UniversalTradeRLU2DevelopmentScopeClosure
    datasets: dict[str, MarketDataset]
    environment_factory: U2ReplayEnvironmentFactory
    _issued_environments: WeakValueDictionary[int, UniversalTradeEnvironment] = field(
        default_factory=WeakValueDictionary,
        init=False,
        repr=False,
    )
    _issued_base_environments: WeakValueDictionary[int, UniversalTradeMarketEnv] = (
        field(
            default_factory=WeakValueDictionary,
            init=False,
            repr=False,
        )
    )

    @property
    def scope_closure_digest(self) -> str:
        return self.scope_closure.digest

    @property
    def evaluation_dataset_ids(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (symbol, dataset.dataset_id) for symbol, dataset in self.datasets.items()
        )

    def scope(self, scope_digest: str) -> UniversalTradeRLU2EvaluationScope:
        require_sha256(scope_digest, field="U2 replay scope digest")
        matches = tuple(
            scope for scope in self.scope_closure.scopes if scope.digest == scope_digest
        )
        if len(matches) != 1:
            raise ValueError("U2 replay scope is not present exactly once")
        return matches[0]

    def _require_canonical_scope(
        self,
        scope: UniversalTradeRLU2EvaluationScope,
    ) -> UniversalTradeRLU2EvaluationScope:
        if not isinstance(scope, UniversalTradeRLU2EvaluationScope):
            raise TypeError("U2 replay requires an evaluation scope")
        canonical = self.scope(scope.digest)
        if canonical != scope:
            raise ValueError(
                "U2 replay scope identity drifted from the canonical closure"
            )
        return canonical

    def _create_verified_environment(
        self,
        scope: UniversalTradeRLU2EvaluationScope,
    ) -> UniversalTradeEnvironment:
        canonical = self._require_canonical_scope(scope)
        dataset = self.datasets.get(canonical.concrete_symbol)
        if not isinstance(dataset, MarketDataset):
            raise TypeError("U2 replay scope dataset is unavailable")
        if dataset.symbols != (canonical.concrete_symbol,):
            raise ValueError("U2 replay dataset symbol mismatch")
        if dataset.dataset_id != canonical.evaluation_dataset_digest:
            raise ValueError("U2 replay dataset identity mismatch")

        environment = self.environment_factory(dataset)
        if not isinstance(environment, UniversalTradeEnvironment):
            raise TypeError(
                "U2 replay environment factory must return UniversalTradeEnvironment"
            )

        object_id = id(environment)
        existing = self._issued_environments.get(object_id)
        if existing is environment:
            raise ValueError(
                "U2 replay requires a fresh mutable U1 environment per variant"
            )
        if existing is not None:
            raise RuntimeError(
                "U2 replay observed a live U1 environment object-id collision"
            )

        base_environment = environment.base_env
        base_id = id(base_environment)
        existing_base = self._issued_base_environments.get(base_id)
        if existing_base is base_environment:
            raise ValueError(
                "U2 replay requires a fresh mutable U1 base environment per variant"
            )
        if existing_base is not None:
            raise RuntimeError("U2 replay observed a live U1 base object-id collision")

        try:
            require_universal_trade_rl_u1_environment_contract(
                contract=self.u1_contract,
                environment=environment,
            )
            if environment.contract.digest != self.policy_contract.digest:
                raise ValueError("U2 replay policy contract mismatch")
            child_normalizer = environment.sequence_normalizer
            if not isinstance(child_normalizer, UniversalTradeSequenceNormalizer):
                raise ValueError("U2 replay requires the frozen sequence normalizer")
            if child_normalizer.digest != self.normalizer.digest:
                raise ValueError("U2 replay normalizer generation mismatch")
            if child_normalizer.provenance_digest != self.normalizer.provenance_digest:
                raise ValueError("U2 replay normalizer provenance mismatch")
            if environment.dataset.dataset_id != canonical.evaluation_dataset_digest:
                raise ValueError("U2 replay environment dataset identity mismatch")
            if environment.dataset.symbols != (canonical.concrete_symbol,):
                raise ValueError("U2 replay environment dataset symbol mismatch")
        except Exception:
            environment.close()
            raise

        self._issued_environments[object_id] = environment
        self._issued_base_environments[base_id] = base_environment
        return environment

    def _reset_scope_environment(
        self,
        environment: UniversalTradeEnvironment,
        scope: UniversalTradeRLU2EvaluationScope,
        *,
        evaluation_seed: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        canonical = self._require_canonical_scope(scope)
        issued = self._issued_environments.get(id(environment))
        if issued is not environment:
            raise ValueError("U2 replay environment was not issued by this session")
        if (
            isinstance(evaluation_seed, bool)
            or not isinstance(evaluation_seed, int)
            or evaluation_seed not in self.u2_contract.training_seeds
        ):
            raise ValueError("U2 replay evaluation seed must be preregistered")

        observation, info = environment.reset(
            seed=evaluation_seed,
            options={"start_idx": canonical.evaluation_start_bar_index},
        )
        expected_start = canonical.evaluation_start_bar_index
        expected_end = canonical.outcome_stop_bar_index_exclusive - 1
        if info.get("start_index") != expected_start:
            raise RuntimeError("U2 replay reset start index drifted")
        if info.get("end_index") != expected_end:
            raise RuntimeError("U2 replay runtime end index drifted")
        base = environment.base_env
        if base.start_index != expected_start or base.current_index != expected_start:
            raise RuntimeError("U2 replay environment start state drifted")
        if base.end_index != expected_end:
            raise RuntimeError("U2 replay environment end state drifted")
        return observation, info

    @staticmethod
    def _candidate_action(model: Any, observation: dict[str, Any]) -> np.ndarray:
        predictor = getattr(model, "predict", None)
        if not callable(predictor):
            raise TypeError("U2 candidate replay model must expose predict()")
        raw_action, _state = predictor(observation, deterministic=True)
        action = np.asarray(raw_action, dtype=np.float32)
        if action.shape != (1,):
            raise ValueError("U2 candidate replay action must have shape (1,)")
        return action

    @staticmethod
    def _step_evidence(
        *,
        decision_bar_index: int,
        before_exposure: float,
        action: np.ndarray,
        info: Mapping[str, Any],
        runtime_exposure: float,
    ) -> UniversalTradeRLU2ReplayStepEvidence:
        risk = info.get("hybrid_risk")
        execution = info.get("hybrid_execution")
        if not isinstance(risk, RiskConstrainedTarget):
            raise TypeError("U2 replay requires maintained U1 hybrid_risk evidence")
        if not isinstance(execution, StatefulExecutionResult):
            raise TypeError(
                "U2 replay requires maintained U1 hybrid_execution evidence"
            )
        if risk.max_abs_weight is None or risk.max_gross is None:
            raise ValueError("U2 replay hard-risk limit metadata is missing")
        if risk.fail_closed_tolerance is None:
            raise ValueError("U2 replay hard-risk tolerance metadata is missing")

        normalized_action = _single_symbol_value(action, field_name="normalized action")
        submitted_target = _single_symbol_value(
            info.get("submitted_target"),
            field_name="submitted target",
        )
        executed_target = _single_symbol_value(
            info.get("executed_target"),
            field_name="executed target",
        )
        risk_projected_target = _single_symbol_value(
            risk.weights,
            field_name="risk-projected target",
        )
        realized_exposure = _single_symbol_value(
            info.get("effective_filled_weights"),
            field_name="effective filled exposure",
        )
        if not math.isclose(
            realized_exposure,
            runtime_exposure,
            rel_tol=0.0,
            abs_tol=1e-10,
        ):
            raise RuntimeError(
                "U2 replay maintained filled exposure drifted from runtime snapshot"
            )

        rejected_events = tuple(
            event for event in execution.order_events if event.event_type == "rejected"
        )
        rejection_reasons = tuple(
            "" if event.reason is None else str(event.reason)
            for event in rejected_events
        )
        rejected_count = _non_negative_count(
            execution.rejected_count,
            field_name="execution rejected count",
        )
        if len(rejected_events) != rejected_count:
            raise RuntimeError(
                "U2 replay maintained execution rejection count does not match events"
            )
        if any(not reason for reason in rejection_reasons):
            raise RuntimeError("U2 replay execution rejection lacks an explicit reason")

        emergency_deleverage = info.get("emergency_deleverage")
        if not isinstance(emergency_deleverage, bool):
            raise TypeError("U2 replay emergency-deleverage evidence must be boolean")
        liquidation = info.get("hybrid_liquidation")
        if liquidation is None:
            if emergency_deleverage:
                raise RuntimeError(
                    "U2 replay emergency deleveraging lacks liquidation evidence"
                )
            liquidation_requested_turnover = 0.0
            liquidation_filled_turnover = 0.0
            liquidation_requested_notional = 0.0
            liquidation_filled_notional = 0.0
            liquidation_fill_count = 0
        else:
            if not emergency_deleverage:
                raise RuntimeError(
                    "U2 replay liquidation evidence appeared outside emergency deleveraging"
                )
            if not isinstance(liquidation, ExecutionResult):
                raise TypeError("U2 replay hybrid_liquidation evidence is invalid")
            liquidation_requested_turnover = _finite_number(
                liquidation.requested_turnover,
                field_name="liquidation requested turnover",
            )
            liquidation_filled_turnover = _finite_number(
                liquidation.filled_turnover,
                field_name="liquidation filled turnover",
            )
            liquidation_requested_notional = _single_symbol_value(
                liquidation.requested_notional_by_symbol,
                field_name="liquidation requested notional",
            )
            liquidation_filled_notional = _single_symbol_value(
                liquidation.filled_notional_by_symbol,
                field_name="liquidation filled notional",
            )
            liquidation_fill_count = _non_negative_count(
                liquidation.fill_count,
                field_name="liquidation fill count",
            )

        risk_scale = _finite_number(risk.risk_scale, field_name="risk scale")
        max_abs_weight = _finite_number(
            risk.max_abs_weight,
            field_name="max abs weight",
        )
        max_gross = _finite_number(risk.max_gross, field_name="max gross")
        fail_closed_tolerance = _finite_number(
            risk.fail_closed_tolerance,
            field_name="fail-closed tolerance",
        )
        hard_risk_violation = _hard_risk_violation(
            projected_target=risk_projected_target,
            risk_scale=risk_scale,
            max_abs_weight=max_abs_weight,
            max_gross=max_gross,
            fail_closed_tolerance=fail_closed_tolerance,
        )
        return UniversalTradeRLU2ReplayStepEvidence(
            decision_bar_index=decision_bar_index,
            normalized_action=normalized_action,
            submitted_target=submitted_target,
            executed_target=executed_target,
            risk_projected_target=risk_projected_target,
            realized_exposure=realized_exposure,
            requested_turnover=_finite_number(
                execution.requested_turnover,
                field_name="requested turnover",
            ),
            filled_turnover=_finite_number(
                execution.filled_turnover,
                field_name="filled turnover",
            ),
            requested_notional=_finite_number(
                execution.requested_notional,
                field_name="requested notional",
            ),
            filled_notional=_finite_number(
                execution.filled_notional,
                field_name="filled notional",
            ),
            fill_count=_non_negative_count(
                execution.fill_count,
                field_name="execution fill count",
            ),
            rejected_count=rejected_count,
            rejection_reasons=rejection_reasons,
            emergency_deleverage=emergency_deleverage,
            liquidation_requested_turnover=liquidation_requested_turnover,
            liquidation_filled_turnover=liquidation_filled_turnover,
            liquidation_requested_notional=liquidation_requested_notional,
            liquidation_filled_notional=liquidation_filled_notional,
            liquidation_fill_count=liquidation_fill_count,
            risk_scale=risk_scale,
            max_abs_weight=max_abs_weight,
            max_gross=max_gross,
            fail_closed_tolerance=fail_closed_tolerance,
            risk_reasons=tuple(str(reason) for reason in risk.reasons),
            hard_risk_violation=hard_risk_violation,
            transition_class=_transition_class(before_exposure, realized_exposure),
        )

    def replay(
        self,
        request: UniversalTradeRLU2ReplayRequest,
        *,
        model: Any | None = None,
    ) -> UniversalTradeRLU2ReplayEvidence:
        """Replay one canonical U2 Development scope through the frozen U1 runtime."""

        if not isinstance(request, UniversalTradeRLU2ReplayRequest):
            raise TypeError("U2 replay request is invalid")
        if request.evaluation_seed not in self.u2_contract.training_seeds:
            raise ValueError("U2 replay evaluation seed is outside the U2 contract")
        candidate = request.policy_variant is UniversalTradeRLU2ReplayVariant.CANDIDATE
        if candidate and model is None:
            raise ValueError("U2 candidate replay requires a model")
        if not candidate and model is not None:
            raise ValueError("U2 baseline replay must not receive a candidate model")

        scope = self.scope(request.scope_digest)
        environment = self._create_verified_environment(scope)
        try:
            observation, _reset_info = self._reset_scope_environment(
                environment,
                scope,
                evaluation_seed=request.evaluation_seed,
            )
            base = environment.base_env
            initial_capital = float(base.hybrid.portfolio_value)
            if base.hybrid.returns_history:
                raise RuntimeError("U2 replay reset produced non-empty return history")

            steps: list[UniversalTradeRLU2ReplayStepEvidence] = []
            observed_decision_count = 0
            terminated = False
            truncated = False
            final_info: dict[str, Any] = {}

            while not terminated and not truncated:
                if observed_decision_count >= scope.decision_count:
                    raise RuntimeError(
                        "U2 replay exceeded the preregistered decision count"
                    )
                if candidate:
                    assert model is not None
                    action = self._candidate_action(model, observation)
                else:
                    action = _BASELINE_ACTIONS[request.policy_variant].copy()
                decision_bar_index = base.current_index
                before_exposure = float(
                    base.universal_trade_runtime_snapshot().current_weight
                )
                observation, _reward, terminated, truncated, info = environment.step(
                    action
                )
                observed_decision_count += 1
                runtime_exposure = float(
                    base.universal_trade_runtime_snapshot().current_weight
                )
                steps.append(
                    self._step_evidence(
                        decision_bar_index=decision_bar_index,
                        before_exposure=before_exposure,
                        action=action,
                        info=info,
                        runtime_exposure=runtime_exposure,
                    )
                )
                final_info = info

            final_current_bar_index = base.current_index
            runtime_start_bar_index = base.start_index
            runtime_end_bar_index = base.end_index
            book = base.hybrid
            net_simple_returns = tuple(float(value) for value in book.returns_history)
            if len(net_simple_returns) != observed_decision_count:
                raise RuntimeError(
                    "U2 replay return history length does not match decisions"
                )

            final_net_portfolio_value = float(book.portfolio_value)
            net_wealth_ratio = final_net_portfolio_value / initial_capital
            wealth_from_returns = math.prod(1.0 + value for value in net_simple_returns)
            if not math.isclose(
                wealth_from_returns,
                net_wealth_ratio,
                rel_tol=0.0,
                abs_tol=1e-10,
            ):
                raise RuntimeError(
                    "U2 replay simple returns do not reconcile to wealth"
                )

            raw_reason = final_info.get("termination_reason")
            termination_reason = (
                None
                if raw_reason is None
                else str(getattr(raw_reason, "value", raw_reason))
            )
            terminal_accounting_mode = str(
                final_info.get("terminal_accounting_mode", "")
            )
            terminal_liquidation_cost = float(
                final_info.get("terminal_liquidation_cost", float("nan"))
            )
            expected_runtime_end = scope.outcome_stop_bar_index_exclusive - 1
            normal_completion = bool(
                observed_decision_count == scope.decision_count
                and not terminated
                and truncated
                and runtime_start_bar_index == scope.evaluation_start_bar_index
                and runtime_end_bar_index == expected_runtime_end
                and final_current_bar_index == expected_runtime_end
                and termination_reason is None
                and terminal_accounting_mode == "mark_to_market"
                and terminal_liquidation_cost == 0.0
            )
            if truncated and not normal_completion:
                raise RuntimeError(
                    "U2 replay time-limit completion drifted from contract"
                )

            step_evidence = tuple(steps)
            normalized_action_trace = tuple(
                step.normalized_action for step in step_evidence
            )
            submitted_trace = tuple(step.submitted_target for step in step_evidence)
            executed_trace = tuple(step.executed_target for step in step_evidence)
            realized_exposure_trace = tuple(
                step.realized_exposure for step in step_evidence
            )
            return UniversalTradeRLU2ReplayEvidence(
                scope_closure_digest=self.scope_closure.digest,
                scope_digest=scope.digest,
                universe_manifest_digest=scope.universe_manifest_digest,
                u1_contract_digest=scope.u1_contract_digest,
                u2_contract_digest=scope.u2_contract_digest,
                source_dataset_digest=scope.source_dataset_digest,
                evaluation_dataset_digest=scope.evaluation_dataset_digest,
                concrete_symbol=scope.concrete_symbol,
                symbol_role=scope.symbol_role.value,
                cell=scope.cell,
                source_window=scope.source_window,
                tile_index=scope.tile_index,
                policy_variant=request.policy_variant.value,
                evaluation_seed=request.evaluation_seed,
                paired_candidate_checkpoint_digest=(
                    request.paired_candidate_checkpoint_digest
                ),
                outcome_start_bar_index=scope.outcome_start_bar_index,
                outcome_stop_bar_index_exclusive=(
                    scope.outcome_stop_bar_index_exclusive
                ),
                evaluation_start_bar_index=scope.evaluation_start_bar_index,
                evaluation_stop_bar_index=scope.evaluation_stop_bar_index,
                runtime_start_bar_index=runtime_start_bar_index,
                runtime_end_bar_index=runtime_end_bar_index,
                final_current_bar_index=final_current_bar_index,
                observed_decision_count=observed_decision_count,
                normal_completion=normal_completion,
                terminated=bool(terminated),
                truncated=bool(truncated),
                termination_reason=termination_reason,
                terminal_accounting_mode=terminal_accounting_mode,
                terminal_liquidation_cost=terminal_liquidation_cost,
                initial_capital=initial_capital,
                final_net_portfolio_value=final_net_portfolio_value,
                net_wealth_ratio=net_wealth_ratio,
                net_simple_returns=net_simple_returns,
                maximum_drawdown=float(book.max_drawdown),
                turnover_total=float(book.turnover_total),
                total_execution_cost=float(book.total_cost),
                funding_pnl=float(book.funding_pnl),
                borrow_cost=float(book.borrow_cost),
                trade_count=int(book.n_trades),
                rebalance_count=int(book.rebalance_events),
                fill_count=int(book.fill_count),
                target_change_count=_change_count(normalized_action_trace),
                submitted_change_count=_change_count(submitted_trace),
                executed_change_count=_change_count(executed_trace),
                sign_flip_count=sum(
                    step.transition_class == "flip" for step in step_evidence
                ),
                hard_risk_violation_count=sum(
                    step.hard_risk_violation for step in step_evidence
                ),
                execution_rejection_count=sum(
                    step.rejected_count for step in step_evidence
                ),
                normalized_action_trace=normalized_action_trace,
                realized_exposure_trace=realized_exposure_trace,
                step_evidence=step_evidence,
            )
        finally:
            environment.close()


def _require_runtime_generation(
    *,
    u2_contract: UniversalTradeRLU2Contract,
    u1_contract: UniversalTradeRLU1Contract,
    policy_contract: UniversalTradePolicyContract,
    normalizer: UniversalTradeSequenceNormalizer,
    environment_factory: U2ReplayEnvironmentFactory,
) -> None:
    if not isinstance(u1_contract, UniversalTradeRLU1Contract):
        raise TypeError("U2 replay requires a U1 contract")
    if not isinstance(policy_contract, UniversalTradePolicyContract):
        raise TypeError("U2 replay requires a policy contract")
    if not isinstance(normalizer, UniversalTradeSequenceNormalizer):
        raise TypeError("U2 replay requires the frozen sequence normalizer")
    if not callable(environment_factory):
        raise TypeError("U2 replay environment factory must be callable")
    if u2_contract.u1_contract_digest != u1_contract.digest:
        raise ValueError("U2 replay U1 contract identity mismatch")
    if u1_contract.policy_contract_digest != policy_contract.digest:
        raise ValueError("U2 replay policy contract identity mismatch")
    if u2_contract.u1_normalizer_digest != normalizer.digest:
        raise ValueError("U2 replay normalizer generation mismatch")
    if normalizer.contract_digest != policy_contract.digest:
        raise ValueError("U2 replay normalizer policy contract mismatch")


def build_universal_trade_rl_u2_development_replay_session(
    *,
    manifest: UniversalTradeRLUniverseManifest,
    time_partition: UniversalTradeRLU2TimePartition,
    u2_contract: UniversalTradeRLU2Contract,
    u1_contract: UniversalTradeRLU1Contract,
    policy_contract: UniversalTradePolicyContract,
    normalizer: UniversalTradeSequenceNormalizer,
    supplied_scope_closure: UniversalTradeRLU2DevelopmentScopeClosure,
    artifact_locators: Mapping[str, U2EvaluationSourceArtifactLocator],
    source_loader: U2EvaluationSourceArtifactLoader,
    environment_factory: U2ReplayEnvironmentFactory,
) -> UniversalTradeRLU2DevelopmentReplaySession:
    """Validate canonical metadata before crossing the Development numeric boundary."""

    if not isinstance(manifest, UniversalTradeRLUniverseManifest):
        raise TypeError("U2 replay requires a U0 manifest")
    if not isinstance(time_partition, UniversalTradeRLU2TimePartition):
        raise TypeError("U2 replay requires a time partition")
    if not isinstance(u2_contract, UniversalTradeRLU2Contract):
        raise TypeError("U2 replay requires a U2 contract")
    if not isinstance(
        supplied_scope_closure,
        UniversalTradeRLU2DevelopmentScopeClosure,
    ):
        raise TypeError("U2 replay requires a Development scope closure")

    canonical_scope_closure = build_universal_trade_rl_u2_development_scope_closure(
        manifest=manifest,
        time_partition=time_partition,
        u2_contract=u2_contract,
    )
    if supplied_scope_closure != canonical_scope_closure:
        raise ValueError("U2 Development replay supplied closure is not canonical")

    _require_runtime_generation(
        u2_contract=u2_contract,
        u1_contract=u1_contract,
        policy_contract=policy_contract,
        normalizer=normalizer,
        environment_factory=environment_factory,
    )
    if not callable(source_loader):
        raise TypeError("U2 replay source loader must be callable")

    datasets = load_universal_trade_rl_u2_development_evaluation_datasets(
        manifest=manifest,
        scope_closure=canonical_scope_closure,
        artifact_locators=artifact_locators,
        loader=source_loader,
    )
    return UniversalTradeRLU2DevelopmentReplaySession(
        manifest=manifest,
        time_partition=time_partition,
        u2_contract=u2_contract,
        u1_contract=u1_contract,
        policy_contract=policy_contract,
        normalizer=normalizer,
        scope_closure=canonical_scope_closure,
        datasets=datasets,
        environment_factory=environment_factory,
    )


__all__ = [
    "U2_REPLAY_EVIDENCE_SCHEMA",
    "U2ReplayEnvironmentFactory",
    "UniversalTradeRLU2DevelopmentReplaySession",
    "UniversalTradeRLU2ReplayEvidence",
    "UniversalTradeRLU2ReplayRequest",
    "UniversalTradeRLU2ReplayStepEvidence",
    "UniversalTradeRLU2ReplayVariant",
    "build_universal_trade_rl_u2_development_replay_session",
]
