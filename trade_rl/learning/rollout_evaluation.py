"""Exact simulator rollouts used by Oracle and behavior-cloning audits."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from trade_rl.data.identity import content_and_arrays_digest
from trade_rl.evaluation.closed_trades import ClosedTradeTracker
from trade_rl.learning.evaluation import (
    ActionPathCollapseEvidence,
    PathPerformanceMetrics,
    evaluate_path_performance,
)


class EvaluationEnvironment(Protocol):
    current_index: int
    dataset: Any

    def reset(
        self, *, options: dict[str, object]
    ) -> tuple[object, dict[str, object]]: ...

    def step(
        self, action: np.ndarray
    ) -> tuple[object, float, bool, bool, dict[str, object]]: ...


_STEP_TRACE_SCHEMA = "action_path_step_trace_v1"
_STEP_TRACE_TOLERANCE = 1e-12


def _trace_matrix(value: object, *, rows: int, field: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).copy()
    if array.ndim != 2 or array.shape[0] != rows or array.shape[1] == 0:
        raise ValueError(f"{field} must be a non-empty step-aligned matrix")
    if not np.isfinite(array).all():
        raise ValueError(f"{field} must be finite")
    array.setflags(write=False)
    return array


def _trace_vector(
    value: object,
    *,
    rows: int,
    field: str,
    dtype: type[np.float64] | type[np.int64] | type[np.bool_] = np.float64,
) -> np.ndarray:
    array = np.asarray(value, dtype=dtype).reshape(-1).copy()
    if array.shape != (rows,):
        raise ValueError(f"{field} must be step-aligned")
    if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
        raise ValueError(f"{field} must be finite")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class ActionPathStepTrace:
    """Immutable per-step decision, risk, execution, and strategy diagnostics."""

    decision_indices: np.ndarray
    current_weights: np.ndarray
    requested_targets: np.ndarray
    projected_targets: np.ndarray
    realized_weights: np.ndarray
    active_risk_caps: np.ndarray
    active_liquidity_caps: np.ndarray
    fast_means: np.ndarray
    fast_stds: np.ndarray
    fast_qualified_directions: np.ndarray
    fast_edge_margins: np.ndarray
    after_cost_entry_objectives: np.ndarray
    slow_means: np.ndarray
    slow_stds: np.ndarray
    slow_directions: np.ndarray
    position_origins: tuple[str, ...]
    hierarchy_reasons: tuple[str, ...]
    gross_returns: np.ndarray
    net_returns: np.ndarray
    costs: np.ndarray
    turnovers: np.ndarray
    submitted: np.ndarray
    suppressed: np.ndarray
    executed: np.ndarray
    schema_version: str = _STEP_TRACE_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        decisions = _trace_vector(
            self.decision_indices,
            rows=len(np.asarray(self.decision_indices).reshape(-1)),
            field="step trace decision_indices",
            dtype=np.int64,
        )
        rows = len(decisions)
        if rows == 0 or np.any(decisions < 0) or np.any(np.diff(decisions) <= 0):
            raise ValueError("step trace decision indices must be increasing")
        matrices = {
            name: _trace_matrix(getattr(self, name), rows=rows, field=f"step trace {name}")
            for name in (
                "current_weights",
                "requested_targets",
                "projected_targets",
                "realized_weights",
                "active_risk_caps",
                "active_liquidity_caps",
            )
        }
        action_shape = matrices["current_weights"].shape
        if any(
            matrices[name].shape != action_shape
            for name in (
                "requested_targets",
                "projected_targets",
                "realized_weights",
                "active_risk_caps",
                "active_liquidity_caps",
            )
        ):
            raise ValueError("step trace action matrices are not aligned")
        vectors = {
            name: _trace_vector(getattr(self, name), rows=rows, field=f"step trace {name}")
            for name in (
                "fast_means",
                "fast_stds",
                "fast_edge_margins",
                "after_cost_entry_objectives",
                "slow_means",
                "slow_stds",
                "gross_returns",
                "net_returns",
                "costs",
                "turnovers",
            )
        }
        for name in ("fast_stds", "slow_stds", "costs", "turnovers"):
            if np.any(vectors[name] < 0.0):
                raise ValueError(f"step trace {name} must be non-negative")
        for name in ("gross_returns", "net_returns"):
            if np.any(vectors[name] <= -1.0):
                raise ValueError(f"step trace {name} must be greater than -1")
        fast_directions = _trace_vector(
            self.fast_qualified_directions,
            rows=rows,
            field="step trace fast_qualified_directions",
            dtype=np.int64,
        )
        slow_directions = _trace_vector(
            self.slow_directions,
            rows=rows,
            field="step trace slow_directions",
            dtype=np.int64,
        )
        if np.any(~np.isin(fast_directions, (-1, 0, 1))) or np.any(
            ~np.isin(slow_directions, (-1, 0, 1))
        ):
            raise ValueError("step trace directions must be -1, 0, or 1")
        bools = {
            name: _trace_vector(
                getattr(self, name),
                rows=rows,
                field=f"step trace {name}",
                dtype=np.bool_,
            )
            for name in ("submitted", "suppressed", "executed")
        }
        origins = tuple(self.position_origins)
        reasons = tuple(self.hierarchy_reasons)
        if len(origins) != rows or len(reasons) != rows:
            raise ValueError("step trace string fields are not aligned")
        if any(not isinstance(value, str) or not value for value in (*origins, *reasons)):
            raise ValueError("step trace string fields must be non-empty")
        if self.schema_version != _STEP_TRACE_SCHEMA:
            raise ValueError("unsupported step trace schema")
        object.__setattr__(self, "decision_indices", decisions)
        for name, array in {**matrices, **vectors, **bools, "fast_qualified_directions": fast_directions, "slow_directions": slow_directions}.items():
            object.__setattr__(self, name, array)
        object.__setattr__(self, "position_origins", origins)
        object.__setattr__(self, "hierarchy_reasons", reasons)
        expected = content_and_arrays_digest(
            {
                "hierarchy_reasons": reasons,
                "position_origins": origins,
                "schema_version": self.schema_version,
            },
            tuple(
                (name, getattr(self, name))
                for name in (
                    "decision_indices",
                    "current_weights",
                    "requested_targets",
                    "projected_targets",
                    "realized_weights",
                    "active_risk_caps",
                    "active_liquidity_caps",
                    "fast_means",
                    "fast_stds",
                    "fast_qualified_directions",
                    "fast_edge_margins",
                    "after_cost_entry_objectives",
                    "slow_means",
                    "slow_stds",
                    "slow_directions",
                    "gross_returns",
                    "net_returns",
                    "costs",
                    "turnovers",
                    "submitted",
                    "suppressed",
                    "executed",
                )
            ),
        )
        if self.digest and self.digest != expected:
            raise ValueError("step trace digest mismatch")
        object.__setattr__(self, "digest", expected)

    @property
    def decision_count(self) -> int:
        return int(self.decision_indices.size)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "active_liquidity_caps": self.active_liquidity_caps.tolist(),
            "active_risk_caps": self.active_risk_caps.tolist(),
            "after_cost_entry_objectives": self.after_cost_entry_objectives.tolist(),
            "costs": self.costs.tolist(),
            "current_weights": self.current_weights.tolist(),
            "decision_indices": self.decision_indices.tolist(),
            "executed": self.executed.tolist(),
            "fast_edge_margins": self.fast_edge_margins.tolist(),
            "fast_means": self.fast_means.tolist(),
            "fast_qualified_directions": self.fast_qualified_directions.tolist(),
            "fast_stds": self.fast_stds.tolist(),
            "gross_returns": self.gross_returns.tolist(),
            "hierarchy_reasons": self.hierarchy_reasons,
            "net_returns": self.net_returns.tolist(),
            "position_origins": self.position_origins,
            "projected_targets": self.projected_targets.tolist(),
            "realized_weights": self.realized_weights.tolist(),
            "requested_targets": self.requested_targets.tolist(),
            "slow_directions": self.slow_directions.tolist(),
            "slow_means": self.slow_means.tolist(),
            "slow_stds": self.slow_stds.tolist(),
            "submitted": self.submitted.tolist(),
            "suppressed": self.suppressed.tolist(),
            "turnovers": self.turnovers.tolist(),
            "schema_version": self.schema_version,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload

    @classmethod
    def from_payload(cls, value: object) -> ActionPathStepTrace:
        if not isinstance(value, Mapping):
            raise ValueError("step trace payload is invalid")
        payload = dict(value)
        raw_digest = payload.pop("artifact_digest", "")
        payload.setdefault("schema_version", _STEP_TRACE_SCHEMA)
        payload["position_origins"] = tuple(payload["position_origins"])
        payload["hierarchy_reasons"] = tuple(payload["hierarchy_reasons"])
        trace = cls(**payload, digest=str(raw_digest))
        return trace


@dataclass(frozen=True, slots=True)
class ActionPathStepEconomics:
    """Simulator-authoritative step economics retained for attribution."""

    gross_returns: np.ndarray
    net_returns: np.ndarray
    costs: np.ndarray
    turnover: np.ndarray
    realized_weights: np.ndarray | None = None

    def __post_init__(self) -> None:
        arrays: dict[str, np.ndarray] = {}
        for name in ("gross_returns", "net_returns", "costs", "turnover"):
            array = np.asarray(getattr(self, name), dtype=np.float64).reshape(-1).copy()
            if array.size == 0 or not np.isfinite(array).all():
                raise ValueError("action path step economics must be non-empty and finite")
            array.setflags(write=False)
            arrays[name] = array
        if len({array.size for array in arrays.values()}) != 1:
            raise ValueError("action path step economics arrays are not aligned")
        if np.any(arrays["gross_returns"] <= -1.0) or np.any(
            arrays["net_returns"] <= -1.0
        ):
            raise ValueError("action path step returns must be greater than -1")
        if np.any(arrays["costs"] < 0.0) or np.any(arrays["turnover"] < 0.0):
            raise ValueError("action path step costs and turnover must be non-negative")
        for name, array in arrays.items():
            object.__setattr__(self, name, array)
        if self.realized_weights is not None:
            realized = np.asarray(self.realized_weights, dtype=np.float64).copy()
            if realized.ndim != 2 or realized.shape[0] != len(arrays["gross_returns"]):
                raise ValueError("realized weights must be step-aligned")
            if realized.shape[1] == 0 or not np.isfinite(realized).all():
                raise ValueError("realized weights must be non-empty and finite")
            realized.setflags(write=False)
            object.__setattr__(self, "realized_weights", realized)


@dataclass(frozen=True, slots=True)
class ActionPathEvaluation:
    actions: np.ndarray
    performance: PathPerformanceMetrics
    collapse_evidence: ActionPathCollapseEvidence
    step_economics: ActionPathStepEconomics | None = None
    step_trace: ActionPathStepTrace | None = None

    def __post_init__(self) -> None:
        actions = np.asarray(self.actions, dtype=np.float32).copy(order="C")
        if (
            actions.ndim != 2
            or len(actions) != self.performance.step_count
            or not np.isfinite(actions).all()
        ):
            raise ValueError("evaluated actions do not match path metrics")
        if self.collapse_evidence.decision_count != self.performance.step_count:
            raise ValueError("collapse evidence does not cover evaluated path")
        if self.collapse_evidence.trade_count != self.performance.trade_count:
            raise ValueError("collapse evidence trade count mismatch")
        if (
            self.collapse_evidence.executed_change_count
            != self.performance.traded_step_count
        ):
            raise ValueError("collapse evidence execution count mismatch")
        if self.step_economics is not None:
            if not isinstance(self.step_economics, ActionPathStepEconomics):
                raise TypeError("step_economics must be ActionPathStepEconomics")
            if len(self.step_economics.gross_returns) != self.performance.step_count:
                raise ValueError("step economics do not cover evaluated path")
        if self.step_trace is not None:
            if not isinstance(self.step_trace, ActionPathStepTrace):
                raise TypeError("step_trace must be ActionPathStepTrace")
            if self.step_trace.decision_count != self.performance.step_count:
                raise ValueError("step trace does not cover evaluated path")
            if not np.allclose(
                actions,
                self.step_trace.requested_targets,
                rtol=0.0,
                atol=1e-6,
            ):
                raise ValueError("step trace requested targets do not reconcile")
            if self.step_economics is not None:
                realized = self.step_economics.realized_weights
                if realized is None or not np.array_equal(
                    realized,
                    self.step_trace.realized_weights,
                ):
                    raise ValueError("step trace realized weights do not reconcile")
                for name in ("gross_returns", "net_returns", "costs", "turnover"):
                    if not np.array_equal(
                        getattr(self.step_economics, name),
                        getattr(self.step_trace, "turnovers" if name == "turnover" else name),
                    ):
                        raise ValueError(f"step trace {name} does not reconcile")
        actions.setflags(write=False)
        object.__setattr__(self, "actions", actions)


def _metric(info: Mapping[str, object], name: str) -> float:
    value = info.get(name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"evaluation info is missing numeric {name}")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"evaluation info {name} is non-finite")
    return result


def _liquidation_metric(info: Mapping[str, object], name: str) -> float:
    liquidation = info.get("hybrid_liquidation")
    if liquidation is None:
        return 0.0
    value = getattr(liquidation, name, None)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"liquidation is missing numeric {name}")
    return float(value)


def _step_vector(
    value: object,
    *,
    shape: tuple[int, ...],
    field: str,
    fallback: np.ndarray | None = None,
) -> np.ndarray:
    """Resolve one authoritative stage vector while keeping legacy fakes usable."""

    if value is None:
        if fallback is None:
            raise ValueError(f"evaluation trace is missing {field}")
        array = np.asarray(fallback, dtype=np.float64).reshape(-1)
    else:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"evaluation trace {field} does not match action dimensions")
    return array.copy()


def _metadata_vector(
    metadata: Mapping[str, object],
    name: str,
    *,
    shape: tuple[int, ...],
    default: float = 0.0,
) -> np.ndarray:
    value = metadata.get(name, default)
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size == 1:
        array = np.full(shape, float(array[0]), dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"evaluation trace metadata {name} is invalid")
    return array.copy()


def _metadata_scalar(metadata: Mapping[str, object], name: str, default: float = 0.0) -> float:
    value = metadata.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"evaluation trace metadata {name} is invalid")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"evaluation trace metadata {name} is non-finite")
    return result


def evaluate_action_path(
    environment: EvaluationEnvironment,
    *,
    evaluation_range: tuple[int, int],
    actions: object | None = None,
    model: object | None = None,
    deterministic: bool = True,
    action_change_tolerance: float = 1e-6,
) -> ActionPathEvaluation:
    """Execute either a declared target path or one causal deterministic policy."""

    start, stop = evaluation_range
    if (
        isinstance(start, bool)
        or isinstance(stop, bool)
        or not isinstance(start, int)
        or not isinstance(stop, int)
        or start < 0
        or stop <= start + 1
    ):
        raise ValueError("evaluation_range must contain at least one decision")
    if (actions is None) == (model is None):
        raise ValueError("provide exactly one of actions or model")
    if not isinstance(deterministic, bool):
        raise TypeError("deterministic must be a boolean")
    if (
        isinstance(action_change_tolerance, bool)
        or not np.isfinite(action_change_tolerance)
        or action_change_tolerance < 0.0
    ):
        raise ValueError("action_change_tolerance must be finite and non-negative")
    expected_count = stop - start - 1
    declared: np.ndarray | None = None
    if actions is not None:
        declared = np.asarray(actions, dtype=np.float32)
        if (
            declared.ndim != 2
            or len(declared) != expected_count
            or not np.isfinite(declared).all()
        ):
            raise ValueError("declared actions do not cover the evaluation range")
    predict = None if model is None else getattr(model, "predict", None)
    if model is not None and not callable(predict):
        raise TypeError("evaluation model must expose predict")

    observation, _ = environment.reset(
        options={
            "start_idx": start,
            "episode_bars": expected_count,
            "initial_state_mode": "cash",
        }
    )
    multipliers = environment.dataset.resolved_array("contract_multipliers")
    trades = ClosedTradeTracker(multipliers)
    initial_book = getattr(environment, "hybrid", None)
    if initial_book is not None:
        trades.seed_positions(
            quantities=initial_book.quantities,
            prices=initial_book.mark_prices,
        )
    evaluated_actions: list[np.ndarray] = []
    gross_returns: list[float] = []
    net_returns: list[float] = []
    rewards: list[float] = []
    turnover: list[float] = []
    costs: list[float] = []
    active_dimension_count = 0
    inactive_dimension_count = 0
    proposal_distance_count = 0
    submitted_change_count = 0
    downstream_no_trade_suppression_count = 0
    execution_rejection_count = 0
    execution_rejection_reasons: Counter[str] = Counter()
    risk_projection_reasons: Counter[str] = Counter()
    executed_change_count = 0
    previous_submitted: np.ndarray | None = None
    action_dimension_count: int | None = None
    trace_decision_indices: list[int] = []
    trace_current_weights: list[np.ndarray] = []
    trace_projected_targets: list[np.ndarray] = []
    trace_realized_weights: list[np.ndarray] = []
    trace_risk_caps: list[np.ndarray] = []
    trace_liquidity_caps: list[np.ndarray] = []
    trace_fast_means: list[float] = []
    trace_fast_stds: list[float] = []
    trace_fast_directions: list[int] = []
    trace_fast_edge_margins: list[float] = []
    trace_entry_objectives: list[float] = []
    trace_slow_means: list[float] = []
    trace_slow_stds: list[float] = []
    trace_slow_directions: list[int] = []
    trace_position_origins: list[str] = []
    trace_hierarchy_reasons: list[str] = []
    trace_submitted: list[bool] = []
    trace_suppressed: list[bool] = []
    trace_executed: list[bool] = []
    for offset in range(expected_count):
        if environment.current_index != start + offset:
            raise ValueError("evaluation environment advanced outside the range")
        if declared is not None:
            action = declared[offset]
        else:
            assert callable(predict)
            raw_action, _ = predict(observation, deterministic=deterministic)
            action = np.asarray(raw_action, dtype=np.float32).reshape(-1)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_dimension_count is None:
            action_dimension_count = int(action.size)
        elif action.size != action_dimension_count:
            raise ValueError("evaluation action dimensions changed within path")
        if not np.isfinite(action).all():
            raise ValueError("evaluation action contains non-finite values")
        reference = (
            np.zeros_like(action) if previous_submitted is None else previous_submitted
        )
        active = np.ones(action.shape, dtype=np.bool_)
        if isinstance(observation, Mapping):
            if "current_weights" in observation:
                reference = np.asarray(
                    observation["current_weights"], dtype=np.float32
                ).reshape(-1)
                if reference.shape != action.shape or not np.isfinite(reference).all():
                    raise ValueError("current weights do not match evaluation action")
            if "active" in observation:
                active_values = np.asarray(observation["active"]).reshape(-1)
                if active_values.shape != action.shape:
                    raise ValueError("active mask does not match evaluation action")
                active = active_values > 0.5
        current_before = np.asarray(reference, dtype=np.float64).copy()
        proposed = active & (np.abs(action - reference) > action_change_tolerance)
        active_dimension_count += int(np.count_nonzero(active))
        inactive_dimension_count += int(np.count_nonzero(~active))
        proposal_distance_count += int(np.count_nonzero(proposed))
        submitted_change = bool(np.any(proposed))
        submitted_change_count += int(submitted_change)
        evaluated_actions.append(action.copy())
        previous_submitted = action.copy()
        observation, reward, terminated, truncated, raw_info = environment.step(action)
        if not isinstance(raw_info, Mapping):
            raise ValueError("evaluation environment info must be a mapping")
        info = raw_info
        execution = info.get("hybrid_execution")
        if execution is None:
            raise ValueError("evaluation info is missing hybrid execution")
        trades.ingest_stateful(execution)
        liquidation = info.get("hybrid_liquidation")
        if liquidation is not None:
            trades.ingest_liquidation(liquidation)
        gross = _metric(info, "interval_gross_return")
        net = _metric(info, "interval_net_return")
        liquidation_gross = _liquidation_metric(info, "interval_gross_return")
        liquidation_net = _liquidation_metric(info, "interval_net_return")
        gross_returns.append((1.0 + gross) * (1.0 + liquidation_gross) - 1.0)
        net_returns.append((1.0 + net) * (1.0 + liquidation_net) - 1.0)
        rewards.append(float(reward))
        requested_turnover = getattr(execution, "requested_turnover", None)
        if isinstance(requested_turnover, bool) or not isinstance(
            requested_turnover, int | float
        ):
            raise ValueError("hybrid execution is missing requested_turnover")
        rejected_count = getattr(execution, "rejected_count", 0)
        if (
            isinstance(rejected_count, bool)
            or not isinstance(rejected_count, int)
            or rejected_count < 0
        ):
            raise ValueError("hybrid execution rejected_count is invalid")
        execution_rejection_count += rejected_count
        rejected_events = 0
        for event in tuple(getattr(execution, "order_events", ())):
            if getattr(event, "event_type", None) != "rejected":
                continue
            reason = getattr(event, "reason", None)
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError("rejected order event is missing a reason")
            execution_rejection_reasons[reason] += 1
            rejected_events += 1
        if rejected_events != rejected_count:
            raise ValueError(
                "hybrid execution rejected_count does not match rejected order events"
            )
        risk = info.get("hybrid_risk")
        risk_pretrade = None if risk is None else getattr(risk, "pretrade_weights", None)
        if risk_pretrade is None and risk is not None:
            risk_pretrade = getattr(risk, "weights", None)
        projected = _step_vector(
            risk_pretrade,
            shape=action.shape,
            field="projected target",
            fallback=np.asarray(action, dtype=np.float64),
        )
        effective = info.get("effective_filled_weights")
        if effective is None:
            book = getattr(execution, "book", None)
            effective = None if book is None else getattr(book, "weights", None)
        if effective is None and isinstance(observation, Mapping):
            effective = observation.get("current_weights")
        realized = _step_vector(
            effective,
            shape=action.shape,
            field="realized weight",
            fallback=projected,
        )
        provider = getattr(model, "last_step_trace_metadata", None)
        if isinstance(provider, Mapping):
            metadata = provider
        elif callable(provider):
            metadata = provider()
        else:
            metadata = {}
        if not isinstance(metadata, Mapping):
            raise ValueError("evaluation model trace metadata must be a mapping")
        risk_caps = _metadata_vector(
            metadata,
            "active_risk_caps",
            shape=action.shape,
            default=_metadata_scalar(
                metadata,
                "active_risk_cap",
                _metadata_scalar(
                    {"value": getattr(risk, "max_gross", 0.0)},
                    "value",
                ),
            ),
        )
        liquidity_caps = _metadata_vector(
            metadata,
            "active_liquidity_caps",
            shape=action.shape,
            default=_metadata_scalar(metadata, "active_liquidity_cap", 0.0),
        )
        fast_mean = _metadata_scalar(metadata, "fast_mean")
        fast_std = _metadata_scalar(metadata, "fast_std")
        fast_direction = int(_metadata_scalar(metadata, "fast_qualified_direction"))
        fast_edge_margin = _metadata_scalar(metadata, "fast_edge_margin")
        entry_objective = _metadata_scalar(metadata, "after_cost_entry_objective")
        slow_mean = _metadata_scalar(metadata, "slow_mean")
        slow_std = _metadata_scalar(metadata, "slow_std")
        slow_direction = int(_metadata_scalar(metadata, "slow_direction"))
        origin = metadata.get("position_origin", "unknown")
        hierarchy_reason = metadata.get("hierarchy_reason", "unavailable")
        if not isinstance(origin, str) or not origin:
            raise ValueError("evaluation trace position origin is invalid")
        if not isinstance(hierarchy_reason, str) or not hierarchy_reason:
            raise ValueError("evaluation trace hierarchy reason is invalid")
        if fast_direction not in (-1, 0, 1) or slow_direction not in (-1, 0, 1):
            raise ValueError("evaluation trace directions must be -1, 0, or 1")
        if risk is not None:
            for reason in tuple(getattr(risk, "reasons", ())):
                if not isinstance(reason, str) or not reason.strip():
                    raise ValueError("hybrid risk projection reason is invalid")
                risk_projection_reasons[reason] += 1
        if submitted_change and float(requested_turnover) <= action_change_tolerance:
            downstream_no_trade_suppression_count += 1
        filled_turnover = getattr(execution, "filled_turnover", None)
        if isinstance(filled_turnover, bool) or not isinstance(
            filled_turnover, int | float
        ):
            raise ValueError("hybrid execution is missing filled_turnover")
        total_filled_turnover = float(filled_turnover) + _liquidation_metric(
            info, "filled_turnover"
        )
        turnover.append(total_filled_turnover)
        executed = total_filled_turnover > action_change_tolerance
        executed_change_count += int(executed)
        costs.append(
            _metric(info, "interval_cost") + _liquidation_metric(info, "interval_cost")
        )
        if (bool(terminated) or bool(truncated)) != (offset == expected_count - 1):
            raise ValueError("evaluation environment ended outside the range")
        trace_decision_indices.append(start + offset)
        trace_current_weights.append(current_before)
        trace_projected_targets.append(projected)
        trace_realized_weights.append(realized)
        trace_risk_caps.append(risk_caps)
        trace_liquidity_caps.append(liquidity_caps)
        trace_fast_means.append(fast_mean)
        trace_fast_stds.append(fast_std)
        trace_fast_directions.append(fast_direction)
        trace_fast_edge_margins.append(fast_edge_margin)
        trace_entry_objectives.append(entry_objective)
        trace_slow_means.append(slow_mean)
        trace_slow_stds.append(slow_std)
        trace_slow_directions.append(slow_direction)
        trace_position_origins.append(origin)
        trace_hierarchy_reasons.append(hierarchy_reason)
        trace_submitted.append(submitted_change)
        trace_suppressed.append(submitted_change and not executed)
        trace_executed.append(executed)
    diagnostics = trades.diagnostics()
    performance = evaluate_path_performance(
        gross_step_returns=gross_returns,
        net_step_returns=net_returns,
        rewards=rewards,
        turnover=turnover,
        costs=costs,
        closed_trade_count=diagnostics.closed_trades,
        winning_trade_count=diagnostics.winning_trades,
        trade_epsilon=action_change_tolerance,
    )
    if action_dimension_count is None:
        raise RuntimeError("evaluation produced no action dimensions")
    step_trace = ActionPathStepTrace(
        decision_indices=np.asarray(trace_decision_indices, dtype=np.int64),
        current_weights=np.stack(trace_current_weights, axis=0),
        requested_targets=np.stack(evaluated_actions, axis=0).astype(np.float64),
        projected_targets=np.stack(trace_projected_targets, axis=0),
        realized_weights=np.stack(trace_realized_weights, axis=0),
        active_risk_caps=np.stack(trace_risk_caps, axis=0),
        active_liquidity_caps=np.stack(trace_liquidity_caps, axis=0),
        fast_means=np.asarray(trace_fast_means, dtype=np.float64),
        fast_stds=np.asarray(trace_fast_stds, dtype=np.float64),
        fast_qualified_directions=np.asarray(trace_fast_directions, dtype=np.int64),
        fast_edge_margins=np.asarray(trace_fast_edge_margins, dtype=np.float64),
        after_cost_entry_objectives=np.asarray(
            trace_entry_objectives,
            dtype=np.float64,
        ),
        slow_means=np.asarray(trace_slow_means, dtype=np.float64),
        slow_stds=np.asarray(trace_slow_stds, dtype=np.float64),
        slow_directions=np.asarray(trace_slow_directions, dtype=np.int64),
        position_origins=tuple(trace_position_origins),
        hierarchy_reasons=tuple(trace_hierarchy_reasons),
        gross_returns=np.asarray(gross_returns, dtype=np.float64),
        net_returns=np.asarray(net_returns, dtype=np.float64),
        costs=np.asarray(costs, dtype=np.float64),
        turnovers=np.asarray(turnover, dtype=np.float64),
        submitted=np.asarray(trace_submitted, dtype=np.bool_),
        suppressed=np.asarray(trace_suppressed, dtype=np.bool_),
        executed=np.asarray(trace_executed, dtype=np.bool_),
    )
    evidence = ActionPathCollapseEvidence(
        decision_count=expected_count,
        action_dimension_count=action_dimension_count,
        active_dimension_count=active_dimension_count,
        inactive_dimension_count=inactive_dimension_count,
        proposal_distance_count=proposal_distance_count,
        submitted_change_count=submitted_change_count,
        downstream_no_trade_suppression_count=(downstream_no_trade_suppression_count),
        execution_rejection_count=execution_rejection_count,
        executed_change_count=executed_change_count,
        trade_count=performance.trade_count,
        constant_submitted_actions=submitted_change_count == 0,
        execution_rejection_reason_counts=tuple(
            sorted(execution_rejection_reasons.items())
        ),
        risk_projection_reason_counts=tuple(sorted(risk_projection_reasons.items())),
        hard_risk_violation=False,
    )
    return ActionPathEvaluation(
        actions=np.stack(evaluated_actions, axis=0),
        performance=performance,
        collapse_evidence=evidence,
        step_economics=ActionPathStepEconomics(
            gross_returns=np.asarray(gross_returns, dtype=np.float64),
            net_returns=np.asarray(net_returns, dtype=np.float64),
            costs=np.asarray(costs, dtype=np.float64),
            turnover=np.asarray(turnover, dtype=np.float64),
            realized_weights=np.stack(trace_realized_weights, axis=0),
        ),
        step_trace=step_trace,
    )


__all__ = [
    "ActionPathEvaluation",
    "ActionPathStepTrace",
    "ActionPathStepEconomics",
    "EvaluationEnvironment",
    "evaluate_action_path",
]
