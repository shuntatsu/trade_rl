"""Manifest-bound Stage A evaluation through the maintained market environment."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Protocol, cast

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.market import MarketDataset
from trade_rl.domain.common import require_sha256
from trade_rl.evaluation.stage_a_zero_shot_contracts import (
    StageAZeroShotEvaluationPlan,
)
from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.funding_evidence import FundingBoundaryEvidence
from trade_rl.simulation.orders import OrderBookState, OrderEvent
from trade_rl.workflows.stage_a_evaluation_dataset_manifest import (
    StageAEvaluationDatasetManifest,
)
from trade_rl.workflows.stage_a_execution_producer import (
    StageAEvaluationEpisodeResult,
)
from trade_rl.workflows.stage_a_funding_evidence import (
    collect_stage_a_funding_evidence,
    validate_stage_a_funding_evidence,
)
from trade_rl.workflows.stage_a_zero_shot_runner_contracts import (
    StageAEvaluationCellRequest,
)

_OBSERVATION_DIGEST_SCHEMA = "stage_a_policy_observation_digest_v1"


class StageAEvaluationDatasetResolver(Protocol):
    """Resolve the full immutable dataset named by one Stage A request."""

    def resolve(self, request: StageAEvaluationCellRequest) -> MarketDataset: ...


class StageAEvaluationEnvironment(Protocol):
    """Executor-facing surface of the maintained market environment."""

    dataset_id: str
    execution_policy_digest: str
    minimum_start_index: int
    current_index: int
    end_index: int
    action_space: object
    hybrid: BookState
    hybrid_order_book: OrderBookState

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[object, dict[str, object]]: ...

    def baseline_action(self) -> np.ndarray: ...

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[object, float, bool, bool, dict[str, object]]: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class StageAEvaluationEnvironmentHandle:
    """Environment plus the exact candidate recipe identity used to build it."""

    environment: StageAEvaluationEnvironment
    candidate_config_digest: str

    def __post_init__(self) -> None:
        require_sha256(
            self.candidate_config_digest,
            field="stage_a_environment_handle.candidate_config_digest",
        )
        if self.environment is None:
            raise ValueError("Stage A evaluation environment is missing")


class StageAEvaluationEnvironmentFactory(Protocol):
    """Build one fresh environment for an exact dataset and recipe identity."""

    def build(
        self,
        *,
        request: StageAEvaluationCellRequest,
        dataset: MarketDataset,
        candidate_config_digest: str,
    ) -> StageAEvaluationEnvironmentHandle: ...


class StageAEvaluationPolicy(Protocol):
    """Canonical deterministic serving-policy surface."""

    def predict(self, observation: object) -> np.ndarray: ...


class MappingStageAEvaluationDatasetResolver:
    """Resolve already-built PostgreSQL triplet datasets by content identity."""

    def __init__(self, datasets: Mapping[str, MarketDataset]) -> None:
        normalized = dict(datasets)
        if not normalized:
            raise ValueError("Stage A dataset resolver requires datasets")
        for dataset_id, dataset in normalized.items():
            require_sha256(dataset_id, field="stage_a_dataset_resolver.dataset_id")
            if dataset.dataset_id != dataset_id:
                raise ValueError("Stage A dataset resolver key does not match dataset")
        self._datasets = MappingProxyType(normalized)

    def resolve(self, request: StageAEvaluationCellRequest) -> MarketDataset:
        try:
            return self._datasets[request.dataset_id]
        except KeyError as error:
            raise ValueError("Stage A evaluation dataset is not registered") from error


class RegisteredStageAEvaluationEnvironmentFactory:
    """Build environments only from builders registered by candidate digest."""

    def __init__(
        self,
        builders: Mapping[
            str,
            Callable[[MarketDataset], StageAEvaluationEnvironment],
        ],
    ) -> None:
        normalized = dict(builders)
        if not normalized:
            raise ValueError("Stage A environment factory requires builders")
        for digest, builder in normalized.items():
            require_sha256(digest, field="stage_a_environment_factory.digest")
            if not callable(builder):
                raise ValueError("Stage A environment builder must be callable")
        self._builders = MappingProxyType(normalized)

    def build(
        self,
        *,
        request: StageAEvaluationCellRequest,
        dataset: MarketDataset,
        candidate_config_digest: str,
    ) -> StageAEvaluationEnvironmentHandle:
        del request
        require_sha256(
            candidate_config_digest,
            field="stage_a_environment_factory.candidate_config_digest",
        )
        try:
            builder = self._builders[candidate_config_digest]
        except KeyError as error:
            raise ValueError(
                "Stage A candidate environment recipe is not registered"
            ) from error
        environment = builder(dataset)
        return StageAEvaluationEnvironmentHandle(
            environment=environment,
            candidate_config_digest=candidate_config_digest,
        )


def _finite_positive_equity(book: BookState, *, field: str) -> float:
    value = float(book.portfolio_value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{field} must be finite and positive")
    return value


def _array_payload(value: object, *, field: str) -> dict[str, object]:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be array-like") from error
    if array.size == 0:
        raise ValueError(f"{field} must not be empty")
    if array.dtype.kind not in {"b", "i", "u", "f"}:
        raise ValueError(f"{field} dtype is unsupported")
    if not np.isfinite(array).all():
        raise ValueError(f"{field} must be finite")
    return {
        "dtype": str(array.dtype),
        "shape": tuple(int(item) for item in array.shape),
        "values": array.tolist(),
    }


def stage_a_observation_digest(observation: object) -> str:
    """Hash one exact flat or structured policy observation."""

    if isinstance(observation, Mapping):
        if not observation:
            raise ValueError("Stage A structured observation must not be empty")
        components: dict[str, object] = {}
        for key in sorted(observation):
            if not isinstance(key, str) or not key:
                raise ValueError("Stage A observation keys must be non-empty strings")
            components[key] = _array_payload(
                observation[key], field=f"Stage A observation component {key}"
            )
    else:
        components = {
            "flat": _array_payload(observation, field="Stage A flat observation")
        }
    return content_digest(
        {
            "components": components,
            "schema_version": _OBSERVATION_DIGEST_SCHEMA,
        }
    )


def _action_shape(environment: StageAEvaluationEnvironment) -> tuple[int, ...]:
    shape = getattr(environment.action_space, "shape", None)
    if (
        not isinstance(shape, tuple)
        or len(shape) != 1
        or isinstance(shape[0], bool)
        or not isinstance(shape[0], int)
        or shape[0] <= 0
    ):
        raise ValueError("Stage A environment action space must be one-dimensional")
    return shape


def _normalized_action(
    value: object,
    *,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    if isinstance(value, tuple):
        raise ValueError(
            "Stage A deterministic policy must return an action array, not state"
        )
    try:
        action = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError("Stage A policy action must be numeric") from error
    if action.shape != expected_shape:
        raise ValueError("Stage A policy action shape mismatch")
    if not np.isfinite(action).all():
        raise ValueError("Stage A policy action must be finite")
    return action.copy()


def _timestamp_ns(dataset: MarketDataset, index: int) -> int:
    return int(dataset.timestamps[index].astype("datetime64[ns]").astype(np.int64))


def _events_from_info(info: Mapping[str, object]) -> tuple[OrderEvent, ...]:
    collected: list[OrderEvent] = []
    for key in ("hybrid_execution", "hybrid_liquidation"):
        source = info.get(key)
        raw_events = () if source is None else getattr(source, "order_events", ())
        events = tuple(raw_events)
        if any(not isinstance(event, OrderEvent) for event in events):
            raise ValueError("Stage A environment returned invalid order events")
        if tuple(event.sequence for event in events) != tuple(range(len(events))):
            raise ValueError("Stage A step order-event sequence is not contiguous")
        collected.extend(events)
    return tuple(collected)


def _normalized_events(
    events: tuple[OrderEvent, ...],
    *,
    request: StageAEvaluationCellRequest,
    dataset: MarketDataset,
) -> tuple[OrderEvent, ...]:
    normalized: list[OrderEvent] = []
    start = request.evaluation_range.start
    stop = request.evaluation_range.stop
    for sequence, event in enumerate(events):
        if event.dataset_id != request.dataset_id:
            raise ValueError("Stage A order event dataset identity mismatch")
        if event.execution_policy_digest != request.execution_identity:
            raise ValueError("Stage A order event execution identity mismatch")
        if event.processing_index < start or event.processing_index > stop:
            raise ValueError(
                "Stage A order event processing index outside authorized range"
            )
        if event.processing_index >= dataset.n_bars:
            raise ValueError("Stage A order event processing index is outside dataset")
        if event.timestamp_ns != _timestamp_ns(dataset, event.processing_index):
            raise ValueError("Stage A order event timestamp mismatch")
        normalized.append(replace(event, sequence=sequence))
    return tuple(normalized)


class StageASB3EvaluationEpisodeExecutor:
    """Execute one exact Stage A cell through a fresh maintained environment."""

    def __init__(
        self,
        *,
        plan: StageAZeroShotEvaluationPlan,
        manifest: StageAEvaluationDatasetManifest,
        dataset_resolver: StageAEvaluationDatasetResolver,
        environment_factory: StageAEvaluationEnvironmentFactory,
    ) -> None:
        plan.validate_manifest(manifest)
        self.plan = plan
        self.manifest = manifest
        self.dataset_resolver = dataset_resolver
        self.environment_factory = environment_factory

    def _validate_dataset(
        self,
        request: StageAEvaluationCellRequest,
        dataset: MarketDataset,
    ) -> None:
        if dataset.dataset_id != request.dataset_id:
            raise ValueError("Stage A evaluation dataset identity mismatch")
        if dataset.feature_config_digest != request.feature_identity:
            raise ValueError("Stage A evaluation feature identity mismatch")
        start = request.evaluation_range.start
        stop = request.evaluation_range.stop
        if start < 0 or start >= stop or stop >= dataset.n_bars:
            raise ValueError("Stage A evaluation range is outside the full dataset")

    @staticmethod
    def _validate_environment(
        *,
        request: StageAEvaluationCellRequest,
        environment: StageAEvaluationEnvironment,
        candidate_config_digest: str,
        handle: StageAEvaluationEnvironmentHandle,
    ) -> None:
        if handle.candidate_config_digest != candidate_config_digest:
            raise ValueError("Stage A environment candidate config digest mismatch")
        if environment.dataset_id != request.dataset_id:
            raise ValueError("Stage A environment dataset identity mismatch")
        if environment.execution_policy_digest != request.execution_identity:
            raise ValueError("Stage A environment execution identity mismatch")
        if environment.minimum_start_index > request.evaluation_range.start:
            raise ValueError(
                "Stage A evaluation range lacks causal environment history"
            )

    @staticmethod
    def _validate_policy_inputs(
        request: StageAEvaluationCellRequest,
        *,
        policy: object | None,
        policy_source_digest: str | None,
    ) -> StageAEvaluationPolicy | None:
        if request.is_baseline:
            if policy is not None or policy_source_digest is not None:
                raise ValueError("Stage A baseline execution must not define a policy")
            return None
        if policy is None or policy_source_digest is None:
            raise ValueError("Stage A policy execution requires policy and source")
        require_sha256(
            policy_source_digest,
            field="stage_a_execution.policy_source_digest",
        )
        predictor = getattr(policy, "predict", None)
        if not callable(predictor):
            raise ValueError("Stage A policy does not expose deterministic predict")
        return cast(StageAEvaluationPolicy, policy)

    def execute(
        self,
        request: StageAEvaluationCellRequest,
        *,
        policy: object | None,
        policy_source_digest: str | None,
        candidate_config_digest: str,
    ) -> StageAEvaluationEpisodeResult:
        """Run the exact scored range while preserving full-dataset observation history."""

        request.validate_manifest(self.plan, self.manifest)
        require_sha256(
            candidate_config_digest,
            field="stage_a_execution.candidate_config_digest",
        )
        predictor = self._validate_policy_inputs(
            request,
            policy=policy,
            policy_source_digest=policy_source_digest,
        )
        dataset = self.dataset_resolver.resolve(request)
        self._validate_dataset(request, dataset)
        handle = self.environment_factory.build(
            request=request,
            dataset=dataset,
            candidate_config_digest=candidate_config_digest,
        )
        environment = handle.environment
        self._validate_environment(
            request=request,
            environment=environment,
            candidate_config_digest=candidate_config_digest,
            handle=handle,
        )
        action_shape = _action_shape(environment)
        start = request.evaluation_range.start
        stop = request.evaluation_range.stop
        actions: list[tuple[float, ...]] = []
        observations: list[str] = []
        equity: list[float] = []
        events: list[OrderEvent] = []
        funding_evidence: list[FundingBoundaryEvidence] = []
        transition_end_indices: list[int] = []
        try:
            observation, _ = environment.reset(
                seed=request.seed,
                options={
                    "episode_bars": stop - start,
                    "initial_state_mode": "cash",
                    "start_idx": start,
                },
            )
            if environment.current_index != start or environment.end_index != stop:
                raise ValueError("Stage A environment reset range mismatch")
            observations.append(stage_a_observation_digest(observation))
            equity.append(
                _finite_positive_equity(
                    environment.hybrid,
                    field="Stage A initial equity",
                )
            )
            maximum_decisions = stop - start + 1
            for _ in range(maximum_decisions):
                if predictor is None:
                    raw_action: object = environment.baseline_action()
                else:
                    raw_action = predictor.predict(observation)
                action = _normalized_action(
                    raw_action,
                    expected_shape=action_shape,
                )
                actions.append(tuple(float(value) for value in action))
                previous_index = environment.current_index
                observation, _, terminated, truncated, info = environment.step(action)
                if environment.current_index <= previous_index:
                    raise ValueError("Stage A environment did not advance")
                if environment.current_index > stop:
                    raise ValueError(
                        "Stage A environment advanced beyond authorized stop"
                    )
                transition_end_indices.append(environment.current_index)
                events.extend(_events_from_info(info))
                funding_evidence.extend(collect_stage_a_funding_evidence(info))
                observations.append(stage_a_observation_digest(observation))
                equity.append(
                    _finite_positive_equity(
                        environment.hybrid,
                        field="Stage A transition equity",
                    )
                )
                if terminated or truncated:
                    if environment.current_index != stop:
                        raise ValueError(
                            "Stage A environment ended before authorized evaluation stop"
                        )
                    break
                if environment.current_index == stop:
                    raise ValueError(
                        "Stage A environment reached authorized stop without termination"
                    )
            else:
                raise ValueError("Stage A environment exceeded decision bound")

            normalized_events = _normalized_events(
                tuple(events),
                request=request,
                dataset=dataset,
            )
            normalized_funding_evidence = validate_stage_a_funding_evidence(
                tuple(funding_evidence),
                request=request,
                dataset=dataset,
            )
            result = StageAEvaluationEpisodeResult(
                request_digest=request.digest,
                policy_source_digest=policy_source_digest,
                candidate_config_digest=candidate_config_digest,
                actions=tuple(actions),
                observation_digests=tuple(observations),
                equity_curve=tuple(equity),
                order_events=normalized_events,
                terminal_book=environment.hybrid,
                terminal_order_book=environment.hybrid_order_book,
                funding_evidence=normalized_funding_evidence,
                transition_end_indices=tuple(transition_end_indices),
            )
            return result.validate_against(
                request,
                expected_policy_source_digest=policy_source_digest,
                expected_candidate_config_digest=candidate_config_digest,
            )
        finally:
            environment.close()


__all__ = [
    "MappingStageAEvaluationDatasetResolver",
    "RegisteredStageAEvaluationEnvironmentFactory",
    "StageAEvaluationDatasetResolver",
    "StageAEvaluationEnvironment",
    "StageAEvaluationEnvironmentFactory",
    "StageAEvaluationEnvironmentHandle",
    "StageAEvaluationPolicy",
    "StageASB3EvaluationEpisodeExecutor",
    "stage_a_observation_digest",
]
