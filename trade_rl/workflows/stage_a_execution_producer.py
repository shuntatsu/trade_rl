"""Strict episode execution and artifact production for Stage A evaluation."""

from __future__ import annotations

import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Self

from trade_rl.domain.common import require_sha256
from trade_rl.evaluation.stage_a_zero_shot_contracts import (
    StageACandidate,
    StageAZeroShotEvaluationPlan,
)
from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.simulation.execution_promotion import (
    execution_evidence_from_cost,
    write_execution_evidence,
)
from trade_rl.simulation.execution_replay import (
    build_execution_event_artifact,
    write_execution_event_artifact,
)
from trade_rl.simulation.funding_evidence import (
    FundingBoundaryEvidence,
    build_funding_evidence_artifact,
)
from trade_rl.simulation.orders import OrderBookState, OrderEvent
from trade_rl.workflows.stage_a_evaluation_dataset_manifest import (
    StageAEvaluationDatasetManifest,
)
from trade_rl.workflows.stage_a_execution_store import StoredStageAExecutionReplay
from trade_rl.workflows.stage_a_policy_source import (
    StageAPolicyRuntimeHandle,
    StageAPolicyRuntimeLoader,
    StageAPolicySourceBinding,
)
from trade_rl.workflows.stage_a_zero_shot_runner_contracts import (
    StageAEvaluationCellRequest,
)


def _finite_float(value: float | int, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite number")
    try:
        resolved = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a finite number") from error
    if not math.isfinite(resolved):
        raise ValueError(f"{field} must be finite")
    return resolved


def _optional_sha256(value: str | None, *, field: str) -> str | None:
    if value is None:
        return None
    require_sha256(value, field=field)
    return value


def _candidate_for_request(
    *,
    plan: StageAZeroShotEvaluationPlan,
    manifest: StageAEvaluationDatasetManifest,
    request: StageAEvaluationCellRequest,
) -> StageACandidate | None:
    request.validate_manifest(plan, manifest)
    if request.is_baseline:
        return None
    candidate_id = request.candidate_id
    if candidate_id is None:
        raise ValueError("Stage A policy execution requires a candidate")
    candidate = plan.candidate(candidate_id)
    if request.checkpoint_digest != candidate.checkpoint_digest(request.seed):
        raise ValueError("Stage A execution request checkpoint mismatch")
    return candidate


@dataclass(frozen=True, slots=True)
class StageAEvaluationEpisodeResult:
    """One validated environment episode before artifact derivation."""

    request_digest: str
    policy_source_digest: str | None
    candidate_config_digest: str
    actions: tuple[tuple[float, ...], ...]
    observation_digests: tuple[str, ...]
    equity_curve: tuple[float, ...]
    order_events: tuple[OrderEvent, ...]
    terminal_book: BookState
    terminal_order_book: OrderBookState
    funding_evidence: tuple[FundingBoundaryEvidence, ...] = ()
    transition_end_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        require_sha256(
            self.request_digest,
            field="stage_a_episode_result.request_digest",
        )
        policy_source_digest = _optional_sha256(
            self.policy_source_digest,
            field="stage_a_episode_result.policy_source_digest",
        )
        require_sha256(
            self.candidate_config_digest,
            field="stage_a_episode_result.candidate_config_digest",
        )

        if not self.actions:
            raise ValueError("Stage A episode actions must not be empty")
        actions: list[tuple[float, ...]] = []
        for step_index, row in enumerate(self.actions):
            if not row:
                raise ValueError("Stage A episode actions must not be empty")
            actions.append(
                tuple(
                    _finite_float(
                        value,
                        field=f"Stage A episode actions[{step_index}]",
                    )
                    for value in row
                )
            )

        observations = tuple(self.observation_digests)
        if not observations:
            raise ValueError("Stage A episode observations must not be empty")
        if len(observations) != len(actions) + 1:
            raise ValueError("Stage A episode observation closure mismatch")
        for index, digest in enumerate(observations):
            require_sha256(
                digest,
                field=f"stage_a_episode_result.observation_digests[{index}]",
            )

        equity = tuple(
            _finite_float(value, field="Stage A episode equity curve")
            for value in self.equity_curve
        )
        if len(equity) != len(observations):
            raise ValueError("Stage A episode equity closure mismatch")
        if any(value <= 0.0 for value in equity):
            raise ValueError("Stage A episode equity curve must be positive")

        transition_end_indices = tuple(self.transition_end_indices)
        if transition_end_indices:
            if len(transition_end_indices) != len(actions):
                raise ValueError(
                    "Stage A episode transition end indices must match actions"
                )
            previous_transition_end: int | None = None
            for value in transition_end_indices:
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(
                        "Stage A episode transition end indices must be non-negative integers"
                    )
                if (
                    previous_transition_end is not None
                    and value <= previous_transition_end
                ):
                    raise ValueError(
                        "Stage A episode transition end indices must be strictly increasing"
                    )
                previous_transition_end = value

        events = tuple(self.order_events)
        if not events:
            raise ValueError("Stage A episode order events must not be empty")
        if any(not isinstance(event, OrderEvent) for event in events):
            raise ValueError(
                "Stage A episode order events must contain OrderEvent values"
            )
        funding = tuple(self.funding_evidence)
        previous_index: int | None = None
        previous_timestamp: int | None = None
        for index, boundary in enumerate(funding):
            if not isinstance(boundary, FundingBoundaryEvidence):
                raise ValueError(
                    f"Stage A episode funding_evidence[{index}] is invalid"
                )
            if previous_index is not None and (
                boundary.processing_index <= previous_index
                or previous_timestamp is None
                or boundary.timestamp_ns <= previous_timestamp
            ):
                raise ValueError(
                    "Stage A episode funding evidence must be strictly increasing"
                )
            previous_index = boundary.processing_index
            previous_timestamp = boundary.timestamp_ns
        if not isinstance(self.terminal_book, BookState):
            raise ValueError("Stage A episode terminal book must be BookState")
        if not isinstance(self.terminal_order_book, OrderBookState):
            raise ValueError(
                "Stage A episode terminal order book must be OrderBookState"
            )
        terminal_equity = _finite_float(
            self.terminal_book.portfolio_value,
            field="Stage A episode terminal book equity",
        )
        tolerance = max(1e-9, abs(equity[-1]) * 1e-12)
        if not math.isclose(
            terminal_equity,
            equity[-1],
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            raise ValueError("Stage A episode terminal equity mismatch")

        object.__setattr__(self, "policy_source_digest", policy_source_digest)
        object.__setattr__(self, "actions", tuple(actions))
        object.__setattr__(self, "observation_digests", observations)
        object.__setattr__(self, "equity_curve", equity)
        object.__setattr__(self, "transition_end_indices", transition_end_indices)
        object.__setattr__(self, "order_events", events)
        object.__setattr__(self, "funding_evidence", funding)

    def validate_against(
        self,
        request: StageAEvaluationCellRequest,
        *,
        expected_policy_source_digest: str | None,
        expected_candidate_config_digest: str,
    ) -> Self:
        """Reconcile the result with the exact request and policy source."""

        if self.request_digest != request.digest:
            raise ValueError("Stage A episode request digest mismatch")
        require_sha256(
            expected_candidate_config_digest,
            field="expected_stage_a_candidate_config_digest",
        )
        if self.candidate_config_digest != expected_candidate_config_digest:
            raise ValueError("Stage A episode candidate config digest mismatch")

        expected_source = _optional_sha256(
            expected_policy_source_digest,
            field="expected_stage_a_policy_source_digest",
        )
        if request.is_baseline:
            if self.policy_source_digest is not None:
                raise ValueError(
                    "Stage A baseline episode must not define a policy source"
                )
            if expected_source is not None:
                raise ValueError(
                    "Stage A baseline request must not expect a policy source"
                )
        else:
            if self.policy_source_digest is None:
                raise ValueError("Stage A policy episode requires a policy source")
            if expected_source is None:
                raise ValueError("Stage A policy request requires a policy source")
        if self.policy_source_digest != expected_source:
            raise ValueError("Stage A episode policy source digest mismatch")

        for event in self.order_events:
            if event.dataset_id != request.dataset_id:
                raise ValueError("Stage A episode order event dataset mismatch")
            if event.execution_policy_digest != request.execution_identity:
                raise ValueError("Stage A episode order event execution mismatch")
        start = request.evaluation_range.start
        stop = request.evaluation_range.stop
        if any(
            value <= start or value > stop for value in self.transition_end_indices
        ):
            raise ValueError(
                "Stage A episode transition end index outside request range"
            )
        if self.transition_end_indices and self.transition_end_indices[-1] != stop:
            raise ValueError("Stage A episode terminal transition end index mismatch")
        if any(
            boundary.processing_index < start or boundary.processing_index > stop
            for boundary in self.funding_evidence
        ):
            raise ValueError("Stage A episode funding evidence outside request range")
        return self


class StageAEvaluationEpisodeExecutor(Protocol):
    """Execute one exact Stage A request against an optional policy."""

    def execute(
        self,
        request: StageAEvaluationCellRequest,
        *,
        policy: object | None,
        policy_source_digest: str | None,
        candidate_config_digest: str,
    ) -> StageAEvaluationEpisodeResult: ...


class StageAPolicySourceReader(Protocol):
    """Read immutable policy-source bindings by request identity."""

    root: Path

    def load(self, request_digest: str) -> StageAPolicySourceBinding: ...


class StageAExecutionCostResolver(Protocol):
    """Resolve the exact execution-cost contract for one request."""

    def resolve(self, request: StageAEvaluationCellRequest) -> ExecutionCostConfig: ...


class StageAExecutionArtifactStore(Protocol):
    """Publish and reload immutable Stage A execution artifacts."""

    def publish(
        self,
        *,
        request: StageAEvaluationCellRequest,
        candidate_config_digest: str,
        actions: tuple[tuple[float, ...], ...],
        observation_digests: tuple[str, ...],
        equity_curve: tuple[float, ...],
        event_artifact_path: str | Path,
        execution_evidence_path: str | Path,
        funding_evidence_path: str | Path | None = None,
    ) -> StoredStageAExecutionReplay: ...

    def load(self, request_digest: str) -> StoredStageAExecutionReplay: ...


def _validate_runtime_handle(
    *,
    binding: StageAPolicySourceBinding,
    request: StageAEvaluationCellRequest,
    handle: StageAPolicyRuntimeHandle,
) -> None:
    if handle.binding_digest != binding.digest:
        raise ValueError("Stage A runtime binding digest mismatch")
    if handle.plan_digest != binding.plan_digest:
        raise ValueError("Stage A runtime plan digest mismatch")
    if handle.request_digest != request.digest:
        raise ValueError("Stage A runtime request digest mismatch")
    if handle.candidate_id != binding.candidate_id:
        raise ValueError("Stage A runtime candidate identity mismatch")
    if handle.seed != binding.seed:
        raise ValueError("Stage A runtime seed identity mismatch")
    if handle.checkpoint_digest != binding.checkpoint_digest:
        raise ValueError("Stage A runtime checkpoint mismatch")
    if handle.candidate_config_digest != binding.candidate_config_digest:
        raise ValueError("Stage A runtime config digest mismatch")
    if handle.checkpoint_policy_digest != binding.checkpoint_policy_digest:
        raise ValueError("Stage A runtime policy digest mismatch")
    if binding.serving_bundle_digest is None:
        raise ValueError("Stage A policy execution requires a serving bundle")
    if handle.serving_bundle_digest != binding.serving_bundle_digest:
        raise ValueError("Stage A runtime serving bundle mismatch")
    if handle.policy is None:
        raise ValueError("Stage A runtime policy is missing")


class StageAExecutionArtifactProducer:
    """Produce one exact policy-bound or baseline Stage A execution artifact."""

    def __init__(
        self,
        *,
        plan: StageAZeroShotEvaluationPlan,
        manifest: StageAEvaluationDatasetManifest,
        policy_source_store: StageAPolicySourceReader,
        policy_runtime_loader: StageAPolicyRuntimeLoader,
        episode_executor: StageAEvaluationEpisodeExecutor,
        execution_store: StageAExecutionArtifactStore,
        execution_cost_resolver: StageAExecutionCostResolver,
        baseline_config_digest: str,
    ) -> None:
        require_sha256(
            baseline_config_digest,
            field="stage_a_baseline_config_digest",
        )
        plan.validate_manifest(manifest)
        self.plan = plan
        self.manifest = manifest
        self.policy_source_store = policy_source_store
        self.policy_runtime_loader = policy_runtime_loader
        self.episode_executor = episode_executor
        self.execution_store = execution_store
        self.execution_cost_resolver = execution_cost_resolver
        self.baseline_config_digest = baseline_config_digest

    def _policy_inputs(
        self,
        *,
        request: StageAEvaluationCellRequest,
        candidate: StageACandidate | None,
    ) -> tuple[object | None, str | None, str]:
        if candidate is None:
            return None, None, self.baseline_config_digest
        binding = self.policy_source_store.load(request.digest)
        binding.validate(
            root=self.policy_source_store.root,
            plan=self.plan,
            manifest=self.manifest,
            request=request,
        )
        if binding.candidate_config_digest != candidate.candidate_config_digest:
            raise ValueError("Stage A policy source config digest mismatch")
        handle = self.policy_runtime_loader.load(
            plan=self.plan,
            manifest=self.manifest,
            request=request,
            binding=binding,
        )
        _validate_runtime_handle(binding=binding, request=request, handle=handle)
        return handle.policy, binding.digest, binding.candidate_config_digest

    def produce(
        self,
        request: StageAEvaluationCellRequest,
    ) -> StoredStageAExecutionReplay:
        """Execute, derive canonical evidence, publish, reload, and return one cell."""

        candidate = _candidate_for_request(
            plan=self.plan, manifest=self.manifest, request=request
        )
        cost = self.execution_cost_resolver.resolve(request)
        if cost.execution_policy_digest != request.execution_identity:
            raise ValueError("Stage A execution cost identity mismatch")
        if cost.path_mode != "conservative":
            raise ValueError("Stage A production execution requires conservative cost")

        policy, policy_source_digest, candidate_config_digest = self._policy_inputs(
            request=request,
            candidate=candidate,
        )
        result = self.episode_executor.execute(
            request,
            policy=policy,
            policy_source_digest=policy_source_digest,
            candidate_config_digest=candidate_config_digest,
        )
        result.validate_against(
            request,
            expected_policy_source_digest=policy_source_digest,
            expected_candidate_config_digest=candidate_config_digest,
        )

        with tempfile.TemporaryDirectory(prefix="stage-a-execution-") as directory:
            source_root = Path(directory)
            event_artifact = build_execution_event_artifact(
                candidate_config_digest=candidate_config_digest,
                evaluation_run_digest=request.digest,
                fold=request.fold,
                seed=request.seed,
                dataset_id=request.dataset_id,
                execution_policy_digest=request.execution_identity,
                actions=result.actions,
                observation_digests=result.observation_digests,
                equity_curve=result.equity_curve,
                order_events=result.order_events,
                terminal_book=result.terminal_book,
                terminal_order_book=result.terminal_order_book,
            )
            event_path = write_execution_event_artifact(
                source_root / "order-events.json",
                event_artifact,
            )
            evidence = execution_evidence_from_cost(
                dataset_id=request.dataset_id,
                cost=cost,
                sensitivity_path_modes=("conservative",),
                order_event_artifact_path=event_path,
            )
            evidence_path = source_root / "execution-evidence.json"
            write_execution_evidence(evidence_path, evidence)
            funding = build_funding_evidence_artifact(
                dataset_id=request.dataset_id,
                execution_policy_digest=request.execution_identity,
                symbol_count=len(result.terminal_book.quantities),
                boundaries=result.funding_evidence,
            )
            funding_path = source_root / "funding-evidence.json"
            funding_path.write_bytes(funding.raw_bytes)
            published = self.execution_store.publish(
                request=request,
                candidate_config_digest=candidate_config_digest,
                actions=result.actions,
                observation_digests=result.observation_digests,
                equity_curve=result.equity_curve,
                event_artifact_path=event_path,
                execution_evidence_path=evidence_path,
                funding_evidence_path=funding_path,
            )
        loaded = self.execution_store.load(request.digest)
        if loaded != published:
            raise ValueError("Stage A execution store reload mismatch")
        return loaded


__all__ = [
    "StageAEvaluationEpisodeExecutor",
    "StageAEvaluationEpisodeResult",
    "StageAExecutionArtifactProducer",
    "StageAExecutionArtifactStore",
    "StageAExecutionCostResolver",
    "StageAPolicySourceReader",
]
