from __future__ import annotations

import hashlib
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from tests.stage_a_helpers import stage_a_test_manifest, stage_a_test_manifest_for_plan
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.stage_a_zero_shot_contracts import (
    StageACandidate,
    StageAZeroShotEvaluationPlan,
    build_stage_a_zero_shot_evaluation_plan,
)
from trade_rl.integrations.stage_a_sb3_evaluation import (
    StageAEvaluationEnvironmentHandle,
    StageASB3EvaluationEpisodeExecutor,
)
from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.simulation.orders import OrderBookState, OrderEvent, OrderStatus
from trade_rl.workflows.stage_a_zero_shot_runner_contracts import (
    StageAEvaluationCellRequest,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _plan_and_manifest() -> tuple[
    StageAZeroShotEvaluationPlan, object
]:
    source_manifest = stage_a_test_manifest(
        symbol_disjoint_manifest_digest=_digest("symbol-manifest"),
        symbol_disjoint_triplet_manifest_digest=_digest("triplet-manifest"),
        feature_identity=_digest("features"),
        validation_triplet_ids=(_digest("validation-triplet"),),
        test_triplet_ids=(_digest("test-triplet"),),
        folds=(0, 1),
    )
    candidate = StageACandidate.create(
        candidate_id="candidate-a",
        candidate_config_digest=_digest("candidate-a:config"),
        final_training_completion_digest=_digest("candidate-a:complete"),
        policy_identity=_digest("candidate-a:policy"),
        checkpoint_digests=(
            (0, _digest("candidate-a:checkpoint:0")),
            (1, _digest("candidate-a:checkpoint:1")),
        ),
    )
    plan = build_stage_a_zero_shot_evaluation_plan(
        symbol_disjoint_manifest_digest=(
            source_manifest.symbol_disjoint_manifest_digest
        ),
        symbol_disjoint_triplet_manifest_digest=(
            source_manifest.symbol_disjoint_triplet_manifest_digest
        ),
        evaluation_dataset_manifest_digest=source_manifest.digest,
        feature_identity=source_manifest.feature_identity,
        execution_identity=ExecutionCostConfig(
            path_mode="conservative"
        ).execution_policy_digest,
        evaluation_identity=_digest("evaluation"),
        candidates=(candidate,),
        seeds=(0, 1),
        folds=(0, 1),
        validation_triplet_ids=(_digest("validation-triplet"),),
        test_triplet_ids=(_digest("test-triplet"),),
        bootstrap_confidence_level=0.95,
        bootstrap_resamples=1_000,
        bootstrap_seed=17,
        minimum_validation_lower_bound=0.0,
        minimum_test_lower_bound=0.0,
        minimum_validation_worst_triplet_excess=0.0,
        minimum_test_worst_triplet_excess=0.0,
        minimum_validation_worst_seed_excess=0.0,
        minimum_test_worst_seed_excess=0.0,
        minimum_validation_triplet_pass_fraction=1.0,
        minimum_test_triplet_pass_fraction=1.0,
    )
    return plan, stage_a_test_manifest_for_plan(plan)


def _request(*, policy: bool = True) -> StageAEvaluationCellRequest:
    plan, manifest = _plan_and_manifest()
    candidate_id = "candidate-a" if policy else None
    checkpoint_digest = (
        plan.candidate("candidate-a").checkpoint_digest(0) if policy else None
    )
    return StageAEvaluationCellRequest(
        plan_digest=plan.digest,
        split="validation",
        triplet_id=plan.validation_triplet_ids[0],
        fold=0,
        seed=0,
        candidate_id=candidate_id,
        checkpoint_digest=checkpoint_digest,
        evaluation_dataset_manifest_digest=manifest.digest,
        dataset_id=manifest.dataset_id_for(
            "validation", plan.validation_triplet_ids[0]
        ),
        evaluation_range=manifest.range_for("validation", 0),
        feature_identity=plan.feature_identity,
        execution_identity=plan.execution_identity,
        evaluation_identity=plan.evaluation_identity,
    )


def _dataset(request: StageAEvaluationCellRequest) -> MarketDataset:
    n_bars = 200
    timestamps = np.datetime64("2026-01-01", "ns") + np.arange(n_bars) * np.timedelta64(
        15, "m"
    )
    close = np.column_stack(
        [
            100.0 * np.exp(np.arange(n_bars) * 0.0002),
            110.0 * np.exp(np.arange(n_bars) * 0.0001),
            90.0 * np.exp(-np.arange(n_bars) * 0.0001),
        ]
    )
    open_price = np.vstack([close[0], close[:-1]])
    return MarketDataset(
        dataset_id=request.dataset_id,
        symbols=("A", "B", "C"),
        timestamps=timestamps,
        features=np.zeros((n_bars, 3, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=open_price,
        high=np.maximum(open_price, close) * 1.001,
        low=np.minimum(open_price, close) * 0.999,
        close=close,
        volume=np.full_like(close, 1_000_000.0),
        funding_rate=np.zeros_like(close),
        tradable=np.ones_like(close, dtype=np.bool_),
        feature_available=np.ones((n_bars, 3, 1), dtype=np.bool_),
        feature_names=("ret",),
        global_feature_names=("regime",),
        periods_per_year=35_040,
        feature_config_digest=request.feature_identity,
    )


def _book(value: float) -> BookState:
    return BookState(
        quantities=np.zeros(3, dtype=np.float64),
        cash=value,
        mark_prices=np.ones(3, dtype=np.float64),
        peak_value=value,
    )


def _events(
    request: StageAEvaluationCellRequest,
    dataset: MarketDataset,
    *,
    processing_index: int | None = None,
    timestamp_ns: int | None = None,
) -> tuple[OrderEvent, ...]:
    index = request.evaluation_range.start + 1 if processing_index is None else processing_index
    timestamp = (
        int(dataset.timestamps[index].astype(np.int64))
        if timestamp_ns is None
        else timestamp_ns
    )
    order_id = _digest("order")
    common = {
        "schema_version": "order_event_v1",
        "order_id": order_id,
        "replaced_order_id": None,
        "dataset_id": request.dataset_id,
        "execution_policy_digest": request.execution_identity,
        "symbol_index": 0,
        "processing_index": index,
        "timestamp_ns": timestamp,
        "requested_quantity": 1.0,
        "capacity_before": 10.0,
        "capacity_after": 9.0,
        "participation_rate": 0.1,
        "trigger_segment": None,
        "available_volume_fraction": 1.0,
        "reason": None,
        "path_mode": "conservative",
        "path_points": (100.0, 101.0, 99.0, 100.5),
    }
    return (
        OrderEvent(
            sequence=0,
            event_type="submitted",
            previous_status=OrderStatus.SUBMITTED,
            new_status=OrderStatus.SUBMITTED,
            remaining_quantity=1.0,
            filled_quantity=0.0,
            execution_price=None,
            filled_notional=0.0,
            **common,
        ),
        OrderEvent(
            sequence=1,
            event_type="eligible",
            previous_status=OrderStatus.SUBMITTED,
            new_status=OrderStatus.ELIGIBLE,
            remaining_quantity=1.0,
            filled_quantity=0.0,
            execution_price=None,
            filled_notional=0.0,
            **common,
        ),
        OrderEvent(
            sequence=2,
            event_type="filled",
            previous_status=OrderStatus.ELIGIBLE,
            new_status=OrderStatus.FILLED,
            remaining_quantity=0.0,
            filled_quantity=1.0,
            execution_price=100.0,
            filled_notional=100.0,
            **common,
        ),
    )


class _Resolver:
    def __init__(self, dataset: MarketDataset) -> None:
        self.dataset = dataset
        self.requests: list[StageAEvaluationCellRequest] = []

    def resolve(self, request: StageAEvaluationCellRequest) -> MarketDataset:
        self.requests.append(request)
        return self.dataset


class _FakeEnvironment:
    def __init__(
        self,
        request: StageAEvaluationCellRequest,
        dataset: MarketDataset,
        *,
        events: tuple[OrderEvent, ...] | None = None,
        final_index: int | None = None,
    ) -> None:
        self.request = request
        self.dataset = dataset
        self.dataset_id = dataset.dataset_id
        self.execution_policy_digest = request.execution_identity
        self.minimum_start_index = 4
        self.action_space = SimpleNamespace(shape=(3,))
        self.hybrid = _book(1_000.0)
        self.hybrid_order_book = OrderBookState.empty()
        self.current_index = 0
        self.end_index = 0
        self.reset_options: dict[str, object] | None = None
        self.baseline_calls = 0
        self.step_actions: list[np.ndarray] = []
        self.closed = False
        self.events = events or _events(request, dataset)
        self.final_index = final_index

    def reset(
        self, *, seed: int | None = None, options: dict[str, object] | None = None
    ) -> tuple[np.ndarray, dict[str, object]]:
        assert seed == self.request.seed
        assert options is not None
        self.reset_options = dict(options)
        self.current_index = int(options["start_idx"])
        self.end_index = self.current_index + int(options["episode_bars"])
        return np.array([1.0, 2.0], dtype=np.float32), {}

    def baseline_action(self) -> np.ndarray:
        self.baseline_calls += 1
        return np.zeros(3, dtype=np.float32)

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        self.step_actions.append(np.asarray(action, dtype=np.float32).copy())
        self.current_index = self.end_index if self.final_index is None else self.final_index
        self.hybrid = _book(1_010.0)
        return (
            np.array([3.0, 4.0], dtype=np.float32),
            0.01,
            False,
            True,
            {"hybrid_execution": SimpleNamespace(order_events=self.events)},
        )

    def close(self) -> None:
        self.closed = True


class _Factory:
    def __init__(self, environment: _FakeEnvironment) -> None:
        self.environment = environment
        self.calls: list[tuple[StageAEvaluationCellRequest, MarketDataset, str]] = []

    def build(
        self,
        *,
        request: StageAEvaluationCellRequest,
        dataset: MarketDataset,
        candidate_config_digest: str,
    ) -> StageAEvaluationEnvironmentHandle:
        self.calls.append((request, dataset, candidate_config_digest))
        return StageAEvaluationEnvironmentHandle(
            environment=self.environment,
            candidate_config_digest=candidate_config_digest,
        )


class _Policy:
    def __init__(self) -> None:
        self.observations: list[object] = []

    def predict(self, observation: object) -> np.ndarray:
        self.observations.append(observation)
        return np.array([0.25, 0.0, -0.25], dtype=np.float32)


def _executor(
    *,
    request: StageAEvaluationCellRequest,
    dataset: MarketDataset,
    environment: _FakeEnvironment,
) -> tuple[StageASB3EvaluationEpisodeExecutor, _Resolver, _Factory]:
    plan, manifest = _plan_and_manifest()
    resolver = _Resolver(dataset)
    factory = _Factory(environment)
    return (
        StageASB3EvaluationEpisodeExecutor(
            plan=plan,
            manifest=manifest,
            dataset_resolver=resolver,
            environment_factory=factory,
        ),
        resolver,
        factory,
    )


def test_executor_uses_full_dataset_and_exact_request_range() -> None:
    request = _request(policy=True)
    dataset = _dataset(request)
    environment = _FakeEnvironment(request, dataset)
    executor, resolver, factory = _executor(
        request=request, dataset=dataset, environment=environment
    )
    policy = _Policy()
    candidate_digest = _plan_and_manifest()[0].candidate(
        "candidate-a"
    ).candidate_config_digest

    result = executor.execute(
        request,
        policy=policy,
        policy_source_digest=_digest("policy-source"),
        candidate_config_digest=candidate_digest,
    )

    assert resolver.requests == [request]
    assert factory.calls == [(request, dataset, candidate_digest)]
    assert environment.reset_options == {
        "start_idx": request.evaluation_range.start,
        "episode_bars": request.evaluation_range.stop
        - request.evaluation_range.start,
        "initial_state_mode": "cash",
    }
    assert len(policy.observations) == 1
    assert result.actions == ((0.25, 0.0, -0.25),)
    assert len(result.observation_digests) == 2
    assert result.equity_curve == (1_000.0, 1_010.0)
    assert tuple(event.sequence for event in result.order_events) == (0, 1, 2)
    assert environment.closed is True


def test_baseline_execution_uses_environment_baseline_action() -> None:
    request = _request(policy=False)
    dataset = _dataset(request)
    environment = _FakeEnvironment(request, dataset)
    executor, _, _ = _executor(
        request=request, dataset=dataset, environment=environment
    )

    result = executor.execute(
        request,
        policy=None,
        policy_source_digest=None,
        candidate_config_digest=_digest("baseline-config"),
    )

    assert environment.baseline_calls == 1
    assert result.policy_source_digest is None
    assert result.actions == ((0.0, 0.0, 0.0),)


def test_executor_rejects_dataset_identity_substitution() -> None:
    request = _request(policy=True)
    dataset = replace(_dataset(request), dataset_id=_digest("other-dataset"))
    environment = _FakeEnvironment(request, dataset)
    executor, _, _ = _executor(
        request=request, dataset=dataset, environment=environment
    )

    with pytest.raises(ValueError, match="dataset identity mismatch"):
        executor.execute(
            request,
            policy=_Policy(),
            policy_source_digest=_digest("policy-source"),
            candidate_config_digest=(
                _plan_and_manifest()[0]
                .candidate("candidate-a")
                .candidate_config_digest
            ),
        )


def test_executor_rejects_environment_that_stops_before_authorized_range() -> None:
    request = _request(policy=True)
    dataset = _dataset(request)
    environment = _FakeEnvironment(
        request,
        dataset,
        final_index=request.evaluation_range.stop - 1,
    )
    executor, _, _ = _executor(
        request=request, dataset=dataset, environment=environment
    )

    with pytest.raises(ValueError, match="authorized evaluation stop"):
        executor.execute(
            request,
            policy=_Policy(),
            policy_source_digest=_digest("policy-source"),
            candidate_config_digest=(
                _plan_and_manifest()[0]
                .candidate("candidate-a")
                .candidate_config_digest
            ),
        )


def test_executor_rejects_order_event_outside_authorized_range() -> None:
    request = _request(policy=True)
    dataset = _dataset(request)
    environment = _FakeEnvironment(
        request,
        dataset,
        events=_events(
            request,
            dataset,
            processing_index=request.evaluation_range.start,
        ),
    )
    executor, _, _ = _executor(
        request=request, dataset=dataset, environment=environment
    )

    with pytest.raises(ValueError, match="processing index outside"):
        executor.execute(
            request,
            policy=_Policy(),
            policy_source_digest=_digest("policy-source"),
            candidate_config_digest=(
                _plan_and_manifest()[0]
                .candidate("candidate-a")
                .candidate_config_digest
            ),
        )


def test_executor_rejects_order_event_timestamp_substitution() -> None:
    request = _request(policy=True)
    dataset = _dataset(request)
    environment = _FakeEnvironment(
        request,
        dataset,
        events=_events(request, dataset, timestamp_ns=1),
    )
    executor, _, _ = _executor(
        request=request, dataset=dataset, environment=environment
    )

    with pytest.raises(ValueError, match="timestamp mismatch"):
        executor.execute(
            request,
            policy=_Policy(),
            policy_source_digest=_digest("policy-source"),
            candidate_config_digest=(
                _plan_and_manifest()[0]
                .candidate("candidate-a")
                .candidate_config_digest
            ),
        )
