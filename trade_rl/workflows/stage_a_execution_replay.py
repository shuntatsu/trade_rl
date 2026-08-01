"""Exact Stage A execution-cell replay artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_non_empty, require_sha256
from trade_rl.evaluation.stage_a_zero_shot_contracts import StageAEvaluationSplit
from trade_rl.evaluation.walk_forward.folds import IndexRange
from trade_rl.simulation.execution_promotion import (
    ExecutionEvidence,
    validate_execution_promotion,
)
from trade_rl.simulation.execution_replay import (
    ExecutionEventArtifact,
    load_execution_event_artifact_bytes,
)
from trade_rl.workflows.stage_a_zero_shot_runner_contracts import (
    StageAEvaluationCellRequest,
)

STAGE_A_EXECUTION_CELL_IDENTITY_SCHEMA = "stage_a_execution_cell_identity_v2"
STAGE_A_EXECUTION_REPLAY_SCHEMA = "stage_a_execution_replay_v2"
_SPLITS = frozenset({"validation", "test"})


def _non_negative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _positive_int(value: object, *, field: str) -> int:
    result = _non_negative_int(value, field=field)
    if result == 0:
        raise ValueError(f"{field} must be positive")
    return result


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{field} must be an object with string keys")
    return cast(Mapping[str, object], value)


def _sequence(value: object, *, field: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{field} must be a sequence")
    return value


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value


def _optional_string(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    return _string(value, field=field)


def _index_range(value: object, *, field: str) -> IndexRange:
    values = tuple(_sequence(value, field=field))
    if len(values) != 2:
        raise ValueError(f"{field} must contain exactly two integers")
    return IndexRange(
        _non_negative_int(values[0], field=f"{field}.start"),
        _non_negative_int(values[1], field=f"{field}.stop"),
    )


def _number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _terminal_portfolio_value(artifact: ExecutionEventArtifact) -> float:
    book = _mapping(artifact.terminal_book, field="terminal_book")
    cash = _number(book.get("cash"), field="terminal_book.cash")
    quantities = tuple(
        _number(item, field="terminal_book.quantities[]")
        for item in _sequence(book.get("quantities"), field="terminal_book.quantities")
    )
    mark_prices = tuple(
        _number(item, field="terminal_book.mark_prices[]")
        for item in _sequence(
            book.get("mark_prices"), field="terminal_book.mark_prices"
        )
    )
    multipliers = tuple(
        _number(item, field="terminal_book.contract_multipliers[]")
        for item in _sequence(
            book.get("contract_multipliers"),
            field="terminal_book.contract_multipliers",
        )
    )
    if not quantities or not (len(quantities) == len(mark_prices) == len(multipliers)):
        raise ValueError("Stage A terminal book vector closure mismatch")
    value = cash + math.fsum(
        quantity * mark * multiplier
        for quantity, mark, multiplier in zip(
            quantities, mark_prices, multipliers, strict=True
        )
    )
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("Stage A terminal portfolio value must be positive and finite")
    return value


def _load_execution_evidence_bytes(raw: bytes) -> ExecutionEvidence:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("execution evidence must be valid JSON") from error
    evidence = ExecutionEvidence.from_mapping(
        _mapping(value, field="execution evidence")
    )
    expected = canonical_json_bytes(evidence.to_mapping()) + b"\n"
    if raw != expected:
        raise ValueError("execution evidence must use canonical encoding")
    return evidence


@dataclass(frozen=True, slots=True)
class StageAExecutionCellIdentity:
    """Complete A6a request and producer configuration identity."""

    request_digest: str
    plan_digest: str
    split: StageAEvaluationSplit
    triplet_id: str
    fold: int
    seed: int
    candidate_id: str | None
    checkpoint_digest: str | None
    candidate_config_digest: str
    evaluation_dataset_manifest_digest: str
    dataset_id: str
    evaluation_range: IndexRange
    feature_identity: str
    execution_identity: str
    evaluation_identity: str
    schema_version: str = STAGE_A_EXECUTION_CELL_IDENTITY_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != STAGE_A_EXECUTION_CELL_IDENTITY_SCHEMA:
            raise ValueError("unsupported Stage A execution cell identity schema")
        if self.split not in _SPLITS:
            raise ValueError("Stage A execution cell split is invalid")
        for field_name, value in (
            ("request_digest", self.request_digest),
            ("plan_digest", self.plan_digest),
            ("triplet_id", self.triplet_id),
            ("candidate_config_digest", self.candidate_config_digest),
            (
                "evaluation_dataset_manifest_digest",
                self.evaluation_dataset_manifest_digest,
            ),
            ("dataset_id", self.dataset_id),
            ("feature_identity", self.feature_identity),
            ("execution_identity", self.execution_identity),
            ("evaluation_identity", self.evaluation_identity),
        ):
            require_sha256(value, field=f"stage_a_execution_cell.{field_name}")
        fold = _non_negative_int(self.fold, field="stage_a_execution_cell.fold")
        seed = _non_negative_int(self.seed, field="stage_a_execution_cell.seed")
        if not isinstance(self.evaluation_range, IndexRange):
            raise ValueError("Stage A execution cell range must be an IndexRange")
        if (self.candidate_id is None) != (self.checkpoint_digest is None):
            raise ValueError(
                "Stage A execution policy cell requires candidate and checkpoint"
            )
        candidate_id = self.candidate_id
        checkpoint_digest = self.checkpoint_digest
        if candidate_id is not None:
            candidate_id = require_non_empty(
                candidate_id, field="stage_a_execution_cell.candidate_id"
            )
            assert checkpoint_digest is not None
            require_sha256(
                checkpoint_digest,
                field="stage_a_execution_cell.checkpoint_digest",
            )
        object.__setattr__(self, "fold", fold)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "candidate_id", candidate_id)
        expected = content_digest(self.digest_payload())
        if self.digest and self.digest != expected:
            raise ValueError("Stage A execution cell identity digest mismatch")
        object.__setattr__(self, "digest", expected)

    @classmethod
    def from_request(
        cls,
        request: StageAEvaluationCellRequest,
        *,
        candidate_config_digest: str,
    ) -> StageAExecutionCellIdentity:
        return cls(
            request_digest=request.digest,
            plan_digest=request.plan_digest,
            split=request.split,
            triplet_id=request.triplet_id,
            fold=request.fold,
            seed=request.seed,
            candidate_id=request.candidate_id,
            checkpoint_digest=request.checkpoint_digest,
            candidate_config_digest=candidate_config_digest,
            evaluation_dataset_manifest_digest=(
                request.evaluation_dataset_manifest_digest
            ),
            dataset_id=request.dataset_id,
            evaluation_range=request.evaluation_range,
            feature_identity=request.feature_identity,
            execution_identity=request.execution_identity,
            evaluation_identity=request.evaluation_identity,
        )

    def to_request(self) -> StageAEvaluationCellRequest:
        return StageAEvaluationCellRequest(
            plan_digest=self.plan_digest,
            split=self.split,
            triplet_id=self.triplet_id,
            fold=self.fold,
            seed=self.seed,
            candidate_id=self.candidate_id,
            checkpoint_digest=self.checkpoint_digest,
            evaluation_dataset_manifest_digest=(
                self.evaluation_dataset_manifest_digest
            ),
            dataset_id=self.dataset_id,
            evaluation_range=self.evaluation_range,
            feature_identity=self.feature_identity,
            execution_identity=self.execution_identity,
            evaluation_identity=self.evaluation_identity,
            digest=self.request_digest,
        )

    def digest_payload(self) -> dict[str, object]:
        return {
            "candidate_config_digest": self.candidate_config_digest,
            "candidate_id": self.candidate_id,
            "checkpoint_digest": self.checkpoint_digest,
            "dataset_id": self.dataset_id,
            "evaluation_dataset_manifest_digest": (
                self.evaluation_dataset_manifest_digest
            ),
            "evaluation_identity": self.evaluation_identity,
            "evaluation_range": (
                self.evaluation_range.start,
                self.evaluation_range.stop,
            ),
            "execution_identity": self.execution_identity,
            "feature_identity": self.feature_identity,
            "fold": self.fold,
            "plan_digest": self.plan_digest,
            "request_digest": self.request_digest,
            "schema_version": self.schema_version,
            "seed": self.seed,
            "split": self.split,
            "triplet_id": self.triplet_id,
        }

    def to_json_dict(self) -> dict[str, object]:
        return {"digest": self.digest, **self.digest_payload()}

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> StageAExecutionCellIdentity:
        required = {
            "candidate_config_digest",
            "candidate_id",
            "checkpoint_digest",
            "dataset_id",
            "digest",
            "evaluation_dataset_manifest_digest",
            "evaluation_identity",
            "evaluation_range",
            "execution_identity",
            "feature_identity",
            "fold",
            "plan_digest",
            "request_digest",
            "schema_version",
            "seed",
            "split",
            "triplet_id",
        }
        if set(value) != required:
            raise ValueError("Stage A execution cell identity field closure mismatch")
        split = _string(value["split"], field="split")
        if split not in _SPLITS:
            raise ValueError("Stage A execution cell split is invalid")
        return cls(
            request_digest=_string(value["request_digest"], field="request_digest"),
            plan_digest=_string(value["plan_digest"], field="plan_digest"),
            split=cast(StageAEvaluationSplit, split),
            triplet_id=_string(value["triplet_id"], field="triplet_id"),
            fold=_non_negative_int(value["fold"], field="fold"),
            seed=_non_negative_int(value["seed"], field="seed"),
            candidate_id=_optional_string(value["candidate_id"], field="candidate_id"),
            checkpoint_digest=_optional_string(
                value["checkpoint_digest"], field="checkpoint_digest"
            ),
            candidate_config_digest=_string(
                value["candidate_config_digest"], field="candidate_config_digest"
            ),
            evaluation_dataset_manifest_digest=_string(
                value["evaluation_dataset_manifest_digest"],
                field="evaluation_dataset_manifest_digest",
            ),
            dataset_id=_string(value["dataset_id"], field="dataset_id"),
            evaluation_range=_index_range(
                value["evaluation_range"], field="evaluation_range"
            ),
            feature_identity=_string(
                value["feature_identity"], field="feature_identity"
            ),
            execution_identity=_string(
                value["execution_identity"], field="execution_identity"
            ),
            evaluation_identity=_string(
                value["evaluation_identity"], field="evaluation_identity"
            ),
            schema_version=_string(value["schema_version"], field="schema_version"),
            digest=_string(value["digest"], field="digest"),
        )


@dataclass(frozen=True, slots=True)
class StageAExecutionReplayArtifact:
    """Canonical replay metadata and verified promotion-file identities."""

    cell_identity: StageAExecutionCellIdentity
    actions: tuple[tuple[float, ...], ...]
    observation_digests: tuple[str, ...]
    equity_curve: tuple[float, ...]
    event_artifact_digest: str
    event_artifact_size_bytes: int
    execution_evidence_digest: str
    execution_evidence_sha256: str
    execution_evidence_size_bytes: int
    schema_version: str = STAGE_A_EXECUTION_REPLAY_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != STAGE_A_EXECUTION_REPLAY_SCHEMA:
            raise ValueError("unsupported Stage A execution replay schema")
        if not isinstance(self.cell_identity, StageAExecutionCellIdentity):
            raise ValueError("Stage A execution replay cell identity is invalid")
        actions = tuple(
            tuple(_number(item, field=f"actions[{index}][]") for item in action)
            for index, action in enumerate(self.actions)
        )
        if not actions or any(not action for action in actions):
            raise ValueError("Stage A execution replay actions must not be empty")
        observations = tuple(self.observation_digests)
        if len(observations) != len(actions) + 1:
            raise ValueError("Stage A execution replay observation closure mismatch")
        for index, digest in enumerate(observations):
            require_sha256(digest, field=f"observation_digests[{index}]")
        equity = tuple(
            _number(item, field=f"equity_curve[{index}]")
            for index, item in enumerate(self.equity_curve)
        )
        if len(equity) != len(observations):
            raise ValueError("Stage A execution replay equity closure mismatch")
        if any(value <= 0.0 for value in equity):
            raise ValueError("Stage A execution replay equity curve must be positive")
        for field_name, value in (
            ("event_artifact_digest", self.event_artifact_digest),
            ("execution_evidence_digest", self.execution_evidence_digest),
            ("execution_evidence_sha256", self.execution_evidence_sha256),
        ):
            require_sha256(value, field=f"stage_a_execution_replay.{field_name}")
        event_size = _positive_int(
            self.event_artifact_size_bytes,
            field="stage_a_execution_replay.event_artifact_size_bytes",
        )
        evidence_size = _positive_int(
            self.execution_evidence_size_bytes,
            field="stage_a_execution_replay.execution_evidence_size_bytes",
        )
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "observation_digests", observations)
        object.__setattr__(self, "equity_curve", equity)
        object.__setattr__(self, "event_artifact_size_bytes", event_size)
        object.__setattr__(self, "execution_evidence_size_bytes", evidence_size)
        expected = content_digest(self.digest_payload())
        if self.digest and self.digest != expected:
            raise ValueError("Stage A execution replay digest mismatch")
        object.__setattr__(self, "digest", expected)

    @property
    def log_growth(self) -> float:
        return math.log(self.equity_curve[-1] / self.equity_curve[0])

    def digest_payload(self) -> dict[str, object]:
        return {
            "actions": self.actions,
            "cell_identity": self.cell_identity.to_json_dict(),
            "equity_curve": self.equity_curve,
            "event_artifact_digest": self.event_artifact_digest,
            "event_artifact_size_bytes": self.event_artifact_size_bytes,
            "execution_evidence_digest": self.execution_evidence_digest,
            "execution_evidence_sha256": self.execution_evidence_sha256,
            "execution_evidence_size_bytes": self.execution_evidence_size_bytes,
            "observation_digests": self.observation_digests,
            "schema_version": self.schema_version,
        }

    def to_json_dict(self) -> dict[str, object]:
        return {"digest": self.digest, **self.digest_payload()}

    @property
    def raw_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_json_dict()) + b"\n"

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> StageAExecutionReplayArtifact:
        required = {
            "actions",
            "cell_identity",
            "digest",
            "equity_curve",
            "event_artifact_digest",
            "event_artifact_size_bytes",
            "execution_evidence_digest",
            "execution_evidence_sha256",
            "execution_evidence_size_bytes",
            "observation_digests",
            "schema_version",
        }
        if set(value) != required:
            raise ValueError("Stage A execution replay field closure mismatch")
        actions = tuple(
            tuple(
                _number(item, field=f"actions[{index}][]")
                for item in _sequence(action, field=f"actions[{index}]")
            )
            for index, action in enumerate(_sequence(value["actions"], field="actions"))
        )
        observations = tuple(
            _string(item, field="observation_digests[]")
            for item in _sequence(
                value["observation_digests"], field="observation_digests"
            )
        )
        equity = tuple(
            _number(item, field="equity_curve[]")
            for item in _sequence(value["equity_curve"], field="equity_curve")
        )
        return cls(
            cell_identity=StageAExecutionCellIdentity.from_mapping(
                _mapping(value["cell_identity"], field="cell_identity")
            ),
            actions=actions,
            observation_digests=observations,
            equity_curve=equity,
            event_artifact_digest=_string(
                value["event_artifact_digest"], field="event_artifact_digest"
            ),
            event_artifact_size_bytes=_positive_int(
                value["event_artifact_size_bytes"], field="event_artifact_size_bytes"
            ),
            execution_evidence_digest=_string(
                value["execution_evidence_digest"],
                field="execution_evidence_digest",
            ),
            execution_evidence_sha256=_string(
                value["execution_evidence_sha256"],
                field="execution_evidence_sha256",
            ),
            execution_evidence_size_bytes=_positive_int(
                value["execution_evidence_size_bytes"],
                field="execution_evidence_size_bytes",
            ),
            schema_version=_string(value["schema_version"], field="schema_version"),
            digest=_string(value["digest"], field="digest"),
        )

    @classmethod
    def from_json_bytes(cls, raw: bytes) -> StageAExecutionReplayArtifact:
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Stage A execution replay must be valid JSON") from error
        artifact = cls.from_mapping(_mapping(value, field="Stage A execution replay"))
        if raw != artifact.raw_bytes:
            raise ValueError("Stage A execution replay must use canonical encoding")
        return artifact


def _validate_promotion_bytes(
    *,
    request: StageAEvaluationCellRequest,
    candidate_config_digest: str,
    event_artifact_bytes: bytes,
    execution_evidence_bytes: bytes,
) -> tuple[ExecutionEvidence, ExecutionEventArtifact]:
    event_artifact = load_execution_event_artifact_bytes(event_artifact_bytes)
    if event_artifact.dataset_id != request.dataset_id:
        raise ValueError("Stage A execution event dataset identity mismatch")
    if event_artifact.execution_policy_digest != request.execution_identity:
        raise ValueError("Stage A execution event policy identity mismatch")
    replay_identity = event_artifact.replay_identity
    if replay_identity.candidate_config_digest != candidate_config_digest:
        raise ValueError("Stage A execution candidate configuration identity mismatch")
    if replay_identity.evaluation_run_digest != request.digest:
        raise ValueError("Stage A execution evaluation run identity mismatch")
    if replay_identity.fold != request.fold:
        raise ValueError("Stage A execution fold identity mismatch")
    if replay_identity.seed != request.seed:
        raise ValueError("Stage A execution seed identity mismatch")
    evidence = _load_execution_evidence_bytes(execution_evidence_bytes)
    if evidence.dataset_id != request.dataset_id:
        raise ValueError("Stage A execution evidence dataset identity mismatch")
    with tempfile.TemporaryDirectory(prefix="trade-rl-stage-a-promotion-") as temporary:
        event_path = Path(temporary) / "order-events.json"
        event_path.write_bytes(event_artifact_bytes)
        validate_execution_promotion(
            evidence,
            expected_policy_digest=request.execution_identity,
            event_artifact_path=event_path,
            expected_candidate_config_digest=candidate_config_digest,
            expected_evaluation_run_digest=request.digest,
            expected_fold=request.fold,
            expected_seed=request.seed,
        )
    return evidence, event_artifact


def _validate_embedded_traces(
    artifact: StageAExecutionReplayArtifact,
    event_artifact: ExecutionEventArtifact,
) -> None:
    if event_artifact.actions != artifact.actions:
        raise ValueError("Stage A execution action trace mismatch")
    if event_artifact.observation_digests != artifact.observation_digests:
        raise ValueError("Stage A execution observation trace mismatch")
    if event_artifact.equity_curve != artifact.equity_curve:
        raise ValueError("Stage A execution equity trace mismatch")


def build_stage_a_execution_replay_artifact(
    *,
    request: StageAEvaluationCellRequest,
    candidate_config_digest: str,
    actions: Sequence[Sequence[float]],
    observation_digests: Sequence[str],
    equity_curve: Sequence[float],
    event_artifact_bytes: bytes,
    execution_evidence_bytes: bytes,
) -> StageAExecutionReplayArtifact:
    """Build one replay only after validating the bound promotion bytes."""

    evidence, event_artifact = _validate_promotion_bytes(
        request=request,
        candidate_config_digest=candidate_config_digest,
        event_artifact_bytes=event_artifact_bytes,
        execution_evidence_bytes=execution_evidence_bytes,
    )
    artifact = StageAExecutionReplayArtifact(
        cell_identity=StageAExecutionCellIdentity.from_request(
            request,
            candidate_config_digest=candidate_config_digest,
        ),
        actions=tuple(tuple(float(item) for item in action) for action in actions),
        observation_digests=tuple(observation_digests),
        equity_curve=tuple(float(item) for item in equity_curve),
        event_artifact_digest=hashlib.sha256(event_artifact_bytes).hexdigest(),
        event_artifact_size_bytes=len(event_artifact_bytes),
        execution_evidence_digest=evidence.digest,
        execution_evidence_sha256=hashlib.sha256(execution_evidence_bytes).hexdigest(),
        execution_evidence_size_bytes=len(execution_evidence_bytes),
    )
    terminal_value = _terminal_portfolio_value(event_artifact)
    tolerance = max(1e-12, abs(terminal_value) * 1e-12)
    if not math.isclose(
        artifact.equity_curve[-1],
        terminal_value,
        rel_tol=0.0,
        abs_tol=tolerance,
    ):
        raise ValueError("Stage A execution replay terminal value mismatch")
    _validate_embedded_traces(artifact, event_artifact)
    return artifact


def validate_stage_a_execution_replay_sources(
    artifact: StageAExecutionReplayArtifact,
    *,
    event_artifact_bytes: bytes,
    execution_evidence_bytes: bytes,
) -> ExecutionEvidence:
    """Rebuild one replay from source bytes and require exact equality."""

    request = artifact.cell_identity.to_request()
    evidence, _ = _validate_promotion_bytes(
        request=request,
        candidate_config_digest=artifact.cell_identity.candidate_config_digest,
        event_artifact_bytes=event_artifact_bytes,
        execution_evidence_bytes=execution_evidence_bytes,
    )
    rebuilt = build_stage_a_execution_replay_artifact(
        request=request,
        candidate_config_digest=artifact.cell_identity.candidate_config_digest,
        actions=artifact.actions,
        observation_digests=artifact.observation_digests,
        equity_curve=artifact.equity_curve,
        event_artifact_bytes=event_artifact_bytes,
        execution_evidence_bytes=execution_evidence_bytes,
    )
    if rebuilt != artifact:
        raise ValueError("Stage A execution replay source identity mismatch")
    return evidence


__all__ = [
    "STAGE_A_EXECUTION_CELL_IDENTITY_SCHEMA",
    "STAGE_A_EXECUTION_REPLAY_SCHEMA",
    "StageAExecutionCellIdentity",
    "StageAExecutionReplayArtifact",
    "build_stage_a_execution_replay_artifact",
    "validate_stage_a_execution_replay_sources",
]
