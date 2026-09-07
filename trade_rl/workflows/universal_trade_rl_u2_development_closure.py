"""Authoritative Universal Trade RL U2 closure before Development access."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Final

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.domain.universal_trade_rl_universe import UniversalTradeRLSymbolRole
from trade_rl.rl.checkpointing import CheckpointManifest
from trade_rl.workflows.universal_trade_rl_u2_contract import (
    U2_TRAINING_SEEDS,
    UniversalTradeRLU2Contract,
)
from trade_rl.workflows.universal_trade_rl_u2_predevelopment import (
    U2_ADMISSION_STATUS,
    U2_PRODUCTION_STATUS,
    UniversalTradeRLU2DevelopmentLock,
    UniversalTradeRLU2PreDevelopmentContract,
    build_universal_trade_rl_u2_predevelopment_contract,
)
from trade_rl.workflows.universal_trade_rl_u2_training import (
    UniversalTradeRLU2SeedTrainingPlan,
    require_universal_trade_rl_u2_selection_checkpoint,
)
from trade_rl.workflows.universal_trade_rl_universe_manifest import (
    UniversalTradeRLUniverseManifest,
)

U2_FINAL_CHECKPOINT_CLOSURE_SCHEMA: Final = (
    "universal_trade_rl_u2_final_checkpoint_closure_v1"
)
U2_TRAINING_EXPOSURE_ROW_SCHEMA: Final = (
    "universal_trade_rl_u2_training_exposure_row_v1"
)
U2_TRAINING_EXPOSURE_EVIDENCE_SCHEMA: Final = (
    "universal_trade_rl_u2_training_exposure_evidence_v1"
)
U2_AUTHORITATIVE_DEVELOPMENT_LOCK_SCHEMA: Final = (
    "universal_trade_rl_u2_authoritative_development_lock_v1"
)
U2_DEVELOPMENT_REPLAY_AUTHORITY_SCHEMA: Final = (
    "universal_trade_rl_u2_development_replay_authority_v1"
)
U2_WORKER_INDICES: Final = tuple(range(8))


def _non_negative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _canonical_digest_pairs(
    values: tuple[tuple[int, str], ...],
    *,
    field: str,
) -> tuple[tuple[int, str], ...]:
    if not isinstance(values, tuple):
        raise TypeError(f"{field} must be an immutable tuple")
    result: list[tuple[int, str]] = []
    for item in values:
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError(f"{field} must contain seed/digest pairs")
        seed, digest = item
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"{field} seed must be an integer")
        require_sha256(digest, field=f"{field} digest")
        result.append((seed, digest))
    resolved = tuple(result)
    if tuple(seed for seed, _digest in resolved) != U2_TRAINING_SEEDS:
        raise ValueError(f"{field} must use exact canonical seeds {U2_TRAINING_SEEDS}")
    return resolved


def _canonical_train_symbols(symbols: tuple[str, ...]) -> tuple[str, ...]:
    if (
        not isinstance(symbols, tuple)
        or not symbols
        or any(not isinstance(symbol, str) or not symbol for symbol in symbols)
    ):
        raise ValueError("U2 training exposure Train symbols are invalid")
    if symbols != tuple(sorted(set(symbols))):
        raise ValueError("U2 training exposure Train symbols must be sorted and unique")
    return symbols


def _role_symbols(
    manifest: UniversalTradeRLUniverseManifest,
    role: UniversalTradeRLSymbolRole,
) -> tuple[str, ...]:
    return tuple(entry.symbol for entry in manifest.entries if entry.role is role)


def _require_predevelopment_u2_identity(
    *,
    predevelopment_contract: UniversalTradeRLU2PreDevelopmentContract,
    u2_contract: UniversalTradeRLU2Contract,
) -> None:
    if not isinstance(
        predevelopment_contract,
        UniversalTradeRLU2PreDevelopmentContract,
    ):
        raise TypeError("U2 closure requires a pre-development contract")
    if not isinstance(u2_contract, UniversalTradeRLU2Contract):
        raise TypeError("U2 closure requires a UniversalTradeRLU2Contract")
    if predevelopment_contract.u2_contract_digest != u2_contract.digest:
        raise ValueError("U2 closure U2 contract identity mismatch")
    if (
        predevelopment_contract.universe_manifest_digest
        != u2_contract.universe_manifest_digest
    ):
        raise ValueError("U2 closure universe manifest identity mismatch")


def build_authoritative_universal_trade_rl_u2_predevelopment_contract(
    *,
    manifest: UniversalTradeRLUniverseManifest,
    u2_contract: UniversalTradeRLU2Contract,
) -> UniversalTradeRLU2PreDevelopmentContract:
    """Freeze pre-Development semantics against a real U2 contract object."""

    if not isinstance(manifest, UniversalTradeRLUniverseManifest):
        raise TypeError("authoritative U2 pre-development closure requires a manifest")
    if not isinstance(u2_contract, UniversalTradeRLU2Contract):
        raise TypeError("authoritative U2 pre-development closure requires U2 contract")
    if u2_contract.universe_manifest_digest != manifest.digest:
        raise ValueError("authoritative U2 universe manifest identity mismatch")
    return build_universal_trade_rl_u2_predevelopment_contract(
        manifest=manifest,
        u2_contract_digest=u2_contract.digest,
        u2_universe_manifest_digest=u2_contract.universe_manifest_digest,
    )


@dataclass(frozen=True, slots=True)
class UniversalTradeRLU2FinalCheckpointClosure:
    """Validated exact-final checkpoint closure for all preregistered seeds."""

    predevelopment_contract_digest: str
    universe_manifest_digest: str
    u2_contract_digest: str
    source_closure_digest: str
    u1_contract_digest: str
    normalizer_digest: str
    time_partition_digest: str
    training_config_digest: str
    training_plan_digests: tuple[tuple[int, str], ...]
    checkpoint_digests: tuple[tuple[int, str], ...]
    environment_digests: tuple[tuple[int, str], ...]
    production_status: str = U2_PRODUCTION_STATUS
    schema_version: str = U2_FINAL_CHECKPOINT_CLOSURE_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != U2_FINAL_CHECKPOINT_CLOSURE_SCHEMA:
            raise ValueError("unsupported U2 final-checkpoint closure schema")
        for field_name, value in (
            ("predevelopment_contract_digest", self.predevelopment_contract_digest),
            ("universe_manifest_digest", self.universe_manifest_digest),
            ("u2_contract_digest", self.u2_contract_digest),
            ("source_closure_digest", self.source_closure_digest),
            ("u1_contract_digest", self.u1_contract_digest),
            ("normalizer_digest", self.normalizer_digest),
            ("time_partition_digest", self.time_partition_digest),
            ("training_config_digest", self.training_config_digest),
        ):
            require_sha256(value, field=f"U2 final-checkpoint closure {field_name}")
        plans = _canonical_digest_pairs(
            self.training_plan_digests,
            field="U2 final-checkpoint training-plan mapping",
        )
        checkpoints = _canonical_digest_pairs(
            self.checkpoint_digests,
            field="U2 final-checkpoint checkpoint mapping",
        )
        environments = _canonical_digest_pairs(
            self.environment_digests,
            field="U2 final-checkpoint environment mapping",
        )
        if self.production_status != U2_PRODUCTION_STATUS:
            raise ValueError("Universal Trade RL U2 remains Production NO-GO")
        object.__setattr__(self, "training_plan_digests", plans)
        object.__setattr__(self, "checkpoint_digests", checkpoints)
        object.__setattr__(self, "environment_digests", environments)
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest:
            require_sha256(self.digest, field="U2 final-checkpoint closure digest")
            if self.digest != expected:
                raise ValueError("U2 final-checkpoint closure digest mismatch")
        object.__setattr__(self, "digest", expected)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "predevelopment_contract_digest": self.predevelopment_contract_digest,
            "universe_manifest_digest": self.universe_manifest_digest,
            "u2_contract_digest": self.u2_contract_digest,
            "source_closure_digest": self.source_closure_digest,
            "u1_contract_digest": self.u1_contract_digest,
            "normalizer_digest": self.normalizer_digest,
            "time_partition_digest": self.time_partition_digest,
            "training_config_digest": self.training_config_digest,
            "training_plan_digests": self.training_plan_digests,
            "checkpoint_digests": self.checkpoint_digests,
            "environment_digests": self.environment_digests,
            "production_status": self.production_status,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


def build_universal_trade_rl_u2_final_checkpoint_closure(
    *,
    predevelopment_contract: UniversalTradeRLU2PreDevelopmentContract,
    u2_contract: UniversalTradeRLU2Contract,
    members: tuple[
        tuple[UniversalTradeRLU2SeedTrainingPlan, CheckpointManifest, str], ...
    ],
) -> UniversalTradeRLU2FinalCheckpointClosure:
    """Validate and bind exact-final checkpoints for seeds 0/1/2."""

    _require_predevelopment_u2_identity(
        predevelopment_contract=predevelopment_contract,
        u2_contract=u2_contract,
    )
    if not isinstance(members, tuple) or len(members) != len(U2_TRAINING_SEEDS):
        raise ValueError(
            "U2 final checkpoint closure requires exactly three seed members"
        )
    if tuple(member[0].seed for member in members) != U2_TRAINING_SEEDS:
        raise ValueError(
            "U2 final checkpoint closure requires exact canonical seed order"
        )

    source_closure_digests: set[str] = set()
    plan_digests: list[tuple[int, str]] = []
    checkpoint_digests: list[tuple[int, str]] = []
    environment_digests: list[tuple[int, str]] = []
    for plan, checkpoint, expected_environment_digest in members:
        if not isinstance(plan, UniversalTradeRLU2SeedTrainingPlan):
            raise TypeError("U2 final checkpoint closure member plan is invalid")
        if not isinstance(checkpoint, CheckpointManifest):
            raise TypeError("U2 final checkpoint closure member checkpoint is invalid")
        require_sha256(
            expected_environment_digest,
            field="U2 final checkpoint expected environment digest",
        )
        if plan.u2_contract_digest != u2_contract.digest:
            raise ValueError("U2 final checkpoint training plan U2 identity mismatch")
        if plan.u1_contract_digest != u2_contract.u1_contract_digest:
            raise ValueError("U2 final checkpoint training plan U1 identity mismatch")
        if plan.normalizer_digest != u2_contract.u1_normalizer_digest:
            raise ValueError("U2 final checkpoint training plan normalizer mismatch")
        if plan.time_partition_digest != u2_contract.time_partition_digest:
            raise ValueError(
                "U2 final checkpoint training plan time partition mismatch"
            )
        if plan.training_config_digest != u2_contract.training_config_digest:
            raise ValueError("U2 final checkpoint training config identity mismatch")
        require_universal_trade_rl_u2_selection_checkpoint(
            plan=plan,
            checkpoint=checkpoint,
            expected_environment_digest=expected_environment_digest,
        )
        source_closure_digests.add(plan.source_closure_digest)
        plan_digests.append((plan.seed, plan.digest))
        checkpoint_digests.append((plan.seed, checkpoint.digest))
        environment_digests.append((plan.seed, expected_environment_digest))

    if len(source_closure_digests) != 1:
        raise ValueError("U2 final checkpoint members must share one source closure")
    source_closure_digest = next(iter(source_closure_digests))
    return UniversalTradeRLU2FinalCheckpointClosure(
        predevelopment_contract_digest=predevelopment_contract.digest,
        universe_manifest_digest=predevelopment_contract.universe_manifest_digest,
        u2_contract_digest=u2_contract.digest,
        source_closure_digest=source_closure_digest,
        u1_contract_digest=u2_contract.u1_contract_digest,
        normalizer_digest=u2_contract.u1_normalizer_digest,
        time_partition_digest=u2_contract.time_partition_digest,
        training_config_digest=u2_contract.training_config_digest,
        training_plan_digests=tuple(plan_digests),
        checkpoint_digests=tuple(checkpoint_digests),
        environment_digests=tuple(environment_digests),
    )


@dataclass(frozen=True, slots=True)
class UniversalTradeRLU2TrainingExposureRow:
    """One seed/worker/symbol training-exposure observation."""

    training_seed: int
    worker_index: int
    concrete_symbol: str
    completed_episode_count: int
    decision_step_count: int
    partial_final_episode_step_count: int
    routing_cycle_count: int
    schema_version: str = U2_TRAINING_EXPOSURE_ROW_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != U2_TRAINING_EXPOSURE_ROW_SCHEMA:
            raise ValueError("unsupported U2 training-exposure row schema")
        if (
            isinstance(self.training_seed, bool)
            or not isinstance(self.training_seed, int)
            or self.training_seed not in U2_TRAINING_SEEDS
        ):
            raise ValueError("U2 training-exposure seed is not preregistered")
        if (
            isinstance(self.worker_index, bool)
            or not isinstance(self.worker_index, int)
            or self.worker_index not in U2_WORKER_INDICES
        ):
            raise ValueError("U2 training-exposure worker index is invalid")
        if not isinstance(self.concrete_symbol, str) or not self.concrete_symbol:
            raise ValueError("U2 training-exposure symbol is invalid")
        for field_name in (
            "completed_episode_count",
            "decision_step_count",
            "partial_final_episode_step_count",
            "routing_cycle_count",
        ):
            _non_negative_int(
                getattr(self, field_name),
                field=f"U2 training-exposure {field_name}",
            )
        if self.partial_final_episode_step_count > self.decision_step_count:
            raise ValueError("U2 training-exposure partial steps exceed decision steps")
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest:
            require_sha256(self.digest, field="U2 training-exposure row digest")
            if self.digest != expected:
                raise ValueError("U2 training-exposure row digest mismatch")
        object.__setattr__(self, "digest", expected)

    @property
    def identity(self) -> tuple[int, int, str]:
        return (self.training_seed, self.worker_index, self.concrete_symbol)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "training_seed": self.training_seed,
            "worker_index": self.worker_index,
            "concrete_symbol": self.concrete_symbol,
            "completed_episode_count": self.completed_episode_count,
            "decision_step_count": self.decision_step_count,
            "partial_final_episode_step_count": self.partial_final_episode_step_count,
            "routing_cycle_count": self.routing_cycle_count,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


@dataclass(frozen=True, slots=True)
class UniversalTradeRLU2TrainingExposureEvidence:
    """Complete seed × worker × Train-symbol exposure closure."""

    predevelopment_contract_digest: str
    universe_manifest_digest: str
    u2_contract_digest: str
    expected_train_symbols: tuple[str, ...]
    rows: tuple[UniversalTradeRLU2TrainingExposureRow, ...]
    posthoc_reweighting_allowed: bool = False
    required_before_development_open: bool = True
    schema_version: str = U2_TRAINING_EXPOSURE_EVIDENCE_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != U2_TRAINING_EXPOSURE_EVIDENCE_SCHEMA:
            raise ValueError("unsupported U2 training-exposure evidence schema")
        for field_name, value in (
            ("predevelopment_contract_digest", self.predevelopment_contract_digest),
            ("universe_manifest_digest", self.universe_manifest_digest),
            ("u2_contract_digest", self.u2_contract_digest),
        ):
            require_sha256(value, field=f"U2 training-exposure {field_name}")
        symbols = _canonical_train_symbols(self.expected_train_symbols)
        rows = tuple(self.rows)
        if any(
            not isinstance(row, UniversalTradeRLU2TrainingExposureRow) for row in rows
        ):
            raise TypeError("U2 training-exposure evidence contains an invalid row")
        if tuple(row.identity for row in rows) != tuple(
            sorted(row.identity for row in rows)
        ):
            raise ValueError(
                "U2 training-exposure rows must be in canonical identity order"
            )
        expected_identities = tuple(
            (seed, worker, symbol)
            for seed, worker, symbol in product(
                U2_TRAINING_SEEDS,
                U2_WORKER_INDICES,
                symbols,
            )
        )
        if tuple(row.identity for row in rows) != expected_identities:
            raise ValueError(
                "U2 training-exposure evidence must cover the complete grid"
            )
        for seed in U2_TRAINING_SEEDS:
            for worker in U2_WORKER_INDICES:
                scoped = tuple(
                    row
                    for row in rows
                    if row.training_seed == seed and row.worker_index == worker
                )
                if sum(row.partial_final_episode_step_count > 0 for row in scoped) > 1:
                    raise ValueError(
                        "U2 training-exposure worker has multiple partial final episodes"
                    )
        if self.posthoc_reweighting_allowed:
            raise ValueError("U2 training exposure forbids post-hoc reweighting")
        if not self.required_before_development_open:
            raise ValueError("U2 training exposure is required before Development open")
        object.__setattr__(self, "expected_train_symbols", symbols)
        object.__setattr__(self, "rows", rows)
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest:
            require_sha256(self.digest, field="U2 training-exposure evidence digest")
            if self.digest != expected:
                raise ValueError("U2 training-exposure evidence digest mismatch")
        object.__setattr__(self, "digest", expected)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "predevelopment_contract_digest": self.predevelopment_contract_digest,
            "universe_manifest_digest": self.universe_manifest_digest,
            "u2_contract_digest": self.u2_contract_digest,
            "expected_train_symbols": self.expected_train_symbols,
            "row_digests": tuple(row.digest for row in self.rows),
            "posthoc_reweighting_allowed": self.posthoc_reweighting_allowed,
            "required_before_development_open": self.required_before_development_open,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


def build_universal_trade_rl_u2_training_exposure_evidence(
    *,
    predevelopment_contract: UniversalTradeRLU2PreDevelopmentContract,
    manifest: UniversalTradeRLUniverseManifest,
    u2_contract: UniversalTradeRLU2Contract,
    rows: tuple[UniversalTradeRLU2TrainingExposureRow, ...],
) -> UniversalTradeRLU2TrainingExposureEvidence:
    """Bind complete routing exposure to the frozen U0/U2 generation."""

    _require_predevelopment_u2_identity(
        predevelopment_contract=predevelopment_contract,
        u2_contract=u2_contract,
    )
    if not isinstance(manifest, UniversalTradeRLUniverseManifest):
        raise TypeError("U2 training exposure requires a universe manifest")
    if manifest.digest != u2_contract.universe_manifest_digest:
        raise ValueError("U2 training exposure universe manifest identity mismatch")
    return UniversalTradeRLU2TrainingExposureEvidence(
        predevelopment_contract_digest=predevelopment_contract.digest,
        universe_manifest_digest=manifest.digest,
        u2_contract_digest=u2_contract.digest,
        expected_train_symbols=_role_symbols(
            manifest,
            UniversalTradeRLSymbolRole.TRAIN,
        ),
        rows=rows,
    )


@dataclass(frozen=True, slots=True)
class UniversalTradeRLU2AuthoritativeDevelopmentLock:
    """Outer Development lock binding validated checkpoints and exposure evidence."""

    base_lock_digest: str
    predevelopment_contract_digest: str
    universe_manifest_digest: str
    u1_contract_digest: str
    u1_normalizer_digest: str
    u2_contract_digest: str
    final_checkpoint_closure_digest: str
    training_exposure_evidence_digest: str
    replay_authority_schema: str = U2_DEVELOPMENT_REPLAY_AUTHORITY_SCHEMA
    admission_status: str = U2_ADMISSION_STATUS
    production_status: str = U2_PRODUCTION_STATUS
    schema_version: str = U2_AUTHORITATIVE_DEVELOPMENT_LOCK_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != U2_AUTHORITATIVE_DEVELOPMENT_LOCK_SCHEMA:
            raise ValueError("unsupported authoritative U2 Development lock schema")
        for field_name, value in (
            ("base_lock_digest", self.base_lock_digest),
            ("predevelopment_contract_digest", self.predevelopment_contract_digest),
            ("universe_manifest_digest", self.universe_manifest_digest),
            ("u1_contract_digest", self.u1_contract_digest),
            ("u1_normalizer_digest", self.u1_normalizer_digest),
            ("u2_contract_digest", self.u2_contract_digest),
            ("final_checkpoint_closure_digest", self.final_checkpoint_closure_digest),
            (
                "training_exposure_evidence_digest",
                self.training_exposure_evidence_digest,
            ),
        ):
            require_sha256(
                value, field=f"authoritative U2 Development lock {field_name}"
            )
        if self.replay_authority_schema != U2_DEVELOPMENT_REPLAY_AUTHORITY_SCHEMA:
            raise ValueError("U2 Development replay authority schema drifted")
        if self.admission_status != U2_ADMISSION_STATUS:
            raise ValueError("U2 Development lock requires Admission SEALED")
        if self.production_status != U2_PRODUCTION_STATUS:
            raise ValueError("Universal Trade RL U2 remains Production NO-GO")
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest:
            require_sha256(
                self.digest, field="authoritative U2 Development lock digest"
            )
            if self.digest != expected:
                raise ValueError("authoritative U2 Development lock digest mismatch")
        object.__setattr__(self, "digest", expected)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "base_lock_digest": self.base_lock_digest,
            "predevelopment_contract_digest": self.predevelopment_contract_digest,
            "universe_manifest_digest": self.universe_manifest_digest,
            "u1_contract_digest": self.u1_contract_digest,
            "u1_normalizer_digest": self.u1_normalizer_digest,
            "u2_contract_digest": self.u2_contract_digest,
            "final_checkpoint_closure_digest": self.final_checkpoint_closure_digest,
            "training_exposure_evidence_digest": self.training_exposure_evidence_digest,
            "replay_authority_schema": self.replay_authority_schema,
            "admission_status": self.admission_status,
            "production_status": self.production_status,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


def build_authoritative_universal_trade_rl_u2_development_lock(
    *,
    base_lock: UniversalTradeRLU2DevelopmentLock,
    checkpoint_closure: UniversalTradeRLU2FinalCheckpointClosure,
    training_exposure_evidence: UniversalTradeRLU2TrainingExposureEvidence,
    manifest: UniversalTradeRLUniverseManifest,
    u2_contract: UniversalTradeRLU2Contract,
) -> UniversalTradeRLU2AuthoritativeDevelopmentLock:
    """Authorize the exact frozen Development evaluation generation, not access itself."""

    if not isinstance(base_lock, UniversalTradeRLU2DevelopmentLock):
        raise TypeError("authoritative U2 Development closure requires a base lock")
    if not isinstance(checkpoint_closure, UniversalTradeRLU2FinalCheckpointClosure):
        raise TypeError(
            "authoritative U2 Development closure requires checkpoint closure"
        )
    if not isinstance(
        training_exposure_evidence,
        UniversalTradeRLU2TrainingExposureEvidence,
    ):
        raise TypeError(
            "authoritative U2 Development closure requires exposure evidence"
        )
    if not isinstance(manifest, UniversalTradeRLUniverseManifest):
        raise TypeError("authoritative U2 Development closure requires a manifest")
    if not isinstance(u2_contract, UniversalTradeRLU2Contract):
        raise TypeError("authoritative U2 Development closure requires U2 contract")
    if manifest.digest != u2_contract.universe_manifest_digest:
        raise ValueError(
            "authoritative U2 Development universe manifest identity mismatch"
        )
    if base_lock.universe_manifest_digest != manifest.digest:
        raise ValueError("authoritative U2 Development base-lock universe mismatch")
    if base_lock.u2_contract_digest != u2_contract.digest:
        raise ValueError("authoritative U2 Development base-lock U2 identity mismatch")
    if base_lock.u1_contract_digest != u2_contract.u1_contract_digest:
        raise ValueError("authoritative U2 Development base-lock U1 identity mismatch")
    if base_lock.u1_normalizer_digest != u2_contract.u1_normalizer_digest:
        raise ValueError("authoritative U2 Development base-lock normalizer mismatch")
    if (
        base_lock.predevelopment_contract_digest
        != checkpoint_closure.predevelopment_contract_digest
        or base_lock.predevelopment_contract_digest
        != training_exposure_evidence.predevelopment_contract_digest
    ):
        raise ValueError(
            "authoritative U2 Development pre-development identity mismatch"
        )
    if checkpoint_closure.u2_contract_digest != u2_contract.digest:
        raise ValueError("authoritative U2 Development checkpoint U2 identity mismatch")
    if training_exposure_evidence.u2_contract_digest != u2_contract.digest:
        raise ValueError("authoritative U2 Development exposure U2 identity mismatch")
    if training_exposure_evidence.universe_manifest_digest != manifest.digest:
        raise ValueError("authoritative U2 Development exposure universe mismatch")
    if base_lock.checkpoint_digests != checkpoint_closure.checkpoint_digests:
        raise ValueError(
            "authoritative U2 Development checkpoint closure mapping mismatch"
        )
    if base_lock.development_numeric_open_count != 0:
        raise ValueError(
            "authoritative U2 Development lock requires zero Development opens"
        )
    if base_lock.admission_numeric_open_count != 0:
        raise ValueError(
            "authoritative U2 Development lock requires zero Admission opens"
        )

    expected_evaluation_symbols = tuple(
        sorted(
            (
                *_role_symbols(manifest, UniversalTradeRLSymbolRole.TRAIN),
                *_role_symbols(manifest, UniversalTradeRLSymbolRole.DEVELOPMENT),
            )
        )
    )
    actual_evaluation_symbols = tuple(
        symbol for symbol, _digest in base_lock.evaluation_dataset_digests
    )
    if actual_evaluation_symbols != expected_evaluation_symbols:
        raise ValueError(
            "authoritative U2 Development evaluation dataset mapping is incomplete"
        )
    admission_symbols = set(
        _role_symbols(manifest, UniversalTradeRLSymbolRole.ADMISSION)
    )
    if set(actual_evaluation_symbols) & admission_symbols:
        raise ValueError(
            "authoritative U2 Development dataset mapping contains Admission"
        )

    return UniversalTradeRLU2AuthoritativeDevelopmentLock(
        base_lock_digest=base_lock.digest,
        predevelopment_contract_digest=base_lock.predevelopment_contract_digest,
        universe_manifest_digest=manifest.digest,
        u1_contract_digest=u2_contract.u1_contract_digest,
        u1_normalizer_digest=u2_contract.u1_normalizer_digest,
        u2_contract_digest=u2_contract.digest,
        final_checkpoint_closure_digest=checkpoint_closure.digest,
        training_exposure_evidence_digest=training_exposure_evidence.digest,
    )


__all__ = [
    "U2_AUTHORITATIVE_DEVELOPMENT_LOCK_SCHEMA",
    "U2_DEVELOPMENT_REPLAY_AUTHORITY_SCHEMA",
    "U2_FINAL_CHECKPOINT_CLOSURE_SCHEMA",
    "U2_TRAINING_EXPOSURE_EVIDENCE_SCHEMA",
    "U2_TRAINING_EXPOSURE_ROW_SCHEMA",
    "U2_WORKER_INDICES",
    "UniversalTradeRLU2AuthoritativeDevelopmentLock",
    "UniversalTradeRLU2FinalCheckpointClosure",
    "UniversalTradeRLU2TrainingExposureEvidence",
    "UniversalTradeRLU2TrainingExposureRow",
    "build_authoritative_universal_trade_rl_u2_development_lock",
    "build_authoritative_universal_trade_rl_u2_predevelopment_contract",
    "build_universal_trade_rl_u2_final_checkpoint_closure",
    "build_universal_trade_rl_u2_training_exposure_evidence",
]
