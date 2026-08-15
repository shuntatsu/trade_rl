"""Durable training-only teacher artifacts for causal alpha V3."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final, Mapping

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.learning.episode_oracle_teacher import (
    EPISODE_ORACLE_BATCH_SCHEMA,
    EPISODE_ORACLE_CONTRACT_SCHEMA,
    EpisodeOracleBatch,
    OracleEpisodeContract,
)

_BATCH_ARTIFACT_SCHEMA: Final = "causal_alpha_v3_teacher_batch_artifact_v1"
_PACKAGE_SCHEMA: Final = "universal_causal_alpha_v3_teacher_package_v2"


def _strict_payload(
    raw: Mapping[str, Any], *, fields: frozenset[str], schema: str, label: str
) -> dict[str, Any]:
    values = dict(raw)
    if set(values) != fields:
        missing = sorted(fields - set(values))
        unknown = sorted(set(values) - fields)
        raise ValueError(
            f"{label} fields mismatch; missing={missing}, unknown={unknown}"
        )
    if values.get("schema_version") != schema:
        raise ValueError(f"{label} schema is unsupported")
    return values


def _frozen_digests(
    values: Mapping[str, str], *, symbols: tuple[str, ...], label: str
) -> Mapping[str, str]:
    resolved = dict(values)
    if set(resolved) != set(symbols):
        raise ValueError(f"{label} must exactly match train_symbols")
    for symbol in symbols:
        require_sha256(resolved[symbol], field=f"{label}[{symbol}]")
    return MappingProxyType(resolved)


def _contract_payload(contract: OracleEpisodeContract) -> dict[str, object]:
    return {
        "artifact_digest": contract.digest,
        "dataset_id": contract.dataset_id,
        "episode_index": contract.episode_index,
        "initial_state_mode": contract.initial_state_mode,
        "initial_weights": contract.initial_weights.tolist(),
        "schema_version": contract.schema_version,
        "start": contract.start,
        "stop": contract.stop,
    }


def _contract_from_payload(raw: Mapping[str, Any]) -> OracleEpisodeContract:
    fields = frozenset(
        {
            "artifact_digest",
            "dataset_id",
            "episode_index",
            "initial_state_mode",
            "initial_weights",
            "schema_version",
            "start",
            "stop",
        }
    )
    values = _strict_payload(
        raw,
        fields=fields,
        schema=EPISODE_ORACLE_CONTRACT_SCHEMA,
        label="V3 teacher contract",
    )
    return OracleEpisodeContract(
        dataset_id=str(values["dataset_id"]),
        episode_index=int(values["episode_index"]),
        start=int(values["start"]),
        stop=int(values["stop"]),
        initial_state_mode=str(values["initial_state_mode"]),
        initial_weights=np.asarray(values["initial_weights"], dtype=np.float64),
        schema_version=str(values["schema_version"]),
        digest=str(values["artifact_digest"]),
    )


def episode_batch_payload(batch: EpisodeOracleBatch) -> dict[str, object]:
    """Serialize a V3 teacher batch without Oracle solver provenance."""

    if batch.solver_provenance is not None:
        raise ValueError("V3 durable teacher batch cannot claim Oracle provenance")
    return {
        "artifact_digest": batch.digest,
        "contracts": tuple(_contract_payload(item) for item in batch.contracts),
        "dataset_id": batch.dataset_id,
        "sampling_config_digest": batch.sampling_config_digest,
        "schema_version": batch.schema_version,
        "solver_provenance": None,
        "targets": tuple(target.tolist() for target in batch.targets),
        "teacher_config_digest": batch.teacher_config_digest,
    }


def episode_batch_from_payload(raw: Mapping[str, Any]) -> EpisodeOracleBatch:
    fields = frozenset(
        {
            "artifact_digest",
            "contracts",
            "dataset_id",
            "sampling_config_digest",
            "schema_version",
            "solver_provenance",
            "targets",
            "teacher_config_digest",
        }
    )
    values = _strict_payload(
        raw,
        fields=fields,
        schema=EPISODE_ORACLE_BATCH_SCHEMA,
        label="V3 teacher batch",
    )
    if values["solver_provenance"] is not None:
        raise ValueError("V3 teacher batch solver provenance must remain null")
    contracts = tuple(
        _contract_from_payload(item) for item in values["contracts"]
    )
    targets = tuple(
        np.asarray(item, dtype=np.float32) for item in values["targets"]
    )
    return EpisodeOracleBatch(
        dataset_id=str(values["dataset_id"]),
        teacher_config_digest=str(values["teacher_config_digest"]),
        sampling_config_digest=str(values["sampling_config_digest"]),
        contracts=contracts,
        targets=targets,
        solver_provenance=None,
        schema_version=str(values["schema_version"]),
        digest=str(values["artifact_digest"]),
    )


@dataclass(frozen=True, slots=True)
class CausalAlphaV3TeacherBatchArtifact:
    symbol: str
    batch: EpisodeOracleBatch
    partition_digest: str
    sample_digest: str
    admission_contract_digest: str
    run_manifest_digest: str
    freeze_digest: str
    selection_digest: str
    teacher_admission_digest: str
    selected_candidate_digest: str
    schema_version: str = _BATCH_ARTIFACT_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("V3 teacher batch symbol must be non-empty")
        if not isinstance(self.batch, EpisodeOracleBatch):
            raise TypeError("V3 teacher batch requires EpisodeOracleBatch")
        if self.batch.solver_provenance is not None:
            raise ValueError("V3 teacher batch cannot claim Oracle provenance")
        for name in (
            "partition_digest",
            "sample_digest",
            "admission_contract_digest",
            "run_manifest_digest",
            "freeze_digest",
            "selection_digest",
            "teacher_admission_digest",
            "selected_candidate_digest",
        ):
            require_sha256(getattr(self, name), field=f"V3 teacher batch {name}")
        if any(
            contract.digest == self.admission_contract_digest
            for contract in self.batch.contracts
        ):
            raise ValueError("V3 training batch contains the admission holdout")
        if self.schema_version != _BATCH_ARTIFACT_SCHEMA:
            raise ValueError("unsupported V3 teacher batch artifact schema")
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V3 teacher batch artifact digest mismatch")
        object.__setattr__(self, "digest", expected)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "admission_contract_digest": self.admission_contract_digest,
            "batch": episode_batch_payload(self.batch),
            "freeze_digest": self.freeze_digest,
            "partition_digest": self.partition_digest,
            "run_manifest_digest": self.run_manifest_digest,
            "sample_digest": self.sample_digest,
            "schema_version": self.schema_version,
            "selected_candidate_digest": self.selected_candidate_digest,
            "selection_digest": self.selection_digest,
            "symbol": self.symbol,
            "teacher_admission_digest": self.teacher_admission_digest,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload

    @classmethod
    def from_payload(cls, raw: Mapping[str, Any]) -> CausalAlphaV3TeacherBatchArtifact:
        fields = frozenset(
            {
                "admission_contract_digest",
                "artifact_digest",
                "batch",
                "freeze_digest",
                "partition_digest",
                "run_manifest_digest",
                "sample_digest",
                "schema_version",
                "selected_candidate_digest",
                "selection_digest",
                "symbol",
                "teacher_admission_digest",
            }
        )
        values = _strict_payload(
            raw,
            fields=fields,
            schema=_BATCH_ARTIFACT_SCHEMA,
            label="V3 teacher batch artifact",
        )
        return cls(
            symbol=str(values["symbol"]),
            batch=episode_batch_from_payload(values["batch"]),
            partition_digest=str(values["partition_digest"]),
            sample_digest=str(values["sample_digest"]),
            admission_contract_digest=str(values["admission_contract_digest"]),
            run_manifest_digest=str(values["run_manifest_digest"]),
            freeze_digest=str(values["freeze_digest"]),
            selection_digest=str(values["selection_digest"]),
            teacher_admission_digest=str(values["teacher_admission_digest"]),
            selected_candidate_digest=str(values["selected_candidate_digest"]),
            schema_version=str(values["schema_version"]),
            digest=str(values["artifact_digest"]),
        )


@dataclass(frozen=True, slots=True)
class UniversalCausalAlphaV3TeacherPackageV2:
    """Reloadable V3 package containing training episodes only."""

    train_symbols: tuple[str, ...]
    batches: Mapping[str, EpisodeOracleBatch]
    partition_digests: Mapping[str, str]
    sample_digests: Mapping[str, str]
    admission_contract_digests: Mapping[str, str]
    run_manifest_digest: str
    freeze_digest: str
    selection_digest: str
    teacher_admission_digest: str
    selected_candidate_digest: str
    generator_code_digest: str
    teacher_admission_passed: bool = True
    research_only: bool = True
    promotion_eligible: bool = False
    schema_version: str = _PACKAGE_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        symbols = tuple(self.train_symbols)
        batches = dict(self.batches)
        if (
            not symbols
            or len(set(symbols)) != len(symbols)
            or set(batches) != set(symbols)
        ):
            raise ValueError("V3 teacher package batch scope must match train_symbols")
        if any(not isinstance(batches[symbol], EpisodeOracleBatch) for symbol in symbols):
            raise TypeError("V3 teacher package contains an invalid episode batch")
        if any(batches[symbol].solver_provenance is not None for symbol in symbols):
            raise ValueError("V3 teacher package cannot claim Oracle provenance")
        partitions = _frozen_digests(
            self.partition_digests, symbols=symbols, label="V3 partition digests"
        )
        samples = _frozen_digests(
            self.sample_digests, symbols=symbols, label="V3 sample digests"
        )
        admissions = _frozen_digests(
            self.admission_contract_digests,
            symbols=symbols,
            label="V3 admission contract digests",
        )
        for name in (
            "run_manifest_digest",
            "freeze_digest",
            "selection_digest",
            "teacher_admission_digest",
            "selected_candidate_digest",
            "generator_code_digest",
        ):
            require_sha256(getattr(self, name), field=f"V3 teacher package {name}")
        if not self.teacher_admission_passed:
            raise ValueError("V3 teacher package requires passed admission")
        if not self.research_only or self.promotion_eligible:
            raise ValueError("V3 teacher package must remain research-only")
        if self.schema_version != _PACKAGE_SCHEMA:
            raise ValueError("unsupported V3 teacher package schema")
        for symbol in symbols:
            if any(
                contract.digest == admissions[symbol]
                for contract in batches[symbol].contracts
            ):
                raise ValueError("V3 teacher package contains an admission holdout")
        object.__setattr__(self, "train_symbols", symbols)
        object.__setattr__(self, "batches", MappingProxyType(batches))
        object.__setattr__(self, "partition_digests", partitions)
        object.__setattr__(self, "sample_digests", samples)
        object.__setattr__(self, "admission_contract_digests", admissions)
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V3 teacher package digest mismatch")
        object.__setattr__(self, "digest", expected)

    def batch_artifact(self, symbol: str) -> CausalAlphaV3TeacherBatchArtifact:
        if symbol not in self.batches:
            raise ValueError("V3 teacher symbol is outside package scope")
        return CausalAlphaV3TeacherBatchArtifact(
            symbol=symbol,
            batch=self.batches[symbol],
            partition_digest=self.partition_digests[symbol],
            sample_digest=self.sample_digests[symbol],
            admission_contract_digest=self.admission_contract_digests[symbol],
            run_manifest_digest=self.run_manifest_digest,
            freeze_digest=self.freeze_digest,
            selection_digest=self.selection_digest,
            teacher_admission_digest=self.teacher_admission_digest,
            selected_candidate_digest=self.selected_candidate_digest,
        )

    @property
    def batch_artifact_digests(self) -> Mapping[str, str]:
        return MappingProxyType(
            {symbol: self.batch_artifact(symbol).digest for symbol in self.train_symbols}
        )

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "admission_contract_digests": dict(self.admission_contract_digests),
            "batch_artifact_digests": dict(self.batch_artifact_digests),
            "batch_digests": {
                symbol: self.batches[symbol].digest for symbol in self.train_symbols
            },
            "freeze_digest": self.freeze_digest,
            "generator_code_digest": self.generator_code_digest,
            "partition_digests": dict(self.partition_digests),
            "promotion_eligible": self.promotion_eligible,
            "research_only": self.research_only,
            "run_manifest_digest": self.run_manifest_digest,
            "sample_digests": dict(self.sample_digests),
            "schema_version": self.schema_version,
            "selected_candidate_digest": self.selected_candidate_digest,
            "selection_digest": self.selection_digest,
            "teacher_admission_digest": self.teacher_admission_digest,
            "teacher_admission_passed": self.teacher_admission_passed,
            "train_symbols": self.train_symbols,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload

    @classmethod
    def from_payload(
        cls,
        raw: Mapping[str, Any],
        *,
        batches: Mapping[str, EpisodeOracleBatch],
    ) -> UniversalCausalAlphaV3TeacherPackageV2:
        fields = frozenset(
            {
                "admission_contract_digests",
                "artifact_digest",
                "batch_artifact_digests",
                "batch_digests",
                "freeze_digest",
                "generator_code_digest",
                "partition_digests",
                "promotion_eligible",
                "research_only",
                "run_manifest_digest",
                "sample_digests",
                "schema_version",
                "selected_candidate_digest",
                "selection_digest",
                "teacher_admission_digest",
                "teacher_admission_passed",
                "train_symbols",
            }
        )
        values = _strict_payload(
            raw,
            fields=fields,
            schema=_PACKAGE_SCHEMA,
            label="V3 teacher package",
        )
        if (
            values["research_only"] is not True
            or values["promotion_eligible"] is not False
            or values["teacher_admission_passed"] is not True
        ):
            raise ValueError("V3 teacher package safety flags are invalid")
        symbols = tuple(str(item) for item in values["train_symbols"])
        package = cls(
            train_symbols=symbols,
            batches=batches,
            partition_digests={
                str(k): str(v) for k, v in dict(values["partition_digests"]).items()
            },
            sample_digests={
                str(k): str(v) for k, v in dict(values["sample_digests"]).items()
            },
            admission_contract_digests={
                str(k): str(v)
                for k, v in dict(values["admission_contract_digests"]).items()
            },
            run_manifest_digest=str(values["run_manifest_digest"]),
            freeze_digest=str(values["freeze_digest"]),
            selection_digest=str(values["selection_digest"]),
            teacher_admission_digest=str(values["teacher_admission_digest"]),
            selected_candidate_digest=str(values["selected_candidate_digest"]),
            generator_code_digest=str(values["generator_code_digest"]),
            teacher_admission_passed=True,
            research_only=True,
            promotion_eligible=False,
            schema_version=str(values["schema_version"]),
            digest=str(values["artifact_digest"]),
        )
        expected_batches = {
            symbol: package.batches[symbol].digest for symbol in symbols
        }
        if dict(values["batch_digests"]) != expected_batches:
            raise ValueError("V3 teacher package batch identity drifted")
        if dict(values["batch_artifact_digests"]) != dict(
            package.batch_artifact_digests
        ):
            raise ValueError("V3 teacher package batch artifact identity drifted")
        return package


__all__ = [
    "CausalAlphaV3TeacherBatchArtifact",
    "UniversalCausalAlphaV3TeacherPackageV2",
    "episode_batch_from_payload",
    "episode_batch_payload",
]
