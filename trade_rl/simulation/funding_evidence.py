"""Immutable funding-boundary evidence emitted by stateful execution."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.domain.common import require_sha256

FUNDING_BOUNDARY_ARTIFACT_SCHEMA = "execution_funding_boundary_artifact_v1"


def _integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _positive_integer(value: object, *, field: str) -> int:
    result = _integer(value, field=field)
    if result <= 0:
        raise ValueError(f"{field} must be positive")
    return result


def _number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{field} must be an object with string keys")
    return value


def _sequence(value: object, *, field: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{field} must be a sequence")
    return value


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value


@dataclass(frozen=True, slots=True)
class FundingBoundaryEvidence:
    """Economic inputs and equity closure for one actual funding boundary."""

    processing_index: int
    timestamp_ns: int
    funding_due: tuple[bool, ...]
    signed_quantities: tuple[float, ...]
    mark_prices: tuple[float, ...]
    contract_multipliers: tuple[float, ...]
    funding_rates: tuple[float, ...]
    funding_amount: float
    equity_before_funding: float
    equity_after_funding: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.processing_index, bool)
            or not isinstance(self.processing_index, int)
            or self.processing_index < 0
        ):
            raise ValueError("processing_index must be a non-negative integer")
        if isinstance(self.timestamp_ns, bool) or not isinstance(
            self.timestamp_ns, int
        ):
            raise ValueError("timestamp_ns must be an integer")

        size = len(self.funding_due)
        if size == 0:
            raise ValueError(
                "funding boundary evidence must contain at least one symbol"
            )
        if not all(isinstance(value, bool) for value in self.funding_due):
            raise ValueError("funding_due values must be booleans")
        if not any(self.funding_due):
            raise ValueError(
                "funding boundary evidence requires at least one due symbol"
            )
        if any(
            len(values) != size
            for values in (
                self.signed_quantities,
                self.mark_prices,
                self.contract_multipliers,
                self.funding_rates,
            )
        ):
            raise ValueError("funding boundary evidence vectors must have equal length")
        if any(not math.isfinite(value) for value in self.signed_quantities):
            raise ValueError("signed_quantities must be finite")
        if any(not math.isfinite(value) or value <= 0.0 for value in self.mark_prices):
            raise ValueError("mark_prices must be finite and positive")
        if any(
            not math.isfinite(value) or value <= 0.0
            for value in self.contract_multipliers
        ):
            raise ValueError("contract_multipliers must be finite and positive")
        if any(not math.isfinite(value) for value in self.funding_rates):
            raise ValueError("funding_rates must be finite")
        for name, value in (
            ("funding_amount", self.funding_amount),
            ("equity_before_funding", self.equity_before_funding),
            ("equity_after_funding", self.equity_after_funding),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")

        expected_funding = -math.fsum(
            quantity * mark * multiplier * rate
            for due, quantity, mark, multiplier, rate in zip(
                self.funding_due,
                self.signed_quantities,
                self.mark_prices,
                self.contract_multipliers,
                self.funding_rates,
                strict=True,
            )
            if due
        )
        if not math.isclose(
            self.funding_amount,
            expected_funding,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("funding_amount does not match boundary mark notional")
        if not math.isclose(
            self.equity_after_funding,
            self.equity_before_funding + self.funding_amount,
            rel_tol=1e-12,
            abs_tol=1e-9,
        ):
            raise ValueError("funding boundary equity closure is inconsistent")

    def to_mapping(self) -> dict[str, object]:
        return {
            "contract_multipliers": self.contract_multipliers,
            "equity_after_funding": self.equity_after_funding,
            "equity_before_funding": self.equity_before_funding,
            "funding_amount": self.funding_amount,
            "funding_due": self.funding_due,
            "funding_rates": self.funding_rates,
            "mark_prices": self.mark_prices,
            "processing_index": self.processing_index,
            "signed_quantities": self.signed_quantities,
            "timestamp_ns": self.timestamp_ns,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> FundingBoundaryEvidence:
        required = {
            "contract_multipliers",
            "equity_after_funding",
            "equity_before_funding",
            "funding_amount",
            "funding_due",
            "funding_rates",
            "mark_prices",
            "processing_index",
            "signed_quantities",
            "timestamp_ns",
        }
        if set(value) != required:
            raise ValueError("funding boundary evidence field closure mismatch")

        raw_due = _sequence(value["funding_due"], field="funding_due")
        if any(not isinstance(item, bool) for item in raw_due):
            raise ValueError("funding_due values must be booleans")

        def vector(field: str) -> tuple[float, ...]:
            return tuple(
                _number(item, field=f"{field}[]")
                for item in _sequence(value[field], field=field)
            )

        return cls(
            processing_index=_integer(
                value["processing_index"], field="processing_index"
            ),
            timestamp_ns=_integer(value["timestamp_ns"], field="timestamp_ns"),
            funding_due=tuple(bool(item) for item in raw_due),
            signed_quantities=vector("signed_quantities"),
            mark_prices=vector("mark_prices"),
            contract_multipliers=vector("contract_multipliers"),
            funding_rates=vector("funding_rates"),
            funding_amount=_number(value["funding_amount"], field="funding_amount"),
            equity_before_funding=_number(
                value["equity_before_funding"], field="equity_before_funding"
            ),
            equity_after_funding=_number(
                value["equity_after_funding"], field="equity_after_funding"
            ),
        )


@dataclass(frozen=True, slots=True)
class FundingEvidenceArtifact:
    """Identity-bound canonical funding-boundary trace for one execution replay."""

    dataset_id: str
    execution_policy_digest: str
    symbol_count: int
    boundaries: tuple[FundingBoundaryEvidence, ...]
    schema_version: str = FUNDING_BOUNDARY_ARTIFACT_SCHEMA

    def __post_init__(self) -> None:
        require_sha256(self.dataset_id, field="funding_artifact.dataset_id")
        require_sha256(
            self.execution_policy_digest,
            field="funding_artifact.execution_policy_digest",
        )
        symbol_count = _positive_integer(
            self.symbol_count, field="funding_artifact.symbol_count"
        )
        boundaries = tuple(self.boundaries)
        previous_index: int | None = None
        previous_timestamp: int | None = None
        for position, boundary in enumerate(boundaries):
            if not isinstance(boundary, FundingBoundaryEvidence):
                raise ValueError(
                    f"funding_artifact.boundaries[{position}] must be funding evidence"
                )
            if len(boundary.funding_due) != symbol_count:
                raise ValueError(
                    "funding artifact boundary vector does not match symbol_count"
                )
            if previous_index is not None and (
                boundary.processing_index <= previous_index
                or previous_timestamp is None
                or boundary.timestamp_ns <= previous_timestamp
            ):
                raise ValueError(
                    "funding artifact boundaries must be strictly increasing"
                )
            previous_index = boundary.processing_index
            previous_timestamp = boundary.timestamp_ns
        if self.schema_version != FUNDING_BOUNDARY_ARTIFACT_SCHEMA:
            raise ValueError("unsupported funding boundary artifact schema")
        object.__setattr__(self, "symbol_count", symbol_count)
        object.__setattr__(self, "boundaries", boundaries)

    @property
    def boundary_count(self) -> int:
        return len(self.boundaries)

    def to_mapping(self) -> dict[str, object]:
        return {
            "boundaries": tuple(boundary.to_mapping() for boundary in self.boundaries),
            "dataset_id": self.dataset_id,
            "execution_policy_digest": self.execution_policy_digest,
            "schema_version": self.schema_version,
            "symbol_count": self.symbol_count,
        }

    @property
    def raw_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_mapping()) + b"\n"

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.raw_bytes).hexdigest()

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> FundingEvidenceArtifact:
        required = {
            "boundaries",
            "dataset_id",
            "execution_policy_digest",
            "schema_version",
            "symbol_count",
        }
        if set(value) != required:
            raise ValueError("funding boundary artifact field closure mismatch")
        raw_boundaries = _sequence(value["boundaries"], field="boundaries")
        boundaries = tuple(
            FundingBoundaryEvidence.from_mapping(
                _mapping(item, field=f"boundaries[{index}]")
            )
            for index, item in enumerate(raw_boundaries)
        )
        return cls(
            dataset_id=_string(value["dataset_id"], field="dataset_id"),
            execution_policy_digest=_string(
                value["execution_policy_digest"], field="execution_policy_digest"
            ),
            symbol_count=_positive_integer(value["symbol_count"], field="symbol_count"),
            boundaries=boundaries,
            schema_version=_string(value["schema_version"], field="schema_version"),
        )


def build_funding_evidence_artifact(
    *,
    dataset_id: str,
    execution_policy_digest: str,
    symbol_count: int,
    boundaries: Sequence[FundingBoundaryEvidence],
) -> FundingEvidenceArtifact:
    """Build one canonical funding-boundary artifact from completed execution."""

    return FundingEvidenceArtifact(
        dataset_id=dataset_id,
        execution_policy_digest=execution_policy_digest,
        symbol_count=symbol_count,
        boundaries=tuple(boundaries),
    )


def load_funding_evidence_artifact_bytes(raw: bytes) -> FundingEvidenceArtifact:
    """Load canonical funding evidence bytes and reject non-canonical encodings."""

    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("funding boundary artifact must be valid JSON") from error
    artifact = FundingEvidenceArtifact.from_mapping(
        _mapping(value, field="funding boundary artifact")
    )
    if raw != artifact.raw_bytes:
        raise ValueError("funding boundary artifact must use canonical encoding")
    return artifact


__all__ = [
    "FUNDING_BOUNDARY_ARTIFACT_SCHEMA",
    "FundingBoundaryEvidence",
    "FundingEvidenceArtifact",
    "build_funding_evidence_artifact",
    "load_funding_evidence_artifact_bytes",
]
