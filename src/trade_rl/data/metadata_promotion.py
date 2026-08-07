"""Fail-closed promotion evidence for point-in-time execution metadata."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.identity import parse_identity_json
from trade_rl.data.market import MarketDataset
from trade_rl.domain.common import require_non_empty, require_sha256

METADATA_PROMOTION_FILE_NAME = "metadata-promotion.json"
METADATA_PROMOTION_SCHEMA = "metadata_promotion_evidence_v1"
_HISTORICAL_MODE = "historical_signed"
_HISTORICAL_AUTHENTICATION = "ed25519"
_HISTORICAL_APPLICATION = "effective-dated-full-interval"
_ZERO_DIGEST = "0" * 64


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return require_non_empty(value, field=field)


def _limitations(value: object) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError("metadata limitations must be a sequence")
    return tuple(_string(item, field="metadata limitations") for item in value)


@dataclass(frozen=True, slots=True)
class MetadataPromotionEvidence:
    """Dataset-bound decision on whether metadata may enter release promotion."""

    dataset_id: str
    mode: str
    metadata_evidence_digest: str
    source_payload_digest: str
    point_in_time: bool
    authentication: str
    coverage_application: str
    limitations: tuple[str, ...]
    promotable: bool
    schema_version: str = METADATA_PROMOTION_SCHEMA

    def __post_init__(self) -> None:
        require_sha256(self.dataset_id, field="metadata_promotion.dataset_id")
        require_sha256(
            self.metadata_evidence_digest,
            field="metadata_promotion.metadata_evidence_digest",
        )
        require_sha256(
            self.source_payload_digest,
            field="metadata_promotion.source_payload_digest",
        )
        require_non_empty(self.mode, field="metadata_promotion.mode")
        require_non_empty(
            self.authentication,
            field="metadata_promotion.authentication",
        )
        require_non_empty(
            self.coverage_application,
            field="metadata_promotion.coverage_application",
        )
        if self.schema_version != METADATA_PROMOTION_SCHEMA:
            raise ValueError("unsupported metadata promotion evidence schema")
        expected = self._expected_promotable()
        if self.promotable is not expected:
            raise ValueError("metadata promotion flag contradicts its evidence")

    def _expected_promotable(self) -> bool:
        return (
            self.mode == _HISTORICAL_MODE
            and self.point_in_time
            and self.authentication == _HISTORICAL_AUTHENTICATION
            and self.coverage_application == _HISTORICAL_APPLICATION
            and not self.limitations
            and self.source_payload_digest != _ZERO_DIGEST
        )

    def require_promotable(self) -> None:
        if not self.promotable:
            raise ValueError(
                "metadata promotion requires historical_signed point-in-time "
                "Ed25519 evidence with effective-dated full-interval coverage"
            )

    def to_mapping(self) -> dict[str, object]:
        return {
            "authentication": self.authentication,
            "coverage_application": self.coverage_application,
            "dataset_id": self.dataset_id,
            "limitations": self.limitations,
            "metadata_evidence_digest": self.metadata_evidence_digest,
            "mode": self.mode,
            "point_in_time": self.point_in_time,
            "promotable": self.promotable,
            "schema_version": self.schema_version,
            "source_payload_digest": self.source_payload_digest,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> MetadataPromotionEvidence:
        point_in_time = value.get("point_in_time")
        promotable = value.get("promotable")
        if not isinstance(point_in_time, bool):
            raise ValueError("metadata promotion point_in_time must be boolean")
        if not isinstance(promotable, bool):
            raise ValueError("metadata promotion promotable must be boolean")
        return cls(
            dataset_id=_string(value.get("dataset_id"), field="dataset_id"),
            mode=_string(value.get("mode"), field="mode"),
            metadata_evidence_digest=_string(
                value.get("metadata_evidence_digest"),
                field="metadata_evidence_digest",
            ),
            source_payload_digest=_string(
                value.get("source_payload_digest"),
                field="source_payload_digest",
            ),
            point_in_time=point_in_time,
            authentication=_string(value.get("authentication"), field="authentication"),
            coverage_application=_string(
                value.get("coverage_application"), field="coverage_application"
            ),
            limitations=_limitations(value.get("limitations")),
            promotable=promotable,
            schema_version=_string(value.get("schema_version"), field="schema_version"),
        )


def metadata_promotion_from_dataset(
    dataset: MarketDataset,
) -> MetadataPromotionEvidence:
    """Derive immutable promotion evidence from the dataset identity payload."""

    if dataset.identity_payload_json is None:
        return MetadataPromotionEvidence(
            dataset_id=dataset.dataset_id,
            mode="unspecified",
            metadata_evidence_digest=content_digest(
                {"dataset_id": dataset.dataset_id, "metadata_evidence": None}
            ),
            source_payload_digest=_ZERO_DIGEST,
            point_in_time=False,
            authentication="none",
            coverage_application="unspecified",
            limitations=("dataset identity has no execution metadata evidence",),
            promotable=False,
        )
    identity = parse_identity_json(dataset.identity_payload_json)
    raw = identity.get("metadata_evidence")
    if not isinstance(raw, Mapping):
        return MetadataPromotionEvidence(
            dataset_id=dataset.dataset_id,
            mode="unspecified",
            metadata_evidence_digest=content_digest(
                {"dataset_id": dataset.dataset_id, "metadata_evidence": None}
            ),
            source_payload_digest=_ZERO_DIGEST,
            point_in_time=False,
            authentication="none",
            coverage_application="unspecified",
            limitations=("dataset identity has no execution metadata evidence",),
            promotable=False,
        )
    point_in_time = raw.get("point_in_time")
    if not isinstance(point_in_time, bool):
        raise ValueError("dataset metadata point_in_time must be boolean")
    coverage = raw.get("coverage")
    if not isinstance(coverage, Mapping):
        raise ValueError("dataset metadata coverage must be a mapping")
    source_payload_digest = _string(
        raw.get("source_payload_digest"), field="source_payload_digest"
    )
    mode = _string(raw.get("mode"), field="mode")
    authentication = _string(raw.get("authentication"), field="authentication")
    coverage_application = _string(
        coverage.get("application"), field="coverage.application"
    )
    limitations = _limitations(raw.get("limitations"))
    promotable = (
        mode == _HISTORICAL_MODE
        and point_in_time
        and authentication == _HISTORICAL_AUTHENTICATION
        and coverage_application == _HISTORICAL_APPLICATION
        and not limitations
        and source_payload_digest != _ZERO_DIGEST
    )
    return MetadataPromotionEvidence(
        dataset_id=dataset.dataset_id,
        mode=mode,
        metadata_evidence_digest=content_digest(raw),
        source_payload_digest=source_payload_digest,
        point_in_time=point_in_time,
        authentication=authentication,
        coverage_application=coverage_application,
        limitations=limitations,
        promotable=promotable,
    )


def write_metadata_promotion_evidence(
    path: Path,
    evidence: MetadataPromotionEvidence,
) -> None:
    if path.exists():
        raise FileExistsError("metadata promotion evidence already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(evidence.to_mapping()) + b"\n")


def load_metadata_promotion_evidence(path: Path) -> MetadataPromotionEvidence:
    if not path.is_file():
        raise FileNotFoundError("metadata promotion evidence is missing")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("metadata promotion evidence must be a mapping")
    return MetadataPromotionEvidence.from_mapping(raw)


__all__ = [
    "METADATA_PROMOTION_FILE_NAME",
    "METADATA_PROMOTION_SCHEMA",
    "MetadataPromotionEvidence",
    "load_metadata_promotion_evidence",
    "metadata_promotion_from_dataset",
    "write_metadata_promotion_evidence",
]
