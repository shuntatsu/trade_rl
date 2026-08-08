"""Persist representative real-window Nautilus parity evidence fail-closed."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.workflows.stage_a_nautilus_economic_comparison import (
    StageANautilusHistoricalEconomicEvidence,
)
from trade_rl.workflows.stage_a_nautilus_historical_differential import (
    StageANautilusHistoricalDifferentialEvidence,
)

STAGE_A_NAUTILUS_REPRESENTATIVE_EVIDENCE_SCHEMA: Final = (
    "stage_a_nautilus_representative_evidence_v1"
)
_REPRESENTATIVE_TIME_QUANTILES: Final = (0.1, 0.5, 0.9)


@dataclass(frozen=True, slots=True)
class RepresentativeNautilusWindowEvidence:
    """Structural and economic evidence for one preselected historical window."""

    time_quantile: float
    structural: StageANautilusHistoricalDifferentialEvidence
    economic: StageANautilusHistoricalEconomicEvidence

    def __post_init__(self) -> None:
        if self.time_quantile not in _REPRESENTATIVE_TIME_QUANTILES:
            raise ValueError("unsupported representative time quantile")
        if self.economic.replay_digest != self.structural.replay_digest:
            raise ValueError("economic replay identity mismatch")
        if self.economic.structural_passed != self.structural.structural_passed:
            raise ValueError("economic structural result mismatch")
        if self.economic.funding_matches != self.structural.funding_matches:
            raise ValueError("economic funding result mismatch")

    @property
    def exact_parity_passed(self) -> bool:
        return self.structural.structural_passed and self.economic.economic_passed


@dataclass(frozen=True, slots=True)
class RepresentativeNautilusEvidence:
    """Immutable evidence covering all maintained representative time windows."""

    digest: str
    source_digest: str
    windows: tuple[RepresentativeNautilusWindowEvidence, ...]
    exact_parity_passed: bool
    schema_version: str = STAGE_A_NAUTILUS_REPRESENTATIVE_EVIDENCE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != STAGE_A_NAUTILUS_REPRESENTATIVE_EVIDENCE_SCHEMA:
            raise ValueError("unsupported representative Nautilus evidence schema")
        require_sha256(self.source_digest, field="source_digest")
        if self.time_quantiles != _REPRESENTATIVE_TIME_QUANTILES:
            raise ValueError("representative time quantiles must be 0.1, 0.5, and 0.9")
        expected_passed = all(window.exact_parity_passed for window in self.windows)
        if self.exact_parity_passed is not expected_passed:
            raise ValueError("representative exact parity result mismatch")
        if self.digest != content_digest(self.digest_payload()):
            raise ValueError("representative Nautilus evidence digest mismatch")

    @property
    def time_quantiles(self) -> tuple[float, ...]:
        return tuple(window.time_quantile for window in self.windows)

    def digest_payload(self) -> dict[str, object]:
        return {
            "exact_parity_passed": self.exact_parity_passed,
            "schema_version": self.schema_version,
            "source_digest": self.source_digest,
            "windows": [_window_mapping(window) for window in self.windows],
        }

    def to_mapping(self) -> dict[str, object]:
        return {"digest": self.digest, **self.digest_payload()}


def build_representative_nautilus_evidence(
    *,
    source_digest: str,
    windows: tuple[RepresentativeNautilusWindowEvidence, ...],
) -> RepresentativeNautilusEvidence:
    """Build exact three-window evidence without changing runtime authority."""

    require_sha256(source_digest, field="source_digest")
    quantiles = tuple(window.time_quantile for window in windows)
    if quantiles != _REPRESENTATIVE_TIME_QUANTILES:
        raise ValueError("representative time quantiles must be 0.1, 0.5, and 0.9")
    exact_parity_passed = all(window.exact_parity_passed for window in windows)
    payload = {
        "exact_parity_passed": exact_parity_passed,
        "schema_version": STAGE_A_NAUTILUS_REPRESENTATIVE_EVIDENCE_SCHEMA,
        "source_digest": source_digest,
        "windows": [_window_mapping(window) for window in windows],
    }
    return RepresentativeNautilusEvidence(
        digest=content_digest(payload),
        source_digest=source_digest,
        windows=windows,
        exact_parity_passed=exact_parity_passed,
    )


def write_representative_nautilus_evidence(
    path: str | Path,
    evidence: RepresentativeNautilusEvidence,
) -> Path:
    """Write immutable canonical evidence; identical re-publication is idempotent."""

    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json_bytes(evidence.to_mapping())
    if resolved.exists():
        if resolved.read_bytes() != encoded:
            raise FileExistsError(
                f"refusing to overwrite representative Nautilus evidence: {resolved}"
            )
        return resolved
    temporary = resolved.with_name(f".{resolved.name}.tmp")
    temporary.write_bytes(encoded)
    try:
        temporary.replace(resolved)
    finally:
        temporary.unlink(missing_ok=True)
    return resolved


def load_representative_nautilus_evidence(
    path: str | Path,
) -> RepresentativeNautilusEvidence:
    """Load and fully revalidate persisted representative evidence."""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("representative Nautilus evidence must be an object")
    try:
        raw_windows = raw["windows"]
        if not isinstance(raw_windows, list):
            raise ValueError("representative windows must be a list")
        windows = tuple(_window_from_mapping(value) for value in raw_windows)
        return RepresentativeNautilusEvidence(
            digest=_require_string(raw, "digest"),
            source_digest=_require_string(raw, "source_digest"),
            windows=windows,
            exact_parity_passed=_require_bool(raw, "exact_parity_passed"),
            schema_version=_require_string(raw, "schema_version"),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("representative Nautilus evidence is invalid") from error


def _window_mapping(window: RepresentativeNautilusWindowEvidence) -> dict[str, object]:
    return {
        "economic": asdict(window.economic),
        "structural": asdict(window.structural),
        "time_quantile": window.time_quantile,
    }


def _window_from_mapping(raw: object) -> RepresentativeNautilusWindowEvidence:
    if not isinstance(raw, Mapping):
        raise ValueError("representative window must be an object")
    structural_raw = raw["structural"]
    economic_raw = raw["economic"]
    if not isinstance(structural_raw, Mapping) or not isinstance(economic_raw, Mapping):
        raise ValueError("representative window evidence must be objects")
    return RepresentativeNautilusWindowEvidence(
        time_quantile=float(raw["time_quantile"]),
        structural=StageANautilusHistoricalDifferentialEvidence(**structural_raw),
        economic=StageANautilusHistoricalEconomicEvidence(**economic_raw),
    )


def _require_string(raw: Mapping[str, object], field: str) -> str:
    value = raw[field]
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value


def _require_bool(raw: Mapping[str, object], field: str) -> bool:
    value = raw[field]
    if type(value) is not bool:
        raise ValueError(f"{field} must be a boolean")
    return value


__all__ = [
    "RepresentativeNautilusEvidence",
    "RepresentativeNautilusWindowEvidence",
    "STAGE_A_NAUTILUS_REPRESENTATIVE_EVIDENCE_SCHEMA",
    "build_representative_nautilus_evidence",
    "load_representative_nautilus_evidence",
    "write_representative_nautilus_evidence",
]
