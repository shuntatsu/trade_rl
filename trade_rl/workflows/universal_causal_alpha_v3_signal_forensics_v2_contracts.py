"""Contracts for read-only Causal Alpha V3 Signal Forensics V2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, Protocol

from trade_rl.artifacts.hashing import content_digest
from trade_rl.workflows.universal_causal_alpha_v3_signal_forensics import (
    CausalAlphaV3SignalForensicsReport,
    CausalAlphaV3UnavailableAnalysis,
)

CAUSAL_ALPHA_V3_SIGNAL_FORENSICS_V2_SCHEMA: Final = (
    "causal_alpha_v3_signal_forensics_v2"
)
CausalAlphaV3SignalForensicsV2SidecarMode = Literal[
    "historical_unavailable", "sidecar_complete"
]
_ALLOWED_SIDECAR_MODES: Final = frozenset(
    {"historical_unavailable", "sidecar_complete"}
)


class _PayloadConvertible(Protocol):
    def to_payload(self) -> dict[str, object]: ...


@dataclass(frozen=True, slots=True)
class CausalAlphaV3SignalForensicsReportV2:
    base_forensics_digest: str
    base_forensics: CausalAlphaV3SignalForensicsReport
    sidecar_mode: CausalAlphaV3SignalForensicsV2SidecarMode
    sidecar_analysis: _PayloadConvertible | None
    unavailable_analyses: tuple[CausalAlphaV3UnavailableAnalysis, ...]
    research_only: bool = True
    promotion_eligible: bool = False
    schema_version: str = CAUSAL_ALPHA_V3_SIGNAL_FORENSICS_V2_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != CAUSAL_ALPHA_V3_SIGNAL_FORENSICS_V2_SCHEMA:
            raise ValueError("unsupported V3 signal forensics V2 schema")
        if self.research_only is not True or self.promotion_eligible is not False:
            raise ValueError("V3 signal forensics V2 must remain research-only")
        if not isinstance(self.base_forensics, CausalAlphaV3SignalForensicsReport):
            raise TypeError("V3 signal forensics V2 base report is invalid")
        if self.base_forensics_digest != self.base_forensics.digest:
            raise ValueError("V3 signal forensics V2 base report digest mismatch")
        if self.sidecar_mode not in _ALLOWED_SIDECAR_MODES:
            raise ValueError("unsupported V3 signal forensics V2 sidecar mode")
        if self.sidecar_mode == "historical_unavailable":
            if self.sidecar_analysis is not None:
                raise ValueError(
                    "historical V3 signal forensics V2 cannot contain sidecar analysis"
                )
        elif self.sidecar_analysis is None:
            raise ValueError(
                "complete V3 signal forensics V2 sidecar mode requires analysis"
            )
        unavailable = tuple(self.unavailable_analyses)
        if any(
            not isinstance(item, CausalAlphaV3UnavailableAnalysis)
            for item in unavailable
        ):
            raise TypeError("V3 signal forensics V2 unavailable analyses are invalid")
        object.__setattr__(self, "unavailable_analyses", unavailable)
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V3 signal forensics V2 digest mismatch")
        object.__setattr__(self, "digest", expected)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "base_forensics": self.base_forensics.to_payload(),
            "base_forensics_digest": self.base_forensics_digest,
            "promotion_eligible": self.promotion_eligible,
            "research_only": self.research_only,
            "schema_version": self.schema_version,
            "sidecar_analysis": (
                None
                if self.sidecar_analysis is None
                else self.sidecar_analysis.to_payload()
            ),
            "sidecar_mode": self.sidecar_mode,
            "unavailable_analyses": tuple(
                item.to_payload() for item in self.unavailable_analyses
            ),
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload
