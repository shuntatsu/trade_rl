"""V7 replay binding around simulator-authoritative V6 economics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.learning.causal_alpha_v6 import CausalAlphaV6Candidate
from trade_rl.learning.causal_alpha_v7 import CausalAlphaV7Candidate
from trade_rl.workflows.universal_causal_alpha_v6_replay import (
    CausalAlphaV6ReplayMetric,
)
from trade_rl.workflows.universal_causal_alpha_v7_attribution import (
    CausalAlphaV7AttributionEvidence,
)

CAUSAL_ALPHA_V7_REPLAY_SCHEMA: Final = "causal_alpha_v7_replay_metric_v1"


@dataclass(frozen=True, slots=True)
class CausalAlphaV7ReplayMetric:
    """Bind one V7 candidate and attribution to maintained replay economics."""

    candidate: CausalAlphaV7Candidate
    v6_metric: CausalAlphaV6ReplayMetric
    attribution: CausalAlphaV7AttributionEvidence
    v7_target_path_digest: str
    source_forecast_digest: str
    calibration_fit_digest: str
    v7_config_digest: str
    schema_version: str = CAUSAL_ALPHA_V7_REPLAY_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        candidate = CausalAlphaV7Candidate(self.candidate)
        if not isinstance(self.v6_metric, CausalAlphaV6ReplayMetric):
            raise TypeError("V7 replay requires a V6 replay metric")
        if self.v6_metric.candidate is not CausalAlphaV6Candidate.FAST_ONLY:
            raise ValueError("V7 replay candidates must share V6 fast-only economics")
        if not isinstance(self.attribution, CausalAlphaV7AttributionEvidence):
            raise TypeError("V7 replay attribution is invalid")
        if self.attribution.candidate is not candidate:
            raise ValueError("V7 replay attribution candidate drifted")
        for name in (
            "v7_target_path_digest",
            "source_forecast_digest",
            "calibration_fit_digest",
            "v7_config_digest",
        ):
            require_sha256(getattr(self, name), field=f"V7 replay {name}")
        if self.attribution.target_path_digest != self.v7_target_path_digest:
            raise ValueError("V7 replay target/attribution identity drifted")
        for observed, expected in (
            (self.attribution.gross_log_return, self.v6_metric.gross_return),
            (self.attribution.net_log_return, self.v6_metric.net_return),
            (
                self.attribution.total_execution_cost,
                self.v6_metric.total_execution_cost,
            ),
        ):
            if not math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("V7 replay attribution economics drifted")
        if self.schema_version != CAUSAL_ALPHA_V7_REPLAY_SCHEMA:
            raise ValueError("unsupported V7 replay schema")
        object.__setattr__(self, "candidate", candidate)
        expected_digest = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected_digest:
            raise ValueError("V7 replay metric digest mismatch")
        object.__setattr__(self, "digest", expected_digest)

    @property
    def identity(self) -> tuple[str, str, int]:
        return (
            self.candidate.value,
            self.v6_metric.symbol,
            self.v6_metric.episode_index,
        )

    @property
    def paired_identity(self) -> tuple[object, ...]:
        metric = self.v6_metric
        return (
            metric.symbol,
            metric.episode_index,
            metric.contract_digest,
            metric.fit_digest,
            self.source_forecast_digest,
            self.calibration_fit_digest,
            self.v7_config_digest,
        )

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "attribution_digest": self.attribution.digest,
            "calibration_fit_digest": self.calibration_fit_digest,
            "candidate": self.candidate.value,
            "schema_version": self.schema_version,
            "source_forecast_digest": self.source_forecast_digest,
            "v6_metric_digest": self.v6_metric.digest,
            "v7_config_digest": self.v7_config_digest,
            "v7_target_path_digest": self.v7_target_path_digest,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


__all__ = ["CAUSAL_ALPHA_V7_REPLAY_SCHEMA", "CausalAlphaV7ReplayMetric"]
