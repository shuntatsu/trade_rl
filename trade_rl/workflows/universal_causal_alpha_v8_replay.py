"""Durable V8 replay binding over simulator-authoritative economics."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final, cast

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.learning.causal_alpha_v6 import CausalAlphaV6Candidate
from trade_rl.learning.causal_alpha_v7 import CausalAlphaV7Candidate
from trade_rl.learning.causal_alpha_v8 import CausalAlphaV8Candidate
from trade_rl.learning.rollout_evaluation import (
    ActionPathLifecycleTrace,
    ActionPathStepTrace,
)
from trade_rl.workflows.universal_causal_alpha_v6_replay import (
    CausalAlphaV6ReplayMetric,
)
from trade_rl.workflows.universal_causal_alpha_v7_attribution import (
    CausalAlphaV7AttributionCell,
    CausalAlphaV7AttributionEvidence,
)
from trade_rl.workflows.universal_causal_alpha_v7_replay import (
    CausalAlphaV7ReplayMetric,
)
from trade_rl.workflows.universal_causal_alpha_v8_attribution import (
    CausalAlphaV8AttributionEvidence,
)

CAUSAL_ALPHA_V8_REPLAY_SCHEMA: Final = "causal_alpha_v8_replay_metric_v1"
_V7_CANDIDATE_BY_V8: Final = {
    CausalAlphaV8Candidate.V7_CONTROL: CausalAlphaV7Candidate.V6_CONTROL,
    CausalAlphaV8Candidate.ROBUST_CONTRARIAN: (
        CausalAlphaV7Candidate.SYMMETRIC_CONTRARIAN
    ),
    CausalAlphaV8Candidate.ROBUST_CALIBRATED: (
        CausalAlphaV7Candidate.CAUSAL_CALIBRATED
    ),
}


def _payload(value: object, *, field: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"V8 replay {field} payload is invalid")
    return {str(key): item for key, item in value.items()}


@dataclass(frozen=True, slots=True)
class CausalAlphaV8ReplayMetric:
    candidate: CausalAlphaV8Candidate
    v6_metric: CausalAlphaV6ReplayMetric
    attribution: CausalAlphaV8AttributionEvidence
    v8_target_path_digest: str
    source_forecast_digest: str
    calibration_fit_digest: str
    v8_config_digest: str
    schema_version: str = CAUSAL_ALPHA_V8_REPLAY_SCHEMA
    digest: str = ""
    step_trace: ActionPathStepTrace | None = None
    lifecycle_trace: ActionPathLifecycleTrace | None = None

    def __post_init__(self) -> None:
        candidate = CausalAlphaV8Candidate(self.candidate)
        if not isinstance(self.v6_metric, CausalAlphaV6ReplayMetric):
            raise TypeError("V8 replay requires a V6 replay metric")
        if self.v6_metric.candidate is not CausalAlphaV6Candidate.FAST_ONLY:
            raise ValueError("V8 replay must use V6 fast-only economics")
        if not isinstance(self.attribution, CausalAlphaV8AttributionEvidence):
            raise TypeError("V8 replay attribution is invalid")
        if self.attribution.candidate is not candidate:
            raise ValueError("V8 replay attribution candidate drifted")
        for name in (
            "v8_target_path_digest",
            "source_forecast_digest",
            "calibration_fit_digest",
            "v8_config_digest",
        ):
            require_sha256(getattr(self, name), field=f"V8 replay {name}")
        if self.attribution.target_path_digest != self.v8_target_path_digest:
            raise ValueError("V8 replay target/attribution identity drifted")
        if self.step_trace is not None:
            if not isinstance(self.step_trace, ActionPathStepTrace):
                raise TypeError("V8 replay step trace is invalid")
            if self.step_trace.decision_count != self.v6_metric.decision_count:
                raise ValueError("V8 replay step trace count drifted")
        if self.lifecycle_trace is not None:
            if not isinstance(self.lifecycle_trace, ActionPathLifecycleTrace):
                raise TypeError("V8 replay lifecycle trace is invalid")
            if self.lifecycle_trace.decision_count != self.v6_metric.decision_count:
                raise ValueError("V8 replay lifecycle trace count drifted")
        for observed, expected in (
            (self.attribution.gross_log_return, self.v6_metric.gross_return),
            (self.attribution.net_log_return, self.v6_metric.net_return),
            (
                self.attribution.total_execution_cost,
                self.v6_metric.total_execution_cost,
            ),
        ):
            if not math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("V8 replay attribution economics drifted")
        if self.schema_version != CAUSAL_ALPHA_V8_REPLAY_SCHEMA:
            raise ValueError("unsupported V8 replay schema")
        object.__setattr__(self, "candidate", candidate)
        expected_digest = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected_digest:
            raise ValueError("V8 replay metric digest mismatch")
        object.__setattr__(self, "digest", expected_digest)

    @property
    def identity(self) -> tuple[str, str, int]:
        return (
            self.candidate.value,
            self.v6_metric.symbol,
            self.v6_metric.episode_index,
        )

    def as_v7_metric(self) -> CausalAlphaV7ReplayMetric:
        candidate = _V7_CANDIDATE_BY_V8[self.candidate]
        source = self.attribution
        attribution = CausalAlphaV7AttributionEvidence(
            candidate=candidate,
            target_path_digest=self.v8_target_path_digest,
            boundaries_digest=source.boundaries_digest,
            step_economics_digest=source.step_economics_digest,
            decision_count=source.decision_count,
            gross_log_return=source.gross_log_return,
            net_log_return=source.net_log_return,
            total_execution_cost=source.total_execution_cost,
            total_exposure_hours=source.total_exposure_hours,
            cells=source.cells,
        )
        return CausalAlphaV7ReplayMetric(
            candidate=candidate,
            v6_metric=self.v6_metric,
            attribution=attribution,
            v7_target_path_digest=self.v8_target_path_digest,
            source_forecast_digest=self.source_forecast_digest,
            calibration_fit_digest=self.calibration_fit_digest,
            v7_config_digest=self.v8_config_digest,
        )

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "attribution": self.attribution.to_payload(),
            "calibration_fit_digest": self.calibration_fit_digest,
            "candidate": self.candidate.value,
            "schema_version": self.schema_version,
            "source_forecast_digest": self.source_forecast_digest,
            "v6_metric": self.v6_metric.to_payload(),
            "v8_config_digest": self.v8_config_digest,
            "v8_target_path_digest": self.v8_target_path_digest,
        }
        if self.step_trace is not None:
            payload["step_trace"] = self.step_trace.to_payload()
        if self.lifecycle_trace is not None:
            payload["lifecycle_trace"] = self.lifecycle_trace.to_payload()
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload

    @classmethod
    def from_payload(cls, value: object) -> CausalAlphaV8ReplayMetric:
        payload = _payload(value, field="root")
        v6_payload = _payload(payload["v6_metric"], field="V6 metric")
        v6_digest = str(v6_payload.pop("artifact_digest"))
        v6_payload["candidate"] = CausalAlphaV6Candidate(str(v6_payload["candidate"]))
        for name in (
            "completed_holding_durations_hours",
            "execution_rejection_reason_counts",
            "risk_projection_reason_counts",
            "target_reason_counts",
        ):
            v6_payload[name] = tuple(cast(Any, v6_payload[name]))
        v6_kwargs: Any = v6_payload
        v6 = CausalAlphaV6ReplayMetric(**v6_kwargs, digest=v6_digest)
        attribution_payload = _payload(payload["attribution"], field="attribution")
        attribution_digest = str(attribution_payload.pop("artifact_digest"))
        cells = tuple(
            CausalAlphaV7AttributionCell(**cast(Any, _payload(cell, field="cell")))
            for cell in tuple(cast(Any, attribution_payload.pop("cells")))
        )
        attribution_payload.pop("dimensions", None)
        attribution_payload["candidate"] = CausalAlphaV8Candidate(
            str(attribution_payload["candidate"])
        )
        attribution_kwargs: Any = attribution_payload
        attribution = CausalAlphaV8AttributionEvidence(
            **attribution_kwargs,
            cells=cells,
            digest=attribution_digest,
        )
        trace_payload = payload.pop("step_trace", None)
        trace = (
            None
            if trace_payload is None
            else ActionPathStepTrace.from_payload(trace_payload)
        )
        lifecycle_payload = payload.pop("lifecycle_trace", None)
        lifecycle = (
            None
            if lifecycle_payload is None
            else ActionPathLifecycleTrace.from_payload(lifecycle_payload)
        )
        root_digest = str(payload.pop("artifact_digest"))
        payload.pop("v6_metric")
        payload.pop("attribution")
        payload["candidate"] = CausalAlphaV8Candidate(str(payload["candidate"]))
        root_kwargs: Any = payload
        return cls(
            **root_kwargs,
            v6_metric=v6,
            attribution=attribution,
            digest=root_digest,
            step_trace=trace,
            lifecycle_trace=lifecycle,
        )


__all__ = ["CAUSAL_ALPHA_V8_REPLAY_SCHEMA", "CausalAlphaV8ReplayMetric"]
