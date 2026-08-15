"""Non-promotable diagnostics derived from authoritative Causal Alpha V3 replays."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256

_DIAGNOSTICS_SCHEMA: Final = "causal_alpha_v3_replay_diagnostics_v1"
_EPSILON: Final = 1e-12


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


def _non_negative_int(value: int, *, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")


def _finite(value: float, *, field: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{field} must be finite")


def _reason_counts(
    value: object, *, field: str
) -> tuple[tuple[str, int], ...]:
    try:
        resolved = tuple((str(reason), int(count)) for reason, count in value)  # type: ignore[misc]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must contain reason/count pairs") from error
    if (
        any(not reason or count < 0 for reason, count in resolved)
        or len({reason for reason, _ in resolved}) != len(resolved)
        or tuple(sorted(resolved)) != resolved
    ):
        raise ValueError(f"{field} contains invalid reason counts")
    return resolved


@dataclass(frozen=True, slots=True)
class CausalAlphaV3ReplayDiagnostics:
    """One immutable descriptive leaf bound to an authoritative replay metric."""

    run_manifest_digest: str
    freeze_digest: str
    candidate_digest: str
    symbol: str
    episode_index: int
    contract_digest: str
    replay_metric_digest: str
    fit_digest: str
    forecast_digest: str
    target_path_digest: str
    decision_count: int
    long_target_count: int
    short_target_count: int
    flat_target_count: int
    positive_forecast_count: int
    negative_forecast_count: int
    near_zero_forecast_count: int
    mean_target: float
    mean_absolute_target: float
    maximum_absolute_target: float
    mean_expected_return: float
    mean_uncertainty: float
    p90_uncertainty: float
    mean_absolute_signal_to_uncertainty: float
    mean_liquidity_weight_cap: float
    mean_objective_improvement: float
    target_reason_counts: tuple[tuple[str, int], ...]
    research_only: bool = True
    promotion_eligible: bool = False
    schema_version: str = _DIAGNOSTICS_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        for name in (
            "run_manifest_digest",
            "freeze_digest",
            "candidate_digest",
            "contract_digest",
            "replay_metric_digest",
            "fit_digest",
            "forecast_digest",
            "target_path_digest",
        ):
            require_sha256(getattr(self, name), field=f"V3 diagnostics {name}")
        if not isinstance(self.symbol, str) or not self.symbol:
            raise ValueError("V3 diagnostics symbol must be non-empty")
        _non_negative_int(self.episode_index, field="V3 diagnostics episode_index")
        if self.decision_count <= 0:
            raise ValueError("V3 diagnostics decision_count must be positive")
        for name in (
            "decision_count",
            "long_target_count",
            "short_target_count",
            "flat_target_count",
            "positive_forecast_count",
            "negative_forecast_count",
            "near_zero_forecast_count",
        ):
            _non_negative_int(getattr(self, name), field=f"V3 diagnostics {name}")
        if (
            self.long_target_count + self.short_target_count + self.flat_target_count
            != self.decision_count
        ):
            raise ValueError("V3 diagnostics target direction counts do not align")
        if (
            self.positive_forecast_count
            + self.negative_forecast_count
            + self.near_zero_forecast_count
            != self.decision_count
        ):
            raise ValueError("V3 diagnostics forecast direction counts do not align")
        for name in (
            "mean_target",
            "mean_absolute_target",
            "maximum_absolute_target",
            "mean_expected_return",
            "mean_uncertainty",
            "p90_uncertainty",
            "mean_absolute_signal_to_uncertainty",
            "mean_liquidity_weight_cap",
            "mean_objective_improvement",
        ):
            _finite(getattr(self, name), field=f"V3 diagnostics {name}")
        for name in (
            "mean_absolute_target",
            "maximum_absolute_target",
            "mean_uncertainty",
            "p90_uncertainty",
            "mean_absolute_signal_to_uncertainty",
            "mean_liquidity_weight_cap",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"V3 diagnostics {name} must be non-negative")
        reasons = _reason_counts(
            self.target_reason_counts, field="V3 diagnostics target reasons"
        )
        if sum(count for _, count in reasons) != self.decision_count:
            raise ValueError("V3 diagnostics target reason counts do not align")
        if not self.research_only or self.promotion_eligible:
            raise ValueError("V3 diagnostics must remain research-only")
        if self.schema_version != _DIAGNOSTICS_SCHEMA:
            raise ValueError("unsupported V3 replay diagnostics schema")
        object.__setattr__(self, "target_reason_counts", reasons)
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V3 replay diagnostics digest mismatch")
        object.__setattr__(self, "digest", expected)

    @property
    def identity(self) -> tuple[str, str, int]:
        return (self.candidate_digest, self.symbol, self.episode_index)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "candidate_digest": self.candidate_digest,
            "contract_digest": self.contract_digest,
            "decision_count": self.decision_count,
            "episode_index": self.episode_index,
            "fit_digest": self.fit_digest,
            "flat_target_count": self.flat_target_count,
            "forecast_digest": self.forecast_digest,
            "freeze_digest": self.freeze_digest,
            "long_target_count": self.long_target_count,
            "maximum_absolute_target": self.maximum_absolute_target,
            "mean_absolute_signal_to_uncertainty": (
                self.mean_absolute_signal_to_uncertainty
            ),
            "mean_absolute_target": self.mean_absolute_target,
            "mean_expected_return": self.mean_expected_return,
            "mean_liquidity_weight_cap": self.mean_liquidity_weight_cap,
            "mean_objective_improvement": self.mean_objective_improvement,
            "mean_target": self.mean_target,
            "mean_uncertainty": self.mean_uncertainty,
            "near_zero_forecast_count": self.near_zero_forecast_count,
            "negative_forecast_count": self.negative_forecast_count,
            "p90_uncertainty": self.p90_uncertainty,
            "positive_forecast_count": self.positive_forecast_count,
            "promotion_eligible": self.promotion_eligible,
            "replay_metric_digest": self.replay_metric_digest,
            "research_only": self.research_only,
            "run_manifest_digest": self.run_manifest_digest,
            "schema_version": self.schema_version,
            "short_target_count": self.short_target_count,
            "symbol": self.symbol,
            "target_path_digest": self.target_path_digest,
            "target_reason_counts": self.target_reason_counts,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload

    @classmethod
    def from_payload(cls, raw: Mapping[str, Any]) -> CausalAlphaV3ReplayDiagnostics:
        fields = frozenset(
            {
                "artifact_digest",
                "candidate_digest",
                "contract_digest",
                "decision_count",
                "episode_index",
                "fit_digest",
                "flat_target_count",
                "forecast_digest",
                "freeze_digest",
                "long_target_count",
                "maximum_absolute_target",
                "mean_absolute_signal_to_uncertainty",
                "mean_absolute_target",
                "mean_expected_return",
                "mean_liquidity_weight_cap",
                "mean_objective_improvement",
                "mean_target",
                "mean_uncertainty",
                "near_zero_forecast_count",
                "negative_forecast_count",
                "p90_uncertainty",
                "positive_forecast_count",
                "promotion_eligible",
                "replay_metric_digest",
                "research_only",
                "run_manifest_digest",
                "schema_version",
                "short_target_count",
                "symbol",
                "target_path_digest",
                "target_reason_counts",
            }
        )
        values = _strict_payload(
            raw,
            fields=fields,
            schema=_DIAGNOSTICS_SCHEMA,
            label="V3 replay diagnostics",
        )
        if not isinstance(values["research_only"], bool) or not isinstance(
            values["promotion_eligible"], bool
        ):
            raise ValueError("V3 replay diagnostics research flags must be boolean")
        return cls(
            run_manifest_digest=str(values["run_manifest_digest"]),
            freeze_digest=str(values["freeze_digest"]),
            candidate_digest=str(values["candidate_digest"]),
            symbol=str(values["symbol"]),
            episode_index=int(values["episode_index"]),
            contract_digest=str(values["contract_digest"]),
            replay_metric_digest=str(values["replay_metric_digest"]),
            fit_digest=str(values["fit_digest"]),
            forecast_digest=str(values["forecast_digest"]),
            target_path_digest=str(values["target_path_digest"]),
            decision_count=int(values["decision_count"]),
            long_target_count=int(values["long_target_count"]),
            short_target_count=int(values["short_target_count"]),
            flat_target_count=int(values["flat_target_count"]),
            positive_forecast_count=int(values["positive_forecast_count"]),
            negative_forecast_count=int(values["negative_forecast_count"]),
            near_zero_forecast_count=int(values["near_zero_forecast_count"]),
            mean_target=float(values["mean_target"]),
            mean_absolute_target=float(values["mean_absolute_target"]),
            maximum_absolute_target=float(values["maximum_absolute_target"]),
            mean_expected_return=float(values["mean_expected_return"]),
            mean_uncertainty=float(values["mean_uncertainty"]),
            p90_uncertainty=float(values["p90_uncertainty"]),
            mean_absolute_signal_to_uncertainty=float(
                values["mean_absolute_signal_to_uncertainty"]
            ),
            mean_liquidity_weight_cap=float(values["mean_liquidity_weight_cap"]),
            mean_objective_improvement=float(values["mean_objective_improvement"]),
            target_reason_counts=_reason_counts(
                values["target_reason_counts"], field="V3 diagnostics target reasons"
            ),
            research_only=values["research_only"],
            promotion_eligible=values["promotion_eligible"],
            schema_version=str(values["schema_version"]),
            digest=str(values["artifact_digest"]),
        )


def summarize_causal_alpha_v3_targets(
    *,
    run_manifest_digest: str,
    freeze_digest: str,
    candidate_digest: str,
    symbol: str,
    episode_index: int,
    contract_digest: str,
    replay_metric_digest: str,
    fit_digest: str,
    forecast_digest: str,
    target_path_digest: str,
    targets: object,
    expected_returns: object,
    uncertainties: object,
    liquidity_weight_caps: object,
    chosen_objectives: object,
    stay_objectives: object,
    reasons: tuple[str, ...],
) -> CausalAlphaV3ReplayDiagnostics:
    """Summarize an already-computed target path without running a counterfactual."""

    arrays = tuple(
        np.asarray(value, dtype=np.float64).reshape(-1)
        for value in (
            targets,
            expected_returns,
            uncertainties,
            liquidity_weight_caps,
            chosen_objectives,
            stay_objectives,
        )
    )
    if not arrays[0].size or any(value.shape != arrays[0].shape for value in arrays):
        raise ValueError("V3 diagnostics target arrays must be non-empty and align")
    if any(not np.isfinite(value).all() for value in arrays):
        raise ValueError("V3 diagnostics target arrays must be finite")
    target_values, forecast_values, uncertainty_values, cap_values, chosen, stay = arrays
    if np.any(uncertainty_values < 0.0) or np.any(cap_values < 0.0):
        raise ValueError("V3 diagnostics uncertainty/liquidity values must be non-negative")
    reason_values = tuple(reasons)
    if len(reason_values) != target_values.size or any(not item for item in reason_values):
        raise ValueError("V3 diagnostics reasons must align with target arrays")

    ratio = np.divide(
        np.abs(forecast_values),
        uncertainty_values,
        out=np.zeros_like(forecast_values),
        where=uncertainty_values > _EPSILON,
    )
    target_reasons = tuple(sorted(Counter(reason_values).items()))
    return CausalAlphaV3ReplayDiagnostics(
        run_manifest_digest=run_manifest_digest,
        freeze_digest=freeze_digest,
        candidate_digest=candidate_digest,
        symbol=symbol,
        episode_index=episode_index,
        contract_digest=contract_digest,
        replay_metric_digest=replay_metric_digest,
        fit_digest=fit_digest,
        forecast_digest=forecast_digest,
        target_path_digest=target_path_digest,
        decision_count=int(target_values.size),
        long_target_count=int(np.count_nonzero(target_values > _EPSILON)),
        short_target_count=int(np.count_nonzero(target_values < -_EPSILON)),
        flat_target_count=int(np.count_nonzero(np.abs(target_values) <= _EPSILON)),
        positive_forecast_count=int(np.count_nonzero(forecast_values > _EPSILON)),
        negative_forecast_count=int(np.count_nonzero(forecast_values < -_EPSILON)),
        near_zero_forecast_count=int(
            np.count_nonzero(np.abs(forecast_values) <= _EPSILON)
        ),
        mean_target=float(np.mean(target_values, dtype=np.float64)),
        mean_absolute_target=float(np.mean(np.abs(target_values), dtype=np.float64)),
        maximum_absolute_target=float(np.max(np.abs(target_values))),
        mean_expected_return=float(np.mean(forecast_values, dtype=np.float64)),
        mean_uncertainty=float(np.mean(uncertainty_values, dtype=np.float64)),
        p90_uncertainty=float(np.quantile(uncertainty_values, 0.90)),
        mean_absolute_signal_to_uncertainty=float(np.mean(ratio, dtype=np.float64)),
        mean_liquidity_weight_cap=float(np.mean(cap_values, dtype=np.float64)),
        mean_objective_improvement=float(np.mean(chosen - stay, dtype=np.float64)),
        target_reason_counts=target_reasons,
    )


__all__ = [
    "CausalAlphaV3ReplayDiagnostics",
    "summarize_causal_alpha_v3_targets",
]
