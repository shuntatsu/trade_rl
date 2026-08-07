"""Deterministic rollout evidence for corrected Lagrangian PPO semantics."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from trade_rl.artifacts.hashing import content_digest
from trade_rl.rl.lagrangian import (
    DualUpdateReport,
    LagrangianConstraintSpec,
    LagrangianSchema,
    canonical_constraint_unit,
)
from trade_rl.rl.lagrangian_diagnostics import (
    ConstraintCorrelationDiagnostics,
    DualStabilityDiagnostics,
)
from trade_rl.rl.lagrangian_probe import CanonicalActionProbeEvidence

_SCHEMA_VERSION = "lagrangian_rollout_evidence_v1"


def _non_negative_integer(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _finite_optional(value: float | None, *, field_name: str) -> float | None:
    if value is None:
        return None
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{field_name} must be finite when present")
    return resolved


def _matrix_payload(value: object) -> list[list[float]]:
    tolist = getattr(value, "tolist", None)
    if not callable(tolist):
        raise TypeError("diagnostic matrix must expose tolist()")
    raw = tolist()
    if not isinstance(raw, list) or any(not isinstance(row, list) for row in raw):
        raise TypeError("diagnostic matrix must be two-dimensional")
    return [[float(item) for item in row] for row in raw]


@dataclass(frozen=True, slots=True)
class LagrangianRolloutEvidence:
    """One finalized rollout's raw actor and dual-control evidence."""

    actor_composition_mode: str
    schema: LagrangianSchema
    correlation_diagnostics: ConstraintCorrelationDiagnostics
    stability_diagnostics: DualStabilityDiagnostics
    dual_reports: tuple[DualUpdateReport, ...]
    probe_evidence: CanonicalActionProbeEvidence
    completed_episode_count: int
    censored_episode_count: int
    digest: str

    @property
    def cost_names(self) -> tuple[str, ...]:
        return self.schema.names

    def _payload_without_digest(self) -> dict[str, object]:
        correlation = self.correlation_diagnostics
        constraints: dict[str, object] = {}
        for index, (spec, report) in enumerate(
            zip(self.schema.specs, self.dual_reports, strict=True)
        ):
            if not isinstance(spec, LagrangianConstraintSpec):
                raise TypeError("rollout evidence schema is missing support metadata")
            consumed = report.consumed_denominator
            beta_effective = (
                spec.ema_beta**consumed if report.updated and consumed > 0 else None
            )
            constraints[spec.name] = {
                "aggregation": spec.aggregation.value,
                "unit": canonical_constraint_unit(spec.name),
                "budget": spec.budget,
                "minimum_completed_episodes": spec.minimum_completed_episodes,
                "ema_beta": spec.ema_beta,
                "raw_cost_advantage_statistics": (
                    correlation.raw_cost_advantage_statistics[index].payload()
                ),
                "raw_effective_penalty_statistics": (
                    correlation.raw_effective_penalty_statistics[index].payload()
                ),
                "raw_estimate": report.raw_estimate,
                "ema_estimate": report.ema_estimate,
                "pending_numerator_before": report.pending_numerator_before,
                "pending_denominator_before": report.pending_denominator_before,
                "consumed_denominator": consumed,
                "censored_episode_count": report.censored_episode_count,
                "beta_effective": beta_effective,
                "constraint_residual": report.constraint_residual,
                "multiplier_before": report.multiplier_before,
                "multiplier_after": report.multiplier_after,
                "updated": report.updated,
                "skip_reason": report.skip_reason,
                "at_lower_bound": report.at_lower_bound,
                "at_upper_cap": report.at_upper_cap,
            }

        normalized = correlation.normalized_cost_advantage_correlation
        return {
            "schema_version": _SCHEMA_VERSION,
            "actor_composition_mode": self.actor_composition_mode,
            "cost_names": list(self.cost_names),
            "lagrangian_schema": self.schema.digest_payload(),
            "lagrangian_schema_digest": self.schema.digest,
            "raw_reward_advantage_statistics": (
                correlation.raw_reward_advantage_statistics.payload()
            ),
            "constraints": constraints,
            "penalty_to_reward_l2_ratio": (correlation.penalty_to_reward_l2_ratio),
            "raw_cost_covariance": _matrix_payload(correlation.raw_cost_covariance),
            "raw_cost_correlation": _matrix_payload(correlation.raw_cost_correlation),
            "normalized_cost_advantage_correlation": (
                None if normalized is None else _matrix_payload(normalized)
            ),
            "dual_stability": self.stability_diagnostics.payload(),
            "probe": {
                "semantic": self.probe_evidence.action_semantic.value,
                "warning": self.probe_evidence.warning,
                "violated_costs": list(self.probe_evidence.violated_costs),
                "payload": self.probe_evidence.digest_payload(),
                "digest": self.probe_evidence.digest,
            },
            "completed_episode_count": self.completed_episode_count,
            "censored_episode_count": self.censored_episode_count,
        }

    def payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload = self._payload_without_digest()
        if include_digest:
            payload["digest"] = self.digest
        return payload


def build_lagrangian_rollout_evidence(
    *,
    actor_composition_mode: str,
    schema: LagrangianSchema,
    correlation_diagnostics: ConstraintCorrelationDiagnostics,
    stability_diagnostics: DualStabilityDiagnostics,
    dual_reports: Mapping[str, DualUpdateReport],
    probe_evidence: CanonicalActionProbeEvidence,
    completed_episode_count: int,
    censored_episode_count: int,
) -> LagrangianRolloutEvidence:
    """Validate and bind all corrected rollout semantics into one evidence digest."""

    if not isinstance(actor_composition_mode, str) or not actor_composition_mode:
        raise ValueError("actor_composition_mode must be a non-empty string")
    if not isinstance(schema, LagrangianSchema):
        raise TypeError("schema must be a LagrangianSchema")
    if not isinstance(
        correlation_diagnostics,
        ConstraintCorrelationDiagnostics,
    ):
        raise TypeError(
            "correlation_diagnostics must be ConstraintCorrelationDiagnostics"
        )
    if correlation_diagnostics.cost_names != schema.names:
        raise ValueError("correlation diagnostic constraint order mismatch")
    if not isinstance(stability_diagnostics, DualStabilityDiagnostics):
        raise TypeError("stability_diagnostics must be DualStabilityDiagnostics")
    if stability_diagnostics.cost_names != schema.names:
        raise ValueError("stability diagnostic constraint order mismatch")
    if not isinstance(dual_reports, Mapping) or tuple(dual_reports) != schema.names:
        raise ValueError("dual report constraint order mismatch")
    ordered_reports: list[DualUpdateReport] = []
    for name in schema.names:
        report = dual_reports[name]
        if not isinstance(report, DualUpdateReport) or report.name != name:
            raise ValueError("dual report constraint order mismatch")
        _finite_optional(
            report.constraint_residual,
            field_name=f"{name}.constraint_residual",
        )
        ordered_reports.append(report)
    if not isinstance(probe_evidence, CanonicalActionProbeEvidence):
        raise TypeError("probe_evidence must be CanonicalActionProbeEvidence")
    completed = _non_negative_integer(
        completed_episode_count,
        field_name="completed_episode_count",
    )
    censored = _non_negative_integer(
        censored_episode_count,
        field_name="censored_episode_count",
    )

    provisional = LagrangianRolloutEvidence(
        actor_composition_mode=actor_composition_mode,
        schema=schema,
        correlation_diagnostics=correlation_diagnostics,
        stability_diagnostics=stability_diagnostics,
        dual_reports=tuple(ordered_reports),
        probe_evidence=probe_evidence,
        completed_episode_count=completed,
        censored_episode_count=censored,
        digest="",
    )
    digest = content_digest(provisional._payload_without_digest())
    return LagrangianRolloutEvidence(
        actor_composition_mode=provisional.actor_composition_mode,
        schema=provisional.schema,
        correlation_diagnostics=provisional.correlation_diagnostics,
        stability_diagnostics=provisional.stability_diagnostics,
        dual_reports=provisional.dual_reports,
        probe_evidence=provisional.probe_evidence,
        completed_episode_count=provisional.completed_episode_count,
        censored_episode_count=provisional.censored_episode_count,
        digest=digest,
    )


__all__ = [
    "LagrangianRolloutEvidence",
    "build_lagrangian_rollout_evidence",
]
