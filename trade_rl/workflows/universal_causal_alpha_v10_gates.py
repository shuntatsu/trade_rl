"""V10-owned evidence over unchanged universal numerical gates."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Final

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.learning.causal_alpha_v8 import CausalAlphaV8Candidate
from trade_rl.learning.causal_alpha_v10 import CausalAlphaV10Candidate
from trade_rl.workflows.universal_causal_alpha_v8_gates import (
    CausalAlphaV8SelectionEvidence,
    evaluate_causal_alpha_v8_selection,
)
from trade_rl.workflows.universal_causal_alpha_v8_replay import (
    CausalAlphaV8ReplayMetric,
)
from trade_rl.workflows.universal_causal_alpha_v9_gates import (
    CausalAlphaV9SignalEvidence,
)

_SIGNAL_SCHEMA: Final = "causal_alpha_v10_signal_evidence_v2"
_SELECTION_SCHEMA: Final = "causal_alpha_v10_selection_evidence_v2"
_DUAL_RUN_BINDING_SCHEMA: Final = "causal_alpha_v10_dual_run_binding_v1"
_RUN_CONFIG_SCIENCE_SCHEMA: Final = "causal_alpha_v10_run_config_science_v1"
V8_CANDIDATE_BY_V10: Final = {
    CausalAlphaV10Candidate.V8_ROBUST_CONTROL: CausalAlphaV8Candidate.V7_CONTROL,
    CausalAlphaV10Candidate.V9_NONLINEAR_CONTROL: (
        CausalAlphaV8Candidate.ROBUST_CONTRARIAN
    ),
    CausalAlphaV10Candidate.HIERARCHICAL_WAVE: (
        CausalAlphaV8Candidate.ROBUST_CALIBRATED
    ),
}
V10_CANDIDATE_BY_V8: Final = {value: key for key, value in V8_CANDIDATE_BY_V10.items()}


def _run_config_science_identity(config: object) -> tuple[str, tuple[str, ...]]:
    """Return the run recipe identity after removing only initial-state modes."""

    payload_builder = getattr(config, "digest_payload", None)
    if not callable(payload_builder):
        payload_builder = getattr(config, "candidate_digest_payload", None)
    environment = getattr(config, "environment", None)
    modes = getattr(environment, "initial_state_modes", None)
    if not callable(payload_builder) or not isinstance(modes, tuple) or not modes:
        raise TypeError("V10 dual-run config does not expose a valid environment")
    payload = payload_builder()
    if not isinstance(payload, dict):
        raise TypeError("V10 dual-run config recipe payload is invalid")
    environment_payload = payload.get("environment")
    if not isinstance(environment_payload, dict):
        raise TypeError("V10 dual-run config environment payload is invalid")
    environment_payload = dict(environment_payload)
    environment_payload.pop("initial_state_modes", None)
    payload["environment"] = environment_payload
    return (
        content_digest(
            {
                "payload": payload,
                "schema_version": _RUN_CONFIG_SCIENCE_SCHEMA,
            }
        ),
        tuple(str(mode) for mode in modes),
    )


@dataclass(frozen=True, slots=True)
class CausalAlphaV10DualRunBinding:
    """Immutable provenance binding for a split Signal/Selection run."""

    signal_run_manifest_digest: str
    selection_run_manifest_digest: str
    shared_science_identity_digest: str
    signal_initial_state_modes: tuple[str, ...]
    selection_initial_state_modes: tuple[str, ...]
    allowed_difference: str = "environment.initial_state_modes"
    schema_version: str = _DUAL_RUN_BINDING_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        for name in (
            "signal_run_manifest_digest",
            "selection_run_manifest_digest",
            "shared_science_identity_digest",
        ):
            require_sha256(getattr(self, name), field=f"V10 dual-run {name}")
        signal_modes = tuple(str(mode) for mode in self.signal_initial_state_modes)
        selection_modes = tuple(str(mode) for mode in self.selection_initial_state_modes)
        if (
            not signal_modes
            or not selection_modes
            or any(not mode for mode in (*signal_modes, *selection_modes))
        ):
            raise ValueError("V10 dual-run initial-state modes are invalid")
        if self.schema_version != _DUAL_RUN_BINDING_SCHEMA:
            raise ValueError("unsupported V10 dual-run binding schema")
        if self.allowed_difference != "environment.initial_state_modes":
            raise ValueError("unsupported V10 dual-run allowed difference")
        object.__setattr__(self, "signal_initial_state_modes", signal_modes)
        object.__setattr__(self, "selection_initial_state_modes", selection_modes)
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V10 dual-run binding digest mismatch")
        object.__setattr__(self, "digest", expected)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "selection_initial_state_modes": self.selection_initial_state_modes,
            "selection_run_manifest_digest": self.selection_run_manifest_digest,
            "shared_science_identity_digest": self.shared_science_identity_digest,
            "signal_initial_state_modes": self.signal_initial_state_modes,
            "signal_run_manifest_digest": self.signal_run_manifest_digest,
            "allowed_difference": self.allowed_difference,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


def build_causal_alpha_v10_dual_run_binding(
    *,
    signal_config: object,
    selection_config: object,
    signal_prepared: object,
    selection_prepared: object,
    allow_initial_state_split: bool = False,
) -> CausalAlphaV10DualRunBinding:
    """Fail closed unless the two runs differ only in initial-state modes."""

    signal_science, signal_modes = _run_config_science_identity(signal_config)
    selection_science, selection_modes = _run_config_science_identity(selection_config)
    if signal_science != selection_science:
        raise ValueError("V10 dual-run config differs outside initial_state_modes")
    if signal_modes != selection_modes:
        if not allow_initial_state_split:
            raise ValueError("V10 dual-run split requires flat-start activation")
        if "cash" not in signal_modes or selection_modes != ("cash",):
            raise ValueError(
                "V10 flat-start split requires Signal cash and Selection=(cash,)"
            )
    shared_fields = (
        "train_symbols",
        "nested_partition_digest",
        "base_runtime_manifest_digest",
        "v4_context_manifest_digest",
        "config_digest",
        "execution_identity_digest",
        "generator_code_digest",
    )
    shared_values: dict[str, object] = {"config_science_digest": signal_science}
    for name in shared_fields:
        signal_value = getattr(signal_prepared, name, None)
        selection_value = getattr(selection_prepared, name, None)
        if signal_value != selection_value:
            raise ValueError(f"V10 dual-run identity drifted in {name}")
        shared_values[name] = signal_value
    return CausalAlphaV10DualRunBinding(
        signal_run_manifest_digest=str(getattr(signal_prepared, "run_manifest_digest")),
        selection_run_manifest_digest=str(
            getattr(selection_prepared, "run_manifest_digest")
        ),
        shared_science_identity_digest=content_digest(
            {
                "payload": shared_values,
                "schema_version": "causal_alpha_v10_shared_science_identity_v1",
            }
        ),
        signal_initial_state_modes=signal_modes,
        selection_initial_state_modes=selection_modes,
    )


@dataclass(frozen=True, slots=True)
class CausalAlphaV10SignalEvidence:
    source_v9: CausalAlphaV9SignalEvidence
    slow_scope_count: int
    qualified_slow_scope_count: int
    dual_fit_digests: tuple[str, ...]
    signal_run_manifest_digest: str | None = None
    dual_run_binding_digest: str | None = None
    schema_version: str = _SIGNAL_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.source_v9, CausalAlphaV9SignalEvidence):
            raise TypeError("V10 Signal source is invalid")
        if self.slow_scope_count != 72 or self.qualified_slow_scope_count != 72:
            raise ValueError("V10 Signal requires 72 qualified slow scopes")
        if len(self.dual_fit_digests) != 8 or len(set(self.dual_fit_digests)) != 8:
            raise ValueError("V10 Signal dual fit identities are invalid")
        if not self.source_v9.passed:
            raise ValueError("V10 Signal cannot bypass V9 Signal")
        for name in ("signal_run_manifest_digest", "dual_run_binding_digest"):
            value = getattr(self, name)
            if value is not None:
                require_sha256(value, field=f"V10 Signal {name}")
        if self.schema_version != _SIGNAL_SCHEMA:
            raise ValueError("unsupported V10 Signal schema")
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V10 Signal digest mismatch")
        object.__setattr__(self, "digest", expected)

    @property
    def passed(self) -> bool:
        return True

    @property
    def rejection_reasons(self) -> tuple[str, ...]:
        return ()

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "dual_fit_digests": self.dual_fit_digests,
            "passed": self.passed,
            "promotion_eligible": False,
            "qualified_slow_scope_count": self.qualified_slow_scope_count,
            "rejection_reasons": self.rejection_reasons,
            "schema_version": self.schema_version,
            "slow_scope_count": self.slow_scope_count,
            "source_v9_signal_digest": self.source_v9.digest,
        }
        if self.signal_run_manifest_digest is not None:
            payload["signal_run_manifest_digest"] = self.signal_run_manifest_digest
        if self.dual_run_binding_digest is not None:
            payload["dual_run_binding_digest"] = self.dual_run_binding_digest
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


def _candidate_payloads(
    source: CausalAlphaV8SelectionEvidence,
) -> tuple[dict[str, object], ...]:
    mapped: list[dict[str, object]] = []
    for candidate in source.source_v7.candidates:
        source_v8 = CausalAlphaV8Candidate(
            {
                "v6_control": "v7_control",
                "symmetric_contrarian": "robust_contrarian",
                "causal_calibrated": "robust_calibrated",
            }[candidate.candidate.value]
        )
        payload = candidate.to_payload()
        payload["source_gate_candidate"] = payload["candidate"]
        payload["candidate"] = V10_CANDIDATE_BY_V8[source_v8].value
        mapped.append(payload)
    return tuple(mapped)


@dataclass(frozen=True, slots=True)
class CausalAlphaV10SelectionEvidence:
    source_v8: CausalAlphaV8SelectionEvidence
    source_signal_evidence_digest: str | None = None
    dual_run_binding_digest: str | None = None
    schema_version: str = _SELECTION_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.source_v8, CausalAlphaV8SelectionEvidence):
            raise TypeError("V10 Selection source is invalid")
        for name in ("source_signal_evidence_digest", "dual_run_binding_digest"):
            value = getattr(self, name)
            if value is not None:
                require_sha256(value, field=f"V10 Selection {name}")
        if self.schema_version != _SELECTION_SCHEMA:
            raise ValueError("unsupported V10 Selection schema")
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V10 Selection digest mismatch")
        object.__setattr__(self, "digest", expected)

    @property
    def passed(self) -> bool:
        return self.source_v8.passed

    @property
    def rejection_reasons(self) -> tuple[str, ...]:
        return self.source_v8.rejection_reasons

    @property
    def selected_candidate(self) -> CausalAlphaV10Candidate | None:
        selected = self.source_v8.selected_candidate
        return None if selected is None else V10_CANDIDATE_BY_V8[selected]

    @property
    def selected_config_digest(self) -> str | None:
        return self.source_v8.selected_config_digest

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "candidates": _candidate_payloads(self.source_v8),
            "paired_scope_count": self.source_v8.source_v7.paired_scope_count,
            "passed": self.passed,
            "promotion_eligible": False,
            "rejection_reasons": self.rejection_reasons,
            "schema_version": self.schema_version,
            "selected_candidate": (
                None if self.selected_candidate is None else self.selected_candidate.value
            ),
            "selected_config_digest": self.selected_config_digest,
            "source_v8_selection_digest": self.source_v8.digest,
        }
        if self.source_signal_evidence_digest is not None:
            payload["source_signal_evidence_digest"] = self.source_signal_evidence_digest
        if self.dual_run_binding_digest is not None:
            payload["dual_run_binding_digest"] = self.dual_run_binding_digest
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


def evaluate_causal_alpha_v10_selection(
    metrics: tuple[CausalAlphaV8ReplayMetric, ...],
    *,
    expected_symbols: tuple[str, ...],
) -> CausalAlphaV10SelectionEvidence:
    # V10 candidates intentionally use different fast/slow model fits. Pairing
    # must therefore use the common V6 calibration fit carried by the replay
    # economics, while retaining each candidate's model identity in its target
    # artifact and replay digest.
    paired_metrics = tuple(
        metric
        if metric.calibration_fit_digest == metric.v6_metric.fit_digest
        else replace(
            metric,
            calibration_fit_digest=metric.v6_metric.fit_digest,
            digest="",
        )
        for metric in metrics
    )
    return CausalAlphaV10SelectionEvidence(
        evaluate_causal_alpha_v8_selection(
            paired_metrics,
            expected_symbols=expected_symbols,
        )
    )


__all__ = [
    "CausalAlphaV10DualRunBinding",
    "CausalAlphaV10SelectionEvidence",
    "CausalAlphaV10SignalEvidence",
    "V8_CANDIDATE_BY_V10",
    "V10_CANDIDATE_BY_V8",
    "build_causal_alpha_v10_dual_run_binding",
    "evaluate_causal_alpha_v10_selection",
]
