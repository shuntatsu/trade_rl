"""Strict binding of Causal Alpha V3 Signal metrics to diagnostic sidecars."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from trade_rl.workflows.universal_causal_alpha_v3_config import (
    CausalAlphaV3ResearchConfig,
)
from trade_rl.workflows.universal_causal_alpha_v3_identity import (
    CausalAlphaV3RunManifestV2,
)
from trade_rl.workflows.universal_causal_alpha_v3_signal import (
    CausalAlphaV3SignalScopeMetric,
)
from trade_rl.workflows.universal_causal_alpha_v3_signal_diagnostic import (
    CausalAlphaV3SignalDiagnosticScope,
)
from trade_rl.workflows.universal_causal_alpha_v3_signal_diagnostic_codec import (
    signal_diagnostic_scope_from_payload,
)
from trade_rl.workflows.universal_causal_alpha_v3_signal_forensics import (
    _fit_order,
    _load_json,
    _load_metrics,
)


@dataclass(frozen=True, slots=True)
class CausalAlphaV3SignalForensicsV2BoundScope:
    metric: CausalAlphaV3SignalScopeMetric
    diagnostic: CausalAlphaV3SignalDiagnosticScope


def _validate_pair(
    *,
    metric: CausalAlphaV3SignalScopeMetric,
    diagnostic: CausalAlphaV3SignalDiagnosticScope,
    manifest: CausalAlphaV3RunManifestV2,
) -> None:
    if diagnostic.run_manifest_digest != manifest.digest:
        raise ValueError("V3 signal diagnostic run manifest identity drifted")
    if diagnostic.run_manifest_digest != metric.run_manifest_digest:
        raise ValueError("V3 signal diagnostic run manifest does not match metric")
    if diagnostic.fit_config_digest != metric.fit_config_digest:
        raise ValueError("V3 signal diagnostic fit config does not match metric")
    if diagnostic.symbol != metric.symbol:
        raise ValueError("V3 signal diagnostic symbol does not match metric")
    if diagnostic.episode_index != metric.episode_index:
        raise ValueError("V3 signal diagnostic episode does not match metric")
    if (
        diagnostic.contract_start != metric.contract_start
        or diagnostic.contract_stop != metric.contract_stop
    ):
        raise ValueError("V3 signal diagnostic contract interval does not match metric")
    if diagnostic.contract_digest != metric.contract_digest:
        raise ValueError("V3 signal diagnostic contract digest does not match metric")
    if diagnostic.signal_metric_digest != metric.digest:
        raise ValueError(
            "V3 signal diagnostic metric digest does not match canonical metric"
        )
    if diagnostic.fit_digest != metric.fit_digest:
        raise ValueError("V3 signal diagnostic fit digest does not match metric")
    if diagnostic.forecast_digest != metric.forecast_digest:
        raise ValueError("V3 signal diagnostic forecast digest does not match metric")
    if diagnostic.canonical_cohort_indices != metric.cohort_indices:
        raise ValueError("V3 signal diagnostic canonical cohort does not match metric")


def load_causal_alpha_v3_signal_forensics_v2_sidecars(
    root: Path,
) -> tuple[CausalAlphaV3SignalForensicsV2BoundScope, ...]:
    """Load a complete metric/diagnostic sidecar graph or fail closed."""

    source_root = Path(root)
    manifest = CausalAlphaV3RunManifestV2.from_payload(
        _load_json(source_root / "run-manifest.json", field="V3 run manifest")
    )
    config = CausalAlphaV3ResearchConfig.from_mapping(
        _load_json(source_root / "authored-config.json", field="V3 authored config")
    )
    if config.digest != manifest.config_digest:
        raise ValueError("V3 signal forensics authored config identity drifted")

    metrics = _load_metrics(source_root, manifest=manifest, config=config)
    metric_by_identity = {metric.identity: metric for metric in metrics}
    if len(metric_by_identity) != len(metrics):
        raise ValueError("V3 signal canonical metric scope identity is duplicated")

    diagnostics_root = source_root / "signal" / "diagnostics"
    if diagnostics_root.is_symlink() or not diagnostics_root.is_dir():
        raise ValueError("V3 signal diagnostic root is not a trusted directory")
    paths = tuple(sorted(diagnostics_root.rglob("*.json")))
    if not paths:
        raise ValueError("V3 signal diagnostic root contains no sidecar evidence")

    diagnostic_by_identity: dict[
        tuple[str, str, int], CausalAlphaV3SignalDiagnosticScope
    ] = {}
    for path in paths:
        diagnostic = signal_diagnostic_scope_from_payload(
            _load_json(path, field="V3 signal diagnostic sidecar")
        )
        expected_path = (
            diagnostics_root
            / diagnostic.fit_config_digest
            / diagnostic.symbol
            / f"{diagnostic.episode_index}.json"
        )
        if path != expected_path:
            raise ValueError("V3 signal diagnostic path identity drifted")
        identity = (
            diagnostic.fit_config_digest,
            diagnostic.symbol,
            diagnostic.episode_index,
        )
        if identity in diagnostic_by_identity:
            raise ValueError("V3 signal diagnostic scope identity is duplicated")
        diagnostic_by_identity[identity] = diagnostic

    metric_identities = set(metric_by_identity)
    diagnostic_identities = set(diagnostic_by_identity)
    if diagnostic_identities != metric_identities:
        missing = sorted(metric_identities - diagnostic_identities)
        extra = sorted(diagnostic_identities - metric_identities)
        raise ValueError(
            "V3 signal diagnostic scope does not exactly match canonical metrics; "
            f"missing={missing}, extra={extra}"
        )

    fit_order = {digest: index for index, digest in enumerate(_fit_order(config))}
    symbol_order = {
        symbol: index for index, symbol in enumerate(manifest.train_symbols)
    }
    ordered_metrics = tuple(
        sorted(
            metrics,
            key=lambda metric: (
                fit_order[metric.fit_config_digest],
                metric.contract_start,
                metric.contract_stop,
                symbol_order[metric.symbol],
                metric.episode_index,
            ),
        )
    )
    bound: list[CausalAlphaV3SignalForensicsV2BoundScope] = []
    for metric in ordered_metrics:
        diagnostic = diagnostic_by_identity[metric.identity]
        _validate_pair(metric=metric, diagnostic=diagnostic, manifest=manifest)
        bound.append(
            CausalAlphaV3SignalForensicsV2BoundScope(
                metric=metric,
                diagnostic=diagnostic,
            )
        )
    return tuple(bound)


__all__ = [
    "CausalAlphaV3SignalForensicsV2BoundScope",
    "load_causal_alpha_v3_signal_forensics_v2_sidecars",
]
