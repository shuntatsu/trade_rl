"""Read-only serving registry and paper inference inspection."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from trade_rl.serving.bundle import load_serving_bundle
from trade_rl.studio.contracts import (
    PaperInferenceSnapshot,
    ServingCheck,
    ServingMonitorReport,
)
from trade_rl.studio.settings import StudioSettings


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a JSON object")
    return value


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _paper_snapshot(path: Path, *, bundle_digest: str, dataset_id: str) -> PaperInferenceSnapshot | None:
    if not path.is_file():
        return None
    payload = _mapping(json.loads(path.read_text(encoding="utf-8")), field="paper snapshot")
    if payload.get("schema_version") != "studio_paper_inference_v1":
        raise ValueError("paper snapshot schema is unsupported")
    if payload.get("bundle_digest") != bundle_digest:
        raise ValueError("paper snapshot bundle identity mismatch")
    if payload.get("dataset_id") != dataset_id:
        raise ValueError("paper snapshot dataset identity mismatch")
    decision_index = payload.get("decision_index")
    latency_ms = payload.get("latency_ms")
    weights = payload.get("target_weights")
    if isinstance(decision_index, bool) or not isinstance(decision_index, int) or decision_index < 0:
        raise ValueError("paper snapshot decision_index must be non-negative")
    if isinstance(latency_ms, bool) or not isinstance(latency_ms, int | float) or not math.isfinite(float(latency_ms)) or float(latency_ms) < 0:
        raise ValueError("paper snapshot latency_ms must be finite and non-negative")
    if not isinstance(weights, Mapping) or not weights:
        raise ValueError("paper snapshot target_weights must be a non-empty object")
    resolved_weights: dict[str, float] = {}
    for key, raw in weights.items():
        if not isinstance(key, str) or not key:
            raise ValueError("paper snapshot weight names must be non-empty strings")
        if isinstance(raw, bool) or not isinstance(raw, int | float) or not math.isfinite(float(raw)):
            raise ValueError("paper snapshot weights must be finite numbers")
        resolved_weights[key] = float(raw)
    return PaperInferenceSnapshot(
        recorded_at=_string(payload.get("recorded_at"), field="paper snapshot recorded_at"),
        bundle_digest=bundle_digest,
        dataset_id=dataset_id,
        decision_index=decision_index,
        target_weights=resolved_weights,
        latency_ms=float(latency_ms),
        snapshot_digest=_string(payload.get("snapshot_digest"), field="paper snapshot digest"),
    )


def inspect_serving(settings: StudioSettings) -> ServingMonitorReport:
    """Inspect, but never activate or execute, the configured serving bundle."""

    root = settings.serving_root
    snapshot_path = settings.paper_snapshot_path
    assert root is not None
    assert snapshot_path is not None
    pointer_path = root / "active.json"
    if not pointer_path.is_file():
        return ServingMonitorReport(
            state="IDLE",
            checks=(
                ServingCheck(
                    key="registry",
                    label="Serving registry",
                    status="WARN",
                    detail="no active bundle pointer is present",
                ),
            ),
        )

    checks: list[ServingCheck] = []
    try:
        pointer = _mapping(json.loads(pointer_path.read_text(encoding="utf-8")), field="active pointer")
        if pointer.get("schema") != "serving_registry_pointer_v1":
            raise ValueError("active registry pointer schema is unsupported")
        expected_digest = _string(pointer.get("bundle_digest"), field="bundle_digest")
        relative = Path(_string(pointer.get("path"), field="path"))
        if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
            raise ValueError("active registry pointer escapes serving root")
        bundle_path = (root / relative).resolve()
        resolved_root = root.resolve()
        if bundle_path != resolved_root and resolved_root not in bundle_path.parents:
            raise ValueError("active registry pointer escapes serving root")
        checks.append(ServingCheck(key="pointer", label="Active pointer", status="PASS", detail="pointer schema and path are valid"))
        bundle = load_serving_bundle(bundle_path)
        manifest = bundle.manifest
        if manifest.bundle_digest != expected_digest:
            raise ValueError("active registry pointer digest mismatch")
        checks.append(ServingCheck(key="closure", label="Bundle closure", status="PASS", detail=f"{len(manifest.files)} declared files verified"))
        checks.append(ServingCheck(key="identity", label="Bundle identity", status="PASS", detail="dataset, action, observation, and policy identities are bound"))
        checks.append(
            ServingCheck(
                key="release",
                label="Release attestation",
                status="PASS" if bundle.release is not None else "WARN",
                detail="detached attestation is present" if bundle.release is not None else "no detached attestation is present",
            )
        )
        paper = None
        try:
            paper = _paper_snapshot(snapshot_path, bundle_digest=manifest.bundle_digest, dataset_id=manifest.dataset_id)
            checks.append(
                ServingCheck(
                    key="paper_snapshot",
                    label="Paper inference snapshot",
                    status="PASS" if paper is not None else "WARN",
                    detail="latest snapshot identity is valid" if paper is not None else "no paper inference snapshot is present",
                )
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            checks.append(ServingCheck(key="paper_snapshot", label="Paper inference snapshot", status="FAIL", detail=str(error)))
        return ServingMonitorReport(
            state="VALID",
            active_bundle_digest=manifest.bundle_digest,
            dataset_id=manifest.dataset_id,
            run_kind=manifest.run_kind,
            policy_digest=manifest.policy_digest,
            action_schema=manifest.action_schema,
            observation_schema=manifest.observation_schema,
            release_attestation_present=bundle.release is not None,
            checks=tuple(checks),
            paper_snapshot=paper,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError) as error:
        checks.append(ServingCheck(key="validation", label="Serving validation", status="FAIL", detail=str(error)))
        return ServingMonitorReport(
            state="INVALID",
            checks=tuple(checks),
            validation_error=str(error),
        )


__all__ = ["inspect_serving"]
