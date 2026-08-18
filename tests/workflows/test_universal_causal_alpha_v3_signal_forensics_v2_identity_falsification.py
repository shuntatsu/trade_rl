from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.workflows.test_universal_causal_alpha_v3_signal_forensics import (
    _digest,
    _write_json,
)
from tests.workflows.test_universal_causal_alpha_v3_signal_forensics_v2_sidecars import (
    _complete_sidecars,
)
from trade_rl.artifacts.hashing import content_digest
from trade_rl.workflows.universal_causal_alpha_v3_signal_forensics_v2_loader import (
    load_causal_alpha_v3_signal_forensics_v2_sidecars,
)


def _rewrite_sidecar(path: Path, **updates: object) -> dict[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw.update(updates)
    body = {key: value for key, value in raw.items() if key != "artifact_digest"}
    payload = {**body, "artifact_digest": content_digest(body)}
    _write_json(path, payload)
    return payload


def _first_sidecar(built: dict[str, object]) -> Path:
    paths = built["diagnostic_paths"]
    assert isinstance(paths, dict)
    path = next(iter(paths.values()))
    assert isinstance(path, Path)
    return path


def test_v2_sidecar_loader_rejects_extra_identity(tmp_path: Path) -> None:
    built = _complete_sidecars(tmp_path)
    source = _first_sidecar(built)
    raw = json.loads(source.read_text(encoding="utf-8"))
    raw["episode_index"] = 999
    body = {key: value for key, value in raw.items() if key != "artifact_digest"}
    extra = source.with_name("999.json")
    _write_json(extra, {**body, "artifact_digest": content_digest(body)})

    with pytest.raises(ValueError, match="exactly match canonical metrics"):
        load_causal_alpha_v3_signal_forensics_v2_sidecars(tmp_path)


def test_v2_sidecar_loader_rejects_run_manifest_identity_drift(
    tmp_path: Path,
) -> None:
    built = _complete_sidecars(tmp_path)
    _rewrite_sidecar(
        _first_sidecar(built),
        run_manifest_digest=_digest("foreign-run-manifest"),
    )

    with pytest.raises(ValueError, match="run manifest"):
        load_causal_alpha_v3_signal_forensics_v2_sidecars(tmp_path)


def test_v2_sidecar_loader_rejects_fit_identity_drift(tmp_path: Path) -> None:
    built = _complete_sidecars(tmp_path)
    _rewrite_sidecar(
        _first_sidecar(built),
        fit_digest=_digest("foreign-fit"),
    )

    with pytest.raises(ValueError, match="fit digest"):
        load_causal_alpha_v3_signal_forensics_v2_sidecars(tmp_path)


def test_v2_sidecar_loader_rejects_forecast_identity_drift(tmp_path: Path) -> None:
    built = _complete_sidecars(tmp_path)
    _rewrite_sidecar(
        _first_sidecar(built),
        forecast_digest=_digest("foreign-forecast"),
    )

    with pytest.raises(ValueError, match="forecast"):
        load_causal_alpha_v3_signal_forensics_v2_sidecars(tmp_path)


def test_v2_sidecar_loader_rejects_canonical_cohort_identity_drift(
    tmp_path: Path,
) -> None:
    built = _complete_sidecars(tmp_path)
    target = _first_sidecar(built)
    raw = json.loads(target.read_text(encoding="utf-8"))
    cohort = raw["canonical_cohort_indices"]
    assert isinstance(cohort, list)
    assert len(cohort) >= 2
    _rewrite_sidecar(target, canonical_cohort_indices=cohort[:-1])

    with pytest.raises(ValueError, match="canonical cohort"):
        load_causal_alpha_v3_signal_forensics_v2_sidecars(tmp_path)
