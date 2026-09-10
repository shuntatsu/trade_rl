from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tests.evaluation.experiments.bootstrap.test_binance import _config
from tests.evaluation.experiments.bootstrap.test_workflow import (
    _config_path,
    _install_fakes,
)
from trade_rl.evaluation.experiments.bootstrap.config import (
    load_canonical_m2_bootstrap_config,
)
from trade_rl.evaluation.experiments.bootstrap.workflow import (
    bootstrap_canonical_m2_study,
)


def _write_payload(tmp_path: Path, payload: dict[str, object], name: str) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_direct_config_construction_cannot_bypass_source_contract(
    tmp_path: Path,
) -> None:
    valid = load_canonical_m2_bootstrap_config(_config_path(tmp_path))

    with pytest.raises(ValueError, match="timeframe|interval"):
        replace(valid, base_timeframe="13m")

    with pytest.raises(ValueError, match="timezone-aware"):
        replace(valid, data_start=datetime(2024, 1, 1))

    jst = timezone(timedelta(hours=9))
    with pytest.raises(ValueError, match="UTC month boundary"):
        replace(valid, data_stop_exclusive=datetime(2024, 3, 1, tzinfo=jst))

    with pytest.raises(ValueError, match="ppo_seeds"):
        replace(valid, ppo_seeds=(False, 1))


def test_output_parent_symlink_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fakes(monkeypatch)
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(ValueError, match="output parent|symlink"):
        bootstrap_canonical_m2_study(
            _config_path(tmp_path),
            linked_parent / "canonical-m2",
        )

    assert not (real_parent / "canonical-m2").exists()


def test_published_dataset_must_cover_preregistered_source_range(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fakes(monkeypatch)
    payload = _config().to_payload()
    payload["data_start"] = "2023-12-01T00:00:00+00:00"
    config_path = _write_payload(tmp_path, payload, "wider-range.json")
    output = tmp_path / "canonical-m2"

    with pytest.raises(ValueError, match="dataset.*range|timestamp range"):
        bootstrap_canonical_m2_study(config_path, output)

    assert not output.exists()


def test_fit_scope_must_have_eligible_training_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fakes(monkeypatch)
    payload = _config().to_payload()
    baseline = payload["baseline"]
    assert isinstance(baseline, dict)
    baseline["fit_cutoff"] = "2024-01-01T02:00:00"
    baseline["evaluation_start"] = "2024-01-01T02:00:00"
    config_path = _write_payload(tmp_path, payload, "empty-fit-scope.json")
    output = tmp_path / "canonical-m2"

    with pytest.raises(ValueError, match="eligible training rows|fit scope"):
        bootstrap_canonical_m2_study(config_path, output)

    assert not output.exists()


def test_end_provenance_is_captured_after_staging_graph_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fakes(monkeypatch)
    from trade_rl.evaluation.experiments.bootstrap import workflow as workflow_module

    real_provenance = workflow_module.build_candidate_run_provenance
    real_inspect = workflow_module.inspect_canonical_m2_bootstrap
    events: list[str] = []

    def provenance():
        events.append("provenance")
        return real_provenance()

    def inspect(root: str | Path):
        name = Path(root).name
        events.append(
            "inspect-staging"
            if name.startswith(".canonical-m2.staging-")
            else "inspect-final"
        )
        return real_inspect(root)

    monkeypatch.setattr(workflow_module, "build_candidate_run_provenance", provenance)
    monkeypatch.setattr(workflow_module, "inspect_canonical_m2_bootstrap", inspect)

    bootstrap_canonical_m2_study(_config_path(tmp_path), tmp_path / "canonical-m2")

    assert events == ["provenance", "inspect-staging", "provenance"]
