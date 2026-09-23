from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from tests.evaluation.experiments.bootstrap.test_final_window_config import (
    _v3_payload,
)
from tests.evaluation.experiments.bootstrap.test_workflow import _install_fakes
from trade_rl.evaluation.experiments import inspect_study
from trade_rl.evaluation.experiments.bootstrap.config import (
    load_canonical_m2_bootstrap_config,
)
from trade_rl.evaluation.experiments.bootstrap.workflow import (
    bootstrap_canonical_m2_study,
    inspect_canonical_m2_bootstrap,
)
from trade_rl.evaluation.experiments.contracts.research import EvidenceUse


def _write(tmp_path: Path, payload: object, name: str = "bootstrap.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _context_payload() -> dict[str, object]:
    return {
        "schema_version": "study_research_context_v1",
        "parent_context_digests": ["c" * 64, "d" * 64],
        "consumed_evidence": [
            {
                "evidence_kind": "EXPERIMENT_DECISION",
                "evidence_digest": "a" * 64,
                "development_start": "2023-01-01T00:00:00.000000000",
                "development_stop_exclusive": "2025-01-01T00:00:00.000000000",
                "uses": [
                    EvidenceUse.HYPOTHESIS_FORMATION.value,
                    EvidenceUse.OBSERVATION_SELECTION.value,
                ],
            }
        ],
    }


def _v4_payload() -> dict[str, object]:
    payload = deepcopy(_v3_payload())
    payload["schema_version"] = "canonical_m2_bootstrap_config_v4"
    payload["research_context"] = _context_payload()
    return payload


def test_v4_binds_research_context_into_config_digest(tmp_path: Path) -> None:
    base = load_canonical_m2_bootstrap_config(
        _write(tmp_path, _v4_payload(), "base.json")
    )
    changed_payload = _v4_payload()
    context = changed_payload["research_context"]
    assert isinstance(context, dict)
    consumed = context["consumed_evidence"]
    assert isinstance(consumed, list)
    first = consumed[0]
    assert isinstance(first, dict)
    first["uses"] = [EvidenceUse.RESULT_INTERPRETATION.value]
    changed = load_canonical_m2_bootstrap_config(
        _write(tmp_path, changed_payload, "changed.json")
    )

    assert base.schema_version == "canonical_m2_bootstrap_config_v4"
    assert base.research_context is not None
    assert base.to_payload()["research_context"] == base.research_context.to_payload()
    assert base.digest != changed.digest


def test_v4_requires_research_context_and_v3_rejects_it(tmp_path: Path) -> None:
    missing = _v4_payload()
    del missing["research_context"]
    with pytest.raises(ValueError, match="research_context|missing"):
        load_canonical_m2_bootstrap_config(_write(tmp_path, missing, "missing.json"))

    legacy = _v3_payload()
    legacy["research_context"] = _context_payload()
    with pytest.raises(ValueError, match="unknown"):
        load_canonical_m2_bootstrap_config(_write(tmp_path, legacy, "legacy.json"))


def test_v4_rejects_consumed_evidence_overlapping_preregistered_final(
    tmp_path: Path,
) -> None:
    payload = _v4_payload()
    context = payload["research_context"]
    assert isinstance(context, dict)
    consumed = context["consumed_evidence"]
    assert isinstance(consumed, list)
    first = consumed[0]
    assert isinstance(first, dict)
    first["development_start"] = "2024-12-15T00:00:00.000000000"
    first["development_stop_exclusive"] = "2025-02-01T00:00:00.000000000"

    with pytest.raises(ValueError, match="final.*consumed|consumed.*final"):
        load_canonical_m2_bootstrap_config(_write(tmp_path, payload))


def test_v4_bootstrap_binds_context_into_study_and_reconstructs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fakes(monkeypatch)
    config_path = _write(tmp_path, _v4_payload(), "v4-bootstrap.json")
    output = tmp_path / "canonical-m2-v4"

    result = bootstrap_canonical_m2_study(config_path, output)
    config = load_canonical_m2_bootstrap_config(config_path)
    snapshot = inspect_study(output / "study")

    assert snapshot.plan.schema_version == "controlled_study_plan_v3"
    assert snapshot.plan.research_context == config.research_context
    assert snapshot.plan.digest == result.study_digest
    assert inspect_canonical_m2_bootstrap(output) == result


def test_v4_bootstrap_manifest_rejects_research_context_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fakes(monkeypatch)
    output = tmp_path / "canonical-m2-v4"
    bootstrap_canonical_m2_study(
        _write(tmp_path, _v4_payload(), "v4-bootstrap.json"),
        output,
    )

    plan_path = output / "study" / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert isinstance(plan, dict)
    context = plan["research_context"]
    assert isinstance(context, dict)
    consumed = context["consumed_evidence"]
    assert isinstance(consumed, list)
    first = consumed[0]
    assert isinstance(first, dict)
    first["uses"] = [EvidenceUse.RESULT_INTERPRETATION.value]
    plan_path.write_text(
        json.dumps(plan, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Study digest|bootstrap manifest|research context",
    ):
        inspect_canonical_m2_bootstrap(output)
