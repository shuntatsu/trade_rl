from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.c3_reporting_fixtures import refreshed, valid_summary_payload, write_summary
from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.evaluation.causal_scenario_c3_reporting import (
    evaluate_phase_a_gate,
    load_c3_aggregate_summary,
    load_c3_report_artifact,
    load_phase_a_gate_artifact,
    write_c3_report_artifact,
    write_phase_a_gate_artifact,
)


def _summary_and_gate(tmp_path: Path):
    source = tmp_path / "input" / "summary.json"
    write_summary(source)
    summary = load_c3_aggregate_summary(source)
    return summary, evaluate_phase_a_gate(summary)


def test_report_artifact_has_exact_canonical_closure(tmp_path: Path) -> None:
    summary, gate = _summary_and_gate(tmp_path)
    root = tmp_path / "report"

    written = write_c3_report_artifact(root, summary, gate)
    loaded = load_c3_report_artifact(root)

    assert {path.name for path in root.iterdir()} == {
        "manifest.json",
        "report.md",
        "summary.json",
    }
    assert loaded.artifact_digest == written.artifact_digest
    assert loaded.summary.summary_digest == summary.summary_digest
    assert loaded.gate.digest == gate.digest
    assert loaded.root == root
    for name in ("manifest.json", "summary.json"):
        raw = json.loads((root / name).read_text(encoding="utf-8"))
        assert (root / name).read_bytes() == canonical_json_bytes(raw)
    assert (root / "report.md").read_text(encoding="utf-8").endswith("\n")


def test_report_artifact_identical_rewrite_is_idempotent(tmp_path: Path) -> None:
    summary, gate = _summary_and_gate(tmp_path)
    root = tmp_path / "report"

    first = write_c3_report_artifact(root, summary, gate)
    first_bytes = {path.name: path.read_bytes() for path in root.iterdir()}
    second = write_c3_report_artifact(root, summary, gate)

    assert second.artifact_digest == first.artifact_digest
    assert {path.name: path.read_bytes() for path in root.iterdir()} == first_bytes


def test_report_artifact_conflicting_rewrite_fails(tmp_path: Path) -> None:
    summary, gate = _summary_and_gate(tmp_path)
    root = tmp_path / "report"
    write_c3_report_artifact(root, summary, gate)
    (root / "report.md").write_text("conflict\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="conflicting C3 report artifact"):
        write_c3_report_artifact(root, summary, gate)


def test_report_artifact_loader_rejects_extra_file(tmp_path: Path) -> None:
    summary, gate = _summary_and_gate(tmp_path)
    root = tmp_path / "report"
    write_c3_report_artifact(root, summary, gate)
    (root / "extra.txt").write_text("unexpected", encoding="utf-8")

    with pytest.raises(ValueError, match="file closure mismatch"):
        load_c3_report_artifact(root)


def test_report_artifact_loader_rejects_symlink_entry(tmp_path: Path) -> None:
    summary, gate = _summary_and_gate(tmp_path)
    root = tmp_path / "report"
    write_c3_report_artifact(root, summary, gate)
    (root / "report.md").unlink()
    try:
        (root / "report.md").symlink_to(tmp_path / "target.md")
    except OSError:
        pytest.skip("symlink creation is unavailable on this platform")

    with pytest.raises(ValueError, match="invalid file entry"):
        load_c3_report_artifact(root)


def test_gate_artifact_binds_report_identity(tmp_path: Path) -> None:
    summary, gate = _summary_and_gate(tmp_path)
    report = write_c3_report_artifact(tmp_path / "report", summary, gate)

    written = write_phase_a_gate_artifact(
        tmp_path / "gate",
        gate,
        report_artifact_digest=report.artifact_digest,
    )
    loaded = load_phase_a_gate_artifact(tmp_path / "gate")

    assert {path.name for path in (tmp_path / "gate").iterdir()} == {
        "gate.json",
        "manifest.json",
    }
    assert loaded.artifact_digest == written.artifact_digest
    assert loaded.report_artifact_digest == report.artifact_digest
    assert loaded.gate.digest == gate.digest
    assert loaded.gate.production_status == "NO-GO"


def test_gate_artifact_rejects_report_identity_substitution(tmp_path: Path) -> None:
    summary, gate = _summary_and_gate(tmp_path)
    report = write_c3_report_artifact(tmp_path / "report", summary, gate)
    root = tmp_path / "gate"
    write_phase_a_gate_artifact(
        root,
        gate,
        report_artifact_digest=report.artifact_digest,
    )
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["report_artifact_digest"] = "9" * 64
    (root / "manifest.json").write_bytes(canonical_json_bytes(manifest))

    with pytest.raises(ValueError, match="artifact digest mismatch"):
        load_phase_a_gate_artifact(root)


def test_report_artifact_detects_summary_digest_substitution(tmp_path: Path) -> None:
    summary, gate = _summary_and_gate(tmp_path)
    root = tmp_path / "report"
    write_c3_report_artifact(root, summary, gate)
    payload = valid_summary_payload()
    payload["mean_uplift"] = 0.99
    payload = refreshed(payload)
    (root / "summary.json").write_bytes(canonical_json_bytes(payload))

    with pytest.raises(ValueError, match="file digest mismatch"):
        load_c3_report_artifact(root)
