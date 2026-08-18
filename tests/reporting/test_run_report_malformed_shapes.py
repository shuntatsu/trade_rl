from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.reporting.test_run_report_collector import (
    _with_digest,
    _write_identity,
    _write_json,
    _write_selection_pass,
    _write_selection_progress,
    _write_signal_pass,
)
from trade_rl.reporting.run_report import RunStageStatus, build_run_report


def _rewrite_with_digest(path: Path, **changes: object) -> None:
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw.pop("artifact_digest", None)
    raw.update(changes)
    _write_json(path, _with_digest(raw))


@pytest.mark.parametrize(
    "field,value",
    (
        ("unavailable_scope_contract_digests", "not-an-array"),
        ("evidence.metric_digests", "not-an-array"),
    ),
)
def test_signal_rejects_malformed_digest_arrays(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    config, manifest = _write_identity(tmp_path)
    _write_signal_pass(tmp_path, config, manifest)
    path = tmp_path / "signal" / f"{config.candidates[0].fit.digest}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    if field == "unavailable_scope_contract_digests":
        raw[field] = value
    else:
        evidence = dict(raw["evidence"])
        evidence.pop("artifact_digest")
        evidence["metric_digests"] = value
        raw["evidence"] = _with_digest(evidence)
    _write_json(path, raw)

    report = build_run_report(tmp_path)

    assert report.stages[0].status is RunStageStatus.INVALID


def test_signal_rejection_requires_fit_results_array(tmp_path: Path) -> None:
    config, _manifest = _write_identity(tmp_path)
    path = tmp_path / "signal" / "rejection.json"
    _write_json(
        path,
        _with_digest(
            {
                "fit_results": "not-an-array",
                "promotion_eligible": False,
                "schema_version": "causal_alpha_v3_signal_rejection_v2",
            }
        ),
    )

    report = build_run_report(tmp_path)

    assert report.stages[0].status is RunStageStatus.INVALID


def test_selection_rejects_candidate_digest_string(tmp_path: Path) -> None:
    config, manifest = _write_identity(tmp_path)
    _write_signal_pass(tmp_path, config, manifest)
    _write_selection_pass(tmp_path, config.candidates[0].digest)
    path = tmp_path / "selection" / "evidence.json"
    _rewrite_with_digest(path, candidate_evidence_digests="2" * 64)

    report = build_run_report(tmp_path)

    assert report.stages[1].status is RunStageStatus.INVALID


def test_selection_progress_requires_integer_replay_counts(tmp_path: Path) -> None:
    config, manifest = _write_identity(tmp_path)
    _write_signal_pass(tmp_path, config, manifest)
    _write_selection_progress(tmp_path)
    path = tmp_path / "selection" / "progress.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["completed_replay_count"] = "2"
    _write_json(path, raw)

    report = build_run_report(tmp_path)

    assert report.stages[1].status is RunStageStatus.INVALID


def _write_admission_pass(root: Path) -> str:
    evidence = _with_digest(
        {
            "aggregate_gross_return": 0.01,
            "aggregate_net_return": 0.005,
            "base_admission_digest": "4" * 64,
            "hard_risk_violation_count": 0,
            "negative_gross_symbol_count": 0,
            "passed": True,
            "promotion_eligible": False,
            "record_digests": ["5" * 64],
            "rejection_reasons": [],
            "schema_version": "causal_alpha_v3_admission_evidence_v3",
            "total_trade_count": 2,
            "unexplained_execution_rejection_count": 0,
            "worst_symbol_net_return": 0.005,
        }
    )
    _write_json(root / "admission" / "evidence.json", evidence)
    return str(evidence["artifact_digest"])


def test_admission_requires_record_digest_array(tmp_path: Path) -> None:
    config, manifest = _write_identity(tmp_path)
    _write_signal_pass(tmp_path, config, manifest)
    _write_selection_pass(tmp_path, config.candidates[0].digest)
    _write_admission_pass(tmp_path)
    path = tmp_path / "admission" / "evidence.json"
    _rewrite_with_digest(path, record_digests="5" * 64)

    report = build_run_report(tmp_path)

    assert report.stages[2].status is RunStageStatus.INVALID


def _write_teacher_package(
    root: Path,
    *,
    run_manifest_digest: str,
    generator_code_digest: str,
    selected_candidate_digest: str,
    selection_digest: str,
    admission_digest: str,
    train_symbols: object,
    admission_contract_digests: object,
) -> None:
    _write_json(
        root / "teacher" / "package.json",
        _with_digest(
            {
                "admission_contract_digests": admission_contract_digests,
                "batch_artifact_digests": {"BTCUSDT": "7" * 64},
                "batch_digests": {"BTCUSDT": "8" * 64},
                "freeze_digest": "3" * 64,
                "generator_code_digest": generator_code_digest,
                "partition_digests": {"BTCUSDT": "9" * 64},
                "promotion_eligible": False,
                "research_only": True,
                "run_manifest_digest": run_manifest_digest,
                "sample_digests": {"BTCUSDT": "a" * 64},
                "schema_version": "universal_causal_alpha_v3_teacher_package_v2",
                "selected_candidate_digest": selected_candidate_digest,
                "selection_digest": selection_digest,
                "teacher_admission_digest": admission_digest,
                "teacher_admission_passed": True,
                "train_symbols": train_symbols,
            }
        ),
    )


@pytest.mark.parametrize(
    "train_symbols,admission_contract_digests",
    (
        ("BTCUSDT", {"BTCUSDT": "6" * 64}),
        (["BTCUSDT"], ["6" * 64]),
    ),
)
def test_teacher_package_rejects_malformed_scope_shapes(
    tmp_path: Path,
    train_symbols: object,
    admission_contract_digests: object,
) -> None:
    config, manifest = _write_identity(tmp_path)
    _write_signal_pass(tmp_path, config, manifest)
    selected = config.candidates[0].digest
    selection_digest = _write_selection_pass(tmp_path, selected)
    admission_digest = _write_admission_pass(tmp_path)
    _write_teacher_package(
        tmp_path,
        run_manifest_digest=manifest.digest,
        generator_code_digest=manifest.generator_code_digest,
        selected_candidate_digest=selected,
        selection_digest=selection_digest,
        admission_digest=admission_digest,
        train_symbols=train_symbols,
        admission_contract_digests=admission_contract_digests,
    )

    report = build_run_report(tmp_path)

    assert report.stages[3].status is RunStageStatus.INVALID
