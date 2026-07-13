from __future__ import annotations

import json
from pathlib import Path

import pytest

from trade_rl.artifacts.legacy_migration import (
    BaselineFallbackStatus,
    PolicyCandidateStatus,
    ReleaseStatus,
    ResearchRunStatus,
    migrate_legacy_research_run,
)
from trade_rl.domain.signals import SignalStatus

FIXTURES = Path(__file__).parents[1] / "fixtures" / "legacy"


def load(name: str) -> dict[str, object]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_realdata_run_is_baseline_analysis_not_a_selected_ppo() -> None:
    result = migrate_legacy_research_run(
        report=load("realdata_20260713_report.json"),
        model_manifest=load("realdata_20260713_manifest.json"),
        signal_metadata=load("realdata_20260713_signal_metadata.json"),
    )

    assert result.research_run_status is ResearchRunStatus.COMPLETED
    assert result.signal_status is SignalStatus.REJECTED
    assert result.policy_candidate_status is PolicyCandidateStatus.NOT_SELECTED
    assert result.baseline_fallback_status is BaselineFallbackStatus.SELECTED_FOR_ANALYSIS
    assert result.release_status is ReleaseStatus.BLOCKED
    assert result.selected_configuration == "A"
    assert result.selected_policy_digest is None
    assert result.policy_ensemble_members == ()
    assert result.holdout_total_return == pytest.approx(0.23499421308975021)
    assert result.cost2x_total_return == pytest.approx(0.1599508186145191)
    assert result.positive_return_p_value == pytest.approx(0.11)


def test_signal_metadata_is_not_mislabeled_as_policy_ensemble() -> None:
    result = migrate_legacy_research_run(
        report=load("realdata_20260713_report.json"),
        model_manifest=load("realdata_20260713_manifest.json"),
        signal_metadata=load("realdata_20260713_signal_metadata.json"),
    )

    assert result.signal_model_kind == "gbm"
    assert result.signal_dataset_id == (
        "b4c41e64daa34b4d184bcc213e7034034d1dc888307a0add49d4c7d42f4ae98a"
    )
    assert "signal metadata is not a PPO ensemble artifact" in result.notes


def test_migration_rejects_inconsistent_dataset_identity() -> None:
    signal = load("realdata_20260713_signal_metadata.json")
    signal["dataset_identity"] = "f" * 64

    with pytest.raises(ValueError, match="dataset identity"):
        migrate_legacy_research_run(
            report=load("realdata_20260713_report.json"),
            model_manifest=load("realdata_20260713_manifest.json"),
            signal_metadata=signal,
        )


def test_baseline_only_report_rejects_selected_model_path() -> None:
    report = load("realdata_20260713_report.json")
    report["selected_model_path"] = "policy/member-000/model.zip"

    with pytest.raises(ValueError, match="baseline_only"):
        migrate_legacy_research_run(
            report=report,
            model_manifest=load("realdata_20260713_manifest.json"),
            signal_metadata=load("realdata_20260713_signal_metadata.json"),
        )
