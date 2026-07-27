from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.evaluation import causal_scenario_c3_markdown as module


def _fold(index: int) -> SimpleNamespace:
    return SimpleNamespace(
        fold_id=f"fold-{index}",
        selection_days=30,
        effective_days=30,
        mean_uplift=0.01 + index * 0.001,
        mean_spearman=0.2,
        mean_regret_margin=0.03,
        scenario_oracle_max_drawdown=0.10,
        trend_max_drawdown=0.12,
        required_adverse_passed=True,
        perfect_information_valid=True,
        failure_reasons=(),
    )


def _report() -> SimpleNamespace:
    return SimpleNamespace(
        digest="1" * 64,
        folds=tuple(_fold(index) for index in range(6)),
        fold_count=6,
        total_selection_days=180,
        total_effective_days=180,
        positive_uplift_folds=6,
        mean_uplift=0.012,
        uplift_lower_ci=0.005,
        uplift_upper_ci=0.020,
        uplift_p_value=0.01,
        mean_spearman=0.2,
        spearman_lower_ci=0.1,
        spearman_upper_ci=0.3,
        mean_regret_margin=0.03,
        regret_margin_lower_ci=0.01,
        regret_margin_upper_ci=0.05,
        worst_scenario_oracle_drawdown=0.10,
        worst_trend_drawdown=0.12,
        neighbor_distance_p50=0.10,
        neighbor_distance_p90=0.20,
        neighbor_distance_p99=0.30,
        unique_anchor_count=100,
        effective_anchor_count=64.0,
        anchor_max_share=0.05,
        historical_coverage_fraction=0.8,
        calibration_buckets=(SimpleNamespace(bucket_index=0, sample_count=180),),
        execution_summaries=(
            SimpleNamespace(
                execution_scenario="adverse_cost_2x",
                policy_kind="scenario_oracle",
                observation_count=180,
                mean_gross_log_return=0.009,
                mean_total_economic_cost=0.002,
                mean_fill_ratio=0.92,
                maximum_drawdown=0.14,
            ),
            SimpleNamespace(
                execution_scenario="nominal",
                policy_kind="scenario_oracle",
                observation_count=180,
                mean_gross_log_return=0.012,
                mean_total_economic_cost=0.001,
                mean_fill_ratio=0.96,
                maximum_drawdown=0.12,
            ),
        ),
        failure_reasons=(),
    )


def _gate(*, report_digest: str = "1" * 64) -> SimpleNamespace:
    conditions = tuple(
        SimpleNamespace(name=f"condition_{index}", passed=True, detail="complete")
        for index in range(9)
    )
    return SimpleNamespace(
        digest="2" * 64,
        report_digest=report_digest,
        config_digest="3" * 64,
        conditions=conditions,
        failed_condition_names=(),
        passed=True,
    )


def test_markdown_renderer_is_deterministic_and_explicitly_no_go() -> None:
    first = module.render_c3_markdown(_report(), _gate())
    second = module.render_c3_markdown(_report(), _gate())

    assert first == second
    assert first.endswith("\n")
    assert "# Causal Scenario C3 Evaluation Report" in first
    assert "Production status: **NO-GO**" in first
    assert "Phase A gate: **PASS**" in first
    assert "fold-0" in first
    assert "adverse_cost_2x" in first
    assert "condition_0" in first


def test_markdown_renderer_rejects_gate_report_substitution() -> None:
    with pytest.raises(ValueError, match="does not bind"):
        module.render_c3_markdown(_report(), _gate(report_digest="9" * 64))


def test_markdown_artifact_binds_verified_core_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_root = tmp_path / "report"
    gate_root = tmp_path / "gate"
    report_root.mkdir()
    gate_root.mkdir()
    report = _report()
    gate = _gate()
    monkeypatch.setattr(
        module,
        "load_c3_aggregate_report_artifact",
        lambda root: SimpleNamespace(
            report=report,
            artifact_digest="4" * 64,
            root=Path(root),
        ),
    )
    monkeypatch.setattr(
        module,
        "load_phase_a_gate_artifact",
        lambda root: SimpleNamespace(
            gate=gate,
            artifact_digest="5" * 64,
            root=Path(root),
        ),
    )

    written = module.write_c3_markdown_artifact(
        tmp_path / "markdown",
        report_root=report_root,
        gate_root=gate_root,
    )
    loaded = module.load_c3_markdown_artifact(
        tmp_path / "markdown",
        report_root=report_root,
        gate_root=gate_root,
    )

    assert written.artifact_digest == loaded.artifact_digest
    assert written.report_artifact_digest == "4" * 64
    assert written.gate_artifact_digest == "5" * 64
    assert {path.name for path in (tmp_path / "markdown").iterdir()} == {
        "manifest.json",
        "report.md",
    }
    manifest = json.loads(
        (tmp_path / "markdown" / "manifest.json").read_text(encoding="utf-8")
    )
    assert (
        tmp_path / "markdown" / "manifest.json"
    ).read_bytes() == canonical_json_bytes(manifest)
    assert manifest["production_status"] == "NO-GO"


def test_markdown_artifact_rejects_extra_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        module,
        "load_c3_aggregate_report_artifact",
        lambda root: SimpleNamespace(
            report=_report(), artifact_digest="4" * 64, root=Path(root)
        ),
    )
    monkeypatch.setattr(
        module,
        "load_phase_a_gate_artifact",
        lambda root: SimpleNamespace(
            gate=_gate(), artifact_digest="5" * 64, root=Path(root)
        ),
    )
    report_root = tmp_path / "report"
    gate_root = tmp_path / "gate"
    report_root.mkdir()
    gate_root.mkdir()
    root = tmp_path / "markdown"
    module.write_c3_markdown_artifact(
        root, report_root=report_root, gate_root=gate_root
    )
    (root / "extra.txt").write_text("unexpected", encoding="utf-8")

    with pytest.raises(ValueError, match="file closure"):
        module.load_c3_markdown_artifact(
            root, report_root=report_root, gate_root=gate_root
        )
