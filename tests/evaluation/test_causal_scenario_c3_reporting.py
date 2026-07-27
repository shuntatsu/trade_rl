from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from trade_rl.evaluation.causal_scenario_c3_reporting import (
    C3PhaseAGateConfig,
    evaluate_phase_a_gate,
    load_c3_aggregate_summary,
    render_c3_markdown,
)


def test_load_c3_aggregate_summary_accepts_canonical_valid_payload(
    tmp_path: Path,
    c3_reporting,
) -> None:
    path = tmp_path / "summary.json"
    payload = c3_reporting.write_summary(path)

    summary = load_c3_aggregate_summary(path)

    assert summary.summary_digest == payload["summary_digest"]
    assert summary.source_run_digest == payload["source_run_digest"]
    assert summary.fold_count == 6
    assert summary.execution_scenario_names == (
        "adverse_spread_2x",
        "nominal",
    )
    assert summary.production_status == "NO-GO"


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda payload: payload.__setitem__("unknown", True), "field closure"),
        (lambda payload: payload.__setitem__("summary_digest", "0" * 64), "digest"),
        (lambda payload: payload.__setitem__("production_status", "GO"), "NO-GO"),
        (
            lambda payload: payload["folds"].__setitem__(
                1, deepcopy(payload["folds"][0])
            ),
            "fold IDs",
        ),
        (
            lambda payload: payload["execution_summaries"].reverse(),
            "sorted",
        ),
        (
            lambda payload: payload.__setitem__("uplift_lower_ci", 0.03),
            "uplift confidence interval",
        ),
        (
            lambda payload: payload.__setitem__("neighbor_distance_p50", 0.25),
            "neighbor distance quantiles",
        ),
    ],
)
def test_load_c3_aggregate_summary_rejects_invalid_evidence(
    tmp_path: Path,
    c3_reporting,
    mutate,
    match: str,
) -> None:
    payload = c3_reporting.valid_summary_payload()
    baseline_digest = payload["summary_digest"]
    mutate(payload)
    if payload.get("summary_digest") == baseline_digest:
        payload = c3_reporting.refreshed(payload)
    path = tmp_path / "summary.json"
    c3_reporting.write_summary(path, payload)

    with pytest.raises(ValueError, match=match):
        load_c3_aggregate_summary(path)


def test_load_c3_aggregate_summary_rejects_non_finite_number(
    tmp_path: Path,
    c3_reporting,
) -> None:
    payload = c3_reporting.valid_summary_payload()
    payload["uplift_lower_ci"] = float("inf")
    path = tmp_path / "summary.json"
    path.write_text(
        json.dumps(payload, allow_nan=True, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="finite"):
        load_c3_aggregate_summary(path)


def test_load_c3_aggregate_summary_rejects_non_canonical_json(
    tmp_path: Path,
    c3_reporting,
) -> None:
    path = tmp_path / "summary.json"
    c3_reporting.write_summary(path, canonical=False)

    with pytest.raises(ValueError, match="canonical JSON"):
        load_c3_aggregate_summary(path)


def test_phase_a_gate_passes_only_complete_supported_evidence(
    tmp_path: Path,
    c3_reporting,
) -> None:
    path = tmp_path / "summary.json"
    c3_reporting.write_summary(path)
    summary = load_c3_aggregate_summary(path)

    gate = evaluate_phase_a_gate(summary)

    assert gate.passed is True
    assert gate.failed_condition_names == ()
    assert len(gate.conditions) == 9
    assert gate.report_digest == summary.summary_digest
    assert gate.config_digest == C3PhaseAGateConfig().digest
    assert gate.production_status == "NO-GO"


def test_phase_a_gate_reports_specific_failed_conditions(
    tmp_path: Path,
    c3_reporting,
) -> None:
    payload = c3_reporting.valid_summary_payload()
    for fold in payload["folds"][2:]:
        fold["mean_uplift"] = -0.001
    payload["folds"][0]["required_adverse_passed"] = False
    payload["positive_uplift_folds"] = 2
    payload["uplift_lower_ci"] = -0.001
    payload["all_required_adverse_passed"] = False
    payload = c3_reporting.refreshed(payload)
    path = tmp_path / "summary.json"
    c3_reporting.write_summary(path, payload)
    summary = load_c3_aggregate_summary(path)

    gate = evaluate_phase_a_gate(summary)

    assert gate.passed is False
    assert gate.failed_condition_names == (
        "positive_uplift_folds",
        "aggregate_uplift_confidence",
        "required_adverse_execution",
    )
    assert gate.production_status == "NO-GO"


def test_render_c3_markdown_is_deterministic_and_complete(
    tmp_path: Path,
    c3_reporting,
) -> None:
    path = tmp_path / "summary.json"
    c3_reporting.write_summary(path)
    summary = load_c3_aggregate_summary(path)
    gate = evaluate_phase_a_gate(summary)

    first = render_c3_markdown(summary, gate)
    second = render_c3_markdown(summary, gate)

    assert first == second
    assert first.endswith("\n")
    assert "# Causal Scenario C3 Evaluation Report" in first
    assert f"Summary digest: `{summary.summary_digest}`" in first
    assert "Production status: **NO-GO**" in first
    assert "Phase A gate: **PASS**" in first
    assert "adverse_spread_2x" in first
    assert "fold-0" in first
    assert "integrity_and_determinism" in first
