from __future__ import annotations

import json

import pytest

from trade_rl.reporting.markdown import render_run_report_markdown
from trade_rl.reporting.run_report import (
    RUN_REPORT_STAGE_ORDER,
    RunReport,
    RunStageReport,
    RunStageStatus,
)


def _empty_stages() -> tuple[RunStageReport, ...]:
    return tuple(
        RunStageReport(name=name, status=RunStageStatus.NOT_RUN)
        for name in RUN_REPORT_STAGE_ORDER
    )


def test_run_report_contract_has_fixed_stage_order_and_deterministic_json() -> None:
    report = RunReport(
        root="/tmp/example-run",
        identities={"run_manifest_digest": "a" * 64},
        stages=_empty_stages(),
    )

    assert report.schema_version == "run_report_v1"
    assert tuple(stage.name for stage in report.stages) == RUN_REPORT_STAGE_ORDER
    assert report.to_payload() == {
        "identities": {"run_manifest_digest": "a" * 64},
        "root": "/tmp/example-run",
        "schema_version": "run_report_v1",
        "stages": [
            {
                "artifact_digests": {},
                "metrics": {},
                "name": name,
                "reasons": [],
                "source_paths": [],
                "status": "NOT_RUN",
            }
            for name in RUN_REPORT_STAGE_ORDER
        ],
    }
    first = report.to_json()
    second = report.to_json()
    assert first == second
    assert json.loads(first) == report.to_payload()
    assert first.endswith("\n")


def test_run_report_rejects_missing_or_reordered_stage_contract() -> None:
    stages = list(_empty_stages())
    stages[0], stages[1] = stages[1], stages[0]

    with pytest.raises(ValueError, match="stage order"):
        RunReport(root="/tmp/run", identities={}, stages=tuple(stages))


def test_run_stage_report_rejects_unknown_status_or_duplicate_reasons() -> None:
    with pytest.raises(ValueError, match="status"):
        RunStageReport(name="signal", status="PASS")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="reasons"):
        RunStageReport(
            name="signal",
            status=RunStageStatus.REJECT,
            reasons=("rank_ic_lower_ci", "rank_ic_lower_ci"),
        )


def test_markdown_renderer_is_fact_only_and_renders_progress_tables() -> None:
    stages = list(_empty_stages())
    stages[0] = RunStageReport(
        name="signal",
        status=RunStageStatus.PASS,
        metrics={
            "fit_count": 2,
            "independent_episode_count": 8,
            "raw_scope_coverage": 1.0,
        },
        artifact_digests={"signal_evidence": "b" * 64},
        source_paths=("signal/fit.json",),
    )
    stages[1] = RunStageReport(
        name="selection",
        status=RunStageStatus.IN_PROGRESS,
        metrics={
            "completed_replay_count": 2,
            "expected_replay_count": 4,
            "completion_fraction": 0.5,
            "candidate_rows": (
                {
                    "name": "baseline",
                    "completed_scope_count": 2,
                    "mean_gross_return": 0.01,
                    "mean_net_return": 0.005,
                    "worst_net_return": -0.01,
                    "mean_turnover_per_day": 0.2,
                    "irrecoverably_rejected": False,
                },
            ),
            "symbol_rows": {
                "BTCUSDT": {
                    "completed_scope_count": 2,
                    "mean_gross_return": 0.01,
                    "mean_net_return": 0.005,
                    "mean_turnover_per_day": 0.2,
                    "total_trade_count": 4,
                }
            },
        },
        source_paths=("selection/progress.json",),
    )
    report = RunReport(
        root="/tmp/example-run",
        identities={
            "run_manifest_digest": "a" * 64,
            "source_tree_digest": "c" * 64,
        },
        stages=tuple(stages),
    )

    rendered = render_run_report_markdown(report)

    assert rendered.startswith("# Machine Run Report\n")
    assert "| signal | PASS |" in rendered
    assert "| selection | IN_PROGRESS |" in rendered
    assert "| baseline | 2 | 1.000000% | 0.500000% | -1.000000% | 0.200000 | false |" in rendered
    assert "| BTCUSDT | 2 | 1.000000% | 0.500000% | 0.200000 | 4 |" in rendered
    assert "recommend" not in rendered.lower()
    assert "should" not in rendered.lower()
