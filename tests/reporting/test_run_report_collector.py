from __future__ import annotations

import json
from pathlib import Path

import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.reporting import run_report as report_module
from trade_rl.reporting.run_report import RunStageStatus
from trade_rl.workflows.universal_causal_alpha_v3_config import (
    CausalAlphaV3ResearchConfig,
)
from trade_rl.workflows.universal_causal_alpha_v3_identity import (
    CausalAlphaV3ExecutionIdentity,
    CausalAlphaV3RunManifestV2,
)
from trade_rl.workflows.universal_causal_alpha_v3_pipeline import (
    authored_config_payload,
)

_EXAMPLE = Path("examples/binance/universal-causal-alpha-v3-research.json")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _with_digest(payload: dict[str, object]) -> dict[str, object]:
    result = dict(payload)
    result["artifact_digest"] = content_digest(payload)
    return result


def _write_identity(
    root: Path,
) -> tuple[CausalAlphaV3ResearchConfig, CausalAlphaV3RunManifestV2]:
    config = CausalAlphaV3ResearchConfig.from_json(_EXAMPLE)
    execution = CausalAlphaV3ExecutionIdentity(
        train_symbols=("BTCUSDT",),
        training_contract_digest="1" * 64,
        instrument_context_schema_digest="2" * 64,
        source_tree_digest="3" * 64,
        shared_clock_digest="4" * 64,
        dependency_lock_digest="5" * 64,
        python_runtime_digest="6" * 64,
        symbol_runtime_digests=(("BTCUSDT", "7" * 64),),
    )
    manifest = CausalAlphaV3RunManifestV2(
        train_symbols=execution.train_symbols,
        config_digest=config.digest,
        catalog_digest="8" * 64,
        partition_digest="9" * 64,
        split_manifest_digest="a" * 64,
        feature_schema_digest="b" * 64,
        statistics_digest="c" * 64,
        generator_code_digest="d" * 64,
        nested_partition_digest="e" * 64,
        execution_identity_digest=execution.digest,
        training_contract_digest=execution.training_contract_digest,
        instrument_context_schema_digest=execution.instrument_context_schema_digest,
    )
    _write_json(root / "execution-identity.json", execution.to_payload())
    _write_json(root / "run-manifest.json", manifest.to_payload())
    _write_json(root / "authored-config.json", authored_config_payload(config))
    return config, manifest


def _bootstrap(*, lower_ci: float = 0.01) -> dict[str, object]:
    return _with_digest(
        {
            "block_size": 1,
            "lower_ci": lower_ci,
            "mean": 0.02,
            "p_value": 0.1,
            "schema_version": "causal_alpha_v3_bootstrap_evidence_v1",
            "upper_ci": 0.03,
        }
    )


def _write_signal_pass(
    root: Path,
    config: CausalAlphaV3ResearchConfig,
    manifest: CausalAlphaV3RunManifestV2,
) -> None:
    fit_digest = config.candidates[0].fit.digest
    evidence = _with_digest(
        {
            "aggregation_mode": "cross_symbol_episode_mean",
            "direction_accuracy_excess": _bootstrap(),
            "expected_independent_episode_count": 8,
            "expected_raw_scope_count": 8,
            "gate_digest": config.signal_gate.digest,
            "independence_unit": "chronological_episode",
            "independent_episode_count": 8,
            "metric_digests": ["f" * 64],
            "passed": True,
            "promotion_eligible": False,
            "rank_ic": _bootstrap(),
            "raw_scope_count": 8,
            "raw_scope_coverage": 1.0,
            "rejection_reasons": [],
            "run_manifest_digest": manifest.digest,
            "schema_version": "causal_alpha_v3_signal_gate_evidence_v2",
            "top_bottom_spread": _bootstrap(),
        }
    )
    _write_json(
        root / "signal" / f"{fit_digest}.json",
        {
            "evidence": evidence,
            "fit_config_digest": fit_digest,
            "passed": True,
            "promotion_eligible": False,
            "schema_version": "causal_alpha_v3_fit_signal_result_v2",
            "unavailable_scope_contract_digests": [],
        },
    )


def _write_signal_reject(root: Path, config: CausalAlphaV3ResearchConfig) -> None:
    fit_result: dict[str, object] = {
        "evidence": None,
        "fit_config_digest": config.candidates[0].fit.digest,
        "passed": False,
        "promotion_eligible": False,
        "schema_version": "causal_alpha_v3_fit_signal_result_v2",
        "unavailable_scope_contract_digests": ["a" * 64],
    }
    _write_json(
        root / "signal" / "rejection.json",
        _with_digest(
            {
                "fit_results": [fit_result],
                "promotion_eligible": False,
                "schema_version": "causal_alpha_v3_signal_rejection_v2",
            }
        ),
    )


def _write_selection_progress(root: Path) -> None:
    _write_json(
        root / "selection" / "progress.json",
        {
            "candidates": [
                {
                    "candidate_digest": "1" * 64,
                    "completed_scope_count": 2,
                    "irrecoverably_rejected": False,
                    "mean_gross_return": 0.01,
                    "mean_net_return": 0.005,
                    "mean_turnover_per_day": 0.2,
                    "name": "baseline",
                    "total_trade_count": 4,
                    "worst_net_return": -0.01,
                }
            ],
            "completed_replay_count": 2,
            "completion_fraction": 0.5,
            "diagnostics_completed_count": 2,
            "expected_replay_count": 4,
            "fit_cache_hits": 1,
            "fit_count": 1,
            "promotion_eligible": False,
            "research_only": True,
            "schema_version": "causal_alpha_v3_selection_progress_v1",
            "symbols": {
                "BTCUSDT": {
                    "completed_scope_count": 2,
                    "mean_gross_return": 0.01,
                    "mean_net_return": 0.005,
                    "mean_turnover_per_day": 0.2,
                    "total_trade_count": 4,
                }
            },
        },
    )


def _write_selection_pass(root: Path, selected_candidate_digest: str) -> str:
    evidence = _with_digest(
        {
            "candidate_evidence_digests": ["2" * 64],
            "freeze_digest": "3" * 64,
            "promotion_eligible": False,
            "schema_version": "causal_alpha_v3_selection_evidence_v1",
            "selected_candidate_digest": selected_candidate_digest,
        }
    )
    _write_json(root / "selection" / "evidence.json", evidence)
    return str(evidence["artifact_digest"])


def _write_admission_reject(root: Path, *, selected_candidate_digest: str) -> None:
    evidence = _with_digest(
        {
            "aggregate_gross_return": 0.01,
            "aggregate_net_return": -0.01,
            "base_admission_digest": "4" * 64,
            "hard_risk_violation_count": 0,
            "negative_gross_symbol_count": 0,
            "passed": False,
            "promotion_eligible": False,
            "record_digests": ["5" * 64],
            "rejection_reasons": ["negative_aggregate_net_return"],
            "schema_version": "causal_alpha_v3_admission_evidence_v3",
            "total_trade_count": 2,
            "unexplained_execution_rejection_count": 0,
            "worst_symbol_net_return": -0.01,
        }
    )
    _write_json(root / "admission" / "evidence.json", evidence)
    _write_json(
        root / "admission" / "rejection.json",
        _with_digest(
            {
                "admission_digest": evidence["artifact_digest"],
                "promotion_eligible": False,
                "schema_version": "causal_alpha_v3_admission_rejection_v2",
                "selected_candidate_digest": selected_candidate_digest,
            }
        ),
    )


def _write_generic_stage(
    root: Path,
    *,
    stage: str,
    status: str,
    reasons: tuple[str, ...] = (),
) -> None:
    _write_json(
        root / "reporting" / "stages" / f"{stage}.json",
        _with_digest(
            {
                "artifact_digests": {"checkpoint": "6" * 64},
                "metrics": {"loss": 0.125, "steps": 100},
                "reasons": list(reasons),
                "schema_version": "run_report_stage_evidence_v1",
                "stage": stage,
                "status": status,
            }
        ),
    )


def _build(root: Path):
    build = getattr(report_module, "build_run_report")
    return build(root)


def _status(report, stage: str) -> RunStageStatus:
    return next(item.status for item in report.stages if item.name == stage)


def test_collector_reports_existing_empty_root_as_missing(tmp_path: Path) -> None:
    report = _build(tmp_path)

    assert all(stage.status is RunStageStatus.MISSING for stage in report.stages)


def test_signal_rejection_proves_all_downstream_stages_not_run(tmp_path: Path) -> None:
    config, _manifest = _write_identity(tmp_path)
    _write_signal_reject(tmp_path, config)

    report = _build(tmp_path)

    assert _status(report, "signal") is RunStageStatus.REJECT
    assert all(stage.status is RunStageStatus.NOT_RUN for stage in report.stages[1:])


def test_selection_progress_without_terminal_evidence_is_in_progress(
    tmp_path: Path,
) -> None:
    config, manifest = _write_identity(tmp_path)
    _write_signal_pass(tmp_path, config, manifest)
    _write_selection_progress(tmp_path)

    report = _build(tmp_path)
    selection = report.stages[1]

    assert report.stages[0].status is RunStageStatus.PASS
    assert selection.status is RunStageStatus.IN_PROGRESS
    assert selection.metrics["completed_replay_count"] == 2
    assert selection.metrics["expected_replay_count"] == 4
    assert selection.metrics["candidate_rows"][0]["name"] == "baseline"
    assert selection.metrics["symbol_rows"]["BTCUSDT"]["total_trade_count"] == 4
    assert all(stage.status is RunStageStatus.MISSING for stage in report.stages[2:])


def test_admission_rejection_copies_persisted_economics_and_blocks_learners(
    tmp_path: Path,
) -> None:
    config, manifest = _write_identity(tmp_path)
    _write_signal_pass(tmp_path, config, manifest)
    selected = config.candidates[0].digest
    _write_selection_pass(tmp_path, selected)
    _write_admission_reject(tmp_path, selected_candidate_digest=selected)

    report = _build(tmp_path)
    admission = report.stages[2]

    assert report.stages[1].status is RunStageStatus.PASS
    assert admission.status is RunStageStatus.REJECT
    assert admission.metrics["aggregate_gross_return"] == pytest.approx(0.01)
    assert admission.metrics["aggregate_net_return"] == pytest.approx(-0.01)
    assert admission.metrics["worst_symbol_net_return"] == pytest.approx(-0.01)
    assert admission.reasons == ("negative_aggregate_net_return",)
    assert all(stage.status is RunStageStatus.NOT_RUN for stage in report.stages[3:])


def test_collector_fails_closed_on_cross_identity_digest_corruption(
    tmp_path: Path,
) -> None:
    _config, _manifest = _write_identity(tmp_path)
    execution_path = tmp_path / "execution-identity.json"
    raw = json.loads(execution_path.read_text(encoding="utf-8"))
    raw["artifact_digest"] = "0" * 64
    _write_json(execution_path, raw)

    report = _build(tmp_path)

    assert all(stage.status is RunStageStatus.INVALID for stage in report.stages[:4])
    assert all(stage.status is RunStageStatus.MISSING for stage in report.stages[4:])


def test_generic_downstream_stage_evidence_populates_without_llm_analysis(
    tmp_path: Path,
) -> None:
    _write_generic_stage(tmp_path, stage="behavior_cloning", status="PASS")

    report = _build(tmp_path)
    bc = report.stages[4]

    assert bc.status is RunStageStatus.PASS
    assert bc.metrics == {"loss": 0.125, "steps": 100}
    assert bc.artifact_digests == {"checkpoint": "6" * 64}
    assert report.stages[5].status is RunStageStatus.MISSING


def test_generic_stage_pass_is_invalid_when_upstream_rejection_proves_blocking(
    tmp_path: Path,
) -> None:
    config, _manifest = _write_identity(tmp_path)
    _write_signal_reject(tmp_path, config)
    _write_generic_stage(tmp_path, stage="behavior_cloning", status="PASS")

    report = _build(tmp_path)

    assert report.stages[4].status is RunStageStatus.INVALID
    assert "upstream_rejection_conflict" in report.stages[4].reasons


def test_malformed_generic_stage_is_invalid_not_silently_missing(
    tmp_path: Path,
) -> None:
    _write_json(
        tmp_path / "reporting" / "stages" / "ppo.json",
        {"schema_version": "wrong", "status": "PASS"},
    )

    report = _build(tmp_path)

    assert report.stages[6].status is RunStageStatus.INVALID
