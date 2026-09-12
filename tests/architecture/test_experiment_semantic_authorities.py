from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from trade_rl.evaluation.experiments.contracts import ResolvedRunConfig, StudyPlan
from trade_rl.evaluation.runs import (
    CandidateRunConfig,
    LeanCandidateConfig,
    ResolvedCandidateRunSpec,
)

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "trade_rl" / "evaluation" / "experiments"


def _spec() -> ResolvedCandidateRunSpec:
    config = CandidateRunConfig(
        signal_name="signal",
        feature_names=("signal", "volatility"),
        fit_symbol_names=("BTCUSDT", "ETHUSDT"),
        fit_cutoff=np.datetime64("2026-01-01", "ns"),
        evaluation_start=np.datetime64("2026-02-01", "ns"),
        evaluation_stop_exclusive=np.datetime64("2026-03-01", "ns"),
        rule_entry_threshold=1.0,
        rule_exit_threshold=0.5,
        forecast_entry_threshold=0.1,
        forecast_exit_threshold=0.05,
        ppo_total_timesteps=8,
        ppo_seed=2,
        gross_budget=0.5,
        initial_capital=100_000.0,
    )
    lean = LeanCandidateConfig(
        signal_index=0,
        feature_indices=(0, 1),
        fit_symbol_indices=(0, 1),
        fit_cutoff=config.fit_cutoff,
        rule_entry_threshold=config.rule_entry_threshold,
        rule_exit_threshold=config.rule_exit_threshold,
        forecast_entry_threshold=config.forecast_entry_threshold,
        forecast_exit_threshold=config.forecast_exit_threshold,
        ppo_total_timesteps=config.ppo_total_timesteps,
        ppo_seed=config.ppo_seed,
    )
    return ResolvedCandidateRunSpec(
        dataset_id="a" * 64,
        dataset_artifact_schema="market_dataset_artifact_v3",
        dataset_artifact_digest="b" * 64,
        config=config,
        lean_config=lean,
        evaluation_start_index=10,
        evaluation_stop_index=20,
    )


def _resolved_constructor_calls(path: Path) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ResolvedRunConfig"
    )


def test_resolved_run_config_owns_candidate_spec_conversion() -> None:
    resolved = ResolvedRunConfig.from_candidate_spec(_spec())

    assert resolved.to_payload() == {
        "schema_version": "resolved_run_config_v1",
        "signal_name": "signal",
        "signal_index": 0,
        "feature_names": ["signal", "volatility"],
        "feature_indices": [0, 1],
        "fit_symbol_names": ["BTCUSDT", "ETHUSDT"],
        "fit_symbol_indices": [0, 1],
        "fit_cutoff": "2026-01-01T00:00:00.000000000",
        "rule_entry_threshold": 1.0,
        "rule_exit_threshold": 0.5,
        "forecast_entry_threshold": 0.1,
        "forecast_exit_threshold": 0.05,
        "ppo_total_timesteps": 8,
        "ppo_seed": 2,
        "evaluation_start": "2026-02-01T00:00:00.000000000",
        "evaluation_stop_exclusive": "2026-03-01T00:00:00.000000000",
        "gross_budget": 0.5,
        "initial_capital": 100_000.0,
        "execution_overlay": "zero_overlay_dataset_fields_authoritative",
    }


def test_candidate_spec_conversion_has_one_production_authority() -> None:
    assert _resolved_constructor_calls(EXPERIMENTS / "evidence.py") == 0
    assert _resolved_constructor_calls(EXPERIMENTS / "codec.py") == 1


def test_study_plan_owns_fixed_resolved_field_roster() -> None:
    assert StudyPlan.FIXED_RESOLVED_FIELDS == (
        "fit_cutoff",
        "evaluation_start",
        "evaluation_stop_exclusive",
        "initial_capital",
        "execution_overlay",
    )

    for name in ("evidence.py", "workflow.py", "delta.py"):
        source = (EXPERIMENTS / name).read_text(encoding="utf-8")
        assert "plan.FIXED_RESOLVED_FIELDS" in source
        assert "_STUDY_FIXED_FIELDS" not in source
        assert "_STUDY_FIXED_CONFIG_PATHS" not in source
