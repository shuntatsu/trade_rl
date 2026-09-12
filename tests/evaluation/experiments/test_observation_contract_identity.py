from __future__ import annotations

import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.experiments.codec import _resolved_from_payload
from trade_rl.evaluation.experiments.contracts import ControlledFactor, StudyPlan
from trade_rl.evaluation.experiments.errors import ArtifactIntegrityError
from trade_rl.strategies.rl.ppo import (
    PPO_GLOBAL_FEATURE_NAMES,
    PPO_OBSERVATION_SCHEMA,
)


def _v1_payload() -> dict[str, object]:
    return {
        "schema_version": "resolved_run_config_v1",
        "signal_name": "signal",
        "signal_index": 0,
        "feature_names": ["signal", "volatility"],
        "feature_indices": [0, 1],
        "fit_symbol_names": ["BTCUSDT", "ETHUSDT"],
        "fit_symbol_indices": [0, 1],
        "fit_cutoff": "2026-01-01T00:00:00.000000000",
        "rule_entry_threshold": 0.1,
        "rule_exit_threshold": 0.02,
        "forecast_entry_threshold": 0.01,
        "forecast_exit_threshold": 0.002,
        "ppo_total_timesteps": 256,
        "ppo_seed": 2,
        "evaluation_start": "2026-02-01T00:00:00.000000000",
        "evaluation_stop_exclusive": "2026-03-01T00:00:00.000000000",
        "gross_budget": 0.5,
        "initial_capital": 100_000.0,
        "execution_overlay": "zero_overlay_dataset_fields_authoritative",
    }


def _v2_payload() -> dict[str, object]:
    payload = _v1_payload()
    payload["schema_version"] = "resolved_run_config_v2"
    payload["ppo_observation_schema"] = PPO_OBSERVATION_SCHEMA
    payload["ppo_global_feature_names"] = list(PPO_GLOBAL_FEATURE_NAMES)
    return payload


def _study_plan(baseline_config) -> StudyPlan:
    return StudyPlan(
        research_question="Does the frozen M2 baseline generalize?",
        dataset_id="a" * 64,
        dataset_artifact_schema="market_dataset_artifact_v3",
        dataset_artifact_digest="b" * 64,
        symbols=("BTCUSDT", "ETHUSDT"),
        baseline_config=baseline_config,
        ppo_seeds=(2, 5),
        allowed_factors=(ControlledFactor.FEATURE_SET,),
        max_experiments=4,
        n_bootstrap=1_000,
        bootstrap_seed=7,
        implementation_digest="c" * 64,
        runtime_environment_digest="d" * 64,
    )


def test_historical_resolved_run_v1_round_trips_without_v2_fields() -> None:
    payload = _v1_payload()

    resolved = _resolved_from_payload(payload, field="legacy")

    assert resolved.to_payload() == payload
    assert resolved.schema_version == "resolved_run_config_v1"
    assert "ppo_observation_schema" not in resolved.to_payload()
    assert "ppo_global_feature_names" not in resolved.to_payload()


def test_resolved_run_v2_round_trips_and_study_digest_binds_observation() -> None:
    payload = _v2_payload()

    resolved = _resolved_from_payload(payload, field="current")
    plan = _study_plan(resolved)

    assert resolved.to_payload() == payload
    assert plan.to_payload()["baseline_config"] == payload
    assert plan.digest == content_digest(plan.to_payload())


def test_resolved_run_v2_rejects_tampered_global_observation_roster() -> None:
    payload = _v2_payload()
    payload["ppo_global_feature_names"] = ["market_return_mean"]

    with pytest.raises(
        ArtifactIntegrityError, match="observation|resolved-run contract"
    ):
        _resolved_from_payload(payload, field="current")
