from __future__ import annotations

from collections.abc import Callable

import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.experiments.codec import (
    _candidate_config_from_resolved,
    _candidate_config_payload,
    _definition_from_payload,
    _resolved_from_payload,
)
from trade_rl.evaluation.experiments.contracts import ControlledFactor, StudyPlan
from trade_rl.evaluation.experiments.errors import ArtifactIntegrityError
from trade_rl.evaluation.experiments.evidence import _check_study_fixed_config
from trade_rl.risk import PreTradeRiskConfig
from trade_rl.strategies.rl.ppo import (
    PPO_GLOBAL_FEATURE_NAMES,
    PPO_OBSERVATION_SCHEMA,
    PPO_OBSERVATION_SCHEMA_V3,
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


def _v3_payload() -> dict[str, object]:
    payload = _v2_payload()
    payload["schema_version"] = "resolved_run_config_v3"
    payload["ppo_training_layout"] = "sequential"
    payload["ppo_rollout_steps_per_env"] = None
    return payload


def _v4_payload() -> dict[str, object]:
    payload = _v3_payload()
    payload["schema_version"] = "resolved_run_config_v4"
    payload["ppo_observation_schema"] = "ppo_observation_v3"
    payload["ppo_minimum_hold_bars"] = 168
    payload["ppo_settle_terminal_position"] = True
    return payload


def _v5_payload() -> dict[str, object]:
    payload = _v4_payload()
    payload["schema_version"] = "resolved_run_config_v5"
    payload["pretrade_risk_config"] = {
        "max_gross": 0.5,
        "max_abs_weight": 0.1,
        "max_turnover": None,
        "drawdown_start": 0.1,
        "drawdown_stop": 0.2,
        "emergency_turnover_override": True,
        "fail_closed_tolerance": 1e-10,
    }
    return payload


def _v7_payload(*, forecast_switch_cost: float | None = None) -> dict[str, object]:
    payload = _v5_payload()
    payload["schema_version"] = "resolved_run_config_v7"
    payload["forecast_switch_cost"] = forecast_switch_cost
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


def test_resolved_run_v3_round_trips_default_training_layout_fields() -> None:
    payload = _v3_payload()

    resolved = _resolved_from_payload(payload, field="current")

    assert resolved.to_payload() == payload
    assert resolved.ppo_training_layout == "sequential"
    assert resolved.ppo_rollout_steps_per_env is None


def test_resolved_run_v4_binds_holding_duration_and_terminal_settlement() -> None:
    payload = _v4_payload()

    resolved = _resolved_from_payload(payload, field="current")
    changed_horizon = dict(payload, ppo_minimum_hold_bars=336)

    assert resolved.to_payload() == payload
    assert resolved.ppo_observation_schema == "ppo_observation_v3"
    assert resolved.ppo_minimum_hold_bars == 168
    assert resolved.ppo_settle_terminal_position is True
    assert (
        resolved.digest
        != _resolved_from_payload(changed_horizon, field="changed").digest
    )


def test_experiment_definition_v5_reconstructs_exact_candidate_semantics() -> None:
    resolved = _resolved_from_payload(_v5_payload(), field="candidate_config")
    candidate = _candidate_config_from_resolved(resolved)
    requested_digest = content_digest(
        _candidate_config_payload(
            candidate,
            resolved_schema_version=resolved.schema_version,
        )
    )
    payload = {
        "schema_version": "controlled_experiment_definition_v1",
        "study_digest": "a" * 64,
        "sequence": 1,
        "hypothesis": "A longer PPO minimum hold improves net returns.",
        "baseline_evidence_digest": "b" * 64,
        "factor": ControlledFactor.PPO_MINIMUM_HOLD.value,
        "candidate_requested_config_digest": requested_digest,
        "candidate_config": resolved.to_payload(),
    }

    definition = _definition_from_payload(payload)

    assert definition.candidate_config.to_payload() == resolved.to_payload()
    assert definition.candidate_config.ppo_minimum_hold_bars == 168
    assert definition.candidate_config.ppo_observation_schema == "ppo_observation_v3"
    assert definition.candidate_config.ppo_settle_terminal_position is True
    assert (
        definition.candidate_config.pretrade_risk_config
        == resolved.pretrade_risk_config
    )


@pytest.mark.parametrize(
    ("resolved_config_payload", "has_holding_fields"),
    [
        pytest.param(_v1_payload, False, id="v1"),
        pytest.param(_v2_payload, False, id="v2"),
        pytest.param(_v3_payload, False, id="v3"),
        pytest.param(_v4_payload, True, id="v4"),
    ],
)
def test_legacy_experiment_definition_keeps_requested_config_digest(
    resolved_config_payload: Callable[[], dict[str, object]],
    has_holding_fields: bool,
) -> None:
    resolved = _resolved_from_payload(
        resolved_config_payload(), field="candidate_config"
    )
    legacy_candidate_payload = {
        "signal_name": resolved.signal_name,
        "feature_names": list(resolved.feature_names),
        "fit_symbol_names": list(resolved.fit_symbol_names),
        "fit_cutoff": resolved.fit_cutoff,
        "evaluation_start": resolved.evaluation_start,
        "evaluation_stop_exclusive": resolved.evaluation_stop_exclusive,
        "rule_entry_threshold": resolved.rule_entry_threshold,
        "rule_exit_threshold": resolved.rule_exit_threshold,
        "forecast_entry_threshold": resolved.forecast_entry_threshold,
        "forecast_exit_threshold": resolved.forecast_exit_threshold,
        "ppo_total_timesteps": resolved.ppo_total_timesteps,
        "ppo_seed": resolved.ppo_seed,
        "gross_budget": resolved.gross_budget,
        "initial_capital": resolved.initial_capital,
    }
    if has_holding_fields:
        legacy_candidate_payload.update(
            {
                "ppo_minimum_hold_bars": resolved.ppo_minimum_hold_bars,
                "ppo_observation_schema": resolved.ppo_observation_schema,
                "ppo_settle_terminal_position": (resolved.ppo_settle_terminal_position),
            }
        )
    legacy_digest = content_digest(legacy_candidate_payload)
    payload = {
        "schema_version": "controlled_experiment_definition_v1",
        "study_digest": "a" * 64,
        "sequence": 1,
        "hypothesis": "The frozen feature set improves net returns.",
        "baseline_evidence_digest": "b" * 64,
        "factor": ControlledFactor.FEATURE_SET.value,
        "candidate_requested_config_digest": legacy_digest,
        "candidate_config": resolved.to_payload(),
    }

    definition = _definition_from_payload(payload)

    assert definition.candidate_config.to_payload() == resolved.to_payload()


def test_resolved_run_v5_binds_explicit_pretrade_risk() -> None:
    payload = _v5_payload()

    resolved = _resolved_from_payload(payload, field="current")
    changed_risk = dict(
        payload,
        pretrade_risk_config={
            **payload["pretrade_risk_config"],  # type: ignore[misc]
            "drawdown_stop": 0.19,
        },
    )

    assert resolved.to_payload() == payload
    assert resolved.schema_version == "resolved_run_config_v5"
    assert resolved.ppo_minimum_hold_bars == 168
    assert resolved.pretrade_risk_config == PreTradeRiskConfig(
        max_gross=0.5,
        max_abs_weight=0.1,
        max_turnover=None,
        drawdown_start=0.1,
        drawdown_stop=0.2,
    )
    assert (
        resolved.digest != _resolved_from_payload(changed_risk, field="changed").digest
    )


def test_resolved_run_v5_age_aware_contract_requires_risk_and_terminal_close() -> None:
    payload = _v5_payload()
    payload["ppo_observation_schema"] = PPO_OBSERVATION_SCHEMA_V3
    payload["ppo_minimum_hold_bars"] = 168

    without_risk = dict(payload)
    without_risk["pretrade_risk_config"] = None
    with pytest.raises(ValueError, match="missing risk|explicit.*risk"):
        _resolved_from_payload(without_risk, field="missing risk")

    above_drawdown_limit = dict(
        payload,
        pretrade_risk_config={
            **payload["pretrade_risk_config"],  # type: ignore[misc]
            "drawdown_stop": 0.21,
        },
    )
    with pytest.raises(ArtifactIntegrityError, match="resolved-run contract"):
        _resolved_from_payload(above_drawdown_limit, field="risk above limit")

    without_close = dict(payload)
    without_close["ppo_settle_terminal_position"] = False
    with pytest.raises(ValueError, match="missing close|terminal settlement"):
        _resolved_from_payload(without_close, field="missing close")


def test_resolved_run_v2_rejects_tampered_global_observation_roster() -> None:
    payload = _v2_payload()
    payload["ppo_global_feature_names"] = ["market_return_mean"]

    with pytest.raises(
        ArtifactIntegrityError, match="observation|resolved-run contract"
    ):
        _resolved_from_payload(payload, field="current")


def test_historical_v1_study_rejects_new_v2_evidence_before_execution() -> None:
    legacy = _resolved_from_payload(_v1_payload(), field="legacy")
    current = _resolved_from_payload(_v2_payload(), field="current")
    plan = _study_plan(legacy)

    with pytest.raises(ArtifactIntegrityError, match="Study-fixed"):
        _check_study_fixed_config(plan, current)


def test_resolved_run_v7_round_trips_switch_cost_identity() -> None:
    payload = _v7_payload(forecast_switch_cost=0.0007)

    resolved = _resolved_from_payload(payload, field="current")
    candidate = _candidate_config_from_resolved(resolved)
    requested = _candidate_config_payload(
        candidate,
        resolved_schema_version=resolved.schema_version,
    )
    disabled = _resolved_from_payload(_v7_payload(), field="disabled")

    assert resolved.to_payload() == payload
    assert resolved.forecast_switch_cost == pytest.approx(0.0007)
    assert requested["forecast_switch_cost"] == pytest.approx(0.0007)
    assert _study_plan(resolved).digest != _study_plan(disabled).digest


def test_resolved_run_v7_round_trips_disabled_switch_cost_identity() -> None:
    payload = _v7_payload()
    resolved = _resolved_from_payload(payload, field="disabled")

    assert resolved.to_payload() == payload
    assert resolved.forecast_switch_cost is None


def test_resolved_run_v7_preserves_interleaved_ppo_training_contract() -> None:
    payload = _v7_payload(forecast_switch_cost=0.0007)
    payload["ppo_training_layout"] = "interleaved"
    payload["ppo_rollout_steps_per_env"] = 16

    resolved = _resolved_from_payload(payload, field="interleaved")
    candidate = _candidate_config_from_resolved(resolved)
    requested = _candidate_config_payload(
        candidate,
        resolved_schema_version=resolved.schema_version,
    )

    assert resolved.to_payload() == payload
    assert resolved.ppo_training_layout == "interleaved"
    assert resolved.ppo_rollout_steps_per_env == 16
    assert requested["ppo_training_layout"] == "interleaved"
    assert requested["ppo_rollout_steps_per_env"] == 16


def test_resolved_run_v5_rejects_forecast_switch_cost_field() -> None:
    payload = _v5_payload()
    payload["forecast_switch_cost"] = 0.0007

    with pytest.raises(ArtifactIntegrityError):
        _resolved_from_payload(payload, field="legacy")


@pytest.mark.parametrize(
    "switch_cost",
    (True, -0.001, float("nan"), float("inf"), "0.0007"),
)
def test_resolved_run_v7_rejects_invalid_switch_cost(switch_cost: object) -> None:
    payload = _v7_payload()
    payload["forecast_switch_cost"] = switch_cost

    with pytest.raises(ArtifactIntegrityError):
        _resolved_from_payload(payload, field="invalid")
