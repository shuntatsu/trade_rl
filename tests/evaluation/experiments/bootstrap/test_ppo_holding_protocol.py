from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from tests.evaluation.experiments.bootstrap.test_execution_economics_fullpath import (
    _profile,
)
from tests.evaluation.experiments.bootstrap.test_research_context_config import (
    _context_payload,
    _install_v4_fakes,
    _workflow_v4_payload,
)
from trade_rl.evaluation.experiments import inspect_study
from trade_rl.evaluation.experiments.bootstrap import workflow as bootstrap_workflow
from trade_rl.evaluation.experiments.bootstrap.config import (
    load_canonical_m2_bootstrap_config,
)
from trade_rl.evaluation.experiments.bootstrap.workflow import (
    bootstrap_canonical_m2_study,
    inspect_canonical_m2_bootstrap,
)
from trade_rl.evaluation.experiments.contracts import (
    ControlledFactor,
    StudyProtocol,
)
from trade_rl.risk import PreTradeRiskConfig
from trade_rl.strategies.rl.ppo import PPO_OBSERVATION_SCHEMA_V3


def _ppo_holding_v5_payload() -> dict[str, object]:
    payload = _workflow_v4_payload()
    payload["schema_version"] = "canonical_m2_bootstrap_config_v5"
    payload["study_protocol"] = StudyProtocol.PPO_HOLDING_DURATION.value
    payload["ppo_seeds"] = [0, 1, 2, 3, 4]
    payload["allowed_factors"] = [ControlledFactor.PPO_MINIMUM_HOLD.value]
    payload["max_experiments"] = 4
    payload["execution_economics"] = _profile().to_payload()

    baseline = payload["baseline"]
    assert isinstance(baseline, dict)
    from tests.evaluation.experiments.bootstrap.test_binance import _config
    from trade_rl.strategies.rl.ppo_training import PPO_TRAINING_LAYOUT_SEQUENTIAL

    candidate = replace(
        _config().baseline,
        ppo_observation_schema=PPO_OBSERVATION_SCHEMA_V3,
        ppo_minimum_hold_bars=0,
        ppo_settle_terminal_position=True,
        ppo_training_layout=PPO_TRAINING_LAYOUT_SEQUENTIAL,
        ppo_rollout_steps_per_env=None,
        pretrade_risk_config=PreTradeRiskConfig(
            max_gross=0.5,
            max_abs_weight=0.1,
            max_turnover=None,
            drawdown_start=0.1,
            drawdown_stop=0.2,
        ),
    )
    serialized = candidate.to_json_payload()
    for name in (
        "ppo_observation_schema",
        "ppo_minimum_hold_bars",
        "ppo_settle_terminal_position",
        "pretrade_risk_config",
    ):
        baseline[name] = serialized[name]
    context = _context_payload()
    evidence = context["consumed_evidence"]
    assert isinstance(evidence, list)
    first = evidence[0]
    assert isinstance(first, dict)
    first["development_stop_exclusive"] = "2024-02-01T00:00:00.000000000"
    payload["research_context"] = context
    return payload


def _ppo_shared_cash_v6_payload() -> dict[str, object]:
    payload = _ppo_holding_v5_payload()
    payload["schema_version"] = "canonical_m2_bootstrap_config_v6"
    payload["study_protocol"] = StudyProtocol.PPO_SHARED_CASH_HOLDING_DURATION.value
    return payload


def _ppo_shared_cash_v7_payload() -> dict[str, object]:
    payload = _ppo_shared_cash_v6_payload()
    payload["schema_version"] = "canonical_m2_bootstrap_config_v7"
    payload["study_protocol"] = StudyProtocol.PPO_SHARED_CASH_HOLDING_DURATION_V3.value
    return payload


def _write_config(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "ppo-holding-bootstrap.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_v5_config_round_trips_protocol_and_complete_ppo_baseline(
    tmp_path: Path,
) -> None:
    payload = _ppo_holding_v5_payload()

    config = load_canonical_m2_bootstrap_config(_write_config(tmp_path, payload))

    assert config.schema_version == "canonical_m2_bootstrap_config_v5"
    assert config.study_protocol is StudyProtocol.PPO_HOLDING_DURATION
    assert config.allowed_factors == (ControlledFactor.PPO_MINIMUM_HOLD,)
    assert config.ppo_seeds == (0, 1, 2, 3, 4)
    assert config.baseline.ppo_observation_schema == PPO_OBSERVATION_SCHEMA_V3
    assert config.baseline.ppo_minimum_hold_bars == 0
    assert config.baseline.ppo_settle_terminal_position is True
    assert config.baseline.pretrade_risk_config == PreTradeRiskConfig(
        max_gross=0.5,
        max_abs_weight=0.1,
        max_turnover=None,
        drawdown_start=0.1,
        drawdown_stop=0.2,
    )
    assert config.to_payload() == payload


def test_v6_config_round_trips_shared_cash_protocol_and_complete_ppo_baseline(
    tmp_path: Path,
) -> None:
    payload = _ppo_shared_cash_v6_payload()

    config = load_canonical_m2_bootstrap_config(_write_config(tmp_path, payload))

    assert config.schema_version == "canonical_m2_bootstrap_config_v6"
    assert config.study_protocol is StudyProtocol.PPO_SHARED_CASH_HOLDING_DURATION
    assert config.allowed_factors == (ControlledFactor.PPO_MINIMUM_HOLD,)
    assert config.ppo_seeds == (0, 1, 2, 3, 4)
    assert config.baseline.ppo_observation_schema == PPO_OBSERVATION_SCHEMA_V3
    assert config.baseline.ppo_minimum_hold_bars == 0
    assert config.baseline.ppo_settle_terminal_position is True
    assert config.to_payload() == payload


def test_v6_shared_cash_protocol_rejects_unregistered_initial_capital(
    tmp_path: Path,
) -> None:
    payload = _ppo_shared_cash_v6_payload()
    baseline = payload["baseline"]
    assert isinstance(baseline, dict)
    baseline["initial_capital"] = 99_999.0

    with pytest.raises(ValueError, match="shared-cash.*capital|capital.*100,000"):
        load_canonical_m2_bootstrap_config(_write_config(tmp_path, payload))


def test_v6_config_rejects_the_independent_account_protocol(
    tmp_path: Path,
) -> None:
    payload = _ppo_shared_cash_v6_payload()
    payload["study_protocol"] = StudyProtocol.PPO_HOLDING_DURATION.value

    with pytest.raises(ValueError, match="requires its matching PPO protocol"):
        load_canonical_m2_bootstrap_config(_write_config(tmp_path, payload))


def test_v7_config_round_trips_ohlc_stress_protocol(tmp_path: Path) -> None:
    payload = _ppo_shared_cash_v7_payload()

    config = load_canonical_m2_bootstrap_config(_write_config(tmp_path, payload))

    assert config.schema_version == "canonical_m2_bootstrap_config_v7"
    assert config.study_protocol is StudyProtocol.PPO_SHARED_CASH_HOLDING_DURATION_V3
    assert config.to_payload() == payload


def test_v7_config_rejects_the_historical_shared_cash_protocol(
    tmp_path: Path,
) -> None:
    payload = _ppo_shared_cash_v7_payload()
    payload["study_protocol"] = StudyProtocol.PPO_SHARED_CASH_HOLDING_DURATION.value

    with pytest.raises(ValueError, match="requires its matching PPO protocol"):
        load_canonical_m2_bootstrap_config(_write_config(tmp_path, payload))


def test_v6_shared_cash_bootstrap_is_read_only(tmp_path: Path) -> None:
    from trade_rl.evaluation.experiments.bootstrap.workflow import (
        bootstrap_canonical_m2_study,
    )

    config_path = _write_config(tmp_path, _ppo_shared_cash_v6_payload())

    with pytest.raises(ValueError, match="historical and read-only"):
        bootstrap_canonical_m2_study(config_path, tmp_path / "old-bootstrap")

    assert not (tmp_path / "old-bootstrap").exists()


def test_v5_rejects_final_start_at_dataset_stop_before_bootstrap(
    tmp_path: Path,
) -> None:
    payload = _ppo_holding_v5_payload()
    payload["final_evaluation_start"] = payload["data_stop_exclusive"]

    with pytest.raises(
        ValueError,
        match="final_evaluation_start must be strictly later than data_stop_exclusive",
    ):
        load_canonical_m2_bootstrap_config(_write_config(tmp_path, payload))


def test_v5_boundary_rejection_precedes_source_and_dataset_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _ppo_holding_v5_payload()
    payload["final_evaluation_start"] = payload["data_stop_exclusive"]
    config_path = _write_config(tmp_path, payload)
    output = tmp_path / "canonical-ppo-holding"

    def fail_if_called(*args: object, **kwargs: object) -> None:
        pytest.fail("invalid config reached source or dataset work")

    monkeypatch.setattr(bootstrap_workflow, "_freeze_binance_source", fail_if_called)
    monkeypatch.setattr(
        bootstrap_workflow, "build_binance_market_dataset", fail_if_called
    )

    with pytest.raises(
        ValueError,
        match="final_evaluation_start must be strictly later than data_stop_exclusive",
    ):
        bootstrap_canonical_m2_study(config_path, output)

    assert not output.exists()


def test_v5_requires_hourly_base_timeframe_before_source_and_dataset_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _ppo_holding_v5_payload()
    payload["base_timeframe"] = "15m"
    config_path = _write_config(tmp_path, payload)
    output = tmp_path / "canonical-ppo-holding"

    def fail_if_called(*args: object, **kwargs: object) -> None:
        pytest.fail("non-hourly config reached source or dataset work")

    monkeypatch.setattr(bootstrap_workflow, "_freeze_binance_source", fail_if_called)
    monkeypatch.setattr(
        bootstrap_workflow, "build_binance_market_dataset", fail_if_called
    )

    with pytest.raises(
        ValueError,
        match="PPO holding-duration protocol requires base_timeframe '1h'",
    ):
        bootstrap_canonical_m2_study(config_path, output)

    assert not output.exists()


def test_v5_baseline_requires_explicit_age_risk_and_settlement_fields(
    tmp_path: Path,
) -> None:
    payload = _ppo_holding_v5_payload()
    baseline = payload["baseline"]
    assert isinstance(baseline, dict)
    del baseline["pretrade_risk_config"]

    with pytest.raises(ValueError, match="baseline keys differ.*missing"):
        load_canonical_m2_bootstrap_config(_write_config(tmp_path, payload))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_gross", 0.6),
        ("max_abs_weight", 0.2),
        ("max_turnover", 0.5),
        ("drawdown_start", 0.05),
        ("drawdown_stop", 0.15),
    ],
)
def test_v5_rejects_deviations_from_the_preregistered_risk_profile(
    tmp_path: Path,
    field: str,
    value: float,
) -> None:
    payload = _ppo_holding_v5_payload()
    baseline = payload["baseline"]
    assert isinstance(baseline, dict)
    risk = baseline["pretrade_risk_config"]
    assert isinstance(risk, dict)
    risk[field] = value

    with pytest.raises(ValueError, match="violates the PPO holding-duration protocol"):
        load_canonical_m2_bootstrap_config(_write_config(tmp_path, payload))


def test_v5_bootstrap_binds_final_eligible_ppo_protocol_before_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_v4_fakes(monkeypatch)
    config_path = _write_config(tmp_path, _ppo_holding_v5_payload())
    output = tmp_path / "canonical-ppo-holding"

    result = bootstrap_canonical_m2_study(config_path, output)

    snapshot = inspect_study(output / "study")
    assert snapshot.baseline is None
    assert snapshot.experiment_sequences == ()
    assert snapshot.plan.schema_version == "controlled_study_plan_v5"
    assert snapshot.plan.protocol is StudyProtocol.PPO_HOLDING_DURATION
    assert snapshot.plan.final_evaluation_start == "2024-03-01T04:00:00.000000000"
    assert (
        snapshot.plan.final_evaluation_stop_exclusive == "2024-04-01T00:00:00.000000000"
    )
    assert snapshot.plan.research_context is not None
    assert snapshot.plan.ppo_seeds == (0, 1, 2, 3, 4)
    assert snapshot.plan.allowed_factors == (ControlledFactor.PPO_MINIMUM_HOLD,)
    assert snapshot.plan.max_experiments == 4
    assert (
        snapshot.plan.baseline_config.ppo_observation_schema
        == PPO_OBSERVATION_SCHEMA_V3
    )
    assert snapshot.plan.baseline_config.ppo_settle_terminal_position is True
    assert snapshot.plan.baseline_config.pretrade_risk_config is not None
    assert snapshot.plan.research_question.endswith(
        "separate one-shot sealed unused-future evaluation for any profitability claim."
    )
    assert inspect_canonical_m2_bootstrap(output) == result


def test_v5_context_cannot_consume_evidence_after_final_start(
    tmp_path: Path,
) -> None:
    payload = _ppo_holding_v5_payload()
    context = deepcopy(payload["research_context"])
    assert isinstance(context, dict)
    evidence = context["consumed_evidence"]
    assert isinstance(evidence, list)
    first = evidence[0]
    assert isinstance(first, dict)
    first["development_stop_exclusive"] = "2024-04-01T00:00:00.000000000"
    payload["research_context"] = context

    with pytest.raises(ValueError, match="consumed development evidence"):
        load_canonical_m2_bootstrap_config(_write_config(tmp_path, payload))


def test_v7_bootstrap_binds_ohlc_stress_selection_before_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_v4_fakes(monkeypatch)
    config_path = _write_config(tmp_path, _ppo_shared_cash_v7_payload())
    output = tmp_path / "canonical-ppo-shared-cash"

    result = bootstrap_canonical_m2_study(config_path, output)

    snapshot = inspect_study(output / "study")
    assert snapshot.baseline is None
    assert snapshot.experiment_sequences == ()
    assert snapshot.plan.schema_version == "controlled_study_plan_v7"
    assert snapshot.plan.protocol is StudyProtocol.PPO_SHARED_CASH_HOLDING_DURATION_V3
    assert snapshot.plan.is_ppo_shared_cash_holding_duration_study
    assert snapshot.plan.final_evaluation_start == "2024-03-01T04:00:00.000000000"
    assert (
        snapshot.plan.final_evaluation_stop_exclusive == "2024-04-01T00:00:00.000000000"
    )
    assert snapshot.plan.research_context is not None
    assert snapshot.plan.ppo_seeds == (0, 1, 2, 3, 4)
    assert snapshot.plan.allowed_factors == (ControlledFactor.PPO_MINIMUM_HOLD,)
    assert snapshot.plan.max_experiments == 4
    assert snapshot.plan.baseline_config.pretrade_risk_config is not None
    assert "one shared-cash portfolio" in snapshot.plan.research_question
    assert inspect_canonical_m2_bootstrap(output) == result
