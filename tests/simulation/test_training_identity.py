"""Public training config identity is the frozen transform's full config digest."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, fields, replace

import pytest

from trade_rl.risk.pretrade import PreTradeRiskConfig
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.simulation.training_identity import (
    training_execution_config_digest,
    training_execution_config_payload,
    training_risk_config_digest,
    training_risk_config_payload,
)


def _literal_digest(payload):
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def test_training_execution_identity_retains_old_literal_zero_digest():
    cost = ExecutionCostConfig.zero()
    payload = training_execution_config_payload(cost)
    assert payload == asdict(cost)
    assert set(payload) == {field.name for field in fields(cost)}
    assert "schema_version" not in payload
    # The zero config now defaults to causal previous-bar capacity.
    assert training_execution_config_digest(cost) == (
        "89907eb0fe96c086386b5d19454a6af6008ce8e931e035b822b912bd154954fa"
    )
    assert cost.execution_policy_digest == (
        "78cf8bef738cca451ff716fb8a644d94027bd93195c2110d556cf7e3059ffaf2"
    )
    assert training_execution_config_digest(cost) != cost.execution_policy_digest
    # The literal digest rule is unchanged for the explicit same-bar policy.
    same_bar = replace(cost, processing_bar_volume_capacity=True)
    assert training_execution_config_digest(same_bar) == (
        "e7180ee4fdda175e579200e3c80145bbc11131d8c7da8cc8e8d24827dbcd2ec8"
    )
    assert same_bar.execution_policy_digest == (
        "5937ffc05c71f2f8410b218db7a7026bc174892b279110a53f8bd94d259467f3"
    )


@pytest.mark.parametrize(
    "config",
    [
        ExecutionCostConfig(),
        replace(
            ExecutionCostConfig.zero(),
            fee_rate=0.003,
            order_latency_bars=1,
            allow_short=False,
        ),
        replace(
            ExecutionCostConfig(),
            trigger_volume_fractions=(1.0, 0.8, 0.6, 0.1),
            random_seed=12,
            path_mode="optimistic",
        ),
    ],
)
def test_complete_execution_config_identity_matches_independent_canonical_oracle(
    config,
):
    expected = asdict(config)
    assert training_execution_config_payload(config) == expected
    assert training_execution_config_digest(config) == _literal_digest(expected)
    payload = training_execution_config_payload(config)
    payload["random_seed"] = 99
    assert training_execution_config_payload(config) == expected


@pytest.mark.parametrize(
    "config",
    [
        PreTradeRiskConfig(),
        PreTradeRiskConfig(max_gross=1.5, max_abs_weight=0.8, max_turnover=None),
        PreTradeRiskConfig(
            drawdown_start=0.2, drawdown_stop=0.5, emergency_turnover_override=False
        ),
    ],
)
def test_risk_identity_matches_complete_training_config(config):
    assert training_risk_config_payload(config) == asdict(config)
    assert training_risk_config_digest(config) == _literal_digest(asdict(config))


def test_wrong_config_types_and_subclasses_fail_closed():
    @dataclass(frozen=True, slots=True)
    class Extended(ExecutionCostConfig):
        unreviewed: int = 42

    for wrong in ({}, PreTradeRiskConfig(), Extended()):
        with pytest.raises(TypeError, match="ExecutionCostConfig"):
            training_execution_config_digest(wrong)
    for wrong in ({}, ExecutionCostConfig()):
        with pytest.raises(TypeError, match="PreTradeRiskConfig"):
            training_risk_config_payload(wrong)


def test_public_training_identity_import_needs_no_optional_runtime():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; "
            "from trade_rl.simulation.training_identity import training_execution_config_digest; "
            "from trade_rl.data.timestamps import utc_timestamp_ns; "
            "from trade_rl.simulation import ExecutionCostConfig; "
            "assert training_execution_config_digest(ExecutionCostConfig.zero()); "
            "assert not any(n.split('.')[0] in {'gymnasium','stable_baselines3','torch','lightgbm'} "
            "for n in sys.modules)",
        ],
        check=True,
    )


def test_public_helpers_match_actual_legacy_transform_bytes():
    from datetime import UTC, datetime, timedelta, timezone

    from trade_rl.artifacts.canonical import canonical_json_bytes
    from trade_rl.artifacts.hashing import content_digest
    from trade_rl.data.timestamps import utc_timestamp_ns
    from trade_rl.strategies.rl.portfolio_transform import (
        _datetime_ns,
        _execution_payload,
        _risk_payload,
    )

    for cost in (
        ExecutionCostConfig(),
        ExecutionCostConfig.zero(),
        replace(ExecutionCostConfig(), fee_rate=0.003, order_latency_bars=1),
    ):
        assert canonical_json_bytes(training_execution_config_payload(cost)) == (
            canonical_json_bytes(_execution_payload(cost))
        )
        assert training_execution_config_digest(cost) == content_digest(
            _execution_payload(cost)
        )
    for risk in (
        PreTradeRiskConfig(),
        PreTradeRiskConfig(max_gross=2, max_abs_weight=0.8, max_turnover=None),
    ):
        assert canonical_json_bytes(training_risk_config_payload(risk)) == (
            canonical_json_bytes(_risk_payload(risk))
        )
        assert training_risk_config_digest(risk) == content_digest(_risk_payload(risk))
    for instant in (
        datetime(1969, 12, 31, 23, 59, 59, 999999, tzinfo=UTC),
        datetime(2024, 2, 29, 9, tzinfo=timezone(timedelta(hours=9))),
        datetime(2026, 11, 25, 15, tzinfo=UTC),
    ):
        assert utc_timestamp_ns(instant) == _datetime_ns(instant)
