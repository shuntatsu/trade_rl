from __future__ import annotations

from types import SimpleNamespace

import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.rl.actions import (
    ACTION_SCHEMA,
    ActionMode,
    ActionSpec,
    ActionValidationMode,
)


def _training_config(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "observation_encoder": "hierarchical_sequence_v2",
        "behavior_cloning_epochs": 2,
        "behavior_cloning_teacher": "oracle",
        "gamma": 1.0,
        "digest_payload": lambda: {"training": "v1"},
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _run_config(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "training": _training_config(),
        "action": ActionSpec(
            mode=ActionMode.TARGET_WEIGHT,
            target_weight_count=1,
            alpha_enabled=False,
            n_factors=0,
            risk_tilt_enabled=False,
            validation_mode=ActionValidationMode.FAIL_CLOSED,
        ),
        "environment": SimpleNamespace(
            structured_sequence_observation=True,
            finite_horizon_observation=True,
            initial_capital=10_000.0,
        ),
        "alpha_artifact": None,
        "factor_artifact": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_validate_universal_training_config_requires_generic_target_weight_surface() -> (
    None
):
    from trade_rl.workflows.universal_training_runner import (
        validate_universal_training_config,
    )

    validate_universal_training_config(_run_config())

    with pytest.raises(ValueError, match="target-weight"):
        validate_universal_training_config(
            _run_config(
                action=ActionSpec(
                    mode=ActionMode.RESIDUAL,
                    validation_mode=ActionValidationMode.FAIL_CLOSED,
                )
            )
        )

    with pytest.raises(ValueError, match="hierarchical_sequence_v2"):
        validate_universal_training_config(
            _run_config(training=_training_config(observation_encoder="asset_set_v1"))
        )


def test_concrete_action_spec_digest_is_symbol_specific_but_deterministic() -> None:
    from trade_rl.workflows.universal_training_runner import (
        concrete_action_spec_digest,
    )

    action = _run_config().action
    first = concrete_action_spec_digest(action, "AAAUSDT")
    second = concrete_action_spec_digest(action, "AAAUSDT")
    other = concrete_action_spec_digest(action, "BBBUSDT")

    assert first == second
    assert first != other
    assert len(first) == 64
    assert first == content_digest(
        {
            "schema_version": ACTION_SCHEMA,
            "alpha_enabled": action.alpha_enabled,
            "mode": "target_weight",
            "risk_tilt_enabled": action.risk_tilt_enabled,
            "n_factors": action.n_factors,
            "names": ("target_weight:AAAUSDT",),
            "residual_scale": action.residual_scale,
            "target_weight_count": action.target_weight_count,
            "validation_mode": action.validation_mode.value,
        }
    )
