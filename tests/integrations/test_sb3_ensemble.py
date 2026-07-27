from __future__ import annotations

import importlib
from types import ModuleType

import numpy as np
import pytest


def _ensemble_module() -> ModuleType:
    return importlib.import_module("trade_rl.integrations.sb3_ensemble")


class _Model:
    def __init__(self, action: object, *, error: Exception | None = None) -> None:
        self.action = action
        self.error = error
        self.deterministic_values: list[bool] = []

    def predict(self, observation: object, deterministic: bool = False):
        self.deterministic_values.append(deterministic)
        if self.error is not None:
            raise self.error
        return self.action, None


def test_deterministic_mean_uses_float64_accumulation_and_returns_float32() -> None:
    module = _ensemble_module()
    models = (
        _Model(np.array([0.2, -0.2], dtype=np.float32)),
        _Model(np.array([0.4, 0.6], dtype=np.float32)),
    )

    result = module.predict_deterministic_mean_action(
        models,
        observation={"current": np.ones(1, dtype=np.float32)},
        action_size=2,
    )

    assert result.dtype == np.float32
    np.testing.assert_allclose(result, np.array([0.3, 0.2], dtype=np.float32))
    assert [model.deterministic_values for model in models] == [[True], [True]]


def test_empty_ensemble_is_rejected() -> None:
    module = _ensemble_module()

    with pytest.raises(ValueError, match="at least one|empty"):
        module.predict_deterministic_mean_action((), observation=object())


def test_member_prediction_failure_is_wrapped_with_index() -> None:
    module = _ensemble_module()
    models = (_Model([0.0]), _Model([0.0], error=RuntimeError("boom")))

    with pytest.raises(ValueError, match="member 1.*prediction") as caught:
        module.predict_deterministic_mean_action(models, observation=object())

    assert isinstance(caught.value.__cause__, RuntimeError)


@pytest.mark.parametrize(
    ("action", "message"),
    (
        ([0.0, np.nan], "finite"),
        ([0.0, 1.1], "bound|range"),
    ),
)
def test_invalid_member_action_is_rejected(action: object, message: str) -> None:
    module = _ensemble_module()

    with pytest.raises(ValueError, match=message):
        module.predict_deterministic_mean_action(
            (_Model(action),),
            observation=object(),
            action_size=2,
        )


def test_declared_action_size_is_enforced() -> None:
    module = _ensemble_module()

    with pytest.raises(ValueError, match="shape|size"):
        module.predict_deterministic_mean_action(
            (_Model([0.0, 0.0, 0.0]),),
            observation=object(),
            action_size=2,
        )


def test_member_shapes_must_agree_when_size_is_inferred() -> None:
    module = _ensemble_module()

    with pytest.raises(ValueError, match="shape.*disagree|same shape"):
        module.predict_deterministic_mean_action(
            (_Model([0.0]), _Model([0.0, 0.0])),
            observation=object(),
        )


def test_context_is_included_in_fail_closed_errors() -> None:
    module = _ensemble_module()

    with pytest.raises(ValueError, match="walk-forward ensemble.*member 0"):
        module.predict_deterministic_mean_action(
            (_Model([2.0]),),
            observation=object(),
            context="walk-forward ensemble",
        )
