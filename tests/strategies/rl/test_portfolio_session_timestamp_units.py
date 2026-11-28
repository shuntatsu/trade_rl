"""Declared session Gym and forward binding preserve UTC clock units (#1357)."""

from dataclasses import replace

import numpy as np
import pytest

from tests.strategies.rl.test_portfolio_session_elapsed import _bound, _cost, _risk
from tests.strategies.rl.test_portfolio_session_elapsed import _dataset as _spot_dataset
from tests.strategies.rl.test_portfolio_session_linear_carry import (
    _dataset as _signed_dataset,
)
from trade_rl.strategies.rl.portfolio_env import (
    ACTION_FLAT,
    ACTION_HOLD,
    ACTION_LONG,
    ACTION_SHORT,
    SharedPortfolioGymEnv,
)
from trade_rl.strategies.rl.portfolio_transform import (
    PortfolioObservationRuntimeBinding,
    fit_portfolio_observation_transform,
)


@pytest.mark.parametrize("unit", ["ns", "ms", "s", "h"])
@pytest.mark.parametrize("signed", [False, True])
def test_session_gym_and_forward_binding_use_actual_utc_boundaries(unit, signed):
    reference = _signed_dataset() if signed else _spot_dataset()
    dataset = replace(
        reference,
        timestamps=reference.timestamps.astype(f"datetime64[{unit}]"),
        identity_payload_json=None,
    ).with_content_identity()
    assert dataset.identity_verified
    assert dataset.timestamps.dtype == np.dtype(f"datetime64[{unit}]")

    def make_env(data, start, stop):
        cost = (
            _cost(allow_short=True, borrow_rate_multiplier=1.0) if signed else _cost()
        )
        return SharedPortfolioGymEnv(
            data,
            start_index=start,
            stop_index=stop,
            bound=_bound(reference, start, stop),
            target_abs_weights=(0.25, 0.25),
            execution_cost=cost,
            risk_config=_risk(),
            native_session_linear_carry=signed,
        )

    actual = make_env(dataset, 0, 5)
    transform = fit_portfolio_observation_transform(actual)
    assert transform.session_boundaries_ns == actual.bound.clock.boundaries_ns
    forward = make_env(dataset, 5, 10)
    binding = PortfolioObservationRuntimeBinding.create(
        transform, actual.bound, forward
    )
    assert binding.session_boundaries_ns == forward.bound.clock.boundaries_ns

    def rewards(env):
        env.reset(seed=42)
        _, reward, done, truncated, info = env.step(
            np.array([ACTION_LONG, ACTION_SHORT if signed else ACTION_FLAT])
        )
        result = [reward]
        while not done:
            assert not truncated
            _, reward, done, truncated, info = env.step(
                np.array([ACTION_HOLD, ACTION_HOLD])
            )
            result.append(reward)
        assert not truncated and info["objective_endpoint_valid"]
        return result

    assert rewards(actual) == pytest.approx(rewards(make_env(reference, 0, 5)))
