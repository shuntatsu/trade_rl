"""Reject invalid explicit protocol before optional backend import."""

from dataclasses import replace

import pytest

from tests.evaluation.test_allocation_rl_env import parameters
from tests.evaluation.test_allocation_rl_observation_v2 import opt_in
from tests.strategies.test_allocation_protocol_receipt import protocol
from trade_rl.evaluation.rl_allocation import training
from trade_rl.evaluation.rl_allocation.env import AllocationTradingEnv


@pytest.mark.parametrize(
    "change", [{"n_steps": 2, "batch_size": 2}, {"gae_lambda": 0.5}]
)
def test_protocol_clock_mismatch_is_rejected_before_backend_import(monkeypatch, change):
    env = AllocationTradingEnv(**opt_in(parameters()))
    monkeypatch.setattr(
        training.importlib, "import_module", lambda _: pytest.fail("backend imported")
    )
    with pytest.raises(ValueError, match="clock"):
        training.build_allocation_ppo(
            env,
            training_protocol=protocol(**({"n_steps": 4, "batch_size": 2} | change)),
        )


def test_protocol_requires_v2_before_backend_import(monkeypatch):
    env = AllocationTradingEnv(**parameters())
    monkeypatch.setattr(
        training.importlib, "import_module", lambda _: pytest.fail("backend imported")
    )
    with pytest.raises(ValueError, match="v2"):
        training.build_allocation_ppo(
            env, training_protocol=protocol(n_steps=4, batch_size=2)
        )


def protocol_env(*, steps=2):
    args = opt_in(parameters(stop=10))
    args["bound"] = replace(
        args["bound"], clock=replace(args["bound"].clock, rollout_steps=steps)
    )
    return AllocationTradingEnv(**args)
