"""Declared fake policy bytes and no-fit native economics, never research evidence."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np

from tests.evaluation.test_allocation_nonrl_walk_forward import fixture
from tests.strategies.test_allocation_schedule_receipt import manifest_v5
from trade_rl.artifacts import content_digest
from trade_rl.evaluation.rl_allocation.env import AllocationTradingEnv
from trade_rl.simulation import MarketExecutor
from trade_rl.strategies.rl.allocation_artifact import save_allocation_policy
from trade_rl.strategies.rl.allocation_manifest import validate_allocation_manifest
from trade_rl.strategies.rl.allocation_model import AllocationPPOPolicy
from trade_rl.strategies.rl.allocation_training_schedule import (
    AllocationTrainingSchedule,
    AllocationTrainingWindow,
)


def native_pair(*, frozen=False, volume=100000, **cost_changes):
    _, environments = fixture(ranges=((6, 10),), signals=(1, 1, 1, 1), volume=volume)
    base = environments[0]
    if frozen:
        from tests.evaluation.test_allocation_preprocessing_runtime import bind_frozen
        from tests.strategies.test_allocation_preprocessing import declaration

        raw = manifest_v5()
        source = raw["training"]["sources"][0]["source"]
        prefix = declaration()

        def ns(hour):
            return int(np.datetime64(f"2025-12-01T0{hour}:00:00", "ns").astype("int64"))

        prefix = replace(
            prefix,
            normalizer=replace(
                prefix.normalizer,
                source_dataset_id=source["dataset_id"],
                mean=(0.0,),
                usable_counts=((3,),),
            ),
            fit_last_event_time_ns=ns(2),
            fit_as_of_ns=ns(3),
            policy_start_index=6,
            policy_start_time_ns=ns(6),
            admitted_row_indices=((0, 1, 2),),
        )
        args = dict(
            dataset=base.dataset,
            stream=base.stream,
            estimates=tuple(base._estimates.values()),
            bound=base.bound,
            action_contract=base.action_contract,
            allocator=base.allocator,
            execution_cost=base.execution_cost,
            risk_config=base.risk_config,
            feature_indices=base.feature_indices,
            symbol_index=base.symbol_index,
            start_index=6,
            stop_index=10,
            account_id=base.account_id,
            observation_schema=base.observation_schema,
        )
        base = AllocationTradingEnv(**bind_frozen(args, prefix))
    cost = replace(base.execution_cost, fee_rate=0.004, **cost_changes)
    economics = MarketExecutor(
        base.dataset, cost, insolvency_valuation="retain_debt"
    ).execution_policy_digest
    recipe = deepcopy(base.recipe)
    recipe["runtime_profile"]["economics_digest"] = economics
    stress = AllocationTradingEnv(
        dataset=base.dataset,
        stream=base.stream,
        estimates=tuple(base._estimates.values()),
        bound=replace(
            base.bound,
            objective=replace(
                base.bound.objective,
                economics_digest=economics,
                deployment_recipe_digest=content_digest(recipe),
            ),
        ),
        action_contract=base.action_contract,
        allocator=base.allocator,
        execution_cost=cost,
        risk_config=base.risk_config,
        feature_indices=base.feature_indices,
        symbol_index=base.symbol_index,
        start_index=base.start_index,
        stop_index=base.stop_index,
        account_id=base.account_id,
        observation_schema=base.observation_schema,
        feature_preprocessing=base.feature_preprocessing,
    )
    return base, stress


def declared_manifest(env):
    """Closed, internally consistent declarations for an explicitly fake model."""
    raw = manifest_v5()
    raw["recipe"] = deepcopy(env.recipe)
    raw["recipe_digest"] = content_digest(raw["recipe"])
    training = raw["training"]
    training["recipe_digest"] = raw["recipe_digest"]
    training["clock"]["economic_horizon_seconds"] = 4 * 3600
    windows = []
    for day, record in enumerate(training["sources"], 1):
        source = record["source"]
        source.update(
            stop_index=10,
            decision_indices=list(range(6, 10)),
            decision_counts=[8] * 4,
            observation_indices=list(range(6, 10)),
            decision_start=f"2025-12-0{day}T06:00:00.000000000",
            terminal_time=f"2025-12-0{day}T10:00:00.000000000",
        )
        envelope = source | {"decision_counts": [1] * 4}
        window = AllocationTrainingWindow(
            "train",
            source["dataset_id"],
            "S0",
            6,
            10,
            int(np.datetime64(source["decision_start"], "ns").astype("int64")),
            int(np.datetime64(source["terminal_time"], "ns").astype("int64")),
            content_digest({k: v for k, v in envelope.items() if k != "dataset_id"}),
        )
        windows.append(window)
        record["window_id"] = window.window_id
        record["objective"] = env.bound.objective.payload() | {
            "evaluation_start": f"2025-12-0{day}T06:00:00Z",
            "evaluation_stop_exclusive": f"2025-12-0{day}T10:00:00Z",
        }
        training["usage"]["windows"][day - 1].update(
            window_id=window.window_id,
            reset_count=9 if day == 1 else 8,
        )
        training["consumption"]["windows"][day - 1].update(
            window_id=window.window_id,
            decision_indices=list(range(6, 10)),
            decision_counts=[8] * 4,
            observation_indices=list(range(6, 10)),
        )
    schedule = AllocationTrainingSchedule(
        tuple(windows), tuple(w.window_id for w in windows)
    )
    training["schedule"] = schedule.payload()
    training["schedule_digest"] = schedule.digest
    training["usage"]["schedule_digest"] = schedule.digest
    if env.feature_preprocessing is not None:
        from trade_rl.strategies.rl.allocation_preprocessing_receipt import (
            preprocessing_fit_payload,
        )

        training["preprocessing_fit"] = preprocessing_fit_payload(
            env.feature_preprocessing
        )
    return validate_allocation_manifest(raw)


class FakeModel:
    def __init__(self, manifest, actions=(2, 0, 0, 0)):
        training = manifest["training"]
        for key, value in training["ppo"].items():
            setattr(self, key, value)
        self.observation_space = gym.spaces.Box(
            -np.inf,
            np.inf,
            (len(manifest["recipe"]["observation"]["fields"]),),
            np.float32,
        )
        self.action_space = gym.spaces.Discrete(4)
        self.weights = {"actor.weight": np.array([[1.0]], dtype=np.float32)}
        self.policy = SimpleNamespace(
            net_arch=training["ppo"]["net_arch"], state_dict=lambda: self.weights
        )
        self.seed, self.num_timesteps, self.n_envs = 7, 64, 1
        self._n_updates = training["optimization"]["epoch_iterations"]
        self.actions, self.calls = iter(actions), []
        self.before_predict = None

    def predict(self, observation, deterministic):
        assert deterministic is True
        self.calls.append(observation.copy())
        if self.before_predict is not None:
            self.before_predict()
        return np.array(next(self.actions)), None

    def save(self, path):
        Path(path).write_bytes(b"fake fee-view archive; no fitted model")


def publish_fake(base, tmp_path, monkeypatch, *, actions=(2, 0, 0, 0)):
    import trade_rl.strategies.rl.allocation_model as owner

    # Only the optional backend class/architecture check is replaced. Ordinary
    # manifest, spaces, PPO fields, full recipe, vector and action guards execute.
    monkeypatch.setattr(owner, "validate_allocation_protocol_model", lambda *_: None)
    raw = declared_manifest(base)
    model = FakeModel(raw, actions)
    policy = AllocationPPOPolicy(model, raw)
    root = tmp_path / "original-policy"
    digest = save_allocation_policy(root, policy)
    return root, digest, model
