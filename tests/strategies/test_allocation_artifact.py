"""Allocation bundle boundary checks; no fit or economic selection."""

from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

from trade_rl.artifacts import canonical_json_bytes, content_digest


def scheduled_bundle(tmp_path, *, frozen=False):
    from hashlib import sha256

    from tests.strategies.test_allocation_schedule_receipt import direct_reader_case

    raw = direct_reader_case(frozen=frozen)
    root = tmp_path / "scheduled"
    root.mkdir()
    model = b"inert policy bytes; pure reader must not deserialize"
    raw["policy_sha256"] = sha256(model).hexdigest()
    (root / "policy.zip").write_bytes(model)
    (root / "manifest.json").write_bytes(canonical_json_bytes(raw))
    return root, raw


@pytest.mark.parametrize("frozen", [False, True])
def test_pinned_pure_reader_and_strict_absolute_training_cutoff(
    tmp_path, monkeypatch, frozen
):
    module = capability("artifact")
    reader = getattr(module, "read_allocation_policy_manifest", None)
    assert callable(reader), "pure pinned manifest reader is absent"
    root, raw = scheduled_bundle(tmp_path, frozen=frozen)
    last = raw["training"]["schedule"]["windows"][-1]["terminal_time_ns"]
    monkeypatch.setattr(
        module, "_load_policy", lambda _: pytest.fail("backend reached")
    )
    pins = dict(
        expected_digest=content_digest(raw), expected_recipe_digest=raw["recipe_digest"]
    )
    assert reader(root, **pins, training_cutoff_ns=last + 1000) == raw
    for cutoff in (last, last - 1000, True, np.int64(last + 1000)):
        with pytest.raises(ValueError):
            reader(root, **pins, training_cutoff_ns=cutoff)

    class Clock(int):
        pass

    with pytest.raises(ValueError):
        reader(root, **pins, training_cutoff_ns=Clock(last + 1000))
    with pytest.raises(ValueError, match="pinned"):
        reader(
            root, expected_digest="f" * 64, expected_recipe_digest=raw["recipe_digest"]
        )
    with pytest.raises(ValueError, match="recipe"):
        reader(
            root,
            expected_digest=pins["expected_digest"],
            expected_recipe_digest="f" * 64,
        )
    # The reader returns detached native bytes, not an in-process trusted object.
    returned = reader(root, **pins)
    returned["recipe"]["feature_names"][0] = "mutated"
    assert reader(root, **pins) == raw


def test_global_cutoff_rejects_historical_bundle_without_changing_default(
    tmp_path, monkeypatch
):
    from hashlib import sha256

    module = capability("artifact")
    reader = getattr(module, "read_allocation_policy_manifest", None)
    assert callable(reader), "pure pinned manifest reader is absent"
    raw = manifest()
    root = tmp_path / "historical"
    root.mkdir()
    model = b"historical policy"
    raw["policy_sha256"] = sha256(model).hexdigest()
    (root / "policy.zip").write_bytes(model)
    (root / "manifest.json").write_bytes(canonical_json_bytes(raw))
    pins = dict(
        expected_digest=content_digest(raw), expected_recipe_digest=raw["recipe_digest"]
    )
    assert reader(root, **pins) == raw
    with pytest.raises(ValueError, match="v5"):
        reader(root, **pins, training_cutoff_ns=10**18)
    monkeypatch.setattr(
        module, "_load_policy", lambda _: pytest.fail("backend reached")
    )
    with pytest.raises(ValueError, match="v5"):
        module.load_allocation_policy(root, **pins, training_cutoff_ns=10**18)


@pytest.mark.parametrize(
    "change", ["source_clock", "normalizer_values", "normalizer_future"]
)
def test_pure_reader_preserves_source_clock_and_frozen_fit_guards(
    tmp_path, monkeypatch, change
):
    module = capability("artifact")
    root, raw = scheduled_bundle(tmp_path, frozen=True)
    cutoff = raw["training"]["schedule"]["windows"][-1]["terminal_time_ns"] + 1000
    if change == "source_clock":
        raw["training"]["sources"][0]["source"]["terminal_time"] = (
            "2026-01-03T22:00:00.000000000"
        )
    else:
        frozen = raw["recipe"]["observation"]["feature_preprocessing"]
        if change == "normalizer_values":
            frozen["statistics"]["mean"][0] += 1.0
        else:
            frozen["fit_as_of_ns"] = cutoff
            frozen["policy_start_time_ns"] = cutoff
        raw["recipe_digest"] = content_digest(raw["recipe"])
        raw["training"]["recipe_digest"] = raw["recipe_digest"]
    (root / "manifest.json").write_bytes(canonical_json_bytes(raw))
    monkeypatch.setattr(
        module, "_load_policy", lambda _: pytest.fail("backend reached")
    )
    with pytest.raises(ValueError):
        module.load_allocation_policy(
            root,
            expected_digest=content_digest(raw),
            expected_recipe_digest=raw["recipe_digest"],
            training_cutoff_ns=cutoff,
        )


def capability(name):
    try:
        return import_module(f"trade_rl.strategies.rl.allocation_{name}")
    except ModuleNotFoundError as error:
        if error.name == f"trade_rl.strategies.rl.allocation_{name}":
            pytest.fail("Allocation PPO frozen policy capability is missing")
        raise


def manifest():
    from dataclasses import asdict

    from trade_rl.strategies.allocation import AfterCostTargetAllocator
    from trade_rl.strategies.allocation_action import AllocationActionContract
    from trade_rl.strategies.rl.allocation_policy import (
        ALLOCATION_OBSERVATION_FIELDS,
        ALLOCATION_OBSERVATION_SCHEMA,
    )

    recipe = {
        "schema": "allocation_ppo_recipe_v1",
        "observation": {
            "schema": ALLOCATION_OBSERVATION_SCHEMA,
            "fields": list(ALLOCATION_OBSERVATION_FIELDS),
        },
        "action": AllocationActionContract("direct", 0.5).payload(),
        "action_mapping": "bounded_direct_or_residual_v1",
        "feature_names": ["signal"],
        "allocator": asdict(AfterCostTargetAllocator()),
        "expected_horizon_seconds": 3600,
        "feature_preprocessing": "raw_available_values_v1",
        "terminal_valuation": "marked_continuation",
        "runtime_profile": {
            "schema": "allocation_runtime_profile_v1",
            "economics_digest": "1" * 64,
            "risk_digest": "2" * 64,
            "initial_capital": 1000.0,
            "currency": "USD",
            "decision_interval_seconds": 3600,
            "economic_horizon_seconds": 16 * 3600,
            "insolvency_valuation": "retain_debt",
            "calendar_kind": "continuous_24_7",
            "execution_bar_hours": 1.0,
        },
    }
    return {
        "schema": "allocation_ppo_inference_bundle_v1",
        "recipe": recipe,
        "recipe_digest": content_digest(recipe),
        "training": {
            "seed": 7,
            "requested_timesteps": 64,
            "actual_timesteps": 64,
            "clock": {
                "schema": "financial_clock_v1",
                "decision_interval_seconds": 3600,
                "execution_interval_seconds": 3600,
                "reward_interval_seconds": 3600,
                "economic_horizon_seconds": 16 * 3600,
                "rollout_steps": 32,
                "gamma": 1.0,
                "gae_lambda": 0.95,
                "reward_schema": "equity_delta_v1",
            },
            "objective": {
                "schema": "net_profit_objective_v1",
                "objective_id": "expected_terminal_net_profit_v1",
                "tax_treatment": "pretax",
                "infrastructure_cost_treatment": "reported_separately",
                "external_cash_flow_convention": "net_deposits_positive_withdrawals_negative",
                "capital": {
                    "mode": "independent_symbol",
                    "currency": "USD",
                    "initial_equities": [1000.0],
                },
                "evaluation_start": "2026-01-01T06:00:00Z",
                "evaluation_stop_exclusive": "2026-01-01T22:00:00Z",
                "terminal_valuation": "marked_continuation",
                "economics_digest": "1" * 64,
                "risk_digest": "2" * 64,
                "deployment_recipe_digest": content_digest(recipe),
                "maximum_drawdown": 0.2,
            },
            "source": {
                "dataset_id": "a" * 64,
                "start_index": 6,
                "stop_index": 22,
                "decision_indices": list(range(6, 22)),
                "decision_counts": [4] * 16,
                "observation_indices": list(range(6, 22)),
                "symbol": "S0",
                "feature_names": ["signal"],
                "decision_start": "2026-01-01T06:00:00.000000000",
                "terminal_time": "2026-01-01T22:00:00.000000000",
                "feature_consumption_digest": "5" * 64,
                "clock_consumption_digest": "6" * 64,
                "execution_consumption_digest": "7" * 64,
                "forecast_consumption_digest": "3" * 64,
                "cost_consumption_digest": "4" * 64,
            },
            "ppo": {
                "gamma": 1.0,
                "gae_lambda": 0.95,
                "n_steps": 32,
                "batch_size": 32,
                "n_epochs": 10,
                "learning_rate": 0.002,
                "net_arch": [32, 32],
                "device": "cpu",
            },
        },
    }


class Policy:
    def __init__(self):
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (17,), np.float32)
        self.action_space = gym.spaces.Discrete(4)
        self.policy = SimpleNamespace(net_arch=[32, 32])
        self.device = "cpu"
        self.n_envs = 1
        self.gamma, self.gae_lambda = 1.0, 0.95
        self.n_steps, self.batch_size, self.n_epochs = 32, 32, 10
        self.num_timesteps, self.learning_rate, self.seed = 64, 0.002, 7
        self.output = np.array(3)

    def predict(self, observation, deterministic):
        assert deterministic is True
        assert observation.dtype == np.float32
        return self.output, None

    def save(self, path):
        Path(path).write_bytes(b"pinned-allocation-policy")


def test_objective_duration_cannot_round_away_a_microsecond_over_a_long_horizon():
    raw = manifest()
    horizon = (datetime(2250, 1, 1) - datetime(1700, 1, 1)).days * 86400
    raw["recipe"]["runtime_profile"]["economic_horizon_seconds"] = horizon
    raw["training"]["clock"]["economic_horizon_seconds"] = horizon
    raw["recipe_digest"] = content_digest(raw["recipe"])
    objective = raw["training"]["objective"]
    objective["deployment_recipe_digest"] = raw["recipe_digest"]
    objective["evaluation_start"] = datetime(1700, 1, 1, tzinfo=UTC).isoformat()
    objective["evaluation_stop_exclusive"] = datetime(
        2250, 1, 1, microsecond=1, tzinfo=UTC
    ).isoformat()
    source = raw["training"]["source"]
    source["stop_index"] = 6 + horizon // 3600
    source["decision_start"] = "1700-01-01T00:00:00.000000000"
    source["terminal_time"] = "2250-01-01T00:00:00.000001000"
    with pytest.raises(ValueError, match="horizon"):
        capability("model").AllocationPPOPolicy(Policy(), raw)


def test_model_snapshot_and_four_action_boundary():
    cls = capability("model").AllocationPPOPolicy
    raw = manifest()
    wrapper = cls(Policy(), raw)
    raw["recipe"]["feature_names"][0] = "changed"
    exposed = wrapper.manifest
    exposed["training"]["seed"] = 999
    assert wrapper.manifest["training"]["seed"] == 7
    assert (
        wrapper.action(
            np.zeros(17), runtime_recipe_digest=wrapper.manifest["recipe_digest"]
        )
        == 3
    )
    with pytest.raises(ValueError, match="recipe"):
        wrapper.action(np.zeros(17), runtime_recipe_digest="0" * 64)
    for observation in (
        np.zeros(16),
        np.zeros((1, 17)),
        np.full(17, np.nan),
        np.full(17, 1e100),
    ):
        with pytest.raises(ValueError, match="observation"):
            wrapper.action(
                observation, runtime_recipe_digest=wrapper.manifest["recipe_digest"]
            )
    for output in (np.array(4), np.array(3.0), np.array([True]), np.array([1, 2])):
        wrapper.model.output = output
        with pytest.raises(ValueError, match="action"):
            wrapper.action(
                np.zeros(17), runtime_recipe_digest=wrapper.manifest["recipe_digest"]
            )


def test_bundle_pins_policy_before_loader_and_is_write_once(tmp_path, monkeypatch):
    model = capability("model")
    artifact = capability("artifact")
    wrapper = model.AllocationPPOPolicy(Policy(), manifest())
    root = tmp_path / "policy"
    digest = artifact.save_allocation_policy(root, wrapper)
    loaded_paths = []

    def load(path):
        loaded_paths.append(Path(path))
        (root / "policy.zip").write_bytes(b"changed-after-verification")
        assert Path(path).read_bytes() == b"pinned-allocation-policy"
        assert Path(path) != root / "policy.zip"
        return Policy()

    monkeypatch.setattr(artifact, "_load_policy", load)
    loaded = artifact.load_allocation_policy(
        root,
        expected_digest=digest,
        expected_recipe_digest=wrapper.manifest["recipe_digest"],
    )
    assert len(loaded_paths) == 1
    assert not loaded_paths[0].exists()
    assert loaded.manifest["policy_sha256"]
    with pytest.raises(FileExistsError):
        artifact.save_allocation_policy(root, wrapper)
    (root / "policy.zip").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="verified copy"):
        artifact.load_allocation_policy(
            root,
            expected_digest=digest,
            expected_recipe_digest=wrapper.manifest["recipe_digest"],
        )
    assert len(loaded_paths) == 1


@pytest.mark.parametrize(
    "mutation", ["schema", "layout", "budget", "clock", "recipe", "extra", "spaces"]
)
def test_rehashed_invalid_contract_is_rejected_before_loading(
    tmp_path, monkeypatch, mutation
):
    model = capability("model")
    artifact = capability("artifact")
    wrapper = model.AllocationPPOPolicy(Policy(), manifest())
    root = tmp_path / "policy"
    artifact.save_allocation_policy(root, wrapper)
    raw = wrapper.manifest
    raw["policy_sha256"] = (
        __import__("hashlib").sha256((root / "policy.zip").read_bytes()).hexdigest()
    )
    if mutation == "schema":
        raw["schema"] = "ppo_inference_bundle_v1"
    if mutation == "layout":
        raw["recipe"]["observation"]["fields"].reverse()
    if mutation == "budget":
        raw["training"]["actual_timesteps"] = 65
    if mutation == "clock":
        raw["training"]["ppo"]["gamma"] = 0.99
    if mutation == "recipe":
        raw["recipe_digest"] = "0" * 64
    if mutation == "extra":
        raw["unexpected"] = "untrusted"
    if mutation == "spaces":
        raw["recipe"]["feature_names"].append("other")
        raw["training"]["source"]["feature_names"].append("other")
    if mutation in ("layout", "spaces"):
        raw["recipe_digest"] = content_digest(raw["recipe"])
        raw["training"]["objective"]["deployment_recipe_digest"] = raw["recipe_digest"]
    (root / "manifest.json").write_bytes(canonical_json_bytes(raw))
    calls = []

    def load(path):
        calls.append(path)
        return Policy()

    monkeypatch.setattr(artifact, "_load_policy", load)
    with pytest.raises(ValueError):
        artifact.load_allocation_policy(
            root,
            expected_digest=content_digest(raw),
            expected_recipe_digest=raw["recipe_digest"],
        )
    assert len(calls) == int(mutation == "spaces")


def test_boolean_ppo_gamma_is_not_numeric_one():
    raw = manifest()
    raw["training"]["ppo"]["gamma"] = True
    with pytest.raises(ValueError, match="gamma"):
        capability("model").AllocationPPOPolicy(Policy(), raw)


def test_raw_observation_space_cannot_be_replaced_by_clipped_box():
    policy = Policy()
    policy.observation_space = gym.spaces.Box(-1.0, 1.0, (17,), np.float32)
    with pytest.raises(ValueError, match="spaces"):
        capability("model").AllocationPPOPolicy(policy, manifest())


def test_noncanonical_manifest_and_directory_extras_do_not_reach_loader(
    tmp_path, monkeypatch
):
    artifact = capability("artifact")
    wrapper = capability("model").AllocationPPOPolicy(Policy(), manifest())
    root = tmp_path / "bundle"
    digest = artifact.save_allocation_policy(root, wrapper)
    calls = []
    monkeypatch.setattr(artifact, "_load_policy", lambda path: calls.append(path))
    original = (root / "manifest.json").read_bytes()
    (root / "manifest.json").write_bytes(original + b"\n")
    with pytest.raises(ValueError, match="canonical"):
        artifact.load_allocation_policy(
            root,
            expected_digest=digest,
            expected_recipe_digest=wrapper.manifest["recipe_digest"],
        )
    (root / "manifest.json").write_bytes(original)
    (root / "extra").write_text("unversioned")
    with pytest.raises(ValueError, match="directory"):
        artifact.load_allocation_policy(
            root,
            expected_digest=digest,
            expected_recipe_digest=wrapper.manifest["recipe_digest"],
        )
    assert calls == []


@pytest.mark.parametrize(
    "name,value",
    [
        ("seed", 8),
        ("num_timesteps", 63),
        ("n_envs", 2),
        ("gamma", 0.99),
        ("gae_lambda", 0.9),
        ("batch_size", 16),
        ("learning_rate", 0.003),
    ],
)
def test_actual_model_settings_cannot_be_substituted(name, value):
    model = Policy()
    setattr(model, name, value)
    with pytest.raises(ValueError, match="actual PPO"):
        capability("model").AllocationPPOPolicy(model, manifest())


@pytest.mark.parametrize(
    "value",
    [
        np.ones(17, dtype=complex) * (1 + 2j),
        np.array(["1"] * 17),
        np.ones(17, dtype=bool),
    ],
)
def test_observation_cannot_silently_discard_values_or_coerce_text(value):
    wrapper = capability("model").AllocationPPOPolicy(Policy(), manifest())
    with pytest.raises(ValueError, match="observation"):
        wrapper.action(value, runtime_recipe_digest=wrapper.manifest["recipe_digest"])


@pytest.mark.parametrize("entry", ["manifest.json", "policy.zip"])
def test_bundle_symlink_files_are_rejected_without_loading(
    tmp_path, monkeypatch, entry
):
    artifact = capability("artifact")
    wrapper = capability("model").AllocationPPOPolicy(Policy(), manifest())
    root = tmp_path / "bundle"
    digest = artifact.save_allocation_policy(root, wrapper)
    target = root / entry
    preserved = tmp_path / (entry + ".preserved")
    target.rename(preserved)
    try:
        target.symlink_to(preserved)
    except OSError as error:
        pytest.skip(f"filesystem does not permit test symlinks: {error}")
    calls = []
    monkeypatch.setattr(artifact, "_load_policy", lambda path: calls.append(path))
    with pytest.raises(ValueError, match="symlink"):
        artifact.load_allocation_policy(
            root,
            expected_digest=digest,
            expected_recipe_digest=wrapper.manifest["recipe_digest"],
        )
    assert calls == []


def test_explicit_sb3_actor_and_critic_architectures_preserve_metadata():
    raw = manifest()
    network = {"pi": [32, 32], "vf": [32, 32]}
    raw["training"]["ppo"]["net_arch"] = network
    model = Policy()
    model.policy.net_arch = network
    wrapper = capability("model").AllocationPPOPolicy(model, raw)
    assert wrapper.manifest["training"]["ppo"]["net_arch"] == network


@pytest.mark.parametrize(
    "network",
    [
        {"pi": [32], "vf": [32], "other": [32]},
        {"pi": [32]},
        {"pi": [], "vf": [32]},
        {"pi": [True], "vf": [32]},
        {"pi": [32.0], "vf": [32]},
        {"pi": [32], "vf": "32"},
    ],
)
def test_actor_critic_architectures_require_exact_sb3_networks(network):
    raw = manifest()
    raw["training"]["ppo"]["net_arch"] = network
    model = Policy()
    model.policy.net_arch = network
    with pytest.raises(ValueError, match="architecture|network"):
        capability("model").AllocationPPOPolicy(model, raw)


def test_out_of_range_objective_cannot_match_wrapped_source_nanoseconds():
    raw = manifest()
    objective, source = raw["training"]["objective"], raw["training"]["source"]
    objective["evaluation_start"] = "2500-01-01T06:00:00Z"
    objective["evaluation_stop_exclusive"] = "2500-01-01T22:00:00Z"
    source["decision_start"] = "1915-06-14T06:25:26.290448384"
    source["terminal_time"] = "1915-06-14T22:25:26.290448384"
    with pytest.raises(ValueError, match="timestamp|nanosecond"):
        capability("model").AllocationPPOPolicy(Policy(), raw)


def test_out_of_range_source_calendar_cannot_match_valid_wrapped_objective():
    raw = manifest()
    objective, source = raw["training"]["objective"], raw["training"]["source"]
    # Adding the 2500-date wrap remainder makes ns parsing exactly the existing
    # 1915 microsecond endpoint. Compare calendar range before narrowing.
    objective["evaluation_start"] = "1915-06-14T06:25:26.290449Z"
    objective["evaluation_stop_exclusive"] = "1915-06-14T22:25:26.290449Z"
    source["decision_start"] = "2500-01-01T06:00:00.000000616"
    source["terminal_time"] = "2500-01-01T22:00:00.000000616"
    with pytest.raises(ValueError, match="timestamp|nanosecond"):
        capability("model").AllocationPPOPolicy(Policy(), raw)


def test_excess_objective_precision_cannot_be_silently_truncated():
    raw = manifest()
    objective = raw["training"]["objective"]
    objective["evaluation_start"] = "2026-01-01T06:00:00.0000001Z"
    objective["evaluation_stop_exclusive"] = "2026-01-01T22:00:00.0000001Z"
    with pytest.raises(ValueError, match="timestamp|precision"):
        capability("model").AllocationPPOPolicy(Policy(), raw)


def test_excess_source_precision_cannot_be_silently_truncated():
    raw = manifest()
    source = raw["training"]["source"]
    source["decision_start"] = "2026-01-01T06:00:00.0000000001"
    source["terminal_time"] = "2026-01-01T22:00:00.0000000001"
    with pytest.raises(ValueError, match="timestamp|precision"):
        capability("model").AllocationPPOPolicy(Policy(), raw)


@pytest.mark.parametrize(
    "text,ticks",
    [
        ("1677-09-21T00:12:43.145224193", -(2**63) + 1),
        ("2262-04-11T23:47:16.854775807", 2**63 - 1),
        (
            "2026-01-01T06:00:00.000000123+09:00",
            int(np.datetime64("2025-12-31T21:00:00.000000123", "ns").astype(np.int64)),
        ),
    ],
)
def test_source_receipt_preserves_exact_finite_ns_and_normalizes_utc(text, ticks):
    from trade_rl.strategies.rl.allocation_receipt_time import parse_source_timestamp

    assert int(parse_source_timestamp(text).astype(np.int64)) == ticks


@pytest.mark.parametrize(
    "text",
    [
        "1677-09-21T00:12:43.145224192",
        "2262-04-11T23:47:16.854775808",
        "NaT",
        "Infinity",
        "2026-02-30T06:00:00",
        "2500-01-01T06:00:00",
        0,
    ],
)
def test_source_receipt_rejects_invalid_calendar_or_nonfinite_ns(text):
    from trade_rl.strategies.rl.allocation_receipt_time import parse_source_timestamp

    with pytest.raises(ValueError, match="timestamp"):
        parse_source_timestamp(text)


def test_source_nanosecond_tail_cannot_approximate_objective():
    raw = manifest()
    raw["training"]["source"]["decision_start"] = "2026-01-01T06:00:00.000000001"
    with pytest.raises(ValueError, match="timestamp"):
        capability("model").AllocationPPOPolicy(Policy(), raw)


def test_long_objective_horizon_cannot_round_away_one_microsecond():
    from datetime import UTC, datetime

    raw = manifest()
    start, stop = datetime(1900, 1, 1, tzinfo=UTC), datetime(2200, 1, 1, tzinfo=UTC)
    horizon = (stop - start).days * 86400
    recipe, training = raw["recipe"], raw["training"]
    recipe["runtime_profile"]["economic_horizon_seconds"] = horizon
    raw["recipe_digest"] = content_digest(recipe)
    training["clock"]["economic_horizon_seconds"] = horizon
    training["objective"].update(
        evaluation_start="1900-01-01T00:00:00Z",
        evaluation_stop_exclusive="2200-01-01T00:00:00.000001Z",
        deployment_recipe_digest=raw["recipe_digest"],
    )
    training["source"].update(
        stop_index=6 + horizon // 3600,
        decision_start="1900-01-01T00:00:00.000000000",
        terminal_time="2200-01-01T00:00:00.000001000",
    )
    with pytest.raises(ValueError, match="timestamp|horizon"):
        capability("model").AllocationPPOPolicy(Policy(), raw)


@pytest.mark.parametrize(
    "indices,counts",
    [
        ([6, 6], [32, 32]),
        ([7, 6], [32, 32]),
        ([6, 22], [32, 32]),
        ([True], [64]),
        ([6], [63]),
        ([6], [True]),
        ([6], [0]),
        ([6], [32, 32]),
        ([], []),
    ],
)
def test_sampled_decision_scope_requires_ordered_rows_and_exact_realized_budget(
    indices, counts
):
    raw = manifest()
    raw["training"]["source"].update(decision_indices=indices, decision_counts=counts)
    with pytest.raises(ValueError, match="decision|budget"):
        capability("model").AllocationPPOPolicy(Policy(), raw)


def test_early_terminal_sampled_scope_can_retain_declared_episode_envelope():
    raw = manifest()
    raw["training"]["source"].update(
        decision_indices=[6], decision_counts=[64], observation_indices=[6]
    )
    wrapper = capability("model").AllocationPPOPolicy(Policy(), raw)
    assert wrapper.manifest["training"]["source"]["stop_index"] == 22


def test_bootstrap_successor_observation_is_part_of_consumed_receipt():
    raw = manifest()
    raw["training"]["source"].update(
        decision_indices=[6, 7], decision_counts=[32, 32], observation_indices=[6, 7, 8]
    )
    wrapper = capability("model").AllocationPPOPolicy(Policy(), raw)
    assert wrapper.manifest["training"]["source"]["observation_indices"] == [6, 7, 8]


@pytest.mark.parametrize(
    "observations",
    [None, [], [6], [6, 7, 7], [7, 6], [6, 7, 22], [True, 7], [6, 7, 10], [6, 7, 8.0]],
)
def test_observation_scope_must_be_ordered_episode_rows_with_decisions_and_successors(
    observations,
):
    raw = manifest()
    raw["training"]["source"].update(
        decision_indices=[6, 7],
        decision_counts=[32, 32],
        observation_indices=observations,
    )
    with pytest.raises(ValueError, match="observation"):
        capability("model").AllocationPPOPolicy(Policy(), raw)


def test_observation_scope_is_required_in_new_unpublished_bundle():
    raw = manifest()
    raw["training"]["source"].pop("observation_indices")
    with pytest.raises(ValueError, match="source"):
        capability("model").AllocationPPOPolicy(Policy(), raw)


@pytest.mark.parametrize(
    "indices,counts,observations",
    [([7], [64], [6, 7]), ([6, 8], [32, 32], [6, 8]), ([6, 7], [1, 63], [6, 7])],
)
def test_sampled_actions_must_follow_reset_prefix_without_increasing_counts(
    indices, counts, observations
):
    raw = manifest()
    raw["training"]["source"].update(
        decision_indices=indices,
        decision_counts=counts,
        observation_indices=observations,
    )
    with pytest.raises(ValueError, match="decision|prefix|count"):
        capability("model").AllocationPPOPolicy(Policy(), raw)
