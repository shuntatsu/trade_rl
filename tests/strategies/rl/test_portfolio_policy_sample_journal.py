from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import gymnasium as gym
import numpy as np
import pytest

from trade_rl.strategies.rl import portfolio_policy_sample_journal as journal_module
from trade_rl.strategies.rl.portfolio_policy_sample_journal import (
    PortfolioPolicySampleJournal,
)


def _sb3() -> Any:
    pytest.importorskip("torch")
    return pytest.importorskip("stable_baselines3")


class _TinyPortfolioEnv(gym.Env[dict[str, np.ndarray], np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        sample_root: Path | None = None,
        nvec: tuple[int, ...] = (4, 4),
        mask_shape: tuple[int, ...] = (2,),
    ) -> None:
        super().__init__()
        self.action_space = gym.spaces.MultiDiscrete(np.asarray(nvec, dtype=np.int64))
        mask_n: int | tuple[int, ...] = (
            mask_shape[0] if len(mask_shape) == 1 else mask_shape
        )
        self.observation_space = gym.spaces.Dict(
            {
                "features": gym.spaces.Box(
                    low=-10.0, high=10.0, shape=(2,), dtype=np.float32
                ),
                "mask": gym.spaces.MultiBinary(mask_n),
            }
        )
        self.sample_root = sample_root
        self._mask_shape = mask_shape
        self.step_calls = 0
        self.sample_counts_at_step: list[int] = []

    def _observation(self) -> dict[str, np.ndarray]:
        return {
            "features": np.asarray([0.25, -0.5], dtype=np.float32),
            "mask": np.asarray([1, 0], dtype=np.int8).reshape(self._mask_shape),
        }

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        return self._observation(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        self.step_calls += 1
        if self.sample_root is not None:
            files = tuple((self.sample_root / "samples").glob("sample-*.json"))
            self.sample_counts_at_step.append(len(files))
        return self._observation(), 0.0, True, False, {}


def _model(
    *,
    sample_root: Path | None = None,
    nvec: tuple[int, ...] = (4, 4),
    mask_shape: tuple[int, ...] = (2,),
) -> tuple[Any, _TinyPortfolioEnv]:
    sb3 = _sb3()
    env = _TinyPortfolioEnv(sample_root=sample_root, nvec=nvec, mask_shape=mask_shape)
    model = sb3.PPO(
        "MultiInputPolicy",
        env,
        n_steps=2,
        batch_size=2,
        n_epochs=1,
        seed=17,
        policy_kwargs={"net_arch": {"pi": [8], "vf": [8]}},
        verbose=0,
    )
    return model, env


def test_multidimensional_multibinary_space_is_recorded_without_loss(
    tmp_path: Path,
) -> None:
    _sb3()
    model, _ = _model(mask_shape=(2, 1))
    root = tmp_path / "multidimensional-space"

    journal = _journal(model, root)
    journal.close()

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    descriptor = manifest["observation_space"]["fields"]["mask"]
    assert descriptor["shape"] == [2, 1]
    assert descriptor["n"] == [2, 1]
    assert all(type(value) is int for value in descriptor["n"])

    scalar_descriptor = journal_module._space_descriptor(gym.spaces.MultiBinary(2))
    assert scalar_descriptor["n"] == 2
    assert type(scalar_descriptor["n"]) is int


def test_missing_optional_ppo_dependencies_raise_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import_module = journal_module.importlib.import_module

    def import_without_torch(name: str) -> Any:
        if name == "torch":
            raise ImportError("torch is unavailable in the core environment")
        return original_import_module(name)

    monkeypatch.setattr(journal_module.importlib, "import_module", import_without_torch)

    with pytest.raises(RuntimeError, match="install the train-sb3 extra"):
        PortfolioPolicySampleJournal(
            object(),
            root=Path("unused"),
            symbols=("AAA",),
            context_digest="a" * 64,
            transform_digest="b" * 64,
            policy_version="rollout-0",
            policy_version_semantics="caller-declared rollout label",
        )


def _journal(model: Any, root: Path) -> PortfolioPolicySampleJournal:
    return PortfolioPolicySampleJournal(
        model,
        root=root,
        symbols=("AAA", "BBB"),
        context_digest="a" * 64,
        transform_digest="b" * 64,
        policy_version="rollout-0",
        policy_version_semantics="caller-declared rollout label; not a checkpoint hash",
    )


def _sample_events(root: Path) -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((root / "samples").glob("sample-*.json"))
    ]


def test_real_sb3_rollout_journals_each_forward_before_native_step(
    tmp_path: Path,
) -> None:
    _sb3()
    root = tmp_path / "samples"
    model, env = _model(sample_root=root)
    journal = _journal(model, root)

    model.learn(total_timesteps=2)
    journal.close()

    assert env.step_calls == 2
    assert env.sample_counts_at_step == [1, 2]
    events = _sample_events(root)
    assert [event["sample_id"] for event in events] == [1, 2]
    assert all(len(event["raw_action_codes"]) == 2 for event in events)
    assert all(
        all(code in range(4) for code in event["raw_action_codes"]) for event in events
    )
    assert all(event["policy_version"] == "rollout-0" for event in events)


def test_hook_preserves_exact_forward_outputs_rng_and_input_tensor_metadata(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    model, _ = _model()
    observation = {
        "features": torch.tensor([[0.25, -0.5]], dtype=torch.float32),
        "mask": torch.tensor([[1, 0]], dtype=torch.int8),
    }
    torch.manual_seed(113)
    with torch.no_grad():
        expected = model.policy(observation)
    expected_rng = torch.random.get_rng_state().clone()

    root = tmp_path / "journal"
    journal = _journal(model, root)
    torch.manual_seed(113)
    with torch.no_grad():
        actual = model.policy(observation)
    actual_rng = torch.random.get_rng_state().clone()
    journal.close()
    event = _sample_events(root)[0]
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))

    assert event["schema"] == "portfolio_policy_sample_v2"
    assert manifest["sample_schema"] == event["schema"]
    assert event["forward_batch_size"] == 1
    assert event["forward_output_metadata"]["actions"] == {
        "dtype": str(actual[0].dtype).removeprefix("torch."),
        "shape": list(actual[0].shape),
    }
    assert event["forward_output_metadata"]["value_prediction"] == {
        "dtype": str(actual[1].dtype).removeprefix("torch."),
        "shape": list(actual[1].shape),
    }
    assert event["forward_output_metadata"]["joint_log_probability"] == {
        "dtype": str(actual[2].dtype).removeprefix("torch."),
        "shape": list(actual[2].shape),
    }
    for expected_tensor, actual_tensor in zip(expected, actual, strict=True):
        torch.testing.assert_close(expected_tensor, actual_tensor, rtol=0, atol=0)
    assert torch.equal(expected_rng, actual_rng)
    assert event["inputs"]["features"] == {
        "dtype": "float32",
        "shape": [2],
        "values": [0.25, -0.5],
    }
    assert event["inputs"]["mask"] == {
        "dtype": "int8",
        "shape": [2],
        "values": [1, 0],
    }
    assert event["raw_action_codes"] == actual[0][0].tolist()
    assert event["joint_log_probability"] == pytest.approx(float(actual[2][0].item()))
    assert event["value_prediction"] == pytest.approx(float(actual[1][0, 0].item()))
    assert "reward" not in event


def test_batched_policy_forward_creates_one_record_per_environment_row(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    model, _ = _model()
    observation = {
        "features": torch.tensor([[0.25, -0.5], [0.75, 0.125]], dtype=torch.float32),
        "mask": torch.tensor([[1, 0], [0, 1]], dtype=torch.int8),
    }
    root = tmp_path / "batch-journal"
    journal = _journal(model, root)
    with torch.no_grad():
        outputs = model.policy(observation)
    journal.close()

    events = _sample_events(root)
    assert [event["sample_id"] for event in events] == [1, 2]
    assert [event["batch_index"] for event in events] == [0, 1]
    assert [event["raw_action_codes"] for event in events] == outputs[0].tolist()
    assert [event["forward_batch_size"] for event in events] == [2, 2]
    assert all(
        event["forward_output_metadata"]
        == {
            "actions": {
                "dtype": str(outputs[0].dtype).removeprefix("torch."),
                "shape": list(outputs[0].shape),
            },
            "value_prediction": {
                "dtype": str(outputs[1].dtype).removeprefix("torch."),
                "shape": list(outputs[1].shape),
            },
            "joint_log_probability": {
                "dtype": str(outputs[2].dtype).removeprefix("torch."),
                "shape": list(outputs[2].shape),
            },
        }
        for event in events
    )
    assert [event["inputs"]["features"]["values"] for event in events] == [
        [0.25, -0.5],
        [0.75, 0.125],
    ]


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("missing_input", "input keys"),
        ("action_shape", "sampled action shape"),
        ("nan_input", "input features must be finite"),
        ("nan_value", "critic values must be finite"),
        ("nan_log_probability", "joint log probabilities must be finite"),
    ],
)
def test_invalid_forward_payload_poisoned_before_any_sample_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    message: str,
) -> None:
    torch = pytest.importorskip("torch")
    model, _ = _model()
    valid_observation = {
        "features": torch.tensor([[0.25, -0.5]], dtype=torch.float32),
        "mask": torch.tensor([[1, 0]], dtype=torch.int8),
    }
    with torch.no_grad():
        valid_output = model.policy(valid_observation)

    malformed_output = valid_output
    observation = valid_observation
    if case == "missing_input":
        observation = {"features": valid_observation["features"]}
    elif case == "action_shape":
        malformed_output = (
            valid_output[0][:, :1],
            valid_output[1],
            valid_output[2],
        )
    elif case == "nan_input":
        observation = {
            "features": torch.tensor([[float("nan"), -0.5]]),
            "mask": valid_observation["mask"],
        }
    elif case == "nan_value":
        malformed_output = (
            valid_output[0],
            torch.full_like(valid_output[1], float("nan")),
            valid_output[2],
        )
    elif case == "nan_log_probability":
        malformed_output = (
            valid_output[0],
            valid_output[1],
            torch.full_like(valid_output[2], float("nan")),
        )

    monkeypatch.setattr(model.policy, "forward", lambda obs: malformed_output)
    root = tmp_path / case
    journal = _journal(model, root)
    with pytest.raises((TypeError, ValueError), match=message):
        with torch.no_grad():
            model.policy(observation)
    assert journal.is_poisoned
    assert journal.sample_count == 0
    assert not tuple((root / "samples").glob("sample-*.json"))
    journal.close()


def test_model_save_and_reload_does_not_reenable_journal_hook(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    sb3 = _sb3()
    model, _ = _model()
    root = tmp_path / "journal"
    journal = _journal(model, root)
    saved_model = tmp_path / "policy-model"
    model.save(saved_model)
    loaded = sb3.PPO.load(saved_model)
    with torch.no_grad():
        loaded.policy(
            {
                "features": torch.tensor([[0.25, -0.5]]),
                "mask": torch.tensor([[1, 0]], dtype=torch.int8),
            }
        )
    assert journal.sample_count == 0
    assert not tuple((root / "samples").glob("sample-*.json"))
    journal.close()


def test_sample_write_failure_prevents_env_step_and_poisons_hook(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _sb3()
    torch = pytest.importorskip("torch")
    root = tmp_path / "failed-journal"
    model, env = _model(sample_root=root)
    journal = _journal(model, root)

    def fail_sample_write(path: str | Path, payload: bytes) -> Path:
        if Path(path).name.startswith("sample-"):
            raise OSError("literal sample write failure")
        return journal_module.atomic_write_bytes(path, payload)

    monkeypatch.setattr(journal_module, "atomic_write_bytes", fail_sample_write)
    with pytest.raises(OSError, match="literal sample write failure"):
        model.learn(total_timesteps=1)

    assert env.step_calls == 0
    assert journal.is_poisoned
    with pytest.raises(RuntimeError, match="poison"):
        model.policy(
            {
                "features": torch.tensor([[0.25, -0.5]]),
                "mask": torch.tensor([[1, 0]], dtype=torch.int8),
            }
        )


def test_later_replacing_policy_hook_is_rejected_before_sample_write(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    model, _ = _model()
    root = tmp_path / "hook-order"
    journal = _journal(model, root)

    def replace_sampled_actions(
        module: object, inputs: object, output: object
    ) -> object:
        actions, values, log_prob = cast(tuple[Any, Any, Any], output)
        changed = actions.clone()
        changed[:, 0] = (changed[:, 0] + 1) % 4
        return changed, values, log_prob

    later_handle = model.policy.register_forward_hook(replace_sampled_actions)
    with pytest.raises(RuntimeError, match="forward-hook layout changed"):
        with torch.no_grad():
            model.policy(
                {
                    "features": torch.tensor([[0.25, -0.5]]),
                    "mask": torch.tensor([[1, 0]], dtype=torch.int8),
                }
            )

    assert journal.is_poisoned
    assert journal.sample_count == 0
    assert not tuple((root / "samples").glob("sample-*.json"))
    later_handle.remove()
    journal.close()


@pytest.mark.parametrize("scope", ["policy", "global"])
def test_preexisting_replacing_forward_hook_is_rejected_before_root_creation(
    tmp_path: Path, scope: str
) -> None:
    torch = pytest.importorskip("torch")
    model, _ = _model()

    def replace_sampled_actions(
        module: object, inputs: object, output: object
    ) -> object:
        if scope == "global" and module is not model.policy:
            return None
        actions, values, log_prob = cast(tuple[Any, Any, Any], output)
        changed = actions.clone()
        changed[:, 0] = (changed[:, 0] + 1) % 4
        return changed, values, log_prob

    if scope == "policy":
        handle = model.policy.register_forward_hook(replace_sampled_actions)
    else:
        handle = torch.nn.modules.module.register_module_forward_hook(
            replace_sampled_actions
        )
    root = tmp_path / f"preexisting-{scope}"
    try:
        with pytest.raises(RuntimeError, match="pre-existing forward hooks"):
            _journal(model, root)
        assert not root.exists()
        if scope == "policy":
            assert handle.id in model.policy._forward_hooks
        else:
            assert handle.id in torch.nn.modules.module._global_forward_hooks
        # The rejected recorder leaves the caller's replacing hook in place.
        with torch.no_grad():
            model.policy(
                {
                    "features": torch.tensor([[0.25, -0.5]]),
                    "mask": torch.tensor([[1, 0]], dtype=torch.int8),
                }
            )
    finally:
        handle.remove()


def test_postattach_global_replacing_hook_poisons_before_sample_and_env_step(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    _sb3()
    root = tmp_path / "global-hook-order"
    model, env = _model(sample_root=root)
    journal = _journal(model, root)

    def replace_policy_output(module: object, inputs: object, output: object) -> object:
        if module is not model.policy:
            return None
        actions, values, log_prob = cast(tuple[Any, Any, Any], output)
        changed = actions.clone()
        changed[:, 0] = (changed[:, 0] + 1) % 4
        return changed, values, log_prob

    handle = torch.nn.modules.module.register_module_forward_hook(replace_policy_output)
    try:
        with pytest.raises(RuntimeError, match="forward-hook layout changed"):
            model.learn(total_timesteps=1)
        assert env.step_calls == 0
        assert env.sample_counts_at_step == []
        assert journal.is_poisoned
        assert journal.sample_count == 0
        assert not tuple((root / "samples").glob("sample-*.json"))
    finally:
        handle.remove()
        journal.close()


def test_failed_postregistration_layout_validation_removes_only_owned_hook(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, _ = _model()
    policy = model.policy
    register = policy.register_forward_hook
    intruder_handle = None

    def register_then_add_intruder(hook: Any) -> Any:
        nonlocal intruder_handle
        owned_handle = register(hook)
        intruder_handle = register(lambda module, inputs, output: None)
        return owned_handle

    monkeypatch.setattr(policy, "register_forward_hook", register_then_add_intruder)
    root = tmp_path / "attachment-race"
    with pytest.raises(RuntimeError, match="forward-hook layout changed"):
        _journal(model, root)

    assert intruder_handle is not None
    assert tuple(policy._forward_hooks) == (intruder_handle.id,)
    intruder_handle.remove()
    monkeypatch.undo()

    journal = _journal(model, tmp_path / "after-cleanup")
    journal.close()


def test_existing_journal_root_is_rejected_without_hook_or_file_changes(
    tmp_path: Path,
) -> None:
    model, _ = _model()
    root = tmp_path / "occupied"
    root.mkdir()
    marker = root / "keep.txt"
    marker.write_text("preserve", encoding="utf-8")
    hooks_before = tuple(model.policy._forward_hooks.items())

    with pytest.raises(FileExistsError, match="journal root already exists"):
        _journal(model, root)

    assert marker.read_text(encoding="utf-8") == "preserve"
    assert tuple(model.policy._forward_hooks.items()) == hooks_before
    # A failed constructor must release its policy reservation.
    journal = _journal(model, tmp_path / "available")
    journal.close()


def test_close_removes_only_owned_hook_and_duplicate_is_rejected(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    model, _ = _model()
    calls: list[int] = []
    root = tmp_path / "journal"
    journal = _journal(model, root)
    with pytest.raises(ValueError, match="already attached"):
        _journal(model, tmp_path / "duplicate")
    assert not (tmp_path / "duplicate").exists()

    other_handle = model.policy.register_forward_hook(
        lambda module, inputs, output: calls.append(1)
    )

    journal.close()
    with torch.no_grad():
        model.policy(
            {
                "features": torch.tensor([[0.25, -0.5]]),
                "mask": torch.tensor([[1, 0]], dtype=torch.int8),
            }
        )
    assert calls == [1]
    assert journal.is_closed
    assert journal.sample_count == 0
    other_handle.remove()


def test_constructor_rejects_non_four_action_dimension_before_root_creation(
    tmp_path: Path,
) -> None:
    model, _ = _model(nvec=(4, 3))
    root = tmp_path / "invalid"
    with pytest.raises(ValueError, match="four action codes"):
        _journal(model, root)
    assert not root.exists()
