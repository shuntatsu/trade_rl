from __future__ import annotations

import importlib
import importlib.util
from types import SimpleNamespace
from typing import Any

import pytest

gym = pytest.importorskip("gymnasium")
np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
stable_baselines3 = pytest.importorskip("stable_baselines3")
PPO = stable_baselines3.PPO
MultiCategoricalDistribution = importlib.import_module(
    "stable_baselines3.common.distributions"
).MultiCategoricalDistribution
Categorical = torch.distributions.Categorical

_PROBE_MODULE = "tools.issue810_portfolio_probability_probe"


def _probe() -> Any:
    return importlib.import_module(_PROBE_MODULE)


class _DictMultiDiscreteEnv(gym.Env[dict[str, np.ndarray], np.ndarray]):
    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        action_nvec: tuple[int, ...] = (4, 4),
        action_start: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Dict(
            {"state": gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)}
        )
        self.action_space = gym.spaces.MultiDiscrete(
            np.asarray(action_nvec),
            start=None if action_start is None else np.asarray(action_start),
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        del options
        return {"state": np.zeros(3, dtype=np.float32)}, {}

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        del action
        return {"state": np.zeros(3, dtype=np.float32)}, 0.0, False, False, {}


def _new_model(
    action_nvec: tuple[int, ...] = (4, 4),
    action_start: tuple[int, ...] | None = None,
) -> PPO:
    env = _DictMultiDiscreteEnv(action_nvec, action_start)
    return PPO(
        "MultiInputPolicy",
        env,
        n_steps=2,
        batch_size=2,
        n_epochs=1,
        seed=11,
        device="cpu",
        verbose=0,
    )


@pytest.fixture
def model() -> PPO:
    return _new_model()


def _replace_distribution(
    model: PPO, monkeypatch: pytest.MonkeyPatch, distribution: object
) -> None:
    monkeypatch.setattr(model.policy, "get_distribution", lambda _obs: distribution)


def test_probability_probe_module_is_available() -> None:
    assert importlib.util.find_spec(_PROBE_MODULE) is not None


def test_extracts_real_per_coordinate_probabilities_in_declared_order(
    model: PPO, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy = model.policy
    modes_before = [module.training for module in policy.modules()]
    rng_before = torch.random.get_rng_state().clone()
    real_get_distribution = policy.get_distribution

    def inspect_eval_no_grad(observation: torch.Tensor) -> MultiCategoricalDistribution:
        assert not torch.is_grad_enabled()
        assert all(not module.training for module in policy.modules())
        return real_get_distribution(observation)

    monkeypatch.setattr(policy, "get_distribution", inspect_eval_no_grad)
    result = _probe().extract_portfolio_action_probabilities(
        model,
        {"state": np.asarray([0.25, -0.5, 0.75], dtype=np.float32)},
        ordered_symbols=("BTCUSDT", "ETHUSDT"),
    )

    assert result.get("schema") == "issue810_portfolio_action_probabilities_v1"
    assert result.get("ordered_symbols") == ["BTCUSDT", "ETHUSDT"]
    assert result.get("action_order") == ["HOLD", "SHORT", "FLAT", "LONG"]
    assert result.get("symbol_order_source") == "caller_declared_not_verified"
    coordinates = result.get("coordinates")
    assert isinstance(coordinates, list)
    assert [item["coordinate_index"] for item in coordinates] == [0, 1]
    assert [item["symbol"] for item in coordinates] == ["BTCUSDT", "ETHUSDT"]
    for coordinate in coordinates:
        probabilities = coordinate["probabilities"]
        assert list(probabilities) == ["HOLD", "SHORT", "FLAT", "LONG"]
        assert all(
            np.isfinite(value) and 0.0 <= value <= 1.0
            for value in probabilities.values()
        )
        assert sum(probabilities.values()) == pytest.approx(1.0, abs=1e-6)
    assert [module.training for module in policy.modules()] == modes_before
    assert torch.equal(torch.random.get_rng_state(), rng_before)


def test_restores_mixed_module_modes_after_distribution_failure(
    model: PPO, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy = model.policy
    modules = list(policy.modules())
    assert len(modules) > 1
    policy.train(True)
    modules[-1].train(False)
    modes_before = [module.training for module in modules]

    def fail_distribution(_obs: torch.Tensor) -> object:
        raise RuntimeError("injected distribution failure")

    monkeypatch.setattr(policy, "get_distribution", fail_distribution)
    with pytest.raises(RuntimeError, match="injected distribution failure"):
        _probe().extract_portfolio_action_probabilities(
            model,
            {"state": np.zeros(3, dtype=np.float32)},
            ordered_symbols=("BTCUSDT", "ETHUSDT"),
        )

    assert [module.training for module in modules] == modes_before


@pytest.mark.parametrize("fail_distribution", [False, True])
def test_restores_shared_child_mode_after_success_and_failure(
    model: PPO,
    monkeypatch: pytest.MonkeyPatch,
    fail_distribution: bool,
) -> None:
    policy = model.policy
    shared_child = torch.nn.Identity()
    left_parent = torch.nn.Sequential(shared_child)
    right_parent = torch.nn.Sequential(shared_child)
    policy.add_module("shared_mode_left", left_parent)
    policy.add_module("shared_mode_right", right_parent)
    policy.train(True)
    shared_child.eval()

    modes_before = (
        policy.training,
        left_parent.training,
        shared_child.training,
        right_parent.training,
    )
    assert modes_before == (True, True, False, True)
    assert sum(module is shared_child for module in policy.modules()) == 1

    if fail_distribution:

        def fail(_observation: torch.Tensor) -> object:
            raise RuntimeError("injected shared-module failure")

        monkeypatch.setattr(policy, "get_distribution", fail)
        with pytest.raises(RuntimeError, match="injected shared-module failure"):
            _probe().extract_portfolio_action_probabilities(
                model,
                {"state": np.zeros(3, dtype=np.float32)},
                ordered_symbols=("BTCUSDT", "ETHUSDT"),
            )
    else:
        _probe().extract_portfolio_action_probabilities(
            model,
            {"state": np.zeros(3, dtype=np.float32)},
            ordered_symbols=("BTCUSDT", "ETHUSDT"),
        )

    modes_after = (
        policy.training,
        left_parent.training,
        shared_child.training,
        right_parent.training,
    )
    assert modes_after == modes_before


def test_rejects_batched_observation_and_symbol_roster_mismatch(model: PPO) -> None:
    with pytest.raises(ValueError, match="unbatched"):
        _probe().extract_portfolio_action_probabilities(
            model,
            {"state": np.zeros((1, 3), dtype=np.float32)},
            ordered_symbols=("BTCUSDT", "ETHUSDT"),
        )
    with pytest.raises(ValueError, match="ordered symbol"):
        _probe().extract_portfolio_action_probabilities(
            model,
            {"state": np.zeros(3, dtype=np.float32)},
            ordered_symbols=("BTCUSDT",),
        )


def test_rejects_action_space_coordinates_without_four_actions() -> None:
    model = _new_model((4, 5))
    with pytest.raises(ValueError, match="exactly four actions"):
        _probe().extract_portfolio_action_probabilities(
            model,
            {"state": np.zeros(3, dtype=np.float32)},
            ordered_symbols=("BTCUSDT", "ETHUSDT"),
        )


def test_rejects_nonzero_multidiscrete_starts() -> None:
    model = _new_model(action_start=(1, 0))
    with pytest.raises(ValueError, match="zero-based"):
        _probe().extract_portfolio_action_probabilities(
            model,
            {"state": np.zeros(3, dtype=np.float32)},
            ordered_symbols=("BTCUSDT", "ETHUSDT"),
        )


def test_rejects_distribution_component_count_mismatch(
    model: PPO, monkeypatch: pytest.MonkeyPatch
) -> None:
    distribution = MultiCategoricalDistribution([4, 4]).proba_distribution(
        torch.zeros((1, 8))
    )
    distribution.distribution.pop()
    _replace_distribution(model, monkeypatch, distribution)

    with pytest.raises(ValueError, match="component count"):
        _probe().extract_portfolio_action_probabilities(
            model,
            {"state": np.zeros(3, dtype=np.float32)},
            ordered_symbols=("BTCUSDT", "ETHUSDT"),
        )


def test_rejects_non_multicategorical_distribution(
    model: PPO, monkeypatch: pytest.MonkeyPatch
) -> None:
    _replace_distribution(model, monkeypatch, Categorical(logits=torch.zeros((1, 4))))
    with pytest.raises(TypeError, match="MultiCategoricalDistribution"):
        _probe().extract_portfolio_action_probabilities(
            model,
            {"state": np.zeros(3, dtype=np.float32)},
            ordered_symbols=("BTCUSDT", "ETHUSDT"),
        )


@pytest.mark.parametrize(
    ("probabilities", "message"),
    [
        (torch.full((1, 3), 1.0 / 3.0), "shape"),
        (torch.tensor([[float("nan"), 0.2, 0.3, 0.5]]), "finite"),
        (torch.tensor([[0.2, 0.2, 0.2, 0.2]]), "sum to one"),
    ],
)
def test_rejects_invalid_probability_rows(
    model: PPO,
    monkeypatch: pytest.MonkeyPatch,
    probabilities: torch.Tensor,
    message: str,
) -> None:
    distribution = MultiCategoricalDistribution([4, 4]).proba_distribution(
        torch.zeros((1, 8))
    )
    distribution.distribution[0] = SimpleNamespace(probs=probabilities)  # type: ignore[assignment]
    _replace_distribution(model, monkeypatch, distribution)

    with pytest.raises(ValueError, match=message):
        _probe().extract_portfolio_action_probabilities(
            model,
            {"state": np.zeros(3, dtype=np.float32)},
            ordered_symbols=("BTCUSDT", "ETHUSDT"),
        )


@pytest.mark.parametrize(
    ("package", "wrong_version"),
    [
        ("stable-baselines3", "2.3.1"),
        ("torch", "2.4.0"),
        ("torch", "2.4.1+cu121"),
    ],
)
def test_rejects_unpinned_dependency_versions_before_inference(
    model: PPO,
    monkeypatch: pytest.MonkeyPatch,
    package: str,
    wrong_version: str,
) -> None:
    probe = _probe()
    original_version = probe.importlib.metadata.version

    def wrong_version_for_one_package(name: str) -> str:
        if name == package:
            return wrong_version
        return original_version(name)

    monkeypatch.setattr(
        probe.importlib.metadata, "version", wrong_version_for_one_package
    )
    monkeypatch.setattr(
        model.policy,
        "obs_to_tensor",
        lambda _obs: pytest.fail("version mismatch must stop before inference"),
    )
    expected = (
        "stable-baselines3 2.3.2" if package == "stable-baselines3" else "torch 2.4.1"
    )
    with pytest.raises(RuntimeError, match=expected):
        probe.extract_portfolio_action_probabilities(
            model,
            {"state": np.zeros(3, dtype=np.float32)},
            ordered_symbols=("BTCUSDT", "ETHUSDT"),
        )


@pytest.mark.parametrize(
    "runtime_version",
    ["2.4.1", "2.4.1+cpu", "2.4.1+cu121"],
)
def test_accepts_pinned_torch_runtime_with_valid_local_build_suffix(
    monkeypatch: pytest.MonkeyPatch,
    runtime_version: str,
) -> None:
    probe = _probe()
    monkeypatch.setattr(torch, "__version__", runtime_version)
    monkeypatch.setattr(
        probe.importlib.metadata,
        "version",
        lambda package: {
            "stable-baselines3": "2.3.2",
            "torch": "2.4.1",
        }[package],
    )

    runtime = probe._load_pinned_runtime()

    assert runtime.torch_version == "2.4.1"
    assert runtime.torch_runtime_version == runtime_version


@pytest.mark.parametrize(
    "runtime_version",
    [
        "2.4.0+cu121",
        "2.5.0+cpu",
        "2.4.1rc1+cu121",
        "2.4.1+cu121?",
    ],
)
def test_rejects_mismatched_or_invalid_torch_runtime_versions(
    monkeypatch: pytest.MonkeyPatch,
    runtime_version: str,
) -> None:
    probe = _probe()
    monkeypatch.setattr(torch, "__version__", runtime_version)
    monkeypatch.setattr(
        probe.importlib.metadata,
        "version",
        lambda package: {
            "stable-baselines3": "2.3.2",
            "torch": "2.4.1",
        }[package],
    )

    with pytest.raises(RuntimeError, match="runtime build does not match"):
        probe._load_pinned_runtime()
