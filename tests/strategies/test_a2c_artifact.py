from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from trade_rl.artifacts import canonical_json_bytes
from trade_rl.strategies import A2CIntentStrategy, PPOIntentStrategy
from trade_rl.strategies.rl.a2c import A2CFitMetadata
from trade_rl.strategies.rl.a2c_artifact import (
    load_a2c_inference_bundle,
    save_a2c_inference_bundle,
)
from trade_rl.strategies.rl.ppo_artifact import (
    load_ppo_inference_bundle,
    save_ppo_inference_bundle,
)


class FakeA2C:
    load_calls: list[tuple[Path, str]] = []
    load_num_timesteps: object = 5
    load_seed: object = 3
    load_n_steps: object = 5
    observation_space = SimpleNamespace(shape=(5,))
    action_space = SimpleNamespace(n=3, start=0)

    def __init__(
        self,
        num_timesteps: object = 5,
        seed: object = 3,
        n_steps: object = 5,
    ) -> None:
        self.num_timesteps = num_timesteps
        self.seed = seed
        self.n_steps = n_steps

    def save(self, path: str) -> None:
        Path(path).write_bytes(b"a2c policy bytes")

    @classmethod
    def load(cls, path: str, *, device: str = "auto") -> FakeA2C:
        cls.load_calls.append((Path(path), device))
        return cls(cls.load_num_timesteps, cls.load_seed, cls.load_n_steps)

    def predict(self, observation: np.ndarray, *, deterministic: bool = True):
        return np.asarray(1), None


class FakePPO:
    load_calls: list[tuple[Path, str]] = []
    observation_space = SimpleNamespace(shape=(5,))
    action_space = SimpleNamespace(n=3, start=0)

    def save(self, path: str) -> None:
        Path(path).write_bytes(b"ppo policy bytes")

    @classmethod
    def load(cls, path: str, *, device: str = "auto") -> FakePPO:
        cls.load_calls.append((Path(path), device))
        return cls()

    def predict(self, observation: np.ndarray, *, deterministic: bool = True):
        return np.asarray(1), None


def install_fake_sb3(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeA2C.load_calls.clear()
    FakeA2C.load_num_timesteps = 5
    FakeA2C.load_seed = 3
    FakeA2C.load_n_steps = 5
    FakePPO.load_calls.clear()
    monkeypatch.setitem(
        sys.modules,
        "stable_baselines3",
        SimpleNamespace(A2C=FakeA2C, PPO=FakePPO),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(set_num_threads=lambda value: None),
    )


def fit_metadata() -> A2CFitMetadata:
    return A2CFitMetadata(
        requested_timesteps=5,
        effective_timesteps=5,
        rollout_steps=5,
        step_rounding="ceil_to_complete_rollout",
        seed=3,
        fit_symbol_indices=(0,),
        fit_symbols=("BTCUSDT",),
        episode_steps=4,
        required_coverage_timesteps=4,
        nominal_full_episodes_per_symbol=1,
    )


def test_a2c_bundle_has_its_own_canonical_algorithm_bound_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_sb3(monkeypatch)
    strategy = A2CIntentStrategy(
        FakeA2C(),
        feature_indices=(0,),
        feature_names=("signal",),
        fit_metadata=fit_metadata(),
    )
    root = tmp_path / "a2c-bundle"

    digest = save_a2c_inference_bundle(
        root,
        strategy,
        feature_names=("signal",),
    )
    manifest_raw = (root / "manifest.json").read_bytes()
    manifest = json.loads(manifest_raw)
    loaded = load_a2c_inference_bundle(
        root,
        expected_digest=digest,
        feature_names=("signal",),
    )

    assert canonical_json_bytes(manifest) == manifest_raw
    assert manifest["schema"] == "a2c_inference_bundle_v1"
    assert manifest["algorithm"] == "a2c"
    assert isinstance(loaded, A2CIntentStrategy)
    assert loaded.feature_names == ("signal",)
    assert len(FakeA2C.load_calls) == 1
    loaded_path, device = FakeA2C.load_calls[0]
    assert loaded_path != root / "policy.zip"
    assert device == "cpu"
    with pytest.raises(FileExistsError):
        save_a2c_inference_bundle(
            root,
            strategy,
            feature_names=("signal",),
        )


def test_cross_family_bundles_fail_before_sb3_deserialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_sb3(monkeypatch)
    ppo_root = tmp_path / "ppo-bundle"
    ppo_strategy = PPOIntentStrategy(
        FakePPO(), feature_indices=(0,), feature_names=("signal",)
    )
    ppo_digest = save_ppo_inference_bundle(
        ppo_root,
        ppo_strategy,
        feature_names=("signal",),
    )

    with pytest.raises(ValueError, match="manifest|schema"):
        load_a2c_inference_bundle(
            ppo_root,
            expected_digest=ppo_digest,
            feature_names=("signal",),
        )
    assert not FakeA2C.load_calls

    a2c_root = tmp_path / "a2c-bundle"
    a2c_digest = save_a2c_inference_bundle(
        a2c_root,
        A2CIntentStrategy(
            FakeA2C(),
            feature_indices=(0,),
            feature_names=("signal",),
            fit_metadata=fit_metadata(),
        ),
        feature_names=("signal",),
    )
    with pytest.raises(ValueError, match="manifest|schema"):
        load_ppo_inference_bundle(
            a2c_root,
            expected_digest=a2c_digest,
            feature_names=("signal",),
        )
    assert not FakePPO.load_calls


def test_tampered_a2c_policy_bytes_fail_before_model_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_sb3(monkeypatch)
    root = tmp_path / "a2c-bundle"
    digest = save_a2c_inference_bundle(
        root,
        A2CIntentStrategy(
            FakeA2C(),
            feature_indices=(0,),
            feature_names=("signal",),
            fit_metadata=fit_metadata(),
        ),
        feature_names=("signal",),
    )
    (root / "policy.zip").write_bytes(b"changed bytes")

    with pytest.raises(ValueError, match="policy"):
        load_a2c_inference_bundle(
            root,
            expected_digest=digest,
            feature_names=("signal",),
        )

    assert not FakeA2C.load_calls


def test_a2c_artifact_cannot_mislabel_ppo_model_before_creating_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_sb3(monkeypatch)
    root = tmp_path / "mislabelled"
    strategy = A2CIntentStrategy(
        FakePPO(),
        feature_indices=(0,),
        feature_names=("signal",),
        fit_metadata=fit_metadata(),
    )

    with pytest.raises(ValueError, match="A2C policy family"):
        save_a2c_inference_bundle(
            root,
            strategy,
            feature_names=("signal",),
        )

    assert not root.exists()


def test_a2c_bundle_requires_fit_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_sb3(monkeypatch)
    root = tmp_path / "unbound-fit"
    strategy = A2CIntentStrategy(
        FakeA2C(),
        feature_indices=(0,),
        feature_names=("signal",),
    )

    with pytest.raises(ValueError, match="fit metadata"):
        save_a2c_inference_bundle(
            root,
            strategy,
            feature_names=("signal",),
        )

    assert not root.exists()


@pytest.mark.parametrize("bad_timesteps", (None, True, 5.0, 10))
def test_a2c_bundle_rejects_policy_timesteps_different_from_fit_metadata_before_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bad_timesteps: object,
) -> None:
    install_fake_sb3(monkeypatch)
    root = tmp_path / "bad-timesteps"
    strategy = A2CIntentStrategy(
        FakeA2C(bad_timesteps),
        feature_indices=(0,),
        feature_names=("signal",),
        fit_metadata=fit_metadata(),
    )

    with pytest.raises(ValueError, match="timestep|fit metadata"):
        save_a2c_inference_bundle(
            root,
            strategy,
            feature_names=("signal",),
        )

    assert not root.exists()


def test_a2c_bundle_rejects_loaded_policy_timestep_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_sb3(monkeypatch)
    root = tmp_path / "load-mismatch"
    digest = save_a2c_inference_bundle(
        root,
        A2CIntentStrategy(
            FakeA2C(),
            feature_indices=(0,),
            feature_names=("signal",),
            fit_metadata=fit_metadata(),
        ),
        feature_names=("signal",),
    )
    FakeA2C.load_num_timesteps = 10

    with pytest.raises(ValueError, match="timestep|fit metadata"):
        load_a2c_inference_bundle(
            root,
            expected_digest=digest,
            feature_names=("signal",),
        )


@pytest.mark.parametrize("bad_seed", (None, True, 3.0, 4))
def test_a2c_bundle_rejects_policy_seed_different_from_fit_metadata_before_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bad_seed: object,
) -> None:
    install_fake_sb3(monkeypatch)
    root = tmp_path / "bad-seed"
    strategy = A2CIntentStrategy(
        FakeA2C(seed=bad_seed),
        feature_indices=(0,),
        feature_names=("signal",),
        fit_metadata=fit_metadata(),
    )

    with pytest.raises(ValueError, match="seed|fit metadata"):
        save_a2c_inference_bundle(
            root,
            strategy,
            feature_names=("signal",),
        )

    assert not root.exists()


def test_a2c_bundle_rejects_loaded_policy_seed_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_sb3(monkeypatch)
    root = tmp_path / "load-seed-mismatch"
    digest = save_a2c_inference_bundle(
        root,
        A2CIntentStrategy(
            FakeA2C(),
            feature_indices=(0,),
            feature_names=("signal",),
            fit_metadata=fit_metadata(),
        ),
        feature_names=("signal",),
    )
    FakeA2C.load_seed = 4

    with pytest.raises(ValueError, match="seed|fit metadata"):
        load_a2c_inference_bundle(
            root,
            expected_digest=digest,
            feature_names=("signal",),
        )


@pytest.mark.parametrize("bad_n_steps", (None, True, 5.0, 10))
def test_a2c_bundle_rejects_policy_rollout_steps_different_from_fit_metadata_before_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bad_n_steps: object,
) -> None:
    install_fake_sb3(monkeypatch)
    root = tmp_path / "bad-rollout-steps"
    strategy = A2CIntentStrategy(
        FakeA2C(n_steps=bad_n_steps),
        feature_indices=(0,),
        feature_names=("signal",),
        fit_metadata=fit_metadata(),
    )

    with pytest.raises(ValueError, match="rollout|n_steps|fit metadata"):
        save_a2c_inference_bundle(
            root,
            strategy,
            feature_names=("signal",),
        )

    assert not root.exists()


def test_a2c_bundle_rejects_loaded_policy_rollout_step_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_sb3(monkeypatch)
    root = tmp_path / "load-rollout-mismatch"
    digest = save_a2c_inference_bundle(
        root,
        A2CIntentStrategy(
            FakeA2C(),
            feature_indices=(0,),
            feature_names=("signal",),
            fit_metadata=fit_metadata(),
        ),
        feature_names=("signal",),
    )
    FakeA2C.load_n_steps = 10

    with pytest.raises(ValueError, match="rollout|n_steps|fit metadata"):
        load_a2c_inference_bundle(
            root,
            expected_digest=digest,
            feature_names=("signal",),
        )
