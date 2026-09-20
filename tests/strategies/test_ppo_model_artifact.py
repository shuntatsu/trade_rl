import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tests.strategies.test_ppo_feature_normalization import _fit
from trade_rl.strategies.rl.ppo import PPOIntentStrategy
from trade_rl.strategies.rl.ppo_artifact import (
    load_normalized_ppo,
    load_ppo_inference_bundle,
    save_normalized_ppo,
    save_ppo_inference_bundle,
)


class Policy:
    loaded = False
    load_device = None
    observation_space = SimpleNamespace(shape=(5,))
    action_space = SimpleNamespace(n=3, start=0)

    def save(self, path):
        Path(path).write_bytes(b"policy bytes")

    @classmethod
    def load(cls, path, *, device="auto"):
        cls.loaded = True
        cls.load_device = device
        return cls()

    def predict(self, observation, *, deterministic=True):
        return np.array(1), None


def _saved(tmp_path, monkeypatch):
    Policy.loaded = False
    Policy.load_device = None
    monkeypatch.setitem(sys.modules, "stable_baselines3", SimpleNamespace(PPO=Policy))
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(set_num_threads=lambda threads: None),
    )
    strategy = PPOIntentStrategy(
        Policy(), feature_indices=(0,), feature_normalizer=_fit()
    )
    root = tmp_path / "policy"
    digest = save_normalized_ppo(root, strategy)
    return root, digest, strategy


def test_saved_model_roundtrip_requires_matching_transform_and_feed_schema(
    tmp_path, monkeypatch
):
    root, digest, strategy = _saved(tmp_path, monkeypatch)
    loaded = load_normalized_ppo(
        root, expected_digest=digest, feature_names=("signal",)
    )
    assert loaded.feature_normalizer == strategy.feature_normalizer
    assert Policy.loaded
    assert Policy.load_device == "cpu"
    Policy.loaded = False
    with pytest.raises(ValueError, match="feature"):
        load_normalized_ppo(root, expected_digest=digest, feature_names=("wrong",))
    assert not Policy.loaded
    with pytest.raises(FileExistsError):
        save_normalized_ppo(root, strategy)


@pytest.mark.parametrize(
    "damage", ["model", "normalizer", "missing_transform", "missing_manifest"]
)
def test_damaged_bundle_is_rejected_before_loading_policy(
    tmp_path, monkeypatch, damage
):
    root, digest, _ = _saved(tmp_path, monkeypatch)
    if damage == "model":
        (root / "policy.zip").write_bytes(b"changed policy")
    elif damage == "missing_manifest":
        (root / "manifest.json").unlink()
    else:
        payload = json.loads((root / "manifest.json").read_bytes())
        if damage == "normalizer":
            payload["normalizer"]["mean"] = [1000.0]
        else:
            del payload["normalizer"]
        (root / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises((ValueError, FileNotFoundError)):
        load_normalized_ppo(root, expected_digest=digest, feature_names=("signal",))
    assert not Policy.loaded


def test_unnormalized_policy_cannot_be_published_as_normalized(tmp_path):
    with pytest.raises(ValueError, match="normalizer"):
        save_normalized_ppo(
            tmp_path / "policy", PPOIntentStrategy(Policy(), feature_indices=(0,))
        )
    assert not (tmp_path / "policy").exists()


def test_raw_ppo_inference_bundle_roundtrip_binds_feed_feature_schema(
    tmp_path, monkeypatch
) -> None:
    Policy.loaded = False
    Policy.load_device = None
    monkeypatch.setitem(sys.modules, "stable_baselines3", SimpleNamespace(PPO=Policy))
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(set_num_threads=lambda threads: None),
    )
    strategy = PPOIntentStrategy(Policy(), feature_indices=(0,))
    root = tmp_path / "raw-policy"

    digest = save_ppo_inference_bundle(
        root,
        strategy,
        feature_names=("signal", "unused"),
    )
    loaded = load_ppo_inference_bundle(
        root,
        expected_digest=digest,
        feature_names=("signal", "unused"),
    )

    assert loaded.feature_indices == (0,)
    assert loaded.feature_normalizer is None
    assert Policy.loaded
    assert Policy.load_device == "cpu"

    Policy.loaded = False
    with pytest.raises(ValueError, match="feature"):
        load_ppo_inference_bundle(
            root,
            expected_digest=digest,
            feature_names=("wrong", "unused"),
        )
    assert not Policy.loaded


def test_normalized_ppo_inference_bundle_roundtrip_preserves_transform(
    tmp_path, monkeypatch
) -> None:
    Policy.loaded = False
    monkeypatch.setitem(sys.modules, "stable_baselines3", SimpleNamespace(PPO=Policy))
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(set_num_threads=lambda threads: None),
    )
    normalizer = _fit()
    strategy = PPOIntentStrategy(
        Policy(),
        feature_indices=(0,),
        feature_normalizer=normalizer,
    )
    root = tmp_path / "normalized-policy"

    digest = save_ppo_inference_bundle(
        root,
        strategy,
        feature_names=("signal",),
    )
    loaded = load_ppo_inference_bundle(
        root,
        expected_digest=digest,
        feature_names=("signal",),
    )

    assert loaded.feature_normalizer == normalizer
    assert loaded.feature_indices == strategy.feature_indices


def test_ppo_inference_bundle_rejects_tampering_before_policy_load(
    tmp_path, monkeypatch
) -> None:
    Policy.loaded = False
    monkeypatch.setitem(sys.modules, "stable_baselines3", SimpleNamespace(PPO=Policy))
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(set_num_threads=lambda threads: None),
    )
    strategy = PPOIntentStrategy(Policy(), feature_indices=(0,))
    root = tmp_path / "policy"
    digest = save_ppo_inference_bundle(
        root,
        strategy,
        feature_names=("signal",),
    )

    (root / "policy.zip").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="policy"):
        load_ppo_inference_bundle(
            root,
            expected_digest=digest,
            feature_names=("signal",),
        )
    assert not Policy.loaded


def test_ppo_inference_bundle_is_write_once_and_requires_matching_selected_names(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setitem(sys.modules, "stable_baselines3", SimpleNamespace(PPO=Policy))
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(set_num_threads=lambda threads: None),
    )
    strategy = PPOIntentStrategy(Policy(), feature_indices=(0,))
    root = tmp_path / "policy"

    save_ppo_inference_bundle(
        root,
        strategy,
        feature_names=("signal",),
    )

    with pytest.raises(FileExistsError):
        save_ppo_inference_bundle(
            root,
            strategy,
            feature_names=("signal",),
        )
    with pytest.raises(ValueError, match="feature"):
        save_ppo_inference_bundle(
            tmp_path / "bad-policy",
            strategy,
            feature_names=(),
        )
