from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.selection import PolicyMode
from trade_rl.integrations.sb3_serving import StableBaselines3PolicyLoader
from trade_rl.rl.actions import ACTION_SCHEMA
from trade_rl.rl.observations import OBSERVATION_SCHEMA
from trade_rl.serving.bundle import (
    ServingBundleManifest,
    load_serving_bundle,
    write_serving_bundle_manifest,
)


class FakeModel:
    def __init__(self, action: np.ndarray) -> None:
        self.action = action

    def predict(self, observation, deterministic=True):
        assert deterministic is True
        assert np.asarray(observation).shape == (5,)
        return self.action.copy(), None


def _bundle(root: Path, member_count: int = 2):
    root.mkdir()
    members: list[str] = []
    for index in range(member_count):
        relative = f"members/member-{index:03d}/policy.zip"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"member-{index}".encode())
        members.append(relative)
    loader_payload = {
        "algorithm": "ppo",
        "members": members,
        "schema_version": "sb3_policy_loader_v1",
    }
    (root / "policy-loader.json").write_text(
        json.dumps(loader_payload), encoding="utf-8"
    )
    action_names = ("fast_tilt", "slow_tilt", "risk_tilt", "alpha_scale")
    action_digest = content_digest({"names": action_names})
    manifest = ServingBundleManifest.build(
        root=root,
        dataset_id="a" * 64,
        action_schema=ACTION_SCHEMA,
        action_size=4,
        action_names=action_names,
        action_spec_digest=action_digest,
        observation_schema=OBSERVATION_SCHEMA,
        observation_size=5,
        environment_digest="b" * 64,
        initial_capital=100_000.0,
        policy_mode=PolicyMode.RESIDUAL_POLICY,
        policy_digest="c" * 64,
        signal_digest="d" * 64,
        selection_digest="e" * 64,
        release_digest=None,
        normalizer_digest=None,
        artifact_paths=tuple([*members, "policy-loader.json"]),
        created_at=datetime(2026, 7, 14, tzinfo=UTC),
    )
    write_serving_bundle_manifest(root, manifest)
    return load_serving_bundle(root)


def test_sb3_loader_averages_all_dynamic_action_members(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "bundle")
    actions = {
        "member-000": np.array([0.2, -0.2, 0.4, 0.0], dtype=np.float32),
        "member-001": np.array([0.4, 0.2, 0.0, 0.6], dtype=np.float32),
    }

    def load(algorithm: str, path: Path):
        assert algorithm == "ppo"
        return FakeModel(actions[path.parent.name])

    policy = StableBaselines3PolicyLoader(model_loader=load).load(bundle)

    np.testing.assert_allclose(
        policy.predict(np.zeros(5, dtype=np.float32)),
        np.array([0.3, 0.0, 0.2, 0.3], dtype=np.float32),
    )


def test_sb3_loader_rejects_member_not_declared_by_bundle(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "bundle", member_count=1)
    loader_path = bundle.root / "policy-loader.json"
    payload = json.loads(loader_path.read_text(encoding="utf-8"))
    payload["members"].append("members/member-999/policy.zip")
    loader_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="bundle|member"):
        StableBaselines3PolicyLoader(model_loader=lambda algorithm, path: None).load(
            load_serving_bundle(bundle.root)
        )


def test_sb3_ensemble_prediction_fails_closed_on_bad_member(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "bundle")
    models = iter(
        (
            FakeModel(np.zeros(4, dtype=np.float32)),
            FakeModel(np.array([0.0, np.nan, 0.0, 0.0], dtype=np.float32)),
        )
    )
    policy = StableBaselines3PolicyLoader(
        model_loader=lambda algorithm, path: next(models)
    ).load(bundle)

    with pytest.raises(ValueError, match="finite|action"):
        policy.predict(np.zeros(5, dtype=np.float32))
