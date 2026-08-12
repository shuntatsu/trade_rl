from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from trade_rl.rl.training_heartbeat import build_training_heartbeat_callback


class FakeLogger:
    name_to_value = {"train/loss": 0.25, "ignored": float("nan")}


def test_heartbeat_is_atomic_finite_and_identity_bound(tmp_path: Path) -> None:
    callback = build_training_heartbeat_callback(
        tmp_path / "training-heartbeat.json",
        seed=1,
        algorithm="ppo",
        identity={"runtime_manifest_digest": "a" * 64},
    )
    callback.model = SimpleNamespace(num_timesteps=2048, logger=FakeLogger())
    callback._on_training_start()
    callback._on_rollout_end()

    payload = json.loads(
        (tmp_path / "training-heartbeat.json").read_text(encoding="utf-8")
    )
    assert payload["phase"] == "training"
    assert payload["global_step"] == 2048
    assert payload["runtime_manifest_digest"] == "a" * 64
    assert payload["seed"] == 1
    assert payload["scalars"] == {"train/loss": 0.25}
