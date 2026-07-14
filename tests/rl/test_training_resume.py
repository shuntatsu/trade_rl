from __future__ import annotations

from pathlib import Path

import pytest

from trade_rl.rl.replay import load_replay_buffer_artifact, write_replay_buffer_artifact


def test_replay_buffer_artifact_round_trip_and_tamper_detection(tmp_path: Path) -> None:
    source = tmp_path / "source.pkl"
    source.write_bytes(b"replay-state")
    artifact = tmp_path / "artifact"

    manifest = write_replay_buffer_artifact(
        artifact,
        source=source,
        algorithm="sac",
        environment_digest="a" * 64,
        training_config_digest="b" * 64,
        timesteps=123,
    )
    loaded, replay_path = load_replay_buffer_artifact(artifact)

    assert loaded == manifest
    assert replay_path.read_bytes() == b"replay-state"

    replay_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="digest|size"):
        load_replay_buffer_artifact(artifact)
