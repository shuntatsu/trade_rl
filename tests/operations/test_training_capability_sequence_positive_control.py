from __future__ import annotations

from pathlib import Path

import pytest

from trade_rl.operations import _training_capability_audit_impl as impl


def test_sequence_positive_control_strengthens_gate_learning_without_relaxing_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, float | int] = {}

    def capture_train(self, *, seed, config, output_path):
        del self, seed, output_path
        observed["behavior_cloning_epochs"] = config.behavior_cloning_epochs
        observed["behavior_cloning_gate_loss_weight"] = (
            config.behavior_cloning_gate_loss_weight
        )
        observed["behavior_cloning_gate_prediction_threshold"] = (
            config.behavior_cloning_gate_prediction_threshold
        )
        raise RuntimeError("captured sequence positive-control config")

    monkeypatch.setattr(impl.StableBaselines3Backend, "train", capture_train)

    with pytest.raises(RuntimeError, match="captured sequence positive-control config"):
        impl._sequence_training(tmp_path)

    assert observed["behavior_cloning_epochs"] == 45
    assert observed["behavior_cloning_gate_loss_weight"] == pytest.approx(4.0)
    assert observed["behavior_cloning_gate_prediction_threshold"] == pytest.approx(0.5)
