from __future__ import annotations

from pathlib import Path

import pytest
import torch

from trade_rl.operations import _training_capability_audit_impl as impl


_ORIGINAL_CONFIG = impl.ResidualTrainingConfig


def _config_override(**overrides: object):
    def build(*args: object, **kwargs: object):
        kwargs.update(overrides)
        return _ORIGINAL_CONFIG(*args, **kwargs)

    return build


def test_sequence_capability_with_single_torch_thread(tmp_path: Path) -> None:
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        record = impl._sequence_training(tmp_path / "single-thread")
    finally:
        torch.set_num_threads(previous_threads)

    assert record["status"] == "pass"


def test_sequence_capability_without_validation_split(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        impl,
        "ResidualTrainingConfig",
        _config_override(behavior_cloning_validation_fraction=0.0),
    )

    record = impl._sequence_training(tmp_path / "no-validation")

    assert record["status"] == "pass"
