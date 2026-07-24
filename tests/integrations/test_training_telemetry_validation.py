from __future__ import annotations

from pathlib import Path

import pytest

from trade_rl.integrations.training_telemetry import TrainingTelemetrySampler


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"seed": -1}, "seed must be non-negative"),
        ({"seed": 1, "sample_every": 0}, "sample_every must be positive"),
        (
            {"seed": 1, "position_threshold": float("nan")},
            "position_threshold must be finite and non-negative",
        ),
    ],
)
def test_sampler_rejects_invalid_configuration(
    tmp_path: Path,
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        TrainingTelemetrySampler(
            tmp_path / "training-telemetry.jsonl",
            **kwargs,
        )
