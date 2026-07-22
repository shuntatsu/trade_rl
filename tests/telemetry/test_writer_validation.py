from __future__ import annotations

from pathlib import Path

import pytest

from trade_rl.telemetry import TrainingTelemetryWriter


@pytest.mark.parametrize("flush_every", (0, -1, True))
def test_writer_rejects_non_positive_or_boolean_flush_interval(
    tmp_path: Path,
    flush_every: int,
) -> None:
    with pytest.raises(ValueError, match="flush_every must be positive"):
        TrainingTelemetryWriter(
            tmp_path / "training-telemetry.jsonl",
            flush_every=flush_every,
        )
