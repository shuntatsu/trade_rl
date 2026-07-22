from __future__ import annotations

from pathlib import Path

import pytest

from tests.telemetry.test_training import record
from trade_rl.telemetry import TrainingTelemetryWriter


def test_writer_rejects_append_after_close(tmp_path: Path) -> None:
    path = tmp_path / "training-telemetry.jsonl"
    writer = TrainingTelemetryWriter(path, flush_every=1)
    writer.close()

    with pytest.raises(RuntimeError, match="telemetry writer is closed"):
        writer.append(record(1))

    writer.flush()
    writer.close()
