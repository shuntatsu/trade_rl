from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from trade_rl.operations.cuda_training_smoke import (
    TINY_TRAINING_UPDATES,
    main,
    run_tiny_training_smoke,
)


def test_tiny_training_smoke_executes_exactly_three_cpu_updates() -> None:
    result = run_tiny_training_smoke(device="cpu", require_cuda=False)

    assert result.updates == TINY_TRAINING_UPDATES == 3
    assert result.device_type == "cpu"
    assert result.dtype == "float64"
    assert len(result.losses) == 3
    assert len(result.gradient_norms) == 3
    assert all(math.isfinite(value) for value in result.losses)
    assert all(value > 0.0 and math.isfinite(value) for value in result.gradient_norms)
    assert result.parameter_delta_l2 > 0.0
    assert result.peak_allocated_bytes is None
    assert result.peak_reserved_bytes is None


def test_tiny_training_smoke_rejects_cpu_when_cuda_is_required() -> None:
    with pytest.raises(RuntimeError, match="CUDA device is required"):
        run_tiny_training_smoke(device="cpu", require_cuda=True)


def test_tiny_training_smoke_cli_writes_canonical_json(tmp_path: Path) -> None:
    output = tmp_path / "tiny-training-smoke.json"

    exit_code = main(
        [
            "--device",
            "cpu",
            "--allow-cpu",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    raw = output.read_text(encoding="utf-8")
    assert raw.endswith("\n")
    payload = json.loads(raw)
    assert payload["schema_version"] == "tiny_cuda_training_smoke_v1"
    assert payload["updates"] == 3
    assert payload["device_type"] == "cpu"
    assert payload["parameter_delta_l2"] > 0.0
