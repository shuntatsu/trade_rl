from __future__ import annotations

import json
from pathlib import Path

import pytest

from trade_rl.workflows.universal_causal_alpha_v7_checkpoint import (
    CausalAlphaV7CheckpointWriter,
)


def _digest(char: str) -> str:
    return char * 64


def test_v7_checkpoint_is_hash_chained_and_durable(tmp_path: Path) -> None:
    writer = CausalAlphaV7CheckpointWriter(
        tmp_path,
        run_manifest_digest=_digest("a"),
        config_digest=_digest("b"),
        generator_code_digest=_digest("c"),
    )
    first = writer.append(
        stage="signal",
        cutoff=100,
        diagnostics={"candidate": "v6_control", "net_log_return": -0.01},
    )
    second = writer.append(
        stage="selection",
        cutoff=200,
        diagnostics={"candidate": "causal_calibrated", "net_log_return": 0.02},
    )

    lines = [json.loads(line) for line in writer.path.read_text().splitlines()]
    assert len(lines) == 3
    assert lines[1]["artifact_digest"] == first
    assert lines[2]["previous_digest"] == first
    assert lines[2]["artifact_digest"] == second
    assert lines[2]["sequence"] == 2


def test_v7_checkpoint_refuses_existing_or_out_of_order_cutoff(tmp_path: Path) -> None:
    writer = CausalAlphaV7CheckpointWriter(
        tmp_path,
        run_manifest_digest=_digest("a"),
        config_digest=_digest("b"),
        generator_code_digest=_digest("c"),
    )
    writer.append(stage="selection", cutoff=200, diagnostics={"x": 1})
    with pytest.raises(ValueError, match="strictly increase"):
        writer.append(stage="selection", cutoff=200, diagnostics={"x": 2})
    with pytest.raises(FileExistsError):
        CausalAlphaV7CheckpointWriter(
            tmp_path,
            run_manifest_digest=_digest("a"),
            config_digest=_digest("b"),
            generator_code_digest=_digest("c"),
        )
