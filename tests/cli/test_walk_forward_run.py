from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

from trade_rl.cli import extended
from trade_rl.workflows.market_walk_forward import WalkForwardRunResult


def test_walk_forward_run_command_emits_published_result(
    monkeypatch, tmp_path: Path
) -> None:
    published = tmp_path / "runs" / "wf-001"

    def fake_execute(**kwargs):
        assert kwargs == {
            "config_path": Path("walk-forward.json"),
            "dataset_path": Path("dataset"),
            "store_root": Path("artifacts"),
            "run_id": "wf-001",
        }
        return WalkForwardRunResult(
            run_id="wf-001",
            status="published",
            path=published,
            run_digest="a" * 64,
            evaluation_digest="b" * 64,
            dataset_id="c" * 64,
        )

    monkeypatch.setattr(extended, "execute_market_walk_forward", fake_execute)
    stdout = StringIO()
    stderr = StringIO()

    exit_code = extended.main(
        [
            "walk-forward",
            "run",
            "--config",
            "walk-forward.json",
            "--dataset",
            "dataset",
            "--output",
            "artifacts",
            "--run-id",
            "wf-001",
        ],
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 0
    assert stderr.getvalue() == ""
    assert json.loads(stdout.getvalue()) == {
        "artifact_path": str(published),
        "dataset_id": "c" * 64,
        "evaluation_digest": "b" * 64,
        "production_status": "NO-GO",
        "run_digest": "a" * 64,
        "run_id": "wf-001",
        "schema": "walk_forward_run_result_v1",
        "status": "published",
    }
