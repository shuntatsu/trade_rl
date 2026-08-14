from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_gpu_control_syncs_market_data_before_supervised_start() -> None:
    workflow = (
        ROOT / ".github" / "workflows" / "launch-binance-frozen-226.yml"
    ).read_text(encoding="utf-8")

    sync = workflow.index(
        "docker compose -f docker/compose.training.yaml run --rm market-data-sync"
    )
    supervised = workflow.index("full_run_supervisor.py start")

    assert sync < supervised
