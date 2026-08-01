from __future__ import annotations

from pathlib import Path


def test_postgres_workflow_tracks_and_executes_live_adapter_regressions() -> None:
    root = Path(__file__).resolve().parents[2]
    workflow = (root / ".github/workflows/postgres-catalog.yml").read_text(
        encoding="utf-8"
    )

    required_paths = (
        "trade_rl/integrations/postgres_indicator_artifacts.py",
        "trade_rl/integrations/postgres_market_dataset.py",
        "tests/integrations/test_postgres_indicator_artifacts.py",
        "tests/integrations/test_postgres_market_dataset.py",
        "tests/integrations/test_postgres_market_dataset_live.py",
    )
    for path in required_paths:
        assert path in workflow, f"PostgreSQL workflow does not track {path}"

    assert "tests/integrations/test_postgres_market_dataset_live.py" in workflow
