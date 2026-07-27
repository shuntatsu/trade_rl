from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("path", "expected"),
    (
        ("examples/binance/walk-forward-smoke.json", "local_exploratory"),
        (
            "examples/binance-multitimeframe/walk-forward-full.json",
            "durable_postgres",
        ),
        (
            "examples/binance-multitimeframe/walk-forward-growth-optimal.json",
            "durable_postgres",
        ),
    ),
)
def test_maintained_configs_pin_sealed_ledger_mode(
    path: str,
    expected: str,
) -> None:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    assert payload["sealed_test_ledger_mode"] == expected
