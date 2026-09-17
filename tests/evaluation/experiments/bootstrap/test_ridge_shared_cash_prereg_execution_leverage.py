from __future__ import annotations

import json

import pytest

from trade_rl.evaluation.experiments.bootstrap.ridge_shared_cash_prereg import (
    canonical_ridge_shared_cash_protocol,
    load_ridge_shared_cash_protocol,
)


def test_protocol_freezes_execution_max_leverage() -> None:
    protocol = canonical_ridge_shared_cash_protocol()
    assert protocol.execution_max_leverage == 1.0


def test_loader_round_trips_execution_max_leverage_and_rejects_drift(tmp_path) -> None:
    protocol = canonical_ridge_shared_cash_protocol()
    payload = protocol.to_payload()
    assert payload["execution_max_leverage"] == 1.0

    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
    loaded = load_ridge_shared_cash_protocol(path)
    assert loaded.execution_max_leverage == 1.0

    payload["execution_max_leverage"] = 0.5
    path.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
    with pytest.raises(ValueError, match="preregistered"):
        load_ridge_shared_cash_protocol(path)


@pytest.mark.parametrize("bad", (True, 0.0, -1.0, float("nan")))
def test_protocol_rejects_invalid_execution_max_leverage(bad: object) -> None:
    protocol = canonical_ridge_shared_cash_protocol()
    payload = protocol.to_payload()
    payload["execution_max_leverage"] = bad
    # constructor/loader share the strict numeric boundary; bool is not a float alias.
    from dataclasses import replace

    with pytest.raises(ValueError):
        replace(protocol, execution_max_leverage=bad)  # type: ignore[arg-type]
