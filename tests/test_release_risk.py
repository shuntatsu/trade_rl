import json
from pathlib import Path

import pytest

from mars_lite.pipeline.release_risk import load_release_risk_policy


def _valid_policy() -> dict[str, object]:
    return {
        "max_leverage": 1.0,
        "max_single_weight": 0.20,
        "max_net_exposure": 0.60,
        "max_worst_case_notional": 100_000.0,
        "min_order_notional": 10.0,
        "symbol_liquidity_caps": {"BTC": 50_000.0, "ETH": 30_000.0},
        "forbidden_symbols": [],
    }


def _write(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "risk.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_loads_complete_release_risk_policy(tmp_path: Path) -> None:
    policy = load_release_risk_policy(
        _write(tmp_path, _valid_policy()), symbols=("BTC", "ETH")
    )

    assert policy.max_single_weight == 0.20
    assert policy.symbol_liquidity_caps["BTC"] == 50_000.0
    assert policy.forbidden_symbols == ()


def test_rejects_missing_symbol_cap(tmp_path: Path) -> None:
    payload = _valid_policy()
    payload["symbol_liquidity_caps"] = {"BTC": 50_000.0}

    with pytest.raises(ValueError, match="missing liquidity caps.*ETH"):
        load_release_risk_policy(_write(tmp_path, payload), symbols=("BTC", "ETH"))


def test_rejects_unknown_symbol_cap(tmp_path: Path) -> None:
    payload = _valid_policy()
    payload["symbol_liquidity_caps"] = {
        "BTC": 50_000.0,
        "ETH": 30_000.0,
        "DOGE": 1_000.0,
    }

    with pytest.raises(ValueError, match="unknown liquidity caps.*DOGE"):
        load_release_risk_policy(_write(tmp_path, payload), symbols=("BTC", "ETH"))


def test_rejects_unbounded_single_weight(tmp_path: Path) -> None:
    payload = _valid_policy()
    payload["max_single_weight"] = 1.5

    with pytest.raises(ValueError, match="max_single_weight"):
        load_release_risk_policy(_write(tmp_path, payload), symbols=("BTC", "ETH"))


def test_rejects_net_exposure_above_leverage(tmp_path: Path) -> None:
    payload = _valid_policy()
    payload["max_net_exposure"] = 1.1

    with pytest.raises(ValueError, match="max_net_exposure"):
        load_release_risk_policy(_write(tmp_path, payload), symbols=("BTC", "ETH"))


def test_rejects_missing_required_field(tmp_path: Path) -> None:
    payload = _valid_policy()
    del payload["max_worst_case_notional"]

    with pytest.raises(ValueError, match="missing release risk fields"):
        load_release_risk_policy(_write(tmp_path, payload), symbols=("BTC", "ETH"))


def test_rejects_non_object_document(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="JSON object"):
        load_release_risk_policy(_write(tmp_path, []), symbols=("BTC", "ETH"))


def test_rejects_unknown_forbidden_symbol(tmp_path: Path) -> None:
    payload = _valid_policy()
    payload["forbidden_symbols"] = ["DOGE"]

    with pytest.raises(ValueError, match="unknown forbidden symbols.*DOGE"):
        load_release_risk_policy(_write(tmp_path, payload), symbols=("BTC", "ETH"))
