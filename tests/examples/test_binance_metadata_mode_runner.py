from __future__ import annotations

import hashlib
import json
import runpy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from trade_rl.integrations.binance import BinanceExchangeInfoSnapshot
from trade_rl.workflows.binance_metadata_modes import (
    BinanceHistoricalSignedScope,
    BinanceMetadataMode,
)

ROOT = Path(__file__).resolve().parents[2]
RUNNER = (
    ROOT
    / "examples"
    / "binance-multitimeframe"
    / "run_full_research_hardened.py"
)
SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT")


def _namespace() -> dict[str, Any]:
    return runpy.run_path(str(RUNNER))


def _snapshot() -> BinanceExchangeInfoSnapshot:
    payload = {
        "symbols": [
            {
                "symbol": symbol,
                "contractStatus": "TRADING",
                "onboardDate": 1_600_000_000_000,
                "filters": [
                    {"filterType": "PRICE_FILTER", "tickSize": "0.1"},
                    {"filterType": "LOT_SIZE", "stepSize": "0.001"},
                    {"filterType": "MIN_NOTIONAL", "notional": "5"},
                ],
            }
            for symbol in SYMBOLS
        ]
    }
    raw = json.dumps(payload).encode("utf-8")
    return BinanceExchangeInfoSnapshot(
        payload=payload,
        raw_payload=raw,
        source_uri="https://fapi.binance.com/fapi/v1/exchangeInfo",
        retrieved_at=datetime(2026, 7, 17, tzinfo=UTC),
        raw_payload_sha256=hashlib.sha256(raw).hexdigest(),
    )


class _Transport:
    def __init__(self) -> None:
        self.calls = 0

    def load_exchange_information_snapshot(
        self, **_: object
    ) -> BinanceExchangeInfoSnapshot:
        self.calls += 1
        return _snapshot()


def test_runner_frozen_mode_does_not_require_signed_history() -> None:
    namespace = _namespace()
    resolve = namespace["_resolve_metadata"]
    resolve.__globals__["_load_signed_history"] = lambda: pytest.fail(
        "frozen mode must not read signed history"
    )
    transport = _Transport()

    result = resolve(
        mode=BinanceMetadataMode.FROZEN_SNAPSHOT,
        transport=transport,
        conservative_static_path=None,
    )

    assert transport.calls == 1
    assert result.mode is BinanceMetadataMode.FROZEN_SNAPSHOT
    assert result.execution_rule_histories is None


def test_runner_historical_mode_preserves_scoped_signed_loader() -> None:
    namespace = _namespace()
    resolve = namespace["_resolve_metadata"]
    rule_type = namespace["InstrumentExecutionRule"]
    start = datetime(2024, 12, 1, tzinfo=UTC)
    end = datetime(2026, 7, 1, tzinfo=UTC)
    rule = rule_type(
        effective_at=start,
        tick_size=0.1,
        lot_size=0.001,
        minimum_notional=5.0,
    )
    metadata = {
        symbol: {
            "listed_at": datetime(2020, 1, 1, tzinfo=UTC).isoformat(),
            "tick_size": 0.1,
            "lot_size": 0.001,
            "minimum_notional": 5.0,
        }
        for symbol in SYMBOLS
    }
    histories = {symbol: (rule,) for symbol in SYMBOLS}
    scope = BinanceHistoricalSignedScope(
        market="usds-m",
        symbols=SYMBOLS,
        coverage_start=start,
        coverage_end=end,
        issued_at=datetime(2026, 7, 17, tzinfo=UTC),
        source_uri="operator://signed-binance-rules",
        payload_digest="a" * 64,
    )
    calls = 0

    def load() -> object:
        nonlocal calls
        calls += 1
        return metadata, histories, scope

    resolve.__globals__["_load_signed_history"] = load

    result = resolve(
        mode=BinanceMetadataMode.HISTORICAL_SIGNED,
        transport=_Transport(),
        conservative_static_path=None,
    )

    assert calls == 1
    assert result.mode is BinanceMetadataMode.HISTORICAL_SIGNED
    assert result.execution_rule_histories == histories
    assert result.identity_evidence["point_in_time"] is True
    assert result.identity_evidence["source_uri"] == scope.source_uri


def test_runner_conservative_mode_requires_explicit_payload() -> None:
    resolve = _namespace()["_resolve_metadata"]

    with pytest.raises(ValueError, match="conservative-static-path"):
        resolve(
            mode=BinanceMetadataMode.CONSERVATIVE_STATIC,
            transport=_Transport(),
            conservative_static_path=None,
        )


def test_runner_source_binds_metadata_and_selection_evidence() -> None:
    content = RUNNER.read_text(encoding="utf-8")

    assert "binance_instrument_rule_history_v3" in content
    assert 'required_purpose="metadata-verification"' in content
    assert "BinanceHistoricalSignedScope" in content
    assert "SelectionAuthorization.create" in content
    assert '"--require-selection-authorization"' in content
    assert '"run_kind"' in content


def test_runner_requires_identity_verified_execution_sensitivity_gate(
    tmp_path: Path,
) -> None:
    namespace = _namespace()
    load_gate = namespace["_SENSITIVITY_GATE"]
    payload = {
        "dataset_id": "a" * 64,
        "experiment_plan_digest": "b" * 64,
        "folds": [],
        "gate": {
            "passed": True,
            "required_scenario": "joint_2x",
            "selected_total_return": 0.01,
            "baseline_uplift": 0.001,
            "maximum_fold_drawdown": 0.1,
        },
        "production_status": "NO-GO",
        "scenario_pack_digest": "c" * 64,
        "schema_version": "execution_sensitivity_v1",
    }
    payload["artifact_digest"] = namespace["content_digest"](payload)
    (tmp_path / "execution-sensitivity.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    (tmp_path / "walk-forward.json").write_text(
        json.dumps(
            {
                "dataset_id": payload["dataset_id"],
                "experiment_plan_digest": payload["experiment_plan_digest"],
                "execution_sensitivity_digest": payload["artifact_digest"],
            }
        ),
        encoding="utf-8",
    )

    passed, gate = load_gate(tmp_path)

    assert passed is True
    assert gate["required_scenario"] == "joint_2x"

    payload["gate"]["selected_total_return"] = -0.1
    (tmp_path / "execution-sensitivity.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    passed, gate = load_gate(tmp_path)
    assert passed is False
    assert "digest" in gate["reason"]


def test_runner_skips_execution_sensitivity_when_not_configured(
    tmp_path: Path,
) -> None:
    load_gate = _namespace()["_SENSITIVITY_GATE"]
    (tmp_path / "walk-forward.json").write_text(
        json.dumps(
            {
                "dataset_id": "a" * 64,
                "experiment_plan_digest": "b" * 64,
                "execution_sensitivity_digest": None,
            }
        ),
        encoding="utf-8",
    )

    passed, gate = load_gate(tmp_path)

    assert passed is True
    assert gate["required"] is False


def test_runner_rejects_execution_sensitivity_identity_mismatch(
    tmp_path: Path,
) -> None:
    namespace = _namespace()
    load_gate = namespace["_SENSITIVITY_GATE"]
    payload = {
        "dataset_id": "a" * 64,
        "experiment_plan_digest": "b" * 64,
        "folds": [],
        "gate": {"passed": True},
        "production_status": "NO-GO",
        "scenario_pack_digest": "c" * 64,
        "schema_version": "execution_sensitivity_v1",
    }
    payload["artifact_digest"] = namespace["content_digest"](payload)
    (tmp_path / "execution-sensitivity.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    (tmp_path / "walk-forward.json").write_text(
        json.dumps(
            {
                "dataset_id": "d" * 64,
                "experiment_plan_digest": payload["experiment_plan_digest"],
                "execution_sensitivity_digest": payload["artifact_digest"],
            }
        ),
        encoding="utf-8",
    )

    passed, gate = load_gate(tmp_path)

    assert passed is False
    assert "dataset_id" in gate["reason"]
