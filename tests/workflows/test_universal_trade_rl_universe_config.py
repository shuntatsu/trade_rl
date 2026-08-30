from __future__ import annotations

import json
from pathlib import Path

import pytest

from trade_rl.workflows.universal_trade_rl_universe_config import (
    UniversalTradeRLSymbolSource,
    load_universal_trade_rl_source_catalog,
    load_universal_trade_rl_universe_config,
    universal_trade_rl_source_catalog_digest,
)

_ROOT = Path(__file__).resolve().parents[2]


def _write(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _config_payload() -> dict[str, object]:
    return {
        "schema_version": "universal_trade_rl_universe_config_v1",
        "train_symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        "development_symbols": ["LINKUSDT"],
        "admission_symbols": ["AVAXUSDT"],
        "excluded_symbols": [
            {
                "symbol": "LUNA2USDT",
                "reason": "insufficient_contiguous_history",
            }
        ],
    }


def _source(symbol: str, char: str) -> dict[str, object]:
    return {
        "symbol": symbol,
        "dataset_digest": char * 64,
        "first_timestamp_ns": 1,
        "last_timestamp_ns": 100,
        "row_count": 100,
    }


def _catalog_payload() -> dict[str, object]:
    return {
        "schema_version": "universal_trade_rl_source_catalog_v1",
        "symbols": [
            _source("AVAXUSDT", "a"),
            _source("BTCUSDT", "b"),
            _source("ETHUSDT", "c"),
            _source("LINKUSDT", "d"),
            _source("LUNA2USDT", "e"),
            _source("SOLUSDT", "f"),
        ],
    }


def test_load_universe_config_round_trips_strict_roles(tmp_path: Path) -> None:
    config = load_universal_trade_rl_universe_config(
        _write(tmp_path / "universe.json", _config_payload())
    )

    assert config.train_symbols == ("BTCUSDT", "ETHUSDT", "SOLUSDT")
    assert config.development_symbols == ("LINKUSDT",)
    assert config.admission_symbols == ("AVAXUSDT",)
    assert tuple(item.symbol for item in config.exclusions) == ("LUNA2USDT",)


def test_config_rejects_unknown_root_keys(tmp_path: Path) -> None:
    payload = _config_payload()
    payload["unexpected"] = True

    with pytest.raises(ValueError, match="exact keys"):
        load_universal_trade_rl_universe_config(
            _write(tmp_path / "universe.json", payload)
        )


def test_config_rejects_unknown_exclusion_keys(tmp_path: Path) -> None:
    payload = _config_payload()
    exclusion = dict(payload["excluded_symbols"][0])  # type: ignore[index]
    exclusion["unexpected"] = True
    payload["excluded_symbols"] = [exclusion]

    with pytest.raises(ValueError, match="exact keys"):
        load_universal_trade_rl_universe_config(
            _write(tmp_path / "universe.json", payload)
        )


def test_config_rejects_non_list_role_values(tmp_path: Path) -> None:
    payload = _config_payload()
    payload["train_symbols"] = "BTCUSDT"

    with pytest.raises(ValueError, match="array"):
        load_universal_trade_rl_universe_config(
            _write(tmp_path / "universe.json", payload)
        )


def test_load_source_catalog_returns_sorted_immutable_records(tmp_path: Path) -> None:
    records = load_universal_trade_rl_source_catalog(
        _write(tmp_path / "catalog.json", _catalog_payload())
    )

    assert all(isinstance(item, UniversalTradeRLSymbolSource) for item in records)
    assert tuple(item.symbol for item in records) == (
        "AVAXUSDT",
        "BTCUSDT",
        "ETHUSDT",
        "LINKUSDT",
        "LUNA2USDT",
        "SOLUSDT",
    )
    assert len(universal_trade_rl_source_catalog_digest(records)) == 64


def test_catalog_rejects_unsorted_symbols(tmp_path: Path) -> None:
    payload = _catalog_payload()
    payload["symbols"] = [
        _source("ETHUSDT", "a"),
        _source("BTCUSDT", "b"),
    ]

    with pytest.raises(ValueError, match="sorted"):
        load_universal_trade_rl_source_catalog(
            _write(tmp_path / "catalog.json", payload)
        )


def test_catalog_rejects_duplicate_symbols(tmp_path: Path) -> None:
    payload = _catalog_payload()
    payload["symbols"] = [
        _source("BTCUSDT", "a"),
        _source("BTCUSDT", "b"),
    ]

    with pytest.raises(ValueError, match="unique"):
        load_universal_trade_rl_source_catalog(
            _write(tmp_path / "catalog.json", payload)
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("first_timestamp_ns", True, "integer"),
        ("last_timestamp_ns", True, "integer"),
        ("row_count", True, "integer"),
        ("first_timestamp_ns", -1, "non-negative"),
        ("row_count", 0, "positive"),
    ),
)
def test_catalog_rejects_invalid_integer_fields(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    payload = _catalog_payload()
    record = dict(payload["symbols"][0])  # type: ignore[index]
    record[field] = value
    payload["symbols"] = [record]

    with pytest.raises(ValueError, match=message):
        load_universal_trade_rl_source_catalog(
            _write(tmp_path / "catalog.json", payload)
        )


def test_catalog_rejects_non_increasing_time_range(tmp_path: Path) -> None:
    payload = _catalog_payload()
    record = dict(payload["symbols"][0])  # type: ignore[index]
    record["last_timestamp_ns"] = record["first_timestamp_ns"]
    payload["symbols"] = [record]

    with pytest.raises(ValueError, match="later"):
        load_universal_trade_rl_source_catalog(
            _write(tmp_path / "catalog.json", payload)
        )


def test_catalog_rejects_invalid_dataset_digest(tmp_path: Path) -> None:
    payload = _catalog_payload()
    record = dict(payload["symbols"][0])  # type: ignore[index]
    record["dataset_digest"] = "not-a-digest"
    payload["symbols"] = [record]

    with pytest.raises(ValueError, match="digest"):
        load_universal_trade_rl_source_catalog(
            _write(tmp_path / "catalog.json", payload)
        )


def test_catalog_rejects_unknown_record_keys(tmp_path: Path) -> None:
    payload = _catalog_payload()
    record = dict(payload["symbols"][0])  # type: ignore[index]
    record["unexpected"] = 1
    payload["symbols"] = [record]

    with pytest.raises(ValueError, match="exact keys"):
        load_universal_trade_rl_source_catalog(
            _write(tmp_path / "catalog.json", payload)
        )


def test_source_catalog_digest_changes_with_source_identity() -> None:
    base = (
        UniversalTradeRLSymbolSource(
            symbol="BTCUSDT",
            dataset_digest="a" * 64,
            first_timestamp_ns=1,
            last_timestamp_ns=100,
            row_count=100,
        ),
    )
    changed = (
        UniversalTradeRLSymbolSource(
            symbol="BTCUSDT",
            dataset_digest="b" * 64,
            first_timestamp_ns=1,
            last_timestamp_ns=100,
            row_count=100,
        ),
    )

    assert universal_trade_rl_source_catalog_digest(base) != (
        universal_trade_rl_source_catalog_digest(changed)
    )


def test_authored_example_inputs_are_strictly_loadable() -> None:
    config = load_universal_trade_rl_universe_config(
        _ROOT / "examples/binance/universal-trade-rl-universe.example.json"
    )
    sources = load_universal_trade_rl_source_catalog(
        _ROOT / "examples/binance/universal-trade-rl-source-catalog.example.json"
    )

    assert config.train_symbols == ("BTCUSDT", "ETHUSDT", "SOLUSDT")
    assert config.development_symbols == ("LINKUSDT",)
    assert config.admission_symbols == ("AVAXUSDT",)
    assert tuple(item.symbol for item in sources) == (
        "AVAXUSDT",
        "BTCUSDT",
        "ETHUSDT",
        "LINKUSDT",
        "LUNA2USDT",
        "SOLUSDT",
    )
