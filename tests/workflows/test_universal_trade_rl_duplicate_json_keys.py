from __future__ import annotations

from pathlib import Path

import pytest

from trade_rl.workflows.universal_trade_rl_universe_config import (
    load_universal_trade_rl_source_catalog,
    load_universal_trade_rl_universe_config,
)


def test_universe_config_rejects_duplicate_json_object_key(tmp_path: Path) -> None:
    path = tmp_path / "universe.json"
    path.write_text(
        """{
  "schema_version": "universal_trade_rl_universe_config_v1",
  "train_symbols": ["ETHUSDT"],
  "train_symbols": ["BTCUSDT"],
  "development_symbols": ["LINKUSDT"],
  "admission_symbols": ["AVAXUSDT"],
  "excluded_symbols": []
}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_universal_trade_rl_universe_config(path)


def test_source_catalog_rejects_duplicate_nested_json_key(tmp_path: Path) -> None:
    path = tmp_path / "catalog.json"
    path.write_text(
        """{
  "schema_version": "universal_trade_rl_source_catalog_v1",
  "symbols": [
    {
      "symbol": "BTCUSDT",
      "dataset_digest": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      "first_timestamp_ns": 1,
      "last_timestamp_ns": 100,
      "row_count": 0,
      "row_count": 100
    }
  ]
}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_universal_trade_rl_source_catalog(path)
