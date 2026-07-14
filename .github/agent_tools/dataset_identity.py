from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEST_PATH = ROOT / "tests/data/test_dataset_content_identity.py"


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one match in {path}, found {count}: {old[:100]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def write_tests() -> None:
    TEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEST_PATH.write_text(
        '''from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from trade_rl.data import load_market_dataset_artifact, write_market_dataset_files
from trade_rl.data.market import MarketDataset


def dataset(*, fee_rate: float = 0.0) -> MarketDataset:
    n_bars = 8
    close = np.full((n_bars, 1), 100.0)
    return MarketDataset(
        dataset_id="0" * 64,
        symbols=("ASSET",),
        timestamps=np.datetime64("2026-01-01T00:00:00", "ns")
        + np.arange(n_bars) * np.timedelta64(1, "h"),
        features=np.zeros((n_bars, 1, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=close,
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=np.full((n_bars, 1), 1_000.0),
        funding_rate=np.zeros((n_bars, 1)),
        tradable=np.ones((n_bars, 1), dtype=np.bool_),
        feature_available=np.ones((n_bars, 1, 1), dtype=np.bool_),
        feature_names=("feature",),
        global_feature_names=("global",),
        periods_per_year=8_760,
        fee_rate=np.full((n_bars, 1), fee_rate),
    )


def test_execution_semantics_are_part_of_dataset_identity() -> None:
    zero_fee = dataset(fee_rate=0.0).with_content_identity()
    charged = dataset(fee_rate=0.001).with_content_identity()
    assert zero_fee.dataset_id != charged.dataset_id


def test_formal_artifact_rejects_unidentified_dataset(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="content identity"):
        write_market_dataset_files(tmp_path, dataset())


def test_content_identity_round_trip_recomputes_all_arrays(tmp_path: Path) -> None:
    original = dataset(fee_rate=0.001).with_content_identity(
        {"source": "unit-test"}
    )
    write_market_dataset_files(tmp_path, original)
    restored = load_market_dataset_artifact(tmp_path)
    assert restored.dataset_id == original.dataset_id
    assert restored.identity_payload_json == original.identity_payload_json

    with pytest.raises(ValueError, match="dataset_id"):
        replace(
            restored,
            dataset_id="f" * 64,
            fee_rate=np.full((restored.n_bars, restored.n_symbols), 0.002),
        )
''',
        encoding="utf-8",
    )


def apply_implementation() -> None:
    identity = ROOT / "trade_rl/data/identity.py"
    text = identity.read_text(encoding="utf-8")
    start = text.index("DATASET_ID_ARRAY_FIELDS = (")
    end = text.index("\n)\n\n\n", start) + 3
    fields = '''DATASET_ID_ARRAY_FIELDS = (
    "timestamps",
    "available_at",
    "information_available",
    "features",
    "feature_available",
    "feature_staleness",
    "feature_staleness_hours",
    "feature_missing_reason",
    "global_features",
    "global_feature_available",
    "global_feature_staleness_hours",
    "global_feature_missing_reason",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "funding_rate",
    "tradable",
    "symbol_active",
    "fee_rate",
    "maker_fee_rate",
    "taker_fee_rate",
    "spread_rate",
    "max_participation_rate",
    "minimum_notional",
    "lot_size",
    "tick_size",
    "borrow_available",
    "borrow_rate",
    "funding_due",
    "buy_allowed",
    "sell_allowed",
    "mark_price",
    "index_price",
    "dividend",
    "split_factor",
    "delisting_recovery",
    "cash_rate",
    "contract_multipliers",
)'''
    identity.write_text(text[:start] + fields + text[end:], encoding="utf-8")

    replace_once(
        "trade_rl/data/market.py",
        "from dataclasses import dataclass, field\n",
        "from collections.abc import Mapping\nfrom dataclasses import dataclass, field, replace\n",
    )
    replace_once(
        "trade_rl/data/market.py",
        '''        identity_payload_json = self.identity_payload_json
        if identity_payload_json is not None:
            payload = parse_identity_json(identity_payload_json)
            canonical_payload_json = canonical_identity_json(payload)
            resolved_id = compute_market_dataset_id(
                payload,
                {
                    "timestamps": timestamps,
                    "available_at": available_at,
                    "information_available": information_available,
                    "features": features,
                    "global_features": global_features,
                    "global_feature_available": global_available,
                    "global_feature_staleness_hours": global_staleness,
                    "global_feature_missing_reason": global_missing_reason,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                    "funding_rate": funding,
                    "tradable": tradable,
                    "symbol_active": asset_active,
                    "feature_available": feature_available,
                    "feature_staleness": feature_staleness,
                },
            )
            if resolved_id != self.dataset_id:
                raise ValueError(
                    "dataset_id does not match identity payload and arrays"
                )
            identity_payload_json = canonical_payload_json

''',
        '''        identity_payload_json = self.identity_payload_json

''',
    )
    replace_once(
        "trade_rl/data/market.py",
        '        object.__setattr__(self, "identity_payload_json", identity_payload_json)\n',
        "",
    )
    replace_once(
        "trade_rl/data/market.py",
        '''        object.__setattr__(self, "_nominal_bar_hours", nominal_bar_hours)

    def resolved_array(self, field_name: str) -> np.ndarray:
''',
        '''        object.__setattr__(self, "_nominal_bar_hours", nominal_bar_hours)
        if identity_payload_json is not None:
            payload = parse_identity_json(identity_payload_json)
            canonical_payload_json = canonical_identity_json(payload)
            resolved_id = compute_market_dataset_id(payload, self.identity_arrays())
            if resolved_id != self.dataset_id:
                raise ValueError(
                    "dataset_id does not match identity payload and arrays"
                )
            identity_payload_json = canonical_payload_json
        object.__setattr__(self, "identity_payload_json", identity_payload_json)

    def identity_contract_payload(self) -> dict[str, object]:
        return {
            "calendar_kind": self.calendar_kind.value,
            "feature_config_digest": self.feature_config_digest,
            "feature_names": self.feature_names,
            "global_feature_names": self.global_feature_names,
            "nominal_bar_hours": self._nominal_bar_hours,
            "normalization_digest": self.normalization_digest,
            "periods_per_year": self.periods_per_year,
            "symbols": self.symbols,
            "volume_units": tuple(value.value for value in self.volume_units),
        }

    def identity_arrays(self) -> dict[str, np.ndarray]:
        return {
            "timestamps": self.timestamps,
            "available_at": self.resolved_array("available_at"),
            "information_available": self.resolved_array("information_available"),
            "features": self.features,
            "feature_available": self.feature_available,
            "feature_staleness": self.resolved_array("feature_staleness"),
            "feature_staleness_hours": self.resolved_array(
                "feature_staleness_hours"
            ),
            "feature_missing_reason": self.resolved_array("feature_missing_reason"),
            "global_features": self.global_features,
            "global_feature_available": self.resolved_array(
                "global_feature_available"
            ),
            "global_feature_staleness_hours": self.resolved_array(
                "global_feature_staleness_hours"
            ),
            "global_feature_missing_reason": self.resolved_array(
                "global_feature_missing_reason"
            ),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "funding_rate": self.funding_rate,
            "tradable": self.tradable,
            "symbol_active": self.resolved_array("symbol_active"),
            "fee_rate": self.resolved_array("fee_rate"),
            "maker_fee_rate": self.resolved_array("maker_fee_rate"),
            "taker_fee_rate": self.resolved_array("taker_fee_rate"),
            "spread_rate": self.resolved_array("spread_rate"),
            "max_participation_rate": self.resolved_array(
                "max_participation_rate"
            ),
            "minimum_notional": self.resolved_array("minimum_notional"),
            "lot_size": self.resolved_array("lot_size"),
            "tick_size": self.resolved_array("tick_size"),
            "borrow_available": self.resolved_array("borrow_available"),
            "borrow_rate": self.resolved_array("borrow_rate"),
            "funding_due": self.resolved_array("funding_due"),
            "buy_allowed": self.resolved_array("buy_allowed"),
            "sell_allowed": self.resolved_array("sell_allowed"),
            "mark_price": self.resolved_array("mark_price"),
            "index_price": self.resolved_array("index_price"),
            "dividend": self.resolved_array("dividend"),
            "split_factor": self.resolved_array("split_factor"),
            "delisting_recovery": self.resolved_array("delisting_recovery"),
            "cash_rate": self.resolved_array("cash_rate"),
            "contract_multipliers": self.resolved_array("contract_multipliers"),
        }

    def with_content_identity(
        self,
        provenance: Mapping[str, object] | None = None,
    ) -> MarketDataset:
        payload = dict(provenance or {})
        payload["schema"] = "market_dataset_identity_v5"
        payload["dataset_contract"] = self.identity_contract_payload()
        dataset_id = compute_market_dataset_id(payload, self.identity_arrays())
        return replace(
            self,
            dataset_id=dataset_id,
            identity_payload_json=canonical_identity_json(payload),
        )

    def resolved_array(self, field_name: str) -> np.ndarray:
''',
    )

    replace_once(
        "trade_rl/data/builder.py",
        '''from trade_rl.data.identity import (
    MARKET_DATASET_IDENTITY_SCHEMA,
    canonical_identity_json,
    compute_market_dataset_id,
    content_and_arrays_digest,
)
''',
        '''from trade_rl.data.identity import (
    MARKET_DATASET_IDENTITY_SCHEMA,
    content_and_arrays_digest,
)
''',
    )
    replace_once(
        "trade_rl/data/builder.py",
        '''        dataset_id = compute_market_dataset_id(
            metadata,
            {
                "timestamps": timestamps,
                "available_at": available_at,
                "information_available": information_available,
                "features": features,
                "global_features": global_features,
                "global_feature_available": global_feature_available,
                "global_feature_staleness_hours": global_feature_staleness,
                "global_feature_missing_reason": global_feature_missing_reason,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "funding_rate": funding_rate,
                "tradable": tradable,
                "symbol_active": symbol_active,
                "feature_available": feature_available,
                "feature_staleness": feature_staleness,
            },
        )
        periods_per_year = int(round(365.0 * 24.0 / self.config.bar_hours))
        return MarketDataset(
            dataset_id=dataset_id,
''',
        '''        periods_per_year = int(round(365.0 * 24.0 / self.config.bar_hours))
        return MarketDataset(
            dataset_id="0" * 64,
''',
    )
    replace_once(
        "trade_rl/data/builder.py",
        '''            normalization_digest=normalization_digest,
            identity_payload_json=canonical_identity_json(metadata),
            periods_per_year=periods_per_year,
        )
''',
        '''            normalization_digest=normalization_digest,
            periods_per_year=periods_per_year,
        ).with_content_identity(metadata)
''',
    )

    replace_once(
        "trade_rl/data/artifact_codec.py",
        '''    root.mkdir(parents=True, exist_ok=True)
    arrays, scalars = _dataset_parts(dataset)
''',
        '''    if dataset.identity_payload_json is None:
        raise ValueError(
            "formal dataset artifacts require a verified content identity"
        )
    root.mkdir(parents=True, exist_ok=True)
    arrays, scalars = _dataset_parts(dataset)
''',
    )

    fixture_replacements = (
        (
            "tests/data/test_market_dataset_artifact.py",
            '''        cash_rate=np.linspace(0.0, 0.001, n_bars),
    )
''',
            '''        cash_rate=np.linspace(0.0, 0.001, n_bars),
    ).with_content_identity()
''',
        ),
        (
            "tests/workflows/test_training_run.py",
            '''        periods_per_year=8_760,
    )


def _config''',
            '''        periods_per_year=8_760,
    ).with_content_identity()


def _config''',
        ),
        (
            "tests/workflows/test_market_walk_forward.py",
            '''        periods_per_year=8_760,
    )


def _candidate_run''',
            '''        periods_per_year=8_760,
    ).with_content_identity()


def _candidate_run''',
        ),
    )
    for path, old, new in fixture_replacements:
        replace_once(path, old, new)

    replace_once(
        "examples/quickstart/create_demo_dataset.py",
        "import hashlib\n",
        "",
    )
    replace_once(
        "examples/quickstart/create_demo_dataset.py",
        '''    identity = hashlib.sha256()
    identity.update(b"trade-rl-quickstart-dataset-v1")
    for array in (
        timestamps.astype("datetime64[ns]").astype(np.int64),
        features,
        global_features,
        *prices.values(),
    ):
        identity.update(np.ascontiguousarray(array).tobytes(order="C"))

''',
        "",
    )
    replace_once(
        "examples/quickstart/create_demo_dataset.py",
        "        dataset_id=identity.hexdigest(),\n",
        '        dataset_id="0" * 64,\n',
    )
    replace_once(
        "examples/quickstart/create_demo_dataset.py",
        '''        borrow_available=np.ones((n_bars, 1), dtype=np.bool_),
    )
''',
        '''        borrow_available=np.ones((n_bars, 1), dtype=np.bool_),
    ).with_content_identity({"source": "quickstart-demo-v2"})
''',
    )


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"tests", "implementation"}:
        raise SystemExit("usage: dataset_identity.py tests|implementation")
    if sys.argv[1] == "tests":
        write_tests()
    else:
        apply_implementation()


if __name__ == "__main__":
    main()
