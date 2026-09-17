from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime

import numpy as np
import pytest

from tests.integrations.test_binance_forward_rules import symbol
from trade_rl.artifacts import canonical_json_bytes
from trade_rl.data.market import MarketDataset
from trade_rl.integrations.binance.metadata import BinanceExchangeInfoSnapshot


def _dataset(symbols=("BTCUSDT",), *, lot=0.001, minimum=50.0):
    shape = (6, len(symbols))
    prices = np.full(shape, 100.0)
    return MarketDataset(
        dataset_id="d" * 64,
        symbols=symbols,
        timestamps=np.datetime64("2024-01-01", "ns")
        + np.arange(6) * np.timedelta64(1, "h"),
        features=np.zeros((*shape, 1), dtype=np.float32),
        global_features=np.zeros((6, 1), dtype=np.float32),
        open=prices,
        high=prices + 10,
        low=prices - 10,
        close=prices,
        volume=np.full(shape, 1000.0),
        funding_rate=np.zeros(shape),
        tradable=np.ones(shape, dtype=bool),
        feature_available=np.ones((*shape, 1), dtype=bool),
        feature_names=("ret",),
        global_feature_names=("regime",),
        periods_per_year=8760,
        minimum_notional=np.full(shape, minimum),
        lot_size=np.full(shape, lot),
    ).with_content_identity({"fixture": "market-order-profile"})


def _snapshot(*, mutate=None):
    payload = {
        "symbols": [symbol("perpetual", name) for name in ("BTCUSDT", "ETHUSDT")]
    }
    if mutate:
        mutate(payload)
    raw = json.dumps(payload).encode()
    return BinanceExchangeInfoSnapshot(
        payload=payload,
        raw_payload=raw,
        source_uri="https://fapi.binance.com/fapi/v1/exchangeInfo",
        retrieved_at=datetime(2026, 9, 18, tzinfo=UTC),
        raw_payload_sha256=hashlib.sha256(raw).hexdigest(),
    )


def _profile(dataset=None, snapshot=None, **overrides):
    from trade_rl.integrations.binance.market_order_profile import (
        build_usdm_market_order_profile,
    )

    values = dict(
        selected_symbols=("BTCUSDT",), account_mode="one_way", reduce_only_exits=True
    )
    values.update(overrides)
    return build_usdm_market_order_profile(
        dataset or _dataset(), snapshot or _snapshot(), **values
    )


def test_profile_binds_verified_dataset_roster_and_both_quantity_rules():
    def mutate(payload):
        row = payload["symbols"][0]
        row["filters"][1].update(stepSize="0.002", minQty="0.004")
        row["filters"][2].update(stepSize="0.003", minQty="0.009", maxQty="10")

    dataset = _dataset(("OTHER", "BTCUSDT"))
    profile = _profile(dataset, _snapshot(mutate=mutate))
    assert profile.dataset_id == dataset.dataset_id
    assert profile.dataset_symbols == ("OTHER", "BTCUSDT")
    assert profile.rule_for(0) is None
    rule = profile.rule_for(1)
    assert rule.lot_size == 0.006
    assert rule.minimum_quantity == 0.009
    assert rule.maximum_quantity == 10.0
    assert rule.minimum_notional == 50.0
    assert profile.reduce_only_exits is True
    assert (
        profile.digest
        != _profile(dataset, _snapshot(mutate=mutate), reduce_only_exits=False).digest
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"account_mode": "hedge"},
        {"account_mode": None},
        {"reduce_only_exits": 1},
        {"selected_symbols": ("MISSING",)},
        {"selected_symbols": ("BTCUSDT", "BTCUSDT")},
        {"selected_symbols": ()},
    ],
)
def test_unsupported_profile_scope_is_rejected(overrides):
    with pytest.raises(ValueError):
        _profile(**overrides)


@pytest.mark.parametrize(
    "change",
    [
        {"source_uri": "https://api.binance.com/api/v3/exchangeInfo"},
        {"raw_payload_sha256": "0" * 64},
        {"raw_payload": b'{"symbols":[],"symbols":[]}'},
    ],
)
def test_profile_rejects_wrong_or_modified_raw_source(change):
    with pytest.raises(ValueError):
        _profile(snapshot=replace(_snapshot(), **change))


def test_profile_requires_content_verified_dataset():
    dataset = replace(_dataset(), identity_payload_json=None)
    with pytest.raises(ValueError, match="identity"):
        _profile(dataset)


def test_profile_rejects_nonunit_selected_contract_multipliers():
    dataset = replace(
        _dataset(), contract_multipliers=np.array([2.0]), identity_payload_json=None
    ).with_content_identity()
    with pytest.raises(ValueError, match="multiplier"):
        _profile(dataset)


def test_profile_cannot_be_replaced_with_fabricated_rules():
    profile = _profile()
    changed = replace(profile.rules[0], minimum_notional=0.0)
    with pytest.raises(ValueError, match="factory"):
        replace(profile, rules=(changed,))


def test_joint_lot_size_preserves_every_grid_and_stress():
    from trade_rl.data.market_order_rules import joint_lot_size

    assert joint_lot_size((0.002, 0.003, 0.003)) == 0.006
    assert joint_lot_size((0.002, 0.003, 0.003), stress_factor=1.5) == 0.018
    assert joint_lot_size((0.0, 0.001, 0.003), stress_factor=2.0) == 0.006


def test_profile_publication_rederives_raw_rules_and_is_write_once(tmp_path):
    from trade_rl.integrations.binance.market_order_profile import (
        load_usdm_market_order_profile,
        publish_usdm_market_order_profile,
    )

    dataset, snapshot = _dataset(), _snapshot()
    profile = _profile(dataset, snapshot)
    root = tmp_path / "profile"
    digest = publish_usdm_market_order_profile(root, profile, snapshot, dataset)
    restored = load_usdm_market_order_profile(root, dataset, expected_digest=digest)
    assert restored == profile
    assert (root / "exchange-info.raw.json").read_bytes() == snapshot.raw_payload
    with pytest.raises(FileExistsError):
        publish_usdm_market_order_profile(root, profile, snapshot, dataset)
    payload = json.loads((root / "profile.json").read_bytes())
    payload["rules"][0]["minimum_notional"] = 0.0
    changed = canonical_json_bytes(payload)
    (root / "profile.json").write_bytes(changed)
    with pytest.raises(ValueError, match="digest"):
        load_usdm_market_order_profile(root, dataset, expected_digest=digest)
    # Even a new outer hash cannot turn fabricated rules into source-derived rules.
    with pytest.raises(ValueError, match="derived"):
        load_usdm_market_order_profile(
            root, dataset, expected_digest=hashlib.sha256(changed).hexdigest()
        )


def test_profile_loader_rejects_different_dataset_and_raw_tampering(tmp_path):
    from trade_rl.integrations.binance.market_order_profile import (
        load_usdm_market_order_profile,
        publish_usdm_market_order_profile,
    )

    dataset, snapshot = _dataset(), _snapshot()
    root = tmp_path / "profile"
    digest = publish_usdm_market_order_profile(
        root, _profile(dataset, snapshot), snapshot, dataset
    )
    with pytest.raises(ValueError, match="dataset"):
        load_usdm_market_order_profile(
            root, _dataset(minimum=40), expected_digest=digest
        )
    (root / "exchange-info.raw.json").write_bytes(snapshot.raw_payload + b" ")
    with pytest.raises(ValueError, match="digest"):
        load_usdm_market_order_profile(root, dataset, expected_digest=digest)
