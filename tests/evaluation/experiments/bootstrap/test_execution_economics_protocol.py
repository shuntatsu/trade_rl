from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tests.evaluation.experiments.bootstrap.test_config import _valid_payload, _write
from tests.integrations.test_binance import FakeTransport
from trade_rl.data.economics import ExecutionEconomicsConfig
from trade_rl.evaluation.experiments.bootstrap.config import (
    load_canonical_m2_bootstrap_config,
)
from trade_rl.integrations.binance import build_binance_market_dataset


def _profile() -> ExecutionEconomicsConfig:
    return ExecutionEconomicsConfig(
        fee_rate=0.0,
        maker_fee_rate=0.0002,
        taker_fee_rate=0.0005,
        spread_rate=0.0002,
        max_participation_rate=0.05,
        borrow_available=True,
        borrow_rate=0.0,
    )


def _v2_payload() -> dict[str, object]:
    payload = _valid_payload()
    payload["schema_version"] = "canonical_m2_bootstrap_config_v2"
    payload["execution_economics"] = _profile().canonical_payload()
    return payload


def test_v1_bootstrap_contract_remains_byte_semantically_compatible(
    tmp_path: Path,
) -> None:
    original = _valid_payload()
    config = load_canonical_m2_bootstrap_config(_write(tmp_path, original))

    assert config.execution_economics is None
    assert config.to_payload() == original
    assert "execution_economics" not in config.to_payload()


def test_v2_requires_and_roundtrips_explicit_execution_economics(
    tmp_path: Path,
) -> None:
    payload = _v2_payload()
    config = load_canonical_m2_bootstrap_config(_write(tmp_path, payload))

    assert config.schema_version == "canonical_m2_bootstrap_config_v2"
    assert config.execution_economics == _profile()
    assert config.to_payload() == payload


@pytest.mark.parametrize("mode", ["missing", "invalid_nested", "v1_mixed"])
def test_bootstrap_execution_economics_contract_fails_closed(
    tmp_path: Path,
    mode: str,
) -> None:
    payload = _v2_payload()
    if mode == "missing":
        del payload["execution_economics"]
    elif mode == "invalid_nested":
        economics = dict(_profile().canonical_payload())
        economics["fee_rate"] = 0.0001
        payload["execution_economics"] = economics
    else:
        payload["schema_version"] = "canonical_m2_bootstrap_config_v1"

    with pytest.raises(ValueError, match="execution_economics|fee_rate|keys differ"):
        load_canonical_m2_bootstrap_config(_write(tmp_path, payload))


def test_binance_dataset_build_applies_explicit_execution_economics() -> None:
    start = FakeTransport().start
    common = {
        "market": "usds-m",
        "symbols": ("BTCUSDT",),
        "interval": "1h",
        "start_time": start,
        "end_time": start.replace(hour=3),
        "transport": FakeTransport(),
    }

    zero = build_binance_market_dataset(**common)
    costed = build_binance_market_dataset(
        **common,
        execution_economics=_profile(),
    )

    assert costed.dataset.dataset_id != zero.dataset.dataset_id
    np.testing.assert_allclose(costed.dataset.resolved_array("fee_rate"), 0.0)
    np.testing.assert_allclose(costed.dataset.resolved_array("maker_fee_rate"), 0.0002)
    np.testing.assert_allclose(costed.dataset.resolved_array("taker_fee_rate"), 0.0005)
    np.testing.assert_allclose(costed.dataset.resolved_array("spread_rate"), 0.0002)
    np.testing.assert_allclose(
        costed.dataset.resolved_array("max_participation_rate"),
        0.05,
    )
    np.testing.assert_allclose(costed.dataset.resolved_array("borrow_rate"), 0.0)
    identity = json.loads(costed.dataset.identity_payload_json or "{}")
    assert identity["execution_economics"] == _profile().canonical_payload()
