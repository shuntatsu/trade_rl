from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tests.evaluation.experiments.bootstrap.test_binance import (
    _config as _bootstrap_fixture_config,
)
from tests.evaluation.experiments.bootstrap.test_config import _valid_payload, _write
from tests.evaluation.experiments.bootstrap.test_workflow import (
    _install_fakes,
    _synthetic_dataset,
)
from tests.integrations.test_binance import FakeTransport
from trade_rl.data.economics import ExecutionEconomicsConfig
from trade_rl.evaluation.experiments.bootstrap.config import (
    load_canonical_m2_bootstrap_config,
)
from trade_rl.evaluation.experiments.bootstrap.workflow import (
    bootstrap_canonical_m2_study,
)
from trade_rl.integrations.binance import (
    BinanceDatasetBuildResult,
    build_binance_market_dataset,
)

# Captured from exact pre-change parent 1fec1a2dbc2ee5af18528d812278c98b6a12090f.
_V1_CONFIG_DIGEST = "0e27cd7608b12261051db0f37ca72a90584ba32a786be93d44de1f6ca820bc0c"


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


def _fixture_v2_payload() -> dict[str, object]:
    payload = _bootstrap_fixture_config().to_payload()
    payload["schema_version"] = "canonical_m2_bootstrap_config_v2"
    payload["execution_economics"] = _profile().canonical_payload()
    return payload


def test_v1_bootstrap_contract_remains_semantically_compatible(
    tmp_path: Path,
) -> None:
    config = load_canonical_m2_bootstrap_config(_write(tmp_path, _valid_payload()))
    normalized = config.to_payload()
    baseline = normalized["baseline"]
    assert isinstance(baseline, dict)

    assert config.execution_economics is None
    assert config.digest == _V1_CONFIG_DIGEST
    assert "execution_economics" not in normalized
    assert baseline["fit_cutoff"] == "2024-07-01T00:00:00.000000000"
    assert baseline["evaluation_start"] == "2024-07-01T00:00:00.000000000"
    assert baseline["evaluation_stop_exclusive"] == "2025-01-01T00:00:00.000000000"


def test_v2_requires_and_roundtrips_explicit_execution_economics(
    tmp_path: Path,
) -> None:
    config = load_canonical_m2_bootstrap_config(_write(tmp_path, _v2_payload()))
    normalized = config.to_payload()

    assert config.schema_version == "canonical_m2_bootstrap_config_v2"
    assert config.execution_economics == _profile()
    assert normalized["execution_economics"] == _profile().canonical_payload()
    assert config.digest != _V1_CONFIG_DIGEST

    roundtrip_path = tmp_path / "roundtrip.json"
    roundtrip_path.write_text(json.dumps(normalized), encoding="utf-8")
    reloaded = load_canonical_m2_bootstrap_config(roundtrip_path)
    assert reloaded.to_payload() == normalized
    assert reloaded.digest == config.digest


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


def test_v2_bootstrap_rejects_silent_zero_economics_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fakes(monkeypatch)
    from trade_rl.evaluation.experiments.bootstrap import workflow as workflow_module

    def zero_economics_build(**kwargs: object) -> BinanceDatasetBuildResult:
        assert kwargs["execution_economics"] == _profile()
        dataset = _synthetic_dataset(kwargs["metadata_evidence"])
        return BinanceDatasetBuildResult(
            dataset=dataset,
            metadata=(),
            sources_used=("frozen:exchange-info", "vision"),
            feature_timeframes=("1h",),
        )

    monkeypatch.setattr(
        workflow_module,
        "build_binance_market_dataset",
        zero_economics_build,
    )
    config_path = tmp_path / "bootstrap-v2.json"
    config_path.write_text(json.dumps(_fixture_v2_payload()), encoding="utf-8")

    with pytest.raises(ValueError, match="execution economics"):
        bootstrap_canonical_m2_study(config_path, tmp_path / "canonical-m2-v2")

    assert not (tmp_path / "canonical-m2-v2").exists()
