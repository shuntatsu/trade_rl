from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from trade_rl.data.v4_context import CausalBetaConfig, V4CrossMarketInputs
from trade_rl.data.v4_context_artifact import load_v4_target_context_artifact
from trade_rl.integrations.binance_v4_context_capability import BinanceV4ProfileCapability
from trade_rl.workflows.universal_causal_alpha_v4_context_generation import (
    materialize_causal_alpha_v4_context_generation,
)
from trade_rl.workflows.universal_causal_alpha_v4_manifest import (
    load_causal_alpha_v4_context_manifest,
)


def _digest(char: str) -> str:
    return char * 64


def _base(symbols: tuple[str, ...]):
    return SimpleNamespace(
        manifest_digest=_digest("a"),
        train_symbols=symbols[: max(1, len(symbols) - 2)],
        validation_symbols=symbols[max(1, len(symbols) - 2) : -1],
        test_symbols=symbols[-1:],
    )


def _capability(symbols: tuple[str, ...]) -> BinanceV4ProfileCapability:
    from datetime import UTC, datetime

    return BinanceV4ProfileCapability(
        symbols=symbols,
        start_time=datetime(2026, 1, 1, tzinfo=UTC),
        end_time=datetime(2026, 1, 2, tzinfo=UTC),
        required_archive_count=len(symbols),
        cached_archive_count=0,
        missing_archive_count=len(symbols),
        invalid_archive_count=0,
        derivative_metrics_complete=False,
        profile_name="cross_market_core_v1",
        source_digest=_digest("d"),
    )


def _input(symbol: str, *, multiplier: float, rows: int = 64) -> V4CrossMarketInputs:
    decisions = np.arange(100, 100 + rows, dtype=np.int64)
    timestamps = np.datetime64("2026-01-01T00:00", "ns") + np.arange(rows) * np.timedelta64(15, "m")
    block_amplitudes = np.asarray([0.0010, 0.0020, -0.0010, 0.0030], dtype=np.float64)
    btc_returns = np.repeat(block_amplitudes, 16)[: rows - 1]
    close = np.exp(
        np.concatenate(
            (np.asarray([0.0], dtype=np.float64), np.cumsum(multiplier * btc_returns))
        )
    ) * 100.0
    spot = close * 0.999
    quote = np.full(rows, 1_000_000.0 + multiplier * 1_000.0, dtype=np.float64)
    taker = quote * 0.55
    funding_rate = np.zeros(rows, dtype=np.float64)
    funding_available = np.zeros(rows, dtype=np.bool_)
    funding_rate[0] = 0.0001
    funding_available[0] = True
    return V4CrossMarketInputs(
        decision_indices=decisions,
        decision_timestamps=timestamps,
        spot_close=spot,
        spot_quote_volume=quote,
        spot_taker_buy_quote_volume=taker,
        spot_row_available=np.ones(rows, dtype=np.bool_),
        perp_close=close,
        perp_mark_price=close * 1.0002,
        perp_quote_volume=quote * 1.5,
        perp_taker_buy_quote_volume=quote * 1.5 * 0.52,
        perp_row_available=np.ones(rows, dtype=np.bool_),
        funding_event_rate=funding_rate,
        funding_event_available=funding_available,
        open_interest_value=None,
        global_long_short_ratio=None,
        top_position_long_short_ratio=None,
        derivatives_available=None,
        derivatives_staleness_hours=None,
        source_digest=_digest({"BTCUSDT": "1", "ETHUSDT": "2", "SOLUSDT": "3"}[symbol]),
    )


def _inputs() -> tuple[tuple[str, ...], dict[str, V4CrossMarketInputs]]:
    symbols = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
    return symbols, {
        "BTCUSDT": _input("BTCUSDT", multiplier=1.0),
        "ETHUSDT": _input("ETHUSDT", multiplier=1.5),
        "SOLUSDT": _input("SOLUSDT", multiplier=0.7),
    }


def _beta_config() -> CausalBetaConfig:
    return CausalBetaConfig(
        return_horizon_hours=4.0,
        lookback_hours=48.0,
        minimum_complete_samples=3,
        minimum_market_variance=1e-12,
        minimum_beta=-3.0,
        maximum_beta=3.0,
    )


def test_materializer_writes_contexts_then_manifest(tmp_path: Path) -> None:
    symbols, inputs = _inputs()
    result = materialize_causal_alpha_v4_context_generation(
        base_runtime=_base(symbols),
        inputs=inputs,
        capability=_capability(symbols),
        output_root=tmp_path / "generation",
        beta_config=_beta_config(),
    )
    assert result.manifest_path == tmp_path / "generation" / "manifest.json"
    manifest = load_causal_alpha_v4_context_manifest(result.manifest_path)
    assert tuple(symbol for symbol, _ in manifest.context_digests) == symbols
    assert manifest.profile_name == "cross_market_core_v1"
    assert manifest.context_artifact_relpath == Path("contexts")
    assert len(result.context_paths) == 3

    loaded = {
        symbol: load_v4_target_context_artifact(path)
        for symbol, path in result.context_paths
    }
    assert len({context.global_market.digest for context in loaded.values()}) == 1
    btc = loaded["BTCUSDT"]
    assert np.any(btc.beta_available)
    np.testing.assert_array_equal(
        btc.beta[btc.beta_available], np.ones(np.count_nonzero(btc.beta_available))
    )
    assert np.any(loaded["ETHUSDT"].beta_available)


def test_materializer_recovers_missing_context_before_manifest_publish(tmp_path: Path) -> None:
    symbols, inputs = _inputs()
    root = tmp_path / "generation"
    first = materialize_causal_alpha_v4_context_generation(
        base_runtime=_base(symbols),
        inputs=inputs,
        capability=_capability(symbols),
        output_root=root,
        beta_config=_beta_config(),
    )
    first.manifest_path.unlink()
    missing_path = root / "contexts" / "SOLUSDT"
    shutil.rmtree(missing_path)

    second = materialize_causal_alpha_v4_context_generation(
        base_runtime=_base(symbols),
        inputs=inputs,
        capability=_capability(symbols),
        output_root=root,
        beta_config=_beta_config(),
    )
    assert second.manifest_path.is_file()
    assert missing_path.is_dir()
    assert second.manifest_digest == first.manifest_digest


def test_materializer_rejects_input_scope_drift(tmp_path: Path) -> None:
    symbols, inputs = _inputs()
    inputs.pop("SOLUSDT")
    with pytest.raises(ValueError, match="scope|symbols"):
        materialize_causal_alpha_v4_context_generation(
            base_runtime=_base(symbols),
            inputs=inputs,
            capability=_capability(symbols),
            output_root=tmp_path / "generation",
            beta_config=_beta_config(),
        )


def test_materializer_rejects_profile_and_derivative_input_mismatch(tmp_path: Path) -> None:
    from datetime import UTC, datetime

    symbols, inputs = _inputs()
    capability = BinanceV4ProfileCapability(
        symbols=symbols,
        start_time=datetime(2026, 1, 1, tzinfo=UTC),
        end_time=datetime(2026, 1, 2, tzinfo=UTC),
        required_archive_count=3,
        cached_archive_count=3,
        missing_archive_count=0,
        invalid_archive_count=0,
        derivative_metrics_complete=True,
        profile_name="cross_market_derivatives_v1",
        source_digest=_digest("d"),
    )
    with pytest.raises(ValueError, match="derivative"):
        materialize_causal_alpha_v4_context_generation(
            base_runtime=_base(symbols),
            inputs=inputs,
            capability=capability,
            output_root=tmp_path / "generation",
            beta_config=_beta_config(),
        )


def test_materializer_rejects_published_manifest_if_context_disappears(tmp_path: Path) -> None:
    symbols, inputs = _inputs()
    root = tmp_path / "generation"
    materialize_causal_alpha_v4_context_generation(
        base_runtime=_base(symbols),
        inputs=inputs,
        capability=_capability(symbols),
        output_root=root,
        beta_config=_beta_config(),
    )
    shutil.rmtree(root / "contexts" / "ETHUSDT")
    with pytest.raises((FileNotFoundError, ValueError), match="context|manifest|missing"):
        materialize_causal_alpha_v4_context_generation(
            base_runtime=_base(symbols),
            inputs=inputs,
            capability=_capability(symbols),
            output_root=root,
            beta_config=_beta_config(),
        )
