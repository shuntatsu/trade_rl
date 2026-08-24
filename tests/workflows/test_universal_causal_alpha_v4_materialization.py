from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from trade_rl.data.v4_context import V4CrossMarketInputs
from trade_rl.integrations.postgres_universal_source import MAINTAINED_SYMBOLS
from trade_rl.workflows import (
    universal_causal_alpha_v4_materialization as materialization,
)
from trade_rl.workflows.universal_causal_alpha_v4_manifest import (
    CausalAlphaV4ContextManifest,
)


def _digest(index: int) -> str:
    return f"{index:064x}"


def _timestamps(rows: int = 128) -> np.ndarray:
    return np.datetime64("2026-01-01T00:15", "ns") + np.arange(rows) * np.timedelta64(
        15, "m"
    )


def _runtime_manifest() -> SimpleNamespace:
    symbols = tuple(MAINTAINED_SYMBOLS)
    return SimpleNamespace(
        manifest_digest=_digest(1),
        train_symbols=symbols[:9],
        validation_symbols=symbols[9:12],
        test_symbols=symbols[12:],
        research_start=datetime(2026, 1, 1, tzinfo=UTC),
        research_end=datetime(2026, 1, 3, tzinfo=UTC),
        metadata_evidence_digest=_digest(2),
        dataset_artifact_relpath=Path("datasets"),
        dataset_digests=(("BTCUSDT", _digest(3)),),
    )


def _dataset() -> SimpleNamespace:
    return SimpleNamespace(
        dataset_id=_digest(3),
        symbols=("BTCUSDT",),
        timestamps=_timestamps(),
        calendar_kind="continuous_24_7",
        nominal_bar_hours=0.25,
    )


def _input(
    *,
    symbol: str,
    decision_indices: object,
    decision_timestamps: object,
) -> V4CrossMarketInputs:
    indices = np.asarray(decision_indices, dtype=np.int64)
    timestamps = np.asarray(decision_timestamps, dtype="datetime64[ns]")
    rows = len(indices)
    multiplier = 1.0 + tuple(MAINTAINED_SYMBOLS).index(symbol) * 0.01
    close = 100.0 * np.exp(
        multiplier * np.arange(rows, dtype=np.float64) * 0.0001
    )
    quote = np.full(rows, 1_000_000.0 * multiplier, dtype=np.float64)
    funding = np.zeros(rows, dtype=np.float64)
    funding_available = np.zeros(rows, dtype=np.bool_)
    funding[0] = 0.0001
    funding_available[0] = True
    return V4CrossMarketInputs(
        decision_indices=indices,
        decision_timestamps=timestamps,
        spot_close=close * 0.999,
        spot_quote_volume=quote,
        spot_taker_buy_quote_volume=quote * 0.55,
        spot_row_available=np.ones(rows, dtype=np.bool_),
        perp_close=close,
        perp_mark_price=close * 1.0002,
        perp_quote_volume=quote * 1.5,
        perp_taker_buy_quote_volume=quote * 1.5 * 0.52,
        perp_row_available=np.ones(rows, dtype=np.bool_),
        funding_event_rate=funding,
        funding_event_available=funding_available,
        open_interest_value=None,
        global_long_short_ratio=None,
        top_position_long_short_ratio=None,
        derivatives_available=None,
        derivatives_staleness_hours=None,
        source_digest=_digest(10 + tuple(MAINTAINED_SYMBOLS).index(symbol)),
    )


def _install_base_fakes(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    runtime = _runtime_manifest()
    monkeypatch.setattr(
        materialization,
        "load_universal_runtime_manifest",
        lambda _path: runtime,
    )
    monkeypatch.setattr(
        materialization,
        "load_market_dataset_artifact",
        lambda _path: _dataset(),
    )
    monkeypatch.setattr(
        materialization,
        "FrozenBinanceExchangeInfoTransport",
        lambda root: root,
    )
    monkeypatch.setattr(
        materialization,
        "resolve_frozen_snapshot",
        lambda **_kwargs: SimpleNamespace(evidence_digest=runtime.metadata_evidence_digest),
    )
    return runtime


def test_binance_v4_materialization_closes_core_source_and_context_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _install_base_fakes(monkeypatch)
    observed_metric_urls: list[tuple[str, ...]] = []
    observed_symbols: list[str] = []
    monkeypatch.setattr(
        materialization,
        "inspect_binance_vision_urls",
        lambda urls, **_kwargs: SimpleNamespace(
            complete=True,
            cached_count=len(tuple(urls)),
            missing_urls=(),
            empty_urls=(),
            invalid_urls=(),
        ),
    )

    def build_inputs(**kwargs: Any) -> V4CrossMarketInputs:
        symbol = str(kwargs["expected_symbol"])
        observed_symbols.append(symbol)
        observed_metric_urls.append(tuple(kwargs["metrics_urls"]))
        return _input(
            symbol=symbol,
            decision_indices=kwargs["decision_indices"],
            decision_timestamps=kwargs["decision_timestamps"],
        )

    monkeypatch.setattr(
        materialization,
        "build_cached_v4_cross_market_inputs",
        build_inputs,
    )

    def publish_generation(**kwargs: Any) -> CausalAlphaV4ContextManifest:
        requested = str(kwargs["requested_profile"])
        capability = kwargs["capability_resolver"](requested)
        contexts = tuple(
            kwargs["context_builder"](symbol, capability.profile_name)
            for symbol in MAINTAINED_SYMBOLS
        )
        assert all(len(context.local.feature_names) == 24 for context in contexts)
        assert all(
            len(context.global_market.feature_names) == 38 for context in contexts
        )
        return CausalAlphaV4ContextManifest(
            base_runtime_manifest_digest=runtime.manifest_digest,
            profile_name=capability.profile_name,
            context_artifact_relpath=Path("contexts"),
            context_digests=tuple(
                (symbol, context.digest)
                for symbol, context in zip(MAINTAINED_SYMBOLS, contexts, strict=True)
            ),
            local_schema_digest=_digest(40),
            global_schema_digest=_digest(41),
            pit_flow_profile=None,
            source_capability_digest=capability.source_digest,
        )

    result = materialization.materialize_binance_causal_alpha_v4_context(
        runtime_manifest_path=tmp_path / "runtime" / "manifest.json",
        frozen_metadata_root=tmp_path / "metadata",
        market_data_cache_root=tmp_path / "cache",
        output_root=tmp_path / "v4",
        requested_profile="core",
        materialize_generation=publish_generation,
    )

    assert result.profile_name == "cross_market_core_v1"
    assert observed_symbols == list(MAINTAINED_SYMBOLS)
    assert observed_metric_urls == [()] * len(MAINTAINED_SYMBOLS)


def test_binance_v4_materialization_fails_before_parse_when_cache_is_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_base_fakes(monkeypatch)
    monkeypatch.setattr(
        materialization,
        "inspect_binance_vision_urls",
        lambda urls, **_kwargs: SimpleNamespace(
            complete=False,
            cached_count=len(tuple(urls)) - 1,
            missing_urls=("missing",),
            empty_urls=(),
            invalid_urls=(),
        ),
    )
    monkeypatch.setattr(
        materialization,
        "build_cached_v4_cross_market_inputs",
        lambda **_kwargs: pytest.fail("incomplete cache must fail before parsing"),
    )

    with pytest.raises(FileNotFoundError, match="cache is incomplete"):
        materialization.materialize_binance_causal_alpha_v4_context(
            runtime_manifest_path=tmp_path / "runtime" / "manifest.json",
            frozen_metadata_root=tmp_path / "metadata",
            market_data_cache_root=tmp_path / "cache",
            output_root=tmp_path / "v4",
            requested_profile="core",
            materialize_generation=lambda **_kwargs: pytest.fail(
                "incomplete cache must fail before publication"
            ),
        )
