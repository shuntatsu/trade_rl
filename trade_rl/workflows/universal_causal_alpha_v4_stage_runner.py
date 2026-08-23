"""Concrete stage assembly helpers for the research-only Causal Alpha V4 runner."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.learning.causal_alpha_v4 import (
    CausalAlphaV4Forecast,
    build_causal_alpha_v4_forecast,
)
from trade_rl.workflows.universal_causal_alpha_v3_signal import (
    split_causal_alpha_v3_partitions,
)
from trade_rl.workflows.universal_causal_alpha_v4_runtime import (
    build_causal_alpha_v4_symbol_samples,
    validate_causal_alpha_v4_train_sample_scope,
)

CAUSAL_ALPHA_V4_STAGE_RUN_IDENTITY_SCHEMA: Final = (
    "causal_alpha_v4_stage_run_identity_v1"
)
_V4_SIGNAL_CONTRACT_COUNT: Final = 8
_V4_MINIMUM_ECONOMIC_CONTRACT_COUNT: Final = 4


@dataclass(frozen=True, slots=True)
class CausalAlphaV4PreparedStageData:
    """V4 train-only samples plus immutable stage/run identity closure."""

    train_symbols: tuple[str, ...]
    samples: Mapping[str, Any]
    nested_partitions: Mapping[str, Any]
    nested_partition_digest: str
    run_manifest_digest: str
    base_runtime_manifest_digest: str
    v4_context_manifest_digest: str
    config_digest: str
    execution_identity_digest: str
    generator_code_digest: str
    prepared_v3: object

    def __post_init__(self) -> None:
        symbols = tuple(self.train_symbols)
        if (
            not symbols
            or len(set(symbols)) != len(symbols)
            or any(not isinstance(symbol, str) or not symbol for symbol in symbols)
        ):
            raise ValueError("V4 prepared train_symbols must be non-empty and unique")
        if set(self.samples) != set(symbols):
            raise ValueError("V4 prepared sample scope must match train_symbols")
        if set(self.nested_partitions) != set(symbols):
            raise ValueError("V4 prepared nested scope must match train_symbols")
        for field_name in (
            "nested_partition_digest",
            "run_manifest_digest",
            "base_runtime_manifest_digest",
            "v4_context_manifest_digest",
            "config_digest",
            "execution_identity_digest",
            "generator_code_digest",
        ):
            require_sha256(getattr(self, field_name), field=f"V4 prepared {field_name}")
        object.__setattr__(self, "train_symbols", symbols)


def slice_causal_alpha_v4_forecast(
    forecast: CausalAlphaV4Forecast,
    row_indices: object,
) -> CausalAlphaV4Forecast:
    """Slice every V4 forecast component by the same ordered source rows."""

    if not isinstance(forecast, CausalAlphaV4Forecast):
        raise TypeError("V4 forecast slice requires CausalAlphaV4Forecast")
    rows = np.asarray(row_indices, dtype=np.int64).reshape(-1)
    if (
        rows.size == 0
        or np.any(rows < 0)
        or np.any(rows >= forecast.decision_indices.size)
        or np.any(np.diff(rows) <= 0)
    ):
        raise ValueError("V4 forecast slice rows must be unique increasing indices")
    horizons = ("4h", "24h", "72h")
    return build_causal_alpha_v4_forecast(
        symbol=forecast.symbol,
        decision_indices=forecast.decision_indices[rows],
        beta=forecast.beta[rows],
        beta_available=forecast.beta_available[rows],
        market_predictions={
            horizon: forecast.market_predictions[horizon][rows] for horizon in horizons
        },
        residual_predictions={
            horizon: forecast.residual_predictions[horizon][rows] for horizon in horizons
        },
        direction_scores={
            horizon: forecast.direction_scores[horizon][rows] for horizon in horizons
        },
        market_model_digests=forecast.market_model_digests,
        residual_model_digests=forecast.residual_model_digests,
        direction_model_digests=forecast.direction_model_digests,
        fit_digest=forecast.fit_digest,
    )


def build_causal_alpha_v4_stage_run_identity(
    *,
    base_runtime_manifest_digest: str,
    v4_context_manifest_digest: str,
    config_digest: str,
    execution_identity_digest: str,
    nested_partition_digest: str,
    generator_code_digest: str,
) -> str:
    """Bind every immutable upstream identity into one V4 run identity."""

    payload = {
        "base_runtime_manifest_digest": base_runtime_manifest_digest,
        "config_digest": config_digest,
        "execution_identity_digest": execution_identity_digest,
        "generator_code_digest": generator_code_digest,
        "nested_partition_digest": nested_partition_digest,
        "schema_version": CAUSAL_ALPHA_V4_STAGE_RUN_IDENTITY_SCHEMA,
        "v4_context_manifest_digest": v4_context_manifest_digest,
    }
    for field_name, value in payload.items():
        if field_name == "schema_version":
            continue
        require_sha256(str(value), field=f"V4 stage {field_name}")
    return content_digest(payload)


def require_causal_alpha_v4_context_scope(
    *,
    train_symbols: tuple[str, ...],
    provider: object | None,
    manifest: object,
) -> dict[str, Any]:
    """Return ordered train contexts after closing the complete manifest/provider scope."""

    symbols = tuple(train_symbols)
    if (
        not symbols
        or len(set(symbols)) != len(symbols)
        or any(not isinstance(symbol, str) or not symbol for symbol in symbols)
    ):
        raise ValueError("V4 stage train_symbols must be non-empty and unique")
    if provider is None:
        raise ValueError("V4 stage requires V4 context provider")
    raw_contexts = getattr(provider, "contexts", None)
    if not isinstance(raw_contexts, Mapping):
        raise TypeError("V4 stage context provider does not expose contexts")
    contexts = dict(raw_contexts)
    raw_digests = getattr(manifest, "context_digests", None)
    if not isinstance(raw_digests, (tuple, list)):
        raise TypeError("V4 stage context manifest does not expose context_digests")
    try:
        expected = tuple((str(symbol), str(digest)) for symbol, digest in raw_digests)
    except (TypeError, ValueError) as error:
        raise ValueError("V4 stage context manifest digests are invalid") from error
    expected_symbols = tuple(symbol for symbol, _ in expected)
    if (
        not expected_symbols
        or len(set(expected_symbols)) != len(expected_symbols)
        or set(contexts) != set(expected_symbols)
        or tuple(symbol for symbol in expected_symbols if symbol in set(symbols)) != symbols
    ):
        raise ValueError("V4 stage context scope does not match train_symbols")

    manifest_contexts: dict[str, Any] = {}
    for symbol, expected_digest in expected:
        require_sha256(expected_digest, field=f"V4 stage context digest {symbol}")
        context = contexts[symbol]
        if getattr(context, "symbol", None) != symbol:
            raise ValueError("V4 stage context artifact symbol identity drifted")
        if getattr(context, "digest", None) != expected_digest:
            raise ValueError("V4 stage context artifact identity drifted")
        manifest_contexts[symbol] = context
    return {symbol: manifest_contexts[symbol] for symbol in symbols}


def _nested_scope_digest(
    *,
    train_symbols: tuple[str, ...],
    nested_partitions: Mapping[str, Any],
) -> str:
    if set(nested_partitions) != set(train_symbols):
        raise ValueError("V4 nested partitions must match train_symbols")
    pairs: list[tuple[str, str]] = []
    for symbol in train_symbols:
        digest = getattr(nested_partitions[symbol], "digest", None)
        if not isinstance(digest, str):
            raise ValueError("V4 nested partition digest is unavailable")
        require_sha256(digest, field=f"V4 nested partition digest {symbol}")
        pairs.append((symbol, digest))
    return content_digest(
        {
            "partitions": tuple(pairs),
            "schema_version": "causal_alpha_v4_nested_scope_v1",
        }
    )


def prepare_causal_alpha_v4_stage_data(
    *,
    config_digest: str,
    generator_code_digest: str,
    runtime_context: object,
    runtime: object,
    prepared_v3: object,
) -> CausalAlphaV4PreparedStageData:
    """Build train-only V4 samples once and close nested/run identities before Signal."""

    require_sha256(config_digest, field="V4 stage config_digest")
    require_sha256(generator_code_digest, field="V4 stage generator_code_digest")
    symbols = tuple(getattr(prepared_v3, "train_symbols", ()))
    if (
        not symbols
        or len(set(symbols)) != len(symbols)
        or any(not isinstance(symbol, str) or not symbol for symbol in symbols)
    ):
        raise ValueError("V4 stage prepared V3 train_symbols are invalid")

    base_manifest = getattr(runtime_context, "manifest", None)
    base_runtime_manifest_digest = getattr(base_manifest, "manifest_digest", None)
    fold_train_range = getattr(base_manifest, "fold_train_range", None)
    if not isinstance(base_runtime_manifest_digest, str):
        raise ValueError("V4 stage base runtime manifest digest is unavailable")
    require_sha256(
        base_runtime_manifest_digest,
        field="V4 stage base runtime manifest digest",
    )
    if (
        not isinstance(fold_train_range, tuple)
        or len(fold_train_range) != 2
        or isinstance(fold_train_range[0], bool)
        or not isinstance(fold_train_range[0], int)
        or isinstance(fold_train_range[1], bool)
        or not isinstance(fold_train_range[1], int)
        or not 0 <= fold_train_range[0] < fold_train_range[1]
    ):
        raise ValueError("V4 stage fold train range is invalid")
    train_stop = fold_train_range[1]

    context_manifest = getattr(runtime_context, "v4_context_manifest", None)
    if context_manifest is None:
        raise ValueError("V4 stage requires V4 context manifest")
    v4_context_manifest_digest = getattr(context_manifest, "manifest_digest", None)
    if not isinstance(v4_context_manifest_digest, str):
        raise ValueError("V4 stage context manifest digest is unavailable")
    require_sha256(
        v4_context_manifest_digest,
        field="V4 stage context manifest digest",
    )

    routed = getattr(runtime, "routed_environment_factory", None)
    provider = getattr(routed, "v4_context_provider", None)
    contexts = require_causal_alpha_v4_context_scope(
        train_symbols=symbols,
        provider=provider,
        manifest=context_manifest,
    )

    base_samples = getattr(prepared_v3, "samples", None)
    environment_factories = getattr(prepared_v3, "environment_factories", None)
    signal_delays = getattr(prepared_v3, "signal_delays", None)
    decision_bars = getattr(prepared_v3, "decision_bars", None)
    partitions = getattr(prepared_v3, "partitions", None)
    for field_name, values in (
        ("samples", base_samples),
        ("environment_factories", environment_factories),
        ("signal_delays", signal_delays),
        ("decision_bars", decision_bars),
        ("partitions", partitions),
    ):
        if not isinstance(values, Mapping) or set(values) != set(symbols):
            raise ValueError(f"V4 stage prepared V3 {field_name} scope is invalid")

    samples: dict[str, Any] = {}
    for symbol in symbols:
        factory = environment_factories[symbol]
        if not callable(factory):
            raise TypeError("V4 stage environment factory must be callable")
        environment = factory()
        close = getattr(environment, "close", None)
        if not callable(close):
            raise TypeError("V4 stage environment must be closable")
        try:
            samples[symbol] = build_causal_alpha_v4_symbol_samples(
                base_samples=base_samples[symbol],
                context=contexts[symbol],
                dataset=getattr(environment, "dataset", None),
                train_stop=train_stop,
                signal_delay_decisions=signal_delays[symbol],
                decision_bars=decision_bars[symbol],
            )
        finally:
            close()
    validated_samples = validate_causal_alpha_v4_train_sample_scope(
        train_symbols=symbols,
        samples=samples,
    )

    nested = split_causal_alpha_v3_partitions(
        partitions,
        train_symbols=symbols,
        signal_contract_count=_V4_SIGNAL_CONTRACT_COUNT,
        minimum_economic_contract_count=_V4_MINIMUM_ECONOMIC_CONTRACT_COUNT,
    )
    nested_digest = _nested_scope_digest(
        train_symbols=symbols,
        nested_partitions=nested,
    )
    execution_identity = getattr(prepared_v3, "execution_identity", None)
    execution_identity_digest = getattr(execution_identity, "digest", None)
    if not isinstance(execution_identity_digest, str):
        raise ValueError("V4 stage execution identity digest is unavailable")
    require_sha256(
        execution_identity_digest,
        field="V4 stage execution identity digest",
    )
    run_manifest_digest = build_causal_alpha_v4_stage_run_identity(
        base_runtime_manifest_digest=base_runtime_manifest_digest,
        v4_context_manifest_digest=v4_context_manifest_digest,
        config_digest=config_digest,
        execution_identity_digest=execution_identity_digest,
        nested_partition_digest=nested_digest,
        generator_code_digest=generator_code_digest,
    )
    return CausalAlphaV4PreparedStageData(
        train_symbols=symbols,
        samples=validated_samples,
        nested_partitions=nested,
        nested_partition_digest=nested_digest,
        run_manifest_digest=run_manifest_digest,
        base_runtime_manifest_digest=base_runtime_manifest_digest,
        v4_context_manifest_digest=v4_context_manifest_digest,
        config_digest=config_digest,
        execution_identity_digest=execution_identity_digest,
        generator_code_digest=generator_code_digest,
        prepared_v3=prepared_v3,
    )


__all__ = [
    "CAUSAL_ALPHA_V4_STAGE_RUN_IDENTITY_SCHEMA",
    "CausalAlphaV4PreparedStageData",
    "build_causal_alpha_v4_stage_run_identity",
    "prepare_causal_alpha_v4_stage_data",
    "require_causal_alpha_v4_context_scope",
    "slice_causal_alpha_v4_forecast",
]
