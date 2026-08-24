from __future__ import annotations

from dataclasses import replace

import numpy as np

from trade_rl.data.universal_features import UNIVERSAL_INSTRUMENT_DESCRIPTOR_NAMES
from trade_rl.data.v4_context import V4ContextBlock
from trade_rl.learning.causal_alpha_v4 import (
    CausalAlphaV4FitConfig,
    CausalAlphaV4SymbolSamples,
)
from trade_rl.workflows import universal_causal_alpha_v4_fitting as fitting
from trade_rl.workflows.universal_causal_alpha_v4_fitting import fit_causal_alpha_v4


def _digest(char: str) -> str:
    return char * 64


def _block(
    *,
    decisions: np.ndarray,
    name: str,
    values: np.ndarray,
    source: str,
) -> V4ContextBlock:
    matrix = values[:, None].astype(np.float64)
    return V4ContextBlock(
        feature_names=(name,),
        decision_indices=decisions,
        values=matrix,
        available=np.ones(matrix.shape, dtype=np.bool_),
        staleness_hours=np.zeros(matrix.shape, dtype=np.float64),
        source_digest=_digest(source),
    )


def _sample(*, symbol: str, mutate_future: bool = False) -> CausalAlphaV4SymbolSamples:
    decisions = np.arange(24, dtype=np.int64)
    global_x = np.linspace(-1.0, 1.0, len(decisions), dtype=np.float64)
    local_x = np.sin(np.linspace(0.2, 3.0, len(decisions), dtype=np.float64))
    target_x = np.cos(np.linspace(0.1, 2.5, len(decisions), dtype=np.float64))[:, None]
    descriptors = np.column_stack(
        tuple(
            0.01 * float(index + 1) + decisions.astype(np.float64) * 1e-4
            for index in range(len(UNIVERSAL_INSTRUMENT_DESCRIPTOR_NAMES))
        )
    )
    beta = np.ones(len(decisions), dtype=np.float64)
    if symbol != "BTCUSDT":
        beta[:] = 1.5

    market_4h = 0.012 * global_x + 0.002
    market_24h = 0.020 * global_x + 0.003
    market_72h = 0.030 * global_x + 0.004
    residual_4h = 0.004 * local_x + 0.001
    residual_24h = 0.006 * local_x + 0.0015
    residual_72h = 0.008 * local_x + 0.002
    if symbol == "BTCUSDT":
        labels_4h = market_4h.copy()
        labels_24h = market_24h.copy()
        labels_72h = market_72h.copy()
    else:
        labels_4h = beta * market_4h + residual_4h
        labels_24h = beta * market_24h + residual_24h
        labels_72h = beta * market_72h + residual_72h
    ends = decisions + 1
    if mutate_future:
        future = ends >= 18
        labels_4h[future] += 10.0
        labels_24h[future] -= 20.0
        labels_72h[future] += 30.0

    return CausalAlphaV4SymbolSamples(
        symbol=symbol,
        dataset_id=_digest("a" if symbol == "BTCUSDT" else "b"),
        target_local_feature_names=("target_x",),
        target_local_feature_schema_digest=_digest("c"),
        source_sample_digest=_digest("d" if not mutate_future else "e"),
        source_context_digest=_digest("f"),
        decision_indices=decisions,
        target_local_features=target_x,
        target_local_available=np.ones(target_x.shape, dtype=np.bool_),
        instrument_descriptor_names=UNIVERSAL_INSTRUMENT_DESCRIPTOR_NAMES,
        instrument_descriptors=descriptors,
        instrument_descriptor_available=np.ones(descriptors.shape, dtype=np.bool_),
        local_context=_block(
            decisions=decisions,
            name="local_x",
            values=local_x,
            source="1",
        ),
        global_context=_block(
            decisions=decisions,
            name="global_x",
            values=global_x,
            source="2",
        ),
        beta=beta,
        beta_available=np.ones(len(decisions), dtype=np.bool_),
        labels_4h=labels_4h,
        label_end_indices_4h=ends,
        labels_24h=labels_24h,
        label_end_indices_24h=ends,
        labels_72h=labels_72h,
        label_end_indices_72h=ends,
    )


def _config() -> CausalAlphaV4FitConfig:
    return CausalAlphaV4FitConfig(
        market_ridge_strength=1.0,
        residual_ridge_strength=0.1,
        direction_ridge_strength=0.1,
    )


def test_v4_pooled_surface_preserves_symbol_row_and_feature_order() -> None:
    samples = {
        "BTCUSDT": _sample(symbol="BTCUSDT"),
        "ETHUSDT": _sample(symbol="ETHUSDT"),
    }

    names, features, available = fitting._pooled_shared_feature_surface(
        samples, ("BTCUSDT", "ETHUSDT")
    )

    expected_blocks = []
    expected_available = []
    for symbol in ("BTCUSDT", "ETHUSDT"):
        sample = samples[symbol]
        expected_blocks.append(
            np.column_stack(
                (
                    sample.target_local_features,
                    sample.local_context.values,
                    sample.global_context.values,
                    sample.instrument_descriptors,
                    sample.beta[:, None],
                )
            )
        )
        expected_available.append(
            np.column_stack(
                (
                    sample.target_local_available,
                    sample.local_context.available,
                    sample.global_context.available,
                    sample.instrument_descriptor_available,
                    sample.beta_available[:, None],
                )
            )
        )
    assert names[-1] == "causal_beta"
    np.testing.assert_array_equal(features, np.vstack(expected_blocks))
    np.testing.assert_array_equal(available, np.vstack(expected_available))


def test_v4_fit_uses_global_only_market_and_one_shared_residual_surface() -> None:
    samples = {
        "BTCUSDT": _sample(symbol="BTCUSDT"),
        "ETHUSDT": _sample(symbol="ETHUSDT"),
    }
    fit = fit_causal_alpha_v4(
        train_symbols=("BTCUSDT", "ETHUSDT"),
        samples=samples,
        knowledge_cutoff=18,
        config=_config(),
    )

    assert set(fit.market_models) == {"4h", "24h", "72h"}
    assert set(fit.residual_models) == {"4h", "24h", "72h"}
    assert set(fit.direction_models) == {"4h", "24h", "72h"}
    for model in fit.market_models.values():
        assert model.feature_names == ("global_x",)
        assert model.config.ridge_strength == 1.0

    expected_shared = (
        "target_x",
        "local_x",
        "global_x",
        *UNIVERSAL_INSTRUMENT_DESCRIPTOR_NAMES,
        "causal_beta",
    )
    for model in fit.residual_models.values():
        assert model.feature_names == expected_shared
        assert model.config.ridge_strength == 0.1
    for model in fit.direction_models.values():
        assert model.feature_names == expected_shared
        assert model.config.ridge_strength == 0.1
    for descriptor in UNIVERSAL_INSTRUMENT_DESCRIPTOR_NAMES:
        assert expected_shared.count(descriptor) == 1


def test_v4_fit_forecast_reconstructs_beta_market_plus_shared_residual() -> None:
    samples = {
        "BTCUSDT": _sample(symbol="BTCUSDT"),
        "ETHUSDT": _sample(symbol="ETHUSDT"),
    }
    fit = fit_causal_alpha_v4(
        train_symbols=("BTCUSDT", "ETHUSDT"),
        samples=samples,
        knowledge_cutoff=18,
        config=_config(),
    )
    forecast = fit.predict(samples["ETHUSDT"])

    for horizon in ("4h", "24h", "72h"):
        np.testing.assert_allclose(
            forecast.final_predictions[horizon],
            forecast.beta_scaled_market_contributions[horizon]
            + forecast.residual_predictions[horizon],
            atol=0.0,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            forecast.beta_scaled_market_contributions[horizon],
            samples["ETHUSDT"].beta * forecast.market_predictions[horizon],
            atol=0.0,
            rtol=0.0,
        )
    assert forecast.symbol == "ETHUSDT"
    assert forecast.fit_digest == fit.digest


def test_v4_fit_digest_ignores_labels_not_realized_before_cutoff() -> None:
    before = {
        "BTCUSDT": _sample(symbol="BTCUSDT"),
        "ETHUSDT": _sample(symbol="ETHUSDT"),
    }
    after = {
        "BTCUSDT": _sample(symbol="BTCUSDT", mutate_future=True),
        "ETHUSDT": _sample(symbol="ETHUSDT", mutate_future=True),
    }

    first = fit_causal_alpha_v4(
        train_symbols=("BTCUSDT", "ETHUSDT"),
        samples=before,
        knowledge_cutoff=18,
        config=_config(),
    )
    second = fit_causal_alpha_v4(
        train_symbols=("BTCUSDT", "ETHUSDT"),
        samples=after,
        knowledge_cutoff=18,
        config=_config(),
    )
    assert first.digest == second.digest


def test_v4_fit_digest_binds_train_symbol_order() -> None:
    samples = {
        "BTCUSDT": _sample(symbol="BTCUSDT"),
        "ETHUSDT": _sample(symbol="ETHUSDT"),
    }
    first = fit_causal_alpha_v4(
        train_symbols=("BTCUSDT", "ETHUSDT"),
        samples=samples,
        knowledge_cutoff=18,
        config=_config(),
    )
    second = fit_causal_alpha_v4(
        train_symbols=("ETHUSDT", "BTCUSDT"),
        samples=samples,
        knowledge_cutoff=18,
        config=_config(),
    )
    assert first.digest != second.digest


def test_v4_direction_fit_excludes_exact_zero_labels() -> None:
    btc = _sample(symbol="BTCUSDT")
    eth = _sample(symbol="ETHUSDT")
    labels = eth.labels_4h.copy()
    labels[4] = 0.0
    eth = replace(eth, labels_4h=labels, digest="")
    fit = fit_causal_alpha_v4(
        train_symbols=("BTCUSDT", "ETHUSDT"),
        samples={"BTCUSDT": btc, "ETHUSDT": eth},
        knowledge_cutoff=18,
        config=_config(),
    )
    assert fit.direction_zero_label_counts["4h"] == 1
