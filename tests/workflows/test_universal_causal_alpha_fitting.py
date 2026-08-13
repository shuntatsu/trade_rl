from __future__ import annotations

import numpy as np
import pytest

import trade_rl.workflows.universal_causal_alpha_teacher as causal_alpha_module
from trade_rl.artifacts.hashing import content_digest
from trade_rl.learning.causal_alpha_teacher import (
    CausalAlphaControllerConfig,
    CausalAlphaHorizonMix,
    CausalAlphaRidgeConfig,
)
from trade_rl.learning.episode_oracle_teacher import OracleEpisodeContract
from trade_rl.workflows.universal_causal_alpha_teacher import (
    CausalAlphaEpisodePartition,
    CausalAlphaSymbolSamples,
    build_causal_alpha_episode_batch,
    fit_expanding_causal_alpha_models,
)


def _samples(symbol: str, offset: float) -> CausalAlphaSymbolSamples:
    decisions = np.arange(2, 30, dtype=np.int64)
    signal = decisions.astype(np.float64) + offset
    features = np.column_stack((signal, np.full(signal.shape, offset + 5.0)))
    labels_24h = 0.01 * signal + 0.1
    labels_72h = -0.02 * signal + 0.5
    return CausalAlphaSymbolSamples(
        symbol=symbol,
        dataset_id=content_digest(f"dataset:{symbol}"),
        feature_names=("signal", "descriptor"),
        feature_schema_digest=content_digest("feature-schema"),
        context_digest=content_digest(f"context:{symbol}"),
        reference_equity_mode="initial_capital",
        reference_equity=1_000.0,
        decision_indices=decisions,
        features=features,
        feature_available=np.ones_like(features, dtype=np.bool_),
        labels_24h=labels_24h,
        label_end_indices_24h=decisions + 2,
        labels_72h=labels_72h,
        label_end_indices_72h=decisions + 4,
    )


def test_expanding_fit_uses_only_labels_realized_before_episode_start() -> None:
    blocks = {
        "AAAUSDT": _samples("AAAUSDT", 0.0),
        "BBBUSDT": _samples("BBBUSDT", 10.0),
    }
    fitted = fit_expanding_causal_alpha_models(
        train_symbols=("AAAUSDT", "BBBUSDT"),
        samples=blocks,
        knowledge_cutoff=16,
        ridge_config=CausalAlphaRidgeConfig(ridge_strength=0.1),
    )

    assert fitted.knowledge_cutoff == 16
    assert fitted.model_24h.knowledge_cutoff == 16
    assert fitted.model_72h.knowledge_cutoff == 16
    assert fitted.max_label_end_24h < 16
    assert fitted.max_label_end_72h < 16
    assert fitted.sample_count_24h == 24
    assert fitted.sample_count_72h == 20
    assert fitted.train_symbols == ("AAAUSDT", "BBBUSDT")


def test_expanding_fit_cache_reuses_identical_scope_cutoff_and_ridge() -> None:
    blocks = {
        "AAAUSDT": _samples("AAAUSDT", 0.0),
        "BBBUSDT": _samples("BBBUSDT", 10.0),
    }
    assert hasattr(causal_alpha_module, "CausalAlphaExpandingFitCache")
    cache_type = causal_alpha_module.CausalAlphaExpandingFitCache
    cache = cache_type(
        train_symbols=("AAAUSDT", "BBBUSDT"),
        samples=blocks,
    )
    config = CausalAlphaRidgeConfig(ridge_strength=0.1)

    first = cache.resolve(knowledge_cutoff=16, ridge_config=config)
    second = cache.resolve(knowledge_cutoff=16, ridge_config=config)

    assert second is first
    assert cache.fit_count == 1
    assert cache.hit_count == 1
    assert cache.entry_count == 1


def test_expanding_fit_scope_is_exactly_train_symbols() -> None:
    blocks = {
        "AAAUSDT": _samples("AAAUSDT", 0.0),
        "BBBUSDT": _samples("BBBUSDT", 10.0),
    }
    with pytest.raises(ValueError, match="exactly match train_symbols"):
        fit_expanding_causal_alpha_models(
            train_symbols=("AAAUSDT",),
            samples=blocks,
            knowledge_cutoff=16,
            ridge_config=CausalAlphaRidgeConfig(ridge_strength=0.1),
        )


def _partition(symbol: str) -> CausalAlphaEpisodePartition:
    dataset_id = content_digest(f"dataset:{symbol}")
    contracts = tuple(
        OracleEpisodeContract(
            dataset_id=dataset_id,
            episode_index=index,
            start=start,
            stop=start + 5,
            initial_state_mode="cash" if index == 0 else "baseline",
            initial_weights=np.asarray([0.0 if index == 0 else 0.25]),
        )
        for index, start in enumerate((10, 20))
    )
    return CausalAlphaEpisodePartition(
        contracts=contracts,
        selection_contracts=contracts[:-1],
        holdout_contract=contracts[-1],
        train_start=2,
        train_stop=25,
    )


def test_episode_batch_fits_each_episode_at_its_own_cutoff_and_preserves_initial_state() -> (
    None
):
    symbol = "AAAUSDT"
    samples = {symbol: _samples(symbol, 0.0)}
    fit_cache = causal_alpha_module.CausalAlphaExpandingFitCache(
        train_symbols=(symbol,),
        samples=samples,
    )
    controller = CausalAlphaControllerConfig(
        horizon_mix=CausalAlphaHorizonMix.EQUAL,
        score_scale=2.0,
        entry_threshold=0.01,
        exit_threshold=0.005,
        no_trade_band=0.0,
        max_target_delta=0.5,
    )
    batch, evidence = build_causal_alpha_episode_batch(
        symbol=symbol,
        train_symbols=(symbol,),
        samples=samples,
        partition=_partition(symbol),
        ridge_config=CausalAlphaRidgeConfig(ridge_strength=0.1),
        controller_config=controller,
        fit_cache=fit_cache,
    )

    assert batch.contracts == _partition(symbol).contracts
    assert batch.episode_count == 2
    assert tuple(item.knowledge_cutoff for item in evidence.episodes) == (10, 20)
    assert all(
        item.max_label_end_24h < item.knowledge_cutoff for item in evidence.episodes
    )
    assert all(
        item.max_label_end_72h < item.knowledge_cutoff for item in evidence.episodes
    )
    assert batch.targets[0].shape == (4, 1)
    assert batch.targets[1].shape == (4, 1)
    assert evidence.episodes[1].initial_weight == pytest.approx(0.25)
    assert batch.teacher_config_digest == evidence.digest
    assert fit_cache.fit_count == 2


def test_sample_identity_drift_changes_fit_digest() -> None:
    original = _samples("AAAUSDT", 0.0)
    changed_features = np.asarray(original.features).copy()
    changed_features[0, 0] += 1.0
    changed = CausalAlphaSymbolSamples(
        symbol=original.symbol,
        dataset_id=original.dataset_id,
        feature_names=original.feature_names,
        feature_schema_digest=original.feature_schema_digest,
        context_digest=original.context_digest,
        reference_equity_mode=original.reference_equity_mode,
        reference_equity=original.reference_equity,
        decision_indices=original.decision_indices,
        features=changed_features,
        feature_available=original.feature_available,
        labels_24h=original.labels_24h,
        label_end_indices_24h=original.label_end_indices_24h,
        labels_72h=original.labels_72h,
        label_end_indices_72h=original.label_end_indices_72h,
    )
    config = CausalAlphaRidgeConfig(ridge_strength=0.1)
    first = fit_expanding_causal_alpha_models(
        train_symbols=("AAAUSDT",),
        samples={"AAAUSDT": original},
        knowledge_cutoff=16,
        ridge_config=config,
    )
    second = fit_expanding_causal_alpha_models(
        train_symbols=("AAAUSDT",),
        samples={"AAAUSDT": changed},
        knowledge_cutoff=16,
        ridge_config=config,
    )
    assert first.digest != second.digest
