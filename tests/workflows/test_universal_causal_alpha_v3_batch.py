from __future__ import annotations

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.learning.causal_alpha_v3 import (
    CausalAlphaV3FitConfig,
    CausalAlphaV3TargetConfig,
)
from trade_rl.learning.episode_oracle_teacher import OracleEpisodeContract
from trade_rl.workflows.universal_causal_alpha_contracts import CausalAlphaSymbolSamples
from trade_rl.workflows.universal_causal_alpha_v3 import (
    CausalAlphaV3FitCache,
    build_causal_alpha_v3_contract_targets,
)
from trade_rl.workflows.universal_causal_alpha_v3_contracts import (
    CausalAlphaV3CandidateConfig,
)


def _samples(symbol: str) -> CausalAlphaSymbolSamples:
    decisions = np.arange(2, 22, dtype=np.int64)
    signal = decisions.astype(np.float64) / 100.0
    features = np.column_stack((signal, np.square(signal)))
    return CausalAlphaSymbolSamples(
        symbol=symbol,
        dataset_id=content_digest(f"dataset:{symbol}"),
        feature_names=("signal", "signal_squared"),
        feature_schema_digest=content_digest("feature-schema"),
        context_digest=content_digest(f"context:{symbol}"),
        reference_equity_mode="initial_capital",
        reference_equity=100_000.0,
        decision_indices=decisions,
        features=features,
        feature_available=np.ones_like(features, dtype=np.bool_),
        labels_24h=0.1 * signal,
        label_end_indices_24h=decisions + 2,
        labels_72h=0.3 * signal,
        label_end_indices_72h=decisions + 4,
    )


def _candidate(*, cadence: int = 2) -> CausalAlphaV3CandidateConfig:
    return CausalAlphaV3CandidateConfig(
        name=f"cadence-{cadence}",
        fit=CausalAlphaV3FitConfig(ridge_strength=0.1),
        target=CausalAlphaV3TargetConfig(
            target_magnitudes=(0.0, 0.1, 0.2),
            uncertainty_multiplier=0.0,
            execution_cost_multiplier=1.0,
            edge_margin=0.0,
            alpha_rebalance_decisions=cadence,
            strong_reversal_threshold=2.0,
            max_target_delta=0.1,
        ),
    )


def _contract(symbol: str) -> OracleEpisodeContract:
    return OracleEpisodeContract(
        dataset_id=content_digest(f"dataset:{symbol}"),
        episode_index=3,
        start=14,
        stop=19,
        initial_state_mode="cash",
        initial_weights=np.asarray([0.0]),
    )


def test_v3_cache_reuses_fit_across_controller_candidates() -> None:
    symbol = "BTCUSDT"
    samples = {symbol: _samples(symbol)}
    cache = CausalAlphaV3FitCache(train_symbols=(symbol,), samples=samples)

    first = cache.resolve(knowledge_cutoff=14, config=_candidate(cadence=2).fit)
    second = cache.resolve(knowledge_cutoff=14, config=_candidate(cadence=4).fit)

    assert second is first
    assert cache.fit_count == 1
    assert cache.fit_hit_count == 1


def test_v3_contract_targets_reuse_prediction_and_obey_liquidity_caps() -> None:
    symbol = "BTCUSDT"
    samples = {symbol: _samples(symbol)}
    cache = CausalAlphaV3FitCache(train_symbols=(symbol,), samples=samples)
    contract = _contract(symbol)
    costs = np.full(4, 0.0009)
    caps = np.full(4, 0.05)

    first = build_causal_alpha_v3_contract_targets(
        symbol=symbol,
        samples=samples,
        contract=contract,
        candidate=_candidate(cadence=2),
        fit_cache=cache,
        one_way_cost_rates=costs,
        liquidity_weight_caps=caps,
    )
    second = build_causal_alpha_v3_contract_targets(
        symbol=symbol,
        samples=samples,
        contract=contract,
        candidate=_candidate(cadence=4),
        fit_cache=cache,
        one_way_cost_rates=costs,
        liquidity_weight_caps=caps,
    )

    assert first.fit_digest == second.fit_digest
    assert first.forecast_digest == second.forecast_digest
    assert cache.fit_count == 1
    assert cache.prediction_count == 1
    assert cache.prediction_hit_count == 1
    assert first.targets.shape == (4, 1)
    assert np.max(np.abs(first.targets)) <= 0.0500001
    assert first.knowledge_cutoff == contract.start
    assert not first.targets.flags.writeable
