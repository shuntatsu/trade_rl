from __future__ import annotations

import numpy as np
import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.learning.causal_alpha_teacher import (
    CausalAlphaControllerConfig,
    CausalAlphaHorizonMix,
    CausalAlphaRidgeConfig,
)
from trade_rl.learning.episode_oracle_teacher import OracleEpisodeContract
from trade_rl.workflows.universal_causal_alpha_contracts import (
    CausalAlphaCandidateConfig,
    CausalAlphaSymbolSamples,
)
from trade_rl.workflows.universal_causal_alpha_selection import (
    _causal_alpha_target_for_contract,
)


def _samples() -> CausalAlphaSymbolSamples:
    # Decision 12 is intentionally absent: it represents an inactive/non-tradable
    # decision. Decision 11 is tradable but has one unavailable feature.
    decisions = np.asarray([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14], dtype=np.int64)
    features = np.column_stack(
        (decisions.astype(np.float64), 0.5 * decisions.astype(np.float64))
    )
    available = np.ones_like(features, dtype=np.bool_)
    available[decisions.tolist().index(11), 1] = False
    return CausalAlphaSymbolSamples(
        symbol="AAAUSDT",
        dataset_id=content_digest("dataset"),
        feature_names=("fast", "slow"),
        feature_schema_digest=content_digest("feature-schema"),
        context_digest=content_digest("context"),
        reference_equity_mode="initial_capital",
        reference_equity=1_000.0,
        decision_indices=decisions,
        features=features,
        feature_available=available,
        labels_24h=0.01 * decisions.astype(np.float64),
        label_end_indices_24h=decisions + 1,
        labels_72h=0.02 * decisions.astype(np.float64),
        label_end_indices_72h=decisions + 2,
    )


def test_target_generation_zero_imputes_missing_features_and_holds_nontradable() -> (
    None
):
    samples = _samples()
    contract = OracleEpisodeContract(
        dataset_id=samples.dataset_id,
        episode_index=0,
        start=10,
        stop=15,
        initial_state_mode="cash",
        initial_weights=np.asarray([0.0], dtype=np.float64),
    )
    candidate = CausalAlphaCandidateConfig(
        name="availability-contract",
        ridge=CausalAlphaRidgeConfig(ridge_strength=0.1),
        controller=CausalAlphaControllerConfig(
            horizon_mix=CausalAlphaHorizonMix.EQUAL,
            score_scale=10.0,
            entry_threshold=0.01,
            exit_threshold=0.005,
            no_trade_band=0.0,
            max_target_delta=2.0,
        ),
    )

    targets = _causal_alpha_target_for_contract(
        symbol="AAAUSDT",
        train_symbols=("AAAUSDT",),
        samples={"AAAUSDT": samples},
        contract=contract,
        candidate=candidate,
    ).reshape(-1)

    assert targets.shape == (4,)
    assert np.isfinite(targets).all()
    # Decision sequence is 10, 11, 12, 13. Decision 12 is not actionable,
    # therefore its target must equal the previous decision's target.
    assert targets[2] == pytest.approx(targets[1])
