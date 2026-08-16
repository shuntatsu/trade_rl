from __future__ import annotations

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.learning.causal_alpha_v3 import (
    CausalAlphaV3FitConfig,
    CausalAlphaV3TargetConfig,
)
from trade_rl.learning.episode_oracle_teacher import OracleEpisodeContract
from trade_rl.workflows.universal_causal_alpha_contracts import CausalAlphaSymbolSamples
from trade_rl.workflows.universal_causal_alpha_v3_config import CausalAlphaV3Candidate
from trade_rl.workflows.universal_causal_alpha_v3_teacher import (
    build_causal_alpha_v3_signal_scope_metric,
)


def _sha(token: str) -> str:
    return token * 64


def test_signal_metric_matches_exact_pre_sidecar_oracle() -> None:
    decisions = np.asarray(
        [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16],
        dtype=np.int64,
    )
    signal = decisions.astype(np.float64)
    features = np.column_stack((signal, 0.5 * signal))
    available = np.ones_like(features, dtype=np.bool_)
    available[decisions.tolist().index(11), 1] = False
    samples = CausalAlphaSymbolSamples(
        symbol="AAAUSDT",
        dataset_id=_sha("d"),
        feature_names=("signal", "descriptor"),
        feature_schema_digest=content_digest("paired-feature-schema"),
        context_digest=content_digest("paired-context"),
        reference_equity_mode="initial_capital",
        reference_equity=1_000.0,
        decision_indices=decisions,
        features=features,
        feature_available=available,
        labels_24h=0.001 * signal,
        label_end_indices_24h=decisions + 1,
        labels_72h=0.003 * signal,
        label_end_indices_72h=decisions + 2,
    )
    candidate = CausalAlphaV3Candidate(
        name="diagnostic",
        fit=CausalAlphaV3FitConfig(ridge_strength=0.1),
        target=CausalAlphaV3TargetConfig(
            target_magnitudes=(0.0, 0.05),
            uncertainty_multiplier=1.0,
            execution_cost_multiplier=1.5,
            edge_margin=0.001,
            alpha_rebalance_decisions=2,
            strong_reversal_threshold=0.02,
            max_target_delta=0.05,
        ),
    )
    contract = OracleEpisodeContract(
        dataset_id=_sha("d"),
        episode_index=0,
        start=10,
        stop=16,
        initial_state_mode="cash",
        initial_weights=np.zeros(1, dtype=np.float64),
    )

    metric = build_causal_alpha_v3_signal_scope_metric(
        run_manifest_digest=_sha("a"),
        symbol="AAAUSDT",
        train_symbols=("AAAUSDT",),
        samples={"AAAUSDT": samples},
        contract=contract,
        candidate=candidate,
    )

    assert metric.to_payload() == {
        "artifact_digest": "b043128cf9888da70bd9bcd6447cba45162290162b90924d74d9f1065967df11",
        "cohort_indices": (10, 13),
        "contract_digest": "d96a3f4e52f4ddc92267bcb4db95fb63cfb9113cf356c66ad2e324e7a245ad2d",
        "contract_start": 10,
        "contract_stop": 16,
        "direction_accuracy": 1.0,
        "episode_index": 0,
        "fit_config_digest": "023d4a368bf74749429413811329ec574618e83c5e29d630de50b04bb5241348",
        "fit_digest": "0c48fa154008f959b8fbdb93bb2cf678739810aa5f87a37dce6f9cf14395b96e",
        "forecast_digest": "e23a7b15fd3a94de4a2624e06878b65043349e855476b830a7a71dab8b8dbd40",
        "rank_correlation": 1.0,
        "run_manifest_digest": _sha("a"),
        "sample_count": 2,
        "schema_version": "causal_alpha_v3_signal_scope_v2",
        "symbol": "AAAUSDT",
        "top_bottom_realized_spread": 0.008,
    }
