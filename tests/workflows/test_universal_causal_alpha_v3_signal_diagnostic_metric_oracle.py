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


def test_signal_metric_matches_pre_sidecar_gate_oracle() -> None:
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

    # The same pre-sidecar commit produced different model/forecast/artifact
    # digests on separate GitHub-hosted runners while these Gate observations
    # remained identical.  The regression oracle therefore fixes the actual
    # Gate inputs and stable scope identities, not backend-sensitive float
    # digests from the ridge solve.
    assert metric.cohort_indices == (10, 13)
    assert metric.contract_digest == (
        "5d45d643da908ee849bdac155b1fc789394bd6657fd0291ac5df7836afd0cf99"
    )
    assert metric.contract_start == 10
    assert metric.contract_stop == 16
    assert metric.direction_accuracy == 1.0
    assert metric.episode_index == 0
    assert metric.fit_config_digest == (
        "a1cac88f40e07de95b1266eb36bf364ef433050adc1bef4642a2ca25805c3590"
    )
    assert metric.rank_correlation == 1.0
    assert metric.run_manifest_digest == _sha("a")
    assert metric.sample_count == 2
    assert metric.symbol == "AAAUSDT"
    assert metric.top_bottom_realized_spread == 0.003000000000000001

    assert len(metric.fit_digest) == 64
    assert len(metric.forecast_digest) == 64
    assert len(metric.digest) == 64
    assert metric.to_payload()["schema_version"] == "causal_alpha_v3_signal_scope_v2"
