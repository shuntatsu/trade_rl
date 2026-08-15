from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.learning.causal_alpha_v3 import (
    CausalAlphaV3FitConfig,
    CausalAlphaV3TargetConfig,
)
from trade_rl.learning.episode_oracle_teacher import OracleEpisodeContract
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.workflows.universal_causal_alpha_contracts import CausalAlphaSymbolSamples
from trade_rl.workflows.universal_causal_alpha_v3_artifact_store import (
    CausalAlphaV3ArtifactStore,
)
from trade_rl.workflows.universal_causal_alpha_v3_config import (
    CausalAlphaV3Candidate,
    CausalAlphaV3SelectionGate,
)
from trade_rl.workflows.universal_causal_alpha_v3_replay import (
    evaluate_causal_alpha_v3_selection,
)
from trade_rl.workflows.universal_causal_alpha_v3_signal import (
    CausalAlphaV3NestedPartition,
)
from trade_rl.workflows.universal_causal_alpha_v3_teacher import (
    CausalAlphaV3ContractTargets,
)


def _candidate() -> CausalAlphaV3Candidate:
    return CausalAlphaV3Candidate(
        name="baseline",
        fit=CausalAlphaV3FitConfig(ridge_strength=0.1),
        target=CausalAlphaV3TargetConfig(
            target_magnitudes=(0.0, 0.05, 0.1),
            uncertainty_multiplier=1.0,
            execution_cost_multiplier=1.5,
            edge_margin=0.001,
            alpha_rebalance_decisions=2,
            strong_reversal_threshold=0.02,
            max_target_delta=0.1,
        ),
    )


def _contract(symbol: str, episode: int, start: int) -> OracleEpisodeContract:
    return OracleEpisodeContract(
        dataset_id=content_digest(f"dataset:{symbol}"),
        episode_index=episode,
        start=start,
        stop=start + 6,
        initial_state_mode="cash",
        initial_weights=np.zeros(1, dtype=np.float64),
    )


def _samples(symbol: str) -> CausalAlphaSymbolSamples:
    decisions = np.arange(2, 30, dtype=np.int64)
    features = np.column_stack(
        (decisions.astype(np.float64), np.ones(decisions.size, dtype=np.float64))
    )
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
        labels_24h=0.001 * decisions,
        label_end_indices_24h=decisions + 1,
        labels_72h=0.002 * decisions,
        label_end_indices_72h=decisions + 2,
    )


def _nested(symbol: str) -> CausalAlphaV3NestedPartition:
    return CausalAlphaV3NestedPartition(
        signal_contracts=(_contract(symbol, 0, 8),),
        economic_contracts=(_contract(symbol, 1, 15), _contract(symbol, 2, 22)),
        holdout_contract=_contract(symbol, 3, 29),
    )


def _gate() -> CausalAlphaV3SelectionGate:
    return CausalAlphaV3SelectionGate(
        minimum_mean_gross_return=0.0,
        minimum_mean_net_return=0.0,
        minimum_symbol_episode_net_return=-0.05,
        maximum_mean_turnover_per_day=1.0,
        maximum_unexplained_execution_rejections=0,
        minimum_positive_gross_episode_fraction=0.5,
    )


def _targets(**kwargs) -> CausalAlphaV3ContractTargets:
    contract = kwargs["contract"]
    size = contract.stop - contract.start - 1
    target_values = np.asarray((-0.1, -0.1, 0.0, 0.1, 0.1), dtype=np.float64)
    assert size == target_values.size
    target_path = SimpleNamespace(
        digest="5" * 64,
        targets=target_values,
        expected_returns=np.asarray((-0.02, -0.01, 0.0, 0.01, 0.02)),
        uncertainties=np.asarray((0.02, 0.02, 0.01, 0.02, 0.02)),
        liquidity_weight_caps=np.full(size, 0.2, dtype=np.float64),
        chosen_objectives=np.asarray((0.01, 0.01, 0.0, 0.01, 0.01)),
        stay_objectives=np.zeros(size, dtype=np.float64),
        submitted_change_count=2,
        sign_flip_count=1,
        liquidity_deleveraging_count=0,
        reasons=("rebalance", "hold", "hold", "hold", "rebalance"),
    )
    return CausalAlphaV3ContractTargets(
        actions=target_values.reshape(-1, 1).astype(np.float32),
        fit_digest="3" * 64,
        forecast_digest="4" * 64,
        target_path=target_path,
    )


def _evaluation(environment, contract, *, actions):
    del environment, contract, actions
    return SimpleNamespace(
        performance=SimpleNamespace(
            gross_return=0.02,
            net_return=0.01,
            turnover_total=0.2,
            cost_total=1.0,
            trade_count=2,
        ),
        collapse_evidence=SimpleNamespace(
            execution_rejection_reason_counts=(),
            risk_projection_reason_counts=(),
            hard_risk_violation=False,
        ),
    )


def test_selection_writes_diagnostics_and_rebuildable_progress_without_replay_backfill(
    tmp_path: Path,
) -> None:
    symbol = "BTCUSDT"
    candidate = _candidate()
    nested = _nested(symbol)
    run_digest = "1" * 64
    freeze_digest = "2" * 64
    store = CausalAlphaV3ArtifactStore(
        tmp_path,
        run_manifest_digest=run_digest,
        freeze_digest=freeze_digest,
    )
    evaluated: list[int] = []

    def evaluate(environment, contract, *, actions):
        evaluated.append(contract.episode_index)
        return _evaluation(environment, contract, actions=actions)

    def factory():
        return SimpleNamespace(
            dataset=SimpleNamespace(n_symbols=1),
            decision_bars=1,
            initial_weights_for_reset=lambda mode, start: np.zeros(1, dtype=np.float64),
            config=SimpleNamespace(
                execution_cost=ExecutionCostConfig(),
                signal_delay_decisions=1,
            ),
            close=lambda: None,
        )

    first = evaluate_causal_alpha_v3_selection(
        train_symbols=(symbol,),
        samples={symbol: _samples(symbol)},
        nested_partitions={symbol: nested},
        candidates=(candidate,),
        environment_factories={symbol: factory},
        episode_hours=24.0,
        thresholds=_gate(),
        run_manifest_digest=run_digest,
        freeze_digest=freeze_digest,
        store=store,
        build_targets=_targets,
        evaluate_path=evaluate,
    )
    assert evaluated == [1, 2]

    expected = {
        (candidate.digest, symbol, contract.episode_index): contract.digest
        for contract in nested.economic_contracts
    }
    metrics = store.load_replay_metrics(expected_contract_digests=expected)
    diagnostics = store.load_replay_diagnostics(
        expected_replay_metric_digests={
            identity: metric.digest for identity, metric in metrics.items()
        }
    )
    assert set(diagnostics) == set(metrics)
    assert all(
        diagnostics[identity].replay_metric_digest == metric.digest
        for identity, metric in metrics.items()
    )

    progress_path = tmp_path / "selection" / "progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert progress["completed_replay_count"] == 2
    assert progress["diagnostics_completed_count"] == 2
    assert progress["candidates"][0]["completed_scope_count"] == 2

    missing_identity = sorted(diagnostics)[0]
    missing_path = (
        tmp_path
        / "selection"
        / "diagnostics"
        / missing_identity[0]
        / missing_identity[1]
        / f"{missing_identity[2]}.json"
    )
    missing_path.unlink()
    evaluated.clear()

    second = evaluate_causal_alpha_v3_selection(
        train_symbols=(symbol,),
        samples={symbol: _samples(symbol)},
        nested_partitions={symbol: nested},
        candidates=(candidate,),
        environment_factories={symbol: factory},
        episode_hours=24.0,
        thresholds=_gate(),
        run_manifest_digest=run_digest,
        freeze_digest=freeze_digest,
        store=store,
        build_targets=_targets,
        evaluate_path=evaluate,
    )
    assert second.digest == first.digest
    assert evaluated == []
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert progress["completed_replay_count"] == 2
    assert progress["diagnostics_completed_count"] == 1
