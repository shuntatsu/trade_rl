from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.learning.causal_alpha_teacher import CausalAlphaTeacherHoldoutMetric
from trade_rl.learning.causal_alpha_v3 import (
    CausalAlphaV3FitConfig,
    CausalAlphaV3TargetConfig,
)
from trade_rl.learning.episode_oracle_teacher import (
    EpisodeOracleBatch,
    OracleEpisodeContract,
)
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.workflows.universal_causal_alpha_contracts import CausalAlphaSymbolSamples
from trade_rl.workflows.universal_causal_alpha_v3_admission import (
    CausalAlphaV3AdmissionRecordV2,
)
from trade_rl.workflows.universal_causal_alpha_v3_artifact_store import (
    CausalAlphaV3ArtifactStore,
)
from trade_rl.workflows.universal_causal_alpha_v3_config import (
    CausalAlphaV3Candidate,
    CausalAlphaV3SelectionGate,
)
from trade_rl.workflows.universal_causal_alpha_v3_contracts import (
    CausalAlphaV3CandidateEvidence,
    CausalAlphaV3SelectionEvidence,
)
from trade_rl.workflows.universal_causal_alpha_v3_runner import (
    evaluate_causal_alpha_v3_admission,
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


def _selection_gate() -> CausalAlphaV3SelectionGate:
    return CausalAlphaV3SelectionGate(
        minimum_mean_gross_return=0.0,
        minimum_mean_net_return=0.0,
        minimum_symbol_episode_net_return=-0.05,
        maximum_mean_turnover_per_day=1.0,
        maximum_unexplained_execution_rejections=0,
        minimum_positive_gross_episode_fraction=0.5,
    )


def _evaluation(*, gross: float = 0.02, net: float = 0.01) -> SimpleNamespace:
    return SimpleNamespace(
        performance=SimpleNamespace(
            gross_return=gross,
            net_return=net,
            turnover_total=0.2,
            cost_total=1.0,
            trade_count=2,
            maximum_drawdown=0.02,
        ),
        collapse_evidence=SimpleNamespace(
            execution_rejection_reason_counts=(),
            risk_projection_reason_counts=(),
            hard_risk_violation=False,
        ),
    )


def test_selection_resumes_completed_scope_and_closes_environment(
    monkeypatch, tmp_path
) -> None:
    import trade_rl.workflows.universal_causal_alpha_v3_runner as module

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
    opened: list[str] = []
    closed: list[str] = []
    evaluated: list[int] = []

    def targets(**kwargs):
        contract = kwargs["contract"]
        return CausalAlphaV3ContractTargets(
            actions=np.zeros(
                (contract.stop - contract.start - 1, 1), dtype=np.float32
            ),
            fit_digest="3" * 64,
            forecast_digest="4" * 64,
            target_path=SimpleNamespace(
                digest="5" * 64,
                submitted_change_count=1,
                sign_flip_count=0,
                liquidity_deleveraging_count=0,
                reasons=("hold",) * (contract.stop - contract.start - 1),
            ),
        )

    def evaluate(environment, contract, *, actions):
        evaluated.append(contract.episode_index)
        return _evaluation()

    monkeypatch.setattr(module, "build_causal_alpha_v3_contract_targets", targets)
    monkeypatch.setattr(
        module, "evaluate_episode_action_path_on_environment", evaluate
    )

    def factory():
        opened.append(symbol)
        return SimpleNamespace(
            symbol=symbol,
            dataset=SimpleNamespace(n_symbols=1),
            decision_bars=1,
            initial_weights_for_reset=lambda mode, start: np.zeros(
                1, dtype=np.float64
            ),
            config=SimpleNamespace(
                execution_cost=ExecutionCostConfig(),
                signal_delay_decisions=1,
            ),
            close=lambda: closed.append(symbol),
        )

    first = evaluate_causal_alpha_v3_selection(
        train_symbols=(symbol,),
        samples={symbol: _samples(symbol)},
        nested_partitions={symbol: nested},
        candidates=(candidate,),
        environment_factories={symbol: factory},
        episode_hours=24.0,
        thresholds=_selection_gate(),
        run_manifest_digest=run_digest,
        freeze_digest=freeze_digest,
        store=store,
    )
    assert first.selected_candidate_digest == candidate.digest
    assert evaluated == [1, 2]
    assert opened == [symbol]
    assert closed == [symbol]

    evaluated.clear()
    opened.clear()
    closed.clear()
    second = evaluate_causal_alpha_v3_selection(
        train_symbols=(symbol,),
        samples={symbol: _samples(symbol)},
        nested_partitions={symbol: nested},
        candidates=(candidate,),
        environment_factories={symbol: factory},
        episode_hours=24.0,
        thresholds=_selection_gate(),
        run_manifest_digest=run_digest,
        freeze_digest=freeze_digest,
        store=store,
    )
    assert second.digest == first.digest
    assert evaluated == []
    assert opened == []
    assert closed == []


def _batch(symbol: str) -> EpisodeOracleBatch:
    contract = _contract(symbol, 3, 22)
    return EpisodeOracleBatch(
        dataset_id=contract.dataset_id,
        teacher_config_digest="a" * 64,
        sampling_config_digest="b" * 64,
        contracts=(contract,),
        targets=(
            np.zeros(
                (contract.stop - contract.start - 1, 1), dtype=np.float32
            ),
        ),
    )


def _selection(
    candidate: CausalAlphaV3Candidate, freeze_digest: str
) -> CausalAlphaV3SelectionEvidence:
    from trade_rl.workflows.universal_causal_alpha_v3_contracts import (
        CausalAlphaV3ReplayMetric,
    )

    metric = CausalAlphaV3ReplayMetric(
        run_manifest_digest="1" * 64,
        freeze_digest=freeze_digest,
        candidate_digest=candidate.digest,
        symbol="AAAUSDT",
        episode_index=1,
        contract_digest="c" * 64,
        fit_digest="d" * 64,
        forecast_digest="e" * 64,
        target_path_digest="f" * 64,
        gross_return=0.02,
        net_return=0.01,
        turnover_per_day=0.2,
        total_execution_cost=1.0,
        trade_count=2,
        submitted_change_count=1,
        sign_flip_count=0,
        liquidity_deleveraging_count=0,
        execution_rejection_reason_counts=(),
        risk_projection_reason_counts=(),
        target_reason_counts=(("hold", 1),),
        hard_risk_violation=False,
    )
    evidence = CausalAlphaV3CandidateEvidence(
        candidate=candidate,
        episode_metrics=(metric,),
        lower_tail_net_return=0.01,
        mean_gross_return=0.02,
        mean_net_return=0.01,
        turnover_per_day=0.2,
        total_execution_cost=1.0,
        positive_gross_episode_fraction=1.0,
        total_trade_count=2,
        unexplained_execution_rejection_count=0,
        hard_risk_violation=False,
        admissible=True,
        rejection_reasons=(),
    )
    return CausalAlphaV3SelectionEvidence(
        candidates=(evidence,),
        selected_candidate_digest=candidate.digest,
        freeze_digest=freeze_digest,
    )


def test_admission_reuses_persisted_symbol_record_exactly_once(
    monkeypatch, tmp_path
) -> None:
    import trade_rl.workflows.universal_causal_alpha_v3_runner as module

    symbols = ("AAAUSDT", "BBBUSDT")
    candidate = _candidate()
    run_digest = "1" * 64
    freeze_digest = "2" * 64
    selection = _selection(candidate, freeze_digest)
    batches = {symbol: _batch(symbol) for symbol in symbols}
    store = CausalAlphaV3ArtifactStore(
        tmp_path,
        run_manifest_digest=run_digest,
        freeze_digest=freeze_digest,
    )
    first_contract = batches[symbols[0]].contracts[-1]
    store.write_admission_record_v2(
        CausalAlphaV3AdmissionRecordV2(
            run_manifest_digest=run_digest,
            freeze_digest=freeze_digest,
            selection_digest=selection.digest,
            selected_candidate_digest=candidate.digest,
            symbol=symbols[0],
            contract_digest=first_contract.digest,
            gross_return=0.01,
            net_return=0.005,
            turnover_per_day=0.1,
            total_execution_cost=1.0,
            trade_count=1,
            maximum_drawdown=0.01,
        )
    )
    evaluated: list[str] = []

    def evaluate(factory, contract, *, actions):
        symbol = next(
            symbol for symbol in symbols if batches[symbol].contracts[-1] == contract
        )
        evaluated.append(symbol)
        return _evaluation()

    monkeypatch.setattr(module, "evaluate_episode_action_path", evaluate)

    def factory() -> SimpleNamespace:
        return SimpleNamespace(
            dataset=SimpleNamespace(n_symbols=1),
            initial_weights_for_reset=lambda mode, start: np.zeros(
                1, dtype=np.float64
            ),
            close=lambda: None,
        )

    evidence = evaluate_causal_alpha_v3_admission(
        train_symbols=symbols,
        batches=batches,
        environment_factories={symbol: factory for symbol in symbols},
        episode_hours=24.0,
        run_manifest_digest=run_digest,
        freeze_digest=freeze_digest,
        selection=selection,
        store=store,
    )
    assert evidence.passed is True
    assert evaluated == [symbols[1]]
    assert tuple(metric.symbol for metric in evidence.metrics) == symbols
    loaded = store.load_admission_records_v2(
        expected_contract_digests={
            symbol: batches[symbol].contracts[-1].digest for symbol in symbols
        },
        selection_digest=selection.digest,
        selected_candidate_digest=candidate.digest,
    )
    assert set(loaded) == set(symbols)


def test_admission_record_maps_back_to_maintained_holdout_metric() -> None:
    record = CausalAlphaV3AdmissionRecordV2(
        run_manifest_digest="1" * 64,
        freeze_digest="2" * 64,
        selection_digest="3" * 64,
        selected_candidate_digest="4" * 64,
        symbol="BTCUSDT",
        contract_digest="5" * 64,
        gross_return=0.01,
        net_return=0.005,
        turnover_per_day=0.1,
        total_execution_cost=1.0,
        trade_count=1,
        maximum_drawdown=0.01,
    )
    maintained = record.to_holdout_metric()
    assert isinstance(maintained, CausalAlphaTeacherHoldoutMetric)
    assert maintained.symbol == record.symbol
    assert maintained.digest != record.digest
