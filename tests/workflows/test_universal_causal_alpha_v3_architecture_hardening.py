from __future__ import annotations

import importlib
import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from trade_rl.learning.episode_oracle_teacher import (
    EpisodeOracleBatch,
    OracleEpisodeContract,
)
from trade_rl.workflows.universal_causal_alpha_v3_config import CausalAlphaV3SignalGate
from trade_rl.workflows.universal_causal_alpha_v3_contracts import (
    CausalAlphaV3AdmissionRecord,
    CausalAlphaV3RunManifest,
    UniversalCausalAlphaV3TeacherPackage,
)
from trade_rl.workflows.universal_causal_alpha_v3_signal import (
    CausalAlphaV3SignalScopeMetric,
    evaluate_causal_alpha_v3_signal_gate,
)
from trade_rl.workflows.universal_causal_alpha_v3_store import CausalAlphaV3RecordStore


def _sha(token: str) -> str:
    return token * 64


def _contract(*, episode_index: int = 0) -> OracleEpisodeContract:
    return OracleEpisodeContract(
        dataset_id=_sha("d"),
        episode_index=episode_index,
        start=10 + episode_index * 10,
        stop=15 + episode_index * 10,
        initial_state_mode="cash",
        initial_weights=np.zeros(1, dtype=np.float64),
    )


def _batch() -> EpisodeOracleBatch:
    contract = _contract()
    return EpisodeOracleBatch(
        dataset_id=contract.dataset_id,
        teacher_config_digest=_sha("t"),
        sampling_config_digest=_sha("s"),
        contracts=(contract,),
        targets=(np.zeros((4, 1), dtype=np.float32),),
    )


def _signal_metric(*, symbol: str, episode_index: int = 0) -> CausalAlphaV3SignalScopeMetric:
    return CausalAlphaV3SignalScopeMetric(
        fit_config_digest=_sha("f"),
        symbol=symbol,
        episode_index=episode_index,
        contract_digest=_sha("c"),
        fit_digest=_sha("a"),
        forecast_digest=_sha("b"),
        sample_count=2,
        rank_correlation=0.2,
        direction_accuracy=0.7,
        top_bottom_realized_spread=0.2,
        cohort_indices=(10, 20),
    )


def _admission_record(**overrides: object) -> CausalAlphaV3AdmissionRecord:
    values: dict[str, object] = {
        "run_manifest_digest": _sha("1"),
        "freeze_digest": _sha("2"),
        "selection_digest": _sha("3"),
        "selected_candidate_digest": _sha("4"),
        "symbol": "BTCUSDT",
        "contract_digest": _sha("5"),
        "gross_return": 0.02,
        "net_return": 0.01,
        "turnover_per_day": 0.2,
        "total_execution_cost": 1.0,
        "trade_count": 2,
        "maximum_drawdown": 0.01,
    }
    values.update(overrides)
    return CausalAlphaV3AdmissionRecord(**values)  # type: ignore[arg-type]


def test_signal_scope_metric_has_strict_round_trip_loader() -> None:
    metric = _signal_metric(symbol="BTCUSDT")
    loader = getattr(CausalAlphaV3SignalScopeMetric, "from_payload")

    restored = loader(metric.to_payload())

    assert restored == metric
    tampered = metric.to_payload()
    tampered["schema_version"] = "tampered"
    with pytest.raises(ValueError, match="schema"):
        loader(tampered)


def test_signal_gate_bootstraps_chronological_episode_clusters_not_symbol_duplicates() -> None:
    metrics = tuple(
        _signal_metric(symbol=symbol)
        for symbol in ("BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT")
    )
    gate = CausalAlphaV3SignalGate(
        minimum_scope_count=2,
        minimum_scope_coverage=1.0,
        minimum_rank_ic_lower_ci=0.1,
        minimum_top_bottom_spread_lower_ci=0.1,
        minimum_direction_accuracy_excess_lower_ci=0.1,
        bootstrap_resamples=100,
        bootstrap_seed=0,
        bootstrap_block_size=1,
    )

    evidence = evaluate_causal_alpha_v3_signal_gate(
        metrics,
        expected_scope_count=len(metrics),
        gate=gate,
    )

    assert evidence.passed is False
    assert "scope_count" in evidence.rejection_reasons


def test_admission_record_rejects_tampered_schema() -> None:
    record = _admission_record()
    raw = record.to_payload()
    raw["schema_version"] = "tampered"

    with pytest.raises(ValueError, match="schema"):
        CausalAlphaV3AdmissionRecord.from_payload(raw)


def test_v3_admission_gate_rejects_net_negative_hard_risk_and_unexplained_rejections() -> None:
    contracts = importlib.import_module(
        "trade_rl.workflows.universal_causal_alpha_v3_contracts"
    )
    gate = getattr(contracts, "evaluate_causal_alpha_v3_admission_gate")
    record = _admission_record(
        gross_return=0.02,
        net_return=-0.01,
        hard_risk_violation=True,
        execution_rejection_reason_counts=(("insufficient_margin", 1),),
        risk_projection_reason_counts=(("market_notional_cap", 1),),
    )

    evidence = gate((record,))

    assert evidence.passed is False
    assert "negative_aggregate_net_return" in evidence.rejection_reasons
    assert "hard_risk_violation" in evidence.rejection_reasons
    assert "unexplained_execution_rejection" in evidence.rejection_reasons


def test_run_manifest_binds_execution_and_runtime_semantics() -> None:
    manifest = CausalAlphaV3RunManifest(
        train_symbols=("BTCUSDT",),
        config_digest=_sha("1"),
        catalog_digest=_sha("2"),
        partition_digest=_sha("3"),
        split_manifest_digest=_sha("4"),
        feature_schema_digest=_sha("5"),
        statistics_digest=_sha("6"),
        generator_code_digest=_sha("7"),
        nested_partition_digest=_sha("8"),
        execution_identity_digest=_sha("9"),
        training_contract_digest=_sha("a"),
        instrument_context_schema_digest=_sha("b"),
    )

    assert manifest.execution_identity_digest == _sha("9")
    assert manifest.training_contract_digest == _sha("a")
    assert manifest.instrument_context_schema_digest == _sha("b")


def test_teacher_package_is_durable_reloadable_and_immutable(tmp_path) -> None:
    package = UniversalCausalAlphaV3TeacherPackage(
        train_symbols=("BTCUSDT",),
        batches={"BTCUSDT": _batch()},
        partition_digests={"BTCUSDT": _sha("6")},
        sample_digests={"BTCUSDT": _sha("7")},
        run_manifest_digest=_sha("1"),
        freeze_digest=_sha("2"),
        selection_digest=_sha("3"),
        teacher_admission_digest=_sha("4"),
        selected_candidate_digest=_sha("5"),
        generator_code_digest=_sha("8"),
    )
    store = CausalAlphaV3RecordStore(
        tmp_path,
        run_manifest_digest=package.run_manifest_digest,
        freeze_digest=package.freeze_digest,
    )
    writer = getattr(store, "write_teacher_package")
    loader = getattr(store, "load_teacher_package")

    writer(package)
    restored = loader()

    assert restored.digest == package.digest
    assert np.array_equal(
        restored.batches["BTCUSDT"].targets[0], package.batches["BTCUSDT"].targets[0]
    )
    with pytest.raises(TypeError):
        package.batches["ETHUSDT"] = _batch()  # type: ignore[index]


def test_signal_scope_records_are_persisted_as_leaf_artifacts(tmp_path) -> None:
    store = CausalAlphaV3RecordStore(tmp_path, run_manifest_digest=_sha("1"))
    metric = _signal_metric(symbol="BTCUSDT")
    writer = getattr(store, "write_signal_scope_metric")
    loader = getattr(store, "load_signal_scope_metrics")

    writer(metric)
    restored = loader(expected={metric.identity: metric.contract_digest})

    assert restored[metric.identity] == metric
    assert (
        tmp_path
        / "signal"
        / "records"
        / metric.fit_config_digest
        / metric.symbol
        / f"{metric.episode_index}.json"
    ).is_file()


def test_replay_requires_contract_initial_state_to_match_environment() -> None:
    replay = importlib.import_module(
        "trade_rl.workflows.universal_causal_alpha_v3_replay"
    )
    assert_initial_state = getattr(replay, "assert_causal_alpha_v3_contract_initial_state")
    contract = _contract()

    class FakeEnvironment:
        dataset = SimpleNamespace(n_symbols=1)

        def initial_weights_for_reset(self, mode: str, start: int) -> np.ndarray:
            assert mode == contract.initial_state_mode
            assert start == contract.start
            return np.array([0.25], dtype=np.float64)

    with pytest.raises(ValueError, match="initial.*state"):
        assert_initial_state(FakeEnvironment(), contract)


def test_v3_output_root_enforces_single_writer(tmp_path) -> None:
    store_module = importlib.import_module(
        "trade_rl.workflows.universal_causal_alpha_v3_store"
    )
    lock_type = getattr(store_module, "CausalAlphaV3RunLock")
    first = lock_type(tmp_path)
    first.acquire()
    try:
        second = lock_type(tmp_path)
        with pytest.raises(RuntimeError, match="lock|writer"):
            second.acquire()
    finally:
        first.release()


def test_runner_is_a_thin_orchestration_facade() -> None:
    runner = importlib.import_module(
        "trade_rl.workflows.universal_causal_alpha_v3_runner"
    )
    source = inspect.getsource(runner)

    assert len(source.splitlines()) <= 180
