from __future__ import annotations

import importlib
import inspect
from collections.abc import MutableMapping
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from trade_rl.learning.episode_oracle_teacher import (
    EpisodeOracleBatch,
    OracleEpisodeContract,
)
from trade_rl.workflows.universal_causal_alpha_v3_admission import (
    CausalAlphaV3AdmissionRecordV2,
    evaluate_causal_alpha_v3_admission_gate,
)
from trade_rl.workflows.universal_causal_alpha_v3_artifact_store import (
    CausalAlphaV3ArtifactStore,
    CausalAlphaV3RunLock,
)
from trade_rl.workflows.universal_causal_alpha_v3_config import CausalAlphaV3SignalGate
from trade_rl.workflows.universal_causal_alpha_v3_identity import (
    CausalAlphaV3ExecutionIdentity,
    CausalAlphaV3RunManifestV2,
)
from trade_rl.workflows.universal_causal_alpha_v3_signal import (
    CausalAlphaV3SignalScopeMetric,
)
from trade_rl.workflows.universal_causal_alpha_v3_signal_v2 import (
    evaluate_causal_alpha_v3_signal_gate_clustered,
    signal_scope_metric_from_payload,
)
from trade_rl.workflows.universal_causal_alpha_v3_teacher_artifacts import (
    UniversalCausalAlphaV3TeacherPackageV2,
)


def _sha(token: str) -> str:
    assert token in "0123456789abcdef"
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


def _training_batch() -> EpisodeOracleBatch:
    contract = _contract(episode_index=0)
    return EpisodeOracleBatch(
        dataset_id=contract.dataset_id,
        teacher_config_digest=_sha("a"),
        sampling_config_digest=_sha("b"),
        contracts=(contract,),
        targets=(np.zeros((4, 1), dtype=np.float32),),
        solver_provenance=None,
    )


def _signal_metric(
    *, symbol: str, episode_index: int = 0
) -> CausalAlphaV3SignalScopeMetric:
    contract = _contract(episode_index=episode_index)
    return CausalAlphaV3SignalScopeMetric(
        fit_config_digest=_sha("f"),
        symbol=symbol,
        episode_index=episode_index,
        contract_start=contract.start,
        contract_stop=contract.stop,
        contract_digest=_sha("c"),
        fit_digest=_sha("a"),
        forecast_digest=_sha("b"),
        sample_count=2,
        rank_correlation=0.2,
        direction_accuracy=0.7,
        top_bottom_realized_spread=0.2,
        cohort_indices=(10, 20),
    )


def _admission_record(**overrides: object) -> CausalAlphaV3AdmissionRecordV2:
    defaults: dict[str, object] = {
        "gross_return": 0.02,
        "net_return": 0.01,
        "execution_rejection_reason_counts": (),
        "risk_projection_reason_counts": (),
        "hard_risk_violation": False,
    }
    defaults.update(overrides)
    return CausalAlphaV3AdmissionRecordV2(
        run_manifest_digest=_sha("1"),
        freeze_digest=_sha("2"),
        selection_digest=_sha("3"),
        selected_candidate_digest=_sha("4"),
        symbol="BTCUSDT",
        contract_digest=_sha("5"),
        gross_return=float(defaults["gross_return"]),
        net_return=float(defaults["net_return"]),
        turnover_per_day=0.2,
        total_execution_cost=1.0,
        trade_count=2,
        maximum_drawdown=0.01,
        execution_rejection_reason_counts=tuple(
            defaults["execution_rejection_reason_counts"]
        ),
        risk_projection_reason_counts=tuple(defaults["risk_projection_reason_counts"]),
        hard_risk_violation=bool(defaults["hard_risk_violation"]),
    )


def _execution_identity(*, source: str = "6") -> CausalAlphaV3ExecutionIdentity:
    return CausalAlphaV3ExecutionIdentity(
        train_symbols=("BTCUSDT",),
        training_contract_digest=_sha("7"),
        instrument_context_schema_digest=_sha("8"),
        source_tree_digest=_sha(source),
        symbol_runtime_digests=(("BTCUSDT", _sha("9")),),
    )


def _manifest(identity: CausalAlphaV3ExecutionIdentity) -> CausalAlphaV3RunManifestV2:
    return CausalAlphaV3RunManifestV2(
        train_symbols=("BTCUSDT",),
        config_digest=_sha("1"),
        catalog_digest=_sha("2"),
        partition_digest=_sha("3"),
        split_manifest_digest=_sha("4"),
        feature_schema_digest=_sha("5"),
        statistics_digest=_sha("6"),
        generator_code_digest=_sha("7"),
        nested_partition_digest=_sha("8"),
        execution_identity_digest=identity.digest,
        training_contract_digest=identity.training_contract_digest,
        instrument_context_schema_digest=identity.instrument_context_schema_digest,
    )


def test_signal_scope_metric_has_strict_round_trip_loader() -> None:
    metric = _signal_metric(symbol="BTCUSDT")
    assert signal_scope_metric_from_payload(metric.to_payload()) == metric
    tampered = metric.to_payload()
    tampered["schema_version"] = "tampered"
    with pytest.raises(ValueError, match="schema"):
        signal_scope_metric_from_payload(tampered)


def test_signal_gate_bootstraps_chronological_episode_clusters_not_symbol_duplicates() -> (
    None
):
    metrics = tuple(
        _signal_metric(symbol=symbol)
        for symbol in ("BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT")
    )
    gate = CausalAlphaV3SignalGate(
        minimum_independent_episode_count=2,
        minimum_raw_scope_coverage=1.0,
        minimum_rank_ic_lower_ci=0.1,
        minimum_top_bottom_spread_lower_ci=0.1,
        minimum_direction_accuracy_excess_lower_ci=0.1,
        bootstrap_resamples=100,
        bootstrap_seed=0,
        bootstrap_block_size=1,
    )
    evidence = evaluate_causal_alpha_v3_signal_gate_clustered(
        metrics,
        expected_raw_scope_count=len(metrics),
        expected_independent_episode_count=1,
        gate=gate,
    )
    assert evidence.passed is False
    assert evidence.raw_scope_count == 4
    assert evidence.independent_episode_count == 1
    assert "independent_episode_count" in evidence.rejection_reasons


def test_admission_record_rejects_tampered_schema() -> None:
    record = _admission_record()
    raw = record.to_payload()
    raw["schema_version"] = "tampered"
    with pytest.raises(ValueError, match="schema"):
        CausalAlphaV3AdmissionRecordV2.from_payload(raw)


def test_v3_admission_gate_rejects_net_negative_hard_risk_and_unexplained_rejections() -> (
    None
):
    evidence = evaluate_causal_alpha_v3_admission_gate(
        (
            _admission_record(
                gross_return=0.02,
                net_return=-0.01,
                hard_risk_violation=True,
                execution_rejection_reason_counts=(("insufficient_margin", 1),),
                risk_projection_reason_counts=(("market_notional_cap", 1),),
            ),
        )
    )
    assert evidence.passed is False
    assert "negative_aggregate_net_return" in evidence.rejection_reasons
    assert "hard_risk_violation" in evidence.rejection_reasons
    assert "unexplained_execution_rejection" in evidence.rejection_reasons


def test_run_manifest_binds_execution_and_runtime_semantics() -> None:
    identity = _execution_identity()
    manifest = _manifest(identity)
    assert manifest.execution_identity_digest == identity.digest
    assert manifest.training_contract_digest == identity.training_contract_digest
    assert (
        manifest.instrument_context_schema_digest
        == identity.instrument_context_schema_digest
    )
    assert CausalAlphaV3RunManifestV2.from_payload(manifest.to_payload()) == manifest


def test_runtime_drift_cannot_reuse_same_output_root(tmp_path) -> None:
    first = _manifest(_execution_identity(source="6"))
    CausalAlphaV3ArtifactStore(
        tmp_path, run_manifest_digest=first.digest
    ).write_exact_artifact("run-manifest.json", first.to_payload())
    second = _manifest(_execution_identity(source="a"))
    with pytest.raises(ValueError, match="identity drifted"):
        CausalAlphaV3ArtifactStore(
            tmp_path, run_manifest_digest=second.digest
        ).write_exact_artifact("run-manifest.json", second.to_payload())


def test_teacher_package_is_durable_reloadable_immutable_and_training_only(
    tmp_path,
) -> None:
    package = UniversalCausalAlphaV3TeacherPackageV2(
        train_symbols=("BTCUSDT",),
        batches={"BTCUSDT": _training_batch()},
        partition_digests={"BTCUSDT": _sha("6")},
        sample_digests={"BTCUSDT": _sha("7")},
        admission_contract_digests={"BTCUSDT": _contract(episode_index=1).digest},
        run_manifest_digest=_sha("1"),
        freeze_digest=_sha("2"),
        selection_digest=_sha("3"),
        teacher_admission_digest=_sha("4"),
        selected_candidate_digest=_sha("5"),
        generator_code_digest=_sha("8"),
    )
    store = CausalAlphaV3ArtifactStore(
        tmp_path,
        run_manifest_digest=package.run_manifest_digest,
        freeze_digest=package.freeze_digest,
    )
    store.write_teacher_package(package)
    restored = store.load_teacher_package()
    assert restored.digest == package.digest
    assert np.array_equal(
        restored.batches["BTCUSDT"].targets[0],
        package.batches["BTCUSDT"].targets[0],
    )
    with pytest.raises(TypeError):
        cast(MutableMapping[str, EpisodeOracleBatch], package.batches)["ETHUSDT"] = (
            _training_batch()
        )
    with pytest.raises(ValueError, match="admission holdout"):
        UniversalCausalAlphaV3TeacherPackageV2(
            train_symbols=("BTCUSDT",),
            batches={"BTCUSDT": _training_batch()},
            partition_digests={"BTCUSDT": _sha("6")},
            sample_digests={"BTCUSDT": _sha("7")},
            admission_contract_digests={
                "BTCUSDT": _training_batch().contracts[0].digest
            },
            run_manifest_digest=_sha("1"),
            freeze_digest=_sha("2"),
            selection_digest=_sha("3"),
            teacher_admission_digest=_sha("4"),
            selected_candidate_digest=_sha("5"),
            generator_code_digest=_sha("8"),
        )


def test_signal_scope_records_are_persisted_as_leaf_artifacts(tmp_path) -> None:
    store = CausalAlphaV3ArtifactStore(tmp_path, run_manifest_digest=_sha("1"))
    metric = _signal_metric(symbol="BTCUSDT")
    store.write_signal_scope_metric(metric)
    restored = store.load_signal_scope_metrics(
        expected={metric.identity: metric.contract_digest}
    )
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
    assert_initial_state = getattr(
        replay, "assert_causal_alpha_v3_contract_initial_state"
    )
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
    first = CausalAlphaV3RunLock(tmp_path)
    first.acquire()
    try:
        with pytest.raises(RuntimeError, match="lock|writer"):
            CausalAlphaV3RunLock(tmp_path).acquire()
    finally:
        first.release()


def test_runner_is_a_thin_orchestration_facade() -> None:
    runner = importlib.import_module(
        "trade_rl.workflows.universal_causal_alpha_v3_runner"
    )
    assert len(inspect.getsource(runner).splitlines()) <= 180
