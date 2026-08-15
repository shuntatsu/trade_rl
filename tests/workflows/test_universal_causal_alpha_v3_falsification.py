from __future__ import annotations

import json
from pathlib import Path

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
from trade_rl.workflows.universal_causal_alpha_v3_identity import (
    CausalAlphaV3ExecutionIdentity,
    CausalAlphaV3RunManifestV2,
)
from trade_rl.workflows.universal_causal_alpha_v3_signal import (
    CausalAlphaV3SignalScopeMetric,
)
from trade_rl.workflows.universal_causal_alpha_v3_teacher_artifacts import (
    UniversalCausalAlphaV3TeacherPackageV2,
)


def _sha(token: str) -> str:
    return token * 64


def _execution_identity() -> CausalAlphaV3ExecutionIdentity:
    return CausalAlphaV3ExecutionIdentity(
        train_symbols=("BTCUSDT",),
        training_contract_digest=_sha("1"),
        instrument_context_schema_digest=_sha("2"),
        source_tree_digest=_sha("3"),
        symbol_runtime_digests=(("BTCUSDT", _sha("4")),),
    )


def _manifest() -> CausalAlphaV3RunManifestV2:
    identity = _execution_identity()
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


def _contract(episode_index: int) -> OracleEpisodeContract:
    return OracleEpisodeContract(
        dataset_id=_sha("d"),
        episode_index=episode_index,
        start=10 + 10 * episode_index,
        stop=15 + 10 * episode_index,
        initial_state_mode="cash",
        initial_weights=np.zeros(1, dtype=np.float64),
    )


def _training_batch() -> EpisodeOracleBatch:
    contract = _contract(0)
    return EpisodeOracleBatch(
        dataset_id=contract.dataset_id,
        teacher_config_digest=_sha("a"),
        sampling_config_digest=_sha("b"),
        contracts=(contract,),
        targets=(np.zeros((4, 1), dtype=np.float32),),
        solver_provenance=None,
    )


def _package() -> UniversalCausalAlphaV3TeacherPackageV2:
    return UniversalCausalAlphaV3TeacherPackageV2(
        train_symbols=("BTCUSDT",),
        batches={"BTCUSDT": _training_batch()},
        partition_digests={"BTCUSDT": _sha("6")},
        sample_digests={"BTCUSDT": _sha("7")},
        admission_contract_digests={"BTCUSDT": _contract(1).digest},
        run_manifest_digest=_sha("1"),
        freeze_digest=_sha("2"),
        selection_digest=_sha("3"),
        teacher_admission_digest=_sha("4"),
        selected_candidate_digest=_sha("5"),
        generator_code_digest=_sha("8"),
    )


def _signal_metric() -> CausalAlphaV3SignalScopeMetric:
    return CausalAlphaV3SignalScopeMetric(
        fit_config_digest=_sha("f"),
        symbol="BTCUSDT",
        episode_index=0,
        contract_digest=_sha("c"),
        fit_digest=_sha("a"),
        forecast_digest=_sha("b"),
        sample_count=2,
        rank_correlation=0.2,
        direction_accuracy=0.7,
        top_bottom_realized_spread=0.02,
        cohort_indices=(10, 20),
    )


def _admission_record(
    *,
    symbol: str = "BTCUSDT",
    gross: float = 0.01,
    net: float = 0.005,
    execution_rejections: tuple[tuple[str, int], ...] = (),
) -> CausalAlphaV3AdmissionRecordV2:
    return CausalAlphaV3AdmissionRecordV2(
        run_manifest_digest=_sha("1"),
        freeze_digest=_sha("2"),
        selection_digest=_sha("3"),
        selected_candidate_digest=_sha("4"),
        symbol=symbol,
        contract_digest=_sha("5"),
        gross_return=gross,
        net_return=net,
        turnover_per_day=0.1,
        total_execution_cost=1.0,
        trade_count=1,
        maximum_drawdown=0.01,
        execution_rejection_reason_counts=execution_rejections,
    )


def test_execution_identity_rejects_scope_order_and_digest_tampering() -> None:
    with pytest.raises(ValueError, match="scope/order"):
        CausalAlphaV3ExecutionIdentity(
            train_symbols=("BTCUSDT", "ETHUSDT"),
            training_contract_digest=_sha("1"),
            instrument_context_schema_digest=_sha("2"),
            source_tree_digest=_sha("3"),
            symbol_runtime_digests=(
                ("ETHUSDT", _sha("4")),
                ("BTCUSDT", _sha("5")),
            ),
        )

    identity = _execution_identity()
    with pytest.raises(ValueError, match="digest mismatch"):
        CausalAlphaV3ExecutionIdentity(
            train_symbols=identity.train_symbols,
            training_contract_digest=identity.training_contract_digest,
            instrument_context_schema_digest=identity.instrument_context_schema_digest,
            source_tree_digest=identity.source_tree_digest,
            symbol_runtime_digests=identity.symbol_runtime_digests,
            digest=_sha("0"),
        )


def test_execution_identity_loader_rejects_missing_unknown_and_wrong_schema() -> None:
    payload = _execution_identity().to_payload()
    missing = dict(payload)
    missing.pop("source_tree_digest")
    with pytest.raises(ValueError, match="fields mismatch"):
        CausalAlphaV3ExecutionIdentity.from_payload(missing)

    unknown = dict(payload)
    unknown["unexpected"] = 1
    with pytest.raises(ValueError, match="fields mismatch"):
        CausalAlphaV3ExecutionIdentity.from_payload(unknown)

    wrong_schema = dict(payload)
    wrong_schema["schema_version"] = "tampered"
    with pytest.raises(ValueError, match="schema"):
        CausalAlphaV3ExecutionIdentity.from_payload(wrong_schema)


def test_run_manifest_loader_rejects_safety_flag_and_schema_tampering() -> None:
    payload = _manifest().to_payload()
    promoted = dict(payload)
    promoted["promotion_eligible"] = True
    with pytest.raises(ValueError, match="safety flags"):
        CausalAlphaV3RunManifestV2.from_payload(promoted)

    wrong_schema = dict(payload)
    wrong_schema["schema_version"] = "tampered"
    with pytest.raises(ValueError, match="schema"):
        CausalAlphaV3RunManifestV2.from_payload(wrong_schema)


def test_run_manifest_constructor_rejects_non_research_and_digest_tampering() -> None:
    manifest = _manifest()
    kwargs = {
        "train_symbols": manifest.train_symbols,
        "config_digest": manifest.config_digest,
        "catalog_digest": manifest.catalog_digest,
        "partition_digest": manifest.partition_digest,
        "split_manifest_digest": manifest.split_manifest_digest,
        "feature_schema_digest": manifest.feature_schema_digest,
        "statistics_digest": manifest.statistics_digest,
        "generator_code_digest": manifest.generator_code_digest,
        "nested_partition_digest": manifest.nested_partition_digest,
        "execution_identity_digest": manifest.execution_identity_digest,
        "training_contract_digest": manifest.training_contract_digest,
        "instrument_context_schema_digest": manifest.instrument_context_schema_digest,
    }
    with pytest.raises(ValueError, match="research-only"):
        CausalAlphaV3RunManifestV2(**kwargs, research_only=False)
    with pytest.raises(ValueError, match="digest mismatch"):
        CausalAlphaV3RunManifestV2(**kwargs, digest=_sha("0"))


def test_signal_store_rejects_unknown_scope_and_contract_drift(tmp_path: Path) -> None:
    metric = _signal_metric()
    store = CausalAlphaV3ArtifactStore(tmp_path, run_manifest_digest=_sha("1"))
    store.write_signal_scope_metric(metric)

    with pytest.raises(ValueError, match="outside the expected scope"):
        store.load_signal_scope_metrics(expected={})
    with pytest.raises(ValueError, match="contract identity drifted"):
        store.load_signal_scope_metrics(expected={metric.identity: _sha("0")})


def test_teacher_loader_rejects_missing_batch_leaf(tmp_path: Path) -> None:
    package = _package()
    store = CausalAlphaV3ArtifactStore(
        tmp_path,
        run_manifest_digest=package.run_manifest_digest,
        freeze_digest=package.freeze_digest,
    )
    store.write_teacher_package(package)
    (tmp_path / "teacher" / "batches" / "BTCUSDT.json").unlink()

    with pytest.raises(ValueError, match="batch artifact missing"):
        store.load_teacher_package()


def test_teacher_loader_rejects_package_schema_tampering(tmp_path: Path) -> None:
    package = _package()
    store = CausalAlphaV3ArtifactStore(
        tmp_path,
        run_manifest_digest=package.run_manifest_digest,
        freeze_digest=package.freeze_digest,
    )
    package_path = store.write_teacher_package(package)
    raw = json.loads(package_path.read_text(encoding="utf-8"))
    raw["schema_version"] = "tampered"
    package_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="schema"):
        store.load_teacher_package()


def test_lock_release_rejects_ownership_tampering(tmp_path: Path) -> None:
    lock = CausalAlphaV3RunLock(tmp_path).acquire()
    lock.path.write_text("different-owner", encoding="utf-8")
    try:
        with pytest.raises(RuntimeError, match="ownership changed"):
            lock.release()
    finally:
        lock.path.unlink(missing_ok=True)


def test_v3_admission_allows_explained_rejections_but_rejects_majority_negative() -> (
    None
):
    explained = _admission_record(execution_rejections=(("below_minimum_notional", 2),))
    evidence = evaluate_causal_alpha_v3_admission_gate((explained,))
    assert evidence.passed is True
    assert evidence.unexplained_execution_rejection_count == 0

    negative_a = _admission_record(symbol="AAAUSDT", gross=-0.01, net=0.01)
    negative_b = _admission_record(symbol="BBBUSDT", gross=-0.01, net=0.01)
    positive = _admission_record(symbol="CCCUSDT", gross=0.03, net=0.01)
    rejected = evaluate_causal_alpha_v3_admission_gate(
        (negative_a, negative_b, positive)
    )
    assert rejected.passed is False
    assert "majority_negative_gross_holdouts" in rejected.rejection_reasons


def test_admission_record_rejects_duplicate_and_unsorted_reason_counts() -> None:
    with pytest.raises(ValueError, match="invalid reason counts"):
        _admission_record(
            execution_rejections=(
                ("zero_quantity_after_rounding", 1),
                ("below_minimum_notional", 1),
            )
        )
    with pytest.raises(ValueError, match="invalid reason counts"):
        _admission_record(
            execution_rejections=(
                ("below_minimum_notional", 1),
                ("below_minimum_notional", 1),
            )
        )
