from __future__ import annotations

import json
from dataclasses import replace

import pytest

from trade_rl.workflows.universal_causal_alpha_v3_artifact_store import (
    CausalAlphaV3ArtifactStore,
)
from trade_rl.workflows.universal_causal_alpha_v3_signal_diagnostic import (
    CausalAlphaV3SignalDiagnosticModel,
    CausalAlphaV3SignalDiagnosticPredictionRow,
    CausalAlphaV3SignalDiagnosticRealizedRow,
    CausalAlphaV3SignalDiagnosticScope,
    signal_diagnostic_scope_from_payload,
)


def _sha(token: str) -> str:
    return token * 64


def _model(token: str) -> CausalAlphaV3SignalDiagnosticModel:
    return CausalAlphaV3SignalDiagnosticModel(
        model_digest=_sha(token),
        feature_names=("fast", "slow"),
        intercept=0.01,
        coefficients=(0.1, -0.2),
        location=(1.0, 2.0),
        scale=(0.5, 1.5),
        constant_mask=(False, False),
        fitted_row_count=10,
        weighted_residual_rmse=0.02,
        pooled_weighted_ess=6.0,
        per_symbol_weighted_ess=(("BTCUSDT", 6.0),),
        overlap_weight_digest=_sha("e"),
    )


def _prediction() -> CausalAlphaV3SignalDiagnosticPredictionRow:
    return CausalAlphaV3SignalDiagnosticPredictionRow(
        decision_index=10,
        actionable=True,
        available_feature_count=2,
        available_feature_fraction=1.0,
        prediction_24h=0.01,
        prediction_72h=0.03,
        prediction_72h_24h_equivalent=0.01,
        expected_return_24h_equivalent=0.01,
        uncertainty_24h_equivalent=0.02,
        signal_to_uncertainty=0.5,
    )


def _realized() -> CausalAlphaV3SignalDiagnosticRealizedRow:
    return CausalAlphaV3SignalDiagnosticRealizedRow(
        decision_index=10,
        label_end_index=12,
        available_feature_count=2,
        available_feature_fraction=1.0,
        prediction=0.01,
        realized_return=0.012,
    )


def _diagnostic(
    *,
    run_manifest_digest: str | None = None,
    contract_digest: str | None = None,
) -> CausalAlphaV3SignalDiagnosticScope:
    return CausalAlphaV3SignalDiagnosticScope(
        run_manifest_digest=run_manifest_digest or _sha("1"),
        fit_config_digest=_sha("2"),
        symbol="BTCUSDT",
        episode_index=3,
        contract_start=10,
        contract_stop=20,
        contract_digest=contract_digest or _sha("3"),
        signal_metric_digest=_sha("4"),
        fit_digest=_sha("5"),
        forecast_digest=_sha("6"),
        feature_schema_digest=_sha("7"),
        model_24h=_model("8"),
        model_72h=_model("9"),
        prediction_rows=(_prediction(),),
        realized_24h_rows=(_realized(),),
        realized_72h_rows=(_realized(),),
        realized_fused_rows=(_realized(),),
        canonical_cohort_indices=(10,),
        per_feature_available_fraction=(1.0, 1.0),
        complete_feature_row_count=1,
        incomplete_feature_row_count=0,
        available_feature_fraction_minimum=1.0,
        available_feature_fraction_mean=1.0,
        available_feature_fraction_maximum=1.0,
    )


def test_signal_diagnostic_payload_round_trips_with_digest_validation() -> None:
    diagnostic = _diagnostic()

    loaded = signal_diagnostic_scope_from_payload(diagnostic.to_payload())

    assert loaded == diagnostic
    assert loaded.digest == diagnostic.digest

    corrupted = diagnostic.to_payload()
    corrupted["forecast_digest"] = _sha("a")
    with pytest.raises(ValueError, match="digest"):
        signal_diagnostic_scope_from_payload(corrupted)


def test_signal_diagnostic_parser_rejects_unknown_fields_and_unsafe_flags() -> None:
    payload = _diagnostic().to_payload()
    payload["unexpected"] = 1
    with pytest.raises(ValueError, match="fields"):
        signal_diagnostic_scope_from_payload(payload)

    promoted = _diagnostic().to_payload()
    promoted["promotion_eligible"] = True
    with pytest.raises(ValueError, match="research-only"):
        signal_diagnostic_scope_from_payload(promoted)


def test_signal_diagnostic_store_writes_and_loads_exact_scope_path(tmp_path) -> None:
    diagnostic = _diagnostic()
    store = CausalAlphaV3ArtifactStore(
        tmp_path,
        run_manifest_digest=diagnostic.run_manifest_digest,
    )

    path = store.write_signal_diagnostic_scope(diagnostic)
    loaded = store.load_signal_diagnostic_scopes(
        expected={diagnostic.identity: diagnostic.contract_digest}
    )

    assert path == (
        tmp_path
        / "signal"
        / "diagnostics"
        / diagnostic.fit_config_digest
        / diagnostic.symbol
        / f"{diagnostic.episode_index}.json"
    )
    assert loaded == {diagnostic.identity: diagnostic}


def test_signal_diagnostic_store_rejects_cross_run_and_contract_drift(tmp_path) -> None:
    diagnostic = _diagnostic()
    store = CausalAlphaV3ArtifactStore(
        tmp_path,
        run_manifest_digest=diagnostic.run_manifest_digest,
    )

    with pytest.raises(ValueError, match="run manifest"):
        store.write_signal_diagnostic_scope(
            replace(diagnostic, run_manifest_digest=_sha("a"), digest="")
        )

    store.write_signal_diagnostic_scope(diagnostic)
    with pytest.raises(ValueError, match="contract"):
        store.load_signal_diagnostic_scopes(
            expected={diagnostic.identity: _sha("b")}
        )


def test_signal_diagnostic_store_fails_closed_on_corrupt_payload(tmp_path) -> None:
    diagnostic = _diagnostic()
    store = CausalAlphaV3ArtifactStore(
        tmp_path,
        run_manifest_digest=diagnostic.run_manifest_digest,
    )
    path = store.write_signal_diagnostic_scope(diagnostic)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["forecast_digest"] = _sha("a")
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="digest"):
        store.load_signal_diagnostic_scopes(
            expected={diagnostic.identity: diagnostic.contract_digest}
        )


def test_signal_diagnostic_store_rejects_unknown_and_wrong_path_records(tmp_path) -> None:
    diagnostic = _diagnostic()
    store = CausalAlphaV3ArtifactStore(
        tmp_path,
        run_manifest_digest=diagnostic.run_manifest_digest,
    )
    path = store.write_signal_diagnostic_scope(diagnostic)

    with pytest.raises(ValueError, match="outside the expected scope"):
        store.load_signal_diagnostic_scopes(expected={})

    wrong = path.parent.parent / "ETHUSDT" / path.name
    wrong.parent.mkdir(parents=True)
    wrong.write_bytes(path.read_bytes())
    path.unlink()
    with pytest.raises(ValueError, match="path identity"):
        store.load_signal_diagnostic_scopes(
            expected={diagnostic.identity: diagnostic.contract_digest}
        )
