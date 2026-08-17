from __future__ import annotations

import inspect
from typing import Any, NoReturn

import numpy as np
import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.learning.causal_alpha_v3 import (
    CausalAlphaV3FitConfig,
    CausalAlphaV3TargetConfig,
)
from trade_rl.learning.episode_oracle_teacher import OracleEpisodeContract
from trade_rl.workflows import universal_causal_alpha_v3_teacher as teacher_module
from trade_rl.workflows.universal_causal_alpha_contracts import CausalAlphaSymbolSamples
from trade_rl.workflows.universal_causal_alpha_v3_config import CausalAlphaV3Candidate
from trade_rl.workflows.universal_causal_alpha_v3_pipeline import (
    run_universal_causal_alpha_v3_research_pipeline,
)
from trade_rl.workflows.universal_causal_alpha_v3_signal_diagnostic_codec import (
    signal_diagnostic_scope_from_payload,
)
from trade_rl.workflows.universal_causal_alpha_v3_teacher import (
    build_causal_alpha_v3_signal_scope,
)


def _sha(token: str) -> str:
    return token * 64


def _samples() -> CausalAlphaSymbolSamples:
    decisions = np.asarray(
        [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16],
        dtype=np.int64,
    )
    signal = decisions.astype(np.float64)
    features = np.column_stack((signal, 0.5 * signal))
    available = np.ones_like(features, dtype=np.bool_)
    available[decisions.tolist().index(11), 1] = False
    return CausalAlphaSymbolSamples(
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


def _candidate() -> CausalAlphaV3Candidate:
    return CausalAlphaV3Candidate(
        name="diagnostic-hardening",
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


def _contract() -> OracleEpisodeContract:
    return OracleEpisodeContract(
        dataset_id=_sha("d"),
        episode_index=0,
        start=10,
        stop=16,
        initial_state_mode="cash",
        initial_weights=np.zeros(1, dtype=np.float64),
    )


def _kwargs() -> dict[str, Any]:
    return {
        "run_manifest_digest": _sha("a"),
        "symbol": "AAAUSDT",
        "train_symbols": ("AAAUSDT",),
        "samples": {"AAAUSDT": _samples()},
        "contract": _contract(),
        "candidate": _candidate(),
    }


def _recompute_outer_digest(payload: dict[str, object]) -> None:
    unsigned = dict(payload)
    unsigned.pop("artifact_digest")
    payload["artifact_digest"] = content_digest(unsigned)


def test_pipeline_signal_scope_builder_uses_explicit_paired_protocol() -> None:
    annotation = (
        inspect.signature(run_universal_causal_alpha_v3_research_pipeline)
        .parameters["signal_scope_builder"]
        .annotation
    )

    assert "CausalAlphaV3SignalScopeBuilder" in str(annotation)
    assert "Any" not in str(annotation)


def test_metric_only_signal_scope_does_not_construct_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def diagnostic_must_not_run(**_: object) -> NoReturn:
        raise AssertionError("diagnostic builder must not run")

    monkeypatch.setattr(
        teacher_module,
        "build_causal_alpha_v3_signal_diagnostic_scope",
        diagnostic_must_not_run,
    )

    metric = teacher_module.build_causal_alpha_v3_signal_scope_metric(**_kwargs())

    assert metric.cohort_indices == (10, 13)
    assert metric.sample_count == 2


def test_paired_signal_scope_performs_one_fit_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_fit = teacher_module.fit_causal_alpha_v3
    fit_calls = 0

    def counted_fit(**kwargs: Any):
        nonlocal fit_calls
        fit_calls += 1
        return real_fit(**kwargs)

    monkeypatch.setattr(teacher_module, "fit_causal_alpha_v3", counted_fit)

    build_causal_alpha_v3_signal_scope(**_kwargs())

    assert fit_calls == 1


def test_strict_sidecar_rejects_forged_forecast_with_recomputed_outer_digest() -> None:
    diagnostic = build_causal_alpha_v3_signal_scope(**_kwargs()).diagnostic
    payload = diagnostic.to_payload()
    prediction_rows = [dict(row) for row in payload["prediction_rows"]]
    prediction_rows[0]["prediction_24h"] = (
        float(prediction_rows[0]["prediction_24h"]) + 0.125
    )
    payload["prediction_rows"] = tuple(prediction_rows)
    _recompute_outer_digest(payload)

    with pytest.raises(ValueError, match="forecast identity"):
        signal_diagnostic_scope_from_payload(payload)


def test_strict_sidecar_rejects_forged_rmse_with_recomputed_outer_digest() -> None:
    diagnostic = build_causal_alpha_v3_signal_scope(**_kwargs()).diagnostic
    payload = diagnostic.to_payload()
    model_24h = dict(payload["model_24h"])
    model_24h["weighted_residual_rmse"] = (
        float(model_24h["weighted_residual_rmse"]) + 0.125
    )
    payload["model_24h"] = model_24h
    _recompute_outer_digest(payload)

    with pytest.raises(ValueError, match="forecast identity"):
        signal_diagnostic_scope_from_payload(payload)
