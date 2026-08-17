from __future__ import annotations

import importlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.workflows.test_universal_causal_alpha_v3_signal_forensics import (
    _build_run,
    _digest,
    _load_metric,
    _write_json,
)
from trade_rl.artifacts.hashing import content_digest
from trade_rl.learning.causal_alpha_v3 import causal_alpha_v3_forecast
from trade_rl.workflows.universal_causal_alpha_v3_signal_diagnostic import (
    CausalAlphaV3SignalDiagnosticModel,
    CausalAlphaV3SignalDiagnosticPredictionRow,
    CausalAlphaV3SignalDiagnosticRealizedRow,
    CausalAlphaV3SignalDiagnosticScope,
)
from trade_rl.workflows.universal_causal_alpha_v3_signal_v2 import (
    evaluate_causal_alpha_v3_signal_gate_clustered,
)


def _model(
    *, fit_index: int, episode_index: int, horizon: str
) -> CausalAlphaV3SignalDiagnosticModel:
    horizon_offset = 0.0 if horizon == "24h" else 0.03
    return CausalAlphaV3SignalDiagnosticModel(
        model_digest=_digest(f"model:{fit_index}:{episode_index}:{horizon}"),
        feature_names=("signal", "descriptor"),
        intercept=0.001 * (fit_index + 1) + horizon_offset,
        coefficients=(
            0.1 + 0.01 * episode_index + horizon_offset,
            -0.2 + 0.005 * episode_index - horizon_offset,
        ),
        location=(1.0 + episode_index, 2.0 + 0.5 * episode_index),
        scale=(0.5 + 0.05 * episode_index, 1.5 + 0.05 * episode_index),
        constant_mask=(False, False),
        fitted_row_count=20 + episode_index,
        weighted_residual_rmse=0.02 + horizon_offset + 0.001 * episode_index,
        pooled_weighted_ess=12.0 + episode_index,
        per_symbol_weighted_ess=(
            ("BTCUSDT", 6.0 + 0.25 * episode_index),
            ("ETHUSDT", 6.5 + 0.25 * episode_index),
        ),
        overlap_weight_digest=_digest(
            f"weights:{fit_index}:{episode_index}:{horizon}"
        ),
    )


def _complete_sidecars(root: Path) -> dict[str, Any]:
    built = _build_run(root)
    updated_by_fit: dict[str, list[Any]] = {
        fit_digest: [] for fit_digest in built["fit_configs"]
    }
    diagnostic_paths: dict[tuple[str, str, int], Path] = {}

    for identity, metric_path in sorted(built["paths_by_identity"].items()):
        fit_config_digest, symbol, episode_index = identity
        fit_index = built["fit_configs"].index(fit_config_digest)
        metric = _load_metric(metric_path)
        model_24h = _model(
            fit_index=fit_index, episode_index=episode_index, horizon="24h"
        )
        model_72h = _model(
            fit_index=fit_index, episode_index=episode_index, horizon="72h"
        )
        decisions = np.asarray(metric.cohort_indices, dtype=np.int64)
        symbol_scale = 1.0 if symbol == "BTCUSDT" else 1.2
        base = np.arange(1, decisions.size + 1, dtype=np.float64)
        prediction_24h = 0.001 * symbol_scale * base
        prediction_72h = 0.0024 * symbol_scale * base
        forecast = causal_alpha_v3_forecast(
            prediction_24h,
            prediction_72h,
            residual_rmse_24h=model_24h.weighted_residual_rmse,
            residual_rmse_72h=model_72h.weighted_residual_rmse,
        )
        updated_metric = replace(metric, forecast_digest=forecast.digest, digest="")
        _write_json(metric_path, updated_metric.to_payload())
        updated_by_fit[fit_config_digest].append(updated_metric)

        prediction_rows: list[CausalAlphaV3SignalDiagnosticPredictionRow] = []
        realized_24h: list[CausalAlphaV3SignalDiagnosticRealizedRow] = []
        realized_72h: list[CausalAlphaV3SignalDiagnosticRealizedRow] = []
        realized_fused: list[CausalAlphaV3SignalDiagnosticRealizedRow] = []
        complete_count = 0
        incomplete_count = 0
        available_fractions: list[float] = []
        for row_index, decision_index in enumerate(decisions.tolist()):
            available_count = 2 if row_index % 2 == 0 else 1
            available_fraction = available_count / 2.0
            available_fractions.append(available_fraction)
            if available_count == 2:
                complete_count += 1
            else:
                incomplete_count += 1
            prediction_rows.append(
                CausalAlphaV3SignalDiagnosticPredictionRow(
                    decision_index=decision_index,
                    actionable=True,
                    available_feature_count=available_count,
                    available_feature_fraction=available_fraction,
                    prediction_24h=float(forecast.prediction_24h[row_index]),
                    prediction_72h=float(forecast.prediction_72h[row_index]),
                    prediction_72h_24h_equivalent=float(
                        forecast.prediction_72h[row_index] / 3.0
                    ),
                    expected_return_24h_equivalent=float(
                        forecast.expected_return_24h_equivalent[row_index]
                    ),
                    uncertainty_24h_equivalent=float(
                        forecast.uncertainty_24h_equivalent[row_index]
                    ),
                    signal_to_uncertainty=float(
                        forecast.signal_to_uncertainty[row_index]
                    ),
                )
            )
            realized_24h_value = float(
                forecast.prediction_24h[row_index]
                + (0.0002 if row_index % 2 == 0 else -0.0001)
            )
            prediction_72h_equivalent = float(forecast.prediction_72h[row_index] / 3.0)
            realized_72h_equivalent = prediction_72h_equivalent + (
                0.00015 if row_index % 2 == 0 else -0.00005
            )
            fused_realized = 0.5 * (realized_24h_value + realized_72h_equivalent)
            realized_24h.append(
                CausalAlphaV3SignalDiagnosticRealizedRow(
                    decision_index=decision_index,
                    label_end_index=decision_index + 1,
                    available_feature_count=available_count,
                    available_feature_fraction=available_fraction,
                    prediction=float(forecast.prediction_24h[row_index]),
                    realized_return=realized_24h_value,
                    raw_prediction=None,
                    raw_realized_return=None,
                )
            )
            realized_72h.append(
                CausalAlphaV3SignalDiagnosticRealizedRow(
                    decision_index=decision_index,
                    label_end_index=decision_index + 3,
                    available_feature_count=available_count,
                    available_feature_fraction=available_fraction,
                    prediction=prediction_72h_equivalent,
                    realized_return=realized_72h_equivalent,
                    raw_prediction=float(forecast.prediction_72h[row_index]),
                    raw_realized_return=3.0 * realized_72h_equivalent,
                )
            )
            realized_fused.append(
                CausalAlphaV3SignalDiagnosticRealizedRow(
                    decision_index=decision_index,
                    label_end_index=decision_index + 3,
                    available_feature_count=available_count,
                    available_feature_fraction=available_fraction,
                    prediction=float(
                        forecast.expected_return_24h_equivalent[row_index]
                    ),
                    realized_return=fused_realized,
                    raw_prediction=None,
                    raw_realized_return=None,
                )
            )

        diagnostic = CausalAlphaV3SignalDiagnosticScope(
            run_manifest_digest=built["manifest"].digest,
            fit_config_digest=fit_config_digest,
            symbol=symbol,
            episode_index=episode_index,
            contract_start=updated_metric.contract_start,
            contract_stop=updated_metric.contract_stop,
            contract_digest=updated_metric.contract_digest,
            signal_metric_digest=updated_metric.digest,
            fit_digest=updated_metric.fit_digest,
            forecast_digest=updated_metric.forecast_digest,
            feature_schema_digest=built["manifest"].feature_schema_digest,
            model_24h=model_24h,
            model_72h=model_72h,
            prediction_rows=tuple(prediction_rows),
            realized_24h_rows=tuple(realized_24h),
            realized_72h_rows=tuple(realized_72h),
            realized_fused_rows=tuple(realized_fused),
            canonical_cohort_indices=updated_metric.cohort_indices,
            per_feature_available_fraction=(
                1.0,
                float(np.mean(np.asarray(available_fractions, dtype=np.float64))),
            ),
            complete_feature_row_count=complete_count,
            incomplete_feature_row_count=incomplete_count,
            available_feature_fraction_minimum=min(available_fractions),
            available_feature_fraction_mean=float(
                np.mean(np.asarray(available_fractions, dtype=np.float64))
            ),
            available_feature_fraction_maximum=max(available_fractions),
        )
        diagnostic_path = (
            root
            / "signal"
            / "diagnostics"
            / fit_config_digest
            / symbol
            / f"{episode_index}.json"
        )
        _write_json(diagnostic_path, diagnostic.to_payload())
        diagnostic_paths[identity] = diagnostic_path

    fit_results: list[dict[str, object]] = []
    for fit_digest in built["fit_configs"]:
        metrics = tuple(
            sorted(
                updated_by_fit[fit_digest],
                key=lambda item: (item.symbol, item.episode_index),
            )
        )
        evidence = evaluate_causal_alpha_v3_signal_gate_clustered(
            metrics,
            expected_raw_scope_count=8,
            expected_independent_episode_count=4,
            gate=built["config"].signal_gate,
        )
        assert evidence.passed is False
        fit_results.append(
            {
                "evidence": evidence.to_payload(),
                "fit_config_digest": fit_digest,
                "passed": False,
                "promotion_eligible": False,
                "schema_version": "causal_alpha_v3_fit_signal_result_v2",
                "unavailable_scope_contract_digests": [],
            }
        )
    rejection_body: dict[str, object] = {
        "fit_results": fit_results,
        "promotion_eligible": False,
        "schema_version": "causal_alpha_v3_signal_rejection_v2",
    }
    _write_json(
        built["rejection_path"],
        {**rejection_body, "artifact_digest": content_digest(rejection_body)},
    )
    built["diagnostic_paths"] = diagnostic_paths
    return built


def _loader() -> Any:
    return importlib.import_module(
        "trade_rl.workflows.universal_causal_alpha_v3_signal_forensics_v2_loader"
    )


def test_v2_sidecar_loader_binds_complete_canonical_scope(tmp_path: Path) -> None:
    built = _complete_sidecars(tmp_path)

    bound = _loader().load_causal_alpha_v3_signal_forensics_v2_sidecars(tmp_path)

    assert len(bound) == 16
    assert {item.metric.identity for item in bound} == set(
        built["diagnostic_paths"]
    )
    assert all(item.metric.digest == item.diagnostic.signal_metric_digest for item in bound)
    assert all(item.metric.forecast_digest == item.diagnostic.forecast_digest for item in bound)


def test_v2_sidecar_loader_rejects_existing_empty_diagnostic_root(
    tmp_path: Path,
) -> None:
    _build_run(tmp_path)
    (tmp_path / "signal" / "diagnostics").mkdir(parents=True)

    with pytest.raises(ValueError, match="diagnostic"):
        _loader().load_causal_alpha_v3_signal_forensics_v2_sidecars(tmp_path)


def test_v2_sidecar_loader_rejects_missing_member(tmp_path: Path) -> None:
    built = _complete_sidecars(tmp_path)
    next(iter(built["diagnostic_paths"].values())).unlink()

    with pytest.raises(ValueError, match="scope"):
        _loader().load_causal_alpha_v3_signal_forensics_v2_sidecars(tmp_path)


def test_v2_sidecar_loader_rejects_wrong_path(tmp_path: Path) -> None:
    built = _complete_sidecars(tmp_path)
    path = next(iter(built["diagnostic_paths"].values()))
    path.rename(path.with_name("99.json"))

    with pytest.raises(ValueError, match="path"):
        _loader().load_causal_alpha_v3_signal_forensics_v2_sidecars(tmp_path)


def test_v2_sidecar_loader_rejects_metric_identity_drift_with_valid_outer_digest(
    tmp_path: Path,
) -> None:
    built = _complete_sidecars(tmp_path)
    path = next(iter(built["diagnostic_paths"].values()))
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["signal_metric_digest"] = _digest("foreign-metric")
    body = {key: value for key, value in raw.items() if key != "artifact_digest"}
    _write_json(path, {**body, "artifact_digest": content_digest(body)})

    with pytest.raises(ValueError, match="metric"):
        _loader().load_causal_alpha_v3_signal_forensics_v2_sidecars(tmp_path)
