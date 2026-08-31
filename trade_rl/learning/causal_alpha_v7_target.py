"""Fixed forecast adapters for Causal Alpha V7 target candidates."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.learning.causal_alpha_v4 import (
    CAUSAL_ALPHA_V4_HORIZONS,
    CausalAlphaV4Forecast,
    build_causal_alpha_v4_forecast,
)
from trade_rl.learning.causal_alpha_v6 import (
    CausalAlphaV6Candidate,
    CausalAlphaV6TargetConfig,
)
from trade_rl.learning.causal_alpha_v6_target import causal_alpha_v6_target_path
from trade_rl.learning.causal_alpha_v7 import (
    CausalAlphaV7Candidate,
    CausalAlphaV7TargetPath,
)
from trade_rl.learning.causal_alpha_v7_calibration import CausalAlphaV7CalibrationFit


def _effective_forecast(
    source: CausalAlphaV4Forecast,
    *,
    expected_return_4h: np.ndarray,
    direction_score_4h: np.ndarray,
    transformation: str,
    return_model_digest: str,
    direction_model_digest: str,
) -> CausalAlphaV4Forecast:
    rows = len(source.decision_indices)
    expected = np.asarray(expected_return_4h, dtype=np.float64).reshape(-1)
    direction = np.asarray(direction_score_4h, dtype=np.float64).reshape(-1)
    if (
        expected.shape != (rows,)
        or direction.shape != (rows,)
        or not np.isfinite(expected).all()
        or not np.isfinite(direction).all()
    ):
        raise ValueError("V7 effective fast forecast is invalid")
    market = {
        horizon: source.market_predictions[horizon]
        for horizon in CAUSAL_ALPHA_V4_HORIZONS
    }
    residual = {
        horizon: source.residual_predictions[horizon]
        for horizon in CAUSAL_ALPHA_V4_HORIZONS
    }
    directions = {
        horizon: source.direction_scores[horizon]
        for horizon in CAUSAL_ALPHA_V4_HORIZONS
    }
    market["4h"] = np.zeros(rows, dtype=np.float64)
    residual["4h"] = expected
    directions["4h"] = direction
    market_digests = dict(source.market_model_digests)
    residual_digests = dict(source.residual_model_digests)
    direction_digests = dict(source.direction_model_digests)
    market_digests["4h"] = content_digest(
        {
            "source_forecast_digest": source.digest,
            "transformation": transformation,
            "component": "zero_market",
        }
    )
    residual_digests["4h"] = return_model_digest
    direction_digests["4h"] = direction_model_digest
    fit_digest = content_digest(
        {
            "source_fit_digest": source.fit_digest,
            "transformation": transformation,
            "return_model_digest": return_model_digest,
            "direction_model_digest": direction_model_digest,
        }
    )
    return build_causal_alpha_v4_forecast(
        symbol=source.symbol,
        decision_indices=source.decision_indices,
        beta=source.beta,
        beta_available=source.beta_available,
        market_predictions=market,
        residual_predictions=residual,
        direction_scores=directions,
        market_model_digests=market_digests,
        residual_model_digests=residual_digests,
        direction_model_digests=direction_digests,
        fit_digest=fit_digest,
    )


def causal_alpha_v7_target_paths(
    *,
    forecast: CausalAlphaV4Forecast,
    calibration_fit: CausalAlphaV7CalibrationFit,
    calibration_features: object,
    calibration_feature_available: object,
    uncertainty: Mapping[str, np.ndarray],
    one_way_cost_rates: object,
    liquidity_weight_caps: object,
    actionable_mask: object,
    config: CausalAlphaV6TargetConfig,
    initial_weight: float,
    risk_weight_caps: object | None = None,
) -> Mapping[CausalAlphaV7Candidate, CausalAlphaV7TargetPath]:
    """Compile all fixed V7 candidates with one shared V6 target compiler."""

    if not isinstance(forecast, CausalAlphaV4Forecast):
        raise TypeError("V7 target paths require a V4 forecast")
    if not isinstance(calibration_fit, CausalAlphaV7CalibrationFit):
        raise TypeError("V7 target paths require a calibration fit")
    calibrated_return, calibrated_direction = calibration_fit.predict(
        calibration_features,
        feature_available=calibration_feature_available,
    )
    raw_return = np.asarray(forecast.final_predictions["4h"], dtype=np.float64)
    raw_direction = np.asarray(forecast.direction_scores["4h"], dtype=np.float64)
    contrarian_return_digest = content_digest(
        {"source_forecast_digest": forecast.digest, "transformation": "negate_return"}
    )
    contrarian_direction_digest = content_digest(
        {
            "source_forecast_digest": forecast.digest,
            "transformation": "negate_direction",
        }
    )
    effective = {
        CausalAlphaV7Candidate.V6_CONTROL: forecast,
        CausalAlphaV7Candidate.SYMMETRIC_CONTRARIAN: _effective_forecast(
            forecast,
            expected_return_4h=-raw_return,
            direction_score_4h=-raw_direction,
            transformation="symmetric_contrarian",
            return_model_digest=contrarian_return_digest,
            direction_model_digest=contrarian_direction_digest,
        ),
        CausalAlphaV7Candidate.CAUSAL_CALIBRATED: _effective_forecast(
            forecast,
            expected_return_4h=calibrated_return,
            direction_score_4h=calibrated_direction,
            transformation="causal_calibrated",
            return_model_digest=calibration_fit.return_model.digest,
            direction_model_digest=calibration_fit.direction_model.digest,
        ),
    }
    result: dict[CausalAlphaV7Candidate, CausalAlphaV7TargetPath] = {}
    for candidate in CausalAlphaV7Candidate:
        target = causal_alpha_v6_target_path(
            effective[candidate],
            uncertainty=uncertainty,
            one_way_cost_rates=one_way_cost_rates,
            liquidity_weight_caps=liquidity_weight_caps,
            actionable_mask=actionable_mask,
            candidate=CausalAlphaV6Candidate.FAST_ONLY,
            config=config,
            initial_weight=initial_weight,
            risk_weight_caps=risk_weight_caps,
        )
        result[candidate] = CausalAlphaV7TargetPath(
            candidate=candidate,
            v6_target_path=target,
            source_forecast_digest=forecast.digest,
            calibration_fit_digest=calibration_fit.digest,
        )
    return MappingProxyType(result)


__all__ = ["causal_alpha_v7_target_paths"]
