from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np

from mars_lite.serving.contracts import InferenceState
from mars_lite.trading.baseline_residual import BaselineResidualComposer
from mars_lite.trading.pipeline import DecisionPipeline, MarketView, PortfolioState
from mars_lite.trading.residual_alpha import FrozenResidualAlpha
from mars_lite.trading.trend_family import TrendFamily, TrendFamilyConfig, TrendTargets


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def build_residual_serving_adapters(bundle, pipeline: DecisionPipeline):
    """Build stateless observation augmentation and context-aware decision functions."""

    run_config = dict(bundle.metadata.get("run_config") or {})
    trend_config = dict(bundle.metadata.get("trend_family") or {})
    trend_config.setdefault("base_timeframe", run_config.get("base_timeframe", "1h"))
    trend_family = TrendFamily(TrendFamilyConfig(**trend_config))
    composer = BaselineResidualComposer(**dict(bundle.metadata.get("composer") or {}))
    alpha = FrozenResidualAlpha.load(
        bundle.root / str(bundle.metadata["residual_alpha_file"])
    )
    expected_symbols = tuple(bundle.metadata["symbols"])
    expected_features = tuple(bundle.preprocessing["feature_names"])
    declared_alpha_enabled = bundle.metadata.get("residual_alpha_enabled")
    if not isinstance(declared_alpha_enabled, bool):
        raise ValueError("baseline residual bundle requires residual_alpha_enabled")
    if alpha.symbols != expected_symbols:
        raise ValueError("residual alpha symbol order does not match bundle")
    if alpha.feature_names != expected_features:
        raise ValueError("residual alpha feature order does not match bundle")
    if declared_alpha_enabled and not alpha.enabled:
        raise ValueError(
            "bundle enables residual alpha but the frozen artifact failed its gate"
        )
    if bundle.metadata.get("policy_mode") == "baseline_only" and declared_alpha_enabled:
        raise ValueError("baseline_only bundle must disable residual alpha")

    def augment_features(snapshot, latest: np.ndarray):
        timestamps = getattr(snapshot, "timestamps", None)
        if timestamps is None:
            raise ValueError("baseline residual serving requires snapshot timestamps")
        timestamp_array = np.asarray(timestamps).astype("datetime64[ns]")
        if timestamp_array.ndim != 1 or len(timestamp_array) != len(
            snapshot.close_history
        ):
            raise ValueError("snapshot timestamps must match close history")
        trend_fs = SimpleNamespace(
            n_bars=len(timestamp_array),
            n_symbols=len(expected_symbols),
            timestamps=timestamp_array,
            close=np.asarray(snapshot.close_history, dtype=np.float64),
        )
        trends = trend_family.targets(trend_fs, trend_fs.n_bars - 1)
        alpha_fs = SimpleNamespace(
            n_bars=1,
            n_symbols=len(expected_symbols),
            symbols=list(expected_symbols),
            feature_names=list(expected_features),
            features=np.asarray(latest, dtype=np.float64).reshape(
                1, len(expected_symbols), len(expected_features)
            ),
        )
        alpha_weights = (
            alpha.predict_at(alpha_fs, 0)
            if declared_alpha_enabled
            else np.zeros(len(expected_symbols), dtype=np.float64)
        )
        augmented = np.concatenate(
            [
                np.asarray(latest, dtype=np.float64),
                trends.fast.reshape(-1, 1),
                trends.base.reshape(-1, 1),
                trends.slow.reshape(-1, 1),
                alpha_weights.reshape(-1, 1),
            ],
            axis=1,
        )
        return augmented, {"trends": trends, "alpha": alpha_weights}

    def decide_with_context(
        raw_action: np.ndarray,
        state: InferenceState,
        recent_returns: np.ndarray | None,
        htf_trend: np.ndarray | None,
        context: Mapping[str, Any],
    ):
        trends = context.get("trends")
        alpha_weights = context.get("alpha")
        if not isinstance(trends, TrendTargets):
            raise ValueError("residual decision context is missing trend targets")
        alpha_array = np.asarray(alpha_weights, dtype=np.float64).reshape(-1)
        composition = composer.compose(
            raw_action,
            trends,
            alpha_array,
            alpha_enabled=declared_alpha_enabled,
        )
        current = state.weights_array(expected_symbols)
        target, info = pipeline.process_proposal(
            composition.proposal,
            PortfolioState(
                weights=current,
                portfolio_value=state.portfolio_value,
                peak_value=state.peak_value,
                disagreement=state.disagreement,
            ),
            MarketView(recent_returns=recent_returns, htf_trend=htf_trend),
        )
        decision_info = {
            "action_schema": "baseline_residual_v1",
            "raw_action": np.asarray(raw_action, dtype=np.float64).tolist(),
            "trend_mix": composition.trend_mix,
            "alpha_enabled": declared_alpha_enabled,
            "alpha_budget": composition.alpha_budget,
            "trend_weights": composition.trend_weights.tolist(),
            "alpha_weights": alpha_array.tolist(),
            "composed_weights": composition.proposal.tolist(),
            "post_process": {
                "processed_gross": info.processed_gross,
                "desired_turnover": info.desired_turnover,
                "executed_turnover": info.executed_turnover,
                "extra": _jsonable(info.extra),
            },
        }
        return target, decision_info

    return augment_features, decide_with_context
