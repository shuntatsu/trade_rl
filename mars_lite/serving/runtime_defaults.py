"""Default adapters from the generic serving runtime to the trading stack."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping, Sequence

import numpy as np

from mars_lite.serving.bundle import ServingBundle
from mars_lite.serving.contracts import InferenceState
from mars_lite.serving.runtime import PolicyLike, RuntimeComponents


def _load_policy(bundle: ServingBundle) -> PolicyLike:
    """Load exactly the policy kind declared by the validated bundle metadata."""
    model_kind = bundle.metadata.get("model_kind")
    if model_kind == "single":
        from stable_baselines3 import PPO

        return PPO.load(str(bundle.model_path), device="cpu")
    if model_kind == "ensemble":
        from mars_lite.learning.policy_ensemble import SeedEnsemble

        return SeedEnsemble.load(bundle.model_path, device="cpu")
    raise ValueError(f"unsupported serving model_kind: {model_kind!r}")


def default_component_factory(bundle: ServingBundle) -> RuntimeComponents:
    from mars_lite.trading.guardrails import (
        GuardrailConfig,
        GuardrailState,
        apply_guardrails,
        evaluate_guardrails,
    )
    from mars_lite.trading.pipeline import DecisionPipeline, MarketView, PortfolioState
    from mars_lite.trading.post_processor import (
        PortfolioPostProcessor,
        PostProcessConfig,
        make_default_processor,
    )
    from mars_lite.trading.pre_trade_risk import (
        PendingOrder,
        PreTradeRejection,
        PreTradeRiskConfig,
        PreTradeRiskVerifier,
    )

    model = _load_policy(bundle)
    post_config = bundle.metadata.get("post_processor")
    if post_config:
        post_processor = PortfolioPostProcessor(PostProcessConfig(**post_config))
    else:
        post_processor = make_default_processor()
    run_config = dict(bundle.metadata.get("run_config") or {})
    pipeline = DecisionPipeline(
        post_processor=post_processor,
        min_trade_delta=float(run_config.get("min_trade_delta", 0.04)),
        htf_threshold=float(run_config.get("htf_threshold", 0.3)),
        htf_neutral_scale=float(run_config.get("htf_neutral_scale", 0.5)),
    )
    guard_config = GuardrailConfig(**dict(bundle.risk.get("guardrails") or {}))
    pretrade_data = dict(bundle.risk.get("pre_trade") or {})
    if pretrade_data.get("forbidden_symbols") is not None:
        pretrade_data["forbidden_symbols"] = set(pretrade_data["forbidden_symbols"])
    verifier = PreTradeRiskVerifier(PreTradeRiskConfig(**pretrade_data))

    def decide(
        raw_action: np.ndarray,
        state: InferenceState,
        recent_returns: np.ndarray | None,
        htf_trend: np.ndarray | None,
    ) -> tuple[np.ndarray, Mapping[str, Any]]:
        current = state.weights_array(tuple(bundle.metadata["symbols"]))
        projected = pipeline.project(raw_action)
        target, info = pipeline.target_weights(
            projected,
            PortfolioState(
                weights=current,
                portfolio_value=state.portfolio_value,
                peak_value=state.peak_value,
                disagreement=state.disagreement,
            ),
            MarketView(recent_returns=recent_returns, htf_trend=htf_trend),
        )
        return target, asdict(info)

    def guard(
        target: np.ndarray,
        current: np.ndarray,
        state: InferenceState,
        data_age_hours: float,
        features: np.ndarray,
    ) -> tuple[np.ndarray, Mapping[str, Any]]:
        turnover = float(np.abs(target - current).sum())
        result = evaluate_guardrails(
            weights=target,
            portfolio_value=state.portfolio_value,
            turnover=turnover,
            data_age_hours=data_age_hours,
            features=features.reshape(-1),
            state=GuardrailState(
                day_start_value=state.day_start_value,
                peak_value=state.peak_value,
                consecutive_losses=state.consecutive_losses,
                turnover_mean=state.turnover_mean,
                turnover_std=state.turnover_std,
            ),
            config=guard_config,
        )
        return apply_guardrails(target, result), result.to_dict()

    def risk(
        target: np.ndarray, state: InferenceState, symbols: Sequence[str]
    ) -> Mapping[str, Any]:
        pending = tuple(
            PendingOrder(
                symbol=order.symbol,
                side=order.side,
                notional=order.notional,
                reduce_only=order.reduce_only,
            )
            for order in state.pending_orders
        )
        try:
            verifier.validate(
                target,
                state.portfolio_value,
                symbols=symbols,
                current_weights=state.weights_array(symbols),
                open_orders=pending,
            )
            return {"approved": True}
        except PreTradeRejection as exc:
            return {
                "approved": False,
                "reason": exc.reason,
                "details": exc.details,
            }

    return RuntimeComponents(
        model=model,
        decide=decide,
        apply_guardrails=guard,
        evaluate_risk=risk,
        include_observation_risk_state=bool(run_config.get("obs_risk_state", False)),
        serving_progress=float(bundle.metadata.get("serving_progress", 0.0)),
        vol_lookback=int(post_processor.cfg.vol_lookback),
        htf_feature_name=("4h_ret_z20" if run_config.get("htf_gate") else None),
    )
