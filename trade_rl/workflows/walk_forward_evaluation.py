"""Range-scoped causal environment construction and policy evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.artifacts.signals import load_signal_artifact
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.evidence import ExecutionDiagnostics
from trade_rl.evaluation.series import ReturnKind, ReturnSeries
from trade_rl.evaluation.walk_forward.folds import IndexRange
from trade_rl.integrations.signal_artifacts import (
    LoadedAlphaArtifact,
    LoadedFactorArtifact,
    load_alpha_artifact,
    load_factor_artifact,
)
from trade_rl.risk.portfolio import PortfolioRiskModel
from trade_rl.risk.pretrade import PreTradeRisk
from trade_rl.rl.environment import ResidualMarketEnv
from trade_rl.rl.episode import minimum_reward_start_index
from trade_rl.rl.normalization import ObservationNormalizer
from trade_rl.rl.sequence_observations import (
    SequenceObservationBuilder,
    SequenceWindowSpec,
)
from trade_rl.strategies.trend import TrendStrategy
from trade_rl.workflows.training_run import TrainingRunConfig


class _NamedSignalRun(Protocol):
    @property
    def run(self) -> TrainingRunConfig: ...


class SignalIdentityConfig(Protocol):
    @property
    def candidates(self) -> Sequence[_NamedSignalRun]: ...

    @property
    def signal_digest(self) -> str: ...


def resolve_signal_digest(config: SignalIdentityConfig, *, dataset_id: str) -> str:
    """Bind walk-forward signal identity to validated artifact content."""

    identities: set[str] = set()
    for candidate in config.candidates:
        run = candidate.run
        alpha_digest: str | None = None
        factor_digest: str | None = None
        if run.alpha_artifact is not None:
            manifest, _ = load_signal_artifact(
                run.alpha_artifact, expected_kind="alpha"
            )
            if manifest.dataset_id != dataset_id:
                raise ValueError("alpha signal artifact dataset identity mismatch")
            alpha_digest = manifest.artifact_digest
        if run.factor_artifact is not None:
            manifest, _ = load_signal_artifact(
                run.factor_artifact, expected_kind="factor"
            )
            if manifest.dataset_id != dataset_id:
                raise ValueError("factor signal artifact dataset identity mismatch")
            factor_digest = manifest.artifact_digest
        identities.add(
            content_digest(
                {
                    "alpha_artifact_digest": alpha_digest,
                    "factor_artifact_digest": factor_digest,
                    "schema_version": "causal_signal_identity_v1",
                    "trend": run.trend,
                }
            )
        )
    if len(identities) != 1:
        raise ValueError("walk-forward candidates must share signal artifacts")
    resolved = next(iter(identities))
    trend_only = content_digest(
        {
            "schema_version": "trend_baseline_signal_v1",
            "trend": config.candidates[0].run.trend.__dict__
            if hasattr(config.candidates[0].run.trend, "__dict__")
            else config.candidates[0].run.trend,
        }
    )
    if config.signal_digest not in {resolved, trend_only}:
        raise ValueError("configured signal_digest does not match signal artifacts")
    return resolved


def factor_names(run: TrainingRunConfig) -> tuple[str, ...]:
    return tuple(f"factor_{index}" for index in range(run.action.n_factors))


def load_signal_providers(
    dataset: MarketDataset,
    run: TrainingRunConfig,
    *,
    evaluation_start: int,
) -> tuple[LoadedAlphaArtifact | None, LoadedFactorArtifact | None]:
    alpha = (
        None
        if run.alpha_artifact is None
        else load_alpha_artifact(
            run.alpha_artifact,
            dataset_id=dataset.dataset_id,
            evaluation_start=evaluation_start,
            expected_symbols=dataset.symbols,
        )
    )
    factor = (
        None
        if run.factor_artifact is None
        else load_factor_artifact(
            run.factor_artifact,
            dataset_id=dataset.dataset_id,
            evaluation_start=evaluation_start,
            expected_names=factor_names(run),
            expected_symbols=dataset.n_symbols,
        )
    )
    return alpha, factor


def bind_signal_providers_to_view(
    source: MarketDataset,
    view: MarketDataset,
    run: TrainingRunConfig,
    *,
    start: int,
    stop: int,
    evaluation_start: int,
) -> tuple[LoadedAlphaArtifact | None, LoadedFactorArtifact | None]:
    alpha, factor = load_signal_providers(
        source, run, evaluation_start=evaluation_start
    )
    return (
        None
        if alpha is None
        else alpha.for_view(start=start, stop=stop, dataset_id=view.dataset_id),
        None
        if factor is None
        else factor.for_view(start=start, stop=stop, dataset_id=view.dataset_id),
    )


def minimum_environment_start(
    dataset: MarketDataset,
    run: TrainingRunConfig,
    *,
    alpha_provider: LoadedAlphaArtifact | None = None,
    factor_provider: LoadedFactorArtifact | None = None,
) -> int:
    """Resolve the first index valid for signals and complete reward pre-roll."""

    minimum = TrendStrategy(run.trend).minimum_history_for(dataset)
    for provider in (alpha_provider, factor_provider):
        if provider is not None:
            minimum = max(minimum, provider.minimum_index)
    if run.reward.baseline_underperformance_weight > 0.0:
        minimum = minimum_reward_start_index(
            dataset,
            signal_minimum=minimum,
            window_hours=run.reward.baseline_window_hours,
        )
    if run.environment.structured_sequence_observation:
        sequence_builder = SequenceObservationBuilder(
            windows=tuple(
                SequenceWindowSpec(timeframe, length)
                for timeframe, length in run.environment.resolved_sequence_windows
            )
        )
        minimum = max(minimum, sequence_builder.minimum_index(dataset))
    return minimum


def build_market_environment(
    dataset: MarketDataset,
    run: TrainingRunConfig,
    *,
    normalizer: ObservationNormalizer | None,
    episode_bars: int,
    liquidate_on_end: bool,
    alpha_provider: LoadedAlphaArtifact | None = None,
    factor_provider: LoadedFactorArtifact | None = None,
) -> ResidualMarketEnv:
    environment_config = replace(
        run.environment,
        episode_bars=episode_bars,
        episode_hour_choices=(),
        initial_state_modes=("cash",),
        liquidate_on_end=liquidate_on_end,
        require_full_reward_preroll=True,
    )
    return ResidualMarketEnv(
        dataset,
        trend_strategy=TrendStrategy(run.trend),
        alpha_provider=alpha_provider,
        alpha_enabled=run.action.alpha_enabled,
        alpha_artifact_digest=(
            None if alpha_provider is None else alpha_provider.artifact_digest
        ),
        alpha_contract=run.alpha_contract,
        factor_basis_provider=factor_provider,
        factor_artifact_digest=(
            None if factor_provider is None else factor_provider.artifact_digest
        ),
        factor_count=run.action.n_factors,
        action_spec=run.action,
        pre_trade_risk=PreTradeRisk(run.risk),
        portfolio_risk=PortfolioRiskModel(run.portfolio_risk),
        normalizer=normalizer,
        config=environment_config,
    )


@dataclass(frozen=True, slots=True)
class RangeEvaluation:
    returns: ReturnSeries
    diagnostics: ExecutionDiagnostics


def evaluate_range_evidence(
    *,
    dataset: MarketDataset,
    evaluation_range: IndexRange,
    run: TrainingRunConfig,
    normalizer: ObservationNormalizer | None,
    model: Any | None,
    baseline: bool,
) -> RangeEvaluation:
    """Evaluate one range and retain execution and economic evidence."""

    start_index = evaluation_range.start - 1
    minimum = TrendStrategy(run.trend).minimum_history_for(dataset)
    if start_index < minimum:
        raise ValueError("evaluation range lacks causal trend history")
    alpha_provider, factor_provider = load_signal_providers(
        dataset, run, evaluation_start=evaluation_range.start
    )
    env = build_market_environment(
        dataset,
        run,
        normalizer=normalizer,
        episode_bars=evaluation_range.size,
        liquidate_on_end=True,
        alpha_provider=alpha_provider,
        factor_provider=factor_provider,
    )
    try:
        observation, _ = env.reset(
            seed=0,
            options={
                "episode_bars": evaluation_range.size,
                "initial_state_mode": "cash",
                "start_idx": start_index,
            },
        )
        terminated = False
        truncated = False
        while not terminated and not truncated:
            if baseline:
                action = np.zeros(run.action.size, dtype=np.float32)
            else:
                if model is None:
                    raise RuntimeError("residual evaluation requires a loaded model")
                raw_action, _ = model.predict(observation, deterministic=True)
                action = np.asarray(raw_action, dtype=np.float32).reshape(-1)
            observation, _, terminated, truncated, _ = env.step(action)
        book = env.shadow if baseline else env.hybrid
        values = tuple(float(value) for value in book.returns_history)
        termination_reasons = (
            ()
            if book.termination_reason is None
            else (
                str(getattr(book.termination_reason, "value", book.termination_reason)),
            )
        )
        diagnostics = ExecutionDiagnostics(
            turnover_total=book.turnover_total,
            total_cost=book.total_cost,
            funding_pnl=book.funding_pnl,
            borrow_cost=book.borrow_cost,
            n_trades=book.n_trades,
            rebalance_events=book.rebalance_events,
            termination_reasons=termination_reasons,
        )
    finally:
        env.close()
    if len(values) != evaluation_range.size:
        raise ValueError(
            "range-restricted environment produced an unexpected return length"
        )
    return RangeEvaluation(
        returns=ReturnSeries(
            values=values,
            kind=ReturnKind.BASE_BAR,
            periods_per_year=dataset.periods_per_year,
        ),
        diagnostics=diagnostics,
    )


def evaluate_range(
    *,
    dataset: MarketDataset,
    evaluation_range: IndexRange,
    run: TrainingRunConfig,
    normalizer: ObservationNormalizer | None,
    model: Any | None,
    baseline: bool,
) -> ReturnSeries:
    """Compatibility wrapper returning only the evaluated return series."""

    return evaluate_range_evidence(
        dataset=dataset,
        evaluation_range=evaluation_range,
        run=run,
        normalizer=normalizer,
        model=model,
        baseline=baseline,
    ).returns


__all__ = [
    "bind_signal_providers_to_view",
    "build_market_environment",
    "evaluate_range",
    "evaluate_range_evidence",
    "RangeEvaluation",
    "factor_names",
    "load_signal_providers",
    "minimum_environment_start",
    "resolve_signal_digest",
]
