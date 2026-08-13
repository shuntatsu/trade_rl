from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.learning.causal_alpha_diagnostics import (
    evaluate_causal_alpha_signal_diagnostics,
)
from trade_rl.learning.causal_alpha_teacher import (
    CausalAlphaControllerConfig,
    CausalAlphaCostAwareConfig,
    CausalAlphaHorizonMix,
    CausalAlphaRidgeConfig,
)
from trade_rl.learning.episode_oracle_teacher import OracleEpisodeContract
from trade_rl.risk.pretrade import PreTradeRiskConfig
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.workflows.universal_causal_alpha_selection import (
    CausalAlphaSelectionRejected,
    CausalAlphaSelectionRejectedV2,
    CausalAlphaSelectionThresholds,
    default_cost_aware_causal_alpha_candidate_grid,
    rank_cost_aware_causal_alpha_candidates,
)
from trade_rl.workflows.universal_causal_alpha_teacher import (
    CausalAlphaCandidateConfig,
    CausalAlphaCandidateEpisodeMetrics,
    CausalAlphaCandidateEpisodeMetricsV2,
    CausalAlphaEpisodePartition,
    CausalAlphaExpandingFitCache,
    CausalAlphaSymbolSamples,
    evaluate_causal_alpha_selection,
    evaluate_cost_aware_causal_alpha_selection,
    load_causal_alpha_selection_checkpoint,
    load_causal_alpha_selection_checkpoint_v2,
    persist_causal_alpha_selection_rejection,
    rank_causal_alpha_candidates,
    write_causal_alpha_selection_checkpoint_metric,
    write_causal_alpha_selection_checkpoint_metric_v2,
)


def _candidate(
    label: str,
    *,
    ridge: float = 0.1,
    scale: float = 1.0,
) -> CausalAlphaCandidateConfig:
    return CausalAlphaCandidateConfig(
        name=label,
        ridge=CausalAlphaRidgeConfig(ridge_strength=ridge),
        controller=CausalAlphaControllerConfig(
            horizon_mix=CausalAlphaHorizonMix.EQUAL,
            score_scale=scale,
            entry_threshold=0.05,
            exit_threshold=0.02,
            no_trade_band=0.01,
            max_target_delta=0.25,
        ),
    )


def _metric(
    candidate: CausalAlphaCandidateConfig,
    episode_index: int,
    *,
    gross: float,
    net: float,
    turnover: float,
    cost: float,
    trades: int = 1,
    risk_violation: bool = False,
) -> CausalAlphaCandidateEpisodeMetrics:
    return CausalAlphaCandidateEpisodeMetrics(
        candidate_digest=candidate.digest,
        symbol=f"S{episode_index}",
        episode_index=episode_index,
        gross_return=gross,
        net_return=net,
        turnover_per_day=turnover,
        total_execution_cost=cost,
        trade_count=trades,
        risk_violation=risk_violation,
    )


def test_candidate_ranking_is_lexicographic_and_worst_case_first() -> None:
    a = _candidate("a")
    b = _candidate("b", ridge=0.2)
    c = _candidate("c", ridge=0.3)
    d = _candidate("d", ridge=0.4)
    evidence = rank_causal_alpha_candidates(
        candidates=(a, b, c, d),
        metrics={
            a.digest: (
                _metric(a, 0, gross=0.1, net=0.01, turnover=3.0, cost=4.0),
                _metric(a, 1, gross=0.1, net=0.03, turnover=3.0, cost=4.0),
            ),
            b.digest: (
                _metric(b, 0, gross=0.1, net=0.00, turnover=0.1, cost=0.1),
                _metric(b, 1, gross=0.1, net=0.20, turnover=0.1, cost=0.1),
            ),
            c.digest: (
                _metric(c, 0, gross=0.1, net=0.01, turnover=2.0, cost=4.0),
                _metric(c, 1, gross=0.1, net=0.05, turnover=2.0, cost=4.0),
            ),
            d.digest: (
                _metric(d, 0, gross=0.1, net=0.01, turnover=1.0, cost=1.0),
                _metric(d, 1, gross=0.1, net=0.05, turnover=1.0, cost=1.0),
            ),
        },
    )

    assert evidence.lower_tail_definition == "minimum_symbol_episode_net_return"
    assert evidence.selected_candidate_digest == d.digest
    selected = next(
        item for item in evidence.candidates if item.candidate.digest == d.digest
    )
    assert selected.lower_tail_net_return == pytest.approx(0.01)
    assert selected.mean_net_return == pytest.approx(0.03)
    assert selected.turnover_per_day == pytest.approx(1.0)
    assert selected.total_execution_cost == pytest.approx(2.0)


def test_candidate_inadmissibility_is_fail_closed() -> None:
    negative = _candidate("negative")
    no_trade = _candidate("no-trade", ridge=0.2)
    risk = _candidate("risk", ridge=0.3)
    metrics = {
        negative.digest: tuple(
            _metric(
                negative,
                index,
                gross=(-0.1 if index < 2 else 0.1),
                net=0.01,
                turnover=1.0,
                cost=1.0,
            )
            for index in range(3)
        ),
        no_trade.digest: (
            _metric(
                no_trade,
                0,
                gross=0.1,
                net=0.1,
                turnover=0.0,
                cost=0.0,
                trades=0,
            ),
        ),
        risk.digest: (
            _metric(
                risk,
                0,
                gross=0.1,
                net=0.1,
                turnover=1.0,
                cost=1.0,
                risk_violation=True,
            ),
        ),
    }
    with pytest.raises(CausalAlphaSelectionRejected) as caught:
        rank_causal_alpha_candidates(
            candidates=(negative, no_trade, risk),
            metrics=metrics,
        )
    payload = caught.value.to_payload()
    assert payload["schema_version"] == "causal_alpha_selection_rejection_v1"
    assert payload["artifact_digest"] == caught.value.digest
    assert [item["rejection_reasons"] for item in payload["candidates"]] == [
        ["majority_negative_gross_return"],
        ["no_meaningful_trades"],
        ["risk_contract_violation"],
    ]


def test_selection_rejection_is_persisted_before_reraising(tmp_path: Path) -> None:
    rejected = _candidate("rejected")
    with pytest.raises(CausalAlphaSelectionRejected) as caught:
        rank_causal_alpha_candidates(
            candidates=(rejected,),
            metrics={
                rejected.digest: (
                    _metric(
                        rejected,
                        0,
                        gross=-0.1,
                        net=-0.2,
                        turnover=2.0,
                        cost=3.0,
                        trades=1,
                    ),
                )
            },
        )
    path = tmp_path / "causal-teacher-selection-rejected.json"

    persist_causal_alpha_selection_rejection(path, caught.value)

    assert json.loads(path.read_text(encoding="utf-8")) == caught.value.to_payload()


def test_selection_checkpoint_round_trips_episode_metrics(tmp_path: Path) -> None:
    candidate = _candidate("checkpoint")
    metric = _metric(
        candidate,
        3,
        gross=0.02,
        net=0.01,
        turnover=0.3,
        cost=4.0,
    )
    path = tmp_path / "causal-teacher-selection-checkpoint.jsonl"

    write_causal_alpha_selection_checkpoint_metric(path, metric)
    restored = load_causal_alpha_selection_checkpoint(path)

    assert restored == {candidate.digest: (metric,)}


def _metric_v2(*, realized_scale: float = 1.0) -> CausalAlphaCandidateEpisodeMetricsV2:
    predicted = np.asarray([-0.02, -0.01, 0.01, 0.02])
    signal = evaluate_causal_alpha_signal_diagnostics(
        predicted, realized_scale * predicted
    )
    return CausalAlphaCandidateEpisodeMetricsV2(
        candidate_digest=content_digest("candidate-v2"),
        symbol="BTCUSDT",
        episode_index=3,
        gross_return=0.02,
        net_return=0.01,
        turnover_per_day=0.3,
        total_execution_cost=4.0,
        trade_count=2,
        signal_24h=signal,
        signal_72h=signal,
        cost_suppressed_change_count=3,
        submitted_change_count=2,
        strong_reversal_count=1,
        command_sign_flip_count=1,
        execution_rejection_count=1,
        execution_rejection_reason_counts=(("minimum_notional", 1),),
        risk_projection_reason_counts=(("no_trade_band", 2),),
        hard_risk_violation=False,
    )


def test_v2_checkpoint_round_trips_and_binds_grid_identity(tmp_path: Path) -> None:
    metric = _metric_v2()
    grid_digest = content_digest("corrected-grid")
    path = tmp_path / "causal-teacher-selection-checkpoint-v2.jsonl"

    write_causal_alpha_selection_checkpoint_metric_v2(
        path, metric, grid_digest=grid_digest
    )

    assert load_causal_alpha_selection_checkpoint_v2(
        path, expected_grid_digest=grid_digest
    ) == {metric.candidate_digest: (metric,)}
    with pytest.raises(ValueError, match="grid digest"):
        load_causal_alpha_selection_checkpoint_v2(
            path, expected_grid_digest=content_digest("different-grid")
        )


def test_v2_checkpoint_rejects_historical_v1_rows(tmp_path: Path) -> None:
    candidate = _candidate("historical")
    path = tmp_path / "causal-teacher-selection-checkpoint.jsonl"
    write_causal_alpha_selection_checkpoint_metric(
        path,
        _metric(
            candidate,
            0,
            gross=0.01,
            net=0.0,
            turnover=0.1,
            cost=1.0,
        ),
    )

    with pytest.raises(ValueError, match="schema"):
        load_causal_alpha_selection_checkpoint_v2(
            path, expected_grid_digest=content_digest("corrected-grid")
        )


def test_v2_metric_digest_changes_with_signal_diagnostics() -> None:
    assert _metric_v2(realized_scale=1.0).digest != _metric_v2(
        realized_scale=-1.0
    ).digest


def test_cost_aware_candidate_grid_has_exact_one_factor_variants() -> None:
    grid = default_cost_aware_causal_alpha_candidate_grid(
        risk_config=PreTradeRiskConfig(max_abs_weight=1.0, no_trade_band=0.05),
        max_position_to_market_notional=0.03,
    )

    assert tuple(item.name for item in grid) == (
        "cost-aware-baseline",
        "horizon-24h",
        "horizon-72h",
        "cost-multiplier-high",
        "edge-margin-high",
        "confirmation-one",
        "confirmation-three",
        "strong-reversal-low",
        "scale-low",
        "exposure-low",
        "no-trade-high",
        "delta-low",
    )
    assert all(isinstance(item.economic_controller, CausalAlphaCostAwareConfig) for item in grid)
    assert all(
        item.economic_controller is not None
        and item.economic_controller.max_position_to_market_notional == pytest.approx(0.03)
        for item in grid
    )

    def identity(candidate: CausalAlphaCandidateConfig) -> dict[str, object]:
        economic = candidate.economic_controller
        assert economic is not None
        return {
            "horizon_mix": candidate.controller.horizon_mix,
            "score_scale": candidate.controller.score_scale,
            "no_trade_band": candidate.controller.no_trade_band,
            "max_target_delta": candidate.controller.max_target_delta,
            "execution_cost_multiplier": economic.execution_cost_multiplier,
            "edge_margin": economic.edge_margin,
            "confirmation_count": economic.confirmation_count,
            "strong_reversal_threshold": economic.strong_reversal_threshold,
            "max_abs_target": economic.max_abs_target,
        }

    baseline = identity(grid[0])
    assert baseline == {
        "horizon_mix": CausalAlphaHorizonMix.EQUAL,
        "score_scale": 25.0,
        "no_trade_band": 0.05,
        "max_target_delta": 0.125,
        "execution_cost_multiplier": 1.5,
        "edge_margin": 0.001,
        "confirmation_count": 2,
        "strong_reversal_threshold": 0.02,
        "max_abs_target": 0.5,
    }
    for candidate in grid[1:]:
        changed = {
            key for key, value in identity(candidate).items() if value != baseline[key]
        }
        assert len(changed) == 1, (candidate.name, changed)


def _cost_metric(
    candidate: CausalAlphaCandidateConfig,
    episode_index: int = 0,
    *,
    gross: float = 0.02,
    net: float = 0.01,
    turnover: float = 0.5,
    trades: int = 2,
    rejection_reason: str | None = None,
    hard_risk: bool = False,
) -> CausalAlphaCandidateEpisodeMetricsV2:
    signal = evaluate_causal_alpha_signal_diagnostics(
        np.asarray([-0.02, -0.01, 0.01, 0.02]),
        np.asarray([-0.01, -0.02, 0.02, 0.01]),
    )
    return CausalAlphaCandidateEpisodeMetricsV2(
        candidate_digest=candidate.digest,
        symbol=f"S{episode_index}",
        episode_index=episode_index,
        gross_return=gross,
        net_return=net,
        turnover_per_day=turnover,
        total_execution_cost=1.0,
        trade_count=trades,
        signal_24h=signal,
        signal_72h=signal,
        cost_suppressed_change_count=1,
        submitted_change_count=2,
        strong_reversal_count=0,
        command_sign_flip_count=0,
        execution_rejection_count=int(rejection_reason is not None),
        execution_rejection_reason_counts=(
            () if rejection_reason is None else ((rejection_reason, 1),)
        ),
        risk_projection_reason_counts=(),
        hard_risk_violation=hard_risk,
    )


def test_cost_aware_ranking_rejects_each_economic_failure_reason() -> None:
    grid = default_cost_aware_causal_alpha_candidate_grid(
        risk_config=PreTradeRiskConfig(max_abs_weight=1.0, no_trade_band=0.05)
    )[:7]
    metrics = {
        grid[0].digest: (_cost_metric(grid[0], hard_risk=True),),
        grid[1].digest: (
            _cost_metric(grid[1], rejection_reason="minimum_notional"),
        ),
        grid[2].digest: (_cost_metric(grid[2], trades=0),),
        grid[3].digest: (_cost_metric(grid[3], net=-0.01),),
        grid[4].digest: (
            _cost_metric(grid[4], 0, net=-0.06),
            _cost_metric(grid[4], 1, net=0.08),
        ),
        grid[5].digest: (_cost_metric(grid[5], turnover=1.01),),
        grid[6].digest: (_cost_metric(grid[6], gross=-0.01),),
    }

    with pytest.raises(CausalAlphaSelectionRejectedV2) as caught:
        rank_cost_aware_causal_alpha_candidates(
            candidates=grid,
            metrics=metrics,
            thresholds=CausalAlphaSelectionThresholds(),
        )

    assert {
        item.rejection_reasons[0] for item in caught.value.candidates
    } == {
        "hard_risk_violation",
        "unexplained_execution_rejection",
        "no_meaningful_trades",
        "negative_mean_net_return",
        "lower_tail_net_return_below_floor",
        "turnover_per_day_above_maximum",
        "majority_negative_gross_return",
    }


def test_cost_aware_ranking_prefers_lower_tail_then_mean_then_turnover() -> None:
    grid = default_cost_aware_causal_alpha_candidate_grid(
        risk_config=PreTradeRiskConfig(max_abs_weight=1.0, no_trade_band=0.05)
    )[:2]
    selected = rank_cost_aware_causal_alpha_candidates(
        candidates=grid,
        metrics={
            grid[0].digest: (
                _cost_metric(grid[0], 0, net=0.01, turnover=0.2),
                _cost_metric(grid[0], 1, net=0.03, turnover=0.2),
            ),
            grid[1].digest: (
                _cost_metric(grid[1], 0, net=0.01, turnover=0.5),
                _cost_metric(grid[1], 1, net=0.05, turnover=0.5),
            ),
        },
        thresholds=CausalAlphaSelectionThresholds(),
    )

    assert selected.selected_candidate_digest == grid[1].digest


def test_cost_aware_ranking_treats_rounded_zero_as_explained_no_fill() -> None:
    candidate = default_cost_aware_causal_alpha_candidate_grid(
        risk_config=PreTradeRiskConfig(max_abs_weight=1.0, no_trade_band=0.05)
    )[0]

    selected = rank_cost_aware_causal_alpha_candidates(
        candidates=(candidate,),
        metrics={
            candidate.digest: (
                _cost_metric(
                    candidate,
                    rejection_reason="zero_quantity_after_rounding",
                ),
            )
        },
        thresholds=CausalAlphaSelectionThresholds(),
    )

    evidence = selected.candidates[0]
    assert evidence.admissible is True
    assert evidence.unexplained_execution_rejection_count == 0
    assert evidence.episode_metrics[0].execution_rejection_count == 1


def test_cost_aware_ranking_treats_below_minimum_notional_as_explained_no_fill() -> None:
    candidate = default_cost_aware_causal_alpha_candidate_grid(
        risk_config=PreTradeRiskConfig(max_abs_weight=1.0, no_trade_band=0.05)
    )[0]

    selected = rank_cost_aware_causal_alpha_candidates(
        candidates=(candidate,),
        metrics={
            candidate.digest: (
                _cost_metric(
                    candidate,
                    rejection_reason="below_minimum_notional",
                ),
            )
        },
        thresholds=CausalAlphaSelectionThresholds(),
    )

    evidence = selected.candidates[0]
    assert evidence.admissible is True
    assert evidence.unexplained_execution_rejection_count == 0
    assert evidence.episode_metrics[0].execution_rejection_count == 1


def _samples(symbol: str) -> CausalAlphaSymbolSamples:
    decisions = np.arange(2, 26, dtype=np.int64)
    features = np.column_stack(
        (decisions.astype(np.float64), np.ones(decisions.size, dtype=np.float64))
    )
    return CausalAlphaSymbolSamples(
        symbol=symbol,
        dataset_id=content_digest(f"dataset:{symbol}"),
        feature_names=("signal", "descriptor"),
        feature_schema_digest=content_digest("feature-schema"),
        context_digest=content_digest(f"context:{symbol}"),
        reference_equity_mode="initial_capital",
        reference_equity=1_000.0,
        decision_indices=decisions,
        features=features,
        feature_available=np.ones_like(features, dtype=np.bool_),
        labels_24h=0.01 * decisions,
        label_end_indices_24h=decisions + 1,
        labels_72h=0.02 * decisions,
        label_end_indices_72h=decisions + 2,
    )


def _partition(symbol: str) -> CausalAlphaEpisodePartition:
    dataset_id = content_digest(f"dataset:{symbol}")
    contracts = tuple(
        OracleEpisodeContract(
            dataset_id=dataset_id,
            episode_index=index,
            start=start,
            stop=start + 5,
            initial_state_mode="cash",
            initial_weights=np.zeros(1, dtype=np.float64),
        )
        for index, start in enumerate((8, 14, 20))
    )
    return CausalAlphaEpisodePartition(
        contracts=contracts,
        selection_contracts=contracts[:2],
        holdout_contract=contracts[2],
        train_start=2,
        train_stop=25,
    )


def test_production_selection_replays_only_selection_contracts(monkeypatch) -> None:
    import trade_rl.workflows.universal_causal_alpha_teacher as module

    symbols = ("AAAUSDT", "BBBUSDT")
    samples = {symbol: _samples(symbol) for symbol in symbols}
    partitions = {symbol: _partition(symbol) for symbol in symbols}
    candidates = (_candidate("selected"), _candidate("alternate", scale=2.0))
    fit_cache = CausalAlphaExpandingFitCache(
        train_symbols=symbols,
        samples=samples,
    )
    evaluated: list[tuple[str, int, int]] = []
    opened: list[str] = []
    closed: list[str] = []
    progress: list[dict[str, object]] = []
    resumed_metric = CausalAlphaCandidateEpisodeMetrics(
        candidate_digest=candidates[0].digest,
        symbol=symbols[0],
        episode_index=0,
        gross_return=0.02,
        net_return=0.01,
        turnover_per_day=0.1,
        total_execution_cost=1.0,
        trade_count=2,
        risk_violation=False,
    )

    def evaluate(environment, contract, *, actions):
        symbol = environment.symbol
        assert contract != partitions[symbol].holdout_contract
        evaluated.append((symbol, contract.episode_index, len(actions)))
        return SimpleNamespace(
            performance=SimpleNamespace(
                gross_return=0.02,
                net_return=0.01,
                turnover_total=3.0,
                cost_total=1.0,
                trade_count=2,
            ),
            collapse_evidence=SimpleNamespace(execution_rejection_count=0),
        )

    monkeypatch.setattr(
        module,
        "evaluate_episode_action_path_on_environment",
        evaluate,
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "evaluate_episode_action_path",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("selection must reuse one open environment per symbol")
        ),
    )

    def environment_factory(symbol: str):
        opened.append(symbol)
        return SimpleNamespace(
            symbol=symbol,
            close=lambda: closed.append(symbol),
        )

    selection = evaluate_causal_alpha_selection(
        train_symbols=symbols,
        samples=samples,
        partitions=partitions,
        candidates=candidates,
        environment_factories={
            symbol: (lambda symbol=symbol: environment_factory(symbol))
            for symbol in symbols
        },
        episode_hours=720.0,
        fit_cache=fit_cache,
        progress_callback=lambda payload: progress.append(dict(payload)),
        initial_metrics={candidates[0].digest: (resumed_metric,)},
    )

    assert set(evaluated) == {
        ("AAAUSDT", 0, 4),
        ("AAAUSDT", 1, 4),
        ("BBBUSDT", 0, 4),
        ("BBBUSDT", 1, 4),
    }
    assert len(evaluated) == 7
    assert evaluated.count(("AAAUSDT", 0, 4)) == 1
    assert selection.selected_candidate_digest in {
        candidate.digest for candidate in candidates
    }
    assert selection.holdout_episode_digests == {
        symbol: partitions[symbol].holdout_contract.digest for symbol in symbols
    }
    assert all(
        record.episode_index in {0, 1}
        for item in selection.candidates
        for record in item.episode_metrics
    )
    assert fit_cache.fit_count == 2
    assert fit_cache.hit_count == 5
    assert opened == list(symbols)
    assert closed == list(symbols)
    last_progress = dict(progress[-1])
    episode_metric = last_progress.pop("episode_metric")
    assert episode_metric["candidate_digest"] == candidates[-1].digest
    assert episode_metric["symbol"] == symbols[-1]
    assert episode_metric["episode_index"] == 1
    assert episode_metric["net_return"] == pytest.approx(0.01)
    assert last_progress == {
        "candidate_digest": candidates[-1].digest,
        "completed_replays": 8,
        "episode_index": partitions[symbols[-1]].selection_contracts[-1].episode_index,
        "fit_cache_entries": 2,
        "fit_cache_hits": 5,
        "fit_count": 2,
        "phase": "causal_teacher_selection",
        "prediction_cache_hits": 3,
        "prediction_count": 4,
        "symbol": symbols[-1],
        "total_replays": 8,
    }


def test_cost_aware_production_selection_persists_complete_v2_metric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import trade_rl.workflows.universal_causal_alpha_teacher as module

    symbol = "AAAUSDT"
    samples = {symbol: _samples(symbol)}
    partitions = {symbol: _partition(symbol)}
    candidate = default_cost_aware_causal_alpha_candidate_grid(
        risk_config=PreTradeRiskConfig(max_abs_weight=1.0, no_trade_band=0.05)
    )[0]
    signal = evaluate_causal_alpha_signal_diagnostics(
        np.asarray([-0.02, -0.01, 0.01, 0.02]),
        np.asarray([-0.01, -0.02, 0.02, 0.01]),
    )
    progress: list[dict[str, object]] = []

    def targets(**kwargs):
        contract = kwargs["contract"]
        return SimpleNamespace(
            actions=np.zeros((contract.stop - contract.start - 1, 1), dtype=np.float32),
            signal_24h=signal,
            signal_72h=signal,
            target_path=SimpleNamespace(
                cost_suppressed_change_count=4,
                liquidity_deleveraging_count=2,
                liquidity_weight_caps=np.asarray([0.02, 0.01, 0.03, 0.02]),
                submitted_change_count=3,
                strong_reversal_count=1,
                sign_flip_count=1,
            ),
        )

    monkeypatch.setattr(module, "_cost_aware_causal_alpha_target_for_contract", targets)
    monkeypatch.setattr(
        module,
        "evaluate_episode_action_path_on_environment",
        lambda *_args, **_kwargs: SimpleNamespace(
            performance=SimpleNamespace(
                gross_return=0.02,
                net_return=0.01,
                turnover_total=0.5,
                cost_total=1.0,
                trade_count=2,
            ),
            collapse_evidence=SimpleNamespace(
                execution_rejection_count=0,
                execution_rejection_reason_counts=(),
                risk_projection_reason_counts=(("no_trade_band", 2),),
                hard_risk_violation=False,
            ),
        ),
    )
    environment = SimpleNamespace(
        dataset=SimpleNamespace(),
        decision_bars=1,
        config=SimpleNamespace(
            execution_cost=ExecutionCostConfig(), signal_delay_decisions=1
        ),
        close=lambda: None,
    )

    selection = evaluate_cost_aware_causal_alpha_selection(
        train_symbols=(symbol,),
        samples=samples,
        partitions=partitions,
        candidates=(candidate,),
        environment_factories={symbol: lambda: environment},
        episode_hours=720.0,
        thresholds=CausalAlphaSelectionThresholds(),
        progress_callback=lambda payload: progress.append(dict(payload)),
    )

    metrics = selection.candidates[0].episode_metrics
    assert len(metrics) == 2
    assert metrics[0].signal_24h.digest == signal.digest
    assert metrics[0].cost_suppressed_change_count == 4
    assert metrics[0].liquidity_deleveraging_count == 2
    assert metrics[0].liquidity_weight_cap_min == pytest.approx(0.01)
    assert metrics[0].liquidity_weight_cap_median == pytest.approx(0.02)
    assert metrics[0].liquidity_weight_cap_max == pytest.approx(0.03)
    assert metrics[0].risk_projection_reason_counts == (("no_trade_band", 2),)
    assert progress[-1]["episode_metric"]["schema_version"] == (
        "causal_alpha_candidate_episode_metrics_v2"
    )
