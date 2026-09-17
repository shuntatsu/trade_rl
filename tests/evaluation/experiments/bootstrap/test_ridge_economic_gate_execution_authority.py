from __future__ import annotations

import json

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.experiments.bootstrap.ridge_economic_gate_evaluation import (
    RidgeEconomicGateEvaluation,
    RidgeEconomicGateSymbolResult,
    canonical_ridge_economic_gate_evaluation_spec,
)
from trade_rl.evaluation.experiments.bootstrap.ridge_economic_gate_execution_authority import (
    build_ridge_economic_gate_pre_pnl_cost_authority,
    canonical_ridge_economic_gate_execution_publication_bytes,
    execute_ridge_economic_gate_with_authority,
)

_AUTHORITY_HEAD = "a" * 40
_ECONOMIC_HEAD = "b" * 40


def _dataset(*, fee_rate: float = 0.0005) -> MarketDataset:
    spec = canonical_ridge_economic_gate_evaluation_spec()
    timestamps = np.arange(
        np.datetime64("2023-01-01", "D"),
        np.datetime64("2025-01-02", "D"),
        dtype="datetime64[D]",
    ).astype("datetime64[ns]")
    n_bars = timestamps.size
    n_symbols = len(spec.symbols)
    close = np.full((n_bars, n_symbols), 100.0)
    return MarketDataset(
        dataset_id=spec.dataset_id,
        symbols=spec.symbols,
        timestamps=timestamps,
        features=np.zeros((n_bars, n_symbols, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=close.copy(),
        high=close.copy(),
        low=close.copy(),
        close=close.copy(),
        volume=np.full((n_bars, n_symbols), 1_000_000.0),
        funding_rate=np.zeros((n_bars, n_symbols)),
        tradable=np.ones((n_bars, n_symbols), dtype=np.bool_),
        feature_available=np.ones((n_bars, n_symbols, 1), dtype=np.bool_),
        feature_names=("unused",),
        global_feature_names=("regime",),
        fee_rate=np.full((n_bars, n_symbols), fee_rate),
        taker_fee_rate=np.zeros((n_bars, n_symbols)),
        spread_rate=np.full((n_bars, n_symbols), spec.market_order_spread_rate),
        max_participation_rate=np.broadcast_to(
            np.asarray(spec.capacity_caps, dtype=np.float64),
            (n_bars, n_symbols),
        ).copy(),
        periods_per_year=365,
    )


def _result() -> RidgeEconomicGateEvaluation:
    spec = canonical_ridge_economic_gate_evaluation_spec()
    rows = tuple(
        RidgeEconomicGateSymbolResult(
            symbol=symbol,
            baseline_total_return=0.10,
            candidate_total_return=0.11,
            excess_total_return=0.01,
            baseline_total_cost=10.0,
            candidate_total_cost=8.0,
            baseline_turnover_total=100.0,
            candidate_turnover_total=80.0,
            baseline_max_drawdown=0.20,
            candidate_max_drawdown=0.19,
            baseline_termination_count=0,
            candidate_termination_count=0,
            baseline_termination_reasons=(),
            candidate_termination_reasons=(),
            new_termination=False,
            baseline_n_periods=2,
            candidate_n_periods=2,
            baseline_return_sha256="c" * 64,
            candidate_return_sha256="d" * 64,
        )
        for symbol in spec.symbols
    )
    return RidgeEconomicGateEvaluation(
        spec_digest=spec.digest,
        dataset_id=spec.dataset_id,
        model_sha256="e" * 64,
        symbols=spec.symbols,
        by_symbol=rows,
        positive_effect_symbols=5,
        median_excess_total_return=0.01,
        cost_reduction_symbols=5,
        turnover_reduction_symbols=5,
        drawdown_nonworse_symbols=5,
        new_termination_symbols=0,
        candidate_positive_total_return_symbols=5,
        research_status="PROMOTE_RESEARCH_REFERENCE",
    )


def test_pre_pnl_authority_is_result_blind_and_content_addressed(monkeypatch) -> None:
    called = False

    def forbidden_evaluator(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("PRE-P&L authority must not fit or replay")

    monkeypatch.setattr(
        "trade_rl.evaluation.experiments.bootstrap."
        "ridge_economic_gate_execution_authority.evaluate_ridge_economic_gate",
        forbidden_evaluator,
    )
    authority = build_ridge_economic_gate_pre_pnl_cost_authority(
        _dataset(),
        authority_implementation_head=_AUTHORITY_HEAD,
        authority_run_id=123,
    )

    assert called is False
    assert authority.result_blind is True
    assert authority.evaluation_pnl_inspected is False
    assert authority.n_evaluation_rows == 731
    assert len(authority.digest) == 64
    assert len(authority.fee_rate_sha256) == 64
    assert len(authority.capacity_sha256) == 64


def test_pre_pnl_authority_rejects_cost_drift_without_economic_execution() -> None:
    with pytest.raises(ValueError, match="fee_rate differs from sealed value"):
        build_ridge_economic_gate_pre_pnl_cost_authority(
            _dataset(fee_rate=0.0006),
            authority_implementation_head=_AUTHORITY_HEAD,
            authority_run_id=123,
        )


def test_execution_rejects_missing_authority_before_evaluator(monkeypatch) -> None:
    called = False

    def forbidden_evaluator(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("evaluator must not run without PRE-P&L authority")

    monkeypatch.setattr(
        "trade_rl.evaluation.experiments.bootstrap."
        "ridge_economic_gate_execution_authority.evaluate_ridge_economic_gate",
        forbidden_evaluator,
    )

    with pytest.raises(TypeError, match="pre_pnl_authority"):
        execute_ridge_economic_gate_with_authority(
            _dataset(),
            pre_pnl_authority=None,  # type: ignore[arg-type]
            economic_implementation_head=_ECONOMIC_HEAD,
            economic_run_id=456,
        )
    assert called is False


def test_execution_rejects_tampered_authority_before_evaluator(monkeypatch) -> None:
    dataset = _dataset()
    authority = build_ridge_economic_gate_pre_pnl_cost_authority(
        dataset,
        authority_implementation_head=_AUTHORITY_HEAD,
        authority_run_id=123,
    )
    object.__setattr__(authority, "fee_rate_sha256", "f" * 64)
    called = False

    def forbidden_evaluator(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("evaluator must not run after authority mismatch")

    monkeypatch.setattr(
        "trade_rl.evaluation.experiments.bootstrap."
        "ridge_economic_gate_execution_authority.evaluate_ridge_economic_gate",
        forbidden_evaluator,
    )

    with pytest.raises(ValueError, match="does not match current Dataset evidence"):
        execute_ridge_economic_gate_with_authority(
            dataset,
            pre_pnl_authority=authority,
            economic_implementation_head=_ECONOMIC_HEAD,
            economic_run_id=456,
        )
    assert called is False


def test_execution_identity_is_validated_before_evaluator(monkeypatch) -> None:
    dataset = _dataset()
    authority = build_ridge_economic_gate_pre_pnl_cost_authority(
        dataset,
        authority_implementation_head=_AUTHORITY_HEAD,
        authority_run_id=123,
    )
    called = False

    def forbidden_evaluator(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("evaluator must not run with invalid execution identity")

    monkeypatch.setattr(
        "trade_rl.evaluation.experiments.bootstrap."
        "ridge_economic_gate_execution_authority.evaluate_ridge_economic_gate",
        forbidden_evaluator,
    )

    with pytest.raises(ValueError, match="economic_implementation_head"):
        execute_ridge_economic_gate_with_authority(
            dataset,
            pre_pnl_authority=authority,
            economic_implementation_head="not-a-sha",
            economic_run_id=456,
        )
    with pytest.raises(ValueError, match="economic_run_id"):
        execute_ridge_economic_gate_with_authority(
            dataset,
            pre_pnl_authority=authority,
            economic_implementation_head=_ECONOMIC_HEAD,
            economic_run_id=True,  # type: ignore[arg-type]
        )
    assert called is False


def test_valid_authority_binds_exact_execution_provenance(monkeypatch) -> None:
    dataset = _dataset()
    authority = build_ridge_economic_gate_pre_pnl_cost_authority(
        dataset,
        authority_implementation_head=_AUTHORITY_HEAD,
        authority_run_id=123,
    )
    expected_result = _result()
    calls = 0

    def fake_evaluator(dataset_arg, spec_arg):
        nonlocal calls
        calls += 1
        assert dataset_arg is dataset
        assert spec_arg.digest == canonical_ridge_economic_gate_evaluation_spec().digest
        return expected_result

    monkeypatch.setattr(
        "trade_rl.evaluation.experiments.bootstrap."
        "ridge_economic_gate_execution_authority.evaluate_ridge_economic_gate",
        fake_evaluator,
    )

    publication = execute_ridge_economic_gate_with_authority(
        dataset,
        pre_pnl_authority=authority,
        economic_implementation_head=_ECONOMIC_HEAD,
        economic_run_id=456,
    )
    encoded = canonical_ridge_economic_gate_execution_publication_bytes(publication)
    document = json.loads(encoded)

    assert calls == 1
    assert publication.pre_pnl_cost_authority_digest == authority.digest
    assert publication.economic_implementation_head == _ECONOMIC_HEAD
    assert publication.economic_run_id == 456
    assert publication.result is expected_result
    assert document["content_digest"] == publication.digest
    assert document["publication"]["economic_implementation_head"] == _ECONOMIC_HEAD
    assert document["publication"]["economic_run_id"] == 456
    assert document["publication"]["pre_pnl_cost_authority_digest"] == authority.digest
