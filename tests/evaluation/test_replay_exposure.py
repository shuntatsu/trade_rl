"""Synthetic native gross evidence and source-free readback contract."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from tests.evaluation.test_shared_portfolio_replay import _market
from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.evaluation.replay import run_shared_portfolio_replay
from trade_rl.evaluation.replay_exposure import mean_pre_interval_gross_from_ledger
from trade_rl.simulation.execution import ExecutionCostConfig, MarketExecutor
from trade_rl.simulation.quantities import project_quantity
from trade_rl.strategies.portfolio import (
    PortfolioDecisionSnapshot,
    PortfolioTargetProposal,
)


class _LongShort:
    def decide_portfolio(
        self, snapshot: PortfolioDecisionSnapshot
    ) -> PortfolioTargetProposal:
        return PortfolioTargetProposal((0.4, -0.2))


def _fixture():
    close = np.array(
        [[100.0, 200.0], [110.0, 195.0], [90.0, 210.0], [120.0, 180.0], [105.0, 205.0]]
    )
    volume = np.full(close.shape, 1_000_000.0)
    volume[1, 0] = 0.2
    return replace(
        _market(close=close, volume=volume), contract_multipliers=np.array([2.0, 3.0])
    )


def _replay(dataset, *, exposure: bool):
    return run_shared_portfolio_replay(
        dataset,
        _LongShort(),
        start_index=0,
        stop_index=4,
        initial_capital=1000.0,
        execution_cost=replace(
            ExecutionCostConfig.zero(), fee_rate=0.001, max_participation_rate=0.5
        ),
        settle_terminal_position=True,
        capture_ledger_evidence=True,
        capture_exposure_evidence=exposure,
    )


def _read(payload, dataset, *, expected_digest=None):
    raw = canonical_json_bytes(payload)
    return mean_pre_interval_gross_from_ledger(
        raw,
        expected_ledger_digest=hashlib.sha256(raw).hexdigest()
        if expected_digest is None
        else expected_digest,
        expected_dataset_id=dataset.dataset_id,
        expected_execution_policy_digest=payload["execution_policy_digest"],
        expected_symbols=dataset.symbols,
        expected_start_index=0,
        expected_stop_index=4,
    )


def test_native_exposure_opt_in_preserves_v1_bytes_and_accounting(monkeypatch):
    dataset = _fixture()
    original = MarketExecutor.execute_interval
    gross = []

    def record(executor, book, target, **kwargs):
        gross.append(book.gross_exposure)
        return original(executor, book, target, **kwargs)

    monkeypatch.setattr(MarketExecutor, "execute_interval", record)
    legacy = _replay(dataset, exposure=False)
    native_gross = tuple(gross)
    gross.clear()
    extended = _replay(dataset, exposure=True)
    assert tuple(gross) == native_gross
    assert legacy.book.exact_quantities == extended.book.exact_quantities
    assert legacy.book.cash == extended.book.cash
    assert legacy.returns == extended.returns
    assert legacy.diagnostics == extended.diagnostics
    assert legacy.decisions == extended.decisions
    assert legacy.ledger_evidence is not None
    assert extended.ledger_evidence is not None
    prior = legacy.ledger_evidence.to_mapping()
    payload = extended.ledger_evidence.to_mapping()
    assert prior["schema_version"] == "shared_cash_portfolio_replay_ledger_v1"
    assert payload["schema_version"] == "shared_cash_portfolio_replay_ledger_v2"
    stripped = dict(payload)
    del stripped["pre_interval_exposure"]
    stripped["schema_version"] = prior["schema_version"]
    assert canonical_json_bytes(stripped) == canonical_json_bytes(prior)
    assert native_gross[0] == 0.0
    assert max(native_gross) > 0.0
    assert len(native_gross) == 4
    assert extended.ledger_evidence.exposure_evidence is not None
    expected_rows = [
        math.fsum(
            abs(float(Fraction(q))) * mark * multiplier / iv.portfolio_value_before
            for q, mark, multiplier in zip(
                iv.exact_quantities_before,
                row.mark_prices,
                row.contract_multipliers,
                strict=True,
            )
        )
        for row, iv in zip(
            extended.ledger_evidence.exposure_evidence.rows,
            extended.ledger_evidence.intervals.intervals,
            strict=True,
        )
        if iv.portfolio_value_before > 0
    ]
    assert _read(payload, dataset) == math.fsum(expected_rows) / len(expected_rows)
    assert extended.ledger_evidence.exposure_evidence is not None
    assert len(extended.ledger_evidence.exposure_evidence.rows) == 4
    assert any(
        event.event_type == "partial_fill"
        for event in extended.ledger_evidence.intervals.intervals[0].order_events
    )
    assert (
        float(
            extended.ledger_evidence.exposure_evidence.rows[1].contract_multipliers[0]
        )
        == 2.0
    )
    assert (
        extended.ledger_evidence.intervals.intervals[1]
        .exact_quantities_before[1]
        .startswith("-")
    )
    assert extended.book.exact_quantities == (0, 0)


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "symbol",
        "row",
        "timestamp",
        "marks",
        "multiplier",
        "quantity",
        "cash",
        "equity",
        "bool",
        "terminal",
        "schema",
    ],
)
def test_source_free_exposure_rejects_inconsistent_authenticated_payload(fault):
    dataset = _fixture()
    result = _replay(dataset, exposure=True)
    assert result.ledger_evidence is not None
    p = json.loads(canonical_json_bytes(result.ledger_evidence.to_mapping()))
    sidecar = p["pre_interval_exposure"]
    rows = sidecar["rows"]
    ivs = p["intervals"]
    if fault == "missing":
        rows.pop()
    elif fault == "symbol":
        sidecar["symbols"].reverse()
    elif fault == "row":
        rows[1]["start_index"] = 0
    elif fault == "timestamp":
        rows[1]["timestamp_ns"] = rows[0]["timestamp_ns"]
    elif fault == "marks":
        rows[1]["mark_prices"][0] = -1.0
    elif fault == "multiplier":
        rows[1]["contract_multipliers"][0] = 0.0
    elif fault == "quantity":
        ivs[1]["exact_quantities_before"][0] = "2/4"
    elif fault == "cash":
        ivs[1]["cash_before"] += 2.0
    elif fault == "equity":
        ivs[1]["portfolio_value_before"] += 1.0
    elif fault == "bool":
        rows[1]["mark_prices"][0] = True
    elif fault == "terminal":
        ivs.pop()
        rows.pop()
    else:
        p["schema_version"] = "shared_cash_portfolio_replay_ledger_v1"
    with pytest.raises(ValueError):
        _read(p, dataset)


def test_exposure_requires_explicit_ledger_and_verified_digest():
    dataset = _fixture()
    with pytest.raises(ValueError, match="ledger"):
        run_shared_portfolio_replay(
            dataset,
            _LongShort(),
            start_index=0,
            stop_index=4,
            capture_exposure_evidence=True,
        )
    result = _replay(dataset, exposure=True)
    assert result.ledger_evidence is not None
    p = json.loads(canonical_json_bytes(result.ledger_evidence.to_mapping()))
    original_digest = hashlib.sha256(canonical_json_bytes(p)).hexdigest()
    assert _read(p, dataset) != 999999.0
    p["pre_interval_exposure"]["rows"][1]["mark_prices"][0] *= 2
    with pytest.raises(ValueError, match="digest"):
        _read(p, dataset, expected_digest=original_digest)


def test_literal_positive_equity_mean_and_terminated_prefix_without_market_data():
    # Read-only fixture: three completed rows, including a zero pre-equity
    # row excluded by the metric. This is not a native replay claim.
    p = {
        "schema_version": "shared_cash_portfolio_replay_ledger_v2",
        "dataset_id": "a" * 64,
        "execution_policy_digest": "b" * 64,
        "start_index": 2,
        "stop_index": 7,
        "termination_reason": "insolvency",
        "pre_interval_exposure": {
            "symbols": ["A", "B"],
            "rows": [
                {
                    "start_index": i,
                    "timestamp_ns": 100 + i,
                    "mark_prices": [10.0, 20.0],
                    "contract_multipliers": [2.0, 3.0],
                }
                for i in range(2, 5)
            ],
        },
        "intervals": [
            {
                "start_index": 2,
                "next_index": 3,
                "exact_quantities_before": ["2", "-1"],
                "cash_before": 120.0,
                "portfolio_value_before": 100.0,
                "termination_reason": None,
            },
            {
                "start_index": 3,
                "next_index": 4,
                "exact_quantities_before": ["0", "0"],
                "cash_before": 0.0,
                "portfolio_value_before": 0.0,
                "termination_reason": None,
            },
            {
                "start_index": 4,
                "next_index": 5,
                "exact_quantities_before": ["0", "0"],
                "cash_before": 100.0,
                "portfolio_value_before": 100.0,
                "termination_reason": "insolvency",
            },
        ],
    }

    def read():
        raw = canonical_json_bytes(p)
        return mean_pre_interval_gross_from_ledger(
            raw,
            expected_ledger_digest=hashlib.sha256(raw).hexdigest(),
            expected_dataset_id="a" * 64,
            expected_execution_policy_digest="b" * 64,
            expected_symbols=("A", "B"),
            expected_start_index=2,
            expected_stop_index=7,
        )

    assert read() == 0.5  # mean(40/100+60/100, 0)
    p["intervals"][1]["cash_before"] = -1.0
    p["intervals"][1]["portfolio_value_before"] = -1.0
    assert read() == 0.5
    p["intervals"][0]["cash_before"] = 20.0
    p["intervals"][0]["portfolio_value_before"] = 0.0
    p["intervals"][2]["cash_before"] = 0.0
    p["intervals"][2]["portfolio_value_before"] = 0.0
    with pytest.raises(ValueError, match="no positive-equity"):
        read()
    del p["termination_reason"]
    with pytest.raises(ValueError, match="missing terminal"):
        read()


@pytest.mark.parametrize(
    "kind",
    [
        "different_policy",
        "different_dataset",
        "different_range",
        "different_roster",
        "noncanonical",
        "duplicate",
    ],
)
def test_reader_binds_independent_expected_identity_and_canonical_bytes(kind):
    dataset = _fixture()
    result = _replay(dataset, exposure=True)
    assert result.ledger_evidence is not None
    payload = result.ledger_evidence.to_mapping()
    raw = canonical_json_bytes(payload)
    expected = dict(
        expected_dataset_id=dataset.dataset_id,
        expected_execution_policy_digest=payload["execution_policy_digest"],
        expected_symbols=dataset.symbols,
        expected_start_index=0,
        expected_stop_index=4,
    )
    if kind == "different_policy":
        expected["expected_execution_policy_digest"] = "0" * 64
    elif kind == "different_dataset":
        expected["expected_dataset_id"] = "0" * 64
    elif kind == "different_range":
        expected["expected_stop_index"] = 5
    elif kind == "different_roster":
        expected["expected_symbols"] = tuple(reversed(dataset.symbols))
    elif kind == "noncanonical":
        raw = raw + b" "
    else:
        raw = b'{"schema_version":"duplicate",' + raw[1:]
    with pytest.raises(ValueError):
        mean_pre_interval_gross_from_ledger(
            raw, expected_ledger_digest=hashlib.sha256(raw).hexdigest(), **expected
        )


def test_reporting_quantity_rounding_preserves_frozen_consumer_formula():
    q = Fraction(5, 6)
    native_quantity = project_quantity(q)
    assert native_quantity < float(q)
    payload = {
        "schema_version": "shared_cash_portfolio_replay_ledger_v2",
        "dataset_id": "a" * 64,
        "execution_policy_digest": "b" * 64,
        "start_index": 0,
        "stop_index": 1,
        "termination_reason": None,
        "pre_interval_exposure": {
            "symbols": ["A"],
            "rows": [
                {
                    "start_index": 0,
                    "timestamp_ns": 1,
                    "mark_prices": [1.0],
                    "contract_multipliers": [1.0],
                }
            ],
        },
        "intervals": [
            {
                "start_index": 0,
                "next_index": 1,
                "exact_quantities_before": ["5/6"],
                "cash_before": 0.0,
                "portfolio_value_before": native_quantity,
                "termination_reason": None,
            }
        ],
    }
    raw = canonical_json_bytes(payload)
    actual = mean_pre_interval_gross_from_ledger(
        raw,
        expected_ledger_digest=hashlib.sha256(raw).hexdigest(),
        expected_dataset_id="a" * 64,
        expected_execution_policy_digest="b" * 64,
        expected_symbols=("A",),
        expected_start_index=0,
        expected_stop_index=1,
    )
    assert actual == abs(float(q)) * 1.0 * 1.0 / native_quantity > 1.0


def test_actual_early_margin_termination_is_a_completed_prefix():
    close = np.array(
        [
            [100.0, 200.0],
            [100.0, 200.0],
            [100.0, 3000.0],
            [100.0, 3100.0],
            [100.0, 3200.0],
        ]
    )
    dataset = _market(close=close)
    result = run_shared_portfolio_replay(
        dataset,
        _LongShort(),
        start_index=0,
        stop_index=4,
        initial_capital=1000.0,
        execution_cost=ExecutionCostConfig.zero(),
        capture_ledger_evidence=True,
        capture_exposure_evidence=True,
    )
    assert result.ledger_evidence is not None
    ledger = result.ledger_evidence.intervals
    assert ledger.termination_reason == "margin_call"
    assert len(ledger.intervals) == 2
    assert ledger.intervals[-1].next_index == 2
    assert result.ledger_evidence.exposure_evidence is not None
    rows = result.ledger_evidence.exposure_evidence.rows
    assert len(rows) == 2
    expected = (
        math.fsum(
            math.fsum(
                abs(float(Fraction(q))) * mark * multiplier / iv.portfolio_value_before
                for q, mark, multiplier in zip(
                    iv.exact_quantities_before,
                    row.mark_prices,
                    row.contract_multipliers,
                    strict=True,
                )
            )
            for iv, row in zip(ledger.intervals, rows, strict=True)
        )
        / 2
    )
    assert _read(result.ledger_evidence.to_mapping(), dataset) == expected
