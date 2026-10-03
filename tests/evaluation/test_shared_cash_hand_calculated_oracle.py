from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.experiments.errors import ArtifactIntegrityError
from trade_rl.evaluation.experiments.evidence import (
    _validate_shared_cash_candidate_source_binding,
)
from trade_rl.evaluation.replay import run_shared_cash_replay
from trade_rl.risk import PreTradeRisk, PreTradeRiskConfig
from trade_rl.simulation import ExecutionCostConfig
from trade_rl.strategies.position_intent import PositionIntent


@dataclass
class _AlwaysLong:
    def decide(self, observation: object) -> PositionIntent:
        del observation
        return PositionIntent.LONG


def _two_symbol_market() -> MarketDataset:
    close = np.asarray(
        [
            [100.0, 200.0],
            [100.0, 200.0],
            [110.0, 190.0],
            [120.0, 180.0],
            [140.0, 160.0],
            [140.0, 160.0],
        ],
        dtype=np.float64,
    )
    open_price = np.vstack((close[0], close[:-1]))
    n_bars, n_symbols = close.shape
    return MarketDataset(
        dataset_id="e" * 64,
        symbols=("ALPHAUSDT", "BETAUSDT"),
        timestamps=np.datetime64("2026-01-01T00:00:00", "ns")
        + np.arange(n_bars) * np.timedelta64(1, "h"),
        features=np.zeros((n_bars, n_symbols, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=open_price,
        high=np.maximum(open_price, close),
        low=np.minimum(open_price, close),
        close=close,
        volume=np.full((n_bars, n_symbols), 1_000_000.0),
        funding_rate=np.zeros((n_bars, n_symbols), dtype=np.float64),
        tradable=np.ones((n_bars, n_symbols), dtype=np.bool_),
        feature_available=np.ones((n_bars, n_symbols, 1), dtype=np.bool_),
        feature_names=("signal",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
        mark_price=close.copy(),
    )


def test_shared_cash_terminal_cash_and_cost_match_hand_calculation() -> None:
    result = run_shared_cash_replay(
        _two_symbol_market(),
        (_AlwaysLong(), _AlwaysLong()),
        start_index=0,
        stop_index=5,
        gross_budget=0.25,
        initial_capital=1_000.0,
        execution_cost=ExecutionCostConfig(
            fee_rate=0.01,
            maker_fee_rate=0.0,
            taker_fee_rate=0.0,
            spread_rate=0.0,
            impact_rate=0.0,
            max_participation_rate=1.0,
        ),
        risk=PreTradeRisk(
            PreTradeRiskConfig(
                max_gross=0.75,
                max_abs_weight=0.5,
                max_turnover=None,
                drawdown_start=1.0,
                drawdown_stop=1.0,
            )
        ),
        settle_terminal_position=True,
        capture_ledger_evidence=True,
    )

    # Both 25% targets buy $250 at the next open: 2.5 ALPHA and 1.25 BETA.
    # Price P&L by the final mark is +$50 and $0, respectively. Entry fees are
    # $5 total; the $550 terminal sale incurs $5.50 in fees.
    expected_terminal_cash = 1_000.0 + 50.0 - 5.0 - 5.5
    expected_total_cost = 5.0 + 5.5

    ledger = result.ledger_evidence
    assert ledger is not None
    assert len(result.decisions) == 4
    assert len(ledger.intervals) == 5
    assert ledger.intervals[0].exact_quantities_after == ("5/2", "5/4")
    assert [interval.interval_cost for interval in ledger.intervals] == pytest.approx(
        [5.0, 0.0, 0.0, 0.0, 5.5]
    )
    np.testing.assert_allclose(result.book.quantities, np.zeros(2))
    assert result.diagnostics.total_cost == pytest.approx(expected_total_cost)
    assert result.book.total_cost == pytest.approx(expected_total_cost)
    assert result.book.cash == pytest.approx(expected_terminal_cash)
    assert ledger.final_total_cost == pytest.approx(expected_total_cost)
    assert ledger.final_cash == pytest.approx(expected_terminal_cash)


def test_shared_cash_dividend_and_cash_interest_match_hand_calculation() -> None:
    base = _two_symbol_market()
    close = np.tile(np.asarray((100.0, 200.0)), (base.n_bars, 1))
    dividend = np.zeros_like(close)
    dividend[2, 0] = 0.4
    cash_rate = np.zeros(base.n_bars, dtype=np.float64)
    cash_rate[1:3] = 0.0876
    dataset = replace(
        base,
        open=close.copy(),
        high=close.copy(),
        low=close.copy(),
        close=close.copy(),
        mark_price=close.copy(),
        dividend=dividend,
        cash_rate=cash_rate,
        identity_payload_json=None,
    )

    result = run_shared_cash_replay(
        dataset,
        (_AlwaysLong(), _AlwaysLong()),
        start_index=0,
        stop_index=2,
        gross_budget=0.25,
        initial_capital=1_000.0,
        execution_cost=ExecutionCostConfig.zero(),
        risk=PreTradeRisk(
            PreTradeRiskConfig(
                max_gross=0.75,
                max_abs_weight=0.5,
                max_turnover=None,
                drawdown_start=1.0,
                drawdown_stop=1.0,
            )
        ),
        settle_terminal_position=False,
        capture_ledger_evidence=True,
    )

    ledger = result.ledger_evidence
    assert ledger is not None
    assert ledger.intervals[0].exact_quantities_after == ("5/2", "5/4")
    assert ledger.intervals[0].interval_dividend == pytest.approx(0.0)
    assert ledger.intervals[0].interval_cash_interest == pytest.approx(0.005)
    assert ledger.intervals[1].interval_dividend == pytest.approx(1.0)
    assert ledger.intervals[1].interval_cash_interest == pytest.approx(0.00501005)
    assert ledger.final_cash == pytest.approx(501.01001005)
    assert ledger.final_portfolio_value == pytest.approx(1_001.01001005)


@pytest.mark.parametrize(
    "source_row",
    ("initial_mark", "open", "mark_price", "open_final", "mark_price_final"),
)
def test_shared_cash_source_binding_rejects_changed_dataset_price_row(
    source_row: str,
) -> None:
    dataset = _two_symbol_market().with_content_identity()
    result = run_shared_cash_replay(
        dataset,
        (_AlwaysLong(), _AlwaysLong()),
        start_index=0,
        stop_index=5,
        gross_budget=0.25,
        initial_capital=1_000.0,
        execution_cost=ExecutionCostConfig(
            fee_rate=0.01,
            maker_fee_rate=0.0,
            taker_fee_rate=0.0,
            spread_rate=0.0,
            impact_rate=0.0,
            max_participation_rate=1.0,
        ),
        risk=PreTradeRisk(
            PreTradeRiskConfig(
                max_gross=0.75,
                max_abs_weight=0.5,
                max_turnover=None,
                drawdown_start=1.0,
                drawdown_stop=1.0,
            )
        ),
        settle_terminal_position=True,
        capture_ledger_evidence=True,
        capture_accounting_evidence=True,
    )
    ledger = result.ledger_evidence
    assert ledger is not None
    artifact_digest = "d" * 64
    source_name = source_row.removesuffix("_final")
    expected_source_name = (
        "mark_price" if source_name == "initial_mark" else source_name
    )
    candidate_summary = {
        "dataset_id": dataset.dataset_id,
        "dataset_artifact": {"artifact_digest": artifact_digest},
        "shared_cash_ppo": {
            "ledger_evidence": {
                "schema_version": ledger.schema_version,
                "payload": ledger.to_mapping(),
            }
        },
    }

    _validate_shared_cash_candidate_source_binding(
        candidate_summary=candidate_summary,
        dataset=dataset,
        expected_dataset_artifact_digest=artifact_digest,
    )

    if source_name == "initial_mark":
        assert dataset.mark_price is not None
        changed_mark = dataset.mark_price.copy()
        changed_mark[ledger.start_index, 0] += 1.0
        changed_dataset = replace(
            dataset,
            mark_price=changed_mark,
            identity_payload_json=None,
        )
    else:
        transition_type = (
            "mark_revaluation" if source_name == "open" else "funding_mark"
        )
        transition = next(
            transition
            for interval in ledger.intervals
            for transition in interval.accounting_transitions
            if transition.transition_type == transition_type
            and (
                not source_row.endswith("_final")
                or transition.processing_index == ledger.stop_index
            )
        )
    if source_name == "open":
        changed_open = dataset.open.copy()
        changed_open[transition.processing_index, 0] += 1.0
        changed_dataset = replace(
            dataset,
            open=changed_open,
            high=np.maximum(dataset.high, changed_open),
            low=np.minimum(dataset.low, changed_open),
            identity_payload_json=None,
        )
    elif source_name == "mark_price":
        assert dataset.mark_price is not None
        changed_mark = dataset.mark_price.copy()
        changed_mark[transition.processing_index, 0] += 1.0
        changed_dataset = replace(
            dataset,
            mark_price=changed_mark,
            identity_payload_json=None,
        )

    with pytest.raises(
        ArtifactIntegrityError,
        match=f"source {expected_source_name} row",
    ):
        _validate_shared_cash_candidate_source_binding(
            candidate_summary=candidate_summary,
            dataset=changed_dataset,
            expected_dataset_artifact_digest=artifact_digest,
        )


def _eventful_shared_cash_candidate() -> tuple[MarketDataset, dict[str, object]]:
    base = _two_symbol_market()
    timestamps = base.timestamps.copy()
    timestamps[2:] += np.timedelta64(2, "h")
    asset_active = np.ones((base.n_bars, base.n_symbols), dtype=np.bool_)
    asset_active[3, 1] = False
    information_available = asset_active.copy()
    tradable = asset_active.copy()
    split_factor = np.ones((base.n_bars, base.n_symbols), dtype=np.float64)
    split_factor[2, 0] = 2.0
    adjusted_prices: dict[str, np.ndarray] = {}
    for field_name in ("open", "high", "low", "close", "mark_price"):
        prices = getattr(base, field_name).copy()
        prices[2:, 0] /= 2.0
        adjusted_prices[field_name] = prices
    dividend = np.zeros_like(split_factor)
    dividend[2, 0] = -0.25
    funding_due = np.zeros_like(asset_active)
    funding_due[2, 0] = True
    funding_rate = np.zeros_like(split_factor)
    funding_rate[2, 0] = 0.01
    delisting_recovery = np.ones_like(split_factor)
    delisting_recovery[3, 1] = 0.5
    borrow_rate = np.full_like(split_factor, 0.02)
    cash_rate = np.full(base.n_bars, 0.03, dtype=np.float64)
    dataset = replace(
        base,
        **adjusted_prices,
        timestamps=timestamps,
        calendar_kind="session_calendar",
        nominal_bar_hours=1.0,
        available_at=np.broadcast_to(
            timestamps[:, None], (base.n_bars, base.n_symbols)
        ).copy(),
        asset_active=asset_active,
        symbol_active=asset_active,
        information_available=information_available,
        tradable=tradable,
        split_factor=split_factor,
        dividend=dividend,
        funding_due=funding_due,
        funding_rate=funding_rate,
        delisting_recovery=delisting_recovery,
        borrow_rate=borrow_rate,
        cash_rate=cash_rate,
    ).with_content_identity()
    result = run_shared_cash_replay(
        dataset,
        (_AlwaysLong(), _AlwaysLong()),
        start_index=0,
        stop_index=5,
        gross_budget=0.25,
        initial_capital=1_000.0,
        execution_cost=ExecutionCostConfig(
            fee_rate=0.01,
            maker_fee_rate=0.0,
            taker_fee_rate=0.0,
            spread_rate=0.0,
            impact_rate=0.0,
            max_participation_rate=1.0,
        ),
        risk=PreTradeRisk(
            PreTradeRiskConfig(
                max_gross=0.75,
                max_abs_weight=0.5,
                max_turnover=None,
                drawdown_start=1.0,
                drawdown_stop=1.0,
            )
        ),
        settle_terminal_position=True,
        capture_ledger_evidence=True,
        capture_accounting_evidence=True,
    )
    ledger = result.ledger_evidence
    assert ledger is not None
    artifact_digest = "d" * 64
    candidate_summary = {
        "dataset_id": dataset.dataset_id,
        "dataset_artifact": {"artifact_digest": artifact_digest},
        "shared_cash_ppo": {
            "ledger_evidence": {
                "schema_version": ledger.schema_version,
                "payload": ledger.to_mapping(),
            }
        },
    }
    return dataset, candidate_summary


def test_shared_cash_source_binding_rejects_dataset_artifact_digest_mismatch() -> None:
    dataset, candidate_summary = _eventful_shared_cash_candidate()

    with pytest.raises(
        ArtifactIntegrityError,
        match="candidate source Dataset artifact digest mismatch",
    ):
        _validate_shared_cash_candidate_source_binding(
            candidate_summary=candidate_summary,
            dataset=dataset,
            expected_dataset_artifact_digest="e" * 64,
        )


def _change_accounting_source_row(
    dataset: MarketDataset,
    source_row: str,
) -> MarketDataset:
    if source_row == "split_factor":
        changed = dataset.split_factor.copy()
        changed[2, 0] += 0.5
        return replace(dataset, split_factor=changed, identity_payload_json=None)
    if source_row == "inactive_mask":
        changed = dataset.asset_active.copy()
        changed[3, 1] = True
        return replace(
            dataset,
            asset_active=changed,
            symbol_active=changed,
            identity_payload_json=None,
        )
    if source_row == "delisting_recovery":
        changed = dataset.delisting_recovery.copy()
        changed[3, 1] = 0.25
        return replace(dataset, delisting_recovery=changed, identity_payload_json=None)
    if source_row == "dividend":
        changed = dataset.dividend.copy()
        changed[2, 0] += 0.1
        return replace(dataset, dividend=changed, identity_payload_json=None)
    if source_row == "cash_rate":
        changed = dataset.cash_rate.copy()
        changed[2] += 0.01
        return replace(dataset, cash_rate=changed, identity_payload_json=None)
    if source_row == "borrow_rate":
        changed = dataset.borrow_rate.copy()
        changed[2, 0] += 0.01
        return replace(dataset, borrow_rate=changed, identity_payload_json=None)
    if source_row == "funding_due":
        changed = dataset.funding_due.copy()
        changed[2, 1] = True
        return replace(dataset, funding_due=changed, identity_payload_json=None)
    if source_row == "funding_rate":
        changed = dataset.funding_rate.copy()
        changed[2, 0] += 0.01
        return replace(dataset, funding_rate=changed, identity_payload_json=None)
    if source_row == "elapsed_time":
        changed = dataset.timestamps.copy()
        changed[2] += np.timedelta64(30, "m")
        return replace(
            dataset,
            timestamps=changed,
            available_at=np.broadcast_to(
                changed[:, None], (dataset.n_bars, dataset.n_symbols)
            ).copy(),
            identity_payload_json=None,
        )
    raise AssertionError(f"unsupported source row: {source_row}")


@pytest.mark.parametrize(
    "source_row",
    (
        "split_factor",
        "inactive_mask",
        "delisting_recovery",
        "dividend",
        "cash_rate",
        "borrow_rate",
        "funding_due",
        "funding_rate",
        "elapsed_time",
    ),
)
def test_shared_cash_source_binding_rejects_changed_accounting_source_rows(
    source_row: str,
) -> None:
    dataset, candidate_summary = _eventful_shared_cash_candidate()
    changed_dataset = _change_accounting_source_row(dataset, source_row)

    with pytest.raises(ArtifactIntegrityError, match="source"):
        _validate_shared_cash_candidate_source_binding(
            candidate_summary=candidate_summary,
            dataset=changed_dataset,
            expected_dataset_artifact_digest="d" * 64,
        )


@pytest.mark.parametrize(
    "source_field",
    ("volume", "max_participation_rate", "fee_rate", "close"),
)
def test_shared_cash_source_binding_rejects_changed_dataset_identity_rows(
    source_field: str,
) -> None:
    dataset, candidate_summary = _eventful_shared_cash_candidate()
    changed_values = getattr(dataset, source_field).copy()
    if source_field == "max_participation_rate":
        changed_values[2, 0] = max(0.01, changed_values[2, 0] * 0.5)
    else:
        changed_values[2, 0] += 0.01
    changes: dict[str, object] = {source_field: changed_values}
    if source_field == "close":
        changes["high"] = np.maximum(dataset.high, changed_values)
    changed_dataset = replace(
        dataset,
        **changes,
        identity_payload_json=None,
    )

    with pytest.raises(
        ArtifactIntegrityError,
        match="candidate source Dataset content identity mismatch",
    ):
        _validate_shared_cash_candidate_source_binding(
            candidate_summary=candidate_summary,
            dataset=changed_dataset,
            expected_dataset_artifact_digest="d" * 64,
        )
