"""Persisted structural differential evidence for Stage A Nautilus replay."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal

from trade_rl.integrations.nautilus.funding_adapter import (
    CanonicalFundingLedger,
    FundingSettlementInput,
    canonicalize_funding_settlement_record,
)
from trade_rl.integrations.nautilus.instrument import MAINTAINED_BTCUSDT_PERPETUAL
from trade_rl.simulation.execution_parity import CanonicalExecutionRecord
from trade_rl.simulation.execution_replay import load_execution_event_artifact
from trade_rl.simulation.funding_evidence import load_funding_evidence_artifact_bytes
from trade_rl.workflows.stage_a_execution_store import StoredStageAExecutionReplay
from trade_rl.workflows.stage_a_nautilus_historical_replay import (
    StageANautilusHistoricalExecutionResult,
)

_SETTLEMENT_CURRENCY_PRECISION = 8
_FUNDING_REFERENCE_PRICE_INCREMENT = Decimal("0.00000001")


@dataclass(frozen=True, slots=True)
class StageANautilusHistoricalDifferentialEvidence:
    """Structural parity bound to one persisted authoritative Stage A replay."""

    replay_digest: str
    request_digest: str
    dataset_id: str
    candidate_runtime_version: str
    legacy_terminal_position_lots: int
    candidate_terminal_position_lots: int
    terminal_position_matches: bool
    terminal_open_orders_passed: bool
    funding_matches: bool
    structural_passed: bool


def build_stage_a_nautilus_historical_differential_evidence(
    stored: StoredStageAExecutionReplay,
    candidate: StageANautilusHistoricalExecutionResult,
) -> StageANautilusHistoricalDifferentialEvidence:
    """Compare persisted legacy structure with one completed Nautilus candidate replay."""

    event_artifact = load_execution_event_artifact(stored.event_path)
    if event_artifact.dataset_id != stored.artifact.cell_identity.dataset_id:
        raise ValueError("persisted historical differential dataset identity mismatch")

    legacy_terminal_position_lots = _legacy_terminal_position_lots(
        event_artifact.terminal_book
    )
    candidate_terminal_position_lots = candidate.execution.terminal_position_lots
    terminal_position_matches = (
        legacy_terminal_position_lots == candidate_terminal_position_lots
    )
    terminal_open_orders_passed = candidate.execution.terminal_open_orders == 0
    funding_matches = _legacy_funding_records(stored) == candidate.funding_records
    structural_passed = (
        terminal_position_matches and terminal_open_orders_passed and funding_matches
    )

    return StageANautilusHistoricalDifferentialEvidence(
        replay_digest=stored.artifact.digest,
        request_digest=stored.artifact.cell_identity.request_digest,
        dataset_id=stored.artifact.cell_identity.dataset_id,
        candidate_runtime_version=candidate.execution.runtime_version,
        legacy_terminal_position_lots=legacy_terminal_position_lots,
        candidate_terminal_position_lots=candidate_terminal_position_lots,
        terminal_position_matches=terminal_position_matches,
        terminal_open_orders_passed=terminal_open_orders_passed,
        funding_matches=funding_matches,
        structural_passed=structural_passed,
    )


def _legacy_terminal_position_lots(terminal_book: dict[str, object]) -> int:
    raw_quantities = terminal_book.get("quantities")
    if not isinstance(raw_quantities, list) or len(raw_quantities) != 1:
        raise ValueError(
            "persisted historical differential requires one terminal quantity"
        )
    quantity = raw_quantities[0]
    if isinstance(quantity, bool) or not isinstance(quantity, (int, float)):
        raise ValueError("persisted historical terminal quantity must be numeric")
    return _exact_grid_units(
        Decimal(str(quantity)),
        Decimal(MAINTAINED_BTCUSDT_PERPETUAL.size_increment),
        field="terminal quantity",
    )


def _legacy_funding_records(
    stored: StoredStageAExecutionReplay,
) -> tuple[CanonicalExecutionRecord, ...]:
    if stored.funding_path is None:
        return ()

    funding = load_funding_evidence_artifact_bytes(stored.funding_path.read_bytes())
    if funding.dataset_id != stored.artifact.cell_identity.dataset_id:
        raise ValueError("persisted historical funding dataset identity mismatch")
    if (
        funding.execution_policy_digest
        != stored.artifact.cell_identity.execution_identity
    ):
        raise ValueError("persisted historical funding execution identity mismatch")
    if funding.symbol_count != 1:
        raise ValueError(
            "persisted historical differential requires one funding symbol"
        )

    spec = MAINTAINED_BTCUSDT_PERPETUAL
    ledger = CanonicalFundingLedger()
    records: list[CanonicalExecutionRecord] = []
    for sequence, boundary in enumerate(funding.boundaries, start=1):
        if any(
            len(values) != 1
            for values in (
                boundary.signed_quantities,
                boundary.mark_prices,
                boundary.contract_multipliers,
                boundary.funding_rates,
            )
        ):
            raise ValueError(
                "persisted historical differential requires single-symbol funding"
            )
        settlement = ledger.settle(
            FundingSettlementInput(
                instrument_id=spec.instrument_id,
                settlement_currency=spec.settlement_currency,
                currency_precision=_SETTLEMENT_CURRENCY_PRECISION,
                signed_quantity=Decimal(str(boundary.signed_quantities[0])),
                settlement_price=Decimal(str(boundary.mark_prices[0])),
                contract_multiplier=Decimal(str(boundary.contract_multipliers[0])),
                funding_rate=Decimal(str(boundary.funding_rates[0])),
                boundary_ns=boundary.timestamp_ns,
            )
        )
        records.append(
            canonicalize_funding_settlement_record(
                settlement,
                sequence=sequence,
                price_tick=_FUNDING_REFERENCE_PRICE_INCREMENT,
                lot_size=Decimal(spec.size_increment),
                equity_before_minor=_currency_minor_units(
                    boundary.equity_before_funding,
                    field="equity_before_funding",
                ),
            )
        )
    return tuple(records)


def _currency_minor_units(value: float, *, field: str) -> int:
    decimal_value = Decimal(str(value))
    if not decimal_value.is_finite():
        raise ValueError(f"persisted historical {field} must be finite")
    scaled = decimal_value * (Decimal(10) ** _SETTLEMENT_CURRENCY_PRECISION)
    return int(scaled.quantize(Decimal("1"), rounding=ROUND_HALF_EVEN))


def _exact_grid_units(value: Decimal, increment: Decimal, *, field: str) -> int:
    units = value / increment
    integral = units.to_integral_value()
    if units != integral:
        raise ValueError(f"persisted historical {field} must align to maintained grid")
    return int(integral)


__all__ = [
    "StageANautilusHistoricalDifferentialEvidence",
    "build_stage_a_nautilus_historical_differential_evidence",
]
