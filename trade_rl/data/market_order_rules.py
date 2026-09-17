"""Immutable, source-verified MARKET rule values for an opt-in execution profile."""

from __future__ import annotations

import hashlib
import math
from dataclasses import InitVar, asdict, dataclass
from datetime import UTC, datetime
from fractions import Fraction
from typing import Any

from trade_rl._validation import require_sha256
from trade_rl.artifacts import canonical_json_bytes
from trade_rl.data.market import MarketDataset

_FACTORY_CAPABILITY = object()


def _nonnegative(value: float) -> Fraction:
    if isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise ValueError("quantity rules must be finite and nonnegative")
    return Fraction(str(float(value)))


def _lcm(steps: tuple[Fraction, ...]) -> Fraction:
    denominator = math.lcm(*(step.denominator for step in steps))
    return Fraction(math.lcm(*(int(step * denominator) for step in steps)), denominator)


def joint_lot_size(steps: tuple[float, ...], *, stress_factor: float = 1.0) -> float:
    """Intersect all decimal grids, plus the stressed common grid."""
    values = tuple(value for step in steps if (value := _nonnegative(step)))
    factor = _nonnegative(stress_factor)
    if not values or factor < 1:
        raise ValueError("joint lot requires a positive grid and stress >= 1")
    base = _lcm(values)
    quantum = _lcm((base, base * factor))
    if Fraction(str(float(quantum))) != quantum:
        raise ValueError("joint lot precision exceeds decimal representation")
    return float(quantum)


@dataclass(frozen=True, slots=True)
class MarketOrderRule:
    symbol_index: int
    symbol: str
    lot_size: float
    minimum_quantity: float
    maximum_quantity: float
    minimum_notional: float

    def __post_init__(self) -> None:
        if type(self.symbol_index) is not int or self.symbol_index < 0:
            raise ValueError("rule symbol index must be a nonnegative integer")
        if not isinstance(self.symbol, str) or not self.symbol:
            raise ValueError("rule symbol must be nonempty")
        lot, low, high, _ = (
            _nonnegative(value)
            for value in (
                self.lot_size,
                self.minimum_quantity,
                self.maximum_quantity,
                self.minimum_notional,
            )
        )
        if lot <= 0 or high <= 0 or max(1, math.ceil(low / lot)) * lot > high:
            raise ValueError("quantity rules have no admissible lot")


@dataclass(frozen=True, slots=True)
class MarketOrderProfile:
    """Construct through the source adapter's builder/loader only.

    The private capability protects supported API paths, not arbitrary reflection.
    In particular dataclass replacement cannot retain verified status.
    """

    dataset_id: str
    dataset_symbols: tuple[str, ...]
    rules: tuple[MarketOrderRule, ...]
    source_uri: str
    source_sha256: str
    retrieved_at: datetime
    account_mode: str
    reduce_only_exits: bool
    _factory: InitVar[object] = None

    def __post_init__(self, _factory: object) -> None:
        if _factory is not _FACTORY_CAPABILITY:
            raise ValueError("market order profile requires a verified source factory")
        require_sha256(self.dataset_id, field="dataset_id")
        require_sha256(self.source_sha256, field="source_sha256")
        if (
            type(self.dataset_symbols) is not tuple
            or not self.dataset_symbols
            or any(not isinstance(s, str) or not s for s in self.dataset_symbols)
            or len(set(self.dataset_symbols)) != len(self.dataset_symbols)
            or type(self.rules) is not tuple
            or not self.rules
            or any(type(rule) is not MarketOrderRule for rule in self.rules)
        ):
            raise ValueError("profile requires immutable unique symbols and rules")
        indices = tuple(rule.symbol_index for rule in self.rules)
        if indices != tuple(sorted(set(indices))) or any(
            rule.symbol_index >= len(self.dataset_symbols)
            or self.dataset_symbols[rule.symbol_index] != rule.symbol
            for rule in self.rules
        ):
            raise ValueError("profile rules must match ordered dataset symbols")
        if self.account_mode != "one_way" or type(self.reduce_only_exits) is not bool:
            raise ValueError("profile requires one-way mode and boolean reduce-only")
        if not self.source_uri or self.retrieved_at.utcoffset() is None:
            raise ValueError("profile requires source URI and aware retrieval time")

    def validate_dataset(self, dataset: MarketDataset) -> None:
        if not dataset.identity_verified:
            raise ValueError("dataset content identity must be verified")
        if (dataset.dataset_id, dataset.symbols) != (
            self.dataset_id,
            self.dataset_symbols,
        ):
            raise ValueError("profile dataset binding mismatch")
        if any(
            dataset.resolved_array("contract_multipliers")[r.symbol_index] != 1.0
            for r in self.rules
        ):
            raise ValueError(
                "source base-asset quantities require unit contract multipliers"
            )

    def rule_for(self, symbol_index: int) -> MarketOrderRule | None:
        return next(
            (rule for rule in self.rules if rule.symbol_index == symbol_index), None
        )

    def canonical_payload(self) -> dict[str, Any]:
        return dict(
            schema="market_order_profile_v1",
            venue="binance_usdm_perpetual",
            historical_application="current_snapshot_assumption",
            dataset_minimum_role="venue_constraint",
            reversal_mode="ordinary",
            dataset_id=self.dataset_id,
            dataset_symbols=list(self.dataset_symbols),
            rules=[asdict(rule) for rule in self.rules],
            source_uri=self.source_uri,
            source_sha256=self.source_sha256,
            retrieved_at=self.retrieved_at.astimezone(UTC).isoformat(),
            account_mode=self.account_mode,
            reduce_only_exits=self.reduce_only_exits,
        )

    @property
    def digest(self) -> str:
        return hashlib.sha256(
            canonical_json_bytes(self.canonical_payload())
        ).hexdigest()
