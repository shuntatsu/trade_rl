"""Opt-in adapters from a small single-symbol strategy set to a portfolio policy."""

from __future__ import annotations

import math
from collections.abc import Callable
from fractions import Fraction
from numbers import Real
from typing import cast

import numpy as np

from trade_rl.strategies.controls import ConstantIntentStrategy
from trade_rl.strategies.forecasts.ridge import RidgeForecastStrategy
from trade_rl.strategies.interface import SingleSymbolStrategy, StrategyObservation
from trade_rl.strategies.portfolio import (
    PortfolioDecisionSnapshot,
    PortfolioTargetProposal,
)
from trade_rl.strategies.position_intent import PositionIntent
from trade_rl.strategies.rules.trend import TrendIntentStrategy

_SUPPORTED_STRATEGY_TYPES = (
    ConstantIntentStrategy,
    TrendIntentStrategy,
    RidgeForecastStrategy,
)


def _fixed_names(values: object, *, name: str) -> tuple[str, ...]:
    if type(values) is not tuple:
        raise TypeError(f"{name} must be a tuple")
    if not values or any(
        not isinstance(value, str) or not value.strip() for value in values
    ):
        raise ValueError(f"{name} must contain non-empty names")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must contain unique names")
    return values


def _validated_budgets(values: object, *, n_symbols: int) -> tuple[float, ...]:
    if type(values) is not tuple or len(values) != n_symbols:
        raise ValueError("intent_gross_budgets must match the fixed symbol roster")
    budgets: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("each intent gross budget must be a real number")
        try:
            budget = float(value)
        except OverflowError as exc:
            raise ValueError(
                "each intent gross budget must be finite and within (0, 1]"
            ) from exc
        if not math.isfinite(budget) or not 0.0 < budget <= 1.0:
            raise ValueError("each intent gross budget must be within (0, 1]")
        budgets.append(budget)
    return tuple(budgets)


def _filled_intent(exact_quantity: str) -> tuple[PositionIntent, float]:
    try:
        quantity = Fraction(exact_quantity)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError("snapshot exact quantity is not a rational value") from exc
    if str(quantity) != exact_quantity:
        raise ValueError("snapshot exact quantity must use its canonical rational form")
    try:
        quantity_float = float(quantity)
    except OverflowError as exc:
        raise ValueError(
            "snapshot exact quantity is outside finite strategy range"
        ) from exc
    if not math.isfinite(quantity_float):
        raise ValueError("snapshot exact quantity is outside finite strategy range")
    if quantity and quantity_float == 0.0:
        raise ValueError("snapshot exact quantity is below finite strategy precision")
    side = (
        PositionIntent.LONG
        if quantity > 0
        else PositionIntent.SHORT
        if quantity < 0
        else PositionIntent.FLAT
    )
    return side, quantity_float


class SingleSymbolPortfolioPolicyAdapter:
    """Map supported built-in single-symbol intents to one numeric portfolio target.

    Only the exact built-in constant, trend, and Ridge strategy classes are
    supported. The adapter derives current side from exact filled quantity and
    supplies an explicit per-symbol intent budget. It does not reproduce legacy
    cached-target, minimum-hold, pending-order, or exact-quantity HOLD behavior;
    the shared portfolio session remains responsible for native risk and order
    reconciliation. Use a direct ``PortfolioPolicy`` for richer semantics.
    Factories must return distinct instances within this roster; comparison
    orchestrators should build fresh instances for every arm because this
    adapter cannot detect sharing across separate adapter objects.
    """

    __slots__ = (
        "_symbols",
        "_feature_names",
        "_global_feature_names",
        "_intent_gross_budgets",
        "_strategies",
        "_last_index",
        "_last_timestamp_ns",
        "_agent_stop_index",
        "_poisoned",
    )

    def __init__(
        self,
        *,
        symbols: tuple[str, ...],
        feature_names: tuple[str, ...],
        global_feature_names: tuple[str, ...],
        intent_gross_budgets: tuple[float, ...],
        strategy_factories: tuple[Callable[[], object], ...],
    ) -> None:
        self._symbols = _fixed_names(symbols, name="symbols")
        self._feature_names = _fixed_names(feature_names, name="feature_names")
        self._global_feature_names = _fixed_names(
            global_feature_names,
            name="global_feature_names",
        )
        self._intent_gross_budgets = _validated_budgets(
            intent_gross_budgets,
            n_symbols=len(self._symbols),
        )
        if type(strategy_factories) is not tuple or len(strategy_factories) != len(
            self._symbols
        ):
            raise ValueError("strategy_factories must match the fixed symbol roster")
        if any(not callable(factory) for factory in strategy_factories):
            raise TypeError("every strategy factory must be callable")

        created = tuple(factory() for factory in strategy_factories)
        for strategy in created:
            if type(strategy) not in _SUPPORTED_STRATEGY_TYPES:
                raise TypeError(
                    "unsupported strategy type; supported strategy types are exact "
                    "ConstantIntentStrategy, TrendIntentStrategy, and RidgeForecastStrategy"
                )
        strategies = cast(tuple[SingleSymbolStrategy, ...], created)
        if len({id(strategy) for strategy in strategies}) != len(strategies):
            raise ValueError("strategy factories must return distinct strategy objects")

        self._strategies = strategies
        self._last_index: int | None = None
        self._last_timestamp_ns: int | None = None
        self._agent_stop_index: int | None = None
        self._poisoned = False

    @property
    def symbols(self) -> tuple[str, ...]:
        return self._symbols

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._feature_names

    @property
    def global_feature_names(self) -> tuple[str, ...]:
        return self._global_feature_names

    @property
    def intent_gross_budgets(self) -> tuple[float, ...]:
        return self._intent_gross_budgets

    def decide_portfolio(
        self, snapshot: PortfolioDecisionSnapshot
    ) -> PortfolioTargetProposal:
        if self._poisoned:
            raise RuntimeError(
                "portfolio policy adapter is poisoned after a callback failure"
            )
        observations = self._build_observations(snapshot)

        try:
            targets: list[float] = []
            for strategy, observation, budget in zip(
                self._strategies,
                observations,
                self._intent_gross_budgets,
                strict=True,
            ):
                intent = strategy.decide(observation)
                if not isinstance(intent, PositionIntent):
                    raise TypeError("strategy output must be a PositionIntent")
                targets.append(float(int(intent)) * budget)
            proposal = PortfolioTargetProposal(target_weights=tuple(targets))
        except BaseException:
            # An interrupted partial callback sequence cannot be retried safely.
            self._poisoned = True
            raise

        self._last_index = snapshot.index
        self._last_timestamp_ns = snapshot.timestamp_ns
        self._agent_stop_index = snapshot.agent_stop_index
        return proposal

    def _build_observations(
        self, snapshot: PortfolioDecisionSnapshot
    ) -> tuple[StrategyObservation, ...]:
        if type(snapshot) is not PortfolioDecisionSnapshot:
            raise TypeError("snapshot must be a PortfolioDecisionSnapshot")
        if (
            snapshot.symbols != self._symbols
            or snapshot.feature_names != self._feature_names
            or snapshot.global_feature_names != self._global_feature_names
        ):
            raise ValueError(
                "snapshot symbols or feature roster does not match the adapter"
            )
        if (
            snapshot.account.insolvent
            or snapshot.account.termination_reason is not None
        ):
            raise ValueError(
                "strategy decisions are unavailable after account termination"
            )
        timestamp = np.datetime64(snapshot.timestamp_ns, "ns")
        if np.isnat(timestamp):
            raise ValueError("snapshot timestamp must not be NaT")
        if self._last_index is not None:
            if self._last_timestamp_ns is None or self._agent_stop_index is None:
                raise RuntimeError("portfolio policy sequence state is incomplete")
            if snapshot.index != self._last_index + 1:
                raise ValueError("snapshot index must advance by exactly one decision")
            if snapshot.timestamp_ns <= self._last_timestamp_ns:
                raise ValueError("snapshot timestamps must increase between decisions")
            if snapshot.agent_stop_index != self._agent_stop_index:
                raise ValueError("snapshot agent stop index must stay fixed")

        observations: list[StrategyObservation] = []
        for symbol_index, budget in enumerate(self._intent_gross_budgets):
            exact_quantity = snapshot.account.exact_quantities[symbol_index]
            if not isinstance(exact_quantity, str):
                raise TypeError("snapshot exact quantities must be strings")
            current_intent, quantity = _filled_intent(exact_quantity)
            reported_quantity = float(snapshot.account.quantities[symbol_index])
            if quantity != reported_quantity:
                raise ValueError("exact and reported snapshot quantities disagree")
            observations.append(
                StrategyObservation(
                    index=snapshot.index,
                    timestamp=timestamp,
                    symbol=self.symbols[symbol_index],
                    features=np.asarray(
                        snapshot.features[symbol_index], dtype=np.float64
                    ),
                    feature_available=np.asarray(
                        snapshot.feature_available[symbol_index], dtype=np.bool_
                    ),
                    feature_staleness=np.asarray(
                        snapshot.feature_staleness[symbol_index], dtype=np.float32
                    ),
                    global_features=np.asarray(
                        snapshot.global_features, dtype=np.float64
                    ),
                    global_feature_available=np.asarray(
                        snapshot.global_feature_available, dtype=np.bool_
                    ),
                    current_intent=current_intent,
                    current_weight=float(snapshot.account.weights[symbol_index]),
                    current_position_quantity=quantity,
                    intent_gross_budget=budget,
                )
            )
        return tuple(observations)


class CashPortfolioPolicy:
    """Return explicit zero targets for one fixed non-empty symbol roster."""

    __slots__ = ("_symbols",)

    def __init__(self, *, symbols: tuple[str, ...]) -> None:
        self._symbols = _fixed_names(symbols, name="symbols")

    @property
    def symbols(self) -> tuple[str, ...]:
        return self._symbols

    def decide_portfolio(
        self, snapshot: PortfolioDecisionSnapshot
    ) -> PortfolioTargetProposal:
        if type(snapshot) is not PortfolioDecisionSnapshot:
            raise TypeError("snapshot must be a PortfolioDecisionSnapshot")
        if snapshot.symbols != self._symbols:
            raise ValueError("snapshot symbols do not match the cash policy roster")
        if (
            snapshot.account.insolvent
            or snapshot.account.termination_reason is not None
        ):
            raise ValueError("cash policy is unavailable after account termination")
        return PortfolioTargetProposal(target_weights=(0.0,) * len(self._symbols))


__all__ = ["CashPortfolioPolicy", "SingleSymbolPortfolioPolicyAdapter"]
