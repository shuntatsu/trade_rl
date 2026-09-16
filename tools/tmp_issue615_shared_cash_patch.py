from __future__ import annotations

from pathlib import Path


path = Path("trade_rl/evaluation/replay.py")
text = path.read_text(encoding="utf-8")

text = text.replace(
    "from __future__ import annotations\n\nimport math\n",
    "from __future__ import annotations\n\nimport math\nfrom collections.abc import Sequence\n",
    1,
)

result_marker = '''@dataclass(frozen=True, slots=True)
class SingleSymbolReplayResult:
    book: BookState
    returns: ReturnSeries
    diagnostics: ExecutionDiagnostics
    decisions: tuple[ReplayDecision, ...]
'''
result_insert = result_marker + '''

@dataclass(frozen=True, slots=True)
class SharedCashReplayDecision:
    """One simultaneous multi-symbol decision before shared execution."""

    index: int
    intents: tuple[PositionIntent, ...]
    changed_intents: tuple[bool, ...]
    proposal_weights: tuple[float, ...]
    target_weights: tuple[float, ...]
    risk_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SharedCashReplayResult:
    """One shared-account multi-symbol replay result."""

    book: BookState
    returns: ReturnSeries
    diagnostics: ExecutionDiagnostics
    decisions: tuple[SharedCashReplayDecision, ...]
'''
if text.count(result_marker) != 1:
    raise SystemExit("shared-cash result insertion marker mismatch")
text = text.replace(result_marker, result_insert, 1)

function_marker = "\n\n__all__ = [\n"
function = r'''


def run_shared_cash_replay(
    dataset: MarketDataset,
    strategies: Sequence[SingleSymbolStrategy],
    *,
    start_index: int,
    stop_index: int,
    gross_budget: float,
    initial_capital: float = 100_000.0,
    execution_cost: ExecutionCostConfig | None = None,
    risk: PreTradeRisk | None = None,
) -> SharedCashReplayResult:
    """Replay all symbols against one shared cash, risk and execution book.

    Every strategy observes the same pre-execution book snapshot for a decision
    bar. The complete proposal vector is then constrained once and executed once,
    so no symbol can consume cash or risk budget before another symbol decides.
    ``strategies`` must contain one distinct instance per dataset symbol to avoid
    hidden mutable strategy state leaking across symbols.
    """

    strategy_tuple = tuple(strategies)
    if len(strategy_tuple) != dataset.n_symbols:
        raise ValueError("shared-cash replay requires one strategy per dataset symbol")
    if len({id(strategy) for strategy in strategy_tuple}) != len(strategy_tuple):
        raise ValueError("shared-cash replay requires a distinct strategy instance per symbol")
    if (
        isinstance(start_index, bool)
        or not isinstance(start_index, int)
        or isinstance(stop_index, bool)
        or not isinstance(stop_index, int)
        or not 0 <= start_index < stop_index < dataset.n_bars
    ):
        raise ValueError("replay range must satisfy 0 <= start < stop < n_bars")
    if not math.isfinite(initial_capital) or initial_capital <= 0.0:
        raise ValueError("initial_capital must be finite and positive")
    target_weight_for_intent(PositionIntent.LONG, gross_budget=gross_budget)

    initial_prices = dataset.resolved_array("mark_price")[start_index]
    book = BookState.zero(
        dataset.n_symbols,
        initial_capital,
        initial_prices,
        contract_multipliers=dataset.contract_multipliers,
    )
    executor = MarketExecutor(dataset, execution_cost or ExecutionCostConfig.zero())
    risk_controller = risk or _default_replay_risk(executor)
    _validate_risk_execution_compatibility(risk_controller, executor)
    current_intents = [PositionIntent.FLAT for _ in range(dataset.n_symbols)]
    desired_quantities = np.zeros(dataset.n_symbols, dtype=np.float64)
    decisions: list[SharedCashReplayDecision] = []
    returns: list[float] = []
    index = start_index

    while index < stop_index:
        intents: list[PositionIntent] = []
        changed_intents: list[bool] = []
        for symbol_index, strategy in enumerate(strategy_tuple):
            observation = _observation(
                dataset,
                index=index,
                symbol_index=symbol_index,
                book=book,
                current_intent=current_intents[symbol_index],
            )
            intent = strategy.decide(observation)
            if not isinstance(intent, PositionIntent):
                raise TypeError("strategy.decide must return PositionIntent")
            changed_intent = intent is not current_intents[symbol_index]
            if changed_intent:
                proposal_weight = target_weight_for_intent(
                    intent,
                    gross_budget=gross_budget,
                )
                desired_quantities[symbol_index] = _desired_quantity_from_weight(
                    book,
                    proposal_weight,
                    symbol_index=symbol_index,
                )
            intents.append(intent)
            changed_intents.append(changed_intent)

        proposal_weights = np.asarray(
            [
                _weight_for_desired_quantity(
                    book,
                    desired_quantities[symbol_index],
                    symbol_index=symbol_index,
                )
                for symbol_index in range(dataset.n_symbols)
            ],
            dtype=np.float64,
        )
        constrained = risk_controller.constrain(
            proposal_weights,
            current=book.weights,
            drawdown=book.max_drawdown,
        )
        if constrained.was_constrained and any(
            reason != "max_turnover" for reason in constrained.reasons
        ):
            desired_quantities = np.asarray(
                [
                    _desired_quantity_from_weight(
                        book,
                        float(constrained.weights[symbol_index]),
                        symbol_index=symbol_index,
                    )
                    for symbol_index in range(dataset.n_symbols)
                ],
                dtype=np.float64,
            )

        decisions.append(
            SharedCashReplayDecision(
                index=index,
                intents=tuple(intents),
                changed_intents=tuple(changed_intents),
                proposal_weights=tuple(float(value) for value in proposal_weights),
                target_weights=tuple(float(value) for value in constrained.weights),
                risk_reasons=constrained.reasons,
            )
        )
        execution = executor.execute_interval(
            book,
            constrained.weights,
            start_index=index,
            bars=1,
        )
        if execution.next_index <= index:
            raise RuntimeError("execution did not advance replay index")
        book = execution.book
        returns.append(execution.interval_net_return)
        current_intents = intents
        index = execution.next_index
        if book.termination_reason is not None:
            break

    termination_reasons: tuple[str, ...] = ()
    reason = book.termination_reason
    if reason is not None:
        reason_value = (
            reason.value if isinstance(reason, EconomicTerminationReason) else reason
        )
        termination_reasons = (reason_value,)
    diagnostics = ExecutionDiagnostics(
        turnover_total=book.turnover_total,
        total_cost=book.total_cost,
        funding_pnl=book.funding_pnl,
        borrow_cost=book.borrow_cost,
        n_trades=book.n_trades,
        rebalance_events=book.rebalance_events,
        termination_reasons=termination_reasons,
    )
    return SharedCashReplayResult(
        book=book.clone(),
        returns=ReturnSeries(
            values=tuple(returns),
            kind=ReturnKind.BASE_BAR,
            periods_per_year=dataset.periods_per_year,
        ),
        diagnostics=diagnostics,
        decisions=tuple(decisions),
    )
'''
if text.count(function_marker) != 1:
    raise SystemExit("shared-cash function insertion marker mismatch")
text = text.replace(function_marker, function + function_marker, 1)

old_all = '''__all__ = [
    "ReplayDecision",
    "SingleSymbolReplayResult",
    "run_single_symbol_replay",
]
'''
new_all = '''__all__ = [
    "ReplayDecision",
    "SharedCashReplayDecision",
    "SharedCashReplayResult",
    "SingleSymbolReplayResult",
    "run_shared_cash_replay",
    "run_single_symbol_replay",
]
'''
if text.count(old_all) != 1:
    raise SystemExit("replay __all__ replacement marker mismatch")
text = text.replace(old_all, new_all, 1)
path.write_text(text, encoding="utf-8")
