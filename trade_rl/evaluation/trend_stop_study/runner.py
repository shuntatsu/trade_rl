"""Multi-symbol runner: every (timeframe, variant, scenario) cell on the shared replay.

All economics go through the existing shared-portfolio replay (``MarketExecutor``,
``BookState``, ``PreTradeRisk``, Dataset funding). This module only builds the policy,
calls the replay, and measures the ledger it returns. It refuses, before any replay:

* a missing or malformed G4 authorization URL;
* a window or Dataset that reaches past the exploratory boundary (2026-07-31);
* a Dataset whose cadence or warm-up violates the protocol, or that holds a symbol
  outside the roster;
* a working tree that is dirty (the result records the code it ran);

and it writes an INVALID result, with no replay, for a Dataset that lacks any roster
symbol (there is no exclusion path and no reduced roster).
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.dev_realdata_study.protocol import PROTOCOL as _DEV_PROTOCOL
from trade_rl.evaluation.dev_realdata_study.protocol import DevStudyProtocol
from trade_rl.evaluation.dev_realdata_study.runner import (
    equity_path,
    exposure_summary,
    perturb_future,
    reconcile_ledger,
)
from trade_rl.evaluation.replay import (
    SharedCashPortfolioReplayResult,
    run_shared_portfolio_replay,
)
from trade_rl.evaluation.trend_stop_study.data import (
    IncompleteRosterError,
    validate_study_dataset,
)
from trade_rl.evaluation.trend_stop_study.metrics import (
    bootstrap_lower_bound,
    closed_trades_from_ledger,
    daily_return_series,
    evaluate_configuration,
    haircut_adjusted_equity,
    performance_summary,
    select_label,
    stop_exit_audit,
)
from trade_rl.evaluation.trend_stop_study.protocol import (
    ARM_BTC_BUY_AND_HOLD,
    ARM_EQUAL_WEIGHT_BUY_AND_HOLD,
    ARM_RULE,
    CELL_SCHEMA,
    LABEL_INVALID,
    PRIOR_EXPOSURE,
    PROTOCOL,
    RESULT_SCHEMA,
    SCENARIO_BASE,
    SCENARIO_COST_X2,
    SCENARIOS,
    STOP_MODEL,
    TrendStopProtocol,
    VariantSpec,
    check_protocol_window,
    hard_risk,
    protocol_digest,
    scenario_cost,
    trend_config,
    validate_authorization_url,
)
from trade_rl.strategies.allocation.buy_and_hold import BuyAndHoldPortfolioPolicy
from trade_rl.strategies.portfolio import (
    PortfolioDecisionSnapshot,
    PortfolioPolicy,
    PortfolioTargetProposal,
)
from trade_rl.strategies.rules.trend_stop import StopExitEvent, TrendStopPortfolioPolicy

Progress = Callable[[str], None]


class SingleSymbolBuyAndHold:
    """Hold one roster symbol: buy at the first decision, then keep exact quantity."""

    def __init__(self, *, symbols: tuple[str, ...], symbol: str, budget: float) -> None:
        if symbol not in symbols:
            raise ValueError("benchmark symbol is not in the roster")
        self._symbols = symbols
        self._index = symbols.index(symbol)
        self._budget = budget
        self._started = False

    def decide_portfolio(
        self, snapshot: PortfolioDecisionSnapshot
    ) -> PortfolioTargetProposal:
        if snapshot.symbols != self._symbols:
            raise ValueError("snapshot symbols do not match the benchmark roster")
        account = snapshot.account
        n_symbols = len(self._symbols)
        if self._started or account.insolvent or account.termination_reason:
            return PortfolioTargetProposal(
                target_weights=account.weights,
                hold_current_quantity_symbols=tuple(range(n_symbols)),
            )
        self._started = True
        weights = [0.0] * n_symbols
        weights[self._index] = self._budget
        return PortfolioTargetProposal(target_weights=tuple(weights))


@dataclass
class CellOutcome:
    """One written cell plus the arrays the gate and the equity file need."""

    summary: dict[str, Any]
    equity: np.ndarray
    engine_equity: np.ndarray
    timestamps_ns: np.ndarray
    daily: np.ndarray


def _ns(text: str) -> int:
    return int(np.datetime64(text, "ns").astype(np.int64))


def _row_of(dataset: MarketDataset, text: str) -> int:
    stamps = dataset.timestamps.astype("datetime64[ns]").astype(np.int64)
    row = int(np.searchsorted(stamps, _ns(text), side="left"))
    if row >= stamps.size or int(stamps[row]) != _ns(text):
        raise ValueError(f"{text} is not a bar close timestamp of the Dataset")
    return row


def window_rows(dataset: MarketDataset, protocol: TrendStopProtocol) -> tuple[int, int]:
    return (
        _row_of(dataset, protocol.evaluation_start),
        _row_of(dataset, protocol.evaluation_end),
    )


class CodeProvenanceError(ValueError):
    """The code that would run cannot be identified, or it is not this checkout."""


class DirtyWorkingTreeError(CodeProvenanceError):
    """The working tree has uncommitted or untracked files: refuse to run."""


@dataclass(frozen=True)
class CodeProvenance:
    """The code a result was produced by.

    ``git_head`` and ``trade_rl_tree`` (the Git tree id of ``trade_rl/`` at HEAD) name
    the committed code; ``dirty`` is true if ``git status`` lists any change or
    untracked file; ``module_digest`` hashes the raw bytes of every ``trade_rl`` module
    loaded when the record is captured (each must lie under the checkout), so an
    installed copy that differs from the checkout cannot hide.
    """

    git_head: str
    trade_rl_tree: str
    dirty: bool
    module_digest: str
    module_count: int

    def require_clean(self) -> None:
        if self.dirty:
            raise DirtyWorkingTreeError(
                "refusing to run from a dirty working tree: commit or remove every "
                "change and untracked file so the result names the exact code"
            )

    def to_mapping(self) -> dict[str, object]:
        return {
            "git_head": self.git_head,
            "trade_rl_tree": self.trade_rl_tree,
            "dirty": self.dirty,
            "module_digest": self.module_digest,
            "module_count": self.module_count,
        }


def repository_root() -> Path:
    """The checkout root that holds this ``trade_rl`` package."""

    return Path(__file__).resolve().parents[3]


def _git(root: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise CodeProvenanceError(
            f"cannot read the Git state of {root}: the code must run from a clean "
            "Git checkout"
        ) from error
    return completed.stdout


def git_state(root: Path) -> tuple[str, str, bool]:
    """``(HEAD commit, tree id of trade_rl/ at HEAD, dirty)`` of the checkout."""

    head = _git(root, "rev-parse", "HEAD").strip()
    tree = _git(root, "rev-parse", "HEAD:trade_rl").strip()
    status = _git(root, "status", "--porcelain", "--untracked-files=all")
    return head, tree, bool(status.strip())


def loaded_module_digest(
    root: Path, modules: Mapping[str, Any] | None = None
) -> tuple[str, int]:
    """SHA-256 over the loaded ``trade_rl`` module files; each must lie under ``root``."""

    loaded = sys.modules if modules is None else modules
    entries: list[list[str]] = []
    resolved_root = root.resolve()
    for name in sorted(loaded):
        if name != "trade_rl" and not name.startswith("trade_rl."):
            continue
        file = getattr(loaded[name], "__file__", None)
        if not file:
            continue
        path = Path(file).resolve()
        if not path.is_relative_to(resolved_root):
            raise CodeProvenanceError(
                f"module {name} is loaded from {path}, outside the checkout "
                f"{resolved_root}"
            )
        entries.append([name, hashlib.sha256(path.read_bytes()).hexdigest()])
    if not entries:
        raise CodeProvenanceError("no trade_rl module is loaded from the checkout")
    return hashlib.sha256(canonical_json_bytes(entries)).hexdigest(), len(entries)


def capture_code_provenance(
    root: Path | None = None, modules: Mapping[str, Any] | None = None
) -> CodeProvenance:
    """Identify the running code from Git and the loaded modules (never the network)."""

    base = repository_root() if root is None else root
    head, tree, dirty = git_state(base)
    digest, count = loaded_module_digest(base, modules)
    return CodeProvenance(
        git_head=head,
        trade_rl_tree=tree,
        dirty=dirty,
        module_digest=digest,
        module_count=count,
    )


def _dev_protocol(protocol: TrendStopProtocol) -> DevStudyProtocol:
    """Reuse the audited ledger reconciliation with this study's capital/tolerance."""

    return replace(
        _DEV_PROTOCOL,
        initial_capital=protocol.initial_capital,
        reconciliation_tolerance=protocol.reconciliation_tolerance,
    )


def _replay(
    dataset: MarketDataset,
    policy: PortfolioPolicy,
    protocol: TrendStopProtocol,
    scenario: str,
    *,
    start_index: int,
    stop_index: int,
) -> SharedCashPortfolioReplayResult:
    return run_shared_portfolio_replay(
        dataset,
        policy,
        start_index=start_index,
        stop_index=stop_index,
        initial_capital=protocol.initial_capital,
        execution_cost=scenario_cost(protocol, scenario),
        risk=hard_risk(protocol),
        settle_terminal_position=True,
        capture_ledger_evidence=True,
    )


def summarize_cell(
    *,
    arm: str,
    timeframe: str,
    variant: str | None,
    scenario: str,
    result: SharedCashPortfolioReplayResult,
    dataset: MarketDataset,
    protocol: TrendStopProtocol,
    stop_events: Sequence[StopExitEvent] = (),
    policy_summary: Mapping[str, int] | None = None,
) -> CellOutcome:
    ledger = result.ledger_evidence
    if ledger is None:
        raise ValueError("cell summaries need ledger evidence")
    engine_equity, stamps = equity_path(result, dataset)
    multipliers = dataset.resolved_array("contract_multipliers")
    intervals = ledger.intervals.intervals
    audit, haircut_by_fill = stop_exit_audit(intervals, stop_events, multipliers)
    # Every gated quantity comes from the haircut-adjusted equity path and closed
    # cycles; the engine (un-haircut) values are kept beside them as ``engine_*``.
    equity = haircut_adjusted_equity(engine_equity, len(intervals), haircut_by_fill)
    performance = performance_summary(stamps, equity, protocol)
    engine_performance = performance_summary(stamps, engine_equity, protocol)
    closed, per_symbol = closed_trades_from_ledger(
        intervals, multipliers, dataset.symbols, haircut_by_fill
    )
    engine_closed, _ = closed_trades_from_ledger(
        intervals, multipliers, dataset.symbols
    )
    reconciliation = reconcile_ledger(result, dataset, _dev_protocol(protocol))
    diagnostics = result.diagnostics
    summary: dict[str, Any] = {
        "schema_version": CELL_SCHEMA,
        "arm": arm,
        "timeframe": timeframe,
        "variant": variant,
        "scenario": scenario,
        "stop_model": STOP_MODEL,
        "dataset_id": dataset.dataset_id,
        "symbols": list(dataset.symbols),
        "protocol_digest": protocol_digest(protocol),
        "performance": performance,
        "engine_performance": engine_performance,
        "haircut": audit,
        "closed_trades": {
            **closed.digest_payload(),
            "win_rate": closed.win_rate,
            "profit_factor": closed.profit_factor,
        },
        "engine_closed_trades": {
            **engine_closed.digest_payload(),
            "win_rate": engine_closed.win_rate,
            "profit_factor": engine_closed.profit_factor,
        },
        "per_symbol": {
            symbol: item.digest_payload() for symbol, item in per_symbol.items()
        },
        "costs": {
            "execution_costs_usdt": float(diagnostics.total_cost),
            "funding_pnl_usdt": float(diagnostics.funding_pnl),
            "borrow_cost_usdt": float(diagnostics.borrow_cost),
            "turnover_total": float(diagnostics.turnover_total),
            "n_trades": int(diagnostics.n_trades),
        },
        "exposure": exposure_summary(result, dataset),
        "policy": dict(policy_summary or {}),
        "terminated": list(diagnostics.termination_reasons),
        "reconciliation": reconciliation,
    }
    return CellOutcome(
        summary=summary,
        equity=equity,
        engine_equity=engine_equity,
        timestamps_ns=stamps,
        daily=daily_return_series(stamps, equity),
    )


def run_rule_cell(
    dataset: MarketDataset,
    protocol: TrendStopProtocol,
    variant: VariantSpec,
    scenario: str,
    timeframe: str,
    *,
    start_index: int,
    stop_index: int,
) -> CellOutcome:
    policy = TrendStopPortfolioPolicy(dataset, trend_config(protocol, variant))
    result = _replay(
        dataset,
        policy,
        protocol,
        scenario,
        start_index=start_index,
        stop_index=stop_index,
    )
    return summarize_cell(
        arm=ARM_RULE,
        timeframe=timeframe,
        variant=variant.name,
        scenario=scenario,
        result=result,
        dataset=dataset,
        protocol=protocol,
        stop_events=policy.stop_events,
        policy_summary=policy.summary(),
    )


def run_benchmark_cell(
    dataset: MarketDataset,
    protocol: TrendStopProtocol,
    arm: str,
    scenario: str,
    timeframe: str,
    *,
    start_index: int,
    stop_index: int,
) -> CellOutcome:
    policy: PortfolioPolicy
    if arm == ARM_BTC_BUY_AND_HOLD:
        policy = SingleSymbolBuyAndHold(
            symbols=tuple(dataset.symbols),
            symbol=protocol.benchmark_symbol,
            budget=protocol.buy_and_hold_gross_budget,
        )
    elif arm == ARM_EQUAL_WEIGHT_BUY_AND_HOLD:
        policy = BuyAndHoldPortfolioPolicy(
            symbols=tuple(dataset.symbols),
            gross_budget=protocol.buy_and_hold_gross_budget,
        )
    else:
        raise ValueError(f"unknown benchmark arm: {arm}")
    result = _replay(
        dataset,
        policy,
        protocol,
        scenario,
        start_index=start_index,
        stop_index=stop_index,
    )
    return summarize_cell(
        arm=arm,
        timeframe=timeframe,
        variant=None,
        scenario=scenario,
        result=result,
        dataset=dataset,
        protocol=protocol,
    )


def _trace(result: SharedCashPortfolioReplayResult) -> list[bytes]:
    ledger = result.ledger_evidence
    assert ledger is not None
    return [
        canonical_json_bytes(interval.to_mapping())
        for interval in ledger.intervals.intervals
    ] + [canonical_json_bytes(decision.to_mapping()) for decision in result.decisions]


def causality_probe(
    dataset: MarketDataset,
    protocol: TrendStopProtocol,
    variant: VariantSpec,
    *,
    start_index: int,
) -> dict[str, Any]:
    """Pre-cut native evidence must be byte-identical under a perturbed future."""

    cut_index = _row_of(dataset, protocol.causality_cut)
    if cut_index <= start_index + 60:
        raise ValueError("causality cut must leave a meaningful pre-cut window")
    perturbed = perturb_future(dataset, after_index=cut_index)
    traces = []
    traded = False
    for data in (dataset, perturbed):
        policy = TrendStopPortfolioPolicy(data, trend_config(protocol, variant))
        result = run_shared_portfolio_replay(
            data,
            policy,
            start_index=start_index,
            stop_index=cut_index,
            initial_capital=protocol.initial_capital,
            execution_cost=scenario_cost(protocol, SCENARIO_BASE),
            risk=hard_risk(protocol),
            capture_ledger_evidence=True,
        )
        traces.append(_trace(result))
        if data is dataset:
            assert result.ledger_evidence is not None
            traded = any(
                any(Fraction(q) != 0 for q in interval.exact_quantities_after)
                for interval in result.ledger_evidence.intervals.intervals
            )
    identical = traces[0] == traces[1]
    return {
        "passed": bool(identical and traded),
        "identical_pre_cut_trace": bool(identical),
        "pre_cut_has_positions": bool(traded),
        "cut": protocol.causality_cut,
        "variant": variant.name,
        "n_trace_rows": len(traces[0]),
    }


def _invalid_result(
    protocol: TrendStopProtocol,
    authorization: str,
    provenance: CodeProvenance,
    reason: str,
    missing_symbols: Sequence[str],
) -> dict[str, Any]:
    """The INVALID outcome for an incomplete roster; no cell is replayed."""

    return {
        "schema_version": RESULT_SCHEMA,
        "status": LABEL_INVALID,
        "label": LABEL_INVALID,
        "reason": reason,
        "protocol_digest": protocol_digest(protocol),
        "g4_authorization_url": authorization,
        "code": provenance.to_mapping(),
        "missing_symbols": list(missing_symbols),
        "prior_exposure": list(PRIOR_EXPOSURE),
        "passing_configurations": [],
        "stop_rule_triggered": False,
        "notes": [
            "INVALID is not NO_WINNER: no replay was run and no rule is rejected",
            "the roster has no exclusion path: a missing archive is INVALID, never a "
            "reduced roster",
        ],
    }


def _cell_valid(summary: Mapping[str, Any]) -> bool:
    return bool(summary["reconciliation"]["passed"]) and not summary["terminated"]


def run_study(
    datasets: Mapping[str, MarketDataset],
    protocol: TrendStopProtocol = PROTOCOL,
    *,
    g4_authorization_url: str | None,
    code_provenance: CodeProvenance,
    progress: Progress | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, np.ndarray]]:
    """Run every cell. Returns ``(result, cells, equity_arrays)``.

    ``cells`` maps a cell name to its summary mapping; ``result`` holds the gate and
    label. Nothing here writes to disk. ``code_provenance`` names the code that runs
    (:func:`capture_code_provenance`); a dirty tree is refused before any replay.
    """

    say = progress or (lambda message: None)
    authorization = validate_authorization_url(g4_authorization_url)
    code_provenance.require_clean()
    check_protocol_window(protocol)
    if set(datasets) != set(protocol.timeframes):
        raise ValueError(
            "a Dataset is required for exactly the preregistered timeframes"
        )
    used_symbols: dict[str, tuple[str, ...]] = {}
    incomplete: list[IncompleteRosterError] = []
    for timeframe in protocol.timeframes:
        try:
            used_symbols[timeframe] = validate_study_dataset(
                datasets[timeframe], protocol, timeframe
            )
        except IncompleteRosterError as error:
            incomplete.append(error)
    if incomplete:
        missing = sorted({name for error in incomplete for name in error.missing})
        return (
            _invalid_result(
                protocol, authorization, code_provenance, str(incomplete[0]), missing
            ),
            {},
            {},
        )

    cells: dict[str, dict[str, Any]] = {}
    outcomes: dict[tuple[str, str, str, str], CellOutcome] = {}
    causality: dict[str, dict[str, Any]] = {}
    for timeframe in protocol.timeframes:
        dataset = datasets[timeframe]
        start_index, stop_index = window_rows(dataset, protocol)
        for variant in protocol.variants:
            say(f"{timeframe}: {variant.name} causality probe")
            causality[f"{timeframe}__{variant.name}"] = causality_probe(
                dataset, protocol, variant, start_index=start_index
            )
        for scenario in SCENARIOS:
            for arm in (ARM_BTC_BUY_AND_HOLD, ARM_EQUAL_WEIGHT_BUY_AND_HOLD):
                say(f"{timeframe}: {arm} / {scenario}")
                outcome = run_benchmark_cell(
                    dataset,
                    protocol,
                    arm,
                    scenario,
                    timeframe,
                    start_index=start_index,
                    stop_index=stop_index,
                )
                outcomes[(timeframe, arm, "", scenario)] = outcome
                cells[f"{timeframe}__{arm}__{scenario}"] = outcome.summary
            for variant in protocol.variants:
                say(f"{timeframe}: {variant.name} / {scenario}")
                outcome = run_rule_cell(
                    dataset,
                    protocol,
                    variant,
                    scenario,
                    timeframe,
                    start_index=start_index,
                    stop_index=stop_index,
                )
                outcomes[(timeframe, ARM_RULE, variant.name, scenario)] = outcome
                cells[f"{timeframe}__{variant.name}__{scenario}"] = outcome.summary

    configurations: dict[str, dict[str, Any]] = {}
    for timeframe in protocol.timeframes:
        benchmark = outcomes[(timeframe, ARM_BTC_BUY_AND_HOLD, "", SCENARIO_BASE)]
        for variant in protocol.variants:
            base = outcomes[(timeframe, ARM_RULE, variant.name, SCENARIO_BASE)]
            stressed = outcomes[(timeframe, ARM_RULE, variant.name, SCENARIO_COST_X2)]
            validity = (
                _cell_valid(base.summary)
                and _cell_valid(stressed.summary)
                and bool(causality[f"{timeframe}__{variant.name}"]["passed"])
            )
            configurations[f"{timeframe}__{variant.name}"] = evaluate_configuration(
                base.summary,
                stressed.summary,
                benchmark.summary,
                bootstrap_lower_bound(base.daily, protocol),
                validity,
                protocol,
            )
    selection = select_label(configurations)
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "status": "development_evidence_only",
        "protocol_digest": protocol_digest(protocol),
        "protocol": protocol.canonical_payload(),
        "g4_authorization_url": authorization,
        "code": code_provenance.to_mapping(),
        "roster": list(protocol.symbols),
        "prior_exposure": list(PRIOR_EXPOSURE),
        "datasets": {
            timeframe: {
                "dataset_id": datasets[timeframe].dataset_id,
                "symbols": list(used_symbols[timeframe]),
                "n_bars": datasets[timeframe].n_bars,
            }
            for timeframe in protocol.timeframes
        },
        "causality_probe": causality,
        "configurations": configurations,
        **selection,
        "notes": [
            "known-used exploratory window with disclosed prior exposure of the "
            "universe and window (prior_exposure): never a winner, never unused-data, "
            "paper or live evidence",
            "every gated metric is computed from the stop-haircut-adjusted equity and "
            "closed cycles; engine_* fields are the un-haircut reference",
            "stop_rule_triggered means: no re-tuning of stops, N, universe or "
            "timeframes on this window; RL step 2 is not started",
        ],
    }
    equity: dict[str, np.ndarray] = {}
    for (timeframe, arm, variant_name, scenario), outcome in outcomes.items():
        name = f"{timeframe}__{variant_name or arm}__{scenario}"
        equity[f"{name}__equity"] = outcome.equity
        equity[f"{name}__engine_equity"] = outcome.engine_equity
        equity[f"{name}__timestamp_ns"] = outcome.timestamps_ns
    return result, cells, equity


__all__ = [
    "CellOutcome",
    "CodeProvenance",
    "CodeProvenanceError",
    "DirtyWorkingTreeError",
    "SingleSymbolBuyAndHold",
    "capture_code_provenance",
    "causality_probe",
    "git_state",
    "loaded_module_digest",
    "repository_root",
    "run_benchmark_cell",
    "run_rule_cell",
    "run_study",
    "summarize_cell",
    "window_rows",
]
