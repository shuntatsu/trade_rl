from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEST_PATH = ROOT / "tests/evaluation/test_execution_evidence.py"


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one match in {path}, found {count}: {old[:100]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def write_tests() -> None:
    TEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEST_PATH.write_text(
        '''from __future__ import annotations

from trade_rl.evaluation.evidence import ExecutionDiagnostics
from trade_rl.evaluation.metrics import evaluate_performance
from trade_rl.evaluation.series import ReturnKind, ReturnSeries
from trade_rl.evaluation.walk_forward.stitching import FoldOOSResult, stitch_oos


def returns(*values: float) -> ReturnSeries:
    return ReturnSeries(
        values=values,
        kind=ReturnKind.BASE_BAR,
        periods_per_year=8_760,
    )


def test_execution_diagnostics_combine_without_losing_economic_evidence() -> None:
    combined = ExecutionDiagnostics.combine(
        (
            ExecutionDiagnostics(
                turnover_total=0.3,
                total_cost=12.0,
                funding_pnl=-2.0,
                borrow_cost=1.5,
                n_trades=2,
                rebalance_events=1,
                termination_reasons=("margin_call",),
            ),
            ExecutionDiagnostics(
                turnover_total=0.4,
                total_cost=8.0,
                funding_pnl=3.0,
                borrow_cost=0.5,
                n_trades=3,
                rebalance_events=2,
                termination_reasons=("drawdown_stop",),
            ),
        )
    )
    assert combined.turnover_total == 0.7
    assert combined.total_cost == 20.0
    assert combined.funding_pnl == 1.0
    assert combined.borrow_cost == 2.0
    assert combined.n_trades == 5
    assert combined.rebalance_events == 3
    assert combined.termination_reasons == ("margin_call", "drawdown_stop")


def test_stitching_and_metrics_preserve_execution_diagnostics() -> None:
    stitched = stitch_oos(
        (
            FoldOOSResult(
                fold_index=0,
                start=10,
                stop=12,
                returns=returns(0.01, -0.005),
                diagnostics=ExecutionDiagnostics(
                    turnover_total=0.2,
                    total_cost=5.0,
                    funding_pnl=-1.0,
                    borrow_cost=0.25,
                    n_trades=2,
                    rebalance_events=1,
                ),
            ),
            FoldOOSResult(
                fold_index=1,
                start=12,
                stop=14,
                returns=returns(0.02, 0.0),
                diagnostics=ExecutionDiagnostics(
                    turnover_total=0.4,
                    total_cost=7.0,
                    funding_pnl=2.0,
                    borrow_cost=0.75,
                    n_trades=3,
                    rebalance_events=2,
                    termination_reasons=("minimum_equity",),
                ),
            ),
        )
    )
    diagnostics = stitched.diagnostics
    metrics = evaluate_performance(
        stitched.returns,
        turnover_total=diagnostics.turnover_total,
        total_cost=diagnostics.total_cost,
        funding_pnl=diagnostics.funding_pnl,
        borrow_cost=diagnostics.borrow_cost,
        n_trades=diagnostics.n_trades,
        rebalance_events=diagnostics.rebalance_events,
        termination_count=diagnostics.termination_count,
    )
    assert metrics.turnover_total == 0.6
    assert metrics.total_cost == 12.0
    assert metrics.funding_pnl == 1.0
    assert metrics.borrow_cost == 1.0
    assert metrics.n_trades == 5
    assert metrics.rebalance_events == 3
    assert metrics.termination_count == 1
''',
        encoding="utf-8",
    )


def apply_implementation() -> None:
    evidence = ROOT / "trade_rl/evaluation/evidence.py"
    evidence.write_text(
        '''"""Immutable execution evidence carried with evaluation return series."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExecutionDiagnostics:
    """Economic and execution diagnostics for one evaluated range."""

    turnover_total: float = 0.0
    total_cost: float = 0.0
    funding_pnl: float = 0.0
    borrow_cost: float = 0.0
    n_trades: int = 0
    rebalance_events: int = 0
    termination_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name, value in (
            ("turnover_total", self.turnover_total),
            ("total_cost", self.total_cost),
            ("borrow_cost", self.borrow_cost),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{field_name} must be finite and non-negative")
        if not math.isfinite(self.funding_pnl):
            raise ValueError("funding_pnl must be finite")
        for field_name, value in (
            ("n_trades", self.n_trades),
            ("rebalance_events", self.rebalance_events),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        if any(not reason for reason in self.termination_reasons):
            raise ValueError("termination reasons must be non-empty")

    @property
    def termination_count(self) -> int:
        return len(self.termination_reasons)

    def digest_payload(self) -> dict[str, object]:
        return {
            "borrow_cost": self.borrow_cost,
            "funding_pnl": self.funding_pnl,
            "n_trades": self.n_trades,
            "rebalance_events": self.rebalance_events,
            "termination_reasons": self.termination_reasons,
            "total_cost": self.total_cost,
            "turnover_total": self.turnover_total,
        }

    @classmethod
    def combine(cls, values: Iterable[ExecutionDiagnostics]) -> ExecutionDiagnostics:
        items = tuple(values)
        return cls(
            turnover_total=sum(item.turnover_total for item in items),
            total_cost=sum(item.total_cost for item in items),
            funding_pnl=sum(item.funding_pnl for item in items),
            borrow_cost=sum(item.borrow_cost for item in items),
            n_trades=sum(item.n_trades for item in items),
            rebalance_events=sum(item.rebalance_events for item in items),
            termination_reasons=tuple(
                reason for item in items for reason in item.termination_reasons
            ),
        )
''',
        encoding="utf-8",
    )

    replace_once(
        "trade_rl/evaluation/__init__.py",
        "from trade_rl.evaluation.gates import resolve_gate\n",
        "from trade_rl.evaluation.evidence import ExecutionDiagnostics\nfrom trade_rl.evaluation.gates import resolve_gate\n",
    )
    replace_once(
        "trade_rl/evaluation/__init__.py",
        '    "CapacityPoint",\n',
        '    "CapacityPoint",\n    "ExecutionDiagnostics",\n',
    )

    replace_once(
        "trade_rl/evaluation/metrics.py",
        '''    funding_pnl: float
    n_trades: int
    n_periods: int
''',
        '''    funding_pnl: float
    borrow_cost: float
    n_trades: int
    rebalance_events: int
    termination_count: int
    n_periods: int
''',
    )
    replace_once(
        "trade_rl/evaluation/metrics.py",
        '''    funding_pnl: float = 0.0,
    n_trades: int = 0,
) -> PerformanceMetrics:
''',
        '''    funding_pnl: float = 0.0,
    borrow_cost: float = 0.0,
    n_trades: int = 0,
    rebalance_events: int = 0,
    termination_count: int = 0,
) -> PerformanceMetrics:
''',
    )
    replace_once(
        "trade_rl/evaluation/metrics.py",
        '''    cost = _require_non_negative(total_cost, field="total_cost")
    if not math.isfinite(funding_pnl):
        raise ValueError("funding_pnl must be finite")
    if n_trades < 0:
        raise ValueError("n_trades must be non-negative")
''',
        '''    cost = _require_non_negative(total_cost, field="total_cost")
    borrow = _require_non_negative(borrow_cost, field="borrow_cost")
    if not math.isfinite(funding_pnl):
        raise ValueError("funding_pnl must be finite")
    for field_name, value in (
        ("n_trades", n_trades),
        ("rebalance_events", rebalance_events),
        ("termination_count", termination_count),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{field_name} must be a non-negative integer")
''',
    )
    replace_once(
        "trade_rl/evaluation/metrics.py",
        '''        total_cost=cost,
        funding_pnl=funding_pnl,
        n_trades=n_trades,
        n_periods=len(values),
''',
        '''        total_cost=cost,
        funding_pnl=funding_pnl,
        borrow_cost=borrow,
        n_trades=n_trades,
        rebalance_events=rebalance_events,
        termination_count=termination_count,
        n_periods=len(values),
''',
    )

    replace_once(
        "trade_rl/evaluation/walk_forward/stitching.py",
        "from dataclasses import dataclass\n",
        "from dataclasses import dataclass, field\n",
    )
    replace_once(
        "trade_rl/evaluation/walk_forward/stitching.py",
        "from trade_rl.evaluation.series import ReturnSeries\n",
        "from trade_rl.evaluation.evidence import ExecutionDiagnostics\nfrom trade_rl.evaluation.series import ReturnSeries\n",
    )
    replace_once(
        "trade_rl/evaluation/walk_forward/stitching.py",
        '''    returns: ReturnSeries
    opening_state_digest: str | None = None
''',
        '''    returns: ReturnSeries
    diagnostics: ExecutionDiagnostics = field(default_factory=ExecutionDiagnostics)
    opening_state_digest: str | None = None
''',
    )
    replace_once(
        "trade_rl/evaluation/walk_forward/stitching.py",
        '''    mode: StitchMode
    gaps: tuple[tuple[int, int], ...]
''',
        '''    mode: StitchMode
    gaps: tuple[tuple[int, int], ...]
    diagnostics: ExecutionDiagnostics = field(default_factory=ExecutionDiagnostics)
''',
    )
    replace_once(
        "trade_rl/evaluation/walk_forward/stitching.py",
        '''        mode=mode,
        gaps=tuple(gaps),
    )
''',
        '''        mode=mode,
        gaps=tuple(gaps),
        diagnostics=ExecutionDiagnostics.combine(
            result.diagnostics for result in ordered
        ),
    )
''',
    )

    replace_once(
        "trade_rl/workflows/fold_runner.py",
        "from dataclasses import dataclass\n",
        "from dataclasses import dataclass, field\n",
    )
    replace_once(
        "trade_rl/workflows/fold_runner.py",
        "from trade_rl.evaluation.series import ReturnSeries\n",
        "from trade_rl.evaluation.evidence import ExecutionDiagnostics\nfrom trade_rl.evaluation.series import ReturnSeries\n",
    )
    replace_once(
        "trade_rl/workflows/fold_runner.py",
        '''    returns: ReturnSeries
    evaluation_digest: str
''',
        '''    returns: ReturnSeries
    evaluation_digest: str
    diagnostics: ExecutionDiagnostics = field(default_factory=ExecutionDiagnostics)
''',
    )
    replace_once(
        "trade_rl/workflows/fold_runner.py",
        '''            stop=fold.test.stop,
            returns=evaluation.returns,
        )
''',
        '''            stop=fold.test.stop,
            returns=evaluation.returns,
            diagnostics=evaluation.diagnostics,
        )
''',
    )

    replace_once(
        "trade_rl/workflows/walk_forward_evaluation.py",
        "from dataclasses import replace\n",
        "from dataclasses import dataclass, replace\n",
    )
    replace_once(
        "trade_rl/workflows/walk_forward_evaluation.py",
        "from trade_rl.evaluation.series import ReturnKind, ReturnSeries\n",
        "from trade_rl.evaluation.evidence import ExecutionDiagnostics\nfrom trade_rl.evaluation.series import ReturnKind, ReturnSeries\n",
    )
    marker = '''def evaluate_range(
    *,
    dataset: MarketDataset,
'''
    if marker not in (ROOT / "trade_rl/workflows/walk_forward_evaluation.py").read_text(encoding="utf-8"):
        raise RuntimeError("evaluate_range marker is missing")
    replace_once(
        "trade_rl/workflows/walk_forward_evaluation.py",
        marker,
        '''@dataclass(frozen=True, slots=True)
class RangeEvaluation:
    returns: ReturnSeries
    diagnostics: ExecutionDiagnostics


def evaluate_range_evidence(
    *,
    dataset: MarketDataset,
''',
    )
    replace_once(
        "trade_rl/workflows/walk_forward_evaluation.py",
        ''') -> ReturnSeries:
    """Evaluate exactly one half-open range without exposing sealed data early."""
''',
        ''') -> RangeEvaluation:
    """Evaluate one range and retain execution and economic evidence."""
''',
    )
    replace_once(
        "trade_rl/workflows/walk_forward_evaluation.py",
        '''        values = tuple(
            float(value)
            for value in (
                env.shadow.returns_history if baseline else env.hybrid.returns_history
            )
        )
    finally:
        env.close()
''',
        '''        book = env.shadow if baseline else env.hybrid
        values = tuple(float(value) for value in book.returns_history)
        termination_reasons = (
            ()
            if book.termination_reason is None
            else (str(book.termination_reason.value),)
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
''',
    )
    replace_once(
        "trade_rl/workflows/walk_forward_evaluation.py",
        '''    return ReturnSeries(
        values=values,
        kind=ReturnKind.BASE_BAR,
        periods_per_year=dataset.periods_per_year,
    )
''',
        '''    return RangeEvaluation(
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
''',
    )
    replace_once(
        "trade_rl/workflows/walk_forward_evaluation.py",
        '    "evaluate_range",\n',
        '    "evaluate_range",\n    "evaluate_range_evidence",\n    "RangeEvaluation",\n',
    )

    replace_once(
        "trade_rl/workflows/market_walk_forward.py",
        '''    evaluate_range,
    minimum_environment_start,
''',
        '''    evaluate_range_evidence,
    minimum_environment_start,
''',
    )
    replace_once(
        "trade_rl/workflows/market_walk_forward.py",
        '''                returns = evaluate_range(
                    dataset=self.dataset,
                    evaluation_range=request.checkpoint_validation,
                    run=run,
                    normalizer=normalizer,
                    model=model,
                    baseline=False,
                )
                score = sum(math.log1p(value) for value in returns.values)
''',
        '''                evidence = evaluate_range_evidence(
                    dataset=self.dataset,
                    evaluation_range=request.checkpoint_validation,
                    run=run,
                    normalizer=normalizer,
                    model=model,
                    baseline=False,
                )
                score = sum(math.log1p(value) for value in evidence.returns.values)
''',
    )
    replace_once(
        "trade_rl/workflows/market_walk_forward.py",
        '''        returns = evaluate_range(
            dataset=self.dataset,
            evaluation_range=request.evaluation_range,
            run=run,
            normalizer=normalizer,
            model=model,
            baseline=baseline,
        )
''',
        '''        evidence = evaluate_range_evidence(
            dataset=self.dataset,
            evaluation_range=request.evaluation_range,
            run=run,
            normalizer=normalizer,
            model=model,
            baseline=baseline,
        )
''',
    )
    replace_once(
        "trade_rl/workflows/market_walk_forward.py",
        '''        score = sum(math.log1p(value) for value in returns.values)
        digest = content_digest(
''',
        '''        score = sum(math.log1p(value) for value in evidence.returns.values)
        digest = content_digest(
''',
    )
    replace_once(
        "trade_rl/workflows/market_walk_forward.py",
        '''                "returns": returns.values,
                "score": score,
                "schema_version": "market_candidate_evaluation_v1",
''',
        '''                "diagnostics": evidence.diagnostics.digest_payload(),
                "returns": evidence.returns.values,
                "score": score,
                "schema_version": "market_candidate_evaluation_v2",
''',
    )
    replace_once(
        "trade_rl/workflows/market_walk_forward.py",
        '''            score=score,
            returns=returns,
            evaluation_digest=digest,
        )
''',
        '''            score=score,
            returns=evidence.returns,
            evaluation_digest=digest,
            diagnostics=evidence.diagnostics,
        )
''',
    )
    replace_once(
        "trade_rl/workflows/market_walk_forward.py",
        '''        "baseline_returns": result.baseline_oos.returns.values,
''',
        '''        "baseline_diagnostics": result.baseline_oos.diagnostics.digest_payload(),
        "baseline_returns": result.baseline_oos.returns.values,
''',
    )
    replace_once(
        "trade_rl/workflows/market_walk_forward.py",
        '''        "selected_returns": result.selected_oos.returns.values,
''',
        '''        "selected_diagnostics": result.selected_oos.diagnostics.digest_payload(),
        "selected_returns": result.selected_oos.returns.values,
''',
    )

    replace_once(
        "trade_rl/workflows/walk_forward.py",
        '''        metrics=evaluate_performance(stitched.returns),
''',
        '''        metrics=evaluate_performance(
            stitched.returns,
            turnover_total=stitched.diagnostics.turnover_total,
            total_cost=stitched.diagnostics.total_cost,
            funding_pnl=stitched.diagnostics.funding_pnl,
            borrow_cost=stitched.diagnostics.borrow_cost,
            n_trades=stitched.diagnostics.n_trades,
            rebalance_events=stitched.diagnostics.rebalance_events,
            termination_count=stitched.diagnostics.termination_count,
        ),
''',
    )
    replace_once(
        "trade_rl/workflows/walk_forward.py",
        '''            "schema_version": "walk_forward_execution_v1",
            "stitch_mode": config.stitch_mode.value,
''',
        '''            "baseline_diagnostics": baseline_stitched.diagnostics.digest_payload(),
            "schema_version": "walk_forward_execution_v2",
            "selected_diagnostics": selected_stitched.diagnostics.digest_payload(),
            "stitch_mode": config.stitch_mode.value,
''',
    )
    replace_once(
        "trade_rl/workflows/walk_forward.py",
        '''        selected_metrics=evaluate_performance(selected_stitched.returns),
        baseline_metrics=evaluate_performance(baseline_stitched.returns),
''',
        '''        selected_metrics=evaluate_performance(
            selected_stitched.returns,
            turnover_total=selected_stitched.diagnostics.turnover_total,
            total_cost=selected_stitched.diagnostics.total_cost,
            funding_pnl=selected_stitched.diagnostics.funding_pnl,
            borrow_cost=selected_stitched.diagnostics.borrow_cost,
            n_trades=selected_stitched.diagnostics.n_trades,
            rebalance_events=selected_stitched.diagnostics.rebalance_events,
            termination_count=selected_stitched.diagnostics.termination_count,
        ),
        baseline_metrics=evaluate_performance(
            baseline_stitched.returns,
            turnover_total=baseline_stitched.diagnostics.turnover_total,
            total_cost=baseline_stitched.diagnostics.total_cost,
            funding_pnl=baseline_stitched.diagnostics.funding_pnl,
            borrow_cost=baseline_stitched.diagnostics.borrow_cost,
            n_trades=baseline_stitched.diagnostics.n_trades,
            rebalance_events=baseline_stitched.diagnostics.rebalance_events,
            termination_count=baseline_stitched.diagnostics.termination_count,
        ),
''',
    )


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"tests", "implementation"}:
        raise SystemExit("usage: evaluation_evidence.py tests|implementation")
    if sys.argv[1] == "tests":
        write_tests()
    else:
        apply_implementation()


if __name__ == "__main__":
    main()
