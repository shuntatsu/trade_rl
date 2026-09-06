"""Frozen U2 Development Selection metric and bootstrap contracts."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from statistics import fmean, median
from typing import Final

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.workflows import universal_trade_rl_u2_contract as u2_contract
from trade_rl.workflows.universal_trade_rl_u2_contract import U2_TRAINING_SEEDS
from trade_rl.workflows.universal_trade_rl_u2_replay import (
    UniversalTradeRLU2ReplayEvidence,
    UniversalTradeRLU2ReplayVariant,
)

U2_SELECTION_LEAF_METRICS_SCHEMA: Final = (
    "universal_trade_rl_u2_selection_leaf_metrics_v1"
)
U2_SELECTION_SYMBOL_METRICS_SCHEMA: Final = (
    "universal_trade_rl_u2_selection_symbol_metrics_v1"
)
U2_SELECTION_SUMMARY_SCHEMA: Final = "universal_trade_rl_u2_selection_summary_v1"
U2_DEVELOPMENT_PANEL_SCHEMA: Final = "universal_trade_rl_u2_development_panel_v1"
U2_DEVELOPMENT_BOOTSTRAP_SCHEMA: Final = (
    "universal_trade_rl_u2_development_segmented_bootstrap_v1"
)
U2_DEVELOPMENT_WINDOWS: Final = (
    "development_future_1",
    "development_future_2",
)
_U2_BOOTSTRAP_RESAMPLES: Final = 2_000
_U2_BOOTSTRAP_SEED: Final = 0
_U2_BOOTSTRAP_CONFIDENCE_LEVEL: Final = 0.95
_U2_BOOTSTRAP_QUANTILE_METHOD: Final = "linear"
_MEANINGFUL_EXECUTION_TURNOVER_TOLERANCE: Final = 1e-6


def _finite(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{field} must be numeric")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{field} must be finite")
    return resolved


def _non_negative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _positive_wealth(log_growth: float, *, field: str) -> float:
    try:
        wealth = math.exp(log_growth)
    except OverflowError as error:
        raise ValueError(f"{field} wealth overflowed") from error
    if not math.isfinite(wealth) or wealth <= 0.0:
        raise ValueError(f"{field} wealth must be finite and positive")
    return wealth


@dataclass(frozen=True, slots=True)
class UniversalTradeRLU2SelectionLeafMetrics:
    """One frozen Selection leaf derived from one candidate replay evidence object."""

    training_seed: int
    cell: str
    concrete_symbol: str
    tile_identity: str
    replay_evidence_digest: str
    leaf_net_log_growth: float
    leaf_gross_log_growth: float
    turnover_per_day: float
    meaningful_execution: bool
    hard_risk_violation_count: int
    unexplained_execution_rejection_count: int
    schema_version: str = U2_SELECTION_LEAF_METRICS_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != U2_SELECTION_LEAF_METRICS_SCHEMA:
            raise ValueError("unsupported U2 Selection leaf metrics schema")
        if (
            isinstance(self.training_seed, bool)
            or not isinstance(self.training_seed, int)
            or self.training_seed not in U2_TRAINING_SEEDS
        ):
            raise ValueError("U2 Selection leaf training seed is not preregistered")
        for field_name, value in (
            ("cell", self.cell),
            ("concrete_symbol", self.concrete_symbol),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"U2 Selection leaf {field_name} must be non-empty")
        require_sha256(self.tile_identity, field="U2 Selection leaf tile identity")
        require_sha256(
            self.replay_evidence_digest,
            field="U2 Selection leaf replay evidence digest",
        )
        object.__setattr__(
            self,
            "leaf_net_log_growth",
            _finite(self.leaf_net_log_growth, field="U2 Selection leaf net log growth"),
        )
        object.__setattr__(
            self,
            "leaf_gross_log_growth",
            _finite(
                self.leaf_gross_log_growth,
                field="U2 Selection leaf gross log growth",
            ),
        )
        turnover_per_day = _finite(
            self.turnover_per_day,
            field="U2 Selection leaf turnover per day",
        )
        if turnover_per_day < 0.0:
            raise ValueError("U2 Selection leaf turnover per day cannot be negative")
        object.__setattr__(self, "turnover_per_day", turnover_per_day)
        if not isinstance(self.meaningful_execution, bool):
            raise TypeError("U2 Selection leaf meaningful execution must be boolean")
        object.__setattr__(
            self,
            "hard_risk_violation_count",
            _non_negative_int(
                self.hard_risk_violation_count,
                field="U2 Selection leaf hard-risk violation count",
            ),
        )
        object.__setattr__(
            self,
            "unexplained_execution_rejection_count",
            _non_negative_int(
                self.unexplained_execution_rejection_count,
                field="U2 Selection leaf unexplained execution rejection count",
            ),
        )
        _positive_wealth(self.leaf_net_log_growth, field="U2 Selection leaf net")
        _positive_wealth(self.leaf_gross_log_growth, field="U2 Selection leaf gross")

        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest:
            require_sha256(self.digest, field="U2 Selection leaf metrics digest")
            if self.digest != expected:
                raise ValueError("U2 Selection leaf metrics digest mismatch")
        object.__setattr__(self, "digest", expected)

    @property
    def identity(self) -> tuple[int, str, str, str]:
        return (
            self.training_seed,
            self.cell,
            self.concrete_symbol,
            self.tile_identity,
        )

    @property
    def leaf_net_wealth(self) -> float:
        return _positive_wealth(self.leaf_net_log_growth, field="U2 Selection leaf net")

    @property
    def leaf_gross_wealth(self) -> float:
        return _positive_wealth(
            self.leaf_gross_log_growth,
            field="U2 Selection leaf gross",
        )

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "training_seed": self.training_seed,
            "cell": self.cell,
            "concrete_symbol": self.concrete_symbol,
            "tile_identity": self.tile_identity,
            "replay_evidence_digest": self.replay_evidence_digest,
            "leaf_net_log_growth": self.leaf_net_log_growth,
            "leaf_gross_log_growth": self.leaf_gross_log_growth,
            "turnover_per_day": self.turnover_per_day,
            "meaningful_execution": self.meaningful_execution,
            "hard_risk_violation_count": self.hard_risk_violation_count,
            "unexplained_execution_rejection_count": (
                self.unexplained_execution_rejection_count
            ),
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


@dataclass(frozen=True, slots=True)
class UniversalTradeRLU2SelectionSymbolMetrics:
    """Equal-vote per-symbol aggregation of Selection leaves."""

    concrete_symbol: str
    leaf_digests: tuple[str, ...]
    symbol_net_log_growth: float
    symbol_gross_log_growth: float
    meaningful_execution: bool
    hard_risk_violation_count: int
    unexplained_execution_rejection_count: int
    schema_version: str = U2_SELECTION_SYMBOL_METRICS_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != U2_SELECTION_SYMBOL_METRICS_SCHEMA:
            raise ValueError("unsupported U2 Selection symbol metrics schema")
        if not isinstance(self.concrete_symbol, str) or not self.concrete_symbol:
            raise ValueError("U2 Selection symbol must be non-empty")
        leaf_digests = tuple(self.leaf_digests)
        if not leaf_digests:
            raise ValueError("U2 Selection symbol metrics require at least one leaf")
        for digest in leaf_digests:
            require_sha256(digest, field="U2 Selection symbol leaf digest")
        if len(set(leaf_digests)) != len(leaf_digests):
            raise ValueError("U2 Selection symbol metrics contain duplicate leaves")
        object.__setattr__(self, "leaf_digests", leaf_digests)
        object.__setattr__(
            self,
            "symbol_net_log_growth",
            _finite(
                self.symbol_net_log_growth,
                field="U2 Selection symbol net log growth",
            ),
        )
        object.__setattr__(
            self,
            "symbol_gross_log_growth",
            _finite(
                self.symbol_gross_log_growth,
                field="U2 Selection symbol gross log growth",
            ),
        )
        if not isinstance(self.meaningful_execution, bool):
            raise TypeError("U2 Selection symbol meaningful execution must be boolean")
        object.__setattr__(
            self,
            "hard_risk_violation_count",
            _non_negative_int(
                self.hard_risk_violation_count,
                field="U2 Selection symbol hard-risk violation count",
            ),
        )
        object.__setattr__(
            self,
            "unexplained_execution_rejection_count",
            _non_negative_int(
                self.unexplained_execution_rejection_count,
                field="U2 Selection symbol unexplained execution rejection count",
            ),
        )
        _positive_wealth(self.symbol_net_log_growth, field="U2 Selection symbol net")
        _positive_wealth(
            self.symbol_gross_log_growth, field="U2 Selection symbol gross"
        )

        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest:
            require_sha256(self.digest, field="U2 Selection symbol metrics digest")
            if self.digest != expected:
                raise ValueError("U2 Selection symbol metrics digest mismatch")
        object.__setattr__(self, "digest", expected)

    @property
    def symbol_net_wealth(self) -> float:
        return _positive_wealth(
            self.symbol_net_log_growth,
            field="U2 Selection symbol net",
        )

    @property
    def symbol_gross_wealth(self) -> float:
        return _positive_wealth(
            self.symbol_gross_log_growth,
            field="U2 Selection symbol gross",
        )

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "concrete_symbol": self.concrete_symbol,
            "leaf_digests": self.leaf_digests,
            "symbol_net_log_growth": self.symbol_net_log_growth,
            "symbol_gross_log_growth": self.symbol_gross_log_growth,
            "meaningful_execution": self.meaningful_execution,
            "hard_risk_violation_count": self.hard_risk_violation_count,
            "unexplained_execution_rejection_count": (
                self.unexplained_execution_rejection_count
            ),
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


@dataclass(frozen=True, slots=True)
class UniversalTradeRLU2SelectionMetricSummary:
    """Frozen Selection metrics for one cell/scope grouping."""

    cell: str
    training_seed: int
    leaf_digests: tuple[str, ...]
    symbol_metrics: tuple[UniversalTradeRLU2SelectionSymbolMetrics, ...]
    leaf_count: int
    symbol_count: int
    symbol_balanced_net_log_growth: float
    symbol_balanced_gross_log_growth: float
    symbol_balanced_net_wealth: float
    symbol_balanced_gross_wealth: float
    median_symbol_net_wealth: float
    minimum_symbol_net_wealth: float
    positive_net_scope_fraction: float
    scope_net_return_cvar10: float
    turnover_per_day_p95: float
    meaningful_execution_symbol_fraction: float
    hard_risk_violation_count: int
    unexplained_execution_rejection_count: int
    positive_gross_log_growth_retention: float | None
    schema_version: str = U2_SELECTION_SUMMARY_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != U2_SELECTION_SUMMARY_SCHEMA:
            raise ValueError("unsupported U2 Selection summary schema")
        if not isinstance(self.cell, str) or not self.cell:
            raise ValueError("U2 Selection summary cell must be non-empty")
        if (
            isinstance(self.training_seed, bool)
            or not isinstance(self.training_seed, int)
            or self.training_seed not in U2_TRAINING_SEEDS
        ):
            raise ValueError("U2 Selection summary training seed is not preregistered")
        leaf_digests = tuple(self.leaf_digests)
        symbols = tuple(self.symbol_metrics)
        if not leaf_digests or not symbols:
            raise ValueError("U2 Selection summary cannot be empty")
        if any(
            not isinstance(row, UniversalTradeRLU2SelectionSymbolMetrics)
            for row in symbols
        ):
            raise TypeError("U2 Selection summary contains invalid symbol metrics")
        symbol_names = tuple(row.concrete_symbol for row in symbols)
        if symbol_names != tuple(sorted(set(symbol_names))):
            raise ValueError(
                "U2 Selection summary symbols must be canonical and unique"
            )
        for digest in leaf_digests:
            require_sha256(digest, field="U2 Selection summary leaf digest")
        if len(set(leaf_digests)) != len(leaf_digests):
            raise ValueError("U2 Selection summary leaf digests must be unique")
        if self.leaf_count != len(leaf_digests):
            raise ValueError("U2 Selection summary leaf count is inconsistent")
        if self.symbol_count != len(symbols):
            raise ValueError("U2 Selection summary symbol count is inconsistent")
        for field_name in (
            "symbol_balanced_net_log_growth",
            "symbol_balanced_gross_log_growth",
            "symbol_balanced_net_wealth",
            "symbol_balanced_gross_wealth",
            "median_symbol_net_wealth",
            "minimum_symbol_net_wealth",
            "positive_net_scope_fraction",
            "scope_net_return_cvar10",
            "turnover_per_day_p95",
            "meaningful_execution_symbol_fraction",
        ):
            object.__setattr__(
                self,
                field_name,
                _finite(getattr(self, field_name), field=f"U2 Selection {field_name}"),
            )
        for field_name in (
            "symbol_balanced_net_wealth",
            "symbol_balanced_gross_wealth",
            "median_symbol_net_wealth",
            "minimum_symbol_net_wealth",
        ):
            if getattr(self, field_name) <= 0.0:
                raise ValueError(f"U2 Selection {field_name} must be positive")
        for field_name in (
            "positive_net_scope_fraction",
            "meaningful_execution_symbol_fraction",
        ):
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"U2 Selection {field_name} must be within [0, 1]")
        if self.turnover_per_day_p95 < 0.0:
            raise ValueError("U2 Selection turnover p95 cannot be negative")
        object.__setattr__(
            self,
            "hard_risk_violation_count",
            _non_negative_int(
                self.hard_risk_violation_count,
                field="U2 Selection summary hard-risk violation count",
            ),
        )
        object.__setattr__(
            self,
            "unexplained_execution_rejection_count",
            _non_negative_int(
                self.unexplained_execution_rejection_count,
                field="U2 Selection summary unexplained execution rejection count",
            ),
        )

        symbol_leaf_digests = tuple(
            digest for row in symbols for digest in row.leaf_digests
        )
        if len(symbol_leaf_digests) != len(set(symbol_leaf_digests)) or set(
            symbol_leaf_digests
        ) != set(leaf_digests):
            raise ValueError(
                "U2 Selection summary symbol leaf-digest closure is inconsistent"
            )

        expected_net_log_growth = float(
            fmean(row.symbol_net_log_growth for row in symbols)
        )
        expected_gross_log_growth = float(
            fmean(row.symbol_gross_log_growth for row in symbols)
        )
        expected_net_wealth = _positive_wealth(
            expected_net_log_growth,
            field="U2 Selection expected balanced net",
        )
        expected_gross_wealth = _positive_wealth(
            expected_gross_log_growth,
            field="U2 Selection expected balanced gross",
        )
        symbol_net_wealth = tuple(row.symbol_net_wealth for row in symbols)
        expected_median_net_wealth = float(median(symbol_net_wealth))
        expected_minimum_net_wealth = float(min(symbol_net_wealth))
        expected_meaningful_fraction = float(
            fmean(row.meaningful_execution for row in symbols)
        )
        expected_hard_risk_count = sum(row.hard_risk_violation_count for row in symbols)
        expected_rejection_count = sum(
            row.unexplained_execution_rejection_count for row in symbols
        )
        if self.hard_risk_violation_count != expected_hard_risk_count:
            raise ValueError(
                "U2 Selection summary hard-risk count is inconsistent with symbol metrics"
            )
        if self.unexplained_execution_rejection_count != expected_rejection_count:
            raise ValueError(
                "U2 Selection summary rejection count is inconsistent with symbol metrics"
            )
        for field_name, observed, expected_value in (
            (
                "symbol_balanced_net_log_growth",
                self.symbol_balanced_net_log_growth,
                expected_net_log_growth,
            ),
            (
                "symbol_balanced_gross_log_growth",
                self.symbol_balanced_gross_log_growth,
                expected_gross_log_growth,
            ),
            (
                "symbol_balanced_net_wealth",
                self.symbol_balanced_net_wealth,
                expected_net_wealth,
            ),
            (
                "symbol_balanced_gross_wealth",
                self.symbol_balanced_gross_wealth,
                expected_gross_wealth,
            ),
            (
                "median_symbol_net_wealth",
                self.median_symbol_net_wealth,
                expected_median_net_wealth,
            ),
            (
                "minimum_symbol_net_wealth",
                self.minimum_symbol_net_wealth,
                expected_minimum_net_wealth,
            ),
            (
                "meaningful_execution_symbol_fraction",
                self.meaningful_execution_symbol_fraction,
                expected_meaningful_fraction,
            ),
        ):
            if not math.isclose(
                observed,
                expected_value,
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise ValueError(
                    f"U2 Selection summary {field_name} is inconsistent with symbol metrics"
                )

        if self.positive_gross_log_growth_retention is not None:
            retention = _finite(
                self.positive_gross_log_growth_retention,
                field="U2 Selection positive gross retention",
            )
            if self.symbol_balanced_gross_log_growth <= 0.0:
                raise ValueError(
                    "U2 Selection retention is undefined for non-positive gross growth"
                )
            expected_retention = expected_net_log_growth / expected_gross_log_growth
            if not math.isclose(
                retention,
                expected_retention,
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise ValueError(
                    "U2 Selection summary retention is inconsistent with symbol metrics"
                )
            object.__setattr__(
                self,
                "positive_gross_log_growth_retention",
                retention,
            )
        elif self.symbol_balanced_gross_log_growth > 0.0:
            raise ValueError(
                "U2 Selection positive gross growth requires retention evidence"
            )

        object.__setattr__(self, "leaf_digests", leaf_digests)
        object.__setattr__(self, "symbol_metrics", symbols)
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest:
            require_sha256(self.digest, field="U2 Selection summary digest")
            if self.digest != expected:
                raise ValueError("U2 Selection summary digest mismatch")
        object.__setattr__(self, "digest", expected)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "cell": self.cell,
            "training_seed": self.training_seed,
            "leaf_digests": self.leaf_digests,
            "symbol_metric_digests": tuple(row.digest for row in self.symbol_metrics),
            "leaf_count": self.leaf_count,
            "symbol_count": self.symbol_count,
            "symbol_balanced_net_log_growth": self.symbol_balanced_net_log_growth,
            "symbol_balanced_gross_log_growth": self.symbol_balanced_gross_log_growth,
            "symbol_balanced_net_wealth": self.symbol_balanced_net_wealth,
            "symbol_balanced_gross_wealth": self.symbol_balanced_gross_wealth,
            "median_symbol_net_wealth": self.median_symbol_net_wealth,
            "minimum_symbol_net_wealth": self.minimum_symbol_net_wealth,
            "positive_net_scope_fraction": self.positive_net_scope_fraction,
            "scope_net_return_cvar10": self.scope_net_return_cvar10,
            "turnover_per_day_p95": self.turnover_per_day_p95,
            "meaningful_execution_symbol_fraction": (
                self.meaningful_execution_symbol_fraction
            ),
            "hard_risk_violation_count": self.hard_risk_violation_count,
            "unexplained_execution_rejection_count": (
                self.unexplained_execution_rejection_count
            ),
            "positive_gross_log_growth_retention": (
                self.positive_gross_log_growth_retention
            ),
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


def build_universal_trade_rl_u2_selection_leaf_metrics(
    *,
    training_seed: int,
    replay_evidence: UniversalTradeRLU2ReplayEvidence,
) -> UniversalTradeRLU2SelectionLeafMetrics:
    """Derive one Selection leaf from the maintained same-path candidate replay."""

    if (
        isinstance(training_seed, bool)
        or not isinstance(training_seed, int)
        or training_seed not in U2_TRAINING_SEEDS
    ):
        raise ValueError("U2 Selection training seed is outside the fixed closure")
    if not isinstance(replay_evidence, UniversalTradeRLU2ReplayEvidence):
        raise TypeError("U2 Selection leaf requires replay evidence")
    if (
        replay_evidence.policy_variant
        != UniversalTradeRLU2ReplayVariant.CANDIDATE.value
    ):
        raise ValueError("U2 Selection leaf requires candidate policy replay evidence")
    if replay_evidence.observed_decision_count <= 0:
        raise ValueError("U2 Selection leaf requires at least one replay decision")

    leaf_net_log_growth = math.fsum(
        math.log1p(value) for value in replay_evidence.net_simple_returns
    )
    leaf_gross_log_growth = math.fsum(
        math.log1p(value) for value in replay_evidence.gross_simple_returns
    )
    turnover_per_day = replay_evidence.turnover_total / (
        replay_evidence.observed_decision_count * 0.25 / 24.0
    )
    meaningful_execution = bool(
        replay_evidence.executed_change_count > 0
        or replay_evidence.turnover_total > _MEANINGFUL_EXECUTION_TURNOVER_TOLERANCE
    )
    return UniversalTradeRLU2SelectionLeafMetrics(
        training_seed=training_seed,
        cell=replay_evidence.cell,
        concrete_symbol=replay_evidence.concrete_symbol,
        tile_identity=replay_evidence.scope_digest,
        replay_evidence_digest=replay_evidence.digest,
        leaf_net_log_growth=leaf_net_log_growth,
        leaf_gross_log_growth=leaf_gross_log_growth,
        turnover_per_day=turnover_per_day,
        meaningful_execution=meaningful_execution,
        hard_risk_violation_count=replay_evidence.hard_risk_violation_count,
        unexplained_execution_rejection_count=(
            replay_evidence.execution_rejection_count
        ),
    )


def summarize_universal_trade_rl_u2_selection_metrics(
    *,
    leaves: tuple[UniversalTradeRLU2SelectionLeafMetrics, ...],
) -> UniversalTradeRLU2SelectionMetricSummary:
    """Execute the preregistered symbol-balanced Selection metric formulas."""

    resolved = tuple(leaves)
    if not resolved:
        raise ValueError("U2 Selection summary requires at least one leaf")
    if any(
        not isinstance(leaf, UniversalTradeRLU2SelectionLeafMetrics)
        for leaf in resolved
    ):
        raise TypeError("U2 Selection summary contains an invalid leaf")
    identities = tuple(leaf.identity for leaf in resolved)
    if len(set(identities)) != len(identities):
        raise ValueError("U2 Selection summary contains duplicate leaf identity")
    cells = {leaf.cell for leaf in resolved}
    if len(cells) != 1:
        raise ValueError("U2 Selection summary must contain one cell grouping")
    cell = next(iter(cells))
    training_seeds = {leaf.training_seed for leaf in resolved}
    if len(training_seeds) != 1:
        raise ValueError("U2 Selection summary must contain one training seed")
    training_seed = next(iter(training_seeds))

    ordered = tuple(sorted(resolved, key=lambda leaf: leaf.identity))
    by_symbol: dict[str, list[UniversalTradeRLU2SelectionLeafMetrics]] = defaultdict(
        list
    )
    for leaf in ordered:
        by_symbol[leaf.concrete_symbol].append(leaf)

    symbol_rows: list[UniversalTradeRLU2SelectionSymbolMetrics] = []
    for concrete_symbol in sorted(by_symbol):
        symbol_leaves = tuple(by_symbol[concrete_symbol])
        symbol_rows.append(
            UniversalTradeRLU2SelectionSymbolMetrics(
                concrete_symbol=concrete_symbol,
                leaf_digests=tuple(leaf.digest for leaf in symbol_leaves),
                symbol_net_log_growth=math.fsum(
                    leaf.leaf_net_log_growth for leaf in symbol_leaves
                ),
                symbol_gross_log_growth=math.fsum(
                    leaf.leaf_gross_log_growth for leaf in symbol_leaves
                ),
                meaningful_execution=any(
                    leaf.meaningful_execution for leaf in symbol_leaves
                ),
                hard_risk_violation_count=sum(
                    leaf.hard_risk_violation_count for leaf in symbol_leaves
                ),
                unexplained_execution_rejection_count=sum(
                    leaf.unexplained_execution_rejection_count for leaf in symbol_leaves
                ),
            )
        )
    symbols = tuple(symbol_rows)

    balanced_net_log_growth = float(fmean(row.symbol_net_log_growth for row in symbols))
    balanced_gross_log_growth = float(
        fmean(row.symbol_gross_log_growth for row in symbols)
    )
    symbol_net_wealth = tuple(row.symbol_net_wealth for row in symbols)
    positive_fraction = float(fmean(leaf.leaf_net_log_growth > 0.0 for leaf in ordered))
    cvar_count = max(1, math.ceil(0.10 * len(ordered)))
    cvar = float(
        fmean(sorted(leaf.leaf_net_log_growth for leaf in ordered)[:cvar_count])
    )
    turnover_p95 = float(
        np.quantile(
            [leaf.turnover_per_day for leaf in ordered],
            0.95,
            method="linear",
        )
    )
    meaningful_symbol_fraction = float(
        fmean(row.meaningful_execution for row in symbols)
    )
    retention = (
        balanced_net_log_growth / balanced_gross_log_growth
        if balanced_gross_log_growth > 0.0
        else None
    )

    return UniversalTradeRLU2SelectionMetricSummary(
        cell=cell,
        training_seed=training_seed,
        leaf_digests=tuple(leaf.digest for leaf in ordered),
        symbol_metrics=symbols,
        leaf_count=len(ordered),
        symbol_count=len(symbols),
        symbol_balanced_net_log_growth=balanced_net_log_growth,
        symbol_balanced_gross_log_growth=balanced_gross_log_growth,
        symbol_balanced_net_wealth=_positive_wealth(
            balanced_net_log_growth,
            field="U2 Selection balanced net",
        ),
        symbol_balanced_gross_wealth=_positive_wealth(
            balanced_gross_log_growth,
            field="U2 Selection balanced gross",
        ),
        median_symbol_net_wealth=float(median(symbol_net_wealth)),
        minimum_symbol_net_wealth=float(min(symbol_net_wealth)),
        positive_net_scope_fraction=positive_fraction,
        scope_net_return_cvar10=cvar,
        turnover_per_day_p95=turnover_p95,
        meaningful_execution_symbol_fraction=meaningful_symbol_fraction,
        hard_risk_violation_count=sum(row.hard_risk_violation_count for row in symbols),
        unexplained_execution_rejection_count=sum(
            row.unexplained_execution_rejection_count for row in symbols
        ),
        positive_gross_log_growth_retention=retention,
    )


U2_PRIMARY_SELECTION_CELL_GATE_SCHEMA: Final = (
    "universal_trade_rl_u2_primary_selection_cell_gate_v1"
)
_U2_PRIMARY_SELECTION_MANDATORY_CELLS: Final = ("B", "C1", "C2", "D1", "D2")


def _u2_primary_selection_rejection_reasons(
    *,
    summary: UniversalTradeRLU2SelectionMetricSummary,
    thresholds: dict[str, object],
) -> tuple[str, ...]:
    gross_min = _finite(
        thresholds["symbol_balanced_gross_wealth_min_exclusive"],
        field="U2 primary gate balanced gross wealth threshold",
    )
    net_min = _finite(
        thresholds["symbol_balanced_net_wealth_min_exclusive"],
        field="U2 primary gate balanced net wealth threshold",
    )
    median_min = _finite(
        thresholds["median_symbol_net_wealth_min_inclusive"],
        field="U2 primary gate median symbol wealth threshold",
    )
    minimum_min = _finite(
        thresholds["minimum_symbol_net_wealth_min_inclusive"],
        field="U2 primary gate minimum symbol wealth threshold",
    )
    positive_fraction_min = _finite(
        thresholds["positive_net_scope_fraction_min_inclusive"],
        field="U2 primary gate positive fraction threshold",
    )
    cvar_min = _finite(
        thresholds["scope_net_return_cvar10_min_inclusive"],
        field="U2 primary gate CVaR10 threshold",
    )
    turnover_max = _finite(
        thresholds["turnover_p95_per_day_max_inclusive"],
        field="U2 primary gate turnover threshold",
    )
    execution_fraction_required = _finite(
        thresholds["meaningful_execution_symbol_fraction_required"],
        field="U2 primary gate execution fraction threshold",
    )
    hard_risk_required = _non_negative_int(
        thresholds["hard_risk_violation_count_required"],
        field="U2 primary gate hard-risk threshold",
    )
    unexplained_rejection_required = _non_negative_int(
        thresholds["unexplained_execution_reject_count_required"],
        field="U2 primary gate unexplained rejection threshold",
    )
    retention_min = _finite(
        thresholds["positive_gross_log_growth_net_retention_min_inclusive"],
        field="U2 primary gate positive-gross retention threshold",
    )

    reasons: list[str] = []
    if summary.symbol_balanced_gross_wealth <= gross_min:
        reasons.append("symbol_balanced_gross_wealth_not_above_cash")
    if summary.symbol_balanced_net_wealth <= net_min:
        reasons.append("symbol_balanced_net_wealth_not_above_cash")
    if summary.median_symbol_net_wealth < median_min:
        reasons.append("median_symbol_net_wealth_below_cash")
    if summary.minimum_symbol_net_wealth < minimum_min:
        reasons.append("minimum_symbol_net_wealth_below_cash")
    if summary.positive_net_scope_fraction < positive_fraction_min:
        reasons.append("positive_net_scope_fraction_below_0_50")
    if summary.scope_net_return_cvar10 < cvar_min:
        reasons.append("scope_net_return_cvar10_below_minus_0_01")
    if summary.turnover_per_day_p95 > turnover_max:
        reasons.append("turnover_p95_per_day_above_1_0")
    if summary.meaningful_execution_symbol_fraction != execution_fraction_required:
        reasons.append("meaningful_execution_symbol_fraction_not_complete")
    if summary.hard_risk_violation_count != hard_risk_required:
        reasons.append("hard_risk_violation_count_nonzero")
    if summary.unexplained_execution_rejection_count != unexplained_rejection_required:
        reasons.append("unexplained_execution_rejection_count_nonzero")
    if summary.symbol_balanced_gross_log_growth > 0.0 and (
        summary.positive_gross_log_growth_retention is None
        or summary.positive_gross_log_growth_retention < retention_min
    ):
        reasons.append("positive_gross_log_growth_retention_below_0_50")
    return tuple(reasons)


@dataclass(frozen=True, slots=True)
class UniversalTradeRLU2PrimarySelectionCellGateEvidence:
    """Digest-bound primary-seed core-gate evidence for one mandatory U2 cell."""

    summary: UniversalTradeRLU2SelectionMetricSummary
    selection_thresholds_digest: str = field(init=False)
    rejection_reasons: tuple[str, ...] = field(init=False)
    passed: bool = field(init=False)
    schema_version: str = U2_PRIMARY_SELECTION_CELL_GATE_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.summary, UniversalTradeRLU2SelectionMetricSummary):
            raise TypeError("U2 primary Selection gate requires a metric summary")
        if self.summary.training_seed != u2_contract.U2_PRIMARY_CANDIDATE_SEED:
            raise ValueError(
                "U2 primary Selection gate requires primary training seed 0"
            )
        if self.summary.cell not in _U2_PRIMARY_SELECTION_MANDATORY_CELLS:
            raise ValueError(
                "U2 primary Selection gate requires mandatory cell B/C1/C2/D1/D2"
            )
        if self.schema_version != U2_PRIMARY_SELECTION_CELL_GATE_SCHEMA:
            raise ValueError("unsupported U2 primary Selection cell gate schema")

        thresholds = u2_contract._selection_thresholds_payload()
        threshold_digest = content_digest(thresholds)
        reasons = _u2_primary_selection_rejection_reasons(
            summary=self.summary,
            thresholds=thresholds,
        )
        object.__setattr__(self, "selection_thresholds_digest", threshold_digest)
        object.__setattr__(self, "rejection_reasons", reasons)
        object.__setattr__(self, "passed", not reasons)

        expected_digest = content_digest(self.to_payload(include_digest=False))
        if self.digest:
            require_sha256(self.digest, field="U2 primary Selection cell gate digest")
            if self.digest != expected_digest:
                raise ValueError("U2 primary Selection cell gate digest mismatch")
        object.__setattr__(self, "digest", expected_digest)

    @property
    def cell(self) -> str:
        return self.summary.cell

    @property
    def training_seed(self) -> int:
        return self.summary.training_seed

    @property
    def summary_digest(self) -> str:
        return self.summary.digest

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "cell": self.cell,
            "training_seed": self.training_seed,
            "summary_digest": self.summary_digest,
            "selection_thresholds_digest": self.selection_thresholds_digest,
            "rejection_reasons": self.rejection_reasons,
            "passed": self.passed,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


def evaluate_universal_trade_rl_u2_primary_cell_gate(
    *,
    summary: UniversalTradeRLU2SelectionMetricSummary,
) -> UniversalTradeRLU2PrimarySelectionCellGateEvidence:
    """Evaluate the preregistered 11-condition primary core gate for one cell."""

    return UniversalTradeRLU2PrimarySelectionCellGateEvidence(summary=summary)


@dataclass(frozen=True, slots=True)
class UniversalTradeRLU2PairedExcessPoint:
    """One Development leaf after same-scope candidate/cash pairing."""

    training_seed: int
    source_window: str
    concrete_symbol: str
    decision_timestamp_ns: int
    candidate_minus_cash_net_log_excess: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.training_seed, bool)
            or self.training_seed not in U2_TRAINING_SEEDS
        ):
            raise ValueError(
                "U2 Development panel seed is outside the fixed seed closure"
            )
        if self.source_window not in U2_DEVELOPMENT_WINDOWS:
            raise ValueError("U2 Development panel source window is not D1/D2")
        if not isinstance(self.concrete_symbol, str) or not self.concrete_symbol:
            raise ValueError("U2 Development panel symbol must be non-empty")
        _non_negative_int(
            self.decision_timestamp_ns,
            field="U2 Development panel decision timestamp",
        )
        object.__setattr__(
            self,
            "candidate_minus_cash_net_log_excess",
            _finite(
                self.candidate_minus_cash_net_log_excess,
                field="U2 Development paired net-log excess",
            ),
        )


@dataclass(frozen=True, slots=True)
class UniversalTradeRLU2ReducedBootstrapSegment:
    """One chronological D1 or D2 series after symbol/seed reduction."""

    source_window: str
    decision_timestamps_ns: tuple[int, ...]
    net_log_excess: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.source_window not in U2_DEVELOPMENT_WINDOWS:
            raise ValueError("U2 bootstrap segment source window is not D1/D2")
        timestamps = tuple(self.decision_timestamps_ns)
        values = tuple(
            _finite(value, field="U2 bootstrap segment net-log excess")
            for value in self.net_log_excess
        )
        if not timestamps or len(timestamps) != len(values):
            raise ValueError("U2 bootstrap segment must be non-empty and aligned")
        if any(
            _non_negative_int(timestamp, field="U2 bootstrap segment timestamp")
            != timestamp
            for timestamp in timestamps
        ):
            raise ValueError("U2 bootstrap segment timestamps are invalid")
        if timestamps != tuple(sorted(set(timestamps))):
            raise ValueError(
                "U2 bootstrap segment timestamps must be sorted and unique"
            )
        object.__setattr__(self, "decision_timestamps_ns", timestamps)
        object.__setattr__(self, "net_log_excess", values)

    @property
    def digest(self) -> str:
        return content_digest(
            {
                "schema_version": U2_DEVELOPMENT_PANEL_SCHEMA,
                "source_window": self.source_window,
                "decision_timestamps_ns": self.decision_timestamps_ns,
                "net_log_excess": self.net_log_excess,
            }
        )


@dataclass(frozen=True, slots=True)
class UniversalTradeRLU2DevelopmentBootstrapResult:
    """Deterministic segmented moving-block bootstrap evidence for D1+D2."""

    segment_digests: tuple[str, ...]
    block_lengths: tuple[int, ...]
    observed_mean: float
    lower_ci: float
    upper_ci: float
    passed: bool
    resamples: int = _U2_BOOTSTRAP_RESAMPLES
    bootstrap_seed: int = _U2_BOOTSTRAP_SEED
    confidence_level: float = _U2_BOOTSTRAP_CONFIDENCE_LEVEL
    quantile_method: str = _U2_BOOTSTRAP_QUANTILE_METHOD
    blocks_may_cross_segment_boundary: bool = False
    schema_version: str = U2_DEVELOPMENT_BOOTSTRAP_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != U2_DEVELOPMENT_BOOTSTRAP_SCHEMA:
            raise ValueError("unsupported U2 Development bootstrap schema")
        if not self.segment_digests or len(self.segment_digests) != len(
            self.block_lengths
        ):
            raise ValueError("U2 Development bootstrap segment evidence is incomplete")
        if any(len(value) != 64 for value in self.segment_digests):
            raise ValueError("U2 Development bootstrap segment digest is invalid")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in self.block_lengths
        ):
            raise ValueError("U2 Development bootstrap block length is invalid")
        for field_name in ("observed_mean", "lower_ci", "upper_ci"):
            object.__setattr__(
                self,
                field_name,
                _finite(getattr(self, field_name), field=f"U2 bootstrap {field_name}"),
            )
        if self.lower_ci > self.upper_ci:
            raise ValueError("U2 Development bootstrap interval is reversed")
        if self.resamples != _U2_BOOTSTRAP_RESAMPLES:
            raise ValueError("U2 Development bootstrap resample count drifted")
        if self.bootstrap_seed != _U2_BOOTSTRAP_SEED:
            raise ValueError("U2 Development bootstrap seed drifted")
        if not math.isclose(
            self.confidence_level,
            _U2_BOOTSTRAP_CONFIDENCE_LEVEL,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError("U2 Development bootstrap confidence level drifted")
        if self.quantile_method != _U2_BOOTSTRAP_QUANTILE_METHOD:
            raise ValueError("U2 Development bootstrap quantile method drifted")
        if self.blocks_may_cross_segment_boundary is not False:
            raise ValueError("U2 Development bootstrap blocks may not cross D1/D2")
        if self.passed is not (self.lower_ci > 0.0):
            raise ValueError("U2 Development bootstrap pass state is inconsistent")

        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("U2 Development bootstrap digest mismatch")
        object.__setattr__(self, "digest", expected)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "segment_digests": self.segment_digests,
            "block_lengths": self.block_lengths,
            "observed_mean": self.observed_mean,
            "lower_ci": self.lower_ci,
            "upper_ci": self.upper_ci,
            "passed": self.passed,
            "resamples": self.resamples,
            "bootstrap_seed": self.bootstrap_seed,
            "confidence_level": self.confidence_level,
            "quantile_method": self.quantile_method,
            "blocks_may_cross_segment_boundary": self.blocks_may_cross_segment_boundary,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


def reduce_universal_trade_rl_u2_development_panel(
    *,
    points: tuple[UniversalTradeRLU2PairedExcessPoint, ...],
    expected_symbols: tuple[str, ...],
) -> tuple[UniversalTradeRLU2ReducedBootstrapSegment, ...]:
    """Reduce D1/D2 leaves as equal-weight symbols then median training seed."""

    resolved_points = tuple(points)
    symbols = tuple(expected_symbols)
    if not resolved_points:
        raise ValueError("U2 Development panel points must be non-empty")
    if not symbols or symbols != tuple(sorted(set(symbols))):
        raise ValueError("U2 Development expected symbols must be sorted and unique")
    if any(
        not isinstance(point, UniversalTradeRLU2PairedExcessPoint)
        for point in resolved_points
    ):
        raise TypeError("U2 Development panel contains an invalid point")

    identities = tuple(
        (
            point.source_window,
            point.decision_timestamp_ns,
            point.training_seed,
            point.concrete_symbol,
        )
        for point in resolved_points
    )
    if len(set(identities)) != len(identities):
        raise ValueError("U2 Development panel contains duplicate leaf identity")
    if {point.concrete_symbol for point in resolved_points} - set(symbols):
        raise ValueError("U2 Development panel contains an unexpected symbol")

    by_window_timestamp: dict[
        tuple[str, int],
        dict[int, dict[str, float]],
    ] = defaultdict(lambda: defaultdict(dict))
    for point in resolved_points:
        by_symbol = by_window_timestamp[
            (point.source_window, point.decision_timestamp_ns)
        ][point.training_seed]
        by_symbol[point.concrete_symbol] = point.candidate_minus_cash_net_log_excess

    segments: list[UniversalTradeRLU2ReducedBootstrapSegment] = []
    for source_window in U2_DEVELOPMENT_WINDOWS:
        timestamps = tuple(
            sorted(
                timestamp
                for window, timestamp in by_window_timestamp
                if window == source_window
            )
        )
        if not timestamps:
            raise ValueError("U2 Development panel requires both D1 and D2 segments")
        reduced: list[float] = []
        for timestamp in timestamps:
            seed_map = by_window_timestamp[(source_window, timestamp)]
            if set(seed_map) != set(U2_TRAINING_SEEDS):
                raise ValueError(
                    "U2 Development timestamp closure is incomplete across training seeds"
                )
            seed_means: list[float] = []
            for seed in U2_TRAINING_SEEDS:
                symbol_map = seed_map[seed]
                if set(symbol_map) != set(symbols):
                    raise ValueError(
                        "U2 Development timestamp closure is incomplete across symbols"
                    )
                seed_means.append(fmean(symbol_map[symbol] for symbol in symbols))
            reduced.append(float(median(seed_means)))
        segments.append(
            UniversalTradeRLU2ReducedBootstrapSegment(
                source_window=source_window,
                decision_timestamps_ns=timestamps,
                net_log_excess=tuple(reduced),
            )
        )

    return tuple(segments)


def bootstrap_universal_trade_rl_u2_development_panel(
    *,
    segments: tuple[UniversalTradeRLU2ReducedBootstrapSegment, ...],
) -> UniversalTradeRLU2DevelopmentBootstrapResult:
    """Bootstrap D1 and D2 independently so a block never crosses their boundary."""

    resolved = tuple(segments)
    if tuple(segment.source_window for segment in resolved) != U2_DEVELOPMENT_WINDOWS:
        raise ValueError("U2 Development bootstrap requires canonical D1/D2 segments")
    if any(
        not isinstance(segment, UniversalTradeRLU2ReducedBootstrapSegment)
        for segment in resolved
    ):
        raise TypeError("U2 Development bootstrap segment is invalid")

    block_lengths = tuple(
        min(
            len(segment.net_log_excess),
            math.ceil(math.sqrt(len(segment.net_log_excess))),
        )
        for segment in resolved
    )
    all_values = tuple(
        value for segment in resolved for value in segment.net_log_excess
    )
    observed_mean = float(fmean(all_values))

    rng = np.random.default_rng(_U2_BOOTSTRAP_SEED)
    means = np.empty(_U2_BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for draw in range(_U2_BOOTSTRAP_RESAMPLES):
        sampled_values: list[float] = []
        for segment, block_length in zip(resolved, block_lengths, strict=True):
            values = segment.net_log_excess
            segment_length = len(values)
            sampled_indices: list[int] = []
            while len(sampled_indices) < segment_length:
                start = int(rng.integers(0, segment_length))
                sampled_indices.extend(
                    (start + offset) % segment_length for offset in range(block_length)
                )
            sampled_values.extend(
                values[index] for index in sampled_indices[:segment_length]
            )
        means[draw] = float(fmean(sampled_values))

    alpha = 1.0 - _U2_BOOTSTRAP_CONFIDENCE_LEVEL
    lower, upper = np.quantile(
        means,
        [alpha / 2.0, 1.0 - alpha / 2.0],
        method=_U2_BOOTSTRAP_QUANTILE_METHOD,
    )
    lower_ci = float(lower)
    upper_ci = float(upper)
    return UniversalTradeRLU2DevelopmentBootstrapResult(
        segment_digests=tuple(segment.digest for segment in resolved),
        block_lengths=block_lengths,
        observed_mean=observed_mean,
        lower_ci=lower_ci,
        upper_ci=upper_ci,
        passed=lower_ci > 0.0,
    )


U2_SEED_ROBUSTNESS_SCHEMA: Final = "universal_trade_rl_u2_seed_robustness_v1"
U2_SEED_ROBUSTNESS_BOOTSTRAP_SCHEMA: Final = (
    "universal_trade_rl_u2_seed_robustness_bootstrap_v1"
)
_U2_SEED_ROBUSTNESS_SCOPES: Final = {
    "D1": (("D1",), ("development_future_1",)),
    "D2": (("D2",), ("development_future_2",)),
    "D1+D2": (
        ("D1", "D2"),
        ("development_future_1", "development_future_2"),
    ),
}
_U2_SEED_ROBUSTNESS_ALLOWED_REASONS: Final = (
    "median_seed_symbol_balanced_net_wealth_not_above_cash",
    "worst_seed_symbol_balanced_net_wealth_below_cash",
    "all_seed_hard_risk_violation_count_nonzero",
    "all_seed_turnover_p95_per_day_above_1_0",
    "paired_excess_bootstrap_lower_ci_not_above_zero",
)


def _u2_cross_seed_robustness_thresholds() -> dict[str, object]:
    thresholds = u2_contract._selection_thresholds_payload()
    raw = thresholds.get("cross_seed_robustness")
    if not isinstance(raw, dict):
        raise ValueError("U2 cross-seed robustness thresholds are missing")
    return dict(raw)


def _u2_positive_int(value: object, *, field: str) -> int:
    resolved = _non_negative_int(value, field=field)
    if resolved <= 0:
        raise ValueError(f"{field} must be positive")
    return resolved


@dataclass(frozen=True, slots=True)
class UniversalTradeRLU2SeedRobustnessBootstrapResult:
    """Digest-bound moving-block bootstrap for one robustness scope."""

    source_windows: tuple[str, ...]
    segment_digests: tuple[str, ...]
    block_lengths: tuple[int, ...]
    observed_mean: float
    lower_ci: float
    upper_ci: float
    lower_ci_min_exclusive: float
    resamples: int
    bootstrap_seed: int
    confidence_level: float
    passed: bool
    quantile_method: str = _U2_BOOTSTRAP_QUANTILE_METHOD
    blocks_may_cross_segment_boundary: bool = False
    schema_version: str = U2_SEED_ROBUSTNESS_BOOTSTRAP_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != U2_SEED_ROBUSTNESS_BOOTSTRAP_SCHEMA:
            raise ValueError("unsupported U2 seed-robustness bootstrap schema")
        source_windows = tuple(self.source_windows)
        segment_digests = tuple(self.segment_digests)
        block_lengths = tuple(self.block_lengths)
        if (
            not source_windows
            or len(source_windows) != len(segment_digests)
            or len(segment_digests) != len(block_lengths)
        ):
            raise ValueError(
                "U2 seed-robustness bootstrap segment evidence is incomplete"
            )
        if len(set(source_windows)) != len(source_windows):
            raise ValueError("U2 seed-robustness bootstrap windows must be unique")
        if any(window not in U2_DEVELOPMENT_WINDOWS for window in source_windows):
            raise ValueError("U2 seed-robustness bootstrap window is invalid")
        for digest in segment_digests:
            require_sha256(digest, field="U2 seed-robustness bootstrap segment digest")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in block_lengths
        ):
            raise ValueError("U2 seed-robustness bootstrap block length is invalid")
        object.__setattr__(self, "source_windows", source_windows)
        object.__setattr__(self, "segment_digests", segment_digests)
        object.__setattr__(self, "block_lengths", block_lengths)
        for field_name in (
            "observed_mean",
            "lower_ci",
            "upper_ci",
            "lower_ci_min_exclusive",
        ):
            object.__setattr__(
                self,
                field_name,
                _finite(
                    getattr(self, field_name),
                    field=f"U2 seed-robustness bootstrap {field_name}",
                ),
            )
        if self.lower_ci > self.upper_ci:
            raise ValueError("U2 seed-robustness bootstrap interval is reversed")
        object.__setattr__(
            self,
            "resamples",
            _u2_positive_int(
                self.resamples,
                field="U2 seed-robustness bootstrap resamples",
            ),
        )
        object.__setattr__(
            self,
            "bootstrap_seed",
            _non_negative_int(
                self.bootstrap_seed,
                field="U2 seed-robustness bootstrap seed",
            ),
        )
        confidence = _finite(
            self.confidence_level,
            field="U2 seed-robustness bootstrap confidence level",
        )
        if not 0.0 < confidence < 1.0:
            raise ValueError(
                "U2 seed-robustness bootstrap confidence level must be in (0,1)"
            )
        object.__setattr__(self, "confidence_level", confidence)
        if self.quantile_method != _U2_BOOTSTRAP_QUANTILE_METHOD:
            raise ValueError("U2 seed-robustness bootstrap quantile method drifted")
        if self.blocks_may_cross_segment_boundary is not False:
            raise ValueError("U2 seed-robustness bootstrap may not cross D1/D2")
        if not isinstance(self.passed, bool):
            raise TypeError("U2 seed-robustness bootstrap passed flag must be boolean")
        expected_passed = self.lower_ci > self.lower_ci_min_exclusive
        if self.passed is not expected_passed:
            raise ValueError("U2 seed-robustness bootstrap pass state is inconsistent")

        expected_digest = content_digest(self.to_payload(include_digest=False))
        if self.digest:
            require_sha256(self.digest, field="U2 seed-robustness bootstrap digest")
            if self.digest != expected_digest:
                raise ValueError("U2 seed-robustness bootstrap digest mismatch")
        object.__setattr__(self, "digest", expected_digest)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "source_windows": self.source_windows,
            "segment_digests": self.segment_digests,
            "block_lengths": self.block_lengths,
            "observed_mean": self.observed_mean,
            "lower_ci": self.lower_ci,
            "upper_ci": self.upper_ci,
            "lower_ci_min_exclusive": self.lower_ci_min_exclusive,
            "resamples": self.resamples,
            "bootstrap_seed": self.bootstrap_seed,
            "confidence_level": self.confidence_level,
            "quantile_method": self.quantile_method,
            "blocks_may_cross_segment_boundary": self.blocks_may_cross_segment_boundary,
            "passed": self.passed,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


def _bootstrap_universal_trade_rl_u2_seed_robustness(
    *,
    segments: tuple[UniversalTradeRLU2ReducedBootstrapSegment, ...],
    thresholds: dict[str, object],
) -> UniversalTradeRLU2SeedRobustnessBootstrapResult:
    resolved = tuple(segments)
    if not resolved:
        raise ValueError("U2 seed-robustness bootstrap requires at least one segment")
    if any(
        not isinstance(segment, UniversalTradeRLU2ReducedBootstrapSegment)
        for segment in resolved
    ):
        raise TypeError("U2 seed-robustness bootstrap segment is invalid")
    source_windows = tuple(segment.source_window for segment in resolved)
    if len(set(source_windows)) != len(source_windows):
        raise ValueError("U2 seed-robustness bootstrap contains duplicate window")

    if thresholds.get("bootstrap_method") != "moving_block_mean_test":
        raise ValueError("U2 seed-robustness bootstrap method drifted")
    if thresholds.get("bootstrap_block_length_rule") != "ceil_sqrt_n_capped":
        raise ValueError("U2 seed-robustness bootstrap block-length rule drifted")
    resamples = _u2_positive_int(
        thresholds.get("bootstrap_resamples"),
        field="U2 seed-robustness bootstrap resamples",
    )
    bootstrap_seed = _non_negative_int(
        thresholds.get("bootstrap_seed"),
        field="U2 seed-robustness bootstrap seed",
    )
    confidence = _finite(
        thresholds.get("bootstrap_confidence_level"),
        field="U2 seed-robustness bootstrap confidence level",
    )
    if not 0.0 < confidence < 1.0:
        raise ValueError(
            "U2 seed-robustness bootstrap confidence level must be in (0,1)"
        )
    lower_ci_min = _finite(
        thresholds.get("paired_excess_vs_cash_bootstrap_lower_ci_min_exclusive"),
        field="U2 seed-robustness bootstrap lower-CI threshold",
    )

    block_lengths = tuple(
        min(
            len(segment.net_log_excess),
            math.ceil(math.sqrt(len(segment.net_log_excess))),
        )
        for segment in resolved
    )
    all_values = tuple(
        value for segment in resolved for value in segment.net_log_excess
    )
    observed_mean = float(fmean(all_values))

    rng = np.random.default_rng(bootstrap_seed)
    means = np.empty(resamples, dtype=np.float64)
    for draw in range(resamples):
        sampled_values: list[float] = []
        for segment, block_length in zip(resolved, block_lengths, strict=True):
            values = segment.net_log_excess
            segment_length = len(values)
            sampled_indices: list[int] = []
            while len(sampled_indices) < segment_length:
                start = int(rng.integers(0, segment_length))
                sampled_indices.extend(
                    (start + offset) % segment_length for offset in range(block_length)
                )
            sampled_values.extend(
                values[index] for index in sampled_indices[:segment_length]
            )
        means[draw] = float(fmean(sampled_values))

    alpha = 1.0 - confidence
    lower, upper = np.quantile(
        means,
        [alpha / 2.0, 1.0 - alpha / 2.0],
        method=_U2_BOOTSTRAP_QUANTILE_METHOD,
    )
    lower_ci = float(lower)
    upper_ci = float(upper)
    return UniversalTradeRLU2SeedRobustnessBootstrapResult(
        source_windows=source_windows,
        segment_digests=tuple(segment.digest for segment in resolved),
        block_lengths=block_lengths,
        observed_mean=observed_mean,
        lower_ci=lower_ci,
        upper_ci=upper_ci,
        lower_ci_min_exclusive=lower_ci_min,
        resamples=resamples,
        bootstrap_seed=bootstrap_seed,
        confidence_level=confidence,
        passed=lower_ci > lower_ci_min,
    )


def _u2_seed_robustness_rejection_reasons(
    *,
    summaries: tuple[UniversalTradeRLU2SelectionMetricSummary, ...],
    bootstrap_result: UniversalTradeRLU2SeedRobustnessBootstrapResult,
    thresholds: dict[str, object],
) -> tuple[str, ...]:
    median_min = _finite(
        thresholds.get("median_seed_net_wealth_min_exclusive"),
        field="U2 seed-robustness median wealth threshold",
    )
    worst_min = _finite(
        thresholds.get("worst_seed_net_wealth_min_inclusive"),
        field="U2 seed-robustness worst wealth threshold",
    )
    hard_risk_required = _non_negative_int(
        thresholds.get("hard_risk_violation_count_required"),
        field="U2 seed-robustness hard-risk threshold",
    )
    turnover_max = _finite(
        thresholds.get("all_seed_turnover_p95_per_day_max_inclusive"),
        field="U2 seed-robustness turnover threshold",
    )
    bootstrap_min = _finite(
        thresholds.get("paired_excess_vs_cash_bootstrap_lower_ci_min_exclusive"),
        field="U2 seed-robustness bootstrap lower-CI threshold",
    )

    seed_wealth = tuple(
        _positive_wealth(
            math.fsum(
                summary.symbol_balanced_net_log_growth
                for summary in summaries
                if summary.training_seed == seed
            ),
            field="U2 seed-robustness seed net",
        )
        for seed in U2_TRAINING_SEEDS
    )
    median_wealth = float(median(seed_wealth))
    worst_wealth = float(min(seed_wealth))
    hard_risk_count = sum(summary.hard_risk_violation_count for summary in summaries)
    turnover_p95 = float(max(summary.turnover_per_day_p95 for summary in summaries))

    reasons: list[str] = []
    if median_wealth <= median_min:
        reasons.append("median_seed_symbol_balanced_net_wealth_not_above_cash")
    if worst_wealth < worst_min:
        reasons.append("worst_seed_symbol_balanced_net_wealth_below_cash")
    if hard_risk_count != hard_risk_required:
        reasons.append("all_seed_hard_risk_violation_count_nonzero")
    if turnover_p95 > turnover_max:
        reasons.append("all_seed_turnover_p95_per_day_above_1_0")
    if bootstrap_result.lower_ci <= bootstrap_min:
        reasons.append("paired_excess_bootstrap_lower_ci_not_above_zero")
    return tuple(reasons)


@dataclass(frozen=True, slots=True)
class UniversalTradeRLU2SeedRobustnessEvidence:
    """Digest-bound D1/D2/D1+D2 robustness evidence over exact U2 seeds."""

    scope: str
    summaries: tuple[UniversalTradeRLU2SelectionMetricSummary, ...]
    scope_closure: tuple[tuple[str, str, str], ...]
    bootstrap_result: UniversalTradeRLU2SeedRobustnessBootstrapResult
    rejection_reasons: tuple[str, ...]
    passed: bool
    scope_closure_digest: str = field(init=False)
    robustness_thresholds_digest: str = field(init=False)
    schema_version: str = U2_SEED_ROBUSTNESS_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != U2_SEED_ROBUSTNESS_SCHEMA:
            raise ValueError("unsupported U2 seed-robustness evidence schema")
        if self.scope not in _U2_SEED_ROBUSTNESS_SCOPES:
            raise ValueError("unsupported U2 seed-robustness scope")
        expected_cells, expected_windows = _U2_SEED_ROBUSTNESS_SCOPES[self.scope]
        summaries = tuple(self.summaries)
        if any(
            not isinstance(summary, UniversalTradeRLU2SelectionMetricSummary)
            for summary in summaries
        ):
            raise TypeError("U2 seed-robustness summary is invalid")
        expected_identities = tuple(
            (seed, cell) for seed in U2_TRAINING_SEEDS for cell in expected_cells
        )
        identities = tuple(
            (summary.training_seed, summary.cell) for summary in summaries
        )
        if identities != expected_identities:
            raise ValueError(
                "U2 seed-robustness summaries do not match fixed seed/cell closure"
            )
        object.__setattr__(self, "summaries", summaries)

        scope_closure = tuple(self.scope_closure)
        if not scope_closure or scope_closure != tuple(sorted(set(scope_closure))):
            raise ValueError(
                "U2 seed-robustness scope closure must be sorted and unique"
            )
        if (
            tuple(sorted({cell for cell, _symbol, _tile in scope_closure}))
            != expected_cells
        ):
            raise ValueError("U2 seed-robustness scope closure has the wrong cells")
        for _cell, symbol, tile_digest in scope_closure:
            if not isinstance(symbol, str) or not symbol:
                raise ValueError("U2 seed-robustness scope closure symbol is invalid")
            require_sha256(tile_digest, field="U2 seed-robustness tile identity")
        object.__setattr__(self, "scope_closure", scope_closure)
        closure_digest = content_digest(
            {
                "schema_version": U2_SEED_ROBUSTNESS_SCHEMA,
                "scope": self.scope,
                "training_seeds": U2_TRAINING_SEEDS,
                "tile_closure": scope_closure,
            }
        )
        object.__setattr__(self, "scope_closure_digest", closure_digest)

        thresholds = _u2_cross_seed_robustness_thresholds()
        threshold_digest = content_digest(thresholds)
        object.__setattr__(self, "robustness_thresholds_digest", threshold_digest)
        if not isinstance(
            self.bootstrap_result,
            UniversalTradeRLU2SeedRobustnessBootstrapResult,
        ):
            raise TypeError("U2 seed-robustness bootstrap result is invalid")
        if self.bootstrap_result.source_windows != expected_windows:
            raise ValueError("U2 seed-robustness bootstrap scope is inconsistent")

        expected_resamples = _u2_positive_int(
            thresholds.get("bootstrap_resamples"),
            field="U2 seed-robustness bootstrap resamples",
        )
        expected_seed = _non_negative_int(
            thresholds.get("bootstrap_seed"),
            field="U2 seed-robustness bootstrap seed",
        )
        expected_confidence = _finite(
            thresholds.get("bootstrap_confidence_level"),
            field="U2 seed-robustness bootstrap confidence level",
        )
        expected_bootstrap_min = _finite(
            thresholds.get("paired_excess_vs_cash_bootstrap_lower_ci_min_exclusive"),
            field="U2 seed-robustness bootstrap lower-CI threshold",
        )
        if (
            self.bootstrap_result.resamples != expected_resamples
            or self.bootstrap_result.bootstrap_seed != expected_seed
            or not math.isclose(
                self.bootstrap_result.confidence_level,
                expected_confidence,
                rel_tol=0.0,
                abs_tol=0.0,
            )
            or not math.isclose(
                self.bootstrap_result.lower_ci_min_exclusive,
                expected_bootstrap_min,
                rel_tol=0.0,
                abs_tol=0.0,
            )
        ):
            raise ValueError("U2 seed-robustness bootstrap settings drifted")

        reasons = tuple(self.rejection_reasons)
        if any(reason not in _U2_SEED_ROBUSTNESS_ALLOWED_REASONS for reason in reasons):
            raise ValueError("U2 seed-robustness contains an unsupported reason")
        if len(set(reasons)) != len(reasons):
            raise ValueError("U2 seed-robustness reasons must be unique")
        expected_reasons = _u2_seed_robustness_rejection_reasons(
            summaries=summaries,
            bootstrap_result=self.bootstrap_result,
            thresholds=thresholds,
        )
        if not isinstance(self.passed, bool):
            raise TypeError("U2 seed-robustness passed flag must be boolean")
        if reasons != expected_reasons or self.passed is not (not expected_reasons):
            raise ValueError("U2 seed-robustness pass/reason state is inconsistent")
        object.__setattr__(self, "rejection_reasons", reasons)

        expected_digest = content_digest(self.to_payload(include_digest=False))
        if self.digest:
            require_sha256(self.digest, field="U2 seed-robustness evidence digest")
            if self.digest != expected_digest:
                raise ValueError("U2 seed-robustness evidence digest mismatch")
        object.__setattr__(self, "digest", expected_digest)

    @property
    def training_seeds(self) -> tuple[int, ...]:
        return U2_TRAINING_SEEDS

    @property
    def seed_symbol_balanced_net_wealth(self) -> tuple[float, ...]:
        return tuple(
            _positive_wealth(
                math.fsum(
                    summary.symbol_balanced_net_log_growth
                    for summary in self.summaries
                    if summary.training_seed == seed
                ),
                field="U2 seed-robustness seed net",
            )
            for seed in U2_TRAINING_SEEDS
        )

    @property
    def median_seed_symbol_balanced_net_wealth(self) -> float:
        return float(median(self.seed_symbol_balanced_net_wealth))

    @property
    def worst_seed_symbol_balanced_net_wealth(self) -> float:
        return float(min(self.seed_symbol_balanced_net_wealth))

    @property
    def all_seed_hard_risk_violation_count(self) -> int:
        return sum(summary.hard_risk_violation_count for summary in self.summaries)

    @property
    def all_seed_turnover_p95_per_day(self) -> float:
        return float(max(summary.turnover_per_day_p95 for summary in self.summaries))

    @property
    def bootstrap_lower_ci(self) -> float:
        return self.bootstrap_result.lower_ci

    @property
    def bootstrap_segment_digests(self) -> tuple[str, ...]:
        return self.bootstrap_result.segment_digests

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "scope": self.scope,
            "training_seeds": self.training_seeds,
            "summary_digests": tuple(summary.digest for summary in self.summaries),
            "scope_closure": self.scope_closure,
            "scope_closure_digest": self.scope_closure_digest,
            "robustness_thresholds_digest": self.robustness_thresholds_digest,
            "bootstrap_result_digest": self.bootstrap_result.digest,
            "seed_symbol_balanced_net_wealth": self.seed_symbol_balanced_net_wealth,
            "median_seed_symbol_balanced_net_wealth": (
                self.median_seed_symbol_balanced_net_wealth
            ),
            "worst_seed_symbol_balanced_net_wealth": (
                self.worst_seed_symbol_balanced_net_wealth
            ),
            "all_seed_hard_risk_violation_count": self.all_seed_hard_risk_violation_count,
            "all_seed_turnover_p95_per_day": self.all_seed_turnover_p95_per_day,
            "bootstrap_lower_ci": self.bootstrap_lower_ci,
            "rejection_reasons": self.rejection_reasons,
            "passed": self.passed,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


def evaluate_universal_trade_rl_u2_seed_robustness(
    *,
    scope: str,
    leaves: tuple[UniversalTradeRLU2SelectionLeafMetrics, ...],
    segments: tuple[UniversalTradeRLU2ReducedBootstrapSegment, ...],
) -> UniversalTradeRLU2SeedRobustnessEvidence:
    """Evaluate preregistered D1/D2/D1+D2 cross-seed robustness."""

    if scope not in _U2_SEED_ROBUSTNESS_SCOPES:
        raise ValueError("unsupported U2 seed-robustness scope")
    expected_cells, expected_windows = _U2_SEED_ROBUSTNESS_SCOPES[scope]
    resolved_leaves = tuple(leaves)
    if not resolved_leaves:
        raise ValueError("U2 seed-robustness requires Selection leaves")
    if any(
        not isinstance(leaf, UniversalTradeRLU2SelectionLeafMetrics)
        for leaf in resolved_leaves
    ):
        raise TypeError("U2 seed-robustness leaf is invalid")
    identities = tuple(leaf.identity for leaf in resolved_leaves)
    if len(set(identities)) != len(identities):
        raise ValueError("U2 seed-robustness contains duplicate leaf identity")
    observed_cells = tuple(sorted({leaf.cell for leaf in resolved_leaves}))
    if observed_cells != expected_cells:
        raise ValueError("U2 seed-robustness cell scope is invalid")
    observed_seeds = tuple(sorted({leaf.training_seed for leaf in resolved_leaves}))
    if observed_seeds != U2_TRAINING_SEEDS:
        raise ValueError("U2 seed-robustness requires complete seed closure (0,1,2)")

    closure_by_seed = {
        seed: {
            (leaf.cell, leaf.concrete_symbol, leaf.tile_identity)
            for leaf in resolved_leaves
            if leaf.training_seed == seed
        }
        for seed in U2_TRAINING_SEEDS
    }
    canonical_closure = closure_by_seed[U2_TRAINING_SEEDS[0]]
    if not canonical_closure or any(
        closure_by_seed[seed] != canonical_closure for seed in U2_TRAINING_SEEDS[1:]
    ):
        raise ValueError("U2 seed-robustness tile/symbol closure drifted across seeds")
    scope_closure = tuple(sorted(canonical_closure))

    summaries: list[UniversalTradeRLU2SelectionMetricSummary] = []
    for seed in U2_TRAINING_SEEDS:
        for cell in expected_cells:
            cell_leaves = tuple(
                leaf
                for leaf in resolved_leaves
                if leaf.training_seed == seed and leaf.cell == cell
            )
            if not cell_leaves:
                raise ValueError("U2 seed-robustness seed/cell closure is incomplete")
            summaries.append(
                summarize_universal_trade_rl_u2_selection_metrics(leaves=cell_leaves)
            )

    resolved_segments = tuple(segments)
    if any(
        not isinstance(segment, UniversalTradeRLU2ReducedBootstrapSegment)
        for segment in resolved_segments
    ):
        raise TypeError("U2 seed-robustness bootstrap segment is invalid")
    observed_windows = tuple(segment.source_window for segment in resolved_segments)
    if len(set(observed_windows)) != len(observed_windows):
        raise ValueError("U2 seed-robustness bootstrap contains duplicate window")
    by_window = {segment.source_window: segment for segment in resolved_segments}
    if any(window not in by_window for window in expected_windows):
        raise ValueError(
            "U2 seed-robustness bootstrap segment/window scope is incomplete"
        )
    selected_segments = tuple(by_window[window] for window in expected_windows)
    if scope == "D1+D2" and set(observed_windows) != set(expected_windows):
        raise ValueError("U2 seed-robustness aggregate bootstrap scope is invalid")

    thresholds = _u2_cross_seed_robustness_thresholds()
    bootstrap_result = _bootstrap_universal_trade_rl_u2_seed_robustness(
        segments=selected_segments,
        thresholds=thresholds,
    )
    summary_tuple = tuple(summaries)
    reasons = _u2_seed_robustness_rejection_reasons(
        summaries=summary_tuple,
        bootstrap_result=bootstrap_result,
        thresholds=thresholds,
    )
    return UniversalTradeRLU2SeedRobustnessEvidence(
        scope=scope,
        summaries=summary_tuple,
        scope_closure=scope_closure,
        bootstrap_result=bootstrap_result,
        rejection_reasons=reasons,
        passed=not reasons,
    )


__all__ = [
    "U2_SEED_ROBUSTNESS_SCHEMA",
    "U2_SEED_ROBUSTNESS_BOOTSTRAP_SCHEMA",
    "UniversalTradeRLU2SeedRobustnessBootstrapResult",
    "UniversalTradeRLU2SeedRobustnessEvidence",
    "evaluate_universal_trade_rl_u2_seed_robustness",
    "U2_PRIMARY_SELECTION_CELL_GATE_SCHEMA",
    "UniversalTradeRLU2PrimarySelectionCellGateEvidence",
    "evaluate_universal_trade_rl_u2_primary_cell_gate",
    "U2_DEVELOPMENT_BOOTSTRAP_SCHEMA",
    "U2_DEVELOPMENT_PANEL_SCHEMA",
    "U2_DEVELOPMENT_WINDOWS",
    "U2_SELECTION_LEAF_METRICS_SCHEMA",
    "U2_SELECTION_SUMMARY_SCHEMA",
    "U2_SELECTION_SYMBOL_METRICS_SCHEMA",
    "UniversalTradeRLU2DevelopmentBootstrapResult",
    "UniversalTradeRLU2PairedExcessPoint",
    "UniversalTradeRLU2ReducedBootstrapSegment",
    "UniversalTradeRLU2SelectionLeafMetrics",
    "UniversalTradeRLU2SelectionMetricSummary",
    "UniversalTradeRLU2SelectionSymbolMetrics",
    "bootstrap_universal_trade_rl_u2_development_panel",
    "build_universal_trade_rl_u2_selection_leaf_metrics",
    "reduce_universal_trade_rl_u2_development_panel",
    "summarize_universal_trade_rl_u2_selection_metrics",
]
