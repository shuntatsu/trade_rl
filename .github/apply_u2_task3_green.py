from pathlib import Path

path = Path("trade_rl/workflows/universal_trade_rl_u2_selection.py")
text = path.read_text(encoding="utf-8")
if "def evaluate_universal_trade_rl_u2_development_seed_robustness(" in text:
    raise SystemExit("Task3 robustness API already exists; refusing duplicate patch")

anchor = "\n\n__all__ = [\n"
if anchor not in text:
    raise SystemExit("Task3 insertion anchor missing")

block = '''

U2_DEVELOPMENT_SEED_ROBUSTNESS_SCHEMA: Final = (
    "universal_trade_rl_u2_development_seed_robustness_v1"
)
_U2_DEVELOPMENT_ROBUSTNESS_SCOPES: Final = {
    "D1": (("D1",), ("development_future_1",)),
    "D2": (("D2",), ("development_future_2",)),
    "D1+D2": (
        ("D1", "D2"),
        ("development_future_1", "development_future_2"),
    ),
}
_U2_DEVELOPMENT_ROBUSTNESS_ALLOWED_REASONS: Final = (
    "median_seed_symbol_balanced_net_wealth_not_above_cash",
    "worst_seed_symbol_balanced_net_wealth_below_cash",
    "all_seed_hard_risk_violation_count_nonzero",
    "all_seed_turnover_p95_per_day_above_1_0",
    "bootstrap_lower_95_ci_not_above_zero",
)


def _bootstrap_universal_trade_rl_u2_scope_segments(
    *,
    segments: tuple[UniversalTradeRLU2ReducedBootstrapSegment, ...],
) -> UniversalTradeRLU2DevelopmentBootstrapResult:
    """Bootstrap one canonical robustness scope without crossing segment boundaries."""

    resolved = tuple(segments)
    if not resolved:
        raise ValueError("U2 robustness bootstrap requires at least one segment")
    if any(
        not isinstance(segment, UniversalTradeRLU2ReducedBootstrapSegment)
        for segment in resolved
    ):
        raise TypeError("U2 robustness bootstrap segment is invalid")

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


@dataclass(frozen=True, slots=True)
class UniversalTradeRLU2DevelopmentSeedRobustnessEvidence:
    """Frozen D1/D2/D1+D2 robustness evidence over exact U2 seeds."""

    scope: str
    summaries: tuple[UniversalTradeRLU2SelectionMetricSummary, ...]
    scope_closure_digest: str
    bootstrap_source_windows: tuple[str, ...]
    bootstrap_result: UniversalTradeRLU2DevelopmentBootstrapResult
    schema_version: str = U2_DEVELOPMENT_SEED_ROBUSTNESS_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != U2_DEVELOPMENT_SEED_ROBUSTNESS_SCHEMA:
            raise ValueError("unsupported U2 Development seed robustness schema")
        if self.scope not in _U2_DEVELOPMENT_ROBUSTNESS_SCOPES:
            raise ValueError("unsupported U2 Development robustness scope")
        expected_cells, expected_windows = _U2_DEVELOPMENT_ROBUSTNESS_SCOPES[
            self.scope
        ]
        summaries = tuple(self.summaries)
        if any(
            not isinstance(summary, UniversalTradeRLU2SelectionMetricSummary)
            for summary in summaries
        ):
            raise TypeError("U2 Development robustness summary is invalid")
        identities = tuple(
            (summary.training_seed, summary.cell) for summary in summaries
        )
        expected_identities = tuple(
            (seed, cell) for seed in U2_TRAINING_SEEDS for cell in expected_cells
        )
        if identities != expected_identities:
            raise ValueError(
                "U2 Development robustness summaries do not match fixed seed/cell closure"
            )
        object.__setattr__(self, "summaries", summaries)
        require_sha256(
            self.scope_closure_digest,
            field="U2 Development robustness scope closure digest",
        )
        windows = tuple(self.bootstrap_source_windows)
        if windows != expected_windows:
            raise ValueError(
                "U2 Development robustness bootstrap windows do not match scope"
            )
        object.__setattr__(self, "bootstrap_source_windows", windows)
        if not isinstance(
            self.bootstrap_result,
            UniversalTradeRLU2DevelopmentBootstrapResult,
        ):
            raise TypeError("U2 Development robustness bootstrap result is invalid")
        if len(self.bootstrap_result.segment_digests) != len(expected_windows):
            raise ValueError(
                "U2 Development robustness bootstrap segment closure is incomplete"
            )

        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest:
            require_sha256(self.digest, field="U2 Development robustness digest")
            if self.digest != expected:
                raise ValueError("U2 Development robustness digest mismatch")
        object.__setattr__(self, "digest", expected)

    @property
    def training_seeds(self) -> tuple[int, ...]:
        return U2_TRAINING_SEEDS

    @property
    def cells(self) -> tuple[str, ...]:
        return _U2_DEVELOPMENT_ROBUSTNESS_SCOPES[self.scope][0]

    @property
    def seed_symbol_balanced_net_wealth(self) -> tuple[tuple[int, float], ...]:
        by_seed: list[tuple[int, float]] = []
        for seed in U2_TRAINING_SEEDS:
            log_growth = math.fsum(
                summary.symbol_balanced_net_log_growth
                for summary in self.summaries
                if summary.training_seed == seed
            )
            by_seed.append(
                (
                    seed,
                    _positive_wealth(
                        log_growth,
                        field="U2 Development robustness seed net",
                    ),
                )
            )
        return tuple(by_seed)

    @property
    def median_seed_symbol_balanced_net_wealth(self) -> float:
        return float(
            median(wealth for _seed, wealth in self.seed_symbol_balanced_net_wealth)
        )

    @property
    def worst_seed_symbol_balanced_net_wealth(self) -> float:
        return float(
            min(wealth for _seed, wealth in self.seed_symbol_balanced_net_wealth)
        )

    @property
    def seed_hard_risk_violation_count(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (
                seed,
                sum(
                    summary.hard_risk_violation_count
                    for summary in self.summaries
                    if summary.training_seed == seed
                ),
            )
            for seed in U2_TRAINING_SEEDS
        )

    @property
    def all_seed_hard_risk_violation_count(self) -> int:
        return sum(count for _seed, count in self.seed_hard_risk_violation_count)

    @property
    def seed_turnover_p95_per_day(self) -> tuple[tuple[int, float], ...]:
        return tuple(
            (
                seed,
                max(
                    summary.turnover_per_day_p95
                    for summary in self.summaries
                    if summary.training_seed == seed
                ),
            )
            for seed in U2_TRAINING_SEEDS
        )

    @property
    def all_seed_turnover_p95_per_day(self) -> float:
        return float(max(value for _seed, value in self.seed_turnover_p95_per_day))

    @property
    def bootstrap_lower_ci(self) -> float:
        return self.bootstrap_result.lower_ci

    @property
    def rejection_reasons(self) -> tuple[str, ...]:
        reasons: list[str] = []
        if self.median_seed_symbol_balanced_net_wealth <= 1.0:
            reasons.append(
                "median_seed_symbol_balanced_net_wealth_not_above_cash"
            )
        if self.worst_seed_symbol_balanced_net_wealth < 1.0:
            reasons.append("worst_seed_symbol_balanced_net_wealth_below_cash")
        if self.all_seed_hard_risk_violation_count != 0:
            reasons.append("all_seed_hard_risk_violation_count_nonzero")
        if self.all_seed_turnover_p95_per_day > 1.0:
            reasons.append("all_seed_turnover_p95_per_day_above_1_0")
        if self.bootstrap_lower_ci <= 0.0:
            reasons.append("bootstrap_lower_95_ci_not_above_zero")
        if any(
            reason not in _U2_DEVELOPMENT_ROBUSTNESS_ALLOWED_REASONS
            for reason in reasons
        ):
            raise AssertionError("internal U2 Development robustness reason drift")
        return tuple(reasons)

    @property
    def passed(self) -> bool:
        return not self.rejection_reasons

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "scope": self.scope,
            "training_seeds": self.training_seeds,
            "cells": self.cells,
            "summary_digests": tuple(summary.digest for summary in self.summaries),
            "scope_closure_digest": self.scope_closure_digest,
            "bootstrap_source_windows": self.bootstrap_source_windows,
            "bootstrap_result_digest": self.bootstrap_result.digest,
            "seed_symbol_balanced_net_wealth": self.seed_symbol_balanced_net_wealth,
            "median_seed_symbol_balanced_net_wealth": (
                self.median_seed_symbol_balanced_net_wealth
            ),
            "worst_seed_symbol_balanced_net_wealth": (
                self.worst_seed_symbol_balanced_net_wealth
            ),
            "seed_hard_risk_violation_count": self.seed_hard_risk_violation_count,
            "all_seed_hard_risk_violation_count": (
                self.all_seed_hard_risk_violation_count
            ),
            "seed_turnover_p95_per_day": self.seed_turnover_p95_per_day,
            "all_seed_turnover_p95_per_day": self.all_seed_turnover_p95_per_day,
            "bootstrap_lower_ci": self.bootstrap_lower_ci,
            "passed": self.passed,
            "rejection_reasons": self.rejection_reasons,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


def evaluate_universal_trade_rl_u2_development_seed_robustness(
    *,
    scope: str,
    leaves: tuple[UniversalTradeRLU2SelectionLeafMetrics, ...],
    bootstrap_segments: tuple[UniversalTradeRLU2ReducedBootstrapSegment, ...],
) -> UniversalTradeRLU2DevelopmentSeedRobustnessEvidence:
    """Evaluate one fixed D1/D2/D1+D2 seed-robustness gate."""

    if scope not in _U2_DEVELOPMENT_ROBUSTNESS_SCOPES:
        raise ValueError("unsupported U2 Development robustness scope")
    expected_cells, expected_windows = _U2_DEVELOPMENT_ROBUSTNESS_SCOPES[scope]
    resolved_leaves = tuple(leaves)
    if not resolved_leaves:
        raise ValueError("U2 Development robustness requires Selection leaves")
    if any(
        not isinstance(leaf, UniversalTradeRLU2SelectionLeafMetrics)
        for leaf in resolved_leaves
    ):
        raise TypeError("U2 Development robustness leaf is invalid")
    identities = tuple(leaf.identity for leaf in resolved_leaves)
    if len(set(identities)) != len(identities):
        raise ValueError("U2 Development robustness contains duplicate leaf identity")
    observed_cells = tuple(sorted({leaf.cell for leaf in resolved_leaves}))
    if observed_cells != expected_cells:
        raise ValueError("U2 Development robustness cell scope is invalid")
    observed_seeds = tuple(sorted({leaf.training_seed for leaf in resolved_leaves}))
    if observed_seeds != U2_TRAINING_SEEDS:
        raise ValueError(
            "U2 Development robustness requires complete fixed seed closure (0,1,2)"
        )

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
        raise ValueError(
            "U2 Development robustness tile/scope identity closure drifted across seeds"
        )
    scope_closure_digest = content_digest(
        {
            "schema_version": U2_DEVELOPMENT_SEED_ROBUSTNESS_SCHEMA,
            "scope": scope,
            "training_seeds": U2_TRAINING_SEEDS,
            "tile_closure": tuple(sorted(canonical_closure)),
        }
    )

    summaries: list[UniversalTradeRLU2SelectionMetricSummary] = []
    for seed in U2_TRAINING_SEEDS:
        for cell in expected_cells:
            cell_leaves = tuple(
                leaf
                for leaf in resolved_leaves
                if leaf.training_seed == seed and leaf.cell == cell
            )
            if not cell_leaves:
                raise ValueError(
                    "U2 Development robustness seed/cell closure is incomplete"
                )
            summaries.append(
                summarize_universal_trade_rl_u2_selection_metrics(
                    leaves=cell_leaves
                )
            )

    segments = tuple(bootstrap_segments)
    if any(
        not isinstance(segment, UniversalTradeRLU2ReducedBootstrapSegment)
        for segment in segments
    ):
        raise TypeError("U2 Development robustness bootstrap segment is invalid")
    observed_windows = tuple(segment.source_window for segment in segments)
    if observed_windows != expected_windows:
        raise ValueError(
            "U2 Development robustness bootstrap segment/window scope is invalid"
        )
    bootstrap_result = _bootstrap_universal_trade_rl_u2_scope_segments(
        segments=segments
    )

    return UniversalTradeRLU2DevelopmentSeedRobustnessEvidence(
        scope=scope,
        summaries=tuple(summaries),
        scope_closure_digest=scope_closure_digest,
        bootstrap_source_windows=observed_windows,
        bootstrap_result=bootstrap_result,
    )
'''

text = text.replace(anchor, block + anchor, 1)
all_anchor = "__all__ = [\n"
exports = (
    '    "U2_DEVELOPMENT_SEED_ROBUSTNESS_SCHEMA",\n'
    '    "UniversalTradeRLU2DevelopmentSeedRobustnessEvidence",\n'
    '    "evaluate_universal_trade_rl_u2_development_seed_robustness",\n'
)
text = text.replace(all_anchor, all_anchor + exports, 1)
path.write_text(text, encoding="utf-8")
