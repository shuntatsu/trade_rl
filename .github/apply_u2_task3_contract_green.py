from pathlib import Path

path = Path("trade_rl/workflows/universal_trade_rl_u2_selection.py")
text = path.read_text(encoding="utf-8")
if "def evaluate_universal_trade_rl_u2_seed_robustness(" in text:
    raise SystemExit("Task3 robustness API already exists; refusing duplicate patch")

start = text.index("def bootstrap_universal_trade_rl_u2_development_panel(\n")
end = text.index("\n\n__all__ = [\n", start)

replacement = '''def _bootstrap_universal_trade_rl_u2_segments(
    *,
    segments: tuple[UniversalTradeRLU2ReducedBootstrapSegment, ...],
) -> UniversalTradeRLU2DevelopmentBootstrapResult:
    """Bootstrap canonical segments independently without crossing boundaries."""

    resolved = tuple(segments)
    if not resolved:
        raise ValueError("U2 Development bootstrap requires at least one segment")
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


def bootstrap_universal_trade_rl_u2_development_panel(
    *,
    segments: tuple[UniversalTradeRLU2ReducedBootstrapSegment, ...],
) -> UniversalTradeRLU2DevelopmentBootstrapResult:
    """Bootstrap D1 and D2 independently so a block never crosses their boundary."""

    resolved = tuple(segments)
    if any(
        not isinstance(segment, UniversalTradeRLU2ReducedBootstrapSegment)
        for segment in resolved
    ):
        raise TypeError("U2 Development bootstrap segment is invalid")
    if tuple(segment.source_window for segment in resolved) != U2_DEVELOPMENT_WINDOWS:
        raise ValueError("U2 Development bootstrap requires canonical D1/D2 segments")
    return _bootstrap_universal_trade_rl_u2_segments(segments=resolved)


U2_DEVELOPMENT_SEED_ROBUSTNESS_SCHEMA: Final = (
    "universal_trade_rl_u2_development_seed_robustness_v1"
)
_U2_DEVELOPMENT_ROBUSTNESS_SCOPES: Final = {
    "D1": ("D1",),
    "D2": ("D2",),
    "D1+D2": ("D1", "D2"),
}
_U2_DEVELOPMENT_ROBUSTNESS_REASONS: Final = (
    "median_seed_symbol_balanced_net_wealth_not_above_cash",
    "worst_seed_symbol_balanced_net_wealth_below_cash",
    "all_seed_hard_risk_violation_count_nonzero",
    "all_seed_turnover_p95_per_day_above_1_0",
    "paired_excess_bootstrap_lower_ci_not_above_zero",
)


def _u2_seed_robustness_rejection_reasons(
    *,
    median_seed_symbol_balanced_net_wealth: float,
    worst_seed_symbol_balanced_net_wealth: float,
    all_seed_hard_risk_violation_count: int,
    all_seed_turnover_p95_per_day: float,
    bootstrap_lower_ci: float,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if median_seed_symbol_balanced_net_wealth <= 1.0:
        reasons.append("median_seed_symbol_balanced_net_wealth_not_above_cash")
    if worst_seed_symbol_balanced_net_wealth < 1.0:
        reasons.append("worst_seed_symbol_balanced_net_wealth_below_cash")
    if all_seed_hard_risk_violation_count != 0:
        reasons.append("all_seed_hard_risk_violation_count_nonzero")
    if all_seed_turnover_p95_per_day > 1.0:
        reasons.append("all_seed_turnover_p95_per_day_above_1_0")
    if bootstrap_lower_ci <= 0.0:
        reasons.append("paired_excess_bootstrap_lower_ci_not_above_zero")
    return tuple(reasons)


@dataclass(frozen=True, slots=True)
class UniversalTradeRLU2DevelopmentSeedRobustnessEvidence:
    """Fail-closed D1/D2/D1+D2 robustness evidence over fixed seeds."""

    scope: str
    training_seeds: tuple[int, ...]
    leaf_digests: tuple[str, ...]
    scope_closure_digest: str
    seed_symbol_balanced_net_wealth: tuple[float, ...]
    median_seed_symbol_balanced_net_wealth: float
    worst_seed_symbol_balanced_net_wealth: float
    seed_hard_risk_violation_count: tuple[int, ...]
    all_seed_hard_risk_violation_count: int
    seed_turnover_p95_per_day: tuple[float, ...]
    all_seed_turnover_p95_per_day: float
    bootstrap_segment_digests: tuple[str, ...]
    bootstrap_result_digest: str
    bootstrap_lower_ci: float
    passed: bool
    rejection_reasons: tuple[str, ...]
    schema_version: str = U2_DEVELOPMENT_SEED_ROBUSTNESS_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != U2_DEVELOPMENT_SEED_ROBUSTNESS_SCHEMA:
            raise ValueError("unsupported U2 Development seed robustness schema")
        if self.scope not in _U2_DEVELOPMENT_ROBUSTNESS_SCOPES:
            raise ValueError("unsupported U2 Development seed robustness scope")
        seeds = tuple(self.training_seeds)
        if seeds != U2_TRAINING_SEEDS:
            raise ValueError("U2 Development seed robustness fixed seed closure drifted")
        object.__setattr__(self, "training_seeds", seeds)

        leaf_digests = tuple(self.leaf_digests)
        if not leaf_digests or len(set(leaf_digests)) != len(leaf_digests):
            raise ValueError("U2 Development seed robustness leaf evidence is invalid")
        for digest in leaf_digests:
            require_sha256(digest, field="U2 Development robustness leaf digest")
        object.__setattr__(self, "leaf_digests", leaf_digests)
        require_sha256(
            self.scope_closure_digest,
            field="U2 Development robustness scope closure digest",
        )

        wealth = tuple(
            _positive_wealth(value, field="U2 Development robustness seed wealth")
            if value <= 0.0
            else _finite(value, field="U2 Development robustness seed wealth")
            for value in self.seed_symbol_balanced_net_wealth
        )
        if len(wealth) != len(U2_TRAINING_SEEDS) or any(value <= 0.0 for value in wealth):
            raise ValueError("U2 Development robustness seed wealth closure is invalid")
        object.__setattr__(self, "seed_symbol_balanced_net_wealth", wealth)
        expected_median = float(median(wealth))
        expected_worst = float(min(wealth))
        object.__setattr__(
            self,
            "median_seed_symbol_balanced_net_wealth",
            _finite(
                self.median_seed_symbol_balanced_net_wealth,
                field="U2 Development robustness median seed wealth",
            ),
        )
        object.__setattr__(
            self,
            "worst_seed_symbol_balanced_net_wealth",
            _finite(
                self.worst_seed_symbol_balanced_net_wealth,
                field="U2 Development robustness worst seed wealth",
            ),
        )
        if not math.isclose(
            self.median_seed_symbol_balanced_net_wealth,
            expected_median,
            rel_tol=0.0,
            abs_tol=1e-12,
        ) or not math.isclose(
            self.worst_seed_symbol_balanced_net_wealth,
            expected_worst,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("U2 Development robustness seed wealth reduction is inconsistent")

        risk_counts = tuple(
            _non_negative_int(
                value,
                field="U2 Development robustness seed hard-risk violation count",
            )
            for value in self.seed_hard_risk_violation_count
        )
        if len(risk_counts) != len(U2_TRAINING_SEEDS):
            raise ValueError("U2 Development robustness risk seed closure is invalid")
        object.__setattr__(self, "seed_hard_risk_violation_count", risk_counts)
        expected_risk_count = sum(risk_counts)
        object.__setattr__(
            self,
            "all_seed_hard_risk_violation_count",
            _non_negative_int(
                self.all_seed_hard_risk_violation_count,
                field="U2 Development robustness all-seed hard-risk violation count",
            ),
        )
        if self.all_seed_hard_risk_violation_count != expected_risk_count:
            raise ValueError("U2 Development robustness hard-risk reduction is inconsistent")

        turnovers = tuple(
            _finite(value, field="U2 Development robustness seed turnover p95")
            for value in self.seed_turnover_p95_per_day
        )
        if len(turnovers) != len(U2_TRAINING_SEEDS) or any(
            value < 0.0 for value in turnovers
        ):
            raise ValueError("U2 Development robustness turnover seed closure is invalid")
        object.__setattr__(self, "seed_turnover_p95_per_day", turnovers)
        expected_turnover = float(max(turnovers))
        object.__setattr__(
            self,
            "all_seed_turnover_p95_per_day",
            _finite(
                self.all_seed_turnover_p95_per_day,
                field="U2 Development robustness all-seed turnover p95",
            ),
        )
        if not math.isclose(
            self.all_seed_turnover_p95_per_day,
            expected_turnover,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("U2 Development robustness turnover reduction is inconsistent")

        segment_digests = tuple(self.bootstrap_segment_digests)
        expected_segment_count = 2 if self.scope == "D1+D2" else 1
        if len(segment_digests) != expected_segment_count:
            raise ValueError("U2 Development robustness bootstrap segment closure is invalid")
        for digest in segment_digests:
            require_sha256(digest, field="U2 Development robustness bootstrap segment digest")
        object.__setattr__(self, "bootstrap_segment_digests", segment_digests)
        require_sha256(
            self.bootstrap_result_digest,
            field="U2 Development robustness bootstrap result digest",
        )
        object.__setattr__(
            self,
            "bootstrap_lower_ci",
            _finite(
                self.bootstrap_lower_ci,
                field="U2 Development robustness bootstrap lower CI",
            ),
        )

        if not isinstance(self.passed, bool):
            raise TypeError("U2 Development robustness passed flag must be boolean")
        reasons = tuple(self.rejection_reasons)
        if len(set(reasons)) != len(reasons) or any(
            reason not in _U2_DEVELOPMENT_ROBUSTNESS_REASONS for reason in reasons
        ):
            raise ValueError("U2 Development robustness rejection reason is invalid")
        expected_reasons = _u2_seed_robustness_rejection_reasons(
            median_seed_symbol_balanced_net_wealth=(
                self.median_seed_symbol_balanced_net_wealth
            ),
            worst_seed_symbol_balanced_net_wealth=(
                self.worst_seed_symbol_balanced_net_wealth
            ),
            all_seed_hard_risk_violation_count=(
                self.all_seed_hard_risk_violation_count
            ),
            all_seed_turnover_p95_per_day=self.all_seed_turnover_p95_per_day,
            bootstrap_lower_ci=self.bootstrap_lower_ci,
        )
        expected_passed = not expected_reasons
        if reasons != expected_reasons or self.passed != expected_passed:
            raise ValueError("U2 Development robustness pass/reason state is inconsistent")
        object.__setattr__(self, "rejection_reasons", reasons)

        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest:
            require_sha256(self.digest, field="U2 Development robustness digest")
            if self.digest != expected:
                raise ValueError("U2 Development robustness digest mismatch")
        object.__setattr__(self, "digest", expected)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "scope": self.scope,
            "training_seeds": self.training_seeds,
            "leaf_digests": self.leaf_digests,
            "scope_closure_digest": self.scope_closure_digest,
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
            "bootstrap_segment_digests": self.bootstrap_segment_digests,
            "bootstrap_result_digest": self.bootstrap_result_digest,
            "bootstrap_lower_ci": self.bootstrap_lower_ci,
            "passed": self.passed,
            "rejection_reasons": self.rejection_reasons,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


def evaluate_universal_trade_rl_u2_seed_robustness(
    *,
    scope: str,
    leaves: tuple[UniversalTradeRLU2SelectionLeafMetrics, ...],
    segments: tuple[UniversalTradeRLU2ReducedBootstrapSegment, ...],
) -> UniversalTradeRLU2DevelopmentSeedRobustnessEvidence:
    """Evaluate D1, D2, or sequential D1+D2 robustness over fixed U2 seeds."""

    if scope not in _U2_DEVELOPMENT_ROBUSTNESS_SCOPES:
        raise ValueError("unsupported U2 Development seed robustness scope")
    expected_cells = _U2_DEVELOPMENT_ROBUSTNESS_SCOPES[scope]
    resolved_leaves = tuple(leaves)
    if not resolved_leaves:
        raise ValueError("U2 Development seed robustness requires Selection leaves")
    if any(
        not isinstance(leaf, UniversalTradeRLU2SelectionLeafMetrics)
        for leaf in resolved_leaves
    ):
        raise TypeError("U2 Development seed robustness leaf is invalid")
    identities = tuple(leaf.identity for leaf in resolved_leaves)
    if len(set(identities)) != len(identities):
        raise ValueError("U2 Development seed robustness contains duplicate leaf identity")
    observed_cells = tuple(sorted({leaf.cell for leaf in resolved_leaves}))
    if observed_cells != expected_cells:
        raise ValueError("U2 Development seed robustness cell scope is invalid")
    observed_seeds = tuple(sorted({leaf.training_seed for leaf in resolved_leaves}))
    if observed_seeds != U2_TRAINING_SEEDS:
        raise ValueError("U2 Development seed robustness seed closure is incomplete")

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
            "U2 Development seed robustness tile/symbol closure drifted across seeds"
        )
    scope_closure_digest = content_digest(
        {
            "schema_version": U2_DEVELOPMENT_SEED_ROBUSTNESS_SCHEMA,
            "scope": scope,
            "training_seeds": U2_TRAINING_SEEDS,
            "tile_symbol_closure": tuple(sorted(canonical_closure)),
        }
    )

    seed_wealth: list[float] = []
    seed_risk_counts: list[int] = []
    seed_turnovers: list[float] = []
    for seed in U2_TRAINING_SEEDS:
        seed_log_growth = 0.0
        for cell in expected_cells:
            cell_leaves = tuple(
                leaf
                for leaf in resolved_leaves
                if leaf.training_seed == seed and leaf.cell == cell
            )
            if not cell_leaves:
                raise ValueError("U2 Development seed robustness seed/cell closure is incomplete")
            summary = summarize_universal_trade_rl_u2_selection_metrics(
                leaves=cell_leaves
            )
            seed_log_growth += summary.symbol_balanced_net_log_growth
        seed_wealth.append(
            _positive_wealth(
                seed_log_growth,
                field="U2 Development robustness seed net",
            )
        )
        seed_rows = tuple(
            leaf for leaf in resolved_leaves if leaf.training_seed == seed
        )
        seed_risk_counts.append(
            sum(leaf.hard_risk_violation_count for leaf in seed_rows)
        )
        seed_turnovers.append(
            float(
                np.quantile(
                    [leaf.turnover_per_day for leaf in seed_rows],
                    0.95,
                    method="linear",
                )
            )
        )

    resolved_segments = tuple(segments)
    if any(
        not isinstance(segment, UniversalTradeRLU2ReducedBootstrapSegment)
        for segment in resolved_segments
    ):
        raise TypeError("U2 Development robustness bootstrap segment is invalid")
    if tuple(segment.source_window for segment in resolved_segments) != U2_DEVELOPMENT_WINDOWS:
        raise ValueError("U2 Development robustness requires canonical D1/D2 segments")
    selected_segments = (
        (resolved_segments[0],)
        if scope == "D1"
        else (resolved_segments[1],)
        if scope == "D2"
        else resolved_segments
    )
    bootstrap_result = _bootstrap_universal_trade_rl_u2_segments(
        segments=selected_segments
    )

    wealth_tuple = tuple(seed_wealth)
    median_wealth = float(median(wealth_tuple))
    worst_wealth = float(min(wealth_tuple))
    risk_tuple = tuple(seed_risk_counts)
    all_risk = sum(risk_tuple)
    turnover_tuple = tuple(seed_turnovers)
    all_turnover = float(max(turnover_tuple))
    reasons = _u2_seed_robustness_rejection_reasons(
        median_seed_symbol_balanced_net_wealth=median_wealth,
        worst_seed_symbol_balanced_net_wealth=worst_wealth,
        all_seed_hard_risk_violation_count=all_risk,
        all_seed_turnover_p95_per_day=all_turnover,
        bootstrap_lower_ci=bootstrap_result.lower_ci,
    )
    ordered_leaves = tuple(sorted(resolved_leaves, key=lambda leaf: leaf.identity))
    return UniversalTradeRLU2DevelopmentSeedRobustnessEvidence(
        scope=scope,
        training_seeds=U2_TRAINING_SEEDS,
        leaf_digests=tuple(leaf.digest for leaf in ordered_leaves),
        scope_closure_digest=scope_closure_digest,
        seed_symbol_balanced_net_wealth=wealth_tuple,
        median_seed_symbol_balanced_net_wealth=median_wealth,
        worst_seed_symbol_balanced_net_wealth=worst_wealth,
        seed_hard_risk_violation_count=risk_tuple,
        all_seed_hard_risk_violation_count=all_risk,
        seed_turnover_p95_per_day=turnover_tuple,
        all_seed_turnover_p95_per_day=all_turnover,
        bootstrap_segment_digests=bootstrap_result.segment_digests,
        bootstrap_result_digest=bootstrap_result.digest,
        bootstrap_lower_ci=bootstrap_result.lower_ci,
        passed=not reasons,
        rejection_reasons=reasons,
    )
'''

text = text[:start] + replacement + text[end:]
all_anchor = "__all__ = [\n"
exports = (
    '    "U2_DEVELOPMENT_SEED_ROBUSTNESS_SCHEMA",\n'
    '    "UniversalTradeRLU2DevelopmentSeedRobustnessEvidence",\n'
    '    "evaluate_universal_trade_rl_u2_seed_robustness",\n'
)
text = text.replace(all_anchor, all_anchor + exports, 1)
path.write_text(text, encoding="utf-8")
