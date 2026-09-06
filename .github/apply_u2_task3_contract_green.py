from pathlib import Path

path = Path("trade_rl/workflows/universal_trade_rl_u2_selection.py")
text = path.read_text(encoding="utf-8")
if "def evaluate_universal_trade_rl_u2_seed_robustness(" in text:
    raise SystemExit("Task3 seed robustness API already exists; refusing duplicate patch")

anchor = "\n\n__all__ = [\n"
if anchor not in text:
    raise SystemExit("Task3 insertion anchor missing")

block = r'''

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
            raise ValueError("U2 seed-robustness bootstrap segment evidence is incomplete")
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
        raise ValueError("U2 seed-robustness bootstrap confidence level must be in (0,1)")
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
        identities = tuple((summary.training_seed, summary.cell) for summary in summaries)
        if identities != expected_identities:
            raise ValueError(
                "U2 seed-robustness summaries do not match fixed seed/cell closure"
            )
        object.__setattr__(self, "summaries", summaries)

        scope_closure = tuple(self.scope_closure)
        if not scope_closure or scope_closure != tuple(sorted(set(scope_closure))):
            raise ValueError("U2 seed-robustness scope closure must be sorted and unique")
        if tuple(sorted({cell for cell, _symbol, _tile in scope_closure})) != expected_cells:
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
        closure_by_seed[seed] != canonical_closure
        for seed in U2_TRAINING_SEEDS[1:]
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
        raise ValueError("U2 seed-robustness bootstrap segment/window scope is incomplete")
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
'''

text = text.replace(anchor, block + anchor, 1)
exports = (
    '    "U2_SEED_ROBUSTNESS_SCHEMA",\n'
    '    "U2_SEED_ROBUSTNESS_BOOTSTRAP_SCHEMA",\n'
    '    "UniversalTradeRLU2SeedRobustnessBootstrapResult",\n'
    '    "UniversalTradeRLU2SeedRobustnessEvidence",\n'
    '    "evaluate_universal_trade_rl_u2_seed_robustness",\n'
)
text = text.replace("__all__ = [\n", "__all__ = [\n" + exports, 1)
path.write_text(text, encoding="utf-8")
