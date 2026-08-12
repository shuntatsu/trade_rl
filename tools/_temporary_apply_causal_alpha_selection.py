from pathlib import Path

path = Path("trade_rl/workflows/universal_causal_alpha_teacher.py")
text = path.read_text(encoding="utf-8")

old = "from trade_rl.learning.episode_oracle_bc import resolve_episode_initial_weights\n"
new = """from trade_rl.learning.episode_oracle_bc import (
    evaluate_episode_action_path,
    resolve_episode_initial_weights,
)
"""
if text.count(old) != 1:
    raise SystemExit("episode oracle bc import target drifted")
text = text.replace(old, new)

marker = "\ndef _train_range(\n"
if text.count(marker) != 1:
    raise SystemExit("selection dataclass insertion marker drifted")
selection_types = r'''

@dataclass(frozen=True, slots=True)
class CausalAlphaCandidateConfig:
    """One member of the bounded, predeclared train-only selection grid."""

    name: str
    ridge: CausalAlphaRidgeConfig
    controller: CausalAlphaControllerConfig
    digest: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("causal alpha candidate name must be non-empty")
        if not isinstance(self.ridge, CausalAlphaRidgeConfig):
            raise TypeError("causal alpha candidate ridge config is invalid")
        if not isinstance(self.controller, CausalAlphaControllerConfig):
            raise TypeError("causal alpha candidate controller config is invalid")
        expected = content_digest(
            {
                "controller_digest": self.controller.digest,
                "name": self.name,
                "ridge_digest": self.ridge.digest,
                "schema_version": "causal_alpha_candidate_v1",
            }
        )
        if self.digest and self.digest != expected:
            raise ValueError("causal alpha candidate digest mismatch")
        object.__setattr__(self, "digest", expected)


@dataclass(frozen=True, slots=True)
class CausalAlphaCandidateEpisodeMetrics:
    candidate_digest: str
    symbol: str
    episode_index: int
    gross_return: float
    net_return: float
    turnover_per_day: float
    total_execution_cost: float
    trade_count: int
    risk_violation: bool
    digest: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_digest, str) or len(self.candidate_digest) != 64:
            raise ValueError("causal alpha candidate metric digest is invalid")
        if not self.symbol:
            raise ValueError("causal alpha candidate metric symbol is empty")
        if (
            isinstance(self.episode_index, bool)
            or not isinstance(self.episode_index, int)
            or self.episode_index < 0
        ):
            raise ValueError("causal alpha candidate episode index is invalid")
        for field, value in (
            ("gross_return", self.gross_return),
            ("net_return", self.net_return),
            ("turnover_per_day", self.turnover_per_day),
            ("total_execution_cost", self.total_execution_cost),
        ):
            if not np.isfinite(value):
                raise ValueError(f"causal alpha {field} must be finite")
        if self.turnover_per_day < 0.0 or self.total_execution_cost < 0.0:
            raise ValueError("causal alpha turnover and cost must be non-negative")
        if (
            isinstance(self.trade_count, bool)
            or not isinstance(self.trade_count, int)
            or self.trade_count < 0
        ):
            raise ValueError("causal alpha trade_count must be non-negative")
        if not isinstance(self.risk_violation, bool):
            raise TypeError("causal alpha risk_violation must be boolean")
        expected = content_digest(
            {
                "candidate_digest": self.candidate_digest,
                "episode_index": self.episode_index,
                "gross_return": self.gross_return,
                "net_return": self.net_return,
                "risk_violation": self.risk_violation,
                "schema_version": "causal_alpha_candidate_episode_metrics_v1",
                "symbol": self.symbol,
                "total_execution_cost": self.total_execution_cost,
                "trade_count": self.trade_count,
                "turnover_per_day": self.turnover_per_day,
            }
        )
        if self.digest and self.digest != expected:
            raise ValueError("causal alpha candidate episode metric digest mismatch")
        object.__setattr__(self, "digest", expected)


@dataclass(frozen=True, slots=True)
class CausalAlphaCandidateEvidence:
    candidate: CausalAlphaCandidateConfig
    episode_metrics: tuple[CausalAlphaCandidateEpisodeMetrics, ...]
    lower_tail_net_return: float
    mean_net_return: float
    turnover_per_day: float
    total_execution_cost: float
    negative_gross_episode_count: int
    total_trade_count: int
    risk_violation: bool
    admissible: bool
    rejection_reasons: tuple[str, ...]
    digest: str = ""

    def __post_init__(self) -> None:
        metrics = tuple(self.episode_metrics)
        if not metrics:
            raise ValueError("causal alpha candidate evidence needs episode metrics")
        if any(item.candidate_digest != self.candidate.digest for item in metrics):
            raise ValueError("causal alpha candidate metric identity drifted")
        scopes = tuple((item.symbol, item.episode_index) for item in metrics)
        if len(set(scopes)) != len(scopes):
            raise ValueError("causal alpha candidate episode metrics are duplicated")
        for value in (
            self.lower_tail_net_return,
            self.mean_net_return,
            self.turnover_per_day,
            self.total_execution_cost,
        ):
            if not np.isfinite(value):
                raise ValueError("causal alpha candidate aggregate metric is non-finite")
        if self.turnover_per_day < 0.0 or self.total_execution_cost < 0.0:
            raise ValueError("causal alpha candidate aggregate cost metrics are invalid")
        if self.negative_gross_episode_count < 0 or self.total_trade_count < 0:
            raise ValueError("causal alpha candidate aggregate counts are invalid")
        if not isinstance(self.risk_violation, bool) or not isinstance(self.admissible, bool):
            raise TypeError("causal alpha candidate gate flags must be boolean")
        reasons = tuple(self.rejection_reasons)
        if self.admissible == bool(reasons):
            raise ValueError("causal alpha candidate admission reasons are inconsistent")
        expected = content_digest(
            {
                "admissible": self.admissible,
                "candidate_digest": self.candidate.digest,
                "episode_metric_digests": tuple(item.digest for item in metrics),
                "lower_tail_net_return": self.lower_tail_net_return,
                "mean_net_return": self.mean_net_return,
                "negative_gross_episode_count": self.negative_gross_episode_count,
                "rejection_reasons": reasons,
                "risk_violation": self.risk_violation,
                "schema_version": "causal_alpha_candidate_evidence_v1",
                "total_execution_cost": self.total_execution_cost,
                "total_trade_count": self.total_trade_count,
                "turnover_per_day": self.turnover_per_day,
            }
        )
        if self.digest and self.digest != expected:
            raise ValueError("causal alpha candidate evidence digest mismatch")
        object.__setattr__(self, "episode_metrics", metrics)
        object.__setattr__(self, "rejection_reasons", reasons)
        object.__setattr__(self, "digest", expected)


@dataclass(frozen=True, slots=True)
class CausalAlphaSelectionEvidence:
    candidates: tuple[CausalAlphaCandidateEvidence, ...]
    selected_candidate_digest: str
    grid_digest: str
    holdout_episode_digests: Mapping[str, str]
    lower_tail_definition: str = "minimum_symbol_episode_net_return"
    digest: str = ""

    def __post_init__(self) -> None:
        candidates = tuple(self.candidates)
        if not candidates:
            raise ValueError("causal alpha selection evidence needs candidates")
        selected = tuple(
            item for item in candidates if item.candidate.digest == self.selected_candidate_digest
        )
        if len(selected) != 1 or not selected[0].admissible:
            raise ValueError("causal alpha selected candidate is not uniquely admissible")
        if self.lower_tail_definition != "minimum_symbol_episode_net_return":
            raise ValueError("causal alpha lower-tail definition is unsupported")
        holdouts = dict(self.holdout_episode_digests)
        if any(not symbol or len(digest) != 64 for symbol, digest in holdouts.items()):
            raise ValueError("causal alpha holdout episode identities are invalid")
        expected = content_digest(
            {
                "candidate_evidence_digests": tuple(item.digest for item in candidates),
                "grid_digest": self.grid_digest,
                "holdout_episode_digests": holdouts,
                "lower_tail_definition": self.lower_tail_definition,
                "schema_version": "causal_alpha_selection_evidence_v1",
                "selected_candidate_digest": self.selected_candidate_digest,
            }
        )
        if self.digest and self.digest != expected:
            raise ValueError("causal alpha selection evidence digest mismatch")
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "holdout_episode_digests", holdouts)
        object.__setattr__(self, "digest", expected)
'''
text = text.replace(marker, selection_types + marker)

marker = "\ndef build_causal_alpha_episode_batch(\n"
if text.count(marker) != 1:
    raise SystemExit("selection function insertion marker drifted")
selection_functions = r'''

def _candidate_evidence(
    candidate: CausalAlphaCandidateConfig,
    metrics: tuple[CausalAlphaCandidateEpisodeMetrics, ...],
) -> CausalAlphaCandidateEvidence:
    if not metrics:
        raise ValueError("causal alpha candidate has no selection metrics")
    net_returns = np.asarray([item.net_return for item in metrics], dtype=np.float64)
    negative_gross = sum(item.gross_return < 0.0 for item in metrics)
    total_trades = sum(item.trade_count for item in metrics)
    risk_violation = any(item.risk_violation for item in metrics)
    reasons: list[str] = []
    if negative_gross > len(metrics) / 2.0:
        reasons.append("majority_negative_gross_return")
    if total_trades == 0:
        reasons.append("no_meaningful_trades")
    if risk_violation:
        reasons.append("risk_contract_violation")
    return CausalAlphaCandidateEvidence(
        candidate=candidate,
        episode_metrics=metrics,
        lower_tail_net_return=float(np.min(net_returns)),
        mean_net_return=float(np.mean(net_returns, dtype=np.float64)),
        turnover_per_day=float(
            np.mean([item.turnover_per_day for item in metrics], dtype=np.float64)
        ),
        total_execution_cost=float(
            np.sum([item.total_execution_cost for item in metrics], dtype=np.float64)
        ),
        negative_gross_episode_count=negative_gross,
        total_trade_count=total_trades,
        risk_violation=risk_violation,
        admissible=not reasons,
        rejection_reasons=tuple(reasons),
    )


def rank_causal_alpha_candidates(
    *,
    candidates: tuple[CausalAlphaCandidateConfig, ...],
    metrics: Mapping[str, tuple[CausalAlphaCandidateEpisodeMetrics, ...]],
    holdout_episode_digests: Mapping[str, str] | None = None,
) -> CausalAlphaSelectionEvidence:
    """Rank a complete candidate grid without consulting causal holdout metrics."""

    candidate_values = tuple(candidates)
    if not candidate_values:
        raise ValueError("causal alpha candidate grid must be non-empty")
    digests = tuple(candidate.digest for candidate in candidate_values)
    if len(set(digests)) != len(digests):
        raise ValueError("causal alpha candidate grid contains duplicate configs")
    if set(metrics) != set(digests):
        raise ValueError("causal alpha candidate metrics must cover the complete grid")
    evidence = tuple(
        _candidate_evidence(candidate, tuple(metrics[candidate.digest]))
        for candidate in candidate_values
    )
    admissible = tuple(item for item in evidence if item.admissible)
    if not admissible:
        raise RuntimeError("no admissible causal alpha candidate")
    selected = max(
        admissible,
        key=lambda item: (
            item.lower_tail_net_return,
            item.mean_net_return,
            -item.turnover_per_day,
            -item.total_execution_cost,
        ),
    )
    grid_digest = content_digest(
        {
            "candidate_digests": digests,
            "lower_tail_definition": "minimum_symbol_episode_net_return",
            "schema_version": "causal_alpha_selection_grid_v1",
        }
    )
    return CausalAlphaSelectionEvidence(
        candidates=evidence,
        selected_candidate_digest=selected.candidate.digest,
        grid_digest=grid_digest,
        holdout_episode_digests=(
            {} if holdout_episode_digests is None else dict(holdout_episode_digests)
        ),
    )


def _causal_alpha_target_for_contract(
    *,
    symbol: str,
    train_symbols: tuple[str, ...],
    samples: Mapping[str, CausalAlphaSymbolSamples],
    contract: OracleEpisodeContract,
    candidate: CausalAlphaCandidateConfig,
) -> np.ndarray:
    fitted = fit_expanding_causal_alpha_models(
        train_symbols=train_symbols,
        samples=samples,
        knowledge_cutoff=contract.start,
        ridge_config=candidate.ridge,
    )
    block = samples[symbol]
    if contract.dataset_id != block.dataset_id:
        raise ValueError("causal alpha selection contract dataset identity drifted")
    decisions = np.arange(contract.start, contract.stop - 1, dtype=np.int64)
    prediction_features = block.features_for_decisions(decisions)
    prediction_24h = fitted.model_24h.predict(prediction_features)
    prediction_72h = fitted.model_72h.predict(prediction_features)
    scores = combine_causal_alpha_predictions(
        prediction_24h,
        prediction_72h,
        candidate.controller.horizon_mix,
    )
    target_path = causal_alpha_target_path(
        scores,
        config=candidate.controller,
        initial_weight=float(contract.initial_weights[0]),
    )
    return np.asarray(target_path.targets, dtype=np.float32).reshape(-1, 1)


def evaluate_causal_alpha_selection(
    *,
    train_symbols: tuple[str, ...],
    samples: Mapping[str, CausalAlphaSymbolSamples],
    partitions: Mapping[str, CausalAlphaEpisodePartition],
    candidates: tuple[CausalAlphaCandidateConfig, ...],
    environment_factories: Mapping[str, Any],
    episode_hours: float,
) -> CausalAlphaSelectionEvidence:
    """Replay only earlier selection episodes through the production environment."""

    symbols, _, _ = _validated_sample_scope(train_symbols, samples)
    partition_values = validate_universal_causal_alpha_partitions(
        train_symbols=symbols,
        partitions=partitions,
    )
    if set(environment_factories) != set(symbols):
        raise ValueError("causal alpha environment factories must exactly match train_symbols")
    if not np.isfinite(episode_hours) or episode_hours <= 0.0:
        raise ValueError("causal alpha episode_hours must be finite and positive")
    episode_days = float(episode_hours) / 24.0
    by_candidate: dict[str, tuple[CausalAlphaCandidateEpisodeMetrics, ...]] = {}
    for candidate in candidates:
        records: list[CausalAlphaCandidateEpisodeMetrics] = []
        for symbol in symbols:
            factory = environment_factories[symbol]
            if not callable(factory):
                raise TypeError("causal alpha selection environment factory is not callable")
            partition = partition_values[symbol]
            for contract in partition.selection_contracts:
                actions = _causal_alpha_target_for_contract(
                    symbol=symbol,
                    train_symbols=symbols,
                    samples=samples,
                    contract=contract,
                    candidate=candidate,
                )
                evaluation = evaluate_episode_action_path(
                    factory,
                    contract,
                    actions=actions,
                )
                performance = evaluation.performance
                collapse = evaluation.collapse_evidence
                records.append(
                    CausalAlphaCandidateEpisodeMetrics(
                        candidate_digest=candidate.digest,
                        symbol=symbol,
                        episode_index=contract.episode_index,
                        gross_return=float(performance.gross_return),
                        net_return=float(performance.net_return),
                        turnover_per_day=(
                            float(performance.turnover_total) / episode_days
                        ),
                        total_execution_cost=float(performance.cost_total),
                        trade_count=int(performance.trade_count),
                        risk_violation=(
                            int(collapse.execution_rejection_count) > 0
                        ),
                    )
                )
        by_candidate[candidate.digest] = tuple(records)
    return rank_causal_alpha_candidates(
        candidates=tuple(candidates),
        metrics=by_candidate,
        holdout_episode_digests={
            symbol: partition_values[symbol].holdout_contract.digest
            for symbol in symbols
        },
    )
'''
text = text.replace(marker, selection_functions + marker)

old = """    "CausalAlphaBatchEvidence",
    "CausalAlphaEpisodeEvidence",
"""
new = """    "CausalAlphaBatchEvidence",
    "CausalAlphaCandidateConfig",
    "CausalAlphaCandidateEpisodeMetrics",
    "CausalAlphaCandidateEvidence",
    "CausalAlphaEpisodeEvidence",
"""
if text.count(old) != 1:
    raise SystemExit("selection type export target drifted")
text = text.replace(old, new)
old = """    "CausalAlphaSymbolSamples",
    "build_causal_alpha_episode_batch",
"""
new = """    "CausalAlphaSelectionEvidence",
    "CausalAlphaSymbolSamples",
    "build_causal_alpha_episode_batch",
"""
if text.count(old) != 1:
    raise SystemExit("selection evidence export target drifted")
text = text.replace(old, new)
old = """    "fit_expanding_causal_alpha_models",
    "latest_complete_episode_split",
"""
new = """    "evaluate_causal_alpha_selection",
    "fit_expanding_causal_alpha_models",
    "latest_complete_episode_split",
    "rank_causal_alpha_candidates",
"""
if text.count(old) != 1:
    raise SystemExit("selection function export target drifted")
text = text.replace(old, new)

path.write_text(text, encoding="utf-8")
