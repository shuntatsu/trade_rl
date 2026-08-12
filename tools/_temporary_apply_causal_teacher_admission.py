from pathlib import Path

# learning-domain admission evidence
learning = Path("trade_rl/learning/causal_alpha_teacher.py")
text = learning.read_text(encoding="utf-8")
marker = "\n\n__all__ = [\n"
if text.count(marker) != 1:
    raise SystemExit("causal alpha learning export marker drifted")
admission = r'''

@dataclass(frozen=True, slots=True)
class CausalAlphaTeacherHoldoutMetric:
    """One untouched train-symbol holdout replay for teacher pre-admission."""

    symbol: str
    gross_return: float
    net_return: float
    turnover_per_day: float
    total_execution_cost: float
    trade_count: int
    maximum_drawdown: float
    digest: str = ""

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("causal alpha teacher holdout symbol must be non-empty")
        for field, value in (
            ("gross_return", self.gross_return),
            ("net_return", self.net_return),
            ("turnover_per_day", self.turnover_per_day),
            ("total_execution_cost", self.total_execution_cost),
            ("maximum_drawdown", self.maximum_drawdown),
        ):
            if not math.isfinite(value):
                raise ValueError(f"causal alpha teacher {field} must be finite")
        if self.turnover_per_day < 0.0 or self.total_execution_cost < 0.0:
            raise ValueError("causal alpha teacher turnover/cost must be non-negative")
        if (
            isinstance(self.trade_count, bool)
            or not isinstance(self.trade_count, int)
            or self.trade_count < 0
        ):
            raise ValueError("causal alpha teacher trade_count must be non-negative")
        expected = content_digest(
            {
                "gross_return": self.gross_return,
                "maximum_drawdown": self.maximum_drawdown,
                "net_return": self.net_return,
                "schema_version": "causal_alpha_teacher_holdout_metric_v1",
                "symbol": self.symbol,
                "total_execution_cost": self.total_execution_cost,
                "trade_count": self.trade_count,
                "turnover_per_day": self.turnover_per_day,
            }
        )
        if self.digest and self.digest != expected:
            raise ValueError("causal alpha teacher holdout metric digest mismatch")
        object.__setattr__(self, "digest", expected)

    def to_payload(self) -> dict[str, object]:
        return {
            "artifact_digest": self.digest,
            "gross_return": self.gross_return,
            "maximum_drawdown": self.maximum_drawdown,
            "net_return": self.net_return,
            "schema_version": "causal_alpha_teacher_holdout_metric_v1",
            "symbol": self.symbol,
            "total_execution_cost": self.total_execution_cost,
            "trade_count": self.trade_count,
            "turnover_per_day": self.turnover_per_day,
        }


@dataclass(frozen=True, slots=True)
class CausalAlphaTeacherAdmissionEvidence:
    metrics: tuple[CausalAlphaTeacherHoldoutMetric, ...]
    aggregate_gross_return: float
    aggregate_net_return: float
    negative_gross_symbol_count: int
    passed: bool
    rejection_reasons: tuple[str, ...]
    digest: str = ""

    def __post_init__(self) -> None:
        metrics = tuple(self.metrics)
        if not metrics or len({item.symbol for item in metrics}) != len(metrics):
            raise ValueError("causal alpha teacher holdout symbols must be unique")
        if not math.isfinite(self.aggregate_gross_return) or not math.isfinite(
            self.aggregate_net_return
        ):
            raise ValueError("causal alpha teacher aggregate returns must be finite")
        if (
            isinstance(self.negative_gross_symbol_count, bool)
            or not isinstance(self.negative_gross_symbol_count, int)
            or not 0 <= self.negative_gross_symbol_count <= len(metrics)
        ):
            raise ValueError("causal alpha teacher negative holdout count is invalid")
        reasons = tuple(self.rejection_reasons)
        if self.passed == bool(reasons):
            raise ValueError("causal alpha teacher admission reasons are inconsistent")
        expected = content_digest(
            {
                "aggregate_gross_return": self.aggregate_gross_return,
                "aggregate_net_return": self.aggregate_net_return,
                "metric_digests": tuple(item.digest for item in metrics),
                "negative_gross_symbol_count": self.negative_gross_symbol_count,
                "passed": self.passed,
                "rejection_reasons": reasons,
                "schema_version": "causal_alpha_teacher_admission_v1",
            }
        )
        if self.digest and self.digest != expected:
            raise ValueError("causal alpha teacher admission digest mismatch")
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "rejection_reasons", reasons)
        object.__setattr__(self, "digest", expected)

    def to_payload(self) -> dict[str, object]:
        return {
            "aggregate_gross_return": self.aggregate_gross_return,
            "aggregate_net_return": self.aggregate_net_return,
            "artifact_digest": self.digest,
            "metrics": [item.to_payload() for item in self.metrics],
            "negative_gross_symbol_count": self.negative_gross_symbol_count,
            "passed": self.passed,
            "rejection_reasons": list(self.rejection_reasons),
            "schema_version": "causal_alpha_teacher_admission_v1",
        }


def evaluate_causal_alpha_teacher_admission(
    metrics: tuple[CausalAlphaTeacherHoldoutMetric, ...],
) -> CausalAlphaTeacherAdmissionEvidence:
    """Apply the maintained pre-BC teacher economics gate to untouched holdouts."""

    values = tuple(metrics)
    if not values or len({item.symbol for item in values}) != len(values):
        raise ValueError("causal alpha teacher holdout symbols must be unique")
    aggregate_gross = float(sum(item.gross_return for item in values))
    aggregate_net = float(sum(item.net_return for item in values))
    negative_count = sum(item.gross_return < 0.0 for item in values)
    reasons: list[str] = []
    if aggregate_gross < 0.0:
        reasons.append("negative_aggregate_gross_return")
    if negative_count > len(values) // 2:
        reasons.append("majority_negative_gross_holdouts")
    return CausalAlphaTeacherAdmissionEvidence(
        metrics=values,
        aggregate_gross_return=aggregate_gross,
        aggregate_net_return=aggregate_net,
        negative_gross_symbol_count=negative_count,
        passed=not reasons,
        rejection_reasons=tuple(reasons),
    )
'''
text = text.replace(marker, admission + marker)
old = '    "CausalAlphaTargetPath",\n'
new = '    "CausalAlphaTargetPath",\n    "CausalAlphaTeacherAdmissionEvidence",\n    "CausalAlphaTeacherHoldoutMetric",\n'
if text.count(old) != 1:
    raise SystemExit("causal alpha learning class exports drifted")
text = text.replace(old, new)
old = '    "fit_causal_alpha_ridge",\n    "forward_log_return_label",\n'
new = '    "evaluate_causal_alpha_teacher_admission",\n    "fit_causal_alpha_ridge",\n    "forward_log_return_label",\n'
if text.count(old) != 1:
    raise SystemExit("causal alpha learning function exports drifted")
learning.write_text(text, encoding="utf-8")

# selection payload + package episode hours
workflow = Path("trade_rl/workflows/universal_causal_alpha_teacher.py")
text = workflow.read_text(encoding="utf-8")
old = '''        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "holdout_episode_digests", holdouts)
        object.__setattr__(self, "digest", expected)


@dataclass(frozen=True, slots=True)
class UniversalCausalAlphaTeacherPackage:
'''
new = '''        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "holdout_episode_digests", holdouts)
        object.__setattr__(self, "digest", expected)

    def to_payload(self) -> dict[str, object]:
        return {
            "artifact_digest": self.digest,
            "candidates": [
                {
                    "admissible": item.admissible,
                    "candidate": {
                        "controller": {
                            "digest": item.candidate.controller.digest,
                            "entry_threshold": item.candidate.controller.entry_threshold,
                            "exit_threshold": item.candidate.controller.exit_threshold,
                            "horizon_mix": item.candidate.controller.horizon_mix.value,
                            "max_target_delta": item.candidate.controller.max_target_delta,
                            "no_trade_band": item.candidate.controller.no_trade_band,
                            "score_scale": item.candidate.controller.score_scale,
                        },
                        "digest": item.candidate.digest,
                        "name": item.candidate.name,
                        "ridge": {
                            "digest": item.candidate.ridge.digest,
                            "ridge_strength": item.candidate.ridge.ridge_strength,
                        },
                    },
                    "episode_metrics": [
                        {
                            "artifact_digest": metric.digest,
                            "episode_index": metric.episode_index,
                            "gross_return": metric.gross_return,
                            "net_return": metric.net_return,
                            "risk_violation": metric.risk_violation,
                            "symbol": metric.symbol,
                            "total_execution_cost": metric.total_execution_cost,
                            "trade_count": metric.trade_count,
                            "turnover_per_day": metric.turnover_per_day,
                        }
                        for metric in item.episode_metrics
                    ],
                    "lower_tail_net_return": item.lower_tail_net_return,
                    "mean_net_return": item.mean_net_return,
                    "negative_gross_episode_count": item.negative_gross_episode_count,
                    "rejection_reasons": list(item.rejection_reasons),
                    "risk_violation": item.risk_violation,
                    "total_execution_cost": item.total_execution_cost,
                    "total_trade_count": item.total_trade_count,
                    "turnover_per_day": item.turnover_per_day,
                }
                for item in self.candidates
            ],
            "grid_digest": self.grid_digest,
            "holdout_episode_digests": dict(self.holdout_episode_digests),
            "lower_tail_definition": self.lower_tail_definition,
            "schema_version": "causal_alpha_selection_evidence_v1",
            "selected_candidate_digest": self.selected_candidate_digest,
        }


@dataclass(frozen=True, slots=True)
class UniversalCausalAlphaTeacherPackage:
'''
if text.count(old) != 1:
    raise SystemExit("selection payload insertion target drifted")
text = text.replace(old, new)
old = '''    selected_candidate_digest: str
    teacher_config_digest: str
    batch_evidence: Mapping[str, CausalAlphaBatchEvidence]
    digest: str = ""
'''
new = '''    selected_candidate_digest: str
    teacher_config_digest: str
    episode_hours: float
    batch_evidence: Mapping[str, CausalAlphaBatchEvidence]
    digest: str = ""
'''
if text.count(old) != 1:
    raise SystemExit("package episode hours field target drifted")
text = text.replace(old, new)
old = '''        for field, value in (
            ("selected_candidate_digest", self.selected_candidate_digest),
            ("teacher_config_digest", self.teacher_config_digest),
        ):
'''
new = '''        if not np.isfinite(self.episode_hours) or self.episode_hours <= 0.0:
            raise ValueError("causal alpha package episode_hours must be positive")
        for field, value in (
            ("selected_candidate_digest", self.selected_candidate_digest),
            ("teacher_config_digest", self.teacher_config_digest),
        ):
'''
if text.count(old) != 1:
    raise SystemExit("package episode hours validation target drifted")
text = text.replace(old, new)
old = '''                "schema_version": "universal_causal_alpha_teacher_package_v1",
                "selected_candidate_digest": self.selected_candidate_digest,
'''
new = '''                "episode_hours": self.episode_hours,
                "schema_version": "universal_causal_alpha_teacher_package_v1",
                "selected_candidate_digest": self.selected_candidate_digest,
'''
if text.count(old) != 1:
    raise SystemExit("package episode hours digest target drifted")
text = text.replace(old, new)
old = '''        selection=selection,
        selected_candidate_digest=selected.digest,
        teacher_config_digest=teacher_config_digest,
        batch_evidence=batch_evidence,
'''
new = '''        selection=selection,
        selected_candidate_digest=selected.digest,
        teacher_config_digest=teacher_config_digest,
        episode_hours=resolved_episode_hours,
        batch_evidence=batch_evidence,
'''
if text.count(old) != 1:
    raise SystemExit("package builder episode hours target drifted")
text = text.replace(old, new)
workflow.write_text(text, encoding="utf-8")

# pass selection evidence through the workflow/integration boundary
runtime = Path("trade_rl/workflows/universal_teacher_runtime.py")
text = runtime.read_text(encoding="utf-8")
old = '''        episode_batches=dict(batches),
    )
'''
new = '''        episode_batches=dict(batches),
        causal_teacher_selection_evidence=(
            None
            if causal_teacher_package is None
            else causal_teacher_package.selection.to_payload()
        ),
        causal_teacher_episode_hours=(
            None
            if causal_teacher_package is None
            else causal_teacher_package.episode_hours
        ),
    )
'''
if text.count(old) != 1:
    raise SystemExit("runtime causal admission bundle target drifted")
runtime.write_text(text.replace(old, new), encoding="utf-8")

# integration bundle + pre-BC gate
integration = Path("trade_rl/integrations/universal_pretraining.py")
text = integration.read_text(encoding="utf-8")
old = '''from trade_rl.learning.direct_bc_evaluation import (
    evaluate_direct_behavior_cloning_gates,
)
'''
new = '''from trade_rl.learning.causal_alpha_teacher import (
    CausalAlphaTeacherHoldoutMetric,
    evaluate_causal_alpha_teacher_admission,
)
from trade_rl.learning.direct_bc_evaluation import (
    evaluate_direct_behavior_cloning_gates,
)
'''
if text.count(old) != 1:
    raise SystemExit("universal pretraining causal import target drifted")
text = text.replace(old, new)
old = '''    aggregate_episode_behavior_cloning_holdouts,
    evaluate_episode_behavior_cloning_holdout,
)
'''
new = '''    aggregate_episode_behavior_cloning_holdouts,
    evaluate_episode_action_path,
    evaluate_episode_behavior_cloning_holdout,
)
'''
if text.count(old) != 1:
    raise SystemExit("universal pretraining action-path import target drifted")
text = text.replace(old, new)
old = '''    teacher_artifact: UniversalTeacherArtifact
    episode_batches: Mapping[str, EpisodeOracleBatch] = field(default_factory=dict)
'''
new = '''    teacher_artifact: UniversalTeacherArtifact
    episode_batches: Mapping[str, EpisodeOracleBatch] = field(default_factory=dict)
    causal_teacher_selection_evidence: Mapping[str, object] | None = None
    causal_teacher_episode_hours: float | None = None
'''
if text.count(old) != 1:
    raise SystemExit("universal bundle causal fields target drifted")
text = text.replace(old, new)
old = '''        if self.teacher_artifact.train_symbols != symbols:
            raise ValueError("teacher artifact train symbol scope mismatch")
        object.__setattr__(self, "train_symbols", symbols)
'''
new = '''        if self.teacher_artifact.train_symbols != symbols:
            raise ValueError("teacher artifact train symbol scope mismatch")
        selection_evidence = self.causal_teacher_selection_evidence
        episode_hours = self.causal_teacher_episode_hours
        if selection_evidence is None:
            if episode_hours is not None:
                raise ValueError(
                    "causal teacher episode hours require selection evidence"
                )
        else:
            selection_evidence = dict(selection_evidence)
            if selection_evidence.get("schema_version") != "causal_alpha_selection_evidence_v1":
                raise ValueError("causal teacher selection evidence schema mismatch")
            artifact_digest = selection_evidence.get("artifact_digest")
            if not isinstance(artifact_digest, str) or len(artifact_digest) != 64:
                raise ValueError("causal teacher selection evidence digest is invalid")
            if episode_hours is None or not math.isfinite(episode_hours) or episode_hours <= 0.0:
                raise ValueError("causal teacher episode hours must be positive")
        object.__setattr__(self, "train_symbols", symbols)
'''
if text.count(old) != 1:
    raise SystemExit("universal bundle causal validation target drifted")
text = text.replace(old, new)
old = '''        object.__setattr__(self, "episode_batches", episode_batches)
        object.__setattr__(self, "critic_targets", targets)
'''
new = '''        object.__setattr__(self, "episode_batches", episode_batches)
        object.__setattr__(
            self, "causal_teacher_selection_evidence", selection_evidence
        )
        object.__setattr__(self, "causal_teacher_episode_hours", episode_hours)
        object.__setattr__(self, "critic_targets", targets)
'''
if text.count(old) != 1:
    raise SystemExit("universal bundle causal normalization target drifted")
text = text.replace(old, new)
old = '''    ) -> dict[str, object]:
        bc_config = _hierarchical_behavior_cloning_config(config)
'''
new = '''    ) -> dict[str, object]:
        if config.behavior_cloning_teacher == "causal_alpha_ridge":
            selection = bundle.causal_teacher_selection_evidence
            episode_hours = bundle.causal_teacher_episode_hours
            if selection is None or episode_hours is None:
                raise RuntimeError("Universal causal teacher selection evidence is unavailable")
            if not bundle.episode_batches:
                raise RuntimeError("Universal causal teacher episode batches are unavailable")
            if set(environment_factories) != set(bundle.train_symbols):
                raise RuntimeError(
                    "Universal causal teacher holdout environment factories are unavailable"
                )
            atomic_write_bytes(
                output_root / "causal-teacher-selection.json",
                canonical_json_bytes(selection) + b"\\n",
            )
            episode_days = episode_hours / 24.0
            teacher_metrics: list[CausalAlphaTeacherHoldoutMetric] = []
            for symbol in bundle.train_symbols:
                batch = bundle.episode_batches[symbol]
                if not batch.contracts or len(batch.targets) != len(batch.contracts):
                    raise RuntimeError(
                        f"Universal causal teacher holdout batch is invalid for {symbol}"
                    )
                evaluation = evaluate_episode_action_path(
                    environment_factories[symbol],
                    batch.contracts[-1],
                    actions=batch.targets[-1],
                )
                performance = evaluation.performance
                teacher_metrics.append(
                    CausalAlphaTeacherHoldoutMetric(
                        symbol=symbol,
                        gross_return=float(performance.gross_return),
                        net_return=float(performance.net_return),
                        turnover_per_day=float(performance.turnover_total) / episode_days,
                        total_execution_cost=float(performance.cost_total),
                        trade_count=int(performance.trade_count),
                        maximum_drawdown=float(performance.maximum_drawdown),
                    )
                )
            teacher_admission = evaluate_causal_alpha_teacher_admission(
                tuple(teacher_metrics)
            )
            atomic_write_bytes(
                output_root / "causal-teacher-admission.json",
                canonical_json_bytes(teacher_admission.to_payload()) + b"\\n",
            )
            if not teacher_admission.passed:
                raise RuntimeError(
                    "Universal causal teacher admission failed before behavior cloning"
                )
        bc_config = _hierarchical_behavior_cloning_config(config)
'''
if text.count(old) != 1:
    raise SystemExit("universal pretraining hook admission insertion target drifted")
text = text.replace(old, new)
integration.write_text(text, encoding="utf-8")

# test fixture requires canonical episode hours
integration_test = Path("tests/integrations/test_universal_causal_teacher_admission.py")
text = integration_test.read_text(encoding="utf-8")
old = '''        causal_teacher_selection_evidence={
            "schema_version": "causal_alpha_selection_evidence_v1",
            "artifact_digest": content_digest("selection-evidence"),
            "selected_candidate_digest": content_digest("candidate"),
        },
    )
'''
new = '''        causal_teacher_selection_evidence={
            "schema_version": "causal_alpha_selection_evidence_v1",
            "artifact_digest": content_digest("selection-evidence"),
            "selected_candidate_digest": content_digest("candidate"),
        },
        causal_teacher_episode_hours=720.0,
    )
'''
if text.count(old) != 1:
    raise SystemExit("teacher admission fixture episode hours target drifted")
integration_test.write_text(text.replace(old, new), encoding="utf-8")
