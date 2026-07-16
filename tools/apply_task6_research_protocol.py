from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"missing Task 6 anchor in {path}: {old[:160]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def append_once(path: str, marker: str, block: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    if marker not in text:
        target.write_text(text.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def add_tests() -> None:
    replace_once(
        "tests/examples/test_binance_multitimeframe_full_assets.py",
        "def test_full_walk_forward_config_has_two_material_folds() -> None:",
        "def test_full_walk_forward_config_has_six_material_folds() -> None:",
    )
    replace_once(
        "tests/examples/test_binance_multitimeframe_full_assets.py",
        '''    assert len(folds) == 2
    assert config.checkpoint_finalists_per_seed == 1
    assert (folds[0].test.start, folds[0].test.stop) == (52_512, 53_952)
    assert (folds[1].test.start, folds[1].test.stop) == (53_952, 55_392)
''',
        '''    assert len(folds) == 6
    assert config.checkpoint_finalists_per_seed == 1
    assert all(fold.test.size == 2_880 for fold in folds)
    assert sum(fold.test.size for fold in folds) == 17_280
    assert (folds[0].test.start, folds[0].test.stop) == (26_336, 29_216)
    assert (folds[-1].test.start, folds[-1].test.stop) == (40_736, 43_616)
''',
    )
    append_once(
        "tests/evaluation/test_research_gate.py",
        "test_strict_research_gate_requires_material_fold_and_oos_evidence",
        '''

def test_strict_research_gate_requires_material_fold_and_oos_evidence() -> None:
    from trade_rl.evaluation.research_gate import ResearchEvidenceRequirements

    requirements = ResearchEvidenceRequirements(
        required_fold_count=6,
        minimum_oos_days=180.0,
        require_positive_bootstrap_lower_bound=True,
    )
    result = _evaluate_research_return_gate(
        selected_mean_return=0.08,
        baseline_mean_return=0.03,
        maximum_fold_drawdown=0.12,
        selected_policy_digests=tuple(chr(97 + index) * 64 for index in range(6)),
        sealed_fold_count=6,
        oos_days=180.0,
        bootstrap_lower_bound=0.0001,
        requirements=requirements,
    )
    assert result.passed
    assert result.conditions["minimum_fold_count_met"]
    assert result.conditions["minimum_oos_days_met"]
    assert result.conditions["bootstrap_lower_bound_positive"]

    insufficient = _evaluate_research_return_gate(
        selected_mean_return=0.08,
        baseline_mean_return=0.03,
        maximum_fold_drawdown=0.12,
        selected_policy_digests=("a" * 64, "b" * 64),
        sealed_fold_count=2,
        oos_days=30.0,
        bootstrap_lower_bound=-0.001,
        requirements=requirements,
    )
    assert not insufficient.passed
    assert not insufficient.conditions["minimum_fold_count_met"]
    assert not insufficient.conditions["minimum_oos_days_met"]
    assert not insufficient.conditions["bootstrap_lower_bound_positive"]


def test_strict_research_gate_requires_fresh_confirmation_when_requested() -> None:
    from trade_rl.evaluation.research_gate import ResearchEvidenceRequirements

    requirements = ResearchEvidenceRequirements(
        required_fold_count=2,
        minimum_oos_days=1.0,
        require_positive_bootstrap_lower_bound=True,
        require_confirmation=True,
        minimum_confirmation_days=30.0,
    )
    result = _evaluate_research_return_gate(
        selected_mean_return=0.08,
        baseline_mean_return=0.03,
        maximum_fold_drawdown=0.12,
        selected_policy_digests=RL_POLICY_DIGESTS,
        sealed_fold_count=2,
        oos_days=30.0,
        bootstrap_lower_bound=0.001,
        confirmation_passed=False,
        confirmation_days=0.0,
        requirements=requirements,
    )
    assert not result.passed
    assert not result.conditions["fresh_confirmation_passed"]
    assert not result.conditions["minimum_confirmation_days_met"]


def test_block_bootstrap_lower_bound_is_deterministic_and_positive_for_growth() -> None:
    from trade_rl.evaluation.research_gate import block_bootstrap_mean_lower_bound

    daily = [0.001] * 200
    first = block_bootstrap_mean_lower_bound(daily, samples=500, seed=7)
    second = block_bootstrap_mean_lower_bound(daily, samples=500, seed=7)
    assert first == pytest.approx(second)
    assert first > 0.0
''',
    )
    append_once(
        "tests/examples/test_binance_multitimeframe_full_assets.py",
        "test_full_runner_preserves_all_selected_recipe_seeds",
        '''

def test_full_runner_preserves_all_selected_recipe_seeds(tmp_path: Path) -> None:
    namespace = _runner_namespace()
    select_recipe = namespace["_selected_walk_forward_recipe"]
    walk_forward_path = tmp_path / "wf"
    _write_walk_forward(walk_forward_path, selected=0.04, baseline=0.01)
    selected_name, seeds, output = select_recipe(
        walk_forward_path,
        EXAMPLE_ROOT / "walk-forward-full.json",
        tmp_path / "selected.json",
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert selected_name == "oracle-bc-ppo-15m-target"
    assert seeds == (0, 1, 2)
    assert payload["training"]["seeds"] == [0, 1, 2]


def test_full_runner_strict_gate_rejects_two_short_folds(tmp_path: Path) -> None:
    namespace = _runner_namespace()
    finalize = namespace["_finalize_research_run"]
    walk_forward_path = tmp_path / "wf"
    _write_walk_forward(walk_forward_path, selected=0.04, baseline=0.01)
    exit_code = finalize(
        work_root=tmp_path,
        walk_forward_path=walk_forward_path,
        summary={"production_status": "NO-GO"},
        strict=True,
    )
    gate = json.loads((tmp_path / "research-gate.json").read_text(encoding="utf-8"))
    assert exit_code != 0
    assert gate["conditions"]["minimum_fold_count_met"] is False
    assert gate["conditions"]["minimum_oos_days_met"] is False
''',
    )


def add_implementation() -> None:
    (ROOT / "trade_rl/evaluation/research_gate.py").write_text(
        '''"""Fail-closed profitability gates for sealed walk-forward evidence."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

from trade_rl.domain.common import require_sha256

PolicyIdentity: TypeAlias = str | None
ObservedValue: TypeAlias = object

_BASE_THRESHOLDS = {
    "selected_mean_return_exclusive_minimum": 0.0,
    "baseline_uplift_minimum": 0.0,
    "maximum_independently_reset_fold_drawdown": 0.20,
    "maximum_turnover_per_day": 1.0,
    "maximum_cost_fraction": 0.03,
}


@dataclass(frozen=True, slots=True)
class ResearchEvidenceRequirements:
    """Materiality requirements applied only to maintained research runs."""

    required_fold_count: int = 2
    minimum_oos_days: float = 0.0
    require_positive_bootstrap_lower_bound: bool = False
    require_confirmation: bool = False
    minimum_confirmation_days: float = 0.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.required_fold_count, bool)
            or not isinstance(self.required_fold_count, int)
            or self.required_fold_count <= 0
        ):
            raise ValueError("required_fold_count must be a positive integer")
        for name, value in (
            ("minimum_oos_days", self.minimum_oos_days),
            ("minimum_confirmation_days", self.minimum_confirmation_days),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name, value in (
            (
                "require_positive_bootstrap_lower_bound",
                self.require_positive_bootstrap_lower_bound,
            ),
            ("require_confirmation", self.require_confirmation),
        ):
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a boolean")


@dataclass(frozen=True, slots=True)
class ResearchReturnGate:
    thresholds: dict[str, float]
    observed: dict[str, ObservedValue]
    conditions: dict[str, bool]
    passed: bool
    evidence_errors: tuple[str, ...]


def block_bootstrap_mean_lower_bound(
    daily_returns: object,
    *,
    confidence: float = 0.95,
    samples: int = 2_000,
    block_size: int = 5,
    seed: int = 0,
) -> float:
    """Deterministic circular block-bootstrap lower bound on mean daily growth."""

    values = np.asarray(daily_returns, dtype=np.float64).reshape(-1)
    if values.size < 2 or not np.isfinite(values).all() or np.any(values < -1.0):
        raise ValueError("daily_returns must contain at least two finite returns")
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must be within (0.5, 1)")
    for name, value in (("samples", samples), ("block_size", block_size)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    log_returns = np.log1p(values)
    rng = np.random.default_rng(seed)
    block_count = math.ceil(values.size / block_size)
    means = np.empty(samples, dtype=np.float64)
    offsets = np.arange(block_size, dtype=np.int64)
    for sample_index in range(samples):
        starts = rng.integers(0, values.size, size=block_count)
        indices = ((starts[:, None] + offsets[None, :]) % values.size).reshape(-1)
        means[sample_index] = float(np.mean(log_returns[indices[: values.size]]))
    lower_log = float(np.quantile(means, 1.0 - confidence))
    return float(np.expm1(lower_log))


def _finite_number(
    value: object, *, field_name: str
) -> tuple[float | None, str | None]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None, f"{field_name} must be a finite number"
    try:
        resolved = float(value)
    except (OverflowError, TypeError, ValueError):
        return None, f"{field_name} must be a finite number"
    if not math.isfinite(resolved):
        return None, f"{field_name} must be a finite number"
    return resolved, None


def _positive_integer(
    value: object, *, field_name: str
) -> tuple[int | None, str | None]:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None, f"{field_name} must be a positive integer"
    return value, None


def _total_return(
    value: object, *, field_name: str
) -> tuple[float | None, str | None]:
    resolved, error = _finite_number(value, field_name=field_name)
    if resolved is not None and resolved < -1.0:
        return None, f"{field_name} must be greater than or equal to -1"
    return resolved, error


def _selected_policy_identities(
    value: object,
    *,
    expected_count: int,
) -> tuple[tuple[PolicyIdentity, ...] | None, tuple[str, ...]]:
    if not isinstance(value, (list, tuple)):
        return None, ("selected_policy_digests must be a sequence",)
    identities: list[PolicyIdentity] = []
    canonical_identities: list[str] = []
    errors: list[str] = []
    if len(value) != expected_count:
        errors.append(
            "selected_policy_digests must contain exactly "
            f"{expected_count} identities"
        )
    for index, identity in enumerate(value):
        if not isinstance(identity, str) or not identity:
            identities.append(None)
            errors.append(
                f"selected_policy_digests[{index}] must be a non-empty string"
            )
        else:
            identities.append(identity)
            try:
                canonical_identities.append(
                    require_sha256(identity, field=f"selected_policy_digests[{index}]")
                )
            except ValueError as exc:
                errors.append(str(exc))
    if len(canonical_identities) == expected_count and len(
        set(canonical_identities)
    ) != len(canonical_identities):
        errors.append("selected_policy_digests must contain unique identities")
    return tuple(identities), tuple(errors)


def evaluate_research_return_gate(
    *,
    selected_mean_return: object,
    baseline_mean_return: object,
    maximum_fold_drawdown: object,
    selected_policy_digests: object,
    maximum_turnover_per_day: object = 0.0,
    maximum_cost_fraction: object = 0.0,
    selection_stability_passed: object = True,
    sealed_fold_count: object = None,
    oos_days: object = None,
    bootstrap_lower_bound: object = None,
    confirmation_passed: object = None,
    confirmation_days: object = None,
    requirements: ResearchEvidenceRequirements | None = None,
) -> ResearchReturnGate:
    """Evaluate base or material research thresholds without raising on evidence."""

    strict = requirements is not None
    resolved_requirements = requirements or ResearchEvidenceRequirements()
    fold_count: int | None = None
    fold_count_error: str | None = None
    if strict:
        fold_count, fold_count_error = _positive_integer(
            sealed_fold_count, field_name="sealed_fold_count"
        )
    expected_count = (
        fold_count
        if strict and fold_count is not None
        else resolved_requirements.required_fold_count
    )
    selected, selected_error = _total_return(
        selected_mean_return, field_name="selected_mean_return"
    )
    baseline, baseline_error = _total_return(
        baseline_mean_return, field_name="baseline_mean_return"
    )
    drawdown, drawdown_error = _finite_number(
        maximum_fold_drawdown, field_name="maximum_fold_drawdown"
    )
    if drawdown is not None and not 0.0 <= drawdown <= 1.0:
        drawdown = None
        drawdown_error = "maximum_fold_drawdown must be between 0 and 1"
    uplift = None if selected is None or baseline is None else selected - baseline
    uplift_error = None
    if uplift is not None and not math.isfinite(uplift):
        uplift = None
        uplift_error = "baseline_uplift must be a finite number"
    policy_identities, policy_identity_errors = _selected_policy_identities(
        selected_policy_digests,
        expected_count=expected_count,
    )
    turnover, turnover_error = _finite_number(
        maximum_turnover_per_day, field_name="maximum_turnover_per_day"
    )
    if turnover is not None and turnover < 0.0:
        turnover = None
        turnover_error = "maximum_turnover_per_day must be non-negative"
    cost_fraction, cost_error = _finite_number(
        maximum_cost_fraction, field_name="maximum_cost_fraction"
    )
    if cost_fraction is not None and cost_fraction < 0.0:
        cost_fraction = None
        cost_error = "maximum_cost_fraction must be non-negative"
    stability_error = (
        None
        if isinstance(selection_stability_passed, bool)
        else "selection_stability_passed must be a boolean"
    )

    strict_errors: list[str] = []
    oos: float | None = None
    bootstrap: float | None = None
    confirmation_duration: float | None = None
    if strict:
        if fold_count_error is not None:
            strict_errors.append(fold_count_error)
        oos, error = _finite_number(oos_days, field_name="oos_days")
        if error is not None:
            strict_errors.append(error)
        elif oos is not None and oos < 0.0:
            oos = None
            strict_errors.append("oos_days must be non-negative")
        if resolved_requirements.require_positive_bootstrap_lower_bound:
            bootstrap, error = _finite_number(
                bootstrap_lower_bound, field_name="bootstrap_lower_bound"
            )
            if error is not None:
                strict_errors.append(error)
        if resolved_requirements.require_confirmation:
            if not isinstance(confirmation_passed, bool):
                strict_errors.append("confirmation_passed must be a boolean")
            confirmation_duration, error = _finite_number(
                confirmation_days, field_name="confirmation_days"
            )
            if error is not None:
                strict_errors.append(error)
            elif confirmation_duration is not None and confirmation_duration < 0.0:
                confirmation_duration = None
                strict_errors.append("confirmation_days must be non-negative")

    errors = (
        tuple(
            error
            for error in (
                selected_error,
                baseline_error,
                drawdown_error,
                uplift_error,
                turnover_error,
                cost_error,
                stability_error,
            )
            if error is not None
        )
        + policy_identity_errors
        + tuple(strict_errors)
    )
    evidence_valid = not errors
    conditions = {
        "selected_mean_return_positive": selected is not None and selected > 0.0,
        "baseline_uplift_nonnegative": uplift is not None and uplift >= 0.0,
        "maximum_fold_drawdown_within_limit": (
            drawdown is not None
            and 0.0 <= drawdown
            <= _BASE_THRESHOLDS["maximum_independently_reset_fold_drawdown"]
        ),
        "rl_policy_selected_all_folds": (
            policy_identities is not None and not policy_identity_errors
        ),
        "turnover_within_limit": (
            turnover is not None
            and turnover <= _BASE_THRESHOLDS["maximum_turnover_per_day"]
        ),
        "cost_fraction_within_limit": (
            cost_fraction is not None
            and cost_fraction <= _BASE_THRESHOLDS["maximum_cost_fraction"]
        ),
        "selection_stability_passed": selection_stability_passed is True,
        "evidence_valid": evidence_valid,
    }
    thresholds = dict(_BASE_THRESHOLDS)
    observed: dict[str, ObservedValue] = {
        "selected_mean_return": selected,
        "baseline_mean_return": baseline,
        "baseline_uplift": uplift,
        "maximum_independently_reset_fold_drawdown": drawdown,
        "selected_policy_digests": policy_identities,
        "maximum_turnover_per_day": turnover,
        "maximum_cost_fraction": cost_fraction,
        "selection_stability_passed": (
            selection_stability_passed
            if isinstance(selection_stability_passed, bool)
            else None
        ),
    }
    if strict:
        thresholds.update(
            {
                "minimum_sealed_fold_count": float(
                    resolved_requirements.required_fold_count
                ),
                "minimum_oos_days": resolved_requirements.minimum_oos_days,
            }
        )
        conditions["minimum_fold_count_met"] = (
            fold_count is not None
            and fold_count >= resolved_requirements.required_fold_count
        )
        conditions["minimum_oos_days_met"] = (
            oos is not None and oos >= resolved_requirements.minimum_oos_days
        )
        observed["sealed_fold_count"] = fold_count
        observed["oos_days"] = oos
        if resolved_requirements.require_positive_bootstrap_lower_bound:
            thresholds["bootstrap_lower_bound_exclusive_minimum"] = 0.0
            conditions["bootstrap_lower_bound_positive"] = (
                bootstrap is not None and bootstrap > 0.0
            )
            observed["bootstrap_lower_bound"] = bootstrap
        if resolved_requirements.require_confirmation:
            thresholds["minimum_confirmation_days"] = (
                resolved_requirements.minimum_confirmation_days
            )
            conditions["fresh_confirmation_passed"] = confirmation_passed is True
            conditions["minimum_confirmation_days_met"] = (
                confirmation_duration is not None
                and confirmation_duration
                >= resolved_requirements.minimum_confirmation_days
            )
            observed["confirmation_passed"] = (
                confirmation_passed if isinstance(confirmation_passed, bool) else None
            )
            observed["confirmation_days"] = confirmation_duration
    return ResearchReturnGate(
        thresholds=thresholds,
        observed=observed,
        conditions=conditions,
        passed=all(conditions.values()),
        evidence_errors=errors,
    )


__all__ = [
    "ResearchEvidenceRequirements",
    "ResearchReturnGate",
    "block_bootstrap_mean_lower_bound",
    "evaluate_research_return_gate",
]
''',
        encoding="utf-8",
    )

    replace_once(
        "examples/binance-multitimeframe/run_full_research.py",
        '''from trade_rl.evaluation.research_gate import (
    ResearchReturnGate,
    evaluate_research_return_gate,
)
''',
        '''from trade_rl.evaluation.research_gate import (
    ResearchEvidenceRequirements,
    ResearchReturnGate,
    block_bootstrap_mean_lower_bound,
    evaluate_research_return_gate,
)
''',
    )
    replace_once(
        "examples/binance-multitimeframe/run_full_research.py",
        '''def _selection_stability_passed(folds: object) -> bool:
    if not isinstance(folds, list) or not folds:
        return False
    selected_configurations: list[str] = []
    selected_seeds: list[int] = []
''',
        '''def _selection_stability_passed(folds: object) -> bool:
    if not isinstance(folds, list) or not folds:
        return False
    selected_configurations: list[str] = []
''',
    )
    replace_once(
        "examples/binance-multitimeframe/run_full_research.py",
        '''        selected_seed = fold.get("selected_seed")
        aggregates = fold.get("candidate_aggregates")
        if (
            not isinstance(selected, str)
            or selected == "baseline"
            or isinstance(selected_seed, bool)
            or not isinstance(selected_seed, int)
            or selected_seed < 0
        ):
            return False
        selected_configurations.append(selected)
        selected_seeds.append(selected_seed)
''',
        '''        aggregates = fold.get("candidate_aggregates")
        if not isinstance(selected, str) or selected == "baseline":
            return False
        selected_configurations.append(selected)
''',
    )
    replace_once(
        "examples/binance-multitimeframe/run_full_research.py",
        '''    return len(set(selected_configurations)) == 1 and len(set(selected_seeds)) == 1


def _evaluate_walk_forward_research_gate(path: Path) -> ResearchReturnGate:
''',
        '''    return len(set(selected_configurations)) == 1


def _selected_daily_returns(folds: object) -> tuple[float, ...] | None:
    if not isinstance(folds, list) or not folds:
        return None
    periods_per_day = 96
    daily: list[float] = []
    for fold in folds:
        if not isinstance(fold, dict):
            return None
        raw_returns = fold.get("selected_returns")
        if not isinstance(raw_returns, (list, tuple)):
            return None
        values: list[float] = []
        for raw in raw_returns:
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                return None
            value = float(raw)
            if not np.isfinite(value) or value < -1.0:
                return None
            values.append(value)
        if len(values) % periods_per_day != 0:
            return None
        for offset in range(0, len(values), periods_per_day):
            wealth = 1.0
            for value in values[offset : offset + periods_per_day]:
                wealth *= 1.0 + value
            if not np.isfinite(wealth):
                return None
            daily.append(wealth - 1.0)
    return tuple(daily)


def _confirmation_evidence(
    path: Path | None,
    *,
    expected_policy_digest: str | None,
) -> tuple[bool, float]:
    if path is None or not path.is_file():
        return False, 0.0
    try:
        payload = _load_json(path)
    except (OSError, ValueError):
        return False, 0.0
    days = payload.get("days")
    total_return = payload.get("total_return")
    maximum_drawdown = payload.get("maximum_drawdown")
    policy_digest = payload.get("policy_digest")
    if (
        payload.get("schema_version") != "fresh_confirmation_evidence_v1"
        or payload.get("sealed") is not True
        or isinstance(days, bool)
        or not isinstance(days, (int, float))
        or isinstance(total_return, bool)
        or not isinstance(total_return, (int, float))
        or isinstance(maximum_drawdown, bool)
        or not isinstance(maximum_drawdown, (int, float))
        or not isinstance(policy_digest, str)
        or len(policy_digest) != 64
    ):
        return False, 0.0
    resolved_days = float(days)
    passed = (
        np.isfinite(resolved_days)
        and resolved_days >= 0.0
        and np.isfinite(float(total_return))
        and float(total_return) > 0.0
        and np.isfinite(float(maximum_drawdown))
        and 0.0 <= float(maximum_drawdown) <= 0.20
        and (
            expected_policy_digest is None
            or policy_digest == expected_policy_digest
        )
    )
    return passed, resolved_days


def _evaluate_walk_forward_research_gate(
    path: Path,
    *,
    strict: bool = False,
    require_confirmation: bool = False,
    confirmation_path: Path | None = None,
    expected_policy_digest: str | None = None,
) -> ResearchReturnGate:
''',
    )
    replace_once(
        "examples/binance-multitimeframe/run_full_research.py",
        '''    folds = payload.get("folds")
    return evaluate_research_return_gate(
''',
        '''    folds = payload.get("folds")
    requirements = None
    fold_count = None
    oos_days = None
    bootstrap_lower_bound = None
    confirmation_passed = None
    confirmation_days = None
    if strict:
        requirements = ResearchEvidenceRequirements(
            required_fold_count=6,
            minimum_oos_days=180.0,
            require_positive_bootstrap_lower_bound=True,
            require_confirmation=require_confirmation,
            minimum_confirmation_days=30.0,
        )
        fold_count = len(folds) if isinstance(folds, list) else None
        daily_returns = _selected_daily_returns(folds)
        if daily_returns is not None:
            oos_days = float(len(daily_returns))
            if len(daily_returns) >= 2:
                bootstrap_lower_bound = block_bootstrap_mean_lower_bound(
                    daily_returns,
                    samples=2_000,
                    block_size=5,
                    seed=0,
                )
        if require_confirmation:
            confirmation_passed, confirmation_days = _confirmation_evidence(
                confirmation_path,
                expected_policy_digest=expected_policy_digest,
            )
    return evaluate_research_return_gate(
''',
    )
    replace_once(
        "examples/binance-multitimeframe/run_full_research.py",
        '''        selection_stability_passed=_selection_stability_passed(folds),
    )
''',
        '''        selection_stability_passed=_selection_stability_passed(folds),
        sealed_fold_count=fold_count,
        oos_days=oos_days,
        bootstrap_lower_bound=bootstrap_lower_bound,
        confirmation_passed=confirmation_passed,
        confirmation_days=confirmation_days,
        requirements=requirements,
    )
''',
    )
    replace_once(
        "examples/binance-multitimeframe/run_full_research.py",
        ''') -> tuple[str, int, Path]:
''',
        ''') -> tuple[str, tuple[int, ...], Path]:
''',
    )
    replace_once(
        "examples/binance-multitimeframe/run_full_research.py",
        '''    selected_seeds = tuple(
        fold.get("selected_seed") for fold in folds if isinstance(fold, dict)
    )
''',
        '''
''',
    )
    replace_once(
        "examples/binance-multitimeframe/run_full_research.py",
        '''    if len(selected_seeds) != len(folds) or any(
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
        for seed in selected_seeds
    ):
        raise RuntimeError("walk-forward selected seed evidence is invalid")
    if len(set(selected_seeds)) != 1:
        raise RuntimeError("walk-forward folds did not agree on one final seed")
    selected_name = str(selected[0])
    selected_seed = int(selected_seeds[0])
''',
        '''    selected_name = str(selected[0])
''',
    )
    replace_once(
        "examples/binance-multitimeframe/run_full_research.py",
        '''    training = dict(training)
    training["seeds"] = [selected_seed]
    selected_run["training"] = training
    _write_json(output_path, selected_run)
    return selected_name, selected_seed, output_path
''',
        '''    training = dict(training)
    raw_seeds = training.get("seeds")
    if not isinstance(raw_seeds, list) or len(raw_seeds) < 2 or any(
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
        for seed in raw_seeds
    ):
        raise RuntimeError("selected training recipe requires multiple fixed seeds")
    seeds = tuple(int(seed) for seed in raw_seeds)
    selected_run["training"] = training
    _write_json(output_path, selected_run)
    return selected_name, seeds, output_path
''',
    )
    replace_once(
        "examples/binance-multitimeframe/run_full_research.py",
        '''def _finalize_research_run(
    *,
    work_root: Path,
    walk_forward_path: Path,
    summary: dict[str, object],
) -> int:
    gate = asdict(_evaluate_walk_forward_research_gate(walk_forward_path))
''',
        '''def _finalize_research_run(
    *,
    work_root: Path,
    walk_forward_path: Path,
    summary: dict[str, object],
    strict: bool = False,
    require_confirmation: bool = False,
    expected_policy_digest: str | None = None,
) -> int:
    gate = asdict(
        _evaluate_walk_forward_research_gate(
            walk_forward_path,
            strict=strict,
            require_confirmation=require_confirmation,
            confirmation_path=work_root / "confirmation-evidence.json",
            expected_policy_digest=expected_policy_digest,
        )
    )
''',
    )
    replace_once(
        "examples/binance-multitimeframe/run_full_research.py",
        '''    preliminary_gate = _evaluate_walk_forward_research_gate(walk_forward_path)
''',
        '''    preliminary_gate = _evaluate_walk_forward_research_gate(
        walk_forward_path,
        strict=True,
    )
''',
    )
    replace_once(
        "examples/binance-multitimeframe/run_full_research.py",
        '''            summary=summary,
        )
''',
        '''            summary=summary,
            strict=True,
        )
''',
    )
    replace_once(
        "examples/binance-multitimeframe/run_full_research.py",
        '''        selected_configuration,
        selected_seed,
        training_config_path,
''',
        '''        selected_configuration,
        selected_seeds,
        training_config_path,
''',
    )
    replace_once(
        "examples/binance-multitimeframe/run_full_research.py",
        '''    summary["selected_training_configuration"] = selected_configuration
    summary["selected_training_seed"] = selected_seed
    summary["training"] = training
    exit_code = _finalize_research_run(
        work_root=work_root,
        walk_forward_path=walk_forward_path,
        summary=summary,
    )
''',
        '''    summary["selected_training_configuration"] = selected_configuration
    summary["selected_training_seeds"] = list(selected_seeds)
    summary["confirmation_required_from"] = _END
    summary["training"] = training
    expected_policy_digest = training.get("artifact_digest")
    exit_code = _finalize_research_run(
        work_root=work_root,
        walk_forward_path=walk_forward_path,
        summary=summary,
        strict=True,
        require_confirmation=True,
        expected_policy_digest=(
            str(expected_policy_digest)
            if isinstance(expected_policy_digest, str)
            else None
        ),
    )
''',
    )

    replace_once(
        "examples/binance-multitimeframe/walk-forward-full.json",
        '''    "train_bars": 43936,
    "checkpoint_bars": 4000,
    "selection_bars": 4000,
    "test_bars": 1440,
    "purge_bars": 192,
    "step_bars": 1440,
    "max_folds": 2,
''',
        '''    "train_bars": 20000,
    "checkpoint_bars": 2880,
    "selection_bars": 2880,
    "test_bars": 2880,
    "purge_bars": 192,
    "step_bars": 2880,
    "max_folds": 6,
''',
    )


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in {"tests", "implementation"}:
        raise SystemExit("usage: apply_task6_research_protocol.py tests|implementation")
    if sys.argv[1] == "tests":
        add_tests()
    else:
        add_implementation()
