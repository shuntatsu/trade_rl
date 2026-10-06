"""Pure common-contract evidence matrix for Issue #810 allocation candidates.

This owner compares already-produced evidence only. It does not execute a strategy,
load a policy, open data, select a winner, mutate a Study, or authorize trading.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from statistics import median
from typing import Any

from trade_rl._validation import require_non_empty, require_sha256
from trade_rl.artifacts import canonical_json_bytes, content_digest


class AllocationCandidateKind(str, Enum):
    NONRL = "nonrl"
    RESIDUAL_PPO = "residual_ppo"
    DIRECT_PPO = "direct_ppo"


class AllocationValidity(str, Enum):
    VALID = "valid"
    INVALID = "invalid"


_CANDIDATE_ROSTER = (
    AllocationCandidateKind.NONRL,
    AllocationCandidateKind.RESIDUAL_PPO,
    AllocationCandidateKind.DIRECT_PPO,
)
_ACCOUNT_MODES = ("independent_symbol", "shared_portfolio")


def _native_int(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


def _closed_mapping(value: object, keys: set[str], field: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise ValueError(f"{field} must contain exactly its declared fields")
    return value


@dataclass(frozen=True, slots=True)
class AllocationComparisonScenario:
    name: str
    digest: str
    required: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "name", require_non_empty(self.name, field="scenario name")
        )
        require_sha256(self.digest, field="scenario digest")
        if type(self.required) is not bool:
            raise ValueError("scenario required must be a native bool")

    def payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "digest": self.digest,
            "required": self.required,
        }

    @classmethod
    def from_payload(cls, value: object) -> AllocationComparisonScenario:
        data = _closed_mapping(value, {"name", "digest", "required"}, "scenario")
        try:
            result = cls(
                name=data["name"],
                digest=data["digest"],
                required=data["required"],
            )
        except TypeError as error:
            raise ValueError("allocation comparison scenario is malformed") from error
        if canonical_json_bytes(result.payload()) != canonical_json_bytes(value):
            raise ValueError("scenario differs from its canonical declaration")
        return result


@dataclass(frozen=True, slots=True)
class AllocationComparisonContract:
    dataset_id: str
    objective_digest: str
    clock_digest: str
    forecast_context_digest: str
    economics_digest: str
    risk_digest: str
    fold_plan_digest: str
    nonrl_recipe_digest: str
    residual_recipe_digest: str
    direct_recipe_digest: str
    account_mode: str
    initial_capital: float
    scenarios: tuple[AllocationComparisonScenario, ...]
    rl_seeds: tuple[int, ...]
    maximum_drawdown: float

    def __post_init__(self) -> None:
        for field in (
            "dataset_id",
            "objective_digest",
            "clock_digest",
            "forecast_context_digest",
            "economics_digest",
            "risk_digest",
            "fold_plan_digest",
            "nonrl_recipe_digest",
            "residual_recipe_digest",
            "direct_recipe_digest",
        ):
            require_sha256(getattr(self, field), field=field)
        if self.account_mode not in _ACCOUNT_MODES:
            raise ValueError("unsupported allocation comparison account_mode")
        capital = _finite(self.initial_capital, "initial_capital")
        if capital <= 0.0:
            raise ValueError("initial_capital must be positive")
        object.__setattr__(self, "initial_capital", capital)

        if (
            type(self.scenarios) is not tuple
            or not self.scenarios
            or any(
                type(value) is not AllocationComparisonScenario
                for value in self.scenarios
            )
        ):
            raise ValueError("comparison scenarios must be a non-empty immutable tuple")
        names = tuple(value.name for value in self.scenarios)
        digests = tuple(value.digest for value in self.scenarios)
        if len(set(names)) != len(names) or len(set(digests)) != len(digests):
            raise ValueError("comparison scenarios must have unique names and digests")
        base = [value for value in self.scenarios if value.name == "base"]
        if len(base) != 1 or not base[0].required:
            raise ValueError("comparison contract requires one required base scenario")

        if type(self.rl_seeds) is not tuple or not self.rl_seeds:
            raise ValueError("rl_seeds must be a non-empty immutable tuple")
        normalized = tuple(_native_int(value, "rl seed") for value in self.rl_seeds)
        if normalized != tuple(sorted(set(normalized))):
            raise ValueError("rl_seeds must be unique and strictly increasing")
        object.__setattr__(self, "rl_seeds", normalized)

        drawdown = _finite(self.maximum_drawdown, "maximum_drawdown")
        if not 0.0 < drawdown <= 1.0:
            raise ValueError("maximum_drawdown must be within (0, 1]")
        object.__setattr__(self, "maximum_drawdown", drawdown)

    @property
    def digest(self) -> str:
        return content_digest(self.payload())

    def payload(self) -> dict[str, object]:
        return {
            "schema": "allocation_comparison_contract_v1",
            "dataset_id": self.dataset_id,
            "objective_digest": self.objective_digest,
            "clock_digest": self.clock_digest,
            "forecast_context_digest": self.forecast_context_digest,
            "economics_digest": self.economics_digest,
            "risk_digest": self.risk_digest,
            "fold_plan_digest": self.fold_plan_digest,
            "candidate_recipes": {
                "nonrl": self.nonrl_recipe_digest,
                "residual_ppo": self.residual_recipe_digest,
                "direct_ppo": self.direct_recipe_digest,
            },
            "account_mode": self.account_mode,
            "initial_capital": self.initial_capital,
            "scenarios": [value.payload() for value in self.scenarios],
            "rl_seeds": list(self.rl_seeds),
            "maximum_drawdown": self.maximum_drawdown,
            "candidate_roster": [value.value for value in _CANDIDATE_ROSTER],
        }

    @classmethod
    def from_payload(cls, value: object) -> AllocationComparisonContract:
        data = _closed_mapping(
            value,
            {
                "schema",
                "dataset_id",
                "objective_digest",
                "clock_digest",
                "forecast_context_digest",
                "economics_digest",
                "risk_digest",
                "fold_plan_digest",
                "candidate_recipes",
                "account_mode",
                "initial_capital",
                "scenarios",
                "rl_seeds",
                "maximum_drawdown",
                "candidate_roster",
            },
            "allocation comparison contract",
        )
        if data["schema"] != "allocation_comparison_contract_v1":
            raise ValueError("unsupported allocation comparison contract schema")
        if data["candidate_roster"] != [value.value for value in _CANDIDATE_ROSTER]:
            raise ValueError("allocation comparison candidate roster changed")
        recipes = _closed_mapping(
            data["candidate_recipes"],
            {"nonrl", "residual_ppo", "direct_ppo"},
            "candidate_recipes",
        )
        if type(data["scenarios"]) is not list or type(data["rl_seeds"]) is not list:
            raise ValueError("comparison contract arrays must be native lists")
        try:
            result = cls(
                dataset_id=data["dataset_id"],
                objective_digest=data["objective_digest"],
                clock_digest=data["clock_digest"],
                forecast_context_digest=data["forecast_context_digest"],
                economics_digest=data["economics_digest"],
                risk_digest=data["risk_digest"],
                fold_plan_digest=data["fold_plan_digest"],
                nonrl_recipe_digest=recipes["nonrl"],
                residual_recipe_digest=recipes["residual_ppo"],
                direct_recipe_digest=recipes["direct_ppo"],
                account_mode=data["account_mode"],
                initial_capital=data["initial_capital"],
                scenarios=tuple(
                    AllocationComparisonScenario.from_payload(item)
                    for item in data["scenarios"]
                ),
                rl_seeds=tuple(data["rl_seeds"]),
                maximum_drawdown=data["maximum_drawdown"],
            )
        except TypeError as error:
            raise ValueError("allocation comparison contract is malformed") from error
        if canonical_json_bytes(result.payload()) != canonical_json_bytes(value):
            raise ValueError("comparison contract differs from canonical declaration")
        return result


@dataclass(frozen=True, slots=True)
class AllocationComparisonEvidence:
    contract_digest: str
    candidate: AllocationCandidateKind
    scenario: str
    seed: int | None
    policy_digest: str | None
    oos_source_digest: str
    ledger_digest: str
    execution_digest: str
    validity_evidence_digest: str
    recipe_digest: str
    opening_state_digest: str
    closing_state_digest: str
    terminal_profit_rate: float
    max_drawdown: float
    validity: AllocationValidity
    coverage_complete: bool
    termination_reason: str | None

    def __post_init__(self) -> None:
        require_sha256(self.contract_digest, field="contract_digest")
        if type(self.candidate) is not AllocationCandidateKind:
            raise ValueError("candidate must be an AllocationCandidateKind")
        object.__setattr__(
            self, "scenario", require_non_empty(self.scenario, field="scenario")
        )
        for field in (
            "oos_source_digest",
            "ledger_digest",
            "execution_digest",
            "validity_evidence_digest",
            "recipe_digest",
            "opening_state_digest",
            "closing_state_digest",
        ):
            require_sha256(getattr(self, field), field=field)
        if self.candidate is AllocationCandidateKind.NONRL:
            if self.seed is not None or self.policy_digest is not None:
                raise ValueError("nonRL evidence must not carry a policy seed/digest")
        else:
            if self.seed is None:
                raise ValueError("RL evidence requires one declared seed")
            _native_int(self.seed, "seed")
            if self.policy_digest is None:
                raise ValueError("RL evidence requires a policy digest")
            require_sha256(self.policy_digest, field="policy_digest")
        object.__setattr__(
            self,
            "terminal_profit_rate",
            _finite(self.terminal_profit_rate, "terminal_profit_rate"),
        )
        drawdown = _finite(self.max_drawdown, "max_drawdown")
        if drawdown < 0.0:
            raise ValueError("max_drawdown must be non-negative")
        object.__setattr__(self, "max_drawdown", drawdown)
        if type(self.validity) is not AllocationValidity:
            raise ValueError("validity must be an AllocationValidity")
        if type(self.coverage_complete) is not bool:
            raise ValueError("coverage_complete must be a native bool")
        if self.termination_reason is not None:
            object.__setattr__(
                self,
                "termination_reason",
                require_non_empty(self.termination_reason, field="termination_reason"),
            )

    def payload(self) -> dict[str, object]:
        return {
            "schema": "allocation_comparison_evidence_v1",
            "contract_digest": self.contract_digest,
            "candidate": self.candidate.value,
            "scenario": self.scenario,
            "seed": self.seed,
            "policy_digest": self.policy_digest,
            "oos_source_digest": self.oos_source_digest,
            "ledger_digest": self.ledger_digest,
            "execution_digest": self.execution_digest,
            "validity_evidence_digest": self.validity_evidence_digest,
            "recipe_digest": self.recipe_digest,
            "opening_state_digest": self.opening_state_digest,
            "closing_state_digest": self.closing_state_digest,
            "terminal_profit_rate": self.terminal_profit_rate,
            "max_drawdown": self.max_drawdown,
            "validity": self.validity.value,
            "coverage_complete": self.coverage_complete,
            "termination_reason": self.termination_reason,
        }

    @classmethod
    def from_payload(cls, value: object) -> AllocationComparisonEvidence:
        data = _closed_mapping(
            value,
            {
                "schema",
                "contract_digest",
                "candidate",
                "scenario",
                "seed",
                "policy_digest",
                "oos_source_digest",
                "ledger_digest",
                "execution_digest",
                "validity_evidence_digest",
                "recipe_digest",
                "opening_state_digest",
                "closing_state_digest",
                "terminal_profit_rate",
                "max_drawdown",
                "validity",
                "coverage_complete",
                "termination_reason",
            },
            "allocation comparison evidence",
        )
        if data["schema"] != "allocation_comparison_evidence_v1":
            raise ValueError("unsupported allocation comparison evidence schema")
        try:
            result = cls(
                contract_digest=data["contract_digest"],
                candidate=AllocationCandidateKind(data["candidate"]),
                scenario=data["scenario"],
                seed=data["seed"],
                policy_digest=data["policy_digest"],
                oos_source_digest=data["oos_source_digest"],
                ledger_digest=data["ledger_digest"],
                execution_digest=data["execution_digest"],
                validity_evidence_digest=data["validity_evidence_digest"],
                recipe_digest=data["recipe_digest"],
                opening_state_digest=data["opening_state_digest"],
                closing_state_digest=data["closing_state_digest"],
                terminal_profit_rate=data["terminal_profit_rate"],
                max_drawdown=data["max_drawdown"],
                validity=AllocationValidity(data["validity"]),
                coverage_complete=data["coverage_complete"],
                termination_reason=data["termination_reason"],
            )
        except (TypeError, ValueError) as error:
            raise ValueError("allocation comparison evidence is malformed") from error
        if canonical_json_bytes(result.payload()) != canonical_json_bytes(value):
            raise ValueError("comparison evidence differs from canonical declaration")
        return result


def _expected_keys(
    contract: AllocationComparisonContract,
) -> tuple[tuple[AllocationCandidateKind, int | None, str], ...]:
    values: list[tuple[AllocationCandidateKind, int | None, str]] = []
    for scenario in contract.scenarios:
        values.append((AllocationCandidateKind.NONRL, None, scenario.name))
    for candidate in (
        AllocationCandidateKind.RESIDUAL_PPO,
        AllocationCandidateKind.DIRECT_PPO,
    ):
        for seed in contract.rl_seeds:
            for scenario in contract.scenarios:
                values.append((candidate, seed, scenario.name))
    return tuple(values)


def validate_allocation_comparison_evidence(
    contract: AllocationComparisonContract,
    evidence: tuple[AllocationComparisonEvidence, ...],
) -> tuple[AllocationComparisonEvidence, ...]:
    if type(contract) is not AllocationComparisonContract:
        raise ValueError("comparison validation requires AllocationComparisonContract")
    if (
        type(evidence) is not tuple
        or not evidence
        or any(type(row) is not AllocationComparisonEvidence for row in evidence)
    ):
        raise ValueError("comparison evidence must be a non-empty immutable tuple")

    scenario_names = {value.name for value in contract.scenarios}
    expected = _expected_keys(contract)
    expected_set = set(expected)
    by_key: dict[
        tuple[AllocationCandidateKind, int | None, str], AllocationComparisonEvidence
    ] = {}
    for row in evidence:
        if row.contract_digest != contract.digest:
            raise ValueError("comparison evidence belongs to another contract")
        if row.scenario not in scenario_names:
            raise ValueError("comparison evidence uses an undeclared scenario")
        if (
            row.candidate is not AllocationCandidateKind.NONRL
            and row.seed not in contract.rl_seeds
        ):
            raise ValueError("comparison evidence uses an undeclared RL seed")
        if row.recipe_digest != _candidate_recipe(contract, row.candidate):
            raise ValueError(
                "comparison evidence candidate recipe differs from contract"
            )
        key = (row.candidate, row.seed, row.scenario)
        if key not in expected_set:
            raise ValueError("comparison evidence row is outside the fixed matrix")
        if key in by_key:
            raise ValueError("comparison evidence matrix contains a duplicate row")
        by_key[key] = row
    if set(by_key) != expected_set:
        raise ValueError("comparison evidence matrix is incomplete")

    for scenario in contract.scenarios:
        rows = [row for row in evidence if row.scenario == scenario.name]
        if len({row.oos_source_digest for row in rows}) != 1:
            raise ValueError("common scenario source differs across candidates")
        if len({row.opening_state_digest for row in rows}) != 1:
            raise ValueError("common scenario opening state differs across candidates")

    for candidate in (
        AllocationCandidateKind.RESIDUAL_PPO,
        AllocationCandidateKind.DIRECT_PPO,
    ):
        for seed in contract.rl_seeds:
            rows = [
                row
                for row in evidence
                if row.candidate is candidate and row.seed == seed
            ]
            if len({row.policy_digest for row in rows}) != 1:
                raise ValueError("one RL seed must use one policy across scenarios")

    return tuple(by_key[key] for key in expected)


def _candidate_recipe(
    contract: AllocationComparisonContract,
    candidate: AllocationCandidateKind,
) -> str:
    if candidate is AllocationCandidateKind.NONRL:
        return contract.nonrl_recipe_digest
    if candidate is AllocationCandidateKind.RESIDUAL_PPO:
        return contract.residual_recipe_digest
    return contract.direct_recipe_digest


@dataclass(frozen=True, slots=True)
class AllocationCandidateSummary:
    candidate: AllocationCandidateKind
    median_base_profit_rate: float
    median_incremental_vs_nonrl_base: float
    validity_passed: bool
    risk_execution_passed: bool
    diagnostic_failures: tuple[str, ...]

    def payload(self) -> dict[str, object]:
        return {
            "candidate": self.candidate.value,
            "median_base_profit_rate": self.median_base_profit_rate,
            "median_incremental_vs_nonrl_base": self.median_incremental_vs_nonrl_base,
            "validity_passed": self.validity_passed,
            "risk_execution_passed": self.risk_execution_passed,
            "diagnostic_failures": list(self.diagnostic_failures),
        }


@dataclass(frozen=True, slots=True)
class AllocationComparisonSummary:
    contract_digest: str
    candidates: tuple[AllocationCandidateSummary, ...]

    def payload(self) -> dict[str, object]:
        return {
            "schema": "allocation_comparison_summary_v1",
            "contract_digest": self.contract_digest,
            "candidates": [value.payload() for value in self.candidates],
        }


def _row_valid(row: AllocationComparisonEvidence) -> bool:
    return row.validity is AllocationValidity.VALID and row.coverage_complete


def _row_risk_execution(
    row: AllocationComparisonEvidence, contract: AllocationComparisonContract
) -> bool:
    return (
        row.max_drawdown <= contract.maximum_drawdown and row.termination_reason is None
    )


def summarize_allocation_comparison(
    contract: AllocationComparisonContract,
    evidence: tuple[AllocationComparisonEvidence, ...],
) -> AllocationComparisonSummary:
    rows = validate_allocation_comparison_evidence(contract, evidence)
    base_nonrl = next(
        row
        for row in rows
        if row.candidate is AllocationCandidateKind.NONRL and row.scenario == "base"
    ).terminal_profit_rate
    scenarios = {value.name: value for value in contract.scenarios}
    summaries: list[AllocationCandidateSummary] = []
    for candidate in _CANDIDATE_ROSTER:
        owned = [row for row in rows if row.candidate is candidate]
        base_values = [
            row.terminal_profit_rate for row in owned if row.scenario == "base"
        ]
        base_median = float(median(base_values))
        required = [row for row in owned if scenarios[row.scenario].required]
        diagnostics: list[str] = []
        for scenario in contract.scenarios:
            if scenario.required:
                continue
            values = [row for row in owned if row.scenario == scenario.name]
            if any(
                not _row_valid(row) or not _row_risk_execution(row, contract)
                for row in values
            ):
                diagnostics.append(scenario.name)
        summaries.append(
            AllocationCandidateSummary(
                candidate=candidate,
                median_base_profit_rate=base_median,
                median_incremental_vs_nonrl_base=base_median - base_nonrl,
                validity_passed=all(_row_valid(row) for row in required),
                risk_execution_passed=all(
                    _row_risk_execution(row, contract) for row in required
                ),
                diagnostic_failures=tuple(diagnostics),
            )
        )
    return AllocationComparisonSummary(contract.digest, tuple(summaries))


__all__ = [
    "AllocationCandidateKind",
    "AllocationCandidateSummary",
    "AllocationComparisonContract",
    "AllocationComparisonEvidence",
    "AllocationComparisonScenario",
    "AllocationComparisonSummary",
    "AllocationValidity",
    "summarize_allocation_comparison",
    "validate_allocation_comparison_evidence",
]
