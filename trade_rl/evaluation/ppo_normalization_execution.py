"""Execution primitives for the sealed PPO normalization replication."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.directional_contract import DIRECTIONAL_BASE_EXECUTION_COST
from trade_rl.evaluation.directional_selection import passes_screen, passes_stress
from trade_rl.evaluation.experiments.store import StudyStore
from trade_rl.strategies.interface import SingleSymbolStrategy
from trade_rl.strategies.rl.ppo import PPOIntentStrategy, fit_ppo_strategy

_SLOT_SCHEMA = "ppo_normalization_replication_slot_v1"
_PREFIT_FAILURE_SCHEMA = "ppo_normalization_replication_prefit_failure_v1"
_CONSUMED_FAILURE_SCHEMA = "ppo_normalization_replication_consumed_failure_v1"


class _ReplicationConfig(Protocol):
    feature_indices: tuple[int, ...]
    fit_symbol_indices: tuple[int, ...]
    fit_cutoff: str


@dataclass(frozen=True, slots=True)
class ReplicationArmSpec:
    """One immutable fresh-fit slot in the paired replication."""

    slot: str
    protocol_arm: str
    seed: int
    normalize_features: bool


def replication_arm_specs() -> tuple[ReplicationArmSpec, ...]:
    """Return the exact ten-slot roster in deterministic order."""

    controls = tuple(
        ReplicationArmSpec(
            slot=f"control_raw_seed{seed}",
            protocol_arm="control_raw",
            seed=seed,
            normalize_features=False,
        )
        for seed in range(5)
    )
    candidates = tuple(
        ReplicationArmSpec(
            slot=f"candidate_normalized_seed{seed}",
            protocol_arm="candidate_normalized",
            seed=seed,
            normalize_features=True,
        )
        for seed in range(5)
    )
    return controls + candidates


def _require_registered_spec(spec: ReplicationArmSpec) -> None:
    if spec not in replication_arm_specs():
        raise ValueError("replication arm is outside the sealed ten-slot roster")


def _fit_stop_index(dataset: MarketDataset, config: _ReplicationConfig) -> int:
    fit_cutoff = np.datetime64(config.fit_cutoff)
    stop_index = int(np.searchsorted(dataset.timestamps, fit_cutoff)) - 1
    if (
        stop_index <= 0
        or stop_index >= dataset.n_bars
        or dataset.timestamps[stop_index] >= fit_cutoff
    ):
        raise ValueError("fit cutoff does not define a valid pre-development window")
    return stop_index


def fit_replication_strategy(
    dataset: MarketDataset,
    config: _ReplicationConfig,
    spec: ReplicationArmSpec,
) -> PPOIntentStrategy:
    """Fit one fresh control/candidate policy under the sealed common contract."""

    _require_registered_spec(spec)
    return fit_ppo_strategy(
        dataset,
        feature_indices=tuple(config.feature_indices),
        fit_symbol_indices=tuple(config.fit_symbol_indices),
        start_index=0,
        stop_index=_fit_stop_index(dataset, config),
        gross_budget=0.1,
        total_timesteps=262_144,
        seed=spec.seed,
        initial_capital=10_000.0,
        execution_cost=DIRECTIONAL_BASE_EXECUTION_COST,
        training_layout="sequential",
        risk_config=None,
        normalize_features=spec.normalize_features,
        settle_terminal_position=True,
    )


def replication_strategy_factory(
    frozen: PPOIntentStrategy,
) -> Callable[[], SingleSymbolStrategy]:
    """Create fresh mutable wrappers while sharing frozen policy/preprocessing."""

    feature_indices = tuple(frozen.feature_indices)
    policy = frozen.policy
    normalizer = frozen.feature_normalizer

    def factory() -> SingleSymbolStrategy:
        return PPOIntentStrategy(
            policy,
            feature_indices=feature_indices,
            feature_normalizer=normalizer,
        )

    return factory


def _require_sha256(value: str, *, field: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _safe_attempt_id(value: str) -> str:
    if (
        not value
        or value in {".", ".."}
        or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
            for character in value
        )
    ):
        raise ValueError("attempt_id contains unsafe characters")
    return value


def _slot_relative(spec: ReplicationArmSpec, name: str) -> Path:
    _require_registered_spec(spec)
    return Path("slots") / spec.slot / name


def record_prefit_failure(
    root: Path,
    spec: ReplicationArmSpec,
    *,
    attempt_id: str,
    error: str,
) -> None:
    """Persist a failed preflight without consuming the economic slot."""

    _require_registered_spec(spec)
    attempt = _safe_attempt_id(attempt_id)
    if not isinstance(error, str) or not error:
        raise ValueError("error must be non-empty text")
    store = StudyStore(root)
    store.publish_json_once(
        Path("prefit-failures") / spec.slot / f"{attempt}.json",
        {
            "schema": _PREFIT_FAILURE_SCHEMA,
            "slot": spec.slot,
            "protocol_arm": spec.protocol_arm,
            "seed": spec.seed,
            "normalize_features": spec.normalize_features,
            "consumed": False,
            "error": error,
        },
    )


def claim_replication_slot(
    root: Path,
    spec: ReplicationArmSpec,
    *,
    activation_digest: str,
    implementation_digest: str,
) -> None:
    """Atomically cross the fit boundary exactly once for one slot."""

    _require_registered_spec(spec)
    activation = _require_sha256(activation_digest, field="activation_digest")
    implementation = _require_sha256(
        implementation_digest,
        field="implementation_digest",
    )
    store = StudyStore(root)
    with store.mutation_lock():
        state = replication_slot_state(root, spec)
        if state["consumed"] or state["failed"] or state["result_published"]:
            raise ValueError(f"replication slot already consumed: {spec.slot}")
        store.publish_json_once(
            _slot_relative(spec, "consumed.json"),
            {
                "schema": _SLOT_SCHEMA,
                "slot": spec.slot,
                "protocol_arm": spec.protocol_arm,
                "seed": spec.seed,
                "normalize_features": spec.normalize_features,
                "activation_digest": activation,
                "implementation_digest": implementation,
                "consumed": True,
            },
        )


def record_consumed_failure(
    root: Path,
    spec: ReplicationArmSpec,
    *,
    error: str,
) -> None:
    """Persist a post-claim failure; the slot remains permanently consumed."""

    if not isinstance(error, str) or not error:
        raise ValueError("error must be non-empty text")
    store = StudyStore(root)
    with store.mutation_lock():
        state = replication_slot_state(root, spec)
        if not state["consumed"]:
            raise ValueError("cannot record consumed failure before slot consumption")
        if state["failed"] or state["result_published"]:
            raise ValueError("consumed slot already has terminal evidence")
        store.publish_json_once(
            _slot_relative(spec, "failed.json"),
            {
                "schema": _CONSUMED_FAILURE_SCHEMA,
                "slot": spec.slot,
                "consumed": True,
                "error": error,
            },
        )


def _regular_json_exists(store: StudyStore, relative: Path) -> bool:
    path = store.root / relative
    if path.is_symlink():
        raise ValueError(f"replication evidence must not be a symlink: {relative}")
    if not path.exists():
        return False
    if not path.is_file():
        raise ValueError(f"replication evidence must be a regular file: {relative}")
    store.read_json(relative)
    return True


def replication_slot_state(
    root: Path,
    spec: ReplicationArmSpec,
) -> dict[str, object]:
    """Inspect one slot without treating pre-fit failures as consumption."""

    _require_registered_spec(spec)
    store = StudyStore(root)
    consumed = _regular_json_exists(store, _slot_relative(spec, "consumed.json"))
    failed = _regular_json_exists(store, _slot_relative(spec, "failed.json"))
    result_published = _regular_json_exists(
        store,
        _slot_relative(spec, "result.json"),
    )
    prefit_root = store.root / "prefit-failures" / spec.slot
    if prefit_root.is_symlink():
        raise ValueError("prefit failure directory must not be a symlink")
    prefit_failure_count = 0
    if prefit_root.exists():
        if not prefit_root.is_dir():
            raise ValueError("prefit failure evidence must be a directory")
        for path in sorted(prefit_root.iterdir(), key=lambda item: item.name):
            if path.is_symlink() or not path.is_file() or path.suffix != ".json":
                raise ValueError("prefit failure evidence contains an unsafe entry")
            relative = path.relative_to(store.root)
            payload = store.read_json(relative)
            if (
                payload.get("schema") != _PREFIT_FAILURE_SCHEMA
                or payload.get("slot") != spec.slot
                or payload.get("consumed") is not False
            ):
                raise ValueError("prefit failure evidence is malformed")
            prefit_failure_count += 1
    if (failed or result_published) and not consumed:
        raise ValueError("terminal slot evidence exists without a consumed claim")
    if failed and result_published:
        raise ValueError("slot cannot contain both failure and result evidence")
    return {
        "slot": spec.slot,
        "consumed": consumed,
        "failed": failed,
        "result_published": result_published,
        "prefit_failure_count": prefit_failure_count,
    }


def _result_total_return(row: dict[str, Any]) -> float:
    metrics = row.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("replication result metrics are missing")
    value = metrics.get("total_return")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("replication total_return must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("replication total_return must be finite")
    return result


def _complete_result(row: dict[str, Any]) -> bool:
    returns = row.get("returns")
    start = row.get("start_index")
    stop = row.get("stop_index")
    return bool(
        isinstance(returns, list)
        and len(returns) == 17_544
        and isinstance(start, int)
        and not isinstance(start, bool)
        and isinstance(stop, int)
        and not isinstance(stop, bool)
        and stop - start == 17_544
    )


def _termination_count(row: dict[str, Any]) -> int:
    reasons = row.get("termination_reasons")
    if not isinstance(reasons, list):
        raise ValueError("termination_reasons must be a list")
    return len(reasons)


def _candidate_base_family(candidate: dict[int, dict[str, Any]]) -> tuple[int, float, dict[str, float]]:
    base_pass = {
        seed: passes_screen(row, require_positive_years=True)
        for seed, row in candidate.items()
    }
    totals = [_result_total_return(candidate[seed]) for seed in range(5)]
    median_total = float(np.median(totals))
    median_years = {
        year: float(
            np.median(
                [
                    float(candidate[seed]["year_returns"][year])
                    for seed in range(5)
                ]
            )
        )
        for year in ("2023", "2024")
    }
    return sum(base_pass.values()), median_total, median_years


def recompute_replication_decision(
    control: dict[int, dict[str, Any]],
    candidate: dict[int, dict[str, Any]],
) -> dict[str, object]:
    """Recompute relative and absolute gates; never trust stored qualified flags."""

    expected_seeds = set(range(5))
    if set(control) != expected_seeds or set(candidate) != expected_seeds:
        raise ValueError("replication comparison requires exactly seeds 0..4")

    paired = {
        seed: _result_total_return(candidate[seed])
        - _result_total_return(control[seed])
        for seed in range(5)
    }
    complete = all(
        _complete_result(control[seed]) and _complete_result(candidate[seed])
        for seed in range(5)
    )
    within_risk = all(
        isinstance(candidate[seed].get("ledger_max_drawdown"), (int, float))
        and not isinstance(candidate[seed].get("ledger_max_drawdown"), bool)
        and 0.0 <= float(candidate[seed]["ledger_max_drawdown"]) <= 0.2
        for seed in range(5)
    )
    no_new_termination = all(
        _termination_count(candidate[seed]) <= _termination_count(control[seed])
        for seed in range(5)
    )
    paired_win_count = sum(value > 0.0 for value in paired.values())
    median_delta = float(np.median(list(paired.values())))
    relative = bool(
        complete
        and within_risk
        and no_new_termination
        and paired_win_count >= 4
        and median_delta > 0.0
    )

    base_pass_count, median_total, median_years = _candidate_base_family(candidate)
    base_family_pass = bool(
        base_pass_count >= 4
        and median_total > 0.0
        and all(value > 0.0 for value in median_years.values())
    )
    stress_pass_count = sum(
        passes_screen(candidate[seed], require_positive_years=True)
        and passes_stress(candidate[seed])
        for seed in range(5)
    )
    full_family_pass = bool(base_family_pass and stress_pass_count >= 4)

    if not relative:
        decision = "KEEP_BASELINE"
    elif not full_family_pass:
        decision = "RELATIVE_IMPROVEMENT_ONLY"
    else:
        decision = "PROSPECTIVE_PAPER_REQUIRED"
    return {
        "schema": "ppo_normalization_replication_comparison_v1",
        "paired_return_deltas": paired,
        "paired_win_count": paired_win_count,
        "median_paired_total_return_delta": median_delta,
        "relative_improvement": relative,
        "candidate_base_pass_count": base_pass_count,
        "candidate_base_median_total_return": median_total,
        "candidate_base_median_year_returns": median_years,
        "candidate_base_and_stress_pass_count": stress_pass_count,
        "candidate_absolute_family_pass": full_family_pass,
        "decision": decision,
        "unused_data_accessed": False,
        "production_eligible": False,
        "live_trading_authorized": False,
    }


__all__ = [
    "ReplicationArmSpec",
    "claim_replication_slot",
    "fit_replication_strategy",
    "recompute_replication_decision",
    "record_consumed_failure",
    "record_prefit_failure",
    "replication_arm_specs",
    "replication_slot_state",
    "replication_strategy_factory",
]
