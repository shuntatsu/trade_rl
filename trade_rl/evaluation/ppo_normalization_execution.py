"""Execution primitives for the sealed PPO normalization replication."""

from __future__ import annotations

import json
import math
import os
import shutil
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from trade_rl._validation import require_sha256
from trade_rl.artifacts import canonical_json_bytes, content_digest
from trade_rl.artifacts.verified_file import file_digest_and_size, open_regular_binary
from trade_rl.data.artifacts import (
    inspect_published_market_dataset_artifact,
    load_market_dataset_artifact,
)
from trade_rl.data.features.price_channels import with_price_channels
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.directional import evaluate_directional_arm
from trade_rl.evaluation.directional_contract import DIRECTIONAL_BASE_EXECUTION_COST
from trade_rl.evaluation.directional_selection import passes_screen, passes_stress
from trade_rl.evaluation.directional_study import development_indices
from trade_rl.evaluation.experiments import ResolvedRunConfig, inspect_study
from trade_rl.evaluation.experiments.store import StudyStore
from trade_rl.evaluation.ppo_normalization_replication import (
    expected_ppo_normalization_protocol,
    ppo_normalization_protocol_bytes,
)
from trade_rl.evaluation.runs import build_candidate_run_provenance
from trade_rl.strategies.interface import SingleSymbolStrategy
from trade_rl.strategies.rl.ppo import PPOIntentStrategy, fit_ppo_strategy
from trade_rl.strategies.rl.ppo_artifact import (
    load_ppo_inference_bundle,
    save_ppo_inference_bundle,
)

_SLOT_SCHEMA = "ppo_normalization_replication_slot_v1"
_PREFIT_FAILURE_SCHEMA = "ppo_normalization_replication_prefit_failure_v1"
_CONSUMED_FAILURE_SCHEMA = "ppo_normalization_replication_consumed_failure_v1"
EXECUTION_ACTIVATION_SCHEMA = "ppo_normalization_execution_activation_v1"
ACTIVATION_AUTHORITY_SCHEMA = "ppo_normalization_execution_activation_authority_v1"
_ACTIVATION_AUTHORITY_PATH = Path(__file__).with_name(
    "ppo_normalization_activation.json"
)
SEALED_PROTOCOL_SHA256 = (
    "0013470ed5858eaa3b9391f97f4b18f772495d21c128832f74e1c50b090df304"
)
SLOT_RESULT_SCHEMA = "ppo_normalization_replication_result_v1"
SLOT_VERIFICATION_SCHEMA = "ppo_normalization_replication_verification_v1"
COMPARISON_SCHEMA = "ppo_normalization_replication_comparison_v1"
VERIFIER_ARTIFACT_AUTHORITY_SCHEMA = (
    "ppo_normalization_replication_verifier_artifact_authority_v1"
)
_VERIFICATION_SET_SCHEMA = "ppo_normalization_replication_verification_set_v1"
_PPO_TIMESTEPS = 262_144


class _ReplicationConfig(Protocol):
    @property
    def feature_indices(self) -> tuple[int, ...]: ...

    @property
    def fit_symbol_indices(self) -> tuple[int, ...]: ...

    @property
    def fit_cutoff(self) -> str: ...


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
        total_timesteps=_PPO_TIMESTEPS,
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
    feature_names = frozen.feature_names
    policy = frozen.policy
    normalizer = frozen.feature_normalizer

    def factory() -> SingleSymbolStrategy:
        return PPOIntentStrategy(
            policy,
            feature_indices=feature_indices,
            feature_names=feature_names,
            feature_normalizer=normalizer,
        )

    return factory


def _evaluate_replication_result(
    dataset: MarketDataset,
    factory: Callable[[], SingleSymbolStrategy],
    *,
    start_index: int,
    stop_index: int,
    spec: ReplicationArmSpec,
) -> dict[str, Any]:
    """Replay one fitted bundle under the sealed base/stress evidence contract."""

    _require_registered_spec(spec)
    result = evaluate_directional_arm(
        dataset,
        factory,
        start_index=start_index,
        stop_index=stop_index,
        initial_capital=10_000.0,
        gross_budget=0.1,
    )
    if spec.protocol_arm != "candidate_normalized" or not passes_screen(
        result,
        require_positive_years=True,
    ):
        return result

    protocol = expected_ppo_normalization_protocol()
    decision = protocol.get("decision")
    if not isinstance(decision, dict):
        raise RuntimeError("sealed normalization decision contract is malformed")
    absolute = decision.get("absolute")
    if not isinstance(absolute, dict):
        raise RuntimeError("sealed normalization absolute gate is malformed")
    stresses = absolute.get("stresses")
    expected_stresses = [
        {"cost_multiplier": 2.0, "latency_bars": 0},
        {"cost_multiplier": 1.0, "latency_bars": 1},
    ]
    if stresses != expected_stresses:
        raise RuntimeError("sealed normalization stress roster drifted")

    result["stress"] = [
        evaluate_directional_arm(
            dataset,
            factory,
            start_index=start_index,
            stop_index=stop_index,
            initial_capital=10_000.0,
            gross_budget=0.1,
            cost_multiplier=float(stress["cost_multiplier"]),
            latency_bars=int(stress["latency_bars"]),
        )
        for stress in expected_stresses
    ]
    result["by_symbol"] = {
        symbol: evaluate_directional_arm(
            dataset,
            factory,
            start_index=start_index,
            stop_index=stop_index,
            symbol_index=index,
            initial_capital=10_000.0,
            gross_budget=0.1,
        )
        for index, symbol in enumerate(dataset.symbols)
    }
    return result


def _safe_attempt_id(value: str) -> str:
    if (
        not value
        or value in {".", ".."}
        or any(
            character
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
            for character in value
        )
    ):
        raise ValueError("attempt_id contains unsafe characters")
    return value


def _slot_relative(spec: ReplicationArmSpec, name: str) -> Path:
    _require_registered_spec(spec)
    return Path("slots") / spec.slot / name


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(path))


def _check_directory_ancestors(path: Path) -> None:
    absolute = _absolute(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"execution path must not contain symlinks: {current}")
        if current.exists() and not current.is_dir():
            raise ValueError(f"execution path parent is not a directory: {current}")


def _existing_execution_store(root: Path) -> StudyStore:
    root = Path(root)
    _check_directory_ancestors(root)
    if root.is_symlink() or not root.exists() or not root.is_dir():
        raise ValueError("replication execution root is not a prepared directory")
    return StudyStore(root)


def _record_prefit_failure(
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
    _validate_execution_root(root, runtime_contract="execution")
    store = _existing_execution_store(root)
    with store.mutation_lock():
        state = _replication_slot_state(store, spec)
        if state["consumed"] or state["failed"] or state["result_published"]:
            raise ValueError("pre-fit failure cannot follow slot consumption")
        store.publish_json_once(
            Path("prefit-failures") / spec.slot / f"{attempt}.json",
            {
                "schema": _PREFIT_FAILURE_SCHEMA,
                "slot": spec.slot,
                "protocol_arm": spec.protocol_arm,
                "seed": spec.seed,
                "normalize_features": spec.normalize_features,
                "attempt_id": attempt,
                "consumed": False,
                "error": error,
            },
        )


def _claim_replication_slot(
    root: Path,
    spec: ReplicationArmSpec,
) -> None:
    """Atomically cross the fit boundary exactly once for one prepared slot."""

    _require_registered_spec(spec)
    activation = _validate_execution_root(root, runtime_contract="execution")
    activation_digest = _sealed_execution_activation_digest()
    implementation = activation.get("implementation_digest")
    if not isinstance(implementation, str):
        raise ValueError("activation implementation digest is malformed")
    implementation_digest = require_sha256(
        implementation,
        field="implementation_digest",
    )
    store = _existing_execution_store(root)
    with store.mutation_lock():
        state = _replication_slot_state(store, spec)
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
                "activation_digest": activation_digest,
                "implementation_digest": implementation_digest,
                "consumed": True,
            },
        )


def _record_consumed_failure(
    root: Path,
    spec: ReplicationArmSpec,
    *,
    error: str,
) -> None:
    """Persist a post-claim failure against the immutable prepared identity."""

    if not isinstance(error, str) or not error:
        raise ValueError("error must be non-empty text")
    store = _existing_execution_store(root)
    with store.mutation_lock():
        claim = _read_consumed_claim(store, spec)
        state = _replication_slot_state(store, spec)
        if claim is None or not state["consumed"]:
            raise ValueError("cannot record consumed failure before slot consumption")
        if state["failed"] or state["result_published"]:
            raise ValueError("consumed slot already has terminal evidence")
        activation_digest = claim.get("activation_digest")
        implementation_digest = claim.get("implementation_digest")
        if not isinstance(activation_digest, str) or not isinstance(
            implementation_digest, str
        ):
            raise ValueError("replication slot claim digests are malformed")
        store.publish_json_once(
            _slot_relative(spec, "failed.json"),
            {
                "schema": _CONSUMED_FAILURE_SCHEMA,
                "slot": spec.slot,
                "protocol_arm": spec.protocol_arm,
                "seed": spec.seed,
                "normalize_features": spec.normalize_features,
                "activation_digest": activation_digest,
                "implementation_digest": implementation_digest,
                "consumed": True,
                "error": error,
            },
        )


def _read_canonical_json(
    store: StudyStore,
    relative: Path,
    *,
    field: str,
) -> tuple[dict[str, object], bytes]:
    path = store._checked_target(relative)
    try:
        with open_regular_binary(path, field=field) as stream:
            raw = stream.read()
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{field} is not valid JSON") from error
    if not isinstance(payload, dict) or canonical_json_bytes(payload) != raw:
        raise ValueError(f"{field} must be canonical JSON")
    return payload, raw


def _publish_json_with_sha256_pair(
    store: StudyStore,
    relative: str | Path,
    payload: dict[str, object],
    *,
    field: str,
) -> None:
    relative_path = Path(relative)
    if (
        relative_path.is_absolute()
        or not relative_path.parts
        or relative_path == Path(".")
        or ".." in relative_path.parts
        or relative_path.suffix != ".json"
    ):
        raise ValueError(f"{field} path must be a safe JSON relative path")
    digest_relative = relative_path.with_name(f"{relative_path.stem}.sha256.json")
    raw = canonical_json_bytes(payload)
    digest_payload = {"sha256": sha256(raw).hexdigest()}

    with store.mutation_lock():
        target = store._checked_target(relative_path)
        digest_target = store._checked_target(digest_relative)

        target_exists = target.exists() or target.is_symlink()
        digest_exists = digest_target.exists() or digest_target.is_symlink()

        if target_exists:
            _existing_payload, existing_raw = _read_canonical_json(
                store,
                relative_path,
                field=field,
            )
            if existing_raw != raw:
                raise ValueError(f"{field} differs from existing evidence")

        if digest_exists:
            existing_digest, _existing_digest_raw = _read_canonical_json(
                store,
                digest_relative,
                field=f"{field} digest",
            )
            if existing_digest != digest_payload:
                raise ValueError(f"{field} digest differs from existing evidence")
        else:
            store.publish_json_once(digest_relative, digest_payload)

        if not target_exists:
            store.publish_json_once(relative_path, payload)


def _regular_json_exists(store: StudyStore, relative: Path) -> bool:
    path = store.root / relative
    if not path.exists() and not path.is_symlink():
        return False
    _read_canonical_json(store, relative, field=f"replication evidence {relative}")
    return True


def _read_consumed_claim(
    store: StudyStore,
    spec: ReplicationArmSpec,
) -> dict[str, object] | None:
    relative = _slot_relative(spec, "consumed.json")
    path = store.root / relative
    if path.is_symlink():
        raise ValueError("replication slot claim must not be a symlink")
    if not path.exists():
        return None
    if not path.is_file():
        raise ValueError("replication slot claim must be a regular file")
    payload, _raw = _read_canonical_json(
        store,
        relative,
        field="replication slot claim",
    )
    expected_keys = {
        "schema",
        "slot",
        "protocol_arm",
        "seed",
        "normalize_features",
        "activation_digest",
        "implementation_digest",
        "consumed",
    }
    if (
        set(payload) != expected_keys
        or payload.get("schema") != _SLOT_SCHEMA
        or payload.get("slot") != spec.slot
        or payload.get("protocol_arm") != spec.protocol_arm
        or payload.get("seed") != spec.seed
        or payload.get("normalize_features") is not spec.normalize_features
        or payload.get("consumed") is not True
    ):
        raise ValueError(
            "replication slot claim is malformed or belongs to another slot"
        )
    activation = payload.get("activation_digest")
    implementation = payload.get("implementation_digest")
    if not isinstance(activation, str) or not isinstance(implementation, str):
        raise ValueError("replication slot claim digests are malformed")
    require_sha256(activation, field="slot claim activation_digest")
    require_sha256(implementation, field="slot claim implementation_digest")
    return payload


def _read_consumed_failure(
    store: StudyStore,
    spec: ReplicationArmSpec,
    claim: dict[str, object],
) -> dict[str, object] | None:
    relative = _slot_relative(spec, "failed.json")
    path = store._checked_target(relative)
    if not path.exists() and not path.is_symlink():
        return None
    payload, _raw = _read_canonical_json(
        store,
        relative,
        field="replication consumed failure evidence",
    )
    expected_keys = {
        "schema",
        "slot",
        "protocol_arm",
        "seed",
        "normalize_features",
        "activation_digest",
        "implementation_digest",
        "consumed",
        "error",
    }
    if (
        set(payload) != expected_keys
        or payload.get("schema") != _CONSUMED_FAILURE_SCHEMA
        or payload.get("slot") != spec.slot
        or payload.get("protocol_arm") != spec.protocol_arm
        or payload.get("seed") != spec.seed
        or payload.get("normalize_features") is not spec.normalize_features
        or payload.get("activation_digest") != claim.get("activation_digest")
        or payload.get("implementation_digest") != claim.get("implementation_digest")
        or payload.get("consumed") is not True
        or not isinstance(payload.get("error"), str)
        or not payload.get("error")
    ):
        raise ValueError("replication consumed failure evidence is malformed")
    return payload


def _replication_slot_state(
    store: StudyStore,
    spec: ReplicationArmSpec,
) -> dict[str, object]:
    _require_registered_spec(spec)
    consumed_claim = _read_consumed_claim(store, spec)
    consumed = consumed_claim is not None
    failed_payload = (
        None
        if consumed_claim is None
        else _read_consumed_failure(store, spec, consumed_claim)
    )
    failed = failed_payload is not None
    failed_path = store._checked_target(_slot_relative(spec, "failed.json"))
    if consumed_claim is None and (failed_path.exists() or failed_path.is_symlink()):
        raise ValueError("terminal slot evidence exists without a consumed claim")
    result_published = _regular_json_exists(
        store,
        _slot_relative(spec, "result.json"),
    )

    probe = store._checked_target(
        Path("prefit-failures") / spec.slot / "__probe__.json"
    )
    prefit_root = probe.parent
    prefit_failure_count = 0
    if prefit_root.exists():
        if prefit_root.is_symlink() or not prefit_root.is_dir():
            raise ValueError("prefit failure evidence must be a regular directory")
        for path in sorted(prefit_root.iterdir(), key=lambda item: item.name):
            if path.is_symlink() or not path.is_file() or path.suffix != ".json":
                raise ValueError("prefit failure evidence contains an unsafe entry")
            relative = path.relative_to(store.root)
            payload, _raw = _read_canonical_json(
                store,
                relative,
                field="replication prefit failure evidence",
            )
            expected_keys = {
                "schema",
                "slot",
                "protocol_arm",
                "seed",
                "normalize_features",
                "attempt_id",
                "consumed",
                "error",
            }
            if (
                set(payload) != expected_keys
                or payload.get("schema") != _PREFIT_FAILURE_SCHEMA
                or payload.get("slot") != spec.slot
                or payload.get("protocol_arm") != spec.protocol_arm
                or payload.get("seed") != spec.seed
                or payload.get("normalize_features") is not spec.normalize_features
                or payload.get("attempt_id") != path.stem
                or payload.get("consumed") is not False
                or not isinstance(payload.get("error"), str)
                or not payload.get("error")
            ):
                raise ValueError("prefit failure evidence is malformed")
            _safe_attempt_id(path.stem)
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


def replication_slot_state(
    root: Path,
    spec: ReplicationArmSpec,
) -> dict[str, object]:
    """Inspect one slot without creating or mutating an execution root."""

    store = _existing_execution_store(root)
    _validate_execution_root(root, runtime_contract="stored")
    return _replication_slot_state(store, spec)


def _read_verified_record(
    store: StudyStore,
    spec: ReplicationArmSpec,
    *,
    result: dict[str, object],
    result_raw: bytes,
) -> dict[str, object]:
    relative = _slot_relative(spec, "verified.json")
    digest_relative = _slot_relative(spec, "verified.sha256.json")
    path = store.root / relative
    digest_path = store.root / digest_relative
    if not path.exists() or not digest_path.exists():
        raise ValueError(f"replication slot is not independently verified: {spec.slot}")
    payload, raw = _read_canonical_json(
        store,
        relative,
        field="replication verification record",
    )
    digest, _digest_raw = _read_canonical_json(
        store,
        digest_relative,
        field="replication verification digest",
    )
    if digest != {"sha256": sha256(raw).hexdigest()}:
        raise ValueError("replication verification digest mismatch")
    expected_keys = {
        "schema",
        "slot",
        "protocol_arm",
        "seed",
        "normalize_features",
        "protocol_sha256",
        "activation_digest",
        "implementation_digest",
        "result_sha256",
        "bundle_digest",
        "bundle_policy_sha256",
        "verifier_provenance",
        "no_refit",
        "replay_verified",
    }
    if (
        set(payload) != expected_keys
        or payload.get("schema") != SLOT_VERIFICATION_SCHEMA
        or payload.get("slot") != spec.slot
        or payload.get("protocol_arm") != spec.protocol_arm
        or payload.get("seed") != spec.seed
        or payload.get("normalize_features") is not spec.normalize_features
        or payload.get("protocol_sha256") != SEALED_PROTOCOL_SHA256
        or payload.get("activation_digest") != result.get("activation_digest")
        or payload.get("implementation_digest") != result.get("implementation_digest")
        or payload.get("result_sha256") != sha256(result_raw).hexdigest()
        or payload.get("bundle_digest") != result.get("bundle_digest")
        or payload.get("bundle_policy_sha256") != result.get("bundle_policy_sha256")
        or payload.get("no_refit") is not True
        or payload.get("replay_verified") is not True
    ):
        raise ValueError(
            "replication verification record differs from published result"
        )
    verifier_provenance = payload.get("verifier_provenance")
    result_provenance = result.get("provenance")
    if not isinstance(verifier_provenance, dict) or not isinstance(
        result_provenance, dict
    ):
        raise ValueError("replication verifier provenance is malformed")
    _validate_verifier_provenance(result_provenance, verifier_provenance)
    return payload


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


def _candidate_base_family(
    candidate: dict[int, dict[str, Any]],
) -> tuple[int, float, dict[str, float]]:
    base_pass = {
        seed: passes_screen(row, require_positive_years=True)
        for seed, row in candidate.items()
    }
    totals = [_result_total_return(candidate[seed]) for seed in range(5)]
    median_total = float(np.median(totals))
    median_years = {
        year: float(
            np.median(
                [float(candidate[seed]["year_returns"][year]) for seed in range(5)]
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
        "schema": COMPARISON_SCHEMA,
        "paired_return_deltas": {str(seed): paired[seed] for seed in range(5)},
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


def _sealed_execution_activation_digest() -> str:
    try:
        with open_regular_binary(
            _ACTIVATION_AUTHORITY_PATH,
            field="PPO normalization activation authority",
        ) as stream:
            raw = stream.read()
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            "PPO normalization activation authority is invalid JSON"
        ) from error

    if not isinstance(payload, dict) or canonical_json_bytes(payload) != raw:
        raise ValueError(
            "PPO normalization activation authority must be canonical JSON"
        )
    if (
        set(payload) != {"schema", "activation_sha256"}
        or payload.get("schema") != ACTIVATION_AUTHORITY_SCHEMA
    ):
        raise ValueError("PPO normalization activation authority shape is unsupported")
    digest = payload.get("activation_sha256")
    if digest is None:
        raise RuntimeError(
            "PPO normalization economic execution activation is not sealed"
        )
    if not isinstance(digest, str):
        raise ValueError("PPO normalization activation digest must be text")
    return require_sha256(digest, field="sealed_execution_activation_sha256")


def _validate_runtime_authority(provenance: dict[str, object]) -> None:
    """Require the activation runtime to match the sealed trainer contract."""

    protocol = expected_ppo_normalization_protocol()
    expected = protocol.get("runtime")
    runtime = provenance.get("runtime_environment")
    if not isinstance(expected, dict) or not isinstance(runtime, dict):
        raise ValueError("execution activation runtime provenance is missing")
    python = runtime.get("python")
    packages = runtime.get("packages")
    if not isinstance(python, dict) or not isinstance(packages, dict):
        raise ValueError("execution activation runtime provenance is malformed")

    expected_python = expected.get("python")
    version = python.get("version")
    if (
        python.get("implementation") != "CPython"
        or not isinstance(expected_python, str)
        or not isinstance(version, str)
        or not version.split()[0].startswith(f"{expected_python}.")
    ):
        raise ValueError("execution activation Python runtime differs from protocol")

    package_fields = {
        "stable-baselines3": "stable_baselines3",
        "torch": "torch",
        "gymnasium": "gymnasium",
    }
    for package_name, protocol_field in package_fields.items():
        expected_version = expected.get(protocol_field)
        if (
            not isinstance(expected_version, str)
            or packages.get(package_name) != expected_version
        ):
            raise ValueError(
                f"execution activation runtime package differs: {package_name}"
            )
    if expected.get("device") != "cpu" or expected.get("torch_num_threads") != 1:
        raise RuntimeError("sealed PPO runtime execution contract drifted")


def _verifier_runtime_identity(provenance: dict[str, object]) -> dict[str, object]:
    runtime = provenance.get("runtime_environment")
    if not isinstance(runtime, dict):
        raise ValueError("verifier runtime provenance is missing")
    python = runtime.get("python")
    os_info = runtime.get("os")
    machine = runtime.get("machine")
    packages = runtime.get("packages")
    if (
        not isinstance(python, dict)
        or not isinstance(os_info, dict)
        or not isinstance(machine, str)
        or not machine
        or not isinstance(packages, dict)
        or any(not isinstance(name, str) or not name for name in packages)
        or any(
            value is not None and not isinstance(value, str)
            for value in packages.values()
        )
    ):
        raise ValueError("verifier runtime provenance is malformed")
    implementation = python.get("implementation")
    version = python.get("version")
    os_family = os_info.get("family")
    os_release = os_info.get("release")
    if (
        not isinstance(implementation, str)
        or not implementation
        or not isinstance(version, str)
        or not version
        or not isinstance(os_family, str)
        or not os_family
        or not isinstance(os_release, str)
        or not os_release
    ):
        raise ValueError("verifier runtime provenance is malformed")
    return {
        "python": {
            "implementation": implementation,
            "version": version,
        },
        "os_family": os_family,
        "machine": machine,
        "packages": {name: packages[name] for name in sorted(packages)},
    }


def _validate_verifier_provenance(
    expected: dict[str, object],
    current: dict[str, object],
) -> None:
    if current.get("implementation_digest") != expected.get("implementation_digest"):
        raise ValueError("verifier implementation digest differs from activation")
    if current.get("research_context_digest") != expected.get(
        "research_context_digest"
    ):
        raise ValueError("verifier research context differs from activation")
    if _verifier_runtime_identity(current) != _verifier_runtime_identity(expected):
        raise ValueError("verifier stable runtime identity differs from activation")


def _validate_activation(
    activation: dict[str, object],
) -> dict[str, object]:
    expected_digest = _sealed_execution_activation_digest()
    expected_keys = {
        "schema",
        "protocol_sha256",
        "implementation_digest",
        "implementation_seal_sha256",
        "fresh_reconstruction_sha256",
        "assurance_review_sha256",
        "provenance",
        "economic_execution_authorized",
        "economic_result_inspected",
        "unused_data_accessed",
        "final_test_accessed",
        "production_eligible",
        "live_trading_authorized",
    }
    if set(activation) != expected_keys:
        raise ValueError("execution activation has an unexpected shape")
    if activation["schema"] != EXECUTION_ACTIVATION_SCHEMA:
        raise ValueError("execution activation schema differs")
    if activation["protocol_sha256"] != SEALED_PROTOCOL_SHA256:
        raise ValueError("execution activation protocol differs")
    implementation = activation["implementation_digest"]
    if not isinstance(implementation, str):
        raise ValueError("implementation_digest must be text")
    require_sha256(implementation, field="implementation_digest")
    for evidence_field in (
        "implementation_seal_sha256",
        "fresh_reconstruction_sha256",
        "assurance_review_sha256",
    ):
        evidence_digest = activation[evidence_field]
        if not isinstance(evidence_digest, str):
            raise ValueError(f"{evidence_field} must be text")
        require_sha256(evidence_digest, field=evidence_field)
    provenance = activation["provenance"]
    if not isinstance(provenance, dict):
        raise ValueError("execution activation provenance is malformed")
    if provenance.get("implementation_digest") != implementation:
        raise ValueError("activation implementation digest differs from provenance")
    _validate_runtime_authority(provenance)
    if content_digest(activation) != expected_digest:
        raise ValueError("execution activation digest mismatch")
    if activation["economic_execution_authorized"] is not True:
        raise ValueError("execution activation does not authorize economics")
    for field in (
        "economic_result_inspected",
        "unused_data_accessed",
        "final_test_accessed",
        "production_eligible",
        "live_trading_authorized",
    ):
        if activation[field] is not False:
            raise ValueError(f"execution activation illegally enables {field}")
    return activation


def prepare_replication_execution(
    root: Path,
    activation: dict[str, object],
) -> None:
    """Atomically publish one fully validated, result-blind execution root."""

    expected_activation_digest = _sealed_execution_activation_digest()
    checked = _validate_activation(activation)
    current_provenance = build_candidate_run_provenance()
    if checked["provenance"] != current_provenance:
        raise ValueError("current provenance differs from execution activation")

    root = Path(root)
    _check_directory_ancestors(root.parent)
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"replication execution root already exists: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    stage = root.parent / f".{root.name}.staging-{uuid.uuid4().hex}"
    if stage.exists() or stage.is_symlink():
        raise FileExistsError("replication execution staging root already exists")

    protocol = expected_ppo_normalization_protocol()
    raw = ppo_normalization_protocol_bytes()
    if sha256(raw).hexdigest() != SEALED_PROTOCOL_SHA256:
        raise RuntimeError("sealed normalization protocol bytes drifted")

    stage.mkdir()
    try:
        store = StudyStore(stage)
        store.publish_json_once("protocol.json", protocol)
        store.publish_json_once(
            "protocol.digest.json",
            {"sha256": SEALED_PROTOCOL_SHA256},
        )
        store.publish_json_once("activation.json", activation)
        store.publish_json_once(
            "activation.digest.json",
            {"digest": expected_activation_digest},
        )
        store.publish_json_once(
            "slots.json",
            {"slots": [spec.slot for spec in replication_arm_specs()]},
        )
        _validate_execution_root(stage)
        if root.exists() or root.is_symlink():
            raise FileExistsError(f"replication execution root already exists: {root}")
        stage.rename(root)
    finally:
        if stage.exists() or stage.is_symlink():
            if stage.is_dir() and not stage.is_symlink():
                shutil.rmtree(stage)
            else:
                stage.unlink(missing_ok=True)


def _validate_execution_root(
    root: Path,
    *,
    runtime_contract: str = "execution",
) -> dict[str, object]:
    if runtime_contract not in {"execution", "verifier", "stored"}:
        raise ValueError("replication runtime contract is unsupported")
    expected_activation_digest = _sealed_execution_activation_digest()
    store = _existing_execution_store(root)
    protocol, protocol_raw = _read_canonical_json(
        store,
        Path("protocol.json"),
        field="replication protocol",
    )
    if protocol_raw != ppo_normalization_protocol_bytes():
        raise ValueError("saved replication protocol differs from sealed bytes")
    digest, _digest_raw = _read_canonical_json(
        store,
        Path("protocol.digest.json"),
        field="replication protocol digest",
    )
    if digest != {"sha256": SEALED_PROTOCOL_SHA256}:
        raise ValueError("saved replication protocol digest differs")
    activation, _activation_raw = _read_canonical_json(
        store,
        Path("activation.json"),
        field="replication activation",
    )
    activation_digest, _activation_digest_raw = _read_canonical_json(
        store,
        Path("activation.digest.json"),
        field="replication activation digest",
    )
    if activation_digest != {"digest": expected_activation_digest}:
        raise ValueError("saved activation digest differs")
    checked = _validate_activation(activation)
    if runtime_contract != "stored":
        current_provenance = build_candidate_run_provenance()
        expected_provenance = checked["provenance"]
        if not isinstance(expected_provenance, dict):
            raise ValueError("execution activation provenance is malformed")
        if runtime_contract == "execution":
            if expected_provenance != current_provenance:
                raise ValueError("source/runtime changed from execution activation")
        else:
            _validate_verifier_provenance(expected_provenance, current_provenance)
    slots, _slots_raw = _read_canonical_json(
        store,
        Path("slots.json"),
        field="replication slot roster",
    )
    if slots != {"slots": [spec.slot for spec in replication_arm_specs()]}:
        raise ValueError("replication slot roster changed")
    return checked


_SOURCE_IDENTITY_PATHS = (
    Path("dataset") / "manifest.json",
    Path("dataset") / "arrays.npz",
    Path("study") / "plan.json",
)


def _source_identity_snapshot(source: Path) -> dict[str, dict[str, object]]:
    source = Path(source)
    snapshot: dict[str, dict[str, object]] = {}
    for relative in _SOURCE_IDENTITY_PATHS:
        path = source / relative
        _check_directory_ancestors(path.parent)
        if path.is_symlink() or not path.is_file():
            raise ValueError(
                f"replication source evidence is missing or unsafe: {relative}"
            )
        digest, size = file_digest_and_size(
            path,
            field=f"replication source {relative.as_posix()}",
        )
        snapshot[relative.as_posix()] = {
            "sha256": digest,
            "size_bytes": size,
        }
    return snapshot


def _load_replication_context(
    source: Path,
) -> tuple[MarketDataset, ResolvedRunConfig, int, int]:
    protocol = expected_ppo_normalization_protocol()
    source_contract = protocol["source"]
    if not isinstance(source_contract, dict):
        raise RuntimeError("sealed source contract is malformed")
    identity = inspect_published_market_dataset_artifact(source / "dataset")
    original = load_market_dataset_artifact(source / "dataset")
    if (
        original.dataset_id != source_contract["dataset_id"]
        or identity.artifact_digest != source_contract["dataset_artifact_digest"]
    ):
        raise ValueError("source Dataset differs from sealed replication authority")
    common = protocol["common"]
    if not isinstance(common, dict):
        raise RuntimeError("sealed common contract is malformed")
    symbols = common["symbols"]
    if not isinstance(symbols, list) or tuple(original.symbols) != tuple(symbols):
        raise ValueError("source symbol roster differs from sealed replication")
    plan = inspect_study(source / "study").plan
    if (
        plan.digest != source_contract["study_digest"]
        or plan.dataset_id != original.dataset_id
    ):
        raise ValueError("source Study differs from sealed replication authority")
    dataset = with_price_channels(original)
    start, stop = development_indices(dataset)
    window = common["development_window"]
    if (
        not isinstance(window, dict)
        or stop - start != window["intervals"]
        or str(dataset.timestamps[start]) != "2023-01-01T00:00:00.000000000"
        or str(dataset.timestamps[stop]) != "2025-01-01T00:00:00.000000000"
    ):
        raise ValueError("development window differs from sealed replication")
    return dataset, plan.baseline_config, start, stop


def _manifest_for_bundle(
    store: StudyStore,
    spec: ReplicationArmSpec,
    *,
    expected_digest: str | None = None,
) -> dict[str, object]:
    manifest, _manifest_raw = _read_canonical_json(
        store,
        Path("slots") / spec.slot / "bundle" / "manifest.json",
        field="PPO replication bundle manifest",
    )
    expected_keys = {
        "schema",
        "observation",
        "feature_indices",
        "feature_names",
        "normalizer",
        "policy_sha256",
    }
    if (
        set(manifest) != expected_keys
        or manifest.get("schema") != "ppo_inference_bundle_v1"
    ):
        raise ValueError("PPO replication bundle manifest shape is unsupported")
    if expected_digest is not None:
        pinned = require_sha256(expected_digest, field="bundle_digest")
        if content_digest(manifest) != pinned:
            raise ValueError("PPO replication bundle manifest digest differs")
    policy_sha256 = manifest.get("policy_sha256")
    if not isinstance(policy_sha256, str):
        raise ValueError("PPO replication bundle policy digest is malformed")
    require_sha256(policy_sha256, field="bundle policy_sha256")
    expected_normalizer = spec.normalize_features
    has_normalizer = manifest.get("normalizer") is not None
    if has_normalizer is not expected_normalizer:
        raise ValueError("bundle normalization state differs from replication arm")
    return manifest


def _strict_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def execute_replication_slot(
    source: Path,
    root: Path,
    slot: str,
) -> dict[str, object]:
    """Consume one slot only under the separately sealed execution activation."""

    expected_activation_digest = _sealed_execution_activation_digest()
    activation = _validate_execution_root(root)
    specs = {spec.slot: spec for spec in replication_arm_specs()}
    if slot not in specs:
        raise ValueError("slot is outside the sealed replication roster")
    spec = specs[slot]
    source_snapshot = _source_identity_snapshot(source)
    dataset, config, start, stop = _load_replication_context(source)
    if _source_identity_snapshot(source) != source_snapshot:
        raise ValueError("replication source changed while loading execution context")
    implementation = activation["implementation_digest"]
    if not isinstance(implementation, str):
        raise ValueError("activation implementation digest is malformed")
    _claim_replication_slot(root, spec)
    store = StudyStore(root)
    try:
        fitted = fit_replication_strategy(dataset, config, spec)
        realized_timesteps = _strict_int(
            getattr(fitted.policy, "num_timesteps"),
            field="realized PPO timesteps",
        )
        if realized_timesteps != _PPO_TIMESTEPS:
            raise ValueError(
                "realized PPO timesteps differ from sealed training budget"
            )
        bundle_root = store._checked_target(Path("slots") / spec.slot / "bundle")
        if bundle_root.exists() or bundle_root.is_symlink():
            raise ValueError("replication bundle destination already exists")
        bundle_digest = save_ppo_inference_bundle(
            bundle_root,
            fitted,
            feature_names=tuple(dataset.feature_names),
        )
        manifest = _manifest_for_bundle(
            store,
            spec,
            expected_digest=bundle_digest,
        )
        loaded = load_ppo_inference_bundle(
            bundle_root,
            expected_digest=bundle_digest,
            feature_names=tuple(dataset.feature_names),
        )
        loaded_timesteps = _strict_int(
            getattr(loaded.policy, "num_timesteps"),
            field="reloaded PPO timesteps",
        )
        if loaded_timesteps != _PPO_TIMESTEPS:
            raise ValueError(
                "reloaded PPO timesteps differ from sealed training budget"
            )
        factory = replication_strategy_factory(loaded)
        result = _evaluate_replication_result(
            dataset,
            factory,
            start_index=start,
            stop_index=stop,
            spec=spec,
        )
        if _source_identity_snapshot(source) != source_snapshot:
            raise ValueError("replication source changed during fit or replay")
        _validate_execution_root(root, runtime_contract="execution")
        payload: dict[str, object] = dict(result)
        payload.update(
            schema=SLOT_RESULT_SCHEMA,
            slot=spec.slot,
            protocol_arm=spec.protocol_arm,
            seed=spec.seed,
            normalize_features=spec.normalize_features,
            protocol_sha256=SEALED_PROTOCOL_SHA256,
            activation_digest=expected_activation_digest,
            implementation_digest=implementation,
            bundle_digest=bundle_digest,
            bundle_policy_sha256=manifest.get("policy_sha256"),
            realized_timesteps=realized_timesteps,
            provenance=activation["provenance"],
        )
        result_raw = canonical_json_bytes(payload)
        store.publish_json_once(
            _slot_relative(spec, "result.sha256.json"),
            {"sha256": sha256(result_raw).hexdigest()},
        )
        result_path = _slot_relative(spec, "result.json")
        store.publish_json_once(result_path, payload)
        return payload
    except BaseException as error:
        state = replication_slot_state(root, spec)
        if state["consumed"] and not state["failed"] and not state["result_published"]:
            _record_consumed_failure(root, spec, error=repr(error))
        raise


_RESULT_METADATA = {
    "schema",
    "slot",
    "protocol_arm",
    "seed",
    "normalize_features",
    "protocol_sha256",
    "activation_digest",
    "implementation_digest",
    "bundle_digest",
    "bundle_policy_sha256",
    "realized_timesteps",
    "provenance",
}


def verify_replication_slot(
    source: Path,
    root: Path,
    slot: str,
) -> dict[str, object]:
    """Reload and replay one published bundle without fitting a model."""

    expected_activation_digest = _sealed_execution_activation_digest()
    activation = _validate_execution_root(root, runtime_contract="verifier")
    specs = {spec.slot: spec for spec in replication_arm_specs()}
    if slot not in specs:
        raise ValueError("slot is outside the sealed replication roster")
    spec = specs[slot]
    state = replication_slot_state(root, spec)
    if not state["consumed"] or state["failed"] or not state["result_published"]:
        raise ValueError("slot does not contain complete published evidence")
    store = StudyStore(root)
    claim = _read_consumed_claim(store, spec)
    if claim is None:
        raise ValueError("slot result exists without a consumed claim")
    result, result_raw = _read_canonical_json(
        store,
        _slot_relative(spec, "result.json"),
        field="replication slot result",
    )
    digest, _digest_raw = _read_canonical_json(
        store,
        _slot_relative(spec, "result.sha256.json"),
        field="replication slot result digest",
    )
    if digest != {"sha256": sha256(result_raw).hexdigest()}:
        raise ValueError("slot result digest mismatch")
    implementation = activation["implementation_digest"]
    if (
        result.get("schema") != SLOT_RESULT_SCHEMA
        or result.get("slot") != spec.slot
        or result.get("protocol_arm") != spec.protocol_arm
        or result.get("seed") != spec.seed
        or result.get("normalize_features") is not spec.normalize_features
        or result.get("protocol_sha256") != SEALED_PROTOCOL_SHA256
        or result.get("activation_digest") != expected_activation_digest
        or result.get("implementation_digest") != implementation
        or result.get("provenance") != activation["provenance"]
        or claim.get("activation_digest") != expected_activation_digest
        or claim.get("implementation_digest") != implementation
    ):
        raise ValueError("slot result or claim identity differs from activation")
    source_snapshot = _source_identity_snapshot(source)
    dataset, _config, start, stop = _load_replication_context(source)
    if _source_identity_snapshot(source) != source_snapshot:
        raise ValueError("replication source changed while loading verifier context")
    bundle_digest = result.get("bundle_digest")
    if not isinstance(bundle_digest, str):
        raise ValueError("slot bundle digest is missing")
    bundle_root = store._checked_target(Path("slots") / spec.slot / "bundle")
    manifest = _manifest_for_bundle(
        store,
        spec,
        expected_digest=bundle_digest,
    )
    loaded = load_ppo_inference_bundle(
        bundle_root,
        expected_digest=bundle_digest,
        feature_names=tuple(dataset.feature_names),
    )
    if result.get("bundle_policy_sha256") != manifest.get("policy_sha256"):
        raise ValueError("slot policy digest differs from bundle manifest")
    realized = _strict_int(
        getattr(loaded.policy, "num_timesteps"),
        field="reloaded PPO timesteps",
    )
    if realized != _PPO_TIMESTEPS:
        raise ValueError("reloaded PPO timesteps differ from sealed training budget")
    if result.get("realized_timesteps") != realized:
        raise ValueError("reloaded PPO timestep count differs from result")
    replay = _evaluate_replication_result(
        dataset,
        replication_strategy_factory(loaded),
        start_index=start,
        stop_index=stop,
        spec=spec,
    )
    recorded_replay = {
        key: value for key, value in result.items() if key not in _RESULT_METADATA
    }
    if canonical_json_bytes(replay) != canonical_json_bytes(recorded_replay):
        raise ValueError("fresh bundle replay differs from published result")
    if _source_identity_snapshot(source) != source_snapshot:
        raise ValueError("replication source changed during verifier replay")
    _validate_execution_root(root, runtime_contract="verifier")
    verifier_provenance = build_candidate_run_provenance()
    expected_provenance = activation.get("provenance")
    if not isinstance(expected_provenance, dict):
        raise ValueError("execution activation provenance is malformed")
    _validate_verifier_provenance(expected_provenance, verifier_provenance)
    verification = {
        "schema": SLOT_VERIFICATION_SCHEMA,
        "slot": spec.slot,
        "protocol_arm": spec.protocol_arm,
        "seed": spec.seed,
        "normalize_features": spec.normalize_features,
        "protocol_sha256": SEALED_PROTOCOL_SHA256,
        "activation_digest": expected_activation_digest,
        "implementation_digest": implementation,
        "result_sha256": sha256(result_raw).hexdigest(),
        "bundle_digest": result.get("bundle_digest"),
        "bundle_policy_sha256": result.get("bundle_policy_sha256"),
        "verifier_provenance": verifier_provenance,
        "no_refit": True,
        "replay_verified": True,
    }
    _publish_json_with_sha256_pair(
        store,
        _slot_relative(spec, "verified.json"),
        verification,
        field="replication verification record",
    )
    return result


def _verification_set_payload(
    records: list[dict[str, str]],
) -> dict[str, object]:
    expected_slots = [spec.slot for spec in replication_arm_specs()]
    if [record.get("slot") for record in records] != expected_slots:
        raise ValueError("replication verification set is incomplete or reordered")
    for record in records:
        if set(record) != {"slot", "sha256"}:
            raise ValueError("replication verification set record is malformed")
        digest = record.get("sha256")
        if not isinstance(digest, str):
            raise ValueError("replication verification record digest is malformed")
        require_sha256(digest, field="verification record SHA-256")
    return {
        "schema": _VERIFICATION_SET_SCHEMA,
        "records": records,
    }


def _read_verifier_artifact_authority(
    store: StudyStore,
    *,
    activation_digest: str,
    implementation_digest: str,
    verification_records: list[dict[str, str]],
) -> tuple[dict[str, object], str, str]:
    relative = Path("verifier-authority.json")
    digest_relative = Path("verifier-authority.sha256.json")
    path = store._checked_target(relative)
    digest_path = store._checked_target(digest_relative)
    if (
        not path.exists()
        or not digest_path.exists()
        or path.is_symlink()
        or digest_path.is_symlink()
    ):
        raise ValueError("fresh verifier artifact authority is missing")
    payload, raw = _read_canonical_json(
        store,
        relative,
        field="fresh verifier artifact authority",
    )
    digest_payload, _digest_raw = _read_canonical_json(
        store,
        digest_relative,
        field="fresh verifier artifact authority digest",
    )
    authority_sha256 = sha256(raw).hexdigest()
    if digest_payload != {"sha256": authority_sha256}:
        raise ValueError("fresh verifier artifact authority digest mismatch")

    verification_set = _verification_set_payload(verification_records)
    verification_set_sha256 = content_digest(verification_set)
    expected_keys = {
        "schema",
        "repository_id",
        "run_id",
        "artifact_id",
        "artifact_sha256",
        "artifact_api_digest",
        "code_sha",
        "workflow_sha",
        "activation_digest",
        "implementation_digest",
        "verification_set_sha256",
    }
    if (
        set(payload) != expected_keys
        or payload.get("schema") != VERIFIER_ARTIFACT_AUTHORITY_SCHEMA
        or payload.get("activation_digest") != activation_digest
        or payload.get("implementation_digest") != implementation_digest
        or payload.get("verification_set_sha256") != verification_set_sha256
    ):
        raise ValueError("fresh verifier artifact authority differs from evidence")
    for field in ("repository_id", "run_id", "artifact_id"):
        value = payload.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"fresh verifier {field} must be a positive integer")
    artifact_sha256 = payload.get("artifact_sha256")
    if not isinstance(artifact_sha256, str):
        raise ValueError("fresh verifier artifact SHA-256 is malformed")
    artifact_sha256 = require_sha256(
        artifact_sha256,
        field="fresh verifier artifact SHA-256",
    )
    if payload.get("artifact_api_digest") != f"sha256:{artifact_sha256}":
        raise ValueError("fresh verifier API digest differs from artifact SHA-256")
    for field in ("code_sha", "workflow_sha"):
        value = payload.get(field)
        if (
            not isinstance(value, str)
            or len(value) != 40
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"fresh verifier {field} must be a full commit SHA")
    return payload, authority_sha256, verification_set_sha256


def publish_replication_decision(root: Path) -> dict[str, object]:
    """Publish comparison only after fresh artifact-bound verification of all slots."""

    activation = _validate_execution_root(root, runtime_contract="verifier")
    implementation = activation.get("implementation_digest")
    if not isinstance(implementation, str):
        raise ValueError("activation implementation digest is malformed")
    activation_digest = _sealed_execution_activation_digest()
    store = _existing_execution_store(root)
    control: dict[int, dict[str, Any]] = {}
    candidate: dict[int, dict[str, Any]] = {}
    verified_slots: list[str] = []
    verification_records: list[dict[str, str]] = []

    for spec in replication_arm_specs():
        state = replication_slot_state(root, spec)
        if not state["consumed"] or state["failed"] or not state["result_published"]:
            raise ValueError(f"replication slot is incomplete: {spec.slot}")
        result, result_raw = _read_canonical_json(
            store,
            _slot_relative(spec, "result.json"),
            field="replication slot result",
        )
        digest, _digest_raw = _read_canonical_json(
            store,
            _slot_relative(spec, "result.sha256.json"),
            field="replication slot result digest",
        )
        if digest != {"sha256": sha256(result_raw).hexdigest()}:
            raise ValueError("slot result digest mismatch")
        if (
            result.get("schema") != SLOT_RESULT_SCHEMA
            or result.get("slot") != spec.slot
            or result.get("protocol_arm") != spec.protocol_arm
            or result.get("seed") != spec.seed
            or result.get("normalize_features") is not spec.normalize_features
            or result.get("protocol_sha256") != SEALED_PROTOCOL_SHA256
            or result.get("activation_digest") != activation_digest
            or result.get("implementation_digest") != implementation
            or result.get("provenance") != activation.get("provenance")
        ):
            raise ValueError("slot result identity differs from comparison authority")
        verification = _read_verified_record(
            store,
            spec,
            result=result,
            result_raw=result_raw,
        )
        verification_records.append(
            {
                "slot": spec.slot,
                "sha256": sha256(canonical_json_bytes(verification)).hexdigest(),
            }
        )
        verified_slots.append(spec.slot)
        economic = {
            key: value for key, value in result.items() if key not in _RESULT_METADATA
        }
        target = control if spec.protocol_arm == "control_raw" else candidate
        target[spec.seed] = economic

    (
        verifier_authority,
        verifier_authority_sha256,
        verification_set_sha256,
    ) = _read_verifier_artifact_authority(
        store,
        activation_digest=activation_digest,
        implementation_digest=implementation,
        verification_records=verification_records,
    )
    report = recompute_replication_decision(control, candidate)
    report.update(
        protocol_sha256=SEALED_PROTOCOL_SHA256,
        activation_digest=activation_digest,
        implementation_digest=implementation,
        verified_slots=verified_slots,
        verification_set_sha256=verification_set_sha256,
        verifier_authority_sha256=verifier_authority_sha256,
        verifier_run_id=verifier_authority["run_id"],
        verifier_artifact_id=verifier_authority["artifact_id"],
        all_slots_independently_verified=True,
        economic_result_inspected=True,
    )
    _publish_json_with_sha256_pair(
        store,
        "comparison.json",
        report,
        field="replication comparison",
    )
    return report


__all__ = [
    "ACTIVATION_AUTHORITY_SCHEMA",
    "EXECUTION_ACTIVATION_SCHEMA",
    "ReplicationArmSpec",
    "SEALED_PROTOCOL_SHA256",
    "SLOT_VERIFICATION_SCHEMA",
    "SLOT_RESULT_SCHEMA",
    "VERIFIER_ARTIFACT_AUTHORITY_SCHEMA",
    "execute_replication_slot",
    "fit_replication_strategy",
    "prepare_replication_execution",
    "publish_replication_decision",
    "recompute_replication_decision",
    "replication_arm_specs",
    "replication_slot_state",
    "replication_strategy_factory",
    "verify_replication_slot",
]
