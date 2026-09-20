"""Execution primitives for the sealed PPO normalization replication."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from trade_rl._validation import require_sha256
from trade_rl.artifacts import canonical_json_bytes, content_digest
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
SEALED_PROTOCOL_SHA256 = (
    "0013470ed5858eaa3b9391f97f4b18f772495d21c128832f74e1c50b090df304"
)
SLOT_RESULT_SCHEMA = "ppo_normalization_replication_result_v1"


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
    activation = require_sha256(activation_digest, field="activation_digest")
    implementation = require_sha256(
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


def _validate_activation(
    activation: dict[str, object],
    *,
    expected_digest: str,
) -> dict[str, object]:
    require_sha256(expected_digest, field="expected_activation_digest")
    expected_keys = {
        "schema",
        "protocol_sha256",
        "implementation_digest",
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
    provenance = activation["provenance"]
    if not isinstance(provenance, dict):
        raise ValueError("execution activation provenance is malformed")
    if provenance.get("implementation_digest") != implementation:
        raise ValueError("activation implementation digest differs from provenance")
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
    *,
    expected_activation_digest: str,
) -> None:
    """Persist a separately authorized one-shot activation without running economics."""

    _validate_activation(activation, expected_digest=expected_activation_digest)
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"replication execution root already exists: {root}")
    root.mkdir(parents=True)
    store = StudyStore(root)
    protocol = expected_ppo_normalization_protocol()
    raw = ppo_normalization_protocol_bytes()
    if sha256(raw).hexdigest() != SEALED_PROTOCOL_SHA256:
        raise RuntimeError("sealed normalization protocol bytes drifted")
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


def _validate_execution_root(
    root: Path,
    *,
    expected_activation_digest: str,
) -> dict[str, object]:
    store = StudyStore(root)
    protocol = store.read_json("protocol.json")
    if canonical_json_bytes(protocol) != ppo_normalization_protocol_bytes():
        raise ValueError("saved replication protocol differs from sealed bytes")
    digest = store.read_json("protocol.digest.json")
    if digest != {"sha256": SEALED_PROTOCOL_SHA256}:
        raise ValueError("saved replication protocol digest differs")
    activation = store.read_json("activation.json")
    activation_digest = store.read_json("activation.digest.json")
    if activation_digest != {"digest": expected_activation_digest}:
        raise ValueError("saved activation digest differs")
    checked = _validate_activation(
        activation,
        expected_digest=expected_activation_digest,
    )
    if checked["provenance"] != build_candidate_run_provenance():
        raise ValueError("source/runtime changed from execution activation")
    slots = store.read_json("slots.json")
    if slots != {"slots": [spec.slot for spec in replication_arm_specs()]}:
        raise ValueError("replication slot roster changed")
    return checked


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
    store: StudyStore, spec: ReplicationArmSpec
) -> dict[str, object]:
    manifest = store.read_json(Path("slots") / spec.slot / "bundle" / "manifest.json")
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
    *,
    expected_activation_digest: str,
) -> dict[str, object]:
    """Consume one slot, fit once, persist a bundle, then replay the saved bundle."""

    activation = _validate_execution_root(
        root,
        expected_activation_digest=expected_activation_digest,
    )
    specs = {spec.slot: spec for spec in replication_arm_specs()}
    if slot not in specs:
        raise ValueError("slot is outside the sealed replication roster")
    spec = specs[slot]
    dataset, config, start, stop = _load_replication_context(source)
    implementation = activation["implementation_digest"]
    if not isinstance(implementation, str):
        raise ValueError("activation implementation digest is malformed")
    claim_replication_slot(
        root,
        spec,
        activation_digest=expected_activation_digest,
        implementation_digest=implementation,
    )
    store = StudyStore(root)
    try:
        fitted = fit_replication_strategy(dataset, config, spec)
        realized_timesteps = _strict_int(
            getattr(fitted.policy, "num_timesteps"),
            field="realized PPO timesteps",
        )
        bundle_root = store.root / "slots" / spec.slot / "bundle"
        bundle_digest = save_ppo_inference_bundle(
            bundle_root,
            fitted,
            feature_names=tuple(dataset.feature_names),
        )
        loaded = load_ppo_inference_bundle(
            bundle_root,
            expected_digest=bundle_digest,
            feature_names=tuple(dataset.feature_names),
        )
        manifest = _manifest_for_bundle(store, spec)
        factory = replication_strategy_factory(loaded)
        result = evaluate_directional_arm(
            dataset,
            factory,
            start_index=start,
            stop_index=stop,
            initial_capital=10_000.0,
            gross_budget=0.1,
        )
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
        result_path = _slot_relative(spec, "result.json")
        store.publish_json_once(result_path, payload)
        result_raw = canonical_json_bytes(payload)
        store.publish_json_once(
            _slot_relative(spec, "result.sha256.json"),
            {"sha256": sha256(result_raw).hexdigest()},
        )
        return payload
    except BaseException as error:
        state = replication_slot_state(root, spec)
        if state["consumed"] and not state["failed"] and not state["result_published"]:
            record_consumed_failure(root, spec, error=repr(error))
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
    *,
    expected_activation_digest: str,
) -> dict[str, object]:
    """Reload and replay one published bundle without fitting a model."""

    activation = _validate_execution_root(
        root,
        expected_activation_digest=expected_activation_digest,
    )
    specs = {spec.slot: spec for spec in replication_arm_specs()}
    if slot not in specs:
        raise ValueError("slot is outside the sealed replication roster")
    spec = specs[slot]
    state = replication_slot_state(root, spec)
    if not state["consumed"] or state["failed"] or not state["result_published"]:
        raise ValueError("slot does not contain complete published evidence")
    store = StudyStore(root)
    result = store.read_json(_slot_relative(spec, "result.json"))
    digest = store.read_json(_slot_relative(spec, "result.sha256.json"))
    if canonical_json_bytes(digest) != canonical_json_bytes(
        {"sha256": sha256(canonical_json_bytes(result)).hexdigest()}
    ):
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
    ):
        raise ValueError("slot result identity differs from activation")
    dataset, _config, start, stop = _load_replication_context(source)
    bundle_digest = result.get("bundle_digest")
    if not isinstance(bundle_digest, str):
        raise ValueError("slot bundle digest is missing")
    bundle_root = store.root / "slots" / spec.slot / "bundle"
    loaded = load_ppo_inference_bundle(
        bundle_root,
        expected_digest=bundle_digest,
        feature_names=tuple(dataset.feature_names),
    )
    manifest = _manifest_for_bundle(store, spec)
    if result.get("bundle_policy_sha256") != manifest.get("policy_sha256"):
        raise ValueError("slot policy digest differs from bundle manifest")
    realized = _strict_int(
        getattr(loaded.policy, "num_timesteps"),
        field="reloaded PPO timesteps",
    )
    if result.get("realized_timesteps") != realized:
        raise ValueError("reloaded PPO timestep count differs from result")
    replay = evaluate_directional_arm(
        dataset,
        replication_strategy_factory(loaded),
        start_index=start,
        stop_index=stop,
        initial_capital=10_000.0,
        gross_budget=0.1,
    )
    recorded_replay = {
        key: value for key, value in result.items() if key not in _RESULT_METADATA
    }
    if canonical_json_bytes(replay) != canonical_json_bytes(recorded_replay):
        raise ValueError("fresh bundle replay differs from published result")
    return result


__all__ = [
    "EXECUTION_ACTIVATION_SCHEMA",
    "ReplicationArmSpec",
    "SEALED_PROTOCOL_SHA256",
    "SLOT_RESULT_SCHEMA",
    "claim_replication_slot",
    "execute_replication_slot",
    "fit_replication_strategy",
    "prepare_replication_execution",
    "recompute_replication_decision",
    "record_consumed_failure",
    "record_prefit_failure",
    "replication_arm_specs",
    "replication_slot_state",
    "replication_strategy_factory",
    "verify_replication_slot",
]
