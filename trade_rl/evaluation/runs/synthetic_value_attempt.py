"""Synthetic-only write-once three-arm comparison using the native portfolio book.

This is a software smoke and never an authorized economic Study or final test.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from numbers import Real
from pathlib import Path
from typing import cast

import numpy as np

from trade_rl._validation import require_sha256
from trade_rl.artifacts.atomic_write import atomic_write_bytes
from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.artifacts.verified_file import open_regular_binary
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.replay import (
    SharedCashPortfolioReplayResult,
    run_shared_portfolio_replay,
)
from trade_rl.risk.pretrade import PreTradeRisk, PreTradeRiskConfig
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.decision_value import (
    LinearDecisionValueModel,
    LinearDecisionValuePolicy,
    LinearDecisionValueRuntimeContext,
    load_linear_decision_value,
)
from trade_rl.strategies.portfolio import (
    PortfolioDecisionSnapshot,
    PortfolioTargetProposal,
)
from trade_rl.strategies.portfolio_adapter import CashPortfolioPolicy
from trade_rl.strategies.portfolio_policy_codec import (
    ACTION_HOLD,
    decode_portfolio_action_codes,
    portfolio_action_to_proposal,
)

_FAMILIES = ("cash", "fixed_long", "linear_value")
_STARTED = "synthetic_value_attempt_started_v1"
_ARM = "synthetic_value_attempt_arm_v1"
_RESULT = "synthetic_value_attempt_result_v1"
_COMPLETE = "synthetic_value_attempt_complete_v1"
_FAILURE = "synthetic_value_attempt_failure_v1"


@dataclass(frozen=True, slots=True)
class SyntheticValueAttemptConfig:
    """Declared, restricted artificial input and signed-model identity."""

    source_scope: str
    dataset: MarketDataset
    expected_dataset_id: str
    model_json: str
    expected_model_sha256: str
    execution_cost: ExecutionCostConfig
    risk_config: PreTradeRiskConfig
    start_index: int
    stop_index: int
    initial_capital: float
    fixed_action: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PublishedSyntheticValueAttempt:
    root: Path
    result_sha256: str


@dataclass(frozen=True, slots=True)
class LoadedSyntheticValueAttempt:
    root: Path
    result_sha256: str
    model_sha256: str
    dataset_id: str
    arms: tuple[dict[str, object], ...]


class _FixedEntryPolicy:
    """One fixed entry followed by exact-quantity HOLD; no nightly rebalancing."""

    def __init__(
        self, action: tuple[int, ...], target_abs_weights: tuple[float, ...]
    ) -> None:
        self._action = action
        self._weights = target_abs_weights
        self._entered = False

    def decide_portfolio(
        self, snapshot: PortfolioDecisionSnapshot
    ) -> PortfolioTargetProposal:
        codes = (
            (ACTION_HOLD,) * len(snapshot.symbols)
            if self._entered
            else self._action
        )
        self._entered = True
        return portfolio_action_to_proposal(codes, snapshot, self._weights)


def _source_fingerprint(dataset: MarketDataset) -> str:
    """Watch all in-memory synthetic inputs for drift; not external attestation."""
    digest = hashlib.sha256()
    for field in fields(dataset):
        if not field.init:
            continue
        value = getattr(dataset, field.name)
        if isinstance(value, np.ndarray):
            item = {
                "kind": "array",
                "dtype": value.dtype.str,
                "shape": value.shape,
                "digest": hashlib.sha256(value.tobytes()).hexdigest(),
            }
        else:
            item = {"kind": "value", "value": value}
        encoded = canonical_json_bytes((field.name, item))
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _preflight(
    config: SyntheticValueAttemptConfig,
    observer: Callable[[str], None] | None,
) -> tuple[LinearDecisionValueModel, str, dict[str, object]]:
    if type(config) is not SyntheticValueAttemptConfig:
        raise TypeError("config must be SyntheticValueAttemptConfig")
    if config.source_scope != "synthetic_only":
        raise ValueError("TD4a accepts synthetic-only input")
    if observer is not None and not callable(observer):
        raise TypeError("observer must be callable")
    dataset = config.dataset
    if type(dataset) is not MarketDataset:
        raise TypeError("dataset must be MarketDataset")
    require_sha256(config.expected_dataset_id, field="expected_dataset_id")
    if dataset.dataset_id != config.expected_dataset_id:
        raise ValueError("dataset identity does not match expected pin")
    if dataset.identity_verified:
        raise ValueError("synthetic trial rejects a verified published Dataset")
    if dataset.n_symbols != 2 or dataset.n_bars > 64:
        raise ValueError("synthetic trial supports two symbols and at most 64 bars")
    if type(config.execution_cost) is not ExecutionCostConfig:
        raise TypeError("execution_cost must be ExecutionCostConfig")
    if type(config.risk_config) is not PreTradeRiskConfig:
        raise TypeError("risk_config must be PreTradeRiskConfig")
    cost = config.execution_cost
    if cost.slippage_std != 0 or cost.tail_slippage_probability != 0:
        raise ValueError("stochastic execution is not supported by TD4a")
    if (
        type(config.start_index) is not int
        or type(config.stop_index) is not int
        or not 0 <= config.start_index < config.stop_index < dataset.n_bars
        or config.stop_index - config.start_index < cost.order_latency_bars + 2
    ):
        raise ValueError("trial window lacks a valid native decision/terminal clock")
    if isinstance(config.initial_capital, bool) or not isinstance(
        config.initial_capital, Real
    ):
        raise TypeError("initial_capital must be a positive real")
    capital = float(config.initial_capital)
    if not math.isfinite(capital) or capital <= 0:
        raise ValueError("initial_capital must be finite and positive")
    model = load_linear_decision_value(
        config.model_json, expected_sha256=config.expected_model_sha256
    )
    if (
        model.symbols != dataset.symbols
        or model.feature_names != dataset.feature_names
        or model.global_feature_names != dataset.global_feature_names
        or model.feature_config_digest != dataset.feature_config_digest
        or model.normalization_digest != dataset.normalization_digest
    ):
        raise ValueError("model and Dataset feature/symbol roster differs")
    if model.execution_policy_digest != cost.execution_policy_digest:
        raise ValueError("execution_policy_digest differs from the fitted model")
    start_ns = int(dataset.timestamps[config.start_index].astype(np.int64))
    if start_ns < model.evaluation_start_ns:
        raise ValueError("evaluation source window precedes the model activation")
    if (
        type(config.fixed_action) is not tuple
        or decode_portfolio_action_codes(
            config.fixed_action, n_symbols=dataset.n_symbols
        ) not in model.candidate_actions[1:]
    ):
        raise ValueError("fixed action must be a declared non-HOLD candidate")
    source_times = dataset.resolved_array("available_at")
    event_times = dataset.timestamps[:, None]
    lower, upper = config.start_index, config.stop_index + 1
    if np.any(source_times[lower:upper] > event_times[lower:upper]):
        raise ValueError("synthetic source unavailable at the native bar time")
    fingerprint = _source_fingerprint(dataset)
    started = {
        "schema_version": _STARTED,
        "source_scope": "synthetic_only",
        "dataset_id": dataset.dataset_id,
        "source_fingerprint": fingerprint,
        "model_sha256": config.expected_model_sha256,
        "execution_policy_digest": cost.execution_policy_digest,
        "start_index": config.start_index,
        "stop_index": config.stop_index,
        "initial_capital": capital,
        "execution_cost": asdict(cost),
        "risk_config": asdict(config.risk_config),
        "fixed_action": config.fixed_action,
        "arm_roster": _FAMILIES,
    }
    return model, fingerprint, started


def _write_record(root: Path, relative: str, value: object) -> str:
    path = root / relative
    if path.exists() or path.is_symlink():
        raise FileExistsError("attempt record already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json_bytes(value)
    atomic_write_bytes(path, encoded)
    return hashlib.sha256(encoded).hexdigest()


def _read_record(root: Path, relative: str) -> tuple[dict[str, object], str]:
    with open_regular_binary(root / relative, field=relative) as handle:
        encoded = handle.read()
    try:
        decoded: object = json.loads(encoded.decode("utf-8"))
    except (UnicodeError, ValueError) as error:
        raise ValueError(f"invalid attempt JSON: {relative}") from error
    if not isinstance(decoded, dict) or canonical_json_bytes(decoded) != encoded:
        raise ValueError(f"attempt JSON is not canonical: {relative}")
    return cast(dict[str, object], decoded), hashlib.sha256(encoded).hexdigest()


def _arm_row(
    config: SyntheticValueAttemptConfig,
    family: str,
    replay: SharedCashPortfolioReplayResult,
) -> dict[str, object]:
    if replay.ledger_evidence is None:
        raise RuntimeError("native replay did not capture ledger evidence")
    ledger = replay.ledger_evidence.to_mapping()
    intervals = ledger.get("intervals")
    if not isinstance(intervals, tuple):
        raise RuntimeError("native ledger intervals are not an ordered tuple")
    if (
        ledger.get("dataset_id") != config.expected_dataset_id
        or ledger.get("execution_policy_digest")
        != config.execution_cost.execution_policy_digest
    ):
        raise ValueError("native Dataset/execution ledger identity changed")
    if ledger.get("final_portfolio_value") != replay.book.portfolio_value:
        raise ValueError("native ledger/book terminal equity differs")
    if ledger.get("final_total_cost") != replay.book.total_cost:
        raise ValueError("native ledger/book execution cost differs")
    valid = (
        len(intervals) == config.stop_index - config.start_index
        and not replay.book.insolvent
        and replay.book.termination_reason is None
        and not ledger["active_order_remainders"]
        and all(q == 0 for q in replay.book.exact_quantities)
    )
    return {
        "schema_version": _ARM,
        "family": family,
        "valid_endpoint": valid,
        "terminal_equity": float(replay.book.portfolio_value),
        "total_cost": float(replay.book.total_cost),
        "ledger": ledger,
    }


def run_synthetic_value_attempt(
    attempt_root: str | Path,
    config: SyntheticValueAttemptConfig,
    *,
    on_arm_completed: Callable[[str], None] | None = None,
) -> PublishedSyntheticValueAttempt:
    """One-shot synthetic comparison; persist each raw arm before next starts."""
    model, fingerprint, started = _preflight(config, on_arm_completed)
    root = Path(attempt_root).expanduser().absolute()
    if root.exists() or root.is_symlink():
        raise FileExistsError("attempt destination already exists")
    if any(parent.is_symlink() for parent in root.parents):
        raise ValueError("attempt parent must not be a symlink")
    if not root.parent.is_dir():
        raise ValueError("attempt parent must already exist")
    root.mkdir()
    completed: list[str] = []
    summaries: list[dict[str, object]] = []
    phase = "started"
    result_digest: str | None = None
    try:
        started_digest = _write_record(root, "started/record.json", started)
        _write_record(root, "input/model.json", json.loads(config.model_json))
        runtime = LinearDecisionValueRuntimeContext(
            source_scope="synthetic_only",
            feature_config_digest=config.dataset.feature_config_digest,
            normalization_digest=config.dataset.normalization_digest,
            execution_policy_digest=config.execution_cost.execution_policy_digest,
            reference_policy_digest=model.reference_policy_digest,
        )
        policies = (
            CashPortfolioPolicy(symbols=config.dataset.symbols),
            _FixedEntryPolicy(config.fixed_action, model.target_abs_weights),
            LinearDecisionValuePolicy(model, runtime_context=runtime),
        )
        for family, policy in zip(_FAMILIES, policies, strict=True):
            phase = family
            if _source_fingerprint(config.dataset) != fingerprint:
                raise RuntimeError("synthetic Dataset changed before native arm")
            native = run_shared_portfolio_replay(
                config.dataset,
                policy,
                start_index=config.start_index,
                stop_index=config.stop_index,
                initial_capital=float(config.initial_capital),
                execution_cost=config.execution_cost,
                risk=PreTradeRisk(config.risk_config),
                settle_terminal_position=True,
                capture_ledger_evidence=True,
            )
            row = _arm_row(config, family, native)
            digest = _write_record(root, f"arms/{family}/record.json", row)
            summaries.append(
                {
                    "family": family,
                    "record_sha256": digest,
                    "terminal_equity": row["terminal_equity"],
                }
            )
            completed.append(family)
            if on_arm_completed is not None:
                on_arm_completed(family)
            if not row["valid_endpoint"]:
                raise RuntimeError(f"native {family} endpoint is incomplete")
        if _source_fingerprint(config.dataset) != fingerprint:
            raise RuntimeError("synthetic Dataset changed after comparison")
        phase = "result"
        result = {
            "schema_version": _RESULT,
            "source_scope": "synthetic_only",
            "dataset_id": config.expected_dataset_id,
            "source_fingerprint": fingerprint,
            "model_sha256": config.expected_model_sha256,
            "started_sha256": started_digest,
            "arms": summaries,
        }
        result_digest = _write_record(root, "result/record.json", result)
        _read_result(
            root,
            expected_result_sha256=result_digest,
            expected_model_sha256=config.expected_model_sha256,
            expected_dataset_id=config.expected_dataset_id,
        )
        phase = "completion"
        _write_record(
            root,
            "complete/record.json",
            {"schema_version": _COMPLETE, "result_sha256": result_digest},
        )
        return PublishedSyntheticValueAttempt(root=root, result_sha256=result_digest)
    except BaseException as error:
        failure = {
            "schema_version": _FAILURE,
            "phase": phase,
            "completed_families": completed,
            "result_sha256": result_digest,
            "exception_type": type(error).__name__,
            "exception_message": str(error),
        }
        try:
            _write_record(root, "failure/record.json", failure)
        except BaseException as secondary:
            error.add_note(f"failure-journal write also failed: {secondary}")
        raise


def _read_result(
    root: Path,
    *,
    expected_result_sha256: str,
    expected_model_sha256: str,
    expected_dataset_id: str,
) -> tuple[dict[str, object], ...]:
    result, digest = _read_record(root, "result/record.json")
    if digest != expected_result_sha256:
        raise ValueError("attempt result digest mismatch")
    started, started_digest = _read_record(root, "started/record.json")
    if (
        result.get("schema_version") != _RESULT
        or result.get("source_scope") != "synthetic_only"
        or result.get("dataset_id") != expected_dataset_id
        or result.get("model_sha256") != expected_model_sha256
        or result.get("started_sha256") != started_digest
        or result.get("source_fingerprint") != started.get("source_fingerprint")
        or started.get("schema_version") != _STARTED
        or started.get("source_scope") != "synthetic_only"
        or started.get("dataset_id") != expected_dataset_id
        or started.get("model_sha256") != expected_model_sha256
    ):
        raise ValueError("attempt source or model identity mismatch")
    with open_regular_binary(root / "input/model.json", field="model") as handle:
        model_bytes = handle.read()
    try:
        model_text = model_bytes.decode("utf-8")
    except UnicodeError as error:
        raise ValueError("model bytes are not UTF-8") from error
    model = load_linear_decision_value(
        model_text, expected_sha256=expected_model_sha256
    )
    arm_items = result.get("arms")
    if not isinstance(arm_items, list) or len(arm_items) != len(_FAMILIES):
        raise ValueError("attempt result has incomplete arm roster")
    rows: list[dict[str, object]] = []
    for index, family in enumerate(_FAMILIES):
        item = arm_items[index]
        if not isinstance(item, dict) or item.get("family") != family:
            raise ValueError("attempt result has wrong arm order")
        expected_row_digest = item.get("record_sha256")
        if not isinstance(expected_row_digest, str):
            raise ValueError("arm record digest must be a SHA-256 string")
        require_sha256(expected_row_digest, field="arm record digest")
        row, digest = _read_record(root, f"arms/{family}/record.json")
        if digest != expected_row_digest:
            raise ValueError("attempt arm record digest mismatch (tamper)")
        ledger = row.get("ledger")
        if not isinstance(ledger, dict):
            raise ValueError("native arm ledger must be a mapping")
        if (
            row.get("schema_version") != _ARM
            or row.get("family") != family
            or row.get("valid_endpoint") is not True
            or item.get("terminal_equity") != row.get("terminal_equity")
            or ledger.get("dataset_id") != expected_dataset_id
            or ledger.get("execution_policy_digest")
            != started.get("execution_policy_digest")
            or ledger.get("final_portfolio_value") != row.get("terminal_equity")
            or ledger.get("final_total_cost") != row.get("total_cost")
            or ledger.get("terminal_exact_quantities") != ["0"] * len(model.symbols)
            or ledger.get("active_order_remainders") != []
        ):
            raise ValueError("attempt arm violates native terminal ledger contract")
        rows.append(row)
    return tuple(rows)


def load_synthetic_value_attempt(
    attempt_root: str | Path,
    *,
    expected_result_sha256: str,
    expected_model_sha256: str,
    expected_dataset_id: str,
) -> LoadedSyntheticValueAttempt:
    """Strictly reload a completed software attempt with external digest pins."""
    for name, value in (
        ("result digest", expected_result_sha256),
        ("model digest", expected_model_sha256),
        ("dataset identity", expected_dataset_id),
    ):
        require_sha256(value, field=name)
    root = Path(attempt_root).expanduser().absolute()
    if (root / "failure").exists() or not (root / "complete/record.json").is_file():
        raise ValueError("attempt has failure evidence or no complete marker")
    complete, _ = _read_record(root, "complete/record.json")
    if (
        complete.get("schema_version") != _COMPLETE
        or complete.get("result_sha256") != expected_result_sha256
    ):
        raise ValueError("attempt complete marker differs from expected result")
    arms = _read_result(
        root,
        expected_result_sha256=expected_result_sha256,
        expected_model_sha256=expected_model_sha256,
        expected_dataset_id=expected_dataset_id,
    )
    return LoadedSyntheticValueAttempt(
        root=root,
        result_sha256=expected_result_sha256,
        model_sha256=expected_model_sha256,
        dataset_id=expected_dataset_id,
        arms=arms,
    )


__all__ = [
    "SyntheticValueAttemptConfig",
    "PublishedSyntheticValueAttempt",
    "LoadedSyntheticValueAttempt",
    "run_synthetic_value_attempt",
    "load_synthetic_value_attempt",
]
