"""Maintained development NONRL/CASH tool; no implicit market execution."""

from __future__ import annotations

import concurrent.futures
import json
import sys
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from trade_rl.artifacts import content_digest
from trade_rl.data import load_market_dataset_artifact
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.allocation_market_inputs import (
    PreparedAllocationMarketInputs,
    prepare_allocation_market_inputs,
)
from trade_rl.evaluation.allocation_settled_forecast_comparison import (
    SettledNonRLComparisonPlan,
    build_settled_nonrl_comparison,
)
from trade_rl.evaluation.allocation_terminal_evidence import (
    TerminalAllocationExecutionReceipt,
)
from trade_rl.evaluation.allocation_terminal_execution import (
    run_terminal_allocation_execution,
)
from trade_rl.evaluation.forecast_allocation import HorizonCostEstimates
from trade_rl.evaluation.runs import build_candidate_run_provenance
from trade_rl.strategies.forecasts.simple_stream_io import (
    PublishedSimpleReturnStreamArtifact,
)

from .progress import ArmProgressSpec, validate_progress_policy
from .progress_io import ProgressJournal, ProgressReader, publish_roster
from .resume import freeze_inputs, preflight_resume
from .tier_d_decomposition import ledger_decomposition
from .tier_d_fixture import write_artificial_dataset
from .tier_d_native import (
    PairDeclaration,
    carrier,
    cost_rows,
    declare_pair,
    execution_folds,
    forecast_blocks,
    synthetic_protocol,
)
from .tier_d_protocol import (
    DATASET_ID,
    DATASET_ROOT,
    IMPLEMENTATION_MODELS,
    INPUT_PINS,
    file_sha256,
    json_sha256,
    market_protocol,
    source_head,
    source_pins,
    validate_committed_source,
    worker_count,
)
from .tier_d_records import (
    append_log,
    arm_state,
    combined_rows,
    initialize_attempt,
    load_gzip,
    native_closure,
    output_manifest,
    read_json,
    save_gzip,
    write_json,
)

CostTables = dict[tuple[int, str], tuple[HorizonCostEstimates, ...]]
PreparedTables = dict[str, PreparedAllocationMarketInputs]


def validate_external_review(
    path: Path | None, protocol: dict[str, Any]
) -> dict[str, Any]:
    if path is None:
        raise ValueError(
            "ROOT different-model review is required before actual costs/fits/PnL"
        )
    review = read_json(path)
    if (
        review.get("scope") != protocol["scope"]
        or review.get("status") != "APPROVED_FOR_TIER_D_DEVELOPMENT_EXECUTION"
        or type(review.get("reviewer_model")) is not str
        or not review["reviewer_model"].strip()
        or review["reviewer_model"].casefold() == "unknown"
        or review["reviewer_model"] in IMPLEMENTATION_MODELS
        or review.get("protocol_digest") != json_sha256(protocol)
        or review.get("source_sha256") != source_pins()
    ):
        raise ValueError(
            "ROOT different-model review must cover this exact protocol and driver"
        )
    # This record is an operator reminder and provenance. It cannot authenticate
    # an AI provider, research authority or data eligibility by itself.
    return {"review_path": str(path), "review_sha256": file_sha256(path), **review}


def validate_market_input(protocol: dict[str, Any]) -> MarketDataset:
    validate_committed_source(protocol)
    if source_head() != protocol["source_head"]:
        raise ValueError(
            "reviewed source HEAD changed; re-review the ordinary prototype"
        )
    manifest = json.loads((DATASET_ROOT / "manifest.json").read_text(encoding="utf-8"))
    if manifest["dataset_id"] != DATASET_ID:
        raise ValueError("Task18 metadata differs from the result-blind protocol")
    for name, digest in INPUT_PINS.items():
        if file_sha256(DATASET_ROOT / name) != digest:
            raise ValueError(f"Task18 input bytes changed: {name}")
    dataset = load_market_dataset_artifact(DATASET_ROOT)
    if (
        [dataset.n_bars, dataset.n_symbols, dataset.n_features] != [9504, 2, 6]
        or not dataset.identity_verified
        or dataset.dataset_id != DATASET_ID
        or list(dataset.feature_names) != protocol["dataset"]["feature_names"]
        or list(dataset.symbols) != protocol["dataset"]["fit_symbols"]
    ):
        raise ValueError("Task18 public Dataset contract changed")
    for date, expected in zip(
        (
            *protocol["actor"]["vintage_start_utc"],
            protocol["actor"]["stop_utc"],
            protocol["terminal"]["utc"],
        ),
        (
            *protocol["actor"]["vintage_starts"],
            protocol["actor"]["stop"],
            protocol["terminal"]["stop_inclusive"],
        ),
        strict=True,
    ):
        rows = np.flatnonzero(dataset.timestamps == np.datetime64(date, "ns"))
        if rows.tolist() != [expected]:
            raise ValueError(
                f"literal UTC clock does not match source-derived index: {date}"
            )
    if not np.array_equal(
        dataset.resolved_array("funding_price_rate"),
        dataset.funding_rate * dataset.resolved_array("mark_price"),
    ):
        raise ValueError("Task18 declared bar-close funding proxy changed")
    return dataset


def prepare_inputs(
    root: Path,
    dataset_root: Path,
    dataset: MarketDataset,
    protocol: dict[str, Any],
) -> tuple[PreparedTables, CostTables]:
    tables: CostTables = {}
    cost_scenarios = (
        ("BASEL0", "COST2L0")
        if protocol["budget"]["native_arms"] == 48
        else ("BASEL0",)
    )
    for horizon in sorted({row["horizon_hours"] for row in protocol["configurations"]}):
        for scenario in cost_scenarios:
            append_log(
                root, "cost_producer_started", horizon=horizon, scenario=scenario
            )
            tables[horizon, scenario] = cost_rows(dataset, protocol, horizon, scenario)
    prepared: PreparedTables = {}
    for config in protocol["configurations"]:
        append_log(root, "fit_started", configuration=config["id"])
        stream_path = root / "forecasts" / f"{config['id']}.json"
        prepared[config["id"]] = prepare_allocation_market_inputs(
            dataset_root,
            stream_path,
            expected_dataset_id=dataset.dataset_id,
            development_start=np.datetime64(
                protocol["dataset"]["development_start"], "ns"
            ),
            development_stop=np.datetime64(
                protocol["dataset"]["development_stop_exclusive"], "ns"
            ),
            feature_names=dataset.feature_names,
            fit_symbols=dataset.symbols,
            blocks=forecast_blocks(
                dataset,
                tuple(protocol["actor"]["vintage_starts"]),
                protocol["actor"]["stop"],
            ),
            horizon_hours=config["horizon_hours"],
            alpha=config["ridge_alpha"],
            cost_rows=tuple(
                row.payload() for row in tables[config["horizon_hours"], "BASEL0"]
            ),
        )
        append_log(
            root,
            "fit_completed",
            configuration=config["id"],
            vintages=len(prepared[config["id"]].stream.vintages),
        )
    return prepared, tables


def symbol_estimates(
    tables: CostTables, horizon: int, scenario: str, symbol: str
) -> tuple[HorizonCostEstimates, ...]:
    return tuple(
        row
        for row in tables[horizon, "COST2L0" if scenario == "COST2L0" else "BASEL0"]
        if row.symbol == symbol
    )


def declare_matrix(
    prepared: PreparedTables,
    tables: CostTables,
    protocol: dict[str, Any],
    provenance: dict[str, Any],
) -> tuple[PairDeclaration, ...]:
    pairs = []
    for config in protocol["configurations"]:
        inputs = prepared[config["id"]]
        symbols = (
            inputs.dataset.symbols
            if protocol["budget"]["native_arms"] == 48
            else inputs.dataset.symbols[:1]
        )
        for index, symbol in enumerate(symbols):
            account = (
                f"{config['id']}_{symbol}"
                if len(symbols) == 2
                else protocol["accounts"][0]
            )
            for scenario in protocol["scenarios"]:
                pairs.append(
                    declare_pair(
                        inputs,
                        symbol_estimates(
                            tables, config["horizon_hours"], scenario, symbol
                        ),
                        protocol,
                        horizon=config["horizon_hours"],
                        symbol_index=index,
                        account=account,
                        scenario=scenario,
                        configuration=config["id"],
                        provenance=provenance,
                    )
                )
    return tuple(pairs)


@dataclass(frozen=True)
class PairExecutionResult:
    configuration: str
    account: str
    scenario: str
    symbol_index: int
    row: dict[str, Any] | None
    closures: dict[str, Any]
    arm_slots: dict[str, dict[str, Any]]
    failure: dict[str, str] | None = None


def _failure(error: Exception) -> dict[str, str]:
    return {"type": type(error).__name__, "message": str(error)}


def _worker_execute_pair(
    root: Path,
    pair: PairDeclaration,
    inputs: PreparedAllocationMarketInputs,
    estimates: tuple[HorizonCostEstimates, ...],
    protocol: dict[str, Any],
    *,
    progress_enabled: bool = False,
    on_progress: Callable[[], None] | None = None,
) -> PairExecutionResult:
    records: list[dict[str, Any]] = []
    closures: dict[str, Any] = {}
    decomposition: dict[str, Any] = {}
    slots: dict[str, dict[str, Any]] = {}
    folds = execution_folds(
        tuple(protocol["actor"]["vintage_starts"]), protocol["actor"]["stop"]
    )
    horizon = next(
        c["horizon_hours"]
        for c in protocol["configurations"]
        if c["id"] == pair.configuration
    )

    def outcome(
        row: dict[str, Any] | None, failure: dict[str, str] | None = None
    ) -> PairExecutionResult:
        return PairExecutionResult(
            pair.configuration,
            pair.account,
            pair.scenario,
            pair.symbol_index,
            row,
            closures,
            slots,
            failure,
        )

    for role, plan in zip(("nonrl", "cash"), pair.controls.plans, strict=True):
        relative = f"receipts/{pair.account}_{pair.scenario}_{role}.json.gz"
        receipt_path = root / relative
        raw = None
        deferred_progress_error: Exception | None = None
        execution_started = False
        # A rejected retained file is never an instruction to execute again.
        if receipt_path.exists():
            try:
                raw = load_gzip(receipt_path)
                receipt = TerminalAllocationExecutionReceipt.from_payload(
                    raw, expected_plan=plan
                )
            except Exception as error:
                if not slots:
                    raise
                failure = _failure(error)
                slots[role] = {"state": "FAILED", "failure": failure, "receipt": None}
                return outcome(None, failure)
            slots[role] = {
                "state": "COMPLETED",
                "receipt_digest": receipt.digest,
                "plan_digest": plan.digest,
                "reused": True,
                "execution_started": False,
                "receipt": {
                    "path": relative,
                    "sha256": file_sha256(receipt_path),
                    "bytes": receipt_path.stat().st_size,
                },
            }
        try:
            env = carrier(
                inputs,
                estimates,
                protocol,
                horizon=horizon,
                symbol_index=pair.symbol_index,
                account=pair.account,
                scenario=pair.scenario,
                role=role,
            )
            if raw is None:
                journal = (
                    ProgressJournal(
                        root,
                        ArmProgressSpec(
                            pair.configuration,
                            pair.account,
                            pair.scenario,
                            role,
                            plan.digest,
                            protocol["actor"]["start"],
                            protocol["actor"]["stop"],
                            protocol["terminal"]["stop_inclusive"],
                        ),
                        on_change=on_progress,
                    )
                    if progress_enabled
                    else nullcontext()
                )
                observed = None
                try:
                    with journal as observer:
                        execution_started = True
                        observed = (
                            run_terminal_allocation_execution(
                                folds, env, plan, on_advance=observer.advance
                            )
                            if observer is not None
                            else run_terminal_allocation_execution(folds, env, plan)
                        )
                except Exception as error:
                    if observed is None:
                        raise
                    # Retain returned facts even when lifecycle publication fails.
                    deferred_progress_error = error
                raw = observed.payload
                receipt = TerminalAllocationExecutionReceipt.from_payload(
                    raw, expected_plan=plan
                )
                metadata = save_gzip(root, relative, raw)
                slots[role] = {
                    "state": "COMPLETED",
                    "receipt_digest": receipt.digest,
                    "plan_digest": plan.digest,
                    "receipt": metadata,
                    "reused": False,
                    "execution_started": True,
                }
                if deferred_progress_error is not None:
                    raise deferred_progress_error
            closures[role] = native_closure(raw)
            decomposition[role] = ledger_decomposition(
                raw, inputs.dataset, env.execution_cost
            )
            records.append(raw)
        except Exception as error:
            failure = _failure(error)
            if role not in slots:
                slots[role] = {
                    "state": "FAILED",
                    "failure": failure,
                    "receipt": None,
                    "execution_started": execution_started,
                }
                partial = (
                    raw
                    if raw is not None
                    else getattr(error, "terminal_execution_receipt", None)
                )
                if partial is not None:
                    try:
                        slots[role]["receipt"] = save_gzip(
                            root,
                            relative.replace(".json.gz", ".partial.json.gz"),
                            partial,
                        )
                    except Exception as publication_error:
                        slots[role]["partial_publication_failure"] = _failure(
                            publication_error
                        )
            if deferred_progress_error is not None:
                slots[role]["progress_publication_failure"] = _failure(
                    deferred_progress_error
                )
            elif (
                diagnostic := getattr(
                    error, "native_progress_publication_failure", None
                )
            ) is not None:
                slots[role]["progress_publication_failure"] = {
                    "message": str(diagnostic)
                }
            return outcome(None, failure)
    try:
        pair_plan = SettledNonRLComparisonPlan(
            (pair.controls,), (pair.account,), (pair.scenario,)
        )
        report = build_settled_nonrl_comparison(pair_plan, tuple(records))
        row = report.payload["groups"][0] | {
            "configuration": pair.configuration,
            "symbol": inputs.dataset.symbols[pair.symbol_index],
            "report_digest": report.digest,
            "decomposition": decomposition,
        }
        return outcome(row)
    except Exception as error:
        return outcome(None, _failure(error))


def _apply_pair_result(
    result: PairExecutionResult, state: dict[str, Any], root: Path
) -> None:
    for role in ("nonrl", "cash"):
        slot = arm_state(state, result.account, result.scenario, role)
        update = result.arm_slots.get(role)
        if update is None:
            if slot["state"] == "SUBMITTED":
                slot["state"] = (
                    "UNATTEMPTED"  # Worker explicitly stopped before this role.
                )
            continue
        slot.update(update)
    # Reconcile every known arm in memory before any fallible publication.
    for role, update in result.arm_slots.items():
        slot = arm_state(state, result.account, result.scenario, role)
        append_log(
            root,
            "arm_reused" if update.get("reused") else "arm_outcome",
            account=result.account,
            scenario=result.scenario,
            role=role,
            state=slot["state"],
        )
    write_json(root / "attempt.json", state, replace=True)


def execute_pairs(
    root: Path,
    pairs: tuple[PairDeclaration, ...],
    prepared: PreparedTables,
    tables: CostTables,
    protocol: dict[str, Any],
    state: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    workers = worker_count(protocol.get("concurrency", {}).get("max_workers", 1))
    groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    roster = tuple((p.configuration, p.account, p.scenario) for p in pairs)
    closures: dict[str, Any] = {}
    failure: dict[str, str] | None = None
    progress_reader = (
        ProgressReader(root, expected_protocol=protocol)
        if "progress_policy" in protocol
        else None
    )
    progress_publication_failed = False

    def publication_failed(error: Exception) -> None:
        nonlocal failure
        failure = failure or _failure(error)
        state.setdefault("publication_failures", []).append(_failure(error))

    def publish_progress() -> None:
        nonlocal progress_publication_failed
        if progress_reader is None or progress_publication_failed:
            return
        try:
            for event in progress_reader.poll():
                record = dict(event)
                append_log(root, record.pop("event"), **record)
        except Exception as error:
            progress_publication_failed = True
            publication_failed(error)
            raise

    def arguments(pair: PairDeclaration) -> tuple[Any, ...]:
        inputs = prepared[pair.configuration]
        horizon = next(
            c["horizon_hours"]
            for c in protocol["configurations"]
            if c["id"] == pair.configuration
        )
        estimates = symbol_estimates(
            tables, horizon, pair.scenario, inputs.dataset.symbols[pair.symbol_index]
        )
        return root, pair, inputs, estimates, protocol

    def submitted(pair: PairDeclaration) -> None:
        for role in ("nonrl", "cash"):
            slot = arm_state(state, pair.account, pair.scenario, role)
            if slot["state"] != "COMPLETED":
                slot["state"] = "SUBMITTED"
        write_json(root / "attempt.json", state, replace=True)
        append_log(root, "pair_submitted", account=pair.account, scenario=pair.scenario)

    def completed(
        pair: PairDeclaration,
        result: PairExecutionResult | None,
        error: Exception | None = None,
        cancelled: bool = False,
        not_started: bool = False,
    ) -> None:
        nonlocal failure, closures
        if result is not None and (
            type(result) is not PairExecutionResult
            or type(result.symbol_index) is not int
            or (
                result.configuration,
                result.account,
                result.scenario,
                result.symbol_index,
            )
            != (pair.configuration, pair.account, pair.scenario, pair.symbol_index)
        ):
            result, error = (
                None,
                ValueError("worker outcome identity differs from its frozen pair"),
            )
        if result is not None:
            if result.row is not None:
                groups[pair.configuration, pair.account, pair.scenario] = result.row
                closures = result.closures
            failure = failure or result.failure
            try:
                _apply_pair_result(result, state, root)
            except Exception as publication_error:
                publication_failed(publication_error)
        else:
            for role in ("nonrl", "cash"):
                slot = arm_state(state, pair.account, pair.scenario, role)
                if slot["state"] != "COMPLETED":
                    slot.update(
                        state="UNATTEMPTED"
                        if not_started
                        else "CANCELLED_BEFORE_START"
                        if cancelled
                        else "INTERRUPTED",
                        failure=None
                        if cancelled
                        else _failure(
                            error or RuntimeError("worker outcome unavailable")
                        ),
                    )
            if cancelled and failure is None:
                failure = {
                    "type": "CancelledError",
                    "message": "worker cancelled before a complete outcome",
                }
            if not cancelled:
                failure = failure or _failure(
                    error or RuntimeError("worker outcome unavailable")
                )
            try:
                append_log(
                    root,
                    "pair_cancelled"
                    if cancelled
                    else "submission_failed"
                    if not_started
                    else "worker_failed",
                    account=pair.account,
                    scenario=pair.scenario,
                )
                write_json(root / "attempt.json", state, replace=True)
            except Exception as publication_error:
                publication_failed(publication_error)
        try:
            write_json(
                root / "pairs.json",
                [groups[key] for key in roster if key in groups],
                replace=True,
            )
        except Exception as publication_error:
            publication_failed(publication_error)

    if progress_reader is not None:
        if not (root / "native-progress-roster.json").exists():
            raise ValueError("declared native progress requires its frozen roster")
        publish_progress()  # Validate bindings before any work is submitted.

    if workers == 1 or len(pairs) == 1:
        for pair in pairs:
            try:
                submitted(pair)
            except Exception as error:
                completed(pair, None, error, not_started=True)
                break
            try:
                result = (
                    _worker_execute_pair(
                        *arguments(pair),
                        progress_enabled=True,
                        on_progress=publish_progress,
                    )
                    if progress_reader is not None
                    else _worker_execute_pair(*arguments(pair))
                )
            except Exception as error:
                completed(pair, None, error)
            else:
                completed(pair, result)
            if failure is not None:
                break
    else:
        append_log(
            root, "parallel_execution_started", max_workers=workers, pairs=len(pairs)
        )
        remaining = iter(pairs)
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            active: dict[Any, PairDeclaration] = {}

            def submit_next() -> bool:
                nonlocal failure
                pair = next(remaining, None)
                if pair is None:
                    return False
                try:
                    submitted(pair)
                    future = (
                        pool.submit(
                            _worker_execute_pair,
                            *arguments(pair),
                            progress_enabled=True,
                        )
                        if progress_reader is not None
                        else pool.submit(_worker_execute_pair, *arguments(pair))
                    )
                    active[future] = pair
                except Exception as error:
                    completed(pair, None, error, not_started=True)
                    return False
                return True

            for _ in range(workers):
                if not submit_next():
                    break
            while active:
                if progress_reader is None:
                    ready = {next(concurrent.futures.as_completed(tuple(active)))}
                else:
                    ready, _ = concurrent.futures.wait(
                        tuple(active),
                        timeout=protocol["progress_policy"]["sample_seconds"],
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    try:
                        publish_progress()
                    except Exception:
                        pass  # Recorded once; drain already-started worker outcomes.
                if failure is not None:
                    for pending, declaration in tuple(active.items()):
                        if pending.cancel():
                            active.pop(pending)
                            completed(declaration, None, cancelled=True)
                if not ready:
                    continue
                # Process one known outcome before considering another submission.
                future = next(iter(ready))
                if future not in active:
                    continue
                pair = active.pop(future)
                try:
                    result = future.result()
                except concurrent.futures.CancelledError:
                    completed(pair, None, cancelled=True)
                except Exception as error:
                    completed(pair, None, error)
                else:
                    completed(pair, result)
                if failure is not None:
                    for pending, declaration in tuple(active.items()):
                        if pending.cancel():
                            active.pop(pending)
                            completed(declaration, None, cancelled=True)
                else:
                    submit_next()
    try:
        publish_progress()
    except Exception:
        pass  # Retain every reconciled result when final progress publication fails.
    if failure is not None:
        raise RuntimeError(f"worker failure: {failure['type']}: {failure['message']}")
    return [groups[key] for key in roster if key in groups], closures


def execute(
    root: Path,
    dataset_root: Path,
    dataset: MarketDataset,
    protocol: dict[str, Any],
    review: dict[str, Any] | None = None,
    *,
    resume: bool = False,
    resume_from: Path | None = None,
) -> dict[str, Any]:
    if resume or root.exists() or root.is_symlink():
        raise FileExistsError(
            "retain prior attempt; resume_from requires a new destination"
        )
    progress_enabled = "progress_policy" in protocol
    if progress_enabled or protocol["scope"] in {
        "CONSUMED_2025_DEVELOPMENT_NONRL48_ONLY",
        "ARTIFICIAL_SOFTWARE_INTEGRATION_ONLY",
    }:
        validate_progress_policy(protocol.get("progress_policy"))
    provenance = build_candidate_run_provenance()
    envelope = {
        "source_head": protocol["source_head"],
        "driver_sha256": source_pins(),
        "protocol_digest": json_sha256(protocol),
        "review": review,
        "python": sys.version,
        "numpy": np.__version__,
        "native": provenance,
    }
    reused = (
        preflight_resume(resume_from, root, protocol, envelope)
        if resume_from is not None
        else None
    )
    state = initialize_attempt(root, protocol)
    groups: list[dict[str, Any]] = []
    closure: dict[str, Any] = {}
    try:
        write_json(root / "provenance.json", envelope)
        if reused is None:
            prepared, tables = prepare_inputs(root, dataset_root, dataset, protocol)
            pairs = declare_matrix(prepared, tables, protocol, provenance)
            matrix = SettledNonRLComparisonPlan(
                tuple(pair.controls for pair in pairs),
                tuple(protocol["accounts"]),
                tuple(protocol["scenarios"]),
            )
            frozen = freeze_inputs(
                root, pairs, prepared, tables, matrix, protocol, envelope
            )
        else:
            reused.materialize(root)
            tables, matrix, frozen = reused.tables, reused.matrix, reused.frozen
            prepared = {}
            for configuration in protocol["configurations"]:
                identity = configuration["id"]
                metadata = frozen["forecasts"][identity]
                prepared[identity] = PreparedAllocationMarketInputs(
                    dataset,
                    reused.streams[identity],
                    PublishedSimpleReturnStreamArtifact(
                        root / metadata["path"],
                        metadata["sha256"],
                        metadata["dataset_id"],
                    ),
                    tuple(
                        tuple(
                            row
                            for row in tables[configuration["horizon_hours"], "BASEL0"]
                            if row.symbol == symbol
                        )
                        for symbol in dataset.symbols
                    ),
                )
            controls = {group.key: group for group in matrix.controls}
            pairs = tuple(
                PairDeclaration(
                    p["account"],
                    p["scenario"],
                    p["configuration"],
                    p["symbol_index"],
                    controls[p["account"], p["scenario"]],
                )
                for p in frozen["pairs"]
            )
        state.update(
            status="EXECUTING",
            matrix_digest=matrix.digest,
            freeze_digest=content_digest(frozen),
        )
        write_json(root / "attempt.json", state, replace=True)
        append_log(root, "complete_matrix_frozen", arms=len(pairs) * 2)
        if progress_enabled:
            publish_roster(
                root,
                protocol,
                tuple(
                    ArmProgressSpec(
                        pair.configuration,
                        pair.account,
                        pair.scenario,
                        role,
                        plan.digest,
                        protocol["actor"]["start"],
                        protocol["actor"]["stop"],
                        protocol["terminal"]["stop_inclusive"],
                    )
                    for pair in pairs
                    for role, plan in zip(
                        ("nonrl", "cash"), pair.controls.plans, strict=True
                    )
                ),
            )

        groups, closure = execute_pairs(root, pairs, prepared, tables, protocol, state)
        state["status"] = "COMPLETED"
        write_json(root / "attempt.json", state, replace=True)
        result = {
            "scope": protocol["scope"],
            "status": state["status"],
            "completed_arms": sum(row["state"] == "COMPLETED" for row in state["arms"]),
            "attempted_arms": sum(
                row.get("execution_started") is True for row in state["arms"]
            ),
            "reused_arms": sum(
                row["state"] == "COMPLETED" and row.get("reused") is True
                for row in state["arms"]
            ),
            "unknown_worker_outcome_arms": sum(
                row["state"] == "INTERRUPTED" for row in state["arms"]
            ),
            "declared_budget": protocol["budget"],
            "reused_vintages": sum(
                len(inputs.stream.vintages) for inputs in prepared.values()
            )
            if reused is not None
            else 0,
            "dataset_dimensions": [
                dataset.n_bars,
                dataset.n_symbols,
                dataset.n_features,
            ],
            "public_calls": {
                "dataset_loads": 1
                + (len(protocol["configurations"]) if reused is None else 0),
                "cost_producers": len(tables) if reused is None else 0,
                "prepare_calls": len(prepared) if reused is None else 0,
                "ridge_vintages": sum(
                    len(inputs.stream.vintages) for inputs in prepared.values()
                )
                if reused is None
                else 0,
            },
            "pairs": groups,
            "combined_ranking": combined_rows(groups, protocol),
            "native_closure": closure
            if len(groups) == 1
            else "full raw closure in each gzip receipt",
            "decision": protocol["decision"],
            "matrix_digest": matrix.digest,
            "comparison_complete": all(
                group["comparison_complete"] for group in groups
            ),
            "qualification": "DEVELOPMENT_ONLY_NO_ADOPTION_OR_WINNER",
        }
        write_json(root / "result.json", result)
        append_log(
            root,
            "attempt_completed",
            arms=result["completed_arms"],
            scope=protocol["scope"],
        )
        write_json(root / "output-manifest.json", output_manifest(root), replace=True)
        return result
    except Exception as error:
        state.update(
            status="FAILED",
            failure={"type": type(error).__name__, "message": str(error)},
        )
        write_json(root / "attempt.json", state, replace=True)
        append_log(root, "attempt_failed", failure=state["failure"])
        write_json(root / "output-manifest.json", output_manifest(root), replace=True)
        raise


def run_synthetic(
    root: Path, *, resume_from: Path | None = None, max_workers: int = 1
) -> dict[str, Any]:
    if root.exists() or root.is_symlink():
        raise FileExistsError("synthetic attempt destination must be new")
    if resume_from is not None:
        protocol = read_json(resume_from / "protocol.json")
        if protocol["scope"] != "ARTIFICIAL_SOFTWARE_INTEGRATION_ONLY" or protocol[
            "concurrency"
        ]["max_workers"] != worker_count(max_workers):
            raise ValueError(
                "synthetic reuse requires its original artificial protocol"
            )
        dataset_root = Path(protocol["dataset"]["root"])
        for name, digest in protocol["dataset"]["input_sha256"].items():
            if file_sha256(dataset_root / name) != digest:
                raise ValueError("artificial source input bytes changed")
        return execute(
            root,
            dataset_root,
            load_market_dataset_artifact(dataset_root),
            protocol,
            resume_from=resume_from,
        )
    dataset_root = root.parent / f"{root.name}-artificial-dataset"
    if dataset_root.exists():
        raise FileExistsError("retain previous artificial input fixture")
    dataset = write_artificial_dataset(dataset_root)
    protocol = synthetic_protocol(market_protocol(max_workers=max_workers), dataset)
    protocol["dataset"].update(
        root=str(dataset_root),
        input_sha256={name: file_sha256(dataset_root / name) for name in INPUT_PINS},
    )
    protocol["economics"].update(
        borrow_rate=0.0876,
        declaration="Artificial manufactured carry for COST2 invariance oracle",
    )
    protocol["funding"].update(
        source="Artificial scheduled 8h events at rate .0001",
        marks="Manufactured rate times artificial close; no venue data",
    )
    # Use the public loader here as well as inside the one public preparation.
    return execute(
        root,
        dataset_root,
        load_market_dataset_artifact(dataset_root),
        protocol,
    )


def run_market(
    root: Path,
    *,
    review_path: Path | None,
    protocol_path: Path | None = None,
    resume_from: Path | None = None,
    max_workers: int | None = None,
) -> dict[str, Any]:
    if review_path is None:
        validate_external_review(None, {})
    protocol = load_market_protocol(protocol_path, max_workers=max_workers)
    review = validate_external_review(review_path, protocol)
    return execute(
        root,
        DATASET_ROOT,
        validate_market_input(protocol),
        protocol,
        review,
        resume_from=resume_from,
    )


def load_market_protocol(
    path: Path | None, *, max_workers: int | None = None
) -> dict[str, Any]:
    if path is None:
        raise ValueError("market execution requires an explicit frozen protocol file")
    protocol = read_json(path)
    count = worker_count(protocol["concurrency"]["max_workers"])
    if max_workers is not None and worker_count(max_workers) != count:
        raise ValueError("CLI concurrency differs from the frozen protocol")
    if json_sha256(protocol) != json_sha256(market_protocol(max_workers=count)):
        raise ValueError(
            "frozen market protocol differs from the exact current source declaration"
        )
    return protocol


if __name__ == "__main__":
    from .tier_d_supervisor import main

    main()
