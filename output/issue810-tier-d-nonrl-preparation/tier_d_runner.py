"""Bounded ignored prototype: public NONRL/CASH comparison, no implicit market run."""

from __future__ import annotations

import concurrent.futures
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tier_d_decomposition import ledger_decomposition
from tier_d_fixture import write_artificial_dataset
from tier_d_native import (
    PairDeclaration,
    carrier,
    cost_rows,
    declare_pair,
    execution_folds,
    forecast_blocks,
    synthetic_protocol,
)
from tier_d_protocol import (
    DATASET_ID,
    DATASET_ROOT,
    IMPLEMENTATION_MODEL,
    INPUT_PINS,
    REPO,
    SOURCE_HEAD,
    file_sha256,
    json_sha256,
    market_protocol,
    source_pins,
)
from tier_d_records import (
    append_log,
    arm_state,
    combined_rows,
    initialize_attempt,
    load_gzip,
    native_closure,
    output_manifest,
    save_gzip,
    write_json,
)

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

CostTables = dict[tuple[int, str], tuple[HorizonCostEstimates, ...]]
PreparedTables = dict[str, PreparedAllocationMarketInputs]


def validate_external_review(
    path: Path | None, protocol: dict[str, Any]
) -> dict[str, Any]:
    if path is None:
        raise ValueError(
            "ROOT different-model review is required before actual costs/fits/PnL"
        )
    review = json.loads(path.read_text(encoding="utf-8"))
    if (
        review.get("scope") != protocol["scope"]
        or review.get("status") != "APPROVED_FOR_TIER_D_DEVELOPMENT_EXECUTION"
        or review.get("reviewer_model") in (None, "", IMPLEMENTATION_MODEL)
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
    current_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    if current_head != SOURCE_HEAD:
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
    *,
    resume: bool = False,
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
        if resume and stream_path.exists():
            stream_path.unlink()
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
    row: dict[str, Any]
    closures: dict[str, Any]
    arm_slots: dict[str, dict[str, Any]]


class ArmExecutionError(RuntimeError):
    def __init__(
        self,
        pair: PairDeclaration,
        role: str,
        update: dict[str, Any],
        cause: Exception,
    ) -> None:
        super().__init__(str(cause))
        self.pair = pair
        self.role = role
        self.update = update
        self.cause = cause


def _worker_execute_pair(
    root: Path,
    pair: PairDeclaration,
    inputs: PreparedAllocationMarketInputs,
    estimates: tuple[HorizonCostEstimates, ...],
    protocol: dict[str, Any],
) -> PairExecutionResult:
    records, closures = [], {}
    decomposition = {}
    arm_slots: dict[str, dict[str, Any]] = {}
    folds = execution_folds(
        tuple(protocol["actor"]["vintage_starts"]), protocol["actor"]["stop"]
    )
    horizon = next(
        c["horizon_hours"]
        for c in protocol["configurations"]
        if c["id"] == pair.configuration
    )
    for role, plan in zip(("nonrl", "cash"), pair.controls.plans, strict=True):
        receipt_path = root / f"receipts/{pair.account}_{pair.scenario}_{role}.json.gz"
        raw = None
        if receipt_path.exists():
            try:
                candidate = load_gzip(receipt_path)
                receipt = TerminalAllocationExecutionReceipt.from_payload(
                    candidate, expected_plan=plan
                )
                raw = candidate
                arm_slots[role] = {
                    "state": "COMPLETED",
                    "receipt_digest": receipt.digest,
                    "plan_digest": plan.digest,
                    "receipt": {
                        "path": f"receipts/{pair.account}_{pair.scenario}_{role}.json.gz",
                        "sha256": file_sha256(receipt_path),
                        "bytes": receipt_path.stat().st_size,
                    },
                    "reused": True,
                }
            except Exception:
                raw = None
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
            try:
                observed = run_terminal_allocation_execution(folds, env, plan)
                raw = observed.payload
                receipt_meta = save_gzip(
                    root, f"receipts/{pair.account}_{pair.scenario}_{role}.json.gz", raw
                )
                receipt = TerminalAllocationExecutionReceipt.from_payload(
                    raw, expected_plan=plan
                )
                arm_slots[role] = {
                    "state": "COMPLETED",
                    "receipt_digest": receipt.digest,
                    "plan_digest": plan.digest,
                    "receipt": receipt_meta,
                    "reused": False,
                }
            except Exception as error:
                failed_update: dict[str, Any] = {
                    "state": "FAILED",
                    "failure": {"type": type(error).__name__, "message": str(error)},
                }
                partial = getattr(error, "terminal_execution_receipt", None)
                if partial is not None:
                    failed_update["receipt"] = save_gzip(
                        root,
                        f"receipts/{pair.account}_{pair.scenario}_{role}.partial.json.gz",
                        partial,
                    )
                raise ArmExecutionError(pair, role, failed_update, error) from error
        closures[role] = native_closure(raw)
        decomposition[role] = ledger_decomposition(
            raw, inputs.dataset, env.execution_cost
        )
        records.append(raw)
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
    return PairExecutionResult(
        configuration=pair.configuration,
        account=pair.account,
        scenario=pair.scenario,
        symbol_index=pair.symbol_index,
        row=row,
        closures=closures,
        arm_slots=arm_slots,
    )


def _handle_arm_failure(
    error: ArmExecutionError, state: dict[str, Any], root: Path
) -> None:
    slot = arm_state(state, error.pair.account, error.pair.scenario, error.role)
    slot.update(
        state="FAILED",
        failure=error.update["failure"],
        receipt=error.update.get("receipt"),
    )
    write_json(root / "attempt.json", state)
    append_log(
        root,
        "arm_failed",
        account=error.pair.account,
        scenario=error.pair.scenario,
        role=error.role,
        failure=slot["failure"],
    )


def _apply_pair_result(
    result: PairExecutionResult,
    state: dict[str, Any],
    root: Path,
) -> None:
    for role, slot_update in result.arm_slots.items():
        slot = arm_state(state, result.account, result.scenario, role)
        slot.update(
            state=slot_update["state"],
            receipt=slot_update["receipt"],
            receipt_digest=slot_update["receipt_digest"],
            plan_digest=slot_update["plan_digest"],
        )
        event = "arm_reused" if slot_update.get("reused") else "arm_completed"
        append_log(
            root,
            event,
            account=result.account,
            scenario=result.scenario,
            role=role,
            receipt_bytes=slot["receipt"]["bytes"],
        )
    write_json(root / "attempt.json", state)


def execute_pair(
    root: Path,
    pair: PairDeclaration,
    inputs: PreparedAllocationMarketInputs,
    estimates: tuple[HorizonCostEstimates, ...],
    protocol: dict[str, Any],
    state: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        result = _worker_execute_pair(root, pair, inputs, estimates, protocol)
    except ArmExecutionError as err:
        _handle_arm_failure(err, state, root)
        raise err.cause from err.cause
    _apply_pair_result(result, state, root)
    return result.row, result.closures


def execute(
    root: Path,
    dataset_root: Path,
    dataset: MarketDataset,
    protocol: dict[str, Any],
    review: dict[str, Any] | None = None,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    state = initialize_attempt(root, protocol, resume=resume)
    provenance = build_candidate_run_provenance()
    write_json(
        root / "provenance.json",
        {
            "source_head": SOURCE_HEAD,
            "driver_sha256": source_pins(),
            "protocol_digest": json_sha256(protocol),
            "review": review,
            "python": sys.version,
            "numpy": np.__version__,
            "native": provenance,
        },
    )
    groups, closure = [], {}
    try:
        prepared, tables = prepare_inputs(
            root, dataset_root, dataset, protocol, resume=resume
        )
        pairs = declare_matrix(prepared, tables, protocol, provenance)
        matrix = SettledNonRLComparisonPlan(
            tuple(pair.controls for pair in pairs),
            tuple(protocol["accounts"]),
            tuple(protocol["scenarios"]),
        )
        if resume and state.get("matrix_digest"):
            if state["matrix_digest"] != matrix.digest:
                raise ValueError(
                    "cannot resume: declared plan digest differs from prior attempt"
                )
        save_gzip(root, "native-plan.json.gz", matrix.payload)
        state.update(status="EXECUTING", matrix_digest=matrix.digest)
        write_json(root / "attempt.json", state)
        append_log(root, "complete_matrix_frozen", arms=len(pairs) * 2)

        max_workers = protocol.get("concurrency", {}).get("max_workers", 1)
        if max_workers <= 1 or len(pairs) <= 1:
            for pair in pairs:
                inputs = prepared[pair.configuration]
                horizon = next(
                    c["horizon_hours"]
                    for c in protocol["configurations"]
                    if c["id"] == pair.configuration
                )
                ests = symbol_estimates(
                    tables,
                    horizon,
                    pair.scenario,
                    inputs.dataset.symbols[pair.symbol_index],
                )
                try:
                    res = _worker_execute_pair(root, pair, inputs, ests, protocol)
                except ArmExecutionError as err:
                    _handle_arm_failure(err, state, root)
                    raise err.cause from err.cause
                _apply_pair_result(res, state, root)
                groups.append(res.row)
                closure = res.closures
                write_json(root / "pairs.json", groups)
        else:
            append_log(
                root,
                "parallel_execution_started",
                max_workers=max_workers,
                pairs=len(pairs),
            )
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=max_workers
            ) as pool:
                futures = {}
                for pair in pairs:
                    inputs = prepared[pair.configuration]
                    horizon = next(
                        c["horizon_hours"]
                        for c in protocol["configurations"]
                        if c["id"] == pair.configuration
                    )
                    ests = symbol_estimates(
                        tables,
                        horizon,
                        pair.scenario,
                        inputs.dataset.symbols[pair.symbol_index],
                    )
                    fut = pool.submit(
                        _worker_execute_pair, root, pair, inputs, ests, protocol
                    )
                    futures[fut] = pair
                for fut in concurrent.futures.as_completed(futures):
                    try:
                        res = fut.result()
                    except ArmExecutionError as err:
                        _handle_arm_failure(err, state, root)
                        raise err.cause from err.cause
                    _apply_pair_result(res, state, root)
                    groups.append(res.row)
                    closure = res.closures
                    write_json(root / "pairs.json", groups)
        state["status"] = "COMPLETED"
    except Exception as error:
        state.update(
            status="FAILED",
            failure={"type": type(error).__name__, "message": str(error)},
        )
        write_json(root / "attempt.json", state)
        append_log(root, "attempt_failed", failure=state["failure"])
        write_json(root / "output-manifest.json", output_manifest(root))
        raise
    write_json(root / "attempt.json", state)
    result = {
        "scope": protocol["scope"],
        "status": state["status"],
        "completed_arms": sum(row["state"] == "COMPLETED" for row in state["arms"]),
        "attempted_arms": sum(row["state"] != "UNATTEMPTED" for row in state["arms"]),
        "dataset_dimensions": [dataset.n_bars, dataset.n_symbols, dataset.n_features],
        "public_calls": {
            "dataset_loads": 1 + len(protocol["configurations"]),
            "cost_producers": len(tables),
            "prepare_calls": len(prepared),
            "ridge_vintages": sum(
                len(inputs.stream.vintages) for inputs in prepared.values()
            ),
        },
        "pairs": groups,
        "combined_ranking": combined_rows(groups, protocol),
        "native_closure": closure
        if len(groups) == 1
        else "full raw closure in each gzip receipt",
        "decision": protocol["decision"],
        "matrix_digest": matrix.digest,
        "comparison_complete": all(group["comparison_complete"] for group in groups),
        "qualification": "DEVELOPMENT_ONLY_NO_ADOPTION_OR_WINNER",
    }
    write_json(root / "result.json", result)
    append_log(
        root,
        "attempt_completed",
        arms=result["completed_arms"],
        scope=protocol["scope"],
    )
    write_json(root / "output-manifest.json", output_manifest(root))
    return result


def run_synthetic(root: Path, *, resume: bool = False) -> dict[str, Any]:
    dataset_root = root / "artificial-dataset"
    dataset = write_artificial_dataset(dataset_root)
    protocol = synthetic_protocol(market_protocol(), dataset)
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
        resume=resume,
    )


def run_market(
    root: Path, *, review_path: Path | None, resume: bool = False
) -> dict[str, Any]:
    protocol = market_protocol()
    review = validate_external_review(review_path, protocol)
    return execute(
        root,
        DATASET_ROOT,
        validate_market_input(protocol),
        protocol,
        review,
        resume=resume,
    )


if __name__ == "__main__":
    from tier_d_supervisor import main

    main()
