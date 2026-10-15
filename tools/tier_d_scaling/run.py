"""Measure native artificial prefixes, preserving unqualified partial evidence."""

from __future__ import annotations

import hashlib
import pstats
from pathlib import Path
from time import perf_counter
from typing import Any

from tools.tier_d_nonrl.progress import ArmProgressSpec
from tools.tier_d_nonrl.progress_io import ProgressJournal, publish_roster
from tools.tier_d_nonrl.tier_d_native import carrier, execution_folds
from tools.tier_d_nonrl.tier_d_records import (
    initialize_attempt,
    output_manifest,
    save_gzip,
    write_json,
)
from trade_rl.artifacts import canonical_json_bytes
from trade_rl.evaluation.allocation_terminal_evidence import (
    TerminalAllocationExecutionReceipt,
)
from trade_rl.evaluation.allocation_terminal_execution import (
    run_terminal_allocation_execution,
)

from .context import load_prepared
from .probe import observe_prefix
from .spec import case_by_name, validate_profile_source


def _error(error: Exception) -> dict[str, str]:
    return {"type": type(error).__name__, "message": str(error)}


def measure(root: Path, prepared_root: Path, name: str) -> dict[str, Any]:
    if root.resolve().is_relative_to(
        prepared_root.resolve()
    ) or prepared_root.resolve().is_relative_to(root.resolve()):
        raise ValueError("measurement and prepared inputs must be disjoint")
    started = perf_counter()
    context = load_prepared(prepared_root)
    protocol, pair, inputs = context.protocol, context.pair, context.inputs
    case = case_by_name(protocol, name)
    state = initialize_attempt(root, protocol)
    report: dict[str, Any] = {
        "status": "FAILED",
        "financial_authority": False,
        "case": case,
        "protocol": protocol,
        "provenance": context.envelope,
        "prepared_root": str(prepared_root.resolve()),
        "execution_started": False,
    }
    try:
        specs = tuple(
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
            for role, plan in (
                ("nonrl", pair.controls.nonrl_plan),
                ("cash", pair.controls.cash_plan),
            )
        )
        publish_roster(root, protocol, specs)
        plan, spec = pair.controls.nonrl_plan, specs[0]
        report["plan_digest"] = plan.digest
        env = carrier(
            inputs,
            inputs.costs_by_symbol[0],
            protocol,
            horizon=72,
            symbol_index=0,
            account=pair.account,
            scenario=pair.scenario,
            role="nonrl",
        )
        folds = execution_folds(
            tuple(protocol["actor"]["vintage_starts"]), protocol["actor"]["stop"]
        )
        journal = ProgressJournal(root, spec)
        native_returned: object | None = None
        state["arms"][0].update(state="RUNNING", plan_digest=plan.digest)
        write_json(root / "attempt.json", state, replace=True)

        def invoke(callback: Any) -> object:
            nonlocal native_returned
            with journal:
                report["execution_started"] = True
                native_returned = run_terminal_allocation_execution(
                    folds, env, plan, on_advance=callback
                )
                return native_returned

        observation = observe_prefix(
            invoke,
            spec,
            target=case["target"],
            on_advance=journal.advance,
            profile_path=root / "native.prof" if case["profile"] else None,
        )
        retained = perf_counter()
        report.update(
            expected_stop=observation.expected_stop,
            count=observation.count,
            actor_advances=observation.actor_advances,
            reserve_advances=observation.reserve_advances,
            processing_index=observation.processing_index,
            elapsed_seconds=observation.elapsed_seconds,
            native_error=_error(observation.error) if observation.error else None,
            profile_publication_failure=observation.profile_publication_failure,
        )
        raw = getattr(native_returned, "payload", None)
        if raw is None:
            raw = getattr(observation.error, "terminal_execution_receipt", None)
        report["partial_receipt_recording_failure"] = getattr(
            observation.error, "terminal_execution_receipt_error", None
        )
        report["progress_publication_failure"] = getattr(
            observation.error, "native_progress_publication_failure", None
        )
        if native_returned is not None and observation.error is not None:
            report["progress_publication_failure"] = _error(observation.error)
        if raw is not None:
            # Preserve caller-returned facts before asking the closed reader.
            try:
                report["receipt"] = save_gzip(root, "receipt.partial.json.gz", raw)
                report["receipt_canonical_sha256"] = hashlib.sha256(
                    canonical_json_bytes(raw)
                ).hexdigest()
            except Exception as error:
                report["receipt_publication_failure"] = _error(error)
            try:
                receipt = TerminalAllocationExecutionReceipt.from_payload(
                    raw, expected_plan=plan
                )
                report["receipt_digest"] = receipt.digest
                report["readback_verified"] = True
            except Exception as error:
                report["readback_failure"] = _error(error)
        else:
            report["readback_failure"] = {
                "type": "MissingPartialReceipt",
                "message": "native call retained no partial receipt",
            }
        try:
            validate_profile_source(protocol)
        except Exception as error:
            report["source_postflight_failure"] = _error(error)
        if (
            observation.expected_stop
            and observation.count == case["target"]
            and report.get("readback_verified") is True
            and raw is not None
            and raw["status"] == "integrity_failure"
            and report.get("receipt")
            and not report.get("progress_publication_failure")
            and not observation.profile_publication_failure
            and not report.get("source_postflight_failure")
        ):
            report["status"] = "PREFIX_OBSERVED"
        if case["profile"] and not observation.profile_publication_failure:
            try:
                with (root / "native-profile.txt").open(
                    "x", encoding="utf-8"
                ) as stream:
                    pstats.Stats(
                        str(root / "native.prof"), stream=stream
                    ).strip_dirs().sort_stats("cumulative").print_stats(60)
            except Exception as error:
                report["profile_text_failure"] = _error(error)
                report["status"] = "FAILED"
        report["retention_readback_seconds"] = perf_counter() - retained
    except Exception as error:
        report["failure"] = _error(error)
    finally:
        # The actual native execution is deliberately failed, even if profiling succeeds.
        state.update(status="FAILED", measurement_status=report["status"])
        state["arms"][0].update(
            state="FAILED" if report["execution_started"] else "UNATTEMPTED",
            receipt=report.get("receipt"),
        )
        write_json(root / "attempt.json", state, replace=True)
        report["total_seconds"] = perf_counter() - started
        write_json(root / "measurement.json", report)
        write_json(root / "output-manifest.json", output_manifest(root))
    return report
