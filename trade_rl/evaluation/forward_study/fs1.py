"""Sealed FS1 evaluator: refuse unless sealed, then replay once and label (Issue #810).

:func:`run_fs1` is the only entrypoint. It

1. refuses (before touching any dataset) unless :func:`~.seal.authorize_scoring`
   returns a permit and the write-once output path is free;
2. loads the dataset through the caller-supplied loader and refuses unless it matches
   the sealed manifest;
3. replays CASH, FS1-A, FS1-B and FS1-C in the base, cost x2 and +1 bar scenarios on
   ONE continuous 10,000 USDT account over the frozen scoring window, with the Phase 3
   policy builders and the development runner unchanged;
4. runs the future-perturbation causality matrix on the loaded data, computes the paired
   block-bootstrap intervals, applies the frozen label rule and writes the result once.

No economic number exists before step 3 and step 3 cannot run before the scoring end
date, an explicit authorization reference and exact parameter / code / dataset matches.
The scoring path is exercised in tests only on synthetic data.
"""

from __future__ import annotations

import json
import math
import traceback
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np

from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.dev_realdata_study import runner
from trade_rl.evaluation.dev_realdata_study.metrics import (
    bootstrap_excess,
    daily_equity,
    daily_returns,
    moving_block_indices,
)
from trade_rl.evaluation.dev_realdata_study.phase3_sources import (
    Phase3Context,
    build_phase3_arm,
    make_phase3_context,
)
from trade_rl.evaluation.dev_realdata_study.protocol import (
    DevStudyProtocol,
    protocol_digest,
)
from trade_rl.evaluation.forward_study.labels import (
    CandidateEvidence,
    classify_candidate,
)
from trade_rl.evaluation.forward_study.registry import (
    FROZEN_PARAMS_DIGEST,
    FS1_REGISTRY,
    RESULT_SCHEMA,
    ROLE_CASH,
    Fs1Registry,
)
from trade_rl.evaluation.forward_study.seal import (
    CodeIdentity,
    Fs1Refusal,
    ScoringPermit,
    _authorize,
    current_code_identity,
    require_permit,
    verify_dataset,
)

Progress = Callable[[str], None]
_NS_PER_DAY: Final = 24 * 3_600 * 1_000_000_000
CASH_PNL_TOLERANCE: Final = 1e-6


def build_fs1_arm(
    arm: str,
    dataset: MarketDataset,
    protocol: DevStudyProtocol,
    context: Phase3Context,
    *,
    registry: Fs1Registry = FS1_REGISTRY,
) -> runner.BuiltArm:
    """Fresh policy for one replay of an FS1 arm (policies hold replay state)."""

    spec = registry.arm(arm)
    if spec.role == ROLE_CASH:
        return runner.BuiltArm(runner.ZeroWeightPolicy(dataset.symbols), None)
    return build_phase3_arm(spec.phase3_arm, dataset, protocol, context)


@dataclass
class ArmRun:
    arm: str
    outcomes: dict[str, runner.ReplayOutcome] = field(default_factory=dict)
    failures: dict[str, dict[str, str]] = field(default_factory=dict)
    probe_matrix: dict[str, dict[str, object]] | None = None


def _failure(exc: BaseException) -> dict[str, str]:
    return {
        "status": "FAILED_TO_RUN",
        "error_type": type(exc).__name__,
        "error": str(exc)[:600],
        "traceback_tail": traceback.format_exc()[-1500:],
    }


def _expected_daily_points(registry: Fs1Registry) -> int:
    start = int(np.datetime64(registry.scoring_start, "ns").astype(np.int64))
    stop = int(np.datetime64(registry.scoring_end, "ns").astype(np.int64))
    if start % _NS_PER_DAY or stop % _NS_PER_DAY or stop <= start:
        raise ValueError("the scoring window must span whole UTC days")
    return (stop - start) // _NS_PER_DAY + 1


def _daily_series(
    outcome: runner.ReplayOutcome, registry: Fs1Registry
) -> np.ndarray | None:
    """Daily returns when the path covers the whole window with finite positive equity."""

    try:
        stamps, values = daily_equity(outcome.timestamps_ns, outcome.equity)
    except ValueError:
        return None
    if values.size != _expected_daily_points(registry):
        return None
    start = int(np.datetime64(registry.scoring_start, "ns").astype(np.int64))
    stop = int(np.datetime64(registry.scoring_end, "ns").astype(np.int64))
    if int(stamps[0]) != start or int(stamps[-1]) != stop:
        return None
    returns = daily_returns(values)
    return returns if bool(np.all(np.isfinite(returns))) else None


def arm_invalid_reasons(
    run: ArmRun,
    registry: Fs1Registry,
    *,
    initial_capital: float,
    needs_probe: bool,
) -> tuple[str, ...]:
    """Measurement-validity failures of one arm (empty tuple = a valid measurement)."""

    reasons: list[str] = []
    scenarios = registry.reported_scenarios
    for scenario in scenarios:
        outcome = run.outcomes.get(scenario)
        if outcome is None:
            reasons.append(f"replay_failed_{scenario}")
            continue
        summary = outcome.summary
        if not bool(summary["reconciliation"]["passed"]):  # type: ignore[index]
            reasons.append(f"ledger_not_reconciled_{scenario}")
        if summary["terminated"]:
            reasons.append(f"account_terminated_{scenario}")
        if _daily_series(outcome, registry) is None:
            reasons.append(f"incomplete_daily_equity_{scenario}")
        if needs_probe and int(summary["n_trades"]) < registry.min_trades:  # type: ignore[call-overload]
            reasons.append(f"too_few_trades_{scenario}")
        decisions = summary.get("allocator_decisions")
        if needs_probe and not isinstance(decisions, Mapping):
            # A candidate or passive control must expose its decision records; without
            # them the fail-closed guard could not run, so the arm is not a valid
            # measurement (never a silent pass).
            reasons.append(f"allocator_decisions_missing_{scenario}")
        elif isinstance(decisions, Mapping) and int(decisions["n_decisions"]) > 0:
            fraction = int(decisions["fail_closed_hold_count"]) / int(
                decisions["n_decisions"]
            )
            if fraction > registry.max_fail_closed_fraction:
                reasons.append(f"fail_closed_fraction_high_{scenario}")
    if needs_probe:
        matrix = run.probe_matrix
        if matrix is None or set(matrix) != set(scenarios):
            reasons.append("causality_matrix_missing")
        else:
            for scenario in scenarios:
                if not bool(matrix[scenario]["identical_pre_cut_trace"]):
                    reasons.append(f"causality_probe_not_identical_{scenario}")
    else:
        for scenario in scenarios:
            outcome = run.outcomes.get(scenario)
            if outcome is not None and (
                abs(float(outcome.summary["net_pnl_usdt"]))  # type: ignore[arg-type]
                > CASH_PNL_TOLERANCE
                or abs(float(outcome.equity[0]) - initial_capital) > CASH_PNL_TOLERANCE
            ):
                reasons.append(f"cash_not_flat_{scenario}")
    return tuple(reasons)


def _score_arm(
    dataset: MarketDataset,
    protocol: DevStudyProtocol,
    registry: Fs1Registry,
    context: Phase3Context,
    arm: str,
    *,
    start_index: int,
    stop_index: int,
    say: Progress,
) -> ArmRun:
    run = ArmRun(arm)

    def builder(
        name: str, data: MarketDataset, proto: DevStudyProtocol, ctx: Any
    ) -> runner.BuiltArm:
        return build_fs1_arm(name, data, proto, ctx, registry=registry)

    for scenario in registry.reported_scenarios:
        say(f"replay {arm} / {scenario}")
        try:
            run.outcomes[scenario] = runner.run_replay(
                dataset,
                protocol,
                arm,
                scenario,
                context,
                start_index=start_index,
                stop_index=stop_index,
                arm_builder=builder,
            )
        except Exception as exc:  # recorded as INVALID, never retried or altered
            run.failures[scenario] = _failure(exc)
            say(f"FAILED {arm} / {scenario}: {type(exc).__name__}: {exc}")
    if registry.arm(arm).role != ROLE_CASH:
        say(f"causality matrix {arm}")
        try:
            run.probe_matrix = runner.causality_probe_matrix(
                dataset,
                protocol,
                context,
                start_index=start_index,
                cut=registry.causality_cut,
                arms=(arm,),
                expect_trades=False,
                arm_builder=builder,
                sources_builder=make_phase3_context,
            )[arm]
        except Exception as exc:
            run.failures["causality_matrix"] = _failure(exc)
            say(f"FAILED {arm} causality matrix: {type(exc).__name__}: {exc}")
    return run


def _json_safe(value: Any) -> Any:
    """Make a result JSON-serialisable without ever losing a one-shot result."""

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return (
            "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
        )
    return value


def _interval(
    excess: np.ndarray, indices: np.ndarray, protocol: DevStudyProtocol
) -> dict[str, float]:
    return bootstrap_excess(excess, indices, protocol).to_mapping()


def assemble_result(
    runs: Mapping[str, ArmRun],
    protocol: DevStudyProtocol,
    registry: Fs1Registry,
) -> dict[str, Any]:
    """Validity, bootstrap and labels from finished arm runs (pure of any I/O)."""

    capital = protocol.initial_capital
    invalid: dict[str, tuple[str, ...]] = {}
    for arm in registry.arms:
        run = runs[arm.fs1_id]
        invalid[arm.fs1_id] = arm_invalid_reasons(
            run,
            registry,
            initial_capital=capital,
            needs_probe=arm.role != ROLE_CASH,
        )
    controls = (registry.cash_arm, registry.passive_arm)
    controls_valid = all(not invalid[arm] for arm in controls)

    bootstrap: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for scenario in registry.reported_scenarios:
        daily: dict[str, np.ndarray] = {}
        for arm in registry.arms:
            outcome = runs[arm.fs1_id].outcomes.get(scenario)
            series = None if outcome is None else _daily_series(outcome, registry)
            if series is not None and not invalid[arm.fs1_id]:
                daily[arm.fs1_id] = series
        table: dict[str, dict[str, dict[str, float]]] = {}
        cash = daily.get(registry.cash_arm)
        if cash is not None and len({len(v) for v in daily.values()}) == 1:
            indices = moving_block_indices(
                len(cash),
                block=protocol.bootstrap_block_days,
                resamples=protocol.bootstrap_resamples,
                seed=protocol.bootstrap_seed,
            )
            passive = daily.get(registry.passive_arm)
            for arm_id, series in daily.items():
                if arm_id == registry.cash_arm:
                    continue
                versus = {"vs_cash": _interval(series - cash, indices, protocol)}
                if passive is not None and arm_id != registry.passive_arm:
                    versus["vs_passive"] = _interval(
                        series - passive, indices, protocol
                    )
                table[arm_id] = versus
        bootstrap[scenario] = table

    labels: dict[str, dict[str, object]] = {}
    for arm_id in registry.candidate_arms:
        scenario_rows = {
            scenario: runs[arm_id].outcomes.get(scenario)
            for scenario in registry.drawdown_scenarios
        }
        drawdowns: dict[str, float | None] = {
            scenario: (
                None if outcome is None else float(outcome.summary["max_drawdown_pct"])  # type: ignore[arg-type]
            )
            for scenario, outcome in scenario_rows.items()
        }
        row = bootstrap.get(registry.excess_scenario, {}).get(arm_id, {})
        cash_interval = row.get("vs_cash", {})
        passive_interval = row.get("vs_passive", {})
        evidence = CandidateEvidence(
            valid=not invalid[arm_id],
            invalid_reasons=invalid[arm_id],
            max_drawdown_pct=drawdowns,
            excess_vs_cash_ci_low=cash_interval.get("ci95_low_annualized"),
            excess_vs_cash_bonferroni_low=cash_interval.get(
                "bonferroni_ci_low_annualized"
            ),
            excess_vs_passive_ci_low=passive_interval.get("ci95_low_annualized"),
        )
        labels[arm_id] = classify_candidate(
            evidence, controls_valid=controls_valid, registry=registry
        ).to_mapping()

    arms_table: dict[str, dict[str, Any]] = {}
    for arm in registry.arms:
        run = runs[arm.fs1_id]
        arms_table[arm.fs1_id] = {
            scenario: (
                run.outcomes[scenario].summary
                if scenario in run.outcomes
                else run.failures.get(scenario, {"status": "FAILED_TO_RUN"})
            )
            for scenario in registry.reported_scenarios
        }
    return {
        "labels": labels,
        "controls_valid": controls_valid,
        "invalid_reasons": {arm: list(reasons) for arm, reasons in invalid.items()},
        "arms": arms_table,
        "bootstrap": bootstrap,
        "causality_matrix": {
            arm.fs1_id: runs[arm.fs1_id].probe_matrix
            for arm in registry.arms
            if arm.role != ROLE_CASH
        },
        "failures": {arm: runs[arm].failures for arm in runs if runs[arm].failures},
        "label_counts": dict(Counter(row["label"] for row in labels.values())),
    }


def run_roster(
    permit: ScoringPermit,
    dataset: MarketDataset,
    *,
    progress: Progress | None = None,
) -> tuple[DevStudyProtocol, dict[str, ArmRun]]:
    """Replay every roster arm in every scenario and run the causality matrices."""

    permit = require_permit(permit)
    verify_dataset(permit, dataset)
    registry = permit.registry
    say = progress or (lambda message: None)
    protocol = registry.protocol(dataset.dataset_id)
    start_index = runner._row_of(dataset, registry.scoring_start)
    stop_index = runner._row_of(dataset, registry.scoring_end)
    context = make_phase3_context(dataset, protocol)
    runs = {
        arm.fs1_id: _score_arm(
            dataset,
            protocol,
            registry,
            context,
            arm.fs1_id,
            start_index=start_index,
            stop_index=stop_index,
            say=say,
        )
        for arm in registry.arms
    }
    return protocol, runs


def score_fs1(
    permit: ScoringPermit,
    dataset: MarketDataset,
    *,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Replay the roster once and label it. Requires a permit and a sealed dataset."""

    protocol, runs = run_roster(permit, dataset, progress=progress)
    registry = permit.registry
    assembled = assemble_result(runs, protocol, registry)
    return {
        "schema_version": RESULT_SCHEMA,
        "status": "forward_study_fs1_scored_once",
        "params_digest": permit.params_digest,
        "protocol_digest": protocol_digest(protocol),
        "code_sha": permit.code_sha,
        "dataset_id": dataset.dataset_id,
        "manifest_sha256": permit.manifest_sha256,
        "fetch_authorization": permit.manifest.authorization_reference,
        "scoring_authorization": permit.authorization,
        "authorized_at_utc": permit.authorized_at_utc,
        "scoring_window": [registry.scoring_start, registry.scoring_end],
        "symbols": list(dataset.symbols),
        **assembled,
        "claim_limits": list(registry.claim_limits),
        "notes": [
            "forward evidence for three configurations selected from public "
            "development results; not a profitability claim, never a WINNER",
            "the authorization reference is recorded, not verified: it names the "
            "human decision that the code cannot check",
        ],
    }


@dataclass(frozen=True)
class _Environment:
    """Everything :func:`run_fs1` takes from outside the call: private and test-only."""

    registry: Fs1Registry
    frozen_params_digest: str
    clock: Callable[[], datetime]
    code_identity: Callable[[], CodeIdentity]
    load_dataset: Callable[[Path], MarketDataset]


def _environment() -> _Environment:
    """Production wiring: the frozen registry and digest, the real clock and git.

    Tests monkeypatch this factory; it is not a parameter of the public entrypoint, so a
    caller of :func:`run_fs1` can never score a non-frozen registry.
    """

    return _Environment(
        registry=FS1_REGISTRY,
        frozen_params_digest=FROZEN_PARAMS_DIGEST,
        clock=lambda: datetime.now(UTC),
        code_identity=current_code_identity,
        load_dataset=lambda path: runner.load_dataset(str(path)),
    )


def run_fs1(
    *,
    authorization: str | None,
    expected_params_digest: str | None,
    sealed_code_sha: str | None,
    manifest_path: Path,
    expected_manifest_sha256: str | None,
    dataset_path: Path,
    output_path: Path,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Authorize, load, verify, score once and write the result (write-once).

    The output path is claimed atomically (``open(..., "xb")``) after authorization and
    before the dataset is read, so a concurrent writer is refused (``output_exists``)
    before any scoring and a computed result can never be lost to a write race. If
    loading, verification or scoring fails, the claimed file is removed: no partial
    output remains.

    Always uses the frozen registry and digest, the real wall clock and the real git
    checkout. A refusal is raised before the dataset is read and before anything is
    written.
    """

    env = _environment()
    permit = None
    reasons: list[str] = []
    manifest_bytes = manifest_path.read_bytes() if manifest_path.is_file() else None
    try:
        permit = _authorize(
            registry=env.registry,
            frozen_params_digest=env.frozen_params_digest,
            now=env.clock(),
            authorization=authorization,
            expected_params_digest=expected_params_digest,
            sealed_code_sha=sealed_code_sha,
            code_identity=env.code_identity(),
            manifest_bytes=manifest_bytes,
            expected_manifest_sha256=expected_manifest_sha256,
        )
    except Fs1Refusal as refusal:
        reasons.extend(refusal.reasons)
    if output_path.exists():
        reasons.append("output_exists")
    if reasons or permit is None:
        raise Fs1Refusal(tuple(dict.fromkeys(reasons)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        stream = output_path.open("xb")
    except FileExistsError:
        raise Fs1Refusal(("output_exists",)) from None
    try:
        try:
            dataset = env.load_dataset(dataset_path)
        except Exception as exc:
            raise Fs1Refusal(
                ("dataset_load_failed",), f"{type(exc).__name__}: {exc}"[:300]
            ) from exc
        result = score_fs1(permit, dataset, progress=progress)
        payload = (
            json.dumps(_json_safe(result), sort_keys=True, indent=2, allow_nan=False)
            + "\n"
        ).encode("utf-8")
        stream.write(payload)
    except BaseException:
        stream.close()
        output_path.unlink(missing_ok=True)
        raise
    stream.close()
    return result


__all__ = [
    "ArmRun",
    "arm_invalid_reasons",
    "assemble_result",
    "build_fs1_arm",
    "run_fs1",
    "run_roster",
    "score_fs1",
]
