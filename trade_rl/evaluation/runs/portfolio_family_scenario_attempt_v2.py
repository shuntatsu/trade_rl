"""Write-once durable attempt wrapper for explicit four-family scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from trade_rl._validation import require_sha256
from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.artifacts.verified_file import open_regular_binary
from trade_rl.evaluation.runs.portfolio_family_comparison import (
    _FAMILY_NAMES,
    PortfolioFamilyReplayOutcome,
)
from trade_rl.evaluation.runs.portfolio_family_result_artifact import (
    _family_payload,
    _require_object,
    _strict_json,
    _validate_family_payload,
)
from trade_rl.evaluation.runs.portfolio_family_scenario_replay_v2 import (
    FourFamilyPortfolioScenarioReplayV2Result,
    run_four_family_shared_portfolio_scenario_replay_v2,
)
from trade_rl.evaluation.runs.portfolio_family_scenario_result_artifact_v2 import (
    FourFamilyPortfolioScenarioAttemptIdentityV2,
    LoadedFourFamilyPortfolioScenarioResultArtifactV2,
    _publish_record,
    capture_four_family_scenario_durable_source_digest_v2,
    four_family_portfolio_scenario_attempt_identity_v2_digest,
    four_family_portfolio_scenario_attempt_identity_v2_payload,
    load_four_family_portfolio_scenario_result_artifact_v2,
    publish_four_family_portfolio_scenario_result_artifact_v2,
)
from trade_rl.evaluation.runs.provenance import build_candidate_run_provenance
from trade_rl.strategies.rl.portfolio_scenario_runtime import (
    capture_portfolio_scenario_inference_source_digest,
)


@dataclass(frozen=True, slots=True)
class FourFamilyPortfolioScenarioAttemptContextV2:
    """External roots plus the complete v2 scenario attempt identity."""

    identity: FourFamilyPortfolioScenarioAttemptIdentityV2
    dataset_artifact_root: str | Path
    ridge_artifact_root: str | Path
    bundle_root: str | Path

    def __post_init__(self) -> None:
        if type(self.identity) is not FourFamilyPortfolioScenarioAttemptIdentityV2:
            raise TypeError(
                "identity must be exactly FourFamilyPortfolioScenarioAttemptIdentityV2"
            )
        self.identity._validate()
        for name in ("dataset_artifact_root", "ridge_artifact_root", "bundle_root"):
            value = getattr(self, name)
            if not isinstance(value, (str, Path)) or not str(value).strip():
                raise ValueError(f"{name} must be a non-empty path")


@dataclass(frozen=True, slots=True)
class PublishedFourFamilyPortfolioScenarioAttemptV2:
    root: Path
    result_artifact_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.root, Path):
            raise TypeError("root must be a pathlib.Path")
        require_sha256(self.result_artifact_digest, field="result_artifact_digest")


@dataclass(frozen=True, slots=True)
class LoadedFourFamilyPortfolioScenarioAttemptV2:
    attempt: PublishedFourFamilyPortfolioScenarioAttemptV2
    result: LoadedFourFamilyPortfolioScenarioResultArtifactV2


def _absolute_root(value: str | Path) -> Path:
    root = Path(value).expanduser().absolute()
    if not root.name:
        raise ValueError("attempt_root must name a directory below its parent")
    return root


def _declared_root(value: str | Path) -> str:
    return Path(value).expanduser().absolute().as_posix()


def _reserve(root: Path) -> None:
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"scenario attempt destination already exists: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir()


def _validate_provenance(
    provenance: Mapping[str, object],
    identity: FourFamilyPortfolioScenarioAttemptIdentityV2,
) -> None:
    common = identity.common_input_declaration
    if provenance.get("implementation_digest") != common.expected_implementation_digest:
        raise ValueError("implementation provenance differs from expected digest")
    if (
        provenance.get("runtime_environment_digest")
        != common.expected_runtime_environment_digest
    ):
        raise ValueError("runtime provenance differs from expected digest")
    if provenance.get("research_context_digest") != common.research_context_digest:
        raise ValueError("research context differs from expected provenance")


def _source_record(
    identity: FourFamilyPortfolioScenarioAttemptIdentityV2,
) -> dict[str, object]:
    durable_source = capture_four_family_scenario_durable_source_digest_v2()
    scenario_source = capture_portfolio_scenario_inference_source_digest()
    if durable_source != identity.expected_durable_source_sha256:
        raise ValueError("current v2 durable source differs from expected digest")
    if scenario_source != identity.expected_scenario_inference_source_sha256:
        raise ValueError(
            "current scenario inference source differs from expected digest"
        )
    return {
        "durable_source_sha256": durable_source,
        "scenario_inference_source_sha256": scenario_source,
        "schema_version": "four_family_portfolio_scenario_source_v2",
    }


def _failure_note(original: BaseException, secondary: BaseException) -> None:
    add_note = getattr(original, "add_note", None)
    if callable(add_note):
        add_note(
            f"scenario attempt failure journal also failed: {type(secondary).__name__}: {secondary}"
        )


def _read_regular_bytes(path: Path, *, field: str) -> bytes:
    if path.is_symlink():
        raise ValueError(f"{field} must not be a symlink")
    with open_regular_binary(path, field=field) as handle:
        return handle.read()


def run_four_family_portfolio_scenario_attempt_v2(
    *,
    attempt_root: str | Path,
    context: FourFamilyPortfolioScenarioAttemptContextV2,
) -> PublishedFourFamilyPortfolioScenarioAttemptV2:
    """Reserve, execute, persist each arm, and complete one strict v2 attempt."""
    if type(context) is not FourFamilyPortfolioScenarioAttemptContextV2:
        raise TypeError(
            "context must be exactly FourFamilyPortfolioScenarioAttemptContextV2"
        )
    identity = context.identity
    identity._validate()
    identity_payload = four_family_portfolio_scenario_attempt_identity_v2_payload(
        identity
    )
    common = identity.common_input_declaration
    root = _absolute_root(attempt_root)
    roots = {
        "bundle_root": _declared_root(context.bundle_root),
        "dataset_artifact_root": _declared_root(context.dataset_artifact_root),
        "ridge_artifact_root": _declared_root(context.ridge_artifact_root),
    }
    _reserve(root)

    phase = "started_record"
    completed_families: list[str] = []
    completed_outcomes: list[PortfolioFamilyReplayOutcome] = []
    before_provenance: Mapping[str, object] | None = None
    after_provenance: Mapping[str, object] | None = None
    before_source: Mapping[str, object] | None = None
    after_source: Mapping[str, object] | None = None
    replay_result: FourFamilyPortfolioScenarioReplayV2Result | None = None
    result_digest: str | None = None
    complete_written = False
    try:
        _publish_record(
            root / "started" / "record.json",
            {
                "declared_roots": roots,
                "identity": identity_payload,
                "schema_version": "four_family_portfolio_scenario_attempt_started_v2",
            },
        )

        phase = "before_provenance"
        before_provenance = build_candidate_run_provenance(
            research_context_digest=common.research_context_digest
        )
        _validate_provenance(before_provenance, identity)
        _publish_record(
            root / "provenance" / "before" / "record.json",
            {
                "provenance": before_provenance,
                "schema_version": "four_family_portfolio_provenance_v1",
            },
        )
        before_source = _source_record(identity)
        _publish_record(
            root / "scenario_source" / "before" / "record.json", before_source
        )

        phase = "four_family_scenario_replay"

        def persist_completed(outcome: PortfolioFamilyReplayOutcome) -> None:
            expected_position = len(completed_families)
            if expected_position >= len(_FAMILY_NAMES):
                raise ValueError(
                    "scenario replay returned more than four family outcomes"
                )
            expected_family = _FAMILY_NAMES[expected_position]
            if (
                type(outcome) is not PortfolioFamilyReplayOutcome
                or outcome.family != expected_family
            ):
                raise ValueError("scenario replay callback violated fixed family order")
            # The replay checks its inference owner at native boundaries. Recheck
            # both durable v2 owners before accepting a row and starting the next arm.
            current_source = _source_record(identity)
            if canonical_json_bytes(current_source) != canonical_json_bytes(
                before_source
            ):
                raise RuntimeError(
                    "scenario source changed during the four-family attempt"
                )
            row = _family_payload(outcome)
            row = _validate_family_payload(
                row,
                expected_family=expected_family,
                identity=common,
            )
            _publish_record(root / "families" / expected_family / "record.json", row)
            completed_families.append(expected_family)
            completed_outcomes.append(outcome)

        replay_result = run_four_family_shared_portfolio_scenario_replay_v2(
            dataset_artifact_root=context.dataset_artifact_root,
            expected_dataset_artifact_sha256=common.expected_dataset_artifact_sha256,
            expected_symbols=common.expected_symbols,
            ridge_artifact_root=context.ridge_artifact_root,
            expected_ridge_artifact_sha256=common.ridge_artifact_sha256,
            expected_ridge_binding=common.expected_ridge_binding,
            bundle_root=context.bundle_root,
            expected_bundle_sha256=common.expected_bundle_sha256,
            expected_manifest_sha256=common.expected_manifest_sha256,
            expected_completion_sha256=common.expected_completion_sha256,
            expected_training_context_sha256=common.expected_training_context_sha256,
            expected_checkpoint_sha256=common.expected_checkpoint_sha256,
            expected_scenario=identity.expected_scenario,
            expected_scenario_sha256=identity.expected_scenario_sha256,
            expected_scenario_inference_source_sha256=(
                identity.expected_scenario_inference_source_sha256
            ),
            bound=common.bound,
            start_index=common.start_index,
            stop_index=common.stop_index,
            initial_capital=common.initial_capital,
            target_abs_weights=common.target_abs_weights,
            trend_config=common.trend_config,
            ridge_entry_threshold=common.ridge_entry_threshold,
            ridge_exit_threshold=common.ridge_exit_threshold,
            ridge_one_way_switch_cost=common.ridge_one_way_switch_cost,
            risk_config=common.risk_config,
            market_order_profile=common.market_order_profile,
            on_family_completed=persist_completed,
        )
        if type(replay_result) is not FourFamilyPortfolioScenarioReplayV2Result:
            raise TypeError("scenario replay owner returned an invalid result type")
        outcomes = tuple(replay_result.common_replay.families)
        if tuple(completed_families) != _FAMILY_NAMES:
            raise ValueError("scenario replay did not persist all four family outcomes")
        if tuple(item.family for item in outcomes) != _FAMILY_NAMES:
            raise ValueError("scenario result family order differs from fixed roster")
        if any(
            outcomes[index] is not outcome
            for index, outcome in enumerate(completed_outcomes)
        ):
            raise ValueError(
                "scenario replay returned family outcomes not observed by callback"
            )

        phase = "after_provenance"
        after_provenance = build_candidate_run_provenance(
            research_context_digest=common.research_context_digest
        )
        _validate_provenance(after_provenance, identity)
        _publish_record(
            root / "provenance" / "after" / "record.json",
            {
                "provenance": after_provenance,
                "schema_version": "four_family_portfolio_provenance_v1",
            },
        )
        after_source = _source_record(identity)
        _publish_record(
            root / "scenario_source" / "after" / "record.json", after_source
        )

        phase = "verify_provenance"
        if canonical_json_bytes(before_provenance) != canonical_json_bytes(
            after_provenance
        ):
            raise RuntimeError("implementation, runtime, or context provenance changed")
        if canonical_json_bytes(before_source) != canonical_json_bytes(after_source):
            raise RuntimeError("scenario source changed during the four-family attempt")

        phase = "publish_result"
        result_digest = publish_four_family_portfolio_scenario_result_artifact_v2(
            root,
            replay_result,
            identity,
            before_provenance=before_provenance,
            after_provenance=after_provenance,
            before_scenario_source=before_source,
            after_scenario_source=after_source,
        )
        phase = "strict_result_reload"
        loaded_result = load_four_family_portfolio_scenario_result_artifact_v2(
            root,
            expected_digest=result_digest,
            expected_identity=identity,
        )
        if loaded_result.artifact_digest != result_digest:
            raise ValueError("strict v2 result loader returned a different digest")

        phase = "completion_record"
        _publish_record(
            root / "complete" / "record.json",
            {
                "after_provenance_digest": content_digest(after_provenance),
                "after_scenario_source_digest": content_digest(after_source),
                "attempt_id": common.attempt_id,
                "before_provenance_digest": content_digest(before_provenance),
                "before_scenario_source_digest": content_digest(before_source),
                "expected_identity_digest": (
                    four_family_portfolio_scenario_attempt_identity_v2_digest(identity)
                ),
                "result_digest": result_digest,
                "schema_version": "four_family_portfolio_scenario_attempt_complete_v2",
            },
        )
        complete_written = True
        phase = "strict_attempt_reload"
        load_four_family_portfolio_scenario_attempt_v2(
            root,
            expected_identity=identity,
            expected_result_digest=result_digest,
        )
        return PublishedFourFamilyPortfolioScenarioAttemptV2(root, result_digest)
    except BaseException as error:
        failure: dict[str, object] = {
            "comparison_returned": replay_result is not None,
            "completed_families": list(completed_families),
            "exception_message": str(error),
            "exception_type": type(error).__name__,
            "phase": phase,
            "result_digest": result_digest,
            "schema_version": "four_family_portfolio_scenario_attempt_failure_v2",
        }
        if complete_written:
            failure["completion_marker_note"] = (
                "completion was written before a later strict reload failed; loader rejects failure/completion conflicts"
            )
        try:
            _publish_record(root / "failure" / "record.json", failure)
        except BaseException as journal_error:
            _failure_note(error, journal_error)
        raise


def load_four_family_portfolio_scenario_attempt_v2(
    attempt_root: str | Path,
    *,
    expected_identity: FourFamilyPortfolioScenarioAttemptIdentityV2,
    expected_result_digest: str,
) -> LoadedFourFamilyPortfolioScenarioAttemptV2:
    """Load only a complete nonfailed v2 attempt with exact file inventory."""
    require_sha256(expected_result_digest, field="expected_result_digest")
    if type(expected_identity) is not FourFamilyPortfolioScenarioAttemptIdentityV2:
        raise TypeError(
            "expected_identity must be exactly FourFamilyPortfolioScenarioAttemptIdentityV2"
        )
    expected_identity._validate()
    root = _absolute_root(attempt_root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("scenario attempt root must be a regular directory")
    failure_path = root / "failure" / "record.json"
    if failure_path.exists() or failure_path.is_symlink():
        raise ValueError("failed scenario attempts cannot be loaded as complete")
    complete_raw = _read_regular_bytes(
        root / "complete" / "record.json", field="scenario completion record"
    )
    complete = _require_object(
        _strict_json(complete_raw, field="scenario completion record"),
        field="scenario completion record",
        keys={
            "after_provenance_digest",
            "after_scenario_source_digest",
            "attempt_id",
            "before_provenance_digest",
            "before_scenario_source_digest",
            "expected_identity_digest",
            "result_digest",
            "schema_version",
        },
    )
    if (
        complete["schema_version"]
        != "four_family_portfolio_scenario_attempt_complete_v2"
    ):
        raise ValueError("scenario completion record schema is unsupported")
    if complete["attempt_id"] != expected_identity.common_input_declaration.attempt_id:
        raise ValueError(
            "scenario completion record identity differs from expected attempt"
        )
    if complete["result_digest"] != expected_result_digest:
        raise ValueError(
            "scenario completion result digest differs from expected digest"
        )
    expected_identity_digest = (
        four_family_portfolio_scenario_attempt_identity_v2_digest(expected_identity)
    )
    if complete["expected_identity_digest"] != expected_identity_digest:
        raise ValueError("scenario completion identity differs from expected attempt")
    result = load_four_family_portfolio_scenario_result_artifact_v2(
        root,
        expected_digest=expected_result_digest,
        expected_identity=expected_identity,
    )
    for field in (
        "before_provenance_digest",
        "after_provenance_digest",
        "before_scenario_source_digest",
        "after_scenario_source_digest",
    ):
        if complete[field] != result.payload[field]:
            raise ValueError(f"scenario completion {field} differs from result")
    inventory: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("scenario attempt tree must not contain symlinks")
        if path.is_file():
            inventory.add(path.relative_to(root).as_posix())
        elif not path.is_dir():
            raise ValueError("scenario attempt tree contains a non-regular entry")
    expected = {
        "started/record.json",
        "provenance/before/record.json",
        "provenance/after/record.json",
        "scenario_source/before/record.json",
        "scenario_source/after/record.json",
        "result/manifest.json",
        "complete/record.json",
        *(f"families/{name}/record.json" for name in _FAMILY_NAMES),
    }
    if inventory != expected:
        raise ValueError(
            "completed scenario attempt tree has missing or unexpected files"
        )
    return LoadedFourFamilyPortfolioScenarioAttemptV2(
        attempt=PublishedFourFamilyPortfolioScenarioAttemptV2(
            root=root,
            result_artifact_digest=result.artifact_digest,
        ),
        result=result,
    )


__all__ = [
    "FourFamilyPortfolioScenarioAttemptContextV2",
    "LoadedFourFamilyPortfolioScenarioAttemptV2",
    "PublishedFourFamilyPortfolioScenarioAttemptV2",
    "load_four_family_portfolio_scenario_attempt_v2",
    "run_four_family_portfolio_scenario_attempt_v2",
]
